// cne-setup: first-run configuration CLI. Detects hardware, previews the
// regime the engine WOULD classify, and lets the user confirm or override
// every setting before anything is written. Detection informs, user decides;
// aborting never leaves a config file behind.
#include "cne/memory_budget.h"
#include "cne/kv_budget.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <sys/statvfs.h>
#include <thread>
#include <unistd.h>
#include <vector>

using json = nlohmann::json;
namespace fs = std::filesystem;

namespace {

struct Item {
    std::string key;      // json key
    std::string label;    // prompt label
    std::string value;    // confirmed value ("" = engine default)
    std::string reason;   // why the suggestion looks like this
};

// Input contract: every field has an accepted grammar; bad values re-prompt
// interactively and abort scripted runs loudly.
bool valid_value(const std::string& key, const std::string& v,
                 std::string& err) {
    auto all_digits = [](const std::string& s) {
        return !s.empty() && s.find_first_not_of("0123456789") ==
                                 std::string::npos;
    };
    if (key == "stream" || key == "think" || key == "kernels")
        err = "on | off";
    else if (key == "dense")
        err = "mmap | warm | anon";
    else if (key == "mtp" || key == "threads" || key == "ctx" ||
             key == "session_max" || key == "session_max_per_user" ||
             key == "api_rpm")
        err = "a whole number" +
              std::string(key == "threads" ? " >= 1" : "");
    else if (key == "port")
        err = "a port 1-65535";
    else if (key == "cache_gib")
        err = "a GiB number";
    else if (key == "max_req_s")
        err = "seconds (e.g. 600)";
    else if (key == "host")
        err = "an IP or hostname";
    else if (key == "api_keys_file")
        err = "a path (relative to models dir)";
    else
        return true;

    if (v.empty()) return true;   // '!' cleared -> engine default
    if (key == "stream" || key == "think" || key == "kernels" ||
        key == "api_mode" || key == "prefetch") {
        if (v != "on" && v != "off" && v != "0" && v != "1") return false;
        return true;
    }
    if (key == "dense")
        return v == "mmap" || v == "warm" || v == "anon";
    if (key == "host")
        return v.find_first_of(" \t/") == std::string::npos;
    if (key == "port")
        return all_digits(v) && atoi(v.c_str()) >= 1 &&
               atoi(v.c_str()) <= 65535;
    if (key == "max_req_s") {
        char* end = nullptr;
        double d = strtod(v.c_str(), &end);
        return end && *end == '\0' && d >= 0;
    }
    // remaining numerics: mtp, threads, ctx, cache_gib
    if (!all_digits(v)) return false;
    long n = atol(v.c_str());
    if (key == "threads" || key == "ctx") return n >= 1;
    if (key == "session_max" || key == "session_max_per_user")
        return n >= 1 && n <= 64;
    if (key == "api_rpm") return n >= 0;
    if (key == "api_keys_file")
        return v.find_first_of(" \t") == std::string::npos;
    return true;   // mtp >= 0, cache_gib >= 0 (= clamp)
}

bool read_line(std::string& out) {
    if (!std::getline(std::cin, out)) return false;
    while (!out.empty() && (out.back() == '\r' || out.back() == ' '))
        out.pop_back();
    return true;
}

std::string gib(uint64_t bytes) {
    char buf[32];
    snprintf(buf, sizeof(buf), "%.1f GiB", (double)bytes / (1ull << 30));
    return buf;
}

std::vector<fs::path> scan_artifacts(const fs::path& dir) {
    std::vector<fs::path> out;
    for (const auto& e : fs::recursive_directory_iterator(dir))
        if (e.is_regular_file() && e.path().extension() == ".gguf")
            out.push_back(e.path());
    // prepared artifacts first: alignment bump + preserved MTP tensors
    std::sort(out.begin(), out.end(), [](const fs::path& a, const fs::path& b) {
        bool pa = a.string().find("-prepared") != std::string::npos;
        bool pb = b.string().find("-prepared") != std::string::npos;
        if (pa != pb) return pa > pb;
        return a.filename() < b.filename();
    });
    return out;
}

// Suggestion table. Sources: SERVER.md section 6 measurements and the
// operational rules in internal-docs/ops/ENGINE_STATUS.md section 7/8.
std::vector<Item> build_suggestions(const fs::path& artifact,
                                    uint64_t model_bytes,
                                    const cne::MemoryBudget& budget,
                                    int hw_threads) {
    cne::Regime regime = cne::classify(model_bytes, budget.mem_available);
    bool deep = regime == cne::Regime::R3_FAR_ABOVE ||
                regime == cne::Regime::R4_EXTREME_EDGE;
    // heuristic: MTP artifacts carry nextn tensors; the release layout marks
    // them in the artifact directory name
    bool has_mtp =
        artifact.string().find("mtp") != std::string::npos;
    const bool mtp_on = has_mtp && !deep;
    cne::ServingKvEstimate kv =
        cne::suggest_serving_kv(budget, model_bytes, regime, mtp_on);
    char kv_note[160];
    snprintf(kv_note, sizeof(kv_note),
             "~%.0f MiB KV est at ctx=%d x %d sessions",
             kv.kv_bytes() / (1024.0 * 1024.0), kv.n_ctx, kv.n_seq_max);

    printf("\nartifact : %s (%s)\n", artifact.filename().c_str(),
           gib(model_bytes).c_str());
    printf("hardware : %s RAM available / %s total, %d logical threads\n",
           gib(budget.mem_available).c_str(), gib(budget.mem_total).c_str(),
           hw_threads);
    printf("regime   : %s (preview only - nothing is auto-applied)\n\n",
           cne::regime_name(regime));

    std::vector<Item> items;
    auto add = [&](const char* k, const char* label, std::string val,
                   std::string reason) {
        items.push_back({k, label, std::move(val), std::move(reason)});
    };

    add("stream", "stream decode", deep ? "on" : "off",
        deep ? "model far above RAM - streaming is the product here"
             : "cache holds the hot set at this regime; naive mmap decode "
               "measured equal-or-better");

    add("prefetch", "prefetch overlap", "off",
        "speculative expert prefetch during streaming; default off "
        "(measured neutral/regressive on current hardware)");

    add("kernels", "custom CPU kernels", "on",
        "fork ggml-cpu fast path; off = stock llama A/B");

    add("dense", "dense residency",
        (has_mtp && !deep) ? "mmap" : "",
        (has_mtp && !deep)
            ? "MTP-era A/B: mmap beat anon under speculation (SERVER.md §6)"
            : "empty = engine auto-classifies per regime");

    add("mtp", "MTP draft depth", mtp_on ? "8" : "0",
        has_mtp
            ? (deep
                   ? "net-negative below ~0.90 expert-cache hit-rate; off in "
                     "deep streaming regimes"
                   : "depth 8 + p_min 0.5 measured ~+20-120% lossless; "
                     "incompatible with conversation_id sessions")
            : "no MTP tensors detected (filename heuristic); leave 0 for chat "
              "sessions");

    add("threads", "threads", std::to_string(std::max(1, hw_threads / 2)),
        "physical-core count beats SMT for both decode arms on this class");

    add("ctx", "context size", std::to_string(kv.n_ctx), kv_note);

    add("session_max", "conversation lanes", std::to_string(kv.n_seq_max),
        mtp_on ? "1 when MTP on (sessions need CNE_MTP=0)"
               : "serial multi-user KV parking; see KV estimate above");

    add("api_mode", "API mode (multi-user)", mtp_on ? "off" : "on",
        mtp_on ? "MTP and API sessions are incompatible"
               : "API key auth + chat_id tenancy; use gateway for client keys");

    add("session_max_per_user", "sessions per user", "2",
        "per-user LRU when API mode on; 0 = global LRU only");

    add("api_rpm", "API rate limit", "120",
        "requests/minute per user in API mode; 0 = off");

    add("api_keys_file", "internal API keys file", "api_keys.txt",
        "gateway-only secret; client keys live in gateway/api_keys.local.txt");

    add("cache_gib", "expert cache cap", "",
        "empty = budget manager clamps to MemAvailable (recommended)");

    add("think", "thinking default", "on",
        "model-native behavior; requests may still toggle per call");

    add("max_req_s", "wall cap per request", "0",
        "0 = off; e.g. 600 bounds abandoned non-streaming generations");

    add("host", "host", "127.0.0.1", "");

    add("port", "port", "8080", "");

    return items;
}

std::string gen_internal_api_key() {
    unsigned char buf[24];
    FILE* f = fopen("/dev/urandom", "rb");
    if (!f) return "change-me-in-production";
    size_t n = fread(buf, 1, sizeof(buf), f);
    fclose(f);
    if (n != sizeof(buf)) return "change-me-in-production";
    static const char hex[] = "0123456789abcdef";
    std::string out = "cne_internal_";
    for (unsigned char b : buf) {
        out += hex[b >> 4];
        out += hex[b & 0xf];
    }
    return out;
}

bool api_mode_on(const std::vector<Item>& items) {
    for (const auto& it : items) {
        if (it.key != "api_mode") continue;
        return it.value == "on" || it.value == "1" || it.value == "true";
    }
    return false;
}

} // namespace

int main(int argc, char** argv) {
    fs::path models_dir = "models";
    fs::path config_path;
    bool accept_all = false;
    std::string model_flag;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "-y" || a == "--accept-all") accept_all = true;
        else if (a == "--config") {
            if (++i >= argc) { fprintf(stderr, "--config needs a path\n"); return 2; }
            config_path = argv[i];
        } else if (a == "--model") {
            if (++i >= argc) { fprintf(stderr, "--model needs a name\n"); return 2; }
            model_flag = argv[i];
        } else if (!a.empty() && a[0] == '-') {
            fprintf(stderr, "unknown flag %s\n", a.c_str());
            return 2;
        } else models_dir = a;
    }

    if (!fs::exists(models_dir)) {
        fprintf(stderr, "models dir not found: %s\n", models_dir.c_str());
        return 1;
    }
    if (config_path.empty()) config_path = models_dir / "server.json";

    auto artifacts = scan_artifacts(models_dir);
    if (artifacts.empty()) {
        fprintf(stderr, "no .gguf artifacts in %s\n", models_dir.c_str());
        return 1;
    }

    auto budget = cne::MemoryBudget::detect();
    int hw_threads = (int)std::thread::hardware_concurrency();

    struct statvfs vfs {};
    uint64_t disk_free = 0;
    if (statvfs(models_dir.c_str(), &vfs) == 0)
        disk_free = (uint64_t)vfs.f_bavail * vfs.f_frsize;

    const fs::path* artifact = nullptr;
    if (!model_flag.empty()) {
        for (const auto& p : artifacts)
            if (p.filename().string().find(model_flag) != std::string::npos) {
                artifact = &p;
                break;
            }
        if (!artifact) {
            fprintf(stderr, "no artifact matches --model %s\n",
                    model_flag.c_str());
            return 1;
        }
    } else if (artifacts.size() == 1 || accept_all) {
        artifact = &artifacts[0];
    } else {
        printf("artifacts in %s:\n", models_dir.c_str());
        for (size_t i = 0; i < artifacts.size(); i++)
            printf("  [%zu] %s (%s)\n", i, artifacts[i].filename().c_str(),
                   gib(fs::file_size(artifacts[i])).c_str());
        printf("pick [0]: ");
        std::string line;
        if (!read_line(line) || line.empty()) line = "0";
        unsigned long idx = 0;
        try { idx = std::stoul(line); }
        catch (...) { fprintf(stderr, "bad index\n"); return 1; }
        if (idx >= artifacts.size()) { fprintf(stderr, "bad index\n"); return 1; }
        artifact = &artifacts[idx];
    }

    uint64_t model_bytes = fs::file_size(*artifact);
    if (disk_free && disk_free < model_bytes)
        printf("WARNING: only %s free on disk, artifact is %s\n",
               gib(disk_free).c_str(), gib(model_bytes).c_str());

    std::vector<Item> items =
        build_suggestions(*artifact, model_bytes, budget, hw_threads);

    printf("confirm each setting (Enter = suggestion, type a value = override,"
           " '!' = engine default; accepted values shown per prompt)\n");
    for (auto& it : items) {
        // scripted mode presets one item via CNE_SETUP_<KEY>
        std::string envkey = "CNE_SETUP_" + it.key;
        for (char& c : envkey)
            if (c >= 'a' && c <= 'z') c = char(c - 'a' + 'A');
        const char* preset = getenv(envkey.c_str());

        if (accept_all || (preset && *preset)) {
            if (preset && *preset) {
                std::string err;
                if (!valid_value(it.key, preset, err)) {
                    fprintf(stderr, "%s: invalid %s preset '%s' "
                                    "(expected: %s)\n",
                            envkey.c_str(), it.key.c_str(), preset, err.c_str());
                    return 2;
                }
                it.value = preset;
            }
            printf("%-22s %-10s # %s\n", it.label.c_str(),
                   it.value.empty() ? "(engine default)" : it.value.c_str(),
                   it.reason.c_str());
            continue;
        }
        if (!isatty(0)) {
            fprintf(stderr,
                    "non-interactive stdin and no %s set for '%s'; "
                    "use flags or -y\n",
                    envkey.c_str(), it.key.c_str());
            return 1;
        }
        std::string err_unused;
        while (true) {
            std::string hint;
            valid_value(it.key, "", hint);
            std::string suffix = hint.empty() ? "" : " (" + hint + ")";
            printf("%-22s [%s]%s : ", it.label.c_str(),
                   it.value.empty() ? "engine default" : it.value.c_str(),
                   suffix.c_str());
            std::string line;
            if (!read_line(line)) {
                fprintf(stderr, "\naborted - no file written\n");
                return 130;
            }
            if (line.empty()) break;              // keep suggestion
            if (line == "!") { it.value.clear(); break; }
            if (!valid_value(it.key, line, err_unused)) {
                printf("  invalid value (expected: %s) - try again\n",
                       hint.c_str());
                continue;
            }
            it.value = line;
            break;
        }
    }

    printf("\nresolved configuration:\n");
    for (const auto& it : items)
        printf("  %-11s = %s\n", it.key.c_str(),
               it.value.empty() ? "(engine default)" : it.value.c_str());
    printf("config file: %s\n", config_path.string().c_str());

    if (!accept_all) {
        printf("write? [Y/n] ");
        std::string line;
        if (!read_line(line)) return 130;
        if (!line.empty() && line != "y" && line != "Y") {
            printf("aborted - no file written\n");
            return 130;
        }
    }

    json cfg;
    cfg["model"] = fs::relative(*artifact, models_dir).string();
    int mtp_k = 0;
    bool api_on = api_mode_on(items);
    std::string api_keys_rel = "api_keys.txt";
    for (const auto& it : items) {
        if (it.value.empty()) continue;
        if (it.key == "stream" || it.key == "think" || it.key == "kernels" ||
            it.key == "prefetch") {
            cfg[it.key] = (it.value == "on" || it.value == "1" ||
                           it.value == "true");
        } else if (it.key == "api_mode") {
            api_on = (it.value == "on" || it.value == "1" ||
                      it.value == "true");
            if (api_on) cfg[it.key] = true;
        } else if (it.key == "threads" || it.key == "ctx" || it.key == "port" ||
                 it.key == "session_max")
            cfg[it.key] = atoi(it.value.c_str());
        else if (it.key == "session_max_per_user" || it.key == "api_rpm") {
            if (!api_on) continue;
            cfg[it.key] = atoi(it.value.c_str());
        } else if (it.key == "max_req_s")
            cfg[it.key] = atof(it.value.c_str());
        else if (it.key == "mtp") {
            mtp_k = atoi(it.value.c_str());
            cfg[it.key] = mtp_k;
        } else if (it.key == "cache_gib") {
            if (atoi(it.value.c_str()) > 0) cfg[it.key] = atoi(it.value.c_str());
        } else if (it.key == "api_keys_file") {
            api_keys_rel = it.value;
            if (api_on) cfg[it.key] = it.value;
        } else
            cfg[it.key] = it.value;
    }
    if (api_on) {
        if (!cfg.contains("session_max") || cfg["session_max"].get<int>() < 2) {
            fprintf(stderr,
                    "[setup] session_max < 2 with API mode — bumping to 2\n");
            cfg["session_max"] = 2;
        }
        const fs::path keys_path = models_dir / api_keys_rel;
        if (!fs::exists(keys_path)) {
            const char* preset = getenv("CNE_SETUP_INTERNAL_API_KEY");
            std::string key =
                (preset && *preset) ? preset : gen_internal_api_key();
            std::ofstream kf(keys_path);
            if (!kf) {
                fprintf(stderr, "cannot write %s\n", keys_path.c_str());
                return 1;
            }
            kf << "# Internal key for cne_gateway → cne_server (not client keys)\n"
               << key << "\n";
            printf("wrote internal API key to %s\n", keys_path.c_str());
        }
    }
    if (mtp_k > 0) cfg["mtp_p_min"] = 0.5;   // measured pairing, SERVER.md §6

    {
        bool stream_on = false, prefetch_on = false;
        for (const auto& it : items) {
            if (it.key == "stream")
                stream_on = (it.value == "on" || it.value == "1" ||
                             it.value == "true");
            if (it.key == "prefetch")
                prefetch_on = (it.value == "on" || it.value == "1" ||
                               it.value == "true");
        }
        if (prefetch_on && !stream_on)
            fprintf(stderr,
                    "[setup] prefetch overlap only applies when stream is on "
                    "(ignored at boot)\n");
    }

    {
        std::ofstream f(config_path);
        if (!f) {
            fprintf(stderr, "cannot write %s\n", config_path.string().c_str());
            return 1;
        }
        f << cfg.dump(2) << "\n";
    }

    std::string envs;
    auto kv = [&](const char* name, const std::string& v) {
        envs += std::string(name) + "=" + v + " ";
    };
    auto env_on = [](const std::string& v) {
        return (v == "on" || v == "1" || v == "true") ? "1" : "0";
    };
    for (const auto& it : items) {
        if (it.value.empty()) continue;
        if (it.key == "stream")         kv("CNE_STREAM", env_on(it.value));
        else if (it.key == "prefetch")  kv("CNE_PREFETCH", env_on(it.value));
        else if (it.key == "kernels")   kv("CNE_KERNELS", env_on(it.value));
        else if (it.key == "dense")     kv("CNE_DENSE", it.value);
        else if (it.key == "mtp")       kv("CNE_MTP", it.value);
        else if (it.key == "threads")   kv("CNE_THREADS", it.value);
        else if (it.key == "ctx")       kv("CNE_CTX", it.value);
        else if (it.key == "session_max") kv("CNE_SESSION_MAX", it.value);
        else if (it.key == "session_max_per_user")
            kv("CNE_SESSION_MAX_PER_USER", it.value);
        else if (it.key == "api_mode")
            kv("CNE_API_MODE", env_on(it.value));
        else if (it.key == "api_rpm") kv("CNE_API_RPM", it.value);
        else if (it.key == "cache_gib") kv("CNE_CACHE_GIB", it.value);
        else if (it.key == "think")     kv("CNE_THINK", env_on(it.value));
        else if (it.key == "max_req_s") kv("CNE_MAX_REQ_S", it.value);
    }
    if (mtp_k > 0) kv("CNE_MTP_P_MIN", "0.5");
    std::string host = "127.0.0.1", port = "8080";
    for (const auto& it : items) {
        if (it.key == "host" && !it.value.empty()) host = it.value;
        if (it.key == "port" && !it.value.empty()) port = it.value;
    }
    printf("\nwrote %s\nwhat will run:\n  %scne_server %s %s %s\n",
           config_path.string().c_str(), envs.c_str(),
           artifact->string().c_str(), host.c_str(), port.c_str());
    if (api_on) {
        printf("\nAPI mode: expose cne_server on 127.0.0.1 only; run cne_gateway "
               "for client keys.\n");
        printf("  gateway internal key: same value as %s\n",
               (models_dir / api_keys_rel).c_str());
        printf("  client keys file: gateway/api_keys.local.txt\n");
        printf("  see docs/GATEWAY.md\n");
    }
    return 0;
}
