// cne-setup: first-run configuration CLI. Detects hardware, previews the
// regime the engine WOULD classify, and lets the user confirm or override
// every setting before anything is written. Detection informs, user decides;
// aborting never leaves a config file behind.
#include "cne/memory_budget.h"

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
// operational rules in ENGINE_STATUS.md section 7/8.
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

    add("dense", "dense residency",
        (has_mtp && !deep) ? "mmap" : "",
        (has_mtp && !deep)
            ? "MTP-era A/B: mmap beat anon under speculation (SERVER.md §6)"
            : "empty = engine auto-classifies per regime");

    add("mtp", "MTP draft depth", (has_mtp && !deep) ? "8" : "0",
        has_mtp
            ? (deep
                   ? "net-negative below ~0.90 expert-cache hit-rate; off in "
                     "deep streaming regimes"
                   : "depth 8 + p_min 0.5 measured ~+20-120% lossless")
            : "no MTP tensors detected (filename heuristic)");

    add("threads", "threads", std::to_string(std::max(1, hw_threads / 2)),
        "physical-core count beats SMT for both decode arms on this class");

    add("ctx", "context size", "1024",
        "serving floor: MTP aborts near token ~250 below this");

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
           " '!' = engine default)\n");
    for (auto& it : items) {
        // scripted mode presets one item via CNE_SETUP_<KEY>
        std::string envkey = "CNE_SETUP_" + it.key;
        for (char& c : envkey)
            if (c >= 'a' && c <= 'z') c = char(c - 'a' + 'A');
        const char* preset = getenv(envkey.c_str());

        if (accept_all || (preset && *preset)) {
            if (preset && *preset) it.value = preset;
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
        printf("%-22s [%s] : ", it.label.c_str(),
               it.value.empty() ? "engine default" : it.value.c_str());
        std::string line;
        if (!read_line(line)) {
            fprintf(stderr, "\naborted - no file written\n");
            return 130;
        }
        if (line.empty()) continue;              // keep suggestion
        if (line == "!") { it.value.clear(); continue; }
        it.value = line;
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
    for (const auto& it : items) {
        if (it.value.empty()) continue;
        if (it.key == "stream" || it.key == "think")
            cfg[it.key] = (it.value == "on" || it.value == "1" ||
                           it.value == "true");
        else if (it.key == "threads" || it.key == "ctx" || it.key == "port")
            cfg[it.key] = atoi(it.value.c_str());
        else if (it.key == "max_req_s")
            cfg[it.key] = atof(it.value.c_str());
        else if (it.key == "mtp") {
            mtp_k = atoi(it.value.c_str());
            cfg[it.key] = mtp_k;
        } else if (it.key == "cache_gib") {
            if (atoi(it.value.c_str()) > 0) cfg[it.key] = atoi(it.value.c_str());
        } else
            cfg[it.key] = it.value;
    }
    if (mtp_k > 0) cfg["mtp_p_min"] = 0.5;   // measured pairing, SERVER.md §6

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
    for (const auto& it : items) {
        if (it.value.empty()) continue;
        if (it.key == "stream")         kv("CNE_STREAM", it.value == "on" ? "1" : "0");
        else if (it.key == "dense")     kv("CNE_DENSE", it.value);
        else if (it.key == "mtp")       kv("CNE_MTP", it.value);
        else if (it.key == "threads")   kv("CNE_THREADS", it.value);
        else if (it.key == "ctx")       kv("CNE_CTX", it.value);
        else if (it.key == "cache_gib") kv("CNE_CACHE_GIB", it.value);
        else if (it.key == "think")     kv("CNE_THINK", it.value == "on" ? "1" : "0");
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
    return 0;
}
