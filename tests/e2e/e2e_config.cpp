#include "e2e_config.h"

#include <nlohmann/json.hpp>

#include <cstdlib>
#include <fstream>
#include <filesystem>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace cne::e2e {

namespace {

bool read_json_file(const std::string& path, json& j, std::string& err) {
    std::ifstream in(path);
    if (!in) {
        err = "cannot open " + path;
        return false;
    }
    try {
        in >> j;
    } catch (const std::exception& e) {
        err = std::string("JSON parse error in ") + path + ": " + e.what();
        return false;
    }
    return true;
}

std::string resolve_path(const std::string& p,
                         const std::string& project_source_dir) {
    fs::path path(p);
    if (path.is_absolute()) return path.string();
    return (fs::path(project_source_dir) / path).lexically_normal().string();
}

} // namespace

std::string discover_path(const char* default_rel,
                          const std::string& project_source_dir) {
    if (const char* p = getenv("CNE_E2E_CONFIG")) return p;
    return (fs::path(project_source_dir) / default_rel).string();
}

bool load(const std::string& path, const std::string& project_source_dir,
          Config& out, std::string& err) {
    json j;
    if (!read_json_file(path, j, err)) return false;

    out = Config{};
    if (j.contains("model") && j["model"].is_string())
        out.model = j["model"].get<std::string>();

    if (j.contains("env") && j["env"].is_object()) {
        for (auto it = j["env"].begin(); it != j["env"].end(); ++it) {
            if (it.value().is_string())
                out.env[it.key()] = it.value().get<std::string>();
            else if (it.value().is_number_integer())
                out.env[it.key()] = std::to_string(it.value().get<int>());
            else if (it.value().is_boolean())
                out.env[it.key()] = it.value().get<bool>() ? "1" : "0";
        }
    }

    if (j.contains("runtime") && j["runtime"].is_object()) {
        const json& r = j["runtime"];
        if (r.contains("cap_gib")) out.runtime.cap_gib = r["cap_gib"].get<size_t>();
        if (r.contains("n_ctx")) out.runtime.n_ctx = r["n_ctx"].get<int>();
        if (r.contains("n_threads"))
            out.runtime.n_threads = r["n_threads"].get<int>();
        if (r.contains("stream_on"))
            out.runtime.stream_on = r["stream_on"].get<bool>();
    }

    if (j.contains("options") && j["options"].is_object()) {
        const json& o = j["options"];
        if (o.contains("port")) out.port = o["port"].get<int>();
        if (o.contains("boot_timeout_s"))
            out.boot_timeout_s = o["boot_timeout_s"].get<int>();
        if (o.contains("prompt_tokens"))
            out.prompt_tokens = o["prompt_tokens"].get<int>();
        if (o.contains("gen_tokens"))
            out.gen_tokens = o["gen_tokens"].get<int>();
    }

    out.model = resolve_path(out.model, project_source_dir);
    return true;
}

void apply_env(const Config& cfg, bool overwrite) {
    for (const auto& kv : cfg.env) {
        if (!overwrite && getenv(kv.first.c_str())) continue;
        setenv(kv.first.c_str(), kv.second.c_str(), 1);
    }
}

std::string resolve_model(const Config& cfg,
                          const std::string& project_source_dir) {
    if (const char* m = getenv("CNE_TEST_MODEL")) {
        if (m[0]) return m;
    }
    if (!cfg.model.empty()) return cfg.model;
    return {};
}

} // namespace cne::e2e
