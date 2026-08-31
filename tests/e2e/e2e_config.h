#pragma once
// JSON-driven env + runtime settings for opt-in live tests.
#include <optional>
#include <string>
#include <unordered_map>

namespace cne::e2e {

struct RuntimeOpts {
    size_t cap_gib   = 0;
    int    n_ctx     = 2048;
    int    n_threads = 4;
    bool   stream_on = false;
};

struct ChatOpts {
    std::string conversation_id = "e2e-session";
    std::string chat_id           = "e2e-chat";
    int         max_tokens      = 12;
    bool        think_off       = false;
};

struct GatewayOpts {
    std::unordered_map<std::string, std::string> env;
    std::string api_keys_file;
    std::string client_api_key; // Bearer key for live tests
    int         port = 0;       // 0 = server port + 1
};

struct Config {
    std::string model;
    std::unordered_map<std::string, std::string> env;
    RuntimeOpts runtime;
    ChatOpts    chat;
    GatewayOpts gateway;
    int port           = 0; // 0 = pick ephemeral
    int boot_timeout_s = 180;
    int prompt_tokens  = 0; // session_bigctx only; 0 = use default in test
    int gen_tokens     = 0;
};

// Load JSON config. Relative model paths resolve from project_source_dir.
bool load(const std::string& path, const std::string& project_source_dir,
          Config& out, std::string& err);

// CNE_E2E_CONFIG if set, else <project_source_dir>/<default_rel>.
std::string discover_path(const char* default_rel,
                          const std::string& project_source_dir);

void apply_env(const Config& cfg, bool overwrite = true);

// CNE_TEST_MODEL wins, then cfg.model (must be non-empty).
std::string resolve_model(const Config& cfg, const std::string& project_source_dir);

} // namespace cne::e2e
