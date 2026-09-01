// HTTP E2E: gateway chat policy (thinking block, max_tokens cap).
// Config: server_gateway_policy_live.json
#include "e2e_config.h"
#include <httplib.h>
#include <nlohmann/json.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <thread>
#include <unordered_map>

#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

using json = nlohmann::json;

#ifndef CNE_SERVER_EXE
#define CNE_SERVER_EXE "cne_server"
#endif

#ifndef CNE_PROJECT_SOURCE_DIR
#define CNE_PROJECT_SOURCE_DIR "."
#endif

namespace {

class ChildProcess {
public:
    bool start_server(const std::string& model, int port,
                      const std::unordered_map<std::string, std::string>& env) {
        return start_impl(env, [&] {
            const std::string host = "127.0.0.1";
            const std::string port_s = std::to_string(port);
            execl(CNE_SERVER_EXE, "cne_server", model.c_str(), host.c_str(),
                  port_s.c_str(), (char*) nullptr);
        });
    }

    bool start_gateway(const std::string& python,
                       const std::string& pythonpath, int port,
                       const std::unordered_map<std::string, std::string>& env) {
        return start_impl(env, [&] {
            setenv("PYTHONPATH", pythonpath.c_str(), 1);
            setenv("CNE_GATEWAY_PORT", std::to_string(port).c_str(), 1);
            setenv("CNE_GATEWAY_HOST", "127.0.0.1", 1);
            execl(python.c_str(), python.c_str(), "-m", "cne_gateway", (char*) nullptr);
        });
    }

    void stop() {
        if (pid_ <= 0) return;
        kill(pid_, SIGTERM);
        int st = 0;
        for (int i = 0; i < 40; i++) {
            if (waitpid(pid_, &st, WNOHANG) == pid_) {
                pid_ = -1;
                return;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(250));
        }
        if (pid_ > 0) {
            kill(pid_, SIGKILL);
            waitpid(pid_, &st, 0);
            pid_ = -1;
        }
    }

    const std::string& log_path() const { return log_path_; }

    ~ChildProcess() { stop(); }

private:
    template <typename Fn>
    bool start_impl(const std::unordered_map<std::string, std::string>& env,
                    Fn child_fn) {
        pid_ = fork();
        if (pid_ < 0) return false;
        if (pid_ == 0) {
            log_path_ = "/tmp/cne_gw_policy_child_" + std::to_string(getpid()) + ".log";
            FILE* log = freopen(log_path_.c_str(), "w", stderr);
            (void) log;
            for (const auto& kv : env) setenv(kv.first.c_str(), kv.second.c_str(), 1);
            child_fn();
            _exit(127);
        }
        log_path_ = "/tmp/cne_gw_policy_child_" + std::to_string(pid_) + ".log";
        return true;
    }

    pid_t       pid_ = -1;
    std::string log_path_;
};

std::string find_gateway_python(const std::string& src) {
    if (const char* p = getenv("CNE_GATEWAY_PYTHON")) {
        if (p[0] && access(p, X_OK) == 0) return p;
    }
    const std::string venv = src + "/gateway/.venv/bin/python";
    if (access(venv.c_str(), X_OK) == 0) return venv;
    if (access("/usr/bin/python3", X_OK) == 0) return "/usr/bin/python3";
    return {};
}

bool gateway_import_ok(const std::string& python, const std::string& pythonpath) {
    pid_t pid = fork();
    if (pid < 0) return false;
    if (pid == 0) {
        setenv("PYTHONPATH", pythonpath.c_str(), 1);
        execl(python.c_str(), python.c_str(), "-c",
              "import cne_gateway.main", (char*) nullptr);
        _exit(1);
    }
    int st = 0;
    waitpid(pid, &st, 0);
    return WIFEXITED(st) && WEXITSTATUS(st) == 0;
}

bool wait_cne_health(httplib::Client& cli, int timeout_s) {
    for (int i = 0; i < timeout_s; i++) {
        if (auto res = cli.Get("/health")) {
            if (res->status == 200) {
                try {
                    json j = json::parse(res->body);
                    if (j.value("status", "") == "ok" &&
                        j["api"]["enabled"].get<bool>())
                        return true;
                } catch (...) {}
            }
        }
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    return false;
}

bool wait_gateway_health(httplib::Client& cli, int timeout_s) {
    for (int i = 0; i < timeout_s; i++) {
        if (auto res = cli.Get("/health")) {
            if (res->status == 200) {
                try {
                    json j = json::parse(res->body);
                    if (j.value("status", "") == "ok" &&
                        j["cne"].value("status", "") == "ok")
                        return true;
                } catch (...) {}
            }
        }
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    return false;
}

bool post_json(httplib::Client& cli, const std::string& path,
               const httplib::Headers& hdrs, const json& body, int expect,
               std::string& err, json* out = nullptr) {
    auto res = cli.Post(path, hdrs, body.dump(), "application/json");
    if (!res) {
        err = "no HTTP response";
        return false;
    }
    if (res->status != expect) {
        err = "HTTP " + std::to_string(res->status) + ": " + res->body;
        return false;
    }
    if (out) {
        try {
            *out = json::parse(res->body);
        } catch (const std::exception& e) {
            err = e.what();
            return false;
        }
    }
    return true;
}

bool check_health_policy(httplib::Client& gw_cli, std::string& err) {
    auto res = gw_cli.Get("/health");
    if (!res || res->status != 200) {
        err = "health request failed";
        return false;
    }
    try {
        json j = json::parse(res->body);
        const json& gw = j.at("gateway");
        if (gw.value("allow_thinking", true)) {
            err = "expected gateway.allow_thinking=false";
            return false;
        }
        if (gw.value("max_tokens_per_request", 0) != 8) {
            err = "expected gateway.max_tokens_per_request=8";
            return false;
        }
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
    return true;
}

bool check_thinking_blocked(httplib::Client& gw_cli,
                            const httplib::Headers& auth, std::string& err) {
    json body = {{"messages", json::array({{{"role", "user"}, {"content", "hi"}}})},
                 {"stream", false},
                 {"chat_template_kwargs", {{"enable_thinking", true}}}};
    return post_json(gw_cli, "/v1/chat/completions", auth, body, 403, err);
}

bool check_max_tokens_clamped(httplib::Client& gw_cli,
                              const httplib::Headers& auth,
                              const std::string& chat_id, bool include_max_tokens,
                              std::string& err) {
    json body = {{"messages",
                  json::array({{{"role", "user"},
                                {"content",
                                 "List ten European capital cities with one "
                                 "sentence each."}}})},
                 {"stream", false},
                 {"temperature", 0},
                 {"chat_id", chat_id}};
    if (include_max_tokens) body["max_tokens"] = 64;

    json resp;
    if (!post_json(gw_cli, "/v1/chat/completions", auth, body, 200, err, &resp))
        return false;

    if (!resp.contains("usage") || !resp["usage"].contains("completion_tokens")) {
        err = "missing usage.completion_tokens";
        return false;
    }
    const int completion = resp["usage"]["completion_tokens"].get<int>();
    if (completion > 8) {
        err = "completion_tokens=" + std::to_string(completion) + " exceeds cap 8";
        return false;
    }
    return true;
}

int run_live(const cne::e2e::Config& cfg, const std::string& model_path,
             const std::string& project_src) {
    const std::string python = find_gateway_python(project_src);
    const std::string pythonpath = project_src + "/gateway";
    if (python.empty() || !gateway_import_ok(python, pythonpath)) {
        fprintf(stderr,
                "skip: server_gateway_policy live (install gateway: "
                "cd gateway && python -m venv .venv && pip install -r "
                "requirements.txt)\n");
        return 0;
    }

    const int server_port =
        cfg.port > 0 ? cfg.port : 21000 + (int) (getpid() % 2000);
    const int gateway_port =
        cfg.gateway.port > 0 ? cfg.gateway.port : server_port + 1;

    std::unordered_map<std::string, std::string> gw_env = cfg.gateway.env;
    gw_env["CNE_UPSTREAM"] = "http://127.0.0.1:" + std::to_string(server_port);
    if (!cfg.gateway.api_keys_file.empty())
        gw_env["CNE_GATEWAY_API_KEYS_FILE"] = cfg.gateway.api_keys_file;

    ChildProcess cne;
    if (!cne.start_server(model_path, server_port, cfg.env)) {
        fprintf(stderr, "FAIL: could not fork cne_server\n");
        return 1;
    }

    httplib::Client cne_cli("127.0.0.1", server_port);
    cne_cli.set_connection_timeout(30, 0);
    cne_cli.set_read_timeout(600, 0);
    if (!wait_cne_health(cne_cli, cfg.boot_timeout_s)) {
        fprintf(stderr, "FAIL: cne_server not healthy (log: %s)\n",
                cne.log_path().c_str());
        return 1;
    }

    ChildProcess gw;
    if (!gw.start_gateway(python, pythonpath, gateway_port, gw_env)) {
        fprintf(stderr, "FAIL: could not fork gateway\n");
        return 1;
    }

    httplib::Client gw_cli("127.0.0.1", gateway_port);
    gw_cli.set_connection_timeout(30, 0);
    gw_cli.set_read_timeout(600, 0);
    if (!wait_gateway_health(gw_cli, cfg.boot_timeout_s)) {
        fprintf(stderr, "FAIL: gateway not healthy (cne log: %s gw log: %s)\n",
                cne.log_path().c_str(), gw.log_path().c_str());
        return 1;
    }

    if (cfg.gateway.client_api_key.empty()) {
        fprintf(stderr, "FAIL: gateway.client_api_key not set in e2e config\n");
        return 1;
    }
    const httplib::Headers auth = {
        {"Authorization", "Bearer " + cfg.gateway.client_api_key},
        {"Content-Type", "application/json"},
    };

    std::string err;
    if (!check_health_policy(gw_cli, err)) {
        fprintf(stderr, "FAIL: health policy (%s)\n", err.c_str());
        return 1;
    }
    if (!check_thinking_blocked(gw_cli, auth, err)) {
        fprintf(stderr, "FAIL: thinking policy (%s)\n", err.c_str());
        return 1;
    }
    if (!check_max_tokens_clamped(gw_cli, auth, cfg.chat.chat_id + "-clamp",
                                  true, err)) {
        fprintf(stderr, "FAIL: max_tokens clamp (%s)\n", err.c_str());
        return 1;
    }
    if (!check_max_tokens_clamped(gw_cli, auth, cfg.chat.chat_id + "-default",
                                  false, err)) {
        fprintf(stderr, "FAIL: max_tokens default (%s)\n", err.c_str());
        return 1;
    }

    gw.stop();
    cne.stop();
    printf("server_gateway_policy live: OK (cne=%d gateway=%d)\n", server_port,
           gateway_port);
    return 0;
}

} // namespace

int main() {
    const std::string src = CNE_PROJECT_SOURCE_DIR;
    const std::string cfg_path =
        cne::e2e::discover_path("tests/e2e/server_gateway_policy_live.json", src);

    cne::e2e::Config cfg;
    std::string err;
    if (!cne::e2e::load(cfg_path, src, cfg, err)) {
        fprintf(stderr, "FAIL: e2e config %s (%s)\n", cfg_path.c_str(),
                err.c_str());
        return 1;
    }
    cne::e2e::apply_env(cfg);

    const std::string model = cne::e2e::resolve_model(cfg, src);
    if (model.empty()) {
        printf("skip: server_gateway_policy live (set model in %s or "
               "CNE_TEST_MODEL)\n",
               cfg_path.c_str());
        return 0;
    }
    if (access(model.c_str(), R_OK) != 0) {
        printf("skip: server_gateway_policy live (model not found: %s)\n",
               model.c_str());
        return 0;
    }
    if (cfg.gateway.api_keys_file.empty() ||
        access(cfg.gateway.api_keys_file.c_str(), R_OK) != 0) {
        printf("skip: server_gateway_policy live (gateway api_keys file missing: "
               "%s)\n",
               cfg.gateway.api_keys_file.c_str());
        return 0;
    }

    fprintf(stderr, "[e2e] config=%s model=%s\n", cfg_path.c_str(),
            model.c_str());
    return run_live(cfg, model, src);
}
