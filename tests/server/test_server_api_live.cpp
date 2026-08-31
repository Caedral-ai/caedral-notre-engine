// HTTP E2E: multi-tenant API mode (auth + chat_id). Config: server_api_live.json
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

class ServerProcess {
public:
    bool start(const std::string& model, int port,
               const std::unordered_map<std::string, std::string>& env) {
        pid_ = fork();
        if (pid_ < 0) return false;
        if (pid_ == 0) {
            log_path_ = "/tmp/cne_server_api_" + std::to_string(getpid()) + ".log";
            FILE* log = freopen(log_path_.c_str(), "w", stderr);
            (void) log;
            for (const auto& kv : env) setenv(kv.first.c_str(), kv.second.c_str(), 1);
            const std::string host = "127.0.0.1";
            const std::string port_s = std::to_string(port);
            execl(CNE_SERVER_EXE, "cne_server", model.c_str(), host.c_str(),
                  port_s.c_str(), (char*) nullptr);
            _exit(127);
        }
        log_path_ = "/tmp/cne_server_api_" + std::to_string(pid_) + ".log";
        return true;
    }

    void stop() {
        if (pid_ <= 0) return;
        kill(pid_, SIGTERM);
        int st = 0;
        for (int i = 0; i < 30; i++) {
            if (waitpid(pid_, &st, WNOHANG) == pid_) {
                pid_ = -1;
                return;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
        if (pid_ > 0) {
            kill(pid_, SIGKILL);
            waitpid(pid_, &st, 0);
            pid_ = -1;
        }
    }

    const std::string& log_path() const { return log_path_; }

    ~ServerProcess() { stop(); }

private:
    pid_t       pid_ = -1;
    std::string log_path_;
};

bool wait_health(httplib::Client& cli, int timeout_s) {
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

httplib::Headers auth_headers(const std::string& user) {
    return { {"Authorization", "Bearer e2e-test-secret"},
             {"X-User-Id", user},
             {"Content-Type", "application/json"} };
}

bool post_chat(httplib::Client& cli, const httplib::Headers& hdrs,
               const json& body, int expect_status, std::string& err) {
    auto res = cli.Post("/v1/chat/completions", hdrs, body.dump(),
                        "application/json");
    if (!res) {
        err = "no HTTP response";
        return false;
    }
    if (res->status != expect_status) {
        err = "HTTP " + std::to_string(res->status) + ": " + res->body;
        return false;
    }
    return true;
}

int run_live(const cne::e2e::Config& cfg, const std::string& model_path) {
    const int port =
        cfg.port > 0 ? cfg.port : 19000 + (int) (getpid() % 2000);
    const int max_tok = cfg.chat.max_tokens > 0 ? cfg.chat.max_tokens : 12;

    ServerProcess srv;
    if (!srv.start(model_path, port, cfg.env)) {
        fprintf(stderr, "FAIL: could not fork cne_server\n");
        return 1;
    }

    httplib::Client cli("127.0.0.1", port);
    cli.set_connection_timeout(30, 0);
    cli.set_read_timeout(600, 0);
    cli.set_write_timeout(30, 0);

    if (!wait_health(cli, cfg.boot_timeout_s)) {
        fprintf(stderr, "FAIL: API server not healthy (log: %s)\n",
                srv.log_path().c_str());
        return 1;
    }

    std::string err;
    json body = { {"messages", json::array({ {{"role", "user"},
                                             {"content", "Say hi."}} })},
                  {"max_tokens", max_tok},
                  {"temperature", 0},
                  {"stream", false} };

    if (post_chat(cli, {}, body, 401, err)) {
        fprintf(stderr, "FAIL: expected 401 without auth\n");
        return 1;
    }

    body["chat_id"] = "chat-1";
    if (!post_chat(cli, auth_headers("alice"), body, 200, err)) {
        fprintf(stderr, "FAIL: authenticated chat (%s)\n", err.c_str());
        return 1;
    }

    body["messages"] = json::array({
        {{"role", "user"}, {"content", "Say hi."}},
        {{"role", "assistant"}, {"content", "Hello."}},
        {{"role", "user"}, {"content", "What did I ask first?"}} });
    if (!post_chat(cli, auth_headers("alice"), body, 200, err)) {
        fprintf(stderr, "FAIL: turn-2 chat (%s)\n", err.c_str());
        return 1;
    }

    std::ifstream log(srv.log_path());
    const std::string log_text((std::istreambuf_iterator<char>(log)),
                               std::istreambuf_iterator<char>());
    if (log_text.find("reused=") == std::string::npos) {
        fprintf(stderr, "FAIL: turn-2 did not reuse KV (see %s)\n",
                srv.log_path().c_str());
        return 1;
    }

    body["messages"] = json::array({ {{"role", "user"}, {"content", "ping"}} });
    body["chat_id"]    = "chat-2";
    if (!post_chat(cli, auth_headers("alice"), body, 200, err)) return 1;
    body["chat_id"] = "chat-3";
    if (!post_chat(cli, auth_headers("alice"), body, 200, err)) return 1;
    body["chat_id"] = "bob-1";
    if (!post_chat(cli, auth_headers("bob"), body, 200, err)) return 1;

    std::ifstream log2(srv.log_path());
    const std::string log_final((std::istreambuf_iterator<char>(log2)),
                                std::istreambuf_iterator<char>());
    if (log_final.find("evicted conv=bob:") != std::string::npos ||
        log_final.find("evicted conv=bob-") != std::string::npos) {
        fprintf(stderr, "FAIL: bob session evicted by alice spam (see %s)\n",
                srv.log_path().c_str());
        return 1;
    }

    srv.stop();
    printf("server_api live: OK (port=%d)\n", port);
    return 0;
}

} // namespace

int main() {
    const std::string src = CNE_PROJECT_SOURCE_DIR;
    const std::string cfg_path =
        cne::e2e::discover_path("tests/e2e/server_api_live.json", src);

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
        printf("skip: server_api live (set model in %s or CNE_TEST_MODEL)\n",
               cfg_path.c_str());
        return 0;
    }
    if (access(model.c_str(), R_OK) != 0) {
        printf("skip: server_api live (model not found: %s)\n", model.c_str());
        return 0;
    }

    fprintf(stderr, "[e2e] config=%s model=%s\n", cfg_path.c_str(),
            model.c_str());
    return run_live(cfg, model);
}
