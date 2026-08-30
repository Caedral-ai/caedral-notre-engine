// HTTP E2E against a live cne_server process. Config: tests/e2e/server_e2e_live.json
// (override with CNE_E2E_CONFIG). Model: json "model" or CNE_TEST_MODEL.
#include "e2e_config.h"
#include <httplib.h>
#include <nlohmann/json.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <thread>

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

std::string read_file(const std::string& path) {
    std::ifstream in(path);
    return std::string((std::istreambuf_iterator<char>(in)),
                       std::istreambuf_iterator<char>());
}

bool log_contains(const std::string& path, const char* needle) {
    return read_file(path).find(needle) != std::string::npos;
}

class ServerProcess {
public:
    bool start(const std::string& model, int port) {
        log_path_ = "/tmp/cne_server_e2e_" + std::to_string(getpid()) + ".log";

        pid_ = fork();
        if (pid_ < 0) return false;
        if (pid_ == 0) {
            FILE* log = freopen(log_path_.c_str(), "w", stderr);
            (void) log;
            const std::string host = "127.0.0.1";
            const std::string port_s = std::to_string(port);
            execl(CNE_SERVER_EXE, "cne_server", model.c_str(), host.c_str(),
                  port_s.c_str(), (char*) nullptr);
            _exit(127);
        }
        return true;
    }

    void stop() {
        if (pid_ <= 0) return;
        kill(pid_, SIGTERM);
        int st = 0;
        for (int i = 0; i < 30; i++) {
            if (waitpid(pid_, &st, WNOHANG) == pid_) {
                pid_ = -1;
                break;
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
                    if (j.value("status", "") == "ok") return true;
                } catch (...) {}
            }
        }
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    return false;
}

json chat_body(const std::string& model, const json& messages,
               const std::string& conv_id, int max_tokens) {
    json body = {{"model", model},
                 {"messages", messages},
                 {"max_tokens", max_tokens},
                 {"temperature", 0},
                 {"stream", false}};
    if (!conv_id.empty()) body["conversation_id"] = conv_id;
    return body;
}

bool post_chat(httplib::Client& cli, const json& body, std::string& content,
               std::string& err) {
    auto res = cli.Post("/v1/chat/completions", body.dump(), "application/json");
    if (!res) {
        err = "no HTTP response";
        return false;
    }
    if (res->status != 200) {
        err = "HTTP " + std::to_string(res->status) + ": " + res->body;
        return false;
    }
    try {
        json j = json::parse(res->body);
        content = j["choices"][0]["message"]["content"].get<std::string>();
        if (content.empty()) {
            err = "empty assistant content";
            return false;
        }
        return true;
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
}

int run_live(const cne::e2e::Config& cfg, const std::string& model_path) {
    const int port =
        cfg.port > 0 ? cfg.port : 18000 + (int) (getpid() % 2000);

    ServerProcess srv;
    if (!srv.start(model_path, port)) {
        fprintf(stderr, "FAIL: could not fork cne_server\n");
        return 1;
    }

    httplib::Client cli("127.0.0.1", port);
    cli.set_connection_timeout(30, 0);
    cli.set_read_timeout(600, 0);
    cli.set_write_timeout(30, 0);

    if (!wait_health(cli, cfg.boot_timeout_s)) {
        fprintf(stderr, "FAIL: server not healthy after %ds (log: %s)\n",
                cfg.boot_timeout_s, srv.log_path().c_str());
        return 1;
    }

    auto health = cli.Get("/health");
    if (!health || health->status != 200) {
        fprintf(stderr, "FAIL: /health\n");
        return 1;
    }
    json h = json::parse(health->body);
    if (!h["sessions"]["enabled"].get<bool>()) {
        fprintf(stderr, "FAIL: sessions not enabled\n");
        return 1;
    }

    auto models = cli.Get("/v1/models");
    if (!models || models->status != 200) {
        fprintf(stderr, "FAIL: /v1/models\n");
        return 1;
    }
    const std::string model_id =
        json::parse(models->body)["data"][0]["id"].get<std::string>();

    const std::string conv = "e2e-lfm2-session";
    std::string content1, err;
    const json turn1 = chat_body(
        model_id,
        json::array({ {{"role", "user"},
                       {"content", "The capital of France is"}} }),
        conv, 12);
    if (!post_chat(cli, turn1, content1, err)) {
        fprintf(stderr, "FAIL: turn-1 chat (%s)\n", err.c_str());
        return 1;
    }

    std::string content2;
    const json turn2 = chat_body(
        model_id,
        json::array({ {{"role", "user"},
                       {"content", "The capital of France is"}},
                      {{"role", "assistant"}, {"content", content1}},
                      {{"role", "user"},
                       {"content", "Name one river in that city."}} }),
        conv, 12);
    if (!post_chat(cli, turn2, content2, err)) {
        fprintf(stderr, "FAIL: turn-2 chat (%s)\n", err.c_str());
        return 1;
    }

    if (!log_contains(srv.log_path(), "reused=")) {
        fprintf(stderr,
                "FAIL: turn-2 did not log session KV reuse (see %s)\n",
                srv.log_path().c_str());
        return 1;
    }

    srv.stop();
    printf("server_e2e live: OK (model=%s port=%d)\n", model_id.c_str(), port);
    return 0;
}

} // namespace

int main() {
    const std::string src = CNE_PROJECT_SOURCE_DIR;
    const std::string cfg_path =
        cne::e2e::discover_path("tests/e2e/server_e2e_live.json", src);

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
        printf("skip: server_e2e live (set model in %s or CNE_TEST_MODEL)\n",
               cfg_path.c_str());
        return 0;
    }
    if (access(model.c_str(), R_OK) != 0) {
        printf("skip: server_e2e live (model not found: %s)\n", model.c_str());
        return 0;
    }

    fprintf(stderr, "[e2e] config=%s model=%s\n", cfg_path.c_str(),
            model.c_str());
    return run_live(cfg, model);
}
