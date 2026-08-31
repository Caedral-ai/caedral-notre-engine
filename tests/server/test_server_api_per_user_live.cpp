// HTTP E2E: session_max_per_user eviction in API mode.
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
#include <sys/stat.h>
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
            log_path_ =
                "/tmp/cne_server_api_per_user_" + std::to_string(getpid()) + ".log";
            FILE* log = freopen(log_path_.c_str(), "w", stderr);
            (void) log;
            for (const auto& kv : env) setenv(kv.first.c_str(), kv.second.c_str(), 1);
            const std::string host = "127.0.0.1";
            const std::string port_s = std::to_string(port);
            execl(CNE_SERVER_EXE, "cne_server", model.c_str(), host.c_str(),
                  port_s.c_str(), (char*) nullptr);
            _exit(127);
        }
        log_path_ = "/tmp/cne_server_api_per_user_" + std::to_string(pid_) + ".log";
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

std::string read_log(const std::string& path) {
    std::ifstream in(path);
    return std::string((std::istreambuf_iterator<char>(in)),
                       std::istreambuf_iterator<char>());
}

bool log_has_fatal_error(const std::string& log) {
    static const char* needles[] = {
        "manifest FAILED",
        "gguf: eof",
        "cannot read API_KEYS_FILE",
        "is not valid JSON",
        "no 'model'",
    };
    for (const char* n : needles) {
        if (log.find(n) != std::string::npos) return true;
    }
    return false;
}

bool wait_health(httplib::Client& cli, const std::string& log_path,
                 int timeout_s) {
    const int step_ms = 500;
    const int steps   = (timeout_s * 1000) / step_ms;
    for (int i = 0; i < steps; i++) {
        if (auto res = cli.Get("/health")) {
            if (res->status == 200) {
                try {
                    json j = json::parse(res->body);
                    if (j.value("status", "") == "ok" &&
                        j["api"]["enabled"].get<bool>()) {
                        fprintf(stderr, "[e2e] server healthy after %.1fs\n",
                                (i * step_ms) / 1000.0);
                        return true;
                    }
                } catch (...) {}
            }
        }
        const std::string log = read_log(log_path);
        if (log_has_fatal_error(log)) {
            fprintf(stderr, "FAIL: server log shows boot error (see %s)\n",
                    log_path.c_str());
            return false;
        }
        if (i > 0 && i % 20 == 0) {
            fprintf(stderr, "[e2e] waiting for health... %.0fs (log %s)\n",
                    (i * step_ms) / 1000.0, log_path.c_str());
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(step_ms));
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

json ping_body(const std::string& chat_id, int max_tokens, bool think_off) {
    json body = { {"messages", json::array({ {{"role", "user"},
                                             {"content", "Say hi."}} })},
                  {"chat_id", chat_id},
                  {"max_tokens", max_tokens},
                  {"temperature", 0},
                  {"stream", false} };
    if (think_off)
        body["chat_template_kwargs"] = {{"enable_thinking", false}};
    return body;
}

int run_live(const cne::e2e::Config& cfg, const std::string& model_path) {
    const int port =
        cfg.port > 0 ? cfg.port : 19500 + (int) (getpid() % 2000);
    const int max_tok = cfg.chat.max_tokens > 0 ? cfg.chat.max_tokens : 4;

    fprintf(stderr, "[e2e] starting cne_server on port %d\n", port);
    ServerProcess srv;
    if (!srv.start(model_path, port, cfg.env)) {
        fprintf(stderr, "FAIL: could not fork cne_server\n");
        return 1;
    }

    httplib::Client cli("127.0.0.1", port);
    cli.set_connection_timeout(10, 0);
    cli.set_read_timeout(120, 0);
    cli.set_write_timeout(30, 0);

    if (!wait_health(cli, srv.log_path(), cfg.boot_timeout_s)) {
        fprintf(stderr, "FAIL: API server not healthy within %ds (log: %s)\n",
                cfg.boot_timeout_s, srv.log_path().c_str());
        return 1;
    }

    std::string err;
    const httplib::Headers alice = auth_headers("alice");

    fprintf(stderr, "[e2e] alice chat-a\n");
    if (!post_chat(cli, alice, ping_body("chat-a", max_tok, cfg.chat.think_off),
                   200, err)) {
        fprintf(stderr, "FAIL: alice chat-a (%s)\n", err.c_str());
        return 1;
    }
    fprintf(stderr, "[e2e] alice chat-b\n");
    if (!post_chat(cli, alice, ping_body("chat-b", max_tok, cfg.chat.think_off),
                   200, err)) {
        fprintf(stderr, "FAIL: alice chat-b (%s)\n", err.c_str());
        return 1;
    }
    fprintf(stderr, "[e2e] alice chat-c (expect per-user eviction)\n");
    if (!post_chat(cli, alice, ping_body("chat-c", max_tok, cfg.chat.think_off),
                   200, err)) {
        fprintf(stderr, "FAIL: alice chat-c (%s)\n", err.c_str());
        return 1;
    }

    const std::string log_mid = read_log(srv.log_path());
    if (log_mid.find("evicted conv=alice:chat-a owner=alice") ==
            std::string::npos ||
        log_mid.find("(per-user LRU)") == std::string::npos) {
        fprintf(stderr,
                "FAIL: expected per-user eviction of alice:chat-a (see %s)\n",
                srv.log_path().c_str());
        return 1;
    }

    fprintf(stderr, "[e2e] bob chat-x\n");
    if (!post_chat(cli, auth_headers("bob"), ping_body("chat-x", max_tok,
                                                       cfg.chat.think_off),
                   200, err)) {
        fprintf(stderr, "FAIL: bob chat-x (%s)\n", err.c_str());
        return 1;
    }

    const std::string log_final = read_log(srv.log_path());
    if (log_final.find("evicted conv=bob:chat-x") != std::string::npos ||
        log_final.find("evicted conv=bob:chat-x owner=bob") !=
            std::string::npos) {
        fprintf(stderr, "FAIL: bob session wrongly evicted (see %s)\n",
                srv.log_path().c_str());
        return 1;
    }

    srv.stop();
    printf("server_api_per_user live: OK (port=%d)\n", port);
    return 0;
}

bool is_regular_file(const std::string& path) {
    struct stat st {};
    return stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

} // namespace

int main() {
    const std::string src = CNE_PROJECT_SOURCE_DIR;
    const std::string cfg_path = cne::e2e::discover_path(
        "tests/e2e/server_api_per_user_live.json", src);

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
        printf("skip: server_api_per_user live (set model in %s or "
               "CNE_TEST_MODEL)\n",
               cfg_path.c_str());
        return 0;
    }
    if (!is_regular_file(model)) {
        fprintf(stderr, "FAIL: model is not a readable file: %s\n",
                model.c_str());
        return 1;
    }

    fprintf(stderr, "[e2e] config=%s model=%s\n", cfg_path.c_str(),
            model.c_str());
    return run_live(cfg, model);
}
