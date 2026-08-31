#pragma once
// Multi-tenant API helpers: API-key auth, rate limits, conversation id rules.
#include <chrono>
#include <cstddef>
#include <deque>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace cne::api {

struct Settings {
    bool        enabled            = false;
    int         rpm_per_user       = 60;  // 0 = unlimited
    size_t      session_max_per_user = 2; // 0 = global LRU only
    bool        require_chat_id    = false; // when false, chat_id optional
};

struct AuthResult {
    bool        ok          = false;
    int         http_status = 401;
    std::string user_id;
    std::string error;
};

// Load from CNE_API_* env and optional server.json fields (see cne_api.cpp).
Settings load_settings();

// Parse CNE_API_KEYS (comma-separated) and/or CNE_API_KEYS_FILE (one key per line).
// Optional second column: key<ws>user_id
bool load_keys(std::unordered_set<std::string>& keys_out,
               std::unordered_map<std::string, std::string>& key_to_user_out);

AuthResult authenticate(const std::unordered_map<std::string, std::string>& headers,
                        const Settings& settings,
                        const std::unordered_set<std::string>& keys,
                        const std::unordered_map<std::string, std::string>& key_to_user);

class RateLimiter {
public:
    explicit RateLimiter(int rpm) : rpm_(rpm) {}

    // Returns false when the user exceeded rpm (if rpm > 0).
    bool allow(const std::string& user_id);

private:
    int rpm_;
    std::mutex mu_;
    std::unordered_map<std::string, std::deque<std::chrono::steady_clock::time_point>>
        hits_;
};

// Server-owned conversation id: "{user_id}:{chat_id}".
std::string make_conversation_id(const std::string& user_id,
                                 const std::string& chat_id);

bool conversation_owned_by(const std::string& conversation_id,
                           const std::string& user_id);

// Lowercase a header name for httplib-style lookup maps.
std::string header_get(const std::unordered_map<std::string, std::string>& headers,
                       const char* name);

} // namespace cne::api
