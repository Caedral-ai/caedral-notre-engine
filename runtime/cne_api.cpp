#include "cne_api.h"

#include "cne/config.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <vector>

namespace cne::api {

namespace {

std::string trim_ws(std::string s) {
    while (!s.empty() && std::isspace((unsigned char) s.front())) s.erase(s.begin());
    while (!s.empty() && std::isspace((unsigned char) s.back())) s.pop_back();
    return s;
}

std::string lower(std::string s) {
    for (char& c : s) c = (char) std::tolower((unsigned char) c);
    return s;
}

void split_csv(const std::string& csv, std::vector<std::string>& out) {
    out.clear();
    std::string item;
    std::istringstream ss(csv);
    while (std::getline(ss, item, ',')) {
        item = trim_ws(item);
        if (!item.empty()) out.push_back(item);
    }
}

bool parse_key_line(const std::string& line, std::string& key,
                    std::string& user) {
    key.clear();
    user.clear();
    std::string t = trim_ws(line);
    if (t.empty() || t[0] == '#') return false;
    const size_t sp = t.find_first_of(" \t");
    if (sp == std::string::npos) {
        key = t;
        return true;
    }
    key  = trim_ws(t.substr(0, sp));
    user = trim_ws(t.substr(sp + 1));
    return !key.empty();
}

} // namespace

Settings load_settings() {
    Settings s;
    if (cne::env("API_MODE")) {
        const char* v = cne::env("API_MODE");
        s.enabled = v[0] != '0';
    }
    if (const char* v = cne::env("API_RPM")) {
        const int n = atoi(v);
        if (n >= 0) s.rpm_per_user = n;
    }
    if (const char* v = cne::env("SESSION_MAX_PER_USER")) {
        const int n = atoi(v);
        if (n >= 0) s.session_max_per_user = (size_t) n;
    }
    if (cne::env("API_REQUIRE_CHAT_ID")) {
        const char* v = cne::env("API_REQUIRE_CHAT_ID");
        s.require_chat_id = v[0] != '0';
    }
    return s;
}

bool load_keys(std::unordered_set<std::string>& keys_out,
               std::unordered_map<std::string, std::string>& key_to_user_out) {
    keys_out.clear();
    key_to_user_out.clear();

    auto ingest = [&](const std::string& key, const std::string& user) {
        if (key.empty()) return;
        keys_out.insert(key);
        if (!user.empty()) key_to_user_out[key] = user;
    };

    if (const char* single = cne::env("API_KEY")) ingest(single, "");

    if (const char* csv = cne::env("API_KEYS")) {
        std::vector<std::string> items;
        split_csv(csv, items);
        for (const auto& item : items) {
            std::string key, user;
            if (parse_key_line(item, key, user)) ingest(key, user);
        }
    }

    const char* path = cne::env("API_KEYS_FILE");
    if (path && path[0]) {
        std::ifstream in(path);
        if (!in) {
            fprintf(stderr, "[api] cannot read API_KEYS_FILE %s\n", path);
            return false;
        }
        std::string line;
        while (std::getline(in, line)) {
            std::string key, user;
            if (parse_key_line(line, key, user)) ingest(key, user);
        }
    }

    return !keys_out.empty();
}

std::string header_get(
    const std::unordered_map<std::string, std::string>& headers,
    const char* name) {
    const std::string want = lower(name);
    for (const auto& kv : headers) {
        if (lower(kv.first) == want) return kv.second;
    }
    return {};
}

AuthResult authenticate(
    const std::unordered_map<std::string, std::string>& headers,
    const Settings& settings, const std::unordered_set<std::string>& keys,
    const std::unordered_map<std::string, std::string>& key_to_user) {
    AuthResult r;
    if (!settings.enabled) {
        r.ok = true;
        return r;
    }
    if (keys.empty()) {
        r.error       = "API mode enabled but no API keys configured";
        r.http_status = 503;
        return r;
    }

    std::string auth = header_get(headers, "Authorization");
    const std::string prefix = "Bearer ";
    if (auth.size() <= prefix.size() ||
        auth.compare(0, prefix.size(), prefix) != 0) {
        r.error = "missing or invalid Authorization header (Bearer token required)";
        return r;
    }
    const std::string token = trim_ws(auth.substr(prefix.size()));
    if (!keys.count(token)) {
        r.error = "invalid API key";
        return r;
    }

    std::string user = header_get(headers, "X-User-Id");
    user = trim_ws(user);
    auto mapped = key_to_user.find(token);
    if (mapped != key_to_user.end()) {
        if (!user.empty() && user != mapped->second) {
            r.error       = "X-User-Id does not match API key";
            r.http_status = 403;
            return r;
        }
        if (user.empty()) user = mapped->second;
    }
    if (user.empty()) {
        r.error       = "X-User-Id header required in API mode";
        r.http_status = 400;
        return r;
    }

    r.ok      = true;
    r.user_id = user;
    return r;
}

bool RateLimiter::allow(const std::string& user_id) {
    if (rpm_ <= 0 || user_id.empty()) return true;
    const auto now = std::chrono::steady_clock::now();
    const auto window = std::chrono::minutes(1);
    std::lock_guard<std::mutex> lock(mu_);
    auto& q = hits_[user_id];
    while (!q.empty() && now - q.front() > window) q.pop_front();
    if ((int) q.size() >= rpm_) return false;
    q.push_back(now);
    return true;
}

std::string make_conversation_id(const std::string& user_id,
                                 const std::string& chat_id) {
    return user_id + ":" + chat_id;
}

bool conversation_owned_by(const std::string& conversation_id,
                           const std::string& user_id) {
    if (user_id.empty()) return false;
    const std::string prefix = user_id + ":";
    return conversation_id.size() > prefix.size() &&
           conversation_id.compare(0, prefix.size(), prefix) == 0;
}

} // namespace cne::api
