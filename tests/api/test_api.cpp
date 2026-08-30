#include "cne_api.h"

#include <cassert>
#include <cstdio>
#include <unordered_map>

int main() {
    using namespace cne::api;

    assert(make_conversation_id("alice", "7") == "alice:7");
    assert(conversation_owned_by("alice:7", "alice"));
    assert(!conversation_owned_by("alice:7", "bob"));
    assert(!conversation_owned_by("alice", "alice"));

    Settings s;
    s.enabled = true;
    std::unordered_set<std::string> keys = {"secret"};
    std::unordered_map<std::string, std::string> key_to_user;

    std::unordered_map<std::string, std::string> headers = {
        {"Authorization", "Bearer secret"},
        {"X-User-Id", "user-1"},
    };
    AuthResult ok = authenticate(headers, s, keys, key_to_user);
    assert(ok.ok);
    assert(ok.user_id == "user-1");

    headers["Authorization"] = "Bearer wrong";
    AuthResult bad = authenticate(headers, s, keys, key_to_user);
    assert(!bad.ok);
    assert(bad.http_status == 401);

    headers["Authorization"] = "Bearer secret";
    headers["X-User-Id"]     = "";
    AuthResult no_user = authenticate(headers, s, keys, key_to_user);
    assert(!no_user.ok);
    assert(no_user.http_status == 400);

    key_to_user["secret"] = "mapped-user";
    headers["X-User-Id"]  = "";
    AuthResult mapped = authenticate(headers, s, keys, key_to_user);
    assert(mapped.ok);
    assert(mapped.user_id == "mapped-user");

    RateLimiter lim(2);
    assert(lim.allow("u"));
    assert(lim.allow("u"));
    assert(!lim.allow("u"));
    assert(lim.allow("v"));

    printf("api_unit: OK\n");
    return 0;
}
