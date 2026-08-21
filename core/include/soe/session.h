#pragma once
// Per-session state. Expert cache is shared per model; KV is session-owned
// and never shared between users. Plan ref: §7.4 / §18.
#include <cstdint>

namespace soe {

class Session {
public:
    explicit Session(uint64_t id) : id_(id) {}
    uint64_t id() const { return id_; }

private:
    uint64_t id_;
};

} // namespace soe
