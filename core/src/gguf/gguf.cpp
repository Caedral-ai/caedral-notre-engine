#include "soe/gguf.h"

#include <cinttypes>
#include <cstdio>
#include <cstring>

namespace soe {

namespace {

constexpr uint32_t kGgufMagic = 0x46554747; // "GGUF"
constexpr size_t kMaxNameLen = 1u << 20;

enum ValueType : uint32_t {
    V_UINT8 = 0,
    V_INT8 = 1,
    V_UINT16 = 2,
    V_INT16 = 3,
    V_UINT32 = 4,
    V_INT32 = 5,
    V_FLOAT32 = 6,
    V_BOOL = 7,
    V_STRING = 8,
    V_ARRAY = 9,
    V_UINT64 = 10,
    V_INT64 = 11,
    V_FLOAT64 = 12,
};

size_t scalar_size(uint32_t t) {
    switch (t) {
    case V_UINT8:
    case V_INT8:
    case V_BOOL:
        return 1;
    case V_UINT16:
    case V_INT16:
        return 2;
    case V_UINT32:
    case V_INT32:
    case V_FLOAT32:
        return 4;
    case V_UINT64:
    case V_INT64:
    case V_FLOAT64:
        return 8;
    default:
        return 0; // unknown/variable-length types are handled by callers
    }
}

uint64_t align_up(uint64_t v, uint64_t a) {
    return a ? (v + a - 1) / a * a : v;
}

}

GgufReader::~GgufReader() {
    if (f_)
        std::fclose(f_);
}

bool GgufReader::fail(const char *msg) {
    error_ = msg;
    if (f_) {
        std::fclose(f_);
        f_ = nullptr;
    }
    return false;
}

bool GgufReader::read_exact(void *dst, size_t n) {
    return std::fread(dst, 1, n, f_) == n;
}

bool GgufReader::read_string(std::string &out) {
    uint64_t len = 0;
    if (!read_exact(&len, 8))
        return fail("eof in string length");
    if (len > kMaxNameLen)
        return fail("string too long");
    out.resize(len);
    if (len && !read_exact(out.data(), (size_t)len))
        return fail("eof in string body");
    return true;
}

bool GgufReader::skip_value(uint32_t t) {
    if (t == V_STRING) {
        std::string s;
        return read_string(s);
    }
    if (t == V_ARRAY) {
        uint32_t elem = 0;
        uint64_t count = 0;
        if (!read_exact(&elem, 4) || !read_exact(&count, 8))
            return fail("eof in array header");
        for (uint64_t i = 0; i < count; i++)
            if (!skip_value(elem))
                return false;
        return true;
    }
    size_t sz = scalar_size(t);
    if (sz == 0)
        return fail("unknown kv value type");
    return std::fseek(f_, (long)sz, SEEK_CUR) == 0 || fail("seek failed");
}

bool GgufReader::parse_kvs(uint64_t count) {
    for (uint64_t i = 0; i < count; i++) {
        std::string key;
        uint32_t type = 0;
        if (!read_string(key))
            return fail("eof in kv key");
        if (!read_exact(&type, 4))
            return fail("eof in kv type");
        // Guard: a zero scalar size here would silently consume nothing and
        // desync the whole parse (fail closed instead).
        if (type != V_STRING && type != V_ARRAY && scalar_size(type) == 0)
            return fail("unknown kv value type");
        if (type == V_UINT8 || type == V_UINT16 || type == V_UINT32 || type == V_UINT64 || type == V_BOOL) {
            uint64_t v = 0;
            if (!read_exact(&v, scalar_size(type)))
                return fail("eof in kv uvalue");
            kv_u_[key] = v;
        } else if (type == V_INT8 || type == V_INT16 || type == V_INT32 || type == V_INT64) {
            int64_t v = 0;
            if (!read_exact(&v, scalar_size(type)))
                return fail("eof in kv ivalue");
            kv_i_[key] = v;
        } else if (type == V_FLOAT32 || type == V_FLOAT64) {
            double v = 0;
            float vf = 0;
            if (type == V_FLOAT32) {
                if (!read_exact(&vf, 4))
                    return fail("eof in kv fvalue");
                v = vf;
            } else if (!read_exact(&v, 8)) {
                return fail("eof in kv fvalue");
            }
            kv_f_[key] = v;
        } else if (type == V_STRING) {
            std::string s;
            if (!read_string(s))
                return false;
            kv_s_[key] = std::move(s);
        } else if (type == V_ARRAY) {
            if (!skip_value(V_ARRAY))
                return false;
        } else {
            return fail("unsupported kv type");
        }
    }
    return true;
}

bool GgufReader::parse_tensor_dir(uint64_t count) {
    tensors_.reserve((size_t)count);
    for (uint64_t i = 0; i < count; i++) {
        RawTensor t;
        uint32_t n_dims = 0;
        if (!read_string(t.name))
            return fail("eof in tensor name");
        if (!read_exact(&n_dims, 4))
            return fail("eof in tensor ndims");
        if (n_dims > 4)
            return fail("tensor ndims > 4");
        t.dims.resize(n_dims);
        for (uint32_t d = 0; d < n_dims; d++)
            if (!read_exact(&t.dims[d], 8))
                return fail("eof in tensor dim");
        if (!read_exact(&t.ggml_type, 4))
            return fail("eof in tensor type");
        if (!read_exact(&t.rel_offset, 8))
            return fail("eof in tensor offset");
        tensors_.push_back(std::move(t));
    }
    return true;
}

bool GgufReader::open(const std::string &path) {
    f_ = std::fopen(path.c_str(), "rb");
    if (!f_)
        return fail("cannot open file");

    uint32_t magic = 0;
    if (!read_exact(&magic, 4))
        return fail("eof at magic");
    if (magic != kGgufMagic)
        return fail("bad magic (not GGUF)");
    if (!read_exact(&version_, 4))
        return fail("eof at version");
    if (version_ < 2 || version_ > 3)
        return fail("unsupported gguf version");

    uint64_t n_tensors = 0, n_kv = 0;
    if (!read_exact(&n_tensors, 8) || !read_exact(&n_kv, 8))
        return fail("eof at counts");

    if (!parse_kvs(n_kv))
        return false;

    uint64_t alignment = 32;
    kv_u64("general.alignment", alignment);
    if (alignment == 0 || (alignment & (alignment - 1)))
        return fail("invalid alignment");
    alignment_ = (size_t)alignment;

    if (!parse_tensor_dir(n_tensors))
        return false;

    long pos = std::ftell(f_);
    if (pos < 0)
        return fail("ftell failed");
    data_offset_ = align_up((uint64_t)pos, alignment);
    error_.clear();
    return true;
}

bool GgufReader::kv_u64(const std::string &key, uint64_t &out) const {
    auto it = kv_u_.find(key);
    if (it == kv_u_.end())
        return false;
    out = it->second;
    return true;
}
bool GgufReader::kv_i64(const std::string &key, int64_t &out) const {
    auto it = kv_i_.find(key);
    if (it == kv_i_.end())
        return false;
    out = it->second;
    return true;
}
bool GgufReader::kv_f64(const std::string &key, double &out) const {
    auto it = kv_f_.find(key);
    if (it == kv_f_.end())
        return false;
    out = it->second;
    return true;
}
bool GgufReader::kv_str(const std::string &key, std::string &out) const {
    auto it = kv_s_.find(key);
    if (it == kv_s_.end())
        return false;
    out = it->second;
    return true;
}
bool GgufReader::kv_bool(const std::string &key, bool &out) const {
    auto it = kv_b_.find(key);
    if (it != kv_b_.end()) {
        out = it->second;
        return true;
    }
    auto iu = kv_u_.find(key);
    if (iu == kv_u_.end())
        return false;
    out = iu->second != 0;
    return true;
}
bool GgufReader::has_kv(const std::string &key) const {
    return kv_u_.count(key) || kv_i_.count(key) || kv_f_.count(key) || kv_s_.count(key) || kv_b_.count(key);
}

const char *GgufReader::ggml_type_name(uint32_t id) {
    static const char *names[] = {
        "f32",   "f16",    "q4_0",  "q4_1", "?4",   "?5",      "q5_0",   "q5_1",    "q8_0",  "q8_1",   "q2_k",
        "q3_k",  "q4_k",   "q5_k",  "q6_k", "q8_k", "iq2_xxs", "iq2_xs", "iq3_xxs", "iq1_s", "iq4_nl", "iq3_s",
        "iq2_s", "iq4_xs", "i8",    "i16",  "i32",  "i64",     "f64",    "iq1_m",   "bf16",  "?31",    "?32",
        "?33",   "tq1_0",  "tq2_0", "?36",  "?37",  "?38",     "mxfp4",  "nvfp4",   "q1_0",  "q2_0",
    };
    if (id < sizeof(names) / sizeof(names[0]))
        return names[id];
    return "unknown";
}

}
