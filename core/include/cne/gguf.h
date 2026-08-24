#pragma once
// Minimal metadata-only GGUF v2/v3 reader.
// Reads header + KV + tensor directory; never touches tensor data.
// Expert byte spans are derived from file layout (offset deltas), so no
// quant-type size tables are embedded here (metadata-driven).
#include "cne/model.h"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace cne {

class GgufReader {
public:
    GgufReader() = default;
    ~GgufReader();
    GgufReader(const GgufReader &) = delete;
    GgufReader &operator=(const GgufReader &) = delete;

    // Parses header, KVs and tensor directory. Returns false on any
    // structural violation; error() explains.
    bool open(const std::string &path);
    const std::string &error() const { return error_; }

    uint32_t version() const { return version_; }
    uint64_t data_offset() const { return data_offset_; }
    size_t io_alignment() const { return alignment_; }
    // File offset where the metadata section ends (tensor dir included);
    // data_offset() is its alignment-padded value.
    uint64_t meta_end() const { return meta_end_; }
    // File offset where the KV section ends / tensor directory begins.
    uint64_t kv_end() const { return kv_end_; }

    // Scalar/string KV access. Arrays are skipped during parse (presence
    // only). Returns false when the key is absent or of incompatible type.
    bool kv_u64(const std::string &key, uint64_t &out) const;
    bool kv_i64(const std::string &key, int64_t &out) const;
    bool kv_f64(const std::string &key, double &out) const;
    bool kv_str(const std::string &key, std::string &out) const;
    bool kv_bool(const std::string &key, bool &out) const;
    bool has_kv(const std::string &key) const;
    // File offset of a scalar KV's value field (for in-place metadata rewriters);
    // false when absent or non-scalar.
    bool kv_scalar_value_pos(const std::string &key, uint64_t &out) const;

    struct RawTensor {
        std::string name;
        std::vector<uint64_t> dims;
        uint32_t ggml_type = 0;
        uint64_t rel_offset = 0; // relative to data section
        // Absolute file offset of the u64 rel_offset field inside the tensor
        // directory (lets layout rewriters patch offsets in place).
        uint64_t offset_field_pos = 0;
    };
    const std::vector<RawTensor> &raw_tensors() const { return tensors_; }

    static const char *ggml_type_name(uint32_t id);

private:
    bool fail(const char *msg);
    bool read_exact(void *dst, size_t n);
    bool read_string(std::string &out);
    bool skip_value(uint32_t type);
    bool parse_kvs(uint64_t count);
    bool parse_tensor_dir(uint64_t count);

    std::FILE *f_ = nullptr;
    std::string error_;
    uint32_t version_ = 0;
    uint64_t data_offset_ = 0;
    uint64_t meta_end_ = 0;
    uint64_t kv_end_ = 0;
    size_t alignment_ = 32;

    std::map<std::string, uint64_t> kv_u_;
    std::map<std::string, int64_t> kv_i_;
    std::map<std::string, double> kv_f_;
    std::map<std::string, std::string> kv_s_;
    std::map<std::string, bool> kv_b_;
    std::map<std::string, uint64_t> kv_pos_; // scalar KV value file positions
    std::vector<RawTensor> tensors_;
};

}
