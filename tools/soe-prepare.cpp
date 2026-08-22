// soe-prepare: one-time GGUF normalization for the O_DIRECT path.
// Strategy: bump general.alignment to 4096 and repack offsets cumulatively in
// directory order - exactly the packing rule upstream llama.cpp enforces
// (gguf.cpp: offset[i] == sum of align_up(nbytes, alignment)). Result: every
// tensor starts at a 4096-aligned absolute offset, tensor bytes are copied
// verbatim, routed-expert spans do not move relative to their content
// (E10: slice sizes are already 4096 multiples), and the file stays a normal
// GGUF any loader accepts.
#include "soe/gguf.h"
#include "soe/model_registry.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include <unistd.h>

namespace {

constexpr uint64_t kAlign = 4096;
constexpr size_t kCopyBytes = 8u << 20;

std::string default_out(const std::string &in) {
    const std::string suffix = ".gguf";
    if (in.size() > suffix.size() && in.compare(in.size() - suffix.size(), suffix.size(), suffix) == 0)
        return in.substr(0, in.size() - suffix.size()) + "-prepared.gguf";
    return in + ".prepared.gguf";
}

uint64_t align_up(uint64_t v, uint64_t a) { return a ? (v + a - 1) / a * a : v; }

} // namespace

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <in.gguf> [out.gguf]\n", argv[0]);
        return 2;
    }
    const std::string in_path = argv[1];
    const std::string out_path = argc > 2 ? argv[2] : default_out(in_path);

    soe::ModelRegistry reg;
    soe::ModelManifest m;
    if (!reg.build(in_path, m)) {
        fprintf(stderr, "soe-prepare: manifest failed: %s\n", reg.error().c_str());
        return 1;
    }

    soe::GgufReader rd;
    if (!rd.open(in_path)) {
        fprintf(stderr, "soe-prepare: %s\n", rd.error().c_str());
        return 1;
    }
    const uint64_t meta_end = rd.meta_end();
    const uint64_t data_off = rd.data_offset();

    // Spans from sorted layout (same rule as the registry: next offset delta,
    // last tensor runs to end of file). Original spans are align_up(nbytes,
    // old_alignment); aligning those to 4096 == align_up(nbytes, 4096).
    std::vector<const soe::GgufReader::RawTensor *> sorted;
    for (const auto &t : rd.raw_tensors())
        sorted.push_back(&t);
    std::sort(sorted.begin(), sorted.end(),
              [](auto *a, auto *b) { return a->rel_offset < b->rel_offset; });
    std::vector<uint64_t> span(sorted.size());
    for (size_t i = 0; i < sorted.size(); i++) {
        uint64_t begin = data_off + sorted[i]->rel_offset;
        span[i] = (i + 1 < sorted.size()) ? data_off + sorted[i + 1]->rel_offset - begin
                                          : m.file_size - begin;
    }

    // Directory order must match ascending-offset order, or the cumulative
    // packing upstream enforces could not be satisfied. Fail closed otherwise.
    const size_t n = rd.raw_tensors().size();
    if (sorted.size() != n)
        return fprintf(stderr, "soe-prepare: internal tensor count mismatch\n"), 1;
    for (size_t i = 0; i < n; i++)
        if (sorted[i] != &rd.raw_tensors()[i])
            return fprintf(stderr, "soe-prepare: tensor dir not sorted by offset "
                                    "(entry %zu: %s) - cannot repack safely\n",
                           i, rd.raw_tensors()[i].name.c_str()), 1;

    // New layout: cumulative in directory order with 4096 padding.
    std::vector<uint64_t> new_rel(n);
    uint64_t cursor = 0;
    for (size_t i = 0; i < n; i++) {
        new_rel[i] = cursor;
        cursor += align_up(span[i], kAlign);
    }

    // Metadata: verbatim copy; ensure general.alignment=4096 exists (insert
    // the KV before the tensor dir when absent), then patch every rel_offset
    // field.
    std::FILE *src = std::fopen(in_path.c_str(), "rb");
    if (!src) {
        fprintf(stderr, "soe-prepare: cannot open %s\n", in_path.c_str());
        return 1;
    }
    std::vector<char> meta((size_t)meta_end);
    if (std::fread(meta.data(), 1, meta.size(), src) != meta.size()) {
        fprintf(stderr, "soe-prepare: short metadata read\n");
        return 1;
    }
    uint64_t kv_shift = 0;
    bool kv_appended = false;
    {
        const char align_key[] = "general.alignment";
        uint32_t al32 = (uint32_t)kAlign;
        uint64_t have_pos = 0;
        if (rd.kv_scalar_value_pos("general.alignment", have_pos)) {
            std::memcpy(meta.data() + have_pos, &al32, sizeof(al32));
        } else {
            // Insert KV BEFORE the tensor directory (GGUF order: KVs first):
            // [u64 klen][key][u32 type=UINT32][u32 value]; bump n_kv in the
            // header (magic u32 + version u32 + n_tensors u64 -> n_kv at
            // byte offset 16). Field positions at/after kv_end shift.
            uint64_t klen = sizeof(align_key) - 1;
            uint32_t vtype = 4; // UINT32
            std::vector<char> add(sizeof(klen) + klen + sizeof(vtype) + sizeof(al32));
            size_t o = 0;
            std::memcpy(add.data() + o, &klen, sizeof(klen)); o += sizeof(klen);
            std::memcpy(add.data() + o, align_key, klen);     o += klen;
            std::memcpy(add.data() + o, &vtype, sizeof(vtype)); o += sizeof(vtype);
            std::memcpy(add.data() + o, &al32, sizeof(al32));
            uint64_t n_kv = 0;
            std::memcpy(&n_kv, meta.data() + 16, sizeof(n_kv));
            n_kv += 1;
            std::memcpy(meta.data() + 16, &n_kv, sizeof(n_kv));
            meta.insert(meta.begin() + rd.kv_end(), add.begin(), add.end());
            kv_shift = add.size();
            kv_appended = true;
        }
    }
    for (size_t i = 0; i < n; i++) {
        uint64_t pos = rd.raw_tensors()[i].offset_field_pos;
        if (!kv_appended || pos >= rd.kv_end())
            pos += kv_shift;
        uint64_t v = new_rel[i];
        std::memcpy(meta.data() + pos, &v, sizeof(v));
    }
    const uint64_t new_data_off = align_up((uint64_t)meta.size(), kAlign);
    const uint64_t new_file_size = new_data_off + cursor;
    if (new_file_size < m.file_size)
        return fprintf(stderr, "soe-prepare: new layout smaller than source (%llu < %llu) - refusing\n",
                       (unsigned long long)new_file_size, (unsigned long long)m.file_size), 1;

    const std::filesystem::path out_fs(out_path);
    {
        auto dir = out_fs.parent_path();
        if (dir.empty())
            dir = ".";
        auto st = std::filesystem::space(dir);
        if (st.available < new_file_size) {
            fprintf(stderr, "soe-prepare: not enough disk space for %s (%.1f GiB needed, %.1f free)\n",
                    out_path.c_str(), new_file_size / 1073741824.0,
                    (double)st.available / 1073741824.0);
            return 1;
        }
    }

    std::FILE *dst = std::fopen(out_path.c_str(), "wb");
    if (!dst) {
        fprintf(stderr, "soe-prepare: cannot create %s\n", out_path.c_str());
        return 1;
    }
    if (std::fwrite(meta.data(), 1, meta.size(), dst) != meta.size()) {
        fprintf(stderr, "soe-prepare: metadata write failed\n");
        return 1;
    }
    if (fseeko(dst, (off_t)new_data_off, SEEK_SET) != 0) {
        fprintf(stderr, "soe-prepare: seek to data section failed\n");
        return 1;
    }

    std::vector<char> buf(kCopyBytes);
    uint64_t copied = 0, next_report = 4ull << 30;
    for (size_t i = 0; i < n; i++) {
        const uint64_t src_abs = data_off + sorted[i]->rel_offset;
        const uint64_t dst_abs = new_data_off + new_rel[i];
        if (fseeko(dst, (off_t)dst_abs, SEEK_SET) != 0 ||
            fseeko(src, (off_t)src_abs, SEEK_SET) != 0) {
            fprintf(stderr, "soe-prepare: seek failed at tensor %zu\n", i);
            return 1;
        }
        uint64_t left = span[i];
        while (left) {
            size_t chunk = (size_t)(left < buf.size() ? left : buf.size());
            if (std::fread(buf.data(), 1, chunk, src) != chunk ||
                std::fwrite(buf.data(), 1, chunk, dst) != chunk) {
                fprintf(stderr, "soe-prepare: copy failure in tensor %s\n",
                        sorted[i]->name.c_str());
                return 1;
            }
            left -= chunk;
            copied += chunk;
            if (copied >= next_report) {
                printf("soe-prepare: %.1f / %.1f GiB (%d%%)\n", copied / 1073741824.0,
                       m.file_size / 1073741824.0, (int)(100 * copied / m.file_size));
                fflush(stdout);
                next_report += 4ull << 30;
            }
        }
    }
    if (std::fflush(dst) != 0 || fsync(fileno(dst)) != 0) {
        fprintf(stderr, "soe-prepare: flush/fsync failed\n");
        return 1;
    }
    std::fclose(dst);
    std::fclose(src);

    // Validation: registry on the prepared file must report every routed
    // slice O_DIRECT-aligned, same counts as the source.
    soe::ModelRegistry vreg;
    soe::ModelManifest vm;
    if (!vreg.build(out_path, vm)) {
        fprintf(stderr, "soe-prepare: VALIDATION FAILED: %s\n", vreg.error().c_str());
        return 1;
    }
    bool ok = vm.routed_expert_tensors == m.routed_expert_tensors &&
              vm.misaligned_for_odirect == 0 && vm.uniform_misalignment == 0 &&
              vm.scattered_alignment == 0 &&
              vm.all_slices_aligned == vm.routed_expert_tensors &&
              vm.io_alignment == kAlign;
    printf("soe-prepare: wrote %s (%.2f GiB)\n", out_path.c_str(),
           new_file_size / 1073741824.0);
    printf("validation: io_alignment=%zu routed=%zu all_aligned=%zu misaligned=%zu uniform=%zu scattered=%zu -> %s\n",
           vm.io_alignment, vm.routed_expert_tensors, vm.all_slices_aligned,
           vm.misaligned_for_odirect, vm.uniform_misalignment, vm.scattered_alignment,
           ok ? "OK" : "FAILED");
    return ok ? 0 : 1;
}
