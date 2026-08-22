// DirectFile: aligned reads on a temp file (O_DIRECT when the filesystem
// supports it, buffered fallback otherwise); misaligned calls rejected;
// short reads fail closed. SliceCache backend mode: fills via Source.
#include "soe/cache.h"
#include "soe/direct_io.h"

#include <sys/mman.h>

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <cstring>
#include <string>

using soe::CacheLimits;
using soe::DirectFile;
using soe::SliceCache;

namespace {
constexpr size_t B = 4096;

std::string tmp_path() { return "soe_test_direct_io.tmp"; }

bool write_pattern(const std::string &path, size_t blocks) {
    std::FILE *f = std::fopen(path.c_str(), "wb");
    if (!f)
        return false;
    std::string blk(B, '\0');
    for (size_t i = 0; i < blocks; i++) {
        std::memset(blk.data(), (int)(0xA0 + i % 16), blk.size());
        if (std::fwrite(blk.data(), 1, blk.size(), f) != blk.size()) {
            std::fclose(f);
            return false;
        }
    }
    // Force writeback: buffered-write -> O_DIRECT-read needs coherent storage,
    // fclose alone only pushes data as far as the page cache.
    bool ok = !std::fflush(f) && !fsync(fileno(f));
    std::fclose(f);
    return ok;
}
} // namespace

int main() {
    const size_t BLOCKS = 8;
    if (!write_pattern(tmp_path(), BLOCKS)) {
        fprintf(stderr, "cannot create temp file in cwd\n");
        return 1;
    }

    DirectFile df;
    if (!df.open_read(tmp_path())) {
        fprintf(stderr, "open failed\n");
        return 1;
    }
    printf("direct(): %s\n", df.direct() ? "yes" : "no (buffered fallback)");

    // aligned slice read spanning blocks 2..3
    auto *page = (char *)aligned_alloc(B, 3 * B);
    if (!df.read_aligned(page, 2 * B, 2 * B)) {
        fprintf(stderr, "aligned read failed\n");
        return 1;
    }
    if ((unsigned char)page[0] != 0xA2 || (unsigned char)page[B - 1] != 0xA2 ||
        (unsigned char)page[B] != 0xA3 || (unsigned char)page[2 * B - 1] != 0xA3) {
        fprintf(stderr, "pattern mismatch in aligned read\n");
        return 1;
    }

    // misaligned offset / length / dest must be rejected, not served wrong
    if (df.read_aligned(page + 1, 2 * B, 2 * B) ||
        df.read_aligned(page, 2 * B + 512, 2 * B) ||
        df.read_aligned(page, 2 * B, 2 * B + 7)) {
        fprintf(stderr, "misaligned read was accepted\n");
        return 1;
    }

    // out-of-range read fails closed (EOF is a corruption signal here)
    if (df.read_aligned(page, B, BLOCKS * B)) {
        fprintf(stderr, "eof read accepted\n");
        return 1;
    }
    std::free(page);

    // SliceCache backend mode: misses call the Source, hits do not.
    static int calls = 0;
    auto src = +[](void *dest, uint64_t off, size_t bytes, void *) -> bool {
        calls++;
        std::memset(dest, (int)(0x10 + off / B), bytes);
        return true;
    };
    void *win = mmap(nullptr, 8 * B, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    SliceCache c(CacheLimits{16 * B});
    c.set_source({src, nullptr});

    assert(!c.touch_at("T", 0, win, 0 * B, B));
    assert(c.touch_at("T", 0, win, 0 * B, B));      // hit: no new call
    assert(!c.touch_at("T", 1, (char *)win + B, 1 * B, B));
    assert(calls == 2);
    assert(((char *)win)[B] == 0x11);

    // batch offsets: experts {3,3,5} -> two fills at off 3B and 5B
    size_t misses = c.touch_batch_at("U", (const int[]){3, 3, 5}, 3, win, 0, B);
    assert(misses == 2);
    assert(calls == 4);
    assert(((char *)win)[5 * B] == 0x15);

    // backend failure fails closed inside fill (abort path); not exercised
    // here because it terminates the process - covered by code review.

    printf("ALL DIRECT IO TESTS PASSED\n");
    std::remove(tmp_path().c_str());
    return 0;
}
