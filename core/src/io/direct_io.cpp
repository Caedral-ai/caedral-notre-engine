#include "soe/direct_io.h"

#include <cerrno>
#include <cstdio>
#include <cstring>

#include <fcntl.h>
#include <unistd.h>

namespace soe {

namespace {
constexpr size_t kAlign = 4096;

bool aligned(uintptr_t v) { return v % kAlign == 0; }
} // namespace

DirectFile::~DirectFile() { close(); }

void DirectFile::close() {
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
}

bool DirectFile::open_read(const std::string &path) {
    close();
#ifdef O_DIRECT
    fd_ = ::open(path.c_str(), O_RDONLY | O_DIRECT | O_CLOEXEC);
    if (fd_ >= 0) {
        direct_ = true;
        return true;
    }
    if (errno != EINVAL && errno != ENOTSUP && errno != EOPNOTSUPP)
        return false;
#endif
    fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd_ >= 0) {
        direct_ = false;
        fprintf(stderr, "[direct-io] O_DIRECT unavailable on %s - buffered fallback\n",
                path.c_str());
        return true;
    }
    return false;
}

bool DirectFile::read_aligned(void *dest, size_t bytes, uint64_t offset) const {
    if (fd_ < 0 || !dest)
        return false;
    const uintptr_t u = reinterpret_cast<uintptr_t>(dest);
    if (!aligned(offset) || !aligned(bytes) || !aligned(u)) {
        fprintf(stderr,
                "[direct-io] misaligned read rejected: off=%llu bytes=%zu dest=%p\n",
                (unsigned long long)offset, bytes, dest);
        return false;
    }
    size_t done = 0;
    while (done < bytes) {
        ssize_t n = ::pread(fd_, (char *)dest + done, bytes - done,
                            (off_t)(offset + done));
        if (n < 0) {
            if (errno == EINTR)
                continue;
            fprintf(stderr, "[direct-io] pread failed: %s\n", strerror(errno));
            return false;
        }
        if (n == 0) { // EOF: short read is a corruption signal, never tolerated
            fprintf(stderr, "[direct-io] short read at +%zu/%zu (offset %llu)\n", done,
                    bytes, (unsigned long long)offset);
            return false;
        }
        done += (size_t)n;
    }
    return true;
}

} // namespace soe
