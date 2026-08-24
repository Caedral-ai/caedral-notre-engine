#pragma once
// Aligned direct-I/O reader for expert slices.
// Opens with O_DIRECT when the filesystem supports it; falls back to buffered
// pread otherwise (telemetry: direct()). All reads must be 4096-aligned in
// offset, length and destination - exactly what cne-prepare + window slices
// guarantee. Short reads fail closed.
#include <cstddef>
#include <cstdint>
#include <string>

namespace cne {

class DirectFile {
public:
    DirectFile() = default;
    ~DirectFile();
    DirectFile(const DirectFile &) = delete;
    DirectFile &operator=(const DirectFile &) = delete;

    // Tries O_RDONLY|O_DIRECT first; on EINVAL/ENOTSUP/EOPNOTSUPP retries
    // buffered and records direct()==false. Returns false when the file
    // cannot be opened at all.
    bool open_read(const std::string &path);
    void close();

    bool valid() const { return fd_ >= 0; }
    bool direct() const { return direct_; }

    // Full-read loop; all three of offset/bytes/dest must be 4096-aligned
    // when direct() (checked, fails closed). Buffered mode relaxes nothing
    // silently: misaligned calls are rejected there too - callers must not
    // rely on kernel copy paths they cannot see.
    bool read_aligned(void *dest, size_t bytes, uint64_t offset) const;

private:
    int fd_ = -1;
    bool direct_ = false;
};

} // namespace cne
