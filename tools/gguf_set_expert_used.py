#!/usr/bin/env python3
# gguf_set_expert_used.py -- patch <arch>.expert_used_count in a GGUF copy.
# Probe tool for self-spec drafting ceilings: the copy degrades output
# quality (fewer experts) but shows the SPEED CEILING of a K'-expert
# drafter. Never run on a serving artifact.
import struct, sys, shutil

SIZES = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}

def skip_value(buf, off, typ):
    if typ == 8:
        (slen,) = struct.unpack_from('<Q', buf, off)
        return off + 8 + slen
    if typ == 9:
        etyp = buf[off]
        (cnt,) = struct.unpack_from('<Q', buf, off + 4)
        off += 12
        if etyp == 8:
            for _ in range(cnt):
                (slen,) = struct.unpack_from('<Q', buf, off)
                off += 8 + slen
            return off
        return off + cnt * SIZES[etyp]
    return off + SIZES[typ]

def find_u32_kv(buf, key):
    kv_count = struct.unpack_from('<Q', buf, 8)[0]
    off = 24
    for _ in range(kv_count):
        (klen,) = struct.unpack_from('<Q', buf, off); off += 8
        kb = bytes(buf[off:off+klen]); off += klen
        (typ,) = struct.unpack_from('<I', buf, off); off += 4
        if kb.decode() == key:
            if typ != 4:
                raise RuntimeError(f'{key} has unexpected type {typ}')
            return off
        off = skip_value(buf, off, typ)
    return None

src, dst, new_k = sys.argv[1], sys.argv[2], int(sys.argv[3])
shutil.copyfile(src, dst)
with open(dst, 'r+b') as f:
    head = f.read(16 * 1024 * 1024)
    pos = find_u32_kv(head, 'lfm2moe.expert_used_count')
    if pos is None:
        sys.exit('key not found')
    old = struct.unpack_from('<I', head, pos)[0]
    f.seek(pos)
    f.write(struct.pack('<I', new_k))
    print(f'patched expert_used_count: {old} -> {new_k} at offset {pos}')
