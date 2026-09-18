"""Fixed signed feature hash with separate bucket/sign mixing streams.

This removes deterministic bucket-parity signs. Separate seeded mixing streams
are an engineering approximation, not a proof of independent hash families or
a Johnson-Lindenstrauss guarantee.
"""

SIGN_STREAM_OFFSET = 0xD1B54A35


def mix_u32(values, seed, *, xp):
    x = values.astype(xp.uint32) + xp.uint32(seed & 0xFFFFFFFF)
    x = (x ^ (x >> xp.uint32(16))) * xp.uint32(0x7FEB352D)
    x = (x ^ (x >> xp.uint32(15))) * xp.uint32(0x846CA68B)
    return x ^ (x >> xp.uint32(16))


def bucket_and_sign(coordinates, *, leaf_index, dimension, seed, xp):
    if dimension <= 0:
        raise ValueError("projection dimension must be positive")
    leaf_seed = seed + 0x9E3779B9 * (leaf_index + 1)
    bucket_hash = mix_u32(coordinates, leaf_seed, xp=xp)
    sign_hash = mix_u32(coordinates, leaf_seed + SIGN_STREAM_OFFSET, xp=xp)
    buckets = (bucket_hash % xp.uint32(dimension)).astype(xp.int32)
    signs = xp.where((sign_hash & xp.uint32(1)) == 0, 1.0, -1.0).astype(xp.float32)
    return buckets, signs
