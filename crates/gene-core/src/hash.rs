//! A small, release-stable hash. Unlike `std::hash::DefaultHasher`, FNV-1a's
//! output is fixed across Rust versions, so content hashes (dataset versions,
//! dedup keys) stay comparable over time.

/// FNV-1a (64-bit) over `bytes`.
pub(crate) fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stable_and_distinct() {
        assert_eq!(fnv1a(b""), 0xcbf2_9ce4_8422_2325);
        assert_ne!(fnv1a(b"a"), fnv1a(b"b"));
        assert_eq!(fnv1a(b"gene"), fnv1a(b"gene"));
    }
}
