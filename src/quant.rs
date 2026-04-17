//! Quantisation matrices for ProRes 422 profiles.
//!
//! SMPTE RDD 36 table 7 defines per-profile luma / chroma quantisation
//! matrices. The HQ profile's matrices are all-4 (essentially
//! untouched), Standard uses the tables below, and Proxy/LT use the
//! same Standard matrices scaled up by a profile-wide quant index.
//!
//! For this minimal implementation we hard-code the **Standard**
//! profile's two matrices and expose a single `quant_index` that
//! scales them multiplicatively. That maps cleanly onto:
//!
//! * Proxy   — quant index around 8 (very coarse)
//! * LT      — quant index around 4-6
//! * Standard — quant index 2-4 (default = 4)
//!
//! Values are from RDD 36 Annex A, 8x8 natural (row-major) order.

/// Luma quantisation matrix for 422 Standard (SMPTE RDD 36 Annex A).
pub const LUMA_QMAT_STANDARD: [u8; 64] = [
    4, 4, 5, 5, 6, 7, 7, 9, //
    4, 4, 5, 6, 7, 7, 9, 9, //
    5, 5, 6, 7, 7, 9, 9, 10, //
    5, 5, 6, 7, 7, 9, 9, 10, //
    5, 6, 7, 7, 8, 9, 10, 12, //
    6, 7, 7, 8, 9, 10, 12, 15, //
    6, 7, 7, 9, 10, 11, 14, 17, //
    7, 7, 9, 10, 11, 14, 17, 21,
];

/// Chroma quantisation matrix for 422 Standard (same source).
pub const CHROMA_QMAT_STANDARD: [u8; 64] = [
    4, 4, 5, 5, 6, 7, 7, 9, //
    4, 4, 5, 6, 7, 7, 9, 9, //
    5, 5, 6, 7, 7, 9, 9, 10, //
    5, 5, 6, 7, 7, 9, 9, 10, //
    5, 6, 7, 7, 8, 9, 10, 12, //
    6, 7, 7, 8, 9, 10, 12, 15, //
    6, 7, 7, 9, 10, 11, 14, 17, //
    7, 7, 9, 10, 11, 14, 17, 21,
];

/// Luma quantisation matrix for 4444 Standard quality (SMPTE RDD 36 Annex B).
///
/// The 4444 profile targets higher quality than 422 Standard, so the
/// matrix is closer to flat (all small values). In the absence of the
/// verbatim RDD 36 bytes we use the conservative "HQ-like" all-4 matrix
/// which is valid (any matrix encoded into the frame header is honoured
/// by the decoder round-trip). If bit-exact RDD 36 compatibility with
/// third-party ProRes files becomes a goal, these values will need to
/// be updated.
pub const LUMA_QMAT_4444: [u8; 64] = [
    4, 4, 4, 4, 4, 4, 4, 4, //
    4, 4, 4, 4, 4, 4, 4, 4, //
    4, 4, 4, 4, 4, 4, 4, 4, //
    4, 4, 4, 4, 4, 4, 4, 4, //
    4, 4, 4, 4, 4, 4, 4, 4, //
    4, 4, 4, 4, 4, 4, 4, 4, //
    4, 4, 4, 4, 4, 4, 4, 4, //
    4, 4, 4, 4, 4, 4, 4, 4,
];

/// Chroma quantisation matrix for 4444 Standard quality.
///
/// Same caveat as [`LUMA_QMAT_4444`] — flat matrix chosen so that
/// encode/decode round-trip matches without a spec-verbatim table.
pub const CHROMA_QMAT_4444: [u8; 64] = [
    4, 4, 4, 4, 4, 4, 4, 4, //
    4, 4, 4, 4, 4, 4, 4, 4, //
    4, 4, 4, 4, 4, 4, 4, 4, //
    4, 4, 4, 4, 4, 4, 4, 4, //
    4, 4, 4, 4, 4, 4, 4, 4, //
    4, 4, 4, 4, 4, 4, 4, 4, //
    4, 4, 4, 4, 4, 4, 4, 4, //
    4, 4, 4, 4, 4, 4, 4, 4,
];

/// ProRes 8x8 zig-zag scan order (from top-left, diagonal sweep).
/// Natural-order index at zig-zag position `k` is `ZIGZAG[k]`.
pub const ZIGZAG: [u8; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, //
    17, 24, 32, 25, 18, 11, 4, 5, //
    12, 19, 26, 33, 40, 48, 41, 34, //
    27, 20, 13, 6, 7, 14, 21, 28, //
    35, 42, 49, 56, 57, 50, 43, 36, //
    29, 22, 15, 23, 30, 37, 44, 51, //
    58, 59, 52, 45, 38, 31, 39, 46, //
    53, 60, 61, 54, 47, 55, 62, 63,
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zigzag_is_permutation() {
        let mut seen = [false; 64];
        for &v in &ZIGZAG {
            assert!(!seen[v as usize]);
            seen[v as usize] = true;
        }
        assert!(seen.iter().all(|&b| b));
    }

    #[test]
    fn matrices_are_8x8() {
        assert_eq!(LUMA_QMAT_STANDARD.len(), 64);
        assert_eq!(CHROMA_QMAT_STANDARD.len(), 64);
        assert_eq!(LUMA_QMAT_4444.len(), 64);
        assert_eq!(CHROMA_QMAT_4444.len(), 64);
    }
}
