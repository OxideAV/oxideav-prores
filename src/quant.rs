//! Quantisation matrices and scan tables for SMPTE RDD 36 ProRes.
//!
//! Per RDD 36 §7.3, when `load_luma_quantization_matrix` /
//! `load_chroma_quantization_matrix` are both 0 the default 8x8
//! all-4s matrix is used for both components — that is what `prores_ks`
//! emits for `apcn` (Standard) and `apch` (HQ) when no custom matrix is
//! requested. Tools that want non-flat matrices include them in the
//! frame header explicitly.
//!
//! `QSCALE_TABLE` (Table 15) maps `quantization_index` (1..=224) to the
//! quantisation scale factor (1..=512). Values 1..=128 are linear, then
//! `128 + 4 * (i - 128)` from index 129 up to 224.
//!
//! `BLOCK_SCAN_PROGRESSIVE` is the 8x8 forward block scan pattern for
//! progressive pictures (Figure 4); `BLOCK_SCAN_INTERLACED` is the
//! field-picture variant (Figure 5). Both store `scan[v][u]` packed in
//! row-major order, giving the scanned-coefficient index in 0..64 for
//! the natural-order position `(u, v)`.

/// Default per-block weight used by ProRes when no custom matrix is
/// loaded. Spec value: all-4s.
pub const DEFAULT_QMAT: [u8; 64] = [4u8; 64];

/// SMPTE RDD 36 progressive block scan pattern (Figure 4).
///
/// `scan[v][u]` (with the table laid out row-major as v*8+u) yields the
/// scanned-coefficient index in 0..64 for the natural-order position
/// `(u, v)`. To go from a scanned index back to natural order, use
/// [`PROGRESSIVE_INV_SCAN`].
pub const BLOCK_SCAN_PROGRESSIVE: [u8; 64] = [
    0, 1, 4, 5, 16, 17, 21, 22, // v=0
    2, 3, 6, 7, 18, 20, 23, 28, // v=1
    8, 9, 12, 13, 19, 24, 27, 29, // v=2
    10, 11, 14, 15, 25, 26, 30, 31, // v=3
    32, 33, 37, 38, 45, 46, 53, 54, // v=4
    34, 36, 39, 44, 47, 52, 55, 60, // v=5
    35, 40, 43, 48, 51, 56, 59, 61, // v=6
    41, 42, 49, 50, 57, 58, 62, 63, // v=7
];

/// SMPTE RDD 36 interlaced block scan pattern (Figure 5). Used when
/// `interlace_mode != 0` (field pictures).
pub const BLOCK_SCAN_INTERLACED: [u8; 64] = [
    0, 2, 8, 10, 32, 34, 35, 41, // v=0
    1, 3, 9, 11, 33, 36, 40, 42, // v=1
    4, 6, 12, 14, 37, 39, 43, 49, // v=2
    5, 7, 13, 15, 38, 44, 48, 50, // v=3
    16, 18, 19, 25, 45, 47, 51, 57, // v=4
    17, 20, 24, 26, 46, 52, 56, 58, // v=5
    21, 23, 27, 30, 53, 55, 59, 62, // v=6
    22, 28, 29, 31, 54, 60, 61, 63, // v=7
];

/// Inverse of [`BLOCK_SCAN_PROGRESSIVE`]: at index k (the scanned
/// position), the natural-order (row-major v*8+u) position.
pub const PROGRESSIVE_INV_SCAN: [u8; 64] = invert_scan(&BLOCK_SCAN_PROGRESSIVE);

/// Inverse of [`BLOCK_SCAN_INTERLACED`].
pub const INTERLACED_INV_SCAN: [u8; 64] = invert_scan(&BLOCK_SCAN_INTERLACED);

const fn invert_scan(scan: &[u8; 64]) -> [u8; 64] {
    let mut out = [0u8; 64];
    let mut i = 0;
    while i < 64 {
        out[scan[i] as usize] = i as u8;
        i += 1;
    }
    out
}

/// Quantisation scale factor `qScale` as a function of
/// `quantization_index` (RDD 36 Table 15). Index 0 is unused; valid
/// indices are 1..=224.
pub fn qscale(quantization_index: u8) -> i32 {
    let i = quantization_index as i32;
    if i <= 128 {
        i
    } else {
        128 + 4 * (i - 128)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn progressive_scan_is_permutation() {
        let mut seen = [false; 64];
        for &v in &BLOCK_SCAN_PROGRESSIVE {
            assert!(!seen[v as usize], "duplicate {v}");
            seen[v as usize] = true;
        }
        assert!(seen.iter().all(|&b| b));
    }

    #[test]
    fn interlaced_scan_is_permutation() {
        let mut seen = [false; 64];
        for &v in &BLOCK_SCAN_INTERLACED {
            assert!(!seen[v as usize], "duplicate {v}");
            seen[v as usize] = true;
        }
        assert!(seen.iter().all(|&b| b));
    }

    #[test]
    fn inverse_progressive_scan_roundtrips() {
        for k in 0..64 {
            let nat = PROGRESSIVE_INV_SCAN[k] as usize;
            assert_eq!(BLOCK_SCAN_PROGRESSIVE[nat], k as u8);
        }
    }

    #[test]
    fn qscale_table15_samples() {
        // Table 15 spot checks.
        assert_eq!(qscale(1), 1);
        assert_eq!(qscale(2), 2);
        assert_eq!(qscale(126), 126);
        assert_eq!(qscale(127), 127);
        assert_eq!(qscale(128), 128);
        assert_eq!(qscale(129), 132);
        assert_eq!(qscale(130), 136);
        assert_eq!(qscale(223), 508);
        assert_eq!(qscale(224), 512);
    }
}
