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

/// Reference perceptual luma quantisation matrix derived from
/// ISO/IEC 10918-1 Annex K Table K.1 (the standard JPEG luma
/// quantisation table), normalised so that the DC weight is 2 — i.e.
/// `clamp(round(K1[v][u] * 2 / 16), 2, 63)` — which gives the
/// encoder twice the DC/low-frequency precision of the spec's flat
/// all-4s default at the same `quantization_index` while letting
/// high-frequency weights climb (rolling off perceptually
/// less-significant detail).
///
/// The shape — small weights at low spatial frequency and large
/// weights toward the bottom-right of the 8x8 grid — exploits the
/// human visual system's reduced sensitivity to high-frequency detail
/// (CSF rolloff). Natural images have a near-1/f² power spectrum, so
/// most signal energy lives at low spatial frequency; preserving it
/// twice as accurately while quantising the high end more coarsely
/// is a strict R-D win on natural-image-like content (smaller
/// bitstream at matched PSNR, see the `perceptual_quant` integration
/// test).
///
/// Indexed in natural (row-major) order: `weight = PERCEPTUAL_LUMA_QMAT[v * 8 + u]`.
pub const PERCEPTUAL_LUMA_QMAT: [u8; 64] = [
    2, 2, 2, 2, 3, 5, 6, 8, // v=0
    2, 2, 2, 2, 3, 7, 8, 7, // v=1
    2, 2, 2, 3, 5, 7, 9, 7, // v=2
    2, 2, 3, 4, 6, 11, 10, 8, // v=3
    2, 3, 5, 7, 9, 14, 13, 10, // v=4
    3, 4, 7, 8, 10, 13, 14, 12, // v=5
    6, 8, 10, 11, 13, 15, 15, 13, // v=6
    9, 12, 12, 12, 14, 13, 13, 12, // v=7
];

/// Reference perceptual chroma quantisation matrix derived from
/// ISO/IEC 10918-1 Annex K Table K.2 (the standard JPEG chroma
/// quantisation table), normalised so that the DC weight is 2 — i.e.
/// `clamp(round(K2[v][u] * 2 / 16), 2, 63)`.
///
/// Chroma resolution sensitivity falls off faster than luma, so the
/// high-frequency weights saturate at a lower ceiling than the luma
/// matrix. Indexed in natural (row-major) order.
pub const PERCEPTUAL_CHROMA_QMAT: [u8; 64] = [
    2, 2, 3, 6, 12, 12, 12, 12, // v=0
    2, 3, 3, 8, 12, 12, 12, 12, // v=1
    3, 3, 7, 12, 12, 12, 12, 12, // v=2
    6, 8, 12, 12, 12, 12, 12, 12, // v=3
    12, 12, 12, 12, 12, 12, 12, 12, // v=4
    12, 12, 12, 12, 12, 12, 12, 12, // v=5
    12, 12, 12, 12, 12, 12, 12, 12, // v=6
    12, 12, 12, 12, 12, 12, 12, 12, // v=7
];

/// Per-component pair of 8x8 quantisation weight matrices for the
/// ProRes encoder.
///
/// Indexed in natural (row-major) order — entry `[v * 8 + u]` is the
/// weight `W[v][u]` from RDD 36 §7.3. Values must be in `2..=63`.
/// `Default::default()` returns the spec's flat all-4s matrices for
/// both components (equivalent to the legacy encoder behaviour with
/// `load_luma_quantization_matrix = load_chroma_quantization_matrix = 0`).
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct QuantMatrices {
    pub luma: [u8; 64],
    pub chroma: [u8; 64],
}

impl QuantMatrices {
    /// All-4s flat matrices — matches what the encoder emits when no
    /// custom matrices are loaded.
    pub const fn flat() -> Self {
        Self {
            luma: DEFAULT_QMAT,
            chroma: DEFAULT_QMAT,
        }
    }

    /// JPEG-derived perceptual matrices from [`PERCEPTUAL_LUMA_QMAT`]
    /// and [`PERCEPTUAL_CHROMA_QMAT`].
    pub const fn perceptual() -> Self {
        Self {
            luma: PERCEPTUAL_LUMA_QMAT,
            chroma: PERCEPTUAL_CHROMA_QMAT,
        }
    }

    /// True when both matrices equal the spec's all-4s default.
    pub fn is_default(&self) -> bool {
        self.luma == DEFAULT_QMAT && self.chroma == DEFAULT_QMAT
    }

    /// True when every weight is in `2..=63`.
    pub fn weights_valid(&self) -> bool {
        self.luma.iter().all(|w| (2..=63).contains(w))
            && self.chroma.iter().all(|w| (2..=63).contains(w))
    }
}

impl Default for QuantMatrices {
    fn default() -> Self {
        Self::flat()
    }
}

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
    fn perceptual_matrices_in_valid_weight_range() {
        // RDD 36 §7.3: weights must be integers in 2..=63.
        let m = QuantMatrices::perceptual();
        assert!(m.weights_valid());
        // DC weight = 2 (twice as fine as the flat all-4 default —
        // this is what lets the perceptual matrix beat flat on PSNR
        // at matched bitrate on natural-image-like spectra).
        assert_eq!(m.luma[0], 2);
        assert_eq!(m.chroma[0], 2);
        // High-frequency luma weight is materially larger than 4.
        assert!(m.luma[63] > 4);
    }

    #[test]
    fn perceptual_matrices_match_jpeg_k1_k2_normalised_dc2() {
        // K.1 and K.2 normalised by 2/16, clamped to 2..=63, with
        // round-half-up. Spot check the most distinctive entries.
        // K.1[0][0] = 16 → round(2.0) = 2
        assert_eq!(PERCEPTUAL_LUMA_QMAT[0], 2);
        // K.1[0][1] = 11 → round(1.375) = 1 → clamp to 2
        assert_eq!(PERCEPTUAL_LUMA_QMAT[1], 2);
        // K.1[0][7] = 61 → round(7.625) = 8
        assert_eq!(PERCEPTUAL_LUMA_QMAT[7], 8);
        // K.1[7][0] = 72 → round(9.0) = 9
        assert_eq!(PERCEPTUAL_LUMA_QMAT[7 * 8], 9);
        // K.2[0][0] = 17 → round(2.125) = 2
        assert_eq!(PERCEPTUAL_CHROMA_QMAT[0], 2);
        // K.2[0][3] = 47 → round(5.875) = 6
        assert_eq!(PERCEPTUAL_CHROMA_QMAT[3], 6);
        // K.2[3][3] = 99 → round(12.375) = 12
        assert_eq!(PERCEPTUAL_CHROMA_QMAT[3 * 8 + 3], 12);
    }

    #[test]
    fn quant_matrices_default_is_flat() {
        assert_eq!(QuantMatrices::default(), QuantMatrices::flat());
        assert!(QuantMatrices::default().is_default());
        assert!(!QuantMatrices::perceptual().is_default());
    }

    #[test]
    fn quant_matrices_weights_valid_rejects_out_of_range() {
        let mut bad = QuantMatrices::flat();
        bad.luma[0] = 1; // below min
        assert!(!bad.weights_valid());
        bad.luma[0] = 64; // above max
        assert!(!bad.weights_valid());
        bad.luma[0] = 4;
        bad.chroma[5] = 0;
        assert!(!bad.weights_valid());
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
