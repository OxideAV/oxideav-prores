//! Textbook f32 8x8 forward/inverse DCT used by ProRes.
//!
//! The ProRes spec specifies a specific integer DCT; for a minimal
//! in-house encoder+decoder pair this textbook float DCT is correct
//! (round-trip PSNR well over 60 dB before quant) and easier to read.
//! If interop with real streams becomes a target we can swap in the
//! integer transform here without disturbing the rest of the stack.

use std::f32::consts::PI;
use std::sync::OnceLock;

fn cos_table() -> &'static [[f32; 8]; 8] {
    static T: OnceLock<[[f32; 8]; 8]> = OnceLock::new();
    T.get_or_init(|| {
        let mut t = [[0.0f32; 8]; 8];
        for k in 0..8 {
            let c_k = if k == 0 {
                (1.0_f32 / 2.0_f32).sqrt()
            } else {
                1.0
            };
            for n in 0..8 {
                t[k][n] = 0.5 * c_k * ((2 * n + 1) as f32 * k as f32 * PI / 16.0).cos();
            }
        }
        t
    })
}

/// Forward DCT of an 8x8 block in natural order, in-place. Input should
/// be level-shifted (subtract the format midpoint — 128 for 8-bit — to
/// centre on 0).
pub fn fdct8x8(block: &mut [f32; 64]) {
    let t = cos_table();
    let mut tmp = [0.0f32; 64];
    for y in 0..8 {
        for k in 0..8 {
            let mut s = 0.0f32;
            for n in 0..8 {
                s += t[k][n] * block[y * 8 + n];
            }
            tmp[y * 8 + k] = s;
        }
    }
    for x in 0..8 {
        for k in 0..8 {
            let mut s = 0.0f32;
            for n in 0..8 {
                s += t[k][n] * tmp[n * 8 + x];
            }
            block[k * 8 + x] = s;
        }
    }
}

/// Inverse DCT of an 8x8 block in natural order, in-place.
pub fn idct8x8(block: &mut [f32; 64]) {
    let t = cos_table();
    let mut tmp = [0.0f32; 64];
    for y in 0..8 {
        for n in 0..8 {
            let mut s = 0.0f32;
            for k in 0..8 {
                s += t[k][n] * block[y * 8 + k];
            }
            tmp[y * 8 + n] = s;
        }
    }
    for x in 0..8 {
        for m in 0..8 {
            let mut s = 0.0f32;
            for k in 0..8 {
                s += t[k][m] * tmp[k * 8 + x];
            }
            block[m * 8 + x] = s;
        }
    }
}

/// Specialised inverse DCT for the **DC-only** case: every entry of
/// `block[1..]` is zero. Equivalent to (but faster than) calling
/// [`idct8x8`] on a buffer with a non-zero `block[0]` and 63 zeros.
///
/// **Derivation (from the cosine basis used by [`idct8x8`]).** With
/// `t[k][n] = 0.5 * c_k * cos((2n+1) k π / 16)` and `c_0 = 1/√2`, the
/// natural-order IDCT of a block whose only non-zero entry is `block[0]`
/// reduces to
///
/// ```text
///   out[m * 8 + x] = block[0] * t[0][m] * t[0][x]
///                  = block[0] * (1/2 * 1/√2) * (1/2 * 1/√2)
///                  = block[0] / 8
/// ```
///
/// for every `(m, x)` because `cos(π/16 * 0) = 1`. So the entire output
/// block is constant `block[0] / 8`. This is what the general-case loop
/// computes too, just at a much higher cost (64 × 16 multiply-adds vs. one
/// multiply + 64 stores).
///
/// Caller is responsible for the DC-only check. The function does not
/// inspect `block[1..]`; if those entries are non-zero, the result is
/// **wrong**. Use [`is_dc_only`] (or an in-place check by the caller) to
/// gate this on the post-dequant block contents.
pub fn idct8x8_dc_only(block: &mut [f32; 64]) {
    let dc = block[0] * 0.125;
    for s in block.iter_mut() {
        *s = dc;
    }
}

/// Predicate matching the [`idct8x8_dc_only`] precondition: every entry
/// of `block[1..]` is exactly zero.
///
/// In ProRes decode this is checked on a dequantised f32 block fed to the
/// IDCT. Dequantisation multiplies by `qmat[k] * qscale / 8`, both of
/// which are positive, so a coefficient is zero **iff** its
/// quantised-domain counterpart was zero — the spec's run/level/sign
/// entropy coder produces those whenever the block is smooth enough.
#[inline]
pub fn is_dc_only(block: &[f32; 64]) -> bool {
    // Bitwise zero check: an f32 produced by `n * qmat * qscale / 8` is
    // exactly 0.0 iff the dequant input was 0, including the +0.0 / -0.0
    // distinction (both compare-equal to 0.0). We rely on the equality
    // semantics rather than `to_bits()` because a -0.0 result is still
    // mathematically zero for the IDCT.
    for &v in &block[1..] {
        if v != 0.0 {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn idct_of_dct_is_identity() {
        let mut block = [0.0f32; 64];
        for (i, b) in block.iter_mut().enumerate() {
            *b = ((i * 7) % 255) as f32 - 128.0;
        }
        let original = block;
        fdct8x8(&mut block);
        idct8x8(&mut block);
        for i in 0..64 {
            assert!((block[i] - original[i]).abs() < 1e-2);
        }
    }

    #[test]
    fn is_dc_only_detects_zero_ac() {
        let mut block = [0.0f32; 64];
        assert!(is_dc_only(&block), "all-zero block is DC-only");
        block[0] = 1024.0;
        assert!(is_dc_only(&block), "DC-only block (AC == 0) is DC-only");
        block[5] = 1e-30;
        assert!(
            !is_dc_only(&block),
            "any non-zero AC coefficient breaks the predicate"
        );
        block[5] = -0.0;
        assert!(
            is_dc_only(&block),
            "negative zero AC is still mathematically zero"
        );
    }

    #[test]
    fn idct_dc_only_matches_general_idct() {
        // Every plausible DC coefficient — including spec-permitted
        // negatives that arise from a level-shifted, quantised, then
        // dequantised DC of a dark sample.
        for &dc in &[-2048.0f32, -512.0, -1.0, 0.0, 1.0, 512.0, 2047.0, 4096.0] {
            let mut a = [0.0f32; 64];
            let mut b = [0.0f32; 64];
            a[0] = dc;
            b[0] = dc;
            idct8x8(&mut a);
            idct8x8_dc_only(&mut b);
            for i in 0..64 {
                assert!(
                    (a[i] - b[i]).abs() < 1e-4,
                    "DC={dc}: idct_dc_only[{i}] = {} vs idct[{i}] = {}",
                    b[i],
                    a[i]
                );
            }
        }
    }

    #[test]
    fn idct_dc_only_output_is_constant_dc_over_eight() {
        let mut block = [0.0f32; 64];
        block[0] = 800.0;
        idct8x8_dc_only(&mut block);
        // 800 / 8 = 100.
        for &s in block.iter() {
            assert!((s - 100.0).abs() < 1e-4);
        }
    }
}
