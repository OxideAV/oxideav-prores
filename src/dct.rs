//! Textbook f32 8x8 forward/inverse DCT used by ProRes.
//!
//! RDD 36 §7.4 prints the IDCT formula in full and explicitly permits
//! either a fixed-point or a floating-point implementation, provided it
//! complies with Annex A ("IDCT Implementation Accuracy Qualification").
//! This f32 transform is qualified against the full Annex A procedure —
//! all three 10,000-block data sets, both sign runs, all five error
//! criteria — by `tests/idct_annex_a.rs`, with the worst measured error
//! term (ppe ≈ 7.1e-4 vs. the 0.15 limit) more than 200× inside the
//! acceptance bound.

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

/// One 8-point forward DCT, scaled per frequency as in Arai, Agui and
/// Nakajima ("A Fast DCT-SQ Scheme for Images", Trans. IEICE E71(11),
/// 1988): 29 additions and 5 multiplications, against 64
/// multiply-adds for the matrix form. The output is
/// `out[k] = 2√2 · s[k] · X[k]` (`s` as in [`aan_scale`]), where `X` is
/// the orthonormal-basis output of [`fdct8x8`]'s row pass. The scale is
/// cancelled by the caller (see [`fdct8x8_aan`]).
///
/// **Derivation.** Write `θ = π/16` and `C[k] = Σₙ d[n]·cos((2n+1)kθ)`.
/// Then `out[0] = C[0]` and, for `k ≥ 1`, `out[k] = 2·cos(kθ)·C[k]`.
///
/// 1. Fold. `cos((2(7−n)+1)kθ) = (−1)ᵏ·cos((2n+1)kθ)`, so for
///    `n = 0..3` even `k` sees only `sₙ = d[n] + d[7−n]` and odd `k`
///    only `aₙ = d[n] − d[7−n]`. Call either `xₙ`.
/// 2. Product to sum. `2·cos(kθ)·cos((2n+1)kθ) = cos(2nkθ) + cos(2(n+1)kθ)`,
///    so, grouping by angle,
///
///    ```text
///    out[k] = x0 + cos(2kθ)(x0+x1) + cos(4kθ)(x1+x2) + cos(6kθ)(x2+x3)
///                + cos(8kθ)·x3
///    ```
///
///    where `cos(8kθ)` is 0 for odd `k` and −1 for `k = 2, 6`.
/// 3. Even outputs, with `cᵢ = cos(iθ)`:
///    `out[0] = s0+s1+s2+s3`, `out[4] = (s0+s3) − (s1+s2)` (the scale of
///    `[0]` and `[4]` is folded so neither needs a multiply), and
///    `out[2], out[6] = (s0−s3) ± c4·((s0−s3) + (s1−s2))`.
/// 4. Odd outputs, with `p = a0+a1`, `q = a1+a2`, `r = a2+a3`:
///
///    ```text
///    out[1], out[7] = (a0 + c4·q) ± (c2·p + c6·r)
///    out[3], out[5] = (a0 − c4·q) ± (c6·p − c2·r)
///    ```
///
///    The two brackets on the right are a reflection of `(p, r)`. With
///    `w = c2·(p+r)` they cost three multiplies instead of four:
///    `c2·p + c6·r = w − (c2−c6)·r` and `c6·p − c2·r = (c2+c6)·p − w`.
#[inline(always)]
fn fdct8_aan(d: [f32; 8]) -> [f32; 8] {
    const C2: f32 = 0.923_879_5; // cos(2θ)
    const C4: f32 = std::f32::consts::FRAC_1_SQRT_2; // cos(4θ)
    const C2_MINUS_C6: f32 = 0.541_196_1; // cos(2θ) - cos(6θ)
    const C2_PLUS_C6: f32 = 1.306_563; // cos(2θ) + cos(6θ)

    // Step 1: fold the input into sums and differences.
    let s0 = d[0] + d[7];
    let s1 = d[1] + d[6];
    let s2 = d[2] + d[5];
    let s3 = d[3] + d[4];
    let a0 = d[0] - d[7];
    let a1 = d[1] - d[6];
    let a2 = d[2] - d[5];
    let a3 = d[3] - d[4];

    // Step 3: even outputs.
    let s03 = s0 + s3;
    let s12 = s1 + s2;
    let s0m3 = s0 - s3;
    let even_c4 = C4 * (s0m3 + (s1 - s2));

    // Step 4: odd outputs.
    let p = a0 + a1;
    let q = a1 + a2;
    let r = a2 + a3;
    let odd_c4 = C4 * q;
    let w = C2 * (p + r);
    let refl_17 = w - C2_MINUS_C6 * r; // c2·p + c6·r
    let refl_35 = C2_PLUS_C6 * p - w; // c6·p − c2·r
    let base_17 = a0 + odd_c4;
    let base_35 = a0 - odd_c4;

    [
        s03 + s12,
        base_17 + refl_17,
        s0m3 + even_c4,
        base_35 + refl_35,
        s03 - s12,
        base_35 - refl_35,
        s0m3 - even_c4,
        base_17 - refl_17,
    ]
}

/// Per-frequency scale of [`fdct8_aan`]: `1` for DC and
/// `√2 · cos(kπ/16)` otherwise.
fn aan_scale() -> &'static [f64; 8] {
    static S: OnceLock<[f64; 8]> = OnceLock::new();
    S.get_or_init(|| {
        let mut s = [1.0f64; 8];
        for (k, v) in s.iter_mut().enumerate().skip(1) {
            *v = std::f64::consts::SQRT_2 * (k as f64 * std::f64::consts::PI / 16.0).cos();
        }
        s
    })
}

/// Factor by which [`fdct8x8_aan`]'s output exceeds [`fdct8x8`]'s at
/// natural-order position `v * 8 + u`: `8 · s[v] · s[u]` with `s` the
/// per-frequency scale of the 1-D AAN transform. Exact to f64.
pub fn fdct8x8_aan_scale(k: usize) -> f64 {
    let s = aan_scale();
    8.0 * s[k / 8] * s[k % 8]
}

/// Scaled forward DCT of an 8x8 block in natural order, in place: the
/// same transform as [`fdct8x8`] with coefficient `k` multiplied by
/// [`fdct8x8_aan_scale`]`(k)`. Encoders fold that factor into the
/// quantiser (one multiply per coefficient either way), so the
/// separable row and column passes need only the 1-D AAN butterflies.
pub fn fdct8x8_aan(block: &mut [f32; 64]) {
    for row in block.as_chunks_mut::<8>().0 {
        *row = fdct8_aan(*row);
    }
    for x in 0..8 {
        let col = fdct8_aan(std::array::from_fn(|y| block[y * 8 + x]));
        for (y, v) in col.into_iter().enumerate() {
            block[y * 8 + x] = v;
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

/// Specialised forward DCT for the **constant-block** case: every entry of
/// `block[..]` is exactly the same scalar `v` (already level-shifted by the
/// caller per RDD 36 §7.5.1). Equivalent to (but faster than) calling
/// [`fdct8x8`] on the buffer.
///
/// **Derivation (from the same cosine basis [`fdct8x8`] uses).** With
/// `t[k][n] = 0.5 * c_k * cos((2n+1) k π / 16)` and `c_0 = 1/√2`,
/// the row pass of a constant-block input collapses to
///
/// ```text
///   tmp[y * 8 + k] = sum_n  t[k][n] * v
///                  = v * sum_n  0.5 * c_k * cos((2n+1) k π / 16)
/// ```
///
/// For `k == 0` every cosine is `1`, so `sum_n = 8 * 0.5 * (1/√2) = 2√2`.
/// For `k > 0` the eight cosine samples sum to exactly zero (one full
/// cycle of cos around eight equally-spaced phases), so the entire AC
/// row of `tmp` is zero. The column pass then produces a single non-zero
/// entry at `(k, m) = (0, 0)` equal to `v * (2√2)² = 8 * v`. So the
/// constant-block forward DCT is one DC coefficient of `8 * v`, all 63 AC
/// coefficients zero.
///
/// Caller is responsible for the constant-block check via
/// [`is_constant_block`] (or an in-place check by the caller). The
/// function does **not** read `block[1..]`; if those entries differ from
/// `block[0]` before the call, the result is wrong.
///
/// Common hit cases inside the encoder: the right-edge / bottom-edge
/// boundary-clamp pad blocks (per RDD 36 §7.5.1 the encoder copies the
/// last sample to fill a partial MB row), large flat areas of natural
/// content at high quant indices, and synthetic gradients whose smooth
/// regions reduce to single-value 8x8 tiles.
pub fn fdct8x8_constant(block: &mut [f32; 64]) {
    let v = block[0];
    let dc = v * 8.0;
    for s in block.iter_mut() {
        *s = 0.0;
    }
    block[0] = dc;
}

/// Predicate matching the [`fdct8x8_constant`] precondition: every entry
/// of `block[..]` equals `block[0]` exactly.
///
/// The encoder calls this on a freshly read-and-level-shifted 8x8 input
/// block. The samples come from `read_block()` which returns f32 values
/// in the centred range `[-256, 256)`; two samples that came from the
/// same source byte produce bit-identical f32, so a constant-source
/// block is detected exactly.
#[inline]
pub fn is_constant_block(block: &[f32; 64]) -> bool {
    let v0 = block[0];
    for &v in &block[1..] {
        if v != v0 {
            return false;
        }
    }
    true
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
    fn aan_fdct_matches_textbook_fdct_after_scaling() {
        let mut seed = 1u32;
        for _ in 0..200 {
            let mut a = [0.0f32; 64];
            for v in a.iter_mut() {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                *v = (seed >> 23) as f32 - 256.0;
            }
            let mut b = a;
            fdct8x8(&mut a);
            fdct8x8_aan(&mut b);
            for k in 0..64 {
                let scaled = (b[k] as f64 / fdct8x8_aan_scale(k)) as f32;
                assert!(
                    (scaled - a[k]).abs() < 2e-3,
                    "k={k}: aan/scale = {scaled} vs textbook {}",
                    a[k]
                );
            }
        }
    }

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

    #[test]
    fn is_constant_block_detects_uniform_input() {
        let block = [42.0f32; 64];
        assert!(is_constant_block(&block));

        let mut diff = [42.0f32; 64];
        diff[37] = 41.999;
        assert!(!is_constant_block(&diff), "any differing entry breaks it");

        // Negative zero != positive zero at the bit level but compares
        // equal as f32; we accept that as a match (DC = 0 either way).
        let mut signed_zero = [0.0f32; 64];
        signed_zero[5] = -0.0;
        assert!(is_constant_block(&signed_zero));
    }

    #[test]
    fn fdct_constant_matches_general_fdct() {
        // Every plausible level-shifted sample value from the read_block
        // range `[-256, 256)` plus exact endpoints. The general fdct8x8
        // and the constant fast path must produce identical buffers to
        // f32 round-off.
        for &v in &[-256.0f32, -200.0, -1.0, 0.0, 1.0, 64.0, 128.0, 255.0] {
            let mut a = [v; 64];
            let mut b = [v; 64];
            fdct8x8(&mut a);
            fdct8x8_constant(&mut b);
            for i in 0..64 {
                assert!(
                    (a[i] - b[i]).abs() < 1e-3,
                    "v={v}: fdct_constant[{i}] = {} vs fdct[{i}] = {}",
                    b[i],
                    a[i]
                );
            }
        }
    }

    #[test]
    fn fdct_constant_emits_dc_equal_to_eight_v() {
        let mut block = [12.5f32; 64];
        fdct8x8_constant(&mut block);
        assert!(
            (block[0] - 100.0).abs() < 1e-4,
            "DC = 8 * 12.5 = 100, got {}",
            block[0]
        );
        for &s in &block[1..] {
            assert_eq!(s, 0.0, "AC must be exactly 0 after constant fdct");
        }
    }

    #[test]
    fn fdct_constant_idct_dc_only_round_trip_is_identity() {
        // Constant input -> constant fdct -> DC-only idct -> constant output.
        // Verifies the two fast paths compose to identity within float
        // round-off, matching the textbook fdct->idct round-trip.
        for &v in &[-256.0f32, -1.0, 0.0, 1.0, 100.0, 255.0] {
            let mut block = [v; 64];
            fdct8x8_constant(&mut block);
            idct8x8_dc_only(&mut block);
            for &s in block.iter() {
                assert!(
                    (s - v).abs() < 1e-3,
                    "round-trip lost the constant: v={v}, got {s}"
                );
            }
        }
    }
}
