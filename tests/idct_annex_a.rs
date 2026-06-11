//! RDD 36 Annex A — IDCT Implementation Accuracy Qualification.
//!
//! RDD 36 §7.4 permits any IDCT implementation (fixed- or floating-point)
//! provided it "shall comply with Annex A, 'IDCT Implementation Accuracy
//! Qualification'". This harness runs the Annex A qualification procedure
//! against this crate's production IDCT ([`oxideav_prores::dct::idct8x8`])
//! and asserts all five acceptance criteria.
//!
//! ### Procedure (Annex A steps 1-7)
//!
//! 1. Generate random integers in `-L ..= +H`, convert to
//!    reference-precision (f64) floats, divide by 8, and pack into 8x8
//!    blocks in row-major order. 10,000 blocks per data set, for the
//!    three (L, H) data sets `(2048, 2047)`, `(40, 40)`, `(2400, 2400)`.
//! 2. Forward-transform every block with a separable, orthogonal FDCT in
//!    reference-precision floating point. The basis is the transpose of
//!    the §7.4 IDCT kernel that RDD 36 prints in full:
//!    `f[y][x] = 1/4 * sum_u sum_v C(u) C(v) F[v][u]
//!     * cos((2x+1)u*pi/16) * cos((2y+1)v*pi/16)` with `C(0) = 1/sqrt(2)`,
//!    `C(n) = 1` otherwise.
//! 3. Round each coefficient to the nearest quarter-integer (multiply by
//!    4, round to nearest, divide by 4) and clip to `-2048.0 ..= +2047.75`
//!    — a signed fixed-point format with 12 integer + 2 fraction bits.
//!    These blocks are the input to both inverse transforms.
//! 4. Reference IDCT: inverse-transform each block in f64, retain full
//!    precision, clip to `-256.0 ..= +256.0`.
//! 5. Test IDCT: run the proposed implementation (`dct::idct8x8`, f32),
//!    promote its output to f64, clip to `-256.0 ..= +256.0`.
//! 6. For each of the 64 pixel positions across the 10,000 blocks,
//!    accumulate peak, mean, and mean-square error between reference and
//!    test output, in f64.
//! 7. Rerun the measurement with the sign changed on every step-1 integer.
//!
//! ### Acceptance criteria (Annex A, error terms per IEEE Std 1180-1990
//! §3.3 naming)
//!
//! 1. ppe  — per-pixel-position peak error      <= 0.15 in magnitude
//! 2. pmse — per-pixel-position mean square err <= 0.002
//! 3. omse — overall mean square error          <= 0.001
//! 4. pme  — per-pixel-position mean error      <= 0.0015 in magnitude
//! 5. ome  — overall mean error                 <= 0.00015 in magnitude
//!
//! ### Random source
//!
//! Annex A step 1 defers to "the random number generator in the appendix
//! of IEEE Std 1180-1990", which is not staged under `docs/`. A
//! crate-local deterministic 64-bit LCG with power-of-two rejection
//! sampling (unbiased uniform over `-L ..= +H`) substitutes for it; the
//! generator choice affects only run-to-run reproducibility (pinned here
//! by a fixed seed), not the statistical validity of the measured error
//! terms. Every range, block count, rounding rule, clip bound, and
//! threshold above is verbatim from Annex A.

use oxideav_prores::dct::idct8x8;

const BLOCKS_PER_SET: usize = 10_000;

/// Deterministic 64-bit LCG (substitute random source — see module doc).
struct Lcg(u64);

impl Lcg {
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // High bits have the longest period / best uniformity in an LCG.
        self.0 >> 16
    }

    /// Unbiased uniform integer in `-l ..= h` via rejection sampling from
    /// the next power of two covering the range width.
    fn uniform(&mut self, l: i32, h: i32) -> i32 {
        let width = (h - (-l) + 1) as u64; // number of representable values
        let mask = width.next_power_of_two() - 1;
        loop {
            let v = self.next_u64() & mask;
            if v < width {
                return -l + v as i32;
            }
        }
    }
}

/// f64 cosine basis `t[k][n] = 0.5 * C(k) * cos((2n+1) k pi / 16)`,
/// `C(0) = 1/sqrt(2)` — the §7.4 kernel at reference precision.
fn basis_f64() -> [[f64; 8]; 8] {
    let mut t = [[0.0f64; 8]; 8];
    for (k, row) in t.iter_mut().enumerate() {
        let c_k = if k == 0 { (0.5f64).sqrt() } else { 1.0 };
        for (n, e) in row.iter_mut().enumerate() {
            *e = 0.5 * c_k * ((2 * n + 1) as f64 * k as f64 * std::f64::consts::PI / 16.0).cos();
        }
    }
    t
}

/// Separable orthogonal FDCT in f64 (transpose of the §7.4 IDCT kernel).
fn fdct8x8_f64(t: &[[f64; 8]; 8], block: &mut [f64; 64]) {
    let mut tmp = [0.0f64; 64];
    for y in 0..8 {
        for k in 0..8 {
            let mut s = 0.0;
            for n in 0..8 {
                s += t[k][n] * block[y * 8 + n];
            }
            tmp[y * 8 + k] = s;
        }
    }
    for x in 0..8 {
        for k in 0..8 {
            let mut s = 0.0;
            for n in 0..8 {
                s += t[k][n] * tmp[n * 8 + x];
            }
            block[k * 8 + x] = s;
        }
    }
}

/// Separable orthogonal IDCT in f64 — the Annex A step-4 reference
/// transform, mirroring the structure of the production `dct::idct8x8`.
fn idct8x8_f64(t: &[[f64; 8]; 8], block: &mut [f64; 64]) {
    let mut tmp = [0.0f64; 64];
    for y in 0..8 {
        for n in 0..8 {
            let mut s = 0.0;
            for k in 0..8 {
                s += t[k][n] * block[y * 8 + k];
            }
            tmp[y * 8 + n] = s;
        }
    }
    for x in 0..8 {
        for m in 0..8 {
            let mut s = 0.0;
            for k in 0..8 {
                s += t[k][m] * tmp[k * 8 + x];
            }
            block[m * 8 + x] = s;
        }
    }
}

/// Accumulated per-pixel-position and overall error terms (step 6).
struct ErrorStats {
    /// ppe — peak error magnitude at the worst pixel position.
    ppe: f64,
    /// pmse — mean square error at the worst pixel position.
    pmse: f64,
    /// omse — mean square error over all positions.
    omse: f64,
    /// pme — mean error magnitude at the worst pixel position.
    pme: f64,
    /// ome — mean error over all positions (signed; criterion is on
    /// magnitude).
    ome: f64,
}

/// Run Annex A steps 1-6 for one `(L, H)` data set with the given sign
/// applied to every step-1 integer (step 7 = rerun with `sign = -1`).
fn measure(l: i32, h: i32, sign: i32, seed: u64) -> ErrorStats {
    let t = basis_f64();
    let mut rng = Lcg(seed);

    let mut sum_err = [0.0f64; 64];
    let mut sum_sq = [0.0f64; 64];
    let mut peak = [0.0f64; 64];

    for _ in 0..BLOCKS_PER_SET {
        // Step 1: random integers in -L..=+H (sign-flipped on the step-7
        // rerun), to f64, divided by 8, row-major.
        let mut ref_block = [0.0f64; 64];
        for s in ref_block.iter_mut() {
            *s = f64::from(sign * rng.uniform(l, h)) / 8.0;
        }

        // Step 2: reference-precision FDCT.
        fdct8x8_f64(&t, &mut ref_block);

        // Step 3: round to nearest quarter-integer, clip to
        // -2048.0..=+2047.75.
        for s in ref_block.iter_mut() {
            *s = ((*s * 4.0).round() / 4.0).clamp(-2048.0, 2047.75);
        }

        // Step 5 input: the same quarter-integer block, demoted to the
        // production IDCT's f32 input type. Every quarter-integer in
        // -2048.0..=+2047.75 is exactly representable in f32 (13-bit
        // significand requirement), so the demotion is lossless and both
        // transforms see identical inputs.
        let mut test_block = [0.0f32; 64];
        for (d, s) in test_block.iter_mut().zip(ref_block.iter()) {
            *d = *s as f32;
        }

        // Step 4: reference IDCT in f64, full precision, clip to
        // -256.0..=+256.0.
        idct8x8_f64(&t, &mut ref_block);
        for s in ref_block.iter_mut() {
            *s = s.clamp(-256.0, 256.0);
        }

        // Step 5: proposed implementation, promoted to f64, same clip.
        idct8x8(&mut test_block);
        let mut test_f64 = [0.0f64; 64];
        for (d, s) in test_f64.iter_mut().zip(test_block.iter()) {
            *d = f64::from(*s).clamp(-256.0, 256.0);
        }

        // Step 6: accumulate per-position error terms in f64.
        for i in 0..64 {
            let e = test_f64[i] - ref_block[i];
            sum_err[i] += e;
            sum_sq[i] += e * e;
            if e.abs() > peak[i] {
                peak[i] = e.abs();
            }
        }
    }

    let n = BLOCKS_PER_SET as f64;
    let mut ppe = 0.0f64;
    let mut pmse = 0.0f64;
    let mut pme = 0.0f64;
    let mut omse = 0.0f64;
    let mut ome = 0.0f64;
    for i in 0..64 {
        let mean = sum_err[i] / n;
        let mse = sum_sq[i] / n;
        ppe = ppe.max(peak[i]);
        pmse = pmse.max(mse);
        pme = pme.max(mean.abs());
        omse += mse;
        ome += mean;
    }
    omse /= 64.0;
    ome /= 64.0;

    ErrorStats {
        ppe,
        pmse,
        omse,
        pme,
        ome,
    }
}

/// Assert the five Annex A acceptance criteria on one measurement run.
fn assert_criteria(l: i32, h: i32, sign: i32, st: &ErrorStats) {
    let tag = format!("(L={l}, H={h}, sign={sign:+})");
    assert!(
        st.ppe <= 0.15,
        "{tag}: criterion 1 — ppe {} exceeds 0.15",
        st.ppe
    );
    assert!(
        st.pmse <= 0.002,
        "{tag}: criterion 2 — pmse {} exceeds 0.002",
        st.pmse
    );
    assert!(
        st.omse <= 0.001,
        "{tag}: criterion 3 — omse {} exceeds 0.001",
        st.omse
    );
    assert!(
        st.pme <= 0.0015,
        "{tag}: criterion 4 — pme {} exceeds 0.0015",
        st.pme
    );
    assert!(
        st.ome.abs() <= 0.00015,
        "{tag}: criterion 5 — |ome| {} exceeds 0.00015",
        st.ome.abs()
    );
    println!(
        "annex A {tag}: ppe={:.3e} pmse={:.3e} omse={:.3e} pme={:.3e} ome={:+.3e}",
        st.ppe, st.pmse, st.omse, st.pme, st.ome
    );
}

/// One qualification run = steps 1-6 plus the step-7 sign-flipped rerun
/// (same integer sequence — same seed — with every value negated).
fn qualify(l: i32, h: i32, seed: u64) {
    for sign in [1, -1] {
        let st = measure(l, h, sign, seed);
        assert_criteria(l, h, sign, &st);
    }
}

#[test]
fn annex_a_idct_qualification_full_range() {
    // Data set 1: (L = 2048, H = 2047) — full 12-bit signed coefficient
    // range; pixels span the full -256..=+255.875 9-bit signed range
    // after the divide-by-8.
    qualify(2048, 2047, 0x5247_4436_0001);
}

#[test]
fn annex_a_idct_qualification_low_amplitude() {
    // Data set 2: (L = H = 40) — low-amplitude pixels (±5 after the
    // divide-by-8), the near-flat-block regime where absolute error
    // tolerances bite hardest relative to signal.
    qualify(40, 40, 0x5247_4436_0002);
}

#[test]
fn annex_a_idct_qualification_overdriven() {
    // Data set 3: (L = H = 2400) — pixels overdrive the ±256 output range
    // (±300 after the divide-by-8), exercising the step-4/5 output clip.
    qualify(2400, 2400, 0x5247_4436_0003);
}

#[test]
fn quarter_integers_are_exact_in_f32() {
    // The step-5 f64 -> f32 demotion claim: every quarter-integer in
    // -2048.0..=+2047.75 needs at most 13 significand bits (value * 4 is
    // an integer of magnitude <= 8192 = 2^13), within f32's 24. Spot-check
    // the extremes and a worst-case odd value.
    for v in [-2048.0f64, 2047.75, -2047.25, 1023.75, 0.25, -0.25] {
        let f = v as f32;
        assert_eq!(f64::from(f), v, "quarter-integer {v} not exact in f32");
    }
}
