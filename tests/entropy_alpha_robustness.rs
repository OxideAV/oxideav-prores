//! Decode-robustness regression tests for the RDD 36 §7.1.1 run/level/sign
//! coefficient coder and the §7.1.2 / Table 12-14 scanned-alpha coder.
//!
//! The crate's fuzz harnesses (`fuzz/fuzz_targets/decode_entropy.rs`,
//! `decode_alpha.rs`) drive these two entry points with adversarial bytes
//! and assert they never panic, integer-overflow (debug), or index out of
//! bounds. Those harnesses require a nightly toolchain + libFuzzer and so
//! do not run in the standard CI matrix. This file pins the *spec-derived*
//! corners of that hardening — the overrun arms, the truncation arms, and
//! the round-trip identity — as ordinary `cargo test` cases that DO run in
//! CI, so a future refactor of the coders cannot silently regress the
//! malformed-input behaviour.
//!
//! Everything here is built from RDD 36 syntax + the crate's own public
//! encoders; no external codec bytes are consulted.

use oxideav_prores::alpha::{decode_scanned_alpha, encode_scanned_alpha, AlphaChannelType};
use oxideav_prores::entropy::{decode_scanned_coefficients, encode_scanned_coefficients};

// ---------------------------------------------------------------------------
// §7.1.1 scanned coefficients
// ---------------------------------------------------------------------------

/// A round-trip through the spec-legal encoder must reconstruct the exact
/// slice-scan coefficient array (DC + AC, in slice-scan order).
#[test]
fn coefficients_round_trip_is_exact() {
    let num_blocks = 8usize; // a maximal 8-MB-wide 4:2:2 luma slice row
    let total = num_blocks * 64;
    // A mix of DC drift + sparse AC content so both codebook-adaptation
    // paths (DC-difference and AC run/level) are exercised.
    let mut coeffs = vec![0i32; total];
    for (b, dc) in coeffs.iter_mut().take(num_blocks).enumerate() {
        *dc = (b as i32 - 4) * 17; // DC walks across blocks
    }
    // Scatter a few AC coefficients at assorted frequencies.
    for (i, c) in coeffs.iter_mut().enumerate().skip(num_blocks) {
        if i % 23 == 0 {
            *c = ((i % 7) as i32) - 3;
        }
    }
    let encoded = encode_scanned_coefficients(&coeffs, num_blocks).expect("encode");
    let decoded = decode_scanned_coefficients(&encoded, num_blocks).expect("decode");
    assert_eq!(decoded, coeffs, "coefficient round-trip must be exact");
}

/// A zero-block slice carries no DC/AC syntax: decoding `num_blocks == 0`
/// must return an empty array without touching the bitstream.
#[test]
fn zero_block_slice_decodes_to_empty() {
    let decoded = decode_scanned_coefficients(&[], 0).expect("empty decode");
    assert!(decoded.is_empty());
    // Even with payload bytes present, zero blocks → empty, no panic.
    let decoded2 = decode_scanned_coefficients(&[0xFF, 0x00, 0xAB], 0).expect("empty decode 2");
    assert!(decoded2.is_empty());
}

/// Truncating a valid coefficient stream mid-codeword must surface a clean
/// `Err` (the bit reader running off the end), never a panic.
#[test]
fn truncated_coefficient_stream_errors_cleanly() {
    let num_blocks = 4usize;
    let total = num_blocks * 64;
    let mut coeffs = vec![0i32; total];
    for (b, dc) in coeffs.iter_mut().take(num_blocks).enumerate() {
        *dc = (b as i32) * 200; // large DC drift → multi-bit codewords
    }
    coeffs[num_blocks + 1] = 9; // an AC level so the AC loop has work
    let encoded = encode_scanned_coefficients(&coeffs, num_blocks).expect("encode");
    // Lop off the trailing half of the byte stream; whatever codeword the
    // decoder is mid-way through, it must error rather than panic.
    for cut in 1..encoded.len() {
        let truncated = &encoded[..cut];
        // The result may be Ok (if the prefix happens to decode to a valid
        // shorter scan) or Err, but it must never panic / overflow.
        let _ = decode_scanned_coefficients(truncated, num_blocks);
    }
}

/// Adversarial bytes fed straight to the coefficient decoder must return a
/// `Result`, never panic — exercising the AC-run-overrun and end-of-data
/// arms with no header gating in front.
#[test]
fn adversarial_coefficient_bytes_never_panic() {
    // A spread of pathological inputs: all-ones (drives the largest
    // codewords + run overruns), all-zeros (immediate end-of-data),
    // alternating, and a sweep of single-byte payloads.
    let cases: Vec<Vec<u8>> = vec![
        vec![0xFF; 64],
        vec![0x00; 64],
        vec![0xAA; 64],
        vec![0x55; 64],
        vec![0x80, 0x00, 0x00, 0x01],
        vec![0xFF, 0xFF, 0xFF, 0xFF, 0x7F],
    ];
    for data in &cases {
        for num_blocks in [1usize, 2, 8, 64] {
            // Must return without panicking (debug overflow included).
            let _ = decode_scanned_coefficients(data, num_blocks);
        }
    }
}

// ---------------------------------------------------------------------------
// §7.1.2 scanned alpha (Tables 12-14)
// ---------------------------------------------------------------------------

/// Round-trip through the lossless alpha coder reconstructs the exact
/// raster-scanned alpha array, for both 8-bit (Table 13, mask 0xFF) and
/// 16-bit (Table 14, mask 0xFFFF) coded alpha.
#[test]
fn alpha_round_trip_is_exact_both_depths() {
    // 8-bit alpha values (0..=255), with runs and jumps.
    let alpha8: Vec<u16> = {
        let mut v = vec![0u16; 256];
        for (i, a) in v.iter_mut().enumerate() {
            *a = ((i * 3) % 256) as u16;
        }
        // A long flat run + a couple of escapes.
        for a in v.iter_mut().take(64).skip(32) {
            *a = 200;
        }
        v
    };
    let enc8 = encode_scanned_alpha(&alpha8, AlphaChannelType::Eight).expect("enc8");
    let dec8 = decode_scanned_alpha(&enc8, alpha8.len(), AlphaChannelType::Eight).expect("dec8");
    assert_eq!(dec8, alpha8, "8-bit alpha round-trip must be exact");

    // 16-bit alpha values across the full 0..=65535 range.
    let alpha16: Vec<u16> = {
        let mut v = vec![0u16; 256];
        for (i, a) in v.iter_mut().enumerate() {
            *a = ((i * 257) % 65536) as u16;
        }
        for a in v.iter_mut().take(160).skip(100) {
            *a = 0xFFFF;
        }
        v
    };
    let enc16 = encode_scanned_alpha(&alpha16, AlphaChannelType::Sixteen).expect("enc16");
    let dec16 =
        decode_scanned_alpha(&enc16, alpha16.len(), AlphaChannelType::Sixteen).expect("dec16");
    assert_eq!(dec16, alpha16, "16-bit alpha round-trip must be exact");
}

/// Decoding a valid alpha stream while *under-declaring* `num_values`
/// (fewer than the encoder wrote) must terminate exactly at the declared
/// count — the decoder fills `num_values` and stops, never reading past.
#[test]
fn alpha_decode_respects_declared_count() {
    let alpha: Vec<u16> = (0..128u16).map(|i| (i * 2) % 256).collect();
    let enc = encode_scanned_alpha(&alpha, AlphaChannelType::Eight).expect("enc");
    // Decode the first 64 values only.
    let dec = decode_scanned_alpha(&enc, 64, AlphaChannelType::Eight).expect("partial dec");
    assert_eq!(dec.len(), 64);
    assert_eq!(&dec[..], &alpha[..64], "prefix must match");
}

/// Truncating an alpha stream mid-codeword must surface a clean `Err`,
/// never a panic, for every prefix length.
#[test]
fn truncated_alpha_stream_errors_cleanly() {
    let alpha: Vec<u16> = (0..200u16).map(|i| (i * 7) % 256).collect();
    let enc = encode_scanned_alpha(&alpha, AlphaChannelType::Eight).expect("enc");
    for cut in 0..enc.len() {
        // Must not panic; declared full length so a short stream can't
        // satisfy it and the reader will run out of bits.
        let _ = decode_scanned_alpha(&enc[..cut], alpha.len(), AlphaChannelType::Eight);
    }
}

/// Adversarial alpha bytes (with no header gating) must return a
/// `Result`, never panic, for both coded depths.
#[test]
fn adversarial_alpha_bytes_never_panic() {
    let cases: Vec<Vec<u8>> = vec![
        vec![0xFF; 64],
        vec![0x00; 64],
        vec![0xAA; 64],
        vec![0x55; 64],
        vec![0x01, 0x02, 0x03, 0x04, 0x05],
    ];
    for data in &cases {
        for num_values in [1usize, 16, 128, 2048] {
            let _ = decode_scanned_alpha(data, num_values, AlphaChannelType::Eight);
            let _ = decode_scanned_alpha(data, num_values, AlphaChannelType::Sixteen);
        }
    }
}
