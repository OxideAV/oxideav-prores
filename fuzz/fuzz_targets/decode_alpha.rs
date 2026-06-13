#![no_main]

//! Panic-free fuzz target for the RDD 36 §7.1.2 / Table 12-14 alpha
//! run-length + difference VLC coder, reached through
//! [`oxideav_prores::alpha::decode_scanned_alpha`].
//!
//! This is the ap4h / ap4x (4:4:4 + alpha) entropy path that the §7.1.1
//! coefficient coder does *not* touch. The decode path covered is the one
//! documented in `docs/video/prores/`:
//!
//! * §7.1.2 per-sample alpha-difference read — the Table 13 (8-bit) /
//!   Table 14 (16-bit) small-codeword form *and* the escape FLC form,
//!   with the `is_modulo` mask-wrap applied to the running `previous_alpha`
//!   accumulator (mask = 0xFF for `Eight`, 0xFFFF for `Sixteen`).
//! * The run reader: a decoded `run` of zero must surface a clear `Err`
//!   ("zero run") and a `run` that walks past `num_values` must surface
//!   the "run overruns alphaValues array" `Err`, never an out-of-bounds
//!   store or a debug overflow on the `n + run` cursor arithmetic.
//! * Bitstreams that end mid-codeword must surface the reader's
//!   end-of-data error rather than panic.
//!
//! ## num_values budget
//!
//! `decode_scanned_alpha` allocates `vec![0u16; num_values]` up front, so
//! `num_values` is the memory-amplification knob. Cap it to a small
//! slice-sized value: a maximal RDD 36 alpha slice is 8 MBs × 256
//! samples/MB = 2048 samples at 4:4:4, which this cap covers with margin.
//! The harness derives both `num_values` and the 8-/16-bit
//! `AlphaChannelType` from the input's own bytes so libFuzzer can steer
//! mutations across both alpha encodings and the run-overrun arm.

use libfuzzer_sys::fuzz_target;
use oxideav_prores::alpha::{self, AlphaChannelType};

const MAX_INPUT_BYTES: usize = 16 * 1024;

/// Per-call sample-count cap (`num_values * 2` bytes allocated). A maximal
/// 8-wide alpha slice is 8 * 256 = 2048 samples; this cap covers it.
const MAX_VALUES: usize = 4096;

fuzz_target!(|data: &[u8]| {
    if data.len() > MAX_INPUT_BYTES {
        return;
    }

    // byte 0 low bit  → AlphaChannelType (Eight vs Sixteen)
    // bytes 1..3      → num_values (BE u16, capped)
    let tag = data.first().copied().unwrap_or(0);
    let act = if tag & 1 == 0 {
        AlphaChannelType::Eight
    } else {
        AlphaChannelType::Sixteen
    };
    let raw_count = u16::from_be_bytes([
        data.get(1).copied().unwrap_or(0),
        data.get(2).copied().unwrap_or(0),
    ]) as usize;
    let num_values = raw_count % (MAX_VALUES + 1);

    // Remaining bytes are the alpha-coded payload; slicing off the three
    // control bytes keeps the knobs from doubling as alpha data.
    let payload = if data.len() > 3 { &data[3..] } else { &[][..] };

    let _ = alpha::decode_scanned_alpha(payload, num_values, act);
});
