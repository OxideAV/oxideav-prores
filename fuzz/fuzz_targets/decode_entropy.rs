#![no_main]

//! Panic-free fuzz target for the RDD 36 §7.1.1 run/level/sign DC + AC
//! coefficient coder, reached through
//! [`oxideav_prores::entropy::decode_scanned_coefficients`].
//!
//! Where `decode_packet` / `parse_headers` exercise the framing and
//! header arithmetic, this target drives the entropy coder *directly* on
//! adversarial bitstream bytes, with no header gating in front of it. The
//! decode path covered is the one documented in `docs/video/prores/`:
//!
//! * §7.1.1.3 first-DC order-5 exp-Golomb of `S(first_dc)`, then the
//!   per-block DC-difference codebook selection driven by the running
//!   `previous_dc_diff` magnitude (and its sign carry).
//! * §7.1.1.4 AC run / abs_level_minus_1 / sign loop, including the
//!   `run_codebook(previous_run)` and `level_codebook(previous_level_symbol)`
//!   adaptive codebook transitions and the Rice / exp-Golomb combo reader
//!   underneath them.
//! * The truncation / overrun arms: a `run` that walks the scanned-array
//!   cursor off the end must produce a clear `Err`, never an out-of-bounds
//!   index or a debug integer overflow; a bitstream that ends mid-codeword
//!   must surface the reader's end-of-data error rather than panic.
//!
//! ## num_blocks budget
//!
//! `decode_scanned_coefficients` allocates `vec![0i32; num_blocks * 64]`
//! up front (the §7.1.1 slice-scan array), so an attacker-chosen
//! `num_blocks` is a memory amplification knob. The decoder itself guards
//! the `* 64` against `usize` overflow, but the harness still caps
//! `num_blocks` to a small slice-sized value so a libFuzzer worker never
//! tries to commit a multi-gigabyte allocation that the *spec-legal* path
//! would also reject at the slice-layout stage. A real RDD 36 slice is at
//! most 8 macroblocks wide; at 4:4:4 + alpha that is well under this cap.

use libfuzzer_sys::fuzz_target;
use oxideav_prores::entropy;

/// Defence-in-depth raw-input cap (libFuzzer default `-max_len` is 4 KiB
/// but stays runner-overridable).
const MAX_INPUT_BYTES: usize = 16 * 1024;

/// Per-call block-count cap. A maximal RDD 36 macroblock at 4:4:4 carries
/// 4 luma + 2 + 2 chroma = 8 blocks; a maximal 8-wide slice is 8 MBs, so
/// 8 * 8 = 64 blocks covers the spec-legal envelope with margin. The cap
/// keeps the harness allocation (`num_blocks * 64 * 4` bytes) bounded.
const MAX_BLOCKS: usize = 256;

fuzz_target!(|data: &[u8]| {
    if data.len() > MAX_INPUT_BYTES {
        return;
    }

    // Derive `num_blocks` from the input's own first byte so libFuzzer can
    // steer mutations across the per-block DC loop length and the AC
    // array-length overrun arm. `num_blocks == 0` is fed through too: the
    // first-DC read still runs, exercising the empty-slice early path.
    let tag = data.first().copied().unwrap_or(0);
    let num_blocks = (tag as usize) % (MAX_BLOCKS + 1);

    // The remaining bytes are the entropy-coded payload. Slicing off byte
    // 0 keeps the block-count knob from doubling as coefficient data, so
    // the corpus minimiser converges on independent (count, payload) pairs.
    let payload = if data.is_empty() { data } else { &data[1..] };

    let _ = entropy::decode_scanned_coefficients(payload, num_blocks);
});
