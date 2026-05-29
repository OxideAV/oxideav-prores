#![no_main]

//! Panic-free fuzz target for the four public RDD-36 header parsers
//! exposed on [`oxideav_prores::frame`]:
//!
//! * [`frame::parse_frame`]          — RDD 36 §5.1 outer framing
//!   (`frame_size` u32 BE + `'icpf'` magic + frame_header()).
//! * [`frame::parse_frame_header`]   — RDD 36 §6.1.1 frame_header(),
//!   including `frame_header_size`, reserved byte, `bitstream_version`
//!   ("shall refuse" > 1), 4-byte encoder id, width / height u16 BE,
//!   packed chroma_format / interlace_mode byte (§6.1.1 Table 2 rejects
//!   `interlace_mode == 3`), aspect_ratio / frame_rate_code, the three
//!   colour-spec bytes, `alpha_channel_type`, the `load_luma_qmat` /
//!   `load_chroma_qmat` flag bits, and the optional 64-byte luma +
//!   chroma quant matrices (each entry constrained to `2..=63` per
//!   §6.1.1).
//! * [`frame::parse_picture_header`] — RDD 36 §6.3 picture_header(),
//!   including `picture_header_size` (≥ 8), `picture_size` u32 BE,
//!   `deprecated_number_of_slices` u16 BE, and the packed
//!   `log2_desired_slice_size_in_mb` field.
//! * [`frame::parse_slice_header`]   — RDD 36 §5.3 slice_header(),
//!   including the `slice_header_size`, `quantization_index` (must be
//!   in `1..=224`), and the `coded_size_of_{y,cb,cr}_data` u16 BE fields
//!   for both the with-alpha and without-alpha shapes.
//!
//! These entry points are reachable from any application that bypasses
//! the top-level [`decoder::decode_packet`] path — for example a tool
//! inspecting `.mov` sample bytes that wants to extract just the header
//! fields. They share the same attacker-controlled byte surface as the
//! full decode path, but are cheaper to drive (no IDCT, no entropy
//! coder, no buffer allocation), so libFuzzer can explore the header
//! arithmetic and quant-matrix loading branches at a higher rate per
//! second than the existing `decode_packet` / `decode_packet_with_depth`
//! harnesses.
//!
//! ## Coverage shape
//!
//! `data` is partitioned into four independent slices, each fed to one
//! of the four parsers, so a single fuzz iteration exercises all four
//! entry points without one entry point's truncation early-out starving
//! the others. The split offsets come from the input's own first byte
//! so libFuzzer can steer mutations toward useful (length, contents)
//! pairs. The `has_alpha` flag fed to `parse_slice_header` is derived
//! from a second input byte so both 6-byte (no-alpha) and 8-byte
//! (with-alpha) slice_header shapes get coverage.
//!
//! ## OOM cap
//!
//! The header parsers do not allocate per-pixel buffers themselves, so
//! the only memory risk here is libFuzzer's own corpus retention. Cap
//! the raw input at 16 KiB, well under the default `-max_len` of 4 KiB
//! that libFuzzer ships with — this is defence in depth against a
//! runner override.

use libfuzzer_sys::fuzz_target;
use oxideav_prores::frame;

const MAX_INPUT_BYTES: usize = 16 * 1024;

/// Split `data` into `n` roughly equal parts using `seed` as a rotation
/// so libFuzzer can steer where the parser-specific contents land.
fn split_n(data: &[u8], n: usize, seed: u8) -> Vec<&[u8]> {
    if data.is_empty() || n == 0 {
        return vec![data; n];
    }
    let base = data.len() / n;
    let offset = (seed as usize) % data.len().max(1);
    let mut out = Vec::with_capacity(n);
    let mut cursor = offset;
    for i in 0..n {
        let start = cursor % data.len();
        // Last chunk takes whatever remains so the four parsers between
        // them cover every byte at least once.
        let end = if i + 1 == n {
            (start + data.len() - (n - 1) * base).min(data.len())
        } else {
            (start + base).min(data.len())
        };
        out.push(&data[start..end]);
        cursor = end;
    }
    out
}

fuzz_target!(|data: &[u8]| {
    if data.len() > MAX_INPUT_BYTES {
        return;
    }

    // Use the first two bytes (when present) as control knobs:
    //   byte 0 → rotation seed for split_n
    //   byte 1 → bit 0 = has_alpha for parse_slice_header
    let seed = data.first().copied().unwrap_or(0);
    let has_alpha = data.get(1).copied().unwrap_or(0) & 1 == 1;

    let chunks = split_n(data, 4, seed);

    let _ = frame::parse_frame(chunks[0]);
    let _ = frame::parse_frame_header(chunks[1]);
    let _ = frame::parse_picture_header(chunks[2]);
    let _ = frame::parse_slice_header(chunks[3], has_alpha);
});
