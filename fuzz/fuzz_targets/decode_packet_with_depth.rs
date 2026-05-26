#![no_main]

//! Panic-free fuzz target for [`oxideav_prores::decoder::decode_packet_with_depth`],
//! the caller-bit-depth path that the registry uses when a downstream
//! container exposes a `pixel_format` of `Yuv422P10Le` / `Yuv422P12Le` /
//! `Yuv444P10Le` / `Yuv444P12Le`.
//!
//! Exercises the same parse chain as `decode_packet.rs` (RDD 36 §5.1
//! frame(), §6.1.1 frame_header() + optional 64-byte luma + chroma
//! quant matrices, §6.3 picture_header() + slice_table(), §5.3 + §7.1.1
//! per-slice slice_header() + run/level/sign coefficient coder, §7.1.2
//! alpha run-length / VLC code), but additionally drives the §7.5.1
//! per-bit-depth sample output formatter:
//!
//! * `BitDepth::Eight` -> 8-bit planar `u8` output, no level shift.
//! * `BitDepth::Ten` -> 10-bit LE `u16` output, `(sample + 256) >> 1`
//!   level shift (b = 10).
//! * `BitDepth::Twelve` -> 12-bit LE `u16` output, `(sample + 64) >> 3`
//!   level shift (b = 12).
//!
//! Combined with the §5.3.3 / §7.1.2 alpha emit path (which runs
//! whenever the frame_header advertises `alpha_channel_type != 0`), the
//! `with_depth` overload widens the per-sample post-IDCT arithmetic
//! surface that the plain `decode_packet` 8-bit path skips. A header
//! requesting 12-bit + 4:4:4 against a 4:2:2 stream — or vice versa —
//! is *not* automatically rejected upstream, so the harness derives
//! both the bit-depth tag and the chroma-format tag from the input's
//! own bytes and lets the decoder validate / cross-check them against
//! the parsed frame header.

use libfuzzer_sys::fuzz_target;
use oxideav_prores::{decoder, frame};

const MAX_CODED_PIXELS: u32 = 256 * 256;
const MAX_INPUT_BYTES: usize = 64 * 1024;

fuzz_target!(|data: &[u8]| {
    if data.len() > MAX_INPUT_BYTES {
        return;
    }

    if data.len() >= 20 {
        let width = u16::from_be_bytes([data[16], data[17]]) as u32;
        let height = u16::from_be_bytes([data[18], data[19]]) as u32;
        if width != 0 && height != 0 {
            let coded = width.saturating_mul(height);
            if coded > MAX_CODED_PIXELS {
                return;
            }
        }
    }

    // Derive a (BitDepth, ChromaFormat) tag from the input's own first
    // byte so libFuzzer can steer mutations toward all six requested
    // (bit-depth, chroma) combinations. The 7th-12th slots circle back
    // to the same combos; the only state the harness reads from `data`
    // is the low 4 bits of byte 0, which is fed back as an attacker-
    // controlled output-format request, exactly as a container demuxer
    // would do when it has its own `pixel_format` declaration.
    let tag = data.first().copied().unwrap_or(0);
    let bit_depth = match tag & 0x3 {
        0 => decoder::BitDepth::Eight,
        1 => decoder::BitDepth::Ten,
        _ => decoder::BitDepth::Twelve,
    };
    let chroma_format = if (tag >> 2) & 1 == 0 {
        frame::ChromaFormat::Y422
    } else {
        frame::ChromaFormat::Y444
    };

    let _ = decoder::decode_packet_with_depth(data, None, Some((bit_depth, chroma_format)));
});
