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
//! * `BitDepth::Sixteen` -> 16-bit LE `u16` output (b = 16, scale 128,
//!   all word bits significant).
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
//!
//! One extra tag bit routes the same input through
//! `decode_packet_with_format` with the alpha-typed
//! `Yuva4(2|4)4P{,10Le,12Le,16Le}` surface at the tagged depth instead
//! — covering the guaranteed-4-plane contract (the §7.5.2 conversion
//! of coded alpha to the surface depth, and the synthesised
//! fully-opaque plane when the attacker's frame header declares
//! `alpha_channel_type == 0`) — and another selects the §7.5.1
//! `OutputRange::Video` clamp.

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

    // Derive the requested output surface from the input's own first
    // byte so libFuzzer can steer mutations toward every request shape.
    // The only state the harness reads from `data` is the low 5 bits of
    // byte 0, fed back as an attacker-controlled output-format request,
    // exactly as a container demuxer would do when it has its own
    // `pixel_format` declaration:
    //
    // * bits 0-1 — bit depth (8 / 10 / 12 / 16);
    // * bit 2    — chroma format (4:2:2 / 4:4:4);
    // * bit 3    — route through the alpha-typed
    //   `Yuva4(2|4)4P{,10Le,12Le,16Le}` surface of
    //   `decode_packet_with_format` (4 planes guaranteed at the tagged
    //   depth) instead of `decode_packet_with_options`;
    // * bit 4    — §7.5.1 `OutputRange::Video` clamp instead of `Full`.
    let tag = data.first().copied().unwrap_or(0);
    let bit_depth = match tag & 0x3 {
        0 => decoder::BitDepth::Eight,
        1 => decoder::BitDepth::Ten,
        2 => decoder::BitDepth::Twelve,
        _ => decoder::BitDepth::Sixteen,
    };
    let y444 = (tag >> 2) & 1 != 0;
    let chroma_format = if y444 {
        frame::ChromaFormat::Y444
    } else {
        frame::ChromaFormat::Y422
    };
    let range = if (tag >> 4) & 1 == 0 {
        decoder::OutputRange::Full
    } else {
        decoder::OutputRange::Video
    };

    if (tag >> 3) & 1 == 0 {
        let _ = decoder::decode_packet_with_options(
            data,
            None,
            Some((bit_depth, chroma_format)),
            range,
        );
    } else {
        use oxideav_core::PixelFormat;
        // `decoder::BitDepth` is `#[non_exhaustive]`, so a `(bool, BitDepth)`
        // tuple can never be matched exhaustively by variant alone. The
        // harness synthesises `bit_depth` from the same `tag & 0x3` two-bit
        // request field used above (0/1/2 -> 8/10/12, everything else -> 16),
        // so route the surface selection off that request field directly:
        // its `_` arm covers the 16-bit request and any future depth the
        // request encoding might grow into, with no unreachable phantom arm.
        let pf = match (y444, tag & 0x3) {
            (false, 0) => PixelFormat::Yuva422P,
            (false, 1) => PixelFormat::Yuva422P10Le,
            (false, 2) => PixelFormat::Yuva422P12Le,
            (false, _) => PixelFormat::Yuva422P16Le,
            (true, 0) => PixelFormat::Yuva444P,
            (true, 1) => PixelFormat::Yuva444P10Le,
            (true, 2) => PixelFormat::Yuva444P12Le,
            (true, _) => PixelFormat::Yuva444P16Le,
        };
        let _ = decoder::decode_packet_with_format(data, None, Some(pf), range);
    }
});
