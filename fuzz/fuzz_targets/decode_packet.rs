#![no_main]

//! Panic-free fuzz target for the full ProRes decode pipeline reached
//! through [`oxideav_prores::decoder::decode_packet`].
//!
//! Exercises the chain documented in `docs/video/prores/`:
//!
//! * RDD 36 §5.1 frame() outer framing (`frame_size` u32 BE + `'icpf'`
//!   magic).
//! * RDD 36 §6.1.1 frame_header() — `frame_header_size`, reserved byte,
//!   `bitstream_version` ("shall refuse" > 1), 4-byte encoder id, width /
//!   height u16 BE, packed chroma_format / interlace_mode byte (§6.1.1
//!   Table 2 rejects interlace_mode == 3), aspect_ratio / frame_rate_code,
//!   color_primaries / transfer_characteristic / matrix_coefficients,
//!   alpha_channel_type, and the `load_luma_qmat` / `load_chroma_qmat`
//!   flag bits driving the optional 64-byte luma + chroma quant matrices
//!   (every entry constrained to `2..=63` per the spec).
//! * RDD 36 §6.4 bitstream_version 0 acceptance rules — v0 streams must
//!   carry chroma_format == 4:2:2 AND alpha_channel_type == 0.
//! * RDD 36 §6.3 picture_header() + §6.3.6 slice_table() — per-row slice
//!   layout, per-slice `slice_size_table` entries, slice boundary
//!   arithmetic.
//! * RDD 36 §5.3 + §7.1.1 per-slice slice_header() + run/level/sign DC
//!   and AC coefficient coding, §7.1.2 + Table 12-14 alpha run-length
//!   / VLC coding for the 4:4:4 + alpha profiles (ap4h / ap4x).
//! * The dispatched decode path for ap4h / ap4x (4:4:4 chroma) + alpha
//!   plane (§5.3.3) AND the §5.1 ProRes RAW (`aprn` / `aprh`) refusal
//!   path which must produce a clear unsupported error rather than
//!   route through the RDD 36 parser.
//!
//! ## OOM cap
//!
//! `decoder::decode_packet` itself caps width × height at 32 768²
//! (MAX_DECODED_PIXELS), but at 32 768 × 32 768 a 4:4:4 + 12-bit plane
//! would still be 6 GB — far beyond a libFuzzer worker's budget. Peek at
//! offsets 16..18 (BE u16 width) and 18..20 (BE u16 height) of the wire
//! before invoking the decoder and bail out if the implied pixel count
//! exceeds a much smaller per-frame budget. libFuzzer learns the cap
//! quickly and steers mutations under it, so this doesn't starve the
//! corpus of useful decode coverage.

use libfuzzer_sys::fuzz_target;
use oxideav_prores::decoder;

/// Per-frame coded-pixel budget. At 256 × 256 a 4:4:4 + 12-bit YUV+alpha
/// plane set is 256 * 256 * 4 * 2 = 524 288 bytes, well under any
/// reasonable libFuzzer worker memory cap.
const MAX_CODED_PIXELS: u32 = 256 * 256;

/// Defence-in-depth raw-input cap. libFuzzer's default max input size
/// is already 4 KiB but stays runner-configurable, so re-enforce in the
/// harness against a runner that overrode `-max_len`.
const MAX_INPUT_BYTES: usize = 64 * 1024;

fuzz_target!(|data: &[u8]| {
    if data.len() > MAX_INPUT_BYTES {
        return;
    }

    // A well-formed frame is at least:
    //   4 (frame_size) + 4 ('icpf') + 20 (frame_header) = 28 bytes.
    // Smaller inputs are rejected cheaply by the parser; we still feed
    // them so the truncation-detection arms get coverage.
    if data.len() >= 20 {
        // Wire bytes 16..18 = width (BE u16), 18..20 = height (BE u16),
        // per the `frame() { frame_size:u32, 'icpf':u32, frame_header()
        // { hdr_size:u16, reserved:u8, ver:u8, enc_id:u32, width:u16,
        // height:u16, … } }` layout in RDD 36 §5.1 / §6.1.1.
        let width = u16::from_be_bytes([data[16], data[17]]) as u32;
        let height = u16::from_be_bytes([data[18], data[19]]) as u32;
        if width != 0 && height != 0 {
            let coded = width.saturating_mul(height);
            if coded > MAX_CODED_PIXELS {
                return;
            }
        }
    }

    let _ = decoder::decode_packet(data, None);
});
