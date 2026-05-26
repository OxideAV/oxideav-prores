#![no_main]

//! Panic-free fuzz target for [`oxideav_prores::frame::parse_frame_header`],
//! the §6.1.1 frame_header() parser in isolation (no picture / slice /
//! coefficient decode work behind it).
//!
//! The existing `decode_packet` and `decode_packet_with_depth` targets
//! both drive the full RDD 36 pipeline. That parses every layer for
//! every input, which is the right test for end-to-end "can the public
//! decode API panic on attacker bytes?" but it lets libFuzzer iterate
//! at only a few thousand exec/s and most of those execs reject the
//! input mid-pipeline, never reaching the deeper-arm coverage. A
//! parser-level target on the header alone runs orders of magnitude
//! faster (~10^5+ exec/s versus ~10^3 for the full pipeline) and lets
//! libFuzzer maximise corpus-discovery on the highest-value attack
//! surface: the byte-counted, version-gated, quant-matrix-loaded
//! §6.1.1 header.
//!
//! ## Why the header is the highest-value parser-level surface
//!
//! RDD 36 §6.1.1 alone encodes every "decoder shall refuse" clause that
//! protects the rest of the pipeline from a malformed sample:
//!
//! * `frame_header_size` (u16 BE at offset 0) — drives a cursor into
//!   `data`; truncation (`< 20`) and overrun (`> data.len()`) must both
//!   be rejected without out-of-bounds reads.
//! * `bitstream_version` (u8 at offset 3) — §6.1.1 "shall abort if it
//!   encounters a bitstream with an unsupported bitstream_version
//!   value"; the spec describes versions 0 and 1 so any byte > 1 must
//!   be rejected.
//! * `chroma_format` (top 2 bits of byte 12) — §6.1.1 Table 1 defines
//!   2 = 4:2:2 and 3 = 4:4:4; codes 0 and 1 are reserved.
//! * `interlace_mode` (bits 3-2 of byte 12) — §6.1.1 Table 2 marks
//!   value 3 reserved; "shall refuse".
//! * `alpha_channel_type` (low 4 bits of byte 17) — §6.4 v0-stream rule
//!   cross-references this against `bitstream_version` and
//!   `chroma_format`. A v0 stream that carries 4:4:4 chroma OR any
//!   alpha is malformed; both arms must be reached.
//! * `load_luma_quantization_matrix` / `load_chroma_quantization_matrix`
//!   (low 2 bits of byte 19) — drive optional 64-byte payload reads.
//!   Truncation (`data.len() < cursor + 64`) and per-entry range
//!   (`2..=63`, §6.1.1 / §7.3) must both be rejected.
//! * The combined branch `load_chroma == 0 && load_luma == 1` — §6.1.1
//!   "the luma matrix shall be used" — exercises the qmat-aliasing
//!   path.
//! * `cursor > frame_header_size` overrun check after qmat reads.
//!
//! Every bullet above is an arm libFuzzer needs to discover and
//! distinguish. At ~10^5 exec/s the corpus saturates each arm in
//! seconds rather than minutes, so this target gives the qmat range +
//! v0 cross-check + size-overrun paths the focused coverage they
//! deserve.
//!
//! ## OOM cap
//!
//! `parse_frame_header` allocates no plane buffers — it only fills two
//! 64-byte stack qmats and returns a `FrameHeader` value. So unlike the
//! full-decode targets there is no width × height OOM risk to gate
//! against; the only input cap is libFuzzer's `-max_len` runner setting,
//! re-enforced in the harness against a runner that overrode it.

use libfuzzer_sys::fuzz_target;
use oxideav_prores::frame;

/// Defence-in-depth raw-input cap. libFuzzer's default max input size
/// is already 4 KiB but stays runner-configurable; re-enforce in the
/// harness against a runner that overrode `-max_len`. A frame_header
/// with both qmats loaded is at most 20 + 64 + 64 = 148 bytes plus the
/// reserved trailing padding `frame_header_size` may declare, so 4 KiB
/// is already overkill — keep the cap small to maximise execs/s.
const MAX_INPUT_BYTES: usize = 4 * 1024;

fuzz_target!(|data: &[u8]| {
    if data.len() > MAX_INPUT_BYTES {
        return;
    }

    // The parser contract: return `Result` rather than panic, integer-
    // overflow (debug), index out of bounds, or read past the end of
    // `data`. We deliberately do not assert anything about the parsed
    // header's values — that's the job of the unit tests under
    // `src/frame.rs`. The fuzzer's only job is to find an input that
    // breaks the panic-free contract.
    let _ = frame::parse_frame_header(data);
});
