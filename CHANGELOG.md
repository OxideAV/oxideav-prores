# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- ffmpeg cross-decode acceptance test for the interlaced encode path.
  `tests/ffmpeg_cross_decode.rs` (5 cases: apch TFF, apch BFF, apcn
  TFF, apcn BFF, apch at 128×96) encodes a synthetic field-distinct
  4:2:2 frame via `encode_frame_interlaced`, wraps the resulting
  `icpf`-prefixed packet in a minimal QuickTime MOV (substituting our
  payload into an ffmpeg-generated template MOV: `mdat` size, embedded
  `frame_size`, and `stsz` sample-size are patched; `stco` first-chunk
  offset remains correct because the patched `mdat` keeps its file
  offset), then asks ffmpeg's `prores_ks` decoder to decode the
  resulting container to raw 10-bit YUV. Measured luma PSNR: 64.17 dB
  on the 64×48 fixtures and 64.18 dB at 128×96 — well above the
  30 dB acceptance bar. Each case additionally verifies that the
  encoded frame header reports the requested `interlace_mode` and
  `picture_count == 2`, and that the decoded even-row luma sum is
  greater than the odd-row luma sum (defends against a swapped
  TFF/BFF field-pair tag). Tests skip gracefully when `ffmpeg` is
  missing.
- Decoder enforcement of RDD 36 §6.4 bitstream-version compatibility
  rules + the qmat / interlace_mode "decoder shall refuse" clauses in
  §6.1.1. `frame::parse_frame_header` now rejects:
  * `bitstream_version > 1` (was already rejected; error message
    extended with the spec reference and version-range hint),
  * a `bitstream_version == 0` stream that carries `chroma_format = 3`
    (4:4:4) — v0 is restricted to 4:2:2 per §6.4,
  * a `bitstream_version == 0` stream that carries any non-zero
    `alpha_channel_type` — v0 forbids encoded alpha per §6.4,
  * `interlace_mode == 3` (RDD 36 §6.1.1 Table 2 marks the value
    reserved),
  * any `luma_quantization_matrix` / `chroma_quantization_matrix` entry
    outside `2..=63` (RDD 36 §6.1.1: "Each entry of the matrix will be
    in the range 2, 3, …, 63"). A custom matrix that violates the
    range cannot be inverse-quantized per §7.3 (the qScale * weight
    product would be 0 or exceed 32256), so a conforming decoder must
    refuse it.
  Mirror `debug_assert!` guards in `frame::write_frame_with_meta`
  prevent the encoder from synthesising any of the same illegal
  combinations. New integration tests in `tests/spec_validation.rs`
  (15 cases) cover every accept + reject combination plus
  v0/v1-aware self-roundtrip through `write_frame` /
  `write_frame_with_alpha`.
- Explicit encoder profile override via
  `encoder::EncoderConfig::with_profile(profile)` /
  `encoder::EncoderConfig::for_profile(profile)`. When set, the supplied
  [`frame::Profile`] is honoured verbatim — the `bit_rate` →
  profile heuristic in `pick_profile` is bypassed. Lets callers pin a
  specific profile that the heuristic would not select, e.g.
  **4444 XQ at bit rates below 400 Mbit/s** (where the heuristic
  defaults to 4444) or any 4:2:2 profile without a `bit_rate` hint.
  The override's `chroma_format` must match the requested
  `PixelFormat`; mismatches return `Error::invalid` at encoder
  construction. Bitstream is unaffected — RDD 36 §5.1.1 carries only
  `chroma_format`, not a profile — so the override only changes the
  default `quantization_index` (per
  `Profile::default_quant_index`). New integration tests in
  `tests/profile_selection.rs` (6 cases): Proxy<HQ size
  monotonicity, 4444<XQ size monotonicity, XQ override at low
  bit_rate, chroma-mismatch rejection, qi-override interplay
  (byte-equal bitstream when qi is pinned), and all-six-profiles
  decode-clean sanity.
- Multi-frame rate-control sequence regression
  `tests/rate_control.rs::rate_ctrl_multi_frame_sequence_average_hits_target`
  — encodes 8 frames of evolving synthetic content through the rate
  controller and asserts the per-frame size **average** lands within
  `RATE_CTRL_TOLERANCE` (±5 %) of the per-frame target. Covers the
  downstream-container view of the rate controller (where individual
  frames may transiently miss tolerance but the long-run average is
  what matters for CBR muxing).

## [0.0.8](https://github.com/OxideAV/oxideav-prores/compare/v0.0.7...v0.0.8) - 2026-05-06

### Other

- reframe FFI claim — HW-engine crates use OS FFI by necessity
- drop dead `linkme` dep
- fix clippy unnecessary_cast (br is already u64)
- two-pass per-frame rate control to ±5 % of nominal ProRes bitrates
- registry calls: rename make_decoder/make_encoder → first_decoder/first_encoder
- auto-register via oxideav_core::register! macro (linkme distributed slice)
- unify entry point on register(&mut RuntimeContext) ([#502](https://github.com/OxideAV/oxideav-prores/pull/502))

### Added

- Two-pass per-frame rate control (`EncoderConfig::with_rate_control`):
  binary-searches `quantization_index` over up to `RATE_CTRL_MAX_PASSES`
  (10) trial encodes per frame to land within `RATE_CTRL_TOLERANCE` (±5 %)
  of the per-frame byte target derived from `CodecParameters::bit_rate /
  frame_rate`. Works across all six ProRes profiles (422 Proxy/LT/Standard/HQ
  and 4444/4444 XQ); degrades gracefully to single-pass when `bit_rate` or
  `frame_rate` is missing. Unreachable targets (target > max achievable at
  qi=1, or target < min achievable at qi=224) return the best candidate
  without panicking. Integration tests in `tests/rate_control.rs` confirm
  ±4–5 % accuracy on both 422 and 4444 at 64×48 and 128×96.
- `EncoderConfig::rate_control` field + `EncoderConfig::with_rate_control()`
  builder; `RATE_CTRL_MAX_PASSES` and `RATE_CTRL_TOLERANCE` public constants.
- `encode_frame_with_rate_control` internal function (not public API)
  implements the binary-search loop.

## [0.0.7](https://github.com/OxideAV/oxideav-prores/compare/v0.0.6...v0.0.7) - 2026-05-04

### Other

- decoder + encoder: alpha plane reads/writes full padded MB-row height
- write frame_rate_code + descriptive metadata into header
- expose quantization_index on EncoderConfig
- derive frame count from data, drop hard-coded n_frames

### Fixed

- Alpha decode against streams whose picture height is not a multiple
  of `MB_SIDE_PX` (16). The decoder now reads `mbs_this_slice * 16 * 16`
  alpha samples per slice (the padded macroblock-row size) regardless
  of visible-row clipping and pastes only the visible rows into the
  cropped output plane, matching the encoder side. The encoder
  symmetrically writes the full padded MB-row alpha (clamped edge
  pixels for partially-visible bottom rows) so self-roundtrip stays
  bit-stable. Pre-fix, the docs-corpus `4444-with-alpha` fixture (1080
  rows = 67.5 MB rows) failed with
  `InvalidData("prores alpha: run overruns alphaValues array")` on
  every slice in the last MB row — the in-tree decoder is now
  interop-clean against ffmpeg-produced ap4h+alpha. RDD 36 §7.5.2.

### Added

- Corpus test driver gains `Tier::DecodesCleanly` (every container
  frame must return Ok from `decode_packet_with_depth` — turns silent
  decode errors into CI-red failures) and `Tier::MinPsnr { min_y_psnr_db,
  min_uv_psnr_db }` (decode + reference-PSNR floor when expected.yuv
  is shipped). Promotions: `4444-with-alpha`, `interlaced-tff`,
  `pal-1080i50`, `4444-1920x1080`, `4444xq-1920x1080`,
  `proxy-1280x720`, `lt-1280x720`, `sq-1920x1080`, `hq-1920x1080` →
  `DecodesCleanly`; `tiny-320x240-sq`, `mxf-container` →
  `MinPsnr { 60.0, 60.0 }` (measured 82.65 / 81.62 dB, ample headroom
  above the floor). Promotes 9 corpus fixtures out of "decoder errors
  out silently" and 2 out of "no quality gate" status.
- `tests/roundtrip.rs::roundtrip_4444_with_alpha_16bit` — focused
  regression for the §7.1.2 16-bit alpha path (8-bit YUV body with
  16-bit alpha; round-trip max diff ≤ 1 LSB after 16-bit→8-bit
  promotion via `round((255 * v) / 65535)`).
- `tests/roundtrip.rs::roundtrip_4444_with_alpha_non_mb_aligned_height`
  — focused regression for the bug above (24-row picture, alpha must
  decode without the "run overruns" error and round-trip identically
  on the visible 24 rows).

- `frame::FrameMeta` carries the descriptive frame-header fields
  (`aspect_ratio_information`, `frame_rate_code`, `color_primaries`,
  `transfer_characteristic`, `matrix_coefficients` per RDD 36 §5.1.1
  / §6.2). New field `encoder::EncoderConfig::meta: Option<FrameMeta>`
  + builder method `EncoderConfig::with_meta(meta)` plus a new
  `frame::write_frame_with_meta` writer. When the field is `None`,
  `make_encoder_with_config` derives `frame_rate_code` from
  `CodecParameters::frame_rate` via the new helper
  `frame::frame_rate_code_from_rational` (Table 4 — 24/1.001, 24, 25,
  30/1.001, 30, 50, 60/1.001, 60, 100, 120/1.001, 120). `Some(meta)`
  overrides every field verbatim. The legacy `write_frame` /
  `write_frame_with_alpha` shims still write all metadata at 0
  ("unknown") so byte-for-byte compatibility is preserved. New tests:
  6 in `tests/frame_meta.rs` (param derivation, override, unknown
  rate, missing rate, byte-compat) + 5 unit tests in `frame::tests`
  (Table 4 mapping for every named rate, unnormalised fractions,
  unknown rates, `FrameMeta` helpers, full round-trip).
- Configurable per-slice `quantization_index` for the encoder (RDD 36
  §7.3 / Table 15). New field
  `encoder::EncoderConfig::quantization_index: Option<u8>` and
  builder method `EncoderConfig::with_quantization_index(qi)`. `None`
  preserves the per-profile default (`8 / 6 / 4 / 2 / 2 / 1` for
  Proxy / LT / Standard / HQ / 4444 / 4444 XQ); `Some(qi)` overrides
  it. Validated against the spec range `1..=224` at encoder
  construction. Lets callers override quality without remapping
  `bit_rate` to a different profile (e.g. "Proxy at qi=2 for archival
  intermediates"). New integration tests in `tests/quant_index.rs`
  (7 cases) cover packet-size monotonicity, PSNR monotonicity, range
  validation, profile-override interplay, and the equivalence
  `EncoderConfig{qi=None} == EncoderConfig{qi=Some(profile_default)}`.

### Changed

- `tests/docs_corpus`: drop hard-coded `n_frames: 2` per fixture; the
  driver now derives the comparable count from the actual container
  (`extract_prores_frames(...).len()`) and the actual reference
  (`expected.yuv.len() / frame_bytes`). Restores green CI on the two
  fixtures shipping a real `expected.yuv` (`tiny-320x240-sq` and
  `mxf-container`), each of which actually carries one frame, not the
  two the trace.txt claimed.

## [0.0.6](https://github.com/OxideAV/oxideav-prores/compare/v0.0.5...v0.0.6) - 2026-05-03

### Other

- wire docs/video/prores/ fixture corpus as docs_corpus.rs
- replace never-match regex with semver_check = false
- migrate to centralized OxideAV/.github reusable workflows
- round 5 — configurable perceptual quantisation matrices
- round 4 — interlaced field pictures + spec §7.5.1 normalization
- round 3 — alpha plane, 12-bit pixel output, ap4h interop
- round 2 — 10-bit Yuv422P10Le / Yuv444P10Le pixel output
- adopt slim VideoFrame shape
- pin release-plz to patch-only bumps

### Added

- Configurable per-component quantisation matrices for the encoder
  (RDD 36 §5.3.4 + §6.3.7 + §7.3). New public types:
  `quant::QuantMatrices` (luma + chroma `[u8; 64]` weights, both
  validated against the spec range `2..=63`) and
  `encoder::EncoderConfig` (carries the optional matrices). Reference
  perceptual matrices `quant::PERCEPTUAL_LUMA_QMAT` /
  `PERCEPTUAL_CHROMA_QMAT` are JPEG K.1/K.2 normalised so DC weight
  is 2 (twice as fine as the flat all-4s default), with HF weights
  rising to ~12 (luma) / 12 (chroma). New entry points:
  `encoder::make_encoder_with_config(params, EncoderConfig)`,
  `encoder::EncoderConfig::{flat, perceptual, with_quant_matrices}`,
  `encoder::encode_frame_with_qmats(...)`. The encoder writes the
  matrices into the frame header (setting `load_luma_qmat = 1` and,
  when `chroma != luma`, `load_chroma_qmat = 1`) so any RDD 36
  decoder picks them up; ffmpeg cross-decodes the resulting bitstream
  cleanly. On broadband content the perceptual matrices cut packet
  size by 20-25% at every `quantization_index` from 2 to 16; PSNR
  trades off because flat is provably PSNR-optimal under uniform
  quantisation, but perceptual quality (JPEG-style CSF rolloff) is
  preserved. New integration tests in `tests/perceptual_quant.rs`
  (7 cases) cover A/B sizes, header roundtrip, ffmpeg cross-decode,
  and config validation.
- Interlaced encode + decode per RDD 36 §5.1 / §6.2 / §7.5.3. The frame
  header `interlace_mode` field is honoured at decode time (mode 1 =
  top-field-first, mode 2 = bottom-field-first) and the two field
  pictures are split / reassembled per §7.5.3 (top field = even rows,
  bottom = odd rows). Each field picture uses the interlaced block
  scan from §7.2 Figure 5 (already provided as `BLOCK_SCAN_INTERLACED`).
  New encoder entry point: `encoder::encode_frame_interlaced`.
- ffmpeg interop tests for `apch` / `apcn` interlaced fixtures
  (`prores_ks` `-flags +ildct -top {0,1}`) reach ≥ 40 dB Y PSNR
  against the raw `testsrc` reference.
- Internal roundtrip tests for top-field-first, bottom-field-first,
  and odd-vertical-size interlaced frames.

### Changed

- Pixel-sample / IDCT level shift is now spec-compliant (RDD 36 §7.5.1):
  encoder maps `s → v = s / 2^(b-9) − 256` and decoder maps
  `v → s = clamp(round(2^b × (v + 256) / 512))`. The previous
  off-by-one normalization (centred on 128 instead of 256, scaled by
  `2^(b-8)` instead of `2^(b-9)`) was self-consistent but cut visual
  contrast in half on third-party-encoded streams. Internal roundtrips
  are unaffected because both ends moved together; ffmpeg-encoded
  fixtures now reproduce the source pixel range (e.g. SMPTE-legal
  64–940 for 10-bit black/white).

## [0.0.5](https://github.com/OxideAV/oxideav-prores/compare/v0.0.4...v0.0.5) - 2026-04-25

### Other

- drop oxideav-codec/oxideav-container shims, import from oxideav-core
- blocked memo for RDD 36 entropy layer (no spec in docs/)
- static FourCC -> CodecId helpers + ffmpeg black-box test

### Added

- `PRORES_FOURCCS` public constant listing the six canonical MP4/MOV
  sample-entry FourCCs (`apco`, `apcs`, `apcn`, `apch`, `ap4h`, `ap4x`).
- `codec_id_for_fourcc()` helper for static fourcc → `CodecId` mapping,
  used by demuxers that cannot consult a `CodecRegistry`.
- `profile_for_fourcc()` helper returning the matching `frame::Profile`
  for a given FourCC.
- Black-box integration test (`tests/mp4_fourcc_dispatch.rs`) that
  generates real ProRes `.mov` files with the `ffmpeg` binary (every
  profile 0..=5) and asserts the FourCC → "prores" mapping via both
  the static and dynamic dispatch paths. Test skips gracefully when
  `ffmpeg` is not on PATH.

### Known limitations

- `SPEC_BLOCKED.md` records that Round 2's planned RDD 36 entropy
  coder (DC differential + AC run/level with Rice/Golomb-Rice codes)
  is blocked: the only ProRes PDFs in `docs/video/prores/`
  (`Apple_ProRes_2022.pdf`, `Apple_ProRes_RAW_2023.pdf`) are marketing
  whitepapers with no bitstream specification, and workspace policy
  forbids consulting third-party source. Slice entropy remains the
  simplified signed exp-Golomb placeholder; real ffmpeg ProRes
  samples still will not decode at the bitstream layer.

## [0.0.4](https://github.com/OxideAV/oxideav-prores/compare/v0.0.3...v0.0.4) - 2026-04-19

### Other

- drop Cargo.lock — this crate is a library
- bump oxideav-core / oxideav-codec dep examples to "0.1"
- bump to oxideav-core 0.1.1 + codec 0.1.1
- migrate register() to CodecInfo builder
- bump oxideav-core + oxideav-codec deps to "0.1"
- claim profile FourCCs via oxideav-codec CodecTag registry
- update README + crate docs for all six profiles
- add integration round-trip tests per profile
- add HQ + 4444 XQ profiles, fix 4444 fourcc to ap4h
