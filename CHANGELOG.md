# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
