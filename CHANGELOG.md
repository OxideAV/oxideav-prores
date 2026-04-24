# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
