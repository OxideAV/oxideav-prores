# oxideav-prores

[![CI](https://github.com/OxideAV/oxideav-prores/actions/workflows/ci.yml/badge.svg)](https://github.com/OxideAV/oxideav-prores/actions/workflows/ci.yml) [![crates.io](https://img.shields.io/crates/v/oxideav-prores.svg)](https://crates.io/crates/oxideav-prores) [![docs.rs](https://docs.rs/oxideav-prores/badge.svg)](https://docs.rs/oxideav-prores) [![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Pure-Rust **Apple ProRes** codec — decoder + encoder for all six
ProRes video profiles (422 Proxy / LT / Standard / HQ and 4444 /
4444 XQ). 8-, 10-, 12-, and 16-bit Y'CbCr; lossless alpha plane on
the 4444 / 4444 XQ profiles, with alpha-typed
`Yuva4(2|4)4P{,10Le,12Le,16Le}` frame surfaces on both codec
directions — up to exact 16-bit alpha. Progressive and interlaced
(TFF/BFF) in every profile.

Part of the [oxideav](https://github.com/OxideAV/oxideav-workspace)
framework but usable standalone. Implemented from SMPTE RDD 36 (no C
codec libraries linked or wrapped, no `*-sys` crates).

## Supported profiles

Every profile decodes and encodes.

| Profile        | FourCC | Pixel formats                                       |
|----------------|--------|-----------------------------------------------------|
| 422 Proxy      | `apco` | `Yuv422P{,10Le,12Le,16Le}`, `Yuva422P{,10Le,12Le,16Le}` |
| 422 LT         | `apcs` | `Yuv422P{,10Le,12Le,16Le}`, `Yuva422P{,10Le,12Le,16Le}` |
| 422 Standard   | `apcn` | `Yuv422P{,10Le,12Le,16Le}`, `Yuva422P{,10Le,12Le,16Le}` |
| 422 HQ         | `apch` | `Yuv422P{,10Le,12Le,16Le}`, `Yuva422P{,10Le,12Le,16Le}` |
| 4444           | `ap4h` | `Yuv444P{,10Le,12Le,16Le}`, `Yuva444P{,10Le,12Le,16Le}` |
| 4444 XQ        | `ap4x` | `Yuv444P{,10Le,12Le,16Le}`, `Yuva444P{,10Le,12Le,16Le}` |

## Using it through the oxideav framework

You don't register anything yourself: `oxideav-meta`'s
`register_all(&mut RuntimeContext)` sets up every enabled sibling, and
ProRes tracks inside MOV/MP4/MKV then route here through the generic
demux→decode flow. The codec id is `"prores"`; all frames are intra
(keyframe-only).

```rust
use oxideav_core::{CodecId, CodecParameters, PixelFormat, RuntimeContext};

let mut ctx = RuntimeContext::new();
oxideav_meta::register_all(&mut ctx);

let mut params = CodecParameters::video(CodecId::new("prores"));
params.width = Some(1920);
params.height = Some(1080);
params.pixel_format = Some(PixelFormat::Yuv422P);
params.bit_rate = Some(220_000_000); // -> 422 HQ

// Priority / preference-aware selection lives in oxideav-pipeline.
let mut enc = oxideav_pipeline::make_encoder(&ctx.codecs, &params)?;
let mut dec = oxideav_pipeline::make_decoder(&ctx.codecs, &params)?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

The encoder picks a profile from `pixel_format` + `bit_rate`:

| `pixel_format`          | `bit_rate` hint (bps)      | Profile  |
|-------------------------|----------------------------|----------|
| `Yuv422P` / `Yuva422P`  | `<= 70_000_000`            | Proxy    |
| `Yuv422P` / `Yuva422P`  | `<= 125_000_000`           | LT       |
| `Yuv422P` / `Yuva422P`  | `<= 180_000_000` or `None` | Standard |
| `Yuv422P` / `Yuva422P`  | `> 180_000_000`            | HQ       |
| `Yuv444P` / `Yuva444P`  | `>= 400_000_000`           | 4444 XQ  |
| `Yuv444P` / `Yuva444P`  | anything else              | 4444     |

## Using the codec directly

Without the framework, pull `oxideav-core` for the shared types and use
the crate's own entry points:

```toml
[dependencies]
oxideav-core = "0.1"
oxideav-prores = "0.0"
```

The direct factories `decoder::make_decoder(&params)` /
`encoder::make_encoder(&params)` take the same `CodecParameters` as
above; `encoder::make_encoder_with_config(&params, cfg)` adds the
[`EncoderConfig`](#encoder-controls) knobs, and the free functions
(`decoder::decode_packet_with_depth` / `decode_packet_with_format` /
`decode_packet_with_options`, `encoder::encode_frame_with_alpha` /
`encode_frame_interlaced`) work packet-by-packet with no state beyond
the call. `decode_packet_with_chroma_border` / `decode_packet_with_coded_width`
are opt-in variants for conversion pipelines that keep the reconstructed
samples to the right of the visible width (one 4:2:2 chroma column, or
the whole right macroblock) instead of cropping them per §7.5. See [docs.rs](https://docs.rs/oxideav-prores) for the full
surface.

## Bit depth and output range

RDD 36 carries no per-frame bit-depth syntax; §7.5.1 defines the
conversion to samples of arbitrary depth `b`. Request the depth via
`CodecParameters::pixel_format`: `Yuv422P`/`Yuv444P` (or nothing) for
8-bit, `*P10Le` for 10-bit (`0..=1023`), `*P12Le` for 12-bit
(`0..=4095`), `*P16Le` for 16-bit (`0..=65535`, every word bit
significant).

The §7.5.1 clamp has two spec-offered choices, exposed as
`decoder::OutputRange`: **`Full`** (default — samples use all `2^b`
levels) and **`Video`** (`1..=2^b−2`, avoiding the BT.601/709
sync-reference codes). Only the Y/Cb/Cr clamp changes; alpha always
maps the full opacity range per §7.5.2.

## Alpha

4444 / 4444 XQ code a lossless per-pixel alpha channel (§5.3.3 +
§7.1.2 run-length + differential VLC); 4:2:2 + alpha is emitted as
`bitstream_version` 1 per §6.4 — a combination this decoder and the
reference decoder both honour even though common encoders never emit
it.

Two ways to consume/produce it:

* **Typed contract** — request `Yuva4(2|4)4P` (8-bit, oxideav-core
  ≥ 0.1.30) or the deep `Yuva4(2|4)4P{10,12,16}Le` formats
  (oxideav-core ≥ 0.1.31): decode *always* returns 4 planes with
  alpha §7.5.2-converted to the surface depth — exact on the `*P16Le`
  surfaces, where 16-bit coded alpha passes through verbatim — and
  streams without alpha get an opaque plane. Encode requires 4-plane
  input at the format's alpha depth and rejects anything else with a
  self-explaining error; deep input codes 16-bit alpha
  (`alpha_channel_type = 2`) via the §7.5.2-mirror promotion, so no
  precision is dropped on the wire (10-/12-bit alpha round-trips
  sample-exact).
* **Untyped convention** — request `Yuv4(2|4)4P{,10Le,12Le,16Le}`: a
  4th plane is appended when the stream codes alpha (detect via
  `planes.len() == 4`), converted to the same surface depth as the
  colour planes, with 16-bit alpha input on the encoder. Because the
  4th plane always matches the surface depth, decoded frames never
  need oxideav-core's per-plane significant-bits side-channel — the
  pixel format already says everything.

The encoder auto-detects 4-plane input on the config path (8- vs
16-bit alpha inferred from the plane), and rate control carries the
lossless alpha blob through every trial. One reference-bitstream
finding worth knowing: the §7.5.3 per-slice alpha array is always the
full 16-row macroblock height — the visible-row exclusion governs
which rows are *written*, not the coded length (pinned directly
against reference bytes; standing DOCS-GAP candidate on the §7.5.3
wording).

## Frame-header metadata

The descriptive header fields (aspect ratio, frame rate, colour
primaries / transfer / matrix) are filled automatically from
`CodecParameters::frame_rate` or explicitly via
`EncoderConfig::with_meta(FrameMeta { .. })`. Each field has symmetric
code⇄value helpers and typed `Option<…>` accessors on the parsed
`FrameHeader` (Tables 3–6, §6.1.1), and `FrameHeader::meta` folds a
parsed header back into a `FrameMeta` so a transcode forwards the
source's metadata verbatim. This crate writes encoder identifier
`oxav`.

**Unknown colour:** RDD 36 gives codes 0 and 2 of the three colour
fields the same meaning ("Unknown/unspecified", Tables 5 and 6), but
readers do not treat them alike — black-box decoders report code 0 as a
reserved value and take matrix code 0 for an RGB identity matrix, which
then fails in colour conversion. When no `FrameMeta` is supplied the
encoder therefore writes code 2 (`FrameMeta::unspecified()`), as every
black-box encoder does; an explicit `FrameMeta` is still written
verbatim.

## Quantisation matrices

The §6.1.1 flags select custom vs default (all-4s) weight matrices,
with the chroma flag falling back to the *luma* matrix when clear. The
parsed header exposes both raw flags plus
`FrameHeader::quantization_matrix_source` (`CustomChroma` /
`LumaCustom` / `Default`). The encoder emits the smallest carriage
that reproduces the configured pair (0, 1, or 2 inline tables — 20-,
84-, or 148-byte header); `EncoderConfig::with_explicit_qmat_carriage`
opts into the always-both-tables form the reference streams ship, for
byte-level header parity. Presets: `EncoderConfig::perceptual*`
(JPEG-derived, quality-blended) and `signature_for_profile` (each
profile's native in-the-wild matrix, pinned byte-for-byte against the
reference corpus). Tables are stored in **natural raster order** (DC
at index 0) — never zigzag.

## Encoder controls

All through `EncoderConfig` + `make_encoder_with_config`:

* `with_quantization_index` — §7.3/Table 15, `1..=224`; per-profile
  defaults `8/6/4/2/2/1`.
* `with_mbs_per_slice` — `{1,2,4,8}` (§5.3), default 8.
* `with_profile` / `for_profile` — override the bit-rate heuristic.
* `with_rate_control` — two-pass per-frame binary search to
  `bit_rate / fps` within ±5 %.
* `with_alpha_channel_type` — explicit 8-/16-bit alpha; default
  auto-detects 4-plane input. Alpha-typed pixel formats fix the coded
  width themselves and refuse a contradictory request.
* `with_quant_matrices` / `perceptual*` / `signature_for_profile` —
  see above.
* `with_min_frame_size` — §5.1.2 constant-frame-size stuffing (padded
  frames decode bit-identically).
* `with_interlace_mode` — 0 progressive / 1 TFF / 2 BFF.
* `with_threads` / `with_all_threads` — code each picture's slices on
  several threads; output is byte-identical for any thread count.

Inputs are validated up front (ranges above, matrix weights `2..=63`,
dimensions `1..=65535` per the u16 header fields — out-of-range is
refused, never truncated).

## Conformance notes

* Every §6.4 "decoder shall refuse" clause is enforced:
  `bitstream_version > 1`, v0 streams that aren't 4:2:2-no-alpha,
  reserved `interlace_mode` 3, matrix entries outside `2..=63`. The
  encoder picks the lowest legal version (v0 / v1).
* **Declared-size advance** (§6.4 version-variant forward
  compatibility): the decoder locates the next syntax structure from
  the *declared* `frame_header_size` / `picture_size` /
  `slice_header_size` — never by summing what it parsed — so future
  version variants that append informative bytes decode identically,
  while payloads exceeding their declared size are refused as corrupt.
* `FrameHeader::picture_geometry` surfaces the §6.2 derivation
  (MB grid, interlaced field split, §7.5.3 crop) for muxer /
  stream-inspection callers.
* **ProRes RAW is detected and refused** cleanly (`aprn`/`aprh`
  FourCCs + in-stream marker → typed `Unsupported`). RAW has no
  public bitstream specification — no SMPTE document, no published
  syntax — so a typed refusal is the correct behaviour until one
  exists.
* FourCC ⇄ profile routing helpers live at the crate root
  (`profile_for_fourcc` / `codec_id_for_fourcc` /
  `fourcc_for_profile`); sample-table assembly belongs to the
  container crates.

## Validation

**Live black-box matrix** (`tests/blackbox_matrix.rs`, generated at run
time, skipped when `ffmpeg` is absent; last full run 1 877 cases, all
passing):

* **Foreign → ours, 1 039 / 1 039.** Streams from the four ProRes
  encoders the tool exposes (three software ones and the platform one):
  all six profiles, alpha at 8 / 16 bits and absent, progressive and
  both field orders on even and odd heights, quantiser and bit-rate
  extremes, every slice size on five widths, every quantisation-matrix
  choice, 2×2 up to 8192×4320 (the 4K / 8K rows need
  `PRORES_MATRIX_LARGE=1`), colour tags, flat / saturated / noise
  content, a 40-frame stream. Each frame is decoded here and by the
  software and the platform black-box decoder. A further 48 option
  combinations were refused by the encoders themselves.
* **Ours → foreign, 838 / 838.** This encoder across every profile ×
  8 / 10 / 12 / 16-bit input × field order, typed and untyped alpha on
  4:4:4 and 4:2:2, 17 quantisation indices from 1 to 224, slice sizes,
  every matrix preset and hand-made tables at weights 2 and 63, rate
  control from 200 kb/s to 2 Gb/s, stuffing, 1×1 up to 8192×4320 and
  every header code. The black-box decoders accept every stream
  without a diagnostic and reproduce this crate's own reconstruction.

The tolerance is the one RDD 36 itself implies: §7.4 fixes no IDCT,
only the Annex A accuracy bound (peak error 0.15 per reconstructed
value), so two conforming decoders may differ by ±1 code at 10 bits and
±3 at 12 bits after the §7.5.1 rounding. Measured, this decoder is
within ±1 code of the software decoder at both depths and within ±1
(10-bit) / ±2 (12-bit) of the platform decoder; alpha is bit-exact
against both. Three things the comparison has to know about the
validators, all input/output facts: the software decoder never emits a
colour code below 4 or above `2^b − 5`; it delivers 16-bit alpha as its
top 12 bits rather than through the §7.5.2 rounding; and it wraps where
§7.5.1 clamps once a coefficient leaves the Annex A range (17 cases,
where each sample is held to the nearer decoder). The platform decoder
is not used as a chroma reference for alpha-bearing pictures (it
returns resampled chroma there) nor for the last column of odd-width
4:2:2; on 6 further cases it alone departs on one or two samples at the
right edge of a partial macroblock under a coarse quantiser.

Not verified against an outside stream: ProRes produced by encoders
other than those four; 4:2:2 + alpha produced by anyone else (no
black-box encoder writes it); bitstreams with `frame_header_size`,
`picture_header_size` or `slice_header_size` larger than the base
syntax (§6.4 version variants), which are covered by this crate's own
tests only.

The full RDD 36 Annex A IDCT accuracy qualification passes with margin,
and reference fixtures under `docs/video/prores/fixtures/` pin decode
and encode SHAs. At equal rate with signature matrices, the encoder
measures **above** the reference encoder on every profile:

| Profile  | reference        | ours @ equal rate | Δ PSNR |
|----------|------------------|-------------------|--------|
| Proxy    | 3 637 B, 46.2 dB | 3 774 B, 46.4 dB  | +0.2   |
| LT       | 10 815 B, 51.0 dB| 11 415 B, 54.9 dB | +3.9   |
| Standard | 15 686 B, 53.8 dB| 14 697 B, 59.3 dB | +5.5   |
| HQ       | 23 801 B, 59.5 dB| 22 589 B, 67.1 dB | +7.6   |
| 4444     | 35 893 B, 59.0 dB| 34 556 B, 64.7 dB | +5.7   |
| 4444 XQ  | 51 640 B, 63.4 dB| 48 572 B, 69.6 dB | +6.2   |

Five panic-free `cargo-fuzz` targets drive the decode entry points,
header parsers, and entropy coders (30-minute daily CI run), with the
malformed-input behaviour also pinned by ordinary `cargo test` cases;
Criterion benches cover the decode/encode hot paths with
constant-block fast-path dispatch verified bit-exact against the
general DCT loops.

## License

MIT — see [LICENSE](LICENSE).
