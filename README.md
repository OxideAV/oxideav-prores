# oxideav-prores

Pure-Rust **Apple ProRes** codec — decoder + encoder for all six
ProRes video profiles (422 Proxy / LT / Standard / HQ and 4444 /
4444 XQ). 8-bit, 10-bit, and 12-bit Y'CbCr; lossless alpha plane on
the 4444 / 4444 XQ profiles.

Part of the [oxideav](https://github.com/OxideAV/oxideav-workspace) framework but usable standalone. Implemented from the spec (no C codec libraries linked or wrapped, no `*-sys` crates).

## Status

| Profile        | FourCC | Pixel formats                                      | State           |
|----------------|--------|----------------------------------------------------|-----------------|
| 422 Proxy      | `apco` | `Yuv422P`, `Yuv422P10Le`, `Yuv422P12Le`            | decode + encode |
| 422 LT         | `apcs` | `Yuv422P`, `Yuv422P10Le`, `Yuv422P12Le`            | decode + encode |
| 422 Standard   | `apcn` | `Yuv422P`, `Yuv422P10Le`, `Yuv422P12Le`            | decode + encode |
| 422 HQ         | `apch` | `Yuv422P`, `Yuv422P10Le`, `Yuv422P12Le`            | decode + encode |
| 4444           | `ap4h` | `Yuv444P`, `Yuv444P10Le`, `Yuv444P12Le` (+ alpha)  | decode + encode |
| 4444 XQ        | `ap4x` | `Yuv444P`, `Yuv444P10Le`, `Yuv444P12Le` (+ alpha)  | decode + encode |

Bit-depth selection follows the stream's `CodecParameters::pixel_format`
— RDD 36 §5 carries no per-frame bit-depth syntax element; §7.5.1
defines the conversion from reconstructed color component values to
pixel samples of arbitrary bit depth `b`. Pass `Yuv422P10Le` /
`Yuv444P10Le` to get 10-bit planar output (LE u16 pairs, valid range
`0..=1023`); pass `Yuv422P12Le` / `Yuv444P12Le` for 12-bit
(`0..=4095`); pass `Yuv422P` / `Yuv444P` (or omit `pixel_format`) for
8-bit.

### Alpha plane

The 4444 / 4444 XQ profiles support a per-pixel alpha channel coded
losslessly per RDD 36 §5.3.3 + §7.1.2 (raster-scan run-length code +
differential VLC, Tables 12-14). The alpha is exposed as a 4th
`VideoPlane` on the decoded `VideoFrame` (after Y/Cb/Cr); the
encoder accepts the same shape on input via
[`encoder::encode_frame_with_alpha`].

The core `PixelFormat` enum does not yet carry `Yuva422P` / `Yuva444P`
variants, so the pixel-format reporting stays as `Yuv4(2|4)4P*` — the
caller checks `frame.planes.len() == 4` to detect alpha.

Downstream stages that need the named Table 7 variant rather than the
raw u4 code call `FrameHeader::alpha_kind()`, which returns
`Option<AlphaChannelType>` (`None` / `Bits8` / `Bits16` for codes
0, 1, 2; outer-Option `None` for the reserved range `3..=15`). The
outer-Option discriminant preserves the wire-level distinction
between "no alpha is present" (`Some(AlphaChannelType::None)`) and
"the field carried a reserved code".

The 4444 and 4444 XQ profiles share the same bitstream structure as
4444 — XQ is selected when the caller requests the highest quality
tier and produces a larger packet at lower quantisation.

ffmpeg-encoded `prores_ks` `apcn` / `apch` (4:2:2) and `ap4h` /
`ap4x` (4444 ± alpha) streams decode interop-clean across 8-/10-/12-bit
and both progressive and interlaced (`-flags +ildct -top {0,1}`)
modes. The picture-height-not-multiple-of-16 alpha edge case is
covered: the decoder reads the full padded macroblock-row alpha
(per RDD 36 §7.5.2) and crops on output. Streams produced by this
crate's own encoder use the spec's entropy coder for color, but the
encoder emits a plain run-length alpha (alternative path permitted by
§7.1.2); the coder is bit-exact with itself and decoder-compatible.

### Frame-header metadata (RDD 36 §5.1.1 / §6.2)

The encoder fills the descriptive frame-header fields
(`aspect_ratio_information`, `frame_rate_code`, `color_primaries`,
`transfer_characteristic`, `matrix_coefficients`) automatically from
[`CodecParameters::frame_rate`] — e.g. `Rational::new(30_000, 1001)`
maps to `frame_rate_code = 4` (29.97 fps NTSC). Set every field
explicitly via [`encoder::EncoderConfig::with_meta`]:

```rust
use oxideav_prores::encoder::{make_encoder_with_config, EncoderConfig};
use oxideav_prores::frame::FrameMeta;

let cfg = EncoderConfig::default().with_meta(FrameMeta {
    aspect_ratio_information: 3,  // 16:9
    frame_rate_code: 8,           // 60 fps
    color_primaries: 9,           // BT.2020
    transfer_characteristic: 16,  // SMPTE ST 2084 (PQ)
    matrix_coefficients: 9,       // BT.2020 NCL
});
let enc = make_encoder_with_config(&params, cfg)?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

The mapping from `Rational` to `frame_rate_code` lives in
[`frame::frame_rate_code_from_rational`] and covers the 11 spec-named
rates (RDD 36 §6.2 Table 4); any other rate yields `0`
("unknown"), which RDD 36 decoders treat as a hint only.

The decoder side gets the symmetric reverse via
[`frame::rational_from_frame_rate_code`] (Table 4) and
[`frame::aspect_ratio_from_code`] (Table 3), returning the spec's
exact symbolic [`oxideav_core::Rational`] for every named code and
`None` for the "unknown" code 0 + the reserved codes. The Option
discriminant is the wire-level distinction between "the stream
explicitly says rate unknown" and "the stream pins 24 fps" — a
downstream pipeline reading a decoded packet (or any consumer
parsing a frame header with [`frame::parse_frame`]) can forward
`frame_rate` along an `oxideav_core` graph without re-implementing
Table 4 itself. The forward + reverse halves are symmetric: every
named code round-trips structurally through both helpers (e.g.
`30000/1001` ⇄ code 4, distinct from `30/1` ⇄ code 5).

The color-metadata bytes get the same treatment via three named
enums + reverse helpers — [`frame::ColorPrimaries`] +
[`frame::color_primaries_from_code`] for RDD 36 §6.1.1 Table 5,
[`frame::MatrixCoefficients`] + [`frame::matrix_coefficients_from_code`]
for Table 6 (with `luma_coefficients()` returning the K_R / K_G / K_B
triple in the spec's exact decimals), and [`frame::AlphaChannelType`] +
[`frame::alpha_channel_type_from_code`] for Table 7 (including a
`has_alpha()` predicate matching the §5.3 slice parser's
`alpha_channel_type != 0` guard). A decoded packet's `FrameHeader` now
surfaces named gamut + matrix + alpha-mode through one cheap helper
call per field — Bt709 / Bt601_625 / Bt601_525 / Bt2020 / DciP3 /
P3D65 on the primaries side; Bt709 / Bt601 / Bt2020Ncl on the matrix
side — and the spec's "unknown / unspecified" codes (0 and 2 for
Tables 5, 6) cleanly surface as `None` so a colour-management stage
can distinguish a stream that pins BT.2020 from one that says
"unknown". `transfer_characteristic` gets the same typed-accessor
treatment via [`frame::TransferCharacteristic`] +
[`frame::transfer_characteristic_from_code`] — RDD 36 §6.1.1 names
only three nonreserved codes (1 = BT.601/709/2020 OETF, here
`Bt1886`; 16 = SMPTE ST 2084 Inverse-EOTF, here `St2084`; 18 =
BT.2100-2 HLG Reference OETF, here `Hlg`) and unlike the primary /
matrix fields it does so in prose rather than a numbered Table. The
raw u8 stays on [`frame::FrameMeta`] for wire-level callers.

Downstream stages that prefer the typed-accessor surface (mirroring
`alpha_kind()` / `interlace_kind()` on the same struct) call
[`FrameHeader::color_primaries_kind`], which folds the
`color_primaries_from_code(fh.color_primaries)` call into a single
method on the parsed header. The accessor returns
`Option<ColorPrimaries>` with the same outer-Option discriminant as
its siblings — `Some(_)` for every named Table 5 code (1/5/6/9/11/12),
`None` for the "unknown" codes (0 and 2) plus every reserved code in
`[3, 4, 7, 8, 10, 13..=255]` — so a consumer reading a parsed packet
can call `fh.color_primaries_kind()` and `fh.alpha_kind()` together
without breaking up the read.

The same shape applies to the Table 6 matrix via
[`FrameHeader::matrix_coefficients_kind`], which returns
`Option<MatrixCoefficients>` — `Some(_)` for the three named codes
(1/6/9, i.e. BT.709 / BT.601 / BT.2020 NCL), `None` for the "unknown"
codes (0 and 2) plus every reserved code in `[3, 4, 5, 7, 8, 10..=255]`.
A Y'CbCr → R'G'B' conversion stage can then evaluate the §6.1.1
derivation formulas off `mc.luma_coefficients()` (the `(K_R, K_G, K_B)`
triple straight from Table 6) without a second table lookup.

The transfer-function field follows the same pattern via
[`FrameHeader::transfer_characteristic_kind`], which returns
`Option<TransferCharacteristic>` — `Some(_)` for the three named
codes (1 / 16 / 18, i.e. BT.1886 / ST 2084 / HLG), `None` for the
"unknown" codes (0 and 2) plus every reserved code in `[3..=15]`,
`17`, and `[19..=255]`. Reading `fh.transfer_characteristic_kind()`,
`fh.matrix_coefficients_kind()`, `fh.color_primaries_kind()`, and
`fh.alpha_kind()` in one chain gives a downstream colour-management
stage every named §6.1.1 field at once.

The §6.2 Table 4 `frame_rate_code` field gets the same typed-accessor
treatment via [`FrameHeader::frame_rate`], which returns
`Option<oxideav_core::Rational>` — `Some(_)` carrying the spec's
exact symbolic fraction (e.g. `Rational::new(30_000, 1001)` for code
4, distinct from `Rational::new(30, 1)` for code 5) for the eleven
named codes (`1..=11`), `None` for the "unknown / unspecified" code
`0` plus every reserved code in `12..=15`. Unlike the colour-metadata
accessors the returned type is the rate fraction itself rather than a
named enum (Table 4 is a list of exact rates with no closer-grained
naming, and `30000/1001` vs `30/1` are wire-distinct), so a downstream
pipeline stage reading a parsed packet can forward
[`CodecParameters::frame_rate`] along an `oxideav_core` graph straight
off `fh.frame_rate()` without precision loss. The accessor is the
natural mirror of the existing reverse helper
[`frame::rational_from_frame_rate_code`] — encoder side already does
the inverse via [`frame::frame_rate_code_from_rational`] when filling
[`FrameMeta::frame_rate_code`] from a caller-supplied rate.

The §6.2 Table 3 `aspect_ratio_information` field — the high-nibble
sibling of `frame_rate_code` in the packed byte 13 — is exposed the
same way via [`FrameHeader::aspect_ratio`], which returns
`Option<oxideav_core::Rational>` — `Some(_)` carrying the spec's
exact ratio (`Rational::new(1, 1)` for code 1 / square pixels,
`Rational::new(4, 3)` for code 2, `Rational::new(16, 9)` for code 3)
for the three named codes, `None` for the "unknown / unspecified"
code `0` plus every reserved code in `4..=15`. A consumer reading a
parsed packet can call `fh.aspect_ratio()` and `fh.frame_rate()`
alongside the §6.1.1 colour-metadata accessors in one chain — the
two §6.2-packed nibbles surface through symmetric outer-Option APIs.
The accessor is the natural mirror of the existing reverse helper
[`frame::aspect_ratio_from_code`].

[`CodecParameters::frame_rate`]: https://docs.rs/oxideav-core/latest/oxideav_core/struct.CodecParameters.html#structfield.frame_rate

### Configurable quantisation index (RDD 36 §7.3 / Table 15)

The encoder picks one `quantization_index` per profile by default
(`8 / 6 / 4 / 2 / 2 / 1` for Proxy / LT / Standard / HQ / 4444 /
4444 XQ — see [`frame::Profile::default_quant_index`]). Lower index =
finer step = higher quality + larger packet. Override the default for
custom rate/quality trade-offs:

```rust
use oxideav_prores::encoder::{make_encoder_with_config, EncoderConfig};

// Highest-quality Proxy: same profile selection (bit_rate hint), but
// the qscale floor of qi=2 instead of the Proxy default qi=8.
let cfg = EncoderConfig::default().with_quantization_index(2);
let enc = make_encoder_with_config(&params, cfg)?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

The valid range is `1..=224` (rejected at encoder construction
otherwise). The override applies to every slice in every encoded frame
and roundtrips through any RDD 36 decoder.

### Configurable macroblocks-per-slice (RDD 36 §5.3)

Every encoded picture is partitioned into slices whose width in
macroblocks is signalled by `picture_header.log2_desired_slice_size_in_mb`
(a 2-bit field, so the legal set is `{1, 2, 4, 8}`). The default is
**8 MBs/slice**, matching every Apple-encoded fixture committed
under `docs/video/prores/fixtures/`. Lowering the value subdivides
each macroblock row into more, smaller slices — useful for finer
error resilience (a corrupted byte taints fewer macroblocks) at the
cost of a slightly larger packet (extra `slice_header` +
`slice_size_table` entries amortise over fewer macroblocks).

```rust
use oxideav_prores::encoder::{make_encoder_with_config, EncoderConfig};

// Single-MB slices: maximum resilience, ~5% larger packet on a
// 128x64 4:2:2 source (5948 B → 6247 B in the regression test).
let cfg = EncoderConfig::default().with_mbs_per_slice(1);
let enc = make_encoder_with_config(&params, cfg)?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

Valid values are `1`, `2`, `4`, `8`; anything else returns
`Error::invalid` at encoder construction. The choice flows into
`picture_header.log2_desired_slice_size_in_mb` so any RDD 36 decoder
(including this crate's own) rebuilds the per-row template via
[`frame::compute_slice_sizes`]. The default path
(`mbs_per_slice == None`) is byte-identical to
`with_mbs_per_slice(DEFAULT_MBS_PER_SLICE)` (8) — the knob is
purely additive. ffmpeg's `prores_ks` exposes the same control
through `-mbs_per_slice {1,2,4,8}`.

Cross-decode acceptance: bitstreams emitted by
`make_encoder_with_config(... with_mbs_per_slice(m))` + `send_frame`
decode through ffmpeg's stock `prores_ks` decoder at 58.8-63.9 dB luma
PSNR for every legal value (Standard at all four `{1, 2, 4, 8}`,
HQ at the `{1, 8}` extremes; 128×48). Each cross-decode case
re-checks `picture_header.log2_desired_slice_size_in_mb == log2(m)`
to confirm the builder threaded through to the emitted bitstream
instead of silently defaulting. Packet sizes grow monotonically as
`mbs_per_slice` shrinks (apcn: 6708 → 7106 bytes from 8 → 1
MBs/slice; ~6%). See `tests/ffmpeg_cross_decode.rs`.

### Explicit profile selection

By default the encoder maps `CodecParameters::bit_rate` to one of the
six profiles via [`encoder::pick_profile`] (see the table below).
Override the heuristic with [`encoder::EncoderConfig::with_profile`] when
the caller wants a specific profile that the bitrate hint would not
pick — e.g. **4444 XQ at low bitrates** (the heuristic only selects XQ
above 400 Mbit/s), or any 4:2:2 profile with no `bit_rate` hint at all.

```rust
use oxideav_prores::encoder::{make_encoder_with_config, EncoderConfig};
use oxideav_prores::frame::Profile;

// Force 4444 XQ for a 4:4:4 stream regardless of bit_rate hint.
let cfg = EncoderConfig::for_profile(Profile::Prores4444Xq);
let enc = make_encoder_with_config(&params, cfg)?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

The override's [`frame::Profile::chroma_format`] must match the
requested `PixelFormat` (HQ/SD/LT/Proxy ↔ 4:2:2; 4444/4444 XQ ↔ 4:4:4) —
mismatches return `Error::invalid` at encoder construction. The bitstream
syntax itself only carries `chroma_format` per RDD 36 §5.1.1; the profile
choice influences only the encoder's default `quantization_index`.

### Two-pass per-frame rate control

Set `CodecParameters::bit_rate` + `frame_rate` and call
`EncoderConfig::with_rate_control()` to enable per-frame binary-search
rate control. The encoder performs up to `RATE_CTRL_MAX_PASSES` (10)
trial encodes per frame, adjusting `quantization_index` to hit the
per-frame byte target derived from `bit_rate / fps` within
`RATE_CTRL_TOLERANCE` (±5 %). When the target is outside the
achievable range for the resolution the encoder returns the best
candidate (finest quality for targets above the maximum, coarsest for
targets below the minimum) — it never emits a broken stream.

```rust
use oxideav_prores::encoder::{make_encoder_with_config, EncoderConfig};

// Hit the nominal 422 HQ bitrate (220 Mbit/s at 29.97 fps) within ±5%.
let cfg = EncoderConfig::default().with_rate_control();
let enc = make_encoder_with_config(&params, cfg)?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

### Configurable quantisation matrices (RDD 36 §5.3.4 + §6.3.7 + §7.3)

The encoder defaults to the spec's flat all-4s quantisation matrix
(`load_luma_qmat = load_chroma_qmat = 0`, 20-byte frame header — same
as `prores_ks` for `apcn` / `apch` when no perceptual preset is
selected). Pass an [`encoder::EncoderConfig`] with a non-default
[`quant::QuantMatrices`] to load custom matrices into the frame header
(making the header 84 or 148 bytes per §7.3) — every RDD 36 decoder,
including ffmpeg's, uses the loaded matrices for dequantisation.

A built-in perceptual preset is provided via
[`quant::QuantMatrices::perceptual`]: JPEG K.1 / K.2 (ISO/IEC 10918-1
Annex K) normalised to a DC weight of 2 and clamped to the spec's
`2..=63` weight range. At every `quantization_index` from 2 to 16 the
perceptual matrices cut packet size by 20-25% on broadband content
because the entropy coder's `endOfData()` semantics (RDD 36 §7.1.1)
turn HF zeros into trailing zero runs that cost no bits. PSNR trades
off slightly because flat is provably PSNR-optimal under uniform
quantisation, but JPEG-style CSF-rolloff matrices preserve perceptual
quality far better than the same byte-count flat encode.

```rust
use oxideav_prores::encoder::{make_encoder_with_config, EncoderConfig};
use oxideav_prores::quant::QuantMatrices;

let enc = make_encoder_with_config(&params, EncoderConfig::perceptual())?;
// Or: EncoderConfig::default().with_quant_matrices(QuantMatrices { luma, chroma })
# Ok::<(), Box<dyn std::error::Error>>(())
```

### Profile-aware perceptual matrices

[`quant::QuantMatrices::perceptual_for_profile`] +
[`encoder::EncoderConfig::perceptual_for_profile`] take a [`frame::Profile`]
and blend the JPEG-derived perceptual matrix toward the flat all-4s
default in proportion to the profile's
[`frame::Profile::default_quant_index`] (blend = qi / 8). Higher-quality
profiles preserve more high-frequency precision; lower-quality profiles
get heavier HF rolloff for tighter packets at matched perceptual
quality.

| Profile     | qi | blend | HF rolloff                       |
|-------------|----|-------|----------------------------------|
| Proxy       |  8 |  8/8  | full (matches `perceptual()`)    |
| LT          |  6 |  6/8  | heavy                            |
| Standard    |  4 |  4/8  | moderate                         |
| HQ          |  2 |  2/8  | light                            |
| 4444        |  2 |  2/8  | light                            |
| 4444 XQ     |  1 |  1/8  | minimal (matrix close to flat)   |

The factory also pins the supplied profile so the chosen tier is
honoured regardless of the `bit_rate` → profile heuristic. Every
blended weight stays in `2..=63` (RDD 36 §7.3), and the matrices are
loaded into the frame header so any RDD 36 decoder dequantises
correctly.

```rust
use oxideav_prores::encoder::{make_encoder_with_config, EncoderConfig};
use oxideav_prores::frame::Profile;

// 4444 XQ with profile-aware perceptual rolloff — matrix sits close
// to flat, giving the encoder near-maximum HF precision while still
// loading custom weights into the frame header.
let cfg = EncoderConfig::perceptual_for_profile(Profile::Prores4444Xq);
let enc = make_encoder_with_config(&params, cfg)?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

At matched `quantization_index`, the HQ-blended preset reconstructs
higher Y-PSNR than the Proxy-blended preset on broadband sources, and
the Proxy-blended preset emits a smaller packet — both monotonic in
the profile's quality tier (see `tests/perceptual_profile.rs`). The
mismatch between an explicit profile and the requested
`PixelFormat` (4:2:2 ↔ Yuv422P; 4:4:4 ↔ Yuv444P) is rejected at
encoder construction.

Cross-decode acceptance: bitstreams emitted by
`encode_frame_with_qmats(..., QuantMatrices::perceptual_for_profile(p))`
decode through ffmpeg's stock `prores` / `prores_ks` decoder at 58.0–63.8 dB
luma PSNR for every profile — Proxy / LT / Standard / HQ at 8-bit (with
HQ also exercised at 10-bit through the §7.5.1 `b = 10` level shift on
the loaded-qmat path); 4444 / 4444 XQ at 8-bit (with 4444 also at 12-bit
through the §7.5.1 `b = 12` level shift). The 4444 XQ corner is the
strongest guard — its 1/8 perceptual + 7/8 flat blend sits closest to
flat yet still triggers the `load_*_qmat = 1` header path
(`frame_header_size == 148`), so a silent collapse to the flat default
would surface immediately. See `tests/ffmpeg_cross_decode.rs`.

### Bitstream version compatibility (RDD 36 §6.4)

The decoder enforces every "decoder shall refuse" clause attached to
the frame-header syntax elements:

| Spec clause | Constraint enforced |
|-------------|---------------------|
| §6.1.1 `bitstream_version` | Reject any value > 1 — only versions 0 and 1 are specified. |
| §6.4 v0 stream rules       | A `bitstream_version = 0` stream must have `chroma_format = 2` (4:2:2) AND `alpha_channel_type = 0`. Any v0 stream carrying 4:4:4 chroma or any alpha is malformed and rejected. |
| §6.1.1 Table 2 `interlace_mode` | Value 3 is reserved — rejected. |
| §6.1.1 qmat entries | Every entry of `luma_quantization_matrix` / `chroma_quantization_matrix` must lie in `2..=63` — out-of-range entries are rejected at parse time. |

The encoder picks the lowest legal `bitstream_version` per the spec's
own recommendation ("encoders should use the lowest bitstream version
appropriate for the frame being encoded"): 4:2:2 no-alpha streams emit
v0 for maximum legacy-decoder reach; any other combination emits v1.
See `tests/spec_validation.rs` (15 cases) for round-trip coverage.

### ProRes RAW is detected and refused, never mis-decoded

Apple **ProRes RAW** (`aprn` / `aprh`) is a *separate* format that
wraps single-plane Bayer/CFA sensor data; it is outside the scope of
SMPTE RDD 36 (which covers only the six YUV/RGB profiles) and uses an
incompatible sample structure. This crate refuses ProRes RAW cleanly
at two layers rather than mis-routing it through the RDD 36 parser:

- **FourCC level.** [`is_prores_raw_fourcc`] recognises `aprn` / `aprh`
  (case-insensitive); [`PRORES_RAW_FOURCCS`] lists them. They
  deliberately resolve to neither a `CodecId` nor a `frame::Profile`,
  so a demuxer/dispatcher can tell "ProRes RAW, unsupported" apart from
  "not ProRes at all" and surface a precise error to the user.
- **In-stream level.** A sample carrying the ProRes RAW marker `aprh`
  at the `icpf` offset (just after the 4-byte `frame_size`) yields a
  specific `Unsupported` error naming ProRes RAW — distinct from the
  generic "magic mismatch" returned for arbitrary non-ProRes bytes.

Decoding ProRes RAW would require Apple's proprietary bitstream
documentation (the staged `Apple_ProRes_RAW_2023.pdf` is a marketing
white paper with no syntax) and is not implemented.

### Interlaced (RDD 36 §5.1, §6.2, §7.5.3)

A frame's `interlace_mode` (0 = progressive, 1 = top-field-first,
2 = bottom-field-first) controls whether the encoded bitstream carries
one picture or two (one per field). Source rows interleave across
fields per §7.5.3: top field = source rows 0, 2, 4, …; bottom field =
rows 1, 3, 5, …. The encoder splits the source into two field pictures
(each at `(height + 1) / 2` or `height / 2` rows) and emits them in
temporal order; the decoder reverses the deinterleave. Each field
picture uses the interlaced block scan (§7.2 Figure 5) instead of the
progressive Figure 4.

Interlaced output is reachable two ways: the free function
[`encoder::encode_frame_interlaced`], or — for callers that only touch
the high-level `Encoder` trait — [`encoder::EncoderConfig::with_interlace_mode`],
which threads the mode through `make_encoder_with_config` → `send_frame`
so a registry-built encoder emits interlaced ProRes (and respects the
mode under two-pass rate control). Value `3` is reserved (Table 2) and
rejected at encoder construction.

Decoder-side, downstream callers reading the parsed `FrameHeader` can
switch on the named scan order via the typed accessor
[`FrameHeader::interlace_kind`] which returns
`Option<InterlaceMode>` (variants `Progressive` / `TopFieldFirst` /
`BottomFieldFirst`, with the `is_interlaced()` predicate matching
`picture_count() == 2`). The raw `interlace_mode: u8` stays on the
struct for wire-level fidelity; the accessor folds the Table 2 reverse
mapping into a single call so a pipeline stage handling field-order
display, deinterleave, or trailing-picture pairing does not have to
re-derive Table 2 at every call site. The reverse helper
`frame::interlace_mode_from_code(u8)` is exposed for callers consuming
a raw byte outside a parsed header.

```rust
use oxideav_prores::encoder::{make_encoder_with_config, EncoderConfig};

// Top-field-first interlaced output through the Encoder trait.
let cfg = EncoderConfig::default().with_interlace_mode(1);
let enc = make_encoder_with_config(&params, cfg)?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

Decoder side: a small synthetic interlaced apcn 10-bit `yuv422p10le`
fixture (128×128, 2 frames, TFF) ships in-tree at
`docs/video/prores/fixtures/interlaced-tff-128x128-apcn/` alongside its
`prores_ks` reference YUV. The test
`tests/docs_corpus.rs::corpus_interlaced_tff_128x128_apcn_per_field_psnr`
splits the decoded luma plane into its top-field rows {0, 2, 4, …} and
bottom-field rows {1, 3, 5, …} per §7.5.3 and scores PSNR for EACH
field independently — both fields must clear 40 dB. Measured **78.19 /
79.42 dB** (frame 0 top / bottom) and **77.84 / 78.69 dB** (frame 1),
catching picture-order swap and silent second-picture skip that
whole-frame aggregate PSNR does not.

The small in-tree 128×128 fixture
(`docs/video/prores/fixtures/interlaced-tff-128x128-apcn/`) is tied down
by `tests/interlaced_decode_sha_128x128.rs`. Both frames of the
2-frame container are pinned by SHA-256 of their decoded `yuv422p10le`
output (`65 536` bytes per frame, frame 0 and frame 1 hashes assert
distinct — a regression where the §5.1 walker silently re-decodes
frame 0 twice would surface as a SHA collision), plus a third pin on
the concatenated `frame0 ‖ frame1` byte stream (`131 072` bytes —
matches the scope of the fixture's `expected.yuv.sha256` manifest, so
the reference fixed-point IDCT SHA is read from the manifest and
reported alongside ours to keep the ~1-LSB float vs fixed-point IDCT
divergence permitted by §7.4 visible). Companion to the existing
per-field PSNR test in `tests/docs_corpus.rs`: PSNR catches a partial
field-mapping drift but cannot catch a same-PSNR byte shift; the SHA
pin catches that.

The two broadcast-scale fixtures —
`docs/video/prores/fixtures/interlaced-tff/` and
`docs/video/prores/fixtures/pal-1080i50/`, both 1920×1080 apcn 10-bit
TFF — are tied down by `tests/interlaced_decode_sha.rs`. That test
pins a SHA-256 of the decoder's Y/Cb/Cr byte stream for frame 0 (`8 294
400` bytes of `yuv422p10le`) so any future change to the §5.1
multi-picture frame walker, the §7.5.3 field-row deinterleave, the
§5.3 slice walker, or the §7.4 IDCT path flips the test red instead
of silently shifting decode output. Both fixtures must hash to the
same value — they share the apcn elementary bitstream wrapped in
differing MOV `moov` metadata, so a cross-fixture identity check
catches container state leaking into the elementary decoder. The pin
is the in-tree float-IDCT SHA; each fixture's `expected.yuv.sha256`
fixed-point reference SHA is read from the manifest and reported in
the test log alongside ours, so the ~1-LSB IDCT divergence permitted
by RDD 36 §7.4 stays visible. A FIPS 180-4 §B.1/§B.2 self-check of
the test-side SHA-256 guards against typos in the hash code that
would mask a real decoder regression.

The **progressive** path gets the same lockstep treatment via
`tests/progressive_decode_sha.rs`, covering frame 0 of the seven
progressive corpus fixtures that ship an `expected.yuv.sha256`
sidecar — `proxy-1280x720` (apco), `lt-1280x720` (apcs),
`sq-1920x1080` (apcn), `hq-1920x1080` (apch), `4444-1920x1080`
(ap4h), `4444xq-1920x1080` (ap4x), and `4444-with-alpha` (ap4h with
`alpha_channel_type = 2`, 4 output planes via §5.3.3 + §7.1.2). Each
test decodes frame 0 at the fixture's pinned `(BitDepth, ChromaFormat)`,
hashes the concatenated Y/Cb/Cr (+A) byte stream, and asserts the SHA
matches a per-fixture float-IDCT constant — locking down the §5.1
single-picture frame container, §5.3 default 8-MBs-per-slice slice
walker, §7.2 Figure 4 progressive scan, §7.4 IDCT scaling, and
§7.5.1 component → pixel sample mapping in one test per profile. As
with the interlaced pin, each fixture's reference (fixed-point) SHA
is reported alongside ours so the permitted ~1-LSB float vs
fixed-point IDCT divergence stays visible without flipping the test
red. The SHA-256 self-check is duplicated against FIPS 180-4 §B.1/§B.2
inside the same binary for the same anti-typo guard.

The **encoder side** gets the same byte-level lockstep treatment via
`tests/encoder_output_sha.rs`, which pins a SHA-256 across every
public encoder free-function entry point: `encode_frame` for all six
profiles (apco / apcs / apcn / apch / ap4h / ap4x) at their
`Profile::default_quant_index`, `encode_frame_with_alpha` for ap4h +
8-bit alpha (exercising §5.3.3 + §7.1.2 `scanned_alpha()` emission at
the tail of each slice), `encode_frame_interlaced` for apcn TFF and
BFF (exercising §5.1 two-picture walker + §6.1.1 Table 2
`interlace_mode` byte + §7.5.3 row-{0,2,…}/{1,3,…} field splitting),
and `encode_frame_with_qmats` for ap4h with
`QuantMatrices::perceptual_for_profile` (exercising the 148-byte
frame_header path where `load_luma_qmat = load_chroma_qmat = 1`). The
synthetic input is a deterministic 128×64 frame (pure function of
`(i, j)` + chroma subsampling — no fixture file), so any encoder-side
drift in DCT, quantisation, slice scan, entropy coder, frame_header /
picture_header / slice_header byte layout, or the §5.1 frame container
surfaces as a SHA mismatch. Each pin is followed by a
`decode_packet_with_depth()` round-trip to catch a SHA-only flipper
(encoder change that mints different bytes but is internally consistent
with a matched decoder change), and `assert_ne!` lines guard that
TFF ≠ BFF and flat ≠ perceptual stay on the wire. Same FIPS 180-4 §B.1
/ §B.2 self-check runs alongside the pins.

The encoder SHA-pin coverage **extends onto the deeper-depth
`read_sample` arms** as well: `encode_frame_with_depth` at
`BitDepth::Ten` and `BitDepth::Twelve` is pinned for both `apcn`
(4:2:2 grid, broadcast canonical interop target) and `ap4h` (the
§7.4 doubled 4:4:4 chroma grid; 12-bit is ffmpeg's native ap4h depth).
Each pin sources from a deterministic synthetic 10-/12-bit input that
is *not* a `4x` / `16x` scaling of the 8-bit pattern (since the §7.5.1
forward level-shift `v = s / 2^(b-9) - 256` exactly cancels that
scaling and would collide wire bytes across depths) — distinct
gradient slopes + offsets per depth keep every pin pegged to a unique
byte stream. `assert_ne!` between the matching 8-bit / 10-bit / 12-bit
pin pairs catches silent depth-flip regressions (e.g. the 10-bit
divisor being applied to a 12-bit input, or `read_sample` reading only
the low byte of a 16-bit sample), and the `decode_packet_with_depth()`
round-trip's Y-plane stride check is bit-depth-aware (1 byte/sample at
8-bit vs 2 bytes/sample LE-packed at 10/12-bit).

Streams produced by `encode_frame_interlaced` for apcn / apch cross-decode
through ffmpeg's `prores_ks` decoder at ≥ 64 dB luma PSNR — both 8-bit
(TFF and BFF, 64×48 and 128×96) and **genuine 10-bit** field-pair
packing (TFF/BFF at 64×48, TFF at 128×96; 64.40-64.47 dB). The 10-bit
cases drive a true 10-bit LE source through `read_sample`'s
`BitDepth::Ten` branch (RDD 36 §7.5.1 level shift for `b = 10`) feeding
the §7.5.3 two-field deinterleave — not an 8-bit value padded into
10-bit storage. The **high-level `Encoder` path** is covered too: streams
emitted by `make_encoder_with_config(... with_interlace_mode(m))` +
`send_frame` cross-decode through `prores_ks` at ≥ 58 dB luma PSNR
(apch TFF/BFF and apcn TFF at 64×48, apch TFF at 128×96), with the
even/odd field-bias check confirming TFF and BFF field order survives
the round-trip.

The **interlaced 4444 + alpha** path (ap4h / ap4x field-pair packing
with a per-pixel alpha plane) also cross-decodes through `prores_ks` at
65.24-65.26 dB luma PSNR (ap4h TFF/BFF at 64×48, ap4h TFF at 128×96,
ap4x TFF at 64×48). These cases combine the four hardest paths the
encoder owns at once — 4:4:4 full-resolution chroma, the genuine 12-bit
`read_sample` branch (RDD 36 §7.5.1 for `b = 12`, matching ffmpeg's
native ap4h/ap4x depth), the §5.3.3 / §7.1.2 / Table 14 16-bit-alpha
entropy coder (per-slice scanned-alpha blob at the padded MB-row height
per §7.5.2), and the §7.5.3 two-field deinterleave. The decoded alpha
gradient round-trips with sub-LSB mean-abs-error (the residual of
ffmpeg's 16→12-bit alpha resample; the bitstream alpha is lossless per
§7.1.2). See `tests/ffmpeg_cross_decode.rs` for the black-box
acceptance harness.

The symmetric **progressive 4444 + alpha** forward path
(`encode_frame_with_alpha`, single picture, `interlace_mode == 0`)
cross-decodes through `prores_ks` at 64.77-64.81 dB luma PSNR (ap4h at
64×48 and 128×96, ap4x at 64×48). It drives the same 4:4:4 + genuine
12-bit + 16-bit-alpha-entropy paths as the interlaced cases but
exercises the §7.2 Figure 4 **progressive** block scan instead of the
§7.5.3 field-pair deinterleave; the decoded alpha gradient again
round-trips with sub-LSB mean-abs-error (≈0.17).

The mainstream **progressive 4:2:2** forward path (`encode_frame_with_depth`,
single picture, §7.2 Figure 4 scan) cross-decodes through ffmpeg's
`prores` decoder for all four base profiles — apco / apcs / apcn / apch —
at 62.4-64.2 dB luma PSNR across 8-, 10-, and 12-bit sources (64×48 and
128×96). The 10-/12-bit cases drive genuine high-bit-depth `read_sample`
input through the §7.5.1 level shift; since ffmpeg's 4:2:2 decode is
10-bit internally, all cases compare at 10-bit (8-bit ⟵ `<< 2`, 12-bit
⟵ `>> 2`), and a left/right luma-ramp assertion guards against a
mis-scanned picture. See `tests/ffmpeg_cross_decode.rs`.

The symmetric **progressive 4:4:4 without alpha** forward path
(`encode_frame_with_depth` with `ChromaFormat::Y444`, single picture,
§7.2 Figure 4 scan, `alpha_channel_type == 0`) cross-decodes through
ffmpeg's `prores_ks` decoder for both 4444 profiles — ap4h and ap4x —
at 64.74-64.97 dB luma PSNR across 8-, 10-, and 12-bit sources (64×48
and 128×96). This is the mainstream no-alpha 4:4:4 path: a
registry-built encoder asked for `PixelFormat::Yuv444P*` with no 4th
alpha plane takes it, and it was previously only validated by
self-roundtrip — the existing 4:4:4 ffmpeg coverage was all on the
4444 + alpha entry point. The full-resolution 4:4:4 chroma (twice the
chroma macroblock grid of 4:2:2) exercises the §7.2 progressive block
scan with the deeper chroma slice payload 4:2:2 cannot reach; ffmpeg's
ap4h/ap4x no-alpha decode is 12-bit internally, so all cases compare
at 12-bit (8-bit ⟵ `<< 4`, 10-bit ⟵ `<< 2`, 12-bit as-is). See
`tests/ffmpeg_cross_decode.rs`.

## Usage

```toml
[dependencies]
oxideav-core = "0.1"
oxideav-codec = "0.1"
oxideav-prores = "0.0"
```

### Profile selection

The encoder picks a profile from the combination of `pixel_format` and
`bit_rate`:

| `pixel_format` | `bit_rate` hint (bps)  | Picked profile |
|----------------|------------------------|----------------|
| `Yuv422P`      | `<= 70_000_000`        | Proxy          |
| `Yuv422P`      | `<= 125_000_000`       | LT             |
| `Yuv422P`      | `<= 180_000_000` or `None` | Standard   |
| `Yuv422P`      | `> 180_000_000`        | HQ             |
| `Yuv444P`      | `>= 400_000_000`       | 4444 XQ        |
| `Yuv444P`      | anything else          | 4444           |

```rust
use oxideav_codec::CodecRegistry;
use oxideav_core::{CodecId, CodecParameters, Frame, PixelFormat};

let mut reg = CodecRegistry::new();
oxideav_prores::register(&mut reg);

let mut params = CodecParameters::video(CodecId::new("prores"));
params.width = Some(1920);
params.height = Some(1080);
params.pixel_format = Some(PixelFormat::Yuv422P);
params.bit_rate = Some(220_000_000); // -> 422 HQ

let mut enc = reg.make_encoder(&params)?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

### Codec id

- Codec: `"prores"`; accepted pixel formats `Yuv422P`, `Yuv444P`.
- Keyframe-only (all ProRes frames are intra).

## Performance

A Criterion benchmark covers the decode hot path. Inputs are
synthesized in-process via this crate's own encoder (no external
fixtures); the encode cost is paid once at setup and excluded from the
measured region. Each case decodes a single 128×96 frame:

| Case                       | Profile / format        | median time |
|----------------------------|-------------------------|-------------|
| `apcn_422_8bit_128x96`     | 422 Standard, 8-bit     | ~68 µs      |
| `ap4h_444_8bit_128x96`     | 4444, 8-bit             | ~106 µs     |
| `apcn_422_10bit_128x96`    | 422 Standard, 10-bit LE | ~69 µs      |

Numbers are a single-machine baseline (Apple Silicon, `--measurement-time
3`); treat them as relative, not absolute. The 4:4:4 case is ~1.5× the
4:2:2 cost as expected (full-width chroma planes carry twice the
coefficient data). 10-bit tracks 8-bit closely since the extra work is
output sample packing, not entropy decode.

Run it:

```sh
cargo bench --bench decode -- --warm-up-time 1 --measurement-time 3
```

A second Criterion benchmark (`benches/encode.rs`) covers the encode
hot path across all six RDD 36 profiles on a synthetic 128×96 gradient,
each at its default `quantization_index`, plus a 10-bit and an
interlaced (top-field-first) case:

| Case                                    | Profile / format          | median time |
|-----------------------------------------|---------------------------|-------------|
| `apco_422_8bit_128x96`                  | 422 Proxy, 8-bit          | ~63 µs      |
| `apcs_422_8bit_128x96`                  | 422 LT, 8-bit             | ~62 µs      |
| `apcn_422_8bit_128x96`                  | 422 Standard, 8-bit       | ~72 µs      |
| `apch_422_8bit_128x96`                  | 422 HQ, 8-bit             | ~85 µs      |
| `ap4h_444_8bit_128x96`                  | 4444, 8-bit               | ~111 µs     |
| `ap4x_444_8bit_128x96`                  | 4444 XQ, 8-bit            | ~136 µs     |
| `apcn_422_10bit_128x96`                 | 422 Standard, 10-bit LE   | ~80 µs      |
| `apcn_422_8bit_interlaced_tff_128x96`   | 422 Standard, interlaced  | ~79 µs      |

Encode cost climbs as the profile's default qi falls (Proxy qi 8 →
4444 XQ qi 1): finer quantisation leaves more non-zero coefficients for
the run/level/sign coder to emit. The 4:4:4 profiles cost ~1.5–2× their
4:2:2 counterparts (full-width chroma planes). 10-bit tracks 8-bit
closely; the interlaced case (two field pictures) matches the
progressive Standard cost since total coefficient work is unchanged.

```sh
cargo bench --bench encode -- --warm-up-time 1 --measurement-time 3
```

### Fast paths in the DCT module

Both the decoder and the encoder probe the 8x8 block before invoking
the textbook forward / inverse DCT and dispatch to a constant-time
fast path on the relevant degenerate case:

| Path                          | Trigger condition                                | Cost vs general                              |
|-------------------------------|--------------------------------------------------|----------------------------------------------|
| [`dct::idct8x8_dc_only`]      | dequantised block has all 63 AC = 0              | 1 multiply + 64 stores vs 64 × 16 mul-adds   |
| [`dct::fdct8x8_constant`]     | input block has all 64 samples bit-identical     | 1 multiply + 63 stores vs 64 × 16 mul-adds   |

The decode-side `idct8x8_dc_only` fires whenever the entropy coder's
`endOfData()` produced a smooth block — common at higher quantisation
indices on natural content. The encode-side `fdct8x8_constant` fires
on boundary-clamp pad blocks (RDD 36 §7.5.1 replicates the last
sample to fill a partial MB row), large flat areas, and the smooth
regions of synthetic gradients. Both paths are verified bit-exact
against the general DCT loops on every plausible input level
(`dct::tests::idct_dc_only_matches_general_idct`,
`dct::tests::fdct_constant_matches_general_fdct`) and end-to-end via
`encoder::tests::constant_flat_frame_decodes_pixel_exact_at_hq`, which
encodes a flat-value-`v` 64×48 frame at HQ (qi=2) and asserts every
Y-plane byte round-trips to `v` exactly.

## Fuzzing

A `cargo-fuzz` harness lives under `fuzz/` with three panic-free
targets that drive arbitrary attacker-controlled bytes through
ProRes's public decode entry points and header parsers:

* `decode_packet` — feeds bytes into [`decoder::decode_packet`]. Covers
  the RDD 36 §5.1 frame() outer framing, §6.1.1 frame_header() (`shall
  refuse` clauses for `bitstream_version > 1`, reserved interlace_mode
  3, v0 stream constraints on chroma_format / alpha_channel_type,
  out-of-range qmat entries), §6.3 picture_header() + slice_table(), the
  §5.3 + §7.1.1 run/level/sign coefficient coder, the §7.1.2 +
  Table 12-14 alpha run-length VLC for ap4h / ap4x, and the §5.1 ProRes
  RAW (`aprn` / `aprh`) refusal path.
* `decode_packet_with_depth` — same parse chain, but additionally
  exercises [`decoder::decode_packet_with_depth`]'s caller-supplied
  `(BitDepth, ChromaFormat)` output formatter (RDD 36 §7.5.1 level
  shifts for `b = 8 / 10 / 12`). The bit-depth and chroma tags are
  derived from the input's own first byte so libFuzzer steers mutations
  across all six (depth × chroma) combinations the registry route can
  reach.
* `parse_headers` — feeds independent input slices into all four
  public header parsers in [`frame`]: [`frame::parse_frame`] (§5.1
  outer framing), [`frame::parse_frame_header`] (§6.1.1 frame_header
  + optional 64-byte luma + chroma quant matrices), [`frame::parse_picture_header`]
  (§6.3 picture_header), and [`frame::parse_slice_header`] (§5.3
  slice_header, both with-alpha and no-alpha shapes). These parsers
  are reachable from any application that bypasses the top-level
  decode path (e.g. a sample-bytes inspector) and are far cheaper
  to drive than the full decode chain, so libFuzzer can explore the
  header arithmetic and quant-matrix loading branches at a higher
  rate per second.

The two decode-pipeline harnesses peek at the wire-stream width /
height (bytes 16..18 and 18..20, BE u16) and bail out if `width ×
height` exceeds 65 536 pixels, so libFuzzer doesn't waste cycles on
inputs that the upstream OOM cap would reject anyway. A daily 30-minute
GitHub Actions run is scheduled under `.github/workflows/fuzz.yml`,
splitting the budget across the three targets.

```sh
# nightly toolchain required by libfuzzer-sys
cd fuzz && cargo +nightly fuzz run decode_packet -- -max_total_time=60
```

## License

MIT — see [LICENSE](LICENSE).
