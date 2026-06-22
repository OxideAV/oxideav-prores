# oxideav-prores

Pure-Rust **Apple ProRes** codec — decoder + encoder for all six
ProRes video profiles (422 Proxy / LT / Standard / HQ and 4444 /
4444 XQ). 8-bit, 10-bit, and 12-bit Y'CbCr; lossless alpha plane on the
4444 / 4444 XQ profiles.

Part of the [oxideav](https://github.com/OxideAV/oxideav-workspace)
framework but usable standalone. Implemented from SMPTE RDD 36 (no C
codec libraries linked or wrapped, no `*-sys` crates).

## Status

| Profile        | FourCC | Pixel formats                                      | State           |
|----------------|--------|----------------------------------------------------|-----------------|
| 422 Proxy      | `apco` | `Yuv422P`, `Yuv422P10Le`, `Yuv422P12Le`            | decode + encode |
| 422 LT         | `apcs` | `Yuv422P`, `Yuv422P10Le`, `Yuv422P12Le`            | decode + encode |
| 422 Standard   | `apcn` | `Yuv422P`, `Yuv422P10Le`, `Yuv422P12Le`            | decode + encode |
| 422 HQ         | `apch` | `Yuv422P`, `Yuv422P10Le`, `Yuv422P12Le`            | decode + encode |
| 4444           | `ap4h` | `Yuv444P`, `Yuv444P10Le`, `Yuv444P12Le` (+ alpha)  | decode + encode |
| 4444 XQ        | `ap4x` | `Yuv444P`, `Yuv444P10Le`, `Yuv444P12Le` (+ alpha)  | decode + encode |

Both progressive and interlaced (top-field-first / bottom-field-first)
modes decode and encode across all six profiles.

## Bit depth

Bit-depth selection follows the stream's `CodecParameters::pixel_format`
— RDD 36 §5 carries no per-frame bit-depth syntax element; §7.5.1
defines the conversion from reconstructed component values to pixel
samples of arbitrary bit depth `b`. Pass `Yuv422P10Le` / `Yuv444P10Le`
for 10-bit planar output (LE u16, range `0..=1023`), `Yuv422P12Le` /
`Yuv444P12Le` for 12-bit (`0..=4095`), or `Yuv422P` / `Yuv444P` (or omit
`pixel_format`) for 8-bit.

## Output range (RDD 36 §7.5.1)

The reconstructed-value → pixel-sample conversion `s = clamp(round(2^b *
(v + 256) / 512))` of §7.5.1 offers two clamp choices. The decoder
exposes both through [`decoder::OutputRange`]:

- **`Full`** (default) — `nmin = 0`, `nmax = 2^b − 1`; samples utilise
  all available quantization levels. Byte-identical to the prior
  behaviour, so [`decoder::decode_packet_with_depth`] and the registry
  path are unchanged.
- **`Video`** — `nmin = 1`, `nmax = 2^b − 2`; confines colour samples to
  the permissible video quantization levels, avoiding the BT.601/BT.709
  synchronization/timing reference codes (the extreme codes `0` and
  `2^b − 1`).

Select it via [`decoder::decode_packet_with_options`] or
`ProResDecoder::set_output_range` on the direct API. The choice affects
only the Y/Cb/Cr clamp bounds; entropy decode, inverse quantisation, the
IDCT, and the alpha plane (§7.5.2 always maps the full opacity range) are
unaffected.

## Alpha plane

The 4444 / 4444 XQ profiles support a per-pixel alpha channel coded
losslessly per RDD 36 §5.3.3 + §7.1.2 (raster-scan run-length code +
differential VLC, Tables 12-14). Alpha is exposed as a 4th `VideoPlane`
on the decoded `VideoFrame` (after Y/Cb/Cr); the encoder accepts the
same shape via [`encoder::encode_frame_with_alpha`]. The core
`PixelFormat` enum does not yet carry `Yuva422P` / `Yuva444P` variants,
so pixel-format reporting stays `Yuv4(2|4)4P*` — the caller checks
`frame.planes.len() == 4` to detect alpha. `FrameHeader::alpha_kind()`
returns the named Table 7 variant (`None` / `Bits8` / `Bits16`).

### §7.5.3 scanned-alpha array length (reference-bitstream note)

A literal reading of RDD 36 §7.5.3 — the per-slice scanned-alpha array
"does not include alpha values for the excess row(s) of pixels at the
bottom of slices with `i = height_in_mb − 1`" — suggests sizing the
bottom macroblock row's alpha array to the *visible* row count. Real
ProRes 4444 bitstreams (including the in-tree `4444-with-alpha`
1920×1080 reference fixture, whose bottom MB row spans only 8 visible
rows) instead carry the **full 16-row** array and let the decoder
discard the excess rows on paste, exactly as it already discards the
excess right-edge *columns*. The §7.5.3 exclusion therefore governs
which rows a decoder writes to the frame buffer, not the coded array
length; the codec reads/writes the full 16-row array to stay
bit-compatible with the reference. `tests/alpha_bit_depth.rs` and
`tests/interlaced_alpha_partial_field.rs` lock this against
non-MB-aligned progressive heights and non-MB-aligned interlaced field
heights respectively (8-bit and 16-bit coded alpha, 8/10/12-bit output).

The conclusion is pinned **directly against the reference bytes** by
`tests/alpha_array_length_reference.rs`: it pulls the raw
`scanned_alpha()` blob of a real bottom-MB-row slice out of the
`4444-with-alpha` `input.mov` and shows that decoding at the
§7.5.3-literal visible-row length (`128 × 8 = 1024` values) overruns the
coded run/level stream, while the full MB-row height (`128 × 16 = 2048`
values) decodes exactly — a single escape diff to `0xFFFF` plus one run
of 2048. The full-height decode holds for every slice of the bottom MB
row and for an interior row. So the §7.5.3 exclusion is a *write*
constraint (which rows reach the frame buffer), not a coded-length
reduction; the array is uniformly `16 * slice_size_in_mb[j] * 16` values.
This is a standing DOCS-GAP candidate against the §7.5.3 wording.

`tests/alpha_plane_reference.rs` closes the loop end-to-end: it
reconstructs the entire frame's alpha plane independently from the same
`4444-with-alpha` bitstream (decode every slice's `scanned_alpha()` blob,
apply the §7.5.2 conversion, place per §7.5.3 with the excess rows and
columns discarded — sharing no placement code with the decoder) and
asserts the **full decoder's** emitted alpha plane matches it
byte-for-byte across all 1020 slices, including the partial bottom MB
row. The comparison runs at the native 12-bit and at the 10-/8-bit
§7.5.2 demote depths the corpus never otherwise exercises.

## Frame-header metadata (RDD 36 §5.1.1 / §6.2)

The encoder fills the descriptive frame-header fields
(`aspect_ratio_information`, `frame_rate_code`, `color_primaries`,
`transfer_characteristic`, `matrix_coefficients`) automatically from
[`CodecParameters::frame_rate`], or explicitly via
[`encoder::EncoderConfig::with_meta`]:

```rust
use oxideav_prores::encoder::{make_encoder_with_config, EncoderConfig};
use oxideav_prores::frame::FrameMeta;

let cfg = EncoderConfig::default().with_meta(FrameMeta {
    aspect_ratio_information: 3,  // 16:9
    frame_rate_code: 8,          // 60 fps
    color_primaries: 9,          // BT.2020
    transfer_characteristic: 16, // SMPTE ST 2084 (PQ)
    matrix_coefficients: 9,      // BT.2020 NCL
});
let enc = make_encoder_with_config(&params, cfg)?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

Each descriptive field has a symmetric forward/reverse helper pair and a
typed accessor on the parsed `FrameHeader` returning `Option<…>` — the
outer-Option discriminant distinguishes the spec's "unknown /
unspecified" codes from named values:

- `frame_rate` ⇄ [`frame::rational_from_frame_rate_code`] /
  [`frame::frame_rate_code_from_rational`] (Table 4).
- `aspect_ratio` ⇄ [`frame::aspect_ratio_from_code`] (Table 3).
- `color_primaries_kind` → [`frame::ColorPrimaries`] (Table 5).
- `matrix_coefficients_kind` → [`frame::MatrixCoefficients`] (Table 6,
  with `luma_coefficients()` returning the K_R/K_G/K_B triple).
- `transfer_characteristic_kind` → [`frame::TransferCharacteristic`]
  (§6.1.1: BT.1886 / ST 2084 / HLG).
- `encoder_identifier` / `encoder_identifier_str` (§6.1.1 f(32) FourCC;
  this crate writes [`frame::ENCODER_IDENTIFIER`] = `oxav`).

For the re-encode direction, [`FrameHeader::meta`] folds all five
descriptive bytes of a parsed header back into a [`frame::FrameMeta`] so
a transcode forwards the source's aspect / rate / colour metadata
verbatim.

[`CodecParameters::frame_rate`]: https://docs.rs/oxideav-core/latest/oxideav_core/struct.CodecParameters.html#structfield.frame_rate

## Quantization-matrix provenance (RDD 36 §6.1.1 / §7.2)

The frame header carries two flags, `load_luma_quantization_matrix` and
`load_chroma_quantization_matrix`, that select whether each component's
quantization weight matrix is custom (carried inline) or the §7.2
default (all 64 weights = 4). The chroma side has a §6.1.1 wrinkle: when
its flag is `0`, the *luma* matrix is reused for chroma — which is itself
the custom luma matrix if that flag is `1`, else the default.

Both raw flags are now surfaced on the parsed [`frame::FrameHeader`]
(`load_luma_quantization_matrix` / `load_chroma_quantization_matrix`),
and [`FrameHeader::quantization_matrix_source`] folds the chroma
derivation into the [`frame::QuantizationMatrixSource`] enum
(`CustomChroma` / `LumaCustom` / `Default`) for stream-inspection and
transcode-provenance callers. The decoder already applies the §6.1.1
fallback when reconstructing `chroma_qmat`; this only exposes which of
the three cases produced it.

## Picture geometry (RDD 36 §6.2)

[`FrameHeader::picture_geometry`] folds the §6.2 derivation of the
*encoded* picture geometry out of the header's source dimensions into a
[`frame::PictureGeometry`]: `width_in_mb` (`ceil(horizontal_size / 16)`,
identical for both fields), the per-picture `picture_vertical_size` field
split for interlaced frames (`top = (h + 1) / 2`, `bottom = h / 2`, with
the leading field selected by the TFF/BFF `interlace_mode`),
`height_in_mb`, the trailing field height, and the §6.2 / §7.5.3
right/bottom crop amounts a decoder discards when the source size is not a
multiple of 16. The slice-partitioning bridges
`PictureGeometry::slice_count(log2)` /
`slices_per_mb_row(log2)` take the picture header's
`log2_desired_slice_size_in_mb`. This is the geometry the decode loop
computes internally, surfaced for stream-inspection / muxer / transcode
callers to size buffers or cross-check container-declared dimensions
without re-deriving the rounding and field-split rules.

## Encoder controls

All controls flow through [`encoder::EncoderConfig`] +
[`encoder::make_encoder_with_config`].

- **Quantisation index** (`with_quantization_index`, §7.3 / Table 15) —
  per-profile defaults `8 / 6 / 4 / 2 / 2 / 1`; valid range `1..=224`.
- **Macroblocks-per-slice** (`with_mbs_per_slice`, §5.3) — `{1, 2, 4, 8}`,
  default 8; smaller values trade packet size for error resilience.
- **Explicit profile** (`with_profile` / `for_profile`) — overrides the
  `bit_rate` → profile heuristic; the profile's chroma format must match
  the requested `PixelFormat`.
- **Two-pass rate control** (`with_rate_control`) — per-frame
  binary-search on `quantization_index` to hit `bit_rate / fps` within
  ±5 %, returning the best candidate when the target is unreachable.
- **Quantisation matrices** (`with_quant_matrices`, §5.3.4 + §7.3) —
  defaults to the flat all-4s matrix; a built-in perceptual preset
  (`EncoderConfig::perceptual` / `perceptual_for_profile`) loads
  JPEG K.1/K.2-derived matrices clamped to the `2..=63` range, blended
  toward flat in proportion to the profile's quality tier.
- **Constant-frame-size stuffing** (`with_min_frame_size`, §5.1.2 +
  §6.1.2) — pads short frames up to a minimum on-wire `frame_size`; a
  padded frame decodes bit-identically to its unpadded twin.
- **Interlacing** (`with_interlace_mode` / [`encoder::encode_frame_interlaced`],
  §5.1 / §6.2 / §7.5.3) — 0 progressive, 1 TFF, 2 BFF; value 3 reserved.

## Bitstream conformance (RDD 36 §6.4)

The decoder enforces every "decoder shall refuse" clause:

| Spec clause | Constraint |
|-------------|-----------|
| §6.1.1 `bitstream_version` | reject any value > 1 |
| §6.4 v0 stream rules | a v0 stream must be 4:2:2 with no alpha |
| §6.1.1 Table 2 `interlace_mode` | value 3 reserved |
| §6.1.1 qmat entries | every entry in `2..=63` |

The encoder picks the lowest legal `bitstream_version` (v0 for 4:2:2
no-alpha; v1 otherwise). Typed accessors `FrameHeader::interlace_kind`,
`PictureHeader::mbs_per_slice`, and `SliceHeader::qscale` fold the
respective reverse mappings onto the parsed headers.

### Version-variant forward compatibility (RDD 36 §6.4)

§6.4 lets a future *version variant* append informative bytes after a
structure's defined syntax without breaking existing decoders, and
mandates that "decoders shall use the specified size — rather than
inference from the syntax itself — to determine the start of the
immediately following syntax structure." The decoder therefore locates
the next `picture()` (the second field of an interlaced frame) from the
first picture's **declared** `picture_size` (§6.2.1) rather than from the
sum of the picture header, slice table, and slice payloads it parsed: a
stream that carries `picture_size > header + slice_table + Σslice` (the
trailing variant bytes) decodes identically to its base-bitstream twin,
while a stream whose parsed payload *exceeds* its declared `picture_size`
is refused as corrupt. Base bitstreams (where the two totals are equal)
are byte-unaffected.

## ProRes RAW is detected and refused

Apple **ProRes RAW** (`aprn` / `aprh`) is a separate format outside the
scope of SMPTE RDD 36. This crate refuses it cleanly at the FourCC level
([`is_prores_raw_fourcc`] / [`PRORES_RAW_FOURCCS`]) and in-stream (the
`aprh` marker at the `icpf` offset yields a specific `Unsupported`
error), so a dispatcher can tell "ProRes RAW, unsupported" apart from
"not ProRes at all" rather than mis-decoding it. Decoding it would
require Apple's proprietary bitstream documentation and is not
implemented.

## FourCC routing helpers

The crate root exposes the FourCC ⇄ profile mapping in both directions
so a demuxer and muxer share one source of truth:
[`profile_for_fourcc`] / [`codec_id_for_fourcc`] (on-wire → profile /
codec id) and [`fourcc_for_profile`] (encoder profile → canonical
lowercase FourCC), with [`PRORES_FOURCCS`] listing the six codes. The
QuickTime / MXF sample-table assembly itself lives in the container
crate.

## Interop validation

The crate self-roundtrips every profile bit-exactly and cross-decodes
against an external ProRes decoder (used as a black-box validator only,
no source consulted) across 8-/10-/12-bit, progressive and interlaced
(TFF/BFF), and 4444 ± alpha at 58–65 dB luma PSNR. In-tree fixtures
under `docs/video/prores/fixtures/` ship a reference YUV sidecar with a
pinned SHA; `tests/{progressive,interlaced}_decode_sha.rs` and
`tests/encoder_output_sha.rs` lock the decode and encode byte streams,
reporting the fixture's fixed-point reference SHA alongside ours so the
~1-LSB float-vs-fixed IDCT divergence permitted by §7.4 stays visible.
`tests/idct_annex_a.rs` runs the full RDD 36 Annex A IDCT accuracy
qualification against the production [`dct::idct8x8`] — all five
acceptance criteria hold with large margin. `tests/alpha_bit_depth.rs`
(plus white-box unit tests in `decoder.rs`) lock the §7.5.2 decoded-alpha
→ pixel-alpha bit-depth conversion `alphaSample = round((2^b − 1) *
alpha ÷ mask)` without the external validator: an 8-bit-alpha 4444 frame
decoded at 8-/10-/12-bit output matches the §7.5.2 formula exactly (alpha
is coded losslessly per §7.1.2), covering the identity, promotion,
demotion, endpoint, and round-half-up cases — including non-MB-aligned
progressive heights and widths (`tests/roundtrip.rs` covers the §7.5.3
partial-bottom-MB-row array, the right-edge column exclusion, and a
both-axes-partial corner MB with 16→12-bit demotion). The interlaced
counterpart in `tests/interlaced_alpha_partial_field.rs` combines the
§6.2 field split, the §7.5.3 top/bottom deinterleave, and the partial-row
alpha array across TFF/BFF at 8/10/12-bit output for **both** 8-bit
(Table 13) and 16-bit (Table 14) coded alpha, and at a non-MB-aligned
width as well as a non-MB-aligned field height (the per-field corner MB).

Streams produced by this crate's encoder use the spec's entropy coder
for colour and a plain run-length code for alpha (the alternative path
permitted by §7.1.2); both are bit-exact with this crate's decoder.

## Usage

```toml
[dependencies]
oxideav-core = "0.1"
oxideav-codec = "0.1"
oxideav-prores = "0.0"
```

The encoder picks a profile from `pixel_format` + `bit_rate`:

| `pixel_format` | `bit_rate` hint (bps)      | Profile  |
|----------------|----------------------------|----------|
| `Yuv422P`      | `<= 70_000_000`            | Proxy    |
| `Yuv422P`      | `<= 125_000_000`           | LT       |
| `Yuv422P`      | `<= 180_000_000` or `None` | Standard |
| `Yuv422P`      | `> 180_000_000`            | HQ       |
| `Yuv444P`      | `>= 400_000_000`           | 4444 XQ  |
| `Yuv444P`      | anything else              | 4444     |

```rust
use oxideav_codec::CodecRegistry;
use oxideav_core::{CodecId, CodecParameters, PixelFormat};

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

All ProRes frames are intra (keyframe-only); accepted pixel formats are
`Yuv422P` and `Yuv444P`.

## Performance

Criterion benchmarks cover the decode and encode hot paths. Inputs are
synthesised in-process via this crate's own encoder (no external
fixtures). The decoder and encoder both probe each 8×8 block and
dispatch to a constant-time fast path — `dct::idct8x8_dc_only` (all 63
AC = 0) on decode, `dct::fdct8x8_constant` (all 64 samples identical) on
encode — each verified bit-exact against the general DCT loops.

```sh
cargo bench --bench decode -- --warm-up-time 1 --measurement-time 3
cargo bench --bench encode -- --warm-up-time 1 --measurement-time 3
```

## Fuzzing

A `cargo-fuzz` harness under `fuzz/` ships five panic-free targets
driving attacker-controlled bytes through the public decode entry
points, header parsers, and entropy coders: `decode_packet`,
`decode_packet_with_depth`, `parse_headers`, `decode_entropy` (the
§7.1.1 run/level/sign coder), and `decode_alpha` (the §7.1.2 alpha VLC).
Pipeline harnesses bail on `width × height > 65_536` and the
entropy-coder harnesses cap the declared block / value counts so a
worker never commits a huge allocation. A daily 30-minute GitHub Actions
run is scheduled under `.github/workflows/fuzz.yml`.

```sh
cd fuzz && cargo +nightly fuzz run decode_packet -- -max_total_time=60
```

The malformed-input behaviour the fuzzers explore is *also* pinned by
ordinary `cargo test` cases that run in the standard (nightly-free) CI
matrix, so a refactor of the coders cannot silently regress it even
between fuzz runs: `tests/entropy_alpha_robustness.rs` asserts the
§7.1.1 coefficient coder and the §7.1.2 / Table 12-14 alpha coder
round-trip exactly, terminate at the declared count, and surface a clean
`Err` (never a panic / debug overflow / out-of-bounds index) on every
truncation and on a spread of adversarial byte patterns. The
§7.3 / Table 15 `qScale` map — including its slope-1 → slope-4
discontinuity at the 128/129 boundary, both printed-anchor rows, and the
reserved-index `None` arm of `SliceHeader::qscale` — is pinned
exhaustively over `1..=224` by `tests/qscale_table15.rs`.

## License

MIT — see [LICENSE](LICENSE).
