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
acceptance criteria hold with large margin.

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

## License

MIT — see [LICENSE](LICENSE).
