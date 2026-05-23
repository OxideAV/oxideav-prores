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

### Interlaced (RDD 36 §5.1, §6.2, §7.5.3)

A frame's `interlace_mode` (0 = progressive, 1 = top-field-first,
2 = bottom-field-first) controls whether the encoded bitstream carries
one picture or two (one per field). Source rows interleave across
fields per §7.5.3: top field = source rows 0, 2, 4, …; bottom field =
rows 1, 3, 5, …. The encoder splits the source into two field pictures
(each at `(height + 1) / 2` or `height / 2` rows) and emits them in
temporal order; the decoder reverses the deinterleave. Each field
picture uses the interlaced block scan (§7.2 Figure 5) instead of the
progressive Figure 4. See [`encoder::encode_frame_interlaced`].

Streams produced by `encode_frame_interlaced` for apcn / apch cross-decode
through ffmpeg's `prores_ks` decoder at ≥ 64 dB luma PSNR — both 8-bit
(TFF and BFF, 64×48 and 128×96) and **genuine 10-bit** field-pair
packing (TFF/BFF at 64×48, TFF at 128×96; 64.40-64.47 dB). The 10-bit
cases drive a true 10-bit LE source through `read_sample`'s
`BitDepth::Ten` branch (RDD 36 §7.5.1 level shift for `b = 10`) feeding
the §7.5.3 two-field deinterleave — not an 8-bit value padded into
10-bit storage.

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

## License

MIT — see [LICENSE](LICENSE).
