# oxideav-prores

Pure-Rust **Apple ProRes** codec — decoder + encoder for all six
ProRes video profiles (422 Proxy / LT / Standard / HQ and 4444 /
4444 XQ). 8-bit and 10-bit Y'CbCr; alpha not currently carried.

Part of the [oxideav](https://github.com/OxideAV/oxideav-workspace)
framework but usable standalone. No C libraries, no FFI wrappers, no
`*-sys` crates.

## Status

| Profile        | FourCC | Pixel formats                    | State           |
|----------------|--------|----------------------------------|-----------------|
| 422 Proxy      | `apco` | `Yuv422P`, `Yuv422P10Le`         | decode + encode |
| 422 LT         | `apcs` | `Yuv422P`, `Yuv422P10Le`         | decode + encode |
| 422 Standard   | `apcn` | `Yuv422P`, `Yuv422P10Le`         | decode + encode |
| 422 HQ         | `apch` | `Yuv422P`, `Yuv422P10Le`         | decode + encode |
| 4444           | `ap4h` | `Yuv444P`, `Yuv444P10Le`         | decode + encode (no alpha) |
| 4444 XQ        | `ap4x` | `Yuv444P`, `Yuv444P10Le`         | decode + encode (no alpha) |

Bit-depth selection follows the stream's `CodecParameters::pixel_format`
— RDD 36 §5 carries no per-frame bit-depth syntax element; §7.5.1
defines the conversion from reconstructed color component values to
pixel samples of arbitrary bit depth `b`. Pass `Yuv422P10Le` /
`Yuv444P10Le` to get 10-bit planar output (LE u16 pairs, valid range
`0..=1023`); pass `Yuv422P` / `Yuv444P` (or omit `pixel_format`) to
get 8-bit planar output. 12-bit + alpha are separate follow-ups.

The 4444 and 4444 XQ profiles share the same bitstream structure as
4444 — XQ is selected when the caller requests the highest quality
tier and produces a larger packet at lower quantisation. Alpha-plane
carriage (RDD 36 A' coding, a `Yuva444P` pixel format) is not
implemented; the core `PixelFormat` enum does not yet carry a 4:4:4:4
variant.

The wire format is RDD 36-structural (frame header with `icpf` magic,
picture header, slice table, per-MB 8x8 DCT blocks) but the entropy
layer is a simplified exp-Golomb coding on zig-zag-scanned
coefficients. Streams are round-trip-exact with this crate but **not**
bit-compatible with Apple ProRes decoders.

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
