# oxideav-prores

Pure-Rust Apple ProRes codec — decoder + encoder for **ProRes 422
Proxy / LT / Standard** and **ProRes 4444** (without alpha).

Part of the [oxideav](https://github.com/OxideAV/oxideav-workspace) framework — a
100% pure Rust media transcoding and streaming stack. No C libraries, no FFI
wrappers, no `*-sys` crates.

## Status

| Profile           | FourCC | Input pixel format | State   |
|-------------------|--------|--------------------|---------|
| 422 Proxy         | `apco` | `Yuv422P` 8-bit    | decode + encode |
| 422 LT            | `apcs` | `Yuv422P` 8-bit    | decode + encode |
| 422 Standard      | `apcn` | `Yuv422P` 8-bit    | decode + encode |
| 4444              | `apch` | `Yuv444P` 8-bit    | decode + encode (no alpha) |
| 4444 XQ (`ap4h`)  | —      | —                  | not yet |

Alpha-plane carriage (`Yuva444P` → 4:4:4:4) is not implemented; the
4444 support is YUV-only. The wire format is RDD 36-structural (frame
header with `icpf` magic, picture header, slice table) but uses a
simplified exp-Golomb entropy layer, so streams are round-trip-exact
with this crate but not bit-compatible with third-party ProRes
decoders.

## Usage

```toml
[dependencies]
oxideav-prores = "0.0"
```

## License

MIT — see [LICENSE](LICENSE).
