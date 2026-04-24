# Blocked: SMPTE RDD 36 bit-level spec not in docs/

**Status:** Blocked — no further entropy-layer work until the normative
SMPTE RDD 36 document is added to `docs/video/prores/`.

## Context

Round 2 aimed to replace the simplified signed exp-Golomb placeholder
entropy coder in this crate with the spec-correct RDD 36 run-level-sign
scheme (DC differential + AC run/level with Rice/Golomb-Rice codes) so
that real ffmpeg-produced ProRes `.mov` samples decode bit-exactly.

Workspace policy for this round explicitly forbids reading third-party
source (ffmpeg `prores_ks` / `prores_aw`, Apple QuickTime, Atomos,
BlackMagic). The ONLY acceptable reference is the SMPTE RDD 36 PDF or
equivalent Apple-authored bitstream specification present in
`docs/video/prores/`.

## Documents checked in `docs/video/prores/`

| File | Date | Pages | Verdict |
| --- | --- | --- | --- |
| `Apple_ProRes_2022.pdf` | April 2022 | 22 | **Marketing whitepaper.** Covers family overview (4444 XQ / 4444 / 422 HQ / 422 / 422 LT / 422 Proxy), target data rates, PSNR charts, performance benchmarks, alpha channel properties, glossary. Zero bitstream syntax, zero codeword tables, zero entropy-coder description. Page 4 explicitly labels FFmpeg an "unauthorized codec implementation". |
| `Apple_ProRes_RAW_2023.pdf` | May 2023 | 26 | **Marketing whitepaper for a different codec** (ProRes RAW, sensor Bayer format). Sections: Introduction / About RAW Video / Data Rate / Performance / Plug-ins / Using in Final Cut Pro / Conclusion. Not applicable to ProRes 422/4444. |

Workspace-wide search (`find docs -iname "*rdd*" -o -iname "*smpte*"`)
returned no additional matches.

## What remains in the crate today

The existing implementation (commit `d2aeec3`, Round 1) uses a
simplified in-house format:

- `src/slice.rs` — each block emits `[DC se(v) | 63 AC se(v)]` using
  unsigned/signed exp-Golomb (H.264-style `ue(v)` / `se(v)`).
- `src/bitstream.rs` — MSB-first bit reader/writer with `read_ue` /
  `read_se` helpers.
- Internal roundtrip (encoder → decoder, same crate) is bit-exact.
- FourCC dispatch correctly tags `apco|apcs|apcn|apch|ap4h|ap4x`
  samples as codec id `"prores"` (Round 1 work), so a demuxer + this
  decoder will agree on identity but NOT on the per-slice coefficient
  bytes for a real ffmpeg-produced file.

## Why this is genuinely blocked, not deferrable

RDD 36 entropy coding is not reconstructable from the material at hand:

- The spec defines per-component Rice-code order tables for DC and AC
  (codebook indexed by previous coded magnitude), switching between
  Rice and Golomb-Rice regions at fixed thresholds, with a
  context-dependent sign bit placement. None of this is in the
  marketing PDFs.
- Implementing "something plausible" would not be interop-compliant
  and would mislead downstream users who see the crate's FourCC
  dispatch succeed.

## Unblock procedure

1. Add the SMPTE RDD 36 PDF (`SMPTE RDD 36:20xx — Apple ProRes Bitstream
   Syntax and Decoding Process`) to `docs/video/prores/`. SMPTE RDD 36
   is a publicly available standard from SMPTE (paid purchase).
2. Re-run this task. The work plan is:
   - Port the per-component Rice-order tables into `src/slice.rs`
     (DC differential, AC run codebook, AC level codebook).
   - Replace `read_ue`/`read_se` call sites in slice decode with
     `read_rice(k)` / `read_golomb_rice(k)` primitives whose `k`
     adapts via the spec's documented state machine.
   - Add `tests/ffmpeg_interop.rs` that runs
     `ffmpeg -f lavfi -i testsrc=...` → `-c:v prores_ks -profile:v 2`,
     decodes with this crate, and asserts luma PSNR >= 35 dB against
     the YUV source.

## Remaining gaps (unchanged from Round 1)

- AC-coefficient run-level-sign entropy layer (this task).
- Per-component quantization matrices (currently uniform stand-in).
- Real DCT (currently a placeholder transform).

None of the above can be advanced without the RDD 36 normative text.
