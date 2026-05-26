# oxideav-prores round-161 profile baseline

This directory captures the profiling-baseline numbers produced by
the `examples/profile_prores.rs` driver that round 161 introduces.
The driver is the durable artefact: any future round (or local
A/B run) can reproduce these numbers + capture per-symbol flame-
graphs against it without re-discovering the harness recipe.

## Headline numbers (round 161, Apple M4 Max, release build)

Each scenario is self-contained (deterministic synthetic gradient
input, no external fixtures). The encode / decode / roundtrip rows
use the same `encode_frame_with_depth` / `decode_packet[_with_depth]`
entry points the `benches/{encode,decode}.rs` Criterion harnesses
target. The `interlaced` row drives the §7.5.3 top-field-first split
(two pictures sharing one frame_header).

```
== encode ==
  encode    apco_422_8bit_128x96           iters=  30    0.151 ms/iter    154.79 MiB/s (raw)  out=1495B/iter (0.061 of input)
  encode    apcn_422_8bit_128x96           iters=  30    0.104 ms/iter    224.96 MiB/s (raw)  out=2536B/iter (0.103 of input)
  encode    apch_422_8bit_128x96           iters=  30    0.194 ms/iter    121.02 MiB/s (raw)  out=3774B/iter (0.154 of input)
  encode    ap4h_444_8bit_128x96           iters=  20    0.218 ms/iter    161.47 MiB/s (raw)  out=4170B/iter (0.113 of input)
  encode    apcn_422_10bit_128x96          iters=  20    0.144 ms/iter    325.77 MiB/s (raw)  out=3325B/iter (0.068 of input)

== decode ==
  decode    apco_422_8bit_128x96           iters=1500    0.062 ms/iter    380.51 MiB/s (raw)
  decode    apcn_422_8bit_128x96           iters=1500    0.064 ms/iter    364.90 MiB/s (raw)
  decode    apch_422_8bit_128x96           iters=1500    0.079 ms/iter    295.93 MiB/s (raw)
  decode    ap4h_444_8bit_128x96           iters=1000    0.100 ms/iter    350.47 MiB/s (raw)
  decode    apcn_422_10bit_128x96          iters=1000    0.064 ms/iter    726.78 MiB/s (raw)

== roundtrip ==
  roundtrip apco_422_8bit_128x96           iters=  30    0.113 ms/iter    207.90 MiB/s (raw)
  roundtrip apcn_422_8bit_128x96           iters=  30    0.138 ms/iter    169.36 MiB/s (raw)
  roundtrip apch_422_8bit_128x96           iters=  30    0.164 ms/iter    143.31 MiB/s (raw)
  roundtrip ap4h_444_8bit_128x96           iters=  20    0.209 ms/iter    168.28 MiB/s (raw)
  roundtrip apcn_422_10bit_128x96          iters=  20    0.145 ms/iter    323.82 MiB/s (raw)

== interlaced (apcn 422 8-bit TFF 128x96) ==
  encode    interlaced-tff                 iters=  20    0.078 ms/iter    302.27 MiB/s (raw)  out=2894B/iter
  decode    interlaced-tff                 iters=1000    0.064 ms/iter    368.28 MiB/s (raw)
```

## Reading the numbers

### Decode

- Decode runs at **300–730 MiB/s of raw output** on the M4 Max for
  128×96 frames. The 10-bit row decoding to packed-LE `Yuv422P10Le`
  reports almost 2× the 8-bit `apcn` row's raw-output throughput —
  same number of compressed input bytes drives twice the output
  bytes per pixel, so the bandwidth number scales accordingly while
  the per-iter wall-time is identical (~64 µs).
- The `apch` (HQ) decode is ~22 % slower per iter than `apcn`
  (Standard) on the same input shape — HQ slices carry more entropy-
  coded coefficients per block (CBP_TABLE expansion + longer AC
  runs), so the entropy decode + dequant loop runs longer per
  64-coeff block.
- `ap4h` 4:4:4 decode is the slowest absolute row (~100 µs) because
  it carries a full-resolution chroma plane (3× luma area instead
  of 2×) — the per-pixel raw throughput is comparable to the 4:2:2
  rows (~350 MiB/s vs ~365 MiB/s on `apcn`), so the cost scales with
  output volume rather than reflecting a 4:4:4-specific slow path.

### Encode

- Encode is dominated by the forward 8×8 DCT + entropy coder.
  Throughput sits in the **120–325 MiB/s range** depending on profile
  and depth.
- The profile ordering `apch (HQ) > apco (Proxy) > apcn (Standard)`
  on per-iter time is counter-intuitive — Proxy carries fewer
  coefficients per slice yet costs more time per frame than
  Standard. The reason is the per-component entropy coder runs the
  same coefficient scan regardless of how many coefficients survive
  quantisation; Standard's larger quant table zeros more high-
  frequency coefficients faster (early-exit on the run-length code's
  EOB), so the inner loop terminates sooner. Profile-specific tuning
  of the AC-EOB shortcut path would close the Proxy vs Standard gap.
- The 10-bit row is **the highest raw-throughput encode** because the
  raw byte count doubles per pixel (~3.3× the byte count of the same
  8-bit input) while the DCT + quant cost stays constant — the DCT
  is depth-agnostic, only the level-shift constant differs (`v = s
  / 2^(b-9) - 256` per RDD 36 §7.5.1).

### Interlaced

- Top-field-first 8-bit 4:2:2 Standard at 128×96 encodes in ~78 µs
  total for the two-picture pair (vs ~104 µs progressive on the same
  shape) — interlaced is **faster per iter** here because each field
  is half-height (48 rows) so each picture's slice grid is smaller;
  the per-picture entropy headers carry the fixed cost, but the
  body shrinks linearly.
- Decode is on par with progressive `apcn` (~64 µs) — the field
  deinterleave (writing rows 0,2,4,… then 1,3,5,…) adds a constant-
  factor row-stride traversal per output plane but doesn't change
  the IDCT or entropy-decode cost.

## Reproducing

```bash
# 1. Build the profile driver in release with debug info.
cargo build --release --example profile_prores \
    -p oxideav-prores

# 2. Run the modes — or `all` for the full sweep.
./target/release/examples/profile_prores all

# Per-mode subsets are useful for sampler runs (samply / perf):
./target/release/examples/profile_prores encode    20
./target/release/examples/profile_prores decode  1000
./target/release/examples/profile_prores interlaced 30
```

### Capturing flamegraphs (samply, no root on macOS)

`samply` is the recommended sampler on macOS — it uses
`task_for_pid` after self-signing, no DTrace / `perf` /
elevated privileges. On Linux substitute `perf record` (root or
`perf_event_paranoid <= 1`) or `samply record` directly.

```bash
cargo install samply
cargo install inferno

# Sample. --unstable-presymbolicate writes a sidecar syms file so
# the JSON profile resolves to source symbols even after the
# binary's debug-info is gone.
samply record --unstable-presymbolicate --save-only \
    -o encode.json.gz \
    -r 1997 \
    -- target/release/examples/profile_prores encode 20

# Convert samply's processed-profile JSON to Brendan-Gregg folded
# stacks, then SVG. (The folded-stacks format is the stable
# interchange artefact — drop the JSON afterwards.)
samply export --output encode.folded encode.json.gz
inferno-flamegraph \
    --title "oxideav-prores encode (round 161)" \
    --subtitle "samply 1997Hz, 20 iters x 5 scenarios" \
    < encode.folded > encode.svg

# Repeat for decode / roundtrip / interlaced.
```

The intermediate `*.json.gz` files are NOT committed — they're a
samply implementation detail. The folded-stack files (`*.folded`)
and SVGs (`*.svg`) are the stable interchange format; future
rounds that capture profiles should commit those alongside this
README baseline.

## Wall

Captured without consulting any external library source or network
queries. `samply` is a sampling profiler that only observes the
OxideAV binary at runtime; the captured stacks reference only the
project's own modules + stdlib + macOS runtime (`libsystem_*`,
`dyld`). No third-party ProRes implementation participated in this
baseline.
