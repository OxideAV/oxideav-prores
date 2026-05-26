// Parallel-array index loops are idiomatic in codec / DCT code; skip
// the lint (mirrors the crate root's `#![allow(clippy::needless_range_loop)]`).
#![allow(clippy::needless_range_loop)]

//! Standalone profiling driver for the ProRes encoder and decoder.
//!
//! Round 161 (depth-mode profiling): the two Criterion harnesses
//! (`benches/{decode,encode}.rs`) measure steady-state throughput in
//! Criterion's sampling framework, but they're a poor target for
//! `samply` / `perf record` / `cargo flamegraph` because Criterion's
//! warm-up + sampling layers + estimator math show up in the profile
//! and bury the real codec hot paths. This example is a flat
//! measure-this-thing driver: it builds a deterministic synthesised
//! input once, then runs a fixed iteration count of whichever path
//! was requested with a single `Instant::now()` / `elapsed()` pair
//! around the whole loop. Throughput is printed at the end so it
//! doubles as a quick A/B harness for the inner tweak-remeasure loop
//! when Criterion's per-run overhead is too coarse.
//!
//! Usage:
//!
//!     cargo run --example profile_prores --release -- <mode> [<iters>]
//!
//! Modes:
//!
//!     decode      — encode each scenario once outside the loop, decode
//!                   N times against the cached bytes
//!     encode      — synth pixels, encode N times (decoder cost excluded)
//!     roundtrip   — synth pixels, encode + decode every iteration
//!     interlaced  — 8-bit 4:2:2 Standard, top-field-first split per
//!                   §7.5.3 (two pictures sharing one frame_header)
//!     all         — run every mode (default)
//!
//! With `samply`:
//!
//!     samply record -- ./target/release/examples/profile_prores encode 20
//!     samply record -- ./target/release/examples/profile_prores decode 200
//!
//! With `cargo flamegraph` (needs `cargo install flamegraph`):
//!
//!     cargo flamegraph --example profile_prores -- encode 20
//!
//! No external files are read — every input is synthesised in-driver
//! from a deterministic gradient pattern matching the Criterion bench
//! harnesses so profile output and bench numbers correspond. Inputs
//! cover the 4:2:2 8/10-bit and 4:4:4 8-bit cost-axis scenarios the
//! workspace README rows track.

use std::env;
use std::io::Write;
use std::time::Instant;

use oxideav_core::frame::VideoPlane;
use oxideav_core::VideoFrame;

use oxideav_prores::decoder::{decode_packet, decode_packet_with_depth, BitDepth};
use oxideav_prores::encoder::{encode_frame_interlaced, encode_frame_with_depth};
use oxideav_prores::frame::{ChromaFormat, Profile};

/// 8-bit 4:2:2 smooth gradient (luma full-width, chroma half-width).
/// Mirrors `benches/{decode,encode}.rs::source_422_8bit` byte for byte.
fn source_422_8bit(width: u32, height: u32) -> VideoFrame {
    let w = width as usize;
    let h = height as usize;
    let cw = w / 2;
    let mut y = vec![0u8; w * h];
    let mut cb = vec![0u8; cw * h];
    let mut cr = vec![0u8; cw * h];
    for j in 0..h {
        for i in 0..w {
            y[j * w + i] = ((i + j) * 255 / (w + h)).min(255) as u8;
        }
        for i in 0..cw {
            cb[j * cw + i] = (128 + ((i as i32 - cw as i32 / 2) * 2).clamp(-64, 64)) as u8;
            cr[j * cw + i] = (128 + ((j as i32 - h as i32 / 2) * 2).clamp(-64, 64)) as u8;
        }
    }
    VideoFrame {
        pts: Some(0),
        planes: vec![
            VideoPlane { stride: w, data: y },
            VideoPlane {
                stride: cw,
                data: cb,
            },
            VideoPlane {
                stride: cw,
                data: cr,
            },
        ],
    }
}

/// 8-bit 4:4:4 smooth gradient (all three planes full-width).
/// Mirrors `benches/{decode,encode}.rs::source_444_8bit`.
fn source_444_8bit(width: u32, height: u32) -> VideoFrame {
    let w = width as usize;
    let h = height as usize;
    let mut y = vec![0u8; w * h];
    let mut cb = vec![0u8; w * h];
    let mut cr = vec![0u8; w * h];
    for j in 0..h {
        for i in 0..w {
            y[j * w + i] = ((i + j) * 255 / (w + h)).min(255) as u8;
            cb[j * w + i] = (128 + ((i as i32 - w as i32 / 2) * 2).clamp(-64, 64)) as u8;
            cr[j * w + i] = (128 + ((j as i32 - h as i32 / 2) * 2).clamp(-64, 64)) as u8;
        }
    }
    VideoFrame {
        pts: Some(0),
        planes: vec![
            VideoPlane { stride: w, data: y },
            VideoPlane {
                stride: w,
                data: cb,
            },
            VideoPlane {
                stride: w,
                data: cr,
            },
        ],
    }
}

/// 10-bit 4:2:2 gradient: each sample is packed little-endian into two
/// bytes, valid range `0..=1023`. Mirrors `benches/*::source_422_10bit`.
fn source_422_10bit(width: u32, height: u32) -> VideoFrame {
    let w = width as usize;
    let h = height as usize;
    let cw = w / 2;
    let mut y = vec![0u8; w * h * 2];
    let mut cb = vec![0u8; cw * h * 2];
    let mut cr = vec![0u8; cw * h * 2];
    for j in 0..h {
        for i in 0..w {
            let v: u16 = ((i * 7 + j * 5) as u16 % 1024).min(1023);
            let off = (j * w + i) * 2;
            y[off] = (v & 0xFF) as u8;
            y[off + 1] = (v >> 8) as u8;
        }
        for i in 0..cw {
            let cb_v: u16 = (512 + ((i as i32 - cw as i32 / 2) * 8).clamp(-256, 256)) as u16;
            let cr_v: u16 = (512 + ((j as i32 - h as i32 / 2) * 8).clamp(-256, 256)) as u16;
            let off = (j * cw + i) * 2;
            cb[off] = (cb_v & 0xFF) as u8;
            cb[off + 1] = (cb_v >> 8) as u8;
            cr[off] = (cr_v & 0xFF) as u8;
            cr[off + 1] = (cr_v >> 8) as u8;
        }
    }
    VideoFrame {
        pts: Some(0),
        planes: vec![
            VideoPlane {
                stride: w * 2,
                data: y,
            },
            VideoPlane {
                stride: cw * 2,
                data: cb,
            },
            VideoPlane {
                stride: cw * 2,
                data: cr,
            },
        ],
    }
}

/// Per-scenario knobs. The picker matches the bench matrix so the
/// profile and Criterion numbers reference identical inputs.
struct Scenario {
    name: &'static str,
    width: u32,
    height: u32,
    chroma: ChromaFormat,
    bit_depth: BitDepth,
    profile: Profile,
    /// Bytes per luma sample (1 for 8-bit, 2 for 10/12-bit). Drives the
    /// throughput print.
    bytes_per_luma_sample: usize,
    /// Default iteration count for the encode path. Decode is much
    /// cheaper so the driver scales this up for decode-only runs.
    encode_iters_default: u32,
}

fn scenarios() -> &'static [Scenario] {
    &[
        Scenario {
            name: "apco_422_8bit_128x96",
            width: 128,
            height: 96,
            chroma: ChromaFormat::Y422,
            bit_depth: BitDepth::Eight,
            profile: Profile::Proxy,
            bytes_per_luma_sample: 1,
            encode_iters_default: 30,
        },
        Scenario {
            name: "apcn_422_8bit_128x96",
            width: 128,
            height: 96,
            chroma: ChromaFormat::Y422,
            bit_depth: BitDepth::Eight,
            profile: Profile::Standard,
            bytes_per_luma_sample: 1,
            encode_iters_default: 30,
        },
        Scenario {
            name: "apch_422_8bit_128x96",
            width: 128,
            height: 96,
            chroma: ChromaFormat::Y422,
            bit_depth: BitDepth::Eight,
            profile: Profile::Hq,
            bytes_per_luma_sample: 1,
            encode_iters_default: 30,
        },
        Scenario {
            name: "ap4h_444_8bit_128x96",
            width: 128,
            height: 96,
            chroma: ChromaFormat::Y444,
            bit_depth: BitDepth::Eight,
            profile: Profile::Prores4444,
            bytes_per_luma_sample: 1,
            encode_iters_default: 20,
        },
        Scenario {
            name: "apcn_422_10bit_128x96",
            width: 128,
            height: 96,
            chroma: ChromaFormat::Y422,
            bit_depth: BitDepth::Ten,
            profile: Profile::Standard,
            bytes_per_luma_sample: 2,
            encode_iters_default: 20,
        },
    ]
}

/// Build the synthesised input for a scenario. The pattern matches the
/// matching `source_*` helper in the Criterion benches so profile and
/// bench inputs are byte-identical.
fn build_input(scen: &Scenario) -> VideoFrame {
    match (scen.chroma, scen.bit_depth) {
        (ChromaFormat::Y422, BitDepth::Eight) => source_422_8bit(scen.width, scen.height),
        (ChromaFormat::Y444, BitDepth::Eight) => source_444_8bit(scen.width, scen.height),
        (ChromaFormat::Y422, BitDepth::Ten) => source_422_10bit(scen.width, scen.height),
        // The remaining (chroma, depth) pairs aren't in the profiling
        // scenario set — they'd require their own deterministic gradient
        // helper before they could be added.
        other => panic!("unsupported scenario chroma/depth combo: {other:?}"),
    }
}

/// One encode pass — returns the encoded byte vector.
fn encode_once(scen: &Scenario, src: &VideoFrame) -> Vec<u8> {
    encode_frame_with_depth(
        src,
        scen.width,
        scen.height,
        scen.chroma,
        scen.bit_depth,
        scen.profile,
        scen.profile.default_quant_index(),
    )
    .expect("encode_frame_with_depth")
}

/// One decode pass. The 8-bit path uses [`decode_packet`]; the 10-bit
/// path uses [`decode_packet_with_depth`] with the matching
/// `(BitDepth, ChromaFormat)` request — same dispatch the bench harness
/// uses, so profile + bench cost line up sample-for-sample.
fn decode_once(scen: &Scenario, bytes: &[u8]) -> usize {
    let frame = if matches!(scen.bit_depth, BitDepth::Eight) {
        decode_packet(bytes, None).expect("decode_packet")
    } else {
        decode_packet_with_depth(bytes, None, Some((scen.bit_depth, scen.chroma)))
            .expect("decode_packet_with_depth")
    };
    // Sum a plane length to keep the optimiser from dropping the call.
    frame.planes.iter().map(|p| p.data.len()).sum()
}

fn print_throughput_line(label: &str, scen: &Scenario, iters: u32, elapsed_secs: f64, extra: &str) {
    // Per-iter raw bytes: luma + chroma. For 4:2:2 that's 2 * w * h * bps;
    // for 4:4:4 that's 3 * w * h * bps.
    let chroma_planes_factor = match scen.chroma {
        ChromaFormat::Y422 => 2usize, // luma + 2 half-width = 2x luma area
        ChromaFormat::Y444 => 3usize,
    };
    let raw_bytes_per_iter = scen.width as usize
        * scen.height as usize
        * scen.bytes_per_luma_sample
        * chroma_planes_factor;
    let total_bytes = raw_bytes_per_iter * iters as usize;
    let per_iter_ms = elapsed_secs * 1000.0 / iters as f64;
    let mib_per_s = (total_bytes as f64) / elapsed_secs / (1024.0 * 1024.0);
    println!(
        "  {label:9} {name:30} iters={iters:>4} {per_iter_ms:8.3} ms/iter  {mib_per_s:8.2} MiB/s (raw){extra}",
        name = scen.name,
    );
}

fn profile_encode(iters_override: Option<u32>) {
    println!("== encode ==");
    for scen in scenarios() {
        let iters = iters_override.unwrap_or(scen.encode_iters_default);
        let src = build_input(scen);

        // One warm-up so any first-call lazy init isn't charged to
        // iteration #1's bucket.
        let _ = encode_once(scen, &src);

        let t = Instant::now();
        let mut total_out_bytes = 0u64;
        for _ in 0..iters {
            let out = std::hint::black_box(encode_once(
                std::hint::black_box(scen),
                std::hint::black_box(&src),
            ));
            total_out_bytes += out.len() as u64;
            std::hint::black_box(out);
        }
        let elapsed = t.elapsed().as_secs_f64();
        let compressed_bytes_per_iter = total_out_bytes / iters.max(1) as u64;
        let chroma_factor = match scen.chroma {
            ChromaFormat::Y422 => 2usize,
            ChromaFormat::Y444 => 3usize,
        };
        let raw_bytes_per_iter =
            scen.width as usize * scen.height as usize * scen.bytes_per_luma_sample * chroma_factor;
        let ratio = compressed_bytes_per_iter as f64 / raw_bytes_per_iter as f64;
        let extra = format!("  out={compressed_bytes_per_iter}B/iter ({ratio:.3} of input)");
        print_throughput_line("encode", scen, iters, elapsed, &extra);
        std::io::stdout().flush().ok();
    }
}

fn profile_decode(iters_override: Option<u32>) {
    println!("== decode ==");
    for scen in scenarios() {
        // Decode is the cheaper side — scale the iteration count up
        // when the caller didn't pick one explicitly.
        let iters = iters_override.unwrap_or(scen.encode_iters_default * 50);
        let src = build_input(scen);

        // Encode once OUTSIDE the timed region.
        let bytes = encode_once(scen, &src);

        // Warm up: one decode pass.
        let _ = decode_once(scen, &bytes);

        let t = Instant::now();
        let mut sink = 0usize;
        for _ in 0..iters {
            sink ^= decode_once(std::hint::black_box(scen), std::hint::black_box(&bytes));
        }
        std::hint::black_box(sink);
        let elapsed = t.elapsed().as_secs_f64();
        print_throughput_line("decode", scen, iters, elapsed, "");
        std::io::stdout().flush().ok();
    }
}

fn profile_roundtrip(iters_override: Option<u32>) {
    println!("== roundtrip ==");
    for scen in scenarios() {
        let iters = iters_override.unwrap_or(scen.encode_iters_default);
        let src = build_input(scen);

        // Warm-up.
        {
            let bytes = encode_once(scen, &src);
            let _ = decode_once(scen, &bytes);
        }

        let t = Instant::now();
        for _ in 0..iters {
            let bytes = std::hint::black_box(encode_once(
                std::hint::black_box(scen),
                std::hint::black_box(&src),
            ));
            let n = decode_once(scen, &bytes);
            std::hint::black_box(n);
        }
        let elapsed = t.elapsed().as_secs_f64();
        print_throughput_line("roundtrip", scen, iters, elapsed, "");
        std::io::stdout().flush().ok();
    }
}

/// Interlaced 8-bit 4:2:2 Standard, top-field-first split per §7.5.3.
/// Two pictures sharing one frame_header() — exercises the alternate
/// block scan (Figure 5) on the encode side and the field-deinterleave
/// path on the decode side.
fn profile_interlaced(iters_override: Option<u32>) {
    println!("== interlaced (apcn 422 8-bit TFF 128x96) ==");
    let iters = iters_override.unwrap_or(20);
    let width: u32 = 128;
    let height: u32 = 96;
    let chroma = ChromaFormat::Y422;
    let depth = BitDepth::Eight;
    let profile = Profile::Standard;
    let qi = profile.default_quant_index();
    let src = source_422_8bit(width, height);

    // Warm-up — one full encode + decode pair.
    {
        let bytes =
            encode_frame_interlaced(&src, width, height, chroma, depth, profile, qi, None, 1)
                .expect("warmup interlaced encode");
        let _ = decode_packet(&bytes, None).expect("warmup interlaced decode");
    }

    let t_enc = Instant::now();
    let mut total_bytes = 0u64;
    for _ in 0..iters {
        let bytes = std::hint::black_box(
            encode_frame_interlaced(
                std::hint::black_box(&src),
                width,
                height,
                chroma,
                depth,
                profile,
                qi,
                None,
                1, // top-field-first
            )
            .expect("interlaced encode"),
        );
        total_bytes += bytes.len() as u64;
        std::hint::black_box(bytes);
    }
    let enc_elapsed = t_enc.elapsed().as_secs_f64();

    // Encode once outside the decode-timed region.
    let bytes = encode_frame_interlaced(&src, width, height, chroma, depth, profile, qi, None, 1)
        .expect("decode setup interlaced encode");
    let dec_iters = iters * 50;
    let t_dec = Instant::now();
    let mut sink = 0usize;
    for _ in 0..dec_iters {
        let f = decode_packet(std::hint::black_box(&bytes), None).expect("interlaced decode");
        sink ^= f.planes.iter().map(|p| p.data.len()).sum::<usize>();
    }
    std::hint::black_box(sink);
    let dec_elapsed = t_dec.elapsed().as_secs_f64();

    let enc_per_iter_ms = enc_elapsed * 1000.0 / iters as f64;
    let dec_per_iter_ms = dec_elapsed * 1000.0 / dec_iters as f64;
    let avg_bytes = total_bytes / iters.max(1) as u64;
    let raw_bytes_per_iter = width as usize * height as usize * 2; // 4:2:2 8-bit
    let enc_mib_per_s =
        (raw_bytes_per_iter * iters as usize) as f64 / enc_elapsed / (1024.0 * 1024.0);
    let dec_mib_per_s =
        (raw_bytes_per_iter * dec_iters as usize) as f64 / dec_elapsed / (1024.0 * 1024.0);
    println!(
        "  encode    interlaced-tff                 iters={iters:>4} {enc_per_iter_ms:8.3} ms/iter  {enc_mib_per_s:8.2} MiB/s (raw)  out={avg_bytes}B/iter"
    );
    println!(
        "  decode    interlaced-tff                 iters={dec_iters:>4} {dec_per_iter_ms:8.3} ms/iter  {dec_mib_per_s:8.2} MiB/s (raw)"
    );
    std::io::stdout().flush().ok();
}

fn main() {
    let mut args = env::args().skip(1);
    let mode = args.next().unwrap_or_else(|| "all".to_string());
    let iters_override: Option<u32> = args.next().and_then(|s| s.parse().ok());

    println!(
        "=== oxideav-prores profile (mode={mode}, iters={}) ===",
        iters_override
            .map(|n| n.to_string())
            .unwrap_or_else(|| "default".to_string()),
    );
    println!();

    match mode.as_str() {
        "encode" => profile_encode(iters_override),
        "decode" => profile_decode(iters_override),
        "roundtrip" => profile_roundtrip(iters_override),
        "interlaced" => profile_interlaced(iters_override),
        "all" => {
            profile_encode(iters_override);
            println!();
            profile_decode(iters_override);
            println!();
            profile_roundtrip(iters_override);
            println!();
            profile_interlaced(iters_override);
        }
        other => {
            eprintln!("unknown mode: {other:?}");
            eprintln!("usage: profile_prores [encode|decode|roundtrip|interlaced|all] [<iters>]");
            std::process::exit(2);
        }
    }
}
