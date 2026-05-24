//! Integration tests against the docs/video/prores/ fixture corpus.
//!
//! Each fixture under `../../docs/video/prores/fixtures/<name>/` ships an
//! `input.mov` (or `input.mxf`) carrying one or more ProRes elementary
//! frames, plus an `expected.yuv.sha256` (and sometimes the actual
//! `expected.yuv` bytes) that pin the FFmpeg/prores_ks reference
//! reconstruction. A `notes.md` describes the bitstream feature focus and
//! a `trace.txt` captures PRORES_TRACE events from an instrumented
//! ffmpeg decode pass.
//!
//! This driver:
//! 1. Reads `input.mov` (or `input.mxf`) bytes via `std::fs::read`.
//! 2. Locates every ProRes elementary frame in the container payload by
//!    searching for the `icpf` magic and consuming the preceding
//!    big-endian `frame_size` field (RDD 36 §5.1). This is sufficient
//!    for both QuickTime/MOV `mdat` atoms and MXF KAG-aligned
//!    GenericContainer essence — neither container interleaves data
//!    inside a ProRes frame, so a linear scan finds them all.
//! 3. Decodes each frame through [`oxideav_prores::decoder::decode_packet_with_depth`],
//!    requesting the bit-depth + chroma-format combination the fixture's
//!    notes pin (yuv422p10le for the 422 profiles; yuv444p12le for 4444
//!    / 4444 XQ; alpha lands as a 4th plane).
//! 4. When the fixture ships `expected.yuv`, compares the decoded YUV
//!    against the reference plane-by-plane (for as many frames as the
//!    reference covers — the corpus may ship a reference for the prefix
//!    only to keep the on-disk size budget). Otherwise (the larger
//!    1080p/720p fixtures keep only the SHA-256 to stay under the corpus
//!    size budget), the test still drives the decoder and reports
//!    plane sizes / errors but skips the pixel comparison.
//!
//! Acceptance:
//! * `Tier::BitExact` — must round-trip exactly. Failure = CI red.
//! * `Tier::ReportOnly` — divergence is logged but the test does NOT
//!   fail. The in-tree decoder uses an Annex A-compliant float IDCT,
//!   which is permitted by RDD 36 §7.4 but cannot guarantee bit-exact
//!   output against ffmpeg's fixed-point IDCT — every fixture therefore
//!   starts as ReportOnly. As individual fixtures are confirmed
//!   bit-exact by the maintainer they may be promoted.
//! * Out-of-scope formats — fixtures the in-tree codec explicitly
//!   excludes (e.g. ProRes RAW) — are covered by a live negative test
//!   that asserts the decoder *refuses* them with a clear `Unsupported`
//!   error rather than mis-decoding (`corpus_proresraw_not_supported`).
//!
//! All fixtures start as ReportOnly per the workspace policy in
//! `feedback_no_external_libs.md`: NO external decoder source (ffmpeg,
//! prores_decoder.c) was consulted while writing this driver — the
//! fixtures are data, the SMPTE RDD 36 PDF is the authority.

use std::fs;
use std::path::PathBuf;

use oxideav_prores::decoder::{decode_packet_with_depth, BitDepth};
use oxideav_prores::frame::ChromaFormat;

/// Locate `docs/video/prores/fixtures/<name>/`. Tests run with CWD set
/// to the crate root; we walk two levels up to reach the workspace
/// root and then into `docs/`.
fn fixture_dir(name: &str) -> PathBuf {
    PathBuf::from("../../docs/video/prores/fixtures").join(name)
}

/// Linear scan over a container payload (MOV mdat or MXF essence
/// element) emitting one elementary ProRes frame per `icpf` magic
/// found. The frame payload is the slice
/// `[size_offset .. size_offset + frame_size]`, where `frame_size` is
/// the big-endian u32 that precedes the magic per RDD 36 §5.1.
fn extract_prores_frames(container: &[u8]) -> Vec<Vec<u8>> {
    let needle = b"icpf";
    let mut out = Vec::new();
    if container.len() < 4 {
        return out;
    }
    let mut i = 4usize;
    while i + 4 <= container.len() {
        if &container[i..i + 4] == needle {
            let size_off = i - 4;
            let frame_size =
                u32::from_be_bytes(container[size_off..size_off + 4].try_into().unwrap()) as usize;
            let end = size_off + frame_size;
            if end <= container.len() && frame_size >= 8 {
                out.push(container[size_off..end].to_vec());
                i = end;
                continue;
            }
        }
        i += 1;
    }
    out
}

/// Per-frame, per-plane comparison summary. All counters are in
/// SAMPLES (not bytes) so 8-bit and HBD outputs share the same
/// reporting axis.
#[derive(Default)]
struct FrameDiff {
    y_total: usize,
    y_exact: usize,
    y_max: i32,
    /// Sum of squared differences across luma — used for PSNR.
    y_sse: f64,
    uv_total: usize,
    uv_exact: usize,
    uv_max: i32,
    uv_sse: f64,
    a_total: usize,
    a_exact: usize,
    a_max: i32,
}

impl FrameDiff {
    fn pct(&self) -> f64 {
        let exact = self.y_exact + self.uv_exact + self.a_exact;
        let total = self.y_total + self.uv_total + self.a_total;
        if total == 0 {
            0.0
        } else {
            exact as f64 / total as f64 * 100.0
        }
    }

    fn merge(&mut self, other: &FrameDiff) {
        self.y_total += other.y_total;
        self.y_exact += other.y_exact;
        self.y_max = self.y_max.max(other.y_max);
        self.y_sse += other.y_sse;
        self.uv_total += other.uv_total;
        self.uv_exact += other.uv_exact;
        self.uv_max = self.uv_max.max(other.uv_max);
        self.uv_sse += other.uv_sse;
        self.a_total += other.a_total;
        self.a_exact += other.a_exact;
        self.a_max = self.a_max.max(other.a_max);
    }
}

/// Compare a decoded plane (in our own decoder's emit format — u8 byte
/// stream that holds either single-byte 8-bit samples or LE-packed
/// 10/12-bit samples) against the same-format reference plane. Returns
/// (n_samples, n_exact, max_abs_diff, sum_sq_diff).
fn diff_plane(our: &[u8], refp: &[u8], bit_depth: BitDepth) -> (usize, usize, i32, f64) {
    let mut ex = 0usize;
    let mut max = 0i32;
    let mut sse = 0.0f64;
    match bit_depth {
        BitDepth::Eight => {
            let n = our.len().min(refp.len());
            for i in 0..n {
                let d = (our[i] as i32 - refp[i] as i32).abs();
                if d == 0 {
                    ex += 1;
                }
                if d > max {
                    max = d;
                }
                sse += (d as f64) * (d as f64);
            }
            (n, ex, max, sse)
        }
        BitDepth::Ten | BitDepth::Twelve => {
            let n_samples = (refp.len() / 2).min(our.len() / 2);
            for i in 0..n_samples {
                let lo_o = our[i * 2];
                let hi_o = our[i * 2 + 1];
                let v_o = u16::from_le_bytes([lo_o, hi_o]) as i32;
                let lo_r = refp[i * 2];
                let hi_r = refp[i * 2 + 1];
                let v_r = u16::from_le_bytes([lo_r, hi_r]) as i32;
                let d = (v_o - v_r).abs();
                if d == 0 {
                    ex += 1;
                }
                if d > max {
                    max = d;
                }
                sse += (d as f64) * (d as f64);
            }
            (n_samples, ex, max, sse)
        }
    }
}

#[derive(Clone, Copy, Debug)]
#[allow(dead_code)] // BitExact reserved for promotion as fixtures are confirmed
enum Tier {
    /// Must decode bit-exactly. Test fails on any divergence.
    BitExact,
    /// Every frame must decode without error AND, when expected.yuv
    /// is shipped, every plane must achieve at least `min_y_psnr_db` /
    /// `min_uv_psnr_db`. Sample-pixel match counts are still logged.
    /// Use this to lock in a permissive quality floor while leaving
    /// room for the float IDCT vs ffmpeg's fixed-point IDCT to diverge
    /// by ~1 LSB.
    MinPsnr {
        min_y_psnr_db: f64,
        min_uv_psnr_db: f64,
    },
    /// Every frame must decode without error. Use this for fixtures
    /// where there's no expected.yuv to compare against — promotes
    /// "decoder errored out" from a silent log line into a CI-red
    /// failure.
    DecodesCleanly,
    /// Decode is permitted to diverge from the reference; per-fixture
    /// stats are logged but the test does not fail.
    ReportOnly,
}

#[derive(Clone, Copy, Debug)]
enum Container {
    /// QuickTime / MOV. Container is searched linearly for `icpf`.
    Mov,
    /// MXF OP1a. Same linear `icpf` scan applies — the inner ProRes
    /// elementary stream is byte-identical to the MOV variant.
    Mxf,
}

impl Container {
    fn input_filename(self) -> &'static str {
        match self {
            Container::Mov => "input.mov",
            Container::Mxf => "input.mxf",
        }
    }
}

struct CorpusCase {
    name: &'static str,
    container: Container,
    width: usize,
    height: usize,
    bit_depth: BitDepth,
    chroma: ChromaFormat,
    /// Number of expected output planes (3 for plain YUV, 4 when alpha
    /// is present per RDD 36 §5.3.3).
    planes: usize,
    tier: Tier,
}

struct DecodeReport {
    /// One entry per input frame: Ok(per-frame diff) when the fixture
    /// shipped expected.yuv AND the decoder produced a frame, or
    /// Err(message) when either step failed.
    per_frame: Vec<Result<FrameDiff, String>>,
    /// Whether expected.yuv was actually present (drives whether we
    /// scored against it or only counted decoder errors).
    had_reference: bool,
    /// First fatal error (if any) — recorded for the report banner.
    fatal: Option<String>,
}

fn decode_fixture(case: &CorpusCase) -> Option<DecodeReport> {
    let dir = fixture_dir(case.name);
    let in_path = dir.join(case.container.input_filename());
    let yuv_path = dir.join("expected.yuv");
    let trace_path = dir.join("trace.txt");
    let container_bytes = match fs::read(&in_path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!(
                "skip {}: missing {} ({e}). docs/ corpus is in the workspace \
                 umbrella repo; the standalone crate checkout has no fixtures.",
                case.name,
                in_path.display()
            );
            return None;
        }
    };
    let yuv_ref = fs::read(&yuv_path).ok();
    eprintln!(
        "fixture {}: {}={} bytes, expected.yuv={}, trace={}",
        case.name,
        case.container.input_filename(),
        container_bytes.len(),
        match &yuv_ref {
            Some(y) => format!("{} bytes", y.len()),
            None => "<sha-only — pixel comparison skipped>".to_string(),
        },
        trace_path.display()
    );

    let frames = extract_prores_frames(&container_bytes);
    if frames.is_empty() {
        return Some(DecodeReport {
            per_frame: vec![Err(format!(
                "{}: no `icpf` frames found in container ({} bytes)",
                case.name,
                container_bytes.len()
            ))],
            had_reference: yuv_ref.is_some(),
            fatal: Some("no icpf frames in container".to_string()),
        });
    }
    eprintln!(
        "  extracted {} elementary frames from container",
        frames.len()
    );

    // Compute per-frame size in BYTES of the reference (post-cropping
    // to the visible picture dims, since ffmpeg's `-f rawvideo` emits
    // the cropped output).
    let bps = match case.bit_depth {
        BitDepth::Eight => 1,
        BitDepth::Ten | BitDepth::Twelve => 2,
    };
    let cw = match case.chroma {
        ChromaFormat::Y422 => case.width.div_ceil(2),
        ChromaFormat::Y444 => case.width,
    };
    let frame_bytes = (case.width * case.height + 2 * cw * case.height) * bps
        + if case.planes == 4 {
            case.width * case.height * bps
        } else {
            0
        };

    // The reference's frame count is whatever fits cleanly into its
    // byte length. The corpus may ship a reference covering fewer
    // frames than the container holds (size budget) — we score the
    // intersection: every container frame for which there's a
    // matching block of expected.yuv pixels.
    let ref_frames = match &yuv_ref {
        Some(yuv) => {
            assert!(
                yuv.len() % frame_bytes == 0,
                "fixture {} expected.yuv size {} is not a whole multiple of frame_bytes {} \
                 [{}x{} {:?}, {}-bit, {} planes]",
                case.name,
                yuv.len(),
                frame_bytes,
                case.width,
                case.height,
                case.chroma,
                case.bit_depth.bits(),
                case.planes,
            );
            yuv.len() / frame_bytes
        }
        None => 0,
    };

    let mut per_frame: Vec<Result<FrameDiff, String>> = Vec::with_capacity(frames.len());
    let mut fatal: Option<String> = None;

    // Score every container frame; pixel-compare against expected.yuv
    // only for the prefix the reference actually covers.
    for (i, frame_bytes_pkt) in frames.iter().enumerate() {
        let requested = Some((case.bit_depth, case.chroma));
        let decoded = match decode_packet_with_depth(frame_bytes_pkt, Some(i as i64), requested) {
            Ok(vf) => vf,
            Err(e) => {
                let msg = format!("frame {i}: decode_packet_with_depth: {e:?}");
                if fatal.is_none() {
                    fatal = Some(msg.clone());
                }
                per_frame.push(Err(msg));
                continue;
            }
        };

        // If reference YUV is unavailable OR doesn't cover this frame
        // index (the corpus may ship a reference for the prefix only),
        // record success and move on.
        let Some(ref yuv) = yuv_ref else {
            per_frame.push(Ok(FrameDiff::default()));
            continue;
        };
        if i >= ref_frames {
            per_frame.push(Ok(FrameDiff::default()));
            continue;
        }

        let ref_off = i * frame_bytes;
        let ref_slice = &yuv[ref_off..ref_off + frame_bytes];
        // Slice the reference into planes in Y/Cb/Cr/[A] order.
        let y_ref_bytes = case.width * case.height * bps;
        let c_ref_bytes = cw * case.height * bps;
        let mut off = 0usize;
        let ref_y = &ref_slice[off..off + y_ref_bytes];
        off += y_ref_bytes;
        let ref_u = &ref_slice[off..off + c_ref_bytes];
        off += c_ref_bytes;
        let ref_v = &ref_slice[off..off + c_ref_bytes];
        off += c_ref_bytes;
        let ref_a = if case.planes == 4 {
            let a_bytes = case.width * case.height * bps;
            Some(&ref_slice[off..off + a_bytes])
        } else {
            None
        };

        // Compare planes one-by-one. The decoder's emit format is
        // already in the same byte layout as the reference for the
        // 8-/10-/12-bit cases (LE-packed for HBD), so the comparison
        // is straight `diff_plane`.
        if decoded.planes.len() < case.planes {
            per_frame.push(Err(format!(
                "frame {i}: decoder produced {} planes, expected {}",
                decoded.planes.len(),
                case.planes
            )));
            continue;
        }
        let mut diff = FrameDiff::default();
        let our_y = decoded.planes[0].data.as_slice();
        let our_u = decoded.planes[1].data.as_slice();
        let our_v = decoded.planes[2].data.as_slice();
        let (yt, ye, ym, yse) = diff_plane(our_y, ref_y, case.bit_depth);
        diff.y_total += yt;
        diff.y_exact += ye;
        diff.y_max = diff.y_max.max(ym);
        diff.y_sse += yse;
        let (ut, ue, um, use_) = diff_plane(our_u, ref_u, case.bit_depth);
        diff.uv_total += ut;
        diff.uv_exact += ue;
        diff.uv_max = diff.uv_max.max(um);
        diff.uv_sse += use_;
        let (vt, ve, vm, vse) = diff_plane(our_v, ref_v, case.bit_depth);
        diff.uv_total += vt;
        diff.uv_exact += ve;
        diff.uv_max = diff.uv_max.max(vm);
        diff.uv_sse += vse;
        if let Some(ra) = ref_a {
            if decoded.planes.len() >= 4 {
                let our_a = decoded.planes[3].data.as_slice();
                let (at, ae, am, _) = diff_plane(our_a, ra, case.bit_depth);
                diff.a_total += at;
                diff.a_exact += ae;
                diff.a_max = diff.a_max.max(am);
            } else {
                per_frame.push(Err(format!(
                    "frame {i}: alpha plane expected but decoder produced only {} planes",
                    decoded.planes.len()
                )));
                continue;
            }
        }
        per_frame.push(Ok(diff));
    }

    Some(DecodeReport {
        per_frame,
        had_reference: yuv_ref.is_some(),
        fatal,
    })
}

/// PSNR in dB given a sum-of-squared-errors and a sample count, scaled
/// to the bit-depth's max value.
fn psnr(sse: f64, n: usize, bit_depth: BitDepth) -> f64 {
    if n == 0 || sse == 0.0 {
        return 120.0;
    }
    let max = bit_depth.max_value() as f64;
    let mse = sse / n as f64;
    10.0 * (max * max / mse).log10()
}

fn evaluate(case: &CorpusCase) {
    let report = match decode_fixture(case) {
        Some(r) => r,
        None => return, // missing fixture — already logged
    };

    let mut agg = FrameDiff::default();
    let mut errors: Vec<String> = Vec::new();
    for (i, r) in report.per_frame.iter().enumerate() {
        match r {
            Ok(d) => {
                if report.had_reference {
                    eprintln!(
                        "  frame {i}: Y {}/{} exact (max diff {}, PSNR {:.2} dB), \
                         UV {}/{} exact (max diff {}, PSNR {:.2} dB), \
                         A {}/{} exact (max diff {}), pct={:.2}%",
                        d.y_exact,
                        d.y_total,
                        d.y_max,
                        psnr(d.y_sse, d.y_total, case.bit_depth),
                        d.uv_exact,
                        d.uv_total,
                        d.uv_max,
                        psnr(d.uv_sse, d.uv_total, case.bit_depth),
                        d.a_exact,
                        d.a_total,
                        d.a_max,
                        d.pct(),
                    );
                } else {
                    eprintln!("  frame {i}: decoded OK (no expected.yuv to compare against)");
                }
                agg.merge(d);
            }
            Err(e) => {
                eprintln!("  frame {i}: ERROR {e}");
                errors.push(format!("frame {i}: {e}"));
            }
        }
    }

    let pct = agg.pct();
    eprintln!(
        "[{:?}] {}: aggregate {}/{} exact ({pct:.2}%), \
         Y max diff {} (PSNR {:.2} dB), \
         UV max diff {} (PSNR {:.2} dB), \
         A max diff {}{}",
        case.tier,
        case.name,
        agg.y_exact + agg.uv_exact + agg.a_exact,
        agg.y_total + agg.uv_total + agg.a_total,
        agg.y_max,
        psnr(agg.y_sse, agg.y_total, case.bit_depth),
        agg.uv_max,
        psnr(agg.uv_sse, agg.uv_total, case.bit_depth),
        agg.a_max,
        match &report.fatal {
            Some(f) => format!(", first_fatal=\"{f}\""),
            None => String::new(),
        }
    );

    match case.tier {
        Tier::BitExact => {
            assert!(
                errors.is_empty(),
                "{}: {} frame errors prevented bit-exact comparison: {:?}",
                case.name,
                errors.len(),
                errors
            );
            assert_eq!(
                agg.y_exact + agg.uv_exact + agg.a_exact,
                agg.y_total + agg.uv_total + agg.a_total,
                "{}: not bit-exact (Y max diff {}, UV max diff {}, A max diff {}; {:.4}% match)",
                case.name,
                agg.y_max,
                agg.uv_max,
                agg.a_max,
                pct
            );
        }
        Tier::MinPsnr {
            min_y_psnr_db,
            min_uv_psnr_db,
        } => {
            assert!(
                errors.is_empty(),
                "{}: {} frame decode errors: {:?}",
                case.name,
                errors.len(),
                errors
            );
            if report.had_reference {
                let y_psnr = psnr(agg.y_sse, agg.y_total, case.bit_depth);
                let uv_psnr = psnr(agg.uv_sse, agg.uv_total, case.bit_depth);
                assert!(
                    y_psnr >= min_y_psnr_db,
                    "{}: luma PSNR {:.2} dB below floor {:.2} dB",
                    case.name,
                    y_psnr,
                    min_y_psnr_db
                );
                assert!(
                    uv_psnr >= min_uv_psnr_db,
                    "{}: chroma PSNR {:.2} dB below floor {:.2} dB",
                    case.name,
                    uv_psnr,
                    min_uv_psnr_db
                );
            }
        }
        Tier::DecodesCleanly => {
            assert!(
                errors.is_empty(),
                "{}: {} frame decode errors: {:?}",
                case.name,
                errors.len(),
                errors
            );
        }
        Tier::ReportOnly => {
            // Don't fail. The eprintln output above is the report.
            // TODO(prores-corpus): tighten to BitExact once the
            // underlying decoder gap for this fixture is closed.
            let _ = pct;
        }
    }
}

// ---------------------------------------------------------------------------
// Per-fixture tests
// ---------------------------------------------------------------------------
//
// All fixtures start as ReportOnly. The in-tree decoder uses the
// Annex-A-compliant float IDCT, which is permitted by RDD 36 §7.4 but
// not bit-exact against ffmpeg's fixed-point IDCT — promotion to
// BitExact requires a pass with the integer-IDCT path enabled.
//
// Trace files (referenced via the eprintln! header in `decode_fixture`)
// live alongside each fixture and capture the PRORES_TRACE event
// sequence (FRAME_CONTAINER / FRAME_HEADER / QUANT_TABLE_LUMA /
// QUANT_TABLE_CHROMA / PICTURE_HEADER / SLICE) emitted by an
// instrumented ffmpeg pass — useful for diffing against our own
// trace output if/when divergence localisation is needed.

/// Smallest practical SQ fixture: 320x240 apcn at yuv422p10le.
/// expected.yuv is kept (~300 KB / frame) so a byte-level comparison
/// runs cheaply.
#[test]
fn corpus_tiny_320x240_sq() {
    evaluate(&CorpusCase {
        name: "tiny-320x240-sq",
        container: Container::Mov,
        width: 320,
        height: 240,
        bit_depth: BitDepth::Ten,
        chroma: ChromaFormat::Y422,
        planes: 3,
        // Measured 82.65 / 81.62 dB; floor at 60 dB leaves headroom
        // for any future float→fixed-point IDCT switch but locks in
        // the current near-bit-exact decode.
        tier: Tier::MinPsnr {
            min_y_psnr_db: 60.0,
            min_uv_psnr_db: 60.0,
        },
    });
}

/// Same SQ bitstream as tiny-320x240-sq but wrapped in MXF OP1a.
/// Verifies the demuxer (here: linear `icpf` scan) is the only moving
/// piece — the ProRes payload should decode to byte-identical output.
#[test]
fn corpus_mxf_container() {
    evaluate(&CorpusCase {
        name: "mxf-container",
        container: Container::Mxf,
        width: 320,
        height: 240,
        bit_depth: BitDepth::Ten,
        chroma: ChromaFormat::Y422,
        planes: 3,
        // Same payload as tiny-320x240-sq — decode quality must match.
        tier: Tier::MinPsnr {
            min_y_psnr_db: 60.0,
            min_uv_psnr_db: 60.0,
        },
    });
}

/// 422 Proxy (apco): lowest data-rate ProRes profile at 1280x720 10-bit.
/// expected.yuv is sha-only (would be ~5.3 MB); we drive the decoder
/// and report errors / plane sizes only.
#[test]
fn corpus_proxy_1280x720() {
    evaluate(&CorpusCase {
        name: "proxy-1280x720",
        container: Container::Mov,
        width: 1280,
        height: 720,
        bit_depth: BitDepth::Ten,
        chroma: ChromaFormat::Y422,
        planes: 3,
        tier: Tier::DecodesCleanly,
    });
}

/// 422 LT (apcs): standard 'light' tier at 1280x720 10-bit.
#[test]
fn corpus_lt_1280x720() {
    evaluate(&CorpusCase {
        name: "lt-1280x720",
        container: Container::Mov,
        width: 1280,
        height: 720,
        bit_depth: BitDepth::Ten,
        chroma: ChromaFormat::Y422,
        planes: 3,
        tier: Tier::DecodesCleanly,
    });
}

/// 422 SQ Standard (apcn) 1920x1080 10-bit — the canonical broadcast
/// progressive layout (1020 slices per frame at slice_mb_width=8).
#[test]
fn corpus_sq_1920x1080() {
    evaluate(&CorpusCase {
        name: "sq-1920x1080",
        container: Container::Mov,
        width: 1920,
        height: 1080,
        bit_depth: BitDepth::Ten,
        chroma: ChromaFormat::Y422,
        planes: 3,
        tier: Tier::DecodesCleanly,
    });
}

/// 422 HQ (apch) 1920x1080 10-bit. Same layout as SQ but higher
/// bitrate budget — exercises entropy decoding throughput.
#[test]
fn corpus_hq_1920x1080() {
    evaluate(&CorpusCase {
        name: "hq-1920x1080",
        container: Container::Mov,
        width: 1920,
        height: 1080,
        bit_depth: BitDepth::Ten,
        chroma: ChromaFormat::Y422,
        planes: 3,
        tier: Tier::DecodesCleanly,
    });
}

/// 4444 (ap4h) 1920x1080 12-bit, no alpha. Confirms the 4:4:4
/// chroma-block-per-MB doubling (log2_chroma_blocks_per_mb=2,
/// mb_x_shift=5).
#[test]
fn corpus_4444_1920x1080() {
    evaluate(&CorpusCase {
        name: "4444-1920x1080",
        container: Container::Mov,
        width: 1920,
        height: 1080,
        bit_depth: BitDepth::Twelve,
        chroma: ChromaFormat::Y444,
        planes: 3,
        // 4:4:4 12-bit decode path (chroma blocks doubled — RDD 36
        // §7.4.2 macroblock layout); guard against silent regressions.
        tier: Tier::DecodesCleanly,
    });
}

/// 4444 (ap4h) 1920x1080 12-bit WITH active alpha plane
/// (alpha_info=2, 16-bit packed alpha). Trips the per-slice
/// `a_data_size` + `unpack_alpha` decode path (RDD 36 §7.1.2 +
/// §7.5.2). Reference output is `yuva444p12le` → 4 planes.
#[test]
fn corpus_4444_with_alpha() {
    evaluate(&CorpusCase {
        name: "4444-with-alpha",
        container: Container::Mov,
        width: 1920,
        height: 1080,
        bit_depth: BitDepth::Twelve,
        chroma: ChromaFormat::Y444,
        planes: 4,
        // Alpha plane is lossless per RDD 36 §7.1.2; the path just
        // needs to decode without a "run overruns" error against
        // ffmpeg-encoded streams whose picture height is not a
        // multiple of MB_SIDE_PX. BitExact on Y/Cb/Cr is gated on
        // the IDCT switch.
        tier: Tier::DecodesCleanly,
    });
}

/// 4444 XQ (ap4x) 1920x1080 12-bit. Highest-bitrate cinema tier — same
/// 4444 geometry but with a permissive quant range so qscale_idx tends
/// to bottom out at 1.
#[test]
fn corpus_4444xq_1920x1080() {
    evaluate(&CorpusCase {
        name: "4444xq-1920x1080",
        container: Container::Mov,
        width: 1920,
        height: 1080,
        bit_depth: BitDepth::Twelve,
        chroma: ChromaFormat::Y444,
        planes: 3,
        tier: Tier::DecodesCleanly,
    });
}

/// Interlaced TFF apcn 1920x1080 10-bit. interlace_mode=1 — each coded
/// frame contains TWO picture headers (first_field then second_field).
#[test]
fn corpus_interlaced_tff() {
    evaluate(&CorpusCase {
        name: "interlaced-tff",
        container: Container::Mov,
        width: 1920,
        height: 1080,
        bit_depth: BitDepth::Ten,
        chroma: ChromaFormat::Y422,
        planes: 3,
        // Two field pictures per coded frame (RDD 36 §5.1, §7.5.3) —
        // both must decode without error.
        tier: Tier::DecodesCleanly,
    });
}

/// PAL-style 1080i50 broadcast — same bitstream layout as
/// interlaced-tff but kept as a separate fixture for exact
/// broadcast-workflow byte sizes.
#[test]
fn corpus_pal_1080i50() {
    evaluate(&CorpusCase {
        name: "pal-1080i50",
        container: Container::Mov,
        width: 1920,
        height: 1080,
        bit_depth: BitDepth::Ten,
        chroma: ChromaFormat::Y422,
        planes: 3,
        tier: Tier::DecodesCleanly,
    });
}

// --- Out-of-scope: explicit negative documentation ---

/// ProRes RAW is a separate Apple format (Bayer sensor data wrapped in
/// the ProRes wavelet/entropy front-end) that the in-tree decoder does
/// NOT cover — see `notes.md` in the fixture dir for the full
/// rationale (different sample structure, separate Apple white paper,
/// not SMPTE-registered, no fully-working open encoder). The fixture
/// dir intentionally ships only `notes.md`.
///
/// `notes.md` § "What a downstream decoder should do" pins the
/// contract: *detect ProRes RAW by MOV codec_tag `aprn` / `aprh` (and
/// the in-stream marker `aprh`) and surface a clear `Unsupported`
/// error rather than attempting to dispatch to the standard ProRes
/// decoder, whose bitstream layout is incompatible.* This test asserts
/// every clause of that contract directly, so it is a live test rather
/// than an `#[ignore]` stub.
#[test]
fn corpus_proresraw_not_supported() {
    use oxideav_prores::{
        codec_id_for_fourcc, is_prores_raw_fourcc, profile_for_fourcc, PRORES_RAW_FOURCCS,
    };

    // The notes.md rationale ships in the fixture dir; confirm it is
    // present when the corpus is available. The docs/ corpus lives in
    // the workspace umbrella repo — the standalone crate checkout (and
    // standalone CI) has no fixtures, so a missing file is a skip, not
    // a failure (mirrors the other corpus_* tests). The contract
    // assertions below need no fixture and always run.
    let dir = fixture_dir("proresraw-not-supported");
    match fs::read(dir.join("notes.md")) {
        Ok(notes) => assert!(!notes.is_empty(), "proresraw notes.md should not be empty"),
        Err(e) => eprintln!(
            "skip notes.md presence check: missing {} ({e}). docs/ corpus is in \
             the workspace umbrella repo; the standalone crate checkout has no fixtures.",
            dir.join("notes.md").display()
        ),
    }

    // Clause 1: detect ProRes RAW by codec_tag aprn / aprh.
    for fc in PRORES_RAW_FOURCCS {
        assert!(is_prores_raw_fourcc(fc), "{fc:?} should be ProRes RAW");
        // Clause 1b: must NOT dispatch to the standard ProRes decoder —
        // these FourCCs resolve to neither a CodecId nor a Profile.
        assert_eq!(codec_id_for_fourcc(fc), None, "{fc:?} must not resolve");
        assert_eq!(profile_for_fourcc(fc), None, "{fc:?} must not map");
    }

    // Clause 2: an in-stream sample carrying the `aprh` marker (at the
    // `icpf` offset) yields a clear Unsupported error naming ProRes RAW.
    let mut sample = Vec::new();
    sample.extend_from_slice(&16u32.to_be_bytes()); // frame_size
    sample.extend_from_slice(b"aprh"); // ProRes RAW in-stream marker
    sample.extend_from_slice(&[0u8; 8]); // padding to frame_size
    let err = decode_packet_with_depth(&sample, None, Some((BitDepth::Eight, ChromaFormat::Y422)))
        .expect_err("ProRes RAW sample must be refused, not decoded");
    assert!(
        err.to_string().contains("ProRes RAW"),
        "error must name ProRes RAW, got: {err}"
    );
}
