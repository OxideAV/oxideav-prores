//! Live black-box interoperability matrix.
//!
//! Two sweeps, both generated at run time in a temporary directory and
//! both skipped when the `ffmpeg` executable is absent:
//!
//! * **foreign → ours** — every ProRes encoder the local `ffmpeg`
//!   exposes writes streams across its bitstream-changing options
//!   (profiles, alpha depth, field order, quantiser / bit-rate extremes,
//!   slice sizes, quantisation matrices, picture sizes from 2×2 to 8K,
//!   colour metadata, picture content). Each coded frame is decoded by
//!   this crate and by the black-box decoders (the software one, and the
//!   platform one where present) and the samples are compared.
//! * **ours → foreign** — this crate's encoder sweeps its own options;
//!   the black-box decoders must accept every stream without a
//!   diagnostic and reproduce this crate's own reconstruction.
//!
//! # Tolerance
//!
//! RDD 36 does not fix the IDCT: §7.4 only requires that an
//! implementation pass the Annex A accuracy qualification, whose
//! criterion (1) bounds the peak error of a reconstructed value at 0.15
//! (in the 9-bit signed reconstruction range) against the
//! reference-precision transform. Two conforming decoders can therefore
//! differ by up to 0.30 before the §7.5.1 conversion
//! `s = round(2^b * (v + 256) / 512)`, i.e. by `0.30 * 2^(b-9)` sample
//! codes plus one for the two independent roundings:
//! ±1 code at 10 bits and ±3 codes at 12 bits. Alpha is lossless
//! (§7.5.2) and is compared exactly at the coded depth.
//!
//! Annex A step 3 qualifies an IDCT only for coefficients inside the
//! 12-bit range −2048..+2047.75. Streams whose dequantised coefficients
//! leave that range (coarse quantiser on saturated content) are outside
//! the qualified domain; those cases are reported separately and held
//! to the looser bound the two black-box decoders show between
//! themselves.
//!
//! Set `PRORES_MATRIX_REPORT=<path>` to write one line per case.
//! Set `PRORES_MATRIX_LARGE=1` to add the 4K / 8K rows (slow in debug
//! builds).

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::Command;

use oxideav_core::frame::VideoPlane;
use oxideav_core::{CodecId, CodecParameters, Frame, MediaType, PixelFormat, Rational, VideoFrame};
use oxideav_prores::alpha::AlphaChannelType;
use oxideav_prores::decoder::{
    OutputRange, decode_packet_with_format, dequantized_coefficient_range,
};
use oxideav_prores::encoder::{EncoderConfig, make_encoder_with_config};
use oxideav_prores::frame::{ChromaFormat, FrameHeader, FrameMeta, Profile, parse_frame};
use oxideav_prores::quant::QuantMatrices;

// ---------------------------------------------------------------------
// Tool discovery
// ---------------------------------------------------------------------

struct Tools {
    encoders: Vec<String>,
    platform_decoder: bool,
}

fn run(cmd: &mut Command) -> Option<(bool, String)> {
    let out = cmd.output().ok()?;
    Some((
        out.status.success(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    ))
}

fn tools() -> Option<Tools> {
    let out = Command::new("ffmpeg")
        .args(["-hide_banner", "-encoders"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let encoders = text
        .lines()
        .filter_map(|l| l.split_whitespace().nth(1))
        .filter(|n| n.starts_with("prores"))
        .map(str::to_owned)
        .collect();
    let hw = Command::new("ffmpeg")
        .args(["-hide_banner", "-hwaccels"])
        .output()
        .ok()?;
    let platform_decoder = String::from_utf8_lossy(&hw.stdout)
        .lines()
        .any(|l| l.trim() == "videotoolbox");
    Some(Tools {
        encoders,
        platform_decoder,
    })
}

struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> Self {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let p = std::env::temp_dir().join(format!(
            "oxideav-prores-matrix-{tag}-{}-{nanos}",
            std::process::id()
        ));
        std::fs::create_dir_all(&p).expect("temp dir");
        Self(p)
    }
    fn path(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        if !keep_files() {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
}

/// `PRORES_MATRIX_KEEP=1` leaves every generated file in place.
fn keep_files() -> bool {
    std::env::var_os("PRORES_MATRIX_KEEP").is_some()
}

fn discard(p: impl AsRef<Path>) {
    if !keep_files() {
        let _ = std::fs::remove_file(p);
    }
}

/// `PRORES_MATRIX_ONLY=<text>` restricts both sweeps to the cases whose
/// label contains the text.
fn selected(label: &str) -> bool {
    std::env::var("PRORES_MATRIX_ONLY").map_or(true, |f| label.contains(&f))
}

fn large_rows() -> bool {
    std::env::var_os("PRORES_MATRIX_LARGE").is_some()
}

fn write_report(name: &str, body: &str) {
    if let Some(p) = std::env::var_os("PRORES_MATRIX_REPORT") {
        let mut path = PathBuf::from(p);
        let stem = path
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        path.set_file_name(format!("{stem}.{name}"));
        let _ = std::fs::write(path, body);
    }
}

// ---------------------------------------------------------------------
// Sample plumbing
// ---------------------------------------------------------------------

/// Native black-box output surface of a stream: 4:2:2 decodes to 10-bit
/// words, 4:4:4 to 12-bit words, plus an alpha plane when one is coded.
#[derive(Clone, Copy, Debug)]
struct Surface {
    chroma444: bool,
    alpha: bool,
    width: usize,
    height: usize,
}

impl Surface {
    fn of(h: &FrameHeader) -> Self {
        Self {
            chroma444: h.chroma_format == ChromaFormat::Y444,
            alpha: h.alpha_channel_type != 0,
            width: h.width as usize,
            height: h.height as usize,
        }
    }
    fn bits(self) -> u32 {
        if self.chroma444 { 12 } else { 10 }
    }
    fn raw_name(self) -> &'static str {
        match (self.chroma444, self.alpha) {
            (false, false) => "yuv422p10le",
            (false, true) => "yuva422p10le",
            (true, false) => "yuv444p12le",
            (true, true) => "yuva444p12le",
        }
    }
    fn pixel_format(self) -> PixelFormat {
        match (self.chroma444, self.alpha) {
            (false, false) => PixelFormat::Yuv422P10Le,
            (false, true) => PixelFormat::Yuva422P10Le,
            (true, false) => PixelFormat::Yuv444P12Le,
            (true, true) => PixelFormat::Yuva444P12Le,
        }
    }
    /// `(width, height)` of every plane, in raw-video order.
    fn plane_dims(self) -> Vec<(usize, usize)> {
        let cw = if self.chroma444 {
            self.width
        } else {
            self.width.div_ceil(2)
        };
        let mut v = vec![
            (self.width, self.height),
            (cw, self.height),
            (cw, self.height),
        ];
        if self.alpha {
            v.push((self.width, self.height));
        }
        v
    }
    fn frame_bytes(self) -> usize {
        self.plane_dims().iter().map(|(w, h)| w * h * 2).sum()
    }
    fn pixel_format16(self) -> PixelFormat {
        if self.chroma444 {
            PixelFormat::Yuva444P16Le
        } else {
            PixelFormat::Yuva422P16Le
        }
    }
    /// Whether the platform decoder's sample in column `x` of colour
    /// plane `plane` is usable as a reference. Three classes are not,
    /// all observed with the software decoder and this crate agreeing
    /// with each other while luma and alpha still agree everywhere:
    ///
    /// * chroma of alpha-bearing 4:4:4 pictures above roughly 40 000
    ///   pixels (agreeing at 300x130 and 512x64, visibly different from
    ///   254x254 up);
    /// * chroma of alpha-bearing 4:2:2 pictures, which that decoder
    ///   returns on a 4:4:4 surface;
    /// * the last luma and chroma column of an odd-width 4:2:2 picture.
    fn platform_sample_usable(self, plane: usize, x: usize) -> bool {
        if self.alpha && plane != 0 {
            self.chroma444 && self.width * self.height <= 40_000
        } else if !self.chroma444 && self.width % 2 == 1 {
            x + 1
                != if plane == 0 {
                    self.width
                } else {
                    self.width.div_ceil(2)
                }
        } else {
            true
        }
    }
    /// Colour-sample tolerance in codes (see the module docs).
    fn tolerance(self) -> i32 {
        if self.chroma444 { 3 } else { 1 }
    }
}

fn frame_planes(vf: &VideoFrame, s: Surface) -> Vec<Vec<u16>> {
    let dims = s.plane_dims();
    assert!(
        vf.planes.len() >= dims.len(),
        "decoder returned {} planes, surface needs {}",
        vf.planes.len(),
        dims.len()
    );
    dims.iter()
        .zip(&vf.planes)
        .map(|(&(w, h), p)| {
            let mut out = Vec::with_capacity(w * h);
            for row in p.data.chunks(p.stride).take(h) {
                out.extend(
                    row[..w * 2]
                        .as_chunks::<2>()
                        .0
                        .iter()
                        .map(|c| u16::from_le_bytes(*c)),
                );
            }
            assert_eq!(out.len(), w * h, "short plane");
            out
        })
        .collect()
}

fn raw_planes(raw: &[u8], s: Surface) -> Vec<Vec<u16>> {
    let mut off = 0;
    s.plane_dims()
        .iter()
        .map(|&(w, h)| {
            let n = w * h * 2;
            let v = raw[off..off + n]
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| u16::from_le_bytes(*c))
                .collect();
            off += n;
            v
        })
        .collect()
}

/// Largest absolute difference per plane and the number of differing
/// samples over all planes.
#[derive(Clone, Copy, Debug, Default)]
struct Diff {
    colour_max: i32,
    alpha_max: i32,
    differing: usize,
}

impl Diff {
    fn merge(&mut self, o: Diff) {
        self.colour_max = self.colour_max.max(o.colour_max);
        self.alpha_max = self.alpha_max.max(o.alpha_max);
        self.differing += o.differing;
    }
}

/// The comparisons of one frame.
#[derive(Clone, Copy, Debug, Default)]
struct FrameDiff {
    /// ours vs the software black-box decoder.
    software: Diff,
    /// ours vs the platform black-box decoder.
    platform: Diff,
    /// software vs platform: how far the two validators are apart.
    validators: Diff,
    /// Per sample, the distance to the *nearer* validator (colour only).
    nearest: i32,
}

fn note(d: &mut Diff, alpha: bool, e: i32) {
    if e != 0 {
        d.differing += 1;
        if alpha {
            d.alpha_max = d.alpha_max.max(e);
        } else {
            d.colour_max = d.colour_max.max(e);
        }
    }
}

/// Compare one frame of this crate (`ours`, native depth, full range;
/// `alpha16` its coded alpha at 16 bits) with the software decoder's
/// planes `sw` and, when present, the platform decoder's 16-bit planes
/// `hw`.
///
/// Three input/output facts about the validators shape the comparison:
///
/// * the software decoder never emits a colour code below 4 or above
///   `2^b - 5`, so this crate's full-range samples are clamped likewise
///   before being compared with it;
/// * the software decoder delivers 16-bit coded alpha as its top `b`
///   bits rather than through the §7.5.2 rounding, so alpha is compared
///   in that form (every alpha bit it exposes is still checked exactly),
///   and against the platform decoder at the full 16 bits;
/// * the platform decoder's samples sit in the top bits of 16-bit words,
///   and some of them are not usable as a reference
///   ([`Surface::platform_sample_usable`]).
fn compare_frame(
    ours: &[Vec<u16>],
    alpha16: Option<&[u16]>,
    sw: &[Vec<u16>],
    hw: Option<&[Vec<u16>]>,
    s: Surface,
) -> FrameDiff {
    let mut d = FrameDiff::default();
    let shift = 16 - s.bits();
    let hi = (1u16 << s.bits()) - 5;
    for (p, &(pw, _)) in s.plane_dims().iter().enumerate().take(3) {
        for (i, (&o, &r)) in ours[p].iter().zip(&sw[p]).enumerate() {
            let e_sw = (i32::from(o.clamp(4, hi)) - i32::from(r)).abs();
            note(&mut d.software, false, e_sw);
            let mut nearest = e_sw;
            if let Some(hw) = hw
                && s.platform_sample_usable(p, i % pw)
            {
                let h = hw[p][i] >> shift;
                let e_hw = (i32::from(o) - i32::from(h)).abs();
                note(&mut d.platform, false, e_hw);
                note(
                    &mut d.validators,
                    false,
                    (i32::from(r) - i32::from(h.clamp(4, hi))).abs(),
                );
                nearest = nearest.min(e_hw);
            }
            d.nearest = d.nearest.max(nearest);
        }
    }
    if let Some(a16) = alpha16 {
        for (i, (&a, &r)) in a16.iter().zip(&sw[3]).enumerate() {
            note(
                &mut d.software,
                true,
                (i32::from(a >> shift) - i32::from(r)).abs(),
            );
            if let Some(hw) = hw {
                note(
                    &mut d.platform,
                    true,
                    (i32::from(a) - i32::from(hw[3][i])).abs(),
                );
                note(
                    &mut d.validators,
                    true,
                    (i32::from(hw[3][i] >> shift) - i32::from(r)).abs(),
                );
            }
        }
    }
    d
}

/// Split the concatenated packets `-f data` writes: every ProRes frame
/// begins with its own big-endian `frame_size` (RDD 36 §5.1).
fn split_frames(data: &[u8]) -> Vec<&[u8]> {
    let mut out = Vec::new();
    let mut off = 0;
    while off + 8 <= data.len() {
        let n = u32::from_be_bytes(data[off..off + 4].try_into().unwrap()) as usize;
        if n < 8 || off + n > data.len() {
            break;
        }
        out.push(&data[off..off + n]);
        off += n;
    }
    out
}

fn ffmpeg() -> Command {
    let mut c = Command::new("ffmpeg");
    c.args(["-hide_banner", "-loglevel", "warning", "-y", "-nostdin"]);
    c
}

/// Software black-box decode of `mov` to the native raw surface.
fn decode_software(mov: &Path, raw: &Path, s: Surface) -> Option<(Vec<u8>, String)> {
    let (ok, err) = run(ffmpeg()
        .arg("-i")
        .arg(mov)
        .args(["-f", "rawvideo", "-pix_fmt", s.raw_name()])
        .arg(raw))?;
    if !ok {
        return Some((Vec::new(), err));
    }
    Some((std::fs::read(raw).ok()?, err))
}

/// Platform black-box decode. Its surfaces carry the samples in the top
/// bits of 16-bit words, so the planar 16-bit raw form is requested and
/// the caller shifts down to the native depth.
fn decode_platform(mov: &Path, raw: &Path, s: Surface) -> Option<Vec<u8>> {
    let name = match (s.chroma444, s.alpha) {
        (false, false) => "yuv422p16le",
        (false, true) => "yuva422p16le",
        (true, false) => "yuv444p16le",
        (true, true) => "yuva444p16le",
    };
    let (ok, _) = run(ffmpeg()
        .args(["-hwaccel", "videotoolbox"])
        .arg("-i")
        .arg(mov)
        .args(["-f", "rawvideo", "-pix_fmt", name])
        .arg(raw))?;
    if !ok {
        return None;
    }
    std::fs::read(raw).ok()
}

// ---------------------------------------------------------------------
// foreign -> ours
// ---------------------------------------------------------------------

#[derive(Clone, Debug)]
struct Foreign {
    label: String,
    encoder: &'static str,
    /// lavfi source with `WxH` substituted for `{S}`.
    source: &'static str,
    /// Filter chain placed before the final `format=`.
    filters: String,
    pix: &'static str,
    width: u32,
    height: u32,
    frames: u32,
    /// Encoder / muxer options.
    opts: Vec<String>,
    /// Expected `interlace_mode`, when the case asks for one.
    interlace: Option<u8>,
    /// Expected colour triple (primaries, transfer, matrix).
    colour: Option<(u8, u8, u8)>,
}

const SRC_PATTERN: &str = "testsrc2=size={S}:rate=25";
const SRC_FRACTAL: &str = "mandelbrot=size={S}:rate=25";

impl Foreign {
    fn new(encoder: &'static str, pix: &'static str, w: u32, h: u32) -> Self {
        Self {
            label: String::new(),
            encoder,
            source: SRC_PATTERN,
            filters: String::new(),
            pix,
            width: w,
            height: h,
            frames: 2,
            opts: Vec::new(),
            interlace: None,
            colour: None,
        }
    }
    fn opt(mut self, k: &str, v: impl ToString) -> Self {
        self.opts.push(k.to_owned());
        self.opts.push(v.to_string());
        self
    }
    fn filter(mut self, f: &str) -> Self {
        if !self.filters.is_empty() {
            self.filters.push(',');
        }
        self.filters.push_str(f);
        self
    }
    fn noise(self) -> Self {
        self.filter("noise=alls=60:allf=t")
    }
    fn fields(mut self, top_first: bool) -> Self {
        self.interlace = Some(if top_first { 1 } else { 2 });
        let f = if top_first {
            "setfield=tff"
        } else {
            "setfield=bff"
        };
        self.filter(f).opt("-flags", "+ildct")
    }
    fn label(&self) -> String {
        format!(
            "{} {} {}x{} n{} [{}] {} {}",
            self.encoder,
            self.pix,
            self.width,
            self.height,
            self.frames,
            self.filters,
            self.opts.join(" "),
            self.label
        )
    }
}

fn foreign_cases(t: &Tools) -> Vec<Foreign> {
    let has = |n: &str| t.encoders.iter().any(|e| e == n);
    let mut v = Vec::new();

    // --- the two software encoder families -------------------------
    for enc in ["prores_ks", "prores_aw", "prores"] {
        if !has(enc) {
            continue;
        }
        let ks = enc == "prores_ks";
        // Profile x pixel format x content.
        let mut combos: Vec<(u32, &'static str)> = (0..4).map(|p| (p, "yuv422p10le")).collect();
        combos.push((4, "yuv444p10le"));
        combos.push((4, "yuva444p10le"));
        combos.push((5, "yuv444p10le"));
        combos.push((5, "yuva444p10le"));
        for &(profile, pix) in &combos {
            for noisy in [false, true] {
                let mut c = Foreign::new(enc, pix, 130, 74).opt("-profile:v", profile);
                if noisy {
                    c = c.noise();
                }
                v.push(c.clone());
                // Field order, on an even and an odd picture height (the
                // two fields then differ in height, §6.2).
                for top in [true, false] {
                    v.push(c.clone().fields(top));
                    let mut odd = c.clone().fields(top);
                    odd.height = 75;
                    v.push(odd);
                }
            }
        }
        // Alpha depth and alpha content.
        if ks {
            for bits in [0, 8, 16] {
                for alpha in [
                    "geq=lum='lum(X,Y)':cb='cb(X,Y)':cr='cr(X,Y)':a='mod(X*37+Y*11,1024)'",
                    "geq=lum='lum(X,Y)':cb='cb(X,Y)':cr='cr(X,Y)':a='1023*gt(X,Y)'",
                    "geq=lum='lum(X,Y)':cb='cb(X,Y)':cr='cr(X,Y)':a='random(1)*1023'",
                    "geq=lum='lum(X,Y)':cb='cb(X,Y)':cr='cr(X,Y)':a='X*4+Y'",
                    "geq=lum='lum(X,Y)':cb='cb(X,Y)':cr='cr(X,Y)':a=0",
                ] {
                    for (w, h) in [(130, 74), (16, 16), (300, 40)] {
                        let mut c = Foreign::new(enc, "yuva444p10le", w, h)
                            .opt("-profile:v", 4)
                            .opt("-alpha_bits", bits);
                        c.filters = format!("format=yuva444p10le,{alpha}");
                        v.push(c.clone());
                        v.push(c.fields(true));
                    }
                }
            }
        } else {
            for alpha in [
                "geq=lum='lum(X,Y)':cb='cb(X,Y)':cr='cr(X,Y)':a='mod(X*37+Y*11,1024)'",
                "geq=lum='lum(X,Y)':cb='cb(X,Y)':cr='cr(X,Y)':a='random(1)*1023'",
                "geq=lum='lum(X,Y)':cb='cb(X,Y)':cr='cr(X,Y)':a=0",
            ] {
                let mut c = Foreign::new(enc, "yuva444p10le", 130, 74).opt("-profile:v", 4);
                c.filters = format!("format=yuva444p10le,{alpha}");
                v.push(c);
            }
        }
        // Quantiser extremes.
        for q in [0, 1, 2, 3, 5, 9, 16, 31, 64, 128, 224, 255] {
            for (profile, pix) in [(3, "yuv422p10le"), (4, "yuva444p10le")] {
                v.push(
                    Foreign::new(enc, pix, 130, 74)
                        .opt("-profile:v", profile)
                        .opt("-qscale:v", q)
                        .noise(),
                );
            }
        }
        // Bit-rate extremes.
        for rate in ["20k", "400k", "5M", "80M", "2000M"] {
            for (profile, pix) in [(0, "yuv422p10le"), (3, "yuv422p10le"), (5, "yuv444p10le")] {
                v.push(
                    Foreign::new(enc, pix, 130, 74)
                        .opt("-profile:v", profile)
                        .opt("-b:v", rate)
                        .noise(),
                );
            }
        }
        if ks {
            for bits in [1, 64, 300, 2000, 8192] {
                v.push(
                    Foreign::new(enc, "yuv422p10le", 130, 74)
                        .opt("-profile:v", 2)
                        .opt("-bits_per_mb", bits)
                        .noise(),
                );
            }
            // Slice size x picture width (the width decides the slice
            // size ladder of §6.2).
            for mbs in 1..=8 {
                for w in [16, 48, 130, 258, 722] {
                    v.push(
                        Foreign::new(enc, "yuv422p10le", w, 34)
                            .opt("-profile:v", 2)
                            .opt("-mbs_per_slice", mbs),
                    );
                }
                v.push(
                    Foreign::new(enc, "yuva444p10le", 130, 34)
                        .opt("-profile:v", 4)
                        .opt("-mbs_per_slice", mbs)
                        .fields(false),
                );
            }
            // Quantisation matrix selection.
            for m in ["auto", "proxy", "lt", "standard", "hq", "default"] {
                for profile in [0, 2, 4] {
                    let pix = if profile == 4 {
                        "yuv444p10le"
                    } else {
                        "yuv422p10le"
                    };
                    v.push(
                        Foreign::new(enc, pix, 130, 74)
                            .opt("-profile:v", profile)
                            .opt("-quant_mat", m)
                            .noise(),
                    );
                }
            }
        }
        // Picture sizes: tiny, non-macroblock-multiple, odd, broadcast.
        let mut sizes = vec![
            (2, 2),
            (4, 2),
            (2, 16),
            (16, 2),
            (16, 16),
            (18, 18),
            (30, 30),
            (32, 16),
            (34, 50),
            (126, 2),
            (2, 126),
            (131, 75),
            (255, 255),
            (720, 486),
            (720, 576),
            (1280, 720),
            (1920, 1080),
        ];
        if large_rows() {
            sizes.extend([(1998, 1080), (2048, 1556), (3840, 2160), (4096, 2160)]);
            if ks {
                sizes.extend([(7680, 4320), (8192, 4320)]);
            }
        }
        for &(w, h) in &sizes {
            for (profile, pix) in [(2, "yuv422p10le"), (4, "yuva444p10le")] {
                let mut c = Foreign::new(enc, pix, w, h).opt("-profile:v", profile);
                c.frames = 1;
                v.push(c.clone());
                if h >= 4 && w * h <= 1920 * 1080 {
                    v.push(c.fields(true));
                }
            }
        }
        // Picture content: saturated extremes, flat fields, fine detail.
        for content in [
            "geq=lum='1023*mod(X+Y,2)':cb='1023*mod(X,2)':cr='1023*mod(Y,2)'",
            "geq=lum='1023*gt(mod(X,16),7)':cb='1023*gt(mod(Y,16),7)':cr=512",
            "geq=lum=0:cb=0:cr=0",
            "geq=lum=1023:cb=1023:cr=1023",
            "geq=lum=64:cb=512:cr=512",
            "geq=lum='random(1)*1023':cb='random(2)*1023':cr='random(3)*1023'",
            "geq=lum='X*8':cb='Y*13':cr='1023-X*8'",
        ] {
            for (profile, pix) in [(0, "yuv422p10le"), (3, "yuv422p10le"), (5, "yuv444p10le")] {
                let mut c = Foreign::new(enc, pix, 130, 74).opt("-profile:v", profile);
                c.filters = format!("format={pix},{content}");
                v.push(c);
            }
        }
        let mut fractal = Foreign::new(enc, "yuv422p10le", 320, 180).opt("-profile:v", 3);
        fractal.source = SRC_FRACTAL;
        v.push(fractal);
        // A longer stream.
        let mut long = Foreign::new(enc, "yuv422p10le", 96, 64)
            .opt("-profile:v", 1)
            .noise();
        long.frames = 40;
        v.push(long);
        // Colour metadata.
        for (prim, trc, mat, codes) in [
            ("bt709", "bt709", "bt709", (1, 1, 1)),
            ("bt470bg", "bt709", "smpte170m", (5, 1, 6)),
            ("smpte170m", "bt709", "smpte170m", (6, 1, 6)),
            ("bt2020", "smpte2084", "bt2020nc", (9, 16, 9)),
            ("bt2020", "arib-std-b67", "bt2020nc", (9, 18, 9)),
            ("smpte431", "bt709", "bt709", (11, 1, 1)),
            ("smpte432", "bt709", "bt709", (12, 1, 1)),
        ] {
            for (profile, pix) in [(3, "yuv422p10le"), (4, "yuv444p10le")] {
                let mut c = Foreign::new(enc, pix, 130, 74)
                    .opt("-profile:v", profile)
                    .filter(&format!(
                        "setparams=color_primaries={prim}:color_trc={trc}:colorspace={mat}"
                    ));
                c.colour = Some(codes);
                c.frames = 1;
                v.push(c);
            }
        }
    }

    // --- the platform encoder --------------------------------------
    if has("prores_videotoolbox") {
        let enc = "prores_videotoolbox";
        let formats_422 = ["p210le", "p216le", "uyvy422", "nv16"];
        let formats_444 = ["p410le", "p416le", "ayuv64le", "bgra", "nv24"];
        let mut sizes = vec![
            (16, 16),
            (18, 18),
            (130, 74),
            (131, 75),
            (720, 486),
            (1920, 1080),
        ];
        if large_rows() {
            sizes.extend([(3840, 2160), (4096, 2160), (7680, 4320), (8192, 4320)]);
        }
        for profile in 0..6 {
            let formats: &[&'static str] = if profile < 4 {
                &formats_422
            } else {
                &formats_444
            };
            for &pix in formats {
                for noisy in [false, true] {
                    let mut c = Foreign::new(enc, pix, 130, 74).opt("-profile:v", profile);
                    if noisy {
                        c = c.noise();
                    }
                    v.push(c.clone());
                    v.push(c.clone().fields(true));
                    v.push(c.fields(false));
                }
            }
            for &(w, h) in &sizes {
                let mut c = Foreign::new(enc, formats[0], w, h).opt("-profile:v", profile);
                c.frames = 1;
                v.push(c);
            }
            for rate in ["100k", "20M", "900M"] {
                v.push(
                    Foreign::new(enc, formats[0], 320, 180)
                        .opt("-profile:v", profile)
                        .opt("-b:v", rate)
                        .noise(),
                );
            }
        }
        // Alpha through the platform encoder.
        for alpha in [
            "geq=r='r(X,Y)':g='g(X,Y)':b='b(X,Y)':a='mod(X*3+Y*5,256)'",
            "geq=r='r(X,Y)':g='g(X,Y)':b='b(X,Y)':a='random(1)*255'",
            "geq=r='r(X,Y)':g='g(X,Y)':b='b(X,Y)':a=0",
        ] {
            for profile in [4, 5] {
                let mut c = Foreign::new(enc, "bgra", 130, 74).opt("-profile:v", profile);
                c.filters = format!("format=bgra,{alpha}");
                v.push(c);
            }
        }
        for (prim, trc, mat, codes) in [
            ("bt709", "bt709", "bt709", (1, 1, 1)),
            ("bt2020", "smpte2084", "bt2020nc", (9, 16, 9)),
            ("bt2020", "arib-std-b67", "bt2020nc", (9, 18, 9)),
            ("smpte432", "bt709", "bt709", (12, 1, 1)),
        ] {
            let mut c = Foreign::new(enc, "p210le", 130, 74)
                .opt("-profile:v", 3)
                .filter(&format!(
                    "setparams=color_primaries={prim}:color_trc={trc}:colorspace={mat}"
                ));
            c.colour = Some(codes);
            c.frames = 1;
            v.push(c);
        }
    }
    v
}

#[derive(Debug, Default)]
struct Outcome {
    /// The black-box encoder refused the option combination.
    refused: bool,
    frames: usize,
    bytes: usize,
    tolerance: i32,
    /// Every dequantised coefficient is inside the Annex A range.
    in_domain: bool,
    /// Largest per-sample distance to the nearer validator.
    nearest: i32,
    /// What the software decoder printed at warning level.
    diagnostics: String,
    header: String,
    /// ours vs the software black-box decoder.
    software: Option<Diff>,
    /// ours vs the platform black-box decoder.
    platform: Option<Diff>,
    /// software vs platform, the disagreement between the two validators.
    validators: Option<Diff>,
    problems: Vec<String>,
}

fn run_foreign(dir: &TempDir, idx: usize, c: &Foreign, t: &Tools) -> Outcome {
    let mut o = Outcome::default();
    let mov = dir.path(&format!("f{idx}.mov"));
    // The lavfi sources only produce even sizes: generate the next even
    // size up and crop to the requested one after format conversion.
    let size = format!(
        "{}x{}",
        c.width.next_multiple_of(2),
        c.height.next_multiple_of(2)
    );
    let mut vf = c.filters.clone();
    if !vf.is_empty() {
        vf.push(',');
    }
    let _ = write!(
        vf,
        "format={},crop={}:{}:0:0:exact=1",
        c.pix, c.width, c.height
    );
    let mut cmd = ffmpeg();
    cmd.args(["-f", "lavfi", "-i", &c.source.replace("{S}", &size)])
        .args(["-vf", &vf])
        .args(["-frames:v", &c.frames.to_string()])
        .args(["-c:v", c.encoder])
        .args(&c.opts)
        .arg(&mov);
    match run(&mut cmd) {
        Some((true, _)) => {}
        _ => {
            o.refused = true;
            return o;
        }
    }
    let bin = dir.path(&format!("f{idx}.bin"));
    let ok = run(ffmpeg()
        .arg("-i")
        .arg(&mov)
        .args(["-map", "0:v", "-c", "copy", "-f", "data"])
        .arg(&bin))
    .is_some_and(|r| r.0);
    let data = std::fs::read(&bin).unwrap_or_default();
    let frames = split_frames(&data);
    if !ok || frames.is_empty() {
        o.refused = true;
        return o;
    }
    o.frames = frames.len();
    let header = match parse_frame(frames[0]) {
        Ok((h, _)) => h,
        Err(e) => {
            o.problems.push(format!("frame header refused: {e}"));
            return o;
        }
    };
    let s = Surface::of(&header);
    o.tolerance = s.tolerance();
    o.header = format!(
        "v{} {} il{} a{} c{}/{}/{} q{}{} {}",
        header.bitstream_version,
        if s.chroma444 { "444" } else { "422" },
        header.interlace_mode,
        header.alpha_channel_type,
        header.color_primaries,
        header.transfer_characteristic,
        header.matrix_coefficients,
        u8::from(header.load_luma_quantization_matrix),
        u8::from(header.load_chroma_quantization_matrix),
        String::from_utf8_lossy(&header.encoder_identifier()),
    );
    if (header.width as u32, header.height as u32) != (c.width, c.height) {
        o.problems.push(format!(
            "header size {}x{} != {}x{}",
            header.width, header.height, c.width, c.height
        ));
    }
    if let Some(il) = c.interlace
        && header.interlace_mode != il
    {
        // The encoder ignored the field request: the stream is still a
        // valid progressive case, so it stays in the matrix as such.
        let _ = write!(o.header, " (asked il{il})");
    }
    if let Some((p, tr, m)) = c.colour
        && (
            header.color_primaries,
            header.transfer_characteristic,
            header.matrix_coefficients,
        ) != (p, tr, m)
    {
        o.problems.push(format!(
            "colour tags {}/{}/{} != requested {p}/{tr}/{m}",
            header.color_primaries, header.transfer_characteristic, header.matrix_coefficients
        ));
    }

    judge(dir, &format!("f{idx}"), &mov, &frames, s, t, &mut o);
    discard(&mov);
    discard(&bin);
    o
}

/// Decode `frames` with this crate and `mov` (the same frames in a
/// container) with the black-box decoders, and record the comparison.
fn judge(
    dir: &TempDir,
    name: &str,
    mov: &Path,
    frames: &[&[u8]],
    s: Surface,
    t: &Tools,
    o: &mut Outcome,
) {
    let raw = dir.path(&format!("{name}.yuv"));
    let Some((reference, diagnostics)) = decode_software(mov, &raw, s) else {
        o.problems.push("software decoder did not run".into());
        return;
    };
    // Only what the decoder says counts: the raw-video writer has its
    // own remarks about one-pixel pictures.
    o.diagnostics = diagnostics
        .lines()
        .filter(|l| !l.contains("rawvideo") && !l.trim().is_empty())
        .collect::<Vec<_>>()
        .join("\n");
    let fb = s.frame_bytes();
    if reference.len() != fb * frames.len() {
        o.problems.push(format!(
            "software decoder wrote {} bytes, expected {}",
            reference.len(),
            fb * frames.len()
        ));
        return;
    }
    let hw_raw = dir.path(&format!("{name}.vt.yuv"));
    let platform_decode = || {
        if !t.platform_decoder {
            return None;
        }
        decode_platform(mov, &hw_raw, s).filter(|p| p.len() == fb * frames.len())
    };
    let mut platform = platform_decode();
    let range = frames
        .iter()
        .map(|pkt| dequantized_coefficient_range(pkt))
        .try_fold((0i64, 0i64), |acc, r| {
            r.map(|(lo, hi)| (acc.0.min(lo), acc.1.max(hi)))
        });
    match range {
        Ok((lo, hi)) => o.in_domain = lo >= -16384 && hi <= 16382,
        Err(e) => {
            o.problems.push(format!("coefficient walk failed: {e}"));
            return;
        }
    }
    let mut decoded = Vec::with_capacity(frames.len());
    for (i, pkt) in frames.iter().enumerate() {
        let ours =
            match decode_packet_with_format(pkt, None, Some(s.pixel_format()), OutputRange::Full) {
                Ok(f) => frame_planes(&f, s),
                Err(e) => {
                    o.problems.push(format!("frame {i}: decode error: {e}"));
                    return;
                }
            };
        // Coded alpha at full 16-bit precision (§7.5.2 at b = 16).
        let alpha16 = if s.alpha {
            match decode_packet_with_format(pkt, None, Some(s.pixel_format16()), OutputRange::Full)
            {
                Ok(f) => frame_planes(&f, s).pop(),
                Err(e) => {
                    o.problems
                        .push(format!("frame {i}: 16-bit decode error: {e}"));
                    return;
                }
            }
        } else {
            None
        };
        decoded.push((ours, alpha16));
    }
    let compare = |platform: Option<&Vec<u8>>| {
        let mut sw = Diff::default();
        let mut hw = Diff::default();
        let mut vv = Diff::default();
        let mut nearest = 0;
        for (i, (ours, alpha16)) in decoded.iter().enumerate() {
            let r = raw_planes(&reference[i * fb..(i + 1) * fb], s);
            let p = platform.map(|p| raw_planes(&p[i * fb..(i + 1) * fb], s));
            let d = compare_frame(ours, alpha16.as_deref(), &r, p.as_deref(), s);
            sw.merge(d.software);
            hw.merge(d.platform);
            vv.merge(d.validators);
            nearest = nearest.max(d.nearest);
        }
        (sw, hw, vv, nearest)
    };
    let (mut sw, mut hw, mut vv, mut nearest) = compare(platform.as_ref());
    if platform.is_some() && (hw.colour_max > s.tolerance() || hw.alpha_max != 0) {
        // The platform decoder occasionally returns a damaged frame when
        // the machine is busy: one retry before the result is believed.
        platform = platform_decode();
        (sw, hw, vv, nearest) = compare(platform.as_ref());
    }
    o.nearest = nearest;
    o.software = Some(sw);
    if platform.is_some() {
        o.platform = Some(hw);
        o.validators = Some(vv);
    }
    discard(&raw);
    discard(&hw_raw);
}

/// The problems of one judged case.
///
/// * Inside the Annex A coefficient domain the software decoder binds
///   on every colour sample.
/// * Every colour sample must agree with at least one validator. This
///   is the whole colour rule outside the Annex A domain, where the
///   validators themselves part ways (the software decoder wraps where
///   §7.5.1 clamps); inside it, it lets an isolated sample on which the
///   platform decoder alone departs be counted (`tally.platform_only`)
///   instead of failing the case.
/// * Alpha is lossless and binds exactly on both validators.
fn verdict(o: &Outcome, tally: &mut Tally) -> Vec<String> {
    let tol = o.tolerance;
    let mut problems = o.problems.clone();
    if let Some(d) = o.software
        && o.in_domain
        && d.colour_max > tol
    {
        problems.push(format!(
            "colour differs from the software decoder by {} (> {tol})",
            d.colour_max
        ));
    }
    for (name, d) in [("software", o.software), ("platform", o.platform)] {
        if let Some(d) = d
            && d.alpha_max != 0
        {
            problems.push(format!(
                "alpha differs from the {name} decoder by {}",
                d.alpha_max
            ));
        }
    }
    if let Some(d) = o.platform {
        tally.platform_ran += 1;
        if o.nearest > tol {
            problems.push(format!(
                "a colour sample is {} (> {tol}) from the nearer decoder",
                o.nearest
            ));
        } else if o.in_domain && d.colour_max > tol {
            tally.platform_only += 1;
        }
    }
    if o.software.is_some() && !o.in_domain {
        tally.out_of_domain += 1;
    }
    problems
}

#[derive(Default)]
struct Tally {
    /// Cases with a coefficient outside the Annex A range.
    out_of_domain: usize,
    /// Cases the platform decoder decoded.
    platform_ran: usize,
    /// In-domain cases where only the platform decoder departs from
    /// this crate and the software decoder.
    platform_only: usize,
}

impl Tally {
    fn describe(&self) -> String {
        format!(
            "{} outside the Annex A coefficient domain; platform decoder ran on {}, alone in \
             departing on {}",
            self.out_of_domain, self.platform_ran, self.platform_only
        )
    }
}

fn fmt_diff(d: Option<Diff>) -> String {
    match d {
        Some(d) => format!("{}/{}/{}", d.colour_max, d.alpha_max, d.differing),
        None => "-".into(),
    }
}

#[test]
fn foreign_streams_decode_like_the_black_box_decoders() {
    let Some(t) = tools() else {
        eprintln!("ffmpeg not found: black-box matrix skipped");
        return;
    };
    if t.encoders.is_empty() {
        eprintln!("ffmpeg exposes no ProRes encoder: black-box matrix skipped");
        return;
    }
    let dir = TempDir::new("foreign");
    let mut cases = foreign_cases(&t);
    cases.retain(|c| selected(&c.label()));
    let mut report = String::new();
    let (mut ran, mut passed, mut refused) = (0, 0, 0);
    let mut tally = Tally::default();
    let mut failures = Vec::new();
    for (idx, c) in cases.iter().enumerate() {
        let o = run_foreign(&dir, idx, c, &t);
        if o.refused {
            refused += 1;
            let _ = writeln!(report, "REFUSED  {}", c.label());
            continue;
        }
        ran += 1;
        let problems = verdict(&o, &mut tally);
        let verdict = if problems.is_empty() { "PASS" } else { "FAIL" };
        let _ = writeln!(
            report,
            "{verdict:8} {} | {} | frames {}{} | sw {} | hw {} | sw-vs-hw {} | {}",
            c.label(),
            o.header,
            o.frames,
            if o.in_domain { "" } else { " OUT-OF-DOMAIN" },
            fmt_diff(o.software),
            fmt_diff(o.platform),
            fmt_diff(o.validators),
            problems.join("; ")
        );
        if problems.is_empty() {
            passed += 1;
        } else {
            failures.push(format!("{}: {}", c.label(), problems.join("; ")));
        }
    }
    let summary = format!(
        "foreign -> ours: {passed}/{ran} cases pass ({}; {refused} option combinations refused \
         by the black-box encoder, {} generated)",
        tally.describe(),
        cases.len()
    );
    eprintln!("{summary}");
    let _ = writeln!(report, "{summary}");
    write_report("foreign", &report);
    assert!(
        failures.is_empty(),
        "{} of {ran} foreign cases fail:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

// ---------------------------------------------------------------------
// ours -> foreign
// ---------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
enum Texture {
    /// Gradients, block edges and mild noise inside the nominal range.
    Pattern,
    /// Uniform noise over every code.
    Noise,
    /// One value, in per-mille of the largest code.
    Flat(u32),
    /// Smallest / largest code alternating per sample.
    Checker,
    /// Smallest / largest code in 8-sample bars.
    Bars,
}

#[derive(Clone, Copy, Debug)]
enum AlphaTexture {
    Ramp,
    Noise,
    Opaque,
    Clear,
    Steps,
}

#[derive(Clone, Debug)]
struct Ours {
    note: &'static str,
    pix: PixelFormat,
    width: u32,
    height: u32,
    frames: u32,
    texture: Texture,
    /// Alpha content, and for the formats without a typed alpha plane
    /// the width of the alpha samples handed to the encoder (8 or 16).
    alpha: Option<(AlphaTexture, u32)>,
    config: EncoderConfig,
    bit_rate: Option<u64>,
    /// Colour names the black-box prober must report.
    colour: Option<[&'static str; 3]>,
}

impl Ours {
    fn new(pix: PixelFormat, width: u32, height: u32, config: EncoderConfig) -> Self {
        let typed_alpha = format_info(pix).2;
        Self {
            note: "",
            pix,
            width,
            height,
            frames: 2,
            texture: Texture::Pattern,
            alpha: typed_alpha.then_some((AlphaTexture::Ramp, 0)),
            config,
            bit_rate: None,
            colour: None,
        }
    }
    fn texture(mut self, t: Texture) -> Self {
        self.texture = t;
        self
    }
    fn alpha(mut self, t: AlphaTexture, untyped_bits: u32) -> Self {
        self.alpha = Some((t, untyped_bits));
        self
    }
    fn note(mut self, n: &'static str) -> Self {
        self.note = n;
        self
    }
    fn label(&self) -> String {
        let c = &self.config;
        format!(
            "{:?} {}x{} n{} {:?} alpha {:?} | profile {:?} qi {:?} il{} mbs {:?} qmat {} explicit {} \
             rc {} rate {:?} min {:?} act {:?} {}",
            self.pix,
            self.width,
            self.height,
            self.frames,
            self.texture,
            self.alpha,
            c.profile,
            c.quantization_index,
            c.interlace_mode,
            c.mbs_per_slice,
            c.quant_matrices.map_or("-".to_owned(), |m| format!(
                "{}/{}",
                m.luma[63], m.chroma[63]
            )),
            c.explicit_qmat_carriage,
            c.rate_control,
            self.bit_rate,
            c.min_frame_size,
            c.alpha_channel_type,
            self.note
        )
    }
}

/// `(bits, 4:4:4, alpha-typed)` of an encoder input format.
fn format_info(pix: PixelFormat) -> (u32, bool, bool) {
    match pix {
        PixelFormat::Yuv422P => (8, false, false),
        PixelFormat::Yuv422P10Le => (10, false, false),
        PixelFormat::Yuv422P12Le => (12, false, false),
        PixelFormat::Yuv422P16Le => (16, false, false),
        PixelFormat::Yuv444P => (8, true, false),
        PixelFormat::Yuv444P10Le => (10, true, false),
        PixelFormat::Yuv444P12Le => (12, true, false),
        PixelFormat::Yuv444P16Le => (16, true, false),
        PixelFormat::Yuva422P => (8, false, true),
        PixelFormat::Yuva422P10Le => (10, false, true),
        PixelFormat::Yuva422P12Le => (12, false, true),
        PixelFormat::Yuva422P16Le => (16, false, true),
        PixelFormat::Yuva444P => (8, true, true),
        PixelFormat::Yuva444P10Le => (10, true, true),
        PixelFormat::Yuva444P12Le => (12, true, true),
        PixelFormat::Yuva444P16Le => (16, true, true),
        other => panic!("not a ProRes encoder input format: {other:?}"),
    }
}

struct Lcg(u32);

impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        self.0 >> 8
    }
}

fn pack(samples: &[u32], w: usize, bits: u32) -> VideoPlane {
    if bits == 8 {
        VideoPlane {
            stride: w,
            data: samples.iter().map(|&v| v as u8).collect(),
        }
    } else {
        VideoPlane {
            stride: w * 2,
            data: samples
                .iter()
                .flat_map(|&v| (v as u16).to_le_bytes())
                .collect(),
        }
    }
}

fn colour_plane(w: usize, h: usize, bits: u32, t: Texture, seed: u32) -> VideoPlane {
    let max = (1u32 << bits) - 1;
    let (lo, hi) = (max / 16, max - max / 16);
    let mut rng = Lcg(seed.wrapping_mul(2_654_435_761).wrapping_add(17));
    let phase = (seed as usize % 7) * 31;
    let mut v = Vec::with_capacity(w * h);
    for y in 0..h {
        for x in 0..w {
            let r = rng.next();
            v.push(match t {
                Texture::Pattern => {
                    let pm = ((x * 9 + y * 5 + phase) % 700) as u32
                        + (((x / 8 + y / 8) % 2) as u32) * 150
                        + r % 150;
                    lo + (hi - lo) * pm / 1000
                }
                Texture::Noise => r % (max + 1),
                Texture::Flat(pm) => max * pm / 1000,
                Texture::Checker => max * ((x + y) as u32 & 1),
                Texture::Bars => max * ((x / 8) as u32 & 1),
            });
        }
    }
    pack(&v, w, bits)
}

fn alpha_plane(w: usize, h: usize, bits: u32, t: AlphaTexture, seed: u32) -> VideoPlane {
    let max = (1u32 << bits) - 1;
    let mut rng = Lcg(seed ^ 0x5EED_A1FA);
    let mut v = Vec::with_capacity(w * h);
    for y in 0..h {
        for x in 0..w {
            let r = rng.next();
            v.push(match t {
                AlphaTexture::Ramp => ((x * 37 + y * 11) as u32 * (max / 255).max(1)) % (max + 1),
                AlphaTexture::Noise => r % (max + 1),
                AlphaTexture::Opaque => max,
                AlphaTexture::Clear => 0,
                AlphaTexture::Steps => ((x / 20 + y / 9) as u32 * (max / 13)) % (max + 1),
            });
        }
    }
    pack(&v, w, bits)
}

fn synth(c: &Ours, frame: u32) -> VideoFrame {
    let (bits, chroma444, typed) = format_info(c.pix);
    let (w, h) = (c.width as usize, c.height as usize);
    let cw = if chroma444 { w } else { w.div_ceil(2) };
    let seed = frame * 97 + 1;
    let mut planes = vec![
        colour_plane(w, h, bits, c.texture, seed),
        colour_plane(cw, h, bits, c.texture, seed + 1),
        colour_plane(cw, h, bits, c.texture, seed + 2),
    ];
    if let Some((t, untyped_bits)) = c.alpha {
        let alpha_bits = if typed { bits } else { untyped_bits };
        planes.push(alpha_plane(w, h, alpha_bits, t, seed + 3));
    }
    VideoFrame {
        pts: Some(i64::from(frame)),
        planes,
    }
}

const PROFILES: [Profile; 6] = [
    Profile::Proxy,
    Profile::Lt,
    Profile::Standard,
    Profile::Hq,
    Profile::Prores4444,
    Profile::Prores4444Xq,
];

fn is_444(p: Profile) -> bool {
    matches!(p, Profile::Prores4444 | Profile::Prores4444Xq)
}

fn plain_format(p: Profile, bits: u32) -> PixelFormat {
    match (is_444(p), bits) {
        (false, 8) => PixelFormat::Yuv422P,
        (false, 10) => PixelFormat::Yuv422P10Le,
        (false, 12) => PixelFormat::Yuv422P12Le,
        (false, _) => PixelFormat::Yuv422P16Le,
        (true, 8) => PixelFormat::Yuv444P,
        (true, 10) => PixelFormat::Yuv444P10Le,
        (true, 12) => PixelFormat::Yuv444P12Le,
        (true, _) => PixelFormat::Yuv444P16Le,
    }
}

fn ours_cases() -> Vec<Ours> {
    let mut v = Vec::new();
    let cfg = EncoderConfig::for_profile;

    // Profile x input depth x field order x content.
    for p in PROFILES {
        for bits in [8, 10, 12, 16] {
            for il in 0..3u8 {
                for t in [Texture::Pattern, Texture::Noise] {
                    let c = cfg(p).with_interlace_mode(il);
                    v.push(Ours::new(plain_format(p, bits), 130, 74, c.clone()).texture(t));
                    if il != 0 && bits == 10 {
                        // Odd height: the two fields differ in size (§6.2).
                        v.push(Ours::new(plain_format(p, bits), 130, 75, c).texture(t));
                    }
                }
            }
        }
    }

    // Alpha: typed surfaces (8-bit codes 8-bit alpha, deep codes 16-bit
    // alpha), the untyped 4-plane convention at both coded widths, on
    // 4:4:4 and on 4:2:2 (bitstream_version 1, §6.4).
    let alpha_textures = [
        AlphaTexture::Ramp,
        AlphaTexture::Noise,
        AlphaTexture::Opaque,
        AlphaTexture::Clear,
        AlphaTexture::Steps,
    ];
    for (pix, p) in [
        (PixelFormat::Yuva444P, Profile::Prores4444),
        (PixelFormat::Yuva444P10Le, Profile::Prores4444),
        (PixelFormat::Yuva444P12Le, Profile::Prores4444Xq),
        (PixelFormat::Yuva444P16Le, Profile::Prores4444),
        (PixelFormat::Yuva422P, Profile::Hq),
        (PixelFormat::Yuva422P10Le, Profile::Standard),
        (PixelFormat::Yuva422P16Le, Profile::Proxy),
    ] {
        for at in alpha_textures {
            for il in 0..3u8 {
                for (w, h) in [(130, 74), (16, 16), (300, 41)] {
                    v.push(Ours::new(pix, w, h, cfg(p).with_interlace_mode(il)).alpha(at, 0));
                }
            }
        }
    }
    for (bits, act) in [
        (8, AlphaChannelType::Eight),
        (16, AlphaChannelType::Sixteen),
    ] {
        for at in alpha_textures {
            for explicit in [false, true] {
                let mut c = cfg(Profile::Prores4444);
                if explicit {
                    c = c.with_alpha_channel_type(act);
                }
                v.push(Ours::new(PixelFormat::Yuv444P10Le, 130, 74, c).alpha(at, bits));
            }
        }
    }

    // Every quantisation index class (§7.3 Table 15), both chroma formats.
    for qi in [
        1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 64, 127, 128, 129, 160, 200, 224,
    ] {
        for (pix, p) in [
            (PixelFormat::Yuv422P10Le, Profile::Hq),
            (PixelFormat::Yuva444P12Le, Profile::Prores4444),
        ] {
            for t in [Texture::Pattern, Texture::Noise] {
                v.push(Ours::new(pix, 130, 74, cfg(p).with_quantization_index(qi)).texture(t));
            }
        }
    }

    // Slice size x picture width.
    for mbs in [1, 2, 4, 8] {
        for w in [16, 48, 130, 258, 722] {
            v.push(Ours::new(
                PixelFormat::Yuv422P10Le,
                w,
                34,
                cfg(Profile::Standard).with_mbs_per_slice(mbs),
            ));
        }
        v.push(Ours::new(
            PixelFormat::Yuva444P10Le,
            130,
            34,
            cfg(Profile::Prores4444)
                .with_mbs_per_slice(mbs)
                .with_interlace_mode(2),
        ));
    }

    // Quantisation matrices: the three presets per profile, both header
    // carriages, and hand-made tables at the weight extremes.
    for p in PROFILES {
        let pix = plain_format(p, 10);
        for (note, c) in [
            ("perceptual", EncoderConfig::perceptual().with_profile(p)),
            (
                "perceptual-for-profile",
                EncoderConfig::perceptual_for_profile(p),
            ),
            ("signature", EncoderConfig::signature_for_profile(p)),
            (
                "signature explicit",
                EncoderConfig::signature_for_profile(p).with_explicit_qmat_carriage(),
            ),
            ("flat explicit", cfg(p).with_explicit_qmat_carriage()),
        ] {
            v.push(
                Ours::new(pix, 130, 74, c)
                    .texture(Texture::Noise)
                    .note(note),
            );
        }
    }
    let ramp: [u8; 64] = std::array::from_fn(|i| 2 + (i as u8 * 61 / 63));
    let reverse: [u8; 64] = std::array::from_fn(|i| 63 - (i as u8 * 61 / 63));
    for (note, luma, chroma) in [
        ("luma only", ramp, ramp),
        ("both", ramp, reverse),
        ("all 2", [2; 64], [2; 64]),
        ("all 63", [63; 64], [63; 64]),
        ("chroma only", [4; 64], reverse),
    ] {
        for (pix, p, il) in [
            (PixelFormat::Yuv422P10Le, Profile::Standard, 0),
            (PixelFormat::Yuv444P12Le, Profile::Prores4444, 1),
        ] {
            v.push(
                Ours::new(
                    pix,
                    130,
                    74,
                    cfg(p)
                        .with_interlace_mode(il)
                        .with_quant_matrices(QuantMatrices { luma, chroma }),
                )
                .texture(Texture::Noise)
                .note(note),
            );
        }
    }

    // Rate control at tiny, ordinary and huge budgets; frame stuffing.
    for p in PROFILES {
        for rate in [200_000u64, 4_000_000, 40_000_000, 2_000_000_000] {
            let mut c = Ours::new(plain_format(p, 10), 320, 180, cfg(p).with_rate_control())
                .texture(Texture::Noise);
            c.bit_rate = Some(rate);
            v.push(c);
        }
        v.push(
            Ours::new(
                plain_format(p, 10),
                130,
                74,
                cfg(p).with_min_frame_size(200_000),
            )
            .note("stuffed"),
        );
    }

    // Picture sizes.
    let mut sizes = vec![
        (1, 1),
        (2, 2),
        (1, 17),
        (17, 1),
        (4, 2),
        (2, 16),
        (16, 2),
        (16, 16),
        (18, 18),
        (30, 30),
        (34, 50),
        (126, 2),
        (2, 126),
        (131, 75),
        (255, 255),
        (720, 486),
        (720, 576),
        (1280, 720),
        (1920, 1080),
    ];
    if large_rows() {
        sizes.extend([(2048, 1556), (3840, 2160), (4096, 2160), (8192, 4320)]);
    }
    for &(w, h) in &sizes {
        for (pix, p) in [
            (PixelFormat::Yuv422P10Le, Profile::Standard),
            (PixelFormat::Yuva444P12Le, Profile::Prores4444),
        ] {
            for il in [0, 1] {
                if il == 1 && (h < 2 || w * h > 1920 * 1080) {
                    continue;
                }
                let mut c = Ours::new(pix, w, h, cfg(p).with_interlace_mode(il));
                c.frames = 1;
                v.push(c);
            }
        }
    }

    // Picture content at the code extremes.
    for t in [
        Texture::Flat(0),
        Texture::Flat(1000),
        Texture::Flat(500),
        Texture::Checker,
        Texture::Bars,
    ] {
        for p in [Profile::Proxy, Profile::Hq, Profile::Prores4444Xq] {
            for bits in [8, 10, 12] {
                v.push(Ours::new(plain_format(p, bits), 130, 74, cfg(p)).texture(t));
            }
        }
    }

    // Header metadata (§6.1.1 Tables 3-6).
    for (prim, trc, mat, names) in [
        (1, 1, 1, ["bt709", "bt709", "bt709"]),
        (5, 1, 6, ["bt470bg", "bt709", "smpte170m"]),
        (6, 1, 6, ["smpte170m", "bt709", "smpte170m"]),
        (9, 16, 9, ["bt2020", "smpte2084", "bt2020nc"]),
        (9, 18, 9, ["bt2020", "arib-std-b67", "bt2020nc"]),
        (11, 1, 1, ["smpte431", "bt709", "bt709"]),
        (12, 1, 1, ["smpte432", "bt709", "bt709"]),
    ] {
        for p in [Profile::Hq, Profile::Prores4444] {
            let meta = FrameMeta::default()
                .with_color_primaries(prim)
                .with_transfer_characteristic(trc)
                .with_matrix_coefficients(mat);
            let mut c = Ours::new(plain_format(p, 10), 130, 74, cfg(p).with_meta(meta));
            c.colour = Some(names);
            c.frames = 1;
            v.push(c);
        }
    }
    for code in 1..=11u8 {
        for aspect in [1, 2, 3] {
            let meta = FrameMeta::default()
                .with_frame_rate_code(code)
                .with_aspect_ratio_information(aspect);
            let mut c = Ours::new(
                PixelFormat::Yuv422P10Le,
                64,
                48,
                cfg(Profile::Lt).with_meta(meta),
            )
            .note("frame rate / aspect codes");
            c.frames = 1;
            v.push(c);
        }
    }

    // A longer stream.
    let mut long =
        Ours::new(PixelFormat::Yuv422P10Le, 96, 64, cfg(Profile::Lt)).texture(Texture::Noise);
    long.frames = 40;
    v.push(long);
    v
}

fn run_ours(dir: &TempDir, idx: usize, c: &Ours, t: &Tools) -> Outcome {
    let mut o = Outcome::default();
    let mut params = CodecParameters::video(CodecId::new("prores"));
    params.media_type = MediaType::Video;
    params.width = Some(c.width);
    params.height = Some(c.height);
    params.pixel_format = Some(c.pix);
    params.frame_rate = Some(Rational::new(25, 1));
    params.bit_rate = c.bit_rate;
    let mut enc = match make_encoder_with_config(&params, c.config.clone()) {
        Ok(e) => e,
        Err(e) => {
            o.problems
                .push(format!("encoder refused the configuration: {e}"));
            return o;
        }
    };
    let mut packets = Vec::new();
    for f in 0..c.frames {
        let packet = enc
            .send_frame(&Frame::Video(synth(c, f)))
            .and_then(|()| enc.receive_packet());
        match packet {
            Ok(p) => packets.push(p.data),
            Err(e) => {
                o.problems.push(format!("frame {f}: encode error: {e}"));
                return o;
            }
        }
    }
    o.frames = packets.len();
    o.bytes = packets.iter().map(Vec::len).sum();
    let header = match parse_frame(&packets[0]) {
        Ok((h, _)) => h,
        Err(e) => {
            o.problems.push(format!("own frame header refused: {e}"));
            return o;
        }
    };
    let s = Surface::of(&header);
    o.tolerance = s.tolerance();
    o.header = format!(
        "v{} {} il{} a{} c{}/{}/{} q{}{}",
        header.bitstream_version,
        if s.chroma444 { "444" } else { "422" },
        header.interlace_mode,
        header.alpha_channel_type,
        header.color_primaries,
        header.transfer_characteristic,
        header.matrix_coefficients,
        u8::from(header.load_luma_quantization_matrix),
        u8::from(header.load_chroma_quantization_matrix),
    );
    for (i, p) in packets.iter().enumerate() {
        std::fs::write(dir.path(&format!("o{idx}_{i}.pro")), p).expect("write packet");
    }
    // Container: the black-box muxer wraps the raw frames unchanged.
    let profile = c.config.profile.expect("every case names its profile");
    let mov = dir.path(&format!("o{idx}.mov"));
    let remuxed = run(ffmpeg()
        .args(["-f", "image2", "-framerate", "25", "-c:v", "prores", "-i"])
        .arg(dir.path(&format!("o{idx}_%d.pro")))
        .args(["-c", "copy", "-tag:v"])
        .arg(String::from_utf8_lossy(profile.fourcc()).as_ref())
        .arg(&mov))
    .is_some_and(|r| r.0);
    if !remuxed {
        o.problems
            .push("the black-box muxer refused the stream".into());
        return o;
    }
    let frames: Vec<&[u8]> = packets.iter().map(Vec::as_slice).collect();
    judge(dir, &format!("o{idx}"), &mov, &frames, s, t, &mut o);
    if !o.diagnostics.is_empty() {
        o.problems.push(format!(
            "software decoder diagnostics: {}",
            o.diagnostics.replace('\n', " / ")
        ));
    }
    if let Some(names) = c.colour {
        let probed = Command::new("ffprobe")
            .args(["-v", "error", "-select_streams", "v:0", "-show_entries"])
            .arg("stream=color_primaries,color_transfer,color_space")
            .args(["-of", "default=nw=1"])
            .arg(&mov)
            .output()
            .map(|out| String::from_utf8_lossy(&out.stdout).into_owned())
            .unwrap_or_default();
        for (key, name) in ["color_primaries", "color_transfer", "color_space"]
            .iter()
            .zip(names)
        {
            if !probed.is_empty() && !probed.contains(&format!("{key}={name}")) {
                o.problems.push(format!(
                    "prober does not report {key}={name}: {}",
                    probed.replace('\n', " ")
                ));
            }
        }
    }
    discard(&mov);
    for i in 0..packets.len() {
        discard(dir.path(&format!("o{idx}_{i}.pro")));
    }
    o
}

#[test]
fn own_streams_decode_identically_in_the_black_box_decoders() {
    let Some(t) = tools() else {
        eprintln!("ffmpeg not found: black-box matrix skipped");
        return;
    };
    let dir = TempDir::new("ours");
    let mut cases = ours_cases();
    cases.retain(|c| selected(&c.label()));
    let mut report = String::new();
    let mut passed = 0;
    let mut tally = Tally::default();
    let mut failures = Vec::new();
    for (idx, c) in cases.iter().enumerate() {
        let o = run_ours(&dir, idx, c, &t);
        let problems = verdict(&o, &mut tally);
        let _ = writeln!(
            report,
            "{:8} {} | {} | frames {} bytes {}{} | sw {} | hw {} | sw-vs-hw {} | {}",
            if problems.is_empty() { "PASS" } else { "FAIL" },
            c.label(),
            o.header,
            o.frames,
            o.bytes,
            if o.in_domain { "" } else { " OUT-OF-DOMAIN" },
            fmt_diff(o.software),
            fmt_diff(o.platform),
            fmt_diff(o.validators),
            problems.join("; ")
        );
        if problems.is_empty() {
            passed += 1;
        } else {
            failures.push(format!("{}: {}", c.label(), problems.join("; ")));
        }
    }
    let summary = format!(
        "ours -> foreign: {passed}/{} cases pass ({})",
        cases.len(),
        tally.describe()
    );
    eprintln!("{summary}");
    let _ = writeln!(report, "{summary}");
    write_report("ours", &report);
    assert!(
        failures.is_empty(),
        "{} of {} own-stream cases fail:\n{}",
        failures.len(),
        cases.len(),
        failures.join("\n")
    );
}
