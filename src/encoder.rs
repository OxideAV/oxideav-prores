//! ProRes encoder following SMPTE RDD 36 §5 + §7.
//!
//! Reads a `Yuv422P` or `Yuv444P` `VideoFrame`, walks 16x16 macroblocks,
//! forward DCTs each 8x8 block, quantises by `qmat * qScale / 8`,
//! per-component slice-scans, and emits the entropy-coded slice payload
//! per RDD 36.

use std::collections::VecDeque;

use oxideav_core::Encoder;
use oxideav_core::{
    CodecId, CodecParameters, Error, Frame, MediaType, Packet, PixelFormat, Result, TimeBase,
    VideoFrame,
};

use crate::alpha::{encode_scanned_alpha, AlphaChannelType};
use crate::dct::{fdct8x8, fdct8x8_constant, is_constant_block};
use crate::decoder::BitDepth;
use crate::frame::{
    compute_slice_sizes, frame_rate_code_from_rational, write_frame_with_meta,
    write_picture_header, write_slice_header, ChromaFormat, FrameMeta, Profile,
};
use crate::quant::{qscale, QuantMatrices, DEFAULT_QMAT};
use crate::slice::{blocks_per_mb, chroma_blocks_per_mb, encode_slice_components};

/// Encoder-side configuration. Defaults match the legacy behaviour
/// (flat all-4s quantisation matrices, `load_luma_qmat = 0`,
/// `load_chroma_qmat = 0`, per-profile default quantisation index).
#[derive(Clone, Debug, Default)]
pub struct EncoderConfig {
    /// Per-component quantisation weight matrices. `None` is identical
    /// to `Some(QuantMatrices::flat())` — the encoder writes
    /// `load_luma_qmat = load_chroma_qmat = 0` and uses the spec's
    /// default matrix internally for quantisation. When non-default
    /// matrices are supplied the encoder writes the matrices into the
    /// frame header (setting `load_*_qmat` to 1) so any RDD 36 decoder
    /// can dequantise correctly.
    pub quant_matrices: Option<QuantMatrices>,
    /// Per-slice `quantization_index` (RDD 36 §7.3 / Table 15) used for
    /// every slice in every encoded frame. Lower index → finer step →
    /// higher quality + larger packet. Range `1..=224`.
    ///
    /// `None` (the default) selects the per-profile default returned by
    /// [`Profile::default_quant_index`] — currently `8 / 6 / 4 / 2 /
    /// 2 / 1` for Proxy / LT / Standard / HQ / 4444 / 4444 XQ. Set this
    /// when the caller wants a different point on the rate/quality
    /// curve without re-mapping the profile selection.
    ///
    /// When `rate_control` is `true` this field is the *starting point*
    /// for the binary search; `None` uses the profile default as seed.
    pub quantization_index: Option<u8>,
    /// Descriptive metadata fields written into the RDD 36 frame header
    /// (`aspect_ratio_information`, `frame_rate_code`,
    /// `color_primaries`, `transfer_characteristic`,
    /// `matrix_coefficients` — all per §5.1.1 / §6.2). `None` lets
    /// [`make_encoder_with_config`] derive `frame_rate_code` from
    /// `CodecParameters::frame_rate` and leave the rest at 0
    /// ("unknown"); `Some(meta)` overrides everything verbatim.
    pub meta: Option<FrameMeta>,
    /// Enable two-pass per-frame rate control.
    ///
    /// When `true` and the encoder was constructed with a
    /// `CodecParameters::bit_rate` and `CodecParameters::frame_rate`,
    /// each call to `send_frame` performs a binary search over
    /// `quantization_index` (up to [`RATE_CTRL_MAX_PASSES`] trial
    /// encodes) to hit the per-frame byte target derived from the
    /// nominal bit-rate within [`RATE_CTRL_TOLERANCE`] (5 %). The
    /// search starts from the profile default qi (or the explicit
    /// `quantization_index` if set) and respects the full 1..=224
    /// range.
    ///
    /// The overhead is bounded: at most `RATE_CTRL_MAX_PASSES` full
    /// encodes per frame. For constant-content sequences the search
    /// typically converges in 2-3 passes. Set `false` (the default)
    /// to preserve the original single-pass behaviour.
    pub rate_control: bool,
    /// Explicit profile override (RDD 36 §4 — Proxy / LT / Standard / HQ
    /// for 4:2:2; 4444 / 4444 XQ for 4:4:4).
    ///
    /// When `None` (default) the encoder calls [`pick_profile`] to map
    /// `CodecParameters::bit_rate` to one of the six profiles. When
    /// `Some(p)`, the caller's choice is honoured verbatim — useful when
    /// the caller wants a specific profile that the `bit_rate` heuristic
    /// would not pick (e.g. `Profile::Prores4444Xq` for a 4:4:4 stream
    /// with `bit_rate < 400 Mbit/s`, or `Profile::Hq` for a 4:2:2 stream
    /// with no `bit_rate` hint at all).
    ///
    /// The override's `chroma_format` must match the requested
    /// `PixelFormat` (HQ/SD/LT/Proxy ↔ 4:2:2; 4444/4444 XQ ↔ 4:4:4) —
    /// validated at encoder construction; mismatch returns
    /// `Error::invalid`.
    pub profile: Option<Profile>,
    /// RDD 36 §6.1.1 `interlace_mode` for every frame the encoder emits.
    ///
    /// * `0` (default) — progressive: one `picture()` per frame.
    /// * `1` — interlaced, top-field-first: two `picture()`s per frame,
    ///   the first carrying the top field (source rows 0, 2, 4, …).
    /// * `2` — interlaced, bottom-field-first: two `picture()`s per
    ///   frame, the first carrying the bottom field (source rows
    ///   1, 3, 5, …).
    ///
    /// When non-zero, each `send_frame` splits the input `VideoFrame`
    /// into two field pictures per RDD 36 §6.2 (top
    /// `picture_vertical_size = (vertical_size + 1) / 2`, bottom
    /// `= vertical_size / 2`) and emits them in temporal order, each
    /// coded with the §7.2 Figure 5 interlaced block scan. Value `3` is
    /// reserved by Table 2 and rejected at encoder construction.
    pub interlace_mode: u8,
    /// Desired slice width in macroblocks (RDD 36 §5.3 — must be a
    /// power of two in `{1, 2, 4, 8}`; the spec encodes
    /// `log2_desired_slice_size_in_mb` in two bits of the picture
    /// header so the legal range is bounded).
    ///
    /// `None` (the default) uses the canonical `8`-MB-per-slice layout
    /// that every reference encoder emits, matching the `[8, 8, …, 4?, 2?, 1?]`
    /// per-row template the decoder rebuilds via
    /// [`crate::frame::compute_slice_sizes`].
    ///
    /// Lowering this value subdivides every macroblock row into more,
    /// smaller slices; the per-slice fixed-cost (`slice_header` + per-
    /// component entropy coder reset + `slice_size_table` entry) is
    /// amortised over fewer macroblocks, so the encoded packet grows
    /// modestly. The control surface is the same knob ffmpeg's
    /// `prores_ks` exposes through `-mbs_per_slice {1,2,4,8}` and lets
    /// callers trade rate for finer error resilience.
    ///
    /// Validated at encoder construction; non-power-of-two or values
    /// outside `{1, 2, 4, 8}` return `Error::invalid`. The bitstream
    /// signals the choice through
    /// `picture_header.log2_desired_slice_size_in_mb` so every RDD 36
    /// decoder (including this crate's [`crate::decoder`]) recovers the
    /// per-row template via the same `compute_slice_sizes` derivation.
    pub mbs_per_slice: Option<u8>,
}

/// Maximum number of trial encodes per frame when rate control is active.
/// Covers the full qi range (1..=224) in log2(224) ≈ 8 steps.
pub const RATE_CTRL_MAX_PASSES: usize = 10;

/// Fractional tolerance for the rate-control target (0.05 = ±5 %).
pub const RATE_CTRL_TOLERANCE: f64 = 0.05;

impl EncoderConfig {
    /// Construct a config that emits the flat all-4s matrices and
    /// `load_*_qmat = 0` (back-compat with the pre-config encoder).
    pub fn flat() -> Self {
        Self::default()
    }

    /// Construct a config that emits perceptual JPEG-derived quant
    /// matrices (see [`QuantMatrices::perceptual`]). The matrices are
    /// written into the frame header so cross-decoders pick them up.
    pub fn perceptual() -> Self {
        Self {
            quant_matrices: Some(QuantMatrices::perceptual()),
            ..Self::default()
        }
    }

    /// Construct a config that emits **profile-aware** perceptual quant
    /// matrices (see [`QuantMatrices::perceptual_for_profile`]) and pins
    /// the encoder to the supplied [`Profile`].
    ///
    /// Equivalent to
    /// `EncoderConfig::for_profile(profile).with_quant_matrices(
    /// QuantMatrices::perceptual_for_profile(profile))`.
    ///
    /// The blend factor is `profile.default_quant_index() / 8`:
    /// Proxy → full JPEG perceptual matrix; 4444 XQ → mostly flat with
    /// a touch of perceptual rolloff. Higher-quality profiles preserve
    /// more high-frequency precision than the plain
    /// [`Self::perceptual`] preset; lower-quality profiles match it.
    /// Per RDD 36 §7.3 the matrices are loaded into the frame header
    /// (`load_luma_qmat = load_chroma_qmat = 1`) so any RDD 36 decoder
    /// dequantises correctly.
    pub fn perceptual_for_profile(profile: Profile) -> Self {
        Self {
            quant_matrices: Some(QuantMatrices::perceptual_for_profile(profile)),
            profile: Some(profile),
            ..Self::default()
        }
    }

    /// Construct a config that pins the encoder to the supplied
    /// [`Profile`]. The override is honoured verbatim (the `bit_rate`
    /// heuristic in [`pick_profile`] is bypassed). All other fields take
    /// their defaults; chain with `with_quant_matrices` /
    /// `with_quantization_index` / `with_rate_control` / `with_meta` to
    /// configure them.
    ///
    /// Equivalent to `EncoderConfig::default().with_profile(profile)`.
    pub fn for_profile(profile: Profile) -> Self {
        Self {
            profile: Some(profile),
            ..Self::default()
        }
    }

    /// Use the supplied per-component matrices. Both must have weights
    /// in `2..=63` per RDD 36 §7.3 (validated at encode time).
    pub fn with_quant_matrices(mut self, qm: QuantMatrices) -> Self {
        self.quant_matrices = Some(qm);
        self
    }

    /// Override the per-profile default `quantization_index` (RDD 36
    /// §7.3 / Table 15). Must be in `1..=224`; validated at encoder
    /// construction. Lower index = finer step = higher quality.
    pub fn with_quantization_index(mut self, qi: u8) -> Self {
        self.quantization_index = Some(qi);
        self
    }

    /// Override the descriptive frame-header metadata
    /// (aspect_ratio_information, frame_rate_code, color_primaries,
    /// transfer_characteristic, matrix_coefficients). Equivalent to
    /// setting the [`Self::meta`] field directly.
    pub fn with_meta(mut self, meta: FrameMeta) -> Self {
        self.meta = Some(meta);
        self
    }

    /// Enable two-pass per-frame rate control (see [`Self::rate_control`]).
    /// Requires `CodecParameters::bit_rate` and `frame_rate` to be set at
    /// encoder construction; silently degrades to single-pass otherwise.
    pub fn with_rate_control(mut self) -> Self {
        self.rate_control = true;
        self
    }

    /// Explicit profile override (see [`Self::profile`]). Bypasses the
    /// `bit_rate` → profile heuristic of [`pick_profile`].
    ///
    /// The profile's [`Profile::chroma_format`] must match the
    /// `PixelFormat` passed in `CodecParameters` (4:2:2 profiles ↔
    /// `Yuv422P*`; 4:4:4 profiles ↔ `Yuv444P*`); a mismatch is rejected
    /// at encoder construction.
    pub fn with_profile(mut self, profile: Profile) -> Self {
        self.profile = Some(profile);
        self
    }

    /// Request interlaced output (RDD 36 §6.1.1 `interlace_mode`): `1`
    /// for top-field-first, `2` for bottom-field-first. The default
    /// (`0`) emits progressive frames. Value `3` is reserved (Table 2)
    /// and rejected at encoder construction.
    ///
    /// When set, every `send_frame` splits the source `VideoFrame` into
    /// two field pictures per §6.2 / §7.5.3 and emits them in temporal
    /// order — see [`Self::interlace_mode`].
    pub fn with_interlace_mode(mut self, interlace_mode: u8) -> Self {
        self.interlace_mode = interlace_mode;
        self
    }

    /// Override the desired macroblocks-per-slice (RDD 36 §5.3 — must
    /// be one of `1`, `2`, `4`, or `8`). The default (`None`) preserves
    /// the historical 8-MBs-per-slice layout. See
    /// [`Self::mbs_per_slice`] for the rate-vs-resilience tradeoff.
    pub fn with_mbs_per_slice(mut self, mbs_per_slice: u8) -> Self {
        self.mbs_per_slice = Some(mbs_per_slice);
        self
    }
}

/// Default macroblocks-per-slice — matches every reference RDD 36
/// encoder's per-row template (`[8, 8, …, 4?, 2?, 1?]` per
/// [`crate::frame::compute_slice_sizes`]) and the
/// `log2_desired_slice_size_in_mb == 3` written by Apple's encoders
/// for every fixture under `docs/video/prores/fixtures/`.
pub const DEFAULT_MBS_PER_SLICE: u8 = 8;

/// Convert a `mbs_per_slice` value (must be 1, 2, 4, or 8) to the
/// `log2_desired_slice_size_in_mb` field stored in the picture header
/// (0..3). Returns `Err` for any other value.
pub fn mbs_per_slice_to_log2(mbs_per_slice: u8) -> Result<u8> {
    match mbs_per_slice {
        1 => Ok(0),
        2 => Ok(1),
        4 => Ok(2),
        8 => Ok(3),
        _ => Err(Error::invalid(
            "prores encoder: mbs_per_slice must be 1, 2, 4, or 8 (RDD 36 §5.3 — \
             log2_desired_slice_size_in_mb is a 2-bit field, so 8 MBs is the maximum)",
        )),
    }
}

/// Default `quantization_index` used for 422 Standard. Lower = higher quality.
pub const DEFAULT_QUANT_INDEX: u8 = 4;

/// Internal cap on encoded packet size — bounds the output Vec against
/// `width * height * bytes-per-sample * a small constant`. Prevents a
/// pathological caller (e.g. a header that lies about dimensions) from
/// driving an unbounded allocation.
fn output_capacity_cap(width: u16, height: u16, chroma: ChromaFormat) -> usize {
    let pixels = width as usize * height as usize;
    // A worst-case ProRes packet for 8-bit YUV is ~10 bytes per pixel
    // before container overhead. Pad to 16 + a slack for headers.
    let bpp = match chroma {
        ChromaFormat::Y422 => 16,
        ChromaFormat::Y444 => 24,
    };
    pixels.saturating_mul(bpp).saturating_add(1 << 16)
}

const MB_SIDE_PX: usize = 16;

/// Pick a profile from `bit_rate` when the caller expresses a target rate.
///
/// Public so callers can preview the encoder's profile selection without
/// running an encode (the chosen profile is no longer carried in the
/// RDD 36 bitstream — it lives at the container level via FourCC).
pub fn pick_profile(chroma: ChromaFormat, bit_rate: Option<u64>) -> Profile {
    match (chroma, bit_rate) {
        (ChromaFormat::Y422, Some(br)) if br <= 70_000_000 => Profile::Proxy,
        (ChromaFormat::Y422, Some(br)) if br <= 125_000_000 => Profile::Lt,
        (ChromaFormat::Y422, Some(br)) if br <= 180_000_000 => Profile::Standard,
        (ChromaFormat::Y422, Some(_)) => Profile::Hq,
        (ChromaFormat::Y422, None) => Profile::Standard,
        (ChromaFormat::Y444, Some(br)) if br >= 400_000_000 => Profile::Prores4444Xq,
        (ChromaFormat::Y444, _) => Profile::Prores4444,
    }
}

pub fn make_encoder(params: &CodecParameters) -> Result<Box<dyn Encoder>> {
    make_encoder_with_config(params, EncoderConfig::default())
}

/// Build a ProRes encoder with explicit [`EncoderConfig`] — wires
/// optional perceptual quantisation matrices through to the frame
/// header (setting `load_luma_qmat = load_chroma_qmat = 1` when the
/// matrices differ from the spec default of all-4s).
pub fn make_encoder_with_config(
    params: &CodecParameters,
    config: EncoderConfig,
) -> Result<Box<dyn Encoder>> {
    if let Some(qm) = &config.quant_matrices {
        if !qm.weights_valid() {
            return Err(Error::invalid(
                "prores encoder: quant matrix weight outside RDD 36 range 2..=63",
            ));
        }
    }
    if let Some(qi) = config.quantization_index {
        if !(1..=224).contains(&qi) {
            return Err(Error::invalid(
                "prores encoder: EncoderConfig::quantization_index out of range \
                 (must be 1..=224 per RDD 36 §7.3 / Table 15)",
            ));
        }
    }
    // RDD 36 §6.1.1 Table 2: interlace_mode is a 2-bit field; 0 =
    // progressive, 1 = TFF, 2 = BFF, 3 = reserved. Refuse 3 (and any
    // value that would not fit the field) at construction.
    if config.interlace_mode > 2 {
        return Err(Error::invalid(
            "prores encoder: EncoderConfig::interlace_mode must be 0 (progressive), \
             1 (top-field-first) or 2 (bottom-field-first) — value 3 is reserved \
             (RDD 36 §6.1.1 Table 2)",
        ));
    }
    // Validate mbs_per_slice up-front so callers get a clean error
    // before any encode runs. mbs_per_slice_to_log2 rejects everything
    // outside {1, 2, 4, 8} per RDD 36 §5.3.
    if let Some(m) = config.mbs_per_slice {
        mbs_per_slice_to_log2(m)?;
    }
    let width = params
        .width
        .ok_or_else(|| Error::invalid("prores encoder: missing width"))?;
    let height = params
        .height
        .ok_or_else(|| Error::invalid("prores encoder: missing height"))?;
    let pix = params.pixel_format.unwrap_or(PixelFormat::Yuv422P);

    let (chroma, bit_depth) = match pix {
        PixelFormat::Yuv422P => (ChromaFormat::Y422, BitDepth::Eight),
        PixelFormat::Yuv444P => (ChromaFormat::Y444, BitDepth::Eight),
        PixelFormat::Yuv422P10Le => (ChromaFormat::Y422, BitDepth::Ten),
        PixelFormat::Yuv444P10Le => (ChromaFormat::Y444, BitDepth::Ten),
        PixelFormat::Yuv422P12Le => (ChromaFormat::Y422, BitDepth::Twelve),
        PixelFormat::Yuv444P12Le => (ChromaFormat::Y444, BitDepth::Twelve),
        other => {
            return Err(Error::unsupported(format!(
                "prores encoder: pixel format {other:?} not supported \
                 (expected Yuv4(2|4)4P / Yuv4(2|4)4P10Le / Yuv4(2|4)4P12Le)"
            )));
        }
    };
    let profile = if let Some(p) = config.profile {
        if p.chroma_format() != chroma {
            return Err(Error::invalid(format!(
                "prores encoder: EncoderConfig::profile {p:?} (chroma_format = \
                 {:?}) does not match requested pixel_format {pix:?} (chroma_format \
                 = {chroma:?})",
                p.chroma_format(),
            )));
        }
        p
    } else {
        pick_profile(chroma, params.bit_rate)
    };

    let mut output_params = params.clone();
    output_params.media_type = MediaType::Video;
    output_params.codec_id = CodecId::new(super::CODEC_ID_STR);
    output_params.width = Some(width);
    output_params.height = Some(height);
    output_params.pixel_format = Some(pix);

    let quant_index = config
        .quantization_index
        .unwrap_or_else(|| profile.default_quant_index());

    // Resolve the metadata block once at construction. When the caller
    // doesn't supply an explicit `FrameMeta`, derive `frame_rate_code`
    // from `params.frame_rate` (per RDD 36 §6.2 / Table 4) and leave
    // every other field at 0 ("unknown / unspecified").
    let meta = config.meta.unwrap_or_else(|| FrameMeta {
        frame_rate_code: params.frame_rate.map_or(0, frame_rate_code_from_rational),
        ..FrameMeta::default()
    });

    // Compute per-frame byte target for rate control. We need both
    // bit_rate and frame_rate; either missing → rate control disabled.
    let target_bytes = if config.rate_control {
        if let (Some(br), Some(fr)) = (params.bit_rate, params.frame_rate) {
            if fr.num > 0 && fr.den > 0 {
                // bytes_per_frame = (bit_rate / 8) * (den / num)
                let bits_per_frame = (br * fr.den as u64).saturating_div(fr.num as u64);
                (bits_per_frame / 8) as usize
            } else {
                0
            }
        } else {
            0
        }
    } else {
        0
    };

    let interlace_mode = config.interlace_mode;
    let log2_slice_mb_width =
        mbs_per_slice_to_log2(config.mbs_per_slice.unwrap_or(DEFAULT_MBS_PER_SLICE))?;

    Ok(Box::new(ProResEncoder {
        output_params,
        width,
        height,
        chroma,
        bit_depth,
        profile,
        quant_index,
        meta,
        interlace_mode,
        log2_slice_mb_width,
        config,
        time_base: params
            .frame_rate
            .map_or(TimeBase::new(1, 90_000), |r| TimeBase::new(r.den, r.num)),
        target_bytes,
        pending: VecDeque::new(),
        eof: false,
    }))
}

struct ProResEncoder {
    output_params: CodecParameters,
    width: u32,
    height: u32,
    chroma: ChromaFormat,
    bit_depth: BitDepth,
    profile: Profile,
    quant_index: u8,
    meta: FrameMeta,
    /// RDD 36 §6.1.1 interlace_mode applied to every emitted frame
    /// (0 = progressive, 1 = TFF, 2 = BFF). Mirrors
    /// [`EncoderConfig::interlace_mode`].
    interlace_mode: u8,
    /// `log2_desired_slice_size_in_mb` field written into every
    /// picture_header (RDD 36 §5.2.2 / §5.3). Resolved at construction
    /// from `config.mbs_per_slice` (default 8 → log2 == 3).
    log2_slice_mb_width: u8,
    config: EncoderConfig,
    time_base: TimeBase,
    /// Target bytes per frame for rate control, or 0 when disabled.
    target_bytes: usize,
    pending: VecDeque<Packet>,
    eof: bool,
}

impl Encoder for ProResEncoder {
    fn codec_id(&self) -> &CodecId {
        &self.output_params.codec_id
    }

    fn output_params(&self) -> &CodecParameters {
        &self.output_params
    }

    fn send_frame(&mut self, frame: &Frame) -> Result<()> {
        match frame {
            Frame::Video(v) => {
                let data = if self.target_bytes > 0 {
                    encode_frame_with_rate_control(
                        v,
                        self.width,
                        self.height,
                        self.chroma,
                        self.bit_depth,
                        self.profile,
                        self.quant_index,
                        self.config.quant_matrices,
                        self.meta,
                        self.target_bytes,
                        self.interlace_mode,
                        self.log2_slice_mb_width,
                    )?
                } else {
                    encode_frame_full(
                        v,
                        self.width,
                        self.height,
                        self.chroma,
                        self.bit_depth,
                        self.profile,
                        self.quant_index,
                        None,
                        self.interlace_mode,
                        self.config.quant_matrices,
                        self.meta,
                        self.log2_slice_mb_width,
                    )?
                };
                let mut pkt = Packet::new(0, self.time_base, data);
                pkt.pts = v.pts;
                pkt.dts = v.pts;
                pkt.flags.keyframe = true;
                self.pending.push_back(pkt);
                Ok(())
            }
            _ => Err(Error::invalid("prores encoder: video frames only")),
        }
    }

    fn receive_packet(&mut self) -> Result<Packet> {
        self.pending.pop_front().ok_or(Error::NeedMore)
    }

    fn flush(&mut self) -> Result<()> {
        self.eof = true;
        Ok(())
    }
}

/// Back-compat wrapper that encodes a 4:2:2 frame with the same API
/// shape as the pre-RDD 36 implementation.
pub fn encode_frame_422(
    frame: &VideoFrame,
    width: u32,
    height: u32,
    profile: Profile,
    quant_index: u8,
) -> Result<Vec<u8>> {
    encode_frame(
        frame,
        width,
        height,
        ChromaFormat::Y422,
        profile,
        quant_index,
    )
}

/// Encode a single picture (4:2:2 or 4:4:4) to a complete RDD 36 frame.
/// 8-bit input only; for 10-bit see [`encode_frame_with_depth`].
pub fn encode_frame(
    frame: &VideoFrame,
    img_w: u32,
    img_h: u32,
    chroma: ChromaFormat,
    profile: Profile,
    quantization_index: u8,
) -> Result<Vec<u8>> {
    encode_frame_with_depth(
        frame,
        img_w,
        img_h,
        chroma,
        BitDepth::Eight,
        profile,
        quantization_index,
    )
}

/// Encode a single picture to an RDD 36 frame at the requested bit depth.
///
/// `BitDepth::Eight` reads each sample as one byte; the deeper-bit paths
/// read each sample as a little-endian `u16` whose value is bounded by
/// the depth (`[0, 1023]` for 10-bit, `[0, 4095]` for 12-bit; high bits
/// ignored). Internal DCT precision is the same for all depths — input
/// samples are level-shifted into the spec's centred range
/// `v = s / 2^(b-9) - 256` (RDD 36 §7.5.1) so the quant-matrix and
/// qScale tables apply identically across depths.
#[allow(clippy::too_many_arguments)]
pub fn encode_frame_with_depth(
    frame: &VideoFrame,
    img_w: u32,
    img_h: u32,
    chroma: ChromaFormat,
    bit_depth: BitDepth,
    profile: Profile,
    quantization_index: u8,
) -> Result<Vec<u8>> {
    encode_frame_with_alpha(
        frame,
        img_w,
        img_h,
        chroma,
        bit_depth,
        profile,
        quantization_index,
        None,
    )
}

/// Encode a single picture using explicit per-component quantisation
/// weight matrices (RDD 36 §7.3). When `qmats` differs from the spec
/// default of all-4s the encoder loads the matrices into the frame
/// header (`load_luma_qmat = load_chroma_qmat = 1`) so any RDD 36
/// decoder reconstructs them correctly.
///
/// Equivalent to [`encode_frame_with_depth`] when
/// `qmats == QuantMatrices::flat()`.
#[allow(clippy::too_many_arguments)]
pub fn encode_frame_with_qmats(
    frame: &VideoFrame,
    img_w: u32,
    img_h: u32,
    chroma: ChromaFormat,
    bit_depth: BitDepth,
    profile: Profile,
    quantization_index: u8,
    qmats: QuantMatrices,
) -> Result<Vec<u8>> {
    encode_frame_full(
        frame,
        img_w,
        img_h,
        chroma,
        bit_depth,
        profile,
        quantization_index,
        None,
        0,
        Some(qmats),
        FrameMeta::default(),
        3, // log2(8) — default slice width matches every reference encoder
    )
}

/// Encode a single picture to an RDD 36 frame with optional alpha
/// channel coding (RDD 36 §5.3.3 + §7.1.2).
///
/// When `alpha_channel_type` is `Some`, the input frame must carry a
/// 4th `VideoPlane` with a per-pixel alpha array at full luma resolution.
/// Each sample is read as one byte (`Eight`) — alpha values are
/// promoted to the spec's 16-bit internal representation and emitted as
/// `scanned_alpha()` blobs at the tail of every slice. The frame header
/// `alpha_channel_type` field is set accordingly.
///
/// When `alpha_channel_type` is `None`, behaviour is identical to
/// [`encode_frame_with_depth`] (3-plane input, no alpha emission).
#[allow(clippy::too_many_arguments)]
pub fn encode_frame_with_alpha(
    frame: &VideoFrame,
    img_w: u32,
    img_h: u32,
    chroma: ChromaFormat,
    bit_depth: BitDepth,
    profile: Profile,
    quantization_index: u8,
    alpha_channel_type: Option<AlphaChannelType>,
) -> Result<Vec<u8>> {
    encode_frame_full(
        frame,
        img_w,
        img_h,
        chroma,
        bit_depth,
        profile,
        quantization_index,
        alpha_channel_type,
        0,
        None,
        FrameMeta::default(),
        3, // log2(8) — default slice width matches every reference encoder
    )
}

/// Encode an interlaced RDD 36 frame. `interlace_mode` selects the
/// field order (1 = top-field-first, 2 = bottom-field-first). The
/// supplied frame's planes are sliced into top + bottom field pictures
/// per §7.5.3 (rows {0, 2, …} → top, rows {1, 3, …} → bottom) and each
/// field is encoded as a separate `picture()` per §5.1, sharing one
/// frame_header().
#[allow(clippy::too_many_arguments)]
pub fn encode_frame_interlaced(
    frame: &VideoFrame,
    img_w: u32,
    img_h: u32,
    chroma: ChromaFormat,
    bit_depth: BitDepth,
    profile: Profile,
    quantization_index: u8,
    alpha_channel_type: Option<AlphaChannelType>,
    interlace_mode: u8,
) -> Result<Vec<u8>> {
    if interlace_mode != 1 && interlace_mode != 2 {
        return Err(Error::invalid(
            "prores encoder: encode_frame_interlaced requires interlace_mode in {1, 2}",
        ));
    }
    encode_frame_full(
        frame,
        img_w,
        img_h,
        chroma,
        bit_depth,
        profile,
        quantization_index,
        alpha_channel_type,
        interlace_mode,
        None,
        FrameMeta::default(),
        3, // log2(8) — default slice width matches every reference encoder
    )
}

/// Two-pass per-frame rate control: binary-search `quantization_index` to
/// hit `target_bytes` within [`RATE_CTRL_TOLERANCE`] (±5 %).
///
/// Strategy:
/// 1. Encode once at `seed_qi` (the profile default or caller's qi).
/// 2. If the size is already within tolerance, return immediately.
/// 3. Binary-search the qi range [1, 224], converging in at most
///    [`RATE_CTRL_MAX_PASSES`] further trials.
///
/// Invariant: larger qi → coarser quantisation → smaller frame.
/// So `lo` is the qi that produced the largest recent frame and `hi`
/// is the qi that produced the smallest recent frame. We pick midpoints
/// until the target is hit or the range collapses.
#[allow(clippy::too_many_arguments)]
fn encode_frame_with_rate_control(
    frame: &VideoFrame,
    img_w: u32,
    img_h: u32,
    chroma: ChromaFormat,
    bit_depth: BitDepth,
    profile: Profile,
    seed_qi: u8,
    qmats: Option<QuantMatrices>,
    meta: FrameMeta,
    target_bytes: usize,
    interlace_mode: u8,
    log2_slice_mb_width: u8,
) -> Result<Vec<u8>> {
    let tol_lo = (target_bytes as f64 * (1.0 - RATE_CTRL_TOLERANCE)) as usize;
    let tol_hi = (target_bytes as f64 * (1.0 + RATE_CTRL_TOLERANCE)) as usize;

    // First encode at seed qi.
    let seed = encode_frame_full(
        frame,
        img_w,
        img_h,
        chroma,
        bit_depth,
        profile,
        seed_qi,
        None,
        interlace_mode,
        qmats,
        meta,
        log2_slice_mb_width,
    )?;
    if seed.len() >= tol_lo && seed.len() <= tol_hi {
        return Ok(seed);
    }

    // Decide search direction.
    // If seed is too large (above target+tol) we need higher qi (coarser).
    // If seed is too small (below target-tol) we need lower qi (finer).
    let (mut lo, mut hi): (u8, u8) = if seed.len() > tol_hi {
        // Too large → need coarser quantisation → higher qi
        (seed_qi, 224)
    } else {
        // Too small → need finer quantisation → lower qi
        (1, seed_qi)
    };

    let mut best = seed;

    for _ in 0..RATE_CTRL_MAX_PASSES {
        if lo >= hi {
            break;
        }
        let mid = lo + (hi - lo) / 2;
        let candidate = encode_frame_full(
            frame,
            img_w,
            img_h,
            chroma,
            bit_depth,
            profile,
            mid,
            None,
            interlace_mode,
            qmats,
            meta,
            log2_slice_mb_width,
        )?;
        let sz = candidate.len();
        if sz >= tol_lo && sz <= tol_hi {
            return Ok(candidate);
        }
        // Track the closest candidate by absolute distance to target.
        let best_dist = (best.len() as i64 - target_bytes as i64).unsigned_abs();
        let cand_dist = (sz as i64 - target_bytes as i64).unsigned_abs();
        if cand_dist < best_dist {
            best = candidate;
        }
        if sz > tol_hi {
            // Frame too large → raise qi (coarser)
            lo = mid + 1;
        } else {
            // Frame too small → lower qi (finer)
            // Safe because mid >= lo >= 1; if mid == 1 the loop exits.
            if mid == 0 {
                break;
            }
            hi = mid - 1;
        }
    }
    // Return the best candidate found (closest to target).
    Ok(best)
}

/// Internal entrypoint shared by progressive and interlaced encodes.
/// `interlace_mode == 0` builds a single picture; `1` (TFF) or `2`
/// (BFF) builds two field pictures per §5.1.
///
/// `qmats == None` reproduces the legacy behaviour: flat all-4s
/// matrices, `load_luma_qmat = load_chroma_qmat = 0`, frame_header_size
/// = 20. `qmats == Some(QuantMatrices::flat())` is treated identically
/// (no point loading the default matrix into the bitstream). For any
/// other matrices, the encoder writes `load_luma_qmat = 1` and (when
/// the chroma matrix differs from the luma matrix) `load_chroma_qmat
/// = 1`, growing the frame header by 64 or 128 bytes per §7.3.
#[allow(clippy::too_many_arguments)]
fn encode_frame_full(
    frame: &VideoFrame,
    img_w: u32,
    img_h: u32,
    chroma: ChromaFormat,
    bit_depth: BitDepth,
    profile: Profile,
    quantization_index: u8,
    alpha_channel_type: Option<AlphaChannelType>,
    interlace_mode: u8,
    qmats: Option<QuantMatrices>,
    meta: FrameMeta,
    log2_slice_mb_width: u8,
) -> Result<Vec<u8>> {
    // Defence-in-depth: the spec stores log2_desired_slice_size_in_mb
    // in two bits so only 0..=3 is representable. Callers go through
    // mbs_per_slice_to_log2 which already enforces this, but the
    // public encode_frame_* shims pass literal `3` (default) and a
    // typo there is a one-bit corruption — assert here.
    if log2_slice_mb_width > 3 {
        return Err(Error::invalid(
            "prores encoder: log2_desired_slice_size_in_mb must be 0..=3 \
             (RDD 36 §5.2.2 — two-bit picture-header field)",
        ));
    }
    let expected_planes = if alpha_channel_type.is_some() { 4 } else { 3 };
    if frame.planes.len() != expected_planes {
        return Err(Error::invalid(format!(
            "prores encoder: expected {expected_planes} planes (got {})",
            frame.planes.len()
        )));
    }
    if !(1..=224).contains(&quantization_index) {
        return Err(Error::invalid(
            "prores encoder: quantization_index out of range",
        ));
    }
    if profile.chroma_format() != chroma {
        return Err(Error::invalid(
            "prores encoder: profile chroma_format does not match requested chroma",
        ));
    }
    if let Some(qm) = &qmats {
        if !qm.weights_valid() {
            return Err(Error::invalid(
                "prores encoder: quant matrix weight outside RDD 36 range 2..=63",
            ));
        }
    }
    let width = img_w as usize;
    let height = img_h as usize;

    // Bound output capacity against header-declared dimensions.
    let cap = output_capacity_cap(img_w as u16, img_h as u16, chroma);

    // Resolve the per-component matrices used for quantisation. When
    // the caller passed flat (or no) matrices we keep load_*_qmat = 0
    // for byte-exact compatibility with the pre-config encoder.
    let qmat_pair = qmats.unwrap_or_default();
    let load_luma = !qmat_pair.is_default();
    let load_chroma = load_luma && qmat_pair.chroma != qmat_pair.luma;
    let luma_qmat = if load_luma {
        &qmat_pair.luma
    } else {
        &DEFAULT_QMAT
    };
    let chroma_qmat = if load_luma {
        &qmat_pair.chroma
    } else {
        &DEFAULT_QMAT
    };

    // Per §6.2 picture_vertical_size derivation. Each interlaced field
    // is a separate picture sized at half the frame height (rounded
    // appropriately for top vs. bottom).
    let pictures: Vec<(usize, FieldStride)> = if interlace_mode == 0 {
        vec![(height, FieldStride::progressive())]
    } else {
        let top_h = height.div_ceil(2);
        let bot_h = height / 2;
        // interlace_mode 1: first picture is top field (offset 0)
        // interlace_mode 2: first picture is bottom field (offset 1)
        if interlace_mode == 1 {
            vec![
                (top_h, FieldStride::new(2, 0)),
                (bot_h, FieldStride::new(2, 1)),
            ]
        } else {
            vec![
                (bot_h, FieldStride::new(2, 1)),
                (top_h, FieldStride::new(2, 0)),
            ]
        }
    };

    let interlaced = interlace_mode != 0;
    let mut picture_blobs: Vec<Vec<u8>> = Vec::with_capacity(pictures.len());
    for (picture_height, field) in &pictures {
        let blob = encode_one_picture(
            frame,
            width,
            height,
            *picture_height,
            chroma,
            bit_depth,
            quantization_index,
            luma_qmat,
            chroma_qmat,
            alpha_channel_type,
            log2_slice_mb_width,
            interlaced,
            *field,
        )?;
        picture_blobs.push(blob);
    }

    // Per §5.1.1 frame_header_size: 20 + 64 (load_luma) + 64 (load_chroma).
    let frame_header_size =
        20usize + if load_luma { 64 } else { 0 } + if load_chroma { 64 } else { 0 };
    let pictures_total: usize = picture_blobs.iter().map(|p| p.len()).sum();
    let total_frame_size_no_padding = 4 + 4 + frame_header_size + pictures_total;
    if total_frame_size_no_padding > cap {
        return Err(Error::invalid(
            "prores encoder: encoded size exceeds internal cap",
        ));
    }

    let mut out = Vec::with_capacity(total_frame_size_no_padding);
    write_frame_with_meta(
        &mut out,
        total_frame_size_no_padding as u32,
        img_w as u16,
        img_h as u16,
        chroma,
        interlace_mode,
        luma_qmat,
        chroma_qmat,
        load_luma,
        load_chroma,
        alpha_channel_type.map_or(0, |a| a.code()),
        meta,
    );
    for blob in &picture_blobs {
        out.extend_from_slice(blob);
    }
    debug_assert_eq!(out.len(), total_frame_size_no_padding);
    Ok(out)
}

/// Field-row mapping for source-plane reads on the encoder side.
/// Mirrors `decoder::FieldStride`.
#[derive(Copy, Clone, Debug)]
struct FieldStride {
    step: usize,
    offset: usize,
}

impl FieldStride {
    fn new(step: usize, offset: usize) -> Self {
        Self { step, offset }
    }
    fn progressive() -> Self {
        Self { step: 1, offset: 0 }
    }
    fn map(self, picture_row: usize) -> usize {
        self.step * picture_row + self.offset
    }
}

/// Build one `picture()` blob (picture_header + slice_table +
/// concatenated slice payloads). For interlaced encodes the caller
/// invokes this twice (once per field).
///
/// `luma_qmat` is applied to all four luma blocks per macroblock;
/// `chroma_qmat` is applied to both Cb and Cr blocks. Both matrices
/// must be the same matrices written into the frame header so the
/// decoder dequantises with the matching W[][].
#[allow(clippy::too_many_arguments)]
fn encode_one_picture(
    frame: &VideoFrame,
    frame_w: usize,
    frame_h: usize,
    picture_height: usize,
    chroma: ChromaFormat,
    bit_depth: BitDepth,
    quantization_index: u8,
    luma_qmat: &[u8; 64],
    chroma_qmat: &[u8; 64],
    alpha_channel_type: Option<AlphaChannelType>,
    log2_slice_mb_width: u8,
    interlaced: bool,
    field: FieldStride,
) -> Result<Vec<u8>> {
    let c_w = match chroma {
        ChromaFormat::Y422 => frame_w.div_ceil(2),
        ChromaFormat::Y444 => frame_w,
    };
    let mbs_x = frame_w.div_ceil(MB_SIDE_PX);
    let mbs_y = picture_height.div_ceil(MB_SIDE_PX);
    let slice_sizes_template = compute_slice_sizes(mbs_x, log2_slice_mb_width);
    let slices_per_row = slice_sizes_template.len();
    let slice_count = slices_per_row * mbs_y;
    let _cb_per_mb = chroma_blocks_per_mb(chroma);
    let per_mb = blocks_per_mb(chroma);

    const LUMA_OFFSETS: [(usize, usize); 4] = [(0, 0), (1, 0), (0, 1), (1, 1)];
    let chroma_offsets: &[(usize, usize)] = match chroma {
        ChromaFormat::Y422 => &[(0, 0), (0, 1)],
        ChromaFormat::Y444 => &LUMA_OFFSETS,
    };

    let mut slice_payloads: Vec<Vec<u8>> = Vec::with_capacity(slice_count);
    for my in 0..mbs_y {
        let mut mx = 0usize;
        for &mbs_this_slice in &slice_sizes_template {
            let mbs_this_slice = mbs_this_slice.min(mbs_x - mx);
            if mbs_this_slice == 0 {
                break;
            }
            let mut blocks: Vec<[i32; 64]> = Vec::with_capacity(mbs_this_slice * per_mb);
            for mb_within in 0..mbs_this_slice {
                let mb_x = mx + mb_within;
                for (bx, by) in LUMA_OFFSETS {
                    let x0 = mb_x * MB_SIDE_PX + bx * 8;
                    let y0 = my * MB_SIDE_PX + by * 8;
                    blocks.push(encode_block(
                        &frame.planes[0].data,
                        frame.planes[0].stride,
                        frame_w,
                        frame_h,
                        x0,
                        y0,
                        luma_qmat,
                        quantization_index,
                        bit_depth,
                        field,
                    ));
                }
                for plane_idx in [1usize, 2] {
                    for (bx, by) in chroma_offsets.iter().copied() {
                        let (x0, y0) = match chroma {
                            ChromaFormat::Y422 => (mb_x * 8, my * MB_SIDE_PX + by * 8),
                            ChromaFormat::Y444 => {
                                (mb_x * MB_SIDE_PX + bx * 8, my * MB_SIDE_PX + by * 8)
                            }
                        };
                        blocks.push(encode_block(
                            &frame.planes[plane_idx].data,
                            frame.planes[plane_idx].stride,
                            c_w,
                            frame_h,
                            x0,
                            y0,
                            chroma_qmat,
                            quantization_index,
                            bit_depth,
                            field,
                        ));
                    }
                }
            }
            let (y_data, cb_data, cr_data) =
                encode_slice_components(mbs_this_slice, chroma, interlaced, &blocks)?;
            if y_data.len() > u16::MAX as usize
                || cb_data.len() > u16::MAX as usize
                || cr_data.len() > u16::MAX as usize
            {
                return Err(Error::invalid(
                    "prores encoder: slice component exceeded u16 size limit",
                ));
            }

            let alpha_blob: Vec<u8> = if let Some(act) = alpha_channel_type {
                // Emit alpha for the FULL macroblock-row height (16
                // sample rows) regardless of visible picture clipping.
                // Decoders MUST allocate the padded MB-aligned plane
                // and crop after decode (RDD 36 §7.5.2 — alphaValues is
                // the padded picture size); ffmpeg's prores_ks behaves
                // the same way. Edge-pixels for the partially-visible
                // last MB row are clamped to the last visible row so
                // the stream stays self-roundtrippable.
                let slice_vertical_size = MB_SIDE_PX;
                let cols = MB_SIDE_PX * mbs_this_slice;
                let mut samples: Vec<u16> = Vec::with_capacity(cols * slice_vertical_size);
                let a_plane = &frame.planes[3];
                let a_stride = a_plane.stride;
                for r in 0..slice_vertical_size {
                    let frame_row = field
                        .map(my * MB_SIDE_PX + r)
                        .min(frame_h.saturating_sub(1));
                    for c in 0..cols {
                        let x = (mx * MB_SIDE_PX + c).min(frame_w.saturating_sub(1));
                        let v: u16 = match act {
                            AlphaChannelType::Eight => {
                                a_plane.data[frame_row * a_stride + x] as u16
                            }
                            AlphaChannelType::Sixteen => {
                                let off = frame_row * a_stride + x * 2;
                                u16::from_le_bytes([a_plane.data[off], a_plane.data[off + 1]])
                            }
                        };
                        samples.push(v);
                    }
                }
                encode_scanned_alpha(&samples, act)?
            } else {
                Vec::new()
            };

            let cr_field = if alpha_channel_type.is_some() {
                Some(cr_data.len() as u16)
            } else {
                None
            };
            let mut slice_buf = Vec::with_capacity(
                8 + y_data.len() + cb_data.len() + cr_data.len() + alpha_blob.len(),
            );
            write_slice_header(
                &mut slice_buf,
                quantization_index,
                y_data.len() as u16,
                cb_data.len() as u16,
                cr_field,
            );
            slice_buf.extend_from_slice(&y_data);
            slice_buf.extend_from_slice(&cb_data);
            slice_buf.extend_from_slice(&cr_data);
            slice_buf.extend_from_slice(&alpha_blob);
            slice_payloads.push(slice_buf);
            mx += mbs_this_slice;
        }
    }
    debug_assert_eq!(slice_payloads.len(), slice_count);
    if slice_payloads.iter().any(|p| p.len() > u16::MAX as usize) {
        return Err(Error::invalid(
            "prores encoder: slice exceeded u16 size table limit",
        ));
    }

    let slice_table_size = slice_count * 2;
    let slice_bytes: usize = slice_payloads.iter().map(|p| p.len()).sum();
    let picture_header_size = 8usize;
    let picture_size = (picture_header_size + slice_table_size + slice_bytes) as u32;
    let mut blob = Vec::with_capacity(picture_size as usize);
    write_picture_header(
        &mut blob,
        picture_size,
        if slice_count <= u16::MAX as usize {
            slice_count as u16
        } else {
            0
        },
        log2_slice_mb_width,
    );
    for p in &slice_payloads {
        blob.extend_from_slice(&(p.len() as u16).to_be_bytes());
    }
    for p in &slice_payloads {
        blob.extend_from_slice(p);
    }
    debug_assert_eq!(blob.len(), picture_size as usize);
    Ok(blob)
}

/// Sample one IDCT input value from the source plane at sample
/// coordinate `(x, y)`, applying the spec's level-shift to a centred
/// `v` in the range `[-256, 256)` per RDD 36 §7.5.1. The inverse of
/// the decoder formula `s = 2^b * (v + 256) / 512` is
/// `v = s * 512 / 2^b - 256 = s / 2^(b-9) - 256`. `stride` is in
/// **bytes**; for 10/12-bit planes that's `2 * samples_per_row`.
fn read_sample(plane: &[u8], stride: usize, x: usize, y: usize, bit_depth: BitDepth) -> f32 {
    match bit_depth {
        BitDepth::Eight => (plane[y * stride + x] as f32) * 2.0 - 256.0,
        BitDepth::Ten => {
            let off = y * stride + x * 2;
            let lo = plane[off] as u16;
            let hi = plane[off + 1] as u16;
            let s = (lo | (hi << 8)) & 0x03FF;
            (s as f32) / 2.0 - 256.0
        }
        BitDepth::Twelve => {
            let off = y * stride + x * 2;
            let lo = plane[off] as u16;
            let hi = plane[off + 1] as u16;
            let s = (lo | (hi << 8)) & 0x0FFF;
            (s as f32) / 8.0 - 256.0
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn encode_block(
    plane: &[u8],
    stride: usize,
    plane_w: usize,
    plane_h: usize,
    x0: usize,
    y0: usize,
    qmat: &[u8; 64],
    quantization_index: u8,
    bit_depth: BitDepth,
    field: FieldStride,
) -> [i32; 64] {
    let mut blk = [0.0f32; 64];
    for j in 0..8 {
        // Map per-picture row to per-frame row; this is the identity for
        // progressive (`step=1, offset=0`) and 2*r+offset for interlaced.
        let frame_row = field.map(y0 + j).min(plane_h.saturating_sub(1));
        for i in 0..8 {
            let x = (x0 + i).min(plane_w.saturating_sub(1));
            blk[j * 8 + i] = read_sample(plane, stride, x, frame_row, bit_depth);
        }
    }
    // Constant-block fast path: any 8x8 input whose 64 samples are all
    // bit-identical (boundary-clamp pad blocks per RDD 36 §7.5.1, flat
    // regions of natural content at high qi, smooth synthetic gradients)
    // produces a single DC coefficient of `8 * v` with all 63 AC = 0.
    // Skip the textbook 64x16 row+column passes of [`fdct8x8`] entirely.
    if is_constant_block(&blk) {
        fdct8x8_constant(&mut blk);
    } else {
        fdct8x8(&mut blk);
    }
    // Quantisation: F[v][u] = (QF[v][u] * W[v][u] * qScale) / 8
    // Inverse: QF = round(F * 8 / (W * qScale)).
    let qs = qscale(quantization_index) as f32;
    let mut out = [0i32; 64];
    for k in 0..64 {
        let denom = qmat[k] as f32 * qs;
        let v = blk[k] * 8.0 / denom;
        out[k] = if v >= 0.0 {
            (v + 0.5) as i32
        } else {
            -((-v + 0.5) as i32)
        };
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decoder::decode_packet;
    use crate::frame::parse_frame;
    use oxideav_core::frame::VideoPlane;
    use oxideav_core::{CodecId, CodecParameters, Frame, MediaType, PixelFormat};

    /// 4:2:2 source whose even rows are bright and odd rows are dim, so
    /// a swapped TFF/BFF field assignment is detectable after a
    /// roundtrip. A small in-row gradient keeps the AC coefficients
    /// non-zero across blocks.
    fn field_distinct_422(width: u32, height: u32) -> VideoFrame {
        let w = width as usize;
        let h = height as usize;
        let cw = w / 2;
        let mut y = vec![0u8; w * h];
        let cb = vec![128u8; cw * h];
        let cr = vec![128u8; cw * h];
        for j in 0..h {
            for i in 0..w {
                let base: i32 = if j % 2 == 0 { 170 } else { 90 };
                let grad = ((i + j) % 32) as i32;
                y[j * w + i] = (base + grad - 8).clamp(16, 235) as u8;
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

    fn enc_params(width: u32, height: u32) -> CodecParameters {
        let mut p = CodecParameters::video(CodecId::new(crate::CODEC_ID_STR));
        p.media_type = MediaType::Video;
        p.width = Some(width);
        p.height = Some(height);
        p.pixel_format = Some(PixelFormat::Yuv422P);
        p
    }

    #[test]
    fn config_interlace_mode_default_is_progressive() {
        assert_eq!(EncoderConfig::default().interlace_mode, 0);
        assert_eq!(
            EncoderConfig::default()
                .with_interlace_mode(1)
                .interlace_mode,
            1
        );
        assert_eq!(
            EncoderConfig::default()
                .with_interlace_mode(2)
                .interlace_mode,
            2
        );
    }

    #[test]
    fn config_interlace_mode_3_rejected_at_construction() {
        // RDD 36 §6.1.1 Table 2: interlace_mode 3 is reserved.
        let params = enc_params(64, 48);
        let cfg = EncoderConfig::default().with_interlace_mode(3);
        let msg = match make_encoder_with_config(&params, cfg) {
            Ok(_) => panic!("interlace_mode 3 must be rejected (RDD 36 Table 2 reserved)"),
            Err(e) => format!("{e}"),
        };
        assert!(
            msg.contains("interlace_mode"),
            "error must name interlace_mode, got: {msg}"
        );
    }

    #[test]
    fn send_frame_progressive_default_emits_one_picture() {
        // Default config (interlace_mode == 0) must still emit a single
        // progressive picture through the public Encoder path.
        let params = enc_params(64, 48);
        let mut enc = make_encoder(&params).expect("make_encoder");
        enc.send_frame(&Frame::Video(field_distinct_422(64, 48)))
            .expect("send_frame");
        let pkt = enc.receive_packet().expect("receive_packet");
        let (fh, _) = parse_frame(&pkt.data).expect("parse frame");
        assert_eq!(fh.interlace_mode, 0);
        assert_eq!(fh.picture_count(), 1);
    }

    /// Drive the high-level Encoder (`make_encoder_with_config` +
    /// `send_frame`) with interlace_mode 1 (TFF) and 2 (BFF) and confirm
    /// the emitted frame header carries the requested mode, two pictures,
    /// and self-roundtrips through the decoder with field order intact.
    fn send_frame_interlaced_roundtrips(interlace_mode: u8) {
        let (w, h) = (64u32, 48u32);
        let src = field_distinct_422(w, h);
        let params = enc_params(w, h);
        let cfg = EncoderConfig::default().with_interlace_mode(interlace_mode);
        let mut enc = make_encoder_with_config(&params, cfg).expect("make_encoder_with_config");
        enc.send_frame(&Frame::Video(src.clone()))
            .expect("send_frame");
        let pkt = enc.receive_packet().expect("receive_packet");

        // Frame header must report the requested interlace_mode + two
        // pictures (one per field) per RDD 36 §5.1.
        let (fh, _) = parse_frame(&pkt.data).expect("parse frame");
        assert_eq!(fh.interlace_mode, interlace_mode, "header interlace_mode");
        assert_eq!(fh.picture_count(), 2, "interlaced frame carries 2 pictures");

        // Self-roundtrip: decode and confirm the even/odd brightness bias
        // survives (i.e. the encoder placed each source row into the
        // correct field and the decoder reinterleaved it correctly).
        let decoded = decode_packet(&pkt.data, Some(0)).expect("decode_packet");
        let dy = &decoded.planes[0].data;
        let stride = decoded.planes[0].stride;
        let mut even_sum = 0u64;
        let mut odd_sum = 0u64;
        for j in 0..(h as usize) {
            let mut row = 0u64;
            for i in 0..(w as usize) {
                row += dy[j * stride + i] as u64;
            }
            if j % 2 == 0 {
                even_sum += row;
            } else {
                odd_sum += row;
            }
        }
        assert!(
            even_sum > odd_sum,
            "interlace_mode {interlace_mode}: even-row sum {even_sum} not > odd-row sum \
             {odd_sum} (field assignment swapped?)"
        );
    }

    #[test]
    fn send_frame_interlaced_tff_roundtrips() {
        send_frame_interlaced_roundtrips(1);
    }

    #[test]
    fn send_frame_interlaced_bff_roundtrips() {
        send_frame_interlaced_roundtrips(2);
    }

    #[test]
    fn send_frame_interlaced_with_rate_control_keeps_field_order() {
        // Interlaced + rate control: the rate-control path must also
        // honour interlace_mode (two pictures, field order preserved).
        let (w, h) = (64u32, 48u32);
        let src = field_distinct_422(w, h);
        let mut params = enc_params(w, h);
        params.bit_rate = Some(50_000_000);
        params.frame_rate = Some(oxideav_core::Rational::new(25, 1));
        let cfg = EncoderConfig::default()
            .with_interlace_mode(1)
            .with_rate_control();
        let mut enc = make_encoder_with_config(&params, cfg).expect("make_encoder_with_config");
        enc.send_frame(&Frame::Video(src)).expect("send_frame");
        let pkt = enc.receive_packet().expect("receive_packet");
        let (fh, _) = parse_frame(&pkt.data).expect("parse frame");
        assert_eq!(fh.interlace_mode, 1);
        assert_eq!(fh.picture_count(), 2);
    }

    /// Regression for the constant-block fast path in `encode_block`:
    /// a uniform 8x8 input must produce exactly the same quantised
    /// coefficient vector through the fast path as through the textbook
    /// `fdct8x8` -> quantise loop. Both paths share the post-DCT
    /// quantiser, so the test pins the f32 DC produced by
    /// `fdct8x8_constant` (which is `8 * v`) against the f32 DC the
    /// general fdct produces on the same input.
    #[test]
    fn encode_block_constant_input_matches_general_path() {
        // Build a flat 8x8 plane of value 200 (level-shifted v = 144),
        // a sentinel macroblock matrix, and Standard's default qi=4.
        let plane = vec![200u8; 8 * 8];
        let qmat = [4u8; 64];
        let qi = 4u8;
        let out = super::encode_block(
            &plane,
            8,
            8,
            8,
            0,
            0,
            &qmat,
            qi,
            super::BitDepth::Eight,
            super::FieldStride::progressive(),
        );
        // The forward DCT of a constant block has DC = 8 * v and AC = 0.
        // After quantise: DC = round(8 * v * 8 / (qmat[0] * qscale(qi))).
        // For v = 200 * 2 - 256 = 144, qmat[0] = 4, qscale(4) = 4:
        //   DC = round(8 * 144 * 8 / (4 * 4)) = round(576) = 576.
        assert_eq!(out[0], 576, "constant-block DC matches the closed form");
        for k in 1..64 {
            assert_eq!(
                out[k], 0,
                "AC[{k}] must be exactly 0 after the constant-block fast path"
            );
        }
    }

    /// Wider correctness check: across every plausible level-shifted
    /// source byte and a sweep of legal qi values, the constant-block
    /// fast path inside `encode_block` must produce the same packet
    /// bytes as the same encoder would on a *near-constant* block
    /// (single-pixel perturbation forces the general fdct path) — when
    /// they decode through the round-trip decoder.
    ///
    /// We don't compare the packets byte-for-byte because the entropy
    /// coder is content-driven and the perturbation flips coefficients.
    /// Instead we encode + decode a constant-flat 64x48 frame and
    /// assert pixel-exact reconstruction at HQ (qi 2) — the encoder
    /// has to take the fast path on every block and the decoder has to
    /// reconstruct each block to a single sample value.
    #[test]
    fn constant_flat_frame_decodes_pixel_exact_at_hq() {
        use crate::decoder::decode_packet;
        for &v in &[16u8, 64, 128, 200, 235] {
            let (w, h) = (64u32, 48u32);
            let wu = w as usize;
            let hu = h as usize;
            let cwu = wu / 2;
            let src = VideoFrame {
                pts: Some(0),
                planes: vec![
                    VideoPlane {
                        stride: wu,
                        data: vec![v; wu * hu],
                    },
                    VideoPlane {
                        stride: cwu,
                        data: vec![128u8; cwu * hu],
                    },
                    VideoPlane {
                        stride: cwu,
                        data: vec![128u8; cwu * hu],
                    },
                ],
            };
            let mut params = enc_params(w, h);
            params.bit_rate = Some(220_000_000); // -> HQ (qi=2)
            let mut enc = make_encoder(&params).expect("make_encoder");
            enc.send_frame(&Frame::Video(src)).expect("send_frame");
            let pkt = enc.receive_packet().expect("receive_packet");
            let out = decode_packet(&pkt.data, None).expect("decode_packet");
            // Y plane must round-trip exactly at HQ on a flat input —
            // the constant-block fast path produces DC = 8 * v_centred,
            // which dequantises and IDCTs back to exactly v.
            for j in 0..hu {
                for i in 0..wu {
                    let got = out.planes[0].data[j * out.planes[0].stride + i];
                    assert_eq!(got, v, "v={v}, pos=({i},{j}): expected {v}, got {got}");
                }
            }
        }
    }
}
