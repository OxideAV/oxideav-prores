//! ProRes decoder following SMPTE RDD 36 §5 + §7.
//!
//! Glues the frame/picture/slice parsers and the entropy coder to the
//! IDCT and pixel emitter, producing a `VideoFrame` of `Yuv422P` (4:2:2)
//! or `Yuv444P` (4:4:4) depending on `frame_header.chroma_format`.
//!
//! ### Bit-depth selection
//!
//! RDD 36 §5 frame_header / picture_header carry NO bit-depth syntax
//! element — the spec leaves `b` (the per-sample bit depth in §7.5.1) up
//! to the decoder. The CodecParameters that built this decoder pin the
//! output format:
//!
//! * `PixelFormat::Yuv422P` / `Yuv444P` (or none) → 8-bit planes
//!   (single-byte samples in `[0, 255]`).
//! * `PixelFormat::Yuv422P10Le` / `Yuv444P10Le` → 10-bit planes
//!   packed as little-endian u16 pairs in `[0, 1023]`.
//! * `PixelFormat::Yuv422P12Le` / `Yuv444P12Le` → 12-bit planes
//!   packed as little-endian u16 pairs in `[0, 4095]`.
//! * `PixelFormat::Yuv422P16Le` / `Yuv444P16Le` → 16-bit planes
//!   packed as little-endian u16 pairs in `[0, 65535]` (§7.5.1 with
//!   `b = 16`; all 16 bits of every word are significant).
//!
//! The `chroma_format` of the requested pixel format must agree with the
//! frame header's `chroma_format`; mismatches return `Error::invalid`.
//!
//! ### Alpha plane (RDD 36 §7.1.2 + §7.5.2)
//!
//! When the frame header's `alpha_channel_type != 0`, every slice carries
//! a per-pixel raster-scanned alpha array decoded via [`crate::alpha`] and
//! converted to the output bit depth per §7.5.2. The resulting alpha
//! plane is appended as the **fourth** entry of `VideoFrame::planes`, in
//! the same per-sample format as the chroma planes (8-bit byte stride for
//! 8-bit output, 16-bit LE for 10/12-bit).
//!
//! Two request shapes expose it:
//!
//! * **Alpha-typed surface** — `PixelFormat::Yuva422P` / `Yuva444P`
//!   (8-bit) and the deep `Yuva4(2|4)4P{10,12,16}Le` formats request a
//!   four-plane frame whose layout is *guaranteed* by the format:
//!   decoded alpha rides plane 3 converted to the surface depth per
//!   §7.5.2, and a stream that carries no coded alpha gets a
//!   synthesised fully-opaque plane so the declared plane count always
//!   holds. On the `*P16Le` surfaces a 16-bit coded alpha channel
//!   (`alpha_channel_type == 2`) passes through **exactly** — the
//!   §7.5.2 conversion at `b = 16` is the identity — so the full
//!   RDD 36 alpha fidelity is expressible in the type system.
//! * **As-coded surface** — the plane-count convention: `Yuv4(2|4)4P*`
//!   requests append the 4th plane only when the stream codes alpha;
//!   callers detect it by checking `frame.planes.len() == 4`. The 4th
//!   plane is converted to the same §7.5.2 surface depth as the colour
//!   planes, so its significant depth never differs from what the
//!   requested `PixelFormat` already declares (which is why no
//!   per-plane significant-bits side-channel is attached to decoded
//!   frames — it would restate the pixel format).

use oxideav_core::Decoder;
use oxideav_core::frame::VideoPlane;
use oxideav_core::{
    CodecId, CodecParameters, Error, ExecutionContext, Frame, Packet, PixelFormat, Result,
    VideoFrame,
};

use crate::alpha::{AlphaChannelType, decode_scanned_alpha_into};
use crate::dct::idct8x8_rows;
use crate::entropy::{AC_PRESENT, ComponentLayout, decode_component_blocks};
use crate::frame::{
    ChromaFormat, compute_slice_sizes, parse_frame, parse_picture_header, parse_slice_header,
};
use crate::quant::{INTERLACED_INV_SCAN, PROGRESSIVE_INV_SCAN, qscale};
use crate::slice::{
    LUMA_BLOCKS_PER_MB, blocks_per_mb, chroma_blocks_per_mb, decode_slice_components,
};

const MB_SIDE_PX: usize = 16;

/// Internal cap on a single decoded frame's plane allocation. Bounds
/// against header-declared dimensions: a malicious header with
/// 1M×1M pixels must NOT trigger a multi-TB allocation. 32k×32k YUV444
/// (3 GB) is far more than any sane container; we cap at that.
const MAX_DECODED_PIXELS: usize = 32_768 * 32_768;

/// Per-component output bit depth.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum BitDepth {
    /// 8-bit planar output (`Yuv422P` / `Yuv444P`). One byte per sample.
    Eight,
    /// 10-bit planar output (`Yuv422P10Le` / `Yuv444P10Le`). Two bytes
    /// per sample, low-byte-first; valid range `0..=1023`.
    Ten,
    /// 12-bit planar output (`Yuv422P12Le` / `Yuv444P12Le`). Two bytes
    /// per sample, low-byte-first; valid range `0..=4095`.
    Twelve,
    /// 16-bit planar output (`Yuv422P16Le` / `Yuv444P16Le`). Two bytes
    /// per sample, low-byte-first; valid range `0..=65535` — all 16
    /// bits of every word are significant (§7.5.1 with `b = 16`).
    Sixteen,
}

impl BitDepth {
    pub fn bytes_per_sample(self) -> usize {
        match self {
            Self::Eight => 1,
            Self::Ten | Self::Twelve | Self::Sixteen => 2,
        }
    }

    pub fn max_value(self) -> u32 {
        match self {
            Self::Eight => 255,
            Self::Ten => 1023,
            Self::Twelve => 4095,
            Self::Sixteen => 65535,
        }
    }

    /// Bit depth as the integer `b` from RDD 36 §7.5.1.
    pub fn bits(self) -> u32 {
        match self {
            Self::Eight => 8,
            Self::Ten => 10,
            Self::Twelve => 12,
            Self::Sixteen => 16,
        }
    }
}

/// Selects the clamping bounds `(nmin, nmax)` of RDD 36 §7.5.1 for the
/// reconstructed-value → pixel-sample conversion
/// `s = clamp(round(2^b * (v + 256) / 512))`.
///
/// §7.5.1 offers two choices for the clamp limits:
///
/// - **Full** — `nmin = 0`, `nmax = 2^b − 1`; "produce pixel component
///   samples that utilize all available quantization levels". This is
///   the default and matches the planar ranges named in [`BitDepth`]
///   (`0..=255` / `0..=1023` / `0..=4095`).
/// - **Video** — `nmin`/`nmax` set to "the smallest and largest
///   permissible video quantization levels for b-bit samples if
///   avoidance of ITU-R BT.601/BT.709 synchronization/timing reference
///   quantization levels is desired". Those reserved timing-reference
///   levels are the extreme codes `0` and `2^b − 1` (the SAV/EAV
///   excursions of the BT.601/BT.709 digital interface), so the
///   permissible video levels span `1 ..= 2^b − 2`.
///
/// Both options round identically; they differ only in whether the two
/// extreme codes may appear in the output. The choice is purely an
/// output-formatting decision and does not affect entropy decoding,
/// inverse quantisation, or the IDCT.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Default)]
#[non_exhaustive]
pub enum OutputRange {
    /// Clamp to `0 ..= 2^b − 1` (all quantization levels). Default.
    #[default]
    Full,
    /// Clamp to `1 ..= 2^b − 2`, avoiding the BT.601/BT.709
    /// synchronization/timing reference quantization levels.
    Video,
}

impl OutputRange {
    /// The `(nmin, nmax)` clamp bounds of RDD 36 §7.5.1 for bit depth
    /// `bd`.
    pub fn bounds(self, bd: BitDepth) -> (u32, u32) {
        let max = bd.max_value();
        match self {
            Self::Full => (0, max),
            // 2^b − 2 is `max - 1`; `max` is always ≥ 255 so this never
            // underflows.
            Self::Video => (1, max - 1),
        }
    }
}

/// Fully resolved output surface for one decode: bit depth, chroma
/// format, and whether the caller's requested pixel format *guarantees*
/// a 4th alpha plane (`Yuva422P` / `Yuva444P`).
type OutputSurface = (BitDepth, ChromaFormat, bool);

/// Resolve a caller-requested `PixelFormat` to the decoder's output
/// surface. The third element is `true` for the alpha-typed formats
/// (`Yuva422P` / `Yuva444P`), whose plane layout promises alpha at plane
/// index 3 regardless of what the stream carries — see
/// [`decode_packet_with_format`].
fn resolve_pixel_format(pf: PixelFormat) -> Result<OutputSurface> {
    Ok(match pf {
        PixelFormat::Yuv422P => (BitDepth::Eight, ChromaFormat::Y422, false),
        PixelFormat::Yuv444P => (BitDepth::Eight, ChromaFormat::Y444, false),
        PixelFormat::Yuv422P10Le => (BitDepth::Ten, ChromaFormat::Y422, false),
        PixelFormat::Yuv444P10Le => (BitDepth::Ten, ChromaFormat::Y444, false),
        PixelFormat::Yuv422P12Le => (BitDepth::Twelve, ChromaFormat::Y422, false),
        PixelFormat::Yuv444P12Le => (BitDepth::Twelve, ChromaFormat::Y444, false),
        // 16-bit surfaces (§7.5.1 with b = 16): every sample word is
        // fully significant. A stream's 16-bit coded alpha reaches the
        // as-coded 4th plane exactly (the §7.5.2 conversion at b = 16
        // is the identity).
        PixelFormat::Yuv422P16Le => (BitDepth::Sixteen, ChromaFormat::Y422, false),
        PixelFormat::Yuv444P16Le => (BitDepth::Sixteen, ChromaFormat::Y444, false),
        // Alpha-typed 8-bit surfaces: full-resolution 8-bit alpha as
        // plane 3. RDD 36 alpha up to 16-bit demotes per §7.5.2, matching
        // how the Y/Cb/Cr planes demote per §7.5.1 at 8-bit output.
        PixelFormat::Yuva422P => (BitDepth::Eight, ChromaFormat::Y422, true),
        PixelFormat::Yuva444P => (BitDepth::Eight, ChromaFormat::Y444, true),
        // Alpha-typed deep surfaces: 16-bit-word planes with the low
        // 10/12 bits significant (or all 16 for the *P16Le pair), and
        // full-resolution alpha at the same depth as plane 3, converted
        // per §7.5.2.
        PixelFormat::Yuva422P10Le => (BitDepth::Ten, ChromaFormat::Y422, true),
        PixelFormat::Yuva444P10Le => (BitDepth::Ten, ChromaFormat::Y444, true),
        PixelFormat::Yuva422P12Le => (BitDepth::Twelve, ChromaFormat::Y422, true),
        PixelFormat::Yuva444P12Le => (BitDepth::Twelve, ChromaFormat::Y444, true),
        PixelFormat::Yuva422P16Le => (BitDepth::Sixteen, ChromaFormat::Y422, true),
        PixelFormat::Yuva444P16Le => (BitDepth::Sixteen, ChromaFormat::Y444, true),
        other => {
            return Err(Error::unsupported(format!(
                "prores decoder: requested pixel_format {other:?} not supported \
                 (expected Yuv4(2|4)4P / Yuva4(2|4)4P, plain or with a 10Le/12Le/16Le \
                 depth suffix)"
            )));
        }
    })
}

/// Resolve the requested output bit depth + chroma format from a caller's
/// `CodecParameters::pixel_format`. Returns `Ok(None)` if no pixel
/// format was set — caller defaults to 8-bit at decode time, derives
/// chroma from the frame header.
fn pick_output_format(params: &CodecParameters) -> Result<Option<OutputSurface>> {
    params.pixel_format.map(resolve_pixel_format).transpose()
}

pub fn make_decoder(params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    let (requested, alpha_typed) = match pick_output_format(params)? {
        Some((bd, cf, alpha)) => (Some((bd, cf)), alpha),
        None => (None, false),
    };
    Ok(Box::new(ProResDecoder {
        codec_id: params.codec_id.clone(),
        requested,
        alpha_typed,
        range: OutputRange::Full,
        threads: 1,
        pending: None,
        eof: false,
    }))
}

/// Concrete ProRes decoder produced by [`make_decoder`]. Exposed so a
/// caller using the direct (non-registry) path can opt into the RDD 36
/// §7.5.1 video-range clamp via [`ProResDecoder::set_output_range`]
/// before driving the [`Decoder`] trait.
pub struct ProResDecoder {
    codec_id: CodecId,
    /// Caller-requested output (bit-depth, chroma) pair. `None` means
    /// "infer from the frame header, default to 8-bit".
    requested: Option<(BitDepth, ChromaFormat)>,
    /// `true` when the caller requested an alpha-typed pixel format
    /// (`Yuva422P` / `Yuva444P`): every emitted frame carries exactly 4
    /// planes, with a synthesised fully-opaque plane when the stream
    /// codes no alpha.
    alpha_typed: bool,
    /// RDD 36 §7.5.1 clamp range applied during sample generation.
    range: OutputRange,
    /// Threads the slices of a picture may be decoded on. 1 until the
    /// caller grants more.
    threads: usize,
    pending: Option<Packet>,
    eof: bool,
}

impl ProResDecoder {
    /// Select the RDD 36 §7.5.1 output clamp `range` for subsequent
    /// frames. Defaults to [`OutputRange::Full`].
    pub fn set_output_range(&mut self, range: OutputRange) {
        self.range = range;
    }

    /// The currently configured §7.5.1 clamp range.
    pub fn output_range(&self) -> OutputRange {
        self.range
    }

    /// Decode the slices of each picture on up to `threads` threads
    /// (`0` and `1` both mean the calling thread only, the default).
    /// The decoded frames do not depend on the value.
    pub fn set_threads(&mut self, threads: usize) {
        self.threads = threads.max(1);
    }

    /// The thread budget set by [`set_threads`](Self::set_threads) or
    /// [`Decoder::set_execution_context`].
    pub fn threads(&self) -> usize {
        self.threads
    }
}

impl Decoder for ProResDecoder {
    fn codec_id(&self) -> &CodecId {
        &self.codec_id
    }

    fn send_packet(&mut self, packet: &Packet) -> Result<()> {
        if self.pending.is_some() {
            return Err(Error::other(
                "prores decoder: receive_frame must be called before sending another packet",
            ));
        }
        self.pending = Some(packet.clone());
        Ok(())
    }

    fn receive_frame(&mut self) -> Result<Frame> {
        let Some(pkt) = self.pending.take() else {
            return if self.eof {
                Err(Error::Eof)
            } else {
                Err(Error::NeedMore)
            };
        };
        let vf = decode_packet_inner(
            &pkt.data,
            pkt.pts,
            self.requested,
            self.range,
            self.alpha_typed,
            false,
            false,
            self.threads,
        )?;
        Ok(Frame::Video(vf))
    }

    fn flush(&mut self) -> Result<()> {
        self.eof = true;
        Ok(())
    }

    fn set_execution_context(&mut self, ctx: &ExecutionContext) {
        self.threads = ctx.threads.max(1);
    }
}

/// Decode a packet at the default output bit depth (8-bit). Equivalent
/// to `decode_packet_with_depth(data, pts, None)`.
pub fn decode_packet(data: &[u8], pts: Option<i64>) -> Result<VideoFrame> {
    decode_packet_with_depth(data, pts, None)
}

/// Decode a packet, emitting the requested `(BitDepth, ChromaFormat)` if
/// supplied. When `requested` is `None`, the chroma format is taken from
/// the frame header and the output is 8-bit.
///
/// Equivalent to [`decode_packet_with_options`] with
/// [`OutputRange::Full`].
pub fn decode_packet_with_depth(
    data: &[u8],
    pts: Option<i64>,
    requested: Option<(BitDepth, ChromaFormat)>,
) -> Result<VideoFrame> {
    decode_packet_with_options(data, pts, requested, OutputRange::Full)
}

/// Decode a packet with explicit control over both the output
/// `(BitDepth, ChromaFormat)` and the RDD 36 §7.5.1 clamp `range`.
///
/// `range` chooses whether reconstructed color component samples may use
/// all available quantization levels ([`OutputRange::Full`]) or are
/// confined to the permissible video levels that avoid the
/// BT.601/BT.709 synchronization/timing reference codes
/// ([`OutputRange::Video`]). The alpha plane is unaffected — §7.5.2
/// always maps decoded alpha across the full opacity range.
pub fn decode_packet_with_options(
    data: &[u8],
    pts: Option<i64>,
    requested: Option<(BitDepth, ChromaFormat)>,
    range: OutputRange,
) -> Result<VideoFrame> {
    decode_packet_inner(data, pts, requested, range, false, false, false, 1)
}

/// Decode a packet to the surface a `PixelFormat` describes — the
/// free-function twin of the registry path's
/// `CodecParameters::pixel_format` handling.
///
/// Accepts the eight `Yuv4(2|4)4P{,10Le,12Le,16Le}` requests of
/// [`decode_packet_with_depth`] plus the alpha-typed formats
/// `Yuva4(2|4)4P{,10Le,12Le,16Le}`. For the alpha-typed formats the
/// returned frame *always* has 4 planes:
///
/// * a stream that codes alpha (RDD 36 §6.1.1 Table 7,
///   `alpha_channel_type` 1 or 2) delivers it as plane 3 at the
///   surface depth, converted per §7.5.2 — consistent with the §7.5.1
///   conversion the Y/Cb/Cr planes undergo at that depth. On the
///   `*P16Le` surfaces 16-bit coded alpha passes through exactly (the
///   §7.5.2 conversion at `b = 16` is the identity);
/// * a stream with no coded alpha gets a synthesised fully-opaque
///   (all-`2^b − 1`) plane, so the declared plane layout is a hard
///   contract.
///
/// `None` behaves like [`decode_packet`] (8-bit, chroma from the frame
/// header, alpha plane only when coded). The chroma format of a `Some`
/// request must match the frame header's, as elsewhere.
pub fn decode_packet_with_format(
    data: &[u8],
    pts: Option<i64>,
    pixel_format: Option<PixelFormat>,
    range: OutputRange,
) -> Result<VideoFrame> {
    let (requested, alpha_typed) = match pixel_format.map(resolve_pixel_format).transpose()? {
        Some((bd, cf, alpha)) => (Some((bd, cf)), alpha),
        None => (None, false),
    };
    decode_packet_inner(data, pts, requested, range, alpha_typed, false, false, 1)
}

/// [`decode_packet_with_format`] with the slices of each picture decoded
/// on up to `threads` threads (`0` and `1` both mean the calling thread
/// only). Slices are independent (RDD 36 §5.3), so the frame is the
/// same for every thread count; only the time to produce it changes.
pub fn decode_packet_with_threads(
    data: &[u8],
    pts: Option<i64>,
    pixel_format: Option<PixelFormat>,
    range: OutputRange,
    threads: usize,
) -> Result<VideoFrame> {
    let (requested, alpha_typed) = match pixel_format.map(resolve_pixel_format).transpose()? {
        Some((bd, cf, alpha)) => (Some((bd, cf)), alpha),
        None => (None, false),
    };
    decode_packet_inner(
        data,
        pts,
        requested,
        range,
        alpha_typed,
        false,
        false,
        threads,
    )
}

/// Decode while retaining one coded right-edge chroma sample when the
/// visible 4:2:2 width is even and ends inside a macroblock. A co-sited
/// chroma upsampler needs the sample to the right of the last visible
/// one, which the ordinary crop drops. Luma, alpha and the visible
/// dimensions are unchanged; the Cb / Cr planes are one sample wider
/// than `ceil(width / 2)` and their `stride` describes it, so a consumer
/// that reads `width / 2` samples per row sees the extra one as padding.
///
/// Opt-in: RDD 36 §7.5 has the decoder discard every sample beyond
/// `horizontal_size`, which [`decode_packet_with_options`] and the
/// registry path still do. No bitstream is edited.
pub fn decode_packet_with_chroma_border(
    data: &[u8],
    pts: Option<i64>,
    requested: Option<(BitDepth, ChromaFormat)>,
    range: OutputRange,
) -> Result<VideoFrame> {
    decode_packet_inner(data, pts, requested, range, false, true, false, 1)
}

/// Decode to the coded width: every plane keeps the whole right
/// macroblock (`16 * width_in_mb` luma samples), as a session API that
/// exposes the coded width inside a fixed allocation would. The height
/// stays the visible height.
///
/// Opt-in: RDD 36 §7.5 has the decoder discard every sample beyond
/// `horizontal_size`, which [`decode_packet_with_options`] and the
/// registry path still do.
pub fn decode_packet_with_coded_width(
    data: &[u8],
    pts: Option<i64>,
    requested: Option<(BitDepth, ChromaFormat)>,
    range: OutputRange,
) -> Result<VideoFrame> {
    decode_packet_inner(data, pts, requested, range, false, false, true, 1)
}

/// Shared decode core. `alpha_typed` requests the 4-plane surface
/// guarantee of the `Yuva*` pixel formats (see
/// [`decode_packet_with_format`]); the coded-stream handling is
/// otherwise identical. `threads` is the number of threads the slices
/// of a picture may be decoded on; the frame is the same for every
/// value.
#[allow(clippy::too_many_arguments)]
fn decode_packet_inner(
    data: &[u8],
    pts: Option<i64>,
    requested: Option<(BitDepth, ChromaFormat)>,
    range: OutputRange,
    alpha_typed: bool,
    retain_chroma_border: bool,
    retain_coded_width: bool,
    threads: usize,
) -> Result<VideoFrame> {
    let (fh, after_frame) = parse_frame(data)?;

    let width = fh.width as usize;
    let height = fh.height as usize;
    if width == 0 || height == 0 {
        return Err(Error::invalid("prores: zero-sized frame"));
    }
    // Bound allocations against the declared dimensions.
    let pixels = width.saturating_mul(height);
    if pixels > MAX_DECODED_PIXELS {
        return Err(Error::invalid(format!(
            "prores: declared frame size ({width}x{height}) exceeds internal cap"
        )));
    }
    let alpha_kind = AlphaChannelType::from_code(fh.alpha_channel_type)?;
    let has_alpha = alpha_kind.is_some();
    let interlaced = fh.interlace_mode != 0;

    let chroma = fh.chroma_format;
    let bit_depth = if let Some((bd, requested_chroma)) = requested {
        if requested_chroma != chroma {
            return Err(Error::invalid(format!(
                "prores: requested pixel_format chroma {:?} does not match \
                 frame chroma {:?}",
                requested_chroma, chroma,
            )));
        }
        bd
    } else {
        BitDepth::Eight
    };
    let bps = bit_depth.bytes_per_sample();
    let mbs_x = width.div_ceil(MB_SIDE_PX);

    // The coded extent (whole macroblocks, and for interlaced frames
    // whole macroblock rows per field) bounds what a frame may ask for.
    let padded_w = mbs_x * MB_SIDE_PX;
    let padded_c_w = match chroma {
        ChromaFormat::Y422 => padded_w / 2,
        ChromaFormat::Y444 => padded_w,
    };
    let padded_frame_h = if interlaced {
        let top_mb_rows = height.div_ceil(2).div_ceil(MB_SIDE_PX);
        let bot_mb_rows = (height / 2).div_ceil(MB_SIDE_PX);
        top_mb_rows.max(bot_mb_rows) * MB_SIDE_PX * 2
    } else {
        height.div_ceil(MB_SIDE_PX) * MB_SIDE_PX
    };
    let y_alloc = padded_w.saturating_mul(padded_frame_h);
    let c_alloc = padded_c_w.saturating_mul(padded_frame_h);
    if y_alloc > MAX_DECODED_PIXELS || c_alloc > MAX_DECODED_PIXELS {
        return Err(Error::invalid("prores: padded plane size exceeds cap"));
    }

    // Output planes hold exactly the samples the caller gets: slices
    // write straight into them and the samples RDD 36 §7.5.3 discards
    // (right of the visible width, below the visible height) are never
    // stored. Session callers can retain the reconstructed right
    // macroblock, or one extra 4:2:2 chroma column.
    let y_w = if retain_coded_width { padded_w } else { width };
    let c_w = match chroma {
        ChromaFormat::Y422 => y_w.div_ceil(2),
        ChromaFormat::Y444 => y_w,
    };
    let c_w = c_w
        + usize::from(
            retain_chroma_border
                && chroma == ChromaFormat::Y422
                && y_w % 2 == 0
                && c_w < padded_c_w,
        );
    let mut y_plane = vec![0u8; y_w * bps * height];
    let mut cb_plane = vec![0u8; c_w * bps * height];
    let mut cr_plane = vec![0u8; c_w * bps * height];
    let mut a_plane = if has_alpha {
        vec![0u8; y_w * bps * height]
    } else {
        Vec::new()
    };

    // Per §6.2 picture_vertical_size: a progressive frame is one
    // picture; an interlaced one is two field pictures, the top field
    // on the even frame rows (§7.5.3), in the order interlace_mode gives.
    let top = (height.div_ceil(2), FieldStride::new(2, 0));
    let bottom = (height / 2, FieldStride::new(2, 1));
    let pictures: &[(usize, FieldStride)] = match fh.interlace_mode {
        0 => &[(height, FieldStride::progressive())],
        1 => &[top, bottom],
        2 => &[bottom, top],
        other => {
            return Err(Error::invalid(format!(
                "prores: invalid interlace_mode {other}"
            )));
        }
    };
    let (nmin, nmax) = range.bounds(bit_depth);
    let mut cursor = after_frame;
    for &(picture_height, field) in pictures {
        let (slices, rest) = picture_slices(cursor, mbs_x, picture_height)?;
        cursor = rest;
        let picture = Picture {
            chroma,
            bit_depth,
            codes: SampleCodes {
                scale: (1u32 << bit_depth.bits()) as f32 / 512.0,
                nmin: nmin as f32,
                nmax: nmax as f32,
            },
            alpha_kind,
            inv_scan: if interlaced {
                &INTERLACED_INV_SCAN
            } else {
                &PROGRESSIVE_INV_SCAN
            },
            luma_weights: fh.luma_qmat.map(f32::from),
            chroma_weights: fh.chroma_qmat.map(f32::from),
            field,
            picture_height,
            y_w,
            c_w,
        };
        picture.decode(
            &slices,
            &mut y_plane,
            &mut cb_plane,
            &mut cr_plane,
            &mut a_plane,
            threads,
        )?;
    }
    // Remaining bytes are stuffing per §5.1 / §6.1.2.

    let mut planes = vec![
        VideoPlane {
            stride: y_w * bps,
            data: y_plane,
        },
        VideoPlane {
            stride: c_w * bps,
            data: cb_plane,
        },
        VideoPlane {
            stride: c_w * bps,
            data: cr_plane,
        },
    ];
    if has_alpha {
        planes.push(VideoPlane {
            stride: y_w * bps,
            data: a_plane,
        });
    } else if alpha_typed {
        // The caller's pixel format (`Yuva422P` / `Yuva444P`) promises a
        // 4-plane layout, but the stream codes no alpha
        // (alpha_channel_type == 0, §6.1.1 Table 7): synthesise a fully
        // opaque plane — §7.5.2 maps full opacity to the all-ones code
        // at the output depth.
        planes.push(opaque_alpha_plane(y_w, height, bit_depth));
    }
    Ok(VideoFrame { pts, planes })
}

/// Build a fully opaque full-resolution alpha plane at the output
/// depth: every sample is the depth's maximum (§7.5.2 maps full opacity
/// to `2^b − 1`).
fn opaque_alpha_plane(width: usize, height: usize, bit_depth: BitDepth) -> VideoPlane {
    let bps = bit_depth.bytes_per_sample();
    let max = bit_depth.max_value();
    let mut data = vec![0u8; width * height * bps];
    if bps == 1 {
        data.fill(max as u8);
    } else {
        let le = (max as u16).to_le_bytes();
        for s in data.as_chunks_mut::<2>().0 {
            s.copy_from_slice(&le);
        }
    }
    VideoPlane {
        stride: width * bps,
        data,
    }
}

/// Field-row mapping. For progressive: `step=1, offset=0` (rows are
/// contiguous). For interlaced: `step=2` and `offset` is 0 (top field)
/// or 1 (bottom field). Picture row `r` is frame row
/// `step * r + offset`.
#[derive(Copy, Clone, Debug)]
struct FieldStride {
    step: usize,
    offset: usize,
}

impl FieldStride {
    const fn new(step: usize, offset: usize) -> Self {
        Self { step, offset }
    }
    const fn progressive() -> Self {
        Self { step: 1, offset: 0 }
    }
    fn map(self, picture_row: usize) -> usize {
        self.step * picture_row + self.offset
    }
}

/// The §7.5.1 conversion of a reconstructed value to a sample code:
/// `clamp(round(scale * (v + 256)))` with `scale = 2^b / 512`.
#[derive(Copy, Clone)]
struct SampleCodes {
    scale: f32,
    nmin: f32,
    nmax: f32,
}

impl SampleCodes {
    /// The sample code of reconstructed value `v`, as an integral
    /// `f32`. Written as selects rather than branches so a row of eight
    /// converts without a jump per sample.
    #[inline(always)]
    fn code(self, v: f32) -> f32 {
        let s = (v + 256.0) * self.scale;
        let r = s.round();
        let r = if s <= self.nmin { self.nmin } else { r };
        if s >= self.nmax { self.nmax } else { r }
    }
}

/// Everything the slices of one `picture()` share.
struct Picture {
    chroma: ChromaFormat,
    bit_depth: BitDepth,
    codes: SampleCodes,
    alpha_kind: Option<AlphaChannelType>,
    /// Natural (row-major) position of each scanned coefficient index.
    inv_scan: &'static [u8; 64],
    /// Quantisation weights in natural order.
    luma_weights: [f32; 64],
    chroma_weights: [f32; 64],
    field: FieldStride,
    picture_height: usize,
    /// Output plane widths in samples.
    y_w: usize,
    c_w: usize,
}

/// One macroblock row of a picture: its slices and the rows of every
/// output plane they can reach. A macroblock row of a progressive
/// picture covers 16 frame rows; that of a field covers the 32 frame
/// rows it shares with the same macroblock row of the other field, of
/// which it writes every second one. Bands of one picture are disjoint,
/// so they can be decoded on different threads.
struct Band<'a, 'd> {
    slices: &'a [SliceRef<'d>],
    y: &'a mut [u8],
    cb: &'a mut [u8],
    cr: &'a mut [u8],
    a: &'a mut [u8],
}

/// Per-thread working storage, grown once to the largest slice.
#[derive(Default)]
struct SliceScratch {
    blocks: Vec<[f32; 64]>,
    masks: Vec<u16>,
    alpha: Vec<u16>,
}

impl Picture {
    /// Decode every slice of the picture into the output planes, on up
    /// to `threads` threads.
    fn decode(
        &self,
        slices: &[SliceRef<'_>],
        y: &mut [u8],
        cb: &mut [u8],
        cr: &mut [u8],
        a: &mut [u8],
        threads: usize,
    ) -> Result<()> {
        let mbs_y = self.picture_height.div_ceil(MB_SIDE_PX);
        if mbs_y == 0 {
            return Ok(());
        }
        let bps = self.bit_depth.bytes_per_sample();
        let band_rows = MB_SIDE_PX * self.field.step;
        let y_band = band_rows * self.y_w * bps;
        let c_band = band_rows * self.c_w * bps;
        let mut alpha_bands = a.chunks_mut(y_band);
        let mut bands: Vec<Band<'_, '_>> = slices
            .chunks(slices.len() / mbs_y)
            .zip(y.chunks_mut(y_band))
            .zip(cb.chunks_mut(c_band).zip(cr.chunks_mut(c_band)))
            .map(|((slices, y), (cb, cr))| Band {
                slices,
                y,
                cb,
                cr,
                a: alpha_bands.next().unwrap_or_default(),
            })
            .collect();
        if bands.len() != mbs_y {
            return Err(Error::invalid(
                "prores: macroblock rows do not match the frame height",
            ));
        }
        let run = |group: &mut [Band<'_, '_>]| -> Result<()> {
            let mut scratch = SliceScratch::default();
            for band in group {
                for slice in band.slices {
                    self.decode_slice(slice, band, &mut scratch)?;
                }
            }
            Ok(())
        };
        let workers = threads.clamp(1, bands.len());
        if workers == 1 {
            return run(&mut bands);
        }
        // Contiguous groups of bands, one per worker; the first error in
        // band order is the one a single thread would have reported.
        let per_worker = bands.len().div_ceil(workers);
        std::thread::scope(|scope| {
            let mut groups = bands.chunks_mut(per_worker);
            let first = groups.next();
            let handles: Vec<_> = groups.map(|g| scope.spawn(|| run(g))).collect();
            let mut result = first.map_or(Ok(()), &run);
            for handle in handles {
                let r = handle
                    .join()
                    .unwrap_or_else(|_| Err(Error::other("prores: slice decoding thread failed")));
                if result.is_ok() {
                    result = r;
                }
            }
            result
        })
    }

    /// Decode one `slice()` (RDD 36 §5.3) into its band.
    fn decode_slice(
        &self,
        slice: &SliceRef<'_>,
        band: &mut Band<'_, '_>,
        scratch: &mut SliceScratch,
    ) -> Result<()> {
        const LUMA_OFFSETS: [(usize, usize); 4] = [(0, 0), (1, 0), (0, 1), (1, 1)];
        let chroma_offsets: &[(usize, usize)] = match self.chroma {
            ChromaFormat::Y422 => &[(0, 0), (0, 1)],
            ChromaFormat::Y444 => &[(0, 0), (0, 1), (1, 0), (1, 1)],
        };
        let has_alpha = self.alpha_kind.is_some();
        let (sh, after_sh) = parse_slice_header(slice.data, has_alpha)?;
        let coded_y = sh.coded_size_of_y_data as usize;
        let coded_cb = sh.coded_size_of_cb_data as usize;
        let cr_data_size = if let Some(sz) = sh.coded_size_of_cr_data {
            sz as usize
        } else {
            slice
                .data
                .len()
                .checked_sub(sh.slice_header_size as usize + coded_y + coded_cb)
                .ok_or_else(|| Error::invalid("prores: cr_data size underflow"))?
        };
        if after_sh.len() < coded_y + coded_cb + cr_data_size {
            return Err(Error::invalid("prores: slice components truncated"));
        }
        let (y_data, rest) = after_sh.split_at(coded_y);
        let (cb_data, rest) = rest.split_at(coded_cb);
        let (cr_data, alpha_data) = rest.split_at(cr_data_size);

        let mbs = slice.mbs;
        let cb_per_mb = chroma_blocks_per_mb(self.chroma);
        let per_mb = blocks_per_mb(self.chroma);
        let SliceScratch {
            blocks,
            masks,
            alpha,
        } = scratch;
        blocks.clear();
        blocks.resize(mbs * per_mb, [0.0; 64]);
        masks.clear();
        masks.resize(mbs * per_mb, 0);
        // F[v][u] = (QF[v][u] * W[v][u] * qScale) / 8 (§7.3), applied to
        // each coefficient as it is decoded.
        let qs = qscale(sh.quantization_index) as f32;
        for (data, blocks_per_mb, offset, weights) in [
            (y_data, LUMA_BLOCKS_PER_MB, 0, &self.luma_weights),
            (cb_data, cb_per_mb, LUMA_BLOCKS_PER_MB, &self.chroma_weights),
            (
                cr_data,
                cb_per_mb,
                LUMA_BLOCKS_PER_MB + cb_per_mb,
                &self.chroma_weights,
            ),
        ] {
            let layout = ComponentLayout {
                mbs,
                blocks_per_mb,
                stride: per_mb,
                offset,
            };
            decode_component_blocks(data, &layout, self.inv_scan, weights, qs, blocks, masks)?;
        }

        let bps = self.bit_depth.bytes_per_sample();
        for (mb, (mb_blocks, mb_masks)) in blocks
            .chunks_exact_mut(per_mb)
            .zip(masks.chunks_exact(per_mb))
            .enumerate()
        {
            let mb_x = slice.mb_x + mb;
            let (luma, chroma_blocks) = mb_blocks.split_at_mut(LUMA_BLOCKS_PER_MB);
            let (cb_blocks, cr_blocks) = chroma_blocks.split_at_mut(cb_per_mb);
            for ((blk, &mask), (bx, by)) in luma.iter_mut().zip(mb_masks).zip(LUMA_OFFSETS) {
                let x0 = mb_x * MB_SIDE_PX + bx * 8;
                self.reconstruct(blk, mask, band.y, self.y_w * bps, self.y_w, x0, by * 8);
            }
            let chroma_masks = &mb_masks[LUMA_BLOCKS_PER_MB..];
            for (plane, plane_blocks, plane_masks) in [
                (&mut *band.cb, cb_blocks, &chroma_masks[..cb_per_mb]),
                (&mut *band.cr, cr_blocks, &chroma_masks[cb_per_mb..]),
            ] {
                for ((blk, &mask), &(bx, by)) in
                    plane_blocks.iter_mut().zip(plane_masks).zip(chroma_offsets)
                {
                    let x0 = match self.chroma {
                        ChromaFormat::Y422 => mb_x * 8,
                        ChromaFormat::Y444 => mb_x * MB_SIDE_PX + bx * 8,
                    };
                    self.reconstruct(blk, mask, plane, self.c_w * bps, self.c_w, x0, by * 8);
                }
            }
        }

        if let Some(act) = self.alpha_kind {
            // How many rows the bottom macroblock row's alpha array
            // holds depends on the encoder. RDD 36 §5.3 (slice syntax)
            // sizes it to the visible rows — `sliceVerticalSize =
            // picture_vertical_size − 16·(height_in_mb − 1)` for
            // `i = height_in_mb − 1` — and §7.5.3 restates it ("does not
            // include alpha values for the excess row(s)"). Streams in
            // the wild carry either that or the full 16 rows, and
            // decoders accept both (black-box, r480), so the 16-row read
            // is tried first and the visible-row count (rounded up to
            // even for a field as a further tolerance) is the fallback.
            // `decode_scanned_alpha`'s end-of-payload check tells the
            // two apart: a 16-row read of a compact payload runs out of
            // bits, and a compact read of a full payload leaves more
            // than the byte-alignment pad behind (or stops a run short).
            // Only the visible rows are stored either way.
            let cols = MB_SIDE_PX * mbs;
            let visible_rows = self
                .picture_height
                .saturating_sub(slice.mb_y * MB_SIDE_PX)
                .min(MB_SIDE_PX);
            let mut decoded = decode_scanned_alpha_into(alpha_data, cols * MB_SIDE_PX, act, alpha);
            for rows in [visible_rows, visible_rows.next_multiple_of(2)] {
                if decoded.is_ok() {
                    break;
                }
                if rows == MB_SIDE_PX {
                    continue;
                }
                decoded = decode_scanned_alpha_into(alpha_data, cols * rows, act, alpha);
            }
            decoded?;
            self.store_alpha(
                band.a,
                slice.mb_x * MB_SIDE_PX,
                cols,
                visible_rows,
                alpha,
                act,
            );
        }
        Ok(())
    }

    /// Inverse-transform one dequantised block (§7.4) and store its
    /// visible samples (§7.5.1, §7.5.3) at `(x0, y0)` of the band, `y0`
    /// in picture rows.
    #[allow(clippy::too_many_arguments)]
    fn reconstruct(
        &self,
        blk: &mut [f32; 64],
        mask: u16,
        plane: &mut [u8],
        byte_stride: usize,
        plane_w: usize,
        x0: usize,
        y0: usize,
    ) {
        let visible = plane_w.saturating_sub(x0).min(8);
        if visible == 0 {
            return;
        }
        let bps = self.bit_depth.bytes_per_sample();
        let codes = self.codes;
        // A block whose AC coefficients are all zero reconstructs to the
        // constant `F[0][0] / 8`: one conversion serves all 64 samples.
        let constant = (mask & AC_PRESENT == 0).then(|| codes.code(blk[0] * 0.125));
        if constant.is_none() {
            idct8x8_rows(blk, mask as u8);
        }
        for (j, row) in blk.as_chunks::<8>().0.iter().enumerate() {
            let start = self.field.map(y0 + j) * byte_stride + x0 * bps;
            let Some(dst) = plane.get_mut(start..start + visible * bps) else {
                // Below the visible height (§7.5.3): not stored.
                break;
            };
            let samples: [f32; 8] = match constant {
                Some(c) => [c; 8],
                None => row.map(|v| codes.code(v)),
            };
            if bps == 1 {
                for (d, s) in dst.iter_mut().zip(samples) {
                    *d = s as u8;
                }
            } else {
                for (d, s) in dst.as_chunks_mut::<2>().0.iter_mut().zip(samples) {
                    *d = (s as u16).to_le_bytes();
                }
            }
        }
    }

    /// Store the visible part of a slice's decoded alpha array
    /// (`cols` values per row) at sample column `x0` of the band,
    /// converted to the output depth per §7.5.2.
    fn store_alpha(
        &self,
        plane: &mut [u8],
        x0: usize,
        cols: usize,
        rows: usize,
        values: &[u16],
        act: AlphaChannelType,
    ) {
        let bps = self.bit_depth.bytes_per_sample();
        let byte_stride = self.y_w * bps;
        let visible = self.y_w.saturating_sub(x0).min(cols);
        let depth = self.bit_depth;
        for (r, row) in values.chunks_exact(cols).take(rows).enumerate() {
            let start = self.field.map(r) * byte_stride + x0 * bps;
            let Some(dst) = plane.get_mut(start..start + visible * bps) else {
                break;
            };
            if bps == 1 {
                for (d, &v) in dst.iter_mut().zip(row) {
                    *d = alpha_code(v, act, depth) as u8;
                }
            } else {
                for (d, &v) in dst.as_chunks_mut::<2>().0.iter_mut().zip(row) {
                    *d = alpha_code(v, act, depth).to_le_bytes();
                }
            }
        }
    }
}

/// Convert a decoded alpha value (8-bit in `0..=255` or 16-bit in
/// `0..=65535` per `act`) to a pixel sample of the requested output bit
/// depth, per RDD 36 §7.5.2: `alphaSample = round((2^b - 1) * alpha /
/// mask)`. The division is resolved per coded width so it compiles to
/// a multiplication by a constant.
#[inline]
fn alpha_code(alpha: u16, act: AlphaChannelType, out_depth: BitDepth) -> u16 {
    let max_out = out_depth.max_value();
    let a = u32::from(alpha);
    match act {
        AlphaChannelType::Eight if max_out == 255 => alpha,
        AlphaChannelType::Sixteen if max_out == 65535 => alpha,
        AlphaChannelType::Eight => ((max_out * a + 127) / 255) as u16,
        AlphaChannelType::Sixteen => ((max_out * a + 32767) / 65535) as u16,
    }
}

/// One coded slice of a picture: its position in the macroblock grid
/// and its bytes (slice header included).
struct SliceRef<'a> {
    mb_x: usize,
    mb_y: usize,
    mbs: usize,
    data: &'a [u8],
}

/// Walk a `picture()` (RDD 36 §5.2): parse the picture header and the
/// slice table and return every slice with its macroblock position,
/// plus the bytes following the picture.
fn picture_slices(
    data: &[u8],
    mbs_x: usize,
    picture_height: usize,
) -> Result<(Vec<SliceRef<'_>>, &[u8])> {
    let mbs_y = picture_height.div_ceil(MB_SIDE_PX);
    let (ph, after_pic) = parse_picture_header(data)?;
    let slice_sizes_template = compute_slice_sizes(mbs_x, ph.log2_desired_slice_size_in_mb);
    let slices_per_row = slice_sizes_template.len();
    let expected_slice_count = slices_per_row * mbs_y;
    let slice_table_bytes = expected_slice_count
        .checked_mul(2)
        .ok_or_else(|| Error::invalid("prores: slice count overflow"))?;
    if after_pic.len() < slice_table_bytes {
        return Err(Error::invalid("prores: slice-size table truncated"));
    }
    let (table, mut cursor) = after_pic.split_at(slice_table_bytes);
    let slice_sizes: Vec<usize> = table
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| u16::from_be_bytes(*b) as usize)
        .collect();
    // The bytes occupied by the defined picture syntax this decoder
    // consumes: the picture header, the slice table, and every slice
    // payload. RDD 36 §6.2.1 makes `picture_size` (which "includes the
    // picture header") the authoritative total — and §6.4 permits a
    // *version variant* to append informative bytes after the defined
    // syntax, inflating `picture_size` beyond this consumed total.
    let consumed_picture_bytes: usize =
        ph.picture_header_size as usize + slice_table_bytes + slice_sizes.iter().sum::<usize>();
    let declared_picture_bytes = ph.picture_size as usize;
    // The defined syntax can never claim more space than the picture
    // declares; if it does the slice table / sizes are corrupt.
    if consumed_picture_bytes > declared_picture_bytes {
        return Err(Error::invalid(
            "prores: header + slice table + payloads exceed declared picture_size",
        ));
    }
    // RDD 36 §6.2.1 / §6.4: "decoders shall use the specified size —
    // rather than inference from the syntax itself — to determine the
    // start of the immediately following syntax structure." The bytes
    // after the picture start at the *declared* `picture_size` (which
    // equals the consumed total for a base bitstream, or exceeds it by
    // the trailing version-variant bytes a forward-compatible stream may
    // carry) so that, for an interlaced frame, the second field's
    // `picture()` starts at the correct offset even when version-variant
    // data sits after the first field's slices.
    if data.len() < declared_picture_bytes {
        return Err(Error::invalid("prores: picture overruns buffer"));
    }
    let mut slices = Vec::with_capacity(expected_slice_count);
    let mut sizes = slice_sizes.iter();
    for mb_y in 0..mbs_y {
        let mut mb_x = 0usize;
        for &template in &slice_sizes_template {
            let mbs = template.min(mbs_x - mb_x);
            if mbs == 0 {
                break;
            }
            let coded_size = *sizes.next().expect("one table entry per slice");
            if cursor.len() < coded_size {
                return Err(Error::invalid("prores: slice payload truncated"));
            }
            let (slice_data, tail) = cursor.split_at(coded_size);
            cursor = tail;
            slices.push(SliceRef {
                mb_x,
                mb_y,
                mbs,
                data: slice_data,
            });
            mb_x += mbs;
        }
    }
    Ok((slices, &data[declared_picture_bytes..]))
}

/// Range of the dequantised DCT coefficients `F[v][u]` of a frame, as
/// `(min, max)` of `QF * W * qScale` — i.e. in eighths (RDD 36 §7.3).
///
/// Annex A qualifies an IDCT only for coefficients in −2048..=+2047.75
/// (`-16384..=16382` here). A frame whose range leaves that interval is
/// outside the domain in which conforming decoders are bound to agree.
#[doc(hidden)]
pub fn dequantized_coefficient_range(data: &[u8]) -> Result<(i64, i64)> {
    let (fh, mut cursor) = parse_frame(data)?;
    let has_alpha = AlphaChannelType::from_code(fh.alpha_channel_type)?.is_some();
    let interlaced = fh.interlace_mode != 0;
    let height = fh.height as usize;
    let mbs_x = (fh.width as usize).div_ceil(MB_SIDE_PX);
    let heights: &[usize] = match fh.interlace_mode {
        0 => &[height],
        1 => &[height.div_ceil(2), height / 2],
        _ => &[height / 2, height.div_ceil(2)],
    };
    let luma_blocks_end = LUMA_BLOCKS_PER_MB;
    let per_mb = blocks_per_mb(fh.chroma_format);
    let (mut lo, mut hi) = (0i64, 0i64);
    for &picture_height in heights {
        let (slices, rest) = picture_slices(cursor, mbs_x, picture_height)?;
        cursor = rest;
        for slice in &slices {
            let (sh, after_sh) = parse_slice_header(slice.data, has_alpha)?;
            let coded_y = sh.coded_size_of_y_data as usize;
            let coded_cb = sh.coded_size_of_cb_data as usize;
            let coded_cr = match sh.coded_size_of_cr_data {
                Some(sz) => sz as usize,
                None => after_sh
                    .len()
                    .checked_sub(coded_y + coded_cb)
                    .ok_or_else(|| Error::invalid("prores: cr_data size underflow"))?,
            };
            if after_sh.len() < coded_y + coded_cb + coded_cr {
                return Err(Error::invalid("prores: slice components truncated"));
            }
            let blocks = decode_slice_components(
                &after_sh[..coded_y],
                &after_sh[coded_y..coded_y + coded_cb],
                &after_sh[coded_y + coded_cb..coded_y + coded_cb + coded_cr],
                slice.mbs,
                fh.chroma_format,
                interlaced,
            )?;
            let qs = i64::from(qscale(sh.quantization_index));
            for (i, blk) in blocks.iter().enumerate() {
                let qmat = if i % per_mb < luma_blocks_end {
                    &fh.luma_qmat
                } else {
                    &fh.chroma_qmat
                };
                for (&q, &w) in blk.iter().zip(qmat) {
                    let f = i64::from(q) * i64::from(w) * qs;
                    lo = lo.min(f);
                    hi = hi.max(f);
                }
            }
        }
    }
    Ok((lo, hi))
}

/// Convert a single reconstructed color component value `v` (the centred
/// IDCT output, nominally in `[-256, 256)` per §7.5.1) to a pixel
/// component sample of the requested output bit depth and clamp `range`,
/// per RDD 36 §7.5.1:
///
/// `s = clamp(round(2^b * (v + 256) / 512))`,
///
/// where `clamp(n)` restricts `n` to `(nmin, nmax)`. For `b = 8` the
/// `2^b / 512` factor is `0.5`, for `b = 10` it is `2.0`, for `b = 12`
/// it is `8.0`, and for `b = 16` it is `128.0`. The clamp bounds are
/// `(0, 2^b − 1)` for
/// [`OutputRange::Full`] and `(1, 2^b − 2)` for [`OutputRange::Video`].
///
/// The branch-per-sample statement of the conversion, kept as the
/// reference the unit tests hold `SampleCodes::code` to.
#[cfg(test)]
fn color_to_sample(v: f32, bit_depth: BitDepth, range: OutputRange) -> u32 {
    let scale = match bit_depth {
        BitDepth::Eight => 0.5,
        BitDepth::Ten => 2.0,
        BitDepth::Twelve => 8.0,
        BitDepth::Sixteen => 128.0,
    };
    let (nmin, nmax) = range.bounds(bit_depth);
    let s = (v + 256.0) * scale;
    if s <= nmin as f32 {
        nmin
    } else if s >= nmax as f32 {
        nmax
    } else {
        s.round() as u32
    }
}

/// Convert a decoded alpha value (8-bit in `0..=255` or 16-bit in
/// `0..=65535` per `act`) to a pixel sample of the requested output bit
/// depth, per RDD 36 §7.5.2:
///
/// `alphaSample = round((2^b - 1) * alpha / mask)`
#[cfg(test)]
fn alpha_to_sample(alpha: u16, act: AlphaChannelType, out_depth: BitDepth) -> u16 {
    let max_out = out_depth.max_value();
    let mask = act.mask();
    // round((max_out * alpha) / mask) using integer math.
    let num = max_out as u64 * alpha as u64;
    let denom = mask as u64;
    ((num + denom / 2) / denom) as u16
}

#[cfg(test)]
mod alpha_sample_tests {
    //! White-box coverage for the RDD 36 §7.5.2 decoded-alpha →
    //! pixel-alpha-sample conversion `alpha_to_sample`. The end-to-end
    //! decode path only ever reaches this code with an alpha-bearing
    //! 4444 stream, and the only prior coverage of the bit-depth
    //! promotion/demotion arm lived in the `ffmpeg_interop` integration
    //! tests — which skip entirely when the external validator binary is
    //! absent (the usual CI case). These tests lock the §7.5.2 formula
    //! `alphaSample = round((2^b − 1) * alpha ÷ mask)` directly, with no
    //! external dependency.
    use super::{AlphaChannelType, BitDepth, alpha_to_sample};

    /// §7.5.2: "If the pixel component sample bit depth matches that of
    /// the decoded alpha values, the pixel alpha component samples shall
    /// be the decoded values themselves." 8-bit alpha (alpha_channel_type
    /// == 1) into an 8-bit sample must be the exact identity across the
    /// full `0..=255` domain.
    #[test]
    fn eight_bit_alpha_to_eight_bit_is_identity() {
        for a in 0u16..=255 {
            assert_eq!(
                alpha_to_sample(a, AlphaChannelType::Eight, BitDepth::Eight),
                a,
                "8-bit alpha {a} must map to itself at 8-bit output"
            );
        }
    }

    /// §7.5.2 endpoints: the smallest and largest decoded alpha values
    /// signify opacities of exactly 0.0 and 1.0 and must map to the
    /// smallest (0) and largest (`2^b − 1`) pixel sample at every output
    /// depth, for both alpha_channel_type values.
    #[test]
    fn endpoints_map_to_full_opacity_range() {
        for act in [AlphaChannelType::Eight, AlphaChannelType::Sixteen] {
            let max_in = act.mask() as u16;
            for depth in [
                BitDepth::Eight,
                BitDepth::Ten,
                BitDepth::Twelve,
                BitDepth::Sixteen,
            ] {
                assert_eq!(
                    alpha_to_sample(0, act, depth),
                    0,
                    "alpha 0 (opacity 0.0) must map to sample 0 ({act:?} -> {depth:?})"
                );
                assert_eq!(
                    u32::from(alpha_to_sample(max_in, act, depth)),
                    depth.max_value(),
                    "alpha {max_in} (opacity 1.0) must map to sample 2^b-1 ({act:?} -> {depth:?})"
                );
            }
        }
    }

    /// §7.5.2 promotion: 8-bit decoded alpha into a 10/12-bit pixel
    /// sample uses `round((2^b − 1) * alpha ÷ 255)`. Spot-check a handful
    /// of values against the formula evaluated independently here with
    /// round-half-away-from-zero (== round-half-up for non-negatives,
    /// the §3 `round(x) = floor(x + 1/2)` convention).
    #[test]
    fn eight_bit_alpha_promotes_to_higher_depth() {
        fn expect(alpha: u32, max_out: u32, mask: u32) -> u16 {
            ((max_out * alpha * 2 + mask) / (mask * 2)) as u16
        }
        for &alpha in &[0u32, 1, 64, 127, 128, 200, 254, 255] {
            // 8-bit -> 10-bit (max 1023, mask 255)
            assert_eq!(
                alpha_to_sample(alpha as u16, AlphaChannelType::Eight, BitDepth::Ten),
                expect(alpha, 1023, 255),
                "8->10 promotion mismatch at alpha {alpha}"
            );
            // 8-bit -> 12-bit (max 4095, mask 255)
            assert_eq!(
                alpha_to_sample(alpha as u16, AlphaChannelType::Eight, BitDepth::Twelve),
                expect(alpha, 4095, 255),
                "8->12 promotion mismatch at alpha {alpha}"
            );
            // 8-bit -> 16-bit (max 65535, mask 255). 65535 = 255 * 257,
            // so the §7.5.2 promotion is the exact integer 257 * alpha.
            assert_eq!(
                alpha_to_sample(alpha as u16, AlphaChannelType::Eight, BitDepth::Sixteen),
                (alpha * 257) as u16,
                "8->16 promotion mismatch at alpha {alpha}"
            );
        }
    }

    /// §7.5.2 at matched 16-bit depths: 16-bit decoded alpha
    /// (alpha_channel_type == 2) into a 16-bit pixel sample is the exact
    /// identity — "If the pixel component sample bit depth matches that
    /// of the decoded alpha values, the pixel alpha component samples
    /// shall be the decoded values themselves." This is the full-alpha-
    /// fidelity guarantee of the `*P16Le` surfaces.
    #[test]
    fn sixteen_bit_alpha_to_sixteen_bit_is_identity() {
        for a in (0u32..=65535).step_by(97).chain([65535]) {
            assert_eq!(
                alpha_to_sample(a as u16, AlphaChannelType::Sixteen, BitDepth::Sixteen),
                a as u16,
                "16-bit alpha {a} must map to itself at 16-bit output"
            );
        }
    }

    /// §7.5.2 demotion: 16-bit decoded alpha (alpha_channel_type == 2)
    /// into 8/10/12-bit samples uses `round((2^b − 1) * alpha ÷ 65535)`.
    /// The output is monotonic non-decreasing in the input and never
    /// exceeds `2^b − 1`.
    #[test]
    fn sixteen_bit_alpha_demotes_monotonically() {
        fn expect(alpha: u32, max_out: u32, mask: u32) -> u16 {
            ((max_out as u64 * alpha as u64 * 2 + mask as u64) / (mask as u64 * 2)) as u16
        }
        for depth in [BitDepth::Eight, BitDepth::Ten, BitDepth::Twelve] {
            let max_out = depth.max_value();
            let mut prev = 0u16;
            for alpha in (0u32..=65535).step_by(257) {
                let got = alpha_to_sample(alpha as u16, AlphaChannelType::Sixteen, depth);
                assert_eq!(
                    got,
                    expect(alpha, max_out, 65535),
                    "16->{}b demotion mismatch at alpha {alpha}",
                    depth.bits()
                );
                assert!(
                    u32::from(got) <= max_out,
                    "demoted sample {got} exceeds 2^b-1 = {max_out}"
                );
                assert!(got >= prev, "demotion must be monotonic non-decreasing");
                prev = got;
            }
        }
    }

    /// §7.5.2 round-half-up boundary: pick an `(alpha, depth)` whose exact
    /// quotient lands on a half-integer and confirm `alpha_to_sample`
    /// rounds it up (the §3 `floor(x + 1/2)` convention), not toward zero.
    #[test]
    fn rounds_half_up_at_a_known_midpoint() {
        // 8-bit alpha 1 into a hypothetical b where (2^b-1)/255 = 0.5
        // exact is not realisable, so construct a concrete half-integer:
        // 16-bit alpha into 8-bit: (255 * alpha) / 65535. Choose alpha so
        // 255*alpha / 65535 = k + 0.5 exactly. 65535 = 255*257, so
        // 255*alpha/65535 = alpha/257; alpha = 257*k + 128.5 is not
        // integral, but alpha = 128 gives 128/257 = 0.498… (-> 0) and the
        // exact half-integer arises for 8-bit alpha into a depth where
        // max_out is odd. Use 8-bit alpha 1 into 10-bit: 1023/255 =
        // 4.0117… (no half). Instead verify the +mask/2 bias directly:
        // for mask 255, alpha 1, max_out chosen so product*2/(2*mask)
        // straddles .5 — use the general invariant against a brute float.
        for act in [AlphaChannelType::Eight, AlphaChannelType::Sixteen] {
            let mask = act.mask();
            for depth in [
                BitDepth::Eight,
                BitDepth::Ten,
                BitDepth::Twelve,
                BitDepth::Sixteen,
            ] {
                let max_out = depth.max_value();
                for alpha in [1u32, 7, 19, 99, 100, mask / 2, mask / 2 + 1] {
                    let alpha = alpha.min(mask);
                    let exact = (max_out as f64 * alpha as f64) / mask as f64;
                    let want = exact.round() as u16; // ties -> away from zero == up here
                    assert_eq!(
                        alpha_to_sample(alpha as u16, act, depth),
                        want,
                        "round mismatch: {act:?} alpha {alpha} -> {depth:?} (exact {exact})"
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod color_sample_tests {
    //! White-box coverage for the RDD 36 §7.5.1 reconstructed-color to
    //! pixel-sample conversion `color_to_sample`, i.e.
    //! `s = clamp(round(2^b * (v + 256) / 512))`. The end-to-end
    //! `output_range` integration test drives this through a full
    //! encode-then-decode round-trip; these cases lock the pure
    //! arithmetic (per-depth scale, round-to-nearest, and both clamp
    //! arms) directly with no codec round-trip, mirroring
    //! `alpha_sample_tests` for §7.5.2.
    use super::{BitDepth, OutputRange, color_to_sample};

    /// The §7.5.1 mid-range value `v = 0` maps to the centre of the
    /// output range: `2^b * 256 / 512 = 2^(b-1)`. For 8/10/12-bit that
    /// is 128 / 512 / 2048 respectively, in either clamp range.
    #[test]
    fn midpoint_v_zero_maps_to_half_scale() {
        for range in [OutputRange::Full, OutputRange::Video] {
            assert_eq!(color_to_sample(0.0, BitDepth::Eight, range), 128);
            assert_eq!(color_to_sample(0.0, BitDepth::Ten, range), 512);
            assert_eq!(color_to_sample(0.0, BitDepth::Twelve, range), 2048);
            assert_eq!(color_to_sample(0.0, BitDepth::Sixteen, range), 32768);
        }
    }

    /// The bottom of the centred IDCT range (`v = -256`) maps to the
    /// clamp floor: 0 for `Full`, 1 for `Video`. A large positive `v`
    /// saturates to the clamp ceiling: `2^b − 1` for `Full`,
    /// `2^b − 2` for `Video`.
    #[test]
    fn clamp_floor_and_ceiling_track_the_range() {
        // Floor: v = -256 → s = 0 before clamp.
        assert_eq!(
            color_to_sample(-256.0, BitDepth::Eight, OutputRange::Full),
            0
        );
        assert_eq!(
            color_to_sample(-256.0, BitDepth::Eight, OutputRange::Video),
            1
        );
        assert_eq!(color_to_sample(-256.0, BitDepth::Ten, OutputRange::Full), 0);
        assert_eq!(
            color_to_sample(-256.0, BitDepth::Twelve, OutputRange::Video),
            1
        );
        assert_eq!(
            color_to_sample(-256.0, BitDepth::Sixteen, OutputRange::Full),
            0
        );
        assert_eq!(
            color_to_sample(-256.0, BitDepth::Sixteen, OutputRange::Video),
            1
        );

        // Ceiling: a large positive v saturates to the top of the range.
        assert_eq!(
            color_to_sample(1000.0, BitDepth::Eight, OutputRange::Full),
            255
        );
        assert_eq!(
            color_to_sample(1000.0, BitDepth::Eight, OutputRange::Video),
            254
        );
        assert_eq!(
            color_to_sample(1000.0, BitDepth::Ten, OutputRange::Full),
            1023
        );
        assert_eq!(
            color_to_sample(1000.0, BitDepth::Ten, OutputRange::Video),
            1022
        );
        assert_eq!(
            color_to_sample(1000.0, BitDepth::Twelve, OutputRange::Full),
            4095
        );
        assert_eq!(
            color_to_sample(1000.0, BitDepth::Twelve, OutputRange::Video),
            4094
        );
        assert_eq!(
            color_to_sample(1000.0, BitDepth::Sixteen, OutputRange::Full),
            65535
        );
        assert_eq!(
            color_to_sample(1000.0, BitDepth::Sixteen, OutputRange::Video),
            65534
        );
    }

    /// Round-to-nearest: at 8-bit the scale is 0.5, so `v = 1` →
    /// `(1 + 256) * 0.5 = 128.5` → rounds to 129 (ties away from zero),
    /// and `v = -1` → `127.5` → 128.
    #[test]
    fn round_to_nearest_at_eight_bit() {
        assert_eq!(
            color_to_sample(1.0, BitDepth::Eight, OutputRange::Full),
            129
        );
        assert_eq!(
            color_to_sample(-1.0, BitDepth::Eight, OutputRange::Full),
            128
        );
        // v = 2 → 129.0 exactly, no tie.
        assert_eq!(
            color_to_sample(2.0, BitDepth::Eight, OutputRange::Full),
            129
        );
    }

    /// The `Video` clamp differs from `Full` only at the two extreme
    /// codes; every interior value is identical between the two ranges.
    #[test]
    fn video_equals_full_away_from_the_extremes() {
        for &v in &[-200.0f32, -64.0, -1.0, 0.0, 1.0, 64.0, 200.0] {
            for depth in [
                BitDepth::Eight,
                BitDepth::Ten,
                BitDepth::Twelve,
                BitDepth::Sixteen,
            ] {
                assert_eq!(
                    color_to_sample(v, depth, OutputRange::Full),
                    color_to_sample(v, depth, OutputRange::Video),
                    "interior v={v} must match across ranges at {depth:?}"
                );
            }
        }
    }
}

#[cfg(test)]
mod fast_conversion_tests {
    //! The conversions the slice decoder uses are restatements of the
    //! reference ones above; hold them to the same results.
    use super::{
        AlphaChannelType, BitDepth, OutputRange, SampleCodes, alpha_code, alpha_to_sample,
        color_to_sample,
    };

    #[test]
    fn sample_codes_match_the_reference_conversion() {
        for depth in [
            BitDepth::Eight,
            BitDepth::Ten,
            BitDepth::Twelve,
            BitDepth::Sixteen,
        ] {
            for range in [OutputRange::Full, OutputRange::Video] {
                let (nmin, nmax) = range.bounds(depth);
                let codes = SampleCodes {
                    scale: (1u32 << depth.bits()) as f32 / 512.0,
                    nmin: nmin as f32,
                    nmax: nmax as f32,
                };
                // Every sixteenth of a code step across and beyond the
                // nominal range, plus far outliers.
                let mut v = -300.0f32;
                while v < 300.0 {
                    assert_eq!(
                        codes.code(v) as u32,
                        color_to_sample(v, depth, range),
                        "{depth:?} {range:?} v = {v}"
                    );
                    v += 1.0 / 64.0;
                }
                for v in [-1.0e9f32, 1.0e9, -256.0, 255.999_98, 0.0] {
                    assert_eq!(codes.code(v) as u32, color_to_sample(v, depth, range));
                }
            }
        }
    }

    #[test]
    fn alpha_codes_match_the_reference_conversion() {
        for act in [AlphaChannelType::Eight, AlphaChannelType::Sixteen] {
            let top = act.mask() as u16;
            for depth in [
                BitDepth::Eight,
                BitDepth::Ten,
                BitDepth::Twelve,
                BitDepth::Sixteen,
            ] {
                for alpha in 0..=top {
                    assert_eq!(
                        alpha_code(alpha, act, depth),
                        alpha_to_sample(alpha, act, depth),
                        "{act:?} {depth:?} alpha {alpha}"
                    );
                }
            }
        }
    }
}
