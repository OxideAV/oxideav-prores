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
//! 8-bit output, 16-bit LE for 10/12-bit). The pixel-format enum still
//! reports `Yuv4..P*` because the core `PixelFormat` does not yet carry
//! `Yuva422P`/`Yuva444P` variants — callers detect alpha by checking
//! `frame.planes.len() == 4`.

use oxideav_core::frame::VideoPlane;
use oxideav_core::Decoder;
use oxideav_core::{
    CodecId, CodecParameters, Error, Frame, Packet, PixelFormat, Result, VideoFrame,
};

use crate::alpha::{decode_scanned_alpha, AlphaChannelType};
use crate::dct::idct8x8;
use crate::frame::{
    compute_slice_sizes, parse_frame, parse_picture_header, parse_slice_header, ChromaFormat,
    FrameHeader,
};
use crate::quant::qscale;
use crate::slice::{
    blocks_per_mb, chroma_blocks_per_mb, decode_slice_components, LUMA_BLOCKS_PER_MB,
};

const MB_SIDE_PX: usize = 16;

/// Internal cap on a single decoded frame's plane allocation. Bounds
/// against header-declared dimensions: a malicious header with
/// 1M×1M pixels must NOT trigger a multi-TB allocation. 32k×32k YUV444
/// (3 GB) is far more than any sane container; we cap at that.
const MAX_DECODED_PIXELS: usize = 32_768 * 32_768;

/// Per-component output bit depth.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum BitDepth {
    /// 8-bit planar output (`Yuv422P` / `Yuv444P`). One byte per sample.
    Eight,
    /// 10-bit planar output (`Yuv422P10Le` / `Yuv444P10Le`). Two bytes
    /// per sample, low-byte-first; valid range `0..=1023`.
    Ten,
    /// 12-bit planar output (`Yuv422P12Le` / `Yuv444P12Le`). Two bytes
    /// per sample, low-byte-first; valid range `0..=4095`.
    Twelve,
}

impl BitDepth {
    pub fn bytes_per_sample(self) -> usize {
        match self {
            Self::Eight => 1,
            Self::Ten | Self::Twelve => 2,
        }
    }

    pub fn max_value(self) -> u32 {
        match self {
            Self::Eight => 255,
            Self::Ten => 1023,
            Self::Twelve => 4095,
        }
    }

    /// Bit depth as the integer `b` from RDD 36 §7.5.1.
    pub fn bits(self) -> u32 {
        match self {
            Self::Eight => 8,
            Self::Ten => 10,
            Self::Twelve => 12,
        }
    }
}

/// Resolve the requested output bit depth + chroma format from a caller's
/// `CodecParameters::pixel_format`. Returns `Ok(None)` if no pixel
/// format was set — caller defaults to 8-bit at decode time, derives
/// chroma from the frame header.
fn pick_output_format(params: &CodecParameters) -> Result<Option<(BitDepth, ChromaFormat)>> {
    let Some(pf) = params.pixel_format else {
        return Ok(None);
    };
    Ok(Some(match pf {
        PixelFormat::Yuv422P => (BitDepth::Eight, ChromaFormat::Y422),
        PixelFormat::Yuv444P => (BitDepth::Eight, ChromaFormat::Y444),
        PixelFormat::Yuv422P10Le => (BitDepth::Ten, ChromaFormat::Y422),
        PixelFormat::Yuv444P10Le => (BitDepth::Ten, ChromaFormat::Y444),
        PixelFormat::Yuv422P12Le => (BitDepth::Twelve, ChromaFormat::Y422),
        PixelFormat::Yuv444P12Le => (BitDepth::Twelve, ChromaFormat::Y444),
        other => {
            return Err(Error::unsupported(format!(
                "prores decoder: requested pixel_format {other:?} \
                 not supported (expected Yuv4(2|4)4P / Yuv4(2|4)4P10Le / Yuv4(2|4)4P12Le)"
            )));
        }
    }))
}

pub fn make_decoder(params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    let requested = pick_output_format(params)?;
    Ok(Box::new(ProResDecoder {
        codec_id: params.codec_id.clone(),
        requested,
        pending: None,
        eof: false,
    }))
}

struct ProResDecoder {
    codec_id: CodecId,
    /// Caller-requested output (bit-depth, chroma) pair. `None` means
    /// "infer from the frame header, default to 8-bit".
    requested: Option<(BitDepth, ChromaFormat)>,
    pending: Option<Packet>,
    eof: bool,
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
        let vf = decode_packet_with_depth(&pkt.data, pkt.pts, self.requested)?;
        Ok(Frame::Video(vf))
    }

    fn flush(&mut self) -> Result<()> {
        self.eof = true;
        Ok(())
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
pub fn decode_packet_with_depth(
    data: &[u8],
    pts: Option<i64>,
    requested: Option<(BitDepth, ChromaFormat)>,
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

    // Allocate full-frame padded planes. For interlaced the two field
    // pictures decode into rows {0, 2, 4, …} (top) and {1, 3, 5, …}
    // (bottom) of these buffers per RDD 36 §7.5.3.
    let padded_w = mbs_x * MB_SIDE_PX;
    let padded_c_w = match chroma {
        ChromaFormat::Y422 => padded_w / 2,
        ChromaFormat::Y444 => padded_w,
    };
    // Padded frame height: round up to a multiple of MB_SIDE_PX, then
    // (for interlaced) ensure each field gets a whole MB row count by
    // rounding the per-field MB-row count up.
    let padded_frame_h = if interlaced {
        let top_h = height.div_ceil(2);
        let bot_h = height / 2;
        let top_mb_rows = top_h.div_ceil(MB_SIDE_PX);
        let bot_mb_rows = bot_h.div_ceil(MB_SIDE_PX);
        // Interleaved padded height = max field padded * 2.
        top_mb_rows.max(bot_mb_rows) * MB_SIDE_PX * 2
    } else {
        height.div_ceil(MB_SIDE_PX) * MB_SIDE_PX
    };
    let y_alloc = padded_w.saturating_mul(padded_frame_h);
    let c_alloc = padded_c_w.saturating_mul(padded_frame_h);
    if y_alloc > MAX_DECODED_PIXELS || c_alloc > MAX_DECODED_PIXELS {
        return Err(Error::invalid("prores: padded plane size exceeds cap"));
    }
    let y_byte_stride = padded_w * bps;
    let c_byte_stride = padded_c_w * bps;
    let a_byte_stride = padded_w * bps;
    let mut y_plane = vec![0u8; y_byte_stride * padded_frame_h];
    let mut cb_plane = vec![0u8; c_byte_stride * padded_frame_h];
    let mut cr_plane = vec![0u8; c_byte_stride * padded_frame_h];
    let mut a_plane: Vec<u8> = if has_alpha {
        vec![0u8; a_byte_stride * padded_frame_h]
    } else {
        Vec::new()
    };

    let mut cursor = after_frame;
    if interlaced {
        // Per §6.2 picture_vertical_size:
        //   topFieldVerticalSize = (vertical_size + 1) / 2
        //   bottomFieldVerticalSize = vertical_size / 2
        // interlace_mode==1 → first picture is top field
        // interlace_mode==2 → first picture is bottom field
        let top_h = height.div_ceil(2);
        let bot_h = height / 2;
        let (first_h, first_field_offset, second_h, second_field_offset) = match fh.interlace_mode {
            1 => (top_h, 0usize, bot_h, 1usize),
            2 => (bot_h, 1usize, top_h, 0usize),
            other => {
                return Err(Error::invalid(format!(
                    "prores: invalid interlace_mode {other}"
                )));
            }
        };
        cursor = decode_picture_into_planes(
            cursor,
            &fh,
            mbs_x,
            first_h,
            chroma,
            bit_depth,
            alpha_kind,
            true,
            FieldStride::new(2, first_field_offset),
            &mut y_plane,
            y_byte_stride,
            &mut cb_plane,
            c_byte_stride,
            &mut cr_plane,
            &mut a_plane,
            a_byte_stride,
        )?;
        cursor = decode_picture_into_planes(
            cursor,
            &fh,
            mbs_x,
            second_h,
            chroma,
            bit_depth,
            alpha_kind,
            true,
            FieldStride::new(2, second_field_offset),
            &mut y_plane,
            y_byte_stride,
            &mut cb_plane,
            c_byte_stride,
            &mut cr_plane,
            &mut a_plane,
            a_byte_stride,
        )?;
    } else {
        cursor = decode_picture_into_planes(
            cursor,
            &fh,
            mbs_x,
            height,
            chroma,
            bit_depth,
            alpha_kind,
            false,
            FieldStride::progressive(),
            &mut y_plane,
            y_byte_stride,
            &mut cb_plane,
            c_byte_stride,
            &mut cr_plane,
            &mut a_plane,
            a_byte_stride,
        )?;
    }
    let _ = cursor; // remaining bytes are stuffing per §5.1 / §6.1.2

    // Crop padded buffers to declared picture size.
    let c_w = match chroma {
        ChromaFormat::Y422 => width.div_ceil(2),
        ChromaFormat::Y444 => width,
    };
    let y_cropped = crop_plane(&y_plane, y_byte_stride, width, height, bps);
    let cb_cropped = crop_plane(&cb_plane, c_byte_stride, c_w, height, bps);
    let cr_cropped = crop_plane(&cr_plane, c_byte_stride, c_w, height, bps);
    let mut planes = vec![
        VideoPlane {
            stride: width * bps,
            data: y_cropped,
        },
        VideoPlane {
            stride: c_w * bps,
            data: cb_cropped,
        },
        VideoPlane {
            stride: c_w * bps,
            data: cr_cropped,
        },
    ];
    if has_alpha {
        let a_cropped = crop_plane(&a_plane, a_byte_stride, width, height, bps);
        planes.push(VideoPlane {
            stride: width * bps,
            data: a_cropped,
        });
    }
    Ok(VideoFrame { pts, planes })
}

/// Field-row mapping. For progressive: `step=1, offset=0` (rows are
/// contiguous). For interlaced: `step=2` and `offset` is 0 (top field)
/// or 1 (bottom field). When pasting block sample row `r`, the actual
/// frame row is `step * r + offset`.
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

/// Decode one `picture()` (header + slice table + slice payloads) into
/// the supplied padded plane buffers. `picture_height` is the picture's
/// luma-sample height (= field height for interlaced, full height
/// otherwise). `field` controls how picture-row indices map onto
/// destination plane rows; for progressive pictures it is the identity.
///
/// Returns a slice positioned just past the consumed picture bytes —
/// the caller advances its cursor with the returned slice.
#[allow(clippy::too_many_arguments)]
fn decode_picture_into_planes<'a>(
    data: &'a [u8],
    fh: &FrameHeader,
    mbs_x: usize,
    picture_height: usize,
    chroma: ChromaFormat,
    bit_depth: BitDepth,
    alpha_kind: Option<AlphaChannelType>,
    interlaced: bool,
    field: FieldStride,
    y_plane: &mut [u8],
    y_byte_stride: usize,
    cb_plane: &mut [u8],
    c_byte_stride: usize,
    cr_plane: &mut [u8],
    a_plane: &mut [u8],
    a_byte_stride: usize,
) -> Result<&'a [u8]> {
    let has_alpha = alpha_kind.is_some();
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
    let mut slice_sizes = Vec::with_capacity(expected_slice_count);
    for i in 0..expected_slice_count {
        let off = i * 2;
        slice_sizes.push(u16::from_be_bytes(after_pic[off..off + 2].try_into().unwrap()) as usize);
    }
    let mut cursor = &after_pic[slice_table_bytes..];
    let total_picture_bytes: usize =
        ph.picture_header_size as usize + slice_table_bytes + slice_sizes.iter().sum::<usize>();
    if (total_picture_bytes as u32) != ph.picture_size {
        return Err(Error::invalid(
            "prores: picture_size mismatch with header + slice table + payloads",
        ));
    }
    if data.len() < total_picture_bytes {
        return Err(Error::invalid("prores: picture overruns buffer"));
    }

    const LUMA_OFFSETS: [(usize, usize); 4] = [(0, 0), (1, 0), (0, 1), (1, 1)];
    let chroma_offsets: &[(usize, usize)] = match chroma {
        ChromaFormat::Y422 => &[(0, 0), (0, 1)],
        ChromaFormat::Y444 => &LUMA_OFFSETS,
    };
    let cb_per_mb = chroma_blocks_per_mb(chroma);
    let per_mb = blocks_per_mb(chroma);

    let mut slice_idx = 0usize;
    for my in 0..mbs_y {
        let mut mx = 0usize;
        for &slice_size_in_mb_template in &slice_sizes_template {
            let mbs_this_slice = slice_size_in_mb_template.min(mbs_x - mx);
            if mbs_this_slice == 0 {
                break;
            }
            let coded_size = slice_sizes[slice_idx];
            if cursor.len() < coded_size {
                return Err(Error::invalid("prores: slice payload truncated"));
            }
            let slice_data = &cursor[..coded_size];
            cursor = &cursor[coded_size..];
            slice_idx += 1;

            let (sh, after_sh) = parse_slice_header(slice_data, has_alpha)?;
            let coded_y = sh.coded_size_of_y_data as usize;
            let coded_cb = sh.coded_size_of_cb_data as usize;
            let cr_data_size = if let Some(sz) = sh.coded_size_of_cr_data {
                sz as usize
            } else {
                slice_data
                    .len()
                    .checked_sub(sh.slice_header_size as usize + coded_y + coded_cb)
                    .ok_or_else(|| Error::invalid("prores: cr_data size underflow"))?
            };
            if after_sh.len() < coded_y + coded_cb + cr_data_size {
                return Err(Error::invalid("prores: slice components truncated"));
            }
            let y_data = &after_sh[..coded_y];
            let cb_data = &after_sh[coded_y..coded_y + coded_cb];
            let cr_data = &after_sh[coded_y + coded_cb..coded_y + coded_cb + cr_data_size];
            let alpha_data: &[u8] = if has_alpha {
                &after_sh[coded_y + coded_cb + cr_data_size..]
            } else {
                &[]
            };

            let blocks = decode_slice_components(
                y_data,
                cb_data,
                cr_data,
                mbs_this_slice,
                chroma,
                interlaced,
            )?;
            if blocks.len() != mbs_this_slice * per_mb {
                return Err(Error::invalid(
                    "prores: decoded block count mismatch in slice",
                ));
            }

            for mb_within in 0..mbs_this_slice {
                let mb_x = mx + mb_within;
                let base = mb_within * per_mb;
                for (i, (bx, by)) in LUMA_OFFSETS.iter().enumerate() {
                    let mut blk_f =
                        dequant_to_f32(&blocks[base + i], &fh.luma_qmat, sh.quantization_index);
                    idct8x8(&mut blk_f);
                    paste_block(
                        y_plane,
                        y_byte_stride,
                        mb_x * MB_SIDE_PX + bx * 8,
                        my * MB_SIDE_PX + by * 8,
                        &blk_f,
                        bit_depth,
                        field,
                    );
                }
                for (i, (bx, by)) in chroma_offsets.iter().enumerate() {
                    let mut blk_f = dequant_to_f32(
                        &blocks[base + LUMA_BLOCKS_PER_MB + i],
                        &fh.chroma_qmat,
                        sh.quantization_index,
                    );
                    idct8x8(&mut blk_f);
                    let (x0, y0) = match chroma {
                        ChromaFormat::Y422 => (mb_x * 8, my * MB_SIDE_PX + by * 8),
                        ChromaFormat::Y444 => {
                            (mb_x * MB_SIDE_PX + bx * 8, my * MB_SIDE_PX + by * 8)
                        }
                    };
                    paste_block(cb_plane, c_byte_stride, x0, y0, &blk_f, bit_depth, field);
                }
                for (i, (bx, by)) in chroma_offsets.iter().enumerate() {
                    let mut blk_f = dequant_to_f32(
                        &blocks[base + LUMA_BLOCKS_PER_MB + cb_per_mb + i],
                        &fh.chroma_qmat,
                        sh.quantization_index,
                    );
                    idct8x8(&mut blk_f);
                    let (x0, y0) = match chroma {
                        ChromaFormat::Y422 => (mb_x * 8, my * MB_SIDE_PX + by * 8),
                        ChromaFormat::Y444 => {
                            (mb_x * MB_SIDE_PX + bx * 8, my * MB_SIDE_PX + by * 8)
                        }
                    };
                    paste_block(cr_plane, c_byte_stride, x0, y0, &blk_f, bit_depth, field);
                }
            }
            if let Some(act) = alpha_kind {
                let slice_vertical_size = if my < mbs_y - 1 {
                    MB_SIDE_PX
                } else {
                    picture_height - my * MB_SIDE_PX
                };
                let cols = MB_SIDE_PX * mbs_this_slice;
                let num_alpha_values = cols * slice_vertical_size;
                let alpha_values = decode_scanned_alpha(alpha_data, num_alpha_values, act)?;
                paste_alpha(
                    a_plane,
                    a_byte_stride,
                    mx * MB_SIDE_PX,
                    my * MB_SIDE_PX,
                    cols,
                    slice_vertical_size,
                    &alpha_values,
                    act,
                    bit_depth,
                    field,
                );
            }
            mx += mbs_this_slice;
        }
    }
    Ok(&data[total_picture_bytes..])
}

fn dequant_to_f32(blk: &[i32; 64], qmat: &[u8; 64], quantization_index: u8) -> [f32; 64] {
    // F[v][u] = (QF[v][u] * W[v][u] * qScale) / 8
    let qs = qscale(quantization_index) as f32;
    let mut out = [0.0f32; 64];
    for k in 0..64 {
        out[k] = (blk[k] as f32 * qmat[k] as f32 * qs) / 8.0;
    }
    out
}

/// Write an 8x8 block of IDCT output `blk` into `plane` at sample
/// position `(x0, y0)`. `byte_stride` is the row stride of `plane` in
/// **bytes** (i.e. samples_per_row * bytes_per_sample).
///
/// Per RDD 36 §7.5.1, the IDCT output `v` is centred in the half-open
/// range `[-256, 256)` (9-bit signed integers). The pixel sample is
/// `s = clamp(round(2^b * (v + 256) / 512))`. For 8-bit that simplifies
/// to `s = clamp(round((v + 256) / 2))`; for 10-bit `s = (v+256) * 2`;
/// for 12-bit `s = (v+256) * 8`.
fn paste_block(
    plane: &mut [u8],
    byte_stride: usize,
    x0: usize,
    y0: usize,
    blk: &[f32; 64],
    bit_depth: BitDepth,
    field: FieldStride,
) {
    let scale = match bit_depth {
        BitDepth::Eight => 0.5,
        BitDepth::Ten => 2.0,
        BitDepth::Twelve => 8.0,
    };
    let max_u32 = bit_depth.max_value();
    match bit_depth {
        BitDepth::Eight => {
            for j in 0..8 {
                let row = field.map(y0 + j);
                for i in 0..8 {
                    let v = (blk[j * 8 + i] + 256.0) * scale;
                    let px = if v <= 0.0 {
                        0
                    } else if v >= max_u32 as f32 {
                        max_u32 as u8
                    } else {
                        v.round() as u8
                    };
                    plane[row * byte_stride + x0 + i] = px;
                }
            }
        }
        BitDepth::Ten | BitDepth::Twelve => {
            let max = max_u32 as f32;
            for j in 0..8 {
                let row = field.map(y0 + j);
                for i in 0..8 {
                    let v = (blk[j * 8 + i] + 256.0) * scale;
                    let px: u16 = if v <= 0.0 {
                        0
                    } else if v >= max {
                        max_u32 as u16
                    } else {
                        v.round() as u16
                    };
                    let off = row * byte_stride + (x0 + i) * 2;
                    plane[off] = (px & 0xFF) as u8;
                    plane[off + 1] = (px >> 8) as u8;
                }
            }
        }
    }
}

/// Convert a decoded alpha value (8-bit in `0..=255` or 16-bit in
/// `0..=65535` per `act`) to a pixel sample of the requested output bit
/// depth, per RDD 36 §7.5.2:
///
/// `alphaSample = round((2^b - 1) * alpha / mask)`
fn alpha_to_sample(alpha: u16, act: AlphaChannelType, out_depth: BitDepth) -> u16 {
    let max_out = out_depth.max_value();
    let mask = act.mask();
    // round((max_out * alpha) / mask) using integer math.
    let num = max_out as u64 * alpha as u64;
    let denom = mask as u64;
    ((num + denom / 2) / denom) as u16
}

/// Paste a slice's worth of decoded alpha values into the padded alpha
/// plane. `cols`/`rows` describe the alpha array shape (in samples).
#[allow(clippy::too_many_arguments)]
fn paste_alpha(
    plane: &mut [u8],
    byte_stride: usize,
    x0: usize,
    y0: usize,
    cols: usize,
    rows: usize,
    values: &[u16],
    act: AlphaChannelType,
    out_depth: BitDepth,
    field: FieldStride,
) {
    debug_assert_eq!(values.len(), cols * rows);
    let bps = out_depth.bytes_per_sample();
    for r in 0..rows {
        let row = field.map(y0 + r);
        for c in 0..cols {
            let s = alpha_to_sample(values[r * cols + c], act, out_depth);
            let off = row * byte_stride + (x0 + c) * bps;
            match out_depth {
                BitDepth::Eight => {
                    plane[off] = s as u8;
                }
                BitDepth::Ten | BitDepth::Twelve => {
                    plane[off] = (s & 0xFF) as u8;
                    plane[off + 1] = (s >> 8) as u8;
                }
            }
        }
    }
}

/// Crop a padded plane to a tight (`dst_w` samples × `dst_h` rows)
/// output buffer. `bps` is the per-sample byte count (1 for 8-bit,
/// 2 for 10-bit packed LE).
fn crop_plane(
    src: &[u8],
    src_byte_stride: usize,
    dst_w: usize,
    dst_h: usize,
    bps: usize,
) -> Vec<u8> {
    let dst_byte_stride = dst_w * bps;
    let mut out = vec![0u8; dst_byte_stride * dst_h];
    for y in 0..dst_h {
        out[y * dst_byte_stride..y * dst_byte_stride + dst_byte_stride]
            .copy_from_slice(&src[y * src_byte_stride..y * src_byte_stride + dst_byte_stride]);
    }
    out
}
