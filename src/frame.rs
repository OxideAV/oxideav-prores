//! ProRes frame, picture, and slice headers (SMPTE RDD 36 §5).
//!
//! Wire layout (big-endian throughout):
//!
//! ```text
//! frame() {
//!   frame_size:        u32  // total bytes including frame_size itself
//!   frame_identifier:  4*u8 // 'icpf' (0x69637066)
//!   frame_header():
//!     frame_header_size:    u16
//!     reserved:              u8
//!     bitstream_version:     u8  // 0 (4:2:2 only) or 1 (4:2:2 / 4:4:4 + alpha)
//!     encoder_identifier:  4*u8
//!     horizontal_size:      u16
//!     vertical_size:        u16
//!     chroma_format:       u2 + reserved u2  // 2=4:2:2, 3=4:4:4
//!     interlace_mode:      u2 + reserved u2
//!     aspect_ratio_info:   u4
//!     frame_rate_code:     u4
//!     color_primaries:      u8
//!     transfer_charac:      u8
//!     matrix_coefficients:  u8
//!     reserved:            u4
//!     alpha_channel_type:  u4
//!     reserved:           u14
//!     load_luma_qmat:      u1
//!     load_chroma_qmat:    u1
//!     [luma_qmat 64*u8]   if load_luma_qmat
//!     [chroma_qmat 64*u8] if load_chroma_qmat
//!   picture():
//!     picture_header():
//!       picture_header_size: u5  // in bytes (= 8 normally)
//!       reserved:           u3
//!       picture_size:      u32  // bytes incl. picture header
//!       deprecated_n_slc:  u16
//!       reserved:           u2
//!       log2_desired_slice_size_in_mb: u2
//!       reserved:           u4
//!     slice_table():
//!       coded_size_of_slice[]: u16 each
//!     slice() ...
//! }
//! ```
//!
//! The `picture_header_size` field is in **bytes** (per §5.2.1), and the
//! 5 bits used to encode it makes the maximum picture_header 31 bytes
//! long. The standard layout above totals 8 bytes — the value emitted
//! is therefore 8.

use oxideav_core::{Error, Result};

/// 'icpf' magic, big-endian. Spec value: 0x69637066.
pub const FRAME_IDENTIFIER: &[u8; 4] = b"icpf";

/// 'oxav' encoder identifier — the four-character code we emit when
/// producing ProRes frames.
pub const ENCODER_IDENTIFIER: &[u8; 4] = b"oxav";

pub const CHROMA_FMT_422_CODE: u8 = 2;
pub const CHROMA_FMT_444_CODE: u8 = 3;

/// Chroma sampling format for a ProRes picture.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ChromaFormat {
    /// 4:2:2 — 2 Cb + 2 Cr blocks per macroblock.
    Y422,
    /// 4:4:4 — 4 Cb + 4 Cr blocks per macroblock.
    Y444,
}

impl ChromaFormat {
    pub fn from_code(c: u8) -> Result<Self> {
        match c {
            CHROMA_FMT_422_CODE => Ok(Self::Y422),
            CHROMA_FMT_444_CODE => Ok(Self::Y444),
            other => Err(Error::unsupported(format!(
                "prores: chroma_format {other} not supported"
            ))),
        }
    }

    pub fn code(self) -> u8 {
        match self {
            Self::Y422 => CHROMA_FMT_422_CODE,
            Self::Y444 => CHROMA_FMT_444_CODE,
        }
    }
}

/// Profile inferred from the container FourCC. Not a bitstream syntax
/// element — RDD 36 frames carry only `chroma_format`, not a profile
/// code. We keep it for the encoder API and to set sensible defaults.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Profile {
    Proxy,
    Lt,
    Standard,
    Hq,
    Prores4444,
    Prores4444Xq,
}

impl Profile {
    pub fn fourcc(self) -> &'static [u8; 4] {
        match self {
            Profile::Proxy => b"apco",
            Profile::Lt => b"apcs",
            Profile::Standard => b"apcn",
            Profile::Hq => b"apch",
            Profile::Prores4444 => b"ap4h",
            Profile::Prores4444Xq => b"ap4x",
        }
    }

    pub fn chroma_format(self) -> ChromaFormat {
        match self {
            Profile::Proxy | Profile::Lt | Profile::Standard | Profile::Hq => ChromaFormat::Y422,
            Profile::Prores4444 | Profile::Prores4444Xq => ChromaFormat::Y444,
        }
    }

    /// Default `quantization_index` used by the encoder when the caller
    /// does not specify one. Lower index → higher quality + larger packets.
    pub fn default_quant_index(self) -> u8 {
        match self {
            Profile::Proxy => 8,
            Profile::Lt => 6,
            Profile::Standard => 4,
            Profile::Hq => 2,
            Profile::Prores4444 => 2,
            Profile::Prores4444Xq => 1,
        }
    }
}

/// Parsed RDD 36 frame header. Fields the rest of the decoder needs.
#[derive(Clone, Debug)]
pub struct FrameHeader {
    pub frame_size: u32,
    pub frame_header_size: u16,
    pub bitstream_version: u8,
    pub width: u16,
    pub height: u16,
    pub chroma_format: ChromaFormat,
    pub interlace_mode: u8,
    pub aspect_ratio_information: u8,
    pub frame_rate_code: u8,
    pub color_primaries: u8,
    pub transfer_characteristic: u8,
    pub matrix_coefficients: u8,
    pub alpha_channel_type: u8,
    pub luma_qmat: [u8; 64],
    pub chroma_qmat: [u8; 64],
}

impl FrameHeader {
    pub fn picture_count(&self) -> u32 {
        if self.interlace_mode == 0 {
            1
        } else {
            2
        }
    }
}

/// Parse the frame() syntax (frame_size + 'icpf' + frame_header()).
/// Returns the parsed header and the remaining bytes (everything after
/// the frame header, i.e. starting at the first picture()).
pub fn parse_frame(data: &[u8]) -> Result<(FrameHeader, &[u8])> {
    if data.len() < 8 {
        return Err(Error::invalid("prores: frame truncated (need 8 bytes)"));
    }
    let frame_size = u32::from_be_bytes(data[0..4].try_into().unwrap());
    if &data[4..8] != FRAME_IDENTIFIER {
        return Err(Error::invalid("prores: frame magic mismatch (not 'icpf')"));
    }
    if (frame_size as usize) > data.len() {
        return Err(Error::invalid(
            "prores: frame_size exceeds available buffer",
        ));
    }
    let frame_data = &data[..frame_size as usize];
    let after_magic = &frame_data[8..];
    let (fh, after_fh) = parse_frame_header(after_magic)?;
    Ok((fh, after_fh))
}

/// Parse just the frame_header() block (assumes `data` starts at the
/// frame_header_size field; i.e. caller has already consumed the 8
/// bytes of frame_size + 'icpf').
pub fn parse_frame_header(data: &[u8]) -> Result<(FrameHeader, &[u8])> {
    if data.len() < 20 {
        return Err(Error::invalid("prores: frame header truncated"));
    }
    let frame_header_size = u16::from_be_bytes(data[0..2].try_into().unwrap());
    if (frame_header_size as usize) < 20 || (frame_header_size as usize) > data.len() {
        return Err(Error::invalid("prores: bad frame_header_size"));
    }
    let _reserved = data[2];
    let bitstream_version = data[3];
    if bitstream_version > 1 {
        return Err(Error::unsupported(format!(
            "prores: unsupported bitstream_version {bitstream_version}"
        )));
    }
    let _enc_id = &data[4..8];
    let width = u16::from_be_bytes(data[8..10].try_into().unwrap());
    let height = u16::from_be_bytes(data[10..12].try_into().unwrap());
    // byte 12: chroma_format (u2) + reserved (u2) + interlace_mode (u2) + reserved (u2)
    let b12 = data[12];
    let chroma_code = (b12 >> 6) & 0x3;
    let interlace_mode = (b12 >> 2) & 0x3;
    let chroma_format = ChromaFormat::from_code(chroma_code)?;
    // byte 13: aspect_ratio_information (u4) + frame_rate_code (u4)
    let b13 = data[13];
    let aspect_ratio_information = (b13 >> 4) & 0xF;
    let frame_rate_code = b13 & 0xF;
    let color_primaries = data[14];
    let transfer_characteristic = data[15];
    let matrix_coefficients = data[16];
    // byte 17: reserved u4 + alpha_channel_type u4
    let b17 = data[17];
    let alpha_channel_type = b17 & 0xF;
    // bytes 18..20: 14 reserved + load_luma + load_chroma (the last two bits)
    let b19 = data[19];
    let load_luma = (b19 >> 1) & 1;
    let load_chroma = b19 & 1;

    let mut luma_qmat = [4u8; 64];
    let mut chroma_qmat = [4u8; 64];
    let mut cursor = 20usize;
    if load_luma == 1 {
        if data.len() < cursor + 64 {
            return Err(Error::invalid("prores: luma_qmat truncated"));
        }
        luma_qmat.copy_from_slice(&data[cursor..cursor + 64]);
        cursor += 64;
    }
    if load_chroma == 1 {
        if data.len() < cursor + 64 {
            return Err(Error::invalid("prores: chroma_qmat truncated"));
        }
        chroma_qmat.copy_from_slice(&data[cursor..cursor + 64]);
        cursor += 64;
    } else if load_luma == 1 {
        // Per §7.3: when load_chroma=0 but load_luma=1, chroma uses
        // the loaded luma matrix.
        chroma_qmat = luma_qmat;
    }
    // Skip up to frame_header_size, allowing trailing reserved bytes.
    if cursor > frame_header_size as usize {
        return Err(Error::invalid(
            "prores: frame header parser overran declared size",
        ));
    }
    // We don't use the frame_size from frame() here (parse_frame_header
    // is also called directly in tests). Use 0 as a placeholder — the
    // encoder fills it in.
    Ok((
        FrameHeader {
            frame_size: 0,
            frame_header_size,
            bitstream_version,
            width,
            height,
            chroma_format,
            interlace_mode,
            aspect_ratio_information,
            frame_rate_code,
            color_primaries,
            transfer_characteristic,
            matrix_coefficients,
            alpha_channel_type,
            luma_qmat,
            chroma_qmat,
        },
        &data[frame_header_size as usize..],
    ))
}

/// Optional descriptive metadata fields written into the frame header.
/// All fields are documented in RDD 36 §5.1.1 / §6.2 and default to 0
/// (= "unknown / unspecified") so RDD 36 decoders treat them as a hint
/// only.
#[derive(Clone, Copy, Debug, Default)]
pub struct FrameMeta {
    /// `aspect_ratio_information` per §6.2 / Table 3 (0 = unknown,
    /// 1 = square pixels, 2 = 4:3, 3 = 16:9, 4..=15 = reserved).
    pub aspect_ratio_information: u8,
    /// `frame_rate_code` per §6.2 / Table 4 — see
    /// [`frame_rate_code_from_rational`] for the mapping. 0 = unknown.
    pub frame_rate_code: u8,
    /// `color_primaries` per §6.2 (BT.601, BT.709, etc — uses the same
    /// codes as ISO/IEC 23001-8 / Rec. ITU-T H.273).
    pub color_primaries: u8,
    /// `transfer_characteristic` per §6.2.
    pub transfer_characteristic: u8,
    /// `matrix_coefficients` per §6.2.
    pub matrix_coefficients: u8,
}

impl FrameMeta {
    /// All fields zeroed = "unknown / unspecified". Identical to
    /// `Default::default()`.
    pub fn unknown() -> Self {
        Self::default()
    }

    /// True when every field is 0 ("unknown / unspecified"). Encoder
    /// uses this to detect the no-op case for short-circuit testing.
    pub fn is_unknown(self) -> bool {
        self.aspect_ratio_information == 0
            && self.frame_rate_code == 0
            && self.color_primaries == 0
            && self.transfer_characteristic == 0
            && self.matrix_coefficients == 0
    }
}

/// Map a frame rate (as a [`oxideav_core::Rational`]) to the RDD 36
/// §6.2 / Table 4 `frame_rate_code` (4-bit field). Returns 0
/// ("unknown") for any rate that is not one of the spec's named codes.
///
/// The spec's named rates (rounded to the table values):
///
/// | code | rate                |
/// |------|---------------------|
/// | 1    | 24000 / 1001 (~23.976) |
/// | 2    | 24                  |
/// | 3    | 25                  |
/// | 4    | 30000 / 1001 (~29.97)  |
/// | 5    | 30                  |
/// | 6    | 50                  |
/// | 7    | 60000 / 1001 (~59.94)  |
/// | 8    | 60                  |
/// | 9    | 100                 |
/// | 10   | 120000 / 1001 (~119.88) |
/// | 11   | 120                 |
///
/// `12..=15` are reserved per the spec — never emitted.
pub fn frame_rate_code_from_rational(r: oxideav_core::Rational) -> u8 {
    if r.num <= 0 || r.den <= 0 {
        return 0;
    }
    // Compare against each named rate by integer cross-product
    // (num * d_named == n_named * den) — exact, drift-free for the
    // canonical fractions that ship as TimeBase / Rational pairs.
    let num = r.num as i128;
    let den = r.den as i128;
    let candidates: &[(u8, i128, i128)] = &[
        (1, 24_000, 1001),
        (2, 24, 1),
        (3, 25, 1),
        (4, 30_000, 1001),
        (5, 30, 1),
        (6, 50, 1),
        (7, 60_000, 1001),
        (8, 60, 1),
        (9, 100, 1),
        (10, 120_000, 1001),
        (11, 120, 1),
    ];
    for &(code, n, d) in candidates {
        if num * d == n * den {
            return code;
        }
    }
    0
}

/// Write a complete frame header (frame_size + 'icpf' + frame_header())
/// with `alpha_channel_type` defaulting to 0 and all metadata fields
/// zeroed ("unknown"). Forwards to [`write_frame_with_meta`].
#[allow(clippy::too_many_arguments)]
pub fn write_frame(
    out: &mut Vec<u8>,
    total_frame_size: u32,
    width: u16,
    height: u16,
    chroma_format: ChromaFormat,
    interlace_mode: u8,
    luma_qmat: &[u8; 64],
    chroma_qmat: &[u8; 64],
    load_luma: bool,
    load_chroma: bool,
) {
    write_frame_with_meta(
        out,
        total_frame_size,
        width,
        height,
        chroma_format,
        interlace_mode,
        luma_qmat,
        chroma_qmat,
        load_luma,
        load_chroma,
        0,
        FrameMeta::default(),
    )
}

/// Write a complete frame header with an explicit `alpha_channel_type`
/// code, all other metadata zeroed. Kept for back-compat with callers
/// that don't carry frame_rate / aspect_ratio info; new code should use
/// [`write_frame_with_meta`].
#[allow(clippy::too_many_arguments)]
pub fn write_frame_with_alpha(
    out: &mut Vec<u8>,
    total_frame_size: u32,
    width: u16,
    height: u16,
    chroma_format: ChromaFormat,
    interlace_mode: u8,
    luma_qmat: &[u8; 64],
    chroma_qmat: &[u8; 64],
    load_luma: bool,
    load_chroma: bool,
    alpha_channel_type: u8,
) {
    write_frame_with_meta(
        out,
        total_frame_size,
        width,
        height,
        chroma_format,
        interlace_mode,
        luma_qmat,
        chroma_qmat,
        load_luma,
        load_chroma,
        alpha_channel_type,
        FrameMeta::default(),
    )
}

/// Write a complete frame header with explicit alpha + descriptive
/// metadata (aspect_ratio_information, frame_rate_code,
/// color_primaries, transfer_characteristic, matrix_coefficients).
///
/// `alpha_channel_type == 0` means no alpha plane; values 1 and 2
/// signal 8-bit and 16-bit alpha respectively (see RDD 36 §5.3.3).
/// When `alpha_channel_type != 0` the bitstream version is forced to 1
/// (alpha is a v1 feature per §6.4).
///
/// All `meta` fields are written verbatim into the corresponding header
/// bytes; only the low 4 bits of `aspect_ratio_information` and
/// `frame_rate_code` are honoured (those are u4 fields per §5.1.1).
#[allow(clippy::too_many_arguments)]
pub fn write_frame_with_meta(
    out: &mut Vec<u8>,
    total_frame_size: u32,
    width: u16,
    height: u16,
    chroma_format: ChromaFormat,
    interlace_mode: u8,
    luma_qmat: &[u8; 64],
    chroma_qmat: &[u8; 64],
    load_luma: bool,
    load_chroma: bool,
    alpha_channel_type: u8,
    meta: FrameMeta,
) {
    debug_assert!(alpha_channel_type <= 2);
    // frame_size + magic
    out.extend_from_slice(&total_frame_size.to_be_bytes());
    out.extend_from_slice(FRAME_IDENTIFIER);
    // frame_header()
    let fh_size: u16 = 20 + if load_luma { 64 } else { 0 } + if load_chroma { 64 } else { 0 };
    out.extend_from_slice(&fh_size.to_be_bytes());
    out.push(0); // reserved
    let bitstream_version: u8 = if alpha_channel_type != 0 {
        1
    } else {
        match chroma_format {
            ChromaFormat::Y422 => 0,
            ChromaFormat::Y444 => 1,
        }
    };
    out.push(bitstream_version);
    out.extend_from_slice(ENCODER_IDENTIFIER);
    out.extend_from_slice(&width.to_be_bytes());
    out.extend_from_slice(&height.to_be_bytes());
    // chroma(2) + reserved(2) + interlace(2) + reserved(2)
    out.push((chroma_format.code() << 6) | ((interlace_mode & 0x3) << 2));
    // aspect_ratio_information(4) + frame_rate_code(4)
    out.push(((meta.aspect_ratio_information & 0x0F) << 4) | (meta.frame_rate_code & 0x0F));
    out.push(meta.color_primaries);
    out.push(meta.transfer_characteristic);
    out.push(meta.matrix_coefficients);
    out.push(alpha_channel_type & 0x0F); // reserved(4) + alpha_channel_type(4)
    out.push(0); // reserved (high 8 of the 14)
                 // last byte: 6 reserved bits + load_luma(1) + load_chroma(1)
    let lb = ((load_luma as u8) << 1) | (load_chroma as u8);
    out.push(lb);
    if load_luma {
        out.extend_from_slice(luma_qmat);
    }
    if load_chroma {
        out.extend_from_slice(chroma_qmat);
    }
}

/// Parsed picture_header().
#[derive(Clone, Debug)]
pub struct PictureHeader {
    pub picture_header_size: u8,
    pub picture_size: u32,
    pub deprecated_number_of_slices: u16,
    pub log2_desired_slice_size_in_mb: u8,
}

/// Parse a picture_header(). Returns the header and a slice that begins
/// at the slice_table().
pub fn parse_picture_header(data: &[u8]) -> Result<(PictureHeader, &[u8])> {
    if data.len() < 8 {
        return Err(Error::invalid("prores: picture header truncated"));
    }
    // byte 0: picture_header_size(5) + reserved(3)
    let b0 = data[0];
    let picture_header_size = (b0 >> 3) & 0x1F;
    if picture_header_size < 8 {
        return Err(Error::invalid("prores: picture_header_size < 8"));
    }
    let picture_size = u32::from_be_bytes(data[1..5].try_into().unwrap());
    let deprecated_number_of_slices = u16::from_be_bytes(data[5..7].try_into().unwrap());
    // byte 7: reserved(2) + log2_desired_slice_size_in_mb(2) + reserved(4)
    let b7 = data[7];
    let log2_desired_slice_size_in_mb = (b7 >> 4) & 0x3;
    if data.len() < picture_header_size as usize {
        return Err(Error::invalid("prores: picture header overruns buffer"));
    }
    Ok((
        PictureHeader {
            picture_header_size,
            picture_size,
            deprecated_number_of_slices,
            log2_desired_slice_size_in_mb,
        },
        &data[picture_header_size as usize..],
    ))
}

pub fn write_picture_header(
    out: &mut Vec<u8>,
    picture_size: u32,
    deprecated_number_of_slices: u16,
    log2_desired_slice_size_in_mb: u8,
) {
    let picture_header_size: u8 = 8;
    out.push(picture_header_size << 3);
    out.extend_from_slice(&picture_size.to_be_bytes());
    out.extend_from_slice(&deprecated_number_of_slices.to_be_bytes());
    // reserved(2) + log2(2) + reserved(4)
    out.push((log2_desired_slice_size_in_mb & 0x3) << 4);
}

/// Parsed slice_header().
#[derive(Clone, Debug)]
pub struct SliceHeader {
    pub slice_header_size: u8,
    pub quantization_index: u8,
    pub coded_size_of_y_data: u16,
    pub coded_size_of_cb_data: u16,
    /// Present only when `alpha_channel_type != 0`. When absent the
    /// caller derives it from `coded_size_of_slice - header - y - cb`.
    pub coded_size_of_cr_data: Option<u16>,
}

/// Parse one slice_header(). `has_alpha` controls whether the
/// `coded_size_of_cr_data` field is present.
pub fn parse_slice_header(data: &[u8], has_alpha: bool) -> Result<(SliceHeader, &[u8])> {
    let min = if has_alpha { 8 } else { 6 };
    if data.len() < min {
        return Err(Error::invalid("prores: slice header truncated"));
    }
    let b0 = data[0];
    let slice_header_size = (b0 >> 3) & 0x1F;
    let quantization_index = data[1];
    if !(1..=224).contains(&quantization_index) {
        return Err(Error::invalid(
            "prores: quantization_index out of range (1..=224)",
        ));
    }
    let coded_size_of_y_data = u16::from_be_bytes(data[2..4].try_into().unwrap());
    let coded_size_of_cb_data = u16::from_be_bytes(data[4..6].try_into().unwrap());
    let coded_size_of_cr_data = if has_alpha {
        Some(u16::from_be_bytes(data[6..8].try_into().unwrap()))
    } else {
        None
    };
    let consumed = slice_header_size as usize;
    if consumed < min {
        return Err(Error::invalid("prores: slice_header_size < required"));
    }
    if data.len() < consumed {
        return Err(Error::invalid("prores: slice header overruns buffer"));
    }
    Ok((
        SliceHeader {
            slice_header_size,
            quantization_index,
            coded_size_of_y_data,
            coded_size_of_cb_data,
            coded_size_of_cr_data,
        },
        &data[consumed..],
    ))
}

pub fn write_slice_header(
    out: &mut Vec<u8>,
    quantization_index: u8,
    coded_size_of_y_data: u16,
    coded_size_of_cb_data: u16,
    coded_size_of_cr_data: Option<u16>,
) {
    let slice_header_size: u8 = if coded_size_of_cr_data.is_some() {
        8
    } else {
        6
    };
    out.push(slice_header_size << 3);
    out.push(quantization_index);
    out.extend_from_slice(&coded_size_of_y_data.to_be_bytes());
    out.extend_from_slice(&coded_size_of_cb_data.to_be_bytes());
    if let Some(cr) = coded_size_of_cr_data {
        out.extend_from_slice(&cr.to_be_bytes());
    }
}

/// Compute the slice_size_in_mb array per §6.2 (the same array applies
/// to every macroblock row). For the typical case `width=128` and
/// `log2_desired_slice_size_in_mb=3`, this returns `[8]` (1 slice per row,
/// 8 MBs each). Returns `(slice_sizes, slice_count)`.
pub fn compute_slice_sizes(width_in_mb: usize, log2_desired_slice_size_in_mb: u8) -> Vec<usize> {
    let mut sizes = Vec::new();
    let mut slice_size = 1usize << log2_desired_slice_size_in_mb;
    let mut remaining = width_in_mb;
    loop {
        while remaining >= slice_size {
            sizes.push(slice_size);
            remaining -= slice_size;
        }
        slice_size /= 2;
        if remaining == 0 {
            break;
        }
        if slice_size == 0 {
            // Defensive — should not occur for width_in_mb > 0.
            break;
        }
    }
    sizes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_roundtrip_422() {
        let luma = [4u8; 64];
        let chroma = [4u8; 64];
        let mut buf = Vec::new();
        // size unknown; we'll write 0 and patch later.
        write_frame(
            &mut buf,
            0,
            128,
            128,
            ChromaFormat::Y422,
            0,
            &luma,
            &chroma,
            false,
            false,
        );
        // Patch frame_size.
        let total = buf.len() as u32;
        buf[0..4].copy_from_slice(&total.to_be_bytes());
        let (fh, _) = parse_frame(&buf).unwrap();
        assert_eq!(fh.width, 128);
        assert_eq!(fh.height, 128);
        assert_eq!(fh.chroma_format, ChromaFormat::Y422);
        assert_eq!(fh.bitstream_version, 0);
        assert_eq!(fh.luma_qmat, [4u8; 64]);
        assert_eq!(fh.chroma_qmat, [4u8; 64]);
    }

    #[test]
    fn frame_roundtrip_444_with_qmats() {
        let mut luma = [0u8; 64];
        let mut chroma = [0u8; 64];
        for i in 0..64 {
            luma[i] = (i + 4) as u8;
            chroma[i] = (i + 8) as u8;
        }
        let mut buf = Vec::new();
        write_frame(
            &mut buf,
            0,
            64,
            64,
            ChromaFormat::Y444,
            0,
            &luma,
            &chroma,
            true,
            true,
        );
        let total = buf.len() as u32;
        buf[0..4].copy_from_slice(&total.to_be_bytes());
        let (fh, _) = parse_frame(&buf).unwrap();
        assert_eq!(fh.chroma_format, ChromaFormat::Y444);
        assert_eq!(fh.bitstream_version, 1);
        assert_eq!(fh.luma_qmat, luma);
        assert_eq!(fh.chroma_qmat, chroma);
    }

    #[test]
    fn picture_header_roundtrip() {
        let mut buf = Vec::new();
        write_picture_header(&mut buf, 1234, 12, 3);
        let (ph, _) = parse_picture_header(&buf).unwrap();
        assert_eq!(ph.picture_header_size, 8);
        assert_eq!(ph.picture_size, 1234);
        assert_eq!(ph.deprecated_number_of_slices, 12);
        assert_eq!(ph.log2_desired_slice_size_in_mb, 3);
    }

    #[test]
    fn slice_header_roundtrip_no_alpha() {
        let mut buf = Vec::new();
        write_slice_header(&mut buf, 4, 100, 50, None);
        let (sh, _) = parse_slice_header(&buf, false).unwrap();
        assert_eq!(sh.slice_header_size, 6);
        assert_eq!(sh.quantization_index, 4);
        assert_eq!(sh.coded_size_of_y_data, 100);
        assert_eq!(sh.coded_size_of_cb_data, 50);
        assert!(sh.coded_size_of_cr_data.is_none());
    }

    #[test]
    fn compute_slice_sizes_examples() {
        // RDD 36 example: width=720→ 45 MBs, slice size 8 → [8,8,8,8,8,4,1].
        assert_eq!(compute_slice_sizes(45, 3), vec![8, 8, 8, 8, 8, 4, 1]);
        // 8 MBs flat
        assert_eq!(compute_slice_sizes(8, 3), vec![8]);
        // 1 MB
        assert_eq!(compute_slice_sizes(1, 3), vec![1]);
        // log2=0 → all 1s
        assert_eq!(compute_slice_sizes(5, 0), vec![1, 1, 1, 1, 1]);
    }

    #[test]
    fn frame_rate_code_named_rates_match_spec_table_4() {
        use oxideav_core::Rational;
        // RDD 36 §6.2 / Table 4 — every named rate must map to its code.
        let cases: &[(Rational, u8)] = &[
            (Rational::new(24_000, 1001), 1),
            (Rational::new(24, 1), 2),
            (Rational::new(25, 1), 3),
            (Rational::new(30_000, 1001), 4),
            (Rational::new(30, 1), 5),
            (Rational::new(50, 1), 6),
            (Rational::new(60_000, 1001), 7),
            (Rational::new(60, 1), 8),
            (Rational::new(100, 1), 9),
            (Rational::new(120_000, 1001), 10),
            (Rational::new(120, 1), 11),
        ];
        for &(r, expected) in cases {
            let got = frame_rate_code_from_rational(r);
            assert_eq!(
                got, expected,
                "rate {}/{} must map to {expected}",
                r.num, r.den
            );
        }
    }

    #[test]
    fn frame_rate_code_unnormalised_fractions_match() {
        use oxideav_core::Rational;
        // 60/2 == 30 → code 5; 50000/1000 == 50 → code 6.
        assert_eq!(frame_rate_code_from_rational(Rational::new(60, 2)), 5);
        assert_eq!(
            frame_rate_code_from_rational(Rational::new(50_000, 1000)),
            6
        );
        // Doubling the 1.001 fraction: 48000/1001 != 24000/1001 (not the same rate).
        assert_eq!(
            frame_rate_code_from_rational(Rational::new(48_000, 1001)),
            0
        );
    }

    #[test]
    fn frame_rate_code_unknown_rates_map_to_zero() {
        use oxideav_core::Rational;
        // 48 fps, 90 fps, 0/0, negative — all "unknown".
        assert_eq!(frame_rate_code_from_rational(Rational::new(48, 1)), 0);
        assert_eq!(frame_rate_code_from_rational(Rational::new(90, 1)), 0);
        assert_eq!(frame_rate_code_from_rational(Rational::new(0, 0)), 0);
        assert_eq!(frame_rate_code_from_rational(Rational::new(-30, 1)), 0);
    }

    #[test]
    fn frame_meta_is_unknown_helpers() {
        assert!(FrameMeta::default().is_unknown());
        assert!(FrameMeta::unknown().is_unknown());
        let m = FrameMeta {
            frame_rate_code: 5,
            ..FrameMeta::default()
        };
        assert!(!m.is_unknown());
    }

    #[test]
    fn frame_with_meta_roundtrips_all_fields() {
        // Pack a non-trivial FrameMeta into a frame header and verify
        // the parser pulls every byte back out unchanged.
        let luma = [4u8; 64];
        let chroma = [4u8; 64];
        let meta = FrameMeta {
            aspect_ratio_information: 3, // 16:9 per Table 3
            frame_rate_code: 4,          // 30/1.001 per Table 4
            color_primaries: 9,          // BT.2020 (H.273)
            transfer_characteristic: 16, // SMPTE ST 2084 (H.273)
            matrix_coefficients: 9,      // BT.2020 non-constant luminance
        };
        let mut buf = Vec::new();
        write_frame_with_meta(
            &mut buf,
            0,
            1920,
            1080,
            ChromaFormat::Y422,
            0,
            &luma,
            &chroma,
            false,
            false,
            0,
            meta,
        );
        let total = buf.len() as u32;
        buf[0..4].copy_from_slice(&total.to_be_bytes());
        let (fh, _) = parse_frame(&buf).unwrap();
        assert_eq!(fh.aspect_ratio_information, meta.aspect_ratio_information);
        assert_eq!(fh.frame_rate_code, meta.frame_rate_code);
        assert_eq!(fh.color_primaries, meta.color_primaries);
        assert_eq!(fh.transfer_characteristic, meta.transfer_characteristic);
        assert_eq!(fh.matrix_coefficients, meta.matrix_coefficients);
    }

    #[test]
    fn frame_with_alpha_back_compat_zeros_meta() {
        // The legacy `write_frame_with_alpha` shim must leave every
        // metadata field at 0 (preserving the byte-exact behaviour the
        // pre-FrameMeta callers depended on).
        let luma = [4u8; 64];
        let chroma = [4u8; 64];
        let mut buf = Vec::new();
        write_frame_with_alpha(
            &mut buf,
            0,
            64,
            64,
            ChromaFormat::Y444,
            0,
            &luma,
            &chroma,
            false,
            false,
            2, // 16-bit alpha
        );
        let total = buf.len() as u32;
        buf[0..4].copy_from_slice(&total.to_be_bytes());
        let (fh, _) = parse_frame(&buf).unwrap();
        assert_eq!(fh.aspect_ratio_information, 0);
        assert_eq!(fh.frame_rate_code, 0);
        assert_eq!(fh.color_primaries, 0);
        assert_eq!(fh.transfer_characteristic, 0);
        assert_eq!(fh.matrix_coefficients, 0);
        assert_eq!(fh.alpha_channel_type, 2);
    }
}
