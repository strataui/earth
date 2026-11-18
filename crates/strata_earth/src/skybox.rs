//! Multi-resolution real telescope space skybox (HEALPix HiPS -> J2000 ICRF cubemap)
//! and `.ssky` binary pack codec for zero-setup CDN / GitHub Release delivery.

use std::collections::HashMap;
use std::io::Cursor;
use std::sync::{Arc, Mutex, RwLock};
use std::time::Duration;

use image::codecs::jpeg::JpegEncoder;
use image::{DynamicImage, ImageFormat, RgbaImage};
use strata_render::scene3d::RetainedCubemapTexture;

use crate::assets_cdn::{http_get_bytes, write_file_atomic, AssetManager, AssetSourceConfig};

const SSKY_MAGIC: &[u8; 8] = b"SSKY0001";

/// Programmatic resolution tiers for the J2000 ICRF telescope space skybox cubemap.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum SpaceSkyboxQuality {
    /// Disable real telescope cubemap (uses lightweight procedural Milky Way + BSC5 stars only).
    Off,
    /// `6 x 512x512` cubemap (~1-2 MB download, ~6 MiB VRAM).
    Low,
    /// `6 x 1024x1024` cubemap (~3-6 MB download, ~24 MiB VRAM).
    Medium,
    /// `6 x 2048x2048` cubemap (~8-15 MB download, ~96 MiB VRAM) — crisp for 1080p/1440p.
    #[default]
    High,
    /// `6 x 4096x4096` cubemap (~25-35 MB download from HiPS Order 3, ~384 MiB VRAM) — 4K-oversampled.
    Ultra,
}

impl SpaceSkyboxQuality {
    /// Face resolution in pixels (`0` when `Off`).
    pub const fn face_size(self) -> u32 {
        match self {
            Self::Off => 0,
            Self::Low => 512,
            Self::Medium => 1024,
            Self::High => 2048,
            Self::Ultra => 4096,
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::Ultra => "ultra",
        }
    }

    pub const fn short_label(self) -> &'static str {
        match self {
            Self::Off => "Off",
            Self::Low => "512",
            Self::Medium => "1K",
            Self::High => "2K",
            Self::Ultra => "4K",
        }
    }

    pub const fn next(self) -> Self {
        match self {
            Self::Off => Self::Low,
            Self::Low => Self::Medium,
            Self::Medium => Self::High,
            Self::High => Self::Ultra,
            Self::Ultra => Self::Off,
        }
    }

    pub const fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::Off),
            1 => Some(Self::Low),
            2 => Some(Self::Medium),
            3 => Some(Self::High),
            4 => Some(Self::Ultra),
            _ => None,
        }
    }

    pub const fn to_u8(self) -> u8 {
        match self {
            Self::Off => 0,
            Self::Low => 1,
            Self::Medium => 2,
            Self::High => 3,
            Self::Ultra => 4,
        }
    }
}

/// Real telescope all-sky survey preset for the space skybox.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum SpaceSkyboxSurvey {
    /// Seamless ESA Gaia DR3 optical Milky Way + blended DSS2/AllWISE nebulae highlights.
    #[default]
    CompositeOptical,
    /// Visible-light Palomar / UK Schmidt Digitized Sky Survey 2 color mosaic.
    Dss2Optical,
    /// NASA/IPAC AllWISE W4/W2/W1 infrared nebulae & interstellar dust mosaic.
    AllWiseInfrared,
    /// ESA Gaia DR3 seamless optical stellar flux & Galactic dust-rift map.
    GaiaDr3Optical,
}

impl SpaceSkyboxSurvey {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::CompositeOptical => "composite",
            Self::Dss2Optical => "dss2",
            Self::AllWiseInfrared => "allwise",
            Self::GaiaDr3Optical => "gaia",
        }
    }

    pub const fn display_label(self) -> &'static str {
        match self {
            Self::CompositeOptical => "Composite",
            Self::Dss2Optical => "DSS2",
            Self::AllWiseInfrared => "AllWISE",
            Self::GaiaDr3Optical => "Gaia DR3",
        }
    }

    pub const fn next(self) -> Self {
        match self {
            Self::CompositeOptical => Self::Dss2Optical,
            Self::Dss2Optical => Self::AllWiseInfrared,
            Self::AllWiseInfrared => Self::GaiaDr3Optical,
            Self::GaiaDr3Optical => Self::CompositeOptical,
        }
    }

    pub const fn to_u8(self) -> u8 {
        match self {
            Self::CompositeOptical => 0,
            Self::Dss2Optical => 1,
            Self::AllWiseInfrared => 2,
            Self::GaiaDr3Optical => 3,
        }
    }

    /// Flat filename stored in the rolling `assets` GitHub Release and `cdn.strata.larvance.com`.
    pub fn cdn_filename(self, quality: SpaceSkyboxQuality) -> String {
        format!("skybox-{}-{}.ssky", self.as_str(), quality.as_str())
    }
}

/// Native sky coordinate system for a HiPS survey.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HipsFrame {
    EquatorialIcrs,
    Galactic,
}

/// Standard IAU/Hipparcos J2000 ICRS -> Galactic `(x_g, y_g, z_g)` direction transform.
#[inline]
pub fn icrs_to_galactic(v: [f64; 3]) -> [f64; 3] {
    [
        -0.054_875_560_4 * v[0] - 0.873_437_090_2 * v[1] - 0.483_835_015_5 * v[2],
        0.494_109_427_9 * v[0] - 0.444_829_630_0 * v[1] + 0.746_982_244_5 * v[2],
        -0.867_666_149_0 * v[0] - 0.198_076_373_4 * v[1] + 0.455_983_776_2 * v[2],
    ]
}

/// Maps WebGPU cubemap `(face 0..6, px, py)` to normalized J2000 ICRF unit vector.
/// Layer order matches WebGPU/Vulkan: `[+X, -X, +Y, -Y, +Z, -Z]`.
#[inline]
pub fn cubemap_face_dir(face: usize, px: u32, py: u32, face_size: u32) -> [f64; 3] {
    let inv = 1.0 / (face_size.max(1) as f64);
    let u = ((px as f64) + 0.5) * inv * 2.0 - 1.0;
    let v = ((py as f64) + 0.5) * inv * 2.0 - 1.0;
    let raw = match face {
        0 => [1.0, -v, -u],  // +X
        1 => [-1.0, -v, u],  // -X
        2 => [u, 1.0, v],    // +Y
        3 => [u, -1.0, -v],  // -Y
        4 => [u, -v, 1.0],   // +Z
        _ => [-u, -v, -1.0], // -Z
    };
    let len = (raw[0] * raw[0] + raw[1] * raw[1] + raw[2] * raw[2])
        .sqrt()
        .max(1e-12);
    [raw[0] / len, raw[1] / len, raw[2] / len]
}

/// Projects a normalized 3D vector to continuous HEALPix `(base_face 0..12, u_ix in [0,1), v_iy in [0,1))`.
#[inline]
pub fn vec_to_healpix_face_uv(v: [f64; 3]) -> (u8, f64, f64) {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-12);
    let vx = v[0] / len;
    let vy = v[1] / len;
    let vz = (v[2] / len).clamp(-1.0, 1.0);
    let za = vz.abs();
    let phi = vy.atan2(vx).rem_euclid(std::f64::consts::TAU);
    let tt = phi / std::f64::consts::FRAC_PI_2;

    if za <= 2.0 / 3.0 {
        let f1 = 0.5 + tt - vz * 0.75;
        let f2 = 0.5 + tt + vz * 0.75;
        let ifp = f1.floor() as i64;
        let ifm = f2.floor() as i64;
        let face = if ifp == ifm {
            ((ifp.rem_euclid(4)) | 4) as u8
        } else if ifp < ifm {
            ifp.rem_euclid(4) as u8
        } else {
            (ifm.rem_euclid(4) + 8) as u8
        };
        let u_ix = (f2 - f2.floor()).clamp(0.0, 0.999_999);
        let v_iy = (1.0 - (f1 - f1.floor())).clamp(0.0, 0.999_999);
        (face, u_ix, v_iy)
    } else {
        let ntt = (tt.floor() as i64).clamp(0, 3) as u8;
        let tp = tt - f64::from(ntt);
        let tmp = (3.0 * (1.0 - za).max(0.0)).sqrt();
        let jp = tp * tmp;
        let jm = (1.0 - tp) * tmp;
        if vz > 0.0 {
            (
                ntt,
                (1.0 - jm).clamp(0.0, 0.999_999),
                (1.0 - jp).clamp(0.0, 0.999_999),
            )
        } else {
            (
                ntt + 8,
                jp.clamp(0.0, 0.999_999),
                jm.clamp(0.0, 0.999_999),
            )
        }
    }
}

/// Encodes `(ix, iy)` inside one base face at `order` into a NESTED tile index.
#[inline]
pub fn healpix_nest_index(base_face: u8, ix: u32, iy: u32, order: u8) -> u64 {
    let mut sub: u64 = 0;
    for bit in 0..order {
        let shift = u32::from(bit);
        let bx = u64::from((ix >> shift) & 1);
        let by = u64::from((iy >> shift) & 1);
        sub |= (bx | (by << 1)) << (2 * shift);
    }
    (u64::from(base_face) << (2 * u32::from(order))) | sub
}

#[inline]
fn sample_bilinear_rgba(buf: &[u8], w: u32, h: u32, fx: f64, fy: f64) -> [f32; 3] {
    let wf = w as f64;
    let hf = h as f64;
    let x = fx.clamp(0.0, wf - 1.0001);
    let y = fy.clamp(0.0, hf - 1.0001);
    let x0 = x.floor() as u32;
    let y0 = y.floor() as u32;
    let x1 = (x0 + 1).min(w - 1);
    let y1 = (y0 + 1).min(h - 1);
    let tx = (x - x0 as f64) as f32;
    let ty = (y - y0 as f64) as f32;

    let read = |px: u32, py: u32| -> [f32; 3] {
        let i = ((py * w + px) * 4) as usize;
        [buf[i] as f32, buf[i + 1] as f32, buf[i + 2] as f32]
    };
    let c00 = read(x0, y0);
    let c10 = read(x1, y0);
    let c01 = read(x0, y1);
    let c11 = read(x1, y1);
    let mut out = [0.0f32; 3];
    for k in 0..3 {
        let top = c00[k] * (1.0 - tx) + c10[k] * tx;
        let bot = c01[k] * (1.0 - tx) + c11[k] * tx;
        out[k] = top * (1.0 - ty) + bot * ty;
    }
    out
}

/// Projects an IVOA HiPS `Norder3/Allsky.{jpg,png}` mosaic (`27x29` grid of `64x64` faces)
/// into 6 WebGPU cubemap faces of `face_size x face_size` RGBA8 bytes.
pub fn project_hips_allsky_to_cubemap(
    allsky_img: &DynamicImage,
    frame: HipsFrame,
    face_size: u32,
) -> [Arc<Vec<u8>>; 6] {
    let rgba = allsky_img.to_rgba8();
    let aw = rgba.width();
    let ah = rgba.height();
    let tile_w = (aw / 27).max(1);
    let tile_h = (ah / 29).max(1);
    let raw = rgba.as_raw();
    let nside = 8.0f64; // 2^3

    std::array::from_fn(|face| {
        let mut out = vec![255u8; (face_size as usize) * (face_size as usize) * 4];
        for py in 0..face_size {
            for px in 0..face_size {
                let d_icrf = cubemap_face_dir(face, px, py, face_size);
                let d_native = match frame {
                    HipsFrame::EquatorialIcrs => d_icrf,
                    HipsFrame::Galactic => icrs_to_galactic(d_icrf),
                };
                let (bf, u_ix, v_iy) = vec_to_healpix_face_uv(d_native);
                let gx = u_ix * nside;
                let gy = v_iy * nside;
                let tx = (gx.floor() as u32).min(7);
                let ty = (gy.floor() as u32).min(7);
                let ipix = healpix_nest_index(bf, tx, ty, 3) as u32;
                let col_grid = ipix % 27;
                let row_grid = ipix / 27;
                // Verified HiPS pixel orientation: px = iy, py = ix
                let local_y = (gx - tx as f64) * ((tile_h - 1) as f64);
                let local_x = (gy - ty as f64) * ((tile_w - 1) as f64);
                let sx = (col_grid * tile_w) as f64 + local_x;
                let sy = (row_grid * tile_h) as f64 + local_y;
                let rgb = sample_bilinear_rgba(raw, aw, ah, sx, sy);
                let dst = ((py * face_size + px) * 4) as usize;
                out[dst] = rgb[0].round().clamp(0.0, 255.0) as u8;
                out[dst + 1] = rgb[1].round().clamp(0.0, 255.0) as u8;
                out[dst + 2] = rgb[2].round().clamp(0.0, 255.0) as u8;
                out[dst + 3] = 255;
            }
        }
        Arc::new(out)
    })
}

/// Projects a map of `Norder{order}` HiPS tiles `(tile_width x tile_width)` into a 6-face cubemap.
pub fn project_hips_tiles_to_cubemap(
    order: u8,
    frame: HipsFrame,
    face_size: u32,
    tiles: &HashMap<u64, RgbaImage>,
) -> [Arc<Vec<u8>>; 6] {
    let nside = (1u32 << u32::from(order)) as f64;
    let max_idx = (1u32 << u32::from(order)).saturating_sub(1);

    std::array::from_fn(|face| {
        let mut out = vec![255u8; (face_size as usize) * (face_size as usize) * 4];
        for py in 0..face_size {
            for px in 0..face_size {
                let d_icrf = cubemap_face_dir(face, px, py, face_size);
                let d_native = match frame {
                    HipsFrame::EquatorialIcrs => d_icrf,
                    HipsFrame::Galactic => icrs_to_galactic(d_icrf),
                };
                let (bf, u_ix, v_iy) = vec_to_healpix_face_uv(d_native);
                let gx = u_ix * nside;
                let gy = v_iy * nside;
                let tx = (gx.floor() as u32).min(max_idx);
                let ty = (gy.floor() as u32).min(max_idx);
                let ipix = healpix_nest_index(bf, tx, ty, order);
                let dst = ((py * face_size + px) * 4) as usize;
                if let Some(tile) = tiles.get(&ipix) {
                    let tw = tile.width();
                    let th = tile.height();
                    let local_y = (gx - tx as f64) * ((th.saturating_sub(1)) as f64);
                    let local_x = (gy - ty as f64) * ((tw.saturating_sub(1)) as f64);
                    let rgb = sample_bilinear_rgba(tile.as_raw(), tw, th, local_x, local_y);
                    out[dst] = rgb[0].round().clamp(0.0, 255.0) as u8;
                    out[dst + 1] = rgb[1].round().clamp(0.0, 255.0) as u8;
                    out[dst + 2] = rgb[2].round().clamp(0.0, 255.0) as u8;
                } else {
                    out[dst] = 0;
                    out[dst + 1] = 0;
                    out[dst + 2] = 0;
                }
                out[dst + 3] = 255;
            }
        }
        Arc::new(out)
    })
}

/// Blends seamless Gaia DR3 optical Milky Way with optional DSS2 optical nebulae and AllWISE dust glow.
pub fn blend_composite_cubemap(
    gaia: &[Arc<Vec<u8>>; 6],
    dss2: Option<&[Arc<Vec<u8>>; 6]>,
    allwise: Option<&[Arc<Vec<u8>>; 6]>,
) -> [Arc<Vec<u8>>; 6] {
    std::array::from_fn(|f| {
        let g_buf = &gaia[f];
        let mut out = vec![255u8; g_buf.len()];
        let d_buf = dss2.map(|d| d[f].as_slice());
        let w_buf = allwise.map(|w| w[f].as_slice());
        for i in (0..g_buf.len()).step_by(4) {
            let mut r = (g_buf[i] as f32) / 255.0;
            let mut g = (g_buf[i + 1] as f32) / 255.0;
            let mut b = (g_buf[i + 2] as f32) / 255.0;

            // Softly screen optical DSS2 nebulae/galaxies above subtle plate noise threshold.
            if let Some(d) = d_buf {
                let dr = ((d[i] as f32) / 255.0 - 0.045).max(0.0) * 0.65;
                let dg = ((d[i + 1] as f32) / 255.0 - 0.045).max(0.0) * 0.65;
                let db = ((d[i + 2] as f32) / 255.0 - 0.045).max(0.0) * 0.65;
                r = 1.0 - (1.0 - r) * (1.0 - dr.min(0.85));
                g = 1.0 - (1.0 - g) * (1.0 - dg.min(0.85));
                b = 1.0 - (1.0 - b) * (1.0 - db.min(0.85));
            }

            // Gently reveal mid-IR star-forming nebulae (Orion, Carina, Cygnus, Rho Oph) from AllWISE.
            if let Some(w) = w_buf {
                let wr = ((w[i] as f32) / 255.0 - 0.06).max(0.0) * 0.34;
                let wg = ((w[i + 1] as f32) / 255.0 - 0.06).max(0.0) * 0.24;
                let wb = ((w[i + 2] as f32) / 255.0 - 0.06).max(0.0) * 0.28;
                r = (r + wr * (1.0 - r)).min(1.0);
                g = (g + wg * (1.0 - g)).min(1.0);
                b = (b + wb * (1.0 - b)).min(1.0);
            }

            out[i] = (r * 255.0).round().clamp(0.0, 255.0) as u8;
            out[i + 1] = (g * 255.0).round().clamp(0.0, 255.0) as u8;
            out[i + 2] = (b * 255.0).round().clamp(0.0, 255.0) as u8;
            out[i + 3] = 255;
        }
        Arc::new(out)
    })
}

/// Encodes 6 square RGBA8 faces into an `.ssky` binary container (`SSKY0001`) using JPEG payloads.
pub fn encode_ssky_jpeg(
    face_size: u32,
    survey: SpaceSkyboxSurvey,
    quality: SpaceSkyboxQuality,
    faces: &[Arc<Vec<u8>>; 6],
    jpeg_quality: u8,
) -> Result<Vec<u8>, String> {
    let mut blobs: [Vec<u8>; 6] = std::array::from_fn(|_| Vec::new());
    for (idx, face) in faces.iter().enumerate() {
        let mut cursor = Cursor::new(Vec::new());
        let mut rgb = Vec::with_capacity((face_size as usize) * (face_size as usize) * 3);
        for px in face.chunks_exact(4) {
            rgb.push(px[0]);
            rgb.push(px[1]);
            rgb.push(px[2]);
        }
        let mut enc = JpegEncoder::new_with_quality(&mut cursor, jpeg_quality);
        enc.encode(&rgb, face_size, face_size, image::ExtendedColorType::Rgb8)
            .map_err(|e| format!("encode face {idx}: {e}"))?;
        blobs[idx] = cursor.into_inner();
    }
    Ok(pack_ssky_blobs(face_size, survey, quality, &blobs))
}

/// Encodes 6 square RGBA8 faces into `.ssky` using lossless PNG face payloads.
pub fn encode_ssky_png(
    face_size: u32,
    survey: SpaceSkyboxSurvey,
    quality: SpaceSkyboxQuality,
    faces: &[Arc<Vec<u8>>; 6],
) -> Result<Vec<u8>, String> {
    let mut blobs: [Vec<u8>; 6] = std::array::from_fn(|_| Vec::new());
    for (idx, face) in faces.iter().enumerate() {
        let img = RgbaImage::from_raw(face_size, face_size, face.to_vec())
            .ok_or_else(|| format!("face {idx} buffer size mismatch"))?;
        let mut cursor = Cursor::new(Vec::new());
        DynamicImage::ImageRgba8(img)
            .write_to(&mut cursor, ImageFormat::Png)
            .map_err(|e| format!("encode png face {idx}: {e}"))?;
        blobs[idx] = cursor.into_inner();
    }
    Ok(pack_ssky_blobs(face_size, survey, quality, &blobs))
}

fn pack_ssky_blobs(
    face_size: u32,
    survey: SpaceSkyboxSurvey,
    quality: SpaceSkyboxQuality,
    blobs: &[Vec<u8>; 6],
) -> Vec<u8> {
    let total_payload: usize = blobs.iter().map(Vec::len).sum();
    let mut out = Vec::with_capacity(40 + total_payload);
    out.extend_from_slice(SSKY_MAGIC);
    out.extend_from_slice(&face_size.to_le_bytes());
    out.push(0); // encoding flag (auto-detected by image::load_from_memory)
    out.push(survey.to_u8());
    out.push(quality.to_u8());
    out.push(0);
    for blob in blobs {
        out.extend_from_slice(&(blob.len() as u32).to_le_bytes());
    }
    for blob in blobs {
        out.extend_from_slice(blob);
    }
    out
}

/// Decodes an `.ssky` binary container into cubemap dimensions and 6 RGBA8 face buffers.
pub fn decode_ssky(bytes: &[u8]) -> Result<(u32, [Arc<Vec<u8>>; 6]), String> {
    if bytes.len() < 40 || &bytes[..8] != SSKY_MAGIC {
        return Err("invalid .ssky header magic".to_string());
    }
    let face_size = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]).max(1);
    let mut lens = [0usize; 6];
    let mut off = 16usize;
    for slot in &mut lens {
        *slot = u32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]])
            as usize;
        off += 4;
    }
    let mut faces: [Option<Arc<Vec<u8>>>; 6] = Default::default();
    for (idx, len) in lens.into_iter().enumerate() {
        if off + len > bytes.len() {
            return Err(format!("truncated .ssky face {idx}"));
        }
        let decoded = image::load_from_memory(&bytes[off..off + len])
            .map_err(|e| format!("decode .ssky face {idx}: {e}"))?;
        let resized = if decoded.width() != face_size || decoded.height() != face_size {
            decoded.resize_exact(
                face_size,
                face_size,
                image::imageops::FilterType::Triangle,
            )
        } else {
            decoded
        };
        let mut raw = resized.to_rgba8().into_raw();
        // Force alpha=255 so SKY_WGSL distinguishes real skybox from 1x1 fallback (alpha=0).
        for px in raw.chunks_exact_mut(4) {
            px[3] = 255;
        }
        faces[idx] = Some(Arc::new(raw));
        off += len;
    }
    Ok((
        face_size,
        [
            faces[0].take().unwrap(),
            faces[1].take().unwrap(),
            faces[2].take().unwrap(),
            faces[3].take().unwrap(),
            faces[4].take().unwrap(),
            faces[5].take().unwrap(),
        ],
    ))
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct SkyboxRequestKey {
    survey: SpaceSkyboxSurvey,
    quality: SpaceSkyboxQuality,
}

type WakeupFn = Arc<dyn Fn() + Send + Sync>;

/// Asynchronous progressive telescope skybox store backed by CDN, GitHub rolling releases,
/// local OS disk cache, and live NASA/IPAC IRSA + ESA Gaia HiPS fallback.
pub struct SpaceSkyboxStore {
    state: Arc<Mutex<SkyboxStoreState>>,
    current: Arc<RwLock<(u64, Option<Arc<RetainedCubemapTexture>>, Option<SkyboxRequestKey>)>>,
    wakeup: Option<WakeupFn>,
}

struct SkyboxStoreState {
    in_flight: Option<SkyboxRequestKey>,
    next_revision: u64,
    failed_keys: HashMap<SkyboxRequestKey, std::time::Instant>,
}

impl std::fmt::Debug for SpaceSkyboxStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SpaceSkyboxStore").finish()
    }
}

impl SpaceSkyboxStore {
    pub fn new(wakeup: Option<WakeupFn>) -> Self {
        Self {
            state: Arc::new(Mutex::new(SkyboxStoreState {
                in_flight: None,
                next_revision: 1,
                failed_keys: HashMap::new(),
            })),
            current: Arc::new(RwLock::new((0, None, None))),
            wakeup,
        }
    }

    /// Returns `(revision, current_cubemap)` for the latest loaded skybox.
    pub fn snapshot(&self) -> (u64, Option<Arc<RetainedCubemapTexture>>) {
        self.current
            .read()
            .map(|g| (g.0, g.1.clone()))
            .unwrap_or((0, None))
    }

    /// Requests `(survey, quality)` asynchronously; clears immediately when `quality == Off`.
    pub fn ensure_loaded(
        &self,
        survey: SpaceSkyboxSurvey,
        quality: SpaceSkyboxQuality,
        cfg: &AssetSourceConfig,
    ) {
        if quality == SpaceSkyboxQuality::Off {
            if let Ok(mut cur) = self.current.write() {
                if cur.1.is_some() {
                    let rev = self
                        .state
                        .lock()
                        .map(|mut s| {
                            s.next_revision += 1;
                            s.next_revision
                        })
                        .unwrap_or(cur.0 + 1);
                    *cur = (rev, None, Some(SkyboxRequestKey { survey, quality }));
                }
            }
            return;
        }

        let key = SkyboxRequestKey { survey, quality };
        if let Ok(cur) = self.current.read() {
            if cur.2.as_ref() == Some(&key) {
                return;
            }
        }

        // Fast synchronous check of local disk cache so cached skyboxes appear on Frame 0!
        let filename = survey.cdn_filename(quality);
        let cache_path = cfg.resolved_cache_dir().join(&filename);
        if let Ok(bytes) = std::fs::read(&cache_path) {
            if let Ok((face_size, faces_rgba8)) = decode_ssky(&bytes) {
                self.commit_cubemap(key, face_size, faces_rgba8);
                return;
            }
        }

        if cfg.offline || !cfg.auto_download {
            return;
        }

        {
            let Ok(mut st) = self.state.lock() else {
                return;
            };
            if st.in_flight.as_ref() == Some(&key) {
                return;
            }
            if let Some(failed_at) = st.failed_keys.get(&key) {
                if failed_at.elapsed() < Duration::from_secs(60) {
                    return;
                }
            }
            st.in_flight = Some(key.clone());
        }

        let cfg_cl = cfg.clone();
        let state_cl = Arc::clone(&self.state);
        let cur_cl = Arc::clone(&self.current);
        let wake_cl = self.wakeup.clone();
        let assets = AssetManager::new();

        std::thread::Builder::new()
            .name("strata-earth-skybox".to_string())
            .spawn(move || {
                let res = fetch_or_synthesize_skybox(&assets, &cfg_cl, survey, quality);
                match res {
                    Ok((face_size, faces_rgba8)) => {
                        let rev = state_cl
                            .lock()
                            .map(|mut s| {
                                s.in_flight = None;
                                s.failed_keys.remove(&key);
                                s.next_revision += 1;
                                s.next_revision
                            })
                            .unwrap_or(1);
                        let id = ((survey.to_u8() as u64) << 8) | (quality.to_u8() as u64) + 100;
                        let cube = Arc::new(RetainedCubemapTexture {
                            id,
                            revision: rev,
                            face_size,
                            faces_rgba8,
                        });
                        if let Ok(mut cur) = cur_cl.write() {
                            *cur = (rev, Some(cube), Some(key));
                        }
                        if let Some(w) = wake_cl {
                            w();
                        }
                    }
                    Err(_) => {
                        if let Ok(mut s) = state_cl.lock() {
                            s.in_flight = None;
                            s.failed_keys.insert(key, std::time::Instant::now());
                        }
                    }
                }
            })
            .ok();
    }

    fn commit_cubemap(
        &self,
        key: SkyboxRequestKey,
        face_size: u32,
        faces_rgba8: [Arc<Vec<u8>>; 6],
    ) {
        let rev = self
            .state
            .lock()
            .map(|mut s| {
                s.next_revision += 1;
                s.next_revision
            })
            .unwrap_or(1);
        let id = ((key.survey.to_u8() as u64) << 8) | (key.quality.to_u8() as u64) + 100;
        let cube = Arc::new(RetainedCubemapTexture {
            id,
            revision: rev,
            face_size,
            faces_rgba8,
        });
        if let Ok(mut cur) = self.current.write() {
            *cur = (rev, Some(cube), Some(key));
        }
    }
}

fn fetch_or_synthesize_skybox(
    assets: &AssetManager,
    cfg: &AssetSourceConfig,
    survey: SpaceSkyboxSurvey,
    quality: SpaceSkyboxQuality,
) -> Result<(u32, [Arc<Vec<u8>>; 6]), String> {
    let filename = survey.cdn_filename(quality);
    let manifest = assets.fetch_manifest(cfg);
    let (legacy, sha) = manifest
        .as_ref()
        .and_then(|m| m.entries.get(&filename))
        .map(|e| (e.legacy_filename.clone(), e.sha256.clone()))
        .unwrap_or((None, None));

    // 1. Try pre-baked `.ssky` pack from CDN / rolling GitHub Releases `assets` & `assets-legacy`.
    if let Ok(bytes) = assets.fetch_or_cache(
        cfg,
        &filename,
        legacy.as_deref(),
        sha.as_deref(),
        Duration::from_secs(30),
        120 * 1024 * 1024,
    ) {
        if let Ok(decoded) = decode_ssky(&bytes) {
            return Ok(decoded);
        }
    }

    if !cfg.allow_direct_hips_fallback {
        return Err(format!("skybox pack {filename} unavailable and HiPS fallback disabled"));
    }

    // 2. Direct live synthesis fallback from NASA/IPAC IRSA & ESA Gaia HiPS `Norder3/Allsky`.
    // Cap live Allsky fallback at 1024 face size to keep boot latency under ~2 seconds.
    let face_size = quality.face_size().min(1024).max(512);
    let faces = match survey {
        SpaceSkyboxSurvey::Dss2Optical => {
            let img = fetch_allsky_cached(
                cfg,
                "hips_dss2_allsky.jpg",
                "https://irsa.ipac.caltech.edu/data/hips/CDS/DSS2/color/Norder3/Allsky.jpg",
            )?;
            project_hips_allsky_to_cubemap(&img, HipsFrame::EquatorialIcrs, face_size)
        }
        SpaceSkyboxSurvey::AllWiseInfrared => {
            let img = fetch_allsky_cached(
                cfg,
                "hips_allwise_allsky.jpg",
                "https://irsa.ipac.caltech.edu/data/hips/CDS/AllWISE/RGB-W4-W2-W1/Norder3/Allsky.jpg",
            )?;
            project_hips_allsky_to_cubemap(&img, HipsFrame::EquatorialIcrs, face_size)
        }
        SpaceSkyboxSurvey::GaiaDr3Optical => {
            let img = fetch_allsky_cached(
                cfg,
                "hips_gaia_dr3_allsky.png",
                "https://skies.esac.esa.int/Gaia/dr3/RGB/Norder3/Allsky.png",
            )?;
            project_hips_allsky_to_cubemap(&img, HipsFrame::Galactic, face_size)
        }
        SpaceSkyboxSurvey::CompositeOptical => {
            let gaia_img = fetch_allsky_cached(
                cfg,
                "hips_gaia_dr3_allsky.png",
                "https://skies.esac.esa.int/Gaia/dr3/RGB/Norder3/Allsky.png",
            )?;
            let gaia = project_hips_allsky_to_cubemap(&gaia_img, HipsFrame::Galactic, face_size);
            let wise = fetch_allsky_cached(
                cfg,
                "hips_allwise_allsky.jpg",
                "https://irsa.ipac.caltech.edu/data/hips/CDS/AllWISE/RGB-W4-W2-W1/Norder3/Allsky.jpg",
            )
            .ok()
            .map(|im| project_hips_allsky_to_cubemap(&im, HipsFrame::EquatorialIcrs, face_size));
            let dss2 = fetch_allsky_cached(
                cfg,
                "hips_dss2_allsky.jpg",
                "https://irsa.ipac.caltech.edu/data/hips/CDS/DSS2/color/Norder3/Allsky.jpg",
            )
            .ok()
            .map(|im| project_hips_allsky_to_cubemap(&im, HipsFrame::EquatorialIcrs, face_size));
            blend_composite_cubemap(&gaia, dss2.as_ref(), wise.as_ref())
        }
    };

    // Persist synthesized `.ssky` to cache if `face_size` matches requested tier.
    if face_size == quality.face_size() {
        if let Ok(encoded) = encode_ssky_jpeg(face_size, survey, quality, &faces, 92) {
            let _ = write_file_atomic(&cfg.resolved_cache_dir().join(&filename), &encoded);
        }
    }

    Ok((face_size, faces))
}

fn fetch_allsky_cached(
    cfg: &AssetSourceConfig,
    cache_name: &str,
    url: &str,
) -> Result<DynamicImage, String> {
    let cache_path = cfg.resolved_cache_dir().join(cache_name);
    if let Ok(bytes) = std::fs::read(&cache_path) {
        if let Ok(img) = image::load_from_memory(&bytes) {
            return Ok(img);
        }
    }
    let bytes = http_get_bytes(url, Duration::from_secs(20), 25 * 1024 * 1024)?;
    let img = image::load_from_memory(&bytes).map_err(|e| format!("decode {url}: {e}"))?;
    let _ = write_file_atomic(&cache_path, &bytes);
    Ok(img)
}
