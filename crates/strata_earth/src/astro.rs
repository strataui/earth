//! Astronomical ephemeris, local OS UTC clock integration, scientific BSC5 star catalog,
//! NASA public-domain Moon & land-mask loaders, and live global 3D cloud-map streaming.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

use strata_render::scene3d::{
    RetainedCubemapTexture, RetainedEnvironmentAssets, RetainedStar, RetainedTileTexture,
};

use crate::tiles::{HttpTransport, StandardHttpTransport};

/// Resolution tier for live global cloud maps (`https://clouds.matteason.co.uk/`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum CloudMapResolution {
    /// 1024×512 (~517 KB PNG, fast load).
    Low,
    /// 2048×1024 (~1.87 MB PNG, crisp default).
    #[default]
    Medium,
    /// 4096×2048 (~6.65 MB PNG, high detail).
    High,
    /// 8192×4096 (~22.88 MB PNG, ultra detail).
    Ultra,
}

impl CloudMapResolution {
    pub const fn url(self) -> &'static str {
        match self {
            Self::Low => "https://clouds.matteason.co.uk/images/1024x512/clouds-alpha.png",
            Self::Medium => "https://clouds.matteason.co.uk/images/2048x1024/clouds-alpha.png",
            Self::High => "https://clouds.matteason.co.uk/images/4096x2048/clouds-alpha.png",
            Self::Ultra => "https://clouds.matteason.co.uk/images/8192x4096/clouds-alpha.png",
        }
    }

    pub const fn short_label(self) -> &'static str {
        match self {
            Self::Low => "1K",
            Self::Medium => "2K",
            Self::High => "4K",
            Self::Ultra => "8K",
        }
    }

    pub const fn next(self) -> Self {
        match self {
            Self::Low => Self::Medium,
            Self::Medium => Self::High,
            Self::High => Self::Ultra,
            Self::Ultra => Self::Low,
        }
    }

    pub fn is_preset_url(url: &str) -> bool {
        matches!(
            url.trim(),
            "https://clouds.matteason.co.uk/images/1024x512/clouds-alpha.png"
                | "https://clouds.matteason.co.uk/images/2048x1024/clouds-alpha.png"
                | "https://clouds.matteason.co.uk/images/4096x2048/clouds-alpha.png"
                | "https://clouds.matteason.co.uk/images/8192x4096/clouds-alpha.png"
        )
    }
}

/// Default open live global cloud-map URL (2048×1024 EUMETSAT/NOAA satellite mosaic).
pub const DEFAULT_CLOUD_MAP_URL: &str = CloudMapResolution::Medium.url();

const BSC5_STARS_BIN: &[u8] = include_bytes!("../assets/bsc5_stars.bin");
const LAND_MASK_PNG: &[u8] = include_bytes!("../assets/land_mask.png");
const NIGHT_LIGHTS_JPG: &[u8] = include_bytes!("../assets/night_lights.jpg");
const MOON_ALBEDO_JPG: &[u8] = include_bytes!("../assets/moon_albedo.jpg");
const CLOUDS_FALLBACK_JPG: &[u8] = include_bytes!("../assets/clouds_fallback.jpg");

static PARSED_STARS: OnceLock<Arc<Vec<RetainedStar>>> = OnceLock::new();
static PARSED_LAND_MASK: OnceLock<Arc<RetainedTileTexture>> = OnceLock::new();
static PARSED_MOON_TEX: OnceLock<Arc<RetainedTileTexture>> = OnceLock::new();
static PARSED_CLOUDS_FALLBACK: OnceLock<Arc<RetainedTileTexture>> = OnceLock::new();

/// Snapshot of the host computer's wall-clock UTC time and local timezone UTC offset.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LocalClockTime {
    /// Seconds since 1970-01-01T00:00:00Z.
    pub unix_utc_s: f64,
    /// Local timezone offset East of UTC in seconds (e.g., `+10800` for UTC+3, `-18000` for UTC-5).
    pub utc_offset_s: i32,
    /// `unix_utc_s + utc_offset_s as f64`.
    pub local_unix_s: f64,
    /// Local fractional hour of day in `[0.0, 24.0)`.
    pub local_hour: f64,
}

/// Reads the computer's live system clock (`SystemTime::now()`) and OS timezone UTC offset.
pub fn system_clock_time() -> LocalClockTime {
    let unix_utc_s = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0);
    let utc_offset_s = current_os_utc_offset_s(unix_utc_s as i64);
    let local_unix_s = unix_utc_s + utc_offset_s as f64;
    let local_hour = (local_unix_s.rem_euclid(86_400.0)) / 3_600.0;
    LocalClockTime {
        unix_utc_s,
        utc_offset_s,
        local_unix_s,
        local_hour,
    }
}

fn current_os_utc_offset_s(unix_sec: i64) -> i32 {
    #[cfg(unix)]
    {
        // SAFETY: `localtime_r` writes into stack-allocated `tm` using `unix_sec` without mutating Rust globals.
        unsafe {
            let t = unix_sec as libc::time_t;
            let mut tm: libc::tm = std::mem::zeroed();
            if !libc::localtime_r(&t, &mut tm).is_null() {
                return tm.tm_gmtoff as i32;
            }
        }
    }
    let _ = unix_sec;
    0
}

/// Converts Unix UTC seconds into Astronomical Julian Date (JD).
#[inline]
pub fn unix_to_julian_date(unix_s: f64) -> f64 {
    2_440_587.5 + unix_s / 86_400.0
}

/// Computes Greenwich Mean Sidereal Time (GMST) in radians `[0, 2*PI)`.
pub fn greenwich_mean_sidereal_time_rad(unix_s: f64) -> f64 {
    let jd = unix_to_julian_date(unix_s);
    let d = jd - 2_451_545.0;
    let t = d / 36_525.0;
    let gmst_deg = 280.460_618_37 + 360.985_647_366_29 * d + 0.000_387_933 * t * t
        - (t * t * t) / 38_710_000.0;
    gmst_deg.rem_euclid(360.0).to_radians()
}

/// Column-major 4x4 rotation matrix transforming J2000 ICRF equatorial directions into ECEF at `unix_s`.
pub fn icrf_to_ecef_matrix(unix_s: f64) -> [[f32; 4]; 4] {
    let gmst = greenwich_mean_sidereal_time_rad(unix_s);
    let (s, c) = gmst.sin_cos();
    let cf = c as f32;
    let sf = s as f32;
    // R_z(-GMST):
    // [  cos  sin  0  0 ]
    // [ -sin  cos  0  0 ]
    // [   0    0   1  0 ]
    // [   0    0   0  1 ]
    [
        [cf, -sf, 0.0, 0.0],
        [sf, cf, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

/// Converts celestial equatorial `(ra_rad, dec_rad)` into normalized ECEF at UTC `unix_s`.
pub fn equatorial_to_ecef(ra_rad: f64, dec_rad: f64, unix_s: f64) -> [f32; 3] {
    let gmst = greenwich_mean_sidereal_time_rad(unix_s);
    let ha = ra_rad - gmst;
    let cos_d = dec_rad.cos();
    let x = cos_d * ha.cos();
    let y = cos_d * ha.sin();
    let z = dec_rad.sin();
    let len = (x * x + y * y + z * z).sqrt().max(1e-12);
    [(x / len) as f32, (y / len) as f32, (z / len) as f32]
}

/// Calculates accurate solar unit direction vector in ECEF for a Unix UTC timestamp.
pub fn sun_direction_from_unix_utc(unix_s: f64) -> [f32; 3] {
    let d = unix_to_julian_date(unix_s) - 2_451_545.0;
    let g = (357.529_11 + 0.985_600_28 * d).to_radians();
    let q = (280.466_46 + 0.985_647_36 * d).to_radians();
    let l = q + (1.914_602 * g.sin() + 0.019_993 * (2.0 * g).sin()).to_radians();
    let e = (23.439_291 - 0.000_000_356_3 * d).to_radians();

    let sin_l = l.sin();
    let ra = (e.cos() * sin_l).atan2(l.cos());
    let dec = (e.sin() * sin_l).asin();
    equatorial_to_ecef(ra, dec, unix_s)
}

/// Calculates accurate solar unit direction vector in ECEF from a local Unix timestamp and UTC offset.
///
/// `local_unix_s` represents wall-clock seconds relative to local midnight (`utc + utc_offset_s`), so
/// `unix_utc_s = local_unix_s - utc_offset_s as f64`. Ready to assign directly to time handlers.
pub fn sun_direction_from_local_time(local_unix_s: f64, utc_offset_s: i32) -> [f32; 3] {
    sun_direction_from_unix_utc(local_unix_s - utc_offset_s as f64)
}

/// Calculates accurate live solar ECEF direction from the computer's OS clock and local UTC offset.
pub fn sun_direction_from_system_clock() -> [f32; 3] {
    let clock = system_clock_time();
    sun_direction_from_local_time(clock.local_unix_s, clock.utc_offset_s)
}

/// Computes an ECEF sunlight direction placing the subsolar point directly overhead `(lat_deg, lon_deg)`.
pub fn sun_direction_from_subsolar_lat_lon(lat_deg: f64, lon_deg: f64) -> [f32; 3] {
    let lat = lat_deg.clamp(-89.9, 89.9).to_radians();
    let lon = lon_deg.to_radians();
    [
        (lat.cos() * lon.cos()) as f32,
        (lat.cos() * lon.sin()) as f32,
        lat.sin() as f32,
    ]
}

/// Calculates geocentric Moon unit direction vector in ECEF at UTC `unix_s` (Meeus ELP2000 truncated).
pub fn moon_direction_from_unix_utc(unix_s: f64) -> [f32; 3] {
    let d = unix_to_julian_date(unix_s) - 2_451_545.0;
    let t = d / 36_525.0;

    let lp = (218.316_447_7 + 13.176_396_48 * d).to_radians();
    let mp = (134.963_396_4 + 13.064_992_95 * d).to_radians();
    let ms = (357.529_109_2 + 0.985_600_28 * d).to_radians();
    let mm_d = (297.850_192_1 + 12.190_749_12 * d).to_radians();
    let f = (93.272_095_0 + 13.229_350_24 * d).to_radians();

    let lon_moon = lp
        + (6.288_774 * mp.sin()
            + 1.274_027 * (2.0 * mm_d - mp).sin()
            + 0.658_314 * (2.0 * mm_d).sin()
            + 0.213_618 * (2.0 * mp).sin()
            - 0.185_116 * ms.sin()
            - 0.114_332 * (2.0 * f).sin())
        .to_radians();

    let lat_moon = (5.128_122 * f.sin()
        + 0.280_602 * (mp + f).sin()
        + 0.277_693 * (mp - f).sin()
        + 0.173_237 * (2.0 * mm_d - f).sin())
    .to_radians();

    let eps = (23.439_291 - 0.013_004_2 * t).to_radians();
    let cos_b = lat_moon.cos();
    let sin_b = lat_moon.sin();
    let cos_l = lon_moon.cos();
    let sin_l = lon_moon.sin();

    let ra = (sin_l * cos_b * eps.cos() - sin_b * eps.sin()).atan2(cos_l * cos_b);
    let dec = (sin_b * eps.cos() + sin_l * cos_b * eps.sin()).asin();
    equatorial_to_ecef(ra, dec, unix_s)
}

/// Decodes and caches the embedded 8,404-star Yale Bright Star Catalog (`bsc5_stars.bin`).
pub fn embedded_bsc5_stars() -> Arc<Vec<RetainedStar>> {
    Arc::clone(PARSED_STARS.get_or_init(|| {
        if BSC5_STARS_BIN.len() < 12 || &BSC5_STARS_BIN[..8] != b"BSCS0001" {
            return Arc::new(Vec::new());
        }
        let count = u32::from_le_bytes([
            BSC5_STARS_BIN[8],
            BSC5_STARS_BIN[9],
            BSC5_STARS_BIN[10],
            BSC5_STARS_BIN[11],
        ]) as usize;
        let mut out = Vec::with_capacity(count);
        let mut off = 12usize;
        for _ in 0..count {
            if off + 12 > BSC5_STARS_BIN.len() {
                break;
            }
            let ix = i16::from_le_bytes([BSC5_STARS_BIN[off], BSC5_STARS_BIN[off + 1]]);
            let iy = i16::from_le_bytes([BSC5_STARS_BIN[off + 2], BSC5_STARS_BIN[off + 3]]);
            let iz = i16::from_le_bytes([BSC5_STARS_BIN[off + 4], BSC5_STARS_BIN[off + 5]]);
            let vm = i16::from_le_bytes([BSC5_STARS_BIN[off + 6], BSC5_STARS_BIN[off + 7]]);
            let r = BSC5_STARS_BIN[off + 8] as f32 / 255.0;
            let g = BSC5_STARS_BIN[off + 9] as f32 / 255.0;
            let b = BSC5_STARS_BIN[off + 10] as f32 / 255.0;
            off += 12;

            let mut x = ix as f32 / 32767.0;
            let mut y = iy as f32 / 32767.0;
            let mut z = iz as f32 / 32767.0;
            let len = (x * x + y * y + z * z).sqrt().max(1e-6);
            x /= len;
            y /= len;
            z /= len;
            out.push(RetainedStar {
                icrf_dir: [x, y, z],
                vmag: vm as f32 / 1000.0,
                color: [r, g, b, 1.0],
            });
        }
        Arc::new(out)
    }))
}

fn decode_image_to_retained_tex(
    id: u64,
    revision: u64,
    bytes: &[u8],
) -> Option<RetainedTileTexture> {
    let img = image::load_from_memory(bytes).ok()?;
    let rgba = img.to_rgba8();
    Some(RetainedTileTexture {
        id,
        revision,
        width: rgba.width(),
        height: rgba.height(),
        rgba8: Arc::new(rgba.into_raw()),
    })
}

/// Embedded Natural Earth public-domain global landmass mask (`.r`: `land=255`, `ocean=0`)
/// packed with NASA VIIRS Black Marble city lights luminance (`.g`).
pub fn embedded_land_mask() -> Arc<RetainedTileTexture> {
    Arc::clone(PARSED_LAND_MASK.get_or_init(|| {
        let mut tex =
            decode_image_to_retained_tex(0xE47A_0001, 1, LAND_MASK_PNG).unwrap_or_else(|| {
                RetainedTileTexture {
                    id: 0xE47A_0001,
                    revision: 1,
                    width: 1,
                    height: 1,
                    rgba8: Arc::new(vec![255, 255, 255, 255]),
                }
            });
        if let Ok(night_img) = image::load_from_memory(NIGHT_LIGHTS_JPG) {
            let luma = night_img.to_luma8();
            let (nw, nh) = (luma.width(), luma.height());
            if (nw, nh) != (tex.width, tex.height) && tex.width > 1 {
                if let Some(land_img) =
                    image::RgbaImage::from_raw(tex.width, tex.height, tex.rgba8.to_vec())
                {
                    let resized = image::imageops::resize(
                        &land_img,
                        nw,
                        nh,
                        image::imageops::FilterType::Triangle,
                    );
                    tex.width = nw;
                    tex.height = nh;
                    tex.rgba8 = Arc::new(resized.into_raw());
                }
            }
            if nw == tex.width && nh == tex.height {
                let raw = Arc::make_mut(&mut tex.rgba8);
                let (chunks, _) = raw.as_chunks_mut::<4>();
                for (px, &lum) in chunks.iter_mut().zip(luma.as_raw().iter()) {
                    let is_land = px[0] > 100;
                    px[1] = if is_land { lum } else { 0 };
                    px[2] = if is_land { lum } else { 0 };
                }
            }
        }
        Arc::new(tex)
    }))
}

/// Embedded NASA SVS Public Domain Lunar Reconnaissance Orbiter Moon albedo texture.
pub fn embedded_moon_texture() -> Arc<RetainedTileTexture> {
    Arc::clone(PARSED_MOON_TEX.get_or_init(|| {
        let tex =
            decode_image_to_retained_tex(0xE47A_0002, 1, MOON_ALBEDO_JPG).unwrap_or_else(|| {
                RetainedTileTexture {
                    id: 0xE47A_0002,
                    revision: 1,
                    width: 1,
                    height: 1,
                    rgba8: Arc::new(vec![196, 196, 196, 255]),
                }
            });
        Arc::new(tex)
    }))
}

/// Embedded fallback global cloud map texture (`clouds_fallback.jpg`).
pub fn embedded_clouds_fallback() -> Arc<RetainedTileTexture> {
    Arc::clone(PARSED_CLOUDS_FALLBACK.get_or_init(|| {
        let tex = decode_image_to_retained_tex(0xE47A_0003, 1, CLOUDS_FALLBACK_JPG).unwrap_or_else(
            || RetainedTileTexture {
                id: 0xE47A_0003,
                revision: 1,
                width: 1,
                height: 1,
                rgba8: Arc::new(vec![255, 255, 255, 0]),
            },
        );
        Arc::new(tex)
    }))
}

fn cloud_url_hash(url: &str) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for b in url.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

/// Asynchronous live global cloud-map manager (serves embedded cloud map immediately,
/// then upgrades smoothly when the live internet cloud map finishes downloading and
/// caches multiple cloud resolutions independently).
pub struct GlobalCloudMapStore {
    current: Arc<RwLock<(u64, Arc<RetainedTileTexture>)>>,
    cached_by_url: Arc<RwLock<HashMap<String, Arc<RetainedTileTexture>>>>,
    active_url: Arc<RwLock<String>>,
    started_urls: Mutex<HashMap<String, bool>>,
    cache_dir: Option<PathBuf>,
    wakeup: Option<Arc<dyn Fn() + Send + Sync>>,
}

impl std::fmt::Debug for GlobalCloudMapStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let rev = self.current.read().map(|g| g.0).unwrap_or(0);
        f.debug_struct("GlobalCloudMapStore")
            .field("revision", &rev)
            .finish()
    }
}

impl GlobalCloudMapStore {
    pub fn new(cache_dir: Option<PathBuf>, wakeup: Option<Arc<dyn Fn() + Send + Sync>>) -> Self {
        let initial = embedded_clouds_fallback();
        Self {
            current: Arc::new(RwLock::new((1, initial))),
            cached_by_url: Arc::new(RwLock::new(HashMap::new())),
            active_url: Arc::new(RwLock::new(String::new())),
            started_urls: Mutex::new(HashMap::new()),
            cache_dir,
            wakeup,
        }
    }

    pub fn ensure_fetched(&self, url: &str) {
        if url.trim().is_empty() {
            return;
        }
        let changed_active = self
            .active_url
            .write()
            .map(|mut a| {
                if *a == url {
                    false
                } else {
                    *a = url.to_string();
                    true
                }
            })
            .unwrap_or(false);

        if changed_active {
            if let Ok(cached_map) = self.cached_by_url.read() {
                if let Some(cached_tex) = cached_map.get(url) {
                    if let Ok(mut cur) = self.current.write() {
                        let next_rev = cur.0 + 1;
                        let swapped = Arc::new(RetainedTileTexture {
                            id: 0xE47A_0003,
                            revision: next_rev,
                            width: cached_tex.width,
                            height: cached_tex.height,
                            rgba8: Arc::clone(&cached_tex.rgba8),
                        });
                        *cur = (next_rev, swapped);
                    }
                    if let Some(ref cb) = self.wakeup {
                        cb();
                    }
                    return;
                }
            }
        }

        if let Ok(mut guard) = self.started_urls.lock() {
            if guard.contains_key(url) {
                return;
            }
            guard.insert(url.to_string(), true);
        } else {
            return;
        }

        let target = Arc::clone(&self.current);
        let cached_by_url = Arc::clone(&self.cached_by_url);
        let active_url = Arc::clone(&self.active_url);
        let url_owned = url.to_string();
        let cache_dir = self.cache_dir.clone();
        let wakeup = self.wakeup.clone();

        std::thread::Builder::new()
            .name("strata-cloudmap-fetch".to_string())
            .spawn(move || {
                let h = cloud_url_hash(&url_owned);
                let cache_file = cache_dir
                    .as_ref()
                    .map(|d| d.join(format!("global_clouds_{h:016x}.bin")));
                let transport = StandardHttpTransport::new().with_timeout_ms(20_000);
                let bytes = match transport.fetch(&url_owned) {
                    Ok(b) if !b.is_empty() => {
                        if let Some(ref cf) = cache_file {
                            if let Some(parent) = cf.parent() {
                                let _ = std::fs::create_dir_all(parent);
                            }
                            let _ = std::fs::write(cf, &b);
                        }
                        Some(b)
                    }
                    _ => cache_file
                        .as_ref()
                        .and_then(|cf| std::fs::read(cf).ok())
                        .filter(|b| !b.is_empty()),
                };

                if let Some(raw) = bytes {
                    if let Some(tex) = decode_image_to_retained_tex(0xE47A_0003, 1, &raw) {
                        let tex_arc = Arc::new(tex);
                        if let Ok(mut cmap) = cached_by_url.write() {
                            cmap.insert(url_owned.clone(), Arc::clone(&tex_arc));
                        }
                        let still_active = active_url
                            .read()
                            .map(|a| *a == url_owned)
                            .unwrap_or(false);
                        if still_active {
                            if let Ok(mut guard) = target.write() {
                                let next_rev = guard.0 + 1;
                                *guard = (
                                    next_rev,
                                    Arc::new(RetainedTileTexture {
                                        id: 0xE47A_0003,
                                        revision: next_rev,
                                        width: tex_arc.width,
                                        height: tex_arc.height,
                                        rgba8: Arc::clone(&tex_arc.rgba8),
                                    }),
                                );
                            }
                            if let Some(cb) = wakeup {
                                cb();
                            }
                        }
                    }
                }
            })
            .ok();
    }

    pub fn snapshot(&self) -> (u64, Arc<RetainedTileTexture>) {
        self.current
            .read()
            .map(|g| (g.0, Arc::clone(&g.1)))
            .unwrap_or_else(|_| (1, embedded_clouds_fallback()))
    }
}

/// Builds the complete [`RetainedEnvironmentAssets`] packet with BSC5 stars,
/// NASA Moon albedo, Natural Earth land mask, global cloud map, and optional skybox cubemap.
pub fn build_environment_assets_with_skybox(
    cloud_tex: Option<Arc<RetainedTileTexture>>,
    skybox_cubemap: Option<Arc<RetainedCubemapTexture>>,
    revision: u64,
) -> Arc<RetainedEnvironmentAssets> {
    Arc::new(RetainedEnvironmentAssets {
        revision,
        land_mask: Some(embedded_land_mask()),
        cloud_map: Some(cloud_tex.unwrap_or_else(embedded_clouds_fallback)),
        moon_texture: Some(embedded_moon_texture()),
        stars: Some(embedded_bsc5_stars()),
        skybox_cubemap,
    })
}

/// Builds [`RetainedEnvironmentAssets`] without a custom skybox cubemap.
pub fn build_environment_assets(
    cloud_tex: Option<Arc<RetainedTileTexture>>,
    revision: u64,
) -> Arc<RetainedEnvironmentAssets> {
    build_environment_assets_with_skybox(cloud_tex, None, revision)
}
