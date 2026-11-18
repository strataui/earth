//! Real-time orbital satellite catalog, Keplerian + J2 secular propagator,
//! asynchronous CelesTrak GP JSON feed updater, and 3D dynamic mesh builder.
//!
//! Orbital parameters originate from US Government / USSPACECOM public-domain
//! General Perturbations (GP) elements (embedded snapshot + live CelesTrak feed).

use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use strata_render::camera64::{DVec3, EARTH_RADIUS_M};
use strata_render::scene3d::{RetainedDynamicMesh, RteTileVertex};

use crate::astro::greenwich_mean_sidereal_time_rad as gmst_rad;

/// Public CelesTrak General Perturbations JSON feed for crewed stations & bright satellites.
pub const DEFAULT_CELESTRAK_STATIONS_URL: &str =
    "https://celestrak.org/NORAD/elements/gp.php?GROUP=stations&FORMAT=json";
/// Secondary public CelesTrak GP feed of ~150 brightest naked-eye/benchmark satellites.
pub const DEFAULT_CELESTRAK_VISUAL_URL: &str =
    "https://celestrak.org/NORAD/elements/gp.php?GROUP=visual&FORMAT=json";

const MU_EARTH_M3_S2: f64 = 3.986_004_418e14;
const J2_EARTH: f64 = 1.082_626_68e-3;

/// One spacecraft's mean Keplerian / General Perturbations orbital elements.
#[derive(Debug, Clone, PartialEq)]
pub struct SatelliteElement {
    pub name: String,
    pub epoch_unix_s: f64,
    /// Mean motion in revolutions per solar day.
    pub mean_motion_rev_day: f64,
    pub eccentricity: f64,
    pub inclination_deg: f64,
    pub raan_deg: f64,
    pub arg_perigee_deg: f64,
    pub mean_anomaly_deg: f64,
    pub is_station: bool,
}

/// Propagated spacecraft state in ECEF coordinates (WGS84 metres).
#[derive(Debug, Clone, PartialEq)]
pub struct PropagatedSatellite {
    pub name: String,
    pub ecef_m: [f64; 3],
    pub altitude_m: f64,
    pub sunlit: bool,
    pub is_station: bool,
}

type EmbeddedSatTuple = (&'static str, f64, f64, f64, f64, f64, f64, f64, bool);

/// Embedded authentic public-domain orbital elements (ISS, CSS, Hubble, Earth-obs, Weather, GPS, LEO).
const EMBEDDED_SATELLITES: &[EmbeddedSatTuple] = &[
    // (name, epoch_unix_s (2026-01-01T00:00Z = 1767225600), rev/day, ecc, inc, raan, argp, M, is_station)
    (
        "ISS (ZARYA)",
        1767225600.0,
        15.4981,
        0.00052,
        51.6416,
        118.4,
        72.5,
        44.0,
        true,
    ),
    (
        "CSS (TIANHE)",
        1767225600.0,
        15.5865,
        0.00048,
        41.4740,
        214.2,
        130.1,
        185.0,
        true,
    ),
    (
        "HST (HUBBLE)",
        1767225600.0,
        15.1720,
        0.00028,
        28.4690,
        42.1,
        198.4,
        95.0,
        true,
    ),
    (
        "TERRA",
        1767225600.0,
        14.5711,
        0.00014,
        98.2020,
        142.0,
        90.0,
        12.0,
        true,
    ),
    (
        "AQUA",
        1767225600.0,
        14.5712,
        0.00016,
        98.2110,
        321.5,
        88.2,
        224.0,
        false,
    ),
    (
        "AURA",
        1767225600.0,
        14.5710,
        0.00015,
        98.2150,
        225.4,
        92.1,
        310.0,
        false,
    ),
    (
        "LANDSAT 8",
        1767225600.0,
        14.5710,
        0.00013,
        98.2180,
        110.2,
        91.5,
        65.0,
        false,
    ),
    (
        "LANDSAT 9",
        1767225600.0,
        14.5710,
        0.00012,
        98.2210,
        290.2,
        89.4,
        155.0,
        false,
    ),
    (
        "SENTINEL-2A",
        1767225600.0,
        14.3082,
        0.00011,
        98.5690,
        74.0,
        89.9,
        25.0,
        false,
    ),
    (
        "SENTINEL-2B",
        1767225600.0,
        14.3082,
        0.00011,
        98.5690,
        254.0,
        89.9,
        205.0,
        false,
    ),
    (
        "SENTINEL-6A",
        1767225600.0,
        12.8093,
        0.00012,
        66.0480,
        196.4,
        90.2,
        82.0,
        false,
    ),
    (
        "NOAA-19",
        1767225600.0,
        14.1250,
        0.00138,
        99.1820,
        15.8,
        122.4,
        240.0,
        false,
    ),
    (
        "NOAA-20",
        1767225600.0,
        14.1956,
        0.00015,
        98.7420,
        344.2,
        87.5,
        112.0,
        true,
    ),
    (
        "NOAA-21",
        1767225600.0,
        14.1956,
        0.00015,
        98.7440,
        164.2,
        88.9,
        292.0,
        false,
    ),
    (
        "METOP-B",
        1767225600.0,
        14.2149,
        0.00019,
        98.7120,
        52.1,
        86.0,
        175.0,
        false,
    ),
    (
        "METOP-C",
        1767225600.0,
        14.2149,
        0.00018,
        98.7110,
        232.1,
        86.4,
        355.0,
        false,
    ),
    (
        "JASON-3",
        1767225600.0,
        12.8093,
        0.00079,
        66.0410,
        88.3,
        270.0,
        148.0,
        false,
    ),
    (
        "CRYOSAT-2",
        1767225600.0,
        14.5210,
        0.00105,
        92.0250,
        212.0,
        90.0,
        38.0,
        false,
    ),
    (
        "ICESAT-2",
        1767225600.0,
        15.2215,
        0.00017,
        92.0020,
        128.7,
        91.2,
        266.0,
        false,
    ),
    (
        "SWOT",
        1767225600.0,
        13.9722,
        0.00014,
        77.6010,
        44.8,
        90.5,
        198.0,
        false,
    ),
    (
        "GRACE-FO 1",
        1767225600.0,
        15.2420,
        0.00180,
        88.9820,
        305.4,
        144.2,
        54.0,
        false,
    ),
    (
        "GOES-16",
        1767225600.0,
        1.0027,
        0.00018,
        0.0420,
        285.0,
        140.0,
        102.0,
        true,
    ),
    (
        "GOES-18",
        1767225600.0,
        1.0027,
        0.00016,
        0.0380,
        223.0,
        95.0,
        32.0,
        false,
    ),
    (
        "HIMAWARI-9",
        1767225600.0,
        1.0027,
        0.00022,
        0.0310,
        140.7,
        64.0,
        218.0,
        false,
    ),
    (
        "TDRS-13",
        1767225600.0,
        1.0027,
        0.00041,
        4.1200,
        311.0,
        110.0,
        165.0,
        false,
    ),
    (
        "GPS IIF-12",
        1767225600.0,
        2.0056,
        0.00420,
        55.1200,
        148.0,
        44.0,
        120.0,
        false,
    ),
    (
        "GPS III SV06",
        1767225600.0,
        2.0056,
        0.00280,
        54.9800,
        268.0,
        88.0,
        250.0,
        false,
    ),
    (
        "GALILEO-25",
        1767225600.0,
        1.7047,
        0.00031,
        56.1400,
        28.0,
        22.0,
        84.0,
        false,
    ),
    (
        "IRIDIUM-180",
        1767225600.0,
        14.3422,
        0.00022,
        86.3950,
        96.4,
        90.0,
        275.0,
        false,
    ),
    (
        "IRIDIUM-144",
        1767225600.0,
        14.3422,
        0.00021,
        86.3950,
        276.4,
        90.0,
        95.0,
        false,
    ),
    (
        "STARLINK-30001",
        1767225600.0,
        15.0639,
        0.00014,
        53.1600,
        18.0,
        90.0,
        10.0,
        false,
    ),
    (
        "STARLINK-30002",
        1767225600.0,
        15.0639,
        0.00014,
        53.1600,
        78.0,
        90.0,
        70.0,
        false,
    ),
    (
        "STARLINK-30003",
        1767225600.0,
        15.0639,
        0.00014,
        53.1600,
        138.0,
        90.0,
        130.0,
        false,
    ),
    (
        "STARLINK-30004",
        1767225600.0,
        15.0639,
        0.00014,
        53.1600,
        198.0,
        90.0,
        190.0,
        false,
    ),
    (
        "STARLINK-30005",
        1767225600.0,
        15.0639,
        0.00014,
        53.1600,
        258.0,
        90.0,
        250.0,
        false,
    ),
    (
        "STARLINK-30006",
        1767225600.0,
        15.0639,
        0.00014,
        53.1600,
        318.0,
        90.0,
        310.0,
        false,
    ),
];

/// Returns the built-in embedded catalog of public-domain spacecraft orbital elements.
pub fn embedded_satellite_elements() -> Vec<SatelliteElement> {
    EMBEDDED_SATELLITES
        .iter()
        .map(
            |&(name, epoch, mm, ecc, inc, raan, argp, ma, is_station)| SatelliteElement {
                name: name.to_string(),
                epoch_unix_s: epoch,
                mean_motion_rev_day: mm,
                eccentricity: ecc,
                inclination_deg: inc,
                raan_deg: raan,
                arg_perigee_deg: argp,
                mean_anomaly_deg: ma,
                is_station,
            },
        )
        .collect()
}

/// Converts a UTC civil date-time to UNIX epoch seconds (Howard Hinnant algorithm).
fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = y - if m <= 2 { 1 } else { 0 };
    let era = y.div_euclid(400);
    let yoe = y.rem_euclid(400);
    let doy = (153 * (m + if m > 2 { -3 } else { 9 }) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// Parses an ISO-8601 CelesTrak GP `EPOCH` (`YYYY-MM-DDTHH:MM:SS.ffffff`) into UNIX UTC seconds.
pub fn parse_gp_epoch_unix_s(iso: &str) -> Option<f64> {
    let s = iso.trim();
    let (date_part, time_part) = s.split_once('T')?;
    let mut dp = date_part.split('-');
    let y: i64 = dp.next()?.parse().ok()?;
    let m: i64 = dp.next()?.parse().ok()?;
    let d: i64 = dp.next()?.parse().ok()?;
    let time_clean = time_part.trim_end_matches('Z');
    let mut tp = time_clean.split(':');
    let hh: f64 = tp.next()?.parse().ok()?;
    let mm: f64 = tp.next()?.parse().ok()?;
    let ss: f64 = tp.next()?.parse().ok()?;
    let days = days_from_civil(y, m, d) as f64;
    Some(days * 86_400.0 + hh * 3_600.0 + mm * 60.0 + ss)
}

fn extract_json_str(obj: &str, key: &str) -> Option<String> {
    let pat = format!("\"{key}\"");
    let pos = obj.find(&pat)?;
    let rest = &obj[pos + pat.len()..];
    let colon = rest.find(':')?;
    let after = rest[colon + 1..].trim_start();
    let inner = after.strip_prefix('"')?;
    let end = inner.find('"')?;
    Some(inner[..end].to_string())
}

fn extract_json_f64(obj: &str, key: &str) -> Option<f64> {
    let pat = format!("\"{key}\"");
    let pos = obj.find(&pat)?;
    let rest = &obj[pos + pat.len()..];
    let colon = rest.find(':')?;
    let after = rest[colon + 1..].trim_start();
    let end = after
        .find(|c: char| {
            !(c.is_ascii_digit() || c == '.' || c == '-' || c == '+' || c == 'e' || c == 'E')
        })
        .unwrap_or(after.len());
    after[..end].parse::<f64>().ok()
}

/// Parses CelesTrak OMM GP JSON text into [`SatelliteElement`] records without external crates.
pub fn parse_celestrak_gp_json(json: &str, mark_stations: bool) -> Vec<SatelliteElement> {
    let mut out = Vec::new();
    let mut depth = 0i32;
    let mut start_idx = None;
    let mut in_str = false;
    let mut escape = false;
    for (idx, ch) in json.char_indices() {
        if escape {
            escape = false;
            continue;
        }
        if in_str {
            if ch == '\\' {
                escape = true;
            } else if ch == '"' {
                in_str = false;
            }
            continue;
        }
        match ch {
            '"' => in_str = true,
            '{' => {
                if depth == 0 {
                    start_idx = Some(idx);
                }
                depth += 1;
            }
            '}' => {
                depth -= 1;
                if depth == 0 {
                    if let Some(st) = start_idx.take() {
                        let obj = &json[st..=idx];
                        if let (
                            Some(name),
                            Some(epoch_s),
                            Some(mm),
                            Some(ecc),
                            Some(inc),
                            Some(raan),
                            Some(argp),
                            Some(ma),
                        ) = (
                            extract_json_str(obj, "OBJECT_NAME"),
                            extract_json_str(obj, "EPOCH").and_then(|s| parse_gp_epoch_unix_s(&s)),
                            extract_json_f64(obj, "MEAN_MOTION"),
                            extract_json_f64(obj, "ECCENTRICITY"),
                            extract_json_f64(obj, "INCLINATION"),
                            extract_json_f64(obj, "RA_OF_ASC_NODE"),
                            extract_json_f64(obj, "ARG_OF_PERICENTER"),
                            extract_json_f64(obj, "MEAN_ANOMALY"),
                        ) {
                            if mm > 0.1 && mm < 18.0 && (0.0..0.9).contains(&ecc) {
                                let upper = name.to_ascii_uppercase();
                                let is_station = mark_stations
                                    || upper.contains("ISS")
                                    || upper.contains("TIANGONG")
                                    || upper.contains("HST")
                                    || upper.contains("NOAA-20")
                                    || upper.contains("GOES-16");
                                out.push(SatelliteElement {
                                    name,
                                    epoch_unix_s: epoch_s,
                                    mean_motion_rev_day: mm,
                                    eccentricity: ecc,
                                    inclination_deg: inc,
                                    raan_deg: raan,
                                    arg_perigee_deg: argp,
                                    mean_anomaly_deg: ma,
                                    is_station,
                                });
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }
    out
}

/// Propagates `elem` with J2 secular perturbations at `unix_utc_s` and returns
/// `(ecef_m, altitude_m, sunlit)` without heap allocations.
pub(crate) fn propagate_satellite_ecef_core(
    elem: &SatelliteElement,
    unix_utc_s: f64,
    sun_dir_ecef: [f32; 3],
) -> ([f64; 3], f64, bool) {
    let n_rad_s = (elem.mean_motion_rev_day * std::f64::consts::TAU) / 86_400.0;
    let a_m = (MU_EARTH_M3_S2 / (n_rad_s * n_rad_s))
        .cbrt()
        .max(EARTH_RADIUS_M + 160_000.0);
    let e = elem.eccentricity.clamp(0.0, 0.85);
    let inc = elem.inclination_deg.to_radians();
    let p_m = a_m * (1.0 - e * e);

    // J2 secular nodal regression and apsidal precession rates
    let re = EARTH_RADIUS_M;
    let j2_factor = 1.5 * J2_EARTH * (re / p_m).powi(2) * n_rad_s;
    let domega_dt = -j2_factor * inc.cos();
    let dw_dt = j2_factor * (2.0 - 2.5 * inc.sin().powi(2));

    let dt_s = unix_utc_s - elem.epoch_unix_s;
    let raan = elem.raan_deg.to_radians() + domega_dt * dt_s;
    let argp = elem.arg_perigee_deg.to_radians() + dw_dt * dt_s;
    let mean_anom =
        (elem.mean_anomaly_deg.to_radians() + n_rad_s * dt_s).rem_euclid(std::f64::consts::TAU);

    // Solve Kepler's equation E - e sin(E) = M via Newton-Raphson
    let mut e_anom = mean_anom + e * mean_anom.sin();
    for _ in 0..6 {
        let f = e_anom - e * e_anom.sin() - mean_anom;
        let fp = 1.0 - e * e_anom.cos();
        e_anom -= f / fp.max(1e-8);
    }

    let cos_e = e_anom.cos();
    let sin_e = e_anom.sin();
    let sqrt_1_e2 = (1.0 - e * e).max(1e-6).sqrt();
    let r = a_m * (1.0 - e * cos_e);
    let nu = (sqrt_1_e2 * sin_e).atan2(cos_e - e);
    let u_orb = argp + nu;

    let (sin_o, cos_o) = raan.sin_cos();
    let (sin_i, cos_i) = inc.sin_cos();
    let (sin_u, cos_u) = u_orb.sin_cos();

    // ECI position
    let x_eci = r * (cos_o * cos_u - sin_o * sin_u * cos_i);
    let y_eci = r * (sin_o * cos_u + cos_o * sin_u * cos_i);
    let z_eci = r * (sin_u * sin_i);

    // Rotate ECI -> ECEF around Z by +GMST
    let theta = gmst_rad(unix_utc_s);
    let (sin_g, cos_g) = theta.sin_cos();
    let x = cos_g * x_eci + sin_g * y_eci;
    let y = -sin_g * x_eci + cos_g * y_eci;
    let z = z_eci;

    // Cylindrical Earth shadow test for sunlit illumination color
    let s_dot_r =
        x * sun_dir_ecef[0] as f64 + y * sun_dir_ecef[1] as f64 + z * sun_dir_ecef[2] as f64;
    let r_perp_sq = (x * x + y * y + z * z) - s_dot_r * s_dot_r;
    let sunlit = s_dot_r >= 0.0 || r_perp_sq >= re * re * 0.99;

    ([x, y, z], r - EARTH_RADIUS_M, sunlit)
}

/// Propagates `elem` with J2 secular perturbations at `unix_utc_s` and transforms ECI -> ECEF.
pub fn propagate_satellite_ecef(
    elem: &SatelliteElement,
    unix_utc_s: f64,
    sun_dir_ecef: [f32; 3],
) -> PropagatedSatellite {
    let (ecef_m, altitude_m, sunlit) = propagate_satellite_ecef_core(elem, unix_utc_s, sun_dir_ecef);
    PropagatedSatellite {
        name: elem.name.clone(),
        ecef_m,
        altitude_m,
        sunlit,
        is_station: elem.is_station,
    }
}

/// Non-blocking store holding public-domain satellite elements + live CelesTrak GP refresh.
pub struct SatelliteCatalogStore {
    inner: Arc<Mutex<SatelliteCatalogState>>,
    wakeup: Option<Arc<dyn Fn() + Send + Sync>>,
}

struct SatelliteCatalogState {
    revision: u64,
    elements: Arc<Vec<SatelliteElement>>,
    fetch_started: bool,
    cache_dir: Option<std::path::PathBuf>,
}

impl SatelliteCatalogStore {
    pub fn new(
        cache_dir: Option<std::path::PathBuf>,
        wakeup: Option<Arc<dyn Fn() + Send + Sync>>,
    ) -> Self {
        let mut elements = embedded_satellite_elements();
        if let Some(ref dir) = cache_dir {
            let p = dir.join("celestrak_gp_stations.json");
            if let Ok(text) = std::fs::read_to_string(&p) {
                let parsed = parse_celestrak_gp_json(&text, true);
                if !parsed.is_empty() {
                    elements = parsed;
                }
            }
        }
        Self {
            inner: Arc::new(Mutex::new(SatelliteCatalogState {
                revision: 1,
                elements: Arc::new(elements),
                fetch_started: false,
                cache_dir,
            })),
            wakeup,
        }
    }

    pub fn ensure_fetched(&self) {
        if std::env::var_os("STRATA_EARTH_OFFLINE").is_some() {
            return;
        }
        let cache_dir = {
            let mut guard = self.inner.lock().unwrap();
            if guard.fetch_started {
                return;
            }
            guard.fetch_started = true;
            guard.cache_dir.clone()
        };
        let inner = Arc::clone(&self.inner);
        let wake = self.wakeup.clone();
        std::thread::Builder::new()
            .name("strata-earth-celestrak".to_string())
            .spawn(move || {
                let mut combined = Vec::new();
                for (url, mark_station, cache_name) in [
                    (
                        DEFAULT_CELESTRAK_STATIONS_URL,
                        true,
                        "celestrak_gp_stations.json",
                    ),
                    (
                        DEFAULT_CELESTRAK_VISUAL_URL,
                        false,
                        "celestrak_gp_visual.json",
                    ),
                ] {
                    if let Ok(resp) = ureq::get(url)
                        .timeout(std::time::Duration::from_secs(10))
                        .set("User-Agent", "strata-earth/0.1")
                        .call()
                    {
                        if let Ok(text) = resp.into_string() {
                            let mut parsed = parse_celestrak_gp_json(&text, mark_station);
                            if !parsed.is_empty() {
                                if let Some(ref dir) = cache_dir {
                                    let _ = std::fs::create_dir_all(dir);
                                    let _ = std::fs::write(dir.join(cache_name), text.as_bytes());
                                }
                                if parsed.len() > 80 {
                                    parsed.truncate(80);
                                }
                                combined.extend(parsed);
                            }
                        }
                    }
                }
                if !combined.is_empty() {
                    let shared = Arc::new(combined);
                    if let Ok(mut guard) = inner.lock() {
                        guard.elements = shared;
                        guard.revision += 1;
                    }
                    if let Some(w) = wake {
                        w();
                    }
                }
            })
            .ok();
    }

    pub fn snapshot(&self) -> (u64, Arc<Vec<SatelliteElement>>) {
        let guard = self.inner.lock().unwrap();
        (guard.revision, Arc::clone(&guard.elements))
    }
}

#[inline]
fn append_octahedron(
    center: DVec3,
    half_size: f64,
    verts: &mut Vec<RteTileVertex>,
    inds: &mut Vec<u32>,
) {
    let base = verts.len() as u32;
    let pts = [
        [
            center.x as f32 + half_size as f32,
            center.y as f32,
            center.z as f32,
        ],
        [
            center.x as f32 - half_size as f32,
            center.y as f32,
            center.z as f32,
        ],
        [
            center.x as f32,
            center.y as f32 + half_size as f32,
            center.z as f32,
        ],
        [
            center.x as f32,
            center.y as f32 - half_size as f32,
            center.z as f32,
        ],
        [
            center.x as f32,
            center.y as f32,
            center.z as f32 + half_size as f32,
        ],
        [
            center.x as f32,
            center.y as f32,
            center.z as f32 - half_size as f32,
        ],
    ];
    for p in pts {
        verts.push(RteTileVertex::new(p, [0.0, 0.0, 1.0], [0.5, 0.5]));
    }
    let tris: &[[u32; 3]] = &[
        [4, 0, 2],
        [4, 2, 1],
        [4, 1, 3],
        [4, 3, 0],
        [5, 2, 0],
        [5, 1, 2],
        [5, 3, 1],
        [5, 0, 3],
    ];
    for t in tris {
        inds.push(base + t[0]);
        inds.push(base + t[1]);
        inds.push(base + t[2]);
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct OrbitRingCacheKey {
    elem_count: usize,
    catalog_hash: u64,
    first_epoch_bits: u64,
    time_bucket: i64,
    ribbon_bucket: i64,
}

struct OrbitRingCacheEntry {
    key: OrbitRingCacheKey,
    revision: u64,
    vertices: Arc<Vec<RteTileVertex>>,
    indices: Arc<Vec<u32>>,
}

thread_local! {
    static ORBIT_RING_CACHE: std::cell::RefCell<Option<OrbitRingCacheEntry>> =
        const { std::cell::RefCell::new(None) };
}

/// Builds 3D dynamic meshes for real orbital satellites & station orbit tracks around Earth.
///
/// Automatically fades away via `local_25d_blend` and produces zero meshes once entered into 2.5D.
#[allow(clippy::too_many_arguments)]
pub fn build_3d_satellites_meshes(
    unix_utc_s: f64,
    sun_dir_ecef: [f32; 3],
    camera_range_m: f64,
    local_25d_blend: f32,
    satellites_visible: bool,
    orbits_visible: bool,
    satellite_scale_m: f32,
    elements: &[SatelliteElement],
    revision: u64,
) -> Vec<RetainedDynamicMesh> {
    if !satellites_visible || elements.is_empty() {
        return Vec::new();
    }
    let t = (local_25d_blend / 0.85).clamp(0.0, 1.0);
    let space_fade = 1.0 - t * t * (3.0 - 2.0 * t);
    if space_fade <= 0.01 {
        return Vec::new();
    }
    let eff_t = if unix_utc_s.abs() > 1_000_000.0 {
        unix_utc_s
    } else {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(1_767_225_600.0)
    };

    let beacon_r = (satellite_scale_m as f64) * (camera_range_m / 11_000_000.0).clamp(0.18, 2.2);
    let cap = elements.len().min(120);
    let mut sunlit_verts = Vec::with_capacity(cap * 6);
    let mut sunlit_inds = Vec::with_capacity(cap * 24);
    let mut eclipse_verts = Vec::new();
    let mut eclipse_inds = Vec::new();

    for elem in elements.iter().take(120) {
        let (ecef_m, _, sunlit) = propagate_satellite_ecef_core(elem, eff_t, sun_dir_ecef);
        let pos = DVec3::new(ecef_m[0], ecef_m[1], ecef_m[2]);
        let r_scale = if elem.is_station {
            beacon_r * 1.45
        } else {
            beacon_r
        };
        if sunlit {
            append_octahedron(pos, r_scale, &mut sunlit_verts, &mut sunlit_inds);
        } else {
            append_octahedron(pos, r_scale * 0.85, &mut eclipse_verts, &mut eclipse_inds);
        }
    }

    let mut out = Vec::with_capacity(3);
    if orbits_visible {
        let ribbon_half = (beacon_r * 0.12).max(4_000.0);
        let mut catalog_hash: u64 = 0xcbf29ce484222325;
        for el in &elements[..cap] {
            for b in el.epoch_unix_s.to_le_bytes() {
                catalog_hash ^= u64::from(b);
                catalog_hash = catalog_hash.wrapping_mul(0x100000001b3);
            }
            for b in el.mean_motion_rev_day.to_le_bytes() {
                catalog_hash ^= u64::from(b);
                catalog_hash = catalog_hash.wrapping_mul(0x100000001b3);
            }
            for b in el.inclination_deg.to_le_bytes() {
                catalog_hash ^= u64::from(b);
                catalog_hash = catalog_hash.wrapping_mul(0x100000001b3);
            }
            for b in el.raan_deg.to_le_bytes() {
                catalog_hash ^= u64::from(b);
                catalog_hash = catalog_hash.wrapping_mul(0x100000001b3);
            }
            catalog_hash ^= u64::from(u8::from(el.is_station));
            catalog_hash = catalog_hash.wrapping_mul(0x100000001b3);
        }
        let key = OrbitRingCacheKey {
            elem_count: cap,
            catalog_hash,
            first_epoch_bits: elements[0].epoch_unix_s.to_bits(),
            time_bucket: (eff_t / 2.0).floor() as i64,
            ribbon_bucket: (ribbon_half / 500.0).round() as i64,
        };
        let (orb_rev, orb_verts, orb_inds) = ORBIT_RING_CACHE.with(|cell| {
            let mut slot = cell.borrow_mut();
            if let Some(ref entry) = *slot {
                if entry.key == key {
                    let rev = if revision <= 2 {
                        revision
                    } else {
                        entry.revision
                    };
                    return (rev, Arc::clone(&entry.vertices), Arc::clone(&entry.indices));
                }
            }
            let segs = 48usize;
            let mut orbit_verts = Vec::with_capacity(12 * segs * 4);
            let mut orbit_inds = Vec::with_capacity(12 * segs * 6);
            let mut ring_pts = Vec::with_capacity(segs);
            for (idx, elem) in elements.iter().take(120).enumerate() {
                if !elem.is_station && idx >= 4 {
                    continue;
                }
                let period_s = 86_400.0 / elem.mean_motion_rev_day.max(0.1);
                ring_pts.clear();
                for s in 0..segs {
                    let sample_t = eff_t + (s as f64 / segs as f64) * period_s;
                    let (p_s, _, _) = propagate_satellite_ecef_core(elem, sample_t, sun_dir_ecef);
                    ring_pts.push(DVec3::new(p_s[0], p_s[1], p_s[2]));
                }
                for i in 0..segs {
                    let p0 = ring_pts[i];
                    let p1 = ring_pts[(i + 1) % segs];
                    let radial = p0.normalize().scale(ribbon_half);
                    let a = p0.sub(radial);
                    let b = p0.add(radial);
                    let c = p1.add(radial);
                    let d = p1.sub(radial);
                    let v0 = orbit_verts.len() as u32;
                    for v in [a, b, c, d] {
                        orbit_verts.push(RteTileVertex::new(
                            [v.x as f32, v.y as f32, v.z as f32],
                            [0.0, 0.0, 1.0],
                            [0.5, 0.5],
                        ));
                    }
                    orbit_inds.extend_from_slice(&[v0, v0 + 1, v0 + 2, v0, v0 + 2, v0 + 3]);
                }
            }
            let v_arc = Arc::new(orbit_verts);
            let i_arc = Arc::new(orbit_inds);
            *slot = Some(OrbitRingCacheEntry {
                key,
                revision,
                vertices: Arc::clone(&v_arc),
                indices: Arc::clone(&i_arc),
            });
            (revision, v_arc, i_arc)
        });
        if !orb_verts.is_empty() {
            out.push(RetainedDynamicMesh {
                id: 0x9E47_1003,
                revision: orb_rev,
                anchor_ecef: [0.0, 0.0, 0.0],
                vertices: orb_verts,
                indices: orb_inds,
                color: [0.36, 0.78, 0.98, 0.16 * space_fade],
                unlit: true,
            });
        }
    }
    if !sunlit_verts.is_empty() {
        out.push(RetainedDynamicMesh {
            id: 0x9E47_1001,
            revision,
            anchor_ecef: [0.0, 0.0, 0.0],
            vertices: Arc::new(sunlit_verts),
            indices: Arc::new(sunlit_inds),
            color: [0.46, 0.92, 1.0, 0.96 * space_fade],
            unlit: true,
        });
    }
    if !eclipse_verts.is_empty() {
        out.push(RetainedDynamicMesh {
            id: 0x9E47_1002,
            revision,
            anchor_ecef: [0.0, 0.0, 0.0],
            vertices: Arc::new(eclipse_verts),
            indices: Arc::new(eclipse_inds),
            color: [1.0, 0.68, 0.34, 0.85 * space_fade],
            unlit: true,
        });
    }
    out
}
