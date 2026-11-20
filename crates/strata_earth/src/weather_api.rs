//! Unified Base Weather API for `strata_earth`.
//!
//! Provides a single optional, pluggable meteorological model ([`WeatherProvider`]) producing
//! a coherent [`WeatherSnapshot`] at the camera's geographic location and altitude. Downstream
//! systems optionally synchronize with this single source (while still allowing independent
//! overrides):
//! - [`crate::precipitation::PrecipitationPlugin`] — temperature, wet-bulb phase, intensity, wind angle & speed, cloud ceiling
//! - [`crate::plugins::WeatherPlugin`] — 2.5D/3D cloud storm darkening, tint, and coverage boost
//! - [`crate::plugins::CelestialPlugin`] — 2.5D sky overcast darkening, sun attenuation, haze, and night star obscuration
//! - [`crate::ocean::OceanWavesPlugin`] — smooth wind-speed and wind-angle wave-train transitions

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::json::Json;
use crate::precipitation::{
    compute_precipitation_phase, isa_pressure_hpa, wet_bulb_temp_c, AtmosphereConfig,
    AtmosphereQuery, AtmosphereSample, OpenMeteoSounding, ISA_LAPSE_RATE_K_PER_KM,
};

/// Horizontal meteorological wind velocity specified by speed (`m/s`) and compass heading (`angle_deg`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WindVector {
    /// Wind speed in metres per second (`0.0..=80.0 m/s`).
    pub speed_mps: f32,
    /// Compass direction in degrees (`0°` = North, `90°` = East, `180°` = South, `270°` = West)
    /// specifying the direction the wind is blowing **TO** (velocity heading vector).
    ///
    /// Use [`Self::from_meteorological_from_deg`] when converting standard weather reports
    /// (which report the compass direction wind is blowing **FROM**).
    pub angle_deg: f32,
}

impl Default for WindVector {
    fn default() -> Self {
        Self {
            speed_mps: 6.0,
            angle_deg: 74.0,
        }
    }
}

impl WindVector {
    /// Creates a wind vector blowing **TO** `angle_deg` (`0°=N, 90°=E`) at `speed_mps`.
    #[inline]
    pub fn new(speed_mps: f32, angle_deg: f32) -> Self {
        Self {
            speed_mps: speed_mps.clamp(0.0, 80.0),
            angle_deg: angle_deg.rem_euclid(360.0),
        }
    }

    /// Creates a wind vector from standard meteorological convention (`dir_from_deg` = where wind
    /// originates FROM, e.g., `270°` West wind blows TO `90°` East).
    #[inline]
    pub fn from_meteorological_from_deg(speed_mps: f32, dir_from_deg: f32) -> Self {
        Self::new(speed_mps, dir_from_deg + 180.0)
    }

    /// Converts `(speed_mps, angle_deg)` to local ENU horizontal velocity `[east_mps, north_mps]`.
    #[inline]
    pub fn to_enu_mps(self) -> [f32; 2] {
        let rad = self.angle_deg.to_radians();
        [self.speed_mps * rad.sin(), self.speed_mps * rad.cos()]
    }

    /// Converts local ENU horizontal velocity `[east_mps, north_mps]` into a [`WindVector`].
    #[inline]
    pub fn from_enu_mps(enu: [f32; 2]) -> Self {
        let speed = (enu[0] * enu[0] + enu[1] * enu[1]).sqrt();
        if speed < 1e-4 {
            Self::new(0.0, 0.0)
        } else {
            let angle = enu[0].atan2(enu[1]).to_degrees().rem_euclid(360.0);
            Self::new(speed, angle)
        }
    }

    /// Returns a concise 8-point compass heading label (`N`, `NE`, `E`, `SE`, `S`, `SW`, `W`, `NW`).
    pub fn compass_label(self) -> &'static str {
        const DIRS: [&str; 8] = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"];
        let idx = ((self.angle_deg.rem_euclid(360.0) + 22.5) / 45.0).floor() as usize % 8;
        DIRS[idx]
    }
}

/// Geographic, altitude, and timestamp query passed to a [`WeatherProvider`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WeatherQuery {
    pub lat_deg: f64,
    pub lon_deg: f64,
    pub altitude_msl_m: f64,
    pub ground_elevation_m: f64,
    pub unix_utc_s: f64,
}

/// Single coherent physical meteorological state evaluated once per frame at the camera.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WeatherSnapshot {
    /// Resolved vertical atmospheric column state (T, Tw, pressure, RH, freezing level, ceiling, phase).
    pub atmosphere: AtmosphereSample,
    /// Horizontal wind velocity (speed + compass direction).
    pub wind: WindVector,
    /// Total sky cloud cover fraction (`0.0` = crystal clear, `1.0` = full overcast).
    pub cloud_cover: f32,
    /// Normalized precipitation intensity (`0.0` = dry, `0.3` = drizzle, `0.85` = steady rain/snow, `1.5+` = storm).
    pub precipitation_intensity: f32,
    /// Optical overcast/storm darkness (`0.0` = sunlit fair weather, `1.0` = brooding nimbostratus storm).
    pub storm_darkness: f32,
    /// Horizontal atmospheric visibility in metres (`35_000 m` crystal clear -> `1_200 m` heavy storm mist).
    pub visibility_m: f32,
    /// Optional WMO weather code if sourced from a live meteorological feed.
    pub wmo_code: Option<u8>,
    /// `true` when backed by a fetched live weather API response rather than static/analytical fallback.
    pub source_live: bool,
}

impl WeatherSnapshot {
    /// Returns normalized atmospheric haze in `[0.0..=1.0]` derived from visibility and precipitation.
    #[inline]
    pub fn normalized_haze(&self) -> f32 {
        let vis_haze = 1.0 - ((self.visibility_m.clamp(800.0, 35_000.0) - 800.0) / 34_200.0);
        let precip_haze = (self.precipitation_intensity * 0.42).clamp(0.0, 0.72);
        vis_haze.max(precip_haze).clamp(0.0, 1.0)
    }

    /// Derives physically consistent optical cloud darkness, cloud cover, and visibility from
    /// `precipitation_intensity` when the caller only specifies base meteorology.
    pub fn with_derived_optics(mut self) -> Self {
        let p = self.precipitation_intensity.clamp(0.0, 2.5);
        if p > 0.02 {
            let precip_t = (p / 1.15).clamp(0.0, 1.0);
            self.cloud_cover = self.cloud_cover.max(0.45 + 0.50 * precip_t).clamp(0.0, 1.0);
            self.storm_darkness = self
                .storm_darkness
                .max(0.28 + 0.58 * precip_t)
                .clamp(0.0, 1.0);
            let target_vis = 18_000.0 - 14_500.0 * precip_t;
            self.visibility_m = self.visibility_m.min(target_vis).clamp(900.0, 45_000.0);
        } else if self.cloud_cover > 0.2 {
            let overcast_dark = ((self.cloud_cover - 0.25) / 0.75).clamp(0.0, 1.0) * 0.42;
            self.storm_darkness = self.storm_darkness.max(overcast_dark).clamp(0.0, 1.0);
        }
        self
    }
}

/// Declarative Unified Base Weather configuration stored on [`crate::scene::EarthInput`].
///
/// When `enabled == true`, `strata_earth` evaluates the configured [`WeatherProvider`] once per
/// frame and shares the resulting [`WeatherSnapshot`] with precipitation, 2.5D/3D clouds,
/// 2.5D sky/haze, and ocean waves (each system can opt in or override independently).
#[derive(Debug, Clone, PartialEq)]
pub struct BaseWeatherConfig {
    /// Enables the unified base weather pipeline (`false` by default for zero overhead when unused).
    pub enabled: bool,
    /// Uses the built-in non-blocking [`OpenMeteoWeatherProvider`] to fetch real-world temperature,
    /// vertical pressure-level soundings, wind speed/angle, cloud cover, and precipitation rate.
    pub use_live_api: bool,
    /// Analytical vertical atmospheric column fallback (base temp, SeaLevel/Terrain anchor, lapse rate, ceiling).
    pub atmosphere: AtmosphereConfig,
    /// Default / fallback horizontal wind vector (speed m/s + compass angle deg).
    pub wind: WindVector,
    /// Base sky cloud cover fraction (`0.0..=1.0`).
    pub cloud_cover: f32,
    /// Base precipitation intensity (`0.0..=2.5`).
    pub precipitation_intensity: f32,
    /// Base storm / overcast cloud & sky darkening (`0.0..=1.0`).
    pub storm_darkness: f32,
    /// Base horizontal visibility in metres (`30_000.0` clear default).
    pub visibility_m: f32,
    /// Automatically derives coherent storm darkening, cloud cover boost, and rain haze from
    /// `precipitation_intensity` (`true` default).
    pub auto_derive_optics: bool,
}

impl Default for BaseWeatherConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            use_live_api: false,
            atmosphere: AtmosphereConfig::default(),
            wind: WindVector::default(),
            cloud_cover: 0.20,
            precipitation_intensity: 0.0,
            storm_darkness: 0.0,
            visibility_m: 32_000.0,
            auto_derive_optics: true,
        }
    }
}

impl BaseWeatherConfig {
    /// Evaluates the static analytical [`WeatherSnapshot`] at `query`.
    pub fn evaluate(&self, query: WeatherQuery) -> WeatherSnapshot {
        let mut atmo = self
            .atmosphere
            .evaluate(query.lat_deg, query.altitude_msl_m, query.ground_elevation_m);
        atmo.wind_mps = Some(self.wind.to_enu_mps());
        let snap = WeatherSnapshot {
            atmosphere: atmo,
            wind: self.wind,
            cloud_cover: self.cloud_cover.clamp(0.0, 1.0),
            precipitation_intensity: self.precipitation_intensity.clamp(0.0, 2.5),
            storm_darkness: self.storm_darkness.clamp(0.0, 1.0),
            visibility_m: self.visibility_m.clamp(500.0, 50_000.0),
            wmo_code: None,
            source_live: false,
        };
        if self.auto_derive_optics {
            snap.with_derived_optics()
        } else {
            snap
        }
    }

    pub fn enabled(mut self) -> Self {
        self.enabled = true;
        self
    }

    pub fn with_live_api(mut self, live: bool) -> Self {
        self.use_live_api = live;
        self
    }

    pub fn with_wind(mut self, speed_mps: f32, angle_deg: f32) -> Self {
        self.wind = WindVector::new(speed_mps, angle_deg);
        self
    }

    pub fn with_precipitation(mut self, intensity: f32) -> Self {
        self.precipitation_intensity = intensity.clamp(0.0, 2.5);
        self
    }

    pub fn with_cloud_cover(mut self, cover: f32) -> Self {
        self.cloud_cover = cover.clamp(0.0, 1.0);
        self
    }

    pub fn with_storm_darkness(mut self, darkness: f32) -> Self {
        self.storm_darkness = darkness.clamp(0.0, 1.0);
        self
    }

    pub fn with_base_temp_c(mut self, temp_c: f64) -> Self {
        self.atmosphere.base_temp_c = temp_c;
        self
    }
}

/// Pluggable unified Base Weather API trait driving precipitation, clouds, sky, and ocean waves.
pub trait WeatherProvider: Send + Sync + 'static {
    fn sample(&self, query: WeatherQuery, fallback: &BaseWeatherConfig) -> WeatherSnapshot;
}

/// Zero-network analytical [`WeatherProvider`] backed directly by [`BaseWeatherConfig`].
#[derive(Debug, Default, Clone, Copy)]
pub struct StaticWeatherProvider;

impl WeatherProvider for StaticWeatherProvider {
    fn sample(&self, query: WeatherQuery, fallback: &BaseWeatherConfig) -> WeatherSnapshot {
        fallback.evaluate(query)
    }
}

/// Adapter turning any closure into a unified [`WeatherProvider`].
pub struct FnWeatherProvider<F>(pub F);

impl<F> WeatherProvider for FnWeatherProvider<F>
where
    F: Fn(WeatherQuery, &BaseWeatherConfig) -> WeatherSnapshot + Send + Sync + 'static,
{
    fn sample(&self, query: WeatherQuery, fallback: &BaseWeatherConfig) -> WeatherSnapshot {
        (self.0)(query, fallback)
    }
}

/// Parsed unified Open-Meteo surface + vertical pressure-level weather report.
#[derive(Debug, Clone, PartialEq)]
pub struct OpenMeteoWeatherReport {
    /// Vertical pressure-level temperature sounding.
    pub sounding: OpenMeteoSounding,
    /// Optional vertical wind profile `(altitude_msl_m, WindVector)`.
    pub wind_levels: Vec<(f64, WindVector)>,
    /// Surface 10m wind vector.
    pub surface_wind: WindVector,
    /// Total cloud cover fraction (`0.0..=1.0`).
    pub cloud_cover: f32,
    /// Surface precipitation rate in mm/h (`rain + snowfall + showers`).
    pub precipitation_mm_h: f32,
    /// Optional WMO weather interpretation code (`0` clear, `51..67` rain, `71..77` snow, `95..99` storm).
    pub wmo_code: Option<u8>,
}

impl OpenMeteoWeatherReport {
    /// Interpolates wind vector at `altitude_msl_m` (or returns `surface_wind` if no upper levels).
    pub fn wind_at_altitude(&self, altitude_msl_m: f64) -> WindVector {
        if self.wind_levels.is_empty() {
            return self.surface_wind;
        }
        if altitude_msl_m <= self.wind_levels[0].0 {
            return self.wind_levels[0].1;
        }
        let last = self.wind_levels[self.wind_levels.len() - 1];
        if altitude_msl_m >= last.0 {
            return last.1;
        }
        for pair in self.wind_levels.windows(2) {
            let (z0, w0) = pair[0];
            let (z1, w1) = pair[1];
            if altitude_msl_m >= z0 && altitude_msl_m <= z1 && (z1 - z0).abs() > 1e-6 {
                let t = ((altitude_msl_m - z0) / (z1 - z0)) as f32;
                let v0 = w0.to_enu_mps();
                let v1 = w1.to_enu_mps();
                return WindVector::from_enu_mps([
                    v0[0] + (v1[0] - v0[0]) * t,
                    v0[1] + (v1[1] - v0[1]) * t,
                ]);
            }
        }
        self.surface_wind
    }

    /// Evaluates a complete [`WeatherSnapshot`] at `query`.
    pub fn to_snapshot(&self, query: WeatherQuery, fallback: &BaseWeatherConfig) -> WeatherSnapshot {
        let mut atmo = self.sounding.to_sample(
            AtmosphereQuery {
                lat_deg: query.lat_deg,
                lon_deg: query.lon_deg,
                altitude_msl_m: query.altitude_msl_m,
                ground_elevation_m: query.ground_elevation_m,
            },
            &fallback.atmosphere,
        );
        let wind = self.wind_at_altitude(query.altitude_msl_m);
        atmo.wind_mps = Some(wind.to_enu_mps());
        atmo.pressure_hpa = isa_pressure_hpa(query.altitude_msl_m);
        atmo.wet_bulb_c = wet_bulb_temp_c(atmo.temperature_c, atmo.relative_humidity);
        atmo.phase = compute_precipitation_phase(atmo.wet_bulb_c);

        // Map mm/h precipitation rate smoothly onto normalized intensity [0.0..=1.8]:
        // 0.2 mm/h = light drizzle (~0.25), 2.5 mm/h = steady rain (~0.85), 10+ mm/h = heavy storm (~1.5).
        let mut precip_intensity = if self.precipitation_mm_h > 0.02 {
            (0.18 + (self.precipitation_mm_h / 4.2).powf(0.62) * 0.92).clamp(0.0, 1.8)
        } else {
            0.0
        };
        let wmo_storm: f32 = match self.wmo_code.unwrap_or(0) {
            95..=99 => 0.92,
            80..=86 => 0.72,
            61..=77 => 0.62,
            51..=57 => 0.45,
            45..=48 => 0.40,
            3 => 0.35,
            2 => 0.16,
            _ => 0.0,
        };
        if precip_intensity < 0.05 && matches!(self.wmo_code, Some(51..=99)) {
            precip_intensity = 0.45;
        }
        let storm_darkness = (wmo_storm.max(self.cloud_cover * 0.48)
            + (precip_intensity * 0.32).min(0.45))
        .clamp(0.0, 1.0);
        let visibility_m = (32_000.0
            - self.cloud_cover * 6_000.0
            - precip_intensity.min(1.5) * 16_000.0)
            .clamp(1_000.0, 35_000.0);

        let snap = WeatherSnapshot {
            atmosphere: atmo,
            wind,
            cloud_cover: self.cloud_cover.clamp(0.0, 1.0),
            precipitation_intensity: precip_intensity,
            storm_darkness,
            visibility_m,
            wmo_code: self.wmo_code,
            source_live: true,
        };
        if fallback.auto_derive_optics {
            snap.with_derived_optics()
        } else {
            snap
        }
    }
}

fn json_field<'a>(obj: &'a [(String, Json)], key: &str) -> Option<&'a Json> {
    obj.iter().find(|(k, _)| k == key).map(|(_, v)| v)
}

fn json_num(val: Option<&Json>) -> Option<f64> {
    match val? {
        Json::Number(n) if n.is_finite() => Some(*n),
        _ => None,
    }
}

fn json_first_num(obj: &[(String, Json)], key: &str) -> Option<f64> {
    let Json::Array(arr) = json_field(obj, key)? else {
        return None;
    };
    arr.first().and_then(|v| json_num(Some(v)))
}

/// Parses an Open-Meteo `/v1/forecast` JSON payload into a unified [`OpenMeteoWeatherReport`].
pub fn parse_open_meteo_weather_json(json_text: &str) -> Result<OpenMeteoWeatherReport, String> {
    let Json::Object(root) =
        Json::parse(json_text).map_err(|e| format!("open-meteo weather json parse error: {e}"))?
    else {
        return Err("open-meteo weather root is not a JSON object".to_string());
    };

    let elev_m = json_num(json_field(&root, "elevation")).unwrap_or(0.0);
    let mut temp_levels: Vec<(f64, f64)> = Vec::new();
    let mut wind_levels: Vec<(f64, WindVector)> = Vec::new();
    let mut relative_humidity = 0.80f64;
    let mut surface_wind = WindVector::default();
    let mut cloud_cover = 0.25f32;
    let mut precipitation_mm_h = 0.0f32;
    let mut wmo_code: Option<u8> = None;

    if let Some(Json::Object(cur)) = json_field(&root, "current") {
        if let Some(t2m) = json_num(json_field(cur, "temperature_2m")) {
            temp_levels.push((elev_m + 2.0, t2m));
        }
        if let Some(rh) = json_num(json_field(cur, "relative_humidity_2m")) {
            relative_humidity = (rh / 100.0).clamp(0.05, 1.0);
        }
        if let (Some(spd_kmh), Some(dir_from)) = (
            json_num(json_field(cur, "wind_speed_10m")),
            json_num(json_field(cur, "wind_direction_10m")),
        ) {
            surface_wind =
                WindVector::from_meteorological_from_deg((spd_kmh / 3.6) as f32, dir_from as f32);
            wind_levels.push((elev_m + 10.0, surface_wind));
        }
        if let Some(cc) = json_num(json_field(cur, "cloud_cover")) {
            cloud_cover = (cc / 100.0).clamp(0.0, 1.0) as f32;
        }
        let precip = json_num(json_field(cur, "precipitation")).unwrap_or(0.0);
        let rain = json_num(json_field(cur, "rain")).unwrap_or(0.0);
        let snow_cm = json_num(json_field(cur, "snowfall")).unwrap_or(0.0);
        precipitation_mm_h = precip.max(rain + snow_cm * 7.0).max(0.0) as f32;
        if let Some(code) = json_num(json_field(cur, "weather_code")) {
            if (0.0..=99.0).contains(&code) {
                wmo_code = Some(code.round() as u8);
            }
        }
    }

    let mut freezing_level_msl_m = None;
    if let Some(Json::Object(hourly)) = json_field(&root, "hourly") {
        freezing_level_msl_m = json_first_num(hourly, "freezing_level_height");
        const LEVELS: &[(&str, &str, &str, &str)] = &[
            (
                "temperature_1000hPa",
                "geopotential_height_1000hPa",
                "wind_speed_1000hPa",
                "wind_direction_1000hPa",
            ),
            (
                "temperature_925hPa",
                "geopotential_height_925hPa",
                "wind_speed_925hPa",
                "wind_direction_925hPa",
            ),
            (
                "temperature_850hPa",
                "geopotential_height_850hPa",
                "wind_speed_850hPa",
                "wind_direction_850hPa",
            ),
            (
                "temperature_700hPa",
                "geopotential_height_700hPa",
                "wind_speed_700hPa",
                "wind_direction_700hPa",
            ),
            (
                "temperature_500hPa",
                "geopotential_height_500hPa",
                "wind_speed_500hPa",
                "wind_direction_500hPa",
            ),
        ];
        for &(tk, zk, wsk, wdk) in LEVELS {
            if let Some(z_m) = json_first_num(hourly, zk) {
                if let Some(t_c) = json_first_num(hourly, tk) {
                    temp_levels.push((z_m, t_c));
                }
                if let (Some(ws_kmh), Some(wd_from)) =
                    (json_first_num(hourly, wsk), json_first_num(hourly, wdk))
                {
                    wind_levels.push((
                        z_m,
                        WindVector::from_meteorological_from_deg(
                            (ws_kmh / 3.6) as f32,
                            wd_from as f32,
                        ),
                    ));
                }
            }
        }
    }

    if temp_levels.is_empty() {
        return Err("open-meteo weather payload contained no temperature levels".to_string());
    }
    temp_levels.sort_by(|a, b| a.0.total_cmp(&b.0));
    temp_levels.dedup_by(|a, b| (a.0 - b.0).abs() < 15.0);
    wind_levels.sort_by(|a, b| a.0.total_cmp(&b.0));
    wind_levels.dedup_by(|a, b| (a.0 - b.0).abs() < 15.0);

    // Ensure freezing level falls back sensibly if omitted from JSON.
    let _ = ISA_LAPSE_RATE_K_PER_KM;
    Ok(OpenMeteoWeatherReport {
        sounding: OpenMeteoSounding {
            levels: temp_levels,
            relative_humidity,
            freezing_level_msl_m,
            wind_mps: Some(surface_wind.to_enu_mps()),
        },
        wind_levels,
        surface_wind,
        cloud_cover,
        precipitation_mm_h,
        wmo_code,
    })
}

const OPEN_METEO_MAX_IN_FLIGHT: usize = 4;
const OPEN_METEO_BASE_BACKOFF_SECS: u64 = 30;
const OPEN_METEO_MAX_BACKOFF_SECS: u64 = 300;

struct CachedWeatherEntry {
    fetched_at: Instant,
    report: OpenMeteoWeatherReport,
}

struct OpenMeteoWeatherCache {
    entries: HashMap<(i32, i32), CachedWeatherEntry>,
    in_flight: HashSet<(i32, i32)>,
    /// Per-cell negative cache `(retry_after, consecutive_failures)` preventing thread storms offline.
    failed_until: HashMap<(i32, i32), (Instant, u32)>,
    /// Global cooldown after consecutive transport failures so panning offline doesn't spawn new threads per cell.
    global_backoff_until: Option<Instant>,
    consecutive_failures: u32,
}

impl OpenMeteoWeatherCache {
    fn record_failure(&mut self, key: (i32, i32), base_backoff: Duration) {
        self.in_flight.remove(&key);
        let cell_fails = self
            .failed_until
            .get(&key)
            .map(|(_, c)| c.saturating_add(1))
            .unwrap_or(1);
        let shift = (cell_fails - 1).min(4);
        let secs = (base_backoff.as_secs().max(1) << shift).min(OPEN_METEO_MAX_BACKOFF_SECS);
        let backoff = Duration::from_secs(secs).max(base_backoff);
        self.failed_until
            .insert(key, (Instant::now() + backoff, cell_fails));
        self.consecutive_failures = self.consecutive_failures.saturating_add(1);
        let g_shift = (self.consecutive_failures - 1).min(4);
        let g_secs = (base_backoff.as_secs().max(1) << g_shift).min(OPEN_METEO_MAX_BACKOFF_SECS);
        let g_backoff = Duration::from_secs(g_secs).max(base_backoff);
        self.global_backoff_until = Some(Instant::now() + g_backoff);
    }

    fn record_success(&mut self, key: (i32, i32), report: OpenMeteoWeatherReport) {
        self.in_flight.remove(&key);
        self.failed_until.remove(&key);
        self.global_backoff_until = None;
        self.consecutive_failures = 0;
        self.entries.insert(
            key,
            CachedWeatherEntry {
                fetched_at: Instant::now(),
                report,
            },
        );
    }
}

/// Optional live unified weather provider backed by Open-Meteo (`api.open-meteo.com`).
///
/// Quantizes geographic lookups to `0.25°` cells with a 10-minute TTL cache and non-blocking
/// background HTTP fetches (falling back cleanly to [`BaseWeatherConfig`] while loading or
/// whenever `STRATA_EARTH_OFFLINE` is set).
#[derive(Clone)]
pub struct OpenMeteoWeatherProvider {
    endpoint: String,
    ttl: Duration,
    base_backoff: Duration,
    transport: Option<Arc<dyn crate::tiles::HttpTransport>>,
    cache: Arc<Mutex<OpenMeteoWeatherCache>>,
}

impl Default for OpenMeteoWeatherProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl OpenMeteoWeatherProvider {
    pub const DEFAULT_ENDPOINT: &'static str = "https://api.open-meteo.com/v1/forecast";

    pub fn new() -> Self {
        Self {
            endpoint: Self::DEFAULT_ENDPOINT.to_string(),
            ttl: Duration::from_secs(600),
            base_backoff: Duration::from_secs(OPEN_METEO_BASE_BACKOFF_SECS),
            transport: None,
            cache: Arc::new(Mutex::new(OpenMeteoWeatherCache {
                entries: HashMap::new(),
                in_flight: HashSet::new(),
                failed_until: HashMap::new(),
                global_backoff_until: None,
                consecutive_failures: 0,
            })),
        }
    }

    pub fn with_transport(transport: Arc<dyn crate::tiles::HttpTransport>) -> Self {
        let mut p = Self::new();
        p.transport = Some(transport);
        p
    }

    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = endpoint.into();
        self
    }

    pub fn with_failure_backoff(mut self, backoff: Duration) -> Self {
        self.base_backoff = backoff.max(Duration::from_millis(50));
        self
    }

    fn quantize_cell(lat_deg: f64, lon_deg: f64) -> (i32, i32) {
        (
            (lat_deg.clamp(-89.75, 89.75) * 4.0).round() as i32,
            (((lon_deg + 180.0).rem_euclid(360.0) - 180.0) * 4.0).round() as i32,
        )
    }

    /// Builds the Open-Meteo forecast query URL for `(lat_deg, lon_deg)`.
    pub fn build_query_url(&self, lat_deg: f64, lon_deg: f64) -> String {
        format!(
            "{}?latitude={:.2}&longitude={:.2}&current=temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m,cloud_cover,precipitation,rain,snowfall,weather_code&hourly=temperature_1000hPa,temperature_925hPa,temperature_850hPa,temperature_700hPa,temperature_500hPa,geopotential_height_1000hPa,geopotential_height_925hPa,geopotential_height_850hPa,geopotential_height_700hPa,geopotential_height_500hPa,wind_speed_1000hPa,wind_speed_925hPa,wind_speed_850hPa,wind_speed_700hPa,wind_speed_500hPa,wind_direction_1000hPa,wind_direction_925hPa,wind_direction_850hPa,wind_direction_700hPa,wind_direction_500hPa,freezing_level_height&forecast_hours=1",
            self.endpoint, lat_deg, lon_deg
        )
    }

    /// Inserts a pre-parsed weather report into the cell cache (for deterministic tests or preload).
    pub fn insert_cached_report(&self, lat_deg: f64, lon_deg: f64, report: OpenMeteoWeatherReport) {
        let key = Self::quantize_cell(lat_deg, lon_deg);
        if let Ok(mut guard) = self.cache.lock() {
            guard.record_success(key, report);
        }
    }

    fn ensure_cell_fetched(&self, key: (i32, i32)) {
        if self.transport.is_none() && std::env::var_os("STRATA_EARTH_OFFLINE").is_some() {
            return;
        }
        let now = Instant::now();
        let should_spawn = if let Ok(mut guard) = self.cache.lock() {
            if let Some(entry) = guard.entries.get(&key) {
                if entry.fetched_at.elapsed() < self.ttl {
                    return;
                }
            }
            if guard
                .global_backoff_until
                .is_some_and(|until| now < until)
            {
                return;
            }
            if guard
                .failed_until
                .get(&key)
                .is_some_and(|(until, _)| now < *until)
            {
                return;
            }
            if guard.in_flight.len() >= OPEN_METEO_MAX_IN_FLIGHT {
                return;
            }
            guard.in_flight.insert(key)
        } else {
            false
        };
        if !should_spawn {
            return;
        }
        let lat = (key.0 as f64) * 0.25;
        let lon = (key.1 as f64) * 0.25;
        let url = self.build_query_url(lat, lon);
        let cache = Arc::clone(&self.cache);
        let base_backoff = self.base_backoff;
        let custom_transport = self.transport.clone();
        let spawn_res = std::thread::Builder::new()
            .name("strata-earth-weather".to_string())
            .spawn(move || {
                let parsed = if let Some(transport) = custom_transport {
                    transport
                        .fetch(&url)
                        .ok()
                        .and_then(|bytes| String::from_utf8(bytes).ok())
                        .and_then(|body| parse_open_meteo_weather_json(&body).ok())
                } else {
                    let agent = ureq::AgentBuilder::new()
                        .timeout_connect(Duration::from_secs(5))
                        .timeout_read(Duration::from_secs(8))
                        .build();
                    agent
                        .get(&url)
                        .set("User-Agent", "strata-earth/0.1 (unified-weather)")
                        .call()
                        .ok()
                        .and_then(|resp| resp.into_string().ok())
                        .and_then(|body| parse_open_meteo_weather_json(&body).ok())
                };
                if let Ok(mut guard) = cache.lock() {
                    if let Some(report) = parsed {
                        guard.record_success(key, report);
                    } else {
                        guard.record_failure(key, base_backoff);
                    }
                }
            });
        if spawn_res.is_err() {
            if let Ok(mut guard) = self.cache.lock() {
                guard.record_failure(key, base_backoff);
            }
        }
    }
}

impl WeatherProvider for OpenMeteoWeatherProvider {
    fn sample(&self, query: WeatherQuery, fallback: &BaseWeatherConfig) -> WeatherSnapshot {
        let key = Self::quantize_cell(query.lat_deg, query.lon_deg);
        let cached = self
            .cache
            .lock()
            .ok()
            .and_then(|g| g.entries.get(&key).map(|e| e.report.clone()));
        self.ensure_cell_fetched(key);
        if let Some(report) = cached {
            report.to_snapshot(query, fallback)
        } else {
            fallback.evaluate(query)
        }
    }
}
