//! Physically accurate atmospheric thermodynamics, optional live altitude temperature soundings,
//! generalized 2D camera-local precipitation layers, and [`PrecipitationPlugin`] for `strata_earth`.
//!
//! # Architecture
//! - **Atmosphere & Thermodynamics**: Evaluates air temperature $T(z)$, barometric pressure $P(z)$,
//!   Stull wet-bulb temperature $T_w(z)$, $0^\circ\text{C}$ freezing level, and precipitating cloud
//!   ceiling across the vertical atmospheric column using either Mean Sea Level (`SeaLevel`, ISA
//!   standard default) or `TerrainSurface` anchoring, with `StandardIsa` (`6.5 K/km`) or exact
//!   Clausius-Clapeyron `MoistAdiabatic` lapse rates.
//! - **Pluggable Temperature Providers**: Supply [`StaticAtmosphereProvider`] (zero-network
//!   default), [`OpenMeteoTemperatureProvider`] (optional live pressure-level atmospheric sounding),
//!   or any custom [`TemperatureProvider`] / [`FnTemperatureProvider`].
//! - **Generalized 2D GPU Particle Layers & Custom Evaluator**: Every precipitation type (fast 2D
//!   rain streaks, simplistic mathematical 6-fold 2D snowflakes, sleet pellets, or arbitrary custom
//!   2D RGBA8 sprite textures) is expressed as a [`PrecipitationLayer`] evaluated per frame by a
//!   composable [`PrecipitationEvaluator`] function and rendered in one GPU-instanced draw call per
//!   layer with world-locked toroidal camera-local panning.

use std::any::Any;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

use strata_render::camera64::DVec3;
use strata_render::scene3d::{
    RetainedParticleBillboard, RetainedParticleLayer, RetainedParticleSprite, RetainedTileTexture,
};

use crate::geodesy::{
    ecef_to_geodetic, project_morphed_point_and_frame, stabilized_surface_morph, Geodetic,
};
use crate::json::Json;
use crate::plugins::{
    EarthParticleContext, EarthPlugin, EarthPluginFrameContext, EarthPluginHooks,
};
use crate::scene::EarthViewMode;

/// Standard ISA mean sea-level atmospheric pressure in hPa (mbar).
pub const ISA_SEA_LEVEL_PRESSURE_HPA: f64 = 1013.25;
/// Standard ISA tropospheric environmental lapse rate in K/km (`6.5 °C/km`).
pub const ISA_LAPSE_RATE_K_PER_KM: f64 = 6.5;
/// Standard ISA tropopause altitude in metres MSL (`11 km`).
pub const ISA_TROPOPAUSE_ALT_M: f64 = 11_000.0;
/// Altitude of the GPU global/local cloud shell (`6,391,500 m - 6,378,137 m = 13,363 m`).
pub const CLOUD_SHELL_ALTITUDE_M: f64 = 13_363.0;
/// Default precipitating cloud deck ceiling altitude in metres MSL (`5,500 m`).
pub const DEFAULT_CLOUD_CEILING_MSL_M: f64 = 5_500.0;
/// Default cloud-base smooth fade transition thickness in metres (`900 m`).
pub const DEFAULT_CLOUD_FADE_BAND_M: f64 = 900.0;

#[inline]
fn smoothstep_f64(edge0: f64, edge1: f64, x: f64) -> f64 {
    if (edge1 - edge0).abs() < 1e-12 {
        return if x >= edge1 { 1.0 } else { 0.0 };
    }
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[inline]
fn smoothstep_f32(edge0: f32, edge1: f32, x: f32) -> f32 {
    smoothstep_f64(edge0 as f64, edge1 as f64, x as f64) as f32
}

// ---------------------------------------------------------------------------
// Atmospheric Physics & Thermodynamics
// ---------------------------------------------------------------------------

/// Altitude reference frame used to anchor base surface temperature.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum TemperatureAnchor {
    /// Base temperature is specified at Mean Sea Level (`z = 0 m` MSL / WGS84 ellipsoid).
    ///
    /// Physically accurate across varied topography (ISO 2533 / ISA standard): moving from a
    /// sea-level valley onto a high mountain automatically cools through the freezing level.
    #[default]
    SeaLevel,
    /// Base temperature is specified at the local terrain surface beneath the camera
    /// (`z = ground_elevation_m`), lapsing with height Above Ground Level (`z - ground_elevation_m`).
    TerrainSurface,
}

/// Vertical tropospheric temperature lapse-rate model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LapseRateModel {
    /// International Standard Atmosphere (`6.5 K/km` up to `11 km` tropopause, isothermal above).
    StandardIsa,
    /// Exact Clausius-Clapeyron saturated (moist) adiabatic lapse rate $\Gamma_m(T, P)$ integrated
    /// vertically — physically authentic inside precipitating saturated air columns.
    MoistAdiabatic,
    /// Custom constant tropospheric lapse rate in `K/km` (positive = cooling with altitude).
    Custom(f64),
}

impl Default for LapseRateModel {
    fn default() -> Self {
        Self::StandardIsa
    }
}

/// Vertical ceiling above which precipitation fades out and stops completely.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CloudCeiling {
    /// Fixed precipitating cloud ceiling in metres above Mean Sea Level (default `5,500 m`).
    FixedMsl(f64),
    /// Precipitating cloud ceiling in metres Above Ground Level (`ground_elevation_m + offset`).
    AboveGround(f64),
    /// Synchronized with the GPU cloud shell altitude (`13,363 m` MSL).
    CloudShell,
}

impl Default for CloudCeiling {
    fn default() -> Self {
        Self::FixedMsl(DEFAULT_CLOUD_CEILING_MSL_M)
    }
}

impl CloudCeiling {
    /// Resolves the active precipitating cloud ceiling altitude in metres MSL.
    #[inline]
    pub fn resolve_msl_m(self, ground_elevation_m: f64) -> f64 {
        match self {
            Self::FixedMsl(alt) => alt.max(50.0),
            Self::AboveGround(agl) => ground_elevation_m.max(0.0) + agl.max(50.0),
            Self::CloudShell => CLOUD_SHELL_ALTITUDE_M,
        }
    }
}

/// Smooth hydrometeor phase distribution across the melting layer (`rain + sleet + snow == 1.0`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PrecipitationPhase {
    pub rain: f32,
    pub sleet: f32,
    pub snow: f32,
}

impl Default for PrecipitationPhase {
    fn default() -> Self {
        Self {
            rain: 1.0,
            sleet: 0.0,
            snow: 0.0,
        }
    }
}

impl PrecipitationPhase {
    /// Concise human-readable phase label for HUD/telemetry displays.
    pub fn label(self) -> &'static str {
        if self.snow >= 0.85 {
            "Snow"
        } else if self.rain >= 0.85 {
            "Rain"
        } else if self.sleet >= 0.25 {
            "Mixed / Sleet"
        } else if self.snow > self.rain {
            "Wet Snow"
        } else {
            "Rain / Mix"
        }
    }
}

/// Computes International Standard Atmosphere barometric pressure in hPa (mbar) at `altitude_msl_m`.
#[inline]
pub fn isa_pressure_hpa(altitude_msl_m: f64) -> f64 {
    let z = altitude_msl_m.clamp(-500.0, 47_000.0);
    if z <= ISA_TROPOPAUSE_ALT_M {
        ISA_SEA_LEVEL_PRESSURE_HPA * (1.0 - (0.0065 * z) / 288.15).max(1e-4).powf(5.255_88)
    } else {
        let p_trop = ISA_SEA_LEVEL_PRESSURE_HPA
            * (1.0 - (0.0065 * ISA_TROPOPAUSE_ALT_M) / 288.15).powf(5.255_88);
        p_trop * (-9.806_65 * (z - ISA_TROPOPAUSE_ALT_M) / (287.058 * 216.65)).exp()
    }
}

/// Computes saturation vapor pressure $e_s(T)$ in Pascals using the Alduchov–Eskridge / Magnus
/// formulation (over liquid water for $T \ge 0^\circ\text{C}$, over ice for $T < 0^\circ\text{C}$).
#[inline]
pub fn saturation_vapor_pressure_pa(temp_c: f64) -> f64 {
    let t = temp_c.clamp(-90.0, 60.0);
    if t >= 0.0 {
        610.94 * ((17.625 * t) / (t + 243.04)).exp()
    } else {
        611.21 * ((22.587 * t) / (t + 273.86)).exp()
    }
}

/// Computes the exact Clausius-Clapeyron saturated (moist) adiabatic lapse rate $\Gamma_m(T, P)$
/// in `K/km` (positive = cooling with height).
#[inline]
pub fn moist_adiabatic_lapse_rate_k_per_km(temp_c: f64, pressure_hpa: f64) -> f64 {
    const G: f64 = 9.806_65;
    const CP: f64 = 1004.685;
    const RD: f64 = 287.058;
    const EPS: f64 = 0.621_98;

    let t_k = (temp_c + 273.15).max(180.0);
    let p_pa = (pressure_hpa * 100.0).max(5_000.0);
    let es = saturation_vapor_pressure_pa(temp_c).min(0.45 * p_pa);
    let rs = EPS * es / (p_pa - es);
    let latent = if temp_c >= 0.0 { 2.501e6 } else { 2.834e6 };

    let num = 1.0 + (latent * rs) / (RD * t_k);
    let den = CP + (latent * latent * rs * EPS) / (RD * t_k * t_k);
    (G * (num / den) * 1_000.0).clamp(3.0, 9.8)
}

/// Computes wet-bulb temperature $T_w$ (°C) using Stull's (2011) closed-form empirical equation,
/// accurate to within $\pm 0.3^\circ\text{C}$ across relative humidity `5%..=100%`.
#[inline]
pub fn wet_bulb_temp_c(temp_c: f64, relative_humidity: f64) -> f64 {
    let rh = (relative_humidity.clamp(0.05, 1.0)) * 100.0;
    let t = temp_c.clamp(-80.0, 60.0);
    let tw = t * (0.151_977 * (rh + 8.313_659).sqrt()).atan()
        + (t + rh).atan()
        - (rh - 1.676_331).atan()
        + 0.003_918_38 * rh.powf(1.5) * (0.023_101 * rh).atan()
        - 4.686_035;
    // Wet-bulb temperature can never physically exceed dry-bulb air temperature.
    tw.min(t)
}

/// Smoothly partitions precipitation between rain, sleet/mixed, and snow across the physical
/// wet-bulb melting layer ($T_w \in [-1.5^\circ\text{C}, +2.5^\circ\text{C}]$).
#[inline]
pub fn compute_precipitation_phase(wet_bulb_c: f64) -> PrecipitationPhase {
    // Snow rises smoothly from 0 at +2.5°C to 1 at -1.5°C.
    let snow_raw = smoothstep_f64(2.5, -1.5, wet_bulb_c) as f32;
    let rain_raw = (1.0 - snow_raw).clamp(0.0, 1.0);
    // Sleet/mixed ice pellets peak cleanly at the center of the melting band (~+0.5°C Tw).
    let sleet = (4.0 * rain_raw * snow_raw * 0.42).clamp(0.0, 0.42);
    let rem = (1.0 - sleet).max(0.0);
    PrecipitationPhase {
        rain: rain_raw * rem,
        sleet,
        snow: snow_raw * rem,
    }
}

/// Climatological annual-mean sea-level surface temperature approximation by latitude (°C).
///
/// Ranges smoothly from ~`+27°C` at the equator to ~`+12°C` at mid-latitudes (`45°`) and
/// ~`-16°C` at the poles.
#[inline]
pub fn climatological_sea_level_temp_c(lat_deg: f64) -> f64 {
    let cos_lat = lat_deg.clamp(-90.0, 90.0).to_radians().cos();
    let cos2 = cos_lat * cos_lat;
    -16.0 + 43.0 * cos2.powf(0.85)
}

/// Parameters governing the vertical atmospheric temperature column & precipitating cloud ceiling.
#[derive(Debug, Clone, PartialEq)]
pub struct AtmosphereConfig {
    /// Base air temperature in °C at [`Self::anchor`] (`12.0°C` at Sea Level by default).
    pub base_temp_c: f64,
    /// Whether [`Self::base_temp_c`] is defined at Mean Sea Level (`SeaLevel`, physically accurate
    /// default) or local ground level (`TerrainSurface`).
    pub anchor: TemperatureAnchor,
    /// Vertical lapse-rate model (`StandardIsa`, `MoistAdiabatic`, or `Custom(K/km)`).
    pub lapse_rate: LapseRateModel,
    /// Relative humidity (`0.05..=1.0`, default `0.85` during precipitation) used for wet-bulb $T_w$.
    pub relative_humidity: f64,
    /// Cloud deck ceiling above which precipitation fades out and stops completely.
    pub cloud_ceiling: CloudCeiling,
    /// Vertical thickness in metres of the smooth fade-out band approaching [`Self::cloud_ceiling`].
    pub cloud_fade_band_m: f64,
    /// When `true`, offsets `base_temp_c` automatically using latitude climatology (warm tropics,
    /// freezing poles) instead of a globally uniform base temperature.
    pub latitude_climatology: bool,
}

impl Default for AtmosphereConfig {
    fn default() -> Self {
        Self {
            base_temp_c: 12.0,
            anchor: TemperatureAnchor::SeaLevel,
            lapse_rate: LapseRateModel::StandardIsa,
            relative_humidity: 0.85,
            cloud_ceiling: CloudCeiling::default(),
            cloud_fade_band_m: DEFAULT_CLOUD_FADE_BAND_M,
            latitude_climatology: false,
        }
    }
}

impl AtmosphereConfig {
    /// Evaluates the full physical atmospheric column at `(lat_deg, altitude_msl_m, ground_elev_m)`.
    pub fn evaluate(&self, lat_deg: f64, altitude_msl_m: f64, ground_elev_m: f64) -> AtmosphereSample {
        let base_t = if self.latitude_climatology {
            climatological_sea_level_temp_c(lat_deg)
        } else {
            self.base_temp_c
        };
        let ref_alt = match self.anchor {
            TemperatureAnchor::SeaLevel => 0.0,
            TemperatureAnchor::TerrainSurface => ground_elev_m.max(0.0),
        };
        let target_alt = altitude_msl_m;
        let pressure_hpa = isa_pressure_hpa(target_alt);
        let rh = self.relative_humidity.clamp(0.05, 1.0);

        let (temp_c, active_lapse, freezing_level_msl_m) = match self.lapse_rate {
            LapseRateModel::StandardIsa => {
                let z_eff = target_alt.min(ISA_TROPOPAUSE_ALT_M);
                let dz = z_eff - ref_alt.min(ISA_TROPOPAUSE_ALT_M);
                let lapse = ISA_LAPSE_RATE_K_PER_KM;
                let t = base_t - (lapse * 1e-3) * dz;
                let freeze_z = ref_alt + base_t / (lapse * 1e-3);
                (
                    t,
                    if target_alt > ISA_TROPOPAUSE_ALT_M {
                        0.0
                    } else {
                        lapse
                    },
                    freeze_z,
                )
            }
            LapseRateModel::Custom(custom_lapse) => {
                let lapse = custom_lapse.clamp(0.1, 15.0);
                let z_eff = target_alt.min(ISA_TROPOPAUSE_ALT_M);
                let dz = z_eff - ref_alt.min(ISA_TROPOPAUSE_ALT_M);
                let t = base_t - (lapse * 1e-3) * dz;
                let freeze_z = ref_alt + base_t / (lapse * 1e-3);
                (
                    t,
                    if target_alt > ISA_TROPOPAUSE_ALT_M {
                        0.0
                    } else {
                        lapse
                    },
                    freeze_z,
                )
            }
            LapseRateModel::MoistAdiabatic => {
                integrate_moist_adiabatic(base_t, ref_alt, target_alt)
            }
        };

        let wet_bulb = wet_bulb_temp_c(temp_c, rh);
        let phase = compute_precipitation_phase(wet_bulb);
        let cloud_ceiling_msl_m = self.cloud_ceiling.resolve_msl_m(ground_elev_m);

        AtmosphereSample {
            temperature_c: temp_c,
            wet_bulb_c: wet_bulb,
            pressure_hpa,
            relative_humidity: rh,
            lapse_rate_k_per_km: active_lapse,
            freezing_level_msl_m,
            cloud_ceiling_msl_m,
            phase,
            wind_mps: None,
            source_live: false,
        }
    }
}

/// Integrates the Clausius-Clapeyron moist adiabatic lapse rate with 50 m midpoint RK2 steps
/// from `ref_alt` to `target_alt`, simultaneously solving for the exact `0°C` freezing level.
fn integrate_moist_adiabatic(base_temp_c: f64, ref_alt: f64, target_alt: f64) -> (f64, f64, f64) {
    let clamped_target = target_alt.min(ISA_TROPOPAUSE_ALT_M);
    let clamped_ref = ref_alt.min(ISA_TROPOPAUSE_ALT_M);
    let integrate_to = |z_end: f64| -> (f64, f64) {
        let total_dz = z_end - clamped_ref;
        if total_dz.abs() < 1e-6 {
            let l0 = moist_adiabatic_lapse_rate_k_per_km(base_temp_c, isa_pressure_hpa(clamped_ref));
            return (base_temp_c, l0);
        }
        let steps = ((total_dz.abs() / 50.0).ceil() as usize).clamp(1, 300);
        let step_dz = total_dz / (steps as f64);
        let mut z = clamped_ref;
        let mut t = base_temp_c;
        for _ in 0..steps {
            let g1 = moist_adiabatic_lapse_rate_k_per_km(t, isa_pressure_hpa(z)) * 1e-3;
            let t_mid = t - 0.5 * step_dz * g1;
            let z_mid = z + 0.5 * step_dz;
            let g2 = moist_adiabatic_lapse_rate_k_per_km(t_mid, isa_pressure_hpa(z_mid)) * 1e-3;
            t -= step_dz * g2;
            z += step_dz;
        }
        let end_lapse = moist_adiabatic_lapse_rate_k_per_km(t, isa_pressure_hpa(z));
        (t, end_lapse)
    };

    let (t_out, end_lapse) = integrate_to(clamped_target);
    let active_lapse = if target_alt > ISA_TROPOPAUSE_ALT_M {
        0.0
    } else {
        end_lapse
    };

    // Solve freezing level (0°C isotherm) by walking from ref_alt in 100 m steps.
    let freeze_z = if base_temp_c <= 0.0 {
        let l0 = moist_adiabatic_lapse_rate_k_per_km(base_temp_c, isa_pressure_hpa(clamped_ref));
        clamped_ref + base_temp_c / (l0 * 1e-3)
    } else {
        let mut z = clamped_ref;
        let mut t = base_temp_c;
        let mut found = ISA_TROPOPAUSE_ALT_M;
        while z < ISA_TROPOPAUSE_ALT_M {
            let step = 100.0;
            let g = moist_adiabatic_lapse_rate_k_per_km(t, isa_pressure_hpa(z + 50.0)) * 1e-3;
            let t_next = t - step * g;
            if t_next <= 0.0 {
                let frac = (t / (t - t_next).max(1e-9)).clamp(0.0, 1.0);
                found = z + step * frac;
                break;
            }
            t = t_next;
            z += step;
        }
        found
    };

    (t_out, active_lapse, freeze_z)
}

// ---------------------------------------------------------------------------
// TemperatureProvider Trait + Static, Closure, and Live Open-Meteo Providers
// ---------------------------------------------------------------------------

/// Input coordinates passed to a [`TemperatureProvider`] for atmospheric column evaluation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AtmosphereQuery {
    pub lat_deg: f64,
    pub lon_deg: f64,
    pub altitude_msl_m: f64,
    pub ground_elevation_m: f64,
}

/// Resolved physical atmospheric state at a queried geographic position and altitude.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AtmosphereSample {
    /// Dry-bulb air temperature at the queried altitude (°C).
    pub temperature_c: f64,
    /// Wet-bulb temperature (°C) — governs physical hydrometeor phase (rain vs sleet vs snow).
    pub wet_bulb_c: f64,
    /// Atmospheric barometric pressure (hPa / mbar).
    pub pressure_hpa: f64,
    /// Relative humidity (`0.05..=1.0`).
    pub relative_humidity: f64,
    /// Active local vertical lapse rate in `K/km` (positive = cooling with height).
    pub lapse_rate_k_per_km: f64,
    /// Computed `0°C` freezing level altitude in metres MSL.
    pub freezing_level_msl_m: f64,
    /// Active precipitating cloud ceiling altitude in metres MSL.
    pub cloud_ceiling_msl_m: f64,
    /// Smooth hydrometeor phase partition (`rain + sleet + snow == 1.0`).
    pub phase: PrecipitationPhase,
    /// Optional real-time wind override `[east_mps, north_mps]` from a live weather source.
    pub wind_mps: Option<[f32; 2]>,
    /// `true` when backed by a fetched live sounding rather than static analytical fallback.
    pub source_live: bool,
}

/// Pluggable atmospheric temperature & sounding provider for [`PrecipitationPlugin`].
pub trait TemperatureProvider: Send + Sync + 'static {
    /// Evaluates the atmospheric state at the queried `(lat, lon, altitude_msl_m, ground_elev_m)`.
    fn sample(&self, query: AtmosphereQuery, fallback: &AtmosphereConfig) -> AtmosphereSample;
}

/// Built-in zero-network physical atmosphere provider using [`AtmosphereConfig`].
#[derive(Debug, Default, Clone, Copy)]
pub struct StaticAtmosphereProvider;

impl TemperatureProvider for StaticAtmosphereProvider {
    fn sample(&self, query: AtmosphereQuery, fallback: &AtmosphereConfig) -> AtmosphereSample {
        fallback.evaluate(query.lat_deg, query.altitude_msl_m, query.ground_elevation_m)
    }
}

/// Adapter turning any closure into a [`TemperatureProvider`].
pub struct FnTemperatureProvider<F>(pub F);

impl<F> TemperatureProvider for FnTemperatureProvider<F>
where
    F: Fn(AtmosphereQuery, &AtmosphereConfig) -> AtmosphereSample + Send + Sync + 'static,
{
    fn sample(&self, query: AtmosphereQuery, fallback: &AtmosphereConfig) -> AtmosphereSample {
        (self.0)(query, fallback)
    }
}

/// Parsed vertical atmospheric sounding fetched from Open-Meteo pressure levels.
#[derive(Debug, Clone, PartialEq)]
pub struct OpenMeteoSounding {
    /// Sorted ascending `(altitude_msl_m, temperature_c)` profile points.
    pub levels: Vec<(f64, f64)>,
    /// Surface relative humidity (`0.05..=1.0`).
    pub relative_humidity: f64,
    /// Reported `0°C` freezing level height in metres MSL (if present).
    pub freezing_level_msl_m: Option<f64>,
    /// Optional surface wind vector `[east_mps, north_mps]`.
    pub wind_mps: Option<[f32; 2]>,
}

impl OpenMeteoSounding {
    /// Piecewise-linearly interpolates temperature (°C) and local lapse rate (`K/km`) at `altitude_msl_m`,
    /// extrapolating cleanly with the standard ISA lapse rate below the lowest or above the highest level.
    pub fn sample_temperature_and_lapse(&self, altitude_msl_m: f64) -> Option<(f64, f64)> {
        if self.levels.is_empty() {
            return None;
        }
        if self.levels.len() == 1 {
            let (z0, t0) = self.levels[0];
            let z_eff = altitude_msl_m.min(ISA_TROPOPAUSE_ALT_M);
            let t = t0 - (ISA_LAPSE_RATE_K_PER_KM * 1e-3) * (z_eff - z0);
            return Some((t, ISA_LAPSE_RATE_K_PER_KM));
        }
        let first = self.levels[0];
        if altitude_msl_m <= first.0 {
            let dz = first.0 - altitude_msl_m;
            return Some((
                first.1 + (ISA_LAPSE_RATE_K_PER_KM * 1e-3) * dz,
                ISA_LAPSE_RATE_K_PER_KM,
            ));
        }
        let last = *self.levels.last().unwrap();
        if altitude_msl_m >= last.0 {
            let z_eff = altitude_msl_m.min(ISA_TROPOPAUSE_ALT_M);
            let dz = (z_eff - last.0).max(0.0);
            let lapse = if altitude_msl_m > ISA_TROPOPAUSE_ALT_M {
                0.0
            } else {
                ISA_LAPSE_RATE_K_PER_KM
            };
            return Some((last.1 - (ISA_LAPSE_RATE_K_PER_KM * 1e-3) * dz, lapse));
        }
        for pair in self.levels.windows(2) {
            let (z0, t0) = pair[0];
            let (z1, t1) = pair[1];
            if altitude_msl_m >= z0 && altitude_msl_m <= z1 && (z1 - z0).abs() > 1e-6 {
                let frac = (altitude_msl_m - z0) / (z1 - z0);
                let temp = t0 + (t1 - t0) * frac;
                let lapse = -((t1 - t0) / (z1 - z0)) * 1_000.0;
                return Some((temp, lapse));
            }
        }
        Some((last.1, ISA_LAPSE_RATE_K_PER_KM))
    }

    /// Evaluates a full [`AtmosphereSample`] from this vertical sounding.
    pub fn to_sample(&self, query: AtmosphereQuery, cfg: &AtmosphereConfig) -> AtmosphereSample {
        let (temp_c, lapse) = self
            .sample_temperature_and_lapse(query.altitude_msl_m)
            .unwrap_or_else(|| {
                let s = cfg.evaluate(query.lat_deg, query.altitude_msl_m, query.ground_elevation_m);
                (s.temperature_c, s.lapse_rate_k_per_km)
            });
        let rh = self.relative_humidity.clamp(0.05, 1.0);
        let wet_bulb = wet_bulb_temp_c(temp_c, rh);
        let phase = compute_precipitation_phase(wet_bulb);
        let freezing_level_msl_m = self.freezing_level_msl_m.unwrap_or_else(|| {
            // Find first 0°C crossing in sounding profile
            for pair in self.levels.windows(2) {
                let (z0, t0) = pair[0];
                let (z1, t1) = pair[1];
                if (t0 >= 0.0 && t1 <= 0.0) || (t0 <= 0.0 && t1 >= 0.0) {
                    let dt = t0 - t1;
                    if dt.abs() > 1e-6 {
                        return z0 + (t0 / dt).clamp(0.0, 1.0) * (z1 - z0);
                    }
                }
            }
            query.altitude_msl_m + temp_c / (ISA_LAPSE_RATE_K_PER_KM * 1e-3)
        });
        AtmosphereSample {
            temperature_c: temp_c,
            wet_bulb_c: wet_bulb,
            pressure_hpa: isa_pressure_hpa(query.altitude_msl_m),
            relative_humidity: rh,
            lapse_rate_k_per_km: lapse,
            freezing_level_msl_m,
            cloud_ceiling_msl_m: cfg.cloud_ceiling.resolve_msl_m(query.ground_elevation_m),
            phase,
            wind_mps: self.wind_mps,
            source_live: true,
        }
    }
}

fn json_obj_field<'a>(obj: &'a [(String, Json)], key: &str) -> Option<&'a Json> {
    obj.iter().find(|(k, _)| k == key).map(|(_, v)| v)
}

fn json_num(val: Option<&Json>) -> Option<f64> {
    match val? {
        Json::Number(n) if n.is_finite() => Some(*n),
        _ => None,
    }
}

fn json_first_array_num(obj: &[(String, Json)], key: &str) -> Option<f64> {
    let Json::Array(arr) = json_obj_field(obj, key)? else {
        return None;
    };
    arr.first().and_then(|v| json_num(Some(v)))
}

/// Parses an Open-Meteo `/v1/forecast` JSON response containing surface `current` weather and
/// `hourly` pressure-level vertical temperature/geopotential height soundings.
pub fn parse_open_meteo_sounding_json(json_text: &str) -> Result<OpenMeteoSounding, String> {
    let Json::Object(root) =
        Json::parse(json_text).map_err(|e| format!("open-meteo json parse error: {e}"))?
    else {
        return Err("open-meteo response root is not a JSON object".to_string());
    };

    let mut levels: Vec<(f64, f64)> = Vec::new();
    let elev_m = json_num(json_obj_field(&root, "elevation")).unwrap_or(0.0);
    let mut relative_humidity = 0.85f64;
    let mut wind_mps: Option<[f32; 2]> = None;

    if let Some(Json::Object(current)) = json_obj_field(&root, "current") {
        if let Some(t2m) = json_num(json_obj_field(current, "temperature_2m")) {
            levels.push((elev_m + 2.0, t2m));
        }
        if let Some(rh_pct) = json_num(json_obj_field(current, "relative_humidity_2m")) {
            relative_humidity = (rh_pct / 100.0).clamp(0.05, 1.0);
        }
        if let (Some(speed_kmh), Some(dir_from_deg)) = (
            json_num(json_obj_field(current, "wind_speed_10m")),
            json_num(json_obj_field(current, "wind_direction_10m")),
        ) {
            // Meteorological direction is where wind blows FROM (0°=N, 90°=E).
            // Velocity vector points TO `(dir_from + 180°)`.
            let speed_mps = (speed_kmh / 3.6).clamp(0.0, 60.0);
            let to_rad = (dir_from_deg + 180.0).to_radians();
            wind_mps = Some([
                (speed_mps * to_rad.sin()) as f32,
                (speed_mps * to_rad.cos()) as f32,
            ]);
        }
    }

    let mut freezing_level_msl_m: Option<f64> = None;
    if let Some(Json::Object(hourly)) = json_obj_field(&root, "hourly") {
        freezing_level_msl_m = json_first_array_num(hourly, "freezing_level_height");
        const PRESSURE_LEVELS: &[(&str, &str)] = &[
            ("temperature_1000hPa", "geopotential_height_1000hPa"),
            ("temperature_925hPa", "geopotential_height_925hPa"),
            ("temperature_850hPa", "geopotential_height_850hPa"),
            ("temperature_700hPa", "geopotential_height_700hPa"),
            ("temperature_500hPa", "geopotential_height_500hPa"),
        ];
        for &(t_key, z_key) in PRESSURE_LEVELS {
            if let (Some(t_c), Some(z_m)) = (
                json_first_array_num(hourly, t_key),
                json_first_array_num(hourly, z_key),
            ) {
                levels.push((z_m, t_c));
            }
        }
    }

    if levels.is_empty() {
        return Err("open-meteo response contained no temperature levels".to_string());
    }
    levels.sort_by(|a, b| a.0.total_cmp(&b.0));
    // Deduplicate levels within 15 m of each other (e.g., surface near 1000 hPa).
    let mut deduped: Vec<(f64, f64)> = Vec::with_capacity(levels.len());
    for (z, t) in levels {
        if let Some(last) = deduped.last_mut() {
            if (z - last.0).abs() < 15.0 {
                last.1 = 0.5 * (last.1 + t);
                continue;
            }
        }
        deduped.push((z, t));
    }

    Ok(OpenMeteoSounding {
        levels: deduped,
        relative_humidity,
        freezing_level_msl_m,
        wind_mps,
    })
}

struct CachedSoundingEntry {
    fetched_at: Instant,
    sounding: OpenMeteoSounding,
}

struct OpenMeteoCache {
    entries: HashMap<(i32, i32), CachedSoundingEntry>,
    in_flight: HashSet<(i32, i32)>,
}

/// Optional live real-world atmospheric sounding provider backed by Open-Meteo (`api.open-meteo.com`).
///
/// Quantizes geographic lookups to `0.25°` cells with a 10-minute TTL cache and non-blocking
/// background fetches (falling back cleanly to [`AtmosphereConfig`] while loading or whenever
/// `STRATA_EARTH_OFFLINE` is set).
#[derive(Clone)]
pub struct OpenMeteoTemperatureProvider {
    endpoint: String,
    ttl: Duration,
    cache: Arc<Mutex<OpenMeteoCache>>,
}

impl Default for OpenMeteoTemperatureProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl OpenMeteoTemperatureProvider {
    pub const DEFAULT_ENDPOINT: &'static str = "https://api.open-meteo.com/v1/forecast";

    pub fn new() -> Self {
        Self {
            endpoint: Self::DEFAULT_ENDPOINT.to_string(),
            ttl: Duration::from_secs(600),
            cache: Arc::new(Mutex::new(OpenMeteoCache {
                entries: HashMap::new(),
                in_flight: HashSet::new(),
            })),
        }
    }

    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = endpoint.into();
        self
    }

    fn quantize_cell(lat_deg: f64, lon_deg: f64) -> (i32, i32) {
        (
            (lat_deg.clamp(-89.75, 89.75) * 4.0).round() as i32,
            (((lon_deg + 180.0).rem_euclid(360.0) - 180.0) * 4.0).round() as i32,
        )
    }

    /// Inserts a pre-parsed sounding into the cell cache (useful for deterministic tests or preload).
    pub fn insert_cached_sounding(&self, lat_deg: f64, lon_deg: f64, sounding: OpenMeteoSounding) {
        let key = Self::quantize_cell(lat_deg, lon_deg);
        if let Ok(mut guard) = self.cache.lock() {
            guard.entries.insert(
                key,
                CachedSoundingEntry {
                    fetched_at: Instant::now(),
                    sounding,
                },
            );
            guard.in_flight.remove(&key);
        }
    }

    fn ensure_cell_fetched(&self, key: (i32, i32)) {
        if std::env::var_os("STRATA_EARTH_OFFLINE").is_some() {
            return;
        }
        let should_spawn = if let Ok(mut guard) = self.cache.lock() {
            if let Some(entry) = guard.entries.get(&key) {
                if entry.fetched_at.elapsed() < self.ttl {
                    return;
                }
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
        let url = format!(
            "{}?latitude={:.2}&longitude={:.2}&current=temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m&hourly=temperature_1000hPa,temperature_925hPa,temperature_850hPa,temperature_700hPa,temperature_500hPa,geopotential_height_1000hPa,geopotential_height_925hPa,geopotential_height_850hPa,geopotential_height_700hPa,geopotential_height_500hPa,freezing_level_height&forecast_hours=1",
            self.endpoint, lat, lon
        );
        let cache = Arc::clone(&self.cache);
        let _ = std::thread::Builder::new()
            .name("strata-earth-meteo".to_string())
            .spawn(move || {
                let agent = ureq::AgentBuilder::new()
                    .timeout_connect(Duration::from_secs(5))
                    .timeout_read(Duration::from_secs(8))
                    .build();
                let parsed = agent
                    .get(&url)
                    .set("User-Agent", "strata-earth/0.1 (atmospheric-sounding)")
                    .call()
                    .ok()
                    .and_then(|resp| resp.into_string().ok())
                    .and_then(|body| parse_open_meteo_sounding_json(&body).ok());
                if let Ok(mut guard) = cache.lock() {
                    guard.in_flight.remove(&key);
                    if let Some(sounding) = parsed {
                        guard.entries.insert(
                            key,
                            CachedSoundingEntry {
                                fetched_at: Instant::now(),
                                sounding,
                            },
                        );
                    }
                }
            });
    }
}

impl TemperatureProvider for OpenMeteoTemperatureProvider {
    fn sample(&self, query: AtmosphereQuery, fallback: &AtmosphereConfig) -> AtmosphereSample {
        let key = Self::quantize_cell(query.lat_deg, query.lon_deg);
        let cached = self
            .cache
            .lock()
            .ok()
            .and_then(|g| g.entries.get(&key).map(|e| e.sounding.clone()));
        self.ensure_cell_fetched(key);
        if let Some(sounding) = cached {
            sounding.to_sample(query, fallback)
        } else {
            fallback.evaluate(query.lat_deg, query.altitude_msl_m, query.ground_elevation_m)
        }
    }
}

// ---------------------------------------------------------------------------
// Generalized 2D Particle Layer Config & Custom Evaluator API
// ---------------------------------------------------------------------------

/// 2D visual model for a [`PrecipitationLayer`].
#[derive(Debug, Clone, PartialEq)]
pub enum ParticleSprite {
    /// Pure mathematical 2D motion-tapered raindrop streak (zero asset files).
    RainStreak,
    /// Pure mathematical simplistic 6-fold geometric 2D snowflake crystal (zero asset files).
    Snowflake,
    /// Pure mathematical translucent sleet / ice pellet (zero asset files).
    SleetPellet,
    /// Arbitrary user-supplied 2D RGBA8 texture (cached on GPU by `(id, revision)`).
    Texture {
        id: u64,
        revision: u64,
        width: u32,
        height: u32,
        rgba8: Arc<Vec<u8>>,
    },
}

/// 2D billboard orientation mode for a [`PrecipitationLayer`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ParticleBillboard {
    /// Aligns quad vertical axis with 3D apparent velocity vector (`wind - gravity`),
    /// stretching cleanly with drop speed and foreshortening naturally when looking downward.
    VelocityAligned,
    /// Camera-facing 2D sprite aligned with camera right/up vectors with per-particle 2D spin.
    CameraFacing,
}

/// Generalized configuration for one GPU-instanced camera-local 2D precipitation particle layer.
///
/// Works identically for fast 2D rain streaks, mathematical 2D snowflakes, sleet pellets,
/// ash, petals, dust, or any custom 2D RGBA8 texture.
#[derive(Debug, Clone, PartialEq)]
pub struct PrecipitationLayer {
    /// Stable unique layer identifier (drives GPU hash seed and toroidal phase tracking).
    pub id: u64,
    /// 2D visual sprite (mathematical SDF or custom 2D RGBA8 texture).
    pub sprite: ParticleSprite,
    /// 2D billboard alignment mode.
    pub billboard: ParticleBillboard,
    /// Normalized intensity/amount scalar (`0.0..=2.0`); scales active particle count & opacity.
    pub intensity: f32,
    /// Maximum GPU-instanced particles in the camera-local wrap box at `intensity = 1.0`.
    pub max_particles: u32,
    /// Downward vertical fall speed in metres per second.
    pub fall_speed_mps: f32,
    /// Horizontal wind velocity `[east_mps, north_mps]` in metres per second.
    pub wind_mps: [f32; 2],
    /// Base 2D particle quad dimensions `[width_m, height_m]` in metres.
    pub particle_size_m: [f32; 2],
    /// Motion elongation multiplier along 3D velocity vector (for `VelocityAligned`).
    pub velocity_stretch: f32,
    /// Horizontal sinusoidal flutter `[amplitude_m, frequency_hz]`.
    pub flutter: [f32; 2],
    /// Per-particle 2D angular spin speed in rad/s (for `CameraFacing`).
    pub spin_rad_s: f32,
    /// Base camera-local toroidal wrap box dimensions `[extent_xy_m, extent_z_m]`.
    pub volume_size_m: [f32; 2],
    /// Screen-space pixel size clamps `[min_px, max_px]` preserving crispness across altitudes.
    pub screen_px_clamp: [f32; 2],
    /// Multiplicative RGBA tint (`alpha` scales layer opacity).
    pub color: [f32; 4],
    /// Deterministic hash seed offset.
    pub seed: u32,
}

impl PrecipitationLayer {
    /// Creates a fast, crisp, mathematical 2D rain streak layer (`VelocityAligned`).
    pub fn rain(intensity: f32) -> Self {
        Self {
            id: 0x5052_0001,
            sprite: ParticleSprite::RainStreak,
            billboard: ParticleBillboard::VelocityAligned,
            intensity: intensity.clamp(0.0, 2.5),
            max_particles: 7_500,
            fall_speed_mps: 28.0,
            wind_mps: [3.5, 1.0],
            particle_size_m: [0.052, 1.05],
            velocity_stretch: 0.038,
            flutter: [0.0, 0.0],
            spin_rad_s: 0.0,
            volume_size_m: [42.0, 32.0],
            screen_px_clamp: [1.8, 62.0],
            color: [0.84, 0.92, 1.0, 0.86],
            seed: 101,
        }
    }

    /// Creates a simplistic, pure mathematical 6-fold 2D snowflake crystal layer (`CameraFacing`).
    pub fn snow(intensity: f32) -> Self {
        Self {
            id: 0x5052_0002,
            sprite: ParticleSprite::Snowflake,
            billboard: ParticleBillboard::CameraFacing,
            intensity: intensity.clamp(0.0, 2.5),
            max_particles: 5_200,
            fall_speed_mps: 2.2,
            wind_mps: [2.2, 0.8],
            particle_size_m: [0.18, 0.18],
            velocity_stretch: 0.0,
            flutter: [0.70, 1.35],
            spin_rad_s: 1.45,
            volume_size_m: [40.0, 28.0],
            screen_px_clamp: [3.8, 16.0],
            color: [0.98, 0.99, 1.0, 0.95],
            seed: 211,
        }
    }

    /// Creates a compact translucent sleet / ice-pellet layer for the freezing-level melting zone.
    pub fn sleet(intensity: f32) -> Self {
        Self {
            id: 0x5052_0003,
            sprite: ParticleSprite::SleetPellet,
            billboard: ParticleBillboard::CameraFacing,
            intensity: intensity.clamp(0.0, 2.5),
            max_particles: 3_600,
            fall_speed_mps: 11.5,
            wind_mps: [3.0, 1.0],
            particle_size_m: [0.10, 0.14],
            velocity_stretch: 0.0,
            flutter: [0.18, 2.1],
            spin_rad_s: 0.6,
            volume_size_m: [40.0, 30.0],
            screen_px_clamp: [2.5, 11.0],
            color: [0.90, 0.96, 1.0, 0.88],
            seed: 307,
        }
    }

    /// Creates a generalized custom 2D textured precipitation particle layer.
    pub fn custom_texture(
        id: u64,
        width: u32,
        height: u32,
        rgba8: Arc<Vec<u8>>,
        intensity: f32,
    ) -> Self {
        Self {
            id,
            sprite: ParticleSprite::Texture {
                id,
                revision: 1,
                width,
                height,
                rgba8,
            },
            billboard: ParticleBillboard::CameraFacing,
            intensity: intensity.clamp(0.0, 2.5),
            max_particles: 1_800,
            fall_speed_mps: 6.0,
            wind_mps: [2.0, 0.5],
            particle_size_m: [0.22, 0.22],
            velocity_stretch: 0.0,
            flutter: [0.35, 1.1],
            spin_rad_s: 0.8,
            volume_size_m: [42.0, 30.0],
            screen_px_clamp: [4.0, 24.0],
            color: [1.0, 1.0, 1.0, 0.90],
            seed: (id as u32).wrapping_mul(31).wrapping_add(17),
        }
    }

    pub fn with_speed(mut self, fall_speed_mps: f32) -> Self {
        self.fall_speed_mps = fall_speed_mps.max(0.0);
        self
    }

    pub fn with_intensity(mut self, intensity: f32) -> Self {
        self.intensity = intensity.clamp(0.0, 2.5);
        self
    }

    pub fn with_wind(mut self, east_mps: f32, north_mps: f32) -> Self {
        self.wind_mps = [east_mps, north_mps];
        self
    }

    pub fn with_color(mut self, rgba: [f32; 4]) -> Self {
        self.color = rgba;
        self
    }

    pub fn with_size(mut self, width_m: f32, height_m: f32) -> Self {
        self.particle_size_m = [width_m.max(0.005), height_m.max(0.005)];
        self
    }

    pub fn with_flutter(mut self, amplitude_m: f32, freq_hz: f32) -> Self {
        self.flutter = [amplitude_m.max(0.0), freq_hz.max(0.0)];
        self
    }

    pub fn with_max_particles(mut self, max_particles: u32) -> Self {
        self.max_particles = max_particles.min(16_384);
        self
    }

    pub fn with_billboard(mut self, billboard: ParticleBillboard) -> Self {
        self.billboard = billboard;
        self
    }
}

/// High-level operating mode for [`PrecipitationConfig`].
#[derive(Debug, Clone, PartialEq, Default)]
pub enum PrecipitationMode {
    /// Automatically determines Rain / Sleet / Snow blend from physical atmospheric temperature
    /// and wet-bulb lapse rate at the camera's exact altitude, stopping above the cloud ceiling.
    #[default]
    Physical,
    /// Forces fast 2D rain streaks (still respecting altitude adaptation & cloud ceiling).
    Rain,
    /// Forces simplistic mathematical 2D snowflakes (still respecting altitude adaptation & cloud ceiling).
    Snow,
    /// Emits explicit user-configured custom 2D particle layers (any sprites, textures, speeds).
    CustomLayers(Vec<PrecipitationLayer>),
}

/// Declarative precipitation configuration stored on [`crate::scene::EarthInput`].
#[derive(Debug, Clone, PartialEq)]
pub struct PrecipitationConfig {
    /// Master enable toggle (`false` by default; zero overhead when off).
    pub enabled: bool,
    /// Precipitation selection mode (`Physical`, `Rain`, `Snow`, or `CustomLayers`).
    pub mode: PrecipitationMode,
    /// Overall precipitation intensity / amount multiplier (`0.0..=2.0`, default `0.85`).
    pub intensity: f32,
    /// Default raindrop fall speed in m/s (`26.0 m/s` default for crisp, fast rain streaks).
    pub rain_speed_mps: f32,
    /// Default snowflake fall speed in m/s (`2.2 m/s` default for gentle drifting flakes).
    pub snow_speed_mps: f32,
    /// Ambient horizontal wind vector `[east_mps, north_mps]` (overridden by live provider if available).
    pub wind_mps: [f32; 2],
    /// Atmospheric column model (base temperature, MSL/Terrain anchor, lapse rate, humidity, cloud ceiling).
    pub atmosphere: AtmosphereConfig,
    /// When `true` (default), precipitation fades smoothly into and stops completely above the cloud ceiling.
    pub respect_cloud_ceiling: bool,
    /// When `true`, enables the built-in non-blocking [`OpenMeteoTemperatureProvider`] to fetch real
    /// vertical pressure-level temperatures at the camera's geographic location & altitude.
    pub use_live_temperature: bool,
}

impl Default for PrecipitationConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            mode: PrecipitationMode::Physical,
            intensity: 0.85,
            rain_speed_mps: 28.0,
            snow_speed_mps: 2.2,
            wind_mps: [3.5, 1.0],
            atmosphere: AtmosphereConfig::default(),
            respect_cloud_ceiling: true,
            use_live_temperature: false,
        }
    }
}

impl PrecipitationConfig {
    pub fn enabled(mut self) -> Self {
        self.enabled = true;
        self
    }

    pub fn with_mode(mut self, mode: PrecipitationMode) -> Self {
        self.mode = mode;
        self
    }

    pub fn with_intensity(mut self, intensity: f32) -> Self {
        self.intensity = intensity.clamp(0.0, 2.5);
        self
    }

    pub fn with_base_temp_c(mut self, temp_c: f64) -> Self {
        self.atmosphere.base_temp_c = temp_c;
        self
    }

    pub fn with_cloud_ceiling(mut self, ceiling: CloudCeiling) -> Self {
        self.atmosphere.cloud_ceiling = ceiling;
        self
    }

    pub fn with_live_temperature(mut self, live: bool) -> Self {
        self.use_live_temperature = live;
        self
    }
}

/// Complete physical & camera context supplied to a [`PrecipitationEvaluator`] every frame.
#[derive(Debug, Clone)]
pub struct PrecipitationEvalContext<'a> {
    pub config: &'a PrecipitationConfig,
    pub camera_lat_deg: f64,
    pub camera_lon_deg: f64,
    pub camera_altitude_msl_m: f64,
    pub ground_elevation_m: f64,
    pub camera_agl_m: f64,
    pub atmosphere: AtmosphereSample,
    /// Smooth cloud-ceiling envelope (`1.0` well below cloud deck, fading to `0.0` at cloud ceiling).
    pub cloud_envelope: f32,
    pub intensity: f32,
    pub time_s: f64,
    pub wind_mps: [f32; 2],
    pub view_mode: EarthViewMode,
    pub map_2d_morph: f32,
}

/// Functional evaluator transforming physical camera & atmosphere state into 0..N 2D particle layers.
pub type PrecipitationEvaluator =
    Arc<dyn Fn(&PrecipitationEvalContext<'_>) -> Vec<PrecipitationLayer> + Send + Sync>;

/// Default physically accurate precipitation evaluator.
///
/// - Stops cleanly above the precipitating cloud ceiling (`cloud_envelope <= 0.002`).
/// - Adapts camera-local wrap volume smoothly to camera height Above Ground Level (`camera_agl_m`).
/// - In [`PrecipitationMode::Physical`], smoothly transitions between fast 2D rain streaks,
///   sleet/mixed pellets, and simplistic mathematical 2D snowflakes across the wet-bulb melting layer.
pub fn physical_precipitation_evaluator(
    ctx: &PrecipitationEvalContext<'_>,
) -> Vec<PrecipitationLayer> {
    if ctx.intensity <= 0.002 {
        return Vec::new();
    }
    let env = if ctx.config.respect_cloud_ceiling {
        ctx.cloud_envelope
    } else {
        1.0
    };
    if env <= 0.002 {
        return Vec::new();
    }

    // Scale the camera-local volume gently with height AGL so particles remain crisp & dense
    // near street level (45 m) while filling the wider vista smoothly up toward the cloud base.
    let agl_scale = 1.0 + 0.32 * smoothstep_f32(300.0, 4_200.0, ctx.camera_agl_m as f32);
    let eff_intensity = (ctx.intensity * env).clamp(0.0, 2.5);
    let wind = ctx.wind_mps;

    let mut out = Vec::with_capacity(3);
    match &ctx.config.mode {
        PrecipitationMode::Physical => {
            let phase = ctx.atmosphere.phase;
            if phase.rain > 0.015 {
                let mut rain = PrecipitationLayer::rain(eff_intensity * phase.rain)
                    .with_speed(ctx.config.rain_speed_mps)
                    .with_wind(wind[0], wind[1]);
                rain.volume_size_m[0] *= agl_scale;
                rain.volume_size_m[1] *= agl_scale;
                rain.max_particles = ((rain.max_particles as f32) * agl_scale).min(12_000.0) as u32;
                rain.color[3] *= (0.45 + 0.55 * env).clamp(0.0, 1.0) * phase.rain.sqrt();
                out.push(rain);
            }
            if phase.sleet > 0.04 {
                let mut sleet = PrecipitationLayer::sleet(eff_intensity * phase.sleet)
                    .with_wind(wind[0] * 0.9, wind[1] * 0.9);
                sleet.volume_size_m[0] *= agl_scale;
                sleet.volume_size_m[1] *= agl_scale;
                sleet.max_particles = ((sleet.max_particles as f32) * agl_scale).min(8_000.0) as u32;
                sleet.color[3] *= (0.45 + 0.55 * env).clamp(0.0, 1.0) * phase.sleet.sqrt();
                out.push(sleet);
            }
            if phase.snow > 0.015 {
                let mut snow = PrecipitationLayer::snow(eff_intensity * phase.snow)
                    .with_speed(ctx.config.snow_speed_mps)
                    .with_wind(wind[0] * 0.75, wind[1] * 0.75);
                snow.volume_size_m[0] *= agl_scale;
                snow.volume_size_m[1] *= agl_scale;
                snow.max_particles = ((snow.max_particles as f32) * agl_scale).min(10_000.0) as u32;
                snow.color[3] *= (0.45 + 0.55 * env).clamp(0.0, 1.0) * phase.snow.sqrt();
                out.push(snow);
            }
        }
        PrecipitationMode::Rain => {
            let mut rain = PrecipitationLayer::rain(eff_intensity)
                .with_speed(ctx.config.rain_speed_mps)
                .with_wind(wind[0], wind[1]);
            rain.volume_size_m[0] *= agl_scale;
            rain.volume_size_m[1] *= agl_scale;
            rain.max_particles = ((rain.max_particles as f32) * agl_scale).min(12_000.0) as u32;
            rain.color[3] *= env;
            out.push(rain);
        }
        PrecipitationMode::Snow => {
            let mut snow = PrecipitationLayer::snow(eff_intensity)
                .with_speed(ctx.config.snow_speed_mps)
                .with_wind(wind[0] * 0.75, wind[1] * 0.75);
            snow.volume_size_m[0] *= agl_scale;
            snow.volume_size_m[1] *= agl_scale;
            snow.max_particles = ((snow.max_particles as f32) * agl_scale).min(10_000.0) as u32;
            snow.color[3] *= env;
            out.push(snow);
        }
        PrecipitationMode::CustomLayers(layers) => {
            for layer in layers {
                let mut l = layer.clone();
                l.intensity *= eff_intensity;
                l.color[3] *= env;
                if l.intensity > 0.002 && l.color[3] > 0.002 {
                    out.push(l);
                }
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// PrecipitationPlugin (First-Class EarthPlugin)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, Default)]
struct LayerToroidalPhase {
    phase_e: f64,
    phase_n: f64,
    phase_z: f64,
}

/// First-class `strata_earth` plugin rendering physically accurate, camera-local 2D precipitation
/// (rain, sleet, mathematical 2D snowflakes, and custom 2D textured layers) on the GPU.
pub struct PrecipitationPlugin {
    pub enabled: bool,
    config_override: Option<PrecipitationConfig>,
    evaluator: PrecipitationEvaluator,
    temperature_provider: Option<Arc<dyn TemperatureProvider>>,
    live_meteo: OpenMeteoTemperatureProvider,
    sim_time_s: f64,
    pending_dt_s: f64,
    last_eye_world: Option<DVec3>,
    layer_phases: HashMap<u64, LayerToroidalPhase>,
    last_sample: Arc<RwLock<Option<AtmosphereSample>>>,
}

impl Default for PrecipitationPlugin {
    fn default() -> Self {
        Self::new()
    }
}

impl PrecipitationPlugin {
    pub const ID: &'static str = "precipitation";

    /// Creates a new `PrecipitationPlugin` using [`physical_precipitation_evaluator`] and
    /// standard ISA sea-level atmospheric thermodynamics.
    pub fn new() -> Self {
        Self {
            enabled: true,
            config_override: None,
            evaluator: Arc::new(physical_precipitation_evaluator),
            temperature_provider: None,
            live_meteo: OpenMeteoTemperatureProvider::new(),
            sim_time_s: 0.0,
            pending_dt_s: 0.0,
            last_eye_world: None,
            layer_phases: HashMap::new(),
            last_sample: Arc::new(RwLock::new(None)),
        }
    }

    /// Attaches a custom [`PrecipitationConfig`] override directly on this plugin instance.
    pub fn with_config(mut self, config: PrecipitationConfig) -> Self {
        self.config_override = Some(config);
        self
    }

    /// Installs a custom functional [`PrecipitationEvaluator`] controlling which 2D layers render.
    pub fn with_evaluator<F>(mut self, evaluator: F) -> Self
    where
        F: Fn(&PrecipitationEvalContext<'_>) -> Vec<PrecipitationLayer> + Send + Sync + 'static,
    {
        self.evaluator = Arc::new(evaluator);
        self
    }

    /// Replaces the active evaluator at runtime.
    pub fn set_evaluator(&mut self, evaluator: PrecipitationEvaluator) {
        self.evaluator = evaluator;
    }

    /// Installs a custom [`TemperatureProvider`] (e.g., [`OpenMeteoTemperatureProvider`] or
    /// [`FnTemperatureProvider`]).
    pub fn with_temperature_provider<P: TemperatureProvider>(mut self, provider: P) -> Self {
        self.temperature_provider = Some(Arc::new(provider));
        self
    }

    /// Replaces or clears the custom [`TemperatureProvider`] at runtime.
    pub fn set_temperature_provider(&mut self, provider: Option<Arc<dyn TemperatureProvider>>) {
        self.temperature_provider = provider;
    }

    /// Returns the most recently evaluated [`AtmosphereSample`] at the camera's position & altitude.
    pub fn last_atmosphere_sample(&self) -> Option<AtmosphereSample> {
        self.last_sample.read().ok().and_then(|g| *g)
    }

    #[inline]
    fn active_config<'a>(&'a self, input_cfg: &'a PrecipitationConfig) -> &'a PrecipitationConfig {
        self.config_override.as_ref().unwrap_or(input_cfg)
    }

    fn compute_camera_geo_and_basis(
        ctx: &EarthParticleContext<'_>,
    ) -> (DVec3, Geodetic, DVec3, DVec3, DVec3) {
        let up_hint = ctx
            .input
            .camera_up_ecef
            .map(|[x, y, z]| DVec3::new(x, y, z));
        let placement = ctx.input.camera.placement_with_up(up_hint);
        let eye_world = placement.eye;

        let flat_t = ctx.low_level.local_25d_blend.max(ctx.low_level.map_2d_morph);
        let morph = stabilized_surface_morph(
            ctx.input.camera.focus,
            flat_t as f64,
            ctx.low_level.map_2d_morph as f64,
            ctx.input.tile_projection,
        );

        let focus_geo = ctx.input.camera.focus;
        let (focus_origin, focus_frame) =
            project_morphed_point_and_frame(focus_geo.lat_deg, focus_geo.lon_deg, 0.0, morph);

        let eye_geo = if flat_t > 0.05 {
            // In 2.5D/2D morphed space, compute exact height & local lat/lon relative to morphed focus frame.
            let rel = eye_world.sub(focus_origin);
            let rel_e = rel.dot(focus_frame.east);
            let rel_n = rel.dot(focus_frame.north);
            let rel_u = rel.dot(focus_frame.up).max(0.0);
            let m_per_deg = crate::geodesy::wgs84::A.to_radians();
            let lat = (focus_geo.lat_deg + rel_n / m_per_deg).clamp(-89.95, 89.95);
            let cos_lat = lat.to_radians().cos().abs().max(0.15);
            let lon = ((focus_geo.lon_deg + rel_e / (m_per_deg * cos_lat)) + 180.0)
                .rem_euclid(360.0)
                - 180.0;
            Geodetic::new(lat, lon, rel_u)
        } else {
            ecef_to_geodetic(eye_world)
        };

        let (_, eye_frame) = project_morphed_point_and_frame(
            eye_geo.lat_deg,
            eye_geo.lon_deg,
            eye_geo.height_m,
            morph,
        );
        (eye_world, eye_geo, eye_frame.east, eye_frame.north, eye_frame.up)
    }

    fn is_below_ceiling_for_tick(&self, ctx: &EarthPluginFrameContext<'_>) -> bool {
        if !self.enabled {
            return false;
        }
        let cfg = self.active_config(&ctx.input.precipitation);
        if !cfg.enabled || cfg.intensity <= 0.002 {
            return false;
        }
        if !cfg.respect_cloud_ceiling {
            return true;
        }
        // Conservative upper altitude test (ground <= 9,000 m) to skip ticking in orbital space.
        let max_ceiling = match cfg.atmosphere.cloud_ceiling {
            CloudCeiling::FixedMsl(alt) => alt,
            CloudCeiling::AboveGround(agl) => agl + 9_000.0,
            CloudCeiling::CloudShell => CLOUD_SHELL_ALTITUDE_M,
        };
        ctx.altitude_m <= max_ceiling + 250.0
    }
}

impl EarthPlugin for PrecipitationPlugin {
    fn id(&self) -> &'static str {
        Self::ID
    }

    fn hooks(&self) -> EarthPluginHooks {
        EarthPluginHooks::TICK | EarthPluginHooks::PARTICLES
    }

    fn is_enabled(&self) -> bool {
        self.enabled
    }

    fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    fn wants_animation_tick(&self, ctx: &EarthPluginFrameContext<'_>) -> bool {
        self.is_below_ceiling_for_tick(ctx)
    }

    fn on_tick(&mut self, dt_s: f64, ctx: &EarthPluginFrameContext<'_>) -> bool {
        if !self.is_below_ceiling_for_tick(ctx) {
            return false;
        }
        let dt = dt_s.clamp(0.0, 0.1);
        self.sim_time_s += dt;
        self.pending_dt_s += dt;
        true
    }

    fn on_particle_layers(
        &mut self,
        ctx: &mut EarthParticleContext<'_>,
        layers_out: &mut Vec<RetainedParticleLayer>,
        textures_out: &mut Vec<Arc<RetainedTileTexture>>,
    ) {
        let cfg = self.active_config(&ctx.input.precipitation).clone();
        if !self.enabled || !cfg.enabled || cfg.intensity <= 0.002 {
            self.pending_dt_s = 0.0;
            self.last_eye_world = None;
            return;
        }

        let (eye_world, eye_geo, east, north, up) = Self::compute_camera_geo_and_basis(ctx);
        let ground_elev_m = if ctx.input.elevation_enabled {
            ctx.sample_ground_elevation_m(eye_geo.lat_deg, eye_geo.lon_deg)
                .max(0.0)
        } else {
            0.0
        };
        let eye_alt_msl = eye_geo.height_m.max(0.0);
        let camera_agl_m = (eye_alt_msl - ground_elev_m).max(0.0);

        let query = AtmosphereQuery {
            lat_deg: eye_geo.lat_deg,
            lon_deg: eye_geo.lon_deg,
            altitude_msl_m: eye_alt_msl,
            ground_elevation_m: ground_elev_m,
        };
        let sample = if let Some(ref custom_provider) = self.temperature_provider {
            custom_provider.sample(query, &cfg.atmosphere)
        } else if cfg.use_live_temperature {
            self.live_meteo.sample(query, &cfg.atmosphere)
        } else {
            StaticAtmosphereProvider.sample(query, &cfg.atmosphere)
        };

        if let Ok(mut guard) = self.last_sample.write() {
            *guard = Some(sample);
        }

        let ceiling_msl = sample.cloud_ceiling_msl_m;
        let fade_band = cfg.atmosphere.cloud_fade_band_m.max(50.0);
        let cloud_envelope = if cfg.respect_cloud_ceiling {
            (1.0 - smoothstep_f64(ceiling_msl - fade_band, ceiling_msl, eye_alt_msl)) as f32
        } else {
            1.0
        };

        // Compute world-locked ENU camera displacement since last emission (panning right -> +delta_e).
        let (delta_e, delta_n, delta_u) = match self.last_eye_world {
            Some(prev) => {
                let d = eye_world.sub(prev);
                if d.length() < 50_000.0 {
                    (d.dot(east), d.dot(north), d.dot(up))
                } else {
                    (0.0, 0.0, 0.0)
                }
            }
            None => (0.0, 0.0, 0.0),
        };
        self.last_eye_world = Some(eye_world);
        let dt_step = self.pending_dt_s;
        self.pending_dt_s = 0.0;

        if cfg.respect_cloud_ceiling && cloud_envelope <= 0.002 {
            return;
        }

        let eff_wind = sample.wind_mps.unwrap_or(cfg.wind_mps);
        let eval_ctx = PrecipitationEvalContext {
            config: &cfg,
            camera_lat_deg: eye_geo.lat_deg,
            camera_lon_deg: eye_geo.lon_deg,
            camera_altitude_msl_m: eye_alt_msl,
            ground_elevation_m: ground_elev_m,
            camera_agl_m,
            atmosphere: sample,
            cloud_envelope,
            intensity: cfg.intensity,
            time_s: self.sim_time_s,
            wind_mps: eff_wind,
            view_mode: ctx.input.view_mode,
            map_2d_morph: ctx.low_level.map_2d_morph,
        };

        let layers = (self.evaluator)(&eval_ctx);
        let mut seen_tex_ids: HashSet<u64> = HashSet::new();
        let wrapped_time_s = (self.sim_time_s % 3_600.0) as f32;

        for layer in layers {
            if layer.intensity <= 0.002 || layer.color[3] <= 0.002 || layer.max_particles == 0 {
                continue;
            }
            let particle_count = ((layer.max_particles as f32) * layer.intensity.min(2.0))
                .round()
                .clamp(1.0, 16_384.0) as u32;
            let extent_xy = (layer.volume_size_m[0] as f64).max(4.0);
            let extent_z = (layer.volume_size_m[1] as f64).max(4.0);

            // Incremental f64 toroidal phase update in normalized [0, 1) cycles:
            // - Camera panning right (+delta_e) subtracts from phase_e -> drops stream left!
            // - Wind blowing east (+wind[0]) adds to phase_e -> drops drift east!
            // - Gravity downward (+fall_speed_mps) subtracts from phase_z -> drops fall down!
            // - Smooth volume resizing across zoom causes zero phase snapping!
            let state = self.layer_phases.entry(layer.id).or_default();
            state.phase_e = (state.phase_e
                + ((layer.wind_mps[0] as f64) * dt_step - delta_e) / extent_xy)
                .rem_euclid(1.0);
            state.phase_n = (state.phase_n
                + ((layer.wind_mps[1] as f64) * dt_step - delta_n) / extent_xy)
                .rem_euclid(1.0);
            state.phase_z = (state.phase_z
                + (-(layer.fall_speed_mps as f64) * dt_step - delta_u) / extent_z)
                .rem_euclid(1.0);

            let sprite = match &layer.sprite {
                ParticleSprite::RainStreak => RetainedParticleSprite::ProceduralStreak,
                ParticleSprite::Snowflake => RetainedParticleSprite::ProceduralSnowflake,
                ParticleSprite::SleetPellet => RetainedParticleSprite::ProceduralPellet,
                ParticleSprite::Texture {
                    id,
                    revision,
                    width,
                    height,
                    rgba8,
                } => {
                    if seen_tex_ids.insert(*id) {
                        textures_out.push(Arc::new(RetainedTileTexture {
                            id: *id,
                            revision: *revision,
                            width: *width,
                            height: *height,
                            rgba8: Arc::clone(rgba8),
                        }));
                    }
                    RetainedParticleSprite::Textured(*id)
                }
            };

            let billboard = match layer.billboard {
                ParticleBillboard::VelocityAligned => RetainedParticleBillboard::VelocityAligned,
                ParticleBillboard::CameraFacing => RetainedParticleBillboard::CameraFacing,
            };

            let ceiling_for_shader = if cfg.respect_cloud_ceiling {
                ceiling_msl as f32
            } else {
                1.0e9_f32
            };

            layers_out.push(RetainedParticleLayer {
                id: layer.id,
                particle_count,
                sprite,
                billboard,
                basis_east: [east.x as f32, east.y as f32, east.z as f32],
                basis_north: [north.x as f32, north.y as f32, north.z as f32],
                basis_up: [up.x as f32, up.y as f32, up.z as f32],
                wrapped_phase: [
                    state.phase_e as f32,
                    state.phase_n as f32,
                    state.phase_z as f32,
                ],
                altitude_bounds: [
                    eye_alt_msl as f32,
                    ground_elev_m as f32,
                    ceiling_for_shader,
                    fade_band as f32,
                ],
                volume_params: [
                    extent_xy as f32,
                    extent_z as f32,
                    0.22,
                    (extent_xy as f32 * 0.032).clamp(0.8, 3.5),
                ],
                motion_params: [
                    layer.fall_speed_mps,
                    layer.wind_mps[0],
                    layer.wind_mps[1],
                    wrapped_time_s,
                ],
                size_params: [
                    layer.particle_size_m[0],
                    layer.particle_size_m[1],
                    layer.velocity_stretch,
                    layer.spin_rad_s,
                ],
                flutter_clamp: [
                    layer.flutter[0],
                    layer.flutter[1],
                    layer.screen_px_clamp[0],
                    layer.screen_px_clamp[1],
                ],
                color: layer.color,
                seed: layer.seed,
            });
        }
    }

    fn as_any(&self) -> Option<&dyn Any> {
        Some(self)
    }

    fn as_any_mut(&mut self) -> Option<&mut dyn Any> {
        Some(self)
    }
}
