//! GPU-native physical ocean wave spectrum plugin with smooth dual-state wind transitions
//! and opt-in O(N) Euclidean Distance Transform coastal shoreline shoaling & periodic surf.

use std::any::Any;
use std::sync::Arc;

use strata_render::scene3d::{
    RetainedOceanSurface, RetainedOceanWaveTrain, RetainedTileTexture,
};

use crate::geodesy::{wgs84, wrap_longitude, Geodetic};
use crate::plugins::{
    EarthOceanContext, EarthPlugin, EarthPluginFrameContext, EarthPluginHooks,
    EarthWorldBuildContext, LoadedTileImagerySnapshot,
};
use crate::scene::{EarthInput, OceanWavesConfig};
use crate::weather_api::WindVector;

use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_COASTAL_SDF_TEX_ID: AtomicU64 = AtomicU64::new(0xE47A_1000);
const COASTAL_SDF_RES: u32 = 128;

fn alloc_coastal_sdf_tex_id() -> u64 {
    NEXT_COASTAL_SDF_TEX_ID.fetch_add(1, Ordering::Relaxed)
}

#[inline]
fn smoothstep_f32(edge0: f32, edge1: f32, x: f32) -> f32 {
    if (edge1 - edge0).abs() < 1e-6 {
        return if x >= edge1 { 1.0 } else { 0.0 };
    }
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

use crate::plugins::sample_land_mask_at;

/// Determines non-land water weight in `[0.0 .. 1.0]` combining global landmass mask,
/// loaded terrain elevation, and coastal tile pixel classification.
#[inline]
pub fn classify_non_land_water(
    lat_deg: f64,
    lon_deg: f64,
    elev_m: f64,
    imagery: Option<&LoadedTileImagerySnapshot>,
) -> f32 {
    let land_r = sample_land_mask_at(lat_deg, lon_deg);
    if land_r >= 0.95 && elev_m > 1.2 {
        return 0.0;
    }
    if elev_m > 2.2 {
        return 0.0;
    }
    let elev_fade = 1.0 - smoothstep_f32(0.25, 1.9, elev_m as f32);
    if elev_fade <= 0.01 {
        return 0.0;
    }

    // Open deep ocean far from continents: trust landmass mask so offshore satellite clouds
    // don't punch dry holes in ocean waves.
    if land_r <= 0.07 {
        return elev_fade;
    }

    // Coastal transition & inland harbors/bays: refine sharply against loaded tile imagery.
    // Physically, chlorophyll foliage & tan beach sand absorb blue (R > B), whereas both deep blue
    // pelagic water and emerald/kelp coastal seawater absorb red (B > R).
    if let Some(snap) = imagery {
        if let Some([r, g, b, _a]) = snap.sample_rgba(lat_deg, lon_deg) {
            let max_rgb = r.max(g).max(b);
            let min_rgb = r.min(g).min(b);
            let cool_water =
                b >= r && b as i32 + 6 >= g as i32 && (b as i32 - r as i32 >= 3 || max_rgb < 110);
            let emerald_water =
                (b as i32 >= r as i32 + 4) && (g as i32 >= r as i32 + 4) && r < 118;
            let surf_foam =
                min_rgb > 142 && (b as i32 >= r as i32 - 3) && elev_m < 0.85 && land_r < 0.72;
            let warm_or_foliage_land =
                (r as i32 - b as i32 >= 6) || (min_rgb > 164 && r as i32 > b as i32 + 2);
            let px_water = if warm_or_foliage_land {
                0.0
            } else if cool_water || emerald_water || surf_foam {
                1.0
            } else {
                0.0
            };
            let mask_water = 1.0 - smoothstep_f32(0.18, 0.72, land_r);
            let blended = if land_r < 0.30 {
                px_water * 0.72 + mask_water * 0.28
            } else {
                px_water
            };
            return (blended * elev_fade).clamp(0.0, 1.0);
        }
    }

    let mask_water = 1.0 - smoothstep_f32(0.16, 0.62, land_r);
    (mask_water * elev_fade).clamp(0.0, 1.0)
}

/// 1D squared Euclidean Distance Transform (Felzenszwalb & Huttenlocher, O(N) exact).
fn edt_1d(f: &[f32], out: &mut [f32], v: &mut [usize], z: &mut [f32]) {
    let n = f.len();
    if n == 0 {
        return;
    }
    let mut k = 0usize;
    v[0] = 0;
    z[0] = f32::NEG_INFINITY;
    z[1] = f32::INFINITY;

    for q in 1..n {
        let fq = f[q];
        loop {
            let vk = v[k];
            let s = ((fq + (q * q) as f32) - (f[vk] + (vk * vk) as f32))
                / (2.0 * (q as f32 - vk as f32));
            if s <= z[k] {
                if k == 0 {
                    z[0] = f32::NEG_INFINITY;
                    break;
                }
                k -= 1;
            } else {
                k += 1;
                v[k] = q;
                z[k] = s;
                z[k + 1] = f32::INFINITY;
                break;
            }
        }
    }

    let mut k = 0usize;
    for q in 0..n {
        while z[k + 1] < q as f32 {
            k += 1;
        }
        let d = q as f32 - v[k] as f32;
        out[q] = d * d + f[v[k]];
    }
}

/// Computes an exact O(N) Euclidean Distance-to-Shore RGBA8 texture from a `width x height`
/// water-mask grid (`water_grid[y * width + x] in [0.0..=1.0]`, where `x` increases East and
/// `y` increases North).
///
/// Output RGBA8 channels per texel:
/// - `R`: water mask (`0` = land, `255` = water)
/// - `G`: normalized distance to nearest land cell (`0` = shoreline edge, `255` = `>= shoaling_range_m`)
/// - `B`: unit shoreward direction East component (`128 + 127 * dir_e`, pointing TOWARD land)
/// - `A`: unit shoreward direction North component (`128 + 127 * dir_n`, pointing TOWARD land)
pub fn compute_coastal_sdf_grid(
    water_grid: &[f32],
    width: usize,
    height: usize,
    cell_size_m: f32,
    shoaling_range_m: f32,
) -> Vec<u8> {
    let n = width * height;
    const INF: f32 = 1.0e14;

    let mut row_in = vec![0.0f32; width.max(height)];
    let mut row_out = vec![0.0f32; width.max(height)];
    let mut v_buf = vec![0usize; width.max(height) + 1];
    let mut z_buf = vec![0.0f32; width.max(height) + 2];

    let mut d2 = vec![INF; n];
    // Pass 1: horizontal 1D EDT per row (land cells have initial distance 0).
    for y in 0..height {
        let base = y * width;
        for x in 0..width {
            row_in[x] = if water_grid[base + x] < 0.45 { 0.0 } else { INF };
        }
        edt_1d(&row_in[..width], &mut row_out[..width], &mut v_buf, &mut z_buf);
        d2[base..base + width].copy_from_slice(&row_out[..width]);
    }
    // Pass 2: vertical 1D EDT per column.
    let mut dist_m = vec![0.0f32; n];
    for x in 0..width {
        for y in 0..height {
            row_in[y] = d2[y * width + x];
        }
        edt_1d(&row_in[..height], &mut row_out[..height], &mut v_buf, &mut z_buf);
        for y in 0..height {
            let sq = row_out[y];
            dist_m[y * width + x] = if sq >= INF * 0.5 {
                shoaling_range_m * 2.0
            } else {
                sq.max(0.0).sqrt() * cell_size_m
            };
        }
    }

    let inv_range = 1.0 / shoaling_range_m.max(10.0);
    let mut rgba = vec![0u8; n * 4];
    for y in 0..height {
        let y0 = y.saturating_sub(1);
        let y1 = (y + 1).min(height - 1);
        for x in 0..width {
            let x0 = x.saturating_sub(1);
            let x1 = (x + 1).min(width - 1);
            let idx = y * width + x;
            let w = water_grid[idx].clamp(0.0, 1.0);
            let dm = dist_m[idx];
            let d_norm = (dm * inv_range).clamp(0.0, 1.0);

            // Gradient points AWAY from land -> negate to obtain unit vector TOWARD nearest shoreline!
            let gx = dist_m[y * width + x1] - dist_m[y * width + x0];
            let gy = dist_m[y1 * width + x] - dist_m[y0 * width + x];
            let glen = (gx * gx + gy * gy).sqrt();
            let (dir_e, dir_n) = if glen > 1e-4 {
                (-gx / glen, -gy / glen)
            } else {
                (0.0, 0.0)
            };

            let o = idx * 4;
            rgba[o] = (w * 255.0).round() as u8;
            rgba[o + 1] = (d_norm * 255.0).round() as u8;
            rgba[o + 2] = ((dir_e * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0).round() as u8;
            rgba[o + 3] = ((dir_n * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0).round() as u8;
        }
    }
    rgba
}

#[derive(Debug, Clone, Copy)]
struct WaveTrainState {
    dir_en: [f32; 2],
    wind_speed_mps: f32,
    phase_time_s: f64,
}

impl WaveTrainState {
    fn from_wind(wind: WindVector) -> Self {
        let rad = wind.angle_deg.to_radians();
        Self {
            dir_en: [rad.sin(), rad.cos()],
            wind_speed_mps: wind.speed_mps.clamp(0.2, 60.0),
            phase_time_s: 0.0,
        }
    }
}

/// GPU-native physical ocean wave spectrum plugin with smooth dual-state wind cross-fading
/// and optional coastal shoreline shoaling & periodic breaking surf.
pub struct OceanWavesPlugin {
    pub enabled: bool,
    pub config_override: Option<OceanWavesConfig>,
    initialized_wind: bool,
    train_a: WaveTrainState,
    train_b: WaveTrainState,
    /// Cross-fade progress (`0.0` = 100% `train_a`, `1.0` = 100% `train_b`).
    blend_t: f32,
    transition_active: bool,
    imagery_snapshot: Option<Arc<LoadedTileImagerySnapshot>>,
    imagery_rev: u64,
    cached_sdf_sig: u64,
    cached_sdf_rev: u64,
    coastal_sdf_tex_id: u64,
    cached_sdf_tex: Option<Arc<RetainedTileTexture>>,
    cached_sdf_center_geo: (f64, f64),
    cached_sdf_half_span_m: f32,
    cached_has_water: bool,
}

impl Default for OceanWavesPlugin {
    fn default() -> Self {
        let def_w = WindVector::default();
        let train = WaveTrainState::from_wind(def_w);
        Self {
            enabled: true,
            config_override: None,
            initialized_wind: false,
            train_a: train,
            train_b: train,
            blend_t: 0.0,
            transition_active: false,
            imagery_snapshot: None,
            imagery_rev: 1,
            cached_sdf_sig: 0,
            cached_sdf_rev: 1,
            coastal_sdf_tex_id: alloc_coastal_sdf_tex_id(),
            cached_sdf_tex: None,
            cached_sdf_center_geo: (0.0, 0.0),
            cached_sdf_half_span_m: 1_500.0,
            cached_has_water: true,
        }
    }
}

impl OceanWavesPlugin {
    pub const ID: &'static str = "ocean_waves";

    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_config(mut self, config: OceanWavesConfig) -> Self {
        self.config_override = Some(config);
        self
    }

    fn effective_config<'a>(&'a self, input: &'a EarthInput) -> &'a OceanWavesConfig {
        self.config_override.as_ref().unwrap_or(&input.ocean_waves)
    }

    fn altitude_visibility(alt_m: f64, map_2d_morph: f32) -> f32 {
        if map_2d_morph > 0.05 {
            return 0.0;
        }
        let alt = alt_m as f32;
        1.0 - smoothstep_f32(1_200_000.0, 3_200_000.0, alt)
    }

    fn target_wind(&self, cfg: &OceanWavesConfig, ctx_weather: Option<&crate::weather_api::WeatherSnapshot>) -> WindVector {
        if cfg.sync_with_weather {
            if let Some(ws) = ctx_weather {
                return ws.wind;
            }
        }
        WindVector::new(cfg.wind_speed_mps, cfg.wind_angle_deg)
    }

    /// Smoothly advances wave phase clocks and cross-fades toward `target` wind with zero phase spin.
    pub fn step_wind_state(&mut self, dt_s: f64, speed_mul: f32, transition_s: f32, target: WindVector) {
        let dt = dt_s.clamp(0.0, 0.15);
        let rate = speed_mul.max(0.05) as f64;
        let t_dir = {
            let rad = target.angle_deg.to_radians();
            [rad.sin(), rad.cos()]
        };
        let t_speed = target.speed_mps.clamp(0.2, 60.0);

        if !self.initialized_wind {
            let init = WaveTrainState {
                dir_en: t_dir,
                wind_speed_mps: t_speed,
                phase_time_s: 0.0,
            };
            self.train_a = init;
            self.train_b = init;
            self.blend_t = 0.0;
            self.transition_active = false;
            self.initialized_wind = true;
        }

        // Advance phase clocks continuously (wrap cleanly at 1 hour so f32 stays razor sharp).
        self.train_a.phase_time_s = (self.train_a.phase_time_s + dt * rate).rem_euclid(3_600.0);
        self.train_b.phase_time_s = (self.train_b.phase_time_s + dt * rate).rem_euclid(3_600.0);

        // Compare target against the currently active destination train.
        let dest = if self.transition_active {
            self.train_b
        } else {
            self.train_a
        };
        let dot_dir = (dest.dir_en[0] * t_dir[0] + dest.dir_en[1] * t_dir[1]).clamp(-1.0, 1.0);
        let ang_diff = dot_dir.acos();
        let speed_diff = (dest.wind_speed_mps - t_speed).abs();

        // Trigger a clean dual-state cross-fade when wind direction rotates > 2.5 deg or speed shifts > 0.6 m/s.
        if !self.transition_active && (ang_diff > 0.044 || speed_diff > 0.6) {
            self.train_b = WaveTrainState {
                dir_en: t_dir,
                wind_speed_mps: t_speed,
                phase_time_s: self.train_a.phase_time_s,
            };
            self.blend_t = 0.0;
            self.transition_active = true;
        } else if self.transition_active {
            // Smoothly update destination speed while cross-fading; if direction jumps again mid-fade,
            // commit current dominant state as train_a only once blend passes 0.5.
            let tau = transition_s.clamp(0.4, 20.0) as f64;
            self.blend_t = (self.blend_t as f64 + dt / tau).clamp(0.0, 1.0) as f32;
            if self.blend_t >= 1.0 - 1e-4 {
                self.train_a = self.train_b;
                self.blend_t = 0.0;
                self.transition_active = false;
            } else if ang_diff > 0.25 && self.blend_t > 0.55 {
                self.train_a = self.train_b;
                self.train_b = WaveTrainState {
                    dir_en: t_dir,
                    wind_speed_mps: t_speed,
                    phase_time_s: self.train_a.phase_time_s,
                };
                self.blend_t = 0.0;
            }
        }
    }

    /// Returns `(weight_a, weight_b)` smoothstep blend weights for the dual wave trains.
    #[inline]
    pub fn current_train_weights(&self) -> (f32, f32) {
        if !self.transition_active {
            (1.0, 0.0)
        } else {
            let b = smoothstep_f32(0.0, 1.0, self.blend_t);
            (1.0 - b, b)
        }
    }

    fn is_currently_active(&self, ctx: &EarthPluginFrameContext<'_>) -> bool {
        let cfg = self.effective_config(ctx.input);
        self.enabled
            && cfg.enabled
            && ctx.input.layers.terrain
            && Self::altitude_visibility(ctx.altitude_m, ctx.low_level.map_2d_morph) > 0.01
    }
}

impl EarthPlugin for OceanWavesPlugin {
    fn id(&self) -> &'static str {
        Self::ID
    }

    fn hooks(&self) -> EarthPluginHooks {
        EarthPluginHooks::TICK | EarthPluginHooks::WORLD_BUILD | EarthPluginHooks::OCEAN
    }

    fn is_enabled(&self) -> bool {
        self.enabled
    }

    fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    fn wants_animation_tick(&self, ctx: &EarthPluginFrameContext<'_>) -> bool {
        self.is_currently_active(ctx)
    }

    fn on_tick(&mut self, dt_s: f64, ctx: &EarthPluginFrameContext<'_>) -> bool {
        if !self.is_currently_active(ctx) {
            return false;
        }
        let cfg = self.effective_config(ctx.input).clone();
        let target = self.target_wind(&cfg, ctx.weather_snapshot.as_ref());
        self.step_wind_state(dt_s, cfg.speed, cfg.wind_transition_s, target);
        true
    }

    fn on_world_build(&mut self, ctx: &mut EarthWorldBuildContext<'_>) {
        if !self.enabled || !self.effective_config(ctx.input).enabled {
            return;
        }
        if let Some(snap) = ctx.imagery_snapshot {
            self.imagery_snapshot = Some(Arc::new(snap.clone()));
            self.imagery_rev = self.imagery_rev.wrapping_add(1).max(1);
        }
    }

    fn on_ocean_surface(&mut self, ctx: &mut EarthOceanContext<'_>) -> Option<RetainedOceanSurface> {
        let cfg = self.effective_config(ctx.input).clone();
        if !self.enabled || !cfg.enabled || !ctx.input.layers.terrain {
            return None;
        }

        let alt_m = ctx.input.camera.altitude_m().max(2.0);
        let alt_fade = Self::altitude_visibility(alt_m, ctx.low_level.map_2d_morph);
        if alt_fade <= 0.01 {
            return None;
        }

        let target = self.target_wind(&cfg, ctx.weather_snapshot.as_ref());
        if !self.initialized_wind {
            self.step_wind_state(0.0, cfg.speed, cfg.wind_transition_s, target);
        }

        let focus = Geodetic::new(
            ctx.input.camera.focus.lat_deg.clamp(-84.5, 84.5),
            wrap_longitude(ctx.input.camera.focus.lon_deg),
            0.0,
        );
        let flat_t = ctx
            .low_level
            .local_25d_blend
            .max(ctx.low_level.map_2d_morph)
            .clamp(0.0, 1.0) as f64;
        let morph = crate::geodesy::stabilized_surface_morph(
            ctx.input.camera.focus,
            flat_t,
            ctx.low_level.map_2d_morph as f64,
            ctx.input.tile_projection,
        );
        // Quantize ocean RTE origin to ~0.05 deg cells (~5 km) so wave spatial phase never drifts
        // during panning and stays well inside f32 precision across 3D, 2.5D, and 2D morphs.
        let q_lat = (focus.lat_deg * 20.0).round() / 20.0;
        let q_lon = wrap_longitude((focus.lon_deg * 20.0).round() / 20.0);
        let (origin_ecef, frame) =
            crate::geodesy::project_morphed_point_and_frame(q_lat, q_lon, 0.0, morph);
        let cos_lat = q_lat.to_radians().cos().abs().max(0.20);

        let dist_to_focus = ctx.input.camera.eye_distance_m().max(alt_m);
        let shoaling_range_m = cfg.shoaling_range_m.clamp(40.0, 2_000.0);

        let mut coastal_sdf = None;
        let mut sdf_half_span_m = (dist_to_focus * 2.2).clamp(600.0, 18_000.0) as f32;
        let mut sdf_center_en_m = [0.0f32, 0.0f32];

        // Opt-in coastal shoreline SDF texture (zero CPU cost when `coastal_shoaling == false`!).
        if cfg.coastal_shoaling && alt_m < 45_000.0 {
            let res = COASTAL_SDF_RES as usize;
            // Quantize SDF window center & span so scrolling smoothly reuses cached SDF without rebuilding every frame.
            let span_q = ((sdf_half_span_m.ln() * 4.0).round() / 4.0).exp().clamp(600.0, 18_000.0);
            let step_deg_lat = ((span_q as f64 * 0.25) / wgs84::A).to_degrees().max(1e-4);
            let step_deg_lon = (step_deg_lat / cos_lat).max(1e-4);
            let c_lat = (focus.lat_deg / step_deg_lat).round() * step_deg_lat;
            let c_lon = wrap_longitude((focus.lon_deg / step_deg_lon).round() * step_deg_lon);

            let sig = {
                let mut h = 0xcbf2_9ce4_8422_2325u64;
                for v in [
                    (c_lat * 10_000.0).round() as i64 as u64,
                    (c_lon * 10_000.0).round() as i64 as u64,
                    (span_q * 10.0).round() as u64,
                    (shoaling_range_m * 10.0).round() as u64,
                    self.imagery_rev,
                ] {
                    h ^= v;
                    h = h.wrapping_mul(0x1000_0000_1b3);
                }
                h
            };

            if sig != self.cached_sdf_sig || self.cached_sdf_tex.is_none() {
                self.cached_sdf_sig = sig;
                self.cached_sdf_center_geo = (c_lat, c_lon);
                self.cached_sdf_half_span_m = span_q;

                let elev_guard = ctx.ground_snapshot.and_then(|l| l.read().ok());
                let elev_ref = elev_guard.as_deref();
                let img_ref = self.imagery_snapshot.as_deref();

                let mut water_grid = vec![0.0f32; res * res];
                let mut any_water = false;
                let c_cos_lat = c_lat.to_radians().cos().abs().max(0.20);
                let cell_size_m = (2.0 * span_q) / (res as f32);

                for y in 0..res {
                    let vn = ((y as f64 + 0.5) / (res as f64) - 0.5) * 2.0 * (span_q as f64);
                    let lat = (c_lat + (vn / wgs84::A).to_degrees()).clamp(-84.9, 84.9);
                    let row_off = y * res;
                    for x in 0..res {
                        let ve = ((x as f64 + 0.5) / (res as f64) - 0.5) * 2.0 * (span_q as f64);
                        let lon = wrap_longitude(c_lon + (ve / (wgs84::A * c_cos_lat)).to_degrees());
                        let elev = elev_ref.map(|s| s.sample_elevation_m(lat, lon)).unwrap_or(0.0);
                        let w = classify_non_land_water(lat, lon, elev, img_ref);
                        if w > 0.05 {
                            any_water = true;
                        }
                        water_grid[row_off + x] = w;
                    }
                }
                self.cached_has_water = any_water;
                if any_water {
                    // Fast 3x3 majority despeckling preserves exact coastlines while healing isolated
                    // 1-2px surf-foam / boat / shadow speckles that would otherwise punch Moire rings.
                    let mut cleaned = water_grid.clone();
                    for y in 1..res.saturating_sub(1) {
                        for x in 1..res.saturating_sub(1) {
                            let mut w_cnt = 0u8;
                            for dy in [y - 1, y, y + 1] {
                                let row = dy * res;
                                for dx in [x - 1, x, x + 1] {
                                    if water_grid[row + dx] >= 0.45 {
                                        w_cnt += 1;
                                    }
                                }
                            }
                            let idx = y * res + x;
                            if w_cnt >= 6 {
                                cleaned[idx] = water_grid[idx].max(0.88);
                            } else if w_cnt <= 3 {
                                cleaned[idx] = water_grid[idx].min(0.12);
                            }
                        }
                    }
                    let rgba = compute_coastal_sdf_grid(
                        &cleaned,
                        res,
                        res,
                        cell_size_m,
                        shoaling_range_m,
                    );
                    self.cached_sdf_rev = self.cached_sdf_rev.wrapping_add(1).max(1);
                    self.cached_sdf_tex = Some(Arc::new(RetainedTileTexture {
                        id: self.coastal_sdf_tex_id,
                        revision: self.cached_sdf_rev,
                        width: res as u32,
                        height: res as u32,
                        rgba8: Arc::new(rgba),
                    }));
                } else {
                    self.cached_sdf_tex = None;
                }
            }

            if self.cached_has_water {
                let (c_lat, c_lon) = self.cached_sdf_center_geo;
                let (c_world, _) =
                    crate::geodesy::project_morphed_point_and_frame(c_lat, c_lon, 0.0, morph);
                let rel_c = c_world.sub(origin_ecef);
                let mut rel_e = rel_c.dot(frame.east);
                if crate::geodesy::is_canonical_2d_morph(morph.map_2d_morph) {
                    let world_circ = 2.0 * std::f64::consts::PI * wgs84::A;
                    rel_e -= (rel_e / world_circ).round() * world_circ;
                }
                sdf_center_en_m = [rel_e as f32, rel_c.dot(frame.north) as f32];
                sdf_half_span_m = self.cached_sdf_half_span_m;
                coastal_sdf = self.cached_sdf_tex.clone();
            } else {
                // Confirmed 100% inland within the SDF patch -> skip GPU ocean shading cleanly.
                return None;
            }
        } else {
            // Fast 5-point inland rejection check when coastal SDF is off (saves fragment work over Denver/Sahara).
            // Use >= 0.28 deg span (>1.5 texels of the 2048x1024 global land mask) plus loaded imagery
            // so coastal beaches, bays (SF Bay), and narrow straits (Bosphorus) are never misclassified as inland.
            let span_deg = ((dist_to_focus * 2.5) / wgs84::A)
                .to_degrees()
                .clamp(0.28, 8.0);
            let probe_pts = [
                (focus.lat_deg, focus.lon_deg),
                ((focus.lat_deg + span_deg).clamp(-84.9, 84.9), focus.lon_deg),
                ((focus.lat_deg - span_deg).clamp(-84.9, 84.9), focus.lon_deg),
                (focus.lat_deg, wrap_longitude(focus.lon_deg + span_deg / cos_lat)),
                (focus.lat_deg, wrap_longitude(focus.lon_deg - span_deg / cos_lat)),
            ];
            let img_ref = self.imagery_snapshot.as_deref();
            let elev_guard = ctx.ground_snapshot.and_then(|l| l.read().ok());
            let elev_ref = elev_guard.as_deref();
            let any_sea = probe_pts.iter().any(|&(la, lo)| {
                if sample_land_mask_at(la, lo) < 0.92 {
                    return true;
                }
                if img_ref.is_some() {
                    let elev = elev_ref.map(|s| s.sample_elevation_m(la, lo)).unwrap_or(0.0);
                    return classify_non_land_water(la, lo, elev, img_ref) > 0.08;
                }
                false
            });
            if !any_sea {
                return None;
            }
        }

        let (wa, wb) = self.current_train_weights();
        Some(RetainedOceanSurface {
            enabled: true,
            origin_ecef: [origin_ecef.x, origin_ecef.y, origin_ecef.z],
            basis_east: [frame.east.x as f32, frame.east.y as f32, frame.east.z as f32],
            basis_north: [
                frame.north.x as f32,
                frame.north.y as f32,
                frame.north.z as f32,
            ],
            basis_up: [frame.up.x as f32, frame.up.y as f32, frame.up.z as f32],
            train_a: RetainedOceanWaveTrain {
                dir_en: self.train_a.dir_en,
                wind_speed_mps: self.train_a.wind_speed_mps,
                phase_time_s: self.train_a.phase_time_s as f32,
                weight: wa,
            },
            train_b: RetainedOceanWaveTrain {
                dir_en: self.train_b.dir_en,
                wind_speed_mps: self.train_b.wind_speed_mps,
                phase_time_s: self.train_b.phase_time_s as f32,
                weight: wb,
            },
            wave_strength: cfg.wave_strength,
            wave_scale: cfg.wave_scale,
            glint_intensity: cfg.glint_intensity,
            opacity: cfg.opacity.clamp(0.05, 1.0),
            alt_fade,
            coastal_shoaling: cfg.coastal_shoaling && coastal_sdf.is_some(),
            shoaling_range_m,
            sdf_half_span_m,
            sdf_center_en_m,
            coastal_sdf,
        })
    }

    fn as_any(&self) -> Option<&dyn Any> {
        Some(self)
    }

    fn as_any_mut(&mut self) -> Option<&mut dyn Any> {
        Some(self)
    }
}
