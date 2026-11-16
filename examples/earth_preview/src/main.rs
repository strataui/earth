// Interactive Earth window example using `strata_earth` and `strata::Canvas3D`.
//
// Opens a single native GPU window featuring the full-viewport Earth scene
// (smooth damped pan, orbit/tilt around Earth's center in 3D, gentle scroll zoom
// with seamless 3D <-> 2.5D ground transition) and floating top-right control
// buttons that programmatically drive [`EarthViewport`] modes, lighting, ephemeris,
// heightmap fetching, orbital satellites, global clouds, and 2.5D weather.

use std::f64::consts::PI;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use strata::prelude::*;
use strata_earth::{
    EarthCamera, EarthDefaultTileSource, EarthInput, EarthViewMode, EarthViewport, Geodetic,
    LayerToggles, SpaceEnvironmentConfig, SunDirectionMode, SurfaceMode, TileProjection,
    WeatherConfig,
};
use strata_layout::Rect;
use strata_render::scene3d::Canvas3D;

const EARTH_RADIUS_M: f64 = 6_378_137.0;

#[inline]
fn smoothstep(edge0: f64, edge1: f64, x: f64) -> f64 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[inline]
fn wrap_lon_delta_deg(delta: f64) -> f64 {
    ((delta + 180.0).rem_euclid(360.0)) - 180.0
}

#[inline]
fn wrap_deg_360_delta(target: f64, current: f64) -> f64 {
    ((target - current + 180.0).rem_euclid(360.0)) - 180.0
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum SunDemoMode {
    SystemClock,
    OrbitalCycle,
    FixedSunset,
}

struct OrbitState {
    lat_deg: f64,
    lon_deg: f64,
    heading_deg: f64,
    tilt_deg: f64,
    range_m: f64,

    target_lat_deg: f64,
    target_lon_deg: f64,
    target_heading_deg: f64,
    target_tilt_deg: f64,
    target_range_m: f64,

    left_down: bool,
    right_down: bool,
    last_pos: Option<(f32, f32)>,
    viewport_w: f64,
    viewport_h: f64,
    sun_demo_mode: SunDemoMode,
}

impl OrbitState {
    fn new(
        lat_deg: f64,
        lon_deg: f64,
        heading_deg: f64,
        tilt_deg: f64,
        range_m: f64,
        w: u32,
        h: u32,
    ) -> Self {
        Self {
            lat_deg,
            lon_deg,
            heading_deg,
            tilt_deg,
            range_m,
            target_lat_deg: lat_deg,
            target_lon_deg: lon_deg,
            target_heading_deg: heading_deg,
            target_tilt_deg: tilt_deg,
            target_range_m: range_m,
            left_down: false,
            right_down: false,
            last_pos: None,
            viewport_w: w as f64,
            viewport_h: h as f64,
            sun_demo_mode: SunDemoMode::SystemClock,
        }
    }

    fn snap_targets(&mut self) {
        self.lat_deg = self.target_lat_deg;
        self.lon_deg = self.target_lon_deg;
        self.heading_deg = self.target_heading_deg;
        self.tilt_deg = self.target_tilt_deg;
        self.range_m = self.target_range_m;
    }

    fn has_pending_motion(&self) -> bool {
        (self.range_m.ln() - self.target_range_m.ln()).abs() > 2e-4
            || (self.lat_deg - self.target_lat_deg).abs() > 1e-4
            || wrap_lon_delta_deg(self.target_lon_deg - self.lon_deg).abs() > 1e-4
            || wrap_deg_360_delta(self.target_heading_deg, self.heading_deg).abs() > 1e-3
            || (self.tilt_deg - self.target_tilt_deg).abs() > 1e-3
    }

    fn step_damping(&mut self, dt_s: f64) -> bool {
        if !self.has_pending_motion() {
            return false;
        }
        let alpha_zoom = 1.0 - (-dt_s * 10.5).exp();
        let alpha_orbit = 1.0 - (-dt_s * 18.0).exp();

        let log_cur = self.range_m.ln();
        let log_tgt = self.target_range_m.ln();
        if (log_tgt - log_cur).abs() <= 2e-4 {
            self.range_m = self.target_range_m;
        } else {
            self.range_m = (log_cur + (log_tgt - log_cur) * alpha_zoom).exp();
        }

        let dlat = self.target_lat_deg - self.lat_deg;
        self.lat_deg = if dlat.abs() <= 1e-4 {
            self.target_lat_deg
        } else {
            self.lat_deg + dlat * alpha_orbit
        };

        let dlon = wrap_lon_delta_deg(self.target_lon_deg - self.lon_deg);
        self.lon_deg = if dlon.abs() <= 1e-4 {
            self.target_lon_deg
        } else {
            wrap_lon_delta_deg(self.lon_deg + dlon * alpha_orbit)
        };

        let dhead = wrap_deg_360_delta(self.target_heading_deg, self.heading_deg);
        self.heading_deg = if dhead.abs() <= 1e-3 {
            self.target_heading_deg
        } else {
            (self.heading_deg + dhead * alpha_orbit).rem_euclid(360.0)
        };

        let dtilt = self.target_tilt_deg - self.tilt_deg;
        self.tilt_deg = if dtilt.abs() <= 1e-3 {
            self.target_tilt_deg
        } else {
            self.tilt_deg + dtilt * alpha_orbit
        };

        true
    }

    fn to_camera(&self) -> EarthCamera {
        // In 3D space (>1,200 km), always aim straight at the center of Earth so the full
        // globe stays centered from every orbit angle; blend smoothly to angled ground 2.5D near surface.
        let ground_weight = 1.0 - smoothstep(95_000.0, 1_150_000.0, self.range_m);
        let eff_tilt_deg = (self.tilt_deg * ground_weight).max(0.05);
        let tilt_rad = eff_tilt_deg.to_radians();
        let head_rad = self.heading_deg.to_radians();
        let horiz = self.range_m * tilt_rad.sin();
        let up = self.range_m * tilt_rad.cos();
        let center_aim_drop = -EARTH_RADIUS_M * (1.0 - ground_weight);
        EarthCamera {
            focus: Geodetic::new(self.lat_deg, self.lon_deg, 0.0),
            aim_offset_m: [0.0, 0.0, center_aim_drop],
            eye_offset_m: [-head_rad.sin() * horiz, -head_rad.cos() * horiz, up],
            fov_y_deg: 55.0,
            near_m: (self.range_m * 0.002).clamp(2.0, 20_000.0),
        }
    }

    fn to_initial_input(
        &self,
        view_mode: EarthViewMode,
        space: SpaceEnvironmentConfig,
        weather: WeatherConfig,
    ) -> EarthInput {
        EarthInput {
            camera: self.to_camera(),
            viewport: Rect::new(0.0, 0.0, self.viewport_w as f32, self.viewport_h as f32),
            view_mode,
            seamless_25d: true,
            tile_projection: TileProjection::WebMercator,
            daylight: 0.96,
            cloud_cover: 0.0,
            surface_mode: SurfaceMode::QuadtreeLod,
            layers: LayerToggles {
                terrain: true,
                buildings: false,
                graticule: false,
                labels: false,
                clouds: true,
                celestial: true,
            },
            elevation_enabled: true,
            space,
            weather,
            tile_grid: 9,
            max_tile_level: 16,
            sse_threshold: 4.5,
            ..EarthInput::default()
        }
    }
}

struct PreviewState {
    viewport: EarthViewport,
    orbit: OrbitState,
    last_is_25d_ui: bool,
}

fn control_btn<F>(label: String, active: bool, on_click: F) -> View
where
    F: Fn(Event) + 'static,
{
    let tone = if active {
        "bg-indigo-600/90 border-indigo-400/70 text-white"
    } else {
        "bg-neutral-900/80 border-neutral-700/70 text-neutral-300 hover:bg-neutral-800/90 hover:text-white"
    };
    rsx! {
        <button
            class={format!("px-2.5 py-1 rounded-lg border text-[11px] font-medium cursor-pointer select-none whitespace-nowrap transition-colors {}", tone)}
            onclick={on_click}
        >
            {label}
        </button>
    }
}

fn is_over_controls(rt: &AppRuntime, x: f32, y: f32) -> bool {
    AppRuntime::hit_test_property(
        &rt.root_blueprint,
        &rt.layout_root,
        x,
        y,
        "data-earth-controls",
    )
    .is_some()
}

fn main() -> Result<(), String> {
    let initial_w = 1280u32;
    let initial_h = 800u32;

    let orbit = OrbitState::new(22.0, 12.0, 0.0, 42.0, 13_200_000.0, initial_w, initial_h);

    let space = SpaceEnvironmentConfig::default().with_system_clock();
    let weather = WeatherConfig {
        global_cloud_spin_rad_s: 0.0022,
        ..WeatherConfig::default()
    };
    let input = orbit.to_initial_input(EarthViewMode::Mode3D, space, weather);

    let tile_ready = Arc::new(AtomicBool::new(false));
    let tile_ready_cb = Arc::clone(&tile_ready);
    let cache_dir = std::env::temp_dir().join("strata_earth_preview_cache");

    let mut source = if std::env::var_os("STRATA_EARTH_OFFLINE").is_some() {
        EarthDefaultTileSource::offline_basemap()
    } else {
        EarthDefaultTileSource::world_satellite().with_disk_cache(&cache_dir)
    };
    source.set_on_tile_ready(Some(Arc::new(move || {
        tile_ready_cb.store(true, Ordering::Release);
    })));

    let viewport = EarthViewport::with_wakeup(&input, Box::new(source), None);
    viewport.sync_now();

    let retained = viewport.retained_scene();
    let state = Arc::new(Mutex::new(PreviewState {
        viewport,
        orbit,
        last_is_25d_ui: false,
    }));

    let config = WindowConfig {
        title: "Strata Earth".to_string(),
        width: initial_w,
        height: initial_h,
        present_mode: PresentMode::Fifo,
        resizable: true,
        ..WindowConfig::default()
    };

    let retained_for_view = retained;
    let state_view = Arc::clone(&state);
    let state_events = Arc::clone(&state);
    let state_frame = Arc::clone(&state);
    let tile_ready_frame = Arc::clone(&tile_ready);

    launch_with_frame(
        move || {
            let (is_2d, is_25d_view, sp, w, elev_on, sun_mode) = {
                let guard = state_view.lock().unwrap();
                (
                    guard.viewport.view_mode() == EarthViewMode::Mode2D,
                    guard.orbit.range_m < 100_000.0,
                    guard.viewport.space_config().clone(),
                    guard.viewport.weather_config().clone(),
                    guard.viewport.elevation_enabled(),
                    guard.orbit.sun_demo_mode,
                )
            };

            let s_mode = Arc::clone(&state_view);
            let btn_mode = control_btn(
                if is_2d { "Mode: 2D" } else { "Mode: 3D" }.to_string(),
                is_2d,
                move |_| {
                    let mut st = s_mode.lock().unwrap();
                    let next = match st.viewport.view_mode() {
                        EarthViewMode::Mode3D => EarthViewMode::Mode2D,
                        EarthViewMode::Mode2D => EarthViewMode::Mode3D,
                    };
                    st.viewport.set_view_mode(next);
                },
            );

            let s_jump = Arc::clone(&state_view);
            let btn_jump = control_btn(
                if is_25d_view {
                    "View: 2.5D SF"
                } else {
                    "View: Space"
                }
                .to_string(),
                is_25d_view,
                move |_| {
                    let mut st = s_jump.lock().unwrap();
                    if st.orbit.range_m > 100_000.0 {
                        st.orbit.lat_deg = 37.7749;
                        st.orbit.lon_deg = -122.4194;
                        st.orbit.tilt_deg = 48.0;
                        st.orbit.range_m = 9_500.0;
                        st.orbit.target_lat_deg = 37.7749;
                        st.orbit.target_lon_deg = -122.4194;
                        st.orbit.target_tilt_deg = 48.0;
                        st.orbit.target_range_m = 9_500.0;
                    } else {
                        st.orbit.lat_deg = 22.0;
                        st.orbit.lon_deg = 12.0;
                        st.orbit.tilt_deg = 42.0;
                        st.orbit.range_m = 13_200_000.0;
                        st.orbit.target_lat_deg = 22.0;
                        st.orbit.target_lon_deg = 12.0;
                        st.orbit.target_tilt_deg = 42.0;
                        st.orbit.target_range_m = 13_200_000.0;
                    }
                    let cam = st.orbit.to_camera();
                    let vw = st.orbit.viewport_w;
                    let vh = st.orbit.viewport_h;
                    st.viewport.update_camera(&cam, vw, vh);
                },
            );

            let s_dn = Arc::clone(&state_view);
            let btn_dn = control_btn("Day/Night".to_string(), sp.day_night_lighting, move |_| {
                let mut st = s_dn.lock().unwrap();
                let mut next = st.viewport.space_config().clone();
                next.day_night_lighting = !next.day_night_lighting;
                st.viewport.set_space_config(next);
            });

            let s_city = Arc::clone(&state_view);
            let btn_city = control_btn("City Lights".to_string(), sp.city_lights, move |_| {
                let mut st = s_city.lock().unwrap();
                let mut next = st.viewport.space_config().clone();
                next.city_lights = !next.city_lights;
                st.viewport.set_space_config(next);
            });

            let s_sun = Arc::clone(&state_view);
            let sun_label = match sun_mode {
                SunDemoMode::SystemClock => "Sun: OS Clock",
                SunDemoMode::OrbitalCycle => "Sun: Cycle",
                SunDemoMode::FixedSunset => "Sun: Sunset",
            };
            let btn_sun = control_btn(
                sun_label.to_string(),
                sun_mode != SunDemoMode::SystemClock,
                move |_| {
                    let mut st = s_sun.lock().unwrap();
                    let mut next = st.viewport.space_config().clone();
                    st.orbit.sun_demo_mode = match st.orbit.sun_demo_mode {
                        SunDemoMode::SystemClock => {
                            next.sun_direction = SunDirectionMode::SimulationTime;
                            SunDemoMode::OrbitalCycle
                        }
                        SunDemoMode::OrbitalCycle => {
                            next.sun_direction = SunDirectionMode::SubsolarPoint {
                                lat_deg: 20.0,
                                lon_deg: st.orbit.lon_deg - 52.0,
                            };
                            SunDemoMode::FixedSunset
                        }
                        SunDemoMode::FixedSunset => {
                            next.sun_direction = SunDirectionMode::SystemClock;
                            SunDemoMode::SystemClock
                        }
                    };
                    st.viewport.set_space_config(next);
                },
            );

            let s_space = Arc::clone(&state_view);
            let btn_stars = control_btn("Stars/Moon".to_string(), sp.stars_visible, move |_| {
                let mut st = s_space.lock().unwrap();
                let mut next = st.viewport.space_config().clone();
                let on = !next.stars_visible;
                next.stars_visible = on;
                next.sun_visible = on;
                next.moon_visible = on;
                st.viewport.set_space_config(next);
            });

            let s_sats = Arc::clone(&state_view);
            let btn_sats =
                control_btn("Satellites".to_string(), sp.satellites_visible, move |_| {
                    let mut st = s_sats.lock().unwrap();
                    let mut next = st.viewport.space_config().clone();
                    next.satellites_visible = !next.satellites_visible;
                    st.viewport.set_space_config(next);
                });

            let s_elev = Arc::clone(&state_view);
            let btn_elev = control_btn("Heightmap".to_string(), elev_on, move |_| {
                let mut st = s_elev.lock().unwrap();
                let next = !st.viewport.elevation_enabled();
                st.viewport.set_elevation_enabled(next);
            });

            let s_gcloud = Arc::clone(&state_view);
            let btn_gcloud = control_btn("3D Clouds".to_string(), w.global_cloud_map, move |_| {
                let mut st = s_gcloud.lock().unwrap();
                let mut next = st.viewport.weather_config().clone();
                next.global_cloud_map = !next.global_cloud_map;
                st.viewport.set_weather_config(next);
            });

            let s_wclouds = Arc::clone(&state_view);
            let btn_wclouds = control_btn("2.5D Clouds".to_string(), w.clouds_25d, move |_| {
                let mut st = s_wclouds.lock().unwrap();
                let mut next = st.viewport.weather_config().clone();
                next.clouds_25d = !next.clouds_25d;
                st.viewport.set_weather_config(next);
            });

            let s_rain = Arc::clone(&state_view);
            let btn_rain = control_btn("Rain".to_string(), w.rain, move |_| {
                let mut st = s_rain.lock().unwrap();
                let mut next = st.viewport.weather_config().clone();
                next.rain = !next.rain;
                st.viewport.set_weather_config(next);
            });

            let s_snow = Arc::clone(&state_view);
            let btn_snow = control_btn("Snow".to_string(), w.snow, move |_| {
                let mut st = s_snow.lock().unwrap();
                let mut next = st.viewport.weather_config().clone();
                next.snow = !next.snow;
                st.viewport.set_weather_config(next);
            });

            let canvas = Canvas3D::builder()
                .retained(retained_for_view.clone())
                .class("absolute inset-0 w-full h-full z-0")
                .build();

            rsx! {
                <div class="relative w-full h-full bg-black overflow-hidden">
                    {canvas}
                    <div
                        data-earth-controls="true"
                        class="absolute top-3 right-3 z-20 flex flex-wrap justify-end gap-1.5 max-w-[420px] p-2 rounded-xl bg-neutral-950/75 border border-neutral-800/80 backdrop-blur-md shadow-2xl"
                    >
                        {btn_mode}
                        {btn_jump}
                        {btn_dn}
                        {btn_city}
                        {btn_sun}
                        {btn_stars}
                        {btn_sats}
                        {btn_elev}
                        {btn_gcloud}
                        {btn_wclouds}
                        {btn_rain}
                        {btn_snow}
                    </div>
                </div>
            }
        },
        config,
        move |ev, rt| {
            let mut st = state_events.lock().unwrap();
            let rw = rt.viewport_width.max(1.0) as f64;
            let rh = rt.viewport_height.max(1.0) as f64;
            if (rw - st.orbit.viewport_w).abs() > 0.5 || (rh - st.orbit.viewport_h).abs() > 0.5 {
                st.orbit.viewport_w = rw;
                st.orbit.viewport_h = rh;
                let cam = st.orbit.to_camera();
                st.viewport.update_camera(&cam, rw, rh);
            }

            match ev {
                PlatformEvent::MouseDown { button, x, y } => {
                    st.orbit.last_pos = Some((*x, *y));
                    if !is_over_controls(rt, *x, *y) {
                        match button {
                            MouseButton::Left => st.orbit.left_down = true,
                            MouseButton::Right | MouseButton::Middle => st.orbit.right_down = true,
                            MouseButton::Other(_) => {}
                        }
                    }
                }
                PlatformEvent::MouseUp { button, .. } => match button {
                    MouseButton::Left => st.orbit.left_down = false,
                    MouseButton::Right | MouseButton::Middle => st.orbit.right_down = false,
                    MouseButton::Other(_) => {}
                },
                PlatformEvent::CursorLeft => {
                    st.orbit.left_down = false;
                    st.orbit.right_down = false;
                }
                PlatformEvent::MouseMove { x, y } => {
                    if let Some((lx, ly)) = st.orbit.last_pos {
                        let dx = (*x - lx) as f64;
                        let dy = (*y - ly) as f64;
                        if st.orbit.left_down {
                            let meters_per_px =
                                (st.orbit.target_range_m * 1.05) / st.orbit.viewport_h.max(1.0);
                            let deg_per_m = 180.0 / (PI * EARTH_RADIUS_M);
                            let cos_lat = st.orbit.target_lat_deg.to_radians().cos().abs().max(0.2);
                            let h_rad = st.orbit.target_heading_deg.to_radians();
                            let north_m = dy * h_rad.cos() - dx * h_rad.sin();
                            let east_m = -dx * h_rad.cos() - dy * h_rad.sin();
                            st.orbit.target_lat_deg = (st.orbit.target_lat_deg
                                + north_m * meters_per_px * deg_per_m)
                                .clamp(-85.0, 85.0);
                            let next_lon = st.orbit.target_lon_deg
                                + east_m * meters_per_px * deg_per_m / cos_lat;
                            st.orbit.target_lon_deg = wrap_lon_delta_deg(next_lon);
                            st.orbit.snap_targets();
                            let cam = st.orbit.to_camera();
                            let vw = st.orbit.viewport_w;
                            let vh = st.orbit.viewport_h;
                            st.viewport.update_camera(&cam, vw, vh);
                        } else if st.orbit.right_down {
                            st.orbit.target_heading_deg =
                                (st.orbit.target_heading_deg + dx * 0.25).rem_euclid(360.0);
                            st.orbit.target_tilt_deg =
                                (st.orbit.target_tilt_deg + dy * 0.25).clamp(0.0, 72.0);
                            st.orbit.snap_targets();
                            let cam = st.orbit.to_camera();
                            let vw = st.orbit.viewport_w;
                            let vh = st.orbit.viewport_h;
                            st.viewport.update_camera(&cam, vw, vh);
                        }
                    }
                    st.orbit.last_pos = Some((*x, *y));
                }
                PlatformEvent::Scroll { delta_y, .. } => {
                    let (mx, my) = rt.last_mouse_pos;
                    if !is_over_controls(rt, mx, my) {
                        // Normalize GTK 110px wheel notch vs +/-1 wheel click units; scroll up (<0) zooms IN.
                        let raw_notches = if delta_y.abs() > 8.0 {
                            (*delta_y as f64) / 110.0
                        } else {
                            *delta_y as f64
                        };
                        let notches = raw_notches.clamp(-3.0, 3.0);
                        let zoom_factor = 1.075f64.powf(notches);
                        st.orbit.target_range_m =
                            (st.orbit.target_range_m * zoom_factor).clamp(1_200.0, 18_000_000.0);
                    }
                }
                _ => {}
            }
            false
        },
        Some(Duration::from_millis(16)),
        move |info, _rt| {
            let mut st = state_frame.lock().unwrap();
            let dt = info.dt.as_secs_f64().clamp(0.0, 0.05);

            if tile_ready_frame.swap(false, Ordering::AcqRel) {
                let cam = st.orbit.to_camera();
                let vw = st.orbit.viewport_w;
                let vh = st.orbit.viewport_h;
                st.viewport.update_camera(&cam, vw, vh);
            }

            if st.orbit.step_damping(dt) {
                let cam = st.orbit.to_camera();
                let vw = st.orbit.viewport_w;
                let vh = st.orbit.viewport_h;
                st.viewport.update_camera(&cam, vw, vh);
            }

            if st.viewport.is_transitioning() {
                st.viewport.step_transition(dt);
            }

            if st.orbit.sun_demo_mode == SunDemoMode::OrbitalCycle {
                let next_t = st.viewport.input().sim_time_unix_s + dt * 3_600.0;
                st.viewport.set_sim_time_unix_s(next_t);
            }

            let tick_satellites = st.orbit.sun_demo_mode != SunDemoMode::FixedSunset;
            let _ = st.viewport.step_dynamic_overlays(dt, tick_satellites);

            let is_25d_ui = st.orbit.range_m < 100_000.0;
            if is_25d_ui != st.last_is_25d_ui {
                st.last_is_25d_ui = is_25d_ui;
                FrameOutcome::Dirty
            } else {
                FrameOutcome::Unchanged
            }
        },
    )
}
