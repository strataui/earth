//! Interactive standalone preview window rendering ONLY the Strata Earth `Canvas3D` component
//! with real-time OS clock solar/lunar ephemeris, BSC5 deep-space starfield, limb-darkened Sun &
//! NASA LRO Moon, land-masked deterministic city lights, live global satellite cloud map in 3D,
//! seamless 3D <-> 2.5D terrain & lighting transition, and CC0 3D cloud/rain/snow models in 2.5D.

use std::f64::consts::PI;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use strata_earth::{
    EarthCamera, EarthDefaultTileSource, EarthInput, EarthViewMode, EarthViewport, Geodetic,
    LayerToggles, SpaceEnvironmentConfig, SunDirectionMode, SurfaceMode, TileProjection,
    WeatherConfig,
};
use strata_layout::Rect;
use strata_platform::{MouseButton, PlatformEvent, PresentMode, WindowConfig};
use strata_render::scene3d::Canvas3D;

const EARTH_RADIUS_M: f64 = 6_378_137.0;

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
    left_down: bool,
    right_down: bool,
    last_pos: Option<(f32, f32)>,
    viewport_w: f64,
    viewport_h: f64,
    sun_demo_mode: SunDemoMode,
}

impl OrbitState {
    fn to_camera(&self) -> EarthCamera {
        let tilt_rad = self.tilt_deg.to_radians();
        let head_rad = self.heading_deg.to_radians();
        let horiz = self.range_m * tilt_rad.sin();
        let up = self.range_m * tilt_rad.cos();
        EarthCamera {
            focus: Geodetic::new(self.lat_deg, self.lon_deg, 0.0),
            aim_offset_m: [0.0, 0.0, 0.0],
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
            space,
            weather,
            tile_grid: 9,
            max_tile_level: 16,
            sse_threshold: 4.5,
            ..EarthInput::default()
        }
    }
}

fn main() -> Result<(), String> {
    let initial_w = 1280u32;
    let initial_h = 800u32;

    let args: Vec<String> = std::env::args().collect();
    let is_25d_rain = args.iter().any(|a| a == "--25d-rain");
    let is_25d_snow = args.iter().any(|a| a == "--25d-snow");
    let has_spin = args.iter().any(|a| a == "--sun-cycle");

    let (lat_deg, lon_deg, tilt_deg, range_m, rain_on, snow_on) = if is_25d_rain || is_25d_snow {
        (37.7749, -122.4194, 48.0, 9_500.0, is_25d_rain, is_25d_snow)
    } else {
        (26.0, 8.0, 22.0, 10_200_000.0, false, false)
    };

    let orbit = OrbitState {
        lat_deg,
        lon_deg,
        heading_deg: 6.0,
        tilt_deg,
        range_m,
        left_down: false,
        right_down: false,
        last_pos: None,
        viewport_w: initial_w as f64,
        viewport_h: initial_h as f64,
        sun_demo_mode: if has_spin {
            SunDemoMode::OrbitalCycle
        } else {
            SunDemoMode::SystemClock
        },
    };

    let space = SpaceEnvironmentConfig::default().with_system_clock();
    let weather = WeatherConfig {
        rain: rain_on,
        snow: snow_on,
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
        strata_platform::post_platform_event(PlatformEvent::AnimationTick);
    })));

    let wakeup = Arc::new(|| {
        strata_platform::post_platform_event(PlatformEvent::AnimationTick);
    });
    let viewport = EarthViewport::with_wakeup(&input, Box::new(source), Some(wakeup));
    viewport.sync_now();

    eprintln!(
        "Strata Earth controls: Drag=Pan | Right-Drag=Orbit/Tilt | Scroll=Zoom(3D<->2.5D) | 2/3/M=2D/3D | Z=Space/2.5D Jump | L=Day/Night | C=CityLights | T=SunMode | G=CloudMap | W=2.5D Clouds | R=Rain | S=Snow | B=Space/Stars/Moon"
    );

    let retained = viewport.retained_scene();
    let last_cam_gen = retained.camera_generation();
    let last_geom_gen = retained.geometry_generation();
    let state = Arc::new(Mutex::new((
        viewport,
        orbit,
        Instant::now(),
        last_cam_gen,
        last_geom_gen,
    )));

    let config = WindowConfig {
        title: "Strata Earth".to_string(),
        width: initial_w,
        height: initial_h,
        present_mode: PresentMode::Immediate,
        resizable: true,
        ..WindowConfig::default()
    };

    let retained_for_view = retained;
    let state_events = Arc::clone(&state);

    if has_spin || rain_on || snow_on {
        strata_platform::post_platform_event(PlatformEvent::AnimationTick);
    }

    strata::launch_with_events(
        move || {
            Canvas3D::builder()
                .retained(retained_for_view.clone())
                .class("w-full h-full")
                .build()
        },
        config,
        move |ev, rt| {
            let mut guard = state_events.lock().unwrap();
            let (vp, orb, last_tick, last_cg, last_gg) = &mut *guard;
            let mut dirty = false;

            let rw = rt.viewport_width.max(1.0) as f64;
            let rh = rt.viewport_height.max(1.0) as f64;
            let resized = (rw - orb.viewport_w).abs() > 0.5 || (rh - orb.viewport_h).abs() > 0.5;
            if resized {
                orb.viewport_w = rw;
                orb.viewport_h = rh;
                let cam = orb.to_camera();
                vp.update_camera(&cam, orb.viewport_w, orb.viewport_h);
                dirty = true;
            }

            match ev {
                PlatformEvent::MouseDown { button, x, y } => {
                    orb.last_pos = Some((*x, *y));
                    match button {
                        MouseButton::Left => orb.left_down = true,
                        MouseButton::Right | MouseButton::Middle => orb.right_down = true,
                        MouseButton::Other(_) => {}
                    }
                }
                PlatformEvent::MouseUp { button, .. } => match button {
                    MouseButton::Left => orb.left_down = false,
                    MouseButton::Right | MouseButton::Middle => orb.right_down = false,
                    MouseButton::Other(_) => {}
                },
                PlatformEvent::MouseMove { x, y } => {
                    if let Some((lx, ly)) = orb.last_pos {
                        let dx = (*x - lx) as f64;
                        let dy = (*y - ly) as f64;
                        if orb.left_down {
                            let meters_per_px = (orb.range_m * 1.15) / orb.viewport_h.max(1.0);
                            let deg_per_m = 180.0 / (PI * EARTH_RADIUS_M);
                            let cos_lat = orb.lat_deg.to_radians().cos().abs().max(0.2);
                            let h_rad = orb.heading_deg.to_radians();
                            let north_m = dy * h_rad.cos() - dx * h_rad.sin();
                            let east_m = -dx * h_rad.cos() - dy * h_rad.sin();
                            orb.lat_deg = (orb.lat_deg + north_m * meters_per_px * deg_per_m)
                                .clamp(-84.0, 84.0);
                            let next_lon =
                                orb.lon_deg + east_m * meters_per_px * deg_per_m / cos_lat;
                            orb.lon_deg = ((next_lon + 180.0).rem_euclid(360.0)) - 180.0;
                            let cam = orb.to_camera();
                            vp.update_camera(&cam, orb.viewport_w, orb.viewport_h);
                            dirty = true;
                        } else if orb.right_down {
                            orb.heading_deg = (orb.heading_deg + dx * 0.25).rem_euclid(360.0);
                            orb.tilt_deg = (orb.tilt_deg + dy * 0.25).clamp(0.0, 72.0);
                            let cam = orb.to_camera();
                            vp.update_camera(&cam, orb.viewport_w, orb.viewport_h);
                            dirty = true;
                        }
                    }
                    orb.last_pos = Some((*x, *y));
                }
                PlatformEvent::Scroll { delta_y, .. } => {
                    let zoom_factor = 0.86f64.powf((*delta_y as f64).clamp(-5.0, 5.0));
                    orb.range_m = (orb.range_m * zoom_factor).clamp(1_200.0, 18_000_000.0);
                    let cam = orb.to_camera();
                    vp.update_camera(&cam, orb.viewport_w, orb.viewport_h);
                    if vp.has_active_weather_animation() {
                        *last_tick = Instant::now();
                        strata_platform::post_platform_event(PlatformEvent::AnimationTick);
                    }
                    dirty = true;
                }
                PlatformEvent::ReceivedCharacter(c) => match c {
                    '2' => {
                        vp.set_view_mode(EarthViewMode::Mode2D);
                        *last_tick = Instant::now();
                        strata_platform::post_platform_event(PlatformEvent::AnimationTick);
                        dirty = true;
                    }
                    '3' => {
                        vp.set_view_mode(EarthViewMode::Mode3D);
                        *last_tick = Instant::now();
                        strata_platform::post_platform_event(PlatformEvent::AnimationTick);
                        dirty = true;
                    }
                    'm' | 'M' | ' ' => {
                        let next = match vp.view_mode() {
                            EarthViewMode::Mode3D => EarthViewMode::Mode2D,
                            EarthViewMode::Mode2D => EarthViewMode::Mode3D,
                        };
                        vp.set_view_mode(next);
                        *last_tick = Instant::now();
                        strata_platform::post_platform_event(PlatformEvent::AnimationTick);
                        dirty = true;
                    }
                    'z' | 'Z' => {
                        if orb.range_m > 100_000.0 {
                            orb.lat_deg = 37.7749;
                            orb.lon_deg = -122.4194;
                            orb.tilt_deg = 48.0;
                            orb.range_m = 9_500.0;
                        } else {
                            orb.lat_deg = 26.0;
                            orb.lon_deg = 8.0;
                            orb.tilt_deg = 22.0;
                            orb.range_m = 10_200_000.0;
                        }
                        let cam = orb.to_camera();
                        vp.update_camera(&cam, orb.viewport_w, orb.viewport_h);
                        if vp.has_active_weather_animation() {
                            *last_tick = Instant::now();
                            strata_platform::post_platform_event(PlatformEvent::AnimationTick);
                        }
                        dirty = true;
                    }
                    'l' | 'L' => {
                        let mut sp = vp.space_config().clone();
                        sp.day_night_lighting = !sp.day_night_lighting;
                        vp.set_space_config(sp);
                        dirty = true;
                    }
                    'c' | 'C' => {
                        let mut sp = vp.space_config().clone();
                        sp.city_lights = !sp.city_lights;
                        vp.set_space_config(sp);
                        dirty = true;
                    }
                    'b' | 'B' => {
                        let mut sp = vp.space_config().clone();
                        let on = !sp.stars_visible;
                        sp.stars_visible = on;
                        sp.sun_visible = on;
                        sp.moon_visible = on;
                        vp.set_space_config(sp);
                        dirty = true;
                    }
                    't' | 'T' => {
                        let mut sp = vp.space_config().clone();
                        orb.sun_demo_mode = match orb.sun_demo_mode {
                            SunDemoMode::SystemClock => {
                                sp.sun_direction = SunDirectionMode::SimulationTime;
                                *last_tick = Instant::now();
                                strata_platform::post_platform_event(PlatformEvent::AnimationTick);
                                SunDemoMode::OrbitalCycle
                            }
                            SunDemoMode::OrbitalCycle => {
                                sp.sun_direction = SunDirectionMode::SubsolarPoint {
                                    lat_deg: 20.0,
                                    lon_deg: orb.lon_deg - 52.0,
                                };
                                SunDemoMode::FixedSunset
                            }
                            SunDemoMode::FixedSunset => {
                                sp.sun_direction = SunDirectionMode::SystemClock;
                                SunDemoMode::SystemClock
                            }
                        };
                        vp.set_space_config(sp);
                        dirty = true;
                    }
                    'g' | 'G' => {
                        let mut w = vp.weather_config().clone();
                        w.global_cloud_map = !w.global_cloud_map;
                        vp.set_weather_config(w);
                        dirty = true;
                    }
                    'w' | 'W' => {
                        let mut w = vp.weather_config().clone();
                        w.clouds_25d = !w.clouds_25d;
                        vp.set_weather_config(w);
                        if vp.has_active_weather_animation() {
                            *last_tick = Instant::now();
                            strata_platform::post_platform_event(PlatformEvent::AnimationTick);
                        }
                        dirty = true;
                    }
                    'r' | 'R' => {
                        let mut w = vp.weather_config().clone();
                        w.rain = !w.rain;
                        vp.set_weather_config(w);
                        if vp.has_active_weather_animation() {
                            *last_tick = Instant::now();
                            strata_platform::post_platform_event(PlatformEvent::AnimationTick);
                        }
                        dirty = true;
                    }
                    's' | 'S' => {
                        let mut w = vp.weather_config().clone();
                        w.snow = !w.snow;
                        vp.set_weather_config(w);
                        if vp.has_active_weather_animation() {
                            *last_tick = Instant::now();
                            strata_platform::post_platform_event(PlatformEvent::AnimationTick);
                        }
                        dirty = true;
                    }
                    _ => {}
                },
                PlatformEvent::AnimationTick => {
                    let now = Instant::now();
                    let dt = (now - *last_tick).as_secs_f64().clamp(1.0 / 240.0, 0.05);
                    *last_tick = now;

                    if tile_ready.swap(false, Ordering::AcqRel) {
                        let cam = orb.to_camera();
                        vp.update_camera(&cam, orb.viewport_w, orb.viewport_h);
                        dirty = true;
                    }

                    let mut keep_ticking = false;
                    if vp.is_transitioning() {
                        vp.step_transition(dt);
                        keep_ticking = true;
                        dirty = true;
                    }
                    if orb.sun_demo_mode == SunDemoMode::OrbitalCycle {
                        let next_t = vp.input().sim_time_unix_s + dt * 3_600.0;
                        vp.set_sim_time_unix_s(next_t);
                        keep_ticking = true;
                        dirty = true;
                    }
                    if vp.step_weather(dt) {
                        keep_ticking = true;
                        dirty = true;
                    }
                    if keep_ticking {
                        strata_platform::post_platform_event(PlatformEvent::AnimationTick);
                    }
                }
                _ => {}
            }

            let ret = vp.retained_scene();
            let cg = ret.camera_generation();
            let gg = ret.geometry_generation();
            if cg != *last_cg || gg != *last_gg {
                *last_cg = cg;
                *last_gg = gg;
                dirty = true;
            }
            dirty
        },
    )
}
