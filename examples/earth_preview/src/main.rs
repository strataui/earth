//! Interactive standalone preview window rendering ONLY the Strata Earth `Canvas3D` component
//! using `EarthDefaultTileSource` (embedded compressed NASA Blue Marble `z=0..2` base world map +
//! live ArcGIS satellite & AWS Terrarium DEM streaming), seamless 3D <-> 2.5D ground morphing,
//! programmatic `2`/`3`/`M` 3D <-> 2D morphing, and direct zero-readback `wgpu` swapchain output.

use std::f64::consts::PI;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use strata_earth::{
    EarthCamera, EarthDefaultTileSource, EarthInput, EarthViewMode, EarthViewport, Geodetic,
    LayerToggles, SurfaceMode, TileProjection,
};
use strata_layout::Rect;
use strata_platform::{MouseButton, PlatformEvent, PresentMode, WindowConfig};
use strata_render::scene3d::Canvas3D;

const EARTH_RADIUS_M: f64 = 6_378_137.0;

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

    fn to_input(&self, view_mode: EarthViewMode) -> EarthInput {
        EarthInput {
            camera: self.to_camera(),
            viewport: Rect::new(0.0, 0.0, self.viewport_w as f32, self.viewport_h as f32),
            view_mode,
            seamless_25d: true,
            tile_projection: TileProjection::WebMercator,
            daylight: 0.94,
            cloud_cover: 0.0,
            surface_mode: SurfaceMode::QuadtreeLod,
            layers: LayerToggles {
                terrain: true,
                buildings: false,
                graticule: false,
                labels: false,
                clouds: false,
                celestial: true,
            },
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

    let orbit = OrbitState {
        lat_deg: 28.0,
        lon_deg: 12.0,
        heading_deg: 6.0,
        tilt_deg: 12.0,
        range_m: 10_200_000.0,
        left_down: false,
        right_down: false,
        last_pos: None,
        viewport_w: initial_w as f64,
        viewport_h: initial_h as f64,
    };

    let input = orbit.to_input(EarthViewMode::Mode3D);

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
                    _ => {}
                },
                PlatformEvent::AnimationTick => {
                    let now = Instant::now();
                    let dt = (now - *last_tick).as_secs_f64().clamp(1.0 / 240.0, 0.05);
                    *last_tick = now;

                    if tile_ready.swap(false, Ordering::AcqRel) {
                        let next_in = orb.to_input(vp.view_mode());
                        vp.update_input(&next_in);
                    }
                    if vp.is_transitioning() {
                        vp.step_transition(dt);
                        strata_platform::post_platform_event(PlatformEvent::AnimationTick);
                        dirty = true;
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
