// Interactive Earth preview embedding `strata_earth::EarthView` alongside floating
// top-right Strata HUD controls, external keybind handling, realistic 2.5D atmosphere,
// unified weather API controls, GPU ocean waves/coastal surf, and live raycast 3D placement.

use std::cell::RefCell;
use std::rc::Rc;

use strata::platform::KeyCode;
use strata::prelude::*;
use strata::WindowConfig;
use strata::{launch_with_frame, AppFrameInfo, FrameOutcome};
use strata_earth::{
    AtmosphereSample, BaseWeatherConfig, Color, EarthDefaultTileSource, EarthGeometry, EarthInput,
    EarthMaterial, EarthObject, EarthOrbitController, EarthView, EarthViewMode, GeoPosition,
    LocalAtmosphereConfig, LocalFrame, OceanWavesConfig, PrecipitationConfig, PrecipitationMode,
    SpaceEnvironmentConfig, SunDirectionMode, TileProviderPreset, WeatherConfig, WeatherSnapshot,
    WindVector, LOCAL_TRANSITION_RANGE_M,
};

/// Example-local continuous exponential zoom rate (natural-log range units per second) for `K` / `L`.
const PREVIEW_KEY_ZOOM_LN_PER_SEC: f64 = 1.95;

#[derive(Default, Clone, Copy, Debug, PartialEq, Eq)]
struct PreviewHeldZoom {
    zoom_in: bool,
    zoom_out: bool,
    fast: bool,
}

impl PreviewHeldZoom {
    #[inline]
    fn is_active(self) -> bool {
        self.zoom_in != self.zoom_out
    }

    #[inline]
    fn step_earth(self, earth: &EarthView, dt_s: f64) {
        let dir = (self.zoom_out as i32 - self.zoom_in as i32) as f64;
        if dir != 0.0 && dt_s > 0.0 {
            let rate = if self.fast {
                PREVIEW_KEY_ZOOM_LN_PER_SEC * 1.6
            } else {
                PREVIEW_KEY_ZOOM_LN_PER_SEC
            };
            earth.zoom_by_ln(dir * rate * dt_s.clamp(0.0, 0.1));
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum SunDemoMode {
    SystemClock,
    LocalNoon,
    GoldenSunset,
    StarryNight,
    OrbitalCycle,
}

impl SunDemoMode {
    fn next(self) -> Self {
        match self {
            Self::SystemClock => Self::LocalNoon,
            Self::LocalNoon => Self::GoldenSunset,
            Self::GoldenSunset => Self::StarryNight,
            Self::StarryNight => Self::OrbitalCycle,
            Self::OrbitalCycle => Self::SystemClock,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::SystemClock => "Sun: OS Clock",
            Self::LocalNoon => "Sun: Noon",
            Self::GoldenSunset => "Sun: Sunset",
            Self::StarryNight => "Sun: Night",
            Self::OrbitalCycle => "Sun: Cycle",
        }
    }
}

fn dir_from_local_az_el(lat_deg: f64, lon_deg: f64, az_deg: f64, el_deg: f64) -> [f32; 3] {
    let f = LocalFrame::at(lat_deg, lon_deg);
    let el = el_deg.to_radians();
    let az = az_deg.to_radians();
    let cos_el = el.cos();
    let v = f
        .east
        .scale(cos_el * az.sin())
        .add(f.north.scale(cos_el * az.cos()))
        .add(f.up.scale(el.sin()))
        .normalize();
    [v.x as f32, v.y as f32, v.z as f32]
}

fn apply_sun_demo_mode(
    mode: SunDemoMode,
    sim_s: f64,
    lat_deg: f64,
    lon_deg: f64,
    heading_deg: f64,
    sp: &mut SpaceEnvironmentConfig,
) {
    match mode {
        SunDemoMode::SystemClock => {
            sp.sun_direction = SunDirectionMode::SystemClock;
            sp.custom_moon_dir = None;
        }
        SunDemoMode::LocalNoon => {
            sp.sun_direction =
                SunDirectionMode::CustomEcef(dir_from_local_az_el(lat_deg, lon_deg, heading_deg + 115.0, 50.0));
            sp.custom_moon_dir = None;
        }
        SunDemoMode::GoldenSunset => {
            // Places low golden Sun +8.5° above horizon slightly ahead-left of camera view for an
            // unobstructed visible Sun disc, Mie aureole, long mountain shadows, and golden ocean glitter!
            sp.sun_direction =
                SunDirectionMode::CustomEcef(dir_from_local_az_el(lat_deg, lon_deg, heading_deg - 14.0, 8.5));
            sp.custom_moon_dir = None;
        }
        SunDemoMode::StarryNight => {
            // Places Sun -28° below local horizon and Moon +8.0° ahead-left for 2.5D stars + moonlight!
            sp.sun_direction =
                SunDirectionMode::CustomEcef(dir_from_local_az_el(lat_deg, lon_deg, heading_deg + 180.0, -28.0));
            sp.custom_moon_dir = Some(dir_from_local_az_el(lat_deg, lon_deg, heading_deg - 14.0, 8.0));
        }
        SunDemoMode::OrbitalCycle => {
            let _ = sim_s;
            sp.sun_direction = SunDirectionMode::SimulationTime;
            sp.custom_moon_dir = None;
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum TempPreset {
    Warm12C,
    Cool2C,
    FreezeNeg6C,
    LiveOpenMeteo,
}

impl TempPreset {
    fn next(self) -> Self {
        match self {
            Self::Warm12C => Self::Cool2C,
            Self::Cool2C => Self::FreezeNeg6C,
            Self::FreezeNeg6C => Self::LiveOpenMeteo,
            Self::LiveOpenMeteo => Self::Warm12C,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Warm12C => "Temp: +12°C",
            Self::Cool2C => "Temp: +2°C",
            Self::FreezeNeg6C => "Temp: -6°C",
            Self::LiveOpenMeteo => "Temp: Live",
        }
    }

    fn apply_to(self, precip: &mut PrecipitationConfig, base: &mut BaseWeatherConfig) {
        match self {
            Self::Warm12C => {
                precip.use_live_temperature = false;
                precip.atmosphere.base_temp_c = 12.0;
                base.use_live_api = false;
                base.atmosphere.base_temp_c = 12.0;
            }
            Self::Cool2C => {
                precip.use_live_temperature = false;
                precip.atmosphere.base_temp_c = 2.0;
                base.use_live_api = false;
                base.atmosphere.base_temp_c = 2.0;
            }
            Self::FreezeNeg6C => {
                precip.use_live_temperature = false;
                precip.atmosphere.base_temp_c = -6.0;
                base.use_live_api = false;
                base.atmosphere.base_temp_c = -6.0;
            }
            Self::LiveOpenMeteo => {
                precip.use_live_temperature = true;
                base.use_live_api = true;
            }
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum WindPreset {
    BreezeEast,
    FreshSW,
    GaleNorth,
    LiveApi,
}

impl WindPreset {
    fn next(self) -> Self {
        match self {
            Self::BreezeEast => Self::FreshSW,
            Self::FreshSW => Self::GaleNorth,
            Self::GaleNorth => Self::LiveApi,
            Self::LiveApi => Self::BreezeEast,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::BreezeEast => "Wind: 7m/s E",
            Self::FreshSW => "Wind: 15m/s SW",
            Self::GaleNorth => "Wind: 24m/s N",
            Self::LiveApi => "Wind: Live",
        }
    }

    fn vector(self) -> WindVector {
        match self {
            Self::BreezeEast => WindVector::new(7.0, 90.0),
            Self::FreshSW => WindVector::new(15.0, 225.0),
            Self::GaleNorth => WindVector::new(24.0, 0.0),
            Self::LiveApi => WindVector::new(11.0, 70.0),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum OceanPreset {
    Off,
    OpenSea,
    CoastalSurf,
}

impl OceanPreset {
    fn next(self) -> Self {
        match self {
            Self::Off => Self::OpenSea,
            Self::OpenSea => Self::CoastalSurf,
            Self::CoastalSurf => Self::Off,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Off => "Ocean: Off",
            Self::OpenSea => "Ocean: Sea",
            Self::CoastalSurf => "Ocean: Coastal",
        }
    }
}

fn format_atmo_status(
    cfg: &PrecipitationConfig,
    ocean: &OceanWavesConfig,
    eye_alt_m: f64,
    sample: Option<AtmosphereSample>,
    weather_snap: Option<WeatherSnapshot>,
) -> Option<String> {
    let active_precip = cfg.enabled
        || weather_snap.is_some_and(|w| w.precipitation_intensity > 0.02);
    if !active_precip && !ocean.enabled {
        return None;
    }

    let wind = weather_snap
        .map(|w| w.wind)
        .unwrap_or_else(|| WindVector::from_enu_mps(cfg.resolved_wind_enu()));
    let wind_str = format!("{:.0}m/s {}", wind.speed_mps, wind.compass_label());

    if !active_precip {
        let mode_s = if ocean.coastal_shoaling {
            "Coastal Surf + SDF"
        } else {
            "Open Sea Swells"
        };
        return Some(format!(
            "Ocean {mode_s} · Wind {wind_str} · Alt {:.0}m",
            eye_alt_m.max(0.0)
        ));
    }

    let s = match sample.or_else(|| weather_snap.map(|w| w.atmosphere)) {
        Some(s) => s,
        None => {
            return Some(format!(
                "Precip Active · Wind {wind_str} · Alt {:.0}m",
                eye_alt_m.max(0.0)
            ));
        }
    };
    let src = if s.source_live { "Live" } else { "ISA" };
    if cfg.respect_cloud_ceiling && eye_alt_m > s.cloud_ceiling_msl_m {
        Some(format!(
            "Above Clouds · Alt {:.0}m > {:.0}m · T {:+.1}°C · Wind {wind_str} ({src})",
            eye_alt_m, s.cloud_ceiling_msl_m, s.temperature_c
        ))
    } else {
        let phase_txt = match &cfg.mode {
            PrecipitationMode::Physical => s.phase.label(),
            PrecipitationMode::Rain => "Rain (Forced)",
            PrecipitationMode::Snow => "Snow (Forced)",
            PrecipitationMode::CustomLayers(_) => "Custom 2D",
        };
        let dark_pct = weather_snap
            .map(|w| (w.storm_darkness * 100.0).round() as u32)
            .unwrap_or_else(|| ((cfg.intensity.clamp(0.0, 1.2) / 1.2) * 76.0).round() as u32);
        Some(format!(
            "{phase_txt} · Alt {:.0}m · T {:+.1}°C (Tw {:+.1}°C) · Wind {wind_str} · Sky Dark {dark_pct}% · Ceil {:.0}m ({src})",
            eye_alt_m.max(0.0),
            s.temperature_c,
            s.wet_bulb_c,
            s.cloud_ceiling_msl_m,
        ))
    }
}

fn control_btn<F>(label: String, active: bool, on_click: F) -> View
where
    F: Fn(strata_dom::Event) + 'static,
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
            <span>{label}</span>
        </button>
    }
}

fn wire_preview_zoom_keys(earth: &EarthView, held: &Rc<RefCell<PreviewHeldZoom>>) {
    let held_down = Rc::clone(held);
    earth.on_key_down(move |ev| {
        if ev.modifiers.ctrl || ev.modifiers.alt || ev.modifiers.meta {
            return;
        }
        let mut h = held_down.borrow_mut();
        h.fast = ev.modifiers.shift;
        match ev.key {
            KeyCode::KeyK => h.zoom_in = true,
            KeyCode::KeyL => h.zoom_out = true,
            _ => {}
        }
    });

    let held_up = Rc::clone(held);
    earth.on_key_up(move |ev| {
        let mut h = held_up.borrow_mut();
        h.fast = ev.modifiers.shift && !ev.key_str.starts_with("Shift");
        match ev.key {
            KeyCode::KeyK => h.zoom_in = false,
            KeyCode::KeyL => h.zoom_out = false,
            _ => {}
        }
    });
}

fn main() -> Result<(), String> {
    let cache_dir = std::env::temp_dir().join("strata_earth_preview_cache");
    let source = if std::env::var_os("STRATA_EARTH_OFFLINE").is_some() {
        EarthDefaultTileSource::offline_basemap()
    } else {
        EarthDefaultTileSource::world_satellite().with_disk_cache(&cache_dir)
    };

    let mut orbit = EarthOrbitController::new();
    orbit.jump_to(26.0, 16.0, 13_500_000.0, 0.0, 0.0);
    orbit.set_viewport_size(1440.0, 900.0);

    let (init_cam, init_up) = orbit.to_camera_and_up(false, 0.0);
    let mut input = EarthInput::globe();
    input.camera = init_cam;
    input.camera_up_ecef = init_up;
    input.viewport = strata::layout::Rect {
        x: 0.0,
        y: 0.0,
        width: 1440.0,
        height: 900.0,
    };
    input.view_mode = EarthViewMode::Mode3D;
    input.elevation_enabled = true;
    input.layers.graticule = false;
    input.space = SpaceEnvironmentConfig::default().with_system_clock();
    input.local_atmosphere = LocalAtmosphereConfig::realistic();
    input.base_weather = BaseWeatherConfig::default().enabled().with_wind(7.0, 90.0);
    input.weather = WeatherConfig::default();

    let mut initial_sun_mode = SunDemoMode::SystemClock;
    let mut initial_wind_preset = WindPreset::BreezeEast;
    let mut initial_ocean_preset = OceanPreset::Off;
    let mut initial_temp_preset = TempPreset::Warm12C;

    // Optional headless showcase scene preset via env var for deterministic visual regression capture.
    if let Ok(preset) = std::env::var("STRATA_EARTH_DEMO_PRESET") {
        match preset.as_str() {
            "sunset_alps" => {
                // Swiss Alps (Eiger / Jungfrau valley) with Golden Sunset Sun disc & dramatic mountain shadows!
                orbit.jump_to(46.5775, 7.9850, 4_200.0, 118.0, 74.0);
                initial_sun_mode = SunDemoMode::GoldenSunset;
                apply_sun_demo_mode(
                    initial_sun_mode,
                    1_735_689_600.0,
                    46.5775,
                    7.9850,
                    118.0,
                    &mut input.space,
                );
            }
            "night_sf" => {
                // San Francisco Bay at Starry Night: Yale BSC5 stars, Milky Way, Moon & city lights!
                orbit.jump_to(37.8050, -122.4300, 3_600.0, 35.0, 82.0);
                initial_sun_mode = SunDemoMode::StarryNight;
                apply_sun_demo_mode(
                    initial_sun_mode,
                    1_735_689_600.0,
                    37.8050,
                    -122.4300,
                    35.0,
                    &mut input.space,
                );
                input.base_weather = BaseWeatherConfig::default()
                    .enabled()
                    .with_wind(9.0, 75.0)
                    .with_cloud_cover(0.04);
                input.ocean_waves = OceanWavesConfig::enabled().with_wind(9.0, 75.0);
                initial_ocean_preset = OceanPreset::OpenSea;
            }
            "storm_rain_curtains" => {
                // Zoomed-out 2.5D valley storm: darkened sky & clouds + 3-shell far-field rain curtains!
                orbit.jump_to(46.5775, 7.9850, 2_400.0, 95.0, 68.0);
                initial_sun_mode = SunDemoMode::LocalNoon;
                apply_sun_demo_mode(
                    initial_sun_mode,
                    1_735_689_600.0,
                    46.5775,
                    7.9850,
                    95.0,
                    &mut input.space,
                );
                initial_wind_preset = WindPreset::FreshSW;
                input.base_weather = BaseWeatherConfig::default()
                    .enabled()
                    .with_wind(15.0, 225.0)
                    .with_precipitation(1.15)
                    .with_cloud_cover(0.92)
                    .with_storm_darkness(0.78)
                    .with_base_temp_c(14.0);
                input.precipitation = PrecipitationConfig::default()
                    .enabled()
                    .with_mode(PrecipitationMode::Rain)
                    .with_intensity(1.15)
                    .with_wind_angle(15.0, 225.0);
            }
            "snow_squall_alps" => {
                // Zoomed-out Alpine snow squall: 3-shell snowflakes + far snow curtains + darkened clouds!
                orbit.jump_to(46.5775, 7.9850, 2_200.0, 95.0, 68.0);
                initial_sun_mode = SunDemoMode::LocalNoon;
                apply_sun_demo_mode(
                    initial_sun_mode,
                    1_735_689_600.0,
                    46.5775,
                    7.9850,
                    95.0,
                    &mut input.space,
                );
                initial_temp_preset = TempPreset::FreezeNeg6C;
                initial_wind_preset = WindPreset::FreshSW;
                input.base_weather = BaseWeatherConfig::default()
                    .enabled()
                    .with_wind(12.0, 225.0)
                    .with_precipitation(1.1)
                    .with_cloud_cover(0.88)
                    .with_storm_darkness(0.66)
                    .with_base_temp_c(-6.0);
                input.precipitation = PrecipitationConfig::default()
                    .enabled()
                    .with_mode(PrecipitationMode::Snow)
                    .with_intensity(1.1)
                    .with_base_temp_c(-6.0)
                    .with_wind_angle(12.0, 225.0);
            }
            "coastal_surf_sunset" => {
                // Marin Headlands / Ocean Beach coast: Golden Sunset reflecting on 8-octave GPU swells + coastal surf!
                orbit.jump_to(37.7680, -122.5180, 1_500.0, 250.0, 68.0);
                initial_sun_mode = SunDemoMode::GoldenSunset;
                apply_sun_demo_mode(
                    initial_sun_mode,
                    1_735_689_600.0,
                    37.7680,
                    -122.5180,
                    250.0,
                    &mut input.space,
                );
                initial_wind_preset = WindPreset::FreshSW;
                initial_ocean_preset = OceanPreset::CoastalSurf;
                input.base_weather = BaseWeatherConfig::default()
                    .enabled()
                    .with_wind(14.0, 75.0)
                    .with_cloud_cover(0.18);
                input.ocean_waves = OceanWavesConfig::enabled()
                    .with_wind(14.0, 75.0)
                    .with_coastal_shoaling(true)
                    .with_shoaling_range_m(380.0)
                    .with_strength(1.25);
            }
            _ => {}
        }
        let (cam, up) = orbit.to_camera_and_up(false, 0.0);
        input.camera = cam;
        input.camera_up_ecef = up;
    }

    let earth = EarthView::with_orbit(&input, Box::new(source), orbit);
    let held_zoom = Rc::new(RefCell::new(PreviewHeldZoom::default()));
    wire_preview_zoom_keys(&earth, &held_zoom);

    let click_place_on = Signal::new(false);
    let placed_count = Signal::new(0usize);
    let last_ray_status: Signal<Option<String>> = Signal::new(None);

    // External click -> world raycast -> geodetic 3D object placement (+5.0m above active terrain/sea level).
    {
        let earth_click = earth.clone();
        let place_sig = click_place_on.clone();
        let count_sig = placed_count.clone();
        let status_sig = last_ray_status.clone();
        earth.on_click(move |ev| {
            if place_sig.untracked_get() || ev.modifiers.shift {
                ev.prevent_default_camera();
                if let Some(hit) = earth_click.raycast(ev.x, ev.y) {
                    earth_click.spawn_object(
                        EarthObject::new(
                            GeoPosition::terrain(hit.lat_deg, hit.lon_deg, 5.0),
                            EarthGeometry::beacon_pin(28.0, 7.5),
                        )
                        .with_material(EarthMaterial::standard(Color::rgb(56, 189, 248)))
                        .with_min_pixel_size(22.0),
                    );
                    let count = earth_click.object_count();
                    count_sig.set(count);
                    status_sig.set(Some(format!(
                        "Object #{count} · {:.4}°, {:.4}° · Surf {:.1}m (+5.0m Terrain)",
                        hit.lat_deg, hit.lon_deg, hit.altitude_m
                    )));
                }
            }
        });
    }

    let sun_demo_mode = Rc::new(RefCell::new(initial_sun_mode));
    let sun_cycle_sim_s = Rc::new(RefCell::new(1_735_689_600.0f64));
    let active_preset = Rc::new(RefCell::new(TileProviderPreset::ArcGisWorldImagery));
    let temp_preset = Rc::new(RefCell::new(initial_temp_preset));
    let wind_preset = Rc::new(RefCell::new(initial_wind_preset));
    let ocean_preset = Rc::new(RefCell::new(initial_ocean_preset));
    let fps_readout = Rc::new(RefCell::new((60u32, 16.7f64)));
    let atmo_readout: Rc<RefCell<Option<String>>> = Rc::new(RefCell::new(None));
    let fly_target_idx = Rc::new(RefCell::new(0usize));

    const FLY_LANDMARKS: &[(&str, f64, f64)] = &[
        ("Alps", 46.5775, 7.9850),
        ("Big Sur", 36.2704, -121.8081),
        ("Tokyo", 35.6762, 139.6503),
        ("Paris", 48.8584, 2.2945),
        ("Rio", -22.9519, -43.2105),
        ("Istanbul", 41.0082, 28.9784),
    ];

    let earth_for_view = earth.clone();
    let mode_for_view = Rc::clone(&sun_demo_mode);
    let sim_for_view = Rc::clone(&sun_cycle_sim_s);
    let preset_for_view = Rc::clone(&active_preset);
    let temp_for_view = Rc::clone(&temp_preset);
    let wind_for_view = Rc::clone(&wind_preset);
    let ocean_for_view = Rc::clone(&ocean_preset);
    let fps_for_view = Rc::clone(&fps_readout);
    let atmo_for_view = Rc::clone(&atmo_readout);
    let fly_idx_for_view = Rc::clone(&fly_target_idx);

    let build_root = move || -> View {
        let (is_2d, at_local_25d, is_flying, sp, la, elev_on, labels_on, w, precip, idle_hi_res_on) =
            earth_for_view.with_state(|st| {
                let inp = st.viewport.input();
                (
                    st.viewport.view_mode() == EarthViewMode::Mode2D,
                    st.orbit.range_m < LOCAL_TRANSITION_RANGE_M,
                    st.orbit.is_flying(),
                    inp.space.clone(),
                    inp.local_atmosphere.clone(),
                    inp.elevation_enabled,
                    inp.layers.labels,
                    inp.weather.clone(),
                    inp.precipitation.clone(),
                    inp.idle_high_res.enabled,
                )
            });
        let cur_place_on = click_place_on.get();
        let cur_placed_count = placed_count.get();
        let cur_ray_status = last_ray_status.get();
        let cur_sun_mode = *mode_for_view.borrow();
        let cur_preset = *preset_for_view.borrow();
        let cur_temp_preset = *temp_for_view.borrow();
        let cur_wind_preset = *wind_for_view.borrow();
        let cur_ocean_preset = *ocean_for_view.borrow();
        let (cur_fps, cur_ms) = *fps_for_view.borrow();
        let cur_atmo_status = atmo_for_view.borrow().clone();
        let next_landmark = FLY_LANDMARKS[*fly_idx_for_view.borrow() % FLY_LANDMARKS.len()];

        let e1 = earth_for_view.clone();
        let btn_mode = control_btn(
            if is_2d { "Mode: 2D" } else { "Mode: 3D" }.to_string(),
            is_2d,
            move |_| {
                e1.with_state_mut(|st| {
                    let next = match st.viewport.view_mode() {
                        EarthViewMode::Mode3D => EarthViewMode::Mode2D,
                        EarthViewMode::Mode2D => EarthViewMode::Mode3D,
                    };
                    st.set_view_mode(next);
                });
            },
        );

        let e2 = earth_for_view.clone();
        let btn_view = control_btn(
            if at_local_25d {
                "View: 2.5D"
            } else {
                "View: Space"
            }
            .to_string(),
            at_local_25d,
            move |_| {
                e2.with_state_mut(|st| {
                    if st.orbit.range_m < LOCAL_TRANSITION_RANGE_M {
                        st.jump_to(26.0, 16.0, 13_500_000.0, 0.0, 0.0);
                    } else {
                        if st.viewport.view_mode() == EarthViewMode::Mode2D {
                            st.viewport.set_view_mode(EarthViewMode::Mode3D);
                            st.orbit.set_is_2d_hint(false);
                        }
                        st.jump_to(37.8199, -122.4980, 2_800.0, 28.0, 66.0);
                    }
                });
            },
        );

        let e_fly = earth_for_view.clone();
        let idx_fly = Rc::clone(&fly_idx_for_view);
        let btn_fly = control_btn(
            format!("Fly To: {}", next_landmark.0),
            is_flying,
            move |_| {
                let idx = *idx_fly.borrow() % FLY_LANDMARKS.len();
                let (_, lat, lon) = FLY_LANDMARKS[idx];
                *idx_fly.borrow_mut() = (idx + 1) % FLY_LANDMARKS.len();
                e_fly.fly_to(lat, lon, None);
            },
        );

        let place_toggle_sig = click_place_on.clone();
        let btn_place = control_btn(
            if cur_place_on {
                "Click Place: On (+5m)"
            } else {
                "Click Place: Off"
            }
            .to_string(),
            cur_place_on,
            move |_| {
                place_toggle_sig.set(!place_toggle_sig.untracked_get());
            },
        );

        let e_clear = earth_for_view.clone();
        let clear_count_sig = placed_count.clone();
        let clear_status_sig = last_ray_status.clone();
        let btn_clear = if cur_placed_count > 0 {
            control_btn(format!("Clear ({cur_placed_count})"), false, move |_| {
                e_clear.clear_objects();
                clear_count_sig.set(0);
                clear_status_sig.set(None);
            })
        } else {
            View::new(strata_dom::NodeBlueprint::fragment(Vec::new()))
        };

        let e_sky25 = earth_for_view.clone();
        let btn_sky_25d = control_btn(
            if la.realistic_sky {
                "2.5D Sky: Real"
            } else {
                "2.5D Sky: Flat"
            }
            .to_string(),
            la.realistic_sky,
            move |_| {
                let mut next_la = e_sky25.local_atmosphere_config();
                let next_on = !next_la.realistic_sky;
                next_la.realistic_sky = next_on;
                next_la.night_stars_sync = next_on;
                next_la.terrain_lighting = next_on;
                e_sky25.set_local_atmosphere_config(next_la);
            },
        );

        let e_shd = earth_for_view.clone();
        let btn_shadows = control_btn(
            if la.terrain_shadows {
                "Shadows: On"
            } else {
                "Shadows: Off"
            }
            .to_string(),
            la.terrain_shadows,
            move |_| {
                let mut next_la = e_shd.local_atmosphere_config();
                next_la.terrain_shadows = !next_la.terrain_shadows;
                if next_la.terrain_shadows {
                    next_la.terrain_lighting = true;
                }
                e_shd.set_local_atmosphere_config(next_la);
            },
        );

        let e3 = earth_for_view.clone();
        let btn_sun_lit = control_btn(
            if sp.day_night_lighting {
                "Sun Light: On"
            } else {
                "Sun Light: Unified"
            }
            .to_string(),
            sp.day_night_lighting,
            move |_| {
                e3.with_state_mut(|st| {
                    let mut sp = st.viewport.space_config().clone();
                    sp.day_night_lighting = !sp.day_night_lighting;
                    st.viewport.set_space_config(sp);
                    st.notify_ui_changed();
                });
            },
        );

        let e4 = earth_for_view.clone();
        let btn_city = control_btn("City Lights".to_string(), sp.city_lights, move |_| {
            e4.with_state_mut(|st| {
                let mut sp = st.viewport.space_config().clone();
                sp.city_lights = !sp.city_lights;
                st.viewport.set_space_config(sp);
                st.notify_ui_changed();
            });
        });

        let e5 = earth_for_view.clone();
        let m5 = Rc::clone(&mode_for_view);
        let s5 = Rc::clone(&sim_for_view);
        let btn_sun_mode = control_btn(
            cur_sun_mode.label().to_string(),
            cur_sun_mode != SunDemoMode::SystemClock,
            move |_| {
                let next = m5.borrow().next();
                *m5.borrow_mut() = next;
                let sim_s = *s5.borrow();
                e5.with_state_mut(|st| {
                    let lat = st.orbit.lat_deg;
                    let lon = st.orbit.lon_deg;
                    let hdg = st.orbit.heading_deg;
                    let mut sp = st.viewport.space_config().clone();
                    apply_sun_demo_mode(next, sim_s, lat, lon, hdg, &mut sp);
                    if next == SunDemoMode::OrbitalCycle {
                        st.viewport.set_sim_time_unix_s(sim_s);
                    }
                    st.viewport.set_space_config(sp);
                    st.notify_ui_changed();
                });
            },
        );

        let e6 = earth_for_view.clone();
        let btn_stars = control_btn("Stars/Moon".to_string(), sp.stars_visible, move |_| {
            e6.with_state_mut(|st| {
                let mut sp = st.viewport.space_config().clone();
                sp.stars_visible = !sp.stars_visible;
                sp.sun_visible = sp.stars_visible;
                sp.moon_visible = sp.stars_visible;
                st.viewport.set_space_config(sp);
                st.notify_ui_changed();
            });
        });

        let e7 = earth_for_view.clone();
        let btn_survey = control_btn(
            format!("Sky: {}", sp.skybox_survey.display_label()),
            sp.stars_visible,
            move |_| {
                e7.with_state_mut(|st| {
                    let mut sp = st.viewport.space_config().clone();
                    sp.skybox_survey = sp.skybox_survey.next();
                    st.viewport.set_space_config(sp);
                    st.notify_ui_changed();
                });
            },
        );

        let e8 = earth_for_view.clone();
        let btn_sky_res = control_btn(
            format!("Res: {}", sp.skybox_quality.short_label()),
            sp.stars_visible,
            move |_| {
                e8.with_state_mut(|st| {
                    let mut sp = st.viewport.space_config().clone();
                    sp.skybox_quality = sp.skybox_quality.next();
                    st.viewport.set_space_config(sp);
                    st.notify_ui_changed();
                });
            },
        );

        let e9 = earth_for_view.clone();
        let btn_sats = control_btn("Satellites".to_string(), sp.satellites_visible, move |_| {
            e9.with_state_mut(|st| {
                let mut sp = st.viewport.space_config().clone();
                sp.satellites_visible = !sp.satellites_visible;
                sp.satellite_orbits_visible = sp.satellites_visible;
                st.viewport.set_space_config(sp);
                st.notify_ui_changed();
            });
        });

        let e10 = earth_for_view.clone();
        let btn_elev = control_btn("Heightmap".to_string(), elev_on, move |_| {
            let next = !e10.elevation_enabled();
            e10.set_elevation_enabled(next);
            e10.with_state(|st| st.notify_ui_changed());
        });

        let e11 = earth_for_view.clone();
        let btn_labels = control_btn("Labels".to_string(), labels_on, move |_| {
            e11.with_state_mut(|st| {
                let next = !st.viewport.labels_enabled();
                st.viewport.set_labels_enabled(next);
                st.notify_ui_changed();
            });
        });

        let e12 = earth_for_view.clone();
        let btn_gcloud = control_btn("Clouds".to_string(), w.global_cloud_map, move |_| {
            e12.with_state_mut(|st| {
                let mut w = st.viewport.weather_config().clone();
                w.global_cloud_map = !w.global_cloud_map;
                st.viewport.set_weather_config(w);
                st.notify_ui_changed();
            });
        });

        let e13 = earth_for_view.clone();
        let btn_cloud_res = control_btn(
            format!("Clouds: {}", w.cloud_resolution.short_label()),
            w.global_cloud_map,
            move |_| {
                e13.with_state_mut(|st| {
                    let mut w = st.viewport.weather_config().clone();
                    w.cloud_resolution = w.cloud_resolution.next();
                    st.viewport.set_weather_config(w);
                    st.notify_ui_changed();
                });
            },
        );

        let e14 = earth_for_view.clone();
        let tp14 = Rc::clone(&temp_for_view);
        let precip_label = if !precip.enabled {
            "Precip: Off"
        } else {
            match &precip.mode {
                PrecipitationMode::Physical => "Precip: Auto",
                PrecipitationMode::Rain => "Precip: Rain",
                PrecipitationMode::Snow => "Precip: Snow",
                PrecipitationMode::CustomLayers(_) => "Precip: Custom",
            }
        };
        let btn_precip = control_btn(precip_label.to_string(), precip.enabled, move |_| {
            let mut cfg = e14.precipitation_config();
            let mut base = e14.base_weather_config();
            if !cfg.enabled {
                cfg.enabled = true;
                cfg.mode = PrecipitationMode::Physical;
                tp14.borrow().apply_to(&mut cfg, &mut base);
                base.precipitation_intensity = cfg.intensity;
            } else {
                match cfg.mode {
                    PrecipitationMode::Physical => {
                        cfg.mode = PrecipitationMode::Rain;
                        base.precipitation_intensity = cfg.intensity;
                    }
                    PrecipitationMode::Rain => {
                        cfg.mode = PrecipitationMode::Snow;
                        base.precipitation_intensity = cfg.intensity;
                    }
                    PrecipitationMode::Snow | PrecipitationMode::CustomLayers(_) => {
                        cfg.enabled = false;
                        cfg.mode = PrecipitationMode::Physical;
                        base.precipitation_intensity = 0.0;
                    }
                }
            }
            e14.set_base_weather_config(base);
            e14.set_precipitation_config(cfg);
        });

        let e15 = earth_for_view.clone();
        let tp15 = Rc::clone(&temp_for_view);
        let btn_temp = control_btn(
            cur_temp_preset.label().to_string(),
            precip.enabled && matches!(precip.mode, PrecipitationMode::Physical),
            move |_| {
                let next = tp15.borrow().next();
                *tp15.borrow_mut() = next;
                let mut cfg = e15.precipitation_config();
                let mut base = e15.base_weather_config();
                next.apply_to(&mut cfg, &mut base);
                e15.set_base_weather_config(base);
                e15.set_precipitation_config(cfg);
            },
        );

        let e_wind = earth_for_view.clone();
        let wp_btn = Rc::clone(&wind_for_view);
        let btn_wind = control_btn(
            cur_wind_preset.label().to_string(),
            cur_wind_preset != WindPreset::BreezeEast,
            move |_| {
                let next = wp_btn.borrow().next();
                *wp_btn.borrow_mut() = next;
                let wv = next.vector();
                let mut base = e_wind.base_weather_config();
                base.enabled = true;
                base.wind = wv;
                base.use_live_api = next == WindPreset::LiveApi;
                e_wind.set_base_weather_config(base);

                let cfg = e_wind
                    .precipitation_config()
                    .with_wind_angle(wv.speed_mps, wv.angle_deg);
                e_wind.set_precipitation_config(cfg);

                let ocfg = e_wind
                    .ocean_waves_config()
                    .with_wind(wv.speed_mps, wv.angle_deg);
                e_wind.set_ocean_waves_config(ocfg);
            },
        );

        let e16 = earth_for_view.clone();
        let p16 = Rc::clone(&preset_for_view);
        let btn_provider = control_btn(
            format!("Map: {}", cur_preset.display_name()),
            cur_preset != TileProviderPreset::ArcGisWorldImagery,
            move |_| {
                let next = p16.borrow().next_public();
                *p16.borrow_mut() = next;
                e16.with_state_mut(|st| {
                    st.viewport.set_tile_preset(next);
                    st.notify_ui_changed();
                });
            },
        );

        let e17 = earth_for_view.clone();
        let op17 = Rc::clone(&ocean_for_view);
        let btn_ocean = control_btn(
            cur_ocean_preset.label().to_string(),
            cur_ocean_preset != OceanPreset::Off,
            move |_| {
                let next = op17.borrow().next();
                *op17.borrow_mut() = next;
                let mut ocfg = e17.ocean_waves_config();
                match next {
                    OceanPreset::Off => {
                        ocfg.enabled = false;
                        ocfg.coastal_shoaling = false;
                    }
                    OceanPreset::OpenSea => {
                        ocfg.enabled = true;
                        ocfg.coastal_shoaling = false;
                    }
                    OceanPreset::CoastalSurf => {
                        ocfg.enabled = true;
                        ocfg.coastal_shoaling = true;
                    }
                }
                e17.set_ocean_waves_config(ocfg);
            },
        );

        let e_idle = earth_for_view.clone();
        let btn_idle_hires = control_btn(
            if idle_hi_res_on {
                "Idle Hi-Res: On"
            } else {
                "Idle Hi-Res: Off"
            }
            .to_string(),
            idle_hi_res_on,
            move |_| {
                let cur = e_idle.idle_high_res_config().enabled;
                e_idle.set_idle_high_res_enabled(!cur);
            },
        );

        let fps_text = format!("{cur_fps} FPS · {cur_ms:.1} ms");
        let atmo_badge = match cur_atmo_status {
            Some(msg) => rsx! {
                <div
                    data-earth-atmo-status="true"
                    class="absolute top-12 left-3 z-20 px-3 py-1.5 rounded-xl bg-neutral-950/85 border border-cyan-500/35 text-cyan-200 text-[11px] font-mono select-none pointer-events-none"
                >
                    <span>{msg}</span>
                </div>
            },
            None => View::new(strata_dom::NodeBlueprint::fragment(Vec::new())),
        };
        let status_badge = match cur_ray_status {
            Some(msg) => rsx! {
                <div
                    data-earth-ray-status="true"
                    class="absolute bottom-3 left-3 z-20 px-3 py-1.5 rounded-xl bg-neutral-950/85 border border-sky-500/40 text-sky-300 text-[11px] font-mono select-none pointer-events-none"
                >
                    <span>{msg}</span>
                </div>
            },
            None => View::new(strata_dom::NodeBlueprint::fragment(Vec::new())),
        };

        rsx! {
            <div class="relative w-full h-full bg-black overflow-hidden">
                {earth_for_view.view()}
                <div
                    data-earth-fps="true"
                    class="absolute top-3 left-3 z-20 flex items-center gap-2 px-3 py-1.5 rounded-xl bg-neutral-950/85 border border-neutral-800/85 text-emerald-400 text-[11px] font-mono font-semibold select-none pointer-events-none"
                >
                    <span>{fps_text}</span>
                </div>
                {atmo_badge}
                {status_badge}
                <div
                    data-earth-controls="true"
                    class="absolute top-3 right-3 z-20 flex flex-wrap justify-end gap-1.5 max-w-[520px] p-2 rounded-xl bg-neutral-950/85 border border-neutral-800/85"
                >
                    {btn_mode}
                    {btn_provider}
                    {btn_view}
                    {btn_fly}
                    {btn_place}
                    {btn_clear}
                    {btn_sky_25d}
                    {btn_shadows}
                    {btn_sun_mode}
                    {btn_ocean}
                    {btn_precip}
                    {btn_temp}
                    {btn_wind}
                    {btn_sun_lit}
                    {btn_city}
                    {btn_stars}
                    {btn_survey}
                    {btn_sky_res}
                    {btn_sats}
                    {btn_elev}
                    {btn_idle_hires}
                    {btn_labels}
                    {btn_gcloud}
                    {btn_cloud_res}
                </div>
            </div>
        }
    };

    let config = WindowConfig {
        title: "Strata Earth".to_string(),
        width: 1440,
        height: 900,
        resizable: true,
        ..Default::default()
    };

    let earth_for_events = earth.clone();
    let earth_for_frame = earth;
    let held_for_frame = Rc::clone(&held_zoom);
    let mode_for_frame = sun_demo_mode;
    let sim_for_frame = sun_cycle_sim_s;
    let fps_for_frame = fps_readout;
    let atmo_for_frame = atmo_readout;
    let mut fps_frames = 0u32;
    let mut fps_window_start = std::time::Instant::now();
    let mut atmo_refresh_start = std::time::Instant::now();
    let mut max_dt_ms = 0.0f64;
    let mut max_step_ms = 0.0f64;
    let mut was_flying = false;
    let mut was_local_25d = false;
    let perf_fps = std::env::var_os("STRATA_PERF_LOG").is_some();

    launch_with_frame(
        build_root,
        config,
        move |event, rt| {
            if rt.active_focus.is_none() {
                earth_for_events.dispatch_platform_event(event);
            }
            false
        },
        None,
        move |info: AppFrameInfo, _rt: &mut AppRuntime| -> FrameOutcome {
            let t_s = std::time::Instant::now();
            let dt = info.dt.as_secs_f64();
            max_dt_ms = max_dt_ms.max(dt * 1000.0);
            fps_frames += 1;
            let mut hud_dirty = false;
            let elapsed_s = fps_window_start.elapsed().as_secs_f64();
            if elapsed_s >= 0.45 {
                let fps = ((fps_frames as f64) / elapsed_s).round().clamp(1.0, 999.0) as u32;
                let ms = (elapsed_s * 1000.0) / (fps_frames.max(1) as f64);
                if perf_fps {
                    eprintln!(
                        "[perf] fps={} avg={:.2}ms max_dt={:.2}ms max_step={:.2}ms frames={}",
                        fps, ms, max_dt_ms, max_step_ms, fps_frames
                    );
                }
                max_dt_ms = 0.0;
                max_step_ms = 0.0;
                let prev = *fps_for_frame.borrow();
                if prev.0 != fps || (prev.1 - ms).abs() >= 0.3 {
                    *fps_for_frame.borrow_mut() = (fps, ms);
                    hud_dirty = true;
                }
                fps_frames = 0;
                fps_window_start = std::time::Instant::now();
            }

            let held_now = *held_for_frame.borrow();
            if held_now.is_active() {
                held_now.step_earth(&earth_for_frame, dt);
            }

            if *mode_for_frame.borrow() == SunDemoMode::OrbitalCycle {
                *sim_for_frame.borrow_mut() += dt * 3_600.0;
                let sim_s = *sim_for_frame.borrow();
                earth_for_frame.with_state_mut(|st| {
                    st.viewport.set_sim_time_unix_s(sim_s);
                });
            }
            let outcome = earth_for_frame.step_frame(dt);
            let (now_flying, now_local_25d, precip_cfg, ocean_cfg, eye_alt_m) =
                earth_for_frame.with_state(|st| {
                    let inp = st.viewport.input();
                    (
                        st.orbit.is_flying(),
                        st.orbit.range_m < LOCAL_TRANSITION_RANGE_M,
                        inp.precipitation.clone(),
                        inp.ocean_waves.clone(),
                        inp.camera.altitude_m(),
                    )
                });
            let fly_edge = now_flying != was_flying;
            was_flying = now_flying;
            let view_edge = now_local_25d != was_local_25d;
            was_local_25d = now_local_25d;

            let weather_snap = earth_for_frame.last_weather_snapshot();
            let need_badge = precip_cfg.enabled
                || ocean_cfg.enabled
                || weather_snap.is_some_and(|w| w.precipitation_intensity > 0.02);
            if !need_badge {
                if atmo_for_frame.borrow().is_some() {
                    *atmo_for_frame.borrow_mut() = None;
                    hud_dirty = true;
                }
            } else if atmo_refresh_start.elapsed().as_secs_f64() >= 0.25 || view_edge {
                atmo_refresh_start = std::time::Instant::now();
                let next_msg = format_atmo_status(
                    &precip_cfg,
                    &ocean_cfg,
                    eye_alt_m,
                    earth_for_frame.last_atmosphere_sample(),
                    weather_snap,
                );
                if *atmo_for_frame.borrow() != next_msg {
                    *atmo_for_frame.borrow_mut() = next_msg;
                    hud_dirty = true;
                }
            }

            max_step_ms = max_step_ms.max(t_s.elapsed().as_secs_f64() * 1000.0);
            if hud_dirty || fly_edge || view_edge {
                FrameOutcome::Dirty
            } else {
                outcome
            }
        },
    )
}

#[cfg(test)]
mod tests {
    use super::{wire_preview_zoom_keys, PreviewHeldZoom};
    use std::cell::RefCell;
    use std::rc::Rc;
    use strata::platform::{KeyCode, Modifiers, PlatformEvent};
    use strata_earth::{
        ecef_to_geodetic, EarthDefaultTileSource, EarthInput, EarthOrbitController, EarthView,
        LocalFrame, MIN_RANGE_M,
    };
    use strata_render::camera64::DVec3;

    #[test]
    fn preview_k_and_l_hold_events_zoom_in_and_out_smoothly_via_external_callbacks() {
        let input = EarthInput::globe();
        let earth = EarthView::new(&input, Box::new(EarthDefaultTileSource::offline_basemap()));
        let held = Rc::new(RefCell::new(PreviewHeldZoom::default()));
        wire_preview_zoom_keys(&earth, &held);

        let start_range = earth.with_state(|st| st.orbit.range_m);

        earth.dispatch_platform_event(&PlatformEvent::KeyDown {
            key: KeyCode::KeyK,
            modifiers: Modifiers::default(),
        });
        assert!(held.borrow().is_active());
        for _ in 0..60 {
            let dt = 1.0 / 120.0;
            held.borrow().step_earth(&earth, dt);
            let _ = earth.step_frame(dt);
        }
        earth.dispatch_platform_event(&PlatformEvent::KeyUp {
            key: KeyCode::KeyK,
            modifiers: Modifiers::default(),
        });
        assert!(!held.borrow().is_active());
        let zoomed_in = earth.with_state(|st| st.orbit.range_m);
        assert!(
            zoomed_in < start_range * 0.50,
            "holding K for 0.5s should zoom closer quickly: start={start_range}, after={zoomed_in}"
        );

        earth.dispatch_platform_event(&PlatformEvent::KeyDown {
            key: KeyCode::KeyL,
            modifiers: Modifiers::default(),
        });
        assert!(held.borrow().is_active());
        for _ in 0..75 {
            let dt = 1.0 / 120.0;
            held.borrow().step_earth(&earth, dt);
            let _ = earth.step_frame(dt);
        }
        earth.dispatch_platform_event(&PlatformEvent::WindowFocused(false));
        assert!(
            !held.borrow().is_active(),
            "losing window focus must synthesize on_key_up and release held K/L keys"
        );
        let zoomed_out = earth.with_state(|st| st.orbit.range_m);
        assert!(
            zoomed_out > zoomed_in * 1.9,
            "holding L should zoom back out smoothly: in={zoomed_in}, out={zoomed_out}"
        );
    }

    #[test]
    fn orbit_zoom_and_tilt_never_roll_camera() {
        let mut orbit = EarthOrbitController::new();
        orbit.set_viewport_size(1440.0, 900.0);
        orbit.jump_to(37.7749, -122.4194, 4_000.0, 0.0, 20.0);
        orbit.handle_mouse_down(2, 720.0, 450.0);
        orbit.handle_mouse_move(780.0, 410.0);
        orbit.handle_mouse_up(2);

        let before_tilt = orbit.tilt_deg;
        let first_step = orbit.step_damping(1.0 / 60.0, false, |_, _| 0.0);
        assert!(first_step, "orbit/tilt must damp smoothly rather than snapping");
        assert!(orbit.tilt_deg >= before_tilt - 1.0);

        let (cam, up_opt) = orbit.to_camera_and_up(false, 0.0);
        let up_v = up_opt
            .map(|u| DVec3::new(u[0], u[1], u[2]))
            .expect("3D camera provides up vector");
        let frame = LocalFrame::at(cam.focus.lat_deg, cam.focus.lon_deg);
        let up_dot = up_v.dot(frame.up);
        assert!(
            up_dot.abs() < 0.05,
            "screen-up hint must lie in local tangent plane, got dot={up_dot}"
        );
        assert!(
            cam.focus.lat_deg.abs() <= 89.95,
            "latitude must stay safely away from polar flip singularity"
        );
    }

    #[test]
    fn pivot_orbit_and_pan_in_25d_work_smoothly() {
        let mut orbit = EarthOrbitController::new();
        orbit.set_viewport_size(1440.0, 900.0);
        orbit.jump_to(37.7749, -122.4194, 4_000.0, 0.0, 45.0);

        let h0 = orbit.heading_deg;
        let t0 = orbit.tilt_deg;
        orbit.handle_mouse_down(2, 720.0, 450.0);
        orbit.handle_mouse_move(820.0, 390.0);
        orbit.handle_mouse_up(2);

        assert!(
            (orbit.heading_deg - h0).abs() > 1.0,
            "horizontal pivot drag must orbit azimuth heading cleanly around pivot, got {}",
            orbit.heading_deg
        );
        assert!(
            orbit.tilt_deg > t0,
            "upward drag must increase tilt cleanly around pivot"
        );

        let lat0 = orbit.lat_deg;
        let lon0 = orbit.lon_deg;
        orbit.handle_mouse_down(0, 720.0, 450.0);
        orbit.handle_mouse_move(600.0, 380.0);
        orbit.handle_mouse_up(0);
        assert!(orbit.step_damping(1.0 / 60.0, false, |_, _| 0.0));
        assert!(
            (orbit.lat_deg - lat0).abs() > 1e-5 || (orbit.lon_deg - lon0).abs() > 1e-5,
            "left drag in 2.5D must pan terrain focus cleanly"
        );
    }

    #[test]
    fn close_ground_zoom_stays_above_mountain_elevation() {
        let mut orbit = EarthOrbitController::new();
        orbit.set_viewport_size(800.0, 600.0);
        orbit.jump_to(46.0, 7.6, 80.0, 15.0, 75.0);
        for _ in 0..20 {
            orbit.handle_scroll(-110.0, 400.0, 300.0);
        }
        let peak_m = |_: f64, _: f64| 4_200.0;
        for _ in 0..90 {
            orbit.step_damping(1.0 / 60.0, false, peak_m);
        }
        assert!(orbit.range_m >= MIN_RANGE_M - 1e-6 && orbit.range_m <= 55.0);
        let (cam, up_opt) = orbit.to_camera_and_up_clamped(false, peak_m);
        let p = cam.placement_with_up(up_opt.map(|u| DVec3::new(u[0], u[1], u[2])));
        let eye_geo = ecef_to_geodetic(p.eye);
        assert!(eye_geo.height_m >= 4_200.0 + 24.5);
    }
}
