// Interactive Earth preview embedding `strata_earth::EarthView` alongside floating
// top-right Strata HUD controls, with zero global window event hooks.

use std::cell::RefCell;
use std::rc::Rc;

use strata::prelude::*;
use strata::WindowConfig;
use strata::{launch_with_frame, AppFrameInfo, FrameOutcome};
use strata_earth::{
    EarthDefaultTileSource, EarthInput, EarthOrbitController, EarthView, EarthViewMode,
    SpaceEnvironmentConfig, SunDirectionMode, TileProviderPreset, WeatherConfig,
};

#[derive(Clone, Copy, PartialEq, Eq)]
enum SunDemoMode {
    SystemClock,
    OrbitalCycle,
    FixedSunset,
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
    input.space = SpaceEnvironmentConfig::default().with_system_clock();
    input.weather = WeatherConfig::default();

    let earth = EarthView::with_orbit(&input, Box::new(source), orbit);
    let sun_demo_mode = Rc::new(RefCell::new(SunDemoMode::SystemClock));
    let sun_cycle_sim_s = Rc::new(RefCell::new(1_735_689_600.0f64));
    let active_preset = Rc::new(RefCell::new(TileProviderPreset::ArcGisWorldImagery));
    let fps_readout = Rc::new(RefCell::new((60u32, 16.7f64)));

    let earth_for_view = earth.clone();
    let mode_for_view = Rc::clone(&sun_demo_mode);
    let sim_for_view = Rc::clone(&sun_cycle_sim_s);
    let preset_for_view = Rc::clone(&active_preset);
    let fps_for_view = Rc::clone(&fps_readout);

    let build_root = move || -> View {
        let (is_2d, at_local_25d, sp, elev_on, labels_on, w, ocean_on) =
            earth_for_view.with_state(|st| {
                let inp = st.viewport.input();
                (
                    st.viewport.view_mode() == EarthViewMode::Mode2D,
                    st.orbit.range_m < 20_000.0,
                    inp.space.clone(),
                    inp.elevation_enabled,
                    inp.layers.labels,
                    inp.weather.clone(),
                    inp.ocean_waves.enabled,
                )
            });
        let cur_sun_mode = *mode_for_view.borrow();
        let cur_preset = *preset_for_view.borrow();
        let (cur_fps, cur_ms) = *fps_for_view.borrow();

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
                "View: 2.5D SF"
            } else {
                "View: Space"
            }
            .to_string(),
            at_local_25d,
            move |_| {
                e2.with_state_mut(|st| {
                    if st.orbit.range_m < 20_000.0 {
                        st.jump_to(26.0, 16.0, 13_500_000.0, 0.0, 0.0);
                    } else {
                        if st.viewport.view_mode() == EarthViewMode::Mode2D {
                            st.viewport.set_view_mode(EarthViewMode::Mode3D);
                            st.orbit.set_is_2d_hint(false);
                        }
                        st.jump_to(37.7749, -122.4194, 3_200.0, 22.0, 58.0);
                    }
                });
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
            match cur_sun_mode {
                SunDemoMode::SystemClock => "Sun: OS Clock",
                SunDemoMode::OrbitalCycle => "Sun: Cycle",
                SunDemoMode::FixedSunset => "Sun: Sunset",
            }
            .to_string(),
            cur_sun_mode != SunDemoMode::FixedSunset,
            move |_| {
                let next = match *m5.borrow() {
                    SunDemoMode::SystemClock => SunDemoMode::OrbitalCycle,
                    SunDemoMode::OrbitalCycle => SunDemoMode::FixedSunset,
                    SunDemoMode::FixedSunset => SunDemoMode::SystemClock,
                };
                *m5.borrow_mut() = next;
                e5.with_state_mut(|st| {
                    let mut sp = st.viewport.space_config().clone();
                    match next {
                        SunDemoMode::SystemClock => {
                            sp.sun_direction = SunDirectionMode::SystemClock;
                        }
                        SunDemoMode::OrbitalCycle => {
                            sp.sun_direction = SunDirectionMode::SimulationTime;
                            st.viewport.set_sim_time_unix_s(*s5.borrow());
                        }
                        SunDemoMode::FixedSunset => {
                            sp.sun_direction = SunDirectionMode::CustomEcef([0.85, 0.48, 0.22]);
                        }
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
            e10.with_state_mut(|st| {
                let next = !st.viewport.elevation_enabled();
                st.viewport.set_elevation_enabled(next);
                st.sync_camera();
                st.notify_ui_changed();
            });
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
        let btn_rain = control_btn("Rain".to_string(), w.rain, move |_| {
            e14.with_state_mut(|st| {
                let mut w = st.viewport.weather_config().clone();
                w.rain = !w.rain;
                st.viewport.set_weather_config(w);
                st.notify_ui_changed();
            });
        });

        let e15 = earth_for_view.clone();
        let btn_snow = control_btn("Snow".to_string(), w.snow, move |_| {
            e15.with_state_mut(|st| {
                let mut w = st.viewport.weather_config().clone();
                w.snow = !w.snow;
                st.viewport.set_weather_config(w);
                st.notify_ui_changed();
            });
        });

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
        let btn_ocean = control_btn("Ocean Waves".to_string(), ocean_on, move |_| {
            e17.set_ocean_waves_enabled(!ocean_on);
        });

        let fps_text = format!("{cur_fps} FPS · {cur_ms:.1} ms");

        rsx! {
            <div class="relative w-full h-full bg-black overflow-hidden">
                {earth_for_view.view()}
                <div
                    data-earth-fps="true"
                    class="absolute top-3 left-3 z-20 flex items-center gap-2 px-3 py-1.5 rounded-xl bg-neutral-950/85 border border-neutral-800/85 text-emerald-400 text-[11px] font-mono font-semibold select-none pointer-events-none"
                >
                    <span>{fps_text}</span>
                </div>
                <div
                    data-earth-controls="true"
                    class="absolute top-3 right-3 z-20 flex flex-wrap justify-end gap-1.5 max-w-[440px] p-2 rounded-xl bg-neutral-950/85 border border-neutral-800/85"
                >
                    {btn_mode}
                    {btn_provider}
                    {btn_view}
                    {btn_sun_lit}
                    {btn_city}
                    {btn_ocean}
                    {btn_sun_mode}
                    {btn_stars}
                    {btn_survey}
                    {btn_sky_res}
                    {btn_sats}
                    {btn_elev}
                    {btn_labels}
                    {btn_gcloud}
                    {btn_cloud_res}
                    {btn_rain}
                    {btn_snow}
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
    let mode_for_frame = sun_demo_mode;
    let sim_for_frame = sun_cycle_sim_s;
    let fps_for_frame = fps_readout;
    let mut fps_frames = 0u32;
    let mut fps_window_start = std::time::Instant::now();
    let mut max_dt_ms = 0.0f64;
    let mut max_step_ms = 0.0f64;
    let perf_fps = std::env::var_os("STRATA_PERF_LOG").is_some();

    launch_with_frame(
        build_root,
        config,
        move |event, _rt| {
            handle_preview_key_event(&earth_for_events, event);
            false
        },
        None,
        move |info: AppFrameInfo, _rt: &mut AppRuntime| -> FrameOutcome {
            let t_s = std::time::Instant::now();
            let dt = info.dt.as_secs_f64();
            max_dt_ms = max_dt_ms.max(dt * 1000.0);
            fps_frames += 1;
            let mut fps_dirty = false;
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
                    fps_dirty = true;
                }
                fps_frames = 0;
                fps_window_start = std::time::Instant::now();
            }

            if *mode_for_frame.borrow() == SunDemoMode::OrbitalCycle {
                *sim_for_frame.borrow_mut() += dt * 3_600.0;
                let sim_s = *sim_for_frame.borrow();
                earth_for_frame.with_state_mut(|st| {
                    st.viewport.set_sim_time_unix_s(sim_s);
                });
            }
            let outcome = earth_for_frame.step_frame(dt);
            max_step_ms = max_step_ms.max(t_s.elapsed().as_secs_f64() * 1000.0);
            if fps_dirty {
                FrameOutcome::Dirty
            } else {
                outcome
            }
        },
    )
}

fn handle_preview_key_event(earth: &EarthView, event: &strata::platform::PlatformEvent) {
    use strata::platform::{KeyCode, PlatformEvent};

    match event {
        PlatformEvent::KeyDown { key, modifiers }
            if !modifiers.ctrl && !modifiers.alt && !modifiers.meta =>
        {
            match key {
                KeyCode::KeyK => earth.with_state_mut(|st| {
                    st.orbit.set_zoom_fast_modifier(modifiers.shift);
                    st.orbit.set_zoom_in_held(true);
                }),
                KeyCode::KeyL => earth.with_state_mut(|st| {
                    st.orbit.set_zoom_fast_modifier(modifiers.shift);
                    st.orbit.set_zoom_out_held(true);
                }),
                _ => {}
            }
        }
        PlatformEvent::KeyUp { key, modifiers } => match key {
            KeyCode::KeyK => earth.with_state_mut(|st| {
                st.orbit.set_zoom_fast_modifier(modifiers.shift);
                st.orbit.set_zoom_in_held(false);
            }),
            KeyCode::KeyL => earth.with_state_mut(|st| {
                st.orbit.set_zoom_fast_modifier(modifiers.shift);
                st.orbit.set_zoom_out_held(false);
            }),
            _ => {}
        },
        PlatformEvent::WindowFocused(false) | PlatformEvent::WindowCloseRequested => {
            earth.with_state_mut(|st| st.orbit.clear_held_keys());
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::handle_preview_key_event;
    use strata::platform::{KeyCode, Modifiers, PlatformEvent};
    use strata_earth::{
        ecef_to_geodetic, EarthDefaultTileSource, EarthInput, EarthOrbitController, EarthView,
        LocalFrame, MIN_RANGE_M,
    };
    use strata_render::camera64::DVec3;

    #[test]
    fn preview_k_and_l_hold_events_zoom_in_and_out_smoothly() {
        let input = EarthInput::globe();
        let earth = EarthView::new(&input, Box::new(EarthDefaultTileSource::offline_basemap()));
        let start_range = earth.with_state(|st| st.orbit.range_m);

        handle_preview_key_event(
            &earth,
            &PlatformEvent::KeyDown {
                key: KeyCode::KeyK,
                modifiers: Modifiers::default(),
            },
        );
        for _ in 0..60 {
            let _ = earth.step_frame(1.0 / 120.0);
        }
        handle_preview_key_event(
            &earth,
            &PlatformEvent::KeyUp {
                key: KeyCode::KeyK,
                modifiers: Modifiers::default(),
            },
        );
        let zoomed_in = earth.with_state(|st| st.orbit.range_m);
        assert!(
            zoomed_in < start_range * 0.45,
            "holding K for 0.5s should zoom closer quickly: start={start_range}, after={zoomed_in}"
        );

        handle_preview_key_event(
            &earth,
            &PlatformEvent::KeyDown {
                key: KeyCode::KeyL,
                modifiers: Modifiers::default(),
            },
        );
        for _ in 0..75 {
            let _ = earth.step_frame(1.0 / 120.0);
        }
        handle_preview_key_event(&earth, &PlatformEvent::WindowFocused(false));
        assert!(
            earth.with_state(|st| !st.orbit.is_zoom_key_held()),
            "losing window focus must release held K/L keys"
        );
        let zoomed_out = earth.with_state(|st| st.orbit.range_m);
        assert!(
            zoomed_out > zoomed_in * 2.2,
            "holding L should zoom back out smoothly: in={zoomed_in}, out={zoomed_out}"
        );
    }

    #[test]
    fn orbit_zoom_and_tilt_never_roll_camera() {
        let mut orbit = EarthOrbitController::new();
        orbit.set_viewport_size(1440.0, 900.0);
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
