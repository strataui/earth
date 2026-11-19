use std::sync::{Arc, Mutex};

use strata_earth::{
    classify_non_land_water, CelestialPlugin, CityLightsPlugin, EarthCamera,
    EarthDynamicOverlayContext, EarthInput, EarthPlugin, EarthPluginFrameContext,
    EarthPluginHooks, EarthPluginRegistry, EarthTextureInjector, EarthTileBuildContext,
    EarthViewport, EarthWorldBuildContext, Geodetic, LoadedTileImagerySnapshot, MemoryTileSource,
    OceanWavesConfig, OceanWavesPlugin, PrecipitationConfig, PrecipitationMode,
    PrecipitationPlugin, RetainedTerrainBuilder, SatellitesPlugin, TileKey, TileProjection,
    TileSource, WeatherPlugin,
};
use strata_render::scene3d::{
    RetainedCameraUniforms, RetainedDynamicMesh, RetainedEnvironmentAssets, RetainedTileInstance,
    RetainedTileTexture, RteTileVertex,
};

struct NightTrackingSource {
    inner: MemoryTileSource,
    night_requests: Arc<Mutex<Vec<TileKey>>>,
}

impl TileSource for NightTrackingSource {
    fn imagery(
        &mut self,
        key: TileKey,
    ) -> Result<Option<strata_earth::TileImage>, strata_earth::TileError> {
        self.inner.imagery(key)
    }

    fn night_lights(
        &mut self,
        key: TileKey,
    ) -> Result<Option<strata_earth::TileImage>, strata_earth::TileError> {
        self.night_requests.lock().unwrap().push(key);
        self.inner.night_lights(key)
    }

    fn elevation(
        &mut self,
        key: TileKey,
    ) -> Result<Option<strata_earth::TileElevation>, strata_earth::TileError> {
        self.inner.elevation(key)
    }

    fn peek_cached_imagery(&self, key: TileKey) -> Option<strata_earth::TileImage> {
        self.inner.peek_cached_imagery(key)
    }

    fn peek_cached_night_lights(&self, key: TileKey) -> Option<strata_earth::TileImage> {
        self.inner.peek_cached_night_lights(key)
    }

    fn peek_cached_elevation(&self, key: TileKey) -> Option<strata_earth::TileElevation> {
        self.inner.peek_cached_elevation(key)
    }
}

fn input_at(lat_deg: f64, lon_deg: f64, range_m: f64) -> EarthInput {
    let mut input = EarthInput::default();
    input.camera = EarthCamera {
        focus: Geodetic::new(lat_deg, lon_deg, 0.0),
        eye_offset_m: [0.0, 0.0, range_m],
        ..Default::default()
    };
    input.elevation_enabled = false;
    input.layers.labels = false;
    input
}

#[test]
fn ocean_waves_emit_animated_gerstner_meshes_over_sea_and_skip_inland() {
    // Open Pacific Ocean near Hawaii
    let mut sea_input = input_at(24.0, -156.0, 8_000.0);
    sea_input.ocean_waves = OceanWavesConfig::enabled();
    let mut vp = EarthViewport::new(&sea_input, Box::new(MemoryTileSource::default()));
    vp.sync_now();

    let frame0 = vp.retained_scene().snapshot_terrain().expect("retained frame");
    let ocean_base0 = frame0
        .dynamic_meshes
        .iter()
        .find(|m| m.id == 0xE47A_0C01)
        .expect("ocean base Gerstner wave mesh present over open ocean");
    assert!(
        frame0.dynamic_meshes.iter().any(|m| m.id == 0xE47A_0C03),
        "ocean trough shadow layer should accompany Gerstner swells"
    );
    assert!(
        frame0.dynamic_meshes.iter().any(|m| m.id == 0xE47A_0C02),
        "ocean crest/glint layer should accompany Gerstner swells"
    );
    assert!(
        ocean_base0.vertices.len() > 200 && ocean_base0.indices.len() >= 300,
        "expected radial Gerstner wave mesh over Pacific, got {} verts / {} indices",
        ocean_base0.vertices.len(),
        ocean_base0.indices.len()
    );
    let verts_before = ocean_base0.vertices.clone();

    // Ticking animation advances wave phase in O(1) and updates wave heights smoothly
    assert!(vp.step_dynamic_overlays(1.0 / 30.0, false));
    let frame1 = vp.retained_scene().snapshot_terrain().expect("updated frame");
    let ocean_base1 = frame1
        .dynamic_meshes
        .iter()
        .find(|m| m.id == 0xE47A_0C01)
        .expect("ocean base mesh after animation step");
    assert_ne!(
        verts_before.as_ref(),
        ocean_base1.vertices.as_ref(),
        "Gerstner wave vertices must animate across frames"
    );

    // Turning ocean waves off clears dynamic ocean wave meshes immediately
    vp.set_ocean_waves_enabled(false);
    let frame_off = vp
        .retained_scene()
        .snapshot_terrain()
        .expect("frame after disable");
    assert!(
        !frame_off
            .dynamic_meshes
            .iter()
            .any(|m| m.id == 0xE47A_0C01 || m.id == 0xE47A_0C02 || m.id == 0xE47A_0C03),
        "disabling OceanWaves must remove ocean wave meshes cleanly"
    );

    // Deep inland North America (Denver) must never render ocean waves even when option is enabled
    let mut land_input = input_at(39.7392, -104.9903, 8_000.0);
    land_input.ocean_waves = OceanWavesConfig::enabled();
    let vp_land = EarthViewport::new(&land_input, Box::new(MemoryTileSource::default()));
    vp_land.sync_now();
    let land_frame = vp_land
        .retained_scene()
        .snapshot_terrain()
        .expect("inland frame");
    assert!(
        !land_frame
            .dynamic_meshes
            .iter()
            .any(|m| m.id == 0xE47A_0C01 || m.id == 0xE47A_0C02 || m.id == 0xE47A_0C03),
        "inland continent must produce zero ocean wave meshes"
    );
}

#[test]
fn coastal_imagery_and_elevation_refine_non_land_wave_masking() {
    // Construct a coastal tile with blue water on West half and green/sandy land on East half
    let tile = TileKey::new(12, 655, 1582); // near San Francisco / Marin coast
    let w = 16u32;
    let h = 16u32;
    let mut rgba = vec![0u8; (w * h * 4) as usize];
    for py in 0..h {
        for px in 0..w {
            let idx = ((py * w + px) * 4) as usize;
            if px < w / 2 {
                // Ocean / Bay water pixel (blue/teal)
                rgba[idx] = 24;
                rgba[idx + 1] = 66;
                rgba[idx + 2] = 104;
                rgba[idx + 3] = 255;
            } else {
                // Land pixel (warm tan / coastal bluff)
                rgba[idx] = 152;
                rgba[idx + 1] = 136;
                rgba[idx + 2] = 98;
                rgba[idx + 3] = 255;
            }
        }
    }
    let snap = LoadedTileImagerySnapshot {
        projection: TileProjection::WebMercator,
        tiles: vec![(
            tile,
            Arc::new(RetainedTileTexture {
                id: 42,
                revision: 1,
                width: w,
                height: h,
                rgba8: Arc::new(rgba),
            }),
        )],
    };

    let b = strata_earth::tile_bounds_mercator(tile);
    let mid_lat = (b.lat0 + b.lat1) * 0.5;
    let west_lon = b.lon0 + (b.lon1 - b.lon0) * 0.20;
    let east_lon = b.lon0 + (b.lon1 - b.lon0) * 0.80;

    let water_score = classify_non_land_water(mid_lat, west_lon, 0.0, Some(&snap));
    let land_score_color = classify_non_land_water(mid_lat, east_lon, 0.0, Some(&snap));
    let land_score_elev = classify_non_land_water(mid_lat, west_lon, 18.0, Some(&snap));

    assert!(
        water_score > 0.6,
        "coastal ocean pixels at 0m elevation must classify as water, got {water_score}"
    );
    assert!(
        land_score_color < 0.05,
        "coastal tan/land imagery pixels must reject ocean waves, got {land_score_color}"
    );
    assert!(
        land_score_elev < 0.01,
        "elevated terrain above sea level must reject ocean waves, got {land_score_elev}"
    );
}

#[test]
fn city_lights_plugin_controls_gpu_intensity_and_skips_night_tile_requests_when_unplugged() {
    let night_requests = Arc::new(Mutex::new(Vec::new()));
    let mut source = NightTrackingSource {
        inner: MemoryTileSource::default(),
        night_requests: Arc::clone(&night_requests),
    };
    let mut input = input_at(35.0, -100.0, 1_200_000.0);
    input.space.day_night_lighting = true;
    input.space.city_lights = true;
    input.space.city_lights_intensity = 1.1;

    let mut builder = RetainedTerrainBuilder::new();
    let (frame_on, _) = builder.build_frame(&input, &mut source, 1, true);
    assert!(
        frame_on.camera.city_lights_intensity > 0.5,
        "CityLightsPlugin active should enable shader city lights intensity"
    );
    assert!(
        !night_requests.lock().unwrap().is_empty(),
        "CityLightsPlugin active should resolve/request night tiles"
    );

    // Disable or unplug CityLightsPlugin: shader uniform drops to 0.0 AND zero night tile requests fire
    night_requests.lock().unwrap().clear();
    assert!(builder.remove_plugin(CityLightsPlugin::ID));
    let (frame_unplugged, _) = builder.build_frame(&input, &mut source, 2, true);
    assert_eq!(
        frame_unplugged.camera.city_lights_intensity, 0.0,
        "Unplugging CityLightsPlugin must zero out city_lights_intensity"
    );
    assert!(
        night_requests.lock().unwrap().is_empty(),
        "Unplugging CityLightsPlugin must prevent night tile fetches cleanly"
    );
    assert!(
        frame_unplugged
            .instances
            .iter()
            .all(|inst| inst.night_texture_id.is_none()),
        "Unplugged CityLightsPlugin must not bind night textures"
    );

    // Re-plugging CityLightsPlugin restores normal night-light operation immediately
    builder.register_plugin(CityLightsPlugin::new(true, 1.5));
    let (frame_replugged, _) = builder.build_frame(&input, &mut source, 3, true);
    assert!(frame_replugged.camera.city_lights_intensity > frame_on.camera.city_lights_intensity);
}

#[test]
fn custom_plugin_mid_work_injection_hooks_every_world_pipeline_stage() {
    #[derive(Clone)]
    struct TracePlugin {
        stages: Arc<Mutex<Vec<&'static str>>>,
        custom_mesh_id: u64,
    }

    impl EarthPlugin for TracePlugin {
        fn id(&self) -> &'static str {
            "custom_trace_injector"
        }

        fn hooks(&self) -> EarthPluginHooks {
            EarthPluginHooks::ALL
        }

        fn wants_animation_tick(&self, _ctx: &EarthPluginFrameContext<'_>) -> bool {
            true
        }

        fn on_tick(&mut self, _dt_s: f64, _ctx: &EarthPluginFrameContext<'_>) -> bool {
            self.stages.lock().unwrap().push("TICK");
            true
        }

        fn on_prepare_camera(
            &mut self,
            _ctx: &EarthPluginFrameContext<'_>,
            camera: &mut RetainedCameraUniforms,
        ) {
            self.stages.lock().unwrap().push("CAMERA");
            camera.stars_intensity *= 1.25;
        }

        fn on_prepare_environment(
            &mut self,
            _ctx: &EarthPluginFrameContext<'_>,
            _env: &mut RetainedEnvironmentAssets,
        ) {
            self.stages.lock().unwrap().push("ENVIRONMENT");
        }

        fn on_tile_begin(&mut self, ctx: &mut EarthTileBuildContext<'_>) {
            let mut st = self.stages.lock().unwrap();
            if st.last().copied() != Some("TILE_BEGIN") {
                st.push("TILE_BEGIN");
            }
            ctx.elevation_enabled = false;
        }

        fn on_tile_instance(
            &mut self,
            ctx: &EarthTileBuildContext<'_>,
            instance: &mut RetainedTileInstance,
            textures: &mut EarthTextureInjector<'_>,
        ) {
            let mut st = self.stages.lock().unwrap();
            if st.last().copied() != Some("TILE_INSTANCE") {
                st.push("TILE_INSTANCE");
            }
            let (tid, trev) = textures.inject_texture(
                0xABCD_0001,
                2,
                2,
                vec![255, 64, 32, 255, 255, 64, 32, 255, 255, 64, 32, 255, 255, 64, 32, 255],
            );
            if ctx.key.x % 2 == 0 {
                instance.texture_id = Some(tid);
                instance.texture_revision = trev;
                instance.tint = [0.25, 0.85, 0.95, 1.0];
            }
        }

        fn on_world_build(&mut self, ctx: &mut EarthWorldBuildContext<'_>) {
            self.stages.lock().unwrap().push("WORLD_BUILD");
            assert!(!ctx.visible_leaves.is_empty());
            ctx.dynamic_meshes.push(RetainedDynamicMesh {
                id: self.custom_mesh_id,
                revision: 1,
                anchor_ecef: [0.0, 0.0, 6_378_137.0],
                vertices: Arc::new(vec![RteTileVertex {
                    pos_u: [0.0, 0.0, 10.0, 0.0],
                    norm_v: [0.0, 0.0, 1.0, 0.0],
                }]),
                indices: Arc::new(vec![0, 0, 0]),
                color: [1.0, 0.2, 0.4, 1.0],
                unlit: true,
            });
        }

        fn on_dynamic_overlays(
            &mut self,
            _ctx: &mut EarthDynamicOverlayContext<'_>,
            dynamic_meshes: &mut Vec<RetainedDynamicMesh>,
        ) {
            self.stages.lock().unwrap().push("DYNAMIC_OVERLAYS");
            dynamic_meshes.push(RetainedDynamicMesh {
                id: self.custom_mesh_id + 1,
                revision: 1,
                anchor_ecef: [0.0, 0.0, 6_378_137.0],
                vertices: Arc::new(vec![RteTileVertex {
                    pos_u: [1.0, 0.0, 10.0, 0.0],
                    norm_v: [0.0, 0.0, 1.0, 0.0],
                }]),
                indices: Arc::new(vec![0, 0, 0]),
                color: [0.2, 1.0, 0.4, 1.0],
                unlit: true,
            });
        }
    }

    let stages = Arc::new(Mutex::new(Vec::new()));
    let plugin = TracePlugin {
        stages: Arc::clone(&stages),
        custom_mesh_id: 0xDEAD_BEEF_0000,
    };

    let mut builder = RetainedTerrainBuilder::new();
    builder.register_plugin(plugin);
    let mut source = MemoryTileSource::default();
    let input = input_at(37.7749, -122.4194, 5_000_000.0);
    let (frame, _) = builder.build_frame(&input, &mut source, 1, true);

    let recorded = stages.lock().unwrap().clone();
    assert_eq!(
        recorded,
        vec![
            "CAMERA",
            "ENVIRONMENT",
            "TILE_BEGIN",
            "TILE_INSTANCE",
            "WORLD_BUILD",
            "DYNAMIC_OVERLAYS"
        ],
        "expected exact mid-work pipeline stage execution order"
    );
    assert!(
        frame
            .instances
            .iter()
            .any(|inst| inst.tint == [0.25, 0.85, 0.95, 1.0] && inst.texture_id.is_some()),
        "mid-work TILE_INSTANCE texture & tint injection must apply cleanly"
    );
    assert!(
        frame
            .textures
            .iter()
            .any(|t| t.width == 2 && t.height == 2 && t.rgba8[0] == 255 && t.rgba8[1] == 64),
        "injected plugin texture must be uploaded with RetainedTerrainFrame"
    );
    assert!(
        frame
            .dynamic_meshes
            .iter()
            .any(|m| m.id == 0xDEAD_BEEF_0000)
            && frame
                .dynamic_meshes
                .iter()
                .any(|m| m.id == 0xDEAD_BEEF_0001),
        "mid-work WORLD_BUILD and DYNAMIC_OVERLAYS meshes must both appear in frame"
    );
}

#[test]
fn empty_plugin_registry_starts_clean_and_allows_selective_composition() {
    let input = {
        let mut inp = input_at(37.7749, -122.4194, 2_500.0);
        inp.view_morph = 0.0;
        inp.seamless_25d = true;
        inp.space.stars_visible = true;
        inp.space.city_lights = true;
        inp.precipitation = PrecipitationConfig {
            enabled: true,
            mode: PrecipitationMode::Rain,
            ..PrecipitationConfig::default()
        };
        inp.ocean_waves = OceanWavesConfig::enabled();
        inp
    };

    let empty = EarthPluginRegistry::empty();
    let mut bare_builder = RetainedTerrainBuilder::new();
    bare_builder.set_plugins(empty.clone());
    let mut source = MemoryTileSource::default();
    let (bare_frame, _) = bare_builder.build_frame(&input, &mut source, 1, true);
    assert_eq!(bare_frame.camera.day_night_strength, 0.0);
    assert_eq!(bare_frame.camera.city_lights_intensity, 0.0);
    assert_eq!(bare_frame.camera.local_cloud_opacity, 0.0);
    assert!(!bare_frame.camera.stars_visible);
    assert!(bare_frame.dynamic_meshes.is_empty());
    assert!(bare_frame.particle_layers.is_empty());

    // Plug in ONLY WeatherPlugin + PrecipitationPlugin + OceanWavesPlugin + SatellitesPlugin + CelestialPlugin selectively
    empty.register(CelestialPlugin::default());
    empty.register(WeatherPlugin::default());
    empty.register(PrecipitationPlugin::default());
    empty.register(SatellitesPlugin::default());
    empty.register(OceanWavesPlugin::default());
    let (composed, _) = bare_builder.build_frame(&input, &mut source, 2, true);
    assert!(composed.camera.stars_visible);
    assert_eq!(composed.camera.city_lights_intensity, 0.0);
    assert!(
        composed.camera.local_cloud_opacity > 0.5,
        "WeatherPlugin should activate 2.5D cloud texture layer when selectively plugged"
    );
    assert!(
        !composed.particle_layers.is_empty(),
        "PrecipitationPlugin should emit GPU 2D rain particle layers when selectively plugged"
    );
}

#[cfg(feature = "widget")]
#[test]
fn earth_view_component_supports_world_plugins_and_ui_overlay_plugins() {
    use strata::prelude::*;
    use strata_earth::{EarthOverlayContext, EarthOverlayPlugin, EarthView};

    struct CustomPinOverlay {
        rendered_coords: Arc<Mutex<Option<(f32, f32)>>>,
    }

    impl EarthOverlayPlugin for CustomPinOverlay {
        fn id(&self) -> &'static str {
            "custom_sf_pin"
        }

        fn render(&self, ctx: &EarthOverlayContext<'_>) -> Option<View> {
            let (sx, sy) = ctx.project_geodetic(37.7749, -122.4194, 0.0)?;
            *self.rendered_coords.lock().unwrap() = Some((sx, sy));
            Some(rsx! {
                <div
                    class="absolute px-2 py-1 rounded bg-emerald-500 text-white text-xs"
                    style={format!("left: {:.1}px; top: {:.1}px;", sx, sy)}
                >
                    <span>"Custom Beacon"</span>
                </div>
            })
        }
    }

    let coords = Arc::new(Mutex::new(None));
    let mut input = input_at(37.7749, -122.4194, 4_000.0);
    input.viewport = strata::layout::Rect::new(0.0, 0.0, 1024.0, 768.0);
    let earth = EarthView::new(&input, Box::new(MemoryTileSource::default()))
        .with_ocean_waves(OceanWavesConfig::enabled())
        .with_overlay_plugin(CustomPinOverlay {
            rendered_coords: Arc::clone(&coords),
        });
    earth.with_state_mut(|st| {
        st.orbit.set_viewport_size(1024.0, 768.0);
        st.orbit.jump_to(37.7749, -122.4194, 4_000.0, 0.0, 0.0);
        st.sync_camera();
    });

    let _view = earth.view();
    let projected = coords
        .lock()
        .unwrap()
        .expect("custom overlay plugin must project center pin");
    assert!(
        (projected.0 - 512.0).abs() < 4.0 && (projected.1 - 384.0).abs() < 4.0,
        "center coordinate must project near viewport center (512, 384), got {projected:?}"
    );
    assert_eq!(earth.is_plugin_enabled("ocean_waves"), Some(true));
    assert!(earth.remove_plugin("city_lights"));
    assert_eq!(earth.is_plugin_enabled("city_lights"), None);
}

#[test]
fn standard_plugins_preserve_space_and_terrain_parity() {
    let input = input_at(26.0, 16.0, 13_500_000.0);
    let cam_stateless = RetainedTerrainBuilder::camera_from_input(&input);
    let std_reg = EarthPluginRegistry::standard();
    let cam_plugins = RetainedTerrainBuilder::camera_from_input_with_plugins(
        &input,
        &std_reg,
        input.weather.weather_time_s,
        0,
        false,
    );
    assert_eq!(cam_stateless, cam_plugins);
    assert!(cam_plugins.stars_visible);
    assert!(cam_plugins.day_night_strength > 0.9);
    assert!(cam_plugins.city_lights_intensity > 0.9);
    assert!(cam_plugins.cloud_layer_opacity > 0.1);

    let vp = EarthViewport::new(&input, Box::new(MemoryTileSource::default()));
    vp.sync_now();
    let frame = vp
        .retained_scene()
        .snapshot_terrain()
        .expect("default EarthViewport frame");
    assert_eq!(frame.camera, cam_stateless);
    assert!(!frame.instances.is_empty(), "terrain tiles must emit");
    let env = frame.environment.as_ref().expect("environment assets");
    assert!(env.land_mask.is_some());
    assert!(env.cloud_map.is_some());
    assert!(env.stars.as_ref().is_some_and(|s| s.len() > 8_000));
}
