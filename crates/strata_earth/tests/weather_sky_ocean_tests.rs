use std::sync::Arc;

use strata_earth::{
    compute_coastal_sdf_grid, parse_open_meteo_weather_json, BaseWeatherConfig, EarthCamera,
    EarthInput, EarthViewport, FnWeatherProvider, Geodetic, LocalAtmosphereConfig,
    MemoryTileSource, OceanWavesConfig, OceanWavesPlugin, OpenMeteoWeatherProvider,
    PrecipitationConfig, PrecipitationMode, RetainedTerrainBuilder, WeatherProvider, WeatherQuery,
    WeatherSnapshot, WindVector,
};
use strata_render::scene3d::RetainedParticleSprite;

fn local_input(lat_deg: f64, lon_deg: f64, alt_m: f64) -> EarthInput {
    let mut input = EarthInput::globe();
    input.camera = EarthCamera {
        focus: Geodetic::new(lat_deg, lon_deg, 0.0),
        eye_offset_m: [0.0, -alt_m * 0.65, alt_m],
        ..Default::default()
    };
    input.elevation_enabled = false;
    input.layers.labels = false;
    input
}

#[test]
fn wind_vector_meteorological_and_enu_conversions_are_exact() {
    // West wind (blowing FROM 270 deg) blows TO 90 deg East => (+east, 0 north)
    let w = WindVector::from_meteorological_from_deg(15.0, 270.0);
    assert!((w.angle_deg - 90.0).abs() < 1e-3);
    assert_eq!(w.compass_label(), "E");
    let enu = w.to_enu_mps();
    assert!((enu[0] - 15.0).abs() < 1e-3 && enu[1].abs() < 1e-3);

    // Roundtrip arbitrary heading (225 deg SW)
    let sw = WindVector::new(18.5, 225.0);
    assert_eq!(sw.compass_label(), "SW");
    let rt = WindVector::from_enu_mps(sw.to_enu_mps());
    assert!((rt.speed_mps - 18.5).abs() < 1e-3);
    assert!((rt.angle_deg - 225.0).abs() < 1e-2);
}

#[test]
fn open_meteo_weather_json_parses_temperature_wind_clouds_and_storm_optics() {
    let sample_json = r#"{
        "elevation": 120.0,
        "current": {
            "temperature_2m": 6.0,
            "relative_humidity_2m": 92.0,
            "wind_speed_10m": 54.0,
            "wind_direction_10m": 225.0,
            "cloud_cover": 94.0,
            "precipitation": 5.2,
            "rain": 5.2,
            "snowfall": 0.0,
            "weather_code": 65
        },
        "hourly": {
            "freezing_level_height": [1100.0],
            "temperature_1000hPa": [6.5],
            "geopotential_height_1000hPa": [110.0],
            "temperature_850hPa": [-2.5],
            "geopotential_height_850hPa": [1450.0],
            "wind_speed_850hPa": [72.0],
            "wind_direction_850hPa": [240.0]
        }
    }"#;

    let report = parse_open_meteo_weather_json(sample_json).expect("valid Open-Meteo weather JSON");
    assert_eq!(report.wmo_code, Some(65));
    assert!(report.cloud_cover > 0.90);
    // 54 km/h = 15 m/s FROM 225 deg (SW) -> blows TO 45 deg (NE)
    assert!((report.surface_wind.speed_mps - 15.0).abs() < 0.05);
    assert!((report.surface_wind.angle_deg - 45.0).abs() < 0.2);

    let provider = OpenMeteoWeatherProvider::new();
    provider.insert_cached_report(46.5, 8.0, report);

    let cfg = BaseWeatherConfig::default().enabled();
    let low_snap = provider.sample(
        WeatherQuery {
            lat_deg: 46.5,
            lon_deg: 8.0,
            altitude_msl_m: 200.0,
            ground_elevation_m: 120.0,
            unix_utc_s: 1_735_689_600.0,
        },
        &cfg,
    );
    let high_snap = provider.sample(
        WeatherQuery {
            lat_deg: 46.5,
            lon_deg: 8.0,
            altitude_msl_m: 1_500.0,
            ground_elevation_m: 120.0,
            unix_utc_s: 1_735_689_600.0,
        },
        &cfg,
    );

    assert!(low_snap.source_live);
    assert!(low_snap.atmosphere.phase.rain > 0.9, "warm valley must be rain");
    assert!(high_snap.atmosphere.phase.snow > 0.85, "cold 1500m altitude must transition to snow");
    assert!(low_snap.storm_darkness > 0.65, "heavy rain WMO 65 must darken sky & clouds");
    assert!(low_snap.precipitation_intensity > 0.8);
    assert!(high_snap.wind.speed_mps > low_snap.wind.speed_mps, "upper 850hPa wind is faster");
}

#[test]
fn unified_base_weather_api_coherently_drives_sky_clouds_precipitation_and_ocean() {
    let storm_provider = Arc::new(FnWeatherProvider(|query: WeatherQuery, fallback: &BaseWeatherConfig| {
        let mut atmo = fallback
            .atmosphere
            .evaluate(query.lat_deg, query.altitude_msl_m, query.ground_elevation_m);
        let wind = WindVector::new(19.0, 135.0);
        atmo.wind_mps = Some(wind.to_enu_mps());
        WeatherSnapshot {
            atmosphere: atmo,
            wind,
            cloud_cover: 0.92,
            precipitation_intensity: 1.2,
            storm_darkness: 0.82,
            visibility_m: 4_200.0,
            wmo_code: Some(65),
            source_live: false,
        }
    }));

    // Off coast of California over open water
    let mut input = local_input(36.2, -122.4, 1_400.0);
    input.local_atmosphere = LocalAtmosphereConfig::realistic();
    input.base_weather = BaseWeatherConfig::default().enabled().with_base_temp_c(12.0);
    input.ocean_waves = OceanWavesConfig::enabled().with_sync_with_weather(true);
    // Note: precipitation.enabled is false, yet sync_with_base_weather == true so base_weather activates it automatically!
    input.precipitation.enabled = false;
    input.precipitation.sync_with_base_weather = true;

    let mut builder = RetainedTerrainBuilder::new();
    builder.set_weather_provider(Some(storm_provider));
    let mut src = MemoryTileSource::default();
    let (frame, _) = builder.build_frame(&input, &mut src, 1, true);

    // 1. 2.5D Sky & atmosphere optics reflect storm darkness, cloud cover, and haze
    assert_eq!(frame.camera.realistic_25d_sky, 1.0);
    assert_eq!(frame.camera.night_stars_25d, 1.0);
    assert_eq!(frame.camera.terrain_25d_lighting, 1.0);
    assert!(frame.camera.terrain_25d_shadows > 0.7);
    assert!(frame.camera.weather_darkness > 0.75);
    assert!(frame.camera.weather_cloud_cover > 0.85);
    assert!(frame.camera.weather_haze > 0.45);

    // 2. 2.5D Clouds darken to brooding storm slate & boost cover
    assert!(
        frame.camera.cloud_tint_rgb[0] < 0.65 && frame.camera.cloud_tint_rgb[2] > frame.camera.cloud_tint_rgb[0],
        "storm clouds should tint darker slate-blue, got {:?}",
        frame.camera.cloud_tint_rgb
    );
    assert!(frame.camera.cloud_25d_cover_boost > 0.7);

    // 3. Precipitation automatically activates with matching wind vector (19 m/s SE = +13.43 E, -13.43 N)
    assert!(!frame.particle_layers.is_empty(), "base weather storm must emit precipitation");
    let p0 = &frame.particle_layers[0];
    assert!(
        (p0.motion_params[1] - 13.435).abs() < 0.3 && (p0.motion_params[2] + 13.435).abs() < 0.3,
        "precipitation wind should match unified weather provider, got [{}, {}]",
        p0.motion_params[1],
        p0.motion_params[2]
    );

    // 4. GPU Ocean waves synchronize direction and Beaufort wind speed from the same WeatherSnapshot
    let ocean = frame.ocean.as_ref().expect("ocean surface active over sea");
    assert!((ocean.train_a.wind_speed_mps - 19.0).abs() < 0.1);
    let expected_dir = WindVector::new(19.0, 135.0).to_enu_mps();
    let dir_dot = (ocean.train_a.dir_en[0] * expected_dir[0]
        + ocean.train_a.dir_en[1] * expected_dir[1])
        / 19.0;
    assert!(dir_dot > 0.99, "ocean wave train direction must align with unified wind");
}

#[test]
fn local_atmosphere_defaults_are_conservative_and_realistic_preset_enables_all_25d_features() {
    let def_input = local_input(37.77, -122.42, 1_800.0);
    let cam_def = RetainedTerrainBuilder::camera_from_input(&def_input);
    assert_eq!(cam_def.realistic_25d_sky, 0.0);
    assert_eq!(cam_def.night_stars_25d, 0.0);
    assert_eq!(cam_def.terrain_25d_lighting, 0.0);
    assert_eq!(cam_def.terrain_25d_shadows, 0.0);
    assert_eq!(cam_def.weather_darkness, 0.0);
    assert_eq!(cam_def.cloud_tint_rgb, [1.0, 1.0, 1.0]);

    let mut real_input = def_input;
    real_input.local_atmosphere = LocalAtmosphereConfig::realistic();
    let cam_real = RetainedTerrainBuilder::camera_from_input(&real_input);
    assert_eq!(cam_real.realistic_25d_sky, 1.0);
    assert_eq!(cam_real.night_stars_25d, 1.0);
    assert_eq!(cam_real.terrain_25d_lighting, 1.0);
    assert!(cam_real.terrain_25d_shadows > 0.8);
    assert!(cam_real.city_lights_intensity > 0.8, "2.5D night city lights allowed with realistic night sync");
}

#[test]
fn zoomed_out_precipitation_emits_far_field_curtains_and_eliminates_panning_state_reset() {
    let mut input = local_input(46.55, 7.98, 1_700.0);
    input.precipitation = PrecipitationConfig::default()
        .enabled()
        .with_mode(PrecipitationMode::Snow)
        .with_wind_angle(10.0, 90.0);

    let mut vp = EarthViewport::new(&input, Box::new(MemoryTileSource::default()));
    vp.sync_now();

    let f0 = vp.retained_scene().snapshot_terrain().expect("initial snow frame");
    assert!(
        f0.particle_layers.len() >= 3,
        "zoomed-out 1700m snow must emit 3 nested shells (near, mid, far curtain), got {}",
        f0.particle_layers.len()
    );
    let has_curtain = f0
        .particle_layers
        .iter()
        .any(|l| matches!(l.sprite, RetainedParticleSprite::ProceduralCurtain));
    assert!(has_curtain, "far-field ProceduralCurtain shell must be emitted at 1700m AGL");

    let has_forward_bias = f0.particle_layers.iter().any(|l| {
        let b = l.forward_bias_world;
        (b[0] * b[0] + b[1] * b[1] + b[2] * b[2]).sqrt() > 10.0
    });
    assert!(has_forward_bias, "oblique camera must shift precipitation volume forward into frustum");

    let near0 = f0
        .particle_layers
        .iter()
        .find(|l| l.id == 0x5052_0002)
        .expect("near snow shell");
    let phase_e0 = near0.wrapped_phase[0];

    // Rapid single-frame eastward pan (~160 m in one frame): sub-Nyquist clamping prevents wrap aliasing/reset!
    let mut panned_cam = input.camera;
    panned_cam.focus.lon_deg += 0.0021;
    vp.update_camera(&panned_cam, 1440.0, 900.0);

    let f1 = vp.retained_scene().snapshot_terrain().expect("panned snow frame");
    let near1 = f1
        .particle_layers
        .iter()
        .find(|l| l.id == 0x5052_0002)
        .expect("near snow shell after pan");
    let mut delta_phase = near1.wrapped_phase[0] - phase_e0;
    if delta_phase > 0.5 {
        delta_phase -= 1.0;
    } else if delta_phase < -0.5 {
        delta_phase += 1.0;
    }
    assert!(
        delta_phase < -0.001 && delta_phase >= -0.17,
        "eastward pan must drift near snowflakes smoothly westward (<0) within sub-Nyquist clamp [-0.16, 0), got {delta_phase}"
    );
}

#[test]
fn ocean_wind_crossfade_is_monotonic_without_phase_jump_and_coastal_sdf_is_exact() {
    // 1. Verify dual-state smooth wind transition (no instant jump or phase spin!)
    let mut plugin = OceanWavesPlugin::new();
    plugin.step_wind_state(0.016, 1.0, 4.0, WindVector::new(8.0, 30.0));
    assert_eq!(plugin.current_train_weights(), (1.0, 0.0));

    // Rotate wind 120 deg and raise speed to 22 m/s -> triggers clean crossfade over 4.0s
    let mut prev_wb = 0.0f32;
    for _ in 0..20 {
        plugin.step_wind_state(0.1, 1.0, 4.0, WindVector::new(22.0, 150.0));
        let (wa, wb) = plugin.current_train_weights();
        assert!((wa + wb - 1.0).abs() < 1e-4, "weights must sum to 1.0");
        assert!(wb >= prev_wb, "target wind weight must increase monotonically during fade");
        prev_wb = wb;
    }
    assert!(prev_wb > 0.4 && prev_wb < 0.65, "at t=2.0s of 4.0s fade, wb should be ~0.5, got {prev_wb}");
    for _ in 0..30 {
        plugin.step_wind_state(0.1, 1.0, 4.0, WindVector::new(22.0, 150.0));
    }
    assert_eq!(plugin.current_train_weights(), (1.0, 0.0), "fade completes cleanly into train_a");

    // 2. Verify O(N) Euclidean Distance Transform coastal SDF grid (land on West x<16, water on East x>=16)
    let w = 64usize;
    let h = 64usize;
    let mut water = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            water[y * w + x] = if x >= 16 { 1.0 } else { 0.0 };
        }
    }
    let cell_m = 10.0f32;
    let range_m = 200.0f32;
    let rgba = compute_coastal_sdf_grid(&water, w, h, cell_m, range_m);

    // At x=26, y=32 -> exactly 11 cells (110 m) east of last land cell x=15 -> norm = 110/200 = 0.55
    let o_mid = (32 * w + 26) * 4;
    assert_eq!(rgba[o_mid], 255, "water pixel must have R=255");
    let d_norm = rgba[o_mid + 1] as f32 / 255.0;
    assert!((d_norm - 0.55).abs() < 0.03, "expected shore distance norm ~0.55, got {d_norm}");
    // Shoreward normal points TOWARD land (West = -1.0 East => B ~ 0, A ~ 128)!
    let dir_e = (rgba[o_mid + 2] as f32 / 255.0) * 2.0 - 1.0;
    let dir_n = (rgba[o_mid + 3] as f32 / 255.0) * 2.0 - 1.0;
    assert!(dir_e < -0.95 && dir_n.abs() < 0.05, "shoreward normal must point West (-1, 0), got ({dir_e}, {dir_n})");

    // 3. Verify coastal_shoaling == false emits zero coastal SDF texture (zero CPU overhead!)
    let mut sea_no_coast = local_input(24.0, -156.0, 3_000.0);
    sea_no_coast.ocean_waves = OceanWavesConfig::enabled().with_coastal_shoaling(false);
    let vp_open = EarthViewport::new(&sea_no_coast, Box::new(MemoryTileSource::default()));
    vp_open.sync_now();
    let f_open = vp_open.retained_scene().snapshot_terrain().expect("open sea frame");
    let oc_open = f_open.ocean.as_ref().expect("open sea ocean");
    assert!(!oc_open.coastal_shoaling && oc_open.coastal_sdf.is_none());

    // 4. Verify coastal_shoaling == true emits cached 128x128 coastal SDF texture off Marin/Golden Gate coast!
    let mut coast_in = local_input(37.77, -122.65, 2_200.0);
    coast_in.ocean_waves = OceanWavesConfig::enabled().with_coastal_shoaling(true);
    let vp_coast = EarthViewport::new(&coast_in, Box::new(MemoryTileSource::default()));
    vp_coast.sync_now();
    let f_coast = vp_coast.retained_scene().snapshot_terrain().expect("coastal frame");
    let oc_coast = f_coast.ocean.as_ref().expect("coastal ocean");
    assert!(oc_coast.coastal_shoaling);
    let sdf_tex = oc_coast.coastal_sdf.as_ref().expect("coastal SDF texture");
    assert_eq!(sdf_tex.width, 128);
    assert_eq!(sdf_tex.height, 128);
}
