use std::sync::Arc;

use strata_earth::{
    climatological_sea_level_temp_c, compute_precipitation_phase, isa_pressure_hpa,
    moist_adiabatic_lapse_rate_k_per_km, parse_open_meteo_sounding_json, wet_bulb_temp_c,
    AtmosphereConfig, AtmosphereQuery, CloudCeiling, EarthCamera, EarthInput, EarthViewport,
    FnTemperatureProvider, Geodetic, LapseRateModel, MemoryTileSource,
    OpenMeteoTemperatureProvider, ParticleBillboard, PrecipitationConfig,
    PrecipitationEvalContext, PrecipitationLayer, PrecipitationMode, PrecipitationPlugin,
    RetainedTerrainBuilder, StaticAtmosphereProvider, TemperatureAnchor, TemperatureProvider,
};
use strata_layout::Rect;
use strata_render::scene3d::{RetainedParticleBillboard, RetainedParticleSprite};

fn low_alt_input(lat: f64, lon: f64, alt_m: f64) -> EarthInput {
    EarthInput {
        camera: EarthCamera {
            focus: Geodetic::new(lat, lon, 0.0),
            aim_offset_m: [0.0, 0.0, 0.0],
            eye_offset_m: [0.0, -alt_m * 0.25, alt_m],
            fov_y_deg: 55.0,
            near_m: (alt_m * 0.002).clamp(1.0, 10_000.0),
        },
        viewport: Rect::new(0.0, 0.0, 1280.0, 800.0),
        seamless_25d: true,
        precipitation: PrecipitationConfig {
            enabled: true,
            mode: PrecipitationMode::Physical,
            ..PrecipitationConfig::default()
        },
        ..EarthInput::default()
    }
}

#[test]
fn isa_and_moist_adiabatic_lapse_rates_and_wet_bulb_are_physically_accurate() {
    // Sea-level ISA pressure = 1013.25 hPa, ~540 hPa at 5 km, ~226 hPa at 11 km tropopause.
    assert!((isa_pressure_hpa(0.0) - 1013.25).abs() < 0.01);
    let p_5km = isa_pressure_hpa(5_000.0);
    assert!(p_5km > 530.0 && p_5km < 550.0, "expected ~540 hPa at 5km, got {p_5km}");
    let p_11km = isa_pressure_hpa(11_000.0);
    assert!(p_11km > 220.0 && p_11km < 235.0, "expected ~226 hPa at 11km, got {p_11km}");

    // Moist adiabatic lapse rate is gentler in warm tropical air (~4-5 K/km) and approaches dry adiabatic (~9 K/km) when very cold.
    let moist_warm = moist_adiabatic_lapse_rate_k_per_km(25.0, 1000.0);
    let moist_cold = moist_adiabatic_lapse_rate_k_per_km(-30.0, 500.0);
    assert!(
        moist_warm > 3.5 && moist_warm < 5.0,
        "warm moist adiabatic lapse expected ~4 K/km, got {moist_warm}"
    );
    assert!(
        moist_cold > 8.0 && moist_cold < 9.8,
        "cold moist adiabatic lapse expected ~9 K/km, got {moist_cold}"
    );

    // Wet-bulb temperature equals dry-bulb at ~100% RH and cools strictly below dry-bulb at lower RH.
    let tw_sat = wet_bulb_temp_c(10.0, 1.0);
    let tw_dry = wet_bulb_temp_c(10.0, 0.35);
    assert!((tw_sat - 10.0).abs() < 0.35, "saturated Tw near T: {tw_sat}");
    assert!(tw_dry < 4.5, "dry air evaporative cooling expected <4.5C, got {tw_dry}");

    // ISA standard column with base_temp_c = +15°C at SeaLevel:
    // At 2,000 m MSL -> +15 - 6.5 * 2 = +2.0°C; above 11 km tropopause -> isothermal.
    let cfg_isa = AtmosphereConfig {
        base_temp_c: 15.0,
        anchor: TemperatureAnchor::SeaLevel,
        lapse_rate: LapseRateModel::StandardIsa,
        relative_humidity: 0.85,
        cloud_ceiling: CloudCeiling::FixedMsl(5_500.0),
        ..AtmosphereConfig::default()
    };
    let s_sea = cfg_isa.evaluate(45.0, 0.0, 0.0);
    let s_2km = cfg_isa.evaluate(45.0, 2_000.0, 0.0);
    let s_11km = cfg_isa.evaluate(45.0, 11_000.0, 0.0);
    let s_15km = cfg_isa.evaluate(45.0, 15_000.0, 0.0);
    assert!((s_sea.temperature_c - 15.0).abs() < 1e-6);
    assert!((s_2km.temperature_c - 2.0).abs() < 1e-6);
    assert!((s_11km.temperature_c - s_15km.temperature_c).abs() < 1e-6);
    assert_eq!(s_15km.lapse_rate_k_per_km, 0.0);
    assert!((s_sea.freezing_level_msl_m - (15.0 / 0.0065)).abs() < 1.0);

    // TerrainSurface anchor vs SeaLevel anchor over a 2,500 m plateau:
    let cfg_terrain = AtmosphereConfig {
        base_temp_c: 10.0,
        anchor: TemperatureAnchor::TerrainSurface,
        ..cfg_isa.clone()
    };
    let over_mtn_msl = cfg_isa.evaluate(45.0, 2_600.0, 2_500.0);
    let over_mtn_sfc = cfg_terrain.evaluate(45.0, 2_600.0, 2_500.0);
    assert!(
        over_mtn_msl.temperature_c < 0.0,
        "SeaLevel anchor at 2600m MSL with 15C base should be below freezing: {}",
        over_mtn_msl.temperature_c
    );
    assert!(
        (over_mtn_sfc.temperature_c - (10.0 - 0.65)).abs() < 0.05,
        "TerrainSurface anchor lapses only by 100m AGL: {}",
        over_mtn_sfc.temperature_c
    );

    // MoistAdiabatic column cools slower than ISA in warm low troposphere.
    let cfg_moist = AtmosphereConfig {
        base_temp_c: 20.0,
        lapse_rate: LapseRateModel::MoistAdiabatic,
        ..cfg_isa.clone()
    };
    let s_isa_2km = AtmosphereConfig {
        base_temp_c: 20.0,
        ..cfg_isa.clone()
    }
    .evaluate(20.0, 2_000.0, 0.0);
    let s_moist_2km = cfg_moist.evaluate(20.0, 2_000.0, 0.0);
    assert!(
        s_moist_2km.temperature_c > s_isa_2km.temperature_c + 1.5,
        "saturated warm air cools slower than 6.5 K/km ISA"
    );

    // Latitude climatology: equator warm, pole freezing.
    assert!(climatological_sea_level_temp_c(0.0) > 22.0);
    assert!(climatological_sea_level_temp_c(85.0) < -10.0);
}

#[test]
fn precipitation_phase_transitions_smoothly_from_rain_through_sleet_to_snow() {
    let warm = compute_precipitation_phase(6.0);
    assert_eq!(warm.rain, 1.0);
    assert_eq!(warm.sleet, 0.0);
    assert_eq!(warm.snow, 0.0);

    let cold = compute_precipitation_phase(-5.0);
    assert_eq!(cold.rain, 0.0);
    assert_eq!(cold.sleet, 0.0);
    assert_eq!(cold.snow, 1.0);

    // Walk finely across the [-1.5°C, +2.5°C] wet-bulb melting band and verify monotonicity + partition of unity.
    let mut prev_snow = 0.0f32;
    let mut saw_sleet = false;
    for step in 0..=40 {
        let tw = 3.0 - (step as f64) * 0.125;
        let phase = compute_precipitation_phase(tw);
        let sum = phase.rain + phase.sleet + phase.snow;
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "phase weights must sum to 1.0 at Tw={tw}, got {sum}"
        );
        assert!(
            phase.snow + 1e-5 >= prev_snow,
            "snow fraction must increase monotonically as air cools"
        );
        if phase.sleet > 0.25 {
            saw_sleet = true;
        }
        prev_snow = phase.snow;
    }
    assert!(saw_sleet, "melting band must emit sleet/mixed transition phase");
}

#[test]
fn ascending_through_freezing_level_transitions_rain_to_snow_and_stops_above_clouds() {
    let mut builder = RetainedTerrainBuilder::new();
    let mut src = MemoryTileSource::new();

    // Base sea-level temp = +12°C -> freezing level at ~1,846 m MSL; cloud ceiling at 5,500 m MSL.
    let mut input = low_alt_input(47.6, -122.3, 400.0);
    input.precipitation.atmosphere.base_temp_c = 12.0;
    input.precipitation.atmosphere.cloud_ceiling = CloudCeiling::FixedMsl(5_500.0);

    // 1. Low altitude (400 m MSL -> ~+9.4°C): fast 2D rain streaks (near, mid, far curtain shells).
    let (frame_rain, _) = builder.build_frame(&input, &mut src, 1, true);
    assert!(
        !frame_rain.particle_layers.is_empty(),
        "rain must emit near + far-field shells"
    );
    assert_eq!(
        frame_rain.particle_layers[0].sprite,
        RetainedParticleSprite::ProceduralStreak
    );
    assert_eq!(
        frame_rain.particle_layers[0].billboard,
        RetainedParticleBillboard::VelocityAligned
    );

    // 2. Melting layer (~1,650 m MSL -> dry ~+1.3°C, Tw ~+0.2°C): mixed rain + sleet + snow!
    input.camera.eye_offset_m = [0.0, -250.0, 1_650.0];
    let (frame_mix, _) = builder.build_frame(&input, &mut src, 2, true);
    assert!(
        frame_mix.particle_layers.len() >= 2,
        "melting layer must smoothly blend multiple hydrometeor layers, got {}",
        frame_mix.particle_layers.len()
    );

    // 3. Cold sub-cloud altitude (3,200 m MSL -> ~-8.8°C): mathematical 2D snowflakes + squall curtains.
    input.camera.eye_offset_m = [0.0, -300.0, 3_200.0];
    let (frame_snow, _) = builder.build_frame(&input, &mut src, 3, true);
    assert!(!frame_snow.particle_layers.is_empty());
    assert_eq!(
        frame_snow.particle_layers[0].sprite,
        RetainedParticleSprite::ProceduralSnowflake
    );
    assert_eq!(
        frame_snow.particle_layers[0].billboard,
        RetainedParticleBillboard::CameraFacing
    );

    // 4. Approaching cloud ceiling (5,250 m MSL inside [4,600..5,500] fade band): reduced opacity/count.
    input.camera.eye_offset_m = [0.0, -300.0, 5_250.0];
    let (frame_fade, _) = builder.build_frame(&input, &mut src, 4, true);
    assert!(!frame_fade.particle_layers.is_empty());
    assert!(
        frame_fade.particle_layers[0].color[3] < frame_snow.particle_layers[0].color[3],
        "opacity must smoothly fade as camera approaches cloud ceiling"
    );

    // 5. Above cloud ceiling (5,700 m MSL > 5,500 m): complete stop, zero GPU draws!
    input.camera.eye_offset_m = [0.0, -300.0, 5_700.0];
    let (frame_above, _) = builder.build_frame(&input, &mut src, 5, true);
    assert!(
        frame_above.particle_layers.is_empty(),
        "precipitation must stop completely above cloud ceiling"
    );
}

#[test]
fn panning_camera_right_shifts_toroidal_rain_phase_left_with_zero_drift() {
    let mut builder = RetainedTerrainBuilder::new();
    let mut src = MemoryTileSource::new();
    let mut input = low_alt_input(37.7749, -122.4194, 800.0);
    input.precipitation.mode = PrecipitationMode::Rain;
    input.precipitation.wind_mps = [0.0, 0.0]; // Isolate pure camera-pan parallax

    let (frame_a, _) = builder.build_frame(&input, &mut src, 1, true);
    assert!(!frame_a.particle_layers.is_empty());
    let phase_a = frame_a.particle_layers[0].wrapped_phase;
    let extent_xy = frame_a.particle_layers[0].volume_params[0];

    // Pan camera +12 m strictly East (camera pans right -> particles must shift left / -East!).
    input.camera.eye_offset_m[0] += 12.0;
    input.camera.aim_offset_m[0] += 12.0;
    let (frame_b, _) = builder.build_frame(&input, &mut src, 2, true);
    let phase_b = frame_b.particle_layers[0].wrapped_phase;

    // Signed shortest delta on periodic [0, 1) torus:
    let mut delta_e = phase_b[0] - phase_a[0];
    if delta_e > 0.5 {
        delta_e -= 1.0;
    } else if delta_e < -0.5 {
        delta_e += 1.0;
    }
    let expected_delta_e = -12.0 / extent_xy;
    assert!(
        (delta_e - expected_delta_e).abs() < 0.015,
        "panning camera +12m East must shift particles -12m/extent ({expected_delta_e:.4}), got {delta_e:.4}"
    );
    assert!(
        delta_e < 0.0,
        "camera panning right (+East) must move particles left (-East)"
    );
}

#[test]
fn custom_2d_texture_sprite_and_custom_evaluator_function_drive_gpu_layers() {
    let mut builder = RetainedTerrainBuilder::new();
    let mut src = MemoryTileSource::new();

    // Install a custom functional precipitation evaluator returning a custom 2D RGBA8 texture sprite.
    let tex_rgba = Arc::new(vec![255, 190, 210, 255, 255, 190, 210, 255, 255, 190, 210, 255, 255, 190, 210, 255]);
    let tex_for_eval = Arc::clone(&tex_rgba);
    let evaluator = Arc::new(move |ctx: &PrecipitationEvalContext<'_>| -> Vec<PrecipitationLayer> {
        if ctx.cloud_envelope <= 0.01 {
            return Vec::new();
        }
        vec![PrecipitationLayer::custom_texture(
            0xC057_0001,
            2,
            2,
            Arc::clone(&tex_for_eval),
            ctx.intensity,
        )
        .with_speed(4.5)
        .with_max_particles(900)]
    });

    assert!(builder.plugins().with_plugin_mut::<PrecipitationPlugin, _>("precipitation", |p| {
        p.set_evaluator(evaluator);
    }).is_some());

    let input = low_alt_input(35.68, 139.76, 500.0);
    let (frame, _) = builder.build_frame(&input, &mut src, 1, true);
    assert_eq!(frame.particle_layers.len(), 1);
    assert_eq!(
        frame.particle_layers[0].sprite,
        RetainedParticleSprite::Textured(0xC057_0001)
    );
    assert!(
        frame
            .particle_textures
            .iter()
            .any(|t| t.id == 0xC057_0001 && t.width == 2 && t.height == 2 && t.rgba8.as_slice() == tex_rgba.as_slice()),
        "custom 2D particle texture must be attached to RetainedTerrainFrame::particle_textures"
    );

    // Also verify declarative CustomLayers mode without a custom evaluator.
    let mut custom_input = low_alt_input(35.68, 139.76, 500.0);
    custom_input.precipitation.mode = PrecipitationMode::CustomLayers(vec![
        PrecipitationLayer::rain(1.0).with_speed(34.0),
        PrecipitationLayer::snow(0.6).with_billboard(ParticleBillboard::CameraFacing),
    ]);
    let mut default_builder = RetainedTerrainBuilder::new();
    let (frame_custom, _) = default_builder.build_frame(&custom_input, &mut src, 1, true);
    assert_eq!(frame_custom.particle_layers.len(), 2);
}

#[test]
fn open_meteo_vertical_sounding_parser_and_provider_interpolate_altitude_offline() {
    let mock_json = r#"{
        "elevation": 120.0,
        "current": {
            "temperature_2m": 11.0,
            "relative_humidity_2m": 88.0,
            "wind_speed_10m": 18.0,
            "wind_direction_10m": 270.0
        },
        "hourly": {
            "freezing_level_height": [1950.0],
            "temperature_1000hPa": [10.5],
            "geopotential_height_1000hPa": [150.0],
            "temperature_925hPa": [6.0],
            "geopotential_height_925hPa": [800.0],
            "temperature_850hPa": [1.5],
            "geopotential_height_850hPa": [1500.0],
            "temperature_700hPa": [-7.5],
            "geopotential_height_700hPa": [3000.0],
            "temperature_500hPa": [-22.0],
            "geopotential_height_500hPa": [5600.0]
        }
    }"#;

    let sounding = parse_open_meteo_sounding_json(mock_json).expect("valid Open-Meteo JSON");
    assert_eq!(sounding.levels.len(), 6);
    assert!((sounding.relative_humidity - 0.88).abs() < 1e-6);
    assert_eq!(sounding.freezing_level_msl_m, Some(1950.0));
    // Wind FROM 270° (West) at 18 km/h (= 5.0 m/s) blows TO +East `[+5.0, 0.0]`.
    let wind = sounding.wind_mps.expect("parsed surface wind");
    assert!((wind[0] - 5.0).abs() < 0.05 && wind[1].abs() < 0.05);

    let provider = OpenMeteoTemperatureProvider::new();
    provider.insert_cached_sounding(47.6, -122.3, sounding);
    let fallback = AtmosphereConfig::default();

    // Sample midpoint between 850 hPa (1500 m, +1.5°C) and 700 hPa (3000 m, -7.5°C) -> 2250 m = -3.0°C -> Snow!
    let sample_2250 = provider.sample(
        AtmosphereQuery {
            lat_deg: 47.6,
            lon_deg: -122.3,
            altitude_msl_m: 2_250.0,
            ground_elevation_m: 120.0,
        },
        &fallback,
    );
    assert!(sample_2250.source_live);
    assert!((sample_2250.temperature_c - (-3.0)).abs() < 0.05);
    assert!(sample_2250.phase.snow > 0.95);

    // Sample low elevation (200 m) -> warm rain!
    let sample_200 = provider.sample(
        AtmosphereQuery {
            lat_deg: 47.6,
            lon_deg: -122.3,
            altitude_msl_m: 200.0,
            ground_elevation_m: 120.0,
        },
        &fallback,
    );
    assert!(sample_200.temperature_c > 9.0);
    assert!(sample_200.phase.rain > 0.99);

    // Custom closure TemperatureProvider works seamlessly too.
    let fn_provider = FnTemperatureProvider(|q: AtmosphereQuery, cfg: &AtmosphereConfig| {
        let mut s = StaticAtmosphereProvider.sample(q, cfg);
        s.temperature_c = -12.0;
        s.wet_bulb_c = -12.5;
        s.phase = compute_precipitation_phase(s.wet_bulb_c);
        s
    });
    let fn_sample = fn_provider.sample(
        AtmosphereQuery {
            lat_deg: 0.0,
            lon_deg: 0.0,
            altitude_msl_m: 100.0,
            ground_elevation_m: 0.0,
        },
        &fallback,
    );
    assert_eq!(fn_sample.phase.snow, 1.0);
}

#[test]
fn earth_viewport_runs_fast_particle_ticks_and_exposes_live_atmosphere_sample() {
    let mut input = low_alt_input(37.7749, -122.4194, 700.0);
    input.precipitation.enabled = true;
    input.precipitation.mode = PrecipitationMode::Physical;
    input.precipitation.atmosphere.base_temp_c = 12.0;

    let mut vp = EarthViewport::new(&input, Box::new(MemoryTileSource::default()));
    vp.sync_now();

    assert!(
        vp.has_active_weather_animation(),
        "enabled precipitation below clouds should mark active weather animation"
    );
    assert!(vp.step_dynamic_overlays(0.016, true));

    let sample = vp
        .last_atmosphere_sample()
        .expect("viewport must expose latest evaluated AtmosphereSample");
    assert!(sample.temperature_c > 5.0 && sample.phase.rain > 0.9);

    let frame = vp
        .retained_scene()
        .snapshot_terrain()
        .expect("terrain frame snapshot");
    assert!(!frame.particle_layers.is_empty());
    assert_eq!(
        frame.particle_layers[0].sprite,
        RetainedParticleSprite::ProceduralStreak
    );

    // Switch base temp to -6°C and step -> immediately transitions to Snow!
    let mut cold_cfg = vp.precipitation_config().clone();
    cold_cfg.atmosphere.base_temp_c = -6.0;
    vp.set_precipitation_config(cold_cfg);
    let cold_sample = vp.last_atmosphere_sample().expect("updated cold sample");
    assert!(cold_sample.temperature_c < -8.0 && cold_sample.phase.snow > 0.99);
    let cold_frame = vp
        .retained_scene()
        .snapshot_terrain()
        .expect("cold terrain frame snapshot");
    assert!(!cold_frame.particle_layers.is_empty());
    assert_eq!(
        cold_frame.particle_layers[0].sprite,
        RetainedParticleSprite::ProceduralSnowflake
    );

    // Turning precipitation off clears particles and returns to 0% idle animation.
    vp.set_precipitation_enabled(false);
    let off_frame = vp
        .retained_scene()
        .snapshot_terrain()
        .expect("off frame snapshot");
    assert!(off_frame.particle_layers.is_empty());
    assert!(!vp.has_active_weather_animation());
}
