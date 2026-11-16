use std::f64::consts::PI;

use std::collections::HashMap;
use strata_earth::{
    build_25d_weather_meshes, cc0_cloud_mesh, cc0_rain_mesh, cc0_snow_mesh, embedded_bsc5_stars,
    embedded_clouds_fallback, embedded_land_mask, embedded_moon_texture,
    greenwich_mean_sidereal_time_rad, icrf_to_ecef_matrix, moon_direction_from_unix_utc,
    parse_glb_mesh, sun_direction_from_local_time, sun_direction_from_subsolar_lat_lon,
    sun_direction_from_system_clock, sun_direction_from_unix_utc, system_clock_time,
    unix_to_julian_date, EarthCamera, EarthDefaultTileSource, EarthInput, EarthViewport, Geodetic,
    LayerToggles, MemoryTileSource, RetainedTerrainBuilder, SpaceEnvironmentConfig,
    SunDirectionMode, SurfaceMode, TileProjection, WeatherConfig,
};
use strata_layout::Rect;

fn norm3(v: [f32; 3]) -> f32 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn sample_mask_at_lat_lon(lat_deg: f64, lon_deg: f64) -> u8 {
    let mask = embedded_land_mask();
    let u = ((lon_deg + 180.0) / 360.0).clamp(0.0, 0.9999);
    let v = ((90.0 - lat_deg) / 180.0).clamp(0.0, 0.9999);
    let px = ((u * mask.width as f64) as usize).min(mask.width as usize - 1);
    let py = ((v * mask.height as f64) as usize).min(mask.height as usize - 1);
    let idx = (py * mask.width as usize + px) * 4;
    mask.rgba8[idx]
}

#[test]
fn embedded_bsc5_catalog_has_8404_scientific_stars() {
    let stars = embedded_bsc5_stars();
    assert_eq!(stars.len(), 8404);
    let mut mag_min = f32::MAX;
    let mut mag_max = f32::MIN;
    let mut blue_count = 0usize;
    let mut red_count = 0usize;
    for s in stars.iter() {
        let len = norm3(s.icrf_dir);
        assert!(
            (len - 1.0).abs() < 1e-3,
            "star direction not normalized: {len}"
        );
        assert!(
            s.vmag <= 6.55 && s.vmag >= -1.6,
            "unexpected vmag {}",
            s.vmag
        );
        mag_min = mag_min.min(s.vmag);
        mag_max = mag_max.max(s.vmag);
        if s.color[2] > s.color[0] + 0.05 {
            blue_count += 1;
        }
        if s.color[0] > s.color[2] + 0.05 {
            red_count += 1;
        }
    }
    assert!(mag_min < 0.0 && mag_max > 6.0);
    assert!(
        blue_count > 500 && red_count > 500,
        "spectral diversity expected"
    );
}

#[test]
fn embedded_land_mask_accurately_separates_land_and_ocean() {
    let mask = embedded_land_mask();
    assert_eq!(mask.width, 1024);
    assert_eq!(mask.height, 512);
    assert_eq!(mask.rgba8.len(), 1024 * 512 * 4);

    // Sahara, Central Europe, Central Asia, Amazonia -> strictly landmass (>200).
    assert!(sample_mask_at_lat_lon(23.0, 12.0) > 200);
    assert!(sample_mask_at_lat_lon(50.0, 10.0) > 200);
    assert!(sample_mask_at_lat_lon(45.0, 75.0) > 200);
    assert!(sample_mask_at_lat_lon(-5.0, -60.0) > 200);

    // Central Pacific, North Atlantic, South Indian Ocean -> strictly ocean (<20).
    assert!(sample_mask_at_lat_lon(0.0, -140.0) < 20);
    assert!(sample_mask_at_lat_lon(35.0, -40.0) < 20);
    assert!(sample_mask_at_lat_lon(-30.0, 80.0) < 20);
}

#[test]
fn embedded_moon_and_cloud_fallback_textures_load_cleanly() {
    let moon = embedded_moon_texture();
    assert!(moon.width >= 512 && moon.height >= 256);
    assert_eq!(moon.rgba8.len(), (moon.width * moon.height * 4) as usize);

    let clouds = embedded_clouds_fallback();
    assert!(clouds.width >= 512 && clouds.height >= 256);
    assert_eq!(
        clouds.rgba8.len(),
        (clouds.width * clouds.height * 4) as usize
    );
    let non_zero_alpha = clouds
        .rgba8
        .iter()
        .skip(3)
        .step_by(4)
        .filter(|&&a| a > 20)
        .count();
    assert!(
        non_zero_alpha > 10_000,
        "cloud map should contain cloud cover"
    );
}

#[test]
fn solar_lunar_and_sidereal_ephemeris_are_accurate() {
    // J2000.0 epoch: 2000-01-01T12:00:00Z = unix 946728000.0 -> JD 2451545.0
    assert!((unix_to_julian_date(946_728_000.0) - 2_451_545.0).abs() < 1e-6);

    let gmst = greenwich_mean_sidereal_time_rad(946_728_000.0);
    assert!((0.0..2.0 * PI).contains(&gmst));

    let mat = icrf_to_ecef_matrix(946_728_000.0);
    let r0 = [mat[0][0], mat[1][0], mat[2][0]];
    let r1 = [mat[0][1], mat[1][1], mat[2][1]];
    let r2 = [mat[0][2], mat[1][2], mat[2][2]];
    assert!((norm3(r0) - 1.0).abs() < 1e-5);
    assert!((norm3(r1) - 1.0).abs() < 1e-5);
    assert!((norm3(r2) - 1.0).abs() < 1e-5);
    assert!(dot3(r0, r1).abs() < 1e-5);

    // Local time + UTC offset helper must agree with UTC conversion.
    let utc = 1_750_507_200.0; // June solstice 2025-06-21T12:00:00Z
    let offset_s = 3 * 3600;
    let sun_utc = sun_direction_from_unix_utc(utc);
    let sun_local = sun_direction_from_local_time(utc + offset_s as f64, offset_s);
    assert!(dot3(sun_utc, sun_local) > 0.999999);
    // At June solstice, Sun declination should be ~+23.4 deg -> z > 0.35.
    assert!(
        sun_utc[2] > 0.35,
        "expected positive summer declination z={}",
        sun_utc[2]
    );
    // At UTC noon, Sun should be roughly over prime meridian (+X).
    assert!(
        sun_utc[0] > 0.85,
        "expected +X prime meridian solar longitude at UTC noon"
    );

    let eq = sun_direction_from_subsolar_lat_lon(0.0, 0.0);
    assert!((eq[0] - 1.0).abs() < 1e-5 && eq[1].abs() < 1e-5 && eq[2].abs() < 1e-5);

    let sys = system_clock_time();
    assert!(sys.unix_utc_s > 1_600_000_000.0);
    let sys_sun = sun_direction_from_system_clock();
    assert!((norm3(sys_sun) - 1.0).abs() < 1e-4);

    let moon = moon_direction_from_unix_utc(utc);
    assert!((norm3(moon) - 1.0).abs() < 1e-4);

    let cfg = SpaceEnvironmentConfig::default().with_local_time(utc + 7200.0, 7200);
    match cfg.sun_direction {
        SunDirectionMode::LocalTimeOffset {
            local_unix_s,
            utc_offset_s,
        } => {
            assert!((local_unix_s - (utc + 7200.0)).abs() < 1e-6);
            assert_eq!(utc_offset_s, 7200);
        }
        other => panic!("unexpected mode: {other:?}"),
    }
}

#[test]
fn cc0_glb_weather_models_parse_and_spawn_only_in_25d() {
    let cloud = cc0_cloud_mesh();
    let rain = cc0_rain_mesh();
    let snow = cc0_snow_mesh();
    assert!(cloud.positions.len() >= 12 && !cloud.indices.is_empty());
    assert!(rain.positions.len() >= 12 && !rain.indices.is_empty());
    assert!(snow.positions.len() >= 12 && !snow.indices.is_empty());

    // Build a tiny valid synthetic GLB in memory to verify custom GLB loading too.
    let mut bin = Vec::new();
    let pos: [[f32; 3]; 3] = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    for p in pos {
        for f in p {
            bin.extend_from_slice(&f.to_le_bytes());
        }
    }
    let norm: [[f32; 3]; 3] = [[0.0, 0.0, 1.0]; 3];
    for n in norm {
        for f in n {
            bin.extend_from_slice(&f.to_le_bytes());
        }
    }
    let idx: [u16; 3] = [0, 1, 2];
    for i in idx {
        bin.extend_from_slice(&i.to_le_bytes());
    }
    while bin.len() % 4 != 0 {
        bin.push(0);
    }
    let json_str = format!(
        r#"{{"asset":{{"version":"2.0"}},"meshes":[{{"primitives":[{{"attributes":{{"POSITION":0,"NORMAL":1}},"indices":2}}]}}],"accessors":[{{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3"}},{{"bufferView":1,"componentType":5126,"count":3,"type":"VEC3"}},{{"bufferView":2,"componentType":5123,"count":3,"type":"SCALAR"}}],"bufferViews":[{{"buffer":0,"byteOffset":0,"byteLength":36}},{{"buffer":0,"byteOffset":36,"byteLength":36}},{{"buffer":0,"byteOffset":72,"byteLength":6}}],"buffers":[{{"byteLength":{}}}]}}"#,
        bin.len()
    );
    let mut json_bytes = json_str.into_bytes();
    while json_bytes.len() % 4 != 0 {
        json_bytes.push(b' ');
    }
    let total_len = 12 + 8 + json_bytes.len() + 8 + bin.len();
    let mut glb = Vec::with_capacity(total_len);
    glb.extend_from_slice(&0x46546C67u32.to_le_bytes());
    glb.extend_from_slice(&2u32.to_le_bytes());
    glb.extend_from_slice(&(total_len as u32).to_le_bytes());
    glb.extend_from_slice(&(json_bytes.len() as u32).to_le_bytes());
    glb.extend_from_slice(&0x4E4F534Au32.to_le_bytes());
    glb.extend_from_slice(&json_bytes);
    glb.extend_from_slice(&(bin.len() as u32).to_le_bytes());
    glb.extend_from_slice(&0x004E4942u32.to_le_bytes());
    glb.extend_from_slice(&bin);

    let custom_tri = parse_glb_mesh(&glb).expect("minimal GLB should parse");
    assert_eq!(custom_tri.positions.len(), 3);
    assert_eq!(custom_tri.indices, vec![0, 1, 2]);

    let weather = WeatherConfig {
        clouds_25d: true,
        rain: true,
        snow: true,
        cloud_count: 6,
        rain_density: 40,
        snow_density: 30,
        ..WeatherConfig::default()
    };
    let focus = Geodetic::new(47.6, -122.3, 0.0);
    let mut cache = HashMap::new();

    // Must be empty in 3D mode (`local_25d_blend == 0.0`).
    let m_3d = build_25d_weather_meshes(focus, 10_000_000.0, 0.0, &weather, None, 1, &mut cache);
    assert!(
        m_3d.is_empty(),
        "2.5D weather must not appear in full 3D globe mode"
    );

    // Must emit 3 distinct CC0 layers when in 2.5D mode.
    let m_25d = build_25d_weather_meshes(focus, 1_800.0, 1.0, &weather, None, 2, &mut cache);
    assert_eq!(m_25d.len(), 3);
    for layer in &m_25d {
        assert!(!layer.vertices.is_empty() && !layer.indices.is_empty());
    }

    // Disabling rain should leave only clouds and snow.
    let mut weather_no_rain = weather.clone();
    weather_no_rain.rain = false;
    let m_filtered =
        build_25d_weather_meshes(focus, 1_800.0, 1.0, &weather_no_rain, None, 3, &mut cache);
    assert_eq!(m_filtered.len(), 2);
}

#[test]
fn transitions_fade_terminator_city_lights_and_clouds_in_25d_and_toggles_work() {
    let mut input_3d = EarthInput {
        camera: EarthCamera {
            focus: Geodetic::new(28.0, 12.0, 0.0),
            aim_offset_m: [0.0, 0.0, 0.0],
            eye_offset_m: [0.0, 0.0, 10_200_000.0],
            fov_y_deg: 55.0,
            near_m: 1_000.0,
        },
        viewport: Rect::new(0.0, 0.0, 1280.0, 800.0),
        seamless_25d: true,
        tile_projection: TileProjection::WebMercator,
        surface_mode: SurfaceMode::QuadtreeLod,
        layers: LayerToggles {
            clouds: true,
            celestial: true,
            ..LayerToggles::default()
        },
        ..EarthInput::default()
    };

    let cam_3d = RetainedTerrainBuilder::camera_from_input(&input_3d);
    assert!(cam_3d.day_night_strength > 0.9);
    assert!(cam_3d.city_lights_intensity > 0.9);
    assert!(cam_3d.cloud_layer_opacity > 0.5);
    assert!(cam_3d.stars_visible);
    assert!(cam_3d.sun_visible);
    assert!(cam_3d.moon_visible);

    // Mid-transition at 135 km altitude (halfway between 250 km and 20 km) -> smoothly reducing contrast.
    input_3d.camera.eye_offset_m = [0.0, 0.0, 135_000.0];
    let cam_mid = RetainedTerrainBuilder::camera_from_input(&input_3d);
    assert!(cam_mid.local_25d_blend > 0.05 && cam_mid.local_25d_blend < 0.95);
    assert!(cam_mid.day_night_strength < cam_3d.day_night_strength);
    assert!(cam_mid.city_lights_intensity < cam_3d.city_lights_intensity);

    // Full 2.5D close to ground (1.5 km) -> shadowless, no city lights, no 3D cloud shell, 2.5D weather present.
    let mut input_25d = input_3d.clone();
    input_25d.camera.eye_offset_m = [0.0, -400.0, 1_500.0];
    input_25d.weather.rain = true;
    input_25d.weather.snow = true;
    let cam_25d = RetainedTerrainBuilder::camera_from_input(&input_25d);
    assert!(cam_25d.local_25d_blend > 0.98);
    assert!(
        cam_25d.day_night_strength < 1e-4,
        "shadows must disappear in 2.5D"
    );
    assert!(
        cam_25d.city_lights_intensity < 1e-4,
        "city lights must disappear in 2.5D"
    );
    assert!(
        cam_25d.cloud_layer_opacity < 1e-4,
        "3D cloud shell must fade out in 2.5D"
    );

    let mut builder = RetainedTerrainBuilder::new();
    let mut src = MemoryTileSource::new();
    let (frame_3d, _) = builder.build_frame(
        &EarthInput {
            camera: EarthCamera {
                eye_offset_m: [0.0, 0.0, 10_200_000.0],
                ..input_3d.camera
            },
            ..input_3d.clone()
        },
        &mut src,
        1,
        true,
    );
    assert!(frame_3d.environment.is_some());
    assert!(
        frame_3d
            .dynamic_meshes
            .iter()
            .all(|m| !(0x9E47_0001..=0x9E47_0003).contains(&m.id)),
        "2.5D weather meshes must remain absent in 3D"
    );
    assert!(
        frame_3d
            .dynamic_meshes
            .iter()
            .any(|m| (0x9E47_1001..=0x9E47_1003).contains(&m.id)),
        "orbital satellite meshes should appear in 3D space"
    );

    let (frame_25d, _) = builder.build_frame(&input_25d, &mut src, 2, true);
    assert_eq!(frame_25d.dynamic_meshes.len(), 3);
    assert!(
        frame_25d
            .dynamic_meshes
            .iter()
            .all(|m| (0x9E47_0001..=0x9E47_0003).contains(&m.id)),
        "only 2.5D weather meshes should remain after orbital satellites fade away in 2.5D"
    );

    // Verify all toggles can disable cleanly even in 3D space.
    input_3d.camera.eye_offset_m = [0.0, 0.0, 10_200_000.0];
    input_3d.space.day_night_lighting = false;
    input_3d.space.city_lights = false;
    input_3d.space.stars_visible = false;
    input_3d.space.sun_visible = false;
    input_3d.space.moon_visible = false;
    input_3d.weather.global_cloud_map = false;
    let cam_off = RetainedTerrainBuilder::camera_from_input(&input_3d);
    assert_eq!(cam_off.day_night_strength, 0.0);
    assert_eq!(cam_off.city_lights_intensity, 0.0);
    assert_eq!(cam_off.cloud_layer_opacity, 0.0);
    assert!(!cam_off.stars_visible);
    assert!(!cam_off.sun_visible);
    assert!(!cam_off.moon_visible);

    // Verify EarthViewport API & calm idle 0% CPU check.
    let mut vp = EarthViewport::new(
        &input_3d,
        Box::new(EarthDefaultTileSource::offline_basemap()),
    );
    vp.sync_now();
    assert!(
        !vp.has_active_weather_animation(),
        "3D calm orbital mode should stay 0% idle"
    );
    assert!(!vp.step_weather(0.016));

    vp.set_space_config(SpaceEnvironmentConfig {
        day_night_lighting: true,
        city_lights: true,
        ..SpaceEnvironmentConfig::default()
    });
    assert!(vp.retained_scene().camera_generation() > 0);
}

#[test]
fn terrain_tiles_project_on_screen_in_25d_mode() {
    let tilt_rad = 48.0f64.to_radians();
    let head_rad = 6.0f64.to_radians();
    let range_m = 9_500.0;
    let horiz = range_m * tilt_rad.sin();
    let up = range_m * tilt_rad.cos();
    let input = EarthInput {
        camera: EarthCamera {
            focus: Geodetic::new(37.7749, -122.4194, 0.0),
            aim_offset_m: [0.0, 0.0, 0.0],
            eye_offset_m: [-head_rad.sin() * horiz, -head_rad.cos() * horiz, up],
            fov_y_deg: 55.0,
            near_m: (range_m * 0.002).clamp(2.0, 20_000.0),
        },
        viewport: Rect::new(0.0, 0.0, 1280.0, 800.0),
        view_mode: strata_earth::EarthViewMode::Mode3D,
        seamless_25d: true,
        tile_projection: TileProjection::WebMercator,
        surface_mode: SurfaceMode::QuadtreeLod,
        tile_grid: 9,
        max_tile_level: 16,
        sse_threshold: 4.5,
        ..EarthInput::default()
    };

    let mut builder = RetainedTerrainBuilder::new();
    let mut source = EarthDefaultTileSource::offline_basemap();
    let (frame, _stats) = builder.build_frame(&input, &mut source, 1, true);
    assert!(
        !frame.instances.is_empty(),
        "expected visible 2.5D tiles at SF Bay"
    );
    assert!(
        !frame.textures.is_empty(),
        "expected ancestor basemap textures resolved for 2.5D tiles"
    );

    let eye = frame.camera.eye_ecef;
    let vp = frame.camera.view_proj_rte;
    let mut on_screen_vertices = 0usize;
    for inst in frame.instances.iter() {
        let Some(mesh) = frame.meshes.iter().find(|m| m.key == inst.key) else {
            continue;
        };
        let cx = (inst.center_ecef[0] - eye[0]) as f32;
        let cy = (inst.center_ecef[1] - eye[1]) as f32;
        let cz = (inst.center_ecef[2] - eye[2]) as f32;
        for v in mesh.vertices.iter() {
            let rx = v.pos_u[0] + cx;
            let ry = v.pos_u[1] + cy;
            let rz = v.pos_u[2] + cz;
            let clip_x = vp[0][0] * rx + vp[1][0] * ry + vp[2][0] * rz + vp[3][0];
            let clip_y = vp[0][1] * rx + vp[1][1] * ry + vp[2][1] * rz + vp[3][1];
            let clip_z = vp[0][2] * rx + vp[1][2] * ry + vp[2][2] * rz + vp[3][2];
            let clip_w = vp[0][3] * rx + vp[1][3] * ry + vp[2][3] * rz + vp[3][3];
            if clip_w > 0.0 {
                let ndc_x = clip_x / clip_w;
                let ndc_y = clip_y / clip_w;
                let ndc_z = clip_z / clip_w;
                if ndc_x.abs() <= 1.0 && ndc_y.abs() <= 1.0 && (0.0..=1.0).contains(&ndc_z) {
                    on_screen_vertices += 1;
                }
            }
        }
    }
    assert!(
        on_screen_vertices > 20,
        "expected terrain vertices inside NDC clip volume"
    );
}

#[test]
fn embedded_basemap_provides_all_z0_to_z4_highres_tiles() {
    let mut total = 0usize;
    for level in 0..=4u8 {
        let n = 1u32 << level;
        for y in 0..n {
            for x in 0..n {
                let tile =
                    strata_earth::embedded_base_tile(strata_earth::TileKey::new(level, x, y))
                        .expect("z=0..4 NASA Blue Marble tile must be built in");
                assert_eq!(tile.width, 256);
                assert_eq!(tile.height, 256);
                total += 1;
            }
        }
    }
    assert_eq!(total, 1 + 4 + 16 + 64 + 256);
}

#[test]
fn real_orbital_satellites_parse_propagate_and_fade_in_25d() {
    let sample_gp = r#"[{"OBJECT_NAME":"ISS (ZARYA)","EPOCH":"2026-11-16T06:00:00.000000","MEAN_MOTION":15.498,"ECCENTRICITY":0.00051,"INCLINATION":51.64,"RA_OF_ASC_NODE":120.0,"ARG_OF_PERICENTER":70.0,"MEAN_ANOMALY":40.0}]"#;
    let elems = strata_earth::parse_celestrak_gp_json(sample_gp, true);
    assert_eq!(elems.len(), 1);
    let iss = strata_earth::propagate_satellite_ecef(&elems[0], 1_767_225_600.0, [1.0, 0.0, 0.0]);
    assert!(
        iss.altitude_m > 350_000.0 && iss.altitude_m < 500_000.0,
        "ISS orbital altitude should sit in LEO (~420 km), got {}",
        iss.altitude_m
    );

    let embedded = strata_earth::embedded_satellite_elements();
    assert!(embedded.len() >= 30);
    let meshes_3d = strata_earth::build_3d_satellites_meshes(
        1_767_225_600.0,
        [1.0, 0.0, 0.0],
        12_000_000.0,
        0.0,
        true,
        true,
        95_000.0,
        &embedded,
        1,
    );
    assert!(!meshes_3d.is_empty());

    let meshes_25d = strata_earth::build_3d_satellites_meshes(
        1_767_225_600.0,
        [1.0, 0.0, 0.0],
        9_500.0,
        1.0,
        true,
        true,
        95_000.0,
        &embedded,
        2,
    );
    assert!(
        meshes_25d.is_empty(),
        "satellites must completely fade out in 2.5D"
    );
}

#[test]
fn panning_in_25d_reuses_tile_meshes_and_ancestor_elevation_prevents_cliff_drops() {
    let mut src = MemoryTileSource::new();
    let img = strata_earth::TileImage {
        width: 4,
        height: 4,
        pixels: [120, 150, 120, 255].repeat(16),
    };
    let parent_tile = strata_earth::TileKey::new(11, 327, 791);
    src.insert_imagery(strata_earth::TileKey::new(0, 0, 0), img);
    src.insert_elevation(
        parent_tile,
        strata_earth::TileElevation {
            width: 4,
            height: 4,
            heights_m: vec![420.0; 16],
        },
    );

    let cam_a = EarthCamera {
        focus: Geodetic::new(37.7749, -122.4194, 0.0),
        aim_offset_m: [0.0, 0.0, 0.0],
        eye_offset_m: [0.0, -5_000.0, 8_000.0],
        fov_y_deg: 55.0,
        near_m: 10.0,
    };
    let mut input = EarthInput {
        camera: cam_a,
        surface_mode: SurfaceMode::QuadtreeLod,
        tile_projection: TileProjection::WebMercator,
        elevation_enabled: true,
        max_tile_level: 14,
        tile_grid: 5,
        sse_threshold: 8.0,
        ..EarthInput::default()
    };

    let mut builder = RetainedTerrainBuilder::new();
    let (frame_a, _) = builder.build_frame(&input, &mut src, 1, true);
    assert!(!frame_a.meshes.is_empty());

    // Pan slightly (~300 m) inside 2.5D: cached meshes should all hit without full rebuild.
    input.camera.focus = Geodetic::new(37.7775, -122.4165, 0.0);
    let (_frame_b, stats_b) = builder.build_frame(&input, &mut src, 2, true);
    assert!(
        stats_b.mesh_cache_hits > 0,
        "2.5D small pan must reuse cached tile meshes"
    );

    // Toggling heightmap off drops relief cleanly back to ellipsoid.
    input.elevation_enabled = false;
    let (frame_flat, _) = builder.build_frame(&input, &mut src, 3, true);
    assert!(!frame_flat.meshes.is_empty());
}
