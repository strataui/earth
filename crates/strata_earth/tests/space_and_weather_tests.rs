use std::f64::consts::PI;

use strata_earth::{
    embedded_bsc5_stars, embedded_clouds_fallback, embedded_land_mask, embedded_moon_texture,
    greenwich_mean_sidereal_time_rad, icrf_to_ecef_matrix, moon_direction_from_unix_utc,
    parse_glb_mesh, sun_direction_from_local_time, sun_direction_from_subsolar_lat_lon,
    sun_direction_from_system_clock, sun_direction_from_unix_utc, system_clock_time,
    unix_to_julian_date, EarthCamera, EarthDefaultTileSource, EarthInput, EarthViewport, Geodetic,
    LayerToggles, MemoryTileSource, PrecipitationConfig, PrecipitationMode, RetainedTerrainBuilder,
    SpaceEnvironmentConfig, SunDirectionMode, SurfaceMode, TileProjection, WeatherConfig,
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
    assert_eq!(mask.width, 4096);
    assert_eq!(mask.height, 2048);
    assert_eq!(mask.rgba8.len(), 4096 * 2048 * 4);

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
fn glb_parser_and_precipitation_layers_spawn_only_below_cloud_ceiling() {
    // Build a tiny valid synthetic GLB in memory to verify custom GLB loading for EarthGeometry::glb.
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

    let mut builder = RetainedTerrainBuilder::new();
    let mut src = MemoryTileSource::new();
    let mut input = EarthInput {
        camera: EarthCamera {
            focus: Geodetic::new(47.6, -122.3, 0.0),
            aim_offset_m: [0.0, 0.0, 0.0],
            eye_offset_m: [0.0, 0.0, 10_000_000.0],
            fov_y_deg: 55.0,
            near_m: 1_000.0,
        },
        precipitation: PrecipitationConfig {
            enabled: true,
            mode: PrecipitationMode::Rain,
            ..PrecipitationConfig::default()
        },
        ..EarthInput::default()
    };

    // Must emit zero precipitation layers in full 3D orbital space above the cloud ceiling.
    let (frame_space, _) = builder.build_frame(&input, &mut src, 1, true);
    assert!(
        frame_space.particle_layers.is_empty(),
        "precipitation must stop completely above the cloud ceiling"
    );

    // Must emit GPU-instanced 2D particle layers below the cloud ceiling with zero CPU weather meshes.
    input.camera.eye_offset_m = [0.0, -400.0, 1_800.0];
    let (frame_low, _) = builder.build_frame(&input, &mut src, 2, true);
    assert_eq!(frame_low.particle_layers.len(), 1);
    assert!(frame_low.particle_layers[0].particle_count > 0);
    assert!(
        frame_low
            .dynamic_meshes
            .iter()
            .all(|m| !(0x9E47_0001..=0x9E47_0003).contains(&m.id)),
        "legacy 3D weather meshes must never spawn"
    );

    // Disabling precipitation clears all particle layers immediately.
    input.precipitation.enabled = false;
    let (frame_off, _) = builder.build_frame(&input, &mut src, 3, true);
    assert!(frame_off.particle_layers.is_empty());
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

    // Full 2.5D close to ground (1.5 km) -> shadowless, no city lights, no 3D cloud shell, GPU precipitation active.
    let mut input_25d = input_3d.clone();
    input_25d.camera.eye_offset_m = [0.0, -400.0, 1_500.0];
    input_25d.precipitation = PrecipitationConfig {
        enabled: true,
        mode: PrecipitationMode::Physical,
        ..PrecipitationConfig::default()
    };
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
    assert!(
        cam_25d.local_cloud_opacity > 0.5,
        "cropped overhead satellite cloud canopy should remain active in 2.5D, got {}",
        cam_25d.local_cloud_opacity
    );

    let mut input_2d = input_3d.clone();
    input_2d.view_mode = strata_earth::EarthViewMode::Mode2D;
    input_2d.view_morph = 1.0;
    let cam_2d = RetainedTerrainBuilder::camera_from_input(&input_2d);
    assert!(!cam_2d.moon_visible, "Moon must be hidden in 2D map mode");
    assert!(!cam_2d.sun_visible, "Sun must be hidden in 2D map mode");
    assert!(!cam_2d.stars_visible, "Stars must be hidden in 2D map mode");
    assert_eq!(cam_2d.skybox_intensity, 0.0);
    assert_eq!(cam_2d.cloud_layer_opacity, 0.0);
    assert_eq!(cam_2d.local_cloud_opacity, 0.0);
    assert_eq!(cam_2d.day_night_strength, 0.0);
    match cam_2d.background {
        strata_render::scene3d::Background3D::VerticalGradient { top, bottom } => {
            assert!(top.r() == 0.0 && top.g() == 0.0 && top.b() == 0.0);
            assert!(bottom.r() == 0.0 && bottom.g() == 0.0 && bottom.b() == 0.0);
        }
        _ => {}
    }

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
        frame_3d.particle_layers.is_empty(),
        "precipitation layers must remain absent in 3D orbital space"
    );
    assert!(
        frame_3d
            .dynamic_meshes
            .iter()
            .any(|m| (0x9E47_1001..=0x9E47_1003).contains(&m.id)),
        "orbital satellite meshes should appear in 3D space"
    );

    let (frame_25d, _) = builder.build_frame(&input_25d, &mut src, 2, true);
    assert!(
        frame_25d.dynamic_meshes.is_empty(),
        "2.5D precipitation uses GPU instanced particle layers, never CPU dynamic meshes"
    );
    assert!(
        !frame_25d.particle_layers.is_empty(),
        "2.5D physical precipitation should emit GPU particle layers below cloud ceiling"
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

#[test]
fn embedded_black_marble_lights_europe_and_keeps_oceans_dark() {
    let mask = embedded_land_mask();
    // Sample central Europe/Paris-Brussels corridor (~48.8N, 2.35E) and Pacific Ocean (0N, -140W)
    let sample_g = |lat_deg: f64, lon_deg: f64| -> u8 {
        let u = ((lon_deg + 180.0) / 360.0).clamp(0.0, 0.9999);
        let v = ((90.0 - lat_deg) / 180.0).clamp(0.0, 0.9999);
        let x = (u * mask.width as f64) as usize;
        let y = (v * mask.height as f64) as usize;
        mask.rgba8[(y * mask.width as usize + x) * 4 + 1]
    };
    let paris_lum = sample_g(48.85, 2.35);
    let pacific_lum = sample_g(0.0, -140.0);
    assert!(
        paris_lum > 40,
        "Paris should show bright NASA Black Marble night luminance, got {paris_lum}"
    );
    assert_eq!(
        pacific_lum, 0,
        "Open Pacific Ocean must have zero night lights"
    );
}

#[test]
fn google_earth_labels_project_fade_and_parse_highways_and_mountains() {
    let all = strata_earth::builtin_world_labels();
    assert!(all
        .iter()
        .any(|l| l.text == "CT05" && l.kind == strata_earth::EarthLabelKind::Highway));
    assert!(all
        .iter()
        .any(|l| l.text == "Mount Tamalpais" && l.kind == strata_earth::EarthLabelKind::Mountain));

    // Verify light green color + MountainRight icon for mountains
    let (m_rgb, m_icon, _) = strata_earth::label_style(strata_earth::EarthLabelKind::Mountain);
    assert!(m_rgb[1] > m_rgb[0] && m_rgb[1] > m_rgb[2]);
    assert_eq!(m_icon, strata_earth::EarthLabelIcon::MountainRight);

    // Space altitude shows continents/oceans/countries
    let mut input = EarthInput {
        camera: EarthCamera {
            focus: Geodetic::new(37.7749, -122.4194, 0.0),
            aim_offset_m: [0.0, 0.0, 0.0],
            eye_offset_m: [0.0, -10.0, 11_000_000.0],
            fov_y_deg: 55.0,
            near_m: 1_000.0,
        },
        layers: strata_earth::LayerToggles {
            labels: true,
            ..Default::default()
        },
        ..EarthInput::default()
    };
    let space_vis = strata_earth::project_earth_labels(&input, &all, 32);
    assert!(space_vis.iter().any(|l| {
        matches!(
            l.kind,
            strata_earth::EarthLabelKind::Continent
                | strata_earth::EarthLabelKind::Country
                | strata_earth::EarthLabelKind::Ocean
        )
    }));

    // Zoomed local view shows peaks and highway shields with smooth alpha
    input.camera.eye_offset_m = [0.0, -8_000.0, 22_000.0];
    let local_vis = strata_earth::project_earth_labels(&input, &all, 32);
    assert!(local_vis
        .iter()
        .any(|l| l.kind == strata_earth::EarthLabelKind::Highway
            || l.kind == strata_earth::EarthLabelKind::Mountain));

    // Live Overpass JSON parser extracts CT05 / peaks / places
    let overpass = r#"{"elements":[{"type":"way","center":{"lat":41.3,"lon":-72.9},"tags":{"highway":"motorway","ref":"CT05"}},{"type":"node","lat":37.9,"lon":-122.59,"tags":{"natural":"peak","name":"East Peak","ele":"784"}}]}"#;
    let parsed = strata_earth::parse_overpass_labels_json(overpass);
    assert_eq!(parsed.len(), 2);
    assert_eq!(parsed[0].text, "CT05");
    assert_eq!(parsed[0].kind, strata_earth::EarthLabelKind::Highway);
    assert_eq!(parsed[1].text, "East Peak");
    assert_eq!(parsed[1].kind, strata_earth::EarthLabelKind::Mountain);
}

#[test]
fn healpix_cubemap_projection_and_ssky_roundtrip() {
    use std::sync::Arc;
    use strata_earth::{
        blend_composite_cubemap, cubemap_face_dir, decode_ssky, encode_ssky_jpeg, encode_ssky_png,
        healpix_nest_index, icrs_to_galactic, vec_to_healpix_face_uv, SpaceSkyboxQuality,
        SpaceSkyboxSurvey,
    };

    // Galactic North Pole J2000 direction should transform close to (0, 0, +1) in Galactic frame.
    let ngp_icrs = [-0.867_666_149, -0.198_076_373, 0.455_983_776];
    let ngp_gal = icrs_to_galactic(ngp_icrs);
    assert!(ngp_gal[2] > 0.999_9, "expected NGP z ~ +1, got {:?}", ngp_gal);

    // Galactic Center (1, 0, 0) lands on HEALPix base face 4!
    let (bf, _u, _v) = vec_to_healpix_face_uv([1.0, 0.0, 0.0]);
    assert_eq!(bf, 4);
    assert_eq!(healpix_nest_index(4, 0, 0, 1), 16);
    assert_eq!(healpix_nest_index(4, 1, 0, 1), 17);

    // Cubemap face direction covers all 6 axis directions cleanly.
    assert!(cubemap_face_dir(0, 4, 4, 8)[0] > 0.9);
    assert!(cubemap_face_dir(1, 4, 4, 8)[0] < -0.9);
    assert!(cubemap_face_dir(2, 4, 4, 8)[1] > 0.9);
    assert!(cubemap_face_dir(3, 4, 4, 8)[1] < -0.9);
    assert!(cubemap_face_dir(4, 4, 4, 8)[2] > 0.9);
    assert!(cubemap_face_dir(5, 4, 4, 8)[2] < -0.9);

    // PNG & JPEG .ssky encode/decode round-trip + composite blending.
    let face_size = 16u32;
    let gaia: [Arc<Vec<u8>>; 6] =
        std::array::from_fn(|i| Arc::new([20 + (i as u8) * 10, 30, 50, 255].repeat(256)));
    let wise: [Arc<Vec<u8>>; 6] = std::array::from_fn(|_| Arc::new([180, 90, 40, 255].repeat(256)));
    let comp = blend_composite_cubemap(&gaia, None, Some(&wise));
    assert!(comp[0][0] > gaia[0][0], "AllWISE highlight should boost composite red");

    let png_bytes = encode_ssky_png(
        face_size,
        SpaceSkyboxSurvey::CompositeOptical,
        SpaceSkyboxQuality::Low,
        &comp,
    )
    .expect("encode ssky png");
    let (dec_size, dec_faces) = decode_ssky(&png_bytes).expect("decode ssky png");
    assert_eq!(dec_size, face_size);
    assert_eq!(dec_faces[0].as_slice(), comp[0].as_slice());

    let jpg_bytes = encode_ssky_jpeg(
        face_size,
        SpaceSkyboxSurvey::AllWiseInfrared,
        SpaceSkyboxQuality::Low,
        &wise,
        92,
    )
    .expect("encode ssky jpeg");
    let (j_size, j_faces) = decode_ssky(&jpg_bytes).expect("decode ssky jpeg");
    assert_eq!(j_size, face_size);
    assert_eq!(j_faces[2][3], 255);
}

#[test]
fn rolling_assets_and_legacy_fallback_caches_atomically() {
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::Arc;
    use std::time::{Duration, SystemTime, UNIX_EPOCH};
    use strata_earth::{
        encode_ssky_png, sha256_hex, versioned_legacy_filename, AssetManager, AssetManifest,
        AssetManifestEntry, AssetSourceConfig, SpaceSkyboxQuality, SpaceSkyboxStore,
        SpaceSkyboxSurvey,
    };

    // 1. Verify default URL chain ordering (CDN assets -> GitHub assets -> CDN legacy -> GitHub legacy).
    let def_cfg = AssetSourceConfig::default();
    let legacy = versioned_legacy_filename("skybox-composite-ultra.ssky", 1);
    assert_eq!(legacy, "skybox-composite-ultra-v1.ssky");
    let urls = def_cfg.candidate_urls_for("skybox-composite-ultra.ssky", Some(&legacy));
    assert_eq!(
        urls,
        vec![
            "https://cdn.strata.larvance.com/earth/assets/skybox-composite-ultra.ssky",
            "https://github.com/strataui/earth/releases/download/assets/skybox-composite-ultra.ssky",
            "https://cdn.strata.larvance.com/earth/assets-legacy/skybox-composite-ultra-v1.ssky",
            "https://github.com/strataui/earth/releases/download/assets-legacy/skybox-composite-ultra-v1.ssky",
        ]
    );

    // 2. Spin up local mock HTTP server serving wrong SHA on /assets/ and valid pinned SHA on /assets-legacy/.
    let faces: [Arc<Vec<u8>>; 6] = std::array::from_fn(|i| Arc::new([40 + (i as u8) * 15, 60, 120, 255].repeat(64)));
    let valid_ssky = encode_ssky_png(
        8,
        SpaceSkyboxSurvey::GaiaDr3Optical,
        SpaceSkyboxQuality::Low,
        &faces,
    )
    .unwrap();
    let expected_sha = sha256_hex(&valid_ssky);

    let mut manifest = AssetManifest {
        schema_version: 1,
        entries: std::collections::HashMap::new(),
    };
    let fname = SpaceSkyboxSurvey::GaiaDr3Optical.cdn_filename(SpaceSkyboxQuality::Low);
    let leg_name = versioned_legacy_filename(&fname, 1);
    manifest.entries.insert(
        fname.clone(),
        AssetManifestEntry {
            key: fname.clone(),
            filename: fname.clone(),
            version: 1,
            sha256: Some(expected_sha.clone()),
            bytes: valid_ssky.len() as u64,
            legacy_filename: Some(leg_name.clone()),
            license: Some("ESA/Gaia/DPAC".to_string()),
        },
    );
    let parsed = AssetManifest::from_json_str(&manifest.to_json_string()).expect("manifest roundtrip");
    assert_eq!(parsed.entries[&fname].sha256.as_deref(), Some(expected_sha.as_str()));

    let listener = TcpListener::bind("127.0.0.1:0").expect("bind mock cdn");
    let addr = listener.local_addr().unwrap();
    let valid_clone = valid_ssky.clone();
    let leg_path = format!("/assets-legacy/{leg_name}");
    let cur_path = format!("/assets/{fname}");
    let srv = std::thread::spawn(move || {
        for _ in 0..4 {
            let Ok((mut stream, _)) = listener.accept() else {
                break;
            };
            let mut req_buf = [0u8; 2048];
            let n = stream.read(&mut req_buf).unwrap_or(0);
            let req = String::from_utf8_lossy(&req_buf[..n]);
            let first_line = req.lines().next().unwrap_or("");
            if first_line.contains(&cur_path) {
                // Return stale content with wrong checksum so client falls back to assets-legacy!
                let body = b"stale-newer-version-bytes";
                let hdr = format!(
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                );
                let _ = stream.write_all(hdr.as_bytes());
                let _ = stream.write_all(body);
            } else if first_line.contains(&leg_path) {
                let hdr = format!(
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    valid_clone.len()
                );
                let _ = stream.write_all(hdr.as_bytes());
                let _ = stream.write_all(&valid_clone);
            } else {
                let _ = stream.write_all(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
            }
        }
    });

    let stamp = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    let tmp_cache = std::env::temp_dir().join(format!("strata_earth_test_cache_{stamp}"));
    let mut cfg = AssetSourceConfig::default()
        .with_cdn_base(Some(format!("http://{addr}")))
        .with_cache_dir(&tmp_cache);
    cfg.github_repo.clear();
    cfg.offline = false;
    cfg.allow_direct_hips_fallback = false;

    let mgr = AssetManager::new();
    let fetched = mgr
        .fetch_or_cache(
            &cfg,
            &fname,
            Some(&leg_name),
            Some(&expected_sha),
            Duration::from_secs(5),
            1024 * 1024,
        )
        .expect("fetch via assets-legacy fallback");
    assert_eq!(fetched, valid_ssky);

    // Offline cache hit works immediately with zero network!
    cfg.offline = true;
    let cached = mgr
        .fetch_or_cache(
            &cfg,
            &fname,
            Some(&leg_name),
            Some(&expected_sha),
            Duration::from_secs(1),
            1024 * 1024,
        )
        .expect("read from local cache while offline");
    assert_eq!(cached, valid_ssky);

    // SpaceSkyboxStore reads synchronously from cache even when offline, and clears when Off!
    let store = SpaceSkyboxStore::new(None);
    store.ensure_loaded(SpaceSkyboxSurvey::GaiaDr3Optical, SpaceSkyboxQuality::Low, &cfg);
    let (rev1, cube1) = store.snapshot();
    assert!(rev1 >= 1 && cube1.as_ref().is_some_and(|c| c.face_size == 8));

    store.ensure_loaded(SpaceSkyboxSurvey::GaiaDr3Optical, SpaceSkyboxQuality::Off, &cfg);
    let (rev2, cube2) = store.snapshot();
    assert!(rev2 > rev1 && cube2.is_none());

    drop(srv);
    let _ = std::fs::remove_dir_all(&tmp_cache);
}

#[test]
fn retained_builder_attaches_streamed_night_lights_on_zoomed_3d_tiles() {
    use strata_earth::TileKey;

    let mut src = MemoryTileSource::new();
    let day_img = strata_earth::TileImage {
        width: 4,
        height: 4,
        pixels: [80, 100, 120, 255].repeat(16),
    };
    let night_img = strata_earth::TileImage {
        width: 4,
        height: 4,
        pixels: [255, 220, 150, 255].repeat(16),
    };
    src.insert_imagery(TileKey::new(0, 0, 0), day_img);
    // NYC area at z=6 (parent of any deeper tile)
    let nyc_z6 = TileKey::new(6, 18, 24);
    src.insert_night_lights(nyc_z6, night_img);

    let mut input = EarthInput {
        camera: EarthCamera {
            focus: Geodetic::new(40.7128, -74.0060, 0.0),
            aim_offset_m: [0.0, 0.0, 0.0],
            eye_offset_m: [0.0, 0.0, 650_000.0],
            fov_y_deg: 55.0,
            near_m: 100.0,
        },
        surface_mode: SurfaceMode::QuadtreeLod,
        tile_projection: TileProjection::WebMercator,
        elevation_enabled: false,
        max_tile_level: 14,
        tile_grid: 5,
        sse_threshold: 8.0,
        space: SpaceEnvironmentConfig {
            day_night_lighting: true,
            city_lights: true,
            ..SpaceEnvironmentConfig::default()
        },
        ..EarthInput::default()
    };

    let mut builder = RetainedTerrainBuilder::new();
    let (frame_on, _) = builder.build_frame(&input, &mut src, 1, false);
    assert!(
        frame_on
            .instances
            .iter()
            .any(|inst| inst.night_texture_id.is_some()),
        "zoomed 3D tiles in range of a cached z=6 night tile must attach night_texture_id"
    );

    // Turning sun lighting (unified daylight) off suppresses night textures cleanly.
    input.space.day_night_lighting = false;
    let (frame_unified, _) = builder.build_frame(&input, &mut src, 2, false);
    assert!(
        frame_unified
            .instances
            .iter()
            .all(|inst| inst.night_texture_id.is_none()),
        "unified sun lighting must omit night_texture_id overlays"
    );
}

#[test]
fn cloud_map_resolution_tiers_and_custom_url_override() {
    use strata_earth::CloudMapResolution;

    let mut res = CloudMapResolution::Low;
    assert_eq!(res.short_label(), "1K");
    assert!(res.url().contains("1024x512"));
    res = res.next();
    assert_eq!(res, CloudMapResolution::Medium);
    assert_eq!(res.short_label(), "2K");
    assert!(res.url().contains("2048x1024"));
    res = res.next();
    assert_eq!(res, CloudMapResolution::High);
    assert_eq!(res.short_label(), "4K");
    assert!(res.url().contains("4096x2048"));
    res = res.next();
    assert_eq!(res, CloudMapResolution::Ultra);
    assert_eq!(res.short_label(), "8K");
    assert!(res.url().contains("8192x4096"));
    assert_eq!(res.next(), CloudMapResolution::Low);

    let cfg = WeatherConfig::default().with_cloud_resolution(CloudMapResolution::High);
    assert_eq!(cfg.resolved_cloud_map_url(), CloudMapResolution::High.url());

    let mut custom = WeatherConfig::default();
    custom.cloud_map_url = "https://example.test/custom_clouds.png".to_string();
    custom = custom.with_cloud_resolution(CloudMapResolution::Ultra);
    assert_eq!(
        custom.resolved_cloud_map_url(),
        "https://example.test/custom_clouds.png"
    );
}

#[test]
fn cloud_opacity_has_no_mid_orbit_dip_and_shader_defines_plane_fade() {
    let base_op = 0.72f32;
    let mut input = EarthInput {
        camera: EarthCamera {
            focus: Geodetic::new(38.5, 24.0, 0.0),
            aim_offset_m: [0.0, 0.0, 0.0],
            eye_offset_m: [0.0, 0.0, 10_000_000.0],
            fov_y_deg: 46.0,
            near_m: 1_000.0,
        },
        viewport: Rect::new(0.0, 0.0, 1280.0, 800.0),
        seamless_25d: true,
        tile_projection: TileProjection::WebMercator,
        surface_mode: SurfaceMode::QuadtreeLod,
        layers: LayerToggles {
            clouds: true,
            ..LayerToggles::default()
        },
        ..EarthInput::default()
    };
    input.weather.global_cloud_map = true;
    input.weather.global_cloud_opacity = base_op;

    for alt_m in [
        10_000_000.0,
        2_000_000.0,
        1_225_000.0,
        600_000.0,
        450_000.0,
        120_000.0,
        13_500.0,
        1_500.0,
    ] {
        input.camera.eye_offset_m = [0.0, 0.0, alt_m];
        let cam = RetainedTerrainBuilder::camera_from_input(&input);
        let combined = cam.cloud_layer_opacity + cam.local_cloud_opacity;
        assert!(
            (combined - base_op).abs() < 1e-4,
            "combined cloud uniform opacity must stay steady at {base_op} without mid-orbit dip at alt={alt_m}m, got {combined}"
        );
    }
}




