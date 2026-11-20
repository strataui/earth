//! Integration tests for CesiumJS-tier Earth features:
//! - Embedded base world map (`z=0..2`) & smooth ancestor fallback until zoomed tiles load
//! - AWS Terrarium DEM heightmap decoding
//! - OpenStreetMap Overpass JSON parsing & `NetworkBuildingSource`
//! - Seamless 3D <-> 2.5D ground-proximity morph & low-level state inspection
//! - Programmatic `EarthViewMode::Mode2D` <-> `Mode3D` animated morph in `EarthViewport`

use std::sync::Arc;

use strata_earth::{
    build_tile_local_rtc_mesh_morphed, decode_terrarium, embedded_base_tile,
    parse_overpass_buildings, resolve_tile_imagery, tile_bounds_mercator, EarthCamera,
    EarthDefaultTileSource, EarthInput, EarthScene, EarthViewMode, EarthViewport,
    ElevationEncoding, EmbeddedBaseMapSource, Geodetic, LayerToggles, MockHttpTransport,
    NetworkBuildingSource, NetworkTileSource, PatchBounds, SurfaceMode, SurfaceMorphParams,
    TileImage, TileKey, TileSource,
};

fn crc32_simple(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &b in data {
        crc ^= b as u32;
        for _ in 0..8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ 0xEDB8_8320;
            } else {
                crc >>= 1;
            }
        }
    }
    !crc
}

fn write_png_chunk(out: &mut Vec<u8>, chunk_type: &[u8; 4], data: &[u8]) {
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    out.extend_from_slice(chunk_type);
    out.extend_from_slice(data);
    let mut crc_buf = Vec::with_capacity(4 + data.len());
    crc_buf.extend_from_slice(chunk_type);
    crc_buf.extend_from_slice(data);
    out.extend_from_slice(&crc32_simple(&crc_buf).to_be_bytes());
}

fn encode_png_rgba(img: &TileImage) -> Vec<u8> {
    use flate2::write::ZlibEncoder;
    use flate2::Compression;
    use std::io::Write;

    let mut out = Vec::new();
    out.extend_from_slice(b"\x89PNG\r\n\x1a\n");

    let mut ihdr = Vec::with_capacity(13);
    ihdr.extend_from_slice(&img.width.to_be_bytes());
    ihdr.extend_from_slice(&img.height.to_be_bytes());
    ihdr.push(8);
    ihdr.push(6);
    ihdr.push(0);
    ihdr.push(0);
    ihdr.push(0);
    write_png_chunk(&mut out, b"IHDR", &ihdr);

    let stride = img.width as usize * 4;
    let mut raw = Vec::new();
    for y in 0..img.height as usize {
        raw.push(0);
        raw.extend_from_slice(&img.pixels[y * stride..(y + 1) * stride]);
    }
    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(&raw).unwrap();
    let compressed = encoder.finish().unwrap();
    write_png_chunk(&mut out, b"IDAT", &compressed);
    write_png_chunk(&mut out, b"IEND", &[]);
    out
}

#[test]
fn embedded_basemap_provides_all_z0_to_z2_world_tiles() {
    let mut src = EmbeddedBaseMapSource::new();
    let mut count = 0usize;
    for level in 0..=2u8 {
        let per_axis = 1u32 << level;
        for y in 0..per_axis {
            for x in 0..per_axis {
                let key = TileKey::new(level, x, y);
                let raw = embedded_base_tile(key).expect("embedded tile should exist");
                assert!(raw.width >= 64 && raw.height >= 64);
                let img = src.imagery(key).expect("ok").expect("present");
                assert_eq!(img.pixels.len(), (img.width * img.height * 4) as usize);
                count += 1;
            }
        }
    }
    assert_eq!(count, 1 + 4 + 16);
}

#[test]
fn default_source_zoom_keeps_base_ancestor_smoothly_until_highres_tile_arrives() {
    let target_z8 = TileKey::new(8, 41, 99);
    let expected_url = format!(
        "https://tiles.example.test/{}/{}/{}.png",
        target_z8.level, target_z8.x, target_z8.y
    );
    let hi_res = TileImage {
        width: 8,
        height: 8,
        pixels: [20, 200, 90, 255].repeat(64),
    };
    let transport = MockHttpTransport::new();
    transport.set_response(&expected_url, encode_png_rgba(&hi_res));

    let temp_dir = std::env::temp_dir().join(format!(
        "strata_earth_cesium_zoom_fallback_{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&temp_dir);
    let net = NetworkTileSource::new("https://tiles.example.test/{z}/{x}/{y}.png")
        .with_transport(transport)
        .with_disk_cache(&temp_dir);
    let mut default_src = EarthDefaultTileSource::with_network(net);

    // Before background download completes, requesting z=8 returns None,
    // while peeking ancestors smoothly resolves to the embedded z=2 base map tile.
    let initial = default_src.imagery(target_z8).expect("request ok");
    if initial.is_none() {
        let fallback = resolve_tile_imagery(target_z8, |candidate| {
            default_src.peek_cached_imagery(candidate).is_some()
        })
        .expect("should smoothly fall back to embedded world basemap");
        assert!(!fallback.exact);
        assert_eq!(fallback.source.level, strata_earth::MAX_EMBEDDED_BASE_LEVEL);
        assert!(fallback.uv_rect.0 >= 0.0 && fallback.uv_rect.2 <= 1.0);
        assert!(fallback.uv_rect.1 >= 0.0 && fallback.uv_rect.3 <= 1.0);
    }

    // Wait briefly for the async worker to deliver z=8.
    let mut arrived = false;
    for _ in 0..80 {
        if let Some(tile) = default_src.imagery(target_z8).expect("poll ok") {
            assert_eq!(tile.width, 8);
            assert_eq!(tile.pixels[0..4], [20, 200, 90, 255]);
            arrived = true;
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    assert!(arrived, "z=8 tile should arrive asynchronously");
    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn decode_terrarium_decodes_sea_level_everest_and_trench() {
    // formula: (R * 256 + G + B / 256) - 32768
    let px = [
        // 0.0 m -> 32768 = 128 * 256 + 0 + 0
        128, 0, 0, 255, // +8848.5 m -> 41616.5 -> 162 * 256 + 144 + 128/256
        162, 144, 128, 255, // -1000.0 m -> 31768 -> 124 * 256 + 24 + 0
        124, 24, 0, 255, // 100.25 m -> 32868.25 -> 128 * 256 + 100 + 64/256
        128, 100, 64, 255,
    ];
    let elev = decode_terrarium(2, 2, &px);
    assert_eq!(elev.width, 2);
    assert_eq!(elev.height, 2);
    assert!((elev.heights_m[0] - 0.0).abs() < 0.02);
    assert!((elev.heights_m[1] - 8848.5).abs() < 0.02);
    assert!((elev.heights_m[2] + 1000.0).abs() < 0.02);
    assert!((elev.heights_m[3] - 100.25).abs() < 0.02);

    // Ensure ElevationEncoding::Terrarium is wired as public variant.
    let _enc = ElevationEncoding::Terrarium;
}

#[test]
fn parse_overpass_buildings_and_network_building_source() {
    let overpass_json = r#"{
        "version": 0.6,
        "elements": [
            { "type": "node", "id": 1, "lat": 37.7740, "lon": -122.4190 },
            { "type": "node", "id": 2, "lat": 37.7740, "lon": -122.4186 },
            { "type": "node", "id": 3, "lat": 37.7744, "lon": -122.4186 },
            { "type": "node", "id": 4, "lat": 37.7744, "lon": -122.4190 },
            {
                "type": "way",
                "id": 100,
                "nodes": [1, 2, 3, 4, 1],
                "tags": { "building": "commercial", "height": "48m" }
            },
            { "type": "node", "id": 5, "lat": 37.7750, "lon": -122.4180 },
            { "type": "node", "id": 6, "lat": 37.7750, "lon": -122.4178 },
            { "type": "node", "id": 7, "lat": 37.7752, "lon": -122.4178 },
            {
                "type": "way",
                "id": 101,
                "nodes": [5, 6, 7, 5],
                "tags": { "building": "yes", "building:levels": "5" }
            }
        ]
    }"#;

    let parsed = parse_overpass_buildings(overpass_json);
    assert_eq!(parsed.len(), 2);
    assert!((parsed[0].height_m - 48.0).abs() < 0.01);
    assert!((parsed[1].height_m - 16.0).abs() < 0.01);

    let mock_transport = MockHttpTransport::new();
    let key = TileKey::new(15, 5241, 12664);
    let bounds = PatchBounds {
        lat0: 37.773,
        lon0: -122.420,
        lat1: 37.776,
        lon1: -122.417,
    };
    let s = bounds.lat0;
    let n = bounds.lat1;
    let w = bounds.lon0;
    let e = bounds.lon1;
    let raw_query = format!(
        "[out:json][timeout:10];(way[\"building\"]({s:.5},{w:.5},{n:.5},{e:.5}););out body;>;out skel qt;"
    );
    let mut encoded = String::new();
    for b in raw_query.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                encoded.push(b as char);
            }
            _ => encoded.push_str(&format!("%{b:02X}")),
        }
    }
    let exact_url = format!("https://overpass.mock.test/api?data={encoded}");
    mock_transport.set_response(&exact_url, overpass_json.as_bytes().to_vec());

    let mut building_src = NetworkBuildingSource::with_endpoint(
        "https://overpass.mock.test/api",
        Arc::new(mock_transport),
    )
    .with_min_fetch_level(14);
    let fetched = building_src.fetch_sync(key, bounds);
    assert_eq!(fetched.len(), 2);
    assert_eq!(building_src.cached_tile_count(), 1);

    let mut scene = EarthScene::new();
    scene.set_building_source(Some(Box::new(building_src)));
    let input = EarthInput {
        camera: EarthCamera {
            focus: Geodetic::new(37.7745, -122.4185, 0.0),
            eye_offset_m: [0.0, -500.0, 900.0],
            ..EarthCamera::default()
        },
        layers: LayerToggles {
            buildings: true,
            ..LayerToggles::default()
        },
        ..EarthInput::default()
    };
    let mut img_src = EmbeddedBaseMapSource::new();
    scene.build(&input, &mut img_src);
    assert!(scene.stats().draw_calls > 0);
}

#[test]
fn seamless_3d_to_25d_blend_and_low_level_override() {
    let far_input = EarthInput {
        camera: EarthCamera {
            eye_offset_m: [0.0, 0.0, 12_000_000.0],
            ..EarthCamera::default()
        },
        surface_mode: SurfaceMode::QuadtreeLod,
        ..EarthInput::default()
    };
    let far_state = far_input.low_level_state();
    assert_eq!(far_state.active_surface_mode, SurfaceMode::QuadtreeLod);
    assert_eq!(far_state.local_25d_blend, 0.0);
    assert_eq!(far_state.map_2d_morph, 0.0);

    let mut mid_input = far_input.clone();
    mid_input.camera.eye_offset_m = [0.0, 0.0, 120_000.0];
    let mid_state = mid_input.low_level_state();
    assert!(mid_state.local_25d_blend > 0.0 && mid_state.local_25d_blend < 1.0);

    let mut near_input = far_input.clone();
    near_input.camera.eye_offset_m = [0.0, 0.0, 5_000.0];
    let near_state = near_input.low_level_state();
    assert_eq!(near_state.local_25d_blend, 1.0);

    // Low-level attribute lets callers override internal surface mode without altering high-level API.
    let mut forced = far_input.clone();
    forced.low_level_surface_override = Some(SurfaceMode::FlatPatches);
    assert_eq!(forced.effective_surface_mode(), SurfaceMode::FlatPatches);
    assert_eq!(
        forced.low_level_state().active_surface_mode,
        SurfaceMode::FlatPatches
    );

    // Morphing a coarse curved tile smoothly reduces chord curvature toward a tangent plane.
    let key = TileKey::new(2, 1, 1);
    let bounds = tile_bounds_mercator(key);
    let curved = build_tile_local_rtc_mesh_morphed(
        key,
        bounds,
        5,
        None,
        key,
        false,
        0.0,
        1,
        0,
        SurfaceMorphParams {
            flat_blend: 0.0,
            map_2d_morph: 0.0,
            focus: Geodetic::new(20.0, -45.0, 0.0),
            ..SurfaceMorphParams::default()
        },
    );
    let flat_25d = build_tile_local_rtc_mesh_morphed(
        key,
        bounds,
        5,
        None,
        key,
        false,
        0.0,
        2,
        0,
        SurfaceMorphParams {
            flat_blend: 1.0,
            map_2d_morph: 0.0,
            focus: Geodetic::new(20.0, -45.0, 0.0),
            ..SurfaceMorphParams::default()
        },
    );
    assert_ne!(curved.vertices[0].pos_u, flat_25d.vertices[0].pos_u);
}

#[test]
fn programmatic_3d_to_2d_morph_transition_and_back() {
    let input = EarthInput {
        surface_mode: SurfaceMode::QuadtreeLod,
        camera: EarthCamera {
            eye_offset_m: [0.0, -4_000_000.0, 8_000_000.0],
            ..EarthCamera::default()
        },
        ..EarthInput::default()
    };
    let src = EmbeddedBaseMapSource::new();
    let mut viewport = EarthViewport::new(&input, Box::new(src));
    viewport.sync_now();

    assert_eq!(viewport.view_mode(), EarthViewMode::Mode3D);
    assert!(!viewport.is_transitioning());
    assert_eq!(viewport.low_level_state().map_2d_morph, 0.0);

    // Programmatically transition from 3D -> 2D map.
    viewport.set_view_mode(EarthViewMode::Mode2D);
    assert_eq!(viewport.view_mode(), EarthViewMode::Mode2D);
    assert!(viewport.is_transitioning());

    let mut saw_intermediate = false;
    for _ in 0..80 {
        let moving = viewport.step_transition(1.0 / 60.0);
        let morph = viewport.low_level_state().map_2d_morph;
        if morph > 0.05 && morph < 0.95 {
            saw_intermediate = true;
        }
        if !moving {
            break;
        }
    }
    assert!(
        saw_intermediate,
        "should step through smooth intermediate morph frames"
    );
    assert!(!viewport.is_transitioning());
    assert!((viewport.low_level_state().map_2d_morph - 1.0).abs() < 1e-4);
    assert_eq!(viewport.low_level_state().local_25d_blend, 0.0);

    // Transition back to 3D globe.
    viewport.set_view_mode(EarthViewMode::Mode3D);
    assert!(viewport.is_transitioning());
    for _ in 0..80 {
        if !viewport.step_transition(1.0 / 60.0) {
            break;
        }
    }
    assert!(!viewport.is_transitioning());
    assert!(viewport.low_level_state().map_2d_morph.abs() < 1e-4);
}

#[test]
fn mode_2d_fast_horizontal_pan_has_zero_rotation_zero_mesh_rebuild_and_seamless_wrap() {
    use strata_earth::{
        EarthGeometry, EarthObject, EarthOrbitController, EarthScreenProjector, GeoPosition,
        RetainedTerrainBuilder,
    };

    let base_input = EarthInput {
        surface_mode: SurfaceMode::QuadtreeLod,
        view_mode: EarthViewMode::Mode2D,
        view_morph: 1.0,
        camera: EarthCamera {
            focus: Geodetic::new(-45.0, 90.0, 0.0),
            eye_offset_m: [0.0, 0.0, 12_000_000.0],
            ..EarthCamera::default()
        },
        ..EarthInput::default()
    };

    // 1. Camera orientation (`view_rot`) must be strictly constant across all longitudes & latitudes
    //    (especially Southern Hemisphere -45° where moving spherical tangent frames caused CCW tilt!).
    let ref_cam = RetainedTerrainBuilder::camera_from_input(&base_input);
    for &(lat, lon) in &[
        (-75.0, -179.0),
        (-45.0, 20.0),
        (-45.0, 145.0),
        (0.0, 0.0),
        (55.0, -120.0),
        (75.0, 179.5),
    ] {
        let mut inp = base_input.clone();
        inp.camera.focus = Geodetic::new(lat, lon, 0.0);
        let c = RetainedTerrainBuilder::camera_from_input(&inp);
        for col in 0..4 {
            for row in 0..4 {
                assert!(
                    (c.view_rot[col][row] - ref_cam.view_rot[col][row]).abs() < 1e-6,
                    "2D view_rot must never rotate or tilt at ({lat}, {lon}): diff at [{col}][{row}]"
                );
            }
        }
    }

    // 2. Rapid panning across Southern Hemisphere must reuse 2D meshes with zero morph invalidation
    //    and keep every mesh normal fixed at +X = (1, 0, 0).
    let mut builder = RetainedTerrainBuilder::new();
    let mut src = EmbeddedBaseMapSource::new();
    let (frame_a, _) = builder.build_frame(&base_input, &mut src, 1, true);
    assert!(!frame_a.meshes.is_empty());
    for mesh in &frame_a.meshes {
        for v in mesh.vertices.iter() {
            let n = v.normal();
            assert!(
                (n[0] - 1.0).abs() < 1e-4 && n[1].abs() < 1e-4 && n[2].abs() < 1e-4,
                "canonical 2D tile mesh normal must stay fixed (1,0,0), got {:?}",
                n
            );
        }
    }

    // Pan rapidly east by 75°, then pan back to original focus: cached meshes must hit 100%!
    let mut panned = base_input.clone();
    panned.camera.focus = Geodetic::new(-45.0, 165.0, 0.0);
    let _ = builder.build_frame(&panned, &mut src, 2, true);
    let (_, stats_return) = builder.build_frame(&base_input, &mut src, 3, true);
    assert_eq!(
        stats_return.mesh_cache_misses, 0,
        "returning after fast 2D pan must hit 100% of cached canonical 2D tile meshes"
    );

    // 3. Antimeridian (+170° viewing Hawaii at -155°): labels, objects, and raycasting must align seamlessly.
    let mut pacific = base_input.clone();
    pacific.camera.focus = Geodetic::new(20.0, 170.0, 0.0);
    pacific.camera.eye_offset_m = [0.0, 0.0, 6_000_000.0];
    let proj = EarthScreenProjector::from_input(&pacific);
    let (sx, sy, ndc_x, _ndc_y, _) = proj
        .project_raw(20.0, -155.0, 0.0)
        .expect("Hawaii (-155°) must project onto screen when 2D camera is at +170°");
    assert!(
        ndc_x > 0.1 && sx > pacific.viewport.width * 0.5,
        "Hawaii (-155°) must appear East (right) of +170°, got sx={sx}, ndc_x={ndc_x}"
    );

    let mut viewport = EarthViewport::new(&pacific, Box::new(EmbeddedBaseMapSource::new()));
    let hawaii_obj = viewport.spawn_object(EarthObject::new(
        GeoPosition::sea_level(20.0, -155.0, 0.0),
        EarthGeometry::sphere(10_000.0),
    ));
    viewport.sync_now();

    let hit = viewport
        .raycast_terrain(sx as f64, sy as f64)
        .expect("raycast onto wrapped Hawaii screen position must hit");
    assert!(
        (hit.lat_deg - 20.0).abs() < 0.05 && (hit.lon_deg - (-155.0)).abs() < 0.05,
        "wrapped 2D raycast must unproject back to (20°, -155°), got ({}, {})",
        hit.lat_deg,
        hit.lon_deg
    );
    let tf = viewport
        .resolve_object_transform(hawaii_obj)
        .expect("resolve Hawaii object");
    let cam_eye_y = viewport.retained_scene().camera().eye_ecef[1];
    assert!(
        (tf.anchor_ecef[1] - cam_eye_y).abs() < 5_000_000.0,
        "Hawaii object must wrap to +205° copy near camera at +170° (diff={} m)",
        (tf.anchor_ecef[1] - cam_eye_y).abs()
    );

    // 4. 2D controller drag speed must stay 1:1 uniform across latitudes (Equator vs -70° Antarctica).
    let mut c_eq = EarthOrbitController::new();
    c_eq.set_viewport_size(1440.0, 900.0);
    c_eq.set_is_2d_hint(true);
    c_eq.jump_to(0.0, 10.0, 8_000_000.0, 0.0, 0.0);
    assert!(c_eq.handle_mouse_down(0, 720.0, 450.0));
    assert!(c_eq.handle_mouse_move(820.0, 450.0));
    let dlon_eq = (c_eq.lon_deg - 10.0).abs();

    let mut c_south = EarthOrbitController::new();
    c_south.set_viewport_size(1440.0, 900.0);
    c_south.set_is_2d_hint(true);
    c_south.jump_to(-70.0, 10.0, 8_000_000.0, 0.0, 0.0);
    assert!(c_south.handle_mouse_down(0, 720.0, 450.0));
    assert!(c_south.handle_mouse_move(820.0, 450.0));
    let dlon_south = (c_south.lon_deg - 10.0).abs();

    assert!(
        (dlon_eq - dlon_south).abs() < 1e-6,
        "2D horizontal drag speed must match flat equirectangular metric at all latitudes: eq={dlon_eq}, south={dlon_south}"
    );
}

