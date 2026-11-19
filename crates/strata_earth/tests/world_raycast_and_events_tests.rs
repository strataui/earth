//! Integration tests for `EarthWorld`, live two-mode geodetic altitude (`SeaLevel` & `Terrain`),
//! hardware-instanced mesh deduplication, terrain/mode/object raycasting, external events, and
//! zero library keybind leakage.

use strata_earth::{
    ecef_to_geodetic, AltitudeMode, Color, EarthCamera, EarthGeometry, EarthInput, EarthMaterial,
    EarthObject, EarthScene, EarthViewMode, EarthViewport, GeoPosition, Geodetic, MemoryTileSource,
    SurfaceMode, TileElevation, TileImage, TileKey,
};
use strata_layout::Rect;
use strata_render::camera64::DVec3;

fn solid_tile(r: u8, g: u8, b: u8) -> TileImage {
    TileImage {
        width: 4,
        height: 4,
        pixels: [r, g, b, 255].repeat(16),
    }
}

fn uniform_elevation(height_m: f32) -> TileElevation {
    TileElevation {
        width: 4,
        height: 4,
        heights_m: vec![height_m; 16],
    }
}

/// Populates levels 0..=15 covering `(lat, lon)` with constant DEM `dem_m` so quadtree LOD
/// and ground snapshots resolve exact terrain elevation immediately without network access.
fn mountain_tile_source(lat_deg: f64, lon_deg: f64, dem_m: f32) -> MemoryTileSource {
    let mut src = MemoryTileSource::new();
    let img = solid_tile(64, 120, 72);
    let elev = uniform_elevation(dem_m);
    for level in 0..=15u8 {
        let n = 1u32 << level;
        let u = ((lon_deg + 180.0) / 360.0).clamp(0.0, 0.999_999);
        // Populate both geographic and Web Mercator tile indices around (lat, lon).
        let v_geo = ((90.0 - lat_deg) / 180.0).clamp(0.0, 0.999_999);
        let lat_rad = lat_deg.clamp(-85.05112878, 85.05112878).to_radians();
        let v_merc = ((1.0 - (lat_rad.tan() + 1.0 / lat_rad.cos()).ln() / std::f64::consts::PI)
            * 0.5)
            .clamp(0.0, 0.999_999);

        let cx = (u * n as f64).floor() as i32;
        for vy in [v_geo, v_merc] {
            let cy = (vy * n as f64).floor() as i32;
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let x = cx.rem_euclid(n as i32) as u32;
                    let y = (cy + dy).clamp(0, n as i32 - 1) as u32;
                    let key = TileKey::new(level, (x as i32 + dx).rem_euclid(n as i32) as u32, y);
                    src.insert_imagery(key, img.clone());
                    src.insert_elevation(key, elev.clone());
                }
            }
        }
    }
    src
}

fn instance_geodetic_height_m(inst: &strata_render::scene3d::RetainedGeoInstance) -> f64 {
    let ecef = DVec3::new(inst.anchor_ecef[0], inst.anchor_ecef[1], inst.anchor_ecef[2]);
    ecef_to_geodetic(ecef).height_m
}

#[test]
fn terrain_and_sea_level_altitude_modes_react_live_to_heightmap_toggle() {
    let (lat, lon) = (46.0, 7.6); // Alpine test coordinate over a 3,200 m DEM surface
    let mut source = mountain_tile_source(lat, lon, 3_200.0);

    let mut input = EarthInput::globe();
    input.camera = EarthCamera {
        focus: Geodetic::new(lat, lon, 0.0),
        eye_offset_m: [0.0, 0.0, 250_000.0], // Pure 3D curved globe regime (zero 2D lift)
        ..Default::default()
    };
    input.viewport = Rect::new(0.0, 0.0, 1280.0, 720.0);
    input.elevation_enabled = true;
    input.surface_mode = SurfaceMode::QuadtreeLod;

    let mut scene = EarthScene::new();
    let terrain_id = scene.spawn_object(
        EarthObject::new(
            GeoPosition::terrain(lat, lon, 5.0),
            EarthGeometry::beacon_pin(24.0, 6.0),
        )
        .with_material(EarthMaterial::standard(Color::rgb(56, 189, 248))),
    );
    let sea_id = scene.spawn_object(
        EarthObject::new(
            GeoPosition::sea_level(lat, lon, 5.0),
            EarthGeometry::sphere(8.0),
        )
        .with_material(EarthMaterial::standard(Color::rgb(250, 204, 21))),
    );

    // 1. Heightmap ON: Terrain(+5m) -> 3,205m, SeaLevel(+5m) -> 5m.
    scene.build_retained(&input, &mut source);
    let tf_terrain_on = scene
        .resolve_object_transform(terrain_id, &input)
        .expect("terrain object transform");
    let tf_sea_on = scene
        .resolve_object_transform(sea_id, &input)
        .expect("sea object transform");
    assert!(
        (tf_terrain_on.resolved_altitude_m - 3_205.0).abs() < 0.5,
        "expected ~3205.0m with heightmap ON, got {}",
        tf_terrain_on.resolved_altitude_m
    );
    assert!(
        (tf_sea_on.resolved_altitude_m - 5.0).abs() < 1e-6,
        "SeaLevel(+5m) must stay at 5.0m, got {}",
        tf_sea_on.resolved_altitude_m
    );

    let snap_on = scene
        .retained_scene()
        .snapshot_terrain()
        .expect("retained terrain frame");
    assert_eq!(snap_on.geo_instances.len(), 2);
    let gpu_terrain_h_on = instance_geodetic_height_m(&snap_on.geo_instances[0]);
    assert!(
        (gpu_terrain_h_on - 3_205.0).abs() < 0.5,
        "GPU instance ECEF height must be ~3205.0m with heightmap ON, got {gpu_terrain_h_on}"
    );

    // 2. Programmatically disable heightmap WITHOUT mutating or recreating the objects:
    //    Terrain(+5m) must immediately drop to exact 5.0m above sea level!
    input.elevation_enabled = false;
    scene.build_retained(&input, &mut source);

    let tf_terrain_off = scene
        .resolve_object_transform(terrain_id, &input)
        .expect("terrain object transform with heightmap OFF");
    assert!(
        (tf_terrain_off.resolved_altitude_m - 5.0).abs() < 1e-6,
        "with heightmap OFF, Terrain(5.0m) must resolve to exact 5.0m above sea level, got {}",
        tf_terrain_off.resolved_altitude_m
    );
    scene.world(|w| {
        let obj = w.get(terrain_id).expect("object still present");
        assert_eq!(
            obj.position.altitude_mode,
            AltitudeMode::Terrain,
            "stored AltitudeMode must remain Terrain (never overwritten to SeaLevel)"
        );
        assert_eq!(obj.position.altitude_m, 5.0);
    });

    let snap_off = scene
        .retained_scene()
        .snapshot_terrain()
        .expect("retained terrain frame OFF");
    let gpu_terrain_h_off = instance_geodetic_height_m(&snap_off.geo_instances[0]);
    assert!(
        (gpu_terrain_h_off - 5.0).abs() < 0.05,
        "GPU instance ECEF height must drop to 5.0m above sea level when heightmap is OFF, got {gpu_terrain_h_off}"
    );

    // 3. Re-enable heightmap: Terrain(+5m) immediately returns to 3,205.0m!
    input.elevation_enabled = true;
    scene.build_retained(&input, &mut source);
    let tf_terrain_reon = scene
        .resolve_object_transform(terrain_id, &input)
        .expect("terrain object transform re-enabled");
    assert!(
        (tf_terrain_reon.resolved_altitude_m - 3_205.0).abs() < 0.5,
        "re-enabling heightmap must restore 3,205.0m altitude, got {}",
        tf_terrain_reon.resolved_altitude_m
    );
}

#[test]
fn earth_viewport_set_elevation_enabled_updates_terrain_objects_synchronously() {
    let (lat, lon) = (46.0, 7.6);
    let source = mountain_tile_source(lat, lon, 3_200.0);

    let mut input = EarthInput::globe();
    input.camera = EarthCamera {
        focus: Geodetic::new(lat, lon, 0.0),
        eye_offset_m: [0.0, 0.0, 250_000.0],
        ..Default::default()
    };
    input.viewport = Rect::new(0.0, 0.0, 1000.0, 800.0);
    input.elevation_enabled = true;

    let mut vp = EarthViewport::new(&input, Box::new(source));
    vp.sync_now();
    let obj_id = vp.spawn_object(EarthObject::new(
        GeoPosition::terrain(lat, lon, 5.0),
        EarthGeometry::beacon_pin(20.0, 5.0),
    ));

    let tf_initial = vp.resolve_object_transform(obj_id).unwrap();
    assert!(
        (tf_initial.resolved_altitude_m - 3_205.0).abs() < 1.0,
        "expected ~3205.0m initially, got {}",
        tf_initial.resolved_altitude_m
    );

    // Toggling heightmap OFF on EarthViewport must synchronously re-evaluate GPU instances on the main thread!
    vp.set_elevation_enabled(false);
    let tf_off = vp.resolve_object_transform(obj_id).unwrap();
    assert!((tf_off.resolved_altitude_m - 5.0).abs() < 1e-6);
    let snap_off = vp.retained_scene().snapshot_terrain().unwrap();
    let gpu_h_off = instance_geodetic_height_m(&snap_off.geo_instances[0]);
    assert!((gpu_h_off - 5.0).abs() < 0.05);

    // Toggling heightmap back ON immediately restores 3,205.0m using the preserved ground snapshot!
    vp.set_elevation_enabled(true);
    let tf_on = vp.resolve_object_transform(obj_id).unwrap();
    assert!((tf_on.resolved_altitude_m - 3_205.0).abs() < 1.0);
}

#[test]
fn hardware_instancing_deduplicates_identical_primitive_meshes() {
    let mut input = EarthInput::globe();
    input.viewport = Rect::new(0.0, 0.0, 800.0, 600.0);
    let mut source = MemoryTileSource::new();
    let mut scene = EarthScene::new();

    for i in 0..50 {
        scene.spawn_object(EarthObject::new(
            GeoPosition::terrain(35.0 + i as f64 * 0.01, 139.0 + i as f64 * 0.01, 5.0),
            EarthGeometry::beacon_pin(28.0, 7.5),
        ));
    }
    scene.build_retained(&input, &mut source);

    let snap = scene
        .retained_scene()
        .snapshot_terrain()
        .expect("retained frame");
    assert_eq!(
        snap.geo_mesh_assets.len(),
        1,
        "all 50 identical BeaconPin objects must share 1 canonical VRAM mesh asset"
    );
    assert_eq!(
        snap.geo_instances.len(),
        50,
        "all 50 BeaconPin objects must emit per-instance transforms"
    );
}

#[test]
fn raycast_respects_dem_relief_heightmap_toggle_planet_limb_2d_map_and_pickable_objects() {
    let (lat, lon) = (46.0, 7.6);
    let mut source = mountain_tile_source(lat, lon, 2_500.0);

    let mut input = EarthInput::globe();
    input.camera = EarthCamera {
        focus: Geodetic::new(lat, lon, 0.0),
        eye_offset_m: [0.0, 0.0, 12_000.0],
        ..Default::default()
    };
    input.viewport = Rect::new(0.0, 0.0, 1000.0, 800.0);
    input.elevation_enabled = true;

    let mut scene = EarthScene::new();
    scene.build_retained(&input, &mut source);

    // 1. Center raycast over 2,500 m mountain with heightmap ON.
    let hit_on = scene
        .raycast(500.0, 400.0, &input)
        .expect("center ray must hit terrain");
    assert!(
        (hit_on.lat_deg - lat).abs() < 0.02 && (hit_on.lon_deg - lon).abs() < 0.02,
        "center ray must hit focus lat/lon, got ({}, {})",
        hit_on.lat_deg,
        hit_on.lon_deg
    );
    assert!(
        (hit_on.altitude_m - 2_500.0).abs() < 1.5,
        "raycast with heightmap ON must hit ~2500m DEM surface, got {}",
        hit_on.altitude_m
    );
    assert!((hit_on.terrain_elevation_m - 2_500.0).abs() < 1.5);
    assert_eq!(hit_on.object_id, None);

    // 2. Same raycast with heightmap OFF hits smooth 0.0 m sea level.
    input.elevation_enabled = false;
    scene.build_retained(&input, &mut source);
    let hit_off = scene
        .raycast(500.0, 400.0, &input)
        .expect("center ray must hit sea level");
    assert!(
        hit_off.altitude_m.abs() < 0.05,
        "raycast with heightmap OFF must hit 0.0m sea level, got {}",
        hit_off.altitude_m
    );
    assert_eq!(hit_off.terrain_elevation_m, 0.0);

    // 3. Pickable 3D object intersection vs terrain-only raycast.
    let box_id = scene.spawn_object(EarthObject::new(
        GeoPosition::sea_level(lat, lon, 0.0),
        EarthGeometry::box_m(120.0, 120.0, 120.0).with_anchor(strata_earth::ObjectAnchor::Base),
    ));
    scene.build_retained(&input, &mut source);

    let hit_obj = scene
        .raycast(500.0, 400.0, &input)
        .expect("center ray must hit 3D box");
    assert_eq!(hit_obj.object_id, Some(box_id));
    assert!(
        (hit_obj.altitude_m - 120.0).abs() < 2.0,
        "nadir ray must hit top face of base-anchored 120m box, got {}",
        hit_obj.altitude_m
    );

    let hit_ground_only = scene
        .raycast_terrain(500.0, 400.0, &input)
        .expect("raycast_terrain must ignore objects");
    assert_eq!(hit_ground_only.object_id, None);
    assert!(hit_ground_only.altitude_m.abs() < 0.05);

    scene.clear_objects();

    // 4. Raycast outside planet limb from space returns None.
    let mut space_input = input.clone();
    space_input.camera.eye_offset_m = [0.0, 0.0, 22_000_000.0];
    scene.build_retained(&space_input, &mut source);
    assert!(
        scene.raycast(4.0, 4.0, &space_input).is_none(),
        "corner ray into deep space beyond Earth's limb must return None"
    );

    // 5. Raycast in 2D Flat Map mode unprojects accurately.
    let mut map2d_input = input.clone();
    map2d_input.view_mode = EarthViewMode::Mode2D;
    map2d_input.view_morph = 1.0;
    map2d_input.camera.eye_offset_m = [0.0, 0.0, 50_000.0];
    scene.build_retained(&map2d_input, &mut source);
    let hit_2d = scene
        .raycast(500.0, 400.0, &map2d_input)
        .expect("2D center raycast must hit map plane");
    assert!(
        (hit_2d.lat_deg - lat).abs() < 0.01 && (hit_2d.lon_deg - lon).abs() < 0.01,
        "2D center raycast must hit focus ({lat}, {lon}), got ({}, {})",
        hit_2d.lat_deg,
        hit_2d.lon_deg
    );
}

#[test]
fn library_crate_contains_zero_hardcoded_zoom_keybinds() {
    // Architectural guard: K / L zoom keybinds must live 100% in example/application code,
    // never inside `strata_earth`'s controller, widget, or scene modules.
    for rel in ["src/controller.rs", "src/widget.rs", "src/scene.rs", "src/worker.rs"] {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(rel);
        let src = std::fs::read_to_string(&path).expect("read source file");
        for forbidden in ["KeyK", "KeyL", "KEY_ZOOM_LN_PER_SEC", "zoom_in_held", "zoom_out_held"] {
            assert!(
                !src.contains(forbidden),
                "{} must not contain hardcoded zoom keybind symbol `{forbidden}`",
                path.display()
            );
        }
    }
}

#[cfg(feature = "widget")]
mod widget_event_tests {
    use std::cell::Cell;
    use std::rc::Rc;

    use strata::platform::{KeyCode, Modifiers, PlatformEvent};
    use strata_earth::{
        EarthGeometry, EarthInput, EarthModifiers, EarthMouseButton, EarthObject,
        EarthOrbitController, EarthView, GeoPosition, MemoryTileSource,
    };
    use strata_layout::Rect;

    fn make_test_view() -> EarthView {
        let mut orbit = EarthOrbitController::new();
        orbit.jump_to(20.0, 10.0, 8_000_000.0, 0.0, 0.0);
        orbit.set_viewport_size(1000.0, 800.0);
        let (cam, up) = orbit.to_camera_and_up(false, 0.0);

        let mut input = EarthInput::globe();
        input.camera = cam;
        input.camera_up_ecef = up;
        input.viewport = Rect::new(0.0, 0.0, 1000.0, 800.0);
        input.elevation_enabled = false;

        EarthView::with_orbit(&input, Box::new(MemoryTileSource::new()), orbit)
    }

    #[test]
    fn clean_click_fires_on_click_and_spawns_object_while_drag_suppresses_click() {
        let earth = make_test_view();
        let clicks = Rc::new(Cell::new(0usize));
        let pointer_ups = Rc::new(Cell::new(0usize));

        {
            let clicks_c = Rc::clone(&clicks);
            let earth_c = earth.clone();
            earth.on_click(move |ev| {
                clicks_c.set(clicks_c.get() + 1);
                if let Some(hit) = earth_c.raycast(ev.x, ev.y) {
                    earth_c.spawn_object(EarthObject::new(
                        GeoPosition::terrain(hit.lat_deg, hit.lon_deg, 5.0),
                        EarthGeometry::beacon_pin(20.0, 5.0),
                    ));
                }
            });
        }
        {
            let ups_c = Rc::clone(&pointer_ups);
            earth.on_pointer_up(move |_| {
                ups_c.set(ups_c.get() + 1);
            });
        }

        // 1. Clean click (2 px jitter <= 5 px threshold) -> fires on_click & spawns 3D object!
        earth.dispatch_pointer_down(
            EarthMouseButton::Left,
            500.0,
            400.0,
            EarthModifiers::default(),
        );
        earth.dispatch_pointer_move(501.5, 401.0, EarthModifiers::default());
        earth.dispatch_pointer_up(
            EarthMouseButton::Left,
            501.5,
            401.0,
            EarthModifiers::default(),
        );
        assert_eq!(clicks.get(), 1);
        assert_eq!(pointer_ups.get(), 1);
        assert_eq!(earth.object_count(), 1);

        // 2. Orbit/pan drag (180 px travel > 5 px threshold) -> moves camera, fires on_pointer_up,
        //    and MUST NOT fire on_click!
        let lon_before = earth.with_state(|st| st.orbit.lon_deg);
        earth.dispatch_pointer_down(
            EarthMouseButton::Left,
            500.0,
            400.0,
            EarthModifiers::default(),
        );
        earth.dispatch_pointer_move(680.0, 400.0, EarthModifiers::default());
        earth.dispatch_pointer_up(
            EarthMouseButton::Left,
            680.0,
            400.0,
            EarthModifiers::default(),
        );
        let lon_after = earth.with_state(|st| st.orbit.lon_deg);
        assert_eq!(
            clicks.get(),
            1,
            "a 180px camera drag must never trigger on_click"
        );
        assert_eq!(pointer_ups.get(), 2);
        assert!(
            (lon_after - lon_before).abs() > 0.5,
            "camera drag should pan longitude"
        );
    }

    #[test]
    fn prevent_default_camera_and_external_key_zoom_work_cleanly() {
        let earth = make_test_view();

        // Intercept pointer_down and wheel with prevent_default_camera(): camera must not move!
        earth.on_pointer_down(|ev| {
            ev.prevent_default_camera();
        });
        earth.on_wheel(|ev| {
            ev.prevent_default_camera();
        });

        let (lat0, lon0, range0) =
            earth.with_state(|st| (st.orbit.lat_deg, st.orbit.lon_deg, st.orbit.target_range_m));
        earth.dispatch_pointer_down(
            EarthMouseButton::Left,
            500.0,
            400.0,
            EarthModifiers::default(),
        );
        earth.dispatch_pointer_move(700.0, 500.0, EarthModifiers::default());
        earth.dispatch_pointer_up(
            EarthMouseButton::Left,
            700.0,
            500.0,
            EarthModifiers::default(),
        );
        earth.dispatch_wheel(0.0, -120.0, 500.0, 400.0, EarthModifiers::default());

        let (lat1, lon1, range1) =
            earth.with_state(|st| (st.orbit.lat_deg, st.orbit.lon_deg, st.orbit.target_range_m));
        assert_eq!((lat0, lon0, range0), (lat1, lon1, range1));

        // External keybind listener implemented purely in caller code (like K/L in earth_preview):
        let held_in = Rc::new(Cell::new(false));
        {
            let h = Rc::clone(&held_in);
            earth.on_key_down(move |ev| {
                if ev.key == KeyCode::KeyK {
                    h.set(true);
                }
            });
        }
        {
            let h = Rc::clone(&held_in);
            earth.on_key_up(move |ev| {
                if ev.key == KeyCode::KeyK {
                    h.set(false);
                }
            });
        }

        earth.dispatch_platform_event(&PlatformEvent::KeyDown {
            key: KeyCode::KeyK,
            modifiers: Modifiers::default(),
        });
        assert!(held_in.get());
        earth.zoom_by_ln(-0.25);
        let _ = earth.step_frame(1.0 / 60.0);
        let range_zoomed = earth.with_state(|st| st.orbit.target_range_m);
        assert!(range_zoomed < range0);

        // Window focus loss synthesizes clean key release so held zoom never sticks!
        earth.dispatch_platform_event(&PlatformEvent::WindowFocused(false));
        assert!(!held_in.get());
    }
}
