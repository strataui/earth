//! Retained GPU pipeline, 6-plane frustum culling, background worker streaming,
//! and 1,000-frame orbit throughput tests for `strata_earth`.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

use strata_earth::{
    tile_bounds, traverse_quadtree_lod_frustum6, EarthCamera, EarthInput, EarthScene,
    EarthViewport, Frustum6, Geodetic, LayerToggles, MemoryTileSource, RetainedTerrainBuilder,
    SurfaceMode, TileElevation, TileImage, TileKey,
};

fn quadtree_input(lat: f64, lon: f64, range_m: f64) -> EarthInput {
    EarthInput {
        camera: EarthCamera {
            focus: Geodetic::new(lat, lon, 0.0),
            eye_offset_m: [0.0, -range_m * 0.4, range_m * 0.9],
            ..EarthCamera::default()
        },
        viewport: strata_layout::Rect::new(0.0, 0.0, 1440.0, 900.0),
        surface_mode: SurfaceMode::QuadtreeLod,
        tile_grid: 9,
        max_tile_level: 14,
        sse_threshold: 4.0,
        layers: LayerToggles {
            buildings: false,
            ..LayerToggles::default()
        },
        ..EarthInput::default()
    }
}

fn populated_source_for_leaves(leaves: &[TileKey]) -> MemoryTileSource {
    let mut source = MemoryTileSource::new();
    for &key in leaves {
        source.insert_imagery(
            key,
            TileImage {
                width: 4,
                height: 4,
                pixels: [64, 140, 210, 255].repeat(16),
            },
        );
        source.insert_elevation(
            key,
            TileElevation {
                width: 5,
                height: 5,
                heights_m: vec![120.0; 25],
            },
        );
    }
    source
}

#[test]
fn frustum6_culls_all_six_clip_planes() {
    let cam = EarthCamera {
        focus: Geodetic::new(37.7749, -122.4194, 0.0),
        eye_offset_m: [0.0, -1_200.0, 2_400.0],
        ..EarthCamera::default()
    };
    let p = cam.placement();
    let frustum = Frustum6::from_camera(&cam, 1440.0, 900.0);

    let forward = p.aim.sub(p.eye).normalize();
    let right = forward.cross(p.up).normalize();
    let true_up = right.cross(forward).normalize();

    let mid_dist = (p.near_m + p.far_m) * 0.5;
    let half_h = mid_dist * (p.fov_y_deg.to_radians() * 0.5).tan();
    let half_w = half_h * (1440.0 / 900.0);

    // Center of frustum must intersect.
    let center_pt = p.eye.add(forward.scale(mid_dist));
    assert!(frustum.intersects_sphere(center_pt, 10.0));

    // Behind eye & inside near gap must be culled.
    assert!(!frustum.intersects_sphere(p.eye.sub(forward.scale(100.0)), 5.0));
    if p.near_m > 20.0 {
        assert!(!frustum.intersects_sphere(p.eye.add(forward.scale(p.near_m * 0.25)), 1.0));
    }

    // Beyond far plane must be culled.
    assert!(!frustum.intersects_sphere(p.eye.add(forward.scale(p.far_m + 50_000.0)), 10.0));

    // Outside Left, Right, Top, Bottom planes must be culled.
    assert!(!frustum.intersects_sphere(center_pt.sub(right.scale(half_w + 5_000.0)), 10.0));
    assert!(!frustum.intersects_sphere(center_pt.add(right.scale(half_w + 5_000.0)), 10.0));
    assert!(!frustum.intersects_sphere(center_pt.add(true_up.scale(half_h + 5_000.0)), 10.0));
    assert!(!frustum.intersects_sphere(center_pt.sub(true_up.scale(half_h + 5_000.0)), 10.0));
}

#[test]
fn traverse_quadtree_lod_frustum6_selects_valid_leaves() {
    let input = quadtree_input(37.7749, -122.4194, 4_000.0);
    let leaves = traverse_quadtree_lod_frustum6(
        &input.camera,
        input.viewport.width as f64,
        input.viewport.height as f64,
        input.sse_threshold,
        input.max_tile_level,
        input.tile_grid,
    );
    assert!(!leaves.is_empty());
    let frustum = Frustum6::from_camera(&input.camera, 1440.0, 900.0);
    for key in &leaves {
        assert!(key.level <= input.max_tile_level);
        let b = tile_bounds(*key);
        let center_geo = Geodetic::new(0.5 * (b.lat0 + b.lat1), 0.5 * (b.lon0 + b.lon1), 0.0);
        let _ = strata_earth::geodetic_to_ecef(center_geo);
        let (sphere_center, sphere_r) = strata_earth::tile_bounding_sphere(*key);
        assert!(frustum.intersects_sphere(sphere_center, sphere_r * 1.25));
    }
}

#[test]
fn retained_terrain_builder_reuses_cached_tile_meshes_and_textures() {
    let input = quadtree_input(37.7749, -122.4194, 8_000.0);
    let leaves = traverse_quadtree_lod_frustum6(
        &input.camera,
        1440.0,
        900.0,
        input.sse_threshold,
        input.max_tile_level,
        input.tile_grid,
    );
    let mut source = populated_source_for_leaves(&leaves);
    let mut builder = RetainedTerrainBuilder::new();

    let (frame1, stats1) = builder.build_frame(&input, &mut source, 1, false);
    assert!(stats1.visible_tiles > 0);
    assert_eq!(stats1.mesh_cache_misses, stats1.visible_tiles);
    assert_eq!(stats1.mesh_cache_hits, 0);

    let (frame2, stats2) = builder.build_frame(&input, &mut source, 2, false);
    assert_eq!(stats2.visible_tiles, stats1.visible_tiles);
    assert_eq!(stats2.mesh_cache_misses, 0);
    assert_eq!(stats2.mesh_cache_hits, stats1.visible_tiles);
    assert_eq!(frame1.instances.len(), frame2.instances.len());
    assert_eq!(frame1.meshes.len(), frame2.meshes.len());
    assert_eq!(frame1.textures.len(), frame2.textures.len());
    for (a, b) in frame1.meshes.iter().zip(frame2.meshes.iter()) {
        assert!(Arc::ptr_eq(a, b));
    }
    for (a, b) in frame1.textures.iter().zip(frame2.textures.iter()) {
        assert!(Arc::ptr_eq(a, b));
    }
}

#[test]
fn earth_viewport_non_blocking_camera_update_and_ancestor_fallback() {
    let mut source = MemoryTileSource::new();
    // Populate only coarse level-0/1/2 ancestor tiles so zoomed-in camera must use fallback.
    for level in 0..=2u8 {
        let count = 1u32 << level;
        for y in 0..count {
            for x in 0..count {
                source.insert_imagery(
                    TileKey::new(level, x, y),
                    TileImage {
                        width: 2,
                        height: 2,
                        pixels: [90, 160, 110, 255].repeat(4),
                    },
                );
            }
        }
    }

    let wakeups = Arc::new(AtomicUsize::new(0));
    let wake_cb = {
        let wakeups = Arc::clone(&wakeups);
        Arc::new(move || {
            wakeups.fetch_add(1, Ordering::Relaxed);
        })
    };

    let input = quadtree_input(37.7749, -122.4194, 5_000.0);
    let mut viewport = EarthViewport::with_wakeup(&input, Box::new(source), Some(wake_cb));
    viewport.sync_now();

    let handle = viewport.retained_scene();
    let frame = handle.snapshot_terrain().expect("initial terrain frame");
    assert!(!frame.instances.is_empty());
    assert!(
        viewport.stats().ancestor_fallbacks > 0,
        "expected coarse ancestor texture fallbacks when finer tiles are missing"
    );
    assert!(wakeups.load(Ordering::Relaxed) >= 1);

    let initial_cam_gen = handle.camera_generation();
    let t0 = Instant::now();
    let mut cam = input.camera;
    for i in 0..100 {
        cam.focus.lon_deg += 0.0001 * (i as f64 + 1.0);
        viewport.update_camera(&cam, 1440.0, 900.0);
    }
    let elapsed_cam_only = t0.elapsed();
    assert_eq!(handle.camera_generation(), initial_cam_gen + 100);
    assert!(
        elapsed_cam_only.as_millis() < 50,
        "update_camera must remain non-blocking O(1), took {:?}",
        elapsed_cam_only
    );

    viewport.sync_now();
    assert!(viewport.stats().worker_builds >= 1);
}

#[test]
fn earth_scene_build_retained_fast_path_1000_frames_quadtree_and_flat() {
    // 1. Synchronous EarthScene::build_retained fast path in QuadtreeLod.
    let mut input = quadtree_input(37.7749, -122.4194, 15_000.0);
    // Keep the 1,000-frame synchronous geometry-stability benchmark independent of async
    // background cloud/skybox/satellite worker completions when STRATA_EARTH_OFFLINE is unset.
    input.layers.clouds = false;
    input.space.stars_visible = false;
    input.space.satellites_visible = false;
    let leaves = traverse_quadtree_lod_frustum6(
        &input.camera,
        1440.0,
        900.0,
        input.sse_threshold,
        input.max_tile_level,
        input.tile_grid,
    );
    let mut source = populated_source_for_leaves(&leaves);
    let mut scene = EarthScene::new();

    // Warm-up initial retained build.
    scene.build_retained(&input, &mut source);
    let handle = scene.retained_scene();
    let start_cam_gen = handle.camera_generation();
    let initial_geom_gen = handle.geometry_generation();

    let base_y = input.camera.eye_offset_m[1];
    let t0 = Instant::now();
    let frames = 1_000u64;
    for i in 0..frames {
        let phase = (i as f64) * 0.005;
        input.camera.eye_offset_m[0] = 2.0 * phase.sin();
        input.camera.eye_offset_m[1] = base_y + 2.0 * phase.cos();
        scene.build_retained(&input, &mut source);
    }
    let elapsed = t0.elapsed();
    let per_frame_us = elapsed.as_secs_f64() * 1e6 / frames as f64;
    eprintln!(
        "[bench earth QuadtreeLod build_retained] 1000 frames in {:.2} ms ({:.2} us/frame)",
        elapsed.as_secs_f64() * 1000.0,
        per_frame_us
    );
    assert_eq!(handle.camera_generation(), start_cam_gen + frames);
    assert_eq!(handle.geometry_generation(), initial_geom_gen);
    assert_eq!(scene.retained_stats().mesh_cache_misses, 0);

    // 2. Non-blocking EarthViewport::update_camera 1,000-frame benchmark for both QuadtreeLod and FlatPatches.
    for mode in [SurfaceMode::QuadtreeLod, SurfaceMode::FlatPatches] {
        let mut mode_input = quadtree_input(37.7749, -122.4194, 15_000.0);
        mode_input.surface_mode = mode;
        let vp_source = populated_source_for_leaves(&leaves);
        let mut viewport = EarthViewport::new(&mode_input, Box::new(vp_source));
        viewport.sync_now();

        let vp_handle = viewport.retained_scene();
        let cam_start = vp_handle.camera_generation();
        let mut cam = mode_input.camera;
        let vp_t0 = Instant::now();
        for i in 0..frames {
            let phase = (i as f64) * 0.01;
            cam.eye_offset_m[0] = 500.0 * phase.sin();
            cam.eye_offset_m[1] = base_y + 300.0 * phase.cos();
            viewport.update_camera(&cam, 1440.0, 900.0);
        }
        let vp_elapsed = vp_t0.elapsed();
        eprintln!(
            "[bench earth {:?} EarthViewport::update_camera] 1000 frames in {:.2} ms ({:.2} us/frame)",
            mode,
            vp_elapsed.as_secs_f64() * 1000.0,
            vp_elapsed.as_secs_f64() * 1e6 / frames as f64
        );
        assert_eq!(vp_handle.camera_generation(), cam_start + frames);
        assert!(
            vp_elapsed.as_millis() < 100,
            "1000 EarthViewport::update_camera calls for {:?} took too long: {:?}",
            mode,
            vp_elapsed
        );
        viewport.sync_now();
    }
}

#[test]
fn heightmap_average_altitude_scales_tile_lod_distance_and_defaults_to_sea_level_when_off() {
    use strata_earth::{TileElevation, TileProjection};

    // 1. Verify TileElevation average elevation across whole tile & UV sub-rectangles.
    // Top half (rows 0..2) = 4,400m, bottom half (rows 2..4) = 3,600m -> overall mean = 4,000m.
    let elev = TileElevation {
        width: 4,
        height: 4,
        heights_m: vec![
            4400.0, 4400.0, 4400.0, 4400.0,
            4400.0, 4400.0, 4400.0, 4400.0,
            3600.0, 3600.0, 3600.0, 3600.0,
            3600.0, 3600.0, 3600.0, 3600.0,
        ],
    };
    assert!((elev.average_elevation_m() - 4000.0).abs() < 1e-3);
    // Northern sub-quadrant [0..0.5] x [0..0.5] should average near 4,400m.
    let north_avg = elev.average_elevation_in_uv_rect_m(0.0, 0.0, 0.5, 0.45);
    assert!(
        north_avg > 4200.0,
        "north sub-rect average should be > 4200m, got {north_avg}"
    );

    // 2. Populate an offline MemoryTileSource with a 4,200m mountain plateau DEM at root ancestor z=0..2.
    let plateau_elev = TileElevation {
        width: 8,
        height: 8,
        heights_m: vec![4_200.0; 64],
    };
    let mut src_elev_off = MemoryTileSource::default();
    let mut src_elev_on = MemoryTileSource::default();
    src_elev_off.insert_elevation(TileKey::new(0, 0, 0), plateau_elev.clone());
    src_elev_on.insert_elevation(TileKey::new(0, 0, 0), plateau_elev);

    // Camera hovering at 4,850 m MSL (only 650 m above the 4,200 m mountain plateau, vs 4,850 m above sea level).
    let mut base_in = quadtree_input(46.55, 7.98, 650.0);
    base_in.tile_projection = TileProjection::WebMercator;
    base_in.camera.focus.height_m = 4_200.0;
    base_in.camera.eye_offset_m = [0.0, 0.0, 650.0];
    base_in.max_tile_level = 18;

    let mut off_in = base_in.clone();
    off_in.elevation_enabled = false;
    let mut builder_off = RetainedTerrainBuilder::new();
    let (frame_off, _) = builder_off.build_frame(&off_in, &mut src_elev_off, 1, false);
    let max_z_off = frame_off
        .instances
        .iter()
        .map(|inst| inst.key.level)
        .max()
        .unwrap_or(0);

    let mut on_in = base_in;
    on_in.elevation_enabled = true;
    let mut builder_on = RetainedTerrainBuilder::new();
    let (frame_on, _) = builder_on.build_frame(&on_in, &mut src_elev_on, 1, false);
    let max_z_on = frame_on
        .instances
        .iter()
        .map(|inst| inst.key.level)
        .max()
        .unwrap_or(0);

    assert!(
        max_z_on >= max_z_off + 2,
        "heightmap average altitude (4,200m) must place tile much closer to 4,850m MSL eye than sea level 0m (max_z_off={max_z_off}, max_z_on={max_z_on})"
    );
}

#[test]
fn idle_high_res_refinement_defaults_off_and_refines_viewed_tiles_when_opted_in() {
    use strata_earth::{IdleHighResConfig, TileProjection};

    // 1. Default config must be OFF with zero extra worker activity while idle.
    let mut input = quadtree_input(46.5775, 7.9850, 8_500.0);
    input.tile_projection = TileProjection::WebMercator;
    input.elevation_enabled = false;
    input.camera.eye_offset_m = [0.0, -6_500.0, 5_500.0];
    assert!(
        !input.idle_high_res.enabled,
        "IdleHighResConfig must be OFF by default"
    );

    let mut vp = EarthViewport::new(&input, Box::new(MemoryTileSource::default()));
    vp.sync_now();
    let base_builds = vp.stats().worker_builds;
    let base_frame = vp
        .retained_scene()
        .snapshot_terrain()
        .expect("baseline frame");
    let base_tiles = base_frame.instances.len();
    let base_max_z = base_frame
        .instances
        .iter()
        .map(|i| i.key.level)
        .max()
        .unwrap_or(0);

    for _ in 0..20 {
        assert!(!vp.step_idle_refinement(0.1, false));
    }
    assert!(!vp.is_idle_high_res_active());
    assert_eq!(vp.stats().worker_builds, base_builds);

    // 2. Opt in to IdleHighResConfig -> staying stationary progressively sharpens viewed tiles!
    vp.set_idle_high_res_config(IdleHighResConfig::enabled().with_idle_delay_s(0.25));
    assert!(!vp.is_idle_high_res_active());

    // Advance past stage 1 (0.25s) and stage 2 (0.55s).
    let mut promoted_count = 0usize;
    for _ in 0..8 {
        if vp.step_idle_refinement(0.1, false) {
            promoted_count += 1;
        }
    }
    assert_eq!(promoted_count, 2, "expected stage 1 and stage 2 promotions");
    assert!(vp.is_idle_high_res_active());
    assert!((vp.idle_refinement_factor() - 1.0).abs() < 1e-6);
    vp.sync_now();

    let refined_frame = vp
        .retained_scene()
        .snapshot_terrain()
        .expect("idle refined frame");
    let refined_tiles = refined_frame.instances.len();
    let refined_max_z = refined_frame
        .instances
        .iter()
        .map(|i| i.key.level)
        .max()
        .unwrap_or(0);
    assert!(
        refined_max_z > base_max_z && refined_tiles > base_tiles,
        "idle high-res refinement must subdivide viewed tiles deeper (base: z={base_max_z} n={base_tiles} -> idle: z={refined_max_z} n={refined_tiles})"
    );

    // 3. Moving the camera immediately resets idle refinement back to fast moving LOD!
    let mut moved_cam = input.camera;
    moved_cam.focus.lon_deg += 0.05;
    vp.update_camera(&moved_cam, 1440.0, 900.0);
    assert!(!vp.is_idle_high_res_active());
    assert_eq!(vp.idle_refinement_factor(), 0.0);
    vp.sync_now();
}

#[test]
fn coons_neighbor_synthesis_and_ancestor_stitching_match_loaded_edges_with_zero_seam() {
    use strata_earth::{
        stitch_elevation_with_neighbors, synthesize_elevation_from_neighbors,
        NeighborElevationEdges, TileElevation,
    };

    let n = 17usize;
    let denom = (n - 1) as f32;
    // Reference smooth tilted mountain ridge: h(u, v) = 2600 + 600*u + 400*v + 180*sin(pi*u)*sin(pi*v)
    let edge_h = |u: f32, v: f32| -> f32 { 2600.0 + 600.0 * u + 400.0 * v };
    let west: Vec<f32> = (0..n).map(|r| edge_h(0.0, r as f32 / denom)).collect();
    let east: Vec<f32> = (0..n).map(|r| edge_h(1.0, r as f32 / denom)).collect();
    let north: Vec<f32> = (0..n).map(|c| edge_h(c as f32 / denom, 0.0)).collect();
    let south: Vec<f32> = (0..n).map(|c| edge_h(c as f32 / denom, 1.0)).collect();

    // 1. All 4 cardinal edges loaded -> 0.001 m exact match along EVERY boundary vertex.
    let full = synthesize_elevation_from_neighbors(
        n as u32,
        &NeighborElevationEdges {
            west: Some(west.clone()),
            east: Some(east.clone()),
            north: Some(north.clone()),
            south: Some(south.clone()),
            ..Default::default()
        },
    )
    .expect("full 4-edge synthesis");
    for i in 0..n {
        assert!((full.heights_m[i * n] - west[i]).abs() < 1e-3);
        assert!((full.heights_m[i * n + (n - 1)] - east[i]).abs() < 1e-3);
        assert!((full.heights_m[i] - north[i]).abs() < 1e-3);
        assert!((full.heights_m[(n - 1) * n + i] - south[i]).abs() < 1e-3);
    }

    // 2. Only 2 adjacent edges loaded (West + North) -> exact match along W & N, interior stays high.
    let corner2 = synthesize_elevation_from_neighbors(
        n as u32,
        &NeighborElevationEdges {
            west: Some(west.clone()),
            north: Some(north.clone()),
            ..Default::default()
        },
    )
    .expect("2-edge synthesis");
    for i in 0..n {
        assert!((corner2.heights_m[i * n] - west[i]).abs() < 1e-3);
        assert!((corner2.heights_m[i] - north[i]).abs() < 1e-3);
    }
    assert!(
        corner2.heights_m.iter().all(|&h| h >= 2_500.0 && h <= 3_750.0),
        "2-edge adaptation must never plunge toward sea level"
    );

    // 3. Only 1 single edge loaded (West ridge) -> exact along West, whole tile stays at ridge height.
    let single1 = synthesize_elevation_from_neighbors(
        n as u32,
        &NeighborElevationEdges {
            west: Some(west.clone()),
            ..Default::default()
        },
    )
    .expect("1-edge synthesis");
    for i in 0..n {
        assert!((single1.heights_m[i * n] - west[i]).abs() < 1e-3);
        assert!(
            single1.heights_m[i * n + (n - 1)] >= 2_550.0,
            "far edge must extrude smoothly from single loaded edge"
        );
    }

    // 4. Only 1 diagonal corner loaded -> lifts whole tile to corner altitude instead of 0.0 m.
    let diag_only = synthesize_elevation_from_neighbors(
        n as u32,
        &NeighborElevationEdges {
            nw: Some(2_850.0),
            ..Default::default()
        },
    )
    .expect("diagonal-only synthesis");
    assert!(diag_only.heights_m.iter().all(|&h| (h - 2_850.0).abs() < 1e-3));

    // 5. Coarse ancestor stitching locks finer East neighbor edge while keeping West & interior peak.
    let mut coarse_vals = Vec::with_capacity(n * n);
    for r in 0..n {
        let v = r as f32 / denom;
        for c in 0..n {
            let u = c as f32 / denom;
            let peak = 650.0 * (std::f32::consts::PI * u).sin() * (std::f32::consts::PI * v).sin();
            coarse_vals.push(2_400.0 + peak);
        }
    }
    let coarse = TileElevation {
        width: n as u32,
        height: n as u32,
        heights_m: coarse_vals,
    };
    let finer_east: Vec<f32> = (0..n).map(|r| 2_950.0 + 120.0 * (r as f32 / denom)).collect();
    let stitched = stitch_elevation_with_neighbors(
        &coarse,
        (0.0, 0.0, 1.0, 1.0),
        n as u32,
        &NeighborElevationEdges {
            east: Some(finer_east.clone()),
            ..Default::default()
        },
    );
    for r in 0..n {
        assert!(
            (stitched.heights_m[r * n + (n - 1)] - finer_east[r]).abs() < 1e-3,
            "stitched East edge must match finer neighbor exactly"
        );
        assert!(
            (stitched.heights_m[r * n] - 2_400.0).abs() < 1e-3,
            "unstitched West edge must stay at coarse ancestor height"
        );
    }
    let center_h = stitched.heights_m[(n / 2) * n + (n / 2)];
    assert!(
        center_h > 3_150.0,
        "interior coarse mountain peak must be preserved after stitching (got {center_h})"
    );
}

#[test]
fn unloaded_single_tile_adapts_watertight_to_surrounding_heightmaps_until_loaded() {
    use strata_earth::{
        geodetic_to_ecef, project_morphed_point_and_frame, stabilized_surface_morph,
        tile_bounds_mercator, Geodetic, TileElevation, TileProjection,
    };

    let focus_lat = 46.0207;
    let focus_lon = 7.7491;
    let mut input = quadtree_input(focus_lat, focus_lon, 5_500.0);
    input.tile_projection = TileProjection::WebMercator;
    input.elevation_enabled = true;
    input.max_tile_level = 12;
    input.tile_grid = 9;
    input.sse_threshold = 4.0;
    input.camera.eye_offset_m = [0.0, -3_200.0, 4_800.0];

    // Probe visible tiles first using an empty source so we know the exact center tile key.
    let mut probe_builder = RetainedTerrainBuilder::new();
    let mut empty_src = MemoryTileSource::default();
    let (probe_frame, _) = probe_builder.build_frame(&input, &mut empty_src, 1, false);
    assert!(probe_frame.instances.len() >= 4);

    // Pick the visible tile covering (focus_lat, focus_lon) as the single unloaded tile.
    let missing_key = probe_frame
        .instances
        .iter()
        .map(|i| TileKey::new(i.key.level, i.key.col, i.key.row))
        .find(|&k| {
            let b = tile_bounds_mercator(k);
            focus_lat >= b.lat0 && focus_lat <= b.lat1 && focus_lon >= b.lon0 && focus_lon <= b.lon1
        })
        .expect("center tile under camera must be visible");

    // Populate ALL surrounding tiles (every other visible tile + 8-neighborhood at missing_key.level)
    // with continuous 2,900..3,400 m alpine relief, while leaving `missing_key` completely unloaded!
    let mut src = MemoryTileSource::default();
    let make_alpine_dem = |k: TileKey| -> TileElevation {
        let b = tile_bounds_mercator(k);
        let res = 9usize;
        let mut heights = Vec::with_capacity(res * res);
        for row in 0..res {
            let v_tex = row as f64 / (res - 1) as f64;
            let lat = b.lat1 + (b.lat0 - b.lat1) * v_tex;
            for col in 0..res {
                let u = col as f64 / (res - 1) as f64;
                let lon = b.lon0 + (b.lon1 - b.lon0) * u;
                let h = 3_050.0 + (lat - focus_lat) * 2_200.0 + (lon - focus_lon) * 1_800.0;
                heights.push(h as f32);
            }
        }
        TileElevation {
            width: res as u32,
            height: res as u32,
            heights_m: heights,
        }
    };

    for inst in &probe_frame.instances {
        let k = TileKey::new(inst.key.level, inst.key.col, inst.key.row);
        if k != missing_key {
            src.insert_elevation(k, make_alpine_dem(k));
        }
    }
    let per_axis = missing_key.tiles_per_axis();
    for dy in -1i32..=1 {
        for dx in -1i32..=1 {
            if dx == 0 && dy == 0 {
                continue;
            }
            let nx = ((missing_key.x as i64 + dx as i64).rem_euclid(per_axis as i64)) as u32;
            let ny = missing_key.y as i64 + dy as i64;
            if ny >= 0 && ny < per_axis as i64 {
                let nk = TileKey::new(missing_key.level, nx, ny as u32);
                src.insert_elevation(nk, make_alpine_dem(nk));
            }
        }
    }

    let mut builder = RetainedTerrainBuilder::new();
    let (frame_missing, _) = builder.build_frame(&input, &mut src, 1, false);

    // 1. Ground elevation sampled right over the unloaded tile must adapt to ~3,050 m (NOT 0 m!).
    let sampled_missing = builder.sample_ground_elevation_m(focus_lat, focus_lon);
    assert!(
        (sampled_missing - 3_050.0).abs() < 120.0,
        "unloaded center tile must adapt to surrounding ~3,050 m terrain, got {sampled_missing} m"
    );

    // 2. Inspect actual GPU vertex heights of the unloaded tile mesh: EVERY top-grid vertex must
    //    sit smoothly near ~3,050 m (zero drop to sea level / undermap exposure!).
    let missing_id = strata_render::scene3d::TileKeyId::quadtree(
        missing_key.level,
        missing_key.x,
        missing_key.y,
    );
    let missing_mesh = frame_missing
        .meshes
        .iter()
        .find(|m| m.key == missing_id)
        .expect("unloaded center tile mesh must be emitted");
    let ll = input.low_level_state();
    let morph = stabilized_surface_morph(
        input.camera.focus,
        ll.local_25d_blend.max(ll.map_2d_morph) as f64,
        ll.map_2d_morph as f64,
        input.tile_projection,
    );
    let (base_p, base_frame) =
        project_morphed_point_and_frame(focus_lat, focus_lon, 0.0, morph);
    let n = missing_mesh.grid_res as usize;
    let center_dvec = strata_render::camera64::DVec3::new(
        missing_mesh.center_ecef[0],
        missing_mesh.center_ecef[1],
        missing_mesh.center_ecef[2],
    );
    for idx in 0..(n * n) {
        let lp = missing_mesh.vertices[idx].position();
        let wp = center_dvec.add(strata_render::camera64::DVec3::new(
            lp[0] as f64,
            lp[1] as f64,
            lp[2] as f64,
        ));
        let h_up = wp.sub(base_p).dot(base_frame.up);
        assert!(
            h_up > 2_600.0 && h_up < 3_600.0,
            "vertex {idx} of unloaded tile dropped out of surrounding alpine height band: h={h_up} m"
        );
    }

    // 3. Verify 0 m watertight match against an adjacent loaded neighbor's shared edge vertices!
    let east_key = TileKey::new(missing_key.level, (missing_key.x + 1) % per_axis, missing_key.y);
    let east_id =
        strata_render::scene3d::TileKeyId::quadtree(east_key.level, east_key.x, east_key.y);
    if let Some(east_mesh) = frame_missing.meshes.iter().find(|m| m.key == east_id) {
        let east_c = strata_render::camera64::DVec3::new(
            east_mesh.center_ecef[0],
            east_mesh.center_ecef[1],
            east_mesh.center_ecef[2],
        );
        for row in 0..n {
            let p_a = missing_mesh.vertices[row * n + (n - 1)].position();
            let p_b = east_mesh.vertices[row * n].position();
            let wa = center_dvec.add(strata_render::camera64::DVec3::new(
                p_a[0] as f64,
                p_a[1] as f64,
                p_a[2] as f64,
            ));
            let wb = east_c.add(strata_render::camera64::DVec3::new(
                p_b[0] as f64,
                p_b[1] as f64,
                p_b[2] as f64,
            ));
            let gap_m = wa.sub(wb).length();
            assert!(
                gap_m < 1.0,
                "shared East edge row {row} must be watertight (<1 m gap), got {gap_m} m"
            );
        }
    }

    // 4. Now simulate the missing tile's real DEM finishing download from the internet (3,950 m peak)
    //    and confirm the tile cleanly upgrades to its true elevation!
    let _ = geodetic_to_ecef(Geodetic::new(focus_lat, focus_lon, 0.0));
    src.insert_elevation(
        missing_key,
        TileElevation {
            width: 9,
            height: 9,
            heights_m: vec![3_950.0; 81],
        },
    );
    let (_frame_loaded, _) = builder.build_frame(&input, &mut src, 2, false);
    let sampled_loaded = builder.sample_ground_elevation_m(focus_lat, focus_lon);
    assert!(
        (sampled_loaded - 3_950.0).abs() < 1.0,
        "after real DEM loads, tile must upgrade to exact 3,950 m height, got {sampled_loaded} m"
    );
}

#[test]
fn frame_zero_bootstrap_populates_root_tile_and_procedural_skybox_synchronously() {
    // 1. Default EarthInput keeps async behavior (frame_zero_bootstrap = false).
    let default_input = quadtree_input(37.7749, -122.4194, 150_000.0);
    assert!(!default_input.frame_zero_bootstrap);

    // 2. Opting in via EarthInput::with_frame_zero_bootstrap(true) publishes a Frame-0
    //    terrain frame immediately on EarthViewport::new without calling sync_now().
    let bootstrap_input = default_input.clone().with_frame_zero_bootstrap(true);
    assert!(bootstrap_input.frame_zero_bootstrap);
    let vp_input = EarthViewport::new(&bootstrap_input, Box::new(MemoryTileSource::default()));
    let frame_vp = vp_input
        .retained_scene()
        .snapshot_terrain()
        .expect("Frame-0 bootstrap via EarthInput flag must populate retained terrain immediately");
    assert_eq!(frame_vp.meshes.len(), 1);
    assert_eq!(frame_vp.instances.len(), 1);
    assert_eq!(
        frame_vp.instances[0].key,
        strata_render::scene3d::TileKeyId::quadtree(0, 0, 0)
    );
    assert_eq!(frame_vp.textures.len(), 1);
    let env_vp = frame_vp
        .environment
        .as_ref()
        .expect("Frame-0 bootstrap must attach procedural skybox & environment assets");
    assert!(env_vp.cloud_map.is_some());
    assert!(env_vp.skybox_cubemap.is_none());

    // 3. Explicit EarthViewport::new_with_bootstrap also populates Frame-0 immediately
    //    even when input.frame_zero_bootstrap is false, and prefers cached root imagery if present.
    let mut cached_src = MemoryTileSource::new();
    let custom_root = TileImage {
        width: 2,
        height: 2,
        pixels: [11, 22, 33, 255].repeat(4),
    };
    cached_src.insert_imagery(TileKey::new(0, 0, 0), custom_root);
    let vp_explicit =
        EarthViewport::new_with_bootstrap(&default_input, Box::new(cached_src), true);
    let frame_explicit = vp_explicit
        .retained_scene()
        .snapshot_terrain()
        .expect("EarthViewport::new_with_bootstrap(true) must populate Frame-0 immediately");
    assert_eq!(frame_explicit.textures.len(), 1);
    assert_eq!(frame_explicit.textures[0].width, 2);
    assert_eq!(frame_explicit.textures[0].rgba8[..4], [11, 22, 33, 255]);

    // 4. On-demand EarthViewport::bootstrap_frame_zero works when unbootstrapped and is idempotent.
    let mut vp_manual =
        EarthViewport::new_with_bootstrap(&default_input, Box::new(MemoryTileSource::default()), false);
    if vp_manual.bootstrap_frame_zero() {
        let f = vp_manual
            .retained_scene()
            .snapshot_terrain()
            .expect("bootstrap_frame_zero() must install Frame-0 frame");
        assert_eq!(f.instances.len(), 1);
        assert!(f.environment.is_some());
        assert!(
            !vp_manual.bootstrap_frame_zero(),
            "second bootstrap_frame_zero() call must be a no-op once a frame exists"
        );
    }

    // 5. Synchronous EarthScene::bootstrap_frame_zero works without a worker and later upgrades
    //    cleanly when build_retained runs.
    let mut earth_scene = EarthScene::new();
    let scene_frame = earth_scene.bootstrap_frame_zero(&default_input, None);
    assert_eq!(scene_frame.meshes.len(), 1);
    assert_eq!(scene_frame.instances.len(), 1);
    assert!(scene_frame.environment.is_some());
    assert!(
        scene_frame
            .environment
            .as_ref()
            .unwrap()
            .skybox_cubemap
            .is_none()
    );
    assert_eq!(
        earth_scene
            .retained_scene()
            .snapshot_terrain()
            .map(|f| f.instances.len()),
        Some(1)
    );

    let mut full_src = MemoryTileSource::new();
    earth_scene.build_retained(&default_input, &mut full_src);
    let upgraded = earth_scene
        .retained_scene()
        .snapshot_terrain()
        .expect("build_retained after Frame-0 bootstrap must publish full LOD frame");
    assert!(upgraded.instances.len() >= 1);
}

#[test]
fn earth_quality_preset_matrix_roundtrip_and_custom_detection() {
    use strata_earth::{
        CloudMapResolution, EarthInput, EarthQualityPreset, EarthRenderQuality, SpaceSkyboxQuality,
    };

    // 1. Canonical preset cycling, parsing, and display labels
    assert_eq!(EarthQualityPreset::Low.next_canonical(), EarthQualityPreset::Medium);
    assert_eq!(EarthQualityPreset::Medium.next_canonical(), EarthQualityPreset::High);
    assert_eq!(EarthQualityPreset::High.next_canonical(), EarthQualityPreset::Ultra);
    assert_eq!(EarthQualityPreset::Ultra.next_canonical(), EarthQualityPreset::Low);
    assert_eq!(EarthQualityPreset::Custom.next_canonical(), EarthQualityPreset::Low);

    assert_eq!(EarthQualityPreset::from_str_loose("LOW"), Some(EarthQualityPreset::Low));
    assert_eq!(EarthQualityPreset::from_str_loose("med"), Some(EarthQualityPreset::Medium));
    assert_eq!(EarthQualityPreset::from_str_loose("high"), Some(EarthQualityPreset::High));
    assert_eq!(EarthQualityPreset::from_str_loose("Ultra"), Some(EarthQualityPreset::Ultra));
    assert_eq!(EarthQualityPreset::from_str_loose("custom"), Some(EarthQualityPreset::Custom));
    assert_eq!(EarthQualityPreset::from_str_loose("unknown"), None);

    // 2. Every canonical preset round-trips cleanly through EarthInput
    for preset in EarthQualityPreset::ALL_CANONICAL {
        let q = EarthRenderQuality::from_preset(preset);
        assert_eq!(q.matching_preset(), preset);

        let input = EarthInput::globe().with_quality_preset(preset);
        assert_eq!(input.quality_preset(), preset);
        assert_eq!(input.render_quality(), q);
    }

    // 3. Tolerance check on sse_threshold (<= 1e-3 still matches; > 1e-3 becomes Custom)
    let mut near = EarthRenderQuality::from_preset(EarthQualityPreset::High);
    near.sse_threshold += 5e-4;
    assert_eq!(near.matching_preset(), EarthQualityPreset::High);
    near.sse_threshold += 2e-3;
    assert_eq!(near.matching_preset(), EarthQualityPreset::Custom);

    // 4. Exhaustive single-field mutation test: mutating ANY of the 12 quality fields yields Custom
    let base = EarthRenderQuality::from_preset(EarthQualityPreset::High);
    let mutations: [(&str, fn(&mut EarthRenderQuality)); 12] = [
        ("tile_grid", |q| q.tile_grid = 21),
        ("sse_threshold", |q| q.sse_threshold = 2.0),
        ("max_tile_level", |q| q.max_tile_level = 17),
        ("realistic_sky_25d", |q| q.realistic_sky_25d = false),
        ("terrain_lighting_25d", |q| q.terrain_lighting_25d = false),
        ("terrain_shadows", |q| q.terrain_shadows = false),
        ("shadow_map_size", |q| q.shadow_map_size = 1024),
        ("skybox_quality", |q| q.skybox_quality = SpaceSkyboxQuality::Low),
        ("cloud_resolution", |q| q.cloud_resolution = CloudMapResolution::Low),
        ("ocean_coastal_shoaling", |q| q.ocean_coastal_shoaling = false),
        ("idle_high_res", |q| q.idle_high_res = true),
        ("idle_extra_leaf_budget", |q| q.idle_extra_leaf_budget = 256),
    ];
    for (field, mutate) in mutations {
        let mut q = base;
        mutate(&mut q);
        assert_eq!(
            q.matching_preset(),
            EarthQualityPreset::Custom,
            "Mutating `{field}` must immediately report EarthQualityPreset::Custom"
        );
    }
}

#[test]
fn earth_viewport_live_quality_preset_and_individual_setter_custom_tracking() {
    use strata_earth::{
        CloudMapResolution, EarthInput, EarthQualityPreset, EarthRenderQuality, EarthViewport,
        MemoryTileSource, SpaceSkyboxQuality,
    };

    let input = EarthInput::globe().with_quality_preset(EarthQualityPreset::High);
    let mut vp = EarthViewport::new(&input, Box::new(MemoryTileSource::new()));

    assert_eq!(vp.quality_preset(), EarthQualityPreset::High);
    assert_eq!(vp.tile_grid(), 25);
    assert!((vp.sse_threshold() - 2.3).abs() < 1e-6);
    assert_eq!(vp.max_tile_level(), 18);

    // Cycle Ultra -> Low -> Medium -> High live
    vp.set_quality_preset(EarthQualityPreset::Ultra);
    assert_eq!(vp.quality_preset(), EarthQualityPreset::Ultra);
    assert_eq!(vp.tile_grid(), 33);
    assert_eq!(vp.local_atmosphere_config().shadow_map_size, 4096);
    assert_eq!(vp.space_config().skybox_quality, SpaceSkyboxQuality::Ultra);
    assert_eq!(vp.weather_config().cloud_resolution, CloudMapResolution::Ultra);
    assert!(vp.idle_high_res_config().enabled);

    vp.set_quality_preset(EarthQualityPreset::Low);
    assert_eq!(vp.quality_preset(), EarthQualityPreset::Low);
    assert_eq!(vp.tile_grid(), 13);
    assert!(!vp.local_atmosphere_config().terrain_shadows);
    assert!(!vp.local_atmosphere_config().realistic_sky);
    assert!(!vp.ocean_waves_config().coastal_shoaling);

    vp.set_quality_preset(EarthQualityPreset::High);
    assert_eq!(vp.quality_preset(), EarthQualityPreset::High);

    // Individual setters automatically diverge into Custom and recover when reverted
    vp.set_idle_high_res_enabled(true);
    assert_eq!(vp.quality_preset(), EarthQualityPreset::Custom);
    vp.set_idle_high_res_enabled(false);
    assert_eq!(vp.quality_preset(), EarthQualityPreset::High);

    let mut atmo = vp.local_atmosphere_config().clone();
    atmo.terrain_shadows = false;
    vp.set_local_atmosphere_config(atmo);
    assert_eq!(vp.quality_preset(), EarthQualityPreset::Custom);

    let mut custom_q = EarthRenderQuality::from_preset(EarthQualityPreset::Medium);
    custom_q.tile_grid = 29;
    vp.set_render_quality(custom_q);
    assert_eq!(vp.quality_preset(), EarthQualityPreset::Custom);
    assert_eq!(vp.tile_grid(), 29);

    vp.set_tile_grid(17);
    assert_eq!(vp.quality_preset(), EarthQualityPreset::Medium);

    vp.set_sse_threshold(1.9);
    assert_eq!(vp.quality_preset(), EarthQualityPreset::Custom);
    vp.set_sse_threshold(2.7);
    assert_eq!(vp.quality_preset(), EarthQualityPreset::Medium);
}

#[cfg(feature = "widget")]
#[test]
fn earth_view_live_quality_preset_and_custom_detection() {
    use strata_earth::{EarthInput, EarthQualityPreset, EarthView, MemoryTileSource};

    let input = EarthInput::globe().with_quality_preset(EarthQualityPreset::High);
    let view = EarthView::new(&input, Box::new(MemoryTileSource::new()));
    assert_eq!(view.quality_preset(), EarthQualityPreset::High);

    view.set_quality_preset(EarthQualityPreset::Ultra);
    assert_eq!(view.quality_preset(), EarthQualityPreset::Ultra);
    assert_eq!(view.render_quality().shadow_map_size, 4096);

    view.set_idle_high_res_enabled(false);
    assert_eq!(view.quality_preset(), EarthQualityPreset::Custom);

    view.set_quality_preset(EarthQualityPreset::Medium);
    assert_eq!(view.quality_preset(), EarthQualityPreset::Medium);
}

#[test]
fn earth_viewport_center_on_coordinates_updates_camera_focus_and_range() {
    use strata_earth::{EarthInput, EarthViewport, MemoryTileSource};

    let input = EarthInput::globe();
    let mut vp = EarthViewport::new(&input, Box::new(MemoryTileSource::new()));
    vp.center_on_coordinates(37.7749, -122.4194, Some(22_000.0));
    let cam = vp.input().camera;
    assert!((cam.focus.lat_deg - 37.7749).abs() < 1e-6);
    assert!((cam.focus.lon_deg - (-122.4194)).abs() < 1e-6);
    let [ex, ey, ez] = cam.eye_offset_m;
    let dist = (ex * ex + ey * ey + ez * ez).sqrt();
    assert!((dist - 22_000.0).abs() < 1e-3);
}

#[cfg(feature = "widget")]
#[test]
fn earth_view_center_on_host_location_async_centers_without_blocking_frame_zero() {
    use strata_device::{DeviceMockController, GeolocationSource};
    use strata_earth::{EarthInput, EarthView, MemoryTileSource};

    let mock = DeviceMockController::new();
    mock.reset_to_defaults();
    mock.set_coordinates(41.0082, 28.9784, Some(40.0), 8.0);

    let input = EarthInput::globe();
    let view = EarthView::new(&input, Box::new(MemoryTileSource::new()));

    // Direct coordinate centering
    view.center_on_coordinates(35.6762, 139.6503, Some(60_000.0));
    let cam0 = view.camera_state();
    assert!((cam0.lat_deg - 35.6762).abs() < 1e-5);
    assert!((cam0.lon_deg - 139.6503).abs() < 1e-5);

    // Non-blocking startup host location probe centers camera when resolved during step_frame
    let t0 = std::time::Instant::now();
    view.center_on_host_location_async(Some(25_000.0), false);
    assert!(
        t0.elapsed() < std::time::Duration::from_millis(50),
        "center_on_host_location_async must never block startup frame 0"
    );

    let mut resolved = false;
    for _ in 0..80 {
        let _ = view.step_frame(1.0 / 60.0);
        if let Some(fix) = view.last_host_location() {
            assert_eq!(fix.source, GeolocationSource::MockOverride);
            assert!((fix.latitude - 41.0082).abs() < 1e-5);
            assert!((fix.longitude - 28.9784).abs() < 1e-5);
            resolved = true;
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    assert!(resolved, "Async host location fix must resolve and apply in step_frame");

    let cam1 = view.camera_state();
    assert!((cam1.lat_deg - 41.0082).abs() < 1e-4);
    assert!((cam1.lon_deg - 28.9784).abs() < 1e-4);
    assert!((cam1.range_m - 25_000.0).abs() < 10.0);

    mock.reset_to_defaults();
}



