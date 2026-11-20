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


