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
