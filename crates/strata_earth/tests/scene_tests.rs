//! Frame assembly: block accounting, the four camera regimes, the frame budget,
//! and the dependency-surface contract.
//!
//! The golden images are produced by `examples/strata_earth_golden`; this suite
//! checks the numbers those goldens depend on (tile resolution, block counts,
//! deterministic scene output) and the frame budget at 1440x900.

use std::time::Instant;
use strata_earth::{
    surrounding_tiles, tile_sampler, EarthInput, EarthScene, EntityMarker, FrameDescriptor,
    Geodetic, MemoryTileSource, TileImage, TileKey, TileSource,
};
use strata_render::overlay3d::OverlayPrimitive;
use strata_render::rasterizer::FrameBuffer;
use strata_render::rasterizer3d::render_scene_3d_with_textures;
use strata_render::scene3d::{Background3D, Geometry3D, Material3D};
use strata_style::types::Color;

const W: usize = 1440;
const H: usize = 900;

fn solid_tile(r: u8, g: u8, b: u8) -> TileImage {
    TileImage {
        width: 2,
        height: 2,
        pixels: (0..4).flat_map(|_| [r, g, b, 255]).collect(),
    }
}

fn source_for(keys: &[TileKey], image: &TileImage) -> MemoryTileSource {
    let mut source = MemoryTileSource::new();
    for key in keys {
        source.insert_imagery(*key, image.clone());
    }
    source
}

fn viewpoint(altitude_m: f64) -> EarthInput {
    EarthInput {
        camera: strata_earth::EarthCamera {
            focus: Geodetic::new(37.7749, -122.4194, 0.0),
            eye_offset_m: [0.0, 0.0, altitude_m],
            ..Default::default()
        },
        viewport: strata_layout::Rect::new(0.0, 0.0, W as f32, H as f32),
        tile_grid: 5,
        ..Default::default()
    }
}

fn render_scene(input: &EarthInput, scene: &EarthScene) -> FrameBuffer {
    let mut fb = FrameBuffer::new(W, H);
    render_scene_3d_with_textures(
        &mut fb,
        input.viewport,
        scene.scene(),
        None,
        Some(scene.tile_registry().store()),
    );
    fb
}

#[test]
fn full_globe_regime_resolves_tiles_and_draws_sky() {
    let input = viewpoint(12_000_000.0);
    let span = strata_earth::tile_patch_span_deg(&input.camera, input.patch_scale);
    let patches = strata_earth::surrounding_patches(input.camera.focus, span);
    let keys: Vec<TileKey> = patches.iter().map(strata_earth::requested_tile).collect();
    let mut source = source_for(&keys, &solid_tile(40, 90, 60));
    let mut scene = EarthScene::new();
    scene.build(&input, &mut source);

    assert!(scene.has_textured_geometry(), "tiles must reach the scene");
    assert_eq!(scene.stats().tiles_requested, keys.len());
    assert_eq!(scene.stats().tiles_exact, keys.len());

    let fb = render_scene(&input, &scene);
    assert!(
        fb.get_pixel(W / 2, 10).a() > 0.5,
        "the sky must fill the top edge"
    );
}

#[test]
fn missing_tiles_fall_back_to_the_loaded_ancestor() {
    // Load a single level-0 world image; every patch asks for a deeper tile.
    let mut input = viewpoint(4_000.0);
    input.patch_scale = 0.002;
    let span = strata_earth::tile_patch_span_deg(&input.camera, input.patch_scale);
    // The requested-tile parent chain ends at this shared ancestor.
    let world = strata_earth::requested_tile(
        &strata_earth::surrounding_patches(input.camera.focus, span)[0],
    )
    .parent()
    .expect("an ancestor above the requested tiles");
    assert!(
        world.level > 0,
        "the test needs an ancestor above the world"
    );
    let mut source = source_for(&[world], &solid_tile(200, 40, 40));

    let mut scene = EarthScene::new();
    scene.build(&input, &mut source);

    assert!(
        scene.stats().tiles_from_ancestor > 0,
        "unloaded patches must resolve to an ancestor, stats were {:?}",
        scene.stats()
    );
    assert!(
        scene.has_textured_geometry(),
        "ancestor imagery must reach the scene"
    );
    for (_, resolution) in scene.tile_resolutions() {
        assert_eq!(resolution.source, world);
        let (u0, v0, u1, v1) = resolution.uv_rect;
        assert!(u1 > u0 && v1 > v0, "the sub-rect must be non-empty");
        assert!(u0 >= 0.0 && u1 <= 1.0 && v0 >= 0.0 && v1 <= 1.0);
    }
}

#[test]
fn no_imagery_leaves_the_scene_textureless_but_renders() {
    let input = viewpoint(120_000.0);
    let mut source = MemoryTileSource::new();
    let mut scene = EarthScene::new();
    scene.build(&input, &mut source);
    assert!(!scene.has_textured_geometry());
    assert_eq!(scene.stats().tiles_from_ancestor, 0);
    let fb = render_scene(&input, &scene);
    assert!(
        fb.get_pixel(W / 2, H / 2).a() > 0.5,
        "the sky must still draw"
    );
}

#[test]
fn terrain_regime_uses_terrain_grid_and_relief() {
    let mut input = viewpoint(2_500.0);
    input.tile_grid = 9;
    let span = strata_earth::tile_patch_span_deg(&input.camera, input.patch_scale);
    let keys: Vec<TileKey> = strata_earth::surrounding_patches(input.camera.focus, span)
        .iter()
        .map(strata_earth::requested_tile)
        .collect();
    let mut source = source_for(&keys, &solid_tile(60, 120, 70));
    // Give every tile a relief so the height path runs.
    source.insert_elevation(
        keys[0],
        strata_earth::TileElevation {
            width: 2,
            height: 2,
            heights_m: vec![420.0, 410.0, 400.0, 395.0],
        },
    );
    let mut scene = EarthScene::new();
    scene.build(&input, &mut source);

    let expected = strata_earth::scene::terrain_triangle_count(keys.len(), 9);
    assert_eq!(scene.stats().terrain_triangles, expected);
    // Four footprints per patch, each four walls plus a roof cap.
    assert_eq!(
        scene.stats().building_triangles,
        strata_earth::scene::building_triangle_count(keys.len() * 4)
    );
}

#[test]
fn night_regime_darkens_the_sky_ramp() {
    let day = strata_earth::scene::sky_gradient(1.0);
    let night = strata_earth::scene::sky_gradient(0.0);
    let Background3D::VerticalGradient {
        top: day_top,
        bottom: day_bottom,
    } = day
    else {
        panic!("expected a gradient");
    };
    let Background3D::VerticalGradient {
        top: night_top,
        bottom: night_bottom,
    } = night
    else {
        panic!("expected a gradient");
    };
    assert!(day_top.r() > night_top.r());
    assert!(day_bottom.b() > night_bottom.b());
}

#[test]
fn scene_build_is_deterministic() {
    let input = viewpoint(50_000.0);
    let keys = surrounding_tiles(input.camera.focus, 8);
    let make = || {
        let mut source = source_for(&keys, &solid_tile(10, 20, 30));
        let mut scene = EarthScene::new();
        scene.build(&input, &mut source);
        (scene.scene().clone(), scene.stats())
    };
    let (a, a_stats) = make();
    let (b, b_stats) = make();
    assert_eq!(a, b, "the same input must build the same scene");
    assert_eq!(a_stats, b_stats);
}

#[test]
fn graticule_and_markers_reach_the_overlay() {
    let mut input = viewpoint(9_000_000.0);
    input.markers.push(EntityMarker {
        position: Geodetic::new(37.7749, -122.4194, 0.0),
        label: "SF".to_string(),
        color: Color::rgba(56, 189, 248, 255),
    });
    let mut source = MemoryTileSource::new();
    let mut scene = EarthScene::new();
    scene.build(&input, &mut source);

    let overlay = scene.scene().overlay().expect("overlay attached");
    assert!(
        overlay
            .primitives
            .iter()
            .any(|p| matches!(p, OverlayPrimitive::Marker { .. })),
        "the marker must be present"
    );
    assert!(
        overlay
            .primitives
            .iter()
            .any(|p| matches!(p, OverlayPrimitive::Polyline { .. })),
        "the graticule must be present"
    );
    assert_eq!(scene.stats().overlay_primitives, overlay.primitives.len());

    // Project with the frame the renderer will use for this scene, so the
    // overlay and the geometry share a camera.
    let frame = scene.renderer_overlay_frame();
    let projected = overlay.project(&frame);
    let markers = projected
        .iter()
        .filter(|c| matches!(c, strata_render::overlay3d::OverlayCommand::Marker { .. }))
        .count();
    assert_eq!(
        markers, 1,
        "exactly one marker should project onto the visible globe"
    );
}

#[test]
fn layer_toggles_remove_their_content() {
    let mut input = viewpoint(9_000_000.0);
    input.layers.graticule = false;
    input.layers.labels = false;
    let mut source = MemoryTileSource::new();
    let mut scene = EarthScene::new();
    scene.build(&input, &mut source);
    assert!(
        scene.scene().overlay().is_none(),
        "no overlay when both toggles are off"
    );

    input.layers.terrain = false;
    input.layers.buildings = false;
    scene.build(&input, &mut source);
    assert_eq!(scene.stats().terrain_triangles, 0);
    assert_eq!(scene.stats().building_triangles, 0);
}

#[test]
fn tile_registry_reuses_handles_and_evicts() {
    let key = TileKey::new(4, 3, 2);
    let mut registry = strata_earth::TileRegistry::new();
    let slot = strata_earth::TextureSlot { key, revision: 0 };
    let image = solid_tile(1, 2, 3);
    let first = registry.ensure(slot, &image).expect("upload");
    let second = registry.ensure(slot, &image).expect("reuse");
    assert_eq!(first, second, "the same slot must reuse its handle");
    assert_eq!(registry.len(), 1);

    let evicted = registry.retain(&std::collections::HashSet::new());
    assert_eq!(evicted, 1);
    assert!(registry.is_empty());
    assert!(registry.store().get(first).is_none());

    let sampler = tile_sampler();
    assert_eq!(
        sampler.filter,
        strata_render::texture3d::FilterMode::Bilinear
    );
}

fn populated_input() -> EarthInput {
    let mut input = viewpoint(3_000.0);
    input.tile_grid = 9;
    input.markers = (0..8)
        .map(|i| EntityMarker {
            position: Geodetic::new(37.0 + i as f64 * 0.2, -122.0 + i as f64 * 0.2, 0.0),
            label: format!("node-{i}"),
            color: Color::rgba(56, 189, 248, 255),
        })
        .collect();
    input
}

fn measure_frame_budget(input: &EarthInput) -> (f64, strata_earth::FrameStats) {
    let keys = surrounding_tiles(input.camera.focus, 18);
    let mut source = source_for(&keys, &solid_tile(40, 90, 60));
    let mut scene = EarthScene::new();

    // Warm-up: buffers, textures, and the rasterizer caches.
    scene.build(input, &mut source);
    let mut fb = FrameBuffer::new(W, H);
    render_scene_3d_with_textures(
        &mut fb,
        input.viewport,
        scene.scene(),
        None,
        Some(scene.tile_registry().store()),
    );

    let mut samples = Vec::with_capacity(30);
    for _ in 0..30 {
        let start = Instant::now();
        scene.build(input, &mut source);
        render_scene_3d_with_textures(
            &mut fb,
            input.viewport,
            scene.scene(),
            None,
            Some(scene.tile_registry().store()),
        );
        samples.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = samples[samples.len() / 2];
    let stats = scene.stats();
    println!(
        "earth frame budget: median {median:.2} ms over {} frames (terrain {} tris, buildings {} tris, overlay {} prims, {} tiles, {} draws)",
        samples.len(),
        stats.terrain_triangles,
        stats.building_triangles,
        stats.overlay_primitives,
        stats.tiles_requested,
        stats.draw_calls
    );
    (median, stats)
}

#[test]
fn frame_budget_at_1440x900_has_headroom() {
    let (median, stats) = measure_frame_budget(&populated_input());
    assert!(
        stats.terrain_triangles > 500 && stats.building_triangles > 50,
        "the budget must cover a populated scene"
    );
    // The measured median sits around 9 ms. The gate allows 25 ms so a noisy
    // CI neighbour cannot fail an unrelated change; the strict target is kept
    // as an ignored benchmark below.
    let max_allowed = if cfg!(debug_assertions) { 100.0 } else { 25.0 };
    assert!(
        median <= max_allowed,
        "median frame time must stay within the {max_allowed} ms headroom gate, was {median:.2} ms"
    );
}

/// Strict budget target, run on a quiet machine.
///
/// `cargo test -p strata_earth --test scene_tests -- --ignored strict_frame_budget`
#[test]
#[ignore]
fn strict_frame_budget_at_1440x900_is_within_16_6_ms() {
    let (median, _) = measure_frame_budget(&populated_input());
    assert!(
        median <= 16.6,
        "median frame time must stay within the 16.6 ms target, was {median:.2} ms"
    );
}

#[test]
fn plugin_has_no_widget_or_shell_dependency() {
    // Structural check of the dependency surface. The plugin may only depend on
    // engine crates; no egui, widget, or process-spawning crate is allowed.
    let manifest = include_str!("../Cargo.toml");
    for banned in [
        "egui",
        "strata-egui",
        "strata_egui",
        "git2",
        "reqwest",
    ] {
        assert!(
            !manifest.to_lowercase().contains(&banned.to_lowercase()),
            "the plugin manifest must not depend on {banned}"
        );
    }
    // And no source file may shell out.
    for entry in std::fs::read_dir(concat!(env!("CARGO_MANIFEST_DIR"), "/src")).expect("src dir") {
        let path = entry.expect("dir entry").path();
        if path.extension().and_then(|e| e.to_str()) != Some("rs") {
            continue;
        }
        let source = std::fs::read_to_string(&path).expect("source");
        for banned in ["std::process::Command", "Command::new", "egui::"] {
            assert!(
                !source.contains(banned),
                "{} must not reference {banned}",
                path.display()
            );
        }
    }
}

#[test]
fn textured_meshes_are_the_only_geometry_the_plugin_emits() {
    let input = viewpoint(9_000_000.0);
    let keys = surrounding_tiles(input.camera.focus, 4);
    let mut source = source_for(&keys, &solid_tile(0, 0, 0));
    let mut scene = EarthScene::new();
    scene.build(&input, &mut source);
    for mesh in &scene.scene().meshes {
        assert!(
            matches!(mesh.geometry, Geometry3D::CustomTextured { .. }),
            "tile geometry must carry UVs"
        );
        assert!(
            matches!(
                mesh.material,
                Material3D::Textured(_)
                    | Material3D::TexturedBatch(_)
                    | Material3D::Standard { .. }
            ),
            "plugin geometry must be textured, batched, or a building prism"
        );
    }
    assert!(
        scene
            .scene()
            .meshes
            .iter()
            .any(|m| matches!(m.material, Material3D::TexturedBatch(_))),
        "terrain must be one batched mesh"
    );
}

#[test]
fn relief_sampling_raises_the_terrain_surface() {
    use strata_render::scene3d::Geometry3D;

    let mut input = viewpoint(4_000.0);
    input.tile_grid = 5;
    let span = strata_earth::tile_patch_span_deg(&input.camera, input.patch_scale);
    let keys: Vec<TileKey> = strata_earth::surrounding_patches(input.camera.focus, span)
        .iter()
        .map(strata_earth::requested_tile)
        .collect();

    let mut source = MemoryTileSource::new();
    for key in &keys {
        source.insert_imagery(*key, solid_tile(80, 130, 90));
        source.insert_elevation(
            *key,
            strata_earth::TileElevation {
                width: 2,
                height: 2,
                heights_m: vec![0.0, 0.0, 800.0, 800.0],
            },
        );
    }

    let mut scene = EarthScene::new();
    scene.build(&input, &mut source);

    let origin = strata_earth::geodetic_to_ecef(input.camera.focus);
    let mesh = scene
        .scene()
        .meshes
        .iter()
        .find(|mesh| matches!(mesh.material, Material3D::TexturedBatch(_)))
        .expect("terrain mesh");
    let Geometry3D::CustomTextured { vertices, .. } = &mesh.geometry else {
        panic!("terrain must be custom textured geometry");
    };
    let radii: Vec<f64> = vertices
        .as_chunks::<8>().0.iter()
        .map(|v| {
            let absolute = origin.add(strata_render::camera64::DVec3::new(
                v[0] as f64,
                v[1] as f64,
                v[2] as f64,
            ));
            absolute.length()
        })
        .collect();
    let min = radii.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = radii.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    assert!(
        max - min > 500.0,
        "relief must move the surface by hundreds of metres, got {:.1} m",
        max - min
    );

    // At least one normal must be tilted away from the radial direction.
    let tilted = vertices.as_chunks::<8>().0.iter().any(|v| {
        let normal = strata_render::camera64::DVec3::new(v[3] as f64, v[4] as f64, v[5] as f64);
        let absolute = origin.add(strata_render::camera64::DVec3::new(
            v[0] as f64,
            v[1] as f64,
            v[2] as f64,
        ));
        normal.normalize().dot(absolute.normalize()).abs() < 0.999
    });
    assert!(tilted, "relief normals must follow the slope");
}

#[test]
fn elevation_gradient_matches_the_reference_sampling() {
    let elevation = strata_earth::TileElevation {
        width: 3,
        height: 3,
        heights_m: vec![0.0, 100.0, 200.0, 0.0, 100.0, 200.0, 0.0, 100.0, 200.0],
    };
    let (height, du, dv) = elevation.sample_bilinear_with_gradient(0.5, 0.5);
    assert!((height - 100.0).abs() < 1e-3, "centre height was {height}");
    assert!(du > 0.0, "east gradient must be positive, got {du}");
    assert!(dv.abs() < 1e-3, "a north-south ridge has no dv, got {dv}");
    assert_eq!(elevation.sample_bilinear(0.0, 0.0), 0.0);
    assert_eq!(elevation.sample_bilinear(1.0, 1.0), 200.0);
}

#[test]
fn terrain_is_one_batched_mesh_and_buildings_are_one_mesh() {
    let input = viewpoint(4_000.0);
    let span = strata_earth::tile_patch_span_deg(&input.camera, input.patch_scale);
    let patches = strata_earth::surrounding_patches(input.camera.focus, span);
    let mut source = MemoryTileSource::new();
    let mut scene = EarthScene::new();
    scene.build(&input, &mut source);

    let batches: Vec<&strata_render::scene3d::TexturedBatchMaterial3D> = scene
        .scene()
        .meshes
        .iter()
        .filter_map(|mesh| match &mesh.material {
            Material3D::TexturedBatch(batch) => Some(batch),
            _ => None,
        })
        .collect();
    assert_eq!(batches.len(), 1, "terrain must be a single batch");
    assert_eq!(
        batches[0].draws.len(),
        patches.len(),
        "each patch must be one draw range"
    );
    assert_eq!(scene.stats().draw_calls, patches.len() + 1);
}

#[test]
fn local_depth_policy_lets_a_twenty_kilometre_quad_paint() {
    use strata_render::camera64::DVec3;
    use strata_render::scene3d::{
        Camera3D, Geometry3D, Material3D, Mesh3D, Scene3D, TexturedMaterial3D,
    };
    use strata_render::texture3d::TextureStore;

    // A terrain camera four kilometres up. The old policy used the globe far
    // plane (about 51 000 km), which left an f32 depth band narrower than a
    // quarter pixel at this range, so a large patch lost the depth test.
    let camera = strata_earth::EarthCamera {
        focus: Geodetic::new(37.6, -122.0, 0.0),
        eye_offset_m: [0.0, 0.0, 4_000.0],
        ..Default::default()
    };
    let placement = camera.placement();
    assert!(
        placement.far_m < 1_000_000.0,
        "a terrain camera must not keep a planet-scale far plane, got {}",
        placement.far_m
    );

    let origin = strata_earth::geodetic_to_ecef(camera.focus);
    let eye = placement.eye.sub(origin);
    let forward = placement.aim.sub(placement.eye).normalize();
    let right = forward.cross(placement.up).normalize();
    let up = right.cross(forward).normalize();

    let span = 10_000.0;
    let mut vertices = Vec::new();
    for (sx, sy, u, v) in [
        (-1.0, -1.0, 0.0, 1.0),
        (1.0, -1.0, 1.0, 1.0),
        (1.0, 1.0, 1.0, 0.0),
        (-1.0, 1.0, 0.0, 0.0),
    ] {
        let world = placement
            .aim
            .add(right.scale(sx * span))
            .add(up.scale(sy * span));
        let local = world.sub(origin);
        vertices.extend_from_slice(&[
            local.x as f32,
            local.y as f32,
            local.z as f32,
            placement.up.x as f32,
            placement.up.y as f32,
            placement.up.z as f32,
            u as f32,
            v as f32,
        ]);
    }

    let mut store = TextureStore::new();
    let texture = store
        .insert_rgba8(2, 2, [180u8, 200, 120, 255].repeat(4))
        .expect("texture");
    let mut scene = Scene3D::new();
    scene.camera = Camera3D::Perspective {
        fov_horizontal_deg: strata_earth::horizontal_fov_deg(placement.fov_y_deg, 1.6) as f32,
        near: placement.near_m as f32,
        far: placement.far_m as f32,
        position: eye.as_f32(),
        target: [0.0, 0.0, 0.0],
    };
    scene.add_mesh(Mesh3D::new(
        Geometry3D::CustomTextured {
            vertices,
            indices: vec![0, 1, 2, 0, 2, 3],
        },
        Material3D::Textured(TexturedMaterial3D::new(texture)),
    ));

    let mut fb = FrameBuffer::new(W, H);
    render_scene_3d_with_textures(
        &mut fb,
        strata_layout::Rect::new(0.0, 0.0, W as f32, H as f32),
        &scene,
        None,
        Some(&store),
    );
    let painted = fb
        .pixels
        .as_chunks::<4>().0.iter()
        .filter(|p| p[3] > 0 && p[0] > 100 && p[1] > 150 && p[2] < 200)
        .count();
    assert!(
        painted > 1_000_000,
        "a 20 km quad four kilometres away must fill the frame, painted {painted} px"
    );
    let _ = DVec3::ZERO;
}

#[test]
fn celestial_bodies_are_declared_depth_test_free() {
    let mut input = viewpoint(12_000_000.0);
    input.layers.clouds = false;
    let keys = surrounding_tiles(input.camera.focus, 18);
    let mut source = source_for(&keys, &solid_tile(40, 90, 60));
    let mut scene = EarthScene::new();
    scene.build(&input, &mut source);

    let bodies = scene.scene().celestial();
    assert!(!bodies.is_empty(), "the globe regime must declare the sun");
    for body in bodies {
        assert!(
            !body.depth_test,
            "celestial bodies lie beyond the terrain far plane and must not depth-test"
        );
    }
}

#[test]
fn cloud_shell_is_a_declaration_not_drawn_geometry() {
    let mut input = viewpoint(4_000.0);
    input.layers.clouds = true;
    let span = strata_earth::tile_patch_span_deg(&input.camera, input.patch_scale);
    let keys: Vec<TileKey> = strata_earth::surrounding_patches(input.camera.focus, span)
        .iter()
        .map(strata_earth::requested_tile)
        .collect();
    let mut source = source_for(&keys, &solid_tile(40, 90, 60));
    let mut scene = EarthScene::new();
    scene.build(&input, &mut source);

    // The software rasterizer does not draw cloud shells; they are declared for
    // a host pass. At terrain altitude the shell radius is far beyond the local
    // far plane, so drawing it here would either clip or cover the frame.
    let shells = scene.scene().cloud_shells();
    assert_eq!(shells.len(), 1, "the cloud layer must still be declared");
    assert!(
        shells[0].radius > 6_000_000.0,
        "the declaration carries the Earth-scale shell radius for the host"
    );
    assert!(
        scene.has_textured_geometry(),
        "clouds must not replace the terrain pass at terrain altitude"
    );
}

#[test]
fn graticule_only_emits_the_eye_facing_hemisphere() {
    let mut input = viewpoint(12_000_000.0);
    input.graticule_step_deg = 30.0;
    let span = strata_earth::tile_patch_span_deg(&input.camera, input.patch_scale);
    let keys: Vec<TileKey> = strata_earth::surrounding_patches(input.camera.focus, span)
        .iter()
        .map(strata_earth::requested_tile)
        .collect();
    let mut source = source_for(&keys, &solid_tile(40, 90, 60));
    let mut scene = EarthScene::new();
    scene.build(&input, &mut source);

    let origin = strata_earth::geodetic_to_ecef(input.camera.focus);
    let eye = input.camera.placement().eye.sub(origin);
    let overlay = scene.scene().overlay().expect("graticule overlay");
    let mut lines = 0;
    let mut points = 0;
    for primitive in &overlay.primitives {
        if let OverlayPrimitive::Polyline {
            points: line_points,
            ..
        } = primitive
        {
            lines += 1;
            for point in line_points {
                let absolute = origin.add(*point);
                let visibility = absolute.dot(eye.sub(*point));
                assert!(
                    visibility >= -1.0,
                    "graticule point {point:?} is on the far hemisphere ({visibility})"
                );
                points += 1;
            }
        }
    }
    assert!(lines > 0, "the globe regime must draw a graticule");
    assert!(
        points > 100,
        "the visible graticule must have sampled points"
    );
}

#[test]
fn vertical_field_of_view_projects_to_the_top_edge() {
    // The reference derives one focal length from height, so its vertical FOV
    // is 2*atan(0.5/0.95) = 55.52 degrees at every aspect.
    let reference_fov_y = 2.0 * (0.5f64 / 0.95).atan().to_degrees();
    assert!((reference_fov_y - 55.52).abs() < 0.01);

    let mut input = viewpoint(4_000.0);
    input.camera.fov_y_deg = reference_fov_y;
    // Offset the eye horizontally so the view direction is not parallel to the
    // up hint. The aim stays at the focus, which keeps the RTC target at zero.
    input.camera.eye_offset_m = [0.0, -2_000.0, 4_000.0];
    let mut source = MemoryTileSource::new();
    let mut scene = EarthScene::new();
    scene.build(&input, &mut source);

    let origin = strata_earth::geodetic_to_ecef(input.camera.focus);
    let placement = input.camera.placement();
    let eye = placement.eye.sub(origin);
    let target = placement.aim.sub(origin);
    let forward = target.sub(eye).normalize();
    let right = forward.cross(placement.up).normalize();
    let up = right.cross(forward).normalize();

    let frame = scene.renderer_overlay_frame();
    let half = reference_fov_y.to_radians() * 0.5;
    let direction = forward.scale(half.cos()).add(up.scale(half.sin()));
    // Place the sample beyond the aim point, so it is in front of the camera.
    let point = eye.add(direction.scale(10_000.0));
    let projected = frame.project(point);

    assert!(
        (projected.x - W as f64 * 0.5).abs() < 1.0,
        "a point straight above the axis must stay horizontally centred, got x={}",
        projected.x
    );
    assert!(
        projected.y.abs() < 1.0,
        "a point at half the vertical FOV must land on the top edge, got y={}",
        projected.y
    );
}

#[test]
fn spherical_globe_mode_builds_closed_sphere_and_surface_descriptor() {
    let mut input = viewpoint(12_000_000.0);
    input.surface_mode = strata_earth::SurfaceMode::SphericalGlobe;
    input.tile_grid = 4;
    let mut source = MemoryTileSource::new();
    let mut scene = EarthScene::new();
    scene.build(&input, &mut source);

    assert!(!scene.patches().is_empty(), "must build sphere tile patches");
    assert!(scene.stats().terrain_triangles > 0, "must emit sphere triangles");

    let descriptor = FrameDescriptor::capture(&scene, &input);
    assert!(descriptor.surface.is_some(), "must declare a surface sphere in descriptor");
    let sphere = descriptor.surface.as_ref().unwrap();
    assert!((sphere.radius_m - 6_378_137.0).abs() < 1e-3);
    assert_eq!(descriptor.patches.len(), 0, "sphere frame omits planar patches");
    assert!(!descriptor.rings.is_empty(), "must declare tile rings");
    assert_eq!(descriptor.rings[0].tiles.len(), descriptor.tiles.len());

    let json_str = descriptor.to_json_string();
    assert!(json_str.contains("\"surface\""));
    assert!(json_str.contains("\"rings\""));
    assert!(!json_str.contains("\"patches\""));

    let restored = descriptor.earth_input();
    assert_eq!(restored.surface_mode, strata_earth::SurfaceMode::SphericalGlobe);
}

#[test]
fn directory_tile_source_round_trips_images_and_elevation() {
    let dir = std::env::temp_dir().join(format!("strata_tile_test_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("2/1")).expect("create test tile dir");

    let key = TileKey::new(2, 1, 1);
    let mut test_image_bytes = vec![0u8; 8 + 64 * 64 * 4];
    test_image_bytes[0..4].copy_from_slice(&64u32.to_le_bytes());
    test_image_bytes[4..8].copy_from_slice(&64u32.to_le_bytes());
    for (i, byte) in test_image_bytes[8..].iter_mut().enumerate() {
        *byte = (i % 255) as u8;
    }
    let image_path = dir.join("2/1/1.raw");
    std::fs::write(&image_path, &test_image_bytes).expect("write test image tile");

    let mut test_elev_bytes = vec![0u8; 8 + 16 * 16 * 4];
    test_elev_bytes[0..4].copy_from_slice(&16u32.to_le_bytes());
    test_elev_bytes[4..8].copy_from_slice(&16u32.to_le_bytes());
    for i in 0..(16 * 16) {
        let val = (i as f32) * 10.5;
        let offset = 8 + i * 4;
        test_elev_bytes[offset..offset + 4].copy_from_slice(&val.to_le_bytes());
    }
    let elev_path = dir.join("2/1/1.elev");
    std::fs::write(&elev_path, &test_elev_bytes).expect("write test elevation tile");

    let mut source = strata_earth::DirectoryTileSource::new(&dir);
    assert_eq!(source.base_dir(), dir.as_path());

    let loaded_img = source.imagery(key).expect("read ok").expect("tile exists");
    assert_eq!(loaded_img.width, 64);
    assert_eq!(loaded_img.height, 64);
    assert_eq!(&loaded_img.pixels[..], &test_image_bytes[8..]);

    let loaded_elev = source.elevation(key).expect("read ok").expect("elev exists");
    assert_eq!(loaded_elev.width, 16);
    assert_eq!(loaded_elev.height, 16);
    assert_eq!(loaded_elev.sample_bilinear(0.0, 0.0), 0.0);

    let missing_key = TileKey::new(5, 10, 10);
    assert!(source.imagery(missing_key).expect("read ok").is_none());
    assert!(source.elevation(missing_key).expect("read ok").is_none());

    let _ = std::fs::remove_dir_all(&dir);
}
