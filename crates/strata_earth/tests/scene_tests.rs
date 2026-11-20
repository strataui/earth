//! Frame assembly: block accounting, the four camera regimes, the frame budget,
//! and the dependency-surface contract.
//!
//! The golden images are produced by `examples/strata_earth_golden`; this suite
//! checks the numbers those goldens depend on (tile resolution, block counts,
//! deterministic scene output) and the frame budget at 1440x900.

use std::time::Instant;
use strata_earth::{
    surrounding_tiles, tile_sampler, EarthGeometry, EarthInput, EarthMaterial, EarthObject,
    EarthScene, FrameDescriptor, GeoPosition, Geodetic, MemoryTileSource, TileImage, TileKey,
    TileSource,
};
use strata_render::compositor::DrawCommand;
use strata_render::overlay3d::OverlayPrimitive;
use strata_render::rasterizer::FrameBuffer;
use strata_render::rasterizer3d::render_scene_3d_with_textures;
use strata_render::scene3d::{Background3D, Geometry3D, Material3D, Scene3D};
use strata_style::types::Color;

const W: usize = 1440;
const H: usize = 900;

fn built_scene3d(scene: &EarthScene, viewport: strata_layout::Rect) -> std::sync::Arc<Scene3D> {
    match scene.to_draw_command(viewport) {
        DrawCommand::DrawViewport3D { scene: s, .. } => s,
        _ => panic!("expected DrawViewport3D"),
    }
}

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

/// Four footprints per patch in a 2x2 grid, matching the old placeholder layout.
fn grid_buildings(patches: &[strata_earth::PatchFrame]) -> strata_earth::MemoryBuildingSource {
    let mut source = strata_earth::MemoryBuildingSource::new();
    for patch in patches {
        let b = patch.bounds;
        for iy in 0..2 {
            for ix in 0..2 {
                let lat = b.lat0 + (b.lat1 - b.lat0) * (0.25 + 0.5 * iy as f64);
                let lon = b.lon0 + (b.lon1 - b.lon0) * (0.25 + 0.5 * ix as f64);
                source.push(strata_earth::BuildingFootprint::new(
                    lat,
                    lon,
                    (40.0, 25.0),
                    60.0,
                ));
            }
        }
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
    let s3d = built_scene3d(scene, input.viewport);
    render_scene_3d_with_textures(
        &mut fb,
        input.viewport,
        &s3d,
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
    input.layers.buildings = true;
    scene.set_building_source(Some(Box::new(grid_buildings(
        &strata_earth::surrounding_patches(input.camera.focus, span),
    ))));
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
        ((*built_scene3d(&scene, input.viewport)).clone(), scene.stats())
    };
    let (a, a_stats) = make();
    let (b, b_stats) = make();
    assert_eq!(a, b, "the same input must build the same scene");
    assert_eq!(a_stats, b_stats);
}

#[test]
fn graticule_and_world_labels_reach_the_overlay() {
    let input = viewpoint(9_000_000.0);
    let mut source = MemoryTileSource::new();
    let mut scene = EarthScene::new();
    scene.spawn_object(
        EarthObject::new(
            GeoPosition::terrain(37.7749, -122.4194, 0.0),
            EarthGeometry::beacon_pin(20.0, 5.0),
        )
        .with_label("SF")
        .with_material(EarthMaterial::standard(Color::rgba(56, 189, 248, 255))),
    );
    scene.build(&input, &mut source);

    let s3d = built_scene3d(&scene, input.viewport);
    let overlay = s3d.overlay().expect("overlay attached");
    assert!(
        overlay
            .primitives
            .iter()
            .any(|p| matches!(p, OverlayPrimitive::Label { .. })),
        "the object label must be present"
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
    let labels = projected
        .iter()
        .filter(|c| matches!(c, strata_render::overlay3d::OverlayCommand::Label { .. }))
        .count();
    assert_eq!(
        labels, 1,
        "exactly one object label should project onto the visible globe"
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
    let s3d = built_scene3d(&scene, input.viewport);
    assert!(
        s3d.overlay().is_none(),
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
    input
}

fn measure_frame_budget(input: &EarthInput) -> (f64, strata_earth::FrameStats) {
    let mut input = input.clone();
    input.layers.buildings = true;
    let input = &input;
    let keys = surrounding_tiles(input.camera.focus, 18);
    let mut source = source_for(&keys, &solid_tile(40, 90, 60));
    let mut scene = EarthScene::new();
    for i in 0..8 {
        scene.spawn_object(
            EarthObject::new(
                GeoPosition::terrain(37.0 + i as f64 * 0.2, -122.0 + i as f64 * 0.2, 5.0),
                EarthGeometry::beacon_pin(20.0, 5.0),
            )
            .with_label(format!("node-{i}"))
            .with_material(EarthMaterial::standard(Color::rgba(56, 189, 248, 255))),
        );
    }
    let span = strata_earth::tile_patch_span_deg(&input.camera, input.patch_scale);
    let patches = strata_earth::surrounding_patches(input.camera.focus, span);
    scene.set_building_source(Some(Box::new(grid_buildings(&patches))));

    // Warm-up: buffers, textures, and the rasterizer caches.
    scene.build(input, &mut source);
    let mut fb = FrameBuffer::new(W, H);
    {
        let s3d = built_scene3d(&scene, input.viewport);
        render_scene_3d_with_textures(
            &mut fb,
            input.viewport,
            &s3d,
            None,
            Some(scene.tile_registry().store()),
        );
    }

    let mut samples = Vec::with_capacity(30);
    for _ in 0..30 {
        let start = Instant::now();
        scene.build(input, &mut source);
        let s3d = built_scene3d(&scene, input.viewport);
        render_scene_3d_with_textures(
            &mut fb,
            input.viewport,
            &s3d,
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
    for banned in ["egui", "strata-egui", "strata_egui", "git2", "reqwest"] {
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
    let s3d = built_scene3d(&scene, input.viewport);
    for mesh in &s3d.meshes {
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
        s3d.meshes
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
    let s3d = built_scene3d(&scene, input.viewport);
    let mesh = s3d
        .meshes
        .iter()
        .find(|mesh| matches!(mesh.material, Material3D::TexturedBatch(_)))
        .expect("terrain mesh");
    let Geometry3D::CustomTextured { vertices, .. } = &mesh.geometry else {
        panic!("terrain must be custom textured geometry");
    };
    let radii: Vec<f64> = vertices
        .as_chunks::<8>()
        .0
        .iter()
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
    let input = {
        let mut input = viewpoint(4_000.0);
        input.layers.buildings = true;
        input
    };
    let span = strata_earth::tile_patch_span_deg(&input.camera, input.patch_scale);
    let patches = strata_earth::surrounding_patches(input.camera.focus, span);
    let mut source = MemoryTileSource::new();
    let mut scene = EarthScene::new();
    scene.set_building_source(Some(Box::new(grid_buildings(&patches))));
    scene.build(&input, &mut source);

    let s3d = built_scene3d(&scene, input.viewport);
    let batches: Vec<&strata_render::scene3d::TexturedBatchMaterial3D> = s3d
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
        .as_chunks::<4>()
        .0
        .iter()
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

    let s3d = built_scene3d(&scene, input.viewport);
    let bodies = s3d.celestial();
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
    let s3d = built_scene3d(&scene, input.viewport);
    let shells = s3d.cloud_shells();
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
    let s3d = built_scene3d(&scene, input.viewport);
    let overlay = s3d.overlay().expect("graticule overlay");
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

    assert!(
        !scene.patches().is_empty(),
        "must build sphere tile patches"
    );
    assert!(
        scene.stats().terrain_triangles > 0,
        "must emit sphere triangles"
    );

    let descriptor = FrameDescriptor::capture(&scene, &input);
    assert!(
        descriptor.surface.is_some(),
        "must declare a surface sphere in descriptor"
    );
    let sphere = descriptor.surface.as_ref().unwrap();
    assert!((sphere.radius_m - 6_378_137.0).abs() < 1e-3);
    assert_eq!(
        descriptor.patches.len(),
        0,
        "sphere frame omits planar patches"
    );
    assert!(!descriptor.rings.is_empty(), "must declare tile rings");
    assert_eq!(descriptor.rings[0].tiles.len(), descriptor.tiles.len());

    let json_str = descriptor.to_json_string();
    assert!(json_str.contains("\"surface\""));
    assert!(json_str.contains("\"rings\""));
    assert!(!json_str.contains("\"patches\""));

    let restored = descriptor.earth_input();
    assert_eq!(
        restored.surface_mode,
        strata_earth::SurfaceMode::SphericalGlobe
    );
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

    let loaded_elev = source
        .elevation(key)
        .expect("read ok")
        .expect("elev exists");
    assert_eq!(loaded_elev.width, 16);
    assert_eq!(loaded_elev.height, 16);
    assert_eq!(loaded_elev.sample_bilinear(0.0, 0.0), 0.0);

    let missing_key = TileKey::new(5, 10, 10);
    assert!(source.imagery(missing_key).expect("read ok").is_none());
    assert!(source.elevation(missing_key).expect("read ok").is_none());

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_earth_scene_build_quadtree_lod_mode() {
    let mut input = viewpoint(2_000_000.0);
    input.surface_mode = strata_earth::SurfaceMode::QuadtreeLod;
    input.sse_threshold = 4.0;
    input.max_tile_level = 16;

    let mut source = MemoryTileSource::new();
    let mut scene = EarthScene::new();

    scene.build(&input, &mut source);

    let stats = scene.stats();
    assert!(
        stats.tiles_requested > 0,
        "quadtree LOD mode must request tiles"
    );
    assert!(
        !scene.patches().is_empty(),
        "quadtree LOD mode must populate patches in the scene"
    );

    // Verify patches are loaded and rendered
    let levels: std::collections::BTreeSet<u8> =
        scene.patches().iter().map(|p| p.key.level).collect();
    assert!(
        !levels.is_empty(),
        "hierarchical quadtree LOD should have resolved tiles, got {:?}",
        levels
    );

    // Verify batched mesh has terrain triangles
    assert!(stats.terrain_triangles > 0);
}

#[test]
fn test_compute_adaptive_skirt_depth_scaling() {
    // 1. Sub-meter / street level tile: 0.001 deg span (~111 m)
    let street_bounds = strata_earth::PatchBounds {
        lon0: -122.420,
        lon1: -122.419,
        lat0: 37.774,
        lat1: 37.775,
    };
    let street_skirt = strata_earth::compute_adaptive_skirt_depth(street_bounds, 0.0);
    assert_eq!(
        street_skirt, 20.0,
        "street level tile must clamp to minimum 20m skirt depth"
    );

    // 2. Mountain relief tile: relief range = 3,000 m
    let mountain_skirt = strata_earth::compute_adaptive_skirt_depth(street_bounds, 3_000.0);
    assert!(
        mountain_skirt >= 1500.0,
        "mountain skirt must be at least half the elevation range, got {mountain_skirt}"
    );

    // 3. Continental tile: 20 deg span (~2,200 km)
    let continental_bounds = strata_earth::PatchBounds {
        lon0: -120.0,
        lon1: -100.0,
        lat0: 20.0,
        lat1: 40.0,
    };
    let continental_skirt = strata_earth::compute_adaptive_skirt_depth(continental_bounds, 500.0);
    assert!(
        continental_skirt > 50_000.0,
        "continental skirt must scale with 5% of tile span, got {continental_skirt}"
    );
}

#[test]
fn test_terrain_skirt_triangle_counting() {
    // For grid = 5: 4 cells per axis -> 4 * 4 * 2 = 32 grid triangles
    // Skirt: 4 borders * 4 segments * 2 triangles = 32 skirt triangles
    assert_eq!(strata_earth::terrain_skirt_triangle_count(5), 32);
    assert_eq!(strata_earth::terrain_triangle_count(1, 5), 32);
    assert_eq!(strata_earth::terrain_triangle_count_with_skirts(1, 5), 64);
    assert_eq!(
        strata_earth::terrain_triangle_count_with_skirts(9, 5),
        9 * 64
    );

    // For grid = 9: 8 cells per axis -> 8 * 8 * 2 = 128 grid triangles
    // Skirt: 4 borders * 8 segments * 2 triangles = 64 skirt triangles
    assert_eq!(strata_earth::terrain_skirt_triangle_count(9), 64);
    assert_eq!(strata_earth::terrain_triangle_count(1, 9), 128);
    assert_eq!(strata_earth::terrain_triangle_count_with_skirts(1, 9), 192);
}

#[test]
fn test_adaptive_terrain_skirts_in_scene_build() {
    let mut input_no_skirts = viewpoint(2_500.0);
    input_no_skirts.tile_grid = 9;
    input_no_skirts.terrain_skirts = Some(false);

    let span =
        strata_earth::tile_patch_span_deg(&input_no_skirts.camera, input_no_skirts.patch_scale);
    let keys: Vec<TileKey> = strata_earth::surrounding_patches(input_no_skirts.camera.focus, span)
        .iter()
        .map(strata_earth::requested_tile)
        .collect();
    let mut source = source_for(&keys, &solid_tile(60, 120, 70));

    let mut scene_no_skirts = EarthScene::new();
    scene_no_skirts.build(&input_no_skirts, &mut source);

    let expected_grid_only = strata_earth::terrain_triangle_count(keys.len(), 9);
    assert_eq!(
        scene_no_skirts.stats().terrain_triangles,
        expected_grid_only
    );

    // Now build with skirts enabled
    let mut input_with_skirts = input_no_skirts.clone();
    input_with_skirts.terrain_skirts = Some(true);

    let mut scene_with_skirts = EarthScene::new();
    scene_with_skirts.build(&input_with_skirts, &mut source);

    let expected_with_skirts = strata_earth::terrain_triangle_count_with_skirts(keys.len(), 9);
    assert_eq!(
        scene_with_skirts.stats().terrain_triangles,
        expected_with_skirts,
        "terrain triangles with skirts must include 64 skirt triangles per patch"
    );
    assert_eq!(
        scene_with_skirts.stats().terrain_triangles - scene_no_skirts.stats().terrain_triangles,
        keys.len() * strata_earth::terrain_skirt_triangle_count(9),
        "difference must equal the total skirt triangle count"
    );

    // Now test QuadtreeLod mode auto-enables skirts when terrain_skirts is None
    let mut input_lod = viewpoint(2_000_000.0);
    input_lod.surface_mode = strata_earth::SurfaceMode::QuadtreeLod;
    input_lod.sse_threshold = 4.0;
    input_lod.max_tile_level = 16;
    input_lod.terrain_skirts = None; // Should auto-enable for QuadtreeLod

    let mut scene_lod_auto = EarthScene::new();
    scene_lod_auto.build(&input_lod, &mut source);

    let mut input_lod_disabled = input_lod.clone();
    input_lod_disabled.terrain_skirts = Some(false); // Explicitly disable skirts

    let mut scene_lod_disabled = EarthScene::new();
    scene_lod_disabled.build(&input_lod_disabled, &mut source);

    assert!(
        scene_lod_auto.stats().terrain_triangles > scene_lod_disabled.stats().terrain_triangles,
        "Quadtree LOD with auto-skirts must have more triangles than with skirts disabled"
    );
}

#[test]
fn test_world_mesh_eviction_on_remove_clear_replace_and_set_polyline_points() {
    use strata_earth::{
        EarthGeometry, EarthMaterial, EarthObject, EarthObjectId, EarthWorld, GeoPosition,
        Geodetic, SurfaceMorphParams, TileProjection,
    };

    let mut world = EarthWorld::new();
    let rev0 = world.revision();
    assert!(
        world.get_mut(EarthObjectId(999_999)).is_none(),
        "missing object must return None"
    );
    assert_eq!(
        world.revision(),
        rev0,
        "get_mut on missing ID must not increment world revision"
    );

    let pts1 = vec![
        GeoPosition::sea_level(37.7749, -122.4194, 100.0),
        GeoPosition::sea_level(37.7849, -122.4094, 200.0),
    ];
    let poly_id = world.spawn(EarthObject::new(
        pts1[0],
        EarthGeometry::Polyline {
            points: pts1,
            radius_m: 15.0,
        },
    ));
    let box_id = world.spawn(
        EarthObject::new(
            GeoPosition::sea_level(37.7749, -122.4194, 50.0),
            EarthGeometry::cube(20.0),
        )
        .with_material(EarthMaterial::standard(Color::rgba(255, 0, 0, 255))),
    );

    let morph = SurfaceMorphParams {
        focus: Geodetic::new(37.7749, -122.4194, 0.0),
        flat_blend: 0.0,
        map_2d_morph: 0.0,
        projection: TileProjection::default(),
    };
    let eye = strata_render::camera64::DVec3::new(0.0, 0.0, 7_000_000.0);
    let _ = world.compile_retained(eye, 900.0, 55.0, false, morph, &|_, _| 0.0);
    assert_eq!(world.mesh_count(), 2, "two distinct meshes should be resident");

    // Mutate polyline points and radius in place without leaking old mesh
    let pts2 = vec![
        GeoPosition::sea_level(37.7749, -122.4194, 100.0),
        GeoPosition::sea_level(37.7900, -122.4000, 250.0),
        GeoPosition::sea_level(37.8000, -122.3900, 300.0),
    ];
    assert!(world.set_polyline_points(poly_id, pts2));
    assert!(world.set_polyline_radius(poly_id, 25.0));
    let _ = world.compile_retained(eye, 900.0, 55.0, false, morph, &|_, _| 0.0);
    assert_eq!(
        world.mesh_count(),
        2,
        "in-place polyline update must evict old polyline mesh"
    );

    // Replace geometry in place
    assert!(world.replace_geometry(poly_id, EarthGeometry::sphere(40.0)));
    assert_eq!(
        world.mesh_count(),
        2,
        "replace_geometry must evict unreferenced polyline mesh"
    );

    // Remove one object -> its mesh must be evicted
    assert!(world.remove(poly_id));
    assert_eq!(world.mesh_count(), 1, "remove must evict unreferenced mesh");

    // Clear all objects -> mesh table must be empty
    world.clear();
    assert_eq!(world.mesh_count(), 0, "clear must evict all meshes");
    let _ = box_id;
}

#[test]
fn test_polyline_25d_pan_zero_rebuild_and_2d_wrapped_altitude_parity() {
    use strata_earth::{
        EarthGeometry, EarthObject, EarthWorld, GeoPosition, Geodetic, SurfaceMorphParams,
        TileProjection,
    };

    let mut world = EarthWorld::new();
    let pts = vec![
        GeoPosition::sea_level(37.7749, -122.4194, 500.0),
        GeoPosition::sea_level(37.7849, -122.4094, 900.0),
    ];
    let _id = world.spawn(EarthObject::new(
        pts[0],
        EarthGeometry::Polyline {
            points: pts,
            radius_m: 12.0,
        },
    ));
    let eye = strata_render::camera64::DVec3::new(-2_700_000.0, -4_200_000.0, 3_800_000.0);

    let morph_a = SurfaceMorphParams {
        focus: Geodetic::new(37.7749, -122.4194, 0.0),
        flat_blend: 1.0,
        map_2d_morph: 0.0,
        projection: TileProjection::default(),
    };
    let (assets_a, insts_a) =
        world.compile_retained(eye, 900.0, 55.0, false, morph_a, &|_, _| 0.0);
    assert_eq!(assets_a.len(), 1);
    assert_eq!(insts_a.len(), 1);

    // Pan focus in 2.5D mode: mesh asset Arc pointer and revision MUST stay identical (zero rebuild!)
    let morph_b = SurfaceMorphParams {
        focus: Geodetic::new(37.8100, -122.3800, 0.0),
        flat_blend: 1.0,
        map_2d_morph: 0.0,
        projection: TileProjection::default(),
    };
    let (assets_b, insts_b) =
        world.compile_retained(eye, 900.0, 55.0, false, morph_b, &|_, _| 0.0);
    assert!(
        std::sync::Arc::ptr_eq(&assets_a[0], &assets_b[0]),
        "2.5D camera pan must not rebuild polyline tube mesh asset"
    );
    assert_ne!(
        insts_a[0].anchor_ecef, insts_b[0].anchor_ecef,
        "2.5D camera pan must update polyline instance anchor relative to new focus tangent plane"
    );

    // Canonical 2D mode with 360-degree wrapped camera focus
    let morph_2d = SurfaceMorphParams {
        focus: Geodetic::new(37.7749, -122.4194 + 360.0, 0.0),
        flat_blend: 1.0,
        map_2d_morph: 1.0,
        projection: TileProjection::WebMercator,
    };
    let (assets_2d, insts_2d) =
        world.compile_retained(eye, 900.0, 55.0, false, morph_2d, &|_, _| 0.0);
    assert_eq!(assets_2d.len(), 1);
    assert_eq!(insts_2d.len(), 1);
}

#[test]
fn test_glb_multi_mesh_node_hierarchy_yup_to_zup_vertex_colors_and_embedded_texture() {
    use strata_earth::{parse_glb_mesh, EarthWorld, GeoPosition, ObjectAnchor};

    // Build a minimal valid 1x1 RGBA PNG in memory using the `image` crate
    let mut png_bytes: Vec<u8> = Vec::new();
    {
        let img = image::RgbaImage::from_pixel(1, 1, image::Rgba([64, 192, 255, 255]));
        let mut cursor = std::io::Cursor::new(&mut png_bytes);
        img.write_to(&mut cursor, image::ImageFormat::Png)
            .expect("encode 1x1 PNG");
    }

    let mut bin: Vec<u8> = Vec::new();
    let positions: [[f32; 3]; 3] = [[0.0, 2.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]];
    for p in positions {
        for f in p {
            bin.extend_from_slice(&f.to_le_bytes());
        }
    }
    let uvs: [[f32; 2]; 3] = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
    for uv in uvs {
        for f in uv {
            bin.extend_from_slice(&f.to_le_bytes());
        }
    }
    for _ in 0..3 {
        for f in [1.0f32, 0.5, 0.25, 1.0] {
            bin.extend_from_slice(&f.to_le_bytes());
        }
    }
    for idx in [0u16, 1, 2] {
        bin.extend_from_slice(&idx.to_le_bytes());
    }
    while bin.len() % 4 != 0 {
        bin.push(0);
    }
    let png_offset = bin.len();
    let png_len = png_bytes.len();
    bin.extend_from_slice(&png_bytes);
    while bin.len() % 4 != 0 {
        bin.push(0);
    }

    let json = format!(
        r#"{{
            "asset": {{"version": "2.0"}},
            "scenes": [{{"nodes": [0, 1]}}],
            "nodes": [
                {{"mesh": 0}},
                {{"mesh": 1, "translation": [0.0, 3.0, 0.0]}}
            ],
            "meshes": [
                {{"primitives": [{{"attributes": {{"POSITION": 0, "TEXCOORD_0": 1, "COLOR_0": 2}}, "indices": 3, "material": 0}}]}},
                {{"primitives": [{{"attributes": {{"POSITION": 0, "TEXCOORD_0": 1}}, "indices": 3, "material": 0}}]}}
            ],
            "materials": [{{
                "pbrMetallicRoughness": {{
                    "baseColorFactor": [1.0, 1.0, 1.0, 1.0],
                    "baseColorTexture": {{"index": 0}},
                    "metallicFactor": 0.2,
                    "roughnessFactor": 0.4
                }}
            }}],
            "textures": [{{"source": 0}}],
            "images": [{{"bufferView": 4, "mimeType": "image/png"}}],
            "accessors": [
                {{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"}},
                {{"bufferView": 1, "componentType": 5126, "count": 3, "type": "VEC2"}},
                {{"bufferView": 2, "componentType": 5126, "count": 3, "type": "VEC4"}},
                {{"bufferView": 3, "componentType": 5123, "count": 3, "type": "SCALAR"}}
            ],
            "bufferViews": [
                {{"buffer": 0, "byteOffset": 0, "byteLength": 36}},
                {{"buffer": 0, "byteOffset": 36, "byteLength": 24}},
                {{"buffer": 0, "byteOffset": 60, "byteLength": 48}},
                {{"buffer": 0, "byteOffset": 108, "byteLength": 6}},
                {{"buffer": 0, "byteOffset": {png_offset}, "byteLength": {png_len}}}
            ],
            "buffers": [{{"byteLength": {total_bin}}}]
        }}"#,
        total_bin = bin.len()
    );
    let mut json_bytes = json.into_bytes();
    while json_bytes.len() % 4 != 0 {
        json_bytes.push(b' ');
    }

    let total_len = 12 + 8 + json_bytes.len() + 8 + bin.len();
    let mut glb = Vec::with_capacity(total_len);
    glb.extend_from_slice(&0x4654_6C67u32.to_le_bytes());
    glb.extend_from_slice(&2u32.to_le_bytes());
    glb.extend_from_slice(&(total_len as u32).to_le_bytes());
    glb.extend_from_slice(&(json_bytes.len() as u32).to_le_bytes());
    glb.extend_from_slice(&0x4E4F_534Au32.to_le_bytes());
    glb.extend_from_slice(&json_bytes);
    glb.extend_from_slice(&(bin.len() as u32).to_le_bytes());
    glb.extend_from_slice(&0x004E_4942u32.to_le_bytes());
    glb.extend_from_slice(&bin);

    let parsed = parse_glb_mesh(&glb).expect("parse multi-mesh textured GLB");
    assert_eq!(
        parsed.positions.len(),
        6,
        "both meshes/nodes in the scene must be parsed (2 x 3 vertices)"
    );
    assert_eq!(parsed.indices.len(), 6);
    // Vertex 0 was (0, 2, 0) in glTF Y-up -> (0, 0, 2) in ENU Z-up!
    assert!((parsed.positions[0][0] - 0.0).abs() < 1e-5);
    assert!((parsed.positions[0][1] - 0.0).abs() < 1e-5);
    assert!((parsed.positions[0][2] - 2.0).abs() < 1e-5);
    // Second node had Y-up translation (0, 3, 0) -> vertex 0 of node 1 is at Y=5 -> Z=5 in ENU!
    assert!((parsed.positions[3][2] - 5.0).abs() < 1e-5);
    assert!(parsed.vertex_colors.is_some(), "COLOR_0 must be decoded");
    let tex = parsed
        .albedo_texture
        .as_ref()
        .expect("embedded PNG baseColorTexture must be decoded");
    assert_eq!(tex.width, 1);
    assert_eq!(tex.height, 1);
    assert_eq!(&tex.rgba8[..], &[64, 192, 255, 255]);

    let mut world = EarthWorld::new();
    let obj_id = world
        .spawn_glb(
            GeoPosition::sea_level(37.7749, -122.4194, 100.0),
            &glb,
            1.0,
            ObjectAnchor::Center,
        )
        .expect("spawn_glb ok");
    assert!(world.get(obj_id).is_some());
    assert!(
        world
            .spawn_glb(
                GeoPosition::sea_level(0.0, 0.0, 0.0),
                b"not a glb",
                1.0,
                ObjectAnchor::Center
            )
            .is_err(),
        "corrupt GLB bytes must return Err from spawn_glb"
    );
}

#[test]
fn test_transparent_objects_sorted_back_to_front_and_albedo_texture_propagated() {
    use strata_earth::{
        EarthGeometry, EarthMaterial, EarthObject, EarthWorld, GeoPosition, Geodetic,
        SurfaceMorphParams, TileProjection,
    };

    let mut world = EarthWorld::new();
    // Spawn near transparent object first, then far transparent object, then opaque textured object
    let near_pos = GeoPosition::sea_level(0.0, 0.0, 500_000.0);
    let far_pos = GeoPosition::sea_level(0.0, 0.0, 10_000.0);
    let opaque_pos = GeoPosition::sea_level(0.0, 0.0, 100_000.0);

    let _near_id = world.spawn(
        EarthObject::new(near_pos, EarthGeometry::cube(100.0))
            .with_material(EarthMaterial::standard(Color::rgba(255, 0, 0, 128))),
    );
    let _far_id = world.spawn(
        EarthObject::new(far_pos, EarthGeometry::cube(100.0))
            .with_material(EarthMaterial::standard(Color::rgba(0, 255, 0, 128))),
    );
    let tex_mat = EarthMaterial::standard(Color::rgba(255, 255, 255, 255))
        .with_albedo_rgba8(1, 1, vec![10, 20, 30, 255]);
    let _opaque_id = world
        .spawn(EarthObject::new(opaque_pos, EarthGeometry::cube(100.0)).with_material(tex_mat));

    // Eye is along +X at 8,000,000 m (directly above lat=0, lon=0)
    let eye = strata_render::camera64::DVec3::new(8_000_000.0, 0.0, 0.0);
    let morph = SurfaceMorphParams {
        focus: Geodetic::new(0.0, 0.0, 0.0),
        flat_blend: 0.0,
        map_2d_morph: 0.0,
        projection: TileProjection::default(),
    };
    let (_assets, instances) =
        world.compile_retained(eye, 900.0, 55.0, false, morph, &|_, _| 0.0);
    assert_eq!(instances.len(), 3);
    // Opaque object must come first and carry its albedo texture
    assert!((instances[0].base_color[3] - 1.0).abs() < 1e-4);
    assert!(instances[0].albedo_texture.is_some());
    // Among transparent objects, far_pos (smaller X) must precede near_pos (larger X)
    assert!(
        instances[1].anchor_ecef[0] < instances[2].anchor_ecef[0],
        "transparent objects must be depth-sorted back-to-front relative to camera eye"
    );
}

#[test]
fn test_custom_mesh_out_of_bounds_indices_rejected_without_raycast_panic() {
    use strata_earth::{
        raycast_viewport, EarthGeometry, EarthObject, EarthVertex, EarthWorld, GeoPosition,
        Geodetic, SurfaceMorphParams, TileProjection,
    };

    let mut world = EarthWorld::new();
    let pos = GeoPosition::sea_level(0.0, 0.0, 0.0);
    world.spawn(EarthObject::new(
        pos,
        EarthGeometry::CustomMesh {
            vertices: vec![
                EarthVertex::new([-100.0, -100.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0]),
                EarthVertex::new([100.0, -100.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0]),
                EarthVertex::new([0.0, 100.0, 100.0], [0.0, 0.0, 1.0], [0.5, 1.0]),
            ],
            indices: vec![0, 1, 2, 0, 99, u32::MAX],
        },
    ));

    let mut inp = viewpoint(500.0);
    inp.camera.focus = Geodetic::new(0.0, 0.0, 0.0);
    inp.camera.eye_offset_m = [0.0, 0.0, 500.0];
    inp.camera.aim_offset_m = [0.0, 0.0, 0.0];
    let morph = SurfaceMorphParams {
        focus: inp.camera.focus,
        flat_blend: 0.0,
        map_2d_morph: 0.0,
        projection: TileProjection::default(),
    };
    let hit = raycast_viewport(
        400.0,
        300.0,
        800.0,
        600.0,
        &inp.camera,
        None,
        false,
        morph,
        Some(&world),
        &|_, _| 0.0,
    );
    assert!(
        hit.is_some(),
        "valid triangle should still be hit without panicking on invalid indices"
    );
}

#[test]
fn test_open_meteo_500hpa_url_and_offline_failure_backoff() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::time::Duration;
    use strata_earth::{
        AtmosphereConfig, AtmosphereQuery, BaseWeatherConfig, HttpTransport,
        OpenMeteoTemperatureProvider, OpenMeteoWeatherProvider, TemperatureProvider, TileError,
        WeatherProvider, WeatherQuery,
    };

    #[derive(Debug)]
    struct CountingFailTransport {
        calls: Arc<AtomicUsize>,
        last_url: std::sync::Mutex<String>,
    }

    impl HttpTransport for CountingFailTransport {
        fn fetch(&self, url: &str) -> Result<Vec<u8>, TileError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            if let Ok(mut g) = self.last_url.lock() {
                *g = url.to_string();
            }
            Err(TileError::Decode("simulated offline error".to_string()))
        }
    }

    let calls = Arc::new(AtomicUsize::new(0));
    let transport = Arc::new(CountingFailTransport {
        calls: Arc::clone(&calls),
        last_url: std::sync::Mutex::new(String::new()),
    });
    let provider = OpenMeteoWeatherProvider::with_transport(transport.clone())
        .with_failure_backoff(Duration::from_secs(60));

    let query = WeatherQuery {
        lat_deg: 46.57,
        lon_deg: 7.98,
        altitude_msl_m: 2500.0,
        ground_elevation_m: 1200.0,
        unix_utc_s: 1_735_689_600.0,
    };
    let fallback_cfg = BaseWeatherConfig::default();

    // Call sample across 10 consecutive simulated frames while offline
    for _ in 0..10 {
        let _ = provider.sample(query, &fallback_cfg);
        std::thread::sleep(Duration::from_millis(5));
    }

    let total_calls = calls.load(Ordering::SeqCst);
    assert_eq!(
        total_calls, 1,
        "offline OpenMeteoWeatherProvider must back off after failure instead of spawning a thread every frame"
    );
    let url = transport.last_url.lock().unwrap().clone();
    assert!(
        url.contains("wind_speed_500hPa") && url.contains("wind_direction_500hPa"),
        "Open-Meteo query URL must include 500hPa wind speed and direction: {url}"
    );

    // Same backoff guarantee for OpenMeteoTemperatureProvider
    let temp_calls = Arc::new(AtomicUsize::new(0));
    let temp_transport = Arc::new(CountingFailTransport {
        calls: Arc::clone(&temp_calls),
        last_url: std::sync::Mutex::new(String::new()),
    });
    let temp_provider = OpenMeteoTemperatureProvider::with_transport(temp_transport)
        .with_failure_backoff(Duration::from_secs(60));
    let atm_query = AtmosphereQuery {
        lat_deg: 46.57,
        lon_deg: 7.98,
        altitude_msl_m: 2500.0,
        ground_elevation_m: 1200.0,
    };
    let atm_fallback = AtmosphereConfig::default();
    for _ in 0..10 {
        let _ = temp_provider.sample(atm_query, &atm_fallback);
        std::thread::sleep(Duration::from_millis(5));
    }
    assert_eq!(
        temp_calls.load(Ordering::SeqCst),
        1,
        "offline OpenMeteoTemperatureProvider must back off after failure"
    );
}

#[test]
fn test_above_ground_cloud_ceiling_optics_gpu_buildings_graticule_object_labels_and_earth_view_setters() {
    use strata_earth::{
        BuildingFootprint, CloudCeiling, EarthGeometry, EarthObject, EarthPluginRegistry,
        EarthViewport, GeoPosition, MemoryBuildingSource, MemoryTileSource, RetainedTerrainBuilder,
    };

    // 1. Verify CloudCeiling::AboveGround uses ground_elevation_m in resolve_weather_optics
    let mut input = viewpoint(4_500.0);
    input.precipitation.enabled = true;
    input.precipitation.intensity = 0.9;
    input.precipitation.respect_cloud_ceiling = true;
    input.precipitation.atmosphere.cloud_ceiling = CloudCeiling::AboveGround(2_000.0);
    let plugins = EarthPluginRegistry::standard();

    // At sea level (ground=0m), ceiling is 2,000m MSL, so camera at 4,500m is ABOVE ceiling (zero storm darkening)
    let cam_sea = RetainedTerrainBuilder::camera_from_input_with_plugins_weather_and_ground(
        &input, &plugins, 0.0, 1, false, None, 0.0,
    );
    // Over 3,500m mountains, ceiling is 5,500m MSL, so camera at 4,500m is BELOW ceiling (storm darkening active)
    let cam_mtn = RetainedTerrainBuilder::camera_from_input_with_plugins_weather_and_ground(
        &input, &plugins, 0.0, 1, false, None, 3_500.0,
    );
    assert!(
        cam_mtn.weather_darkness > cam_sea.weather_darkness + 0.2,
        "AboveGround cloud ceiling over 3500m terrain must darken sky at 4500m camera altitude (got mtn={}, sea={})",
        cam_mtn.weather_darkness,
        cam_sea.weather_darkness
    );

    // 2. Verify GPU buildings and graticule dynamic meshes in RetainedTerrainBuilder
    let mut b_input = viewpoint(1_200.0);
    b_input.layers.buildings = true;
    b_input.layers.graticule = true;
    b_input.graticule_step_deg = 15.0;
    let mut b_source = MemoryBuildingSource::new();
    b_source.push(BuildingFootprint::new(
        b_input.camera.focus.lat_deg,
        b_input.camera.focus.lon_deg,
        (40.0, 30.0),
        85.0,
    ));
    let mut tile_src = MemoryTileSource::new();
    let mut builder = RetainedTerrainBuilder::new();
    builder.set_building_source(Some(Box::new(b_source)));
    let (frame, _) = builder.build_frame(&b_input, &mut tile_src, 1, true);
    assert!(
        frame.dynamic_meshes.len() >= 2,
        "RetainedTerrainBuilder must emit GPU dynamic meshes for both buildings and graticule (got {})",
        frame.dynamic_meshes.len()
    );

    // 3. Verify EarthViewport object label projection and layer toggles
    let mut vp = EarthViewport::new(&viewpoint(2_000.0), Box::new(MemoryTileSource::new()));
    vp.set_buildings_enabled(true);
    assert!(vp.buildings_enabled());
    vp.set_graticule_enabled(false);
    assert!(!vp.graticule_enabled());
    let focus = vp.input().camera.focus;
    let vp_obj = vp.spawn_object(
        EarthObject::new(
            GeoPosition::sea_level(focus.lat_deg, focus.lon_deg, 10.0),
            EarthGeometry::sphere(15.0),
        )
        .with_label("ALPHA-UAV-01"),
    );
    let projected = vp.project_object_labels();
    assert_eq!(projected.len(), 1);
    assert_eq!(projected[0].text, "ALPHA-UAV-01");
    assert!(vp.remove_object(vp_obj));

    // 4. Verify EarthView top-level convenience setters when widget feature is active
    #[cfg(feature = "widget")]
    {
        use strata_earth::{
            EarthView, EarthViewMode, SpaceEnvironmentConfig, TileProviderPreset, WeatherConfig,
        };
        let view = EarthView::new(&viewpoint(2_000.0), Box::new(MemoryTileSource::new()));
        view.set_labels_enabled(false);
        assert!(!view.labels_enabled());
        view.set_buildings_enabled(true);
        assert!(view.buildings_enabled());
        view.set_graticule_enabled(false);
        assert!(!view.graticule_enabled());
        view.set_sim_time_unix_s(1_740_000_000.0);
        assert!((view.sim_time_unix_s() - 1_740_000_000.0).abs() < 1e-3);
        view.set_space_config(SpaceEnvironmentConfig::default());
        view.set_weather_config(WeatherConfig::default());
        view.set_tile_preset(TileProviderPreset::OpenStreetMap);
        view.set_view_mode_immediate(EarthViewMode::Mode3D);
        assert_eq!(view.view_mode(), EarthViewMode::Mode3D);
    }
}
