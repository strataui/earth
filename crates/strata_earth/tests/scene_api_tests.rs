//! Host-facing scene API: no mock buildings by default, textures carried with
//! the scene, native geodetic `EarthWorld` object management, and ENU / Heading-Pitch-Roll transforms.

use strata_earth::{
    BuildingFootprint, EarthCamera, EarthGeometry, EarthInput, EarthMaterial, EarthObject,
    EarthScene, GeoOrientation, GeoPosition, Geodetic, MemoryBuildingSource, MemoryTileSource,
    TileImage,
};
use strata_render::camera64::DVec3;
use strata_render::compositor::{DisplayList, DrawCommand};
use strata_render::rasterizer::FrameBuffer;
use strata_render::scene3d::Material3D;
use strata_style::types::Color;

const W: usize = 400;
const H: usize = 300;

fn input(altitude_m: f64) -> EarthInput {
    EarthInput {
        camera: EarthCamera {
            focus: Geodetic::new(37.7749, -122.4194, 0.0),
            eye_offset_m: [0.0, 0.0, altitude_m],
            ..Default::default()
        },
        viewport: strata_layout::Rect::new(0.0, 0.0, W as f32, H as f32),
        tile_grid: 5,
        ..Default::default()
    }
}

fn built_scene3d(
    scene: &EarthScene,
    viewport: strata_layout::Rect,
) -> std::sync::Arc<strata_render::scene3d::Scene3D> {
    match scene.to_draw_command(viewport) {
        DrawCommand::DrawViewport3D { scene: s, .. } => s,
        _ => panic!("expected DrawViewport3D"),
    }
}

fn red_source(input: &EarthInput) -> MemoryTileSource {
    let span = strata_earth::tile_patch_span_deg(&input.camera, input.patch_scale);
    let mut source = MemoryTileSource::new();
    for patch in strata_earth::surrounding_patches(input.camera.focus, span) {
        source.insert_imagery(
            strata_earth::requested_tile(&patch),
            TileImage {
                width: 2,
                height: 2,
                pixels: (0..4).flat_map(|_| [220, 20, 20, 255]).collect(),
            },
        );
    }
    source
}

#[test]
fn default_input_spawns_no_mock_buildings() {
    assert!(!strata_earth::LayerToggles::default().buildings);
    let input = input(2_500.0);
    let mut scene = EarthScene::new();
    scene.build(&input, &mut MemoryTileSource::new());
    assert_eq!(scene.stats().building_triangles, 0);
}

#[test]
fn enabled_buildings_without_a_source_stay_empty() {
    let mut input = input(2_500.0);
    input.layers.buildings = true;
    let mut scene = EarthScene::new();
    scene.build(&input, &mut MemoryTileSource::new());
    assert_eq!(scene.stats().building_triangles, 0);
}

#[test]
fn building_source_supplies_buildings() {
    let mut input = input(2_500.0);
    input.layers.buildings = true;
    let mut source = MemoryBuildingSource::new();
    source.push(BuildingFootprint::new(
        input.camera.focus.lat_deg,
        input.camera.focus.lon_deg,
        (30.0, 20.0),
        50.0,
    ));
    let mut scene = EarthScene::new();
    scene.set_building_source(Some(Box::new(source)));
    scene.build(&input, &mut MemoryTileSource::new());
    assert_eq!(
        scene.stats().building_triangles,
        strata_earth::building_triangle_count(1)
    );
}

#[test]
fn scene_carries_textures_and_draw_command_renders_them() {
    let input = input(2_500.0);
    let mut source = red_source(&input);
    let mut scene = EarthScene::new();
    scene.build(&input, &mut source);
    let s3d = built_scene3d(&scene, input.viewport);
    assert!(s3d.textures().is_some());

    let mut list = DisplayList::new();
    scene.append_to_display_list(&mut list, input.viewport);
    let mut fb = FrameBuffer::new(W, H);
    fb.execute_display_list(&list);
    let px = fb.get_pixel(W / 2, H / 2);
    assert!(
        px.r() > px.g() && px.r() > px.b(),
        "satellite tile must tint the terrain red, got {px:?}"
    );
}

#[test]
fn unchanged_frames_share_the_texture_store() {
    let input = input(2_500.0);
    let mut source = red_source(&input);
    let mut scene = EarthScene::new();
    scene.build(&input, &mut source);
    scene.build(&input, &mut source);
    let a = built_scene3d(&scene, input.viewport)
        .textures()
        .cloned()
        .expect("textures");
    scene.build(&input, &mut source);
    let b = built_scene3d(&scene, input.viewport)
        .textures()
        .cloned()
        .expect("textures");
    assert!(std::sync::Arc::ptr_eq(&a, &b));
}

#[test]
fn earth_world_spawns_and_removes_geodetic_objects_cleanly() {
    let input = input(2_500.0);
    let mut scene = EarthScene::new();
    scene.build(&input, &mut MemoryTileSource::new());
    let base_draws = scene.stats().draw_calls;

    let box_id = scene.spawn_object(
        EarthObject::new(
            GeoPosition::terrain(37.7749, -122.4194, 15.0),
            EarthGeometry::box_m(20.0, 20.0, 30.0),
        )
        .with_label("Tower")
        .with_material(EarthMaterial::standard(Color::rgb(56, 189, 248))),
    );
    let line_id = scene.spawn_object(
        EarthObject::new(
            GeoPosition::sea_level(37.7749, -122.4194, 40.0),
            EarthGeometry::Polyline {
                points: vec![
                    GeoPosition::sea_level(37.7749, -122.4194, 40.0),
                    GeoPosition::sea_level(37.7780, -122.4150, 80.0),
                ],
                radius_m: 2.5,
            },
        )
        .with_material(EarthMaterial::unlit(Color::rgb(255, 80, 80))),
    );
    assert_eq!(scene.world(|w| w.len()), 2);

    scene.build(&input, &mut MemoryTileSource::new());
    assert_eq!(scene.stats().draw_calls, base_draws + 2);

    assert!(scene.remove_object(box_id));
    assert!(scene.remove_object(line_id));
    assert_eq!(scene.world(|w| w.len()), 0);
}

fn dot_col(col: [f64; 3], d: DVec3) -> f64 {
    col[0] * d.x + col[1] * d.y + col[2] * d.z
}

#[test]
fn geodetic_object_transform_aligns_local_axes_with_enu() {
    let inp = input(12_000_000.0); // Pure 3D globe regime (zero flat morph)
    let mut scene = EarthScene::new();
    scene.build(&inp, &mut MemoryTileSource::new());

    for (lat, lon) in [(37.77, -122.42), (-33.9, 151.2), (64.1, -21.9), (0.0, 0.0)] {
        let id = scene.spawn_object(EarthObject::new(
            GeoPosition::sea_level(lat, lon, 100.0),
            EarthGeometry::box_m(1.0, 1.0, 1.0),
        ));
        let tf = scene
            .resolve_object_transform(id, &inp)
            .expect("resolved transform");
        let frame = strata_earth::geodesy::LocalFrame::at(lat, lon);
        assert!(
            dot_col(tf.model_cols[0], frame.east) > 0.9999,
            "+X must align with East at {lat},{lon}"
        );
        assert!(
            dot_col(tf.model_cols[1], frame.north) > 0.9999,
            "+Y must align with North at {lat},{lon}"
        );
        assert!(
            dot_col(tf.model_cols[2], frame.up) > 0.9999,
            "+Z must align with Up at {lat},{lon}"
        );
    }
}

#[test]
fn geodetic_object_transform_applies_heading_pitch_roll() {
    let inp = input(12_000_000.0);
    let mut scene = EarthScene::new();
    scene.build(&inp, &mut MemoryTileSource::new());

    let (lat, lon) = (37.77, -122.42);
    let frame = strata_earth::geodesy::LocalFrame::at(lat, lon);

    let eval = |scene: &mut EarthScene, h: f64, p: f64, r: f64| {
        let id = scene.spawn_object(
            EarthObject::new(
                GeoPosition::sea_level(lat, lon, 100.0),
                EarthGeometry::box_m(1.0, 1.0, 1.0),
            )
            .with_orientation(GeoOrientation::new(h, p, r)),
        );
        scene
            .resolve_object_transform(id, &inp)
            .expect("resolved transform")
    };

    // Heading 90° clockwise from North rotates +Y (forward) onto +East.
    let h90 = eval(&mut scene, 90.0, 0.0, 0.0);
    assert!(dot_col(h90.model_cols[1], frame.east) > 0.999);

    // Pitch +30° tilts +Y (forward) upward toward +Up.
    let p30 = eval(&mut scene, 0.0, 30.0, 0.0);
    assert!(dot_col(p30.model_cols[1], frame.up) > 0.45);

    // Roll +30° banks +X (right wing) downward toward -Up.
    let r30 = eval(&mut scene, 0.0, 0.0, 30.0);
    assert!(dot_col(r30.model_cols[0], frame.up) < -0.45);

    // 90° roll preserves orthonormal forward (+North) and downward right wing (-Up).
    let r90 = eval(&mut scene, 0.0, 0.0, 90.0);
    assert!(dot_col(r90.model_cols[0], frame.up) < -0.99);
    assert!(dot_col(r90.model_cols[1], frame.north) > 0.99);
}

#[test]
fn fallback_tint_is_configurable_and_neutral_by_default() {
    let default = EarthInput::default();
    assert_eq!(default.fallback_tint.r(), default.fallback_tint.g());
    assert_eq!(default.fallback_tint.g(), default.fallback_tint.b());

    let mut input = input(2_500.0);
    input.fallback_tint = Color::rgba(10, 20, 200, 255);
    let mut scene = EarthScene::new();
    scene.build(&input, &mut MemoryTileSource::new());
    let s3d = built_scene3d(&scene, input.viewport);
    let tinted = s3d.meshes.iter().any(|mesh| match &mesh.material {
        Material3D::TexturedBatch(batch) => batch
            .draws
            .iter()
            .all(|d| d.material.tint == input.fallback_tint),
        _ => false,
    });
    assert!(
        tinted,
        "patches without imagery must use the configured tint"
    );
}
