//! Host-facing scene API: no mock buildings by default, textures carried with
//! the scene, mutable scene access, and ENU transform helpers.

use strata_earth::{
    BuildingFootprint, EarthCamera, EarthInput, EarthScene, Geodetic, MemoryBuildingSource,
    MemoryTileSource, TileImage,
};
use strata_render::compositor::DisplayList;
use strata_render::math3d::{Mat4, Vec3};
use strata_render::overlay3d::OverlayPrimitive;
use strata_render::rasterizer::FrameBuffer;
use strata_render::scene3d::{Geometry3D, Material3D, Mesh3D};

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
    assert!(scene.scene().textures().is_some());

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
    let a = scene.scene().textures().cloned().expect("textures");
    scene.build(&input, &mut source);
    let b = scene.scene().textures().cloned().expect("textures");
    assert!(std::sync::Arc::ptr_eq(&a, &b));
}

#[test]
fn scene_mut_and_overlay_mut_append_without_rebuilding_terrain() {
    let input = input(2_500.0);
    let mut scene = EarthScene::new();
    scene.build(&input, &mut MemoryTileSource::new());
    let meshes = scene.scene().meshes.len();

    scene.scene_mut().meshes.push(Mesh3D::new(
        Geometry3D::Box {
            width: 1.0,
            height: 1.0,
            depth: 1.0,
        },
        Material3D::default(),
    ));
    assert_eq!(scene.scene().meshes.len(), meshes + 1);

    let before = scene.overlay_mut().primitives.len();
    scene
        .overlay_mut()
        .primitives
        .push(OverlayPrimitive::Polyline {
            points: vec![
                strata_render::camera64::DVec3 {
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                },
                strata_render::camera64::DVec3 {
                    x: 1.0,
                    y: 1.0,
                    z: 1.0,
                },
            ],
            color: strata_style::types::Color::rgb(255, 0, 0),
            width: 2.0,
        });
    assert_eq!(scene.overlay_mut().primitives.len(), before + 1);

    let taken = scene.take_scene_arc();
    assert_eq!(taken.meshes.len(), meshes + 1);
}

fn model_matrix(scene: &EarthScene, pos: Geodetic, ypr: [f64; 3]) -> Mat4 {
    let t = scene.rtc_transform(pos, ypr, [1.0, 1.0, 1.0]);
    Mat4::from_trs(
        Vec3::new(t.position[0], t.position[1], t.position[2]),
        Vec3::new(t.rotation[0], t.rotation[1], t.rotation[2]),
        Vec3::new(t.scale[0], t.scale[1], t.scale[2]),
    )
}

fn dot(v: Vec3, d: strata_render::camera64::DVec3) -> f32 {
    v.x * d.x as f32 + v.y * d.y as f32 + v.z * d.z as f32
}

#[test]
fn rtc_transform_aligns_model_axes_with_enu() {
    let mut scene = EarthScene::new();
    scene.build(&input(2_500.0), &mut MemoryTileSource::new());
    for (lat, lon) in [(37.77, -122.42), (-33.9, 151.2), (64.1, -21.9), (0.0, 0.0)] {
        let pos = Geodetic::new(lat, lon, 100.0);
        let frame = strata_earth::geodesy::LocalFrame::at(lat, lon);
        let m = model_matrix(&scene, pos, [0.0, 0.0, 0.0]);
        let east = m.transform_vector(Vec3::new(1.0, 0.0, 0.0));
        let up = m.transform_vector(Vec3::new(0.0, 1.0, 0.0));
        let nose = m.transform_vector(Vec3::new(0.0, 0.0, -1.0));
        assert!(dot(east, frame.east) > 0.999, "east at {lat},{lon}");
        assert!(dot(up, frame.up) > 0.999, "up at {lat},{lon}");
        assert!(dot(nose, frame.north) > 0.999, "north at {lat},{lon}");
    }
}

#[test]
fn rtc_transform_applies_yaw_pitch_roll() {
    let mut scene = EarthScene::new();
    scene.build(&input(2_500.0), &mut MemoryTileSource::new());
    let pos = Geodetic::new(37.77, -122.42, 100.0);
    let frame = strata_earth::geodesy::LocalFrame::at(pos.lat_deg, pos.lon_deg);
    let nose = Vec3::new(0.0, 0.0, -1.0);
    let wing = Vec3::new(1.0, 0.0, 0.0);

    let yaw = model_matrix(&scene, pos, [90.0, 0.0, 0.0]);
    assert!(dot(yaw.transform_vector(nose), frame.east) > 0.999);

    let pitch = model_matrix(&scene, pos, [0.0, 30.0, 0.0]);
    assert!(dot(pitch.transform_vector(nose), frame.up) > 0.45);

    let roll = model_matrix(&scene, pos, [0.0, 0.0, 30.0]);
    assert!(dot(roll.transform_vector(wing), frame.up) < -0.45);

    // Gimbal lock: heading 0, 90 degrees roll still yields a valid frame.
    let lock = model_matrix(&scene, pos, [0.0, 0.0, 90.0]);
    assert!(dot(lock.transform_vector(wing), frame.up) < -0.99);
    assert!(dot(lock.transform_vector(nose), frame.north) > 0.99);
}
