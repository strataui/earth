//! Golden descriptor: canonical JSON, round-trip, and replay.

use strata_earth::{
    replay, EarthCamera, EarthInput, EarthScene, EntityMarker, FrameDescriptor, Geodetic,
    MemoryTileSource, TileElevation, TileImage, TileKey,
};
use strata_layout::Rect;
use strata_style::types::Color;

fn source() -> (MemoryTileSource, Vec<TileKey>) {
    let input = input();
    let span = strata_earth::tile_patch_span_deg(&input.camera, input.patch_scale);
    let keys: Vec<TileKey> = strata_earth::surrounding_patches(input.camera.focus, span)
        .iter()
        .map(strata_earth::requested_tile)
        .collect();
    let mut source = MemoryTileSource::new();
    for (index, key) in keys.iter().enumerate() {
        source.insert_imagery(
            *key,
            TileImage {
                width: 2,
                height: 2,
                pixels: (0..4)
                    .flat_map(|_| [40 + index as u8, 90, 60, 255])
                    .collect(),
            },
        );
        source.insert_elevation(
            *key,
            TileElevation {
                width: 2,
                height: 2,
                heights_m: vec![100.0, 200.0, 150.0, 250.0],
            },
        );
    }
    (source, keys)
}

fn input() -> EarthInput {
    EarthInput {
        camera: EarthCamera {
            focus: Geodetic::new(37.7749, -122.4194, 0.0),
            eye_offset_m: [0.0, 0.0, 4_000.0],
            ..Default::default()
        },
        viewport: Rect::new(0.0, 0.0, 1440.0, 900.0),
        sim_time_unix_s: 1_767_225_600.0,
        cloud_cover: 0.4,
        daylight: 0.7,
        tile_grid: 5,
        markers: vec![EntityMarker {
            position: Geodetic::new(37.7749, -122.4194, 0.0),
            label: "SF".to_string(),
            color: Color::rgba(56, 189, 248, 255),
        }],
        ..Default::default()
    }
}

#[test]
fn descriptor_round_trips_through_json() {
    let (mut source, _) = source();
    let input = input();
    let mut scene = EarthScene::new();
    scene.build(&input, &mut source);

    let descriptor = FrameDescriptor::capture(&scene, &input);
    let text = descriptor.to_json_string();
    let parsed = FrameDescriptor::from_json_str(&text).expect("descriptor parses");
    assert_eq!(parsed, descriptor, "a descriptor must round-trip exactly");
    assert_eq!(
        parsed.to_json_string(),
        text,
        "serialization must be canonical"
    );

    // Core inputs survive the round trip, not just the JSON shape.
    let rebuilt = parsed.earth_input();
    assert_eq!(rebuilt.camera.focus.lat_deg, input.camera.focus.lat_deg);
    assert_eq!(rebuilt.camera.eye_offset_m, input.camera.eye_offset_m);
    assert_eq!(rebuilt.layers, input.layers);
    assert_eq!(rebuilt.markers.len(), 1);
    assert_eq!(rebuilt.markers[0].label, "SF");
}

#[test]
fn descriptor_records_resolved_tiles_and_uv_rects() {
    let (mut source, keys) = source();
    let input = input();
    let mut scene = EarthScene::new();
    scene.build(&input, &mut source);
    let descriptor = FrameDescriptor::capture(&scene, &input);

    assert_eq!(descriptor.patches.len(), keys.len());
    for patch in &descriptor.patches {
        assert!(patch.exact, "every patch has its own image");
        let rect = patch.uv_rect.expect("imagery resolved");
        assert_eq!(rect, [0.0, 0.0, 1.0, 1.0]);
    }
}

#[test]
fn replay_reproduces_the_same_tile_decisions() {
    let (mut first_source, _) = source();
    let input = input();
    let mut first = EarthScene::new();
    first.build(&input, &mut first_source);
    let descriptor = FrameDescriptor::capture(&first, &input);

    let (mut second_source, _) = source();
    let mut second = EarthScene::new();
    assert!(
        replay(&mut second, &descriptor, &mut second_source),
        "a replayed descriptor must resolve the same tiles and UVs"
    );
    assert_eq!(second.stats(), first.stats());
}

#[test]
fn a_tampered_descriptor_is_rejected_by_the_replay_check() {
    let (mut scene_source, _) = source();
    let input = input();
    let mut scene = EarthScene::new();
    scene.build(&input, &mut scene_source);
    let mut descriptor = FrameDescriptor::capture(&scene, &input);
    descriptor.patches[0].uv_rect = Some([0.25, 0.25, 0.5, 0.5]);

    let (mut replay_source, _) = source();
    let mut replay_scene = EarthScene::new();
    assert!(
        !replay(&mut replay_scene, &descriptor, &mut replay_source),
        "a UV rect that does not match the resolver must fail the check"
    );
}

#[test]
fn capture_is_deterministic() {
    let (mut first_source, _) = source();
    let (mut second_source, _) = source();
    let input = input();
    let mut first = EarthScene::new();
    first.build(&input, &mut first_source);
    let mut second = EarthScene::new();
    second.build(&input, &mut second_source);

    let a = FrameDescriptor::capture(&first, &input).to_json_string();
    let b = FrameDescriptor::capture(&second, &input).to_json_string();
    assert_eq!(a, b, "the same frame must serialize identically");
}

#[test]
fn malformed_descriptors_are_reported() {
    assert!(FrameDescriptor::from_json_str("{}").is_err());
    let mut text = FrameDescriptor::capture(&EarthScene::new(), &input()).to_json_string();
    text = text.replace("\"version\":2", "\"version\":99");
    assert!(
        FrameDescriptor::from_json_str(&text).is_err(),
        "an unsupported version must be rejected"
    );
}

#[test]
fn compare_reports_no_delta_for_the_same_frame() {
    let (mut source, _) = source();
    let input = input();
    let mut scene = EarthScene::new();
    scene.build(&input, &mut source);
    let descriptor = FrameDescriptor::capture(&scene, &input);

    let delta = strata_earth::compare(&descriptor, &scene);
    assert_eq!(delta.summary.patches, descriptor.patches.len());
    assert_eq!(delta.summary.patches_key_matched, descriptor.patches.len());
    assert_eq!(delta.summary.patches_uncovered, 0);
    assert_eq!(delta.summary.patches_extra, 0);
    assert_eq!(delta.summary.uv_mismatches, 0);
    assert_eq!(delta.summary.requested_mismatches, 0);
    assert!(delta.summary.camera_matches, "the scene camera must match");
    assert!(delta.camera.near_m.abs() < 0.1);
    assert!(delta.camera.far_m.abs() < 0.1);
}

#[test]
fn compare_reports_a_foreign_descriptor_as_a_delta_not_a_failure() {
    let (mut scene_source, _) = source();
    let input = input();
    let mut scene = EarthScene::new();
    scene.build(&input, &mut scene_source);
    let mut descriptor = FrameDescriptor::capture(&scene, &input);

    // A reference frame may resolve different tiles. The descriptor must still
    // parse, and replay should reject it while compare measures the difference.
    descriptor.patches[0].key = TileKey::new(3, 1, 1);
    descriptor.patches[0].requested = TileKey::new(3, 1, 1);
    descriptor.patches[0].bounds = [0.0, 1.0, 0.0, 1.0];
    descriptor.patches[0].uv_rect = Some([0.25, 0.25, 0.5, 0.5]);
    let text = descriptor.to_json_string();
    let parsed = FrameDescriptor::from_json_str(&text).expect("foreign patch keys must parse");

    let delta = strata_earth::compare(&parsed, &scene);
    assert_eq!(
        delta.summary.patches_uncovered, 1,
        "one descriptor patch is absent from the scene"
    );
    assert_eq!(
        delta.summary.patches_extra, 1,
        "one scene patch is absent from the descriptor"
    );
    let missing = delta
        .patches
        .iter()
        .find(|patch| patch.match_kind == strata_earth::PatchMatch::Uncovered)
        .expect("uncovered patch entry");
    assert_eq!(missing.key, TileKey::new(3, 1, 1));
    assert_eq!(missing.reference_uv_rect, Some([0.25, 0.25, 0.5, 0.5]));

    let (mut replay_source, _) = source();
    let mut replayed = EarthScene::new();
    assert!(
        !replay(&mut replayed, &parsed, &mut replay_source),
        "replay stays strict for tamper detection"
    );
}

#[test]
fn compare_surfaces_a_camera_policy_difference() {
    let (mut source, _) = source();
    let input = input();
    let mut scene = EarthScene::new();
    scene.build(&input, &mut source);
    let mut descriptor = FrameDescriptor::capture(&scene, &input);

    // Pretend the reference declared a different depth range.
    descriptor.camera.near_m = 1.0;
    descriptor.camera.far_m = 51_000_000.0;
    descriptor.camera.fov_y_deg = 45.0;
    let delta = strata_earth::compare(&descriptor, &scene);
    assert!(!delta.summary.camera_matches);
    assert!(delta.camera.near_m > 0.0, "scene near is larger than 1 m");
    assert!(
        delta.camera.far_m < 0.0,
        "scene far is smaller than 51 000 km"
    );
    assert!((delta.camera.fov_y_deg - 15.0).abs() < 1e-3);
}

#[test]
fn a_one_level_difference_reads_as_overlap_not_a_missing_patch() {
    let (mut source, _) = source();
    let input = input();
    let mut scene = EarthScene::new();
    scene.build(&input, &mut source);
    let mut descriptor = FrameDescriptor::capture(&scene, &input);

    // Give the reference patch the child level while keeping the same bounds,
    // which is what a one-level resolver difference looks like.
    let original = descriptor.patches[0].key;
    descriptor.patches[0].key = TileKey::new(original.level + 1, original.x * 2, original.y * 2);
    descriptor.patches[0].requested = descriptor.patches[0].key;

    let delta = strata_earth::compare(&descriptor, &scene);
    let patch = delta
        .patches
        .iter()
        .find(|patch| patch.reference_key == Some(descriptor.patches[0].key))
        .expect("reference patch entry");
    assert_eq!(patch.match_kind, strata_earth::PatchMatch::Overlap);
    assert!(
        patch.overlap_fraction >= 0.95,
        "overlap was {}",
        patch.overlap_fraction
    );
    assert_eq!(patch.level_delta, 1);
    assert_eq!(delta.summary.patches_uncovered, 0);
    assert_eq!(delta.summary.max_level_delta, 1);
}

#[test]
fn the_v1_reader_still_accepts_a_descriptor_without_tiles() {
    use strata_earth::json::Json;

    let (mut source, _) = source();
    let input = input();
    let mut scene = EarthScene::new();
    scene.build(&input, &mut source);
    let descriptor = FrameDescriptor::capture(&scene, &input);

    let mut value = descriptor.to_json();
    if let Json::Object(entries) = &mut value {
        entries.retain(|(key, _)| key != "tiles");
        for (key, value) in entries.iter_mut() {
            if key == "version" {
                *value = Json::Number(1.0);
            }
        }
    }
    let text = value.to_string();
    let parsed = FrameDescriptor::from_json_str(&text).expect("v1 must still parse");
    assert!(parsed.tiles.is_empty());
    assert!(parsed.surface.is_none());
    assert!(!parsed.patches.is_empty());
    assert_eq!(parsed.patches.len(), descriptor.patches.len());
}

#[test]
fn a_sphere_descriptor_omits_patches_and_round_trips() {
    let (mut source, _) = source();
    let input = input();
    let mut scene = EarthScene::new();
    scene.build(&input, &mut source);
    let mut descriptor = FrameDescriptor::capture(&scene, &input);
    descriptor.patches.clear();
    descriptor.tiles.clear();
    descriptor.surface = Some(strata_earth::DescriptorSphere {
        centre: [0.0, 0.0, 0.0],
        radius_m: 6_378_137.0,
        segments: [128, 64],
    });
    descriptor.rings = vec![strata_earth::DescriptorRing {
        level: 2,
        subdiv: 8,
        tiles: vec![strata_earth::DescriptorTile {
            level: 2,
            key: TileKey::new(2, 1, 1),
            source: TileKey::new(2, 1, 1),
            uv_rect: Some([0.0, 0.0, 1.0, 1.0]),
            exact: true,
        }],
    }];

    let text = descriptor.to_json_string();
    assert!(
        !text.contains("\"patches\""),
        "a sphere frame omits patches"
    );
    assert!(text.contains("\"surface\""));
    assert!(text.contains("\"rings\""));
    let parsed = FrameDescriptor::from_json_str(&text).expect("sphere descriptor parses");
    assert_eq!(parsed, descriptor);
    assert_eq!(parsed.rings[0].tiles[0].key, TileKey::new(2, 1, 1));
}
