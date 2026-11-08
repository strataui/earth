//! Acceptance evaluation for the ground parity pair.

use strata_earth::{
    evaluate_acceptance, AcceptanceCriteria, EarthCamera, EarthInput, EarthScene, EntityMarker,
    Geodetic, MemoryTileSource, PixelDelta, TileElevation, TileImage, TileKey,
};
use strata_layout::Rect;
use strata_style::types::Color;

fn source_for(input: &EarthInput) -> (MemoryTileSource, Vec<TileKey>) {
    let span = strata_earth::tile_patch_span_deg(&input.camera, input.patch_scale);
    let keys: Vec<TileKey> = strata_earth::surrounding_patches(input.camera.focus, span)
        .iter()
        .map(strata_earth::requested_tile)
        .collect();
    let mut source = MemoryTileSource::new();
    for key in &keys {
        source.insert_imagery(
            *key,
            TileImage {
                width: 2,
                height: 2,
                pixels: [40, 90, 60, 255].repeat(4),
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
            fov_y_deg: 55.52,
            ..Default::default()
        },
        viewport: Rect::new(0.0, 0.0, 1440.0, 900.0),
        sim_time_unix_s: 1_767_225_600.0,
        cloud_cover: 0.0,
        daylight: 0.6,
        tile_grid: 5,
        markers: vec![EntityMarker {
            position: Geodetic::new(37.7749, -122.4194, 0.0),
            label: "SF".to_string(),
            color: Color::rgba(56, 189, 248, 255),
        }],
        ..Default::default()
    }
}

fn same_frame_delta() -> strata_earth::DescriptorDelta {
    let input = input();
    let (mut source, _) = source_for(&input);
    let mut scene = EarthScene::new();
    scene.build(&input, &mut source);
    let descriptor = strata_earth::FrameDescriptor::capture(&scene, &input);
    strata_earth::compare(&descriptor, &scene)
}

#[test]
fn a_matching_frame_passes_once_pixels_are_measured() {
    let delta = same_frame_delta();
    let criteria = AcceptanceCriteria::default();
    let report = evaluate_acceptance(
        &delta,
        Some(PixelDelta {
            mean_abs_channel: 0.01,
            max_region_fraction_over_24: 0.0,
        }),
        Some(6.5),
        &criteria,
    );
    assert!(report.camera);
    assert!(report.patches);
    assert!(report.uvs);
    assert_eq!(report.pixels, Some(true));
    assert_eq!(report.median_ms, Some(6.5));
    assert!(report.passed);
}

#[test]
fn a_frame_without_pixel_comparison_is_not_accepted() {
    let delta = same_frame_delta();
    let report = evaluate_acceptance(&delta, None, Some(6.5), &AcceptanceCriteria::default());
    assert!(report.camera && report.patches && report.uvs);
    assert_eq!(report.pixels, None);
    assert!(!report.passed, "pixel parity is required for acceptance");
}

#[test]
fn pixel_and_uv_thresholds_are_enforced() {
    let mut delta = same_frame_delta();
    let criteria = AcceptanceCriteria::default();

    delta.tiles.tiles[0].uv_delta = Some([0.001, 0.0, 0.0, 0.0]);
    assert!(evaluate_acceptance(&delta, None, None, &criteria).uvs);
    delta.tiles.tiles[0].uv_delta = Some([0.01, 0.0, 0.0, 0.0]);
    assert!(!evaluate_acceptance(&delta, None, None, &criteria).uvs);

    let failing_pixels = PixelDelta {
        mean_abs_channel: 6.0 / 255.0 + 0.001,
        max_region_fraction_over_24: 0.0,
    };
    let report = evaluate_acceptance(&delta, Some(failing_pixels), None, &criteria);
    assert_eq!(report.pixels, Some(false));
    assert!(!report.passed);

    let region_failure = PixelDelta {
        mean_abs_channel: 0.0,
        max_region_fraction_over_24: 0.006,
    };
    assert_eq!(
        evaluate_acceptance(&delta, Some(region_failure), None, &criteria).pixels,
        Some(false)
    );
}
