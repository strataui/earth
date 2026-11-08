//! Acceptance thresholds for the ground parity frames.
//!
//! The reference owner decides pass or fail from five numbers: camera, patch
//! overlap and level, UV accuracy, pixel difference, and frame time. This
//! module evaluates the first four from a [`DescriptorDelta`] plus caller-supplied
//! pixel statistics, and reports timing without gating it.

use crate::descriptor::DescriptorDelta;
use crate::descriptor::PatchMatch;

/// Pixel difference statistics computed by the parity tool.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PixelDelta {
    /// Mean absolute channel difference over the whole frame, in `0..=1`.
    pub mean_abs_channel: f64,
    /// Largest connected region above 24/255, as a fraction of frame area.
    pub max_region_fraction_over_24: f64,
}

/// Thresholds from the parity decision.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AcceptanceCriteria {
    pub near_far_m: f64,
    pub fov_deg: f64,
    pub eye_m: f64,
    pub patch_overlap: f64,
    pub max_level_delta: i64,
    /// Tile edge in pixels, used to turn one texel into a normalized UV bound.
    pub tile_texels: f64,
    pub mean_abs_channel: f64,
    pub max_region_fraction: f64,
}

impl Default for AcceptanceCriteria {
    fn default() -> Self {
        Self {
            near_far_m: 0.1,
            fov_deg: 0.01,
            eye_m: 0.1,
            patch_overlap: 0.95,
            max_level_delta: 1,
            tile_texels: 256.0,
            mean_abs_channel: 6.0 / 255.0,
            max_region_fraction: 0.005,
        }
    }
}

/// Per-regime acceptance result. A `None` pixel result means the reference PNG
/// has not been compared yet; the frame cannot pass until it has.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AcceptanceReport {
    pub camera: bool,
    pub patches: bool,
    pub uvs: bool,
    pub pixels: Option<bool>,
    /// Plugin median frame time in milliseconds, reported but not gated.
    pub median_ms: Option<f64>,
    pub passed: bool,
}

/// Evaluates the ground-frame criteria.
pub fn evaluate(
    delta: &DescriptorDelta,
    pixels: Option<PixelDelta>,
    median_ms: Option<f64>,
    criteria: &AcceptanceCriteria,
) -> AcceptanceReport {
    let camera = delta.camera.scene_camera_present
        && delta.camera.near_m.abs() <= criteria.near_far_m
        && delta.camera.far_m.abs() <= criteria.near_far_m
        && delta.camera.fov_y_deg.abs() <= criteria.fov_deg
        && delta
            .camera
            .eye_m
            .iter()
            .all(|component| component.abs() <= criteria.eye_m);

    let reference_patches: Vec<_> = delta
        .patches
        .iter()
        .filter(|patch| patch.reference_key.is_some())
        .collect();
    let patches = !reference_patches.is_empty()
        && reference_patches.iter().all(|patch| {
            patch.match_kind != PatchMatch::Uncovered
                && patch.overlap_fraction >= criteria.patch_overlap
                && patch.level_delta.abs() <= criteria.max_level_delta
        });

    let uv_tolerance = 1.0 / criteria.tile_texels.max(1.0);
    let uvs = delta.tiles.tiles.iter().all(|tile| match tile.uv_delta {
        Some(uv) => uv.iter().all(|value| value.abs() <= uv_tolerance),
        None => true,
    });

    let pixels = pixels.map(|pixel| {
        pixel.mean_abs_channel <= criteria.mean_abs_channel
            && pixel.max_region_fraction_over_24 <= criteria.max_region_fraction
    });

    AcceptanceReport {
        camera,
        patches,
        uvs,
        pixels,
        median_ms,
        passed: camera && patches && uvs && pixels == Some(true),
    }
}
