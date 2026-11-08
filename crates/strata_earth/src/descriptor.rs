//! Deterministic golden-frame descriptor for Earth rendering parity and validation.
//!
//! The descriptor records every input that decides the rendered frame: camera
//! placement, resolved tile keys and UV sub-rects, terrain tessellation, layer
//! toggles, markers, sky ramp, and the celestial and cloud declarations. It is
//! written as canonical JSON so the same frame can be replayed on either side
//! and diffed byte for byte.
//!
//! It deliberately does *not* embed pixel data. Imagery and elevation still come
//! from a [`crate::tiles::TileSource`]; both sides key those from the tile
//! entries in the descriptor.

use crate::geodesy::{geodetic_to_ecef, EarthCamera, Geodetic, TileKey};
use crate::json::Json;
use crate::scene::{EarthInput, EarthScene, LayerToggles};
use crate::tiles::TileSource;
use strata_render::scene3d::{Background3D, Camera3D, CelestialBody3D, CloudShell3D};
use strata_style::types::Color;

/// Format tag written into every descriptor.
pub const DESCRIPTOR_FORMAT: &str = "strata-earth-frame";
/// Descriptor schema version.
pub const DESCRIPTOR_VERSION: u32 = 2;
/// First schema version, still accepted by the reader.
pub const DESCRIPTOR_VERSION_V1: u32 = 1;

fn round_f32(value: f32) -> f64 {
    // Keep f32-derived inputs readable and stable across languages instead of
    // emitting artefacts such as 0.550000011920929.
    ((value as f64) * 1_000_000.0).round() / 1_000_000.0
}

fn color_channels(color: Color) -> [f64; 4] {
    let (r, g, b, a) = color.to_rgba8();
    [r as f64, g as f64, b as f64, a as f64]
}

fn color_from_channels(channels: &[Json]) -> Result<Color, String> {
    let values = numbers_from(channels, 4, "color")?;
    Ok(color_from_f64([values[0], values[1], values[2], values[3]]))
}

fn color_from_f64(values: [f64; 4]) -> Color {
    Color::rgba(
        values[0].clamp(0.0, 255.0).round() as u8,
        values[1].clamp(0.0, 255.0).round() as u8,
        values[2].clamp(0.0, 255.0).round() as u8,
        values[3].clamp(0.0, 255.0).round() as u8,
    )
}

fn numbers_from(values: &[Json], count: usize, label: &str) -> Result<Vec<f64>, String> {
    if values.len() != count {
        return Err(format!(
            "{label} needs {count} numbers, got {}",
            values.len()
        ));
    }
    values
        .iter()
        .map(|value| {
            value
                .as_f64()
                .ok_or_else(|| format!("{label} contains a non-number"))
        })
        .collect()
}

fn tile_from_json(value: &Json) -> Result<TileKey, String> {
    let values = value
        .as_array()
        .ok_or_else(|| "tile key must be an array".to_string())?;
    let values = numbers_from(values, 3, "tile key")?;
    Ok(TileKey::new(
        values[0].clamp(0.0, 30.0) as u8,
        values[1].max(0.0) as u32,
        values[2].max(0.0) as u32,
    ))
}

fn tile_to_json(key: TileKey) -> Json {
    Json::numbers(&[key.level as f64, key.x as f64, key.y as f64])
}

fn uv_rect_from_json(value: Option<&Json>) -> Result<Option<[f64; 4]>, String> {
    match value {
        Some(Json::Null) | None => Ok(None),
        Some(value) => {
            let values = numbers_from(
                value
                    .as_array()
                    .ok_or_else(|| "uv_rect must be an array".to_string())?,
                4,
                "uv_rect",
            )?;
            Ok(Some([values[0], values[1], values[2], values[3]]))
        }
    }
}

fn tile_from_descriptor_json(value: &Json) -> Result<DescriptorTile, String> {
    let key = tile_from_json(
        value
            .get("key")
            .ok_or_else(|| "missing tile.key".to_string())?,
    )?;
    let level = value
        .get("level")
        .and_then(Json::as_f64)
        .map(|level| level.clamp(0.0, 30.0) as u8)
        .unwrap_or(key.level);
    Ok(DescriptorTile {
        level,
        key,
        source: tile_from_json(
            value
                .get("source")
                .ok_or_else(|| "missing tile.source".to_string())?,
        )?,
        uv_rect: uv_rect_from_json(value.get("uv_rect"))?,
        exact: value.get("exact").and_then(Json::as_bool).unwrap_or(false),
    })
}

fn surface_from_json(value: &Json) -> Result<DescriptorSphere, String> {
    let kind = value
        .get("kind")
        .and_then(Json::as_str)
        .ok_or_else(|| "missing surface.kind".to_string())?;
    if kind != "sphere" {
        return Err(format!("unsupported surface kind '{kind}'"));
    }
    let centre = numbers_from(
        value
            .get("centre")
            .and_then(Json::as_array)
            .ok_or_else(|| "missing surface.centre".to_string())?,
        3,
        "surface.centre",
    )?;
    let segments = numbers_from(
        value
            .get("segments")
            .and_then(Json::as_array)
            .ok_or_else(|| "missing surface.segments".to_string())?,
        2,
        "surface.segments",
    )?;
    Ok(DescriptorSphere {
        centre: [centre[0], centre[1], centre[2]],
        radius_m: value
            .get("radius_m")
            .and_then(Json::as_f64)
            .ok_or_else(|| "missing surface.radius_m".to_string())?,
        segments: [segments[0].max(3.0) as u32, segments[1].max(2.0) as u32],
    })
}

/// Camera fields needed to rebuild the frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DescriptorCamera {
    /// `[lat_deg, lon_deg, height_m]`.
    pub focus: [f64; 3],
    pub aim_offset_m: [f64; 3],
    pub eye_offset_m: [f64; 3],
    pub fov_y_deg: f64,
    pub near_m: f64,
    pub far_m: f64,
}

/// One resolved terrain patch.
#[derive(Debug, Clone, PartialEq)]
pub struct DescriptorPatch {
    pub key: TileKey,
    pub requested: TileKey,
    /// `[lon0, lon1, lat0, lat1]` in degrees.
    pub bounds: [f64; 4],
    /// `[u_min, v_min, u_max, v_max]`, or `None` when no imagery resolved.
    pub uv_rect: Option<[f64; 4]>,
    /// True when the patch's own tile supplied the image.
    pub exact: bool,
}

/// One texture bound to visible geometry, at the patch's clipmap level.
#[derive(Debug, Clone, PartialEq)]
pub struct DescriptorTile {
    pub level: u8,
    pub key: TileKey,
    /// The tile actually sampled: `key` when exact, an ancestor otherwise.
    pub source: TileKey,
    /// `[u0, v0, u1, v1]`, or `None` when no imagery resolved.
    pub uv_rect: Option<[f64; 4]>,
    /// True only when `source == key`.
    pub exact: bool,
}

/// A spherical globe surface.
#[derive(Debug, Clone, PartialEq)]
pub struct DescriptorSphere {
    pub centre: [f64; 3],
    pub radius_m: f64,
    /// `[longitude_segments, latitude_segments]`.
    pub segments: [u32; 2],
}

/// One clipmap ring of a spherical surface.
#[derive(Debug, Clone, PartialEq)]
pub struct DescriptorRing {
    pub level: u8,
    pub subdiv: u32,
    pub tiles: Vec<DescriptorTile>,
}

/// One overlay marker.
#[derive(Debug, Clone, PartialEq)]
pub struct DescriptorMarker {
    pub position: [f64; 3],
    pub label: String,
    pub color: [f64; 4],
}

/// One celestial body declaration.
#[derive(Debug, Clone, PartialEq)]
pub struct DescriptorCelestial {
    pub position: [f64; 3],
    pub radius: f64,
    pub color: [f64; 4],
    pub glow: f64,
    pub depth_test: bool,
}

/// The cloud-shell declaration.
#[derive(Debug, Clone, PartialEq)]
pub struct DescriptorClouds {
    pub centre: [f64; 3],
    pub radius: f64,
    pub color: [f64; 4],
    pub coverage: f32,
    pub opacity: f32,
}

/// Everything needed to rebuild and compare one Earth frame.
#[derive(Debug, Clone, PartialEq)]
pub struct FrameDescriptor {
    pub viewport: [f64; 2],
    pub camera: DescriptorCamera,
    pub sim_time_unix_s: f64,
    pub cloud_cover: f32,
    pub daylight: f32,
    pub layers: [bool; 6],
    pub graticule_step_deg: f64,
    pub max_tile_level: u8,
    pub tile_grid: u32,
    pub patch_scale: f64,
    pub patches: Vec<DescriptorPatch>,
    /// Every texture bound to visible geometry in this frame.
    pub tiles: Vec<DescriptorTile>,
    /// Present for a spherical globe frame; `patches` is empty in that case.
    pub surface: Option<DescriptorSphere>,
    /// Per-level tile supply for a spherical surface.
    pub rings: Vec<DescriptorRing>,
    pub markers: Vec<DescriptorMarker>,
    pub sky_top: [f64; 4],
    pub sky_bottom: [f64; 4],
    pub celestial: Vec<DescriptorCelestial>,
    pub clouds: Option<DescriptorClouds>,
}

fn patch_to_json(patch: &DescriptorPatch) -> Json {
    Json::object(vec![
        ("key", tile_to_json(patch.key)),
        ("requested", tile_to_json(patch.requested)),
        ("bounds", Json::numbers(&patch.bounds)),
        (
            "uv_rect",
            match patch.uv_rect {
                Some(rect) => Json::numbers(&rect),
                None => Json::Null,
            },
        ),
        ("exact", Json::Bool(patch.exact)),
    ])
}

fn tile_to_descriptor_json(tile: &DescriptorTile) -> Json {
    Json::object(vec![
        ("level", Json::number(tile.level as f64)),
        ("key", tile_to_json(tile.key)),
        ("source", tile_to_json(tile.source)),
        (
            "uv_rect",
            match tile.uv_rect {
                Some(rect) => Json::numbers(&rect),
                None => Json::Null,
            },
        ),
        ("exact", Json::Bool(tile.exact)),
    ])
}

fn surface_to_json(surface: &DescriptorSphere) -> Json {
    Json::object(vec![
        ("kind", Json::string("sphere")),
        ("centre", Json::numbers(&surface.centre)),
        ("radius_m", Json::number(surface.radius_m)),
        (
            "segments",
            Json::numbers(&[surface.segments[0] as f64, surface.segments[1] as f64]),
        ),
    ])
}

fn ring_to_json(ring: &DescriptorRing) -> Json {
    Json::object(vec![
        ("level", Json::number(ring.level as f64)),
        ("subdiv", Json::number(ring.subdiv as f64)),
        (
            "tiles",
            Json::array(ring.tiles.iter().map(tile_to_descriptor_json).collect()),
        ),
    ])
}

impl FrameDescriptor {
    /// Captures the inputs and resolved patch state of the most recent build.
    pub fn capture(scene: &EarthScene, input: &EarthInput) -> Self {
        let placement = input.camera.placement();
        let sky = crate::scene::sky_gradient(input.daylight);
        let (sky_top, sky_bottom) = match sky {
            Background3D::VerticalGradient { top, bottom } => {
                (color_channels(top), color_channels(bottom))
            }
            Background3D::Solid(color) => (color_channels(color), color_channels(color)),
            Background3D::Transparent => ([0.0; 4], [0.0; 4]),
        };
        let celestial = if input.layers.celestial {
            crate::scene::celestial_bodies(input.sim_time_unix_s, input.daylight)
                .into_iter()
                .map(celestial_to_descriptor)
                .collect()
        } else {
            Vec::new()
        };
        let clouds = input.layers.clouds.then(|| DescriptorClouds {
            centre: [0.0; 3],
            radius: 6_378_137.0 + 12_000.0,
            color: color_channels(Color::rgba(255, 255, 255, 255)),
            coverage: input.cloud_cover,
            opacity: 0.85,
        });
        let resolved: Vec<(TileKey, crate::geodesy::TileTextureResolution)> =
            scene.tile_resolutions();
        let patches: Vec<DescriptorPatch> = scene
            .patches()
            .iter()
            .map(|patch| {
                let resolution = resolved
                    .iter()
                    .find(|(key, _)| *key == patch.key)
                    .map(|(_, resolution)| *resolution);
                DescriptorPatch {
                    key: patch.key,
                    requested: patch.requested,
                    bounds: [
                        patch.bounds.lon0,
                        patch.bounds.lon1,
                        patch.bounds.lat0,
                        patch.bounds.lat1,
                    ],
                    uv_rect: resolution
                        .map(|r| [r.uv_rect.0, r.uv_rect.1, r.uv_rect.2, r.uv_rect.3]),
                    exact: resolution.map(|r| r.exact).unwrap_or(false),
                }
            })
            .collect();
        let tiles: Vec<DescriptorTile> = patches
            .iter()
            .filter_map(|patch| {
                resolved
                    .iter()
                    .find(|(key, _)| *key == patch.key)
                    .map(|(_, resolution)| DescriptorTile {
                        level: patch.key.level,
                        key: patch.key,
                        source: resolution.source,
                        uv_rect: Some([
                            resolution.uv_rect.0,
                            resolution.uv_rect.1,
                            resolution.uv_rect.2,
                            resolution.uv_rect.3,
                        ]),
                        exact: resolution.source == patch.key,
                    })
            })
            .collect();
        Self {
            viewport: [input.viewport.width as f64, input.viewport.height as f64],
            camera: DescriptorCamera {
                focus: [
                    input.camera.focus.lat_deg,
                    input.camera.focus.lon_deg,
                    input.camera.focus.height_m,
                ],
                aim_offset_m: input.camera.aim_offset_m,
                eye_offset_m: input.camera.eye_offset_m,
                fov_y_deg: placement.fov_y_deg,
                near_m: placement.near_m,
                far_m: placement.far_m,
            },
            sim_time_unix_s: input.sim_time_unix_s,
            cloud_cover: input.cloud_cover,
            daylight: input.daylight,
            layers: [
                input.layers.terrain,
                input.layers.buildings,
                input.layers.graticule,
                input.layers.labels,
                input.layers.clouds,
                input.layers.celestial,
            ],
            graticule_step_deg: input.graticule_step_deg,
            max_tile_level: input.max_tile_level,
            tile_grid: input.tile_grid,
            patch_scale: input.patch_scale,
            surface: if input.surface_mode == crate::scene::SurfaceMode::SphericalGlobe {
                Some(DescriptorSphere {
                    centre: [0.0, 0.0, 0.0],
                    radius_m: crate::geodesy::wgs84::A,
                    segments: [input.tile_grid.max(2) * 16, input.tile_grid.max(2) * 8],
                })
            } else {
                None
            },
            rings: if input.surface_mode == crate::scene::SurfaceMode::SphericalGlobe {
                let level = crate::geodesy::level_for_altitude(input.camera.altitude_m(), 2);
                vec![DescriptorRing {
                    level,
                    subdiv: input.tile_grid,
                    tiles: tiles.clone(),
                }]
            } else {
                Vec::new()
            },
            patches: if input.surface_mode == crate::scene::SurfaceMode::SphericalGlobe {
                Vec::new()
            } else {
                patches
            },
            tiles,
            markers: input
                .markers
                .iter()
                .map(|marker| DescriptorMarker {
                    position: [
                        marker.position.lat_deg,
                        marker.position.lon_deg,
                        marker.position.height_m,
                    ],
                    label: marker.label.clone(),
                    color: color_channels(marker.color),
                })
                .collect(),
            sky_top,
            sky_bottom,
            celestial,
            clouds,
        }
    }

    /// Rebuilds an [`EarthInput`] from the descriptor.
    pub fn earth_input(&self) -> EarthInput {
        EarthInput {
            camera: EarthCamera {
                focus: Geodetic::new(
                    self.camera.focus[0],
                    self.camera.focus[1],
                    self.camera.focus[2],
                ),
                aim_offset_m: self.camera.aim_offset_m,
                eye_offset_m: self.camera.eye_offset_m,
                fov_y_deg: self.camera.fov_y_deg,
                near_m: self.camera.near_m,
            },
            viewport: strata_layout::Rect::new(
                0.0,
                0.0,
                self.viewport[0] as f32,
                self.viewport[1] as f32,
            ),
            sim_time_unix_s: self.sim_time_unix_s,
            cloud_cover: self.cloud_cover,
            daylight: self.daylight,
            layers: LayerToggles {
                terrain: self.layers[0],
                buildings: self.layers[1],
                graticule: self.layers[2],
                labels: self.layers[3],
                clouds: self.layers[4],
                celestial: self.layers[5],
            },
            graticule_step_deg: self.graticule_step_deg,
            max_tile_level: self.max_tile_level,
            tile_grid: self.tile_grid,
            patch_scale: self.patch_scale,
            markers: self
                .markers
                .iter()
                .map(|marker| crate::scene::EntityMarker {
                    position: Geodetic::new(
                        marker.position[0],
                        marker.position[1],
                        marker.position[2],
                    ),
                    label: marker.label.clone(),
                    color: color_from_channels(marker.color.map(Json::Number).as_ref())
                        .unwrap_or(Color::WHITE),
                })
                .collect(),
            surface_mode: if self.surface.is_some() {
                crate::scene::SurfaceMode::SphericalGlobe
            } else {
                crate::scene::SurfaceMode::FlatPatches
            },
        }
    }

    /// Serializes to canonical JSON.
    pub fn to_json(&self) -> Json {
        let mut entries: Vec<(&str, Json)> = vec![
            ("format", Json::string(DESCRIPTOR_FORMAT)),
            ("version", Json::number(DESCRIPTOR_VERSION as f64)),
            ("viewport", Json::numbers(&self.viewport)),
            (
                "camera",
                Json::object(vec![
                    ("focus", Json::numbers(&self.camera.focus)),
                    ("aim_offset_m", Json::numbers(&self.camera.aim_offset_m)),
                    ("eye_offset_m", Json::numbers(&self.camera.eye_offset_m)),
                    ("fov_y_deg", Json::number(self.camera.fov_y_deg)),
                    ("near_m", Json::number(self.camera.near_m)),
                    ("far_m", Json::number(self.camera.far_m)),
                ]),
            ),
            ("sim_time_unix_s", Json::number(self.sim_time_unix_s)),
            ("cloud_cover", Json::number(round_f32(self.cloud_cover))),
            ("daylight", Json::number(round_f32(self.daylight))),
            (
                "layers",
                Json::object(vec![
                    ("terrain", Json::Bool(self.layers[0])),
                    ("buildings", Json::Bool(self.layers[1])),
                    ("graticule", Json::Bool(self.layers[2])),
                    ("labels", Json::Bool(self.layers[3])),
                    ("clouds", Json::Bool(self.layers[4])),
                    ("celestial", Json::Bool(self.layers[5])),
                ]),
            ),
            ("graticule_step_deg", Json::number(self.graticule_step_deg)),
            ("max_tile_level", Json::number(self.max_tile_level as f64)),
            ("tile_grid", Json::number(self.tile_grid as f64)),
            ("patch_scale", Json::number(self.patch_scale)),
        ];
        // A sphere frame omits patches; a ground frame omits surface and rings.
        if self.surface.is_none() {
            entries.push((
                "patches",
                Json::array(self.patches.iter().map(patch_to_json).collect()),
            ));
        }
        entries.push((
            "tiles",
            Json::array(self.tiles.iter().map(tile_to_descriptor_json).collect()),
        ));
        if let Some(surface) = &self.surface {
            entries.push(("surface", surface_to_json(surface)));
            entries.push((
                "rings",
                Json::array(self.rings.iter().map(ring_to_json).collect()),
            ));
        }
        entries.extend([
            (
                "markers",
                Json::array(
                    self.markers
                        .iter()
                        .map(|marker| {
                            Json::object(vec![
                                ("position", Json::numbers(&marker.position)),
                                ("label", Json::string(marker.label.clone())),
                                ("color", Json::numbers(&marker.color)),
                            ])
                        })
                        .collect(),
                ),
            ),
            (
                "sky",
                Json::object(vec![
                    ("top", Json::numbers(&self.sky_top)),
                    ("bottom", Json::numbers(&self.sky_bottom)),
                ]),
            ),
            (
                "celestial",
                Json::array(
                    self.celestial
                        .iter()
                        .map(|body| {
                            Json::object(vec![
                                ("position", Json::numbers(&body.position)),
                                ("radius", Json::number(body.radius)),
                                ("color", Json::numbers(&body.color)),
                                ("glow", Json::number(body.glow)),
                                ("depth_test", Json::Bool(body.depth_test)),
                            ])
                        })
                        .collect(),
                ),
            ),
            (
                "clouds",
                match &self.clouds {
                    Some(cloud) => Json::object(vec![
                        ("centre", Json::numbers(&cloud.centre)),
                        ("radius", Json::number(cloud.radius)),
                        ("color", Json::numbers(&cloud.color)),
                        ("coverage", Json::number(round_f32(cloud.coverage))),
                        ("opacity", Json::number(round_f32(cloud.opacity))),
                    ]),
                    None => Json::Null,
                },
            ),
        ]);
        Json::object(entries)
    }

    /// Serializes to a JSON string.
    pub fn to_json_string(&self) -> String {
        self.to_json().to_string()
    }

    /// Parses and validates a descriptor.
    pub fn from_json(value: &Json) -> Result<Self, String> {
        let format = value
            .get("format")
            .and_then(Json::as_str)
            .ok_or_else(|| "missing format".to_string())?;
        if format != DESCRIPTOR_FORMAT {
            return Err(format!("unexpected format '{format}'"));
        }
        let version = value
            .get("version")
            .and_then(Json::as_f64)
            .ok_or_else(|| "missing version".to_string())?;
        let version = version as u32;
        if version != DESCRIPTOR_VERSION && version != DESCRIPTOR_VERSION_V1 {
            return Err(format!("unsupported version {version}"));
        }
        let viewport = numbers_from(
            value
                .get("viewport")
                .and_then(Json::as_array)
                .ok_or_else(|| "missing viewport".to_string())?,
            2,
            "viewport",
        )?;
        let camera = value
            .get("camera")
            .ok_or_else(|| "missing camera".to_string())?;
        let focus = numbers_from(
            camera
                .get("focus")
                .and_then(Json::as_array)
                .ok_or_else(|| "missing camera.focus".to_string())?,
            3,
            "camera.focus",
        )?;
        let aim = numbers_from(
            camera
                .get("aim_offset_m")
                .and_then(Json::as_array)
                .ok_or_else(|| "missing camera.aim_offset_m".to_string())?,
            3,
            "camera.aim_offset_m",
        )?;
        let eye = numbers_from(
            camera
                .get("eye_offset_m")
                .and_then(Json::as_array)
                .ok_or_else(|| "missing camera.eye_offset_m".to_string())?,
            3,
            "camera.eye_offset_m",
        )?;
        let number = |name: &str| -> Result<f64, String> {
            camera
                .get(name)
                .and_then(Json::as_f64)
                .ok_or_else(|| format!("missing camera.{name}"))
        };
        let layers = value
            .get("layers")
            .ok_or_else(|| "missing layers".to_string())?;
        let layer =
            |name: &str| -> bool { layers.get(name).and_then(Json::as_bool).unwrap_or(false) };
        let patches = value
            .get("patches")
            .and_then(Json::as_array)
            .unwrap_or(&[])
            .iter()
            .map(|patch| {
                let bounds = numbers_from(
                    patch
                        .get("bounds")
                        .and_then(Json::as_array)
                        .ok_or_else(|| "missing patch.bounds".to_string())?,
                    4,
                    "patch.bounds",
                )?;
                let uv_rect = uv_rect_from_json(patch.get("uv_rect"))?;
                Ok(DescriptorPatch {
                    key: tile_from_json(
                        patch
                            .get("key")
                            .ok_or_else(|| "missing patch.key".to_string())?,
                    )?,
                    requested: tile_from_json(
                        patch
                            .get("requested")
                            .ok_or_else(|| "missing patch.requested".to_string())?,
                    )?,
                    bounds: [bounds[0], bounds[1], bounds[2], bounds[3]],
                    uv_rect,
                    exact: patch.get("exact").and_then(Json::as_bool).unwrap_or(false),
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        let tiles = value
            .get("tiles")
            .and_then(Json::as_array)
            .unwrap_or(&[])
            .iter()
            .map(tile_from_descriptor_json)
            .collect::<Result<Vec<_>, String>>()?;
        let surface = match value.get("surface") {
            Some(Json::Null) | None => None,
            Some(value) => Some(surface_from_json(value)?),
        };
        let rings = value
            .get("rings")
            .and_then(Json::as_array)
            .unwrap_or(&[])
            .iter()
            .map(|ring| {
                Ok(DescriptorRing {
                    level: ring
                        .get("level")
                        .and_then(Json::as_f64)
                        .unwrap_or(0.0)
                        .clamp(0.0, 30.0) as u8,
                    subdiv: ring
                        .get("subdiv")
                        .and_then(Json::as_f64)
                        .unwrap_or(1.0)
                        .max(1.0) as u32,
                    tiles: ring
                        .get("tiles")
                        .and_then(Json::as_array)
                        .unwrap_or(&[])
                        .iter()
                        .map(tile_from_descriptor_json)
                        .collect::<Result<Vec<_>, String>>()?,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        let markers = value
            .get("markers")
            .and_then(Json::as_array)
            .unwrap_or(&[])
            .iter()
            .map(|marker| {
                let position = numbers_from(
                    marker
                        .get("position")
                        .and_then(Json::as_array)
                        .ok_or_else(|| "missing marker.position".to_string())?,
                    3,
                    "marker.position",
                )?;
                let color = color_from_channels(
                    marker
                        .get("color")
                        .and_then(Json::as_array)
                        .ok_or_else(|| "missing marker.color".to_string())?,
                )?;
                Ok(DescriptorMarker {
                    position: [position[0], position[1], position[2]],
                    label: marker
                        .get("label")
                        .and_then(Json::as_str)
                        .unwrap_or("")
                        .to_string(),
                    color: color_channels(color),
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        let sky = value.get("sky").ok_or_else(|| "missing sky".to_string())?;
        let sky_top = numbers_from(
            sky.get("top")
                .and_then(Json::as_array)
                .ok_or_else(|| "missing sky.top".to_string())?,
            4,
            "sky.top",
        )?;
        let sky_bottom = numbers_from(
            sky.get("bottom")
                .and_then(Json::as_array)
                .ok_or_else(|| "missing sky.bottom".to_string())?,
            4,
            "sky.bottom",
        )?;
        let celestial = value
            .get("celestial")
            .and_then(Json::as_array)
            .unwrap_or(&[])
            .iter()
            .map(|body| {
                let position = numbers_from(
                    body.get("position")
                        .and_then(Json::as_array)
                        .ok_or_else(|| "missing celestial.position".to_string())?,
                    3,
                    "celestial.position",
                )?;
                Ok(DescriptorCelestial {
                    position: [position[0], position[1], position[2]],
                    radius: body.get("radius").and_then(Json::as_f64).unwrap_or(0.0),
                    color: color_from_channels(
                        body.get("color")
                            .and_then(Json::as_array)
                            .ok_or_else(|| "missing celestial.color".to_string())?,
                    )
                    .map(color_channels)?,
                    glow: body.get("glow").and_then(Json::as_f64).unwrap_or(0.0),
                    depth_test: body
                        .get("depth_test")
                        .and_then(Json::as_bool)
                        .unwrap_or(false),
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        let clouds = match value.get("clouds") {
            Some(Json::Null) | None => None,
            Some(cloud) => {
                let centre = numbers_from(
                    cloud
                        .get("centre")
                        .and_then(Json::as_array)
                        .ok_or_else(|| "missing clouds.centre".to_string())?,
                    3,
                    "clouds.centre",
                )?;
                Some(DescriptorClouds {
                    centre: [centre[0], centre[1], centre[2]],
                    radius: cloud.get("radius").and_then(Json::as_f64).unwrap_or(0.0),
                    color: color_from_channels(
                        cloud
                            .get("color")
                            .and_then(Json::as_array)
                            .ok_or_else(|| "missing clouds.color".to_string())?,
                    )
                    .map(color_channels)?,
                    coverage: cloud.get("coverage").and_then(Json::as_f64).unwrap_or(0.0) as f32,
                    opacity: cloud.get("opacity").and_then(Json::as_f64).unwrap_or(0.0) as f32,
                })
            }
        };

        Ok(Self {
            viewport: [viewport[0], viewport[1]],
            camera: DescriptorCamera {
                focus: [focus[0], focus[1], focus[2]],
                aim_offset_m: [aim[0], aim[1], aim[2]],
                eye_offset_m: [eye[0], eye[1], eye[2]],
                fov_y_deg: number("fov_y_deg")?,
                near_m: number("near_m")?,
                far_m: number("far_m")?,
            },
            sim_time_unix_s: value
                .get("sim_time_unix_s")
                .and_then(Json::as_f64)
                .unwrap_or(0.0),
            cloud_cover: value
                .get("cloud_cover")
                .and_then(Json::as_f64)
                .unwrap_or(0.0) as f32,
            daylight: value.get("daylight").and_then(Json::as_f64).unwrap_or(1.0) as f32,
            layers: [
                layer("terrain"),
                layer("buildings"),
                layer("graticule"),
                layer("labels"),
                layer("clouds"),
                layer("celestial"),
            ],
            graticule_step_deg: value
                .get("graticule_step_deg")
                .and_then(Json::as_f64)
                .unwrap_or(30.0),
            max_tile_level: value
                .get("max_tile_level")
                .and_then(Json::as_f64)
                .unwrap_or(18.0) as u8,
            tile_grid: value.get("tile_grid").and_then(Json::as_f64).unwrap_or(5.0) as u32,
            patch_scale: value
                .get("patch_scale")
                .and_then(Json::as_f64)
                .unwrap_or(0.25),
            patches,
            tiles,
            surface,
            rings,
            markers,
            sky_top: [sky_top[0], sky_top[1], sky_top[2], sky_top[3]],
            sky_bottom: [sky_bottom[0], sky_bottom[1], sky_bottom[2], sky_bottom[3]],
            celestial,
            clouds,
        })
    }

    /// Parses a descriptor from a JSON string.
    pub fn from_json_str(text: &str) -> Result<Self, String> {
        Self::from_json(&Json::parse(text)?)
    }

    /// Whether the most recent build reproduced this descriptor's tile keys and
    /// UV sub-rects.
    pub fn matches(&self, scene: &EarthScene) -> bool {
        // A spherical surface has no plugin geometry yet, so strict replay
        // cannot certify it.
        if self.surface.is_some() {
            return false;
        }
        let resolved = scene.tile_resolutions();
        if scene.patches().len() != self.patches.len() {
            return false;
        }
        for (actual, expected) in scene.patches().iter().zip(&self.patches) {
            if actual.key != expected.key || actual.requested != expected.requested {
                return false;
            }
            let actual_uv = resolved
                .iter()
                .find(|(key, _)| *key == actual.key).map(|(_, resolution)| [
                        resolution.uv_rect.0,
                        resolution.uv_rect.1,
                        resolution.uv_rect.2,
                        resolution.uv_rect.3,
                    ]);
            match (actual_uv, expected.uv_rect) {
                (None, None) => {}
                (Some(a), Some(b)) => {
                    if a.iter().zip(b.iter()).any(|(a, b)| (a - b).abs() > 1e-9) {
                        return false;
                    }
                }
                _ => return false,
            }
        }
        true
    }
}

fn celestial_to_descriptor(body: CelestialBody3D) -> DescriptorCelestial {
    DescriptorCelestial {
        position: body.position,
        radius: body.radius,
        color: color_channels(body.color),
        glow: body.glow as f64,
        depth_test: body.depth_test,
    }
}

/// Rebuilds a scene from a descriptor and checks the tile decisions round-trip.
pub fn replay(
    scene: &mut EarthScene,
    descriptor: &FrameDescriptor,
    source: &mut dyn TileSource,
) -> bool {
    let input = descriptor.earth_input();
    scene.build(&input, source);
    descriptor.matches(scene)
}

/// Camera difference between a descriptor and the scene it is compared with.
///
/// The deltas are `scene - descriptor`. A near/far delta is the parity
/// measurement: the reference declares its own depth range, and the plugin's
/// policy may legitimately choose a different one.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CameraDelta {
    /// Whether the scene carried a perspective camera.
    pub scene_camera_present: bool,
    pub near_m: f64,
    pub far_m: f64,
    pub fov_y_deg: f64,
    /// Scene eye minus the descriptor's resolved eye, relative to the focus.
    pub eye_m: [f64; 3],
}

impl Default for CameraDelta {
    fn default() -> Self {
        Self {
            scene_camera_present: false,
            near_m: 0.0,
            far_m: 0.0,
            fov_y_deg: 0.0,
            eye_m: [0.0; 3],
        }
    }
}

impl CameraDelta {
    /// Whether every compared field is within tolerance.
    pub fn matches(&self, tolerance: f64) -> bool {
        self.scene_camera_present
            && self.near_m.abs() <= tolerance
            && self.far_m.abs() <= tolerance
            && self.fov_y_deg.abs() <= tolerance
            && self.eye_m.iter().all(|delta| delta.abs() <= tolerance)
    }
}

/// How a reference patch matched the plugin's patch set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PatchMatch {
    /// Both sides use the same tile key.
    Key,
    /// Keys differ but the bounds overlap.
    Overlap,
    /// No plugin patch overlaps this reference patch.
    Uncovered,
    /// A plugin patch with no reference counterpart.
    Extra,
}

/// Difference for one reference patch, or one extra plugin patch.
#[derive(Debug, Clone, PartialEq)]
pub struct PatchDelta {
    /// The reference key, or the plugin key for an extra patch.
    pub key: TileKey,
    pub reference_key: Option<TileKey>,
    pub scene_key: Option<TileKey>,
    pub reference_requested: Option<TileKey>,
    pub scene_requested: Option<TileKey>,
    pub reference_bounds: Option<[f64; 4]>,
    pub scene_bounds: Option<[f64; 4]>,
    pub reference_uv_rect: Option<[f64; 4]>,
    pub scene_uv_rect: Option<[f64; 4]>,
    /// Per-component `scene - reference` UV delta, only for a key match.
    pub uv_delta: Option<[f64; 4]>,
    /// Bounds intersection as a fraction of the reference patch area.
    pub overlap_fraction: f64,
    /// `reference.level - scene.level`, zero for an extra patch.
    pub level_delta: i64,
    pub match_kind: PatchMatch,
}

impl PatchDelta {
    /// Whether both sides request the same tile.
    pub fn requested_mismatch(&self) -> bool {
        self.reference_requested != self.scene_requested
    }

    /// Whether a key-matched patch resolved imagery to a different UV sub-rect.
    pub fn uv_mismatch(&self) -> bool {
        match self.uv_delta {
            Some(delta) => delta.iter().any(|value| value.abs() > 1e-9),
            None => false,
        }
    }
}

/// Difference for one tile key in the supply list.
#[derive(Debug, Clone, PartialEq)]
pub struct TileDelta {
    pub key: TileKey,
    pub reference_source: Option<TileKey>,
    pub scene_source: Option<TileKey>,
    pub reference_uv_rect: Option<[f64; 4]>,
    pub scene_uv_rect: Option<[f64; 4]>,
    pub uv_delta: Option<[f64; 4]>,
}

/// Supply coverage of the reference tile list against the plugin's bound tiles.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TileSupply {
    pub matched: usize,
    pub reference_only: usize,
    pub plugin_only: usize,
    pub uv_mismatches: usize,
    pub tiles: Vec<TileDelta>,
}

/// Counts over a [`DescriptorDelta`].
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct DeltaSummary {
    /// Number of reference patches (zero for a sphere frame).
    pub patches: usize,
    pub patches_key_matched: usize,
    pub patches_overlapping: usize,
    pub patches_uncovered: usize,
    pub patches_extra: usize,
    pub uv_mismatches: usize,
    pub requested_mismatches: usize,
    /// Largest absolute level difference among key and overlap matches.
    pub max_level_delta: i64,
    /// Smallest reference-patch overlap fraction; 1.0 when there are no patches.
    pub min_overlap_fraction: f64,
    pub camera_matches: bool,
}

/// Reporting form of a descriptor-versus-scene comparison.
///
/// Unlike [`replay`], this never fails: a reference descriptor may legitimately
/// resolve different tiles or UV rects, and that difference is the parity
/// measurement.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DescriptorDelta {
    pub patches: Vec<PatchDelta>,
    pub tiles: TileSupply,
    pub camera: CameraDelta,
    pub summary: DeltaSummary,
}

fn bounds_area(bounds: [f64; 4]) -> f64 {
    let width = (bounds[1] - bounds[0]).max(0.0);
    let height = (bounds[3] - bounds[2]).max(0.0);
    width * height
}

fn overlap_fraction(reference: [f64; 4], scene: [f64; 4]) -> f64 {
    let area = bounds_area(reference);
    if area <= 0.0 {
        return 0.0;
    }
    let lon = (reference[1].min(scene[1]) - reference[0].max(scene[0])).max(0.0);
    let lat = (reference[3].min(scene[3]) - reference[2].max(scene[2])).max(0.0);
    (lon * lat / area).clamp(0.0, 1.0)
}

fn uv_delta(reference: Option<[f64; 4]>, scene: Option<[f64; 4]>) -> Option<[f64; 4]> {
    match (reference, scene) {
        (Some(a), Some(b)) => Some([b[0] - a[0], b[1] - a[1], b[2] - a[2], b[3] - a[3]]),
        _ => None,
    }
}

fn patch_bounds(patch: &crate::scene::PatchFrame) -> [f64; 4] {
    [
        patch.bounds.lon0,
        patch.bounds.lon1,
        patch.bounds.lat0,
        patch.bounds.lat1,
    ]
}

fn scene_tiles(scene: &EarthScene) -> Vec<DescriptorTile> {
    let resolved = scene.tile_resolutions();
    scene
        .patches()
        .iter()
        .filter_map(|patch| {
            resolved
                .iter()
                .find(|(key, _)| *key == patch.key)
                .map(|(_, resolution)| DescriptorTile {
                    level: patch.key.level,
                    key: patch.key,
                    source: resolution.source,
                    uv_rect: Some([
                        resolution.uv_rect.0,
                        resolution.uv_rect.1,
                        resolution.uv_rect.2,
                        resolution.uv_rect.3,
                    ]),
                    exact: resolution.source == patch.key,
                })
        })
        .collect()
}

/// Compares a descriptor against a built scene without failing.
///
/// Patches match by key first, then by bounds overlap; a one-level difference
/// reads as a full-overlap match with a level delta, not a missing patch.
/// `replay` stays strict for tamper detection.
pub fn compare(descriptor: &FrameDescriptor, scene: &EarthScene) -> DescriptorDelta {
    let scene_patches = scene.patches();
    let mut used = vec![false; scene_patches.len()];
    let mut patches = Vec::with_capacity(descriptor.patches.len() + scene_patches.len());

    for expected in &descriptor.patches {
        let mut exact = None;
        let mut best: Option<(usize, f64)> = None;
        for (index, actual) in scene_patches.iter().enumerate() {
            if used[index] {
                continue;
            }
            if actual.key == expected.key {
                exact = Some(index);
                break;
            }
            let fraction = overlap_fraction(expected.bounds, patch_bounds(actual));
            if fraction > 0.0 && best.map(|(_, f)| fraction > f).unwrap_or(true) {
                best = Some((index, fraction));
            }
        }
        if let Some(index) = exact {
            used[index] = true;
            let actual = &scene_patches[index];
            let scene_uv = scene_tiles(scene)
                .into_iter()
                .find(|tile| tile.key == actual.key)
                .and_then(|tile| tile.uv_rect);
            patches.push(PatchDelta {
                key: expected.key,
                reference_key: Some(expected.key),
                scene_key: Some(actual.key),
                reference_requested: Some(expected.requested),
                scene_requested: Some(actual.requested),
                reference_bounds: Some(expected.bounds),
                scene_bounds: Some(patch_bounds(actual)),
                reference_uv_rect: expected.uv_rect,
                scene_uv_rect: scene_uv,
                uv_delta: uv_delta(expected.uv_rect, scene_uv),
                overlap_fraction: 1.0,
                level_delta: 0,
                match_kind: PatchMatch::Key,
            });
        } else if let Some((index, fraction)) = best {
            used[index] = true;
            let actual = &scene_patches[index];
            patches.push(PatchDelta {
                key: expected.key,
                reference_key: Some(expected.key),
                scene_key: Some(actual.key),
                reference_requested: Some(expected.requested),
                scene_requested: Some(actual.requested),
                reference_bounds: Some(expected.bounds),
                scene_bounds: Some(patch_bounds(actual)),
                reference_uv_rect: expected.uv_rect,
                scene_uv_rect: scene_tiles(scene)
                    .into_iter()
                    .find(|tile| tile.key == actual.key)
                    .and_then(|tile| tile.uv_rect),
                uv_delta: None,
                overlap_fraction: fraction,
                level_delta: expected.key.level as i64 - actual.key.level as i64,
                match_kind: PatchMatch::Overlap,
            });
        } else {
            patches.push(PatchDelta {
                key: expected.key,
                reference_key: Some(expected.key),
                scene_key: None,
                reference_requested: Some(expected.requested),
                scene_requested: None,
                reference_bounds: Some(expected.bounds),
                scene_bounds: None,
                reference_uv_rect: expected.uv_rect,
                scene_uv_rect: None,
                uv_delta: None,
                overlap_fraction: 0.0,
                level_delta: 0,
                match_kind: PatchMatch::Uncovered,
            });
        }
    }

    for (index, actual) in scene_patches.iter().enumerate() {
        if used[index] {
            continue;
        }
        patches.push(PatchDelta {
            key: actual.key,
            reference_key: None,
            scene_key: Some(actual.key),
            reference_requested: None,
            scene_requested: Some(actual.requested),
            reference_bounds: None,
            scene_bounds: Some(patch_bounds(actual)),
            reference_uv_rect: None,
            scene_uv_rect: scene_tiles(scene)
                .into_iter()
                .find(|tile| tile.key == actual.key)
                .and_then(|tile| tile.uv_rect),
            uv_delta: None,
            overlap_fraction: 0.0,
            level_delta: 0,
            match_kind: PatchMatch::Extra,
        });
    }

    let input = descriptor.earth_input();
    let origin = geodetic_to_ecef(input.camera.focus);
    let placement = input.camera.placement();
    let descriptor_eye = placement.eye.sub(origin);
    let mut camera = CameraDelta {
        scene_camera_present: false,
        near_m: descriptor.camera.near_m,
        far_m: descriptor.camera.far_m,
        fov_y_deg: descriptor.camera.fov_y_deg,
        eye_m: [descriptor_eye.x, descriptor_eye.y, descriptor_eye.z],
    };
    if let Camera3D::Perspective {
        fov_horizontal_deg,
        near,
        far,
        position,
        ..
    } = scene.scene().camera
    {
        let aspect = descriptor.viewport[0] / descriptor.viewport[1].max(1.0);
        camera.scene_camera_present = true;
        camera.near_m = near as f64 - descriptor.camera.near_m;
        camera.far_m = far as f64 - descriptor.camera.far_m;
        camera.fov_y_deg = crate::scene::vertical_fov_deg(fov_horizontal_deg as f64, aspect)
            - descriptor.camera.fov_y_deg;
        camera.eye_m = [
            position[0] as f64 - descriptor_eye.x,
            position[1] as f64 - descriptor_eye.y,
            position[2] as f64 - descriptor_eye.z,
        ];
    }

    let mut tiles = TileSupply::default();
    let plugin_tiles = scene_tiles(scene);
    let mut tile_used = vec![false; plugin_tiles.len()];
    for reference in &descriptor.tiles {
        let index = plugin_tiles
            .iter()
            .enumerate()
            .position(|(index, tile)| !tile_used[index] && tile.key == reference.key);
        match index {
            Some(index) => {
                tile_used[index] = true;
                let plugin = &plugin_tiles[index];
                let delta = uv_delta(reference.uv_rect, plugin.uv_rect);
                if delta
                    .map(|delta| delta.iter().any(|value| value.abs() > 1e-9))
                    .unwrap_or(false)
                {
                    tiles.uv_mismatches += 1;
                }
                tiles.matched += 1;
                tiles.tiles.push(TileDelta {
                    key: reference.key,
                    reference_source: Some(reference.source),
                    scene_source: Some(plugin.source),
                    reference_uv_rect: reference.uv_rect,
                    scene_uv_rect: plugin.uv_rect,
                    uv_delta: delta,
                });
            }
            None => {
                tiles.reference_only += 1;
                tiles.tiles.push(TileDelta {
                    key: reference.key,
                    reference_source: Some(reference.source),
                    scene_source: None,
                    reference_uv_rect: reference.uv_rect,
                    scene_uv_rect: None,
                    uv_delta: None,
                });
            }
        }
    }
    for (index, plugin) in plugin_tiles.iter().enumerate() {
        if tile_used[index] {
            continue;
        }
        tiles.plugin_only += 1;
        tiles.tiles.push(TileDelta {
            key: plugin.key,
            reference_source: None,
            scene_source: Some(plugin.source),
            reference_uv_rect: None,
            scene_uv_rect: plugin.uv_rect,
            uv_delta: None,
        });
    }

    let reference_patches: Vec<&PatchDelta> = patches
        .iter()
        .filter(|patch| patch.reference_key.is_some())
        .collect();
    let summary = DeltaSummary {
        patches: reference_patches.len(),
        patches_key_matched: patches
            .iter()
            .filter(|patch| patch.match_kind == PatchMatch::Key)
            .count(),
        patches_overlapping: patches
            .iter()
            .filter(|patch| patch.match_kind == PatchMatch::Overlap)
            .count(),
        patches_uncovered: patches
            .iter()
            .filter(|patch| patch.match_kind == PatchMatch::Uncovered)
            .count(),
        patches_extra: patches
            .iter()
            .filter(|patch| patch.match_kind == PatchMatch::Extra)
            .count(),
        uv_mismatches: patches.iter().filter(|patch| patch.uv_mismatch()).count(),
        requested_mismatches: patches
            .iter()
            .filter(|patch| patch.reference_key.is_some() && patch.requested_mismatch())
            .count(),
        max_level_delta: reference_patches
            .iter()
            .map(|patch| patch.level_delta.abs())
            .max()
            .unwrap_or(0),
        min_overlap_fraction: reference_patches
            .iter()
            .map(|patch| patch.overlap_fraction)
            .fold(1.0f64, f64::min),
        camera_matches: camera.matches(0.1),
    };

    DescriptorDelta {
        patches,
        tiles,
        camera,
        summary,
    }
}

/// Rebuilds the `Scene3D` declarations that come only from the descriptor.
///
/// Used by the descriptor test suite to prove the sky, celestial, and cloud
/// declarations survive a JSON round-trip, independently of a tile source.
pub fn declared_clouds(descriptor: &FrameDescriptor) -> Option<CloudShell3D> {
    descriptor.clouds.as_ref().map(|cloud| CloudShell3D {
        centre: cloud.centre,
        radius: cloud.radius,
        color: color_from_f64(cloud.color),
        coverage: cloud.coverage,
        opacity: cloud.opacity,
    })
}
