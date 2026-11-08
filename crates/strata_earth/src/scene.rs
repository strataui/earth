//! Frame assembly: turn an [`EarthInput`] into a renderer-ready scene.

use crate::geodesy::{
    geodetic_to_ecef, resolve_tile_imagery, EarthCamera, Geodetic, TileKey, TileTextureResolution,
};
use crate::tiles::{TextureSlot, TileRegistry, TileSource};
use strata_layout::Rect;
use strata_render::camera64::{Camera64, DVec3, RenderFrame64};
use strata_render::overlay3d::{
    LabelAnchor, MarkerKind, Overlay3D, OverlayPrimitive, OverlayTextStyle,
};
use strata_render::scene3d::{
    Background3D, Camera3D, CelestialBody3D, CloudShell3D, Material3D, Mesh3D, Scene3D,
    TexturedBatchMaterial3D, TexturedDraw3D, TexturedMaterial3D, Transform3D, UvTransform,
};
use strata_render::terrain3d::MeshBuilder;
use strata_style::types::Color;

/// Which layers a frame draws.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LayerToggles {
    pub terrain: bool,
    pub buildings: bool,
    pub graticule: bool,
    pub labels: bool,
    pub clouds: bool,
    pub celestial: bool,
}

impl Default for LayerToggles {
    fn default() -> Self {
        Self {
            terrain: true,
            buildings: true,
            graticule: true,
            labels: true,
            clouds: true,
            celestial: true,
        }
    }
}

/// An entity marker drawn by the overlay pass.
#[derive(Debug, Clone, PartialEq)]
pub struct EntityMarker {
    pub position: Geodetic,
    pub label: String,
    pub color: Color,
}

/// Surface geometry mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SurfaceMode {
    /// 3x3 planar terrain patches tangential to the camera focus point (default).
    #[default]
    FlatPatches,
    /// True spherical Earth globe geometry at WGS84 radius.
    SphericalGlobe,
}

/// Everything a frame needs.
#[derive(Debug, Clone, PartialEq)]
pub struct EarthInput {
    pub camera: EarthCamera,
    pub viewport: Rect,
    /// Simulation time in Unix seconds, used for sun and moon direction.
    pub sim_time_unix_s: f64,
    /// Cloud coverage, 0..=1.
    pub cloud_cover: f32,
    /// Daylight factor, 0..=1; the caller may derive it from `sim_time_unix_s`.
    pub daylight: f32,
    pub layers: LayerToggles,
    pub graticule_step_deg: f64,
    /// Highest tile level the view may request.
    pub max_tile_level: u8,
    /// Grid resolution of one terrain tile patch.
    pub tile_grid: u32,
    /// Patch size multiplier relative to the camera's ground coverage. Larger
    /// values mean fewer, bigger patches.
    pub patch_scale: f64,
    pub markers: Vec<EntityMarker>,
    /// Geometry mode for surface rendering.
    pub surface_mode: SurfaceMode,
}

impl Default for EarthInput {
    fn default() -> Self {
        Self {
            camera: EarthCamera::default(),
            viewport: Rect::new(0.0, 0.0, 1440.0, 900.0),
            sim_time_unix_s: 0.0,
            cloud_cover: 0.0,
            daylight: 1.0,
            layers: LayerToggles::default(),
            graticule_step_deg: 30.0,
            max_tile_level: 18,
            tile_grid: 5,
            patch_scale: 0.25,
            markers: Vec::new(),
            surface_mode: SurfaceMode::default(),
        }
    }
}

/// Per-frame statistics, published for the budget report.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FrameStats {
    pub tiles_requested: usize,
    pub tiles_exact: usize,
    pub tiles_from_ancestor: usize,
    pub terrain_triangles: usize,
    pub building_triangles: usize,
    pub overlay_primitives: usize,
    /// Mesh draws submitted to the rasterizer: one per visible patch, plus one
    /// for all buildings.
    pub draw_calls: usize,
}

/// The plugin's reusable frame builder.
///
/// One instance is held by the host and reused every frame so mesh buffers and
/// the texture registry keep their capacity.
#[derive(Debug)]
pub struct EarthScene {
    scene: Scene3D,
    terrain: MeshBuilder,
    buildings: MeshBuilder,
    tiles: TileRegistry,
    stats: FrameStats,
    patches: Vec<PatchFrame>,
    frame_width: f32,
    frame_height: f32,
}

impl Default for EarthScene {
    fn default() -> Self {
        Self::new()
    }
}

impl EarthScene {
    pub fn new() -> Self {
        Self {
            scene: Scene3D::new(),
            terrain: MeshBuilder::new(),
            buildings: MeshBuilder::new(),
            tiles: TileRegistry::new(),
            stats: FrameStats::default(),
            patches: Vec::new(),
            frame_width: 1440.0,
            frame_height: 900.0,
        }
    }

    /// Number of tile images currently uploaded.
    pub fn uploaded_tiles(&self) -> usize {
        self.tiles.len()
    }

    /// Statistics for the most recent build.
    pub fn stats(&self) -> FrameStats {
        self.stats
    }

    /// The scene produced by the most recent build.
    pub fn scene(&self) -> &Scene3D {
        &self.scene
    }

    /// The texture registry the renderer should sample from.
    pub fn tile_registry(&self) -> &TileRegistry {
        &self.tiles
    }

    /// The exact frame the renderer uses to project this scene's overlay.
    ///
    /// Building the frame from `EarthInput` directly can differ from this when
    /// the renderer re-derives the camera, so anchor maths should use this.
    pub fn renderer_overlay_frame(&self) -> RenderFrame64 {
        strata_render::rasterizer3d::overlay_frame(&self.scene, self.viewport_for_frame())
    }

    fn viewport_for_frame(&self) -> Rect {
        Rect::new(0.0, 0.0, self.frame_width, self.frame_height)
    }

    /// The f64 overlay frame matching the camera, for anchor projection.
    pub fn overlay_frame(&self, input: &EarthInput) -> RenderFrame64 {
        let placement = input.camera.placement();
        // Scene coordinates are relative to the focus point, so the frame's
        // camera must use the same origin.
        let origin = geodetic_to_ecef(input.camera.focus);
        let aspect = (input.viewport.width / input.viewport.height.max(1.0)).max(0.01);
        let camera = Camera64 {
            eye: placement.eye.sub(origin),
            target: placement.aim.sub(origin),
            up: placement.up,
            fov_horizontal_deg: horizontal_fov_deg(placement.fov_y_deg, aspect as f64),
            near: placement.near_m,
            far: placement.far_m,
        };
        RenderFrame64::from_camera(
            &camera,
            input.viewport.width as f64,
            input.viewport.height as f64,
        )
    }

    /// Rebuilds the scene for this frame.
    ///
    /// Tile resolution is pure and can be inspected through [`FrameStats`] and
    /// [`EarthScene::tile_resolutions`] without any GPU work.
    pub fn build(&mut self, input: &EarthInput, source: &mut dyn TileSource) {
        let placement = input.camera.placement();
        let mut stats = FrameStats::default();
        // The whole scene is authored relative to the focus point, so the
        // camera is too. This is the RTC contract: subtract in f64 first, then
        // let the f32 rasterizer work on small coordinates.
        let origin = geodetic_to_ecef(input.camera.focus);

        // 1. Camera. The engine camera declares a horizontal field of view, so
        // the plugin's vertical FOV is converted at this boundary.
        let aspect = (input.viewport.width / input.viewport.height.max(1.0)).max(0.01);
        let scene_camera = Camera3D::Perspective {
            fov_horizontal_deg: horizontal_fov_deg(placement.fov_y_deg, aspect as f64) as f32,
            near: placement.near_m as f32,
            far: placement.far_m as f32,
            position: placement.eye.sub(origin).as_f32(),
            target: placement.aim.sub(origin).as_f32(),
        };

        // 2. Background: daylight drives the sky ramp.
        let sky = sky_gradient(input.daylight);
        let mut scene = Scene3D::new();
        scene.camera = scene_camera;
        // Space lighting: a bright ambient fill plus a directional key from the
        // camera's side so terrain and buildings read without a sun shadow pass.
        scene.add_light(strata_render::scene3d::Light3D::Ambient {
            color: Color::rgba(150, 170, 200, 255),
            intensity: 0.25 + 0.2 * input.daylight.clamp(0.0, 1.0),
        });
        let key_dir = placement.eye.sub(placement.aim).normalize();
        scene.add_light(strata_render::scene3d::Light3D::Directional {
            position: [
                key_dir.x as f32 + 0.35,
                key_dir.y as f32 + 0.35,
                key_dir.z as f32 + 0.25,
            ],
            color: Color::rgba(255, 244, 214, 255),
            intensity: 0.15 + 0.2 * input.daylight.clamp(0.0, 1.0),
            cast_shadow: false,
        });
        scene = scene
            .with_background(sky)
            .with_camera_up(placement.up.as_f32().map(|v| v as f64));

        if input.layers.celestial {
            for body in celestial_bodies(input.sim_time_unix_s, input.daylight) {
                scene = scene.with_celestial(body);
            }
        }
        if input.layers.clouds {
            scene = scene.with_cloud_shell(CloudShell3D {
                centre: [0.0; 3],
                radius: 6_378_137.0 + 12_000.0,
                color: Color::rgba(255, 255, 255, 255),
                coverage: input.cloud_cover,
                opacity: 0.85,
            });
        }

        // 3. Terrain patches around the focus point. Patch size follows the
        // camera's ground coverage, so the frame budget and the rasterizer's
        // per-triangle clipping stay consistent at every zoom level.
        self.terrain.clear();
        self.buildings.clear();
        self.patches.clear();
        self.terrain.set_origin(origin);
        self.buildings.set_origin(origin);

        if input.layers.terrain {
            match input.surface_mode {
                SurfaceMode::FlatPatches => {
                    let span_deg = tile_patch_span_deg(&input.camera, input.patch_scale);
                    let bounds = surrounding_patches(input.camera.focus, span_deg);
                    for mut patch in bounds {
                        stats.tiles_requested += 1;
                        let requested = patch.requested;
                        let resolution = resolve_tile_imagery(requested, |candidate| {
                            matches!(source.imagery(candidate), Ok(Some(_)))
                        });
                        if let Some(resolution) = resolution {
                            if resolution.exact {
                                stats.tiles_exact += 1;
                            } else {
                                stats.tiles_from_ancestor += 1;
                            }
                        }

                        let elevation = source.elevation(requested).ok().flatten();
                        let (index_start, index_count) = push_tile_patch(
                            &mut self.terrain,
                            patch.bounds,
                            input.tile_grid.max(2),
                            elevation.as_ref(),
                            requested,
                        );
                        patch.resolution = resolution;
                        patch.index_start = index_start;
                        patch.index_count = index_count;

                        if input.layers.buildings {
                            push_placeholder_buildings(
                                &mut self.buildings,
                                origin,
                                patch.bounds,
                                elevation.as_ref(),
                                requested,
                            );
                        }
                        self.patches.push(patch);
                    }
                }
                SurfaceMode::SphericalGlobe => {
                    let level = crate::geodesy::level_for_altitude(input.camera.altitude_m(), 2);
                    let per_axis = 1u32 << level;
                    let lon_span = 360.0 / per_axis as f64;
                    let lat_span = 180.0 / per_axis as f64;
                    for y in 0..per_axis {
                        let lat1 = 90.0 - y as f64 * lat_span;
                        let lat0 = lat1 - lat_span;
                        for x in 0..per_axis {
                            let lon0 = -180.0 + x as f64 * lon_span;
                            let lon1 = lon0 + lon_span;
                            let key = TileKey::new(level, x, y);
                            let bounds = PatchBounds {
                                lon0,
                                lon1,
                                lat0,
                                lat1,
                            };
                            stats.tiles_requested += 1;
                            let requested = key;
                            let resolution = resolve_tile_imagery(requested, |candidate| {
                                matches!(source.imagery(candidate), Ok(Some(_)))
                            });
                            if let Some(resolution) = resolution {
                                if resolution.exact {
                                    stats.tiles_exact += 1;
                                } else {
                                    stats.tiles_from_ancestor += 1;
                                }
                            }

                            let elevation = source.elevation(requested).ok().flatten();
                            let (index_start, index_count) = push_tile_patch(
                                &mut self.terrain,
                                bounds,
                                input.tile_grid.max(2),
                                elevation.as_ref(),
                                requested,
                            );
                            self.patches.push(PatchFrame {
                                key,
                                requested,
                                bounds,
                                resolution,
                                index_start,
                                index_count,
                            });
                        }
                    }
                }
            }
        }

        // 4. Attach the whole terrain as one batched textured mesh, and all
        // buildings as one untextured mesh. Both buffers were filled above.
        if let Some(geometry) = self.terrain.finish() {
            let mut batch = TexturedBatchMaterial3D::new();
            let fallback = self.tiles.fallback_texture();
            for patch in &self.patches {
                if let Some(resolution) = &patch.resolution {
                    let slot = TextureSlot {
                        key: resolution.source,
                        revision: 0,
                    };
                    let texture = match self.tiles.get(slot) {
                        Some(id) => Some(id),
                        None => match source.imagery(resolution.source) {
                            Ok(Some(image)) => self.tiles.ensure(slot, &image),
                            _ => None,
                        },
                    };
                    let (texture, tint) = match texture {
                        Some(id) => (id, Color::WHITE),
                        None => (fallback, Color::rgba(70, 110, 80, 255)),
                    };
                    let (u0, v0, u1, v1) = resolution.uv_rect;
                    batch.push(TexturedDraw3D::new(
                        TexturedMaterial3D::new(texture)
                            .with_uv_transform(UvTransform::from_sub_rect(
                                u0 as f32, v0 as f32, u1 as f32, v1 as f32,
                            ))
                            .with_tint(tint),
                        patch.index_start,
                        patch.index_count,
                    ));
                } else {
                    batch.push(TexturedDraw3D::new(
                        TexturedMaterial3D::new(fallback).with_tint(Color::rgba(70, 110, 80, 255)),
                        patch.index_start,
                        patch.index_count,
                    ));
                }
            }
            if !batch.is_empty() {
                scene.add_mesh(
                    Mesh3D::new(geometry, Material3D::TexturedBatch(batch))
                        .with_transform(Transform3D::default()),
                );
            }
        }
        if let Some(geometry) = self.buildings.finish() {
            scene.add_mesh(
                Mesh3D::new(
                    geometry,
                    Material3D::Standard {
                        color: Color::rgba(178, 184, 196, 255),
                        roughness: 0.8,
                        metalness: 0.0,
                        emissive: None,
                    },
                )
                .with_transform(Transform3D::default()),
            );
        }

        stats.terrain_triangles = self.terrain.triangle_count();
        stats.building_triangles = self.buildings.triangle_count();
        stats.draw_calls = self.patches.len() + usize::from(stats.building_triangles > 0);

        // 5. Overlay: graticule, markers, labels.
        let mut overlay = Overlay3D::new();
        if input.layers.graticule {
            push_graticule(
                &mut overlay,
                input.graticule_step_deg,
                origin,
                placement.eye.sub(origin),
            );
        }
        if input.layers.labels {
            for marker in &input.markers {
                let anchor = geodetic_to_ecef(marker.position).sub(origin);
                overlay.push(OverlayPrimitive::Marker {
                    anchor,
                    kind: MarkerKind::Diamond,
                    color: marker.color,
                    size: 7.0,
                });
                overlay.push(OverlayPrimitive::Label {
                    anchor,
                    text: marker.label.clone(),
                    style: OverlayTextStyle {
                        color: marker.color,
                        scale: 1,
                    },
                    placement: LabelAnchor::Above,
                    leader: true,
                });
            }
        }
        stats.overlay_primitives = overlay.primitives.len();
        if !overlay.is_empty() {
            scene = scene.with_overlay(overlay);
        }

        self.frame_width = input.viewport.width;
        self.frame_height = input.viewport.height;
        self.scene = scene;
        self.stats = stats;
    }

    /// Terrain patches built during the most recent frame, in draw order.
    pub fn patches(&self) -> &[PatchFrame] {
        &self.patches
    }

    /// Tile resolutions computed during the most recent build, one per patch
    /// that had usable imagery.
    pub fn tile_resolutions(&self) -> Vec<(TileKey, TileTextureResolution)> {
        self.patches
            .iter()
            .filter_map(|p| p.resolution.as_ref().map(|r| (p.key, *r)))
            .collect()
    }

    /// Whether any built tile quad is a usable texture target. A tile quad is
    /// only attached when its image is uploaded, so `false` means imagery is
    /// still missing for every visible tile.
    pub fn has_textured_geometry(&self) -> bool {
        let fallback = self.tiles.fallback_texture();
        self.scene.meshes.iter().any(|mesh| match &mesh.material {
            Material3D::Textured(material) => material.texture != fallback,
            Material3D::TexturedBatch(batch) => batch
                .draws
                .iter()
                .any(|draw| draw.material.texture != fallback),
            _ => false,
        })
    }
}

/// Shared terrain triangles: one patch per tile, two triangles per grid cell.
pub fn terrain_triangle_count(tile_count: usize, grid: u32) -> usize {
    let cells = (grid.max(2) as usize - 1).pow(2);
    tile_count * cells * 2
}

/// Shared building triangles: four walls plus the roof cap per footprint.
pub fn building_triangle_count(footprints: usize) -> usize {
    footprints * 10
}

/// Daylight sky ramp: deep space to atmospheric zenith.
pub fn sky_gradient(daylight: f32) -> Background3D {
    let t = daylight.clamp(0.0, 1.0);
    let mix = |a: u8, b: u8| -> u8 {
        (a as f32 + (b as f32 - a as f32) * t)
            .round()
            .clamp(0.0, 255.0) as u8
    };
    Background3D::VerticalGradient {
        top: Color::rgba(mix(2, 30), mix(4, 64), mix(14, 140), 255),
        bottom: Color::rgba(mix(6, 120), mix(10, 180), mix(24, 255), 255),
    }
}

/// Sun and moon positions for a simulation time, in ECEF metres.
///
/// This is a deliberately small ephemeris: a solar and a lunar orbit model with
/// the right gross behaviour (matching seasonal latitude drift and monthly moon
/// phase) without a full almanac.
pub fn celestial_bodies(sim_time_unix_s: f64, daylight: f32) -> Vec<CelestialBody3D> {
    let days = sim_time_unix_s / 86_400.0;
    let sun_angle = (days * std::f64::consts::TAU / 365.256_363).rem_euclid(std::f64::consts::TAU);
    let moon_angle = (days * std::f64::consts::TAU / 27.321_661).rem_euclid(std::f64::consts::TAU);
    let tilt = 23.44f64.to_radians();

    let sun_dir = DVec3::new(
        sun_angle.cos(),
        sun_angle.sin() * tilt.sin(),
        sun_angle.sin() * tilt.cos(),
    );
    let moon_dir = DVec3::new(
        moon_angle.cos(),
        moon_angle.sin() * 0.9,
        moon_angle.sin() * 0.436,
    );

    let bright = daylight.clamp(0.0, 1.0);
    vec![
        CelestialBody3D {
            position: [
                sun_dir.normalize().x * 1.496e11,
                sun_dir.normalize().y * 1.496e11,
                sun_dir.normalize().z * 1.496e11,
            ],
            radius: 4.65e9,
            color: Color::rgba(255, 247, 226, 255),
            glow: 2.0 + bright * 2.0,
            depth_test: false,
        },
        CelestialBody3D {
            position: [
                moon_dir.normalize().x * 3.844e8,
                moon_dir.normalize().y * 3.844e8,
                moon_dir.normalize().z * 3.844e8,
            ],
            radius: 3.47e6,
            color: Color::rgba(203, 213, 225, 255),
            glow: 1.0,
            depth_test: false,
        },
    ]
}

/// Converts a vertical field of view to the horizontal field the scene camera
/// declares at `aspect`.
pub fn horizontal_fov_deg(vertical_fov_deg: f64, aspect: f64) -> f64 {
    let half = (vertical_fov_deg.to_radians() * 0.5).tan() * aspect.max(1e-6);
    2.0 * half.atan().to_degrees()
}

/// Converts a horizontal field of view to the vertical field at `aspect`.
pub fn vertical_fov_deg(horizontal_fov_deg: f64, aspect: f64) -> f64 {
    let half = (horizontal_fov_deg.to_radians() * 0.5).tan() / aspect.max(1e-6);
    2.0 * half.atan().to_degrees()
}

/// Width of one terrain patch in degrees of longitude.
///
/// The patch size follows the camera's ground coverage, scaled by
/// `patch_scale`. Nine patches then cover the view without rasterizing several
/// screens' worth of overlapping geometry. The camera's depth range comes from
/// [`crate::geodesy::depth_range`], so patch size no longer has to compensate
/// for a planet-scale far plane.
pub fn tile_patch_span_deg(camera: &EarthCamera, patch_scale: f64) -> f64 {
    let metres = camera.ground_span_m() * patch_scale;
    let degrees = metres / (crate::geodesy::wgs84::A * std::f64::consts::PI / 180.0);
    degrees.clamp(1e-6, 20.0)
}

/// One terrain patch in the frame.
#[derive(Debug, Clone, PartialEq)]
pub struct PatchFrame {
    /// Tile key used for imagery and reporting.
    pub key: TileKey,
    /// Tile whose imagery and elevation the patch addresses.
    pub requested: TileKey,
    pub bounds: PatchBounds,
    pub resolution: Option<TileTextureResolution>,
    /// Index range this patch occupies in the shared terrain mesh.
    pub index_start: u32,
    pub index_count: u32,
}

/// The patch containing `focus` plus its eight neighbours.
///
/// `span_deg` is the patch width, as returned by [`tile_patch_span_deg`]. Keys
/// carry the enclosing tile level and index so texture resolution and reporting
/// stay in tile space.
pub fn surrounding_patches(focus: Geodetic, span_deg: f64) -> Vec<PatchFrame> {
    // Patches are tile-aligned and sized within a factor of two of `span_deg`.
    // That keeps imagery UVs exact and stops nine patches from covering several
    // screens at once, which would rasterize the same pixels repeatedly.
    let level = ((360.0 / span_deg.max(1e-9)).log2().round().max(0.0) as i64).clamp(0, 30) as u8;
    let per_axis = 1u32 << level;
    // A quadtree level has 2^level tiles across 360 degrees of longitude and
    // across 180 degrees of latitude.
    let lon_span = 360.0 / per_axis as f64;
    let lat_span = 180.0 / per_axis as f64;
    let cx = ((focus.lon_deg + 180.0) / lon_span).floor() as i64;
    let cy = ((90.0 - focus.lat_deg) / lat_span).floor() as i64;

    let mut out = Vec::with_capacity(9);
    for dy in -1..=1 {
        let row = cy + dy;
        if row < 0 || row >= per_axis as i64 {
            continue;
        }
        for dx in -1..=1 {
            let column = (cx + dx).rem_euclid(per_axis as i64);
            let lon0 = -180.0 + column as f64 * lon_span;
            let lat1 = 90.0 - row as f64 * lat_span;
            let key = TileKey::new(level, column as u32, row as u32);
            out.push(PatchFrame {
                key,
                requested: key,
                bounds: PatchBounds {
                    lon0,
                    lon1: lon0 + lon_span,
                    lat0: lat1 - lat_span,
                    lat1,
                },
                resolution: None,
                index_start: 0,
                index_count: 0,
            });
        }
    }
    out
}

/// The tile whose imagery and elevation a patch addresses.
///
/// This is the patch's own tile, so its UV mapping and height samples cover the
/// whole patch. Missing imagery resolves through the ancestors of this key.
pub fn requested_tile(patch: &PatchFrame) -> TileKey {
    patch.requested
}

/// Geographic bounds of a tile at its own level.
pub fn tile_bounds(key: TileKey) -> PatchBounds {
    let per_axis = key.tiles_per_axis() as f64;
    let lon_span = 360.0 / per_axis;
    let lat_span = 180.0 / per_axis;
    let lon0 = key.x as f64 * lon_span - 180.0;
    let lat1 = 90.0 - key.y as f64 * lat_span;
    PatchBounds {
        lon0,
        lon1: lon0 + lon_span,
        lat0: lat1 - lat_span,
        lat1,
    }
}

/// Maps a geographic point into a tile's normalized `(u, v)` space.
///
/// `u` runs west to east and `v` north to south, matching imagery and
/// elevation row order.
pub fn uv_in_tile(key: TileKey, lat_deg: f64, lon_deg: f64) -> (f32, f32) {
    let bounds = tile_bounds(key);
    let u = ((lon_deg - bounds.lon0) / (bounds.lon1 - bounds.lon0)).clamp(0.0, 1.0);
    let v = ((bounds.lat1 - lat_deg) / (bounds.lat1 - bounds.lat0)).clamp(0.0, 1.0);
    (u as f32, v as f32)
}

/// The tile at `focus` plus its eight neighbours at `level`.
pub fn surrounding_tiles(focus: Geodetic, level: u8) -> Vec<TileKey> {
    let per_axis = 1u32 << level;
    let lon = focus.lon_deg.rem_euclid(360.0) / 360.0;
    let lat = (90.0 - focus.lat_deg.clamp(-85.0, 85.0)) / 180.0;
    let cx = ((lon * per_axis as f64) as i64).rem_euclid(per_axis as i64);
    let cy = ((lat * per_axis as f64) as i64).clamp(0, per_axis as i64 - 1);

    let mut out = Vec::with_capacity(9);
    for dy in -1i64..=1 {
        for dx in -1i64..=1 {
            let x = (cx + dx).rem_euclid(per_axis as i64) as u32;
            let y = cy + dy;
            if (0..per_axis as i64).contains(&y) {
                out.push(TileKey::new(level, x, y as u32));
            }
        }
    }
    out
}

/// Geographic bounds of one terrain patch.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PatchBounds {
    /// West edge longitude in degrees.
    pub lon0: f64,
    /// East edge longitude in degrees.
    pub lon1: f64,
    /// South edge latitude in degrees.
    pub lat0: f64,
    /// North edge latitude in degrees.
    pub lat1: f64,
}

/// Appends one relief patch and returns its index range in the shared mesh.
fn push_tile_patch(
    builder: &mut MeshBuilder,
    bounds: PatchBounds,
    grid: u32,
    elevation: Option<&crate::tiles::TileElevation>,
    elevation_tile: TileKey,
) -> (u32, u32) {
    let index_start = builder.index_count() as u32;
    let cols = grid.max(2) as usize;
    let rows = cols;
    let mut positions = Vec::with_capacity(cols * rows);
    for row in 0..rows {
        let v = row as f64 / (rows - 1) as f64;
        for col in 0..cols {
            let u = col as f64 / (cols - 1) as f64;
            let lon = bounds.lon0 + (bounds.lon1 - bounds.lon0) * u;
            let lat = (bounds.lat0 + (bounds.lat1 - bounds.lat0) * v).clamp(-89.9, 89.9);
            let height = elevation
                .map(|grid| {
                    let (tu, tv) = uv_in_tile(elevation_tile, lat, lon);
                    grid.sample_bilinear(tu, tv) as f64
                })
                .unwrap_or(0.0);
            positions.push(geodetic_to_ecef(Geodetic::new(lat, lon, height)));
        }
    }

    // Derive normals from the surface itself so relief catches the light.
    let mut normals = Vec::with_capacity(cols * rows);
    for row in 0..rows {
        for col in 0..cols {
            let index = row * cols + col;
            let radial = positions[index].normalize();
            let east = positions[row * cols + (col + 1).min(cols - 1)]
                .sub(positions[row * cols + col.saturating_sub(1)]);
            // Latitude rises with the row index, so row+1 is north of row-1.
            let north = positions[(row + 1).min(rows - 1) * cols + col]
                .sub(positions[row.saturating_sub(1) * cols + col]);
            let cross = east.cross(north);
            let normal = if cross.length() < 1e-9 {
                radial
            } else {
                let normal = cross.normalize();
                if normal.dot(radial) < 0.0 {
                    normal.scale(-1.0)
                } else {
                    normal
                }
            };
            normals.push(normal.as_f32());
        }
    }

    builder.push_grid_with_normals(&positions, &normals, cols, rows, (0.0, 0.0, 1.0, 1.0));
    let index_count = builder.index_count() as u32 - index_start;
    (index_start, index_count)
}

fn push_placeholder_buildings(
    builder: &mut MeshBuilder,
    origin: DVec3,
    bounds: PatchBounds,
    elevation: Option<&crate::tiles::TileElevation>,
    elevation_tile: TileKey,
) {
    // Deterministic footprints inside the patch; real footprints arrive with
    // the building source. Reporting them keeps the budget honest.
    let samples = 2;
    for row in 0..samples {
        for col in 0..samples {
            let u = (col as f64 + 0.5) / samples as f64;
            let v = (row as f64 + 0.5) / samples as f64;
            let lon = bounds.lon0 + (bounds.lon1 - bounds.lon0) * u;
            let lat = bounds.lat0 + (bounds.lat1 - bounds.lat0) * v;
            let height = elevation
                .map(|grid| {
                    let (tu, tv) = uv_in_tile(elevation_tile, lat, lon);
                    grid.sample_bilinear(tu, tv) as f64
                })
                .unwrap_or(0.0);
            let base = geodetic_to_ecef(Geodetic::new(lat, lon, height)).sub(origin);
            builder.push_building(
                base,
                (40.0, 25.0),
                60.0,
                (0.0, 0.0, 0.5, 0.5),
                (0.5, 0.0, 1.0, 1.0),
            );
        }
    }
}

/// Builds a graticule of meridians and parallels at patch altitude.
///
/// Points are expressed relative to `origin`, the same RTC origin the camera
/// and terrain use. `eye` must be the camera eye in the same relative space.
///
/// Only the hemisphere facing the eye is emitted. A meridian or parallel that
/// crosses the horizon is cut at the crossing and may produce more than one
/// polyline; without this, a far-side line drew straight across the globe.
pub fn push_graticule(overlay: &mut Overlay3D, step_deg: f64, origin: DVec3, eye: DVec3) {
    if step_deg <= 0.0 || step_deg > 90.0 {
        return;
    }
    const SAMPLES: usize = 144;
    let color = Color::rgba(148, 163, 184, 90);
    let altitude = 2_000.0;

    let mut emit = |points: Vec<DVec3>| {
        for run in clip_to_visible_hemisphere(&points, origin, eye) {
            overlay.push(OverlayPrimitive::Polyline {
                points: run,
                color,
                width: 1.0,
            });
        }
    };

    let mut meridian = step_deg;
    while meridian < 180.0 {
        for sign in [-1.0, 1.0] {
            let points = (0..=SAMPLES)
                .map(|i| {
                    let lat = -85.0 + 170.0 * i as f64 / SAMPLES as f64;
                    geodetic_to_ecef(Geodetic::new(lat, meridian * sign, altitude)).sub(origin)
                })
                .collect();
            emit(points);
        }
        meridian += step_deg;
    }

    let mut parallel = 0.0;
    while parallel <= 90.0 - step_deg {
        for sign in [-1.0, 1.0] {
            let lat = parallel * sign;
            let points = (0..=SAMPLES)
                .map(|i| {
                    let lon = 180.0 * i as f64 / SAMPLES as f64;
                    geodetic_to_ecef(Geodetic::new(lat, lon, altitude)).sub(origin)
                })
                .collect();
            emit(points);
        }
        parallel += step_deg;
    }
}

/// Signed visibility of a surface point from the eye.
///
/// Positive means the point is on the eye-facing hemisphere. `q` is relative
/// to `origin`, and `eye` is the eye in the same relative space.
fn hemisphere_visibility(q: DVec3, origin: DVec3, eye: DVec3) -> f64 {
    let absolute = origin.add(q);
    absolute.dot(eye.sub(q))
}

/// Splits a polyline into the runs that are visible from `eye`.
///
/// Crossings are interpolated, so a cut line reaches the horizon instead of
/// stopping a sample short of it.
fn clip_to_visible_hemisphere(points: &[DVec3], origin: DVec3, eye: DVec3) -> Vec<Vec<DVec3>> {
    let mut runs = Vec::new();
    let mut current: Vec<DVec3> = Vec::new();
    let mut previous = points.first().copied();
    if let Some(point) = previous {
        if hemisphere_visibility(point, origin, eye) >= 0.0 {
            current.push(point);
        }
    }
    for &point in points.iter().skip(1) {
        let Some(previous_point) = previous else {
            previous = Some(point);
            continue;
        };
        let a = hemisphere_visibility(previous_point, origin, eye);
        let b = hemisphere_visibility(point, origin, eye);
        if a >= 0.0 && b >= 0.0 {
            current.push(point);
        } else if a >= 0.0 && b < 0.0 {
            let t = (a / (a - b)).clamp(0.0, 1.0);
            current.push(previous_point.lerp(point, t));
            if current.len() >= 2 {
                runs.push(std::mem::take(&mut current));
            } else {
                current.clear();
            }
        } else if a < 0.0 && b >= 0.0 {
            let t = (a / (a - b)).clamp(0.0, 1.0);
            current.push(previous_point.lerp(point, t));
            current.push(point);
        }
        previous = Some(point);
    }
    if current.len() >= 2 {
        runs.push(current);
    }
    runs
}
