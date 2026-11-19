//! Frame assembly: turn an [`EarthInput`] into a renderer-ready scene.

use crate::geodesy::{
    geodetic_to_ecef, resolve_tile_imagery, stabilized_surface_morph, EarthCamera, Geodetic,
    TileKey, TileTextureResolution,
};
pub use crate::geodesy::{tile_bounds, PatchBounds};
use crate::raycast::{raycast_surface_ray, raycast_viewport, EarthRayHit};
use crate::tiles::{TextureSlot, TileRegistry, TileSource};
use crate::world::{EarthObject, EarthObjectId, EarthWorld, ResolvedObjectTransform};
use strata_layout::Rect;
use strata_render::camera64::{Camera64, DVec3, RenderFrame64};
use strata_render::compositor::{DisplayList, DrawCommand};
use strata_render::overlay3d::{LabelAnchor, Overlay3D, OverlayPrimitive, OverlayTextStyle};
use strata_render::rasterizer::FrameBuffer;
use strata_render::scene3d::{
    Background3D, Camera3D, CelestialBody3D, CloudShell3D, Material3D, Mesh3D, Scene3D,
    TexturedBatchMaterial3D, TexturedDraw3D, TexturedMaterial3D, Transform3D, UvTransform,
};
use strata_render::swapchain::{PresentError, SurfacePresenter};
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
            buildings: false,
            graticule: true,
            labels: true,
            clouds: true,
            celestial: true,
        }
    }
}

/// High-level view mode exposed to application callers.
///
/// Inside [`EarthViewMode::Mode3D`], `strata_earth` seamlessly blends from curved
/// WGS84 globe geometry down to a flat local 2.5D tangent plane when approaching
/// the ground; inspectable via the low-level attribute ([`EarthLowLevelState`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EarthViewMode {
    /// 3D WGS84 globe with automatic seamless 2.5D flattening near ground level.
    #[default]
    Mode3D,
    /// Flat 2D map plane (programmatically morphed smoothly from/to 3D).
    Mode2D,
}

/// Low-level internal surface state inside 3D/2D rendering.
///
/// Exposed for diagnostics or explicit low-level override via [`EarthInput::low_level_surface_override`].
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct EarthLowLevelState {
    pub active_surface_mode: SurfaceMode,
    /// Seamless 3D-to-2.5D ground-proximity flatness blend (`0.0` = curved 3D, `1.0` = local 2.5D plane).
    pub local_25d_blend: f32,
    /// Smooth 3D-to-2D map morph progress (`0.0` = 3D/2.5D, `1.0` = flattened 2D map).
    pub map_2d_morph: f32,
}

/// Selectable astronomical or custom solar lighting source.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SunDirectionMode {
    /// Dynamically computes solar ephemeris & sidereal rotation using OS wall-clock UTC and timezone offset.
    SystemClock,
    /// Computes astronomical solar position from [`EarthInput::sim_time_unix_s`].
    SimulationTime,
    /// Computes solar position from an explicit local timestamp and UTC offset (ready for time handlers).
    LocalTimeOffset {
        local_unix_s: f64,
        utc_offset_s: i32,
    },
    /// Places noon directly overhead a chosen geodetic `(lat_deg, lon_deg)`.
    SubsolarPoint { lat_deg: f64, lon_deg: f64 },
    /// Explicit normalized ECEF sunlight vector `[x, y, z]`.
    CustomEcef([f32; 3]),
}

/// Toggleable and customizable planetary lighting, space skybox, Sun/Moon, and land-masked city lights.
#[derive(Debug, Clone, PartialEq)]
pub struct SpaceEnvironmentConfig {
    /// Enables physical day/night terminator shadows (`true` by default; auto-fades to shadowless daylight in 2.5D).
    pub day_night_lighting: bool,
    /// Day/night shadow contrast `[0.0..=1.0]` (`1.0` default).
    pub shadow_contrast: f32,
    /// Enables deterministic night city lights masked strictly to landmass (`true` default).
    pub city_lights: bool,
    /// Brightness multiplier for deterministic city lights (`1.0` default).
    pub city_lights_intensity: f32,
    /// Solar position & lighting source mode (defaults to [`SunDirectionMode::SystemClock`]).
    pub sun_direction: SunDirectionMode,
    /// Render physical limb-darkened Sun disc & solar corona (`true` default).
    pub sun_visible: bool,
    /// Apparent Sun angular radius in radians (default `0.018`).
    pub sun_angular_radius_rad: f32,
    /// Render NASA LRO textured Moon with accurate orbital lunar phase (`true` default).
    pub moon_visible: bool,
    /// Apparent Moon angular radius in radians (default `0.022`).
    pub moon_angular_radius_rad: f32,
    /// Optional custom ECEF direction for Moon; `None` computes astronomical lunar ephemeris.
    pub custom_moon_dir: Option<[f32; 3]>,
    /// Render pitch-black deep-space skybox with Yale BSC5 scientific stars & Milky Way (`true` default).
    pub stars_visible: bool,
    /// Starfield & Milky Way brightness multiplier (`1.0` default).
    pub stars_intensity: f32,
    /// Real telescope skybox survey preset (`CompositeOptical` default).
    pub skybox_survey: crate::skybox::SpaceSkyboxSurvey,
    /// Programmatic skybox cubemap resolution (`High` `2048^2` default; `Ultra` `4096^2` for 4K).
    pub skybox_quality: crate::skybox::SpaceSkyboxQuality,
    /// Brightness multiplier for real telescope skybox cubemap (`1.0` default).
    pub skybox_intensity: f32,
    /// Remote CDN / GitHub rolling-release (`assets` & `assets-legacy`) asset configuration.
    pub assets: crate::assets_cdn::AssetSourceConfig,
    /// Render live / propagated real orbital satellites (ISS, Hubble, NOAA, GPS, etc.) in 3D space (fades out in 2.5D).
    pub satellites_visible: bool,
    /// Render orbital ground-track arcs for flagship spacecraft in 3D space.
    pub satellite_orbits_visible: bool,
    /// Base radius scale in metres for 3D spacecraft beacons from orbital distance.
    pub satellite_scale_m: f32,
}

impl Default for SpaceEnvironmentConfig {
    fn default() -> Self {
        Self {
            day_night_lighting: true,
            shadow_contrast: 1.0,
            city_lights: true,
            city_lights_intensity: 1.0,
            sun_direction: SunDirectionMode::SimulationTime,
            sun_visible: true,
            sun_angular_radius_rad: 0.018,
            moon_visible: true,
            moon_angular_radius_rad: 0.022,
            custom_moon_dir: None,
            stars_visible: true,
            stars_intensity: 1.0,
            skybox_survey: crate::skybox::SpaceSkyboxSurvey::default(),
            skybox_quality: crate::skybox::SpaceSkyboxQuality::default(),
            skybox_intensity: 1.0,
            assets: crate::assets_cdn::AssetSourceConfig::default(),
            satellites_visible: true,
            satellite_orbits_visible: true,
            satellite_scale_m: 38_000.0,
        }
    }
}

impl SpaceEnvironmentConfig {
    /// Sets the telescope skybox survey and programmatic resolution tier.
    pub fn with_skybox(
        mut self,
        survey: crate::skybox::SpaceSkyboxSurvey,
        quality: crate::skybox::SpaceSkyboxQuality,
    ) -> Self {
        self.skybox_survey = survey;
        self.skybox_quality = quality;
        self
    }

    /// Overrides the remote CDN base URL for downloadable assets (e.g. `https://cdn.strata.larvance.com/earth`).
    pub fn with_asset_cdn(mut self, cdn_base: Option<impl Into<String>>) -> Self {
        self.assets = self.assets.with_cdn_base(cdn_base);
        self
    }

    /// Uses the host OS wall clock and local UTC offset to compute solar/lunar direction.
    pub fn with_system_clock(mut self) -> Self {
        self.sun_direction = SunDirectionMode::SystemClock;
        self
    }

    /// Pre-ready helper to wire local timestamp and UTC offset directly from a time handler.
    pub fn with_local_time(mut self, local_unix_s: f64, utc_offset_s: i32) -> Self {
        self.sun_direction = SunDirectionMode::LocalTimeOffset {
            local_unix_s,
            utc_offset_s,
        };
        self
    }

    /// Pre-ready helper to wire UTC timestamp and OS/explicit UTC offset directly from a time handler.
    pub fn with_utc_and_offset(mut self, unix_utc_s: f64, utc_offset_s: i32) -> Self {
        self.sun_direction = SunDirectionMode::LocalTimeOffset {
            local_unix_s: unix_utc_s + utc_offset_s as f64,
            utc_offset_s,
        };
        self
    }
}

/// Optional animated wavy ocean rendering on non-land water surfaces.
#[derive(Debug, Clone, PartialEq)]
pub struct OceanWavesConfig {
    /// Enables animated multi-octave Gerstner waves and sun-glint crests on non-land surfaces (`false` default).
    pub enabled: bool,
    /// Wave height / slope amplitude multiplier (`1.0` default).
    pub wave_strength: f32,
    /// Spatial wave frequency multiplier (`1.0` default; higher = tighter waves).
    pub wave_scale: f32,
    /// Specular sun-glint & whitecap crest highlight intensity (`1.0` default).
    pub glint_intensity: f32,
    /// Animation speed multiplier (`1.0` default).
    pub speed: f32,
    /// Translucent wave surface opacity over underlying bathymetry/satellite water (`0.36` default).
    pub opacity: f32,
}

impl Default for OceanWavesConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            wave_strength: 1.0,
            wave_scale: 1.0,
            glint_intensity: 1.0,
            speed: 1.0,
            opacity: 0.36,
        }
    }
}

impl OceanWavesConfig {
    /// Returns a preset with wavy ocean rendering enabled at nominal strength.
    pub fn enabled() -> Self {
        Self {
            enabled: true,
            ..Self::default()
        }
    }

    pub fn with_enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }

    pub fn with_strength(mut self, strength: f32) -> Self {
        self.wave_strength = strength.max(0.0);
        self
    }

    pub fn with_scale(mut self, scale: f32) -> Self {
        self.wave_scale = scale.max(0.1);
        self
    }

    pub fn with_speed(mut self, speed: f32) -> Self {
        self.speed = speed.max(0.0);
        self
    }

    pub fn with_glint(mut self, glint: f32) -> Self {
        self.glint_intensity = glint.max(0.0);
        self
    }

    pub fn with_opacity(mut self, opacity: f32) -> Self {
        self.opacity = opacity.clamp(0.0, 1.0);
        self
    }
}

/// Low-level surface geometry mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SurfaceMode {
    /// 3x3 planar terrain patches tangential to the camera focus point (default).
    #[default]
    FlatPatches,
    /// True spherical Earth globe geometry at WGS84 radius.
    SphericalGlobe,
    /// Continuous hierarchical WGS84 ellipsoid quadtree tile LOD pyramid with horizon culling and SSE subdivision.
    QuadtreeLod,
}

/// Everything a frame needs.
#[derive(Debug, Clone, PartialEq)]
pub struct EarthInput {
    pub camera: EarthCamera,
    /// Optional explicit ECEF camera-up vector for gimbal-free 3D arcball/trackball orientation.
    pub camera_up_ecef: Option<[f64; 3]>,
    pub viewport: Rect,
    /// High-level view mode (`Mode3D` or `Mode2D`).
    pub view_mode: EarthViewMode,
    /// Programmatic transition morph factor (`0.0` = 3D, `1.0` = 2D map).
    pub view_morph: f32,
    /// Enables seamless transition between curved 3D globe and flat 2.5D tangent plane near ground.
    pub seamless_25d: bool,
    /// Tile latitude projection (`WebMercator` default for standard Slippy/XYZ tiles).
    pub tile_projection: crate::geodesy::TileProjection,
    /// Simulation time in Unix seconds, used for sun, moon, and sidereal rotation.
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
    /// Geometry mode for surface rendering.
    pub surface_mode: SurfaceMode,
    /// Low-level attribute allowing explicit internal surface regime override/inspection.
    pub low_level_surface_override: Option<SurfaceMode>,
    /// Target Screen-Space Error threshold in pixels for quadtree subdivision.
    pub sse_threshold: f64,
    /// Whether to generate adaptive terrain skirts around tile borders to eliminate LOD cracks.
    ///
    /// When `None` (the default), skirts are automatically generated for multi-resolution
    /// LOD modes such as [`SurfaceMode::QuadtreeLod`]. When `Some(true)` or `Some(false)`,
    /// skirt generation is explicitly forced on or off for all surface modes.
    pub terrain_skirts: Option<bool>,
    /// Tint applied to terrain patches that have no imagery loaded. Neutral
    /// mid-gray by default so hosts can theme it.
    pub fallback_tint: Color,
    /// Enable heightmap (DEM) fetching and terrain elevation displacement (`true` default).
    pub elevation_enabled: bool,
    /// Toggleable space skybox, Sun/Moon, day/night terminator, satellites, and city lights.
    pub space: SpaceEnvironmentConfig,
    /// Toggleable global and local 2.5D cloud-shell map configuration.
    pub weather: crate::gltf::WeatherConfig,
    /// Physically accurate, GPU-instanced 2D camera-local precipitation configuration.
    pub precipitation: crate::precipitation::PrecipitationConfig,
    /// Optional animated wavy ocean renders on non-land water surfaces.
    pub ocean_waves: OceanWavesConfig,
}

impl EarthInput {
    /// Returns the effective internal [`SurfaceMode`] respecting [`Self::low_level_surface_override`].
    #[inline]
    pub fn effective_surface_mode(&self) -> SurfaceMode {
        self.low_level_surface_override.unwrap_or(self.surface_mode)
    }

    /// Resolves `(sun_dir_ecef, moon_dir_ecef, unix_utc_s)` for the current configuration.
    pub fn resolved_celestial_state(&self) -> ([f32; 3], [f32; 3], f64) {
        let (sun_dir, unix_utc_s) = match self.space.sun_direction {
            SunDirectionMode::SystemClock => {
                let clock = crate::astro::system_clock_time();
                (
                    crate::astro::sun_direction_from_local_time(
                        clock.local_unix_s,
                        clock.utc_offset_s,
                    ),
                    clock.unix_utc_s,
                )
            }
            SunDirectionMode::SimulationTime => (
                crate::astro::sun_direction_from_unix_utc(self.sim_time_unix_s),
                self.sim_time_unix_s,
            ),
            SunDirectionMode::LocalTimeOffset {
                local_unix_s,
                utc_offset_s,
            } => (
                crate::astro::sun_direction_from_local_time(local_unix_s, utc_offset_s),
                local_unix_s - utc_offset_s as f64,
            ),
            SunDirectionMode::SubsolarPoint { lat_deg, lon_deg } => (
                crate::astro::sun_direction_from_subsolar_lat_lon(lat_deg, lon_deg),
                self.sim_time_unix_s,
            ),
            SunDirectionMode::CustomEcef(v) => {
                let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
                let dir = if len > 1e-6 {
                    [v[0] / len, v[1] / len, v[2] / len]
                } else {
                    [1.0, 0.0, 0.0]
                };
                (dir, self.sim_time_unix_s)
            }
        };
        let moon_dir = match self.space.custom_moon_dir {
            Some(v) => {
                let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
                if len > 1e-6 {
                    [v[0] / len, v[1] / len, v[2] / len]
                } else {
                    crate::astro::moon_direction_from_unix_utc(unix_utc_s)
                }
            }
            None => crate::astro::moon_direction_from_unix_utc(unix_utc_s),
        };
        (sun_dir, moon_dir, unix_utc_s)
    }

    /// Computes the current low-level internal surface state (3D vs seamless 2.5D vs 2D morph).
    pub fn low_level_state(&self) -> EarthLowLevelState {
        let mode = self.effective_surface_mode();
        let map_2d_morph = if self.view_morph > 0.0 {
            self.view_morph.clamp(0.0, 1.0)
        } else {
            match self.view_mode {
                EarthViewMode::Mode3D => 0.0,
                EarthViewMode::Mode2D => 1.0,
            }
        };
        let local_25d_blend = if !self.seamless_25d {
            if mode == SurfaceMode::FlatPatches {
                1.0
            } else {
                0.0
            }
        } else {
            crate::geodesy::compute_seamless_25d_blend(self.camera.altitude_m())
        };
        EarthLowLevelState {
            active_surface_mode: mode,
            local_25d_blend,
            map_2d_morph,
        }
    }
}

impl Default for EarthInput {
    fn default() -> Self {
        Self {
            camera: EarthCamera::default(),
            camera_up_ecef: None,
            viewport: Rect::new(0.0, 0.0, 1440.0, 900.0),
            view_mode: EarthViewMode::Mode3D,
            view_morph: 0.0,
            seamless_25d: true,
            tile_projection: crate::geodesy::TileProjection::WebMercator,
            sim_time_unix_s: 0.0,
            cloud_cover: 0.0,
            daylight: 1.0,
            layers: LayerToggles::default(),
            graticule_step_deg: 30.0,
            max_tile_level: 18,
            tile_grid: 5,
            patch_scale: 0.25,
            surface_mode: SurfaceMode::default(),
            low_level_surface_override: None,
            sse_threshold: 4.0,
            terrain_skirts: None,
            fallback_tint: Color::rgba(128, 128, 128, 255),
            elevation_enabled: true,
            space: SpaceEnvironmentConfig::default(),
            weather: crate::gltf::WeatherConfig::default(),
            precipitation: crate::precipitation::PrecipitationConfig::default(),
            ocean_waves: OceanWavesConfig::default(),
        }
    }
}

impl EarthInput {
    /// Returns a full-globe spherical LOD preset (`SurfaceMode::QuadtreeLod`) suitable for
    /// interactive 3D/2.5D/2D `<EarthView>` components across planetary and local scales.
    pub fn globe() -> Self {
        Self {
            surface_mode: SurfaceMode::QuadtreeLod,
            tile_grid: 17,
            ..Self::default()
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

#[derive(Debug, Clone, PartialEq)]
struct RetainedLeafSig {
    mode: SurfaceMode,
    tile_grid: u32,
    max_tile_level: u8,
    sse_threshold: f64,
    with_skirts: bool,
    tile_projection: crate::geodesy::TileProjection,
    flat_q: u32,
    morph_q: u32,
    focus_lat_q: i64,
    focus_lon_q: i64,
    world_rev: u64,
    elevation_enabled: bool,
    leaves: Vec<TileKey>,
}

/// The planetary engine's reusable frame and geodetic [`EarthWorld`] builder.
#[derive(Debug)]
pub struct EarthScene {
    scene: Scene3D,
    world: std::sync::Arc<std::sync::RwLock<EarthWorld>>,
    terrain: MeshBuilder,
    buildings: MeshBuilder,
    tiles: TileRegistry,
    stats: FrameStats,
    patches: Vec<PatchFrame>,
    frame_width: f32,
    frame_height: f32,
    rtc_origin: DVec3,
    building_source: Option<Box<dyn BuildingSource>>,
    retained_builder: crate::worker::RetainedTerrainBuilder,
    retained_handle: strata_render::scene3d::RetainedScene3D,
    retained_geom_rev: u64,
    last_retained_stats: crate::worker::RetainedBuildStats,
    last_retained_leaves: Option<RetainedLeafSig>,
}

impl Default for EarthScene {
    fn default() -> Self {
        Self::new()
    }
}

impl EarthScene {
    pub fn new() -> Self {
        let world = std::sync::Arc::new(std::sync::RwLock::new(EarthWorld::new()));
        let mut retained_builder = crate::worker::RetainedTerrainBuilder::new();
        retained_builder.set_world(std::sync::Arc::clone(&world));
        Self {
            scene: Scene3D::new(),
            world,
            terrain: MeshBuilder::new(),
            buildings: MeshBuilder::new(),
            tiles: TileRegistry::new(),
            stats: FrameStats::default(),
            patches: Vec::new(),
            frame_width: 1440.0,
            frame_height: 900.0,
            rtc_origin: geodetic_to_ecef(Geodetic::default()),
            building_source: None,
            retained_builder,
            retained_handle: strata_render::scene3d::RetainedScene3D::new(
                strata_render::scene3d::RetainedCameraUniforms::default(),
            ),
            retained_geom_rev: 0,
            last_retained_stats: crate::worker::RetainedBuildStats::default(),
            last_retained_leaves: None,
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

    pub(crate) fn internal_scene(&self) -> &Scene3D {
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
        // Release the previous scene's shared texture handle so uploads below
        // mutate the registry in place instead of cloning the whole store.
        self.scene.set_textures(None);
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
            let effective_mode = input.effective_surface_mode();
            let with_skirts = input
                .terrain_skirts
                .unwrap_or(effective_mode == SurfaceMode::QuadtreeLod);

            fn resolve_non_flooding(
                requested: TileKey,
                src: &mut dyn TileSource,
            ) -> Option<TileTextureResolution> {
                if matches!(src.imagery(requested), Ok(Some(_))) {
                    Some(TileTextureResolution {
                        source: requested,
                        exact: true,
                        uv_rect: (0.0, 0.0, 1.0, 1.0),
                    })
                } else {
                    resolve_tile_imagery(requested, |candidate| {
                        src.peek_cached_imagery(candidate).is_some()
                    })
                }
            }

            fn resolve_elevation_non_flooding(
                requested: TileKey,
                src: &mut dyn TileSource,
            ) -> (Option<crate::tiles::TileElevation>, TileKey) {
                if let Ok(Some(elev)) = src.elevation(requested) {
                    return (Some(elev), requested);
                }
                let mut probe = requested;
                if requested.level > 15 {
                    let mut p15 = requested;
                    while p15.level > 15 {
                        p15 = match p15.parent() {
                            Some(p) => p,
                            None => break,
                        };
                    }
                    probe = p15;
                    if let Ok(Some(elev)) = src.elevation(p15) {
                        return (Some(elev), p15);
                    }
                }
                let mut anc = probe.parent();
                while let Some(a) = anc {
                    if let Some(elev) = src.peek_cached_elevation(a) {
                        return (Some(elev), a);
                    }
                    anc = a.parent();
                }
                (None, requested)
            }

            match effective_mode {
                SurfaceMode::FlatPatches => {
                    let span_deg = tile_patch_span_deg(&input.camera, input.patch_scale);
                    let bounds = surrounding_patches(input.camera.focus, span_deg);
                    for mut patch in bounds {
                        stats.tiles_requested += 1;
                        let requested = patch.requested;
                        let resolution = resolve_non_flooding(requested, source);
                        if let Some(resolution) = resolution {
                            if resolution.exact {
                                stats.tiles_exact += 1;
                            } else {
                                stats.tiles_from_ancestor += 1;
                            }
                        }

                        let (elevation, elev_tile) =
                            resolve_elevation_non_flooding(requested, source);
                        let (index_start, index_count) = push_tile_patch(
                            &mut self.terrain,
                            patch.bounds,
                            input.tile_grid.max(2),
                            elevation.as_ref(),
                            elev_tile,
                            with_skirts,
                        );
                        patch.resolution = resolution;
                        patch.index_start = index_start;
                        patch.index_count = index_count;

                        if input.layers.buildings {
                            push_footprints(
                                &mut self.buildings,
                                self.building_source.as_mut(),
                                origin,
                                patch.bounds,
                                elevation.as_ref(),
                                elev_tile,
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
                            let bounds = if input.tile_projection
                                == crate::geodesy::TileProjection::WebMercator
                            {
                                crate::geodesy::tile_bounds_mercator(key)
                            } else {
                                PatchBounds {
                                    lon0,
                                    lon1,
                                    lat0,
                                    lat1,
                                }
                            };
                            stats.tiles_requested += 1;
                            let requested = key;
                            let resolution = resolve_non_flooding(requested, source);
                            if let Some(resolution) = resolution {
                                if resolution.exact {
                                    stats.tiles_exact += 1;
                                } else {
                                    stats.tiles_from_ancestor += 1;
                                }
                            }

                            let (elevation, elev_tile) =
                                resolve_elevation_non_flooding(requested, source);
                            let (index_start, index_count) = push_tile_patch(
                                &mut self.terrain,
                                bounds,
                                input.tile_grid.max(2),
                                elevation.as_ref(),
                                elev_tile,
                                with_skirts,
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
                SurfaceMode::QuadtreeLod => {
                    let mut leaves = Vec::new();
                    let horizon_cull = input.low_level_state().map_2d_morph < 0.15;
                    let raw_leaves = crate::geodesy::traverse_quadtree_lod_frustum6_with_options(
                        &input.camera,
                        input.viewport.width.max(1.0) as f64,
                        input.viewport.height.max(1.0) as f64,
                        input.sse_threshold.max(0.5),
                        input.max_tile_level,
                        input.tile_grid.max(2),
                        horizon_cull,
                        crate::geodesy::MAX_QUADTREE_LEAVES,
                    );
                    for key in raw_leaves {
                        let bounds = if input.tile_projection
                            == crate::geodesy::TileProjection::WebMercator
                        {
                            crate::geodesy::tile_bounds_mercator(key)
                        } else {
                            tile_bounds(key)
                        };
                        leaves.push((key, bounds));
                    }

                    for (key, bounds) in leaves {
                        stats.tiles_requested += 1;
                        let requested = key;
                        let resolution = resolve_non_flooding(requested, source);
                        if let Some(res) = resolution {
                            if res.exact {
                                stats.tiles_exact += 1;
                            } else {
                                stats.tiles_from_ancestor += 1;
                            }
                        }

                        let (elevation, elev_tile) =
                            resolve_elevation_non_flooding(requested, source);
                        let (index_start, index_count) = push_tile_patch(
                            &mut self.terrain,
                            bounds,
                            input.tile_grid.max(2),
                            elevation.as_ref(),
                            elev_tile,
                            with_skirts,
                        );

                        if input.layers.buildings {
                            push_footprints(
                                &mut self.buildings,
                                self.building_source.as_mut(),
                                origin,
                                bounds,
                                elevation.as_ref(),
                                requested,
                            );
                        }

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
                        None => (fallback, input.fallback_tint),
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
                        TexturedMaterial3D::new(fallback).with_tint(input.fallback_tint),
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

        // 5. Retained GPU frame + geodetic EarthWorld compilation.
        self.retained_geom_rev += 1;
        let (retained_frame, r_stats) =
            self.retained_builder
                .build_frame(input, source, self.retained_geom_rev, false);
        self.last_retained_stats = r_stats;
        self.retained_handle
            .update_terrain(std::sync::Arc::clone(&retained_frame));

        let low_state = input.low_level_state();
        let flat_t = low_state.local_25d_blend.max(low_state.map_2d_morph);
        let morph = stabilized_surface_morph(
            input.camera.focus,
            flat_t as f64,
            low_state.map_2d_morph as f64,
            input.tile_projection,
        );
        let elev_enabled = input.elevation_enabled;
        let rb_ref = &self.retained_builder;
        let dem_sampler = |lat: f64, lon: f64| -> f64 {
            if elev_enabled {
                rb_ref.sample_ground_elevation_m(lat, lon)
            } else {
                0.0
            }
        };

        let mut overlay = Overlay3D::new();
        if input.layers.graticule {
            push_graticule(
                &mut overlay,
                input.graticule_step_deg,
                origin,
                placement.eye.sub(origin),
            );
        }

        if let Ok(mut world_guard) = self.world.write() {
            if !world_guard.is_empty() {
                let cpu_meshes = world_guard.emit_cpu_meshes(
                    origin,
                    placement.eye,
                    input.viewport.height.max(1.0) as f64,
                    input.camera.fov_y_deg,
                    elev_enabled,
                    morph,
                    &dem_sampler,
                );
                for (geom, mat) in cpu_meshes {
                    let m3d = if mat.unlit {
                        Material3D::Basic { color: mat.color }
                    } else {
                        Material3D::Standard {
                            color: mat.color,
                            roughness: mat.roughness,
                            metalness: mat.metalness,
                            emissive: (mat.emissive.a() > 0.001).then_some(mat.emissive),
                        }
                    };
                    scene.meshes.push(Mesh3D::new(geom, m3d));
                    stats.draw_calls += 1;
                }

                if input.layers.labels {
                    for (id, obj) in world_guard.iter() {
                        if !obj.visible {
                            continue;
                        }
                        if let Some(ref text) = obj.label {
                            if !text.is_empty() {
                                if let Some(tf) = world_guard.resolve_object_transform(
                                    id,
                                    placement.eye,
                                    input.viewport.height.max(1.0) as f64,
                                    input.camera.fov_y_deg,
                                    elev_enabled,
                                    morph,
                                    &dem_sampler,
                                ) {
                                    let anchor = DVec3::new(
                                        tf.anchor_ecef[0],
                                        tf.anchor_ecef[1],
                                        tf.anchor_ecef[2],
                                    )
                                    .sub(origin);
                                    overlay.push(OverlayPrimitive::Label {
                                        anchor,
                                        text: text.clone(),
                                        style: OverlayTextStyle {
                                            color: obj.material.color,
                                            scale: 1,
                                        },
                                        placement: LabelAnchor::Above,
                                        leader: true,
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }

        stats.overlay_primitives = overlay.primitives.len();
        if !overlay.is_empty() {
            scene = scene.with_overlay(overlay);
        }

        scene.set_retained_terrain(Some(retained_frame));
        scene.set_retained_handle(Some(self.retained_handle.clone()));

        self.frame_width = input.viewport.width;
        self.frame_height = input.viewport.height;
        scene.set_textures(Some(self.tiles.store_arc()));
        self.rtc_origin = origin;
        self.scene = scene;
        self.stats = stats;
    }

    /// Returns the persistent [`strata_render::scene3d::RetainedScene3D`] handle owned by this scene.
    pub fn retained_scene(&self) -> strata_render::scene3d::RetainedScene3D {
        self.retained_handle.clone()
    }

    /// Returns cache hit/miss statistics from the latest retained terrain frame build.
    pub fn retained_stats(&self) -> crate::worker::RetainedBuildStats {
        self.last_retained_stats
    }

    /// GPU-first retained build that updates camera uniforms in O(1) when visible LOD leaves
    /// and world revision remain identical, and skips CPU [`MeshBuilder`] construction altogether.
    pub fn build_retained(&mut self, input: &EarthInput, source: &mut dyn TileSource) {
        let cam_u = crate::worker::RetainedTerrainBuilder::camera_from_input(input);
        let effective_mode = input.effective_surface_mode();
        let low_state = input.low_level_state();
        let flat_q = (low_state.local_25d_blend * 1024.0).round() as u32;
        let morph_q = (low_state.map_2d_morph * 1024.0).round() as u32;
        let (focus_lat_q, focus_lon_q) = if flat_q > 0 {
            (
                (input.camera.focus.lat_deg * 1e5).round() as i64,
                (input.camera.focus.lon_deg * 1e5).round() as i64,
            )
        } else {
            (0, 0)
        };
        let world_rev = self.world.read().map(|w| w.revision()).unwrap_or(0);
        let has_world = self.world.read().map(|w| !w.is_empty()).unwrap_or(false);
        if input.layers.terrain && effective_mode == SurfaceMode::QuadtreeLod {
            let leaves = crate::geodesy::traverse_quadtree_lod_frustum6_projected(
                &input.camera,
                input.viewport.width.max(1.0) as f64,
                input.viewport.height.max(1.0) as f64,
                input.sse_threshold,
                input.max_tile_level,
                input.tile_grid.max(2),
                low_state.map_2d_morph < 0.15,
                crate::geodesy::MAX_QUADTREE_LEAVES,
                input.tile_projection,
            );
            let with_skirts = input.terrain_skirts.unwrap_or(true);
            let sig = RetainedLeafSig {
                mode: effective_mode,
                tile_grid: input.tile_grid.max(2),
                max_tile_level: input.max_tile_level,
                sse_threshold: input.sse_threshold,
                with_skirts,
                tile_projection: input.tile_projection,
                flat_q,
                morph_q,
                focus_lat_q,
                focus_lon_q,
                world_rev,
                elevation_enabled: input.elevation_enabled,
                leaves,
            };
            if !has_world
                && self.last_retained_leaves.as_ref() == Some(&sig)
                && self.last_retained_stats.ancestor_fallbacks == 0
                && self.retained_builder.frame_cache_valid_for_fast_path(input)
                && self.retained_handle.snapshot_terrain().is_some()
            {
                self.retained_handle.update_camera(cam_u);
                self.last_retained_stats = crate::worker::RetainedBuildStats {
                    visible_tiles: sig.leaves.len(),
                    mesh_cache_hits: sig.leaves.len(),
                    mesh_cache_misses: 0,
                    texture_cache_hits: sig.leaves.len(),
                    texture_cache_misses: 0,
                    ancestor_fallbacks: 0,
                };
                self.scene
                    .set_retained_handle(Some(self.retained_handle.clone()));
                return;
            }
            self.last_retained_leaves = Some(sig);
        } else {
            self.last_retained_leaves = None;
        }

        self.retained_geom_rev += 1;
        let (frame, r_stats) =
            self.retained_builder
                .build_frame(input, source, self.retained_geom_rev, true);
        self.last_retained_stats = r_stats;
        self.retained_handle
            .update_terrain(std::sync::Arc::clone(&frame));
        let mut scene = Scene3D::new()
            .with_background(sky_gradient(input.daylight))
            .with_retained_terrain(frame)
            .with_retained_handle(self.retained_handle.clone());
        let placement = input.camera.placement();
        let origin = geodetic_to_ecef(input.camera.focus);
        let aspect = (input.viewport.width / input.viewport.height.max(1.0)).max(0.01);
        scene.camera = Camera3D::Perspective {
            fov_horizontal_deg: horizontal_fov_deg(placement.fov_y_deg, aspect as f64) as f32,
            near: placement.near_m as f32,
            far: placement.far_m as f32,
            position: placement.eye.sub(origin).as_f32(),
            target: placement.aim.sub(origin).as_f32(),
        };
        self.frame_width = input.viewport.width;
        self.frame_height = input.viewport.height;
        self.rtc_origin = origin;
        self.scene = scene;
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

    /// Registers the building footprint provider. Without one, no buildings are
    /// generated even when `layers.buildings` is on.
    pub fn set_building_source(&mut self, source: Option<Box<dyn BuildingSource>>) {
        self.building_source = source;
    }

    /// Borrows the geodetic [`EarthWorld`] owned by this scene.
    pub fn world<R>(&self, f: impl FnOnce(&EarthWorld) -> R) -> R {
        let guard = self.world.read().expect("EarthWorld lock poisoned");
        f(&guard)
    }

    /// Mutates the geodetic [`EarthWorld`] owned by this scene.
    pub fn world_mut<R>(&mut self, f: impl FnOnce(&mut EarthWorld) -> R) -> R {
        let mut guard = self.world.write().expect("EarthWorld lock poisoned");
        f(&mut guard)
    }

    /// Spawns a geodetic [`EarthObject`] into this scene's [`EarthWorld`].
    pub fn spawn_object(&mut self, object: EarthObject) -> EarthObjectId {
        self.world_mut(|w| w.spawn(object))
    }

    /// Removes a geodetic [`EarthObject`] by ID, returning `true` if it existed.
    pub fn remove_object(&mut self, id: EarthObjectId) -> bool {
        self.world_mut(|w| w.remove(id))
    }

    /// Clears all geodetic [`EarthObject`]s from this scene's [`EarthWorld`].
    pub fn clear_objects(&mut self) {
        self.world_mut(|w| w.clear());
    }

    /// Evaluates the world transform of `id` (including live `Terrain` vs `SeaLevel` altitude)
    /// against `input` and the most recently loaded DEM elevation snapshot.
    pub fn resolve_object_transform(
        &self,
        id: EarthObjectId,
        input: &EarthInput,
    ) -> Option<ResolvedObjectTransform> {
        let low = input.low_level_state();
        let flat_t = low.local_25d_blend.max(low.map_2d_morph);
        let morph = stabilized_surface_morph(
            input.camera.focus,
            flat_t as f64,
            low.map_2d_morph as f64,
            input.tile_projection,
        );
        let up = input.camera_up_ecef.map(|[x, y, z]| DVec3::new(x, y, z));
        let placement = input.camera.placement_with_up(up);
        let elev_on = input.elevation_enabled;
        let rb = &self.retained_builder;
        let guard = self.world.read().ok()?;
        guard.resolve_object_transform(
            id,
            placement.eye,
            input.viewport.height.max(1.0) as f64,
            input.camera.fov_y_deg,
            elev_on,
            morph,
            &|lat, lon| {
                if elev_on {
                    rb.sample_ground_elevation_m(lat, lon)
                } else {
                    0.0
                }
            },
        )
    }

    /// Raycasts `(screen_x, screen_y)` against both pickable [`EarthWorld`] objects and the
    /// planetary surface using the DEM snapshot from the latest build.
    pub fn raycast(&self, screen_x: f64, screen_y: f64, input: &EarthInput) -> Option<EarthRayHit> {
        let low = input.low_level_state();
        let flat_t = low.local_25d_blend.max(low.map_2d_morph);
        let morph = stabilized_surface_morph(
            input.camera.focus,
            flat_t as f64,
            low.map_2d_morph as f64,
            input.tile_projection,
        );
        let up = input.camera_up_ecef.map(|[x, y, z]| DVec3::new(x, y, z));
        let elev_on = input.elevation_enabled;
        let rb = &self.retained_builder;
        let guard = self.world.read().ok()?;
        raycast_viewport(
            screen_x,
            screen_y,
            input.viewport.width as f64,
            input.viewport.height as f64,
            &input.camera,
            up,
            elev_on,
            morph,
            Some(&guard),
            &|lat, lon| {
                if elev_on {
                    rb.sample_ground_elevation_m(lat, lon)
                } else {
                    0.0
                }
            },
        )
    }

    /// Raycasts `(screen_x, screen_y)` strictly against the planetary surface.
    pub fn raycast_terrain(
        &self,
        screen_x: f64,
        screen_y: f64,
        input: &EarthInput,
    ) -> Option<EarthRayHit> {
        let low = input.low_level_state();
        let flat_t = low.local_25d_blend.max(low.map_2d_morph);
        let morph = stabilized_surface_morph(
            input.camera.focus,
            flat_t as f64,
            low.map_2d_morph as f64,
            input.tile_projection,
        );
        let up = input.camera_up_ecef.map(|[x, y, z]| DVec3::new(x, y, z));
        let (ray_origin, ray_dir) = crate::geodesy::screen_to_ray_with_up(
            screen_x,
            screen_y,
            input.viewport.width as f64,
            input.viewport.height as f64,
            &input.camera,
            up,
        );
        let elev_on = input.elevation_enabled;
        let rb = &self.retained_builder;
        raycast_surface_ray(ray_origin, ray_dir, elev_on, morph, &|lat, lon| {
            if elev_on {
                rb.sample_ground_elevation_m(lat, lon)
            } else {
                0.0
            }
        })
    }

    /// Converts the current 3D scene into a compositor `DrawViewport3D` command.
    pub fn to_draw_command(&self, bounds: Rect) -> DrawCommand {
        DrawCommand::DrawViewport3D {
            bounds,
            scene: std::sync::Arc::new(self.scene.clone()),
        }
    }

    /// Appends the viewport draw command for the current scene to a display list.
    pub fn append_to_display_list(&self, list: &mut DisplayList, bounds: Rect) {
        list.push(self.to_draw_command(bounds));
    }

    /// Software-rasterizes the current 3D scene into an offscreen RGBA framebuffer.
    pub fn render_to_framebuffer(&self, width: u32, height: u32) -> FrameBuffer {
        let width = width.max(1);
        let height = height.max(1);
        let mut fb = FrameBuffer::new(width as usize, height as usize);
        let mut dl = DisplayList::new();
        self.append_to_display_list(&mut dl, Rect::new(0.0, 0.0, width as f32, height as f32));
        fb.execute_display_list(&dl);
        fb
    }

    /// Presents the rendered scene to an abstract surface presenter.
    pub fn present_to_surface(
        &self,
        presenter: &mut dyn SurfacePresenter,
    ) -> Result<(), PresentError> {
        self.present_to_surface_with_damage(presenter, None)
    }

    /// Presents the rendered scene to an abstract surface presenter with optional damage clipping.
    pub fn present_to_surface_with_damage(
        &self,
        presenter: &mut dyn SurfacePresenter,
        damage: Option<Rect>,
    ) -> Result<(), PresentError> {
        let (width, height) = presenter.surface_size();
        let width = width.max(1);
        let height = height.max(1);
        let fb = self.render_to_framebuffer(width, height);
        presenter.present(&fb, damage)
    }
}

/// Shared terrain triangles: one patch per tile, two triangles per grid cell.
pub fn terrain_triangle_count(tile_count: usize, grid: u32) -> usize {
    let cells = (grid.max(2) as usize - 1).pow(2);
    tile_count * cells * 2
}

/// Number of skirt triangles generated along the perimeter of one tile patch.
pub fn terrain_skirt_triangle_count(grid: u32) -> usize {
    let cells = grid.max(2) as usize - 1;
    // 4 edges * (grid - 1) segments * 2 triangles per quad
    cells * 8
}

/// Shared terrain triangles with skirts enabled: one patch per tile plus 4 boundary skirts.
pub fn terrain_triangle_count_with_skirts(tile_count: usize, grid: u32) -> usize {
    tile_count * (terrain_triangle_count(1, grid) + terrain_skirt_triangle_count(grid))
}

/// Computes the adaptive skirt depth in metres for a terrain tile patch.
///
/// Skirt depth scales with tile ground footprint span and topographic relief variance,
/// guaranteeing that crack seams between adjacent tiles of differing LOD levels are
/// occluded under oblique camera viewing angles without protruding into space.
pub fn compute_adaptive_skirt_depth(bounds: PatchBounds, elevation_relief_range: f64) -> f64 {
    let lat_mid_rad = ((bounds.lat0 + bounds.lat1) * 0.5).to_radians();
    let lon_span_deg = (bounds.lon1 - bounds.lon0).abs();
    let lat_span_deg = (bounds.lat1 - bounds.lat0).abs();
    let span_x_m = lon_span_deg * 111_319.5 * lat_mid_rad.cos().abs();
    let span_y_m = lat_span_deg * 111_132.9;
    let tile_span_m = span_x_m.max(span_y_m);

    (tile_span_m * 0.05)
        .max(elevation_relief_range * 0.5)
        .clamp(20.0, 500_000.0)
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

/// Appends one relief patch and returns its index range in the shared mesh.
fn push_tile_patch(
    builder: &mut MeshBuilder,
    bounds: PatchBounds,
    grid: u32,
    elevation: Option<&crate::tiles::TileElevation>,
    elevation_tile: TileKey,
    generate_skirts: bool,
) -> (u32, u32) {
    let index_start = builder.index_count() as u32;
    let cols = grid.max(2) as usize;
    let rows = cols;
    let mut positions = Vec::with_capacity(cols * rows);
    let mut geodetics = Vec::with_capacity(cols * rows);
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
            let geo = Geodetic::new(lat, lon, height);
            positions.push(geodetic_to_ecef(geo));
            geodetics.push(geo);
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

    if generate_skirts {
        let elev_range = elevation
            .map(|grid| {
                let mut min_h = f32::MAX;
                let mut max_h = f32::MIN;
                for &h in &grid.heights_m {
                    if h < min_h {
                        min_h = h;
                    }
                    if h > max_h {
                        max_h = h;
                    }
                }
                if max_h >= min_h {
                    (max_h - min_h) as f64
                } else {
                    0.0
                }
            })
            .unwrap_or(0.0);
        let skirt_depth = compute_adaptive_skirt_depth(bounds, elev_range);

        // 1. South edge: row = 0, col from 0 to cols - 1 (West to East)
        let mut south_top = Vec::with_capacity(cols);
        let mut south_bot = Vec::with_capacity(cols);
        let mut south_uvs = Vec::with_capacity(cols);
        for col in 0..cols {
            let idx = col;
            let geo = geodetics[idx];
            let u = col as f32 / (cols - 1) as f32;
            south_top.push(positions[idx]);
            south_bot.push(geodetic_to_ecef(Geodetic::new(
                geo.lat_deg,
                geo.lon_deg,
                geo.height_m - skirt_depth,
            )));
            south_uvs.push((u, 0.0));
        }
        builder.push_skirt(&south_top, &south_bot, &south_uvs);

        // 2. East edge: col = cols - 1, row from 0 to rows - 1 (South to North)
        let mut east_top = Vec::with_capacity(rows);
        let mut east_bot = Vec::with_capacity(rows);
        let mut east_uvs = Vec::with_capacity(rows);
        for row in 0..rows {
            let idx = row * cols + (cols - 1);
            let geo = geodetics[idx];
            let v = row as f32 / (rows - 1) as f32;
            east_top.push(positions[idx]);
            east_bot.push(geodetic_to_ecef(Geodetic::new(
                geo.lat_deg,
                geo.lon_deg,
                geo.height_m - skirt_depth,
            )));
            east_uvs.push((1.0, v));
        }
        builder.push_skirt(&east_top, &east_bot, &east_uvs);

        // 3. North edge: row = rows - 1, col from cols - 1 down to 0 (East to West)
        let mut north_top = Vec::with_capacity(cols);
        let mut north_bot = Vec::with_capacity(cols);
        let mut north_uvs = Vec::with_capacity(cols);
        for col in (0..cols).rev() {
            let idx = (rows - 1) * cols + col;
            let geo = geodetics[idx];
            let u = col as f32 / (cols - 1) as f32;
            north_top.push(positions[idx]);
            north_bot.push(geodetic_to_ecef(Geodetic::new(
                geo.lat_deg,
                geo.lon_deg,
                geo.height_m - skirt_depth,
            )));
            north_uvs.push((u, 1.0));
        }
        builder.push_skirt(&north_top, &north_bot, &north_uvs);

        // 4. West edge: col = 0, row from rows - 1 down to 0 (North to South)
        let mut west_top = Vec::with_capacity(rows);
        let mut west_bot = Vec::with_capacity(rows);
        let mut west_uvs = Vec::with_capacity(rows);
        for row in (0..rows).rev() {
            let idx = row * cols;
            let geo = geodetics[idx];
            let v = row as f32 / (rows - 1) as f32;
            west_top.push(positions[idx]);
            west_bot.push(geodetic_to_ecef(Geodetic::new(
                geo.lat_deg,
                geo.lon_deg,
                geo.height_m - skirt_depth,
            )));
            west_uvs.push((0.0, v));
        }
        builder.push_skirt(&west_top, &west_bot, &west_uvs);
    }

    let index_count = builder.index_count() as u32 - index_start;
    (index_start, index_count)
}

/// One real-world building footprint, anchored at its centre.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BuildingFootprint {
    pub lat: f64,
    pub lon: f64,
    /// Footprint extent in metres as `(east-west, north-south)`.
    pub size_m: (f32, f32),
    /// Roof height above the terrain in metres.
    pub height_m: f32,
}

impl BuildingFootprint {
    pub fn new(lat: f64, lon: f64, size_m: (f32, f32), height_m: f32) -> Self {
        Self {
            lat,
            lon,
            size_m,
            height_m,
        }
    }
}

/// Supplies building footprints for a terrain patch.
///
/// The scene never invents buildings: with no source registered, or with
/// `layers.buildings` off, no building geometry is produced.
pub trait BuildingSource: std::fmt::Debug {
    fn buildings(&mut self, key: TileKey, bounds: PatchBounds) -> Vec<BuildingFootprint>;
}

/// In-memory [`BuildingSource`] that returns every stored footprint whose
/// centre lies inside the requested patch.
#[derive(Debug, Clone, Default)]
pub struct MemoryBuildingSource {
    footprints: Vec<BuildingFootprint>,
}

impl MemoryBuildingSource {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, footprint: BuildingFootprint) -> &mut Self {
        self.footprints.push(footprint);
        self
    }

    pub fn len(&self) -> usize {
        self.footprints.len()
    }

    pub fn is_empty(&self) -> bool {
        self.footprints.is_empty()
    }
}

impl BuildingSource for MemoryBuildingSource {
    fn buildings(&mut self, _key: TileKey, bounds: PatchBounds) -> Vec<BuildingFootprint> {
        let (lat_lo, lat_hi) = (bounds.lat0.min(bounds.lat1), bounds.lat0.max(bounds.lat1));
        let (lon_lo, lon_hi) = (bounds.lon0.min(bounds.lon1), bounds.lon0.max(bounds.lon1));
        self.footprints
            .iter()
            .copied()
            .filter(|f| f.lat >= lat_lo && f.lat < lat_hi && f.lon >= lon_lo && f.lon < lon_hi)
            .collect()
    }
}

/// Asynchronous non-blocking [`BuildingSource`] streaming real-world 3D building footprints
/// from OpenStreetMap Overpass JSON endpoints (with pluggable [`crate::tiles::HttpTransport`]).
pub struct NetworkBuildingSource {
    endpoint_url: String,
    min_fetch_level: u8,
    transport: std::sync::Arc<dyn crate::tiles::HttpTransport>,
    cache: std::collections::HashMap<TileKey, Vec<BuildingFootprint>>,
    pending: std::sync::Arc<std::sync::Mutex<std::collections::HashSet<TileKey>>>,
    results_rx: std::sync::mpsc::Receiver<(TileKey, Vec<BuildingFootprint>)>,
    task_tx: std::sync::mpsc::Sender<(TileKey, PatchBounds, String)>,
}

impl std::fmt::Debug for NetworkBuildingSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NetworkBuildingSource")
            .field("endpoint_url", &self.endpoint_url)
            .field("min_fetch_level", &self.min_fetch_level)
            .field("cached_tiles", &self.cache.len())
            .finish()
    }
}

impl Default for NetworkBuildingSource {
    fn default() -> Self {
        Self::overpass()
    }
}

impl NetworkBuildingSource {
    /// Creates a building streamer querying the public OpenStreetMap Overpass API.
    pub fn overpass() -> Self {
        Self::with_endpoint(
            "https://overpass-api.de/api/interpreter",
            std::sync::Arc::new(crate::tiles::StandardHttpTransport::new()),
        )
    }

    /// Creates a building streamer targeting a custom Overpass-compatible endpoint and transport.
    pub fn with_endpoint(
        endpoint_url: impl Into<String>,
        transport: std::sync::Arc<dyn crate::tiles::HttpTransport>,
    ) -> Self {
        let (task_tx, task_rx) = std::sync::mpsc::channel::<(TileKey, PatchBounds, String)>();
        let (results_tx, results_rx) =
            std::sync::mpsc::channel::<(TileKey, Vec<BuildingFootprint>)>();
        let worker_transport = std::sync::Arc::clone(&transport);
        std::thread::Builder::new()
            .name("strata-building-fetcher".to_string())
            .spawn(move || {
                while let Ok((key, bounds, endpoint)) = task_rx.recv() {
                    let buildings =
                        Self::fetch_for_bounds(worker_transport.as_ref(), &endpoint, bounds);
                    let _ = results_tx.send((key, buildings));
                }
            })
            .expect("failed to spawn strata-building-fetcher thread");

        Self {
            endpoint_url: endpoint_url.into(),
            min_fetch_level: 14,
            transport,
            cache: std::collections::HashMap::new(),
            pending: std::sync::Arc::new(std::sync::Mutex::new(std::collections::HashSet::new())),
            results_rx,
            task_tx,
        }
    }

    pub fn with_min_fetch_level(mut self, level: u8) -> Self {
        self.min_fetch_level = level;
        self
    }

    pub fn cached_tile_count(&self) -> usize {
        self.cache.len()
    }

    /// Synchronously fetches and caches building footprints for `key` / `bounds`.
    pub fn fetch_sync(&mut self, key: TileKey, bounds: PatchBounds) -> Vec<BuildingFootprint> {
        let list = Self::fetch_for_bounds(self.transport.as_ref(), &self.endpoint_url, bounds);
        self.cache.insert(key, list.clone());
        list
    }

    fn build_query_url(endpoint: &str, bounds: PatchBounds) -> String {
        let s = bounds.lat0.min(bounds.lat1);
        let n = bounds.lat0.max(bounds.lat1);
        let w = bounds.lon0.min(bounds.lon1);
        let e = bounds.lon0.max(bounds.lon1);
        let query = format!(
            "[out:json][timeout:10];(way[\"building\"]({s:.5},{w:.5},{n:.5},{e:.5}););out body;>;out skel qt;"
        );
        let mut encoded = String::with_capacity(query.len() * 2);
        for b in query.bytes() {
            match b {
                b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                    encoded.push(b as char);
                }
                _ => {
                    use std::fmt::Write;
                    let _ = write!(encoded, "%{b:02X}");
                }
            }
        }
        if endpoint.contains('?') {
            format!("{endpoint}&data={encoded}")
        } else {
            format!("{endpoint}?data={encoded}")
        }
    }

    fn fetch_for_bounds(
        transport: &dyn crate::tiles::HttpTransport,
        endpoint: &str,
        bounds: PatchBounds,
    ) -> Vec<BuildingFootprint> {
        let url = Self::build_query_url(endpoint, bounds);
        let Ok(bytes) = transport.fetch(&url) else {
            return Vec::new();
        };
        let Ok(text) = std::str::from_utf8(&bytes) else {
            return Vec::new();
        };
        parse_overpass_buildings(text)
    }

    fn poll_completed(&mut self) {
        while let Ok((key, buildings)) = self.results_rx.try_recv() {
            if let Ok(mut guard) = self.pending.lock() {
                guard.remove(&key);
            }
            if self.cache.len() > 256 {
                self.cache.clear();
            }
            self.cache.insert(key, buildings);
        }
    }
}

impl BuildingSource for NetworkBuildingSource {
    fn buildings(&mut self, key: TileKey, bounds: PatchBounds) -> Vec<BuildingFootprint> {
        self.poll_completed();
        if let Some(cached) = self.cache.get(&key) {
            return cached.clone();
        }
        if key.level < self.min_fetch_level {
            return Vec::new();
        }
        let mut enqueue = false;
        if let Ok(mut pending) = self.pending.lock() {
            if !pending.contains(&key) {
                pending.insert(key);
                enqueue = true;
            }
        }
        if enqueue {
            let _ = self.task_tx.send((key, bounds, self.endpoint_url.clone()));
        }
        Vec::new()
    }
}

/// Parses OpenStreetMap Overpass API JSON (`elements` array of nodes and building ways)
/// into extruded [`BuildingFootprint`] entries.
pub fn parse_overpass_buildings(json_text: &str) -> Vec<BuildingFootprint> {
    use crate::json::Json;
    let Ok(root) = Json::parse(json_text) else {
        return Vec::new();
    };
    let Json::Object(top) = root else {
        return Vec::new();
    };
    let Some(Json::Array(elements)) = top.iter().find(|(k, _)| k == "elements").map(|(_, v)| v)
    else {
        return Vec::new();
    };

    fn get_field<'a>(
        entries: &'a [(String, crate::json::Json)],
        key: &str,
    ) -> Option<&'a crate::json::Json> {
        entries.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }
    let as_f64 = |val: &Json| -> Option<f64> {
        match val {
            Json::Number(n) => Some(*n),
            Json::String(s) => s.trim().trim_end_matches('m').trim().parse::<f64>().ok(),
            _ => None,
        }
    };

    let mut nodes: std::collections::HashMap<i64, (f64, f64)> = std::collections::HashMap::new();
    for el in elements {
        let Json::Object(obj) = el else { continue };
        if matches!(get_field(obj, "type"), Some(Json::String(t)) if t == "node") {
            if let (Some(Json::Number(id)), Some(Json::Number(lat)), Some(Json::Number(lon))) = (
                get_field(obj, "id"),
                get_field(obj, "lat"),
                get_field(obj, "lon"),
            ) {
                nodes.insert(*id as i64, (*lat, *lon));
            }
        }
    }

    let mut out = Vec::new();
    for el in elements {
        let Json::Object(obj) = el else { continue };
        if !matches!(get_field(obj, "type"), Some(Json::String(t)) if t == "way") {
            continue;
        }
        let Some(Json::Object(tags)) = get_field(obj, "tags") else {
            continue;
        };
        if get_field(tags, "building").is_none() {
            continue;
        }
        let Some(Json::Array(node_refs)) = get_field(obj, "nodes") else {
            continue;
        };
        let mut lat_min = f64::INFINITY;
        let mut lat_max = f64::NEG_INFINITY;
        let mut lon_min = f64::INFINITY;
        let mut lon_max = f64::NEG_INFINITY;
        let mut count = 0usize;
        for nr in node_refs {
            if let Json::Number(nid) = nr {
                if let Some(&(lat, lon)) = nodes.get(&(*nid as i64)) {
                    lat_min = lat_min.min(lat);
                    lat_max = lat_max.max(lat);
                    lon_min = lon_min.min(lon);
                    lon_max = lon_max.max(lon);
                    count += 1;
                }
            }
        }
        if count < 3 || !lat_min.is_finite() || !lon_min.is_finite() {
            continue;
        }
        let center_lat = (lat_min + lat_max) * 0.5;
        let center_lon = (lon_min + lon_max) * 0.5;
        let meters_per_deg_lat = 111_320.0;
        let meters_per_deg_lon = 111_320.0 * center_lat.to_radians().cos().abs().max(0.2);
        let size_e_m = ((lon_max - lon_min) * meters_per_deg_lon).clamp(4.0, 250.0) as f32;
        let size_n_m = ((lat_max - lat_min) * meters_per_deg_lat).clamp(4.0, 250.0) as f32;
        let height_m = get_field(tags, "height")
            .and_then(as_f64)
            .or_else(|| {
                get_field(tags, "building:levels")
                    .and_then(as_f64)
                    .map(|lv| lv * 3.2)
            })
            .unwrap_or(12.0)
            .clamp(3.0, 600.0) as f32;
        out.push(BuildingFootprint::new(
            center_lat,
            center_lon,
            (size_e_m, size_n_m),
            height_m,
        ));
    }
    out
}

fn push_footprints(
    builder: &mut MeshBuilder,
    source: Option<&mut Box<dyn BuildingSource>>,
    origin: DVec3,
    bounds: PatchBounds,
    elevation: Option<&crate::tiles::TileElevation>,
    elevation_tile: TileKey,
) {
    let Some(source) = source else {
        return;
    };
    for footprint in source.buildings(elevation_tile, bounds) {
        let ground = elevation
            .map(|grid| {
                let (tu, tv) = uv_in_tile(elevation_tile, footprint.lat, footprint.lon);
                grid.sample_bilinear(tu, tv) as f64
            })
            .unwrap_or(0.0);
        let base =
            geodetic_to_ecef(Geodetic::new(footprint.lat, footprint.lon, ground)).sub(origin);
        builder.push_building(
            base,
            (
                footprint.size_m.0 as f64 * 0.5,
                footprint.size_m.1 as f64 * 0.5,
            ),
            footprint.height_m as f64,
            (0.0, 0.0, 0.5, 0.5),
            (0.5, 0.0, 1.0, 1.0),
        );
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
