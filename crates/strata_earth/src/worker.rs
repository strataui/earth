//! Asynchronous tile streaming, frustum-culled LOD selection, and retained GPU mesh builder.

use crate::astro::{build_environment_assets_with_skybox, GlobalCloudMapStore};
use crate::geodesy::{
    build_tile_local_rtc_mesh_morphed, geodetic_to_ecef, is_canonical_2d_morph, level_for_altitude,
    retained_camera_uniforms_for_morph, screen_to_ray_for_morph, stabilized_surface_morph,
    tile_bounding_sphere, tile_bounds, tile_bounds_mercator, traverse_quadtree_lod_2d_flat,
    traverse_quadtree_lod_frustum6_with_relief, wgs84, EarthCamera, Frustum6, Geodetic,
    LocalFrame, PatchBounds, SurfaceMorphParams, TileKey, TileProjection,
    CANONICAL_2D_MORPH_THRESHOLD, MAX_QUADTREE_LEAVES,
};
use crate::gltf::WeatherConfig;
use crate::plugins::{
    CachedPluginTexEntry, EarthDynamicOverlayContext, EarthOceanContext, EarthParticleContext,
    EarthPlugin, EarthPluginFrameContext, EarthPluginHooks, EarthPluginRegistry,
    EarthTextureInjector, EarthTileBuildContext, EarthWorldBuildContext,
    LoadedTileImagerySnapshot, OceanWavesPlugin, WeatherPlugin,
};
use crate::precipitation::{
    AtmosphereSample, PrecipitationConfig, PrecipitationEvaluator, PrecipitationPlugin,
    TemperatureProvider,
};
use crate::raycast::{raycast_surface_ray, raycast_viewport, EarthRayHit};
use crate::satellites::{SatelliteCatalogStore, SatelliteElement};
use crate::scene::{
    compute_adaptive_skirt_depth, requested_tile, sky_gradient, surrounding_patches,
    tile_patch_span_deg, EarthInput, EarthLowLevelState, EarthViewMode, IdleHighResConfig,
    LocalAtmosphereConfig, OceanWavesConfig, SpaceEnvironmentConfig, SurfaceMode,
};
use crate::skybox::SpaceSkyboxStore;
use crate::tiles::{
    stitch_elevation_with_neighbors, synthesize_elevation_from_neighbors, NeighborElevationEdges,
    TileElevation, TileImage, TileProviderConfig, TileProviderPreset, TileSource,
};
use crate::weather_api::{
    BaseWeatherConfig, OpenMeteoWeatherProvider, StaticWeatherProvider, WeatherProvider,
    WeatherQuery, WeatherSnapshot,
};
use crate::world::{EarthObject, EarthObjectId, EarthWorld, ResolvedObjectTransform};
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, RwLock};
use std::thread::JoinHandle;
use strata_render::camera64::DVec3;
use strata_render::scene3d::{
    Background3D, RetainedCameraUniforms, RetainedDynamicMesh, RetainedEnvironmentAssets,
    RetainedOceanSurface, RetainedParticleLayer, RetainedScene3D, RetainedTerrainFrame,
    RetainedTileInstance, RetainedTileMesh, RetainedTileTexture, TileKeyId,
};
use strata_style::types::Color;

fn evaluate_base_weather_snapshot(
    input: &EarthInput,
    ground_elev_m: f64,
    custom_provider: Option<&Arc<dyn WeatherProvider>>,
    live_provider: Option<&OpenMeteoWeatherProvider>,
) -> Option<WeatherSnapshot> {
    if !input.base_weather.enabled {
        return None;
    }
    let (_, _, unix_utc_s) = input.resolved_celestial_state();
    let query = WeatherQuery {
        lat_deg: input.camera.focus.lat_deg,
        lon_deg: input.camera.focus.lon_deg,
        altitude_msl_m: input.camera.altitude_m().max(0.0),
        ground_elevation_m: ground_elev_m.max(0.0),
        unix_utc_s,
    };
    let snap = if let Some(provider) = custom_provider {
        provider.sample(query, &input.base_weather)
    } else if input.base_weather.use_live_api {
        if let Some(live) = live_provider {
            live.sample(query, &input.base_weather)
        } else {
            StaticWeatherProvider.sample(query, &input.base_weather)
        }
    } else {
        StaticWeatherProvider.sample(query, &input.base_weather)
    };
    Some(snap)
}

#[inline]
fn hash_bytes_64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    // Sample stride keeps hashing O(1) (~256 samples max) on large tiles while catching edits.
    let step = (bytes.len() / 256).max(1);
    let mut i = 0;
    while i < bytes.len() {
        h ^= bytes[i] as u64;
        h = h.wrapping_mul(0x100000001b3);
        i += step;
    }
    h ^= bytes.len() as u64;
    h
}

#[inline]
fn hash_elevation_64(elev: Option<&TileElevation>) -> u64 {
    let Some(e) = elev else {
        return 0;
    };
    let mut h: u64 = 0xcbf29ce484222325 ^ ((e.width as u64) << 32) ^ (e.height as u64);
    let step = (e.heights_m.len() / 64).max(1);
    let mut i = 0;
    while i < e.heights_m.len() {
        h ^= e.heights_m[i].to_bits() as u64;
        h = h.wrapping_mul(0x100000001b3);
        i += step;
    }
    h
}

#[inline]
fn flat_patch_key_id(bounds: PatchBounds, level: u8) -> TileKeyId {
    let bits = [
        bounds.lat0.to_bits(),
        bounds.lat1.to_bits(),
        bounds.lon0.to_bits(),
        bounds.lon1.to_bits(),
    ];
    let mut h: u64 = 0xcbf29ce484222325;
    for b in bits {
        h ^= b;
        h = h.wrapping_mul(0x100000001b3);
    }
    TileKeyId::new(level, (h >> 32) as u32, h as u32, 1)
}

#[inline]
fn elevation_range(elev: Option<&TileElevation>) -> f64 {
    let Some(grid) = elev else {
        return 0.0;
    };
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
}

/// Statistics from a single retained terrain build pass.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RetainedBuildStats {
    pub visible_tiles: usize,
    pub mesh_cache_hits: usize,
    pub mesh_cache_misses: usize,
    pub texture_cache_hits: usize,
    pub texture_cache_misses: usize,
    pub ancestor_fallbacks: usize,
}

#[inline]
fn stabilized_morph_params(
    focus: Geodetic,
    flat_blend: f64,
    map_2d_morph: f64,
    projection: TileProjection,
) -> SurfaceMorphParams {
    stabilized_surface_morph(focus, flat_blend, map_2d_morph, projection)
}

#[inline]
fn mesh_morph_signature(morph: SurfaceMorphParams) -> u64 {
    let proj_bit = match morph.projection {
        TileProjection::WebMercator => 1u64,
        TileProjection::Geographic4326 => 2u64,
    };
    if morph.flat_blend <= 1e-6 {
        return proj_bit;
    }
    let flat_q = (morph.flat_blend.clamp(0.0, 1.0) * 500.0).round() as u64;
    let map_q = (morph.map_2d_morph.clamp(0.0, 1.0) * 500.0).round() as u64;
    let lat_q = ((morph.focus.lat_deg + 90.0) * 1_000.0).round() as u64;
    let lon_q = ((morph.focus.lon_deg + 180.0) * 1_000.0).round() as u64;
    let mut h = 0xcbf29ce484222325u64 ^ proj_bit;
    for v in [flat_q, map_q, lat_q, lon_q] {
        h ^= v;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

struct CachedMeshEntry {
    elev_tile: TileKey,
    elev_hash: u64,
    morph_sig: u64,
    last_used_frame: u64,
    mesh: Arc<RetainedTileMesh>,
}

struct CachedTexEntry {
    id: u64,
    pixel_hash: u64,
    revision: u64,
    last_used_frame: u64,
    texture: Arc<RetainedTileTexture>,
}

#[derive(Clone)]
pub(crate) struct GroundElevSnapshot {
    pub(crate) projection: TileProjection,
    pub(crate) patches: Vec<(TileKey, Arc<TileElevation>)>,
}

impl Default for GroundElevSnapshot {
    fn default() -> Self {
        Self {
            projection: TileProjection::WebMercator,
            patches: Vec::new(),
        }
    }
}

pub(crate) fn sample_tile_uv(
    key: TileKey,
    projection: TileProjection,
    lat_deg: f64,
    lon_norm: f64,
) -> Option<(f32, f32)> {
    let per_axis = key.tiles_per_axis() as f64;
    match projection {
        TileProjection::WebMercator => {
            let mx = ((lon_norm + 180.0) / 360.0).clamp(0.0, 1.0);
            let sin_lat = lat_deg.clamp(-85.05112878, 85.05112878).to_radians().sin();
            let my = (0.5 - ((1.0 + sin_lat) / (1.0 - sin_lat)).ln() / (4.0 * std::f64::consts::PI))
                .clamp(0.0, 1.0);
            let tx = mx * per_axis - key.x as f64;
            let ty = my * per_axis - key.y as f64;
            if (0.0..=1.0).contains(&tx) && (0.0..=1.0).contains(&ty) {
                Some((tx as f32, ty as f32))
            } else {
                None
            }
        }
        TileProjection::Geographic4326 => {
            let bounds = tile_bounds(key);
            if lon_norm >= bounds.lon0
                && lon_norm <= bounds.lon1
                && lat_deg >= bounds.lat0
                && lat_deg <= bounds.lat1
            {
                let u = ((lon_norm - bounds.lon0) / (bounds.lon1 - bounds.lon0).max(1e-12)) as f32;
                let v = ((bounds.lat1 - lat_deg) / (bounds.lat1 - bounds.lat0).max(1e-12)) as f32;
                Some((u.clamp(0.0, 1.0), v.clamp(0.0, 1.0)))
            } else {
                None
            }
        }
    }
}

impl GroundElevSnapshot {
    pub(crate) fn sample_elevation_m(&self, lat_deg: f64, lon_deg: f64) -> f64 {
        let lon_norm = (lon_deg + 180.0).rem_euclid(360.0) - 180.0;
        for (tile, elev) in &self.patches {
            if let Some((u, v)) = sample_tile_uv(*tile, self.projection, lat_deg, lon_norm) {
                return elev.sample_bilinear(u, v) as f64;
            }
        }
        0.0
    }
}

fn record_frame_elevation(
    arc_cache: &mut HashMap<TileKey, (u64, Arc<TileElevation>)>,
    frame_elevs: &mut HashMap<TileKey, Arc<TileElevation>>,
    elev_tile: TileKey,
    elev_hash: u64,
    elev: Option<&TileElevation>,
) {
    let Some(grid) = elev else {
        return;
    };
    if frame_elevs.contains_key(&elev_tile) {
        return;
    }
    let arc = match arc_cache.get(&elev_tile) {
        Some((h, existing)) if *h == elev_hash => Arc::clone(existing),
        _ => {
            let created = Arc::new(grid.clone());
            arc_cache.insert(elev_tile, (elev_hash, Arc::clone(&created)));
            created
        }
    };
    frame_elevs.insert(elev_tile, arc);
}

#[inline]
fn tile_sub_uv_in_ancestor(child: TileKey, ancestor: TileKey) -> (f32, f32, f32, f32) {
    let dl = child.level.saturating_sub(ancestor.level);
    if dl == 0 {
        return (0.0, 0.0, 1.0, 1.0);
    }
    let shift = dl.min(24);
    let scale = (1u64 << shift) as f64;
    let dx = (child.x as u64).saturating_sub((ancestor.x as u64) << shift) as f64;
    let dy = (child.y as u64).saturating_sub((ancestor.y as u64) << shift) as f64;
    let u0 = (dx / scale).clamp(0.0, 1.0) as f32;
    let v0 = (dy / scale).clamp(0.0, 1.0) as f32;
    let u1 = ((dx + 1.0) / scale).clamp(0.0, 1.0) as f32;
    let v1 = ((dy + 1.0) / scale).clamp(0.0, 1.0) as f32;
    (u0, v0, u1, v1)
}

/// Persistent CPU cache converting visible LOD leaves into GPU-ready tile-local RTC meshes and textures.
pub struct RetainedTerrainBuilder {
    meshes: HashMap<(TileKeyId, u32, bool), CachedMeshEntry>,
    textures: HashMap<TileKey, CachedTexEntry>,
    night_textures: HashMap<TileKey, CachedTexEntry>,
    plugin_textures: HashMap<u64, CachedPluginTexEntry>,
    cached_elev_arcs: HashMap<TileKey, (u64, Arc<TileElevation>)>,
    elev_mean_cache: HashMap<(TileKey, u64), f64>,
    ground_snapshot: Option<Arc<RwLock<GroundElevSnapshot>>>,
    world: Arc<RwLock<EarthWorld>>,
    next_tex_id: u64,
    next_revision: u64,
    frame_counter: u64,
    cloud_store: Arc<GlobalCloudMapStore>,
    satellite_store: Arc<SatelliteCatalogStore>,
    skybox_store: Arc<SpaceSkyboxStore>,
    plugins: EarthPluginRegistry,
    include_particles: bool,
    include_ocean: bool,
    weather_provider: Option<Arc<dyn WeatherProvider>>,
    building_source: Option<Box<dyn crate::scene::BuildingSource + Send>>,
    last_buildings_mesh: Option<Arc<RwLock<Option<RetainedDynamicMesh>>>>,
    dynamic_mesh_id_base: u64,
    live_weather: OpenMeteoWeatherProvider,
    last_plugin_rev: u64,
    last_ocean_waves: OceanWavesConfig,
    weather_ids: HashMap<u64, u64>,
    weather_rev: u64,
    last_cloud_rev: u64,
    last_sat_rev: u64,
    last_skybox_rev: u64,
    cached_env: Option<((u64, u64), Arc<RetainedEnvironmentAssets>)>,
}

static NEXT_BUILDER_TEX_ID_BLOCK: AtomicU64 = AtomicU64::new(1);
static NEXT_BUILDER_DYN_MESH_ID_BLOCK: AtomicU64 = AtomicU64::new(0x9E48_0000);

impl std::fmt::Debug for RetainedTerrainBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RetainedTerrainBuilder")
            .field("cached_meshes", &self.meshes.len())
            .field("cached_textures", &self.textures.len())
            .field("cached_night_textures", &self.night_textures.len())
            .field("plugin_textures", &self.plugin_textures.len())
            .field("next_revision", &self.next_revision)
            .finish()
    }
}

impl Default for RetainedTerrainBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl RetainedTerrainBuilder {
    pub fn new() -> Self {
        Self::with_stores(
            Arc::new(GlobalCloudMapStore::new(None, None)),
            Arc::new(SatelliteCatalogStore::new(None, None)),
        )
    }

    pub fn with_cloud_store(cloud_store: Arc<GlobalCloudMapStore>) -> Self {
        Self::with_stores(
            cloud_store,
            Arc::new(SatelliteCatalogStore::new(None, None)),
        )
    }

    pub fn with_stores(
        cloud_store: Arc<GlobalCloudMapStore>,
        satellite_store: Arc<SatelliteCatalogStore>,
    ) -> Self {
        Self::with_all_stores(
            cloud_store,
            satellite_store,
            Arc::new(SpaceSkyboxStore::new(None)),
        )
    }

    pub fn with_all_stores(
        cloud_store: Arc<GlobalCloudMapStore>,
        satellite_store: Arc<SatelliteCatalogStore>,
        skybox_store: Arc<SpaceSkyboxStore>,
    ) -> Self {
        let plugins = EarthPluginRegistry::standard();
        let last_plugin_rev = plugins.revision();
        let next_tex_id = NEXT_BUILDER_TEX_ID_BLOCK.fetch_add(1_000_000_000, Ordering::Relaxed);
        let dynamic_mesh_id_base = NEXT_BUILDER_DYN_MESH_ID_BLOCK.fetch_add(16, Ordering::Relaxed);
        Self {
            meshes: HashMap::new(),
            textures: HashMap::new(),
            night_textures: HashMap::new(),
            plugin_textures: HashMap::new(),
            cached_elev_arcs: HashMap::new(),
            elev_mean_cache: HashMap::new(),
            ground_snapshot: Some(Arc::new(RwLock::new(GroundElevSnapshot::default()))),
            world: Arc::new(RwLock::new(EarthWorld::new())),
            next_tex_id,
            next_revision: 1,
            frame_counter: 0,
            cloud_store,
            satellite_store,
            skybox_store,
            plugins,
            include_particles: true,
            include_ocean: true,
            weather_provider: None,
            building_source: None,
            last_buildings_mesh: None,
            dynamic_mesh_id_base,
            live_weather: OpenMeteoWeatherProvider::new(),
            last_plugin_rev,
            last_ocean_waves: OceanWavesConfig::default(),
            weather_ids: HashMap::new(),
            weather_rev: 0,
            last_cloud_rev: 0,
            last_sat_rev: 0,
            last_skybox_rev: 0,
            cached_env: None,
        }
    }

    #[inline]
    pub fn plugins(&self) -> &EarthPluginRegistry {
        &self.plugins
    }

    pub fn set_plugins(&mut self, plugins: EarthPluginRegistry) {
        self.plugins = plugins;
        self.last_plugin_rev = 0;
        self.cached_env = None;
    }

    pub(crate) fn set_include_particles(&mut self, include: bool) {
        self.include_particles = include;
    }

    pub(crate) fn set_include_ocean(&mut self, include: bool) {
        self.include_ocean = include;
    }

    pub fn set_weather_provider(&mut self, provider: Option<Arc<dyn WeatherProvider>>) {
        self.weather_provider = provider;
    }

    /// Installs or clears a [`crate::scene::BuildingSource`] used when `input.layers.buildings` is `true`.
    pub fn set_building_source(
        &mut self,
        source: Option<Box<dyn crate::scene::BuildingSource + Send>>,
    ) {
        self.building_source = source;
    }

    pub(crate) fn set_last_buildings_mesh(
        &mut self,
        slot: Arc<RwLock<Option<RetainedDynamicMesh>>>,
    ) {
        self.last_buildings_mesh = Some(slot);
    }

    pub(crate) fn set_dynamic_mesh_id_base(&mut self, base: u64) {
        self.dynamic_mesh_id_base = base;
    }

    pub(crate) fn set_deterministic_ids_for_cpu_scene(&mut self) {
        self.next_tex_id = 1;
        self.dynamic_mesh_id_base = 0x9E48_0000;
    }

    pub fn register_plugin<P: EarthPlugin>(&mut self, plugin: P) {
        self.plugins.register(plugin);
    }

    pub fn remove_plugin(&mut self, id: &str) -> bool {
        self.plugins.remove(id)
    }

    pub fn set_plugin_enabled(&mut self, id: &str, enabled: bool) -> bool {
        self.plugins.set_enabled(id, enabled)
    }

    pub(crate) fn set_ground_snapshot(&mut self, snap: Arc<RwLock<GroundElevSnapshot>>) {
        self.ground_snapshot = Some(snap);
    }

    pub(crate) fn set_world(&mut self, world: Arc<RwLock<EarthWorld>>) {
        self.world = world;
    }

    pub fn sample_ground_elevation_m(&self, lat_deg: f64, lon_deg: f64) -> f64 {
        self.ground_snapshot
            .as_ref()
            .and_then(|s| s.read().ok())
            .map(|g| g.sample_elevation_m(lat_deg, lon_deg))
            .unwrap_or(0.0)
    }

    /// Computes the average heightmap altitude in metres for `key` from already-loaded DEM
    /// tiles (or their nearest loaded ancestor over `key`'s exact UV footprint), returning
    /// `0.0` (sea level) whenever `elevation_enabled == false` or no DEM tile is loaded yet.
    ///
    /// Never triggers network I/O (`peek_cached_elevation` only).
    pub(crate) fn tile_average_altitude_m(
        &mut self,
        key: TileKey,
        elevation_enabled: bool,
        source: &dyn TileSource,
    ) -> f64 {
        if !elevation_enabled {
            return 0.0;
        }
        let mut probe = key;
        loop {
            if let Some((h, elev_arc)) = self.cached_elev_arcs.get(&probe) {
                let h_val = *h;
                let elev_arc = Arc::clone(elev_arc);
                return if probe == key {
                    *self
                        .elev_mean_cache
                        .entry((probe, h_val))
                        .or_insert_with(|| elev_arc.average_elevation_m())
                } else {
                    let (u0, v0, u1, v1) = tile_sub_uv_in_ancestor(key, probe);
                    elev_arc.average_elevation_in_uv_rect_m(u0, v0, u1, v1)
                };
            }
            if let Some(elev) = source.peek_cached_elevation(probe) {
                let h_val = hash_elevation_64(Some(&elev));
                let elev_arc = Arc::new(elev);
                self.cached_elev_arcs
                    .insert(probe, (h_val, Arc::clone(&elev_arc)));
                return if probe == key {
                    let mean = elev_arc.average_elevation_m();
                    self.elev_mean_cache.insert((probe, h_val), mean);
                    mean
                } else {
                    let (u0, v0, u1, v1) = tile_sub_uv_in_ancestor(key, probe);
                    elev_arc.average_elevation_in_uv_rect_m(u0, v0, u1, v1)
                };
            }
            let Some(parent) = probe.parent() else {
                break;
            };
            probe = parent;
        }
        0.0
    }

    /// Clears cached GPU tile textures (and optionally elevation meshes) when switching tile providers.
    pub fn clear_provider_caches(&mut self, clear_elevation: bool) {
        self.textures.clear();
        self.night_textures.clear();
        self.plugin_textures.clear();
        if clear_elevation {
            self.meshes.clear();
            self.cached_elev_arcs.clear();
            self.elev_mean_cache.clear();
            if let Some(ref snap_lock) = self.ground_snapshot {
                if let Ok(mut guard) = snap_lock.write() {
                    *guard = GroundElevSnapshot::default();
                }
            }
        }
    }

    #[inline]
    pub fn cloud_revision(&self) -> u64 {
        self.cloud_store.snapshot().0
    }

    #[inline]
    pub fn last_built_cloud_rev(&self) -> u64 {
        self.last_cloud_rev
    }

    /// Returns `true` when neither background cloudmap/skybox/satellites/plugins nor ocean waves require rebuilding.
    pub fn frame_cache_valid_for_fast_path(&self, input: &EarthInput) -> bool {
        if self.last_plugin_rev != self.plugins.revision() {
            return false;
        }
        if self.last_ocean_waves != input.ocean_waves {
            return false;
        }
        if self.last_cloud_rev != self.cloud_store.snapshot().0 {
            return false;
        }
        if self.last_skybox_rev != self.skybox_store.snapshot().0 {
            return false;
        }
        if input.space.satellites_visible && self.last_sat_rev != self.satellite_store.snapshot().0
        {
            return false;
        }
        true
    }

    pub(crate) fn base_camera_from_input(input: &EarthInput) -> RetainedCameraUniforms {
        let ll = input.low_level_state();
        let map_t = ll.map_2d_morph as f64;
        let mut cam = input.camera;
        if map_t > 1e-4 {
            cam.focus.lat_deg = cam.focus.lat_deg.clamp(-85.051129, 85.051129);
            let [ex, ey, ez] = cam.eye_offset_m;
            let dist = (ex * ex + ey * ey + ez * ez).sqrt().max(cam.altitude_m());
            cam.eye_offset_m = [
                ex * (1.0 - map_t),
                ey * (1.0 - map_t),
                ez + (dist - ez) * map_t,
            ];
            cam.aim_offset_m = [
                cam.aim_offset_m[0] * (1.0 - map_t),
                cam.aim_offset_m[1] * (1.0 - map_t),
                cam.aim_offset_m[2] * (1.0 - map_t),
            ];
        }
        let morph_flat_t = ll.local_25d_blend.max(ll.map_2d_morph).clamp(0.0, 1.0);
        let lighting_25d = (ll.local_25d_blend * (1.0 - ll.map_2d_morph)).clamp(0.0, 1.0);
        let mut bg = sky_gradient(input.daylight);
        if map_t > 1e-4 {
            let k = (1.0 - map_t as f32).clamp(0.0, 1.0);
            let scale_col = |c: Color| {
                let to_u8 = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
                Color::rgba(to_u8(c.r() * k), to_u8(c.g() * k), to_u8(c.b() * k), to_u8(c.a()))
            };
            bg = match bg {
                Background3D::VerticalGradient { top, bottom } => Background3D::VerticalGradient {
                    top: scale_col(top),
                    bottom: scale_col(bottom),
                },
                Background3D::Solid(c) => Background3D::Solid(scale_col(c)),
                other => other,
            };
        }
        let sun_color = [1.0, 0.97, 0.92];
        let ambient_color = [0.16, 0.19, 0.24];
        let up_override = if is_canonical_2d_morph(map_t) {
            None
        } else {
            input.camera_up_ecef.map(|u| {
                let u0 = DVec3::new(u[0], u[1], u[2]);
                if map_t > 1e-4 {
                    let target_north = LocalFrame::at(cam.focus.lat_deg, cam.focus.lon_deg).north;
                    let blended = DVec3::new(
                        u0.x + (target_north.x - u0.x) * map_t,
                        u0.y + (target_north.y - u0.y) * map_t,
                        u0.z + (target_north.z - u0.z) * map_t,
                    );
                    if blended.length() > 1e-6 {
                        blended.normalize()
                    } else {
                        target_north
                    }
                } else {
                    u0
                }
            })
        };
        let morph = stabilized_morph_params(
            cam.focus,
            morph_flat_t as f64,
            map_t,
            input.tile_projection,
        );
        let mut u = retained_camera_uniforms_for_morph(
            &cam,
            morph,
            up_override,
            input.viewport.width as f64,
            input.viewport.height as f64,
            [0.577_350_26, 0.577_350_26, 0.577_350_26],
            sun_color,
            ambient_color,
            bg,
        );
        u.local_25d_blend = lighting_25d;
        u.map_2d_morph = ll.map_2d_morph.clamp(0.0, 1.0);
        u.day_night_strength = 0.0;
        u.city_lights_intensity = 0.0;
        u.sun_visible = false;
        u.moon_visible = false;
        u.stars_visible = false;
        u.stars_intensity = 0.0;
        u.skybox_intensity = 0.0;
        u.cloud_layer_opacity = 0.0;
        u.local_cloud_opacity = 0.0;
        u.cloud_lon_offset_rad = 0.0;
        u.time_s = input.weather.weather_time_s as f32;
        u
    }

    pub fn camera_from_input_with_plugins_weather_and_ground(
        input: &EarthInput,
        plugins: &EarthPluginRegistry,
        time_s: f64,
        overlay_rev: u64,
        is_dragging: bool,
        weather_snapshot: Option<WeatherSnapshot>,
        ground_elevation_m: f64,
    ) -> RetainedCameraUniforms {
        let mut u = Self::base_camera_from_input(input);
        let ctx = EarthPluginFrameContext {
            input,
            low_level: input.low_level_state(),
            altitude_m: input.camera.altitude_m(),
            ground_elevation_m,
            time_s,
            overlay_rev,
            is_dragging,
            weather_snapshot,
        };
        plugins.run_prepare_camera(&ctx, &mut u);
        u
    }

    pub fn camera_from_input_with_plugins_and_weather(
        input: &EarthInput,
        plugins: &EarthPluginRegistry,
        time_s: f64,
        overlay_rev: u64,
        is_dragging: bool,
        weather_snapshot: Option<WeatherSnapshot>,
    ) -> RetainedCameraUniforms {
        Self::camera_from_input_with_plugins_weather_and_ground(
            input,
            plugins,
            time_s,
            overlay_rev,
            is_dragging,
            weather_snapshot,
            0.0,
        )
    }

    pub fn camera_from_input_with_plugins(
        input: &EarthInput,
        plugins: &EarthPluginRegistry,
        time_s: f64,
        overlay_rev: u64,
        is_dragging: bool,
    ) -> RetainedCameraUniforms {
        let weather_snapshot = evaluate_base_weather_snapshot(input, 0.0, None, None);
        Self::camera_from_input_with_plugins_and_weather(
            input,
            plugins,
            time_s,
            overlay_rev,
            is_dragging,
            weather_snapshot,
        )
    }

    pub fn camera_from_input(input: &EarthInput) -> RetainedCameraUniforms {
        let mut u = Self::base_camera_from_input(input);
        let weather_snapshot = evaluate_base_weather_snapshot(input, 0.0, None, None);
        let ctx = EarthPluginFrameContext {
            input,
            low_level: input.low_level_state(),
            altitude_m: input.camera.altitude_m(),
            ground_elevation_m: 0.0,
            time_s: input.weather.weather_time_s,
            overlay_rev: 0,
            is_dragging: false,
            weather_snapshot,
        };
        crate::plugins::CelestialPlugin::default().on_prepare_camera(&ctx, &mut u);
        crate::plugins::CityLightsPlugin::default().on_prepare_camera(&ctx, &mut u);
        WeatherPlugin::default().on_prepare_camera(&ctx, &mut u);
        u
    }

    fn cache_tile_texture(
        map: &mut HashMap<TileKey, CachedTexEntry>,
        next_tex_id: &mut u64,
        frame_counter: u64,
        key: TileKey,
        img: &TileImage,
        stats: &mut RetainedBuildStats,
    ) -> (u64, u64, Arc<RetainedTileTexture>) {
        let pixel_hash = hash_bytes_64(&img.pixels);
        if let Some(entry) = map.get_mut(&key) {
            if entry.pixel_hash == pixel_hash
                && entry.texture.width == img.width
                && entry.texture.height == img.height
            {
                entry.last_used_frame = frame_counter;
                stats.texture_cache_hits += 1;
                return (entry.id, entry.revision, Arc::clone(&entry.texture));
            }
            let rev = entry.revision + 1;
            let tex = Arc::new(RetainedTileTexture {
                id: entry.id,
                revision: rev,
                width: img.width,
                height: img.height,
                rgba8: Arc::new(img.pixels.clone()),
            });
            let id = entry.id;
            map.insert(
                key,
                CachedTexEntry {
                    id,
                    pixel_hash,
                    revision: rev,
                    last_used_frame: frame_counter,
                    texture: Arc::clone(&tex),
                },
            );
            stats.texture_cache_misses += 1;
            return (id, rev, tex);
        }

        let id = *next_tex_id;
        *next_tex_id += 1;
        let tex = Arc::new(RetainedTileTexture {
            id,
            revision: 1,
            width: img.width,
            height: img.height,
            rgba8: Arc::new(img.pixels.clone()),
        });
        map.insert(
            key,
            CachedTexEntry {
                id,
                pixel_hash,
                revision: 1,
                last_used_frame: frame_counter,
                texture: Arc::clone(&tex),
            },
        );
        stats.texture_cache_misses += 1;
        (id, 1, tex)
    }

    fn resolve_or_cache_texture(
        &mut self,
        key: TileKey,
        img: &TileImage,
        stats: &mut RetainedBuildStats,
    ) -> (u64, u64, Arc<RetainedTileTexture>) {
        Self::cache_tile_texture(
            &mut self.textures,
            &mut self.next_tex_id,
            self.frame_counter,
            key,
            img,
            stats,
        )
    }

    fn ancestor_uv_rect(leaf: TileKey, anc: TileKey) -> [f32; 4] {
        if leaf == anc {
            return [0.0, 0.0, 1.0, 1.0];
        }
        let (u0, v0) = ancestor_uv_point(leaf, anc, 0.0, 0.0);
        let (u1, v1) = ancestor_uv_point(leaf, anc, 1.0, 1.0);
        [u0, v0, u1, v1]
    }

    fn ancestor_at_level(mut key: TileKey, target_level: u8) -> Option<TileKey> {
        while key.level > target_level {
            key = key.parent()?;
        }
        Some(key)
    }

    fn resolve_imagery_or_ancestor(
        &mut self,
        key: TileKey,
        source: &mut dyn TileSource,
        max_level: u8,
        stats: &mut RetainedBuildStats,
    ) -> Option<(TileKey, u64, u64, [f32; 4], Arc<RetainedTileTexture>)> {
        let probe = Self::ancestor_at_level(key, max_level)?;

        // Request exact leaf once (queues async network download if not cached).
        if let Ok(Some(img)) = source.imagery(probe) {
            if !img.pixels.is_empty() {
                let (id, rev, tex) = self.resolve_or_cache_texture(probe, &img, stats);
                if probe != key {
                    stats.ancestor_fallbacks += 1;
                }
                return Some((probe, id, rev, Self::ancestor_uv_rect(key, probe), tex));
            }
        }
        // Instant placeholder/offline fallback at the exact requested level (e.g. z=0..=4)
        // while the active network provider downloads that tile asynchronously.
        if let Some(img) = source.peek_cached_imagery(probe) {
            if !img.pixels.is_empty() {
                let (id, rev, tex) = self.resolve_or_cache_texture(probe, &img, stats);
                if probe != key {
                    stats.ancestor_fallbacks += 1;
                }
                return Some((probe, id, rev, Self::ancestor_uv_rect(key, probe), tex));
            }
        }

        // Walk ancestors without issuing spurious network fetches for intermediate missing levels,
        // while keeping coarse-to-fine stepping-stone tiers queued whenever the zoom gap > 2 levels.
        let request_steps = |src: &mut dyn TileSource, best_lvl: u8| {
            if probe.level <= best_lvl.saturating_add(2) {
                return;
            }
            let mut step_lvl = best_lvl.saturating_add(3);
            let max_step = probe.level.saturating_sub(2);
            while step_lvl <= max_step {
                if let Some(step) = Self::ancestor_at_level(probe, step_lvl) {
                    let _ = src.imagery(step);
                }
                step_lvl = step_lvl.saturating_add(3);
            }
        };

        let mut anc = probe.parent();
        while let Some(a) = anc {
            if a.level <= crate::basemap::MAX_EMBEDDED_BASE_LEVEL {
                request_steps(source, a.level);
                if let Ok(Some(img)) = source.imagery(a) {
                    if !img.pixels.is_empty() {
                        let (id, rev, tex) = self.resolve_or_cache_texture(a, &img, stats);
                        stats.ancestor_fallbacks += 1;
                        return Some((a, id, rev, Self::ancestor_uv_rect(key, a), tex));
                    }
                }
            }
            if let Some(entry) = self.textures.get_mut(&a) {
                request_steps(source, a.level);
                entry.last_used_frame = self.frame_counter;
                stats.ancestor_fallbacks += 1;
                return Some((
                    a,
                    entry.id,
                    entry.revision,
                    Self::ancestor_uv_rect(key, a),
                    Arc::clone(&entry.texture),
                ));
            }
            if let Some(img) = source.peek_cached_imagery(a) {
                if !img.pixels.is_empty() {
                    request_steps(source, a.level);
                    let (id, rev, tex) = self.resolve_or_cache_texture(a, &img, stats);
                    stats.ancestor_fallbacks += 1;
                    return Some((a, id, rev, Self::ancestor_uv_rect(key, a), tex));
                }
            }
            anc = a.parent();
        }
        None
    }

    fn resolve_night_lights_or_ancestor(
        &mut self,
        key: TileKey,
        source: &mut dyn TileSource,
        stats: &mut RetainedBuildStats,
    ) -> Option<(u64, u64, [f32; 4], Arc<RetainedTileTexture>)> {
        let top = Self::ancestor_at_level(key, 8)?;
        if top.level < 5 {
            return None;
        }
        if let Ok(Some(img)) = source.night_lights(top) {
            if !img.pixels.is_empty() {
                let (id, rev, tex) = Self::cache_tile_texture(
                    &mut self.night_textures,
                    &mut self.next_tex_id,
                    self.frame_counter,
                    top,
                    &img,
                    stats,
                );
                return Some((id, rev, Self::ancestor_uv_rect(key, top), tex));
            }
        }
        let mut anc = top.parent();
        while let Some(a) = anc {
            if a.level < 5 {
                break;
            }
            if let Some(entry) = self.night_textures.get_mut(&a) {
                entry.last_used_frame = self.frame_counter;
                return Some((
                    entry.id,
                    entry.revision,
                    Self::ancestor_uv_rect(key, a),
                    Arc::clone(&entry.texture),
                ));
            }
            if let Some(img) = source.peek_cached_night_lights(a) {
                if !img.pixels.is_empty() {
                    let (id, rev, tex) = Self::cache_tile_texture(
                        &mut self.night_textures,
                        &mut self.next_tex_id,
                        self.frame_counter,
                        a,
                        &img,
                        stats,
                    );
                    return Some((id, rev, Self::ancestor_uv_rect(key, a), tex));
                }
            }
            anc = a.parent();
        }
        None
    }

    fn resolve_night_for_tile(
        &mut self,
        key: TileKey,
        want_night_tiles: bool,
        source: &mut dyn TileSource,
        stats: &mut RetainedBuildStats,
        textures_out: &mut Vec<Arc<RetainedTileTexture>>,
        seen_tex_ids: &mut HashSet<u64>,
    ) -> (Option<u64>, u64, [f32; 4]) {
        if !want_night_tiles {
            return (None, 0, [0.0, 0.0, 1.0, 1.0]);
        }
        if let Some((nid, nrev, nrect, ntex)) =
            self.resolve_night_lights_or_ancestor(key, source, stats)
        {
            if seen_tex_ids.insert(nid) {
                textures_out.push(ntex);
            }
            (Some(nid), nrev, nrect)
        } else {
            (None, 0, [0.0, 0.0, 1.0, 1.0])
        }
    }

    fn prune_lru_caches(&mut self) {
        const MAX_CACHED_MESHES: usize = 768;
        const MAX_CACHED_TEXTURES: usize = 1024;
        const MAX_CACHED_NIGHT: usize = 256;
        const MAX_CACHED_ELEV_MEANS: usize = 1024;

        fn trim_oldest<K: Clone + Eq + std::hash::Hash, V>(
            map: &mut HashMap<K, V>,
            cap: usize,
            frame_of: fn(&V) -> u64,
        ) {
            if map.len() <= cap {
                return;
            }
            let target = cap * 3 / 4;
            let mut entries: Vec<(K, u64)> = map
                .iter()
                .map(|(k, v)| (k.clone(), frame_of(v)))
                .collect();
            entries.sort_by_key(|&(_, f)| f);
            let remove_n = entries.len().saturating_sub(target);
            for (k, _) in entries.into_iter().take(remove_n) {
                map.remove(&k);
            }
        }

        trim_oldest(&mut self.meshes, MAX_CACHED_MESHES, |m| m.last_used_frame);
        trim_oldest(&mut self.textures, MAX_CACHED_TEXTURES, |t| t.last_used_frame);
        trim_oldest(&mut self.night_textures, MAX_CACHED_NIGHT, |t| t.last_used_frame);
        trim_oldest(&mut self.plugin_textures, MAX_CACHED_TEXTURES, |t| t.last_used_frame);
        if self.elev_mean_cache.len() > MAX_CACHED_ELEV_MEANS {
            self.elev_mean_cache.clear();
        }
    }

    fn downsample_cached_children(
        key: TileKey,
        res: u32,
        source: &dyn TileSource,
    ) -> Option<TileElevation> {
        if key.level >= 15 {
            return None;
        }
        let mut kids: [Option<TileElevation>; 4] = [None, None, None, None];
        for (q, slot) in kids.iter_mut().enumerate() {
            let ck = key.child(q as u8)?;
            *slot = Some(source.peek_cached_elevation(ck)?);
        }
        let n = res.clamp(2, 65) as usize;
        let denom = (n - 1) as f32;
        let mut heights_m = Vec::with_capacity(n * n);
        for row in 0..n {
            let v = row as f32 / denom;
            let qy = if v >= 0.5 { 1usize } else { 0usize };
            let cv = (v * 2.0 - qy as f32).clamp(0.0, 1.0);
            for col in 0..n {
                let u = col as f32 / denom;
                let qx = if u >= 0.5 { 1usize } else { 0usize };
                let cu = (u * 2.0 - qx as f32).clamp(0.0, 1.0);
                let q_idx = (qy << 1) | qx;
                let h = kids[q_idx]
                    .as_ref()
                    .map(|k| k.sample_bilinear(cu, cv))
                    .unwrap_or(0.0);
                heights_m.push(h);
            }
        }
        Some(TileElevation {
            width: n as u32,
            height: n as u32,
            heights_m,
        })
    }

    fn resolve_elevation_or_ancestor(
        &self,
        key: TileKey,
        enabled: bool,
        source: &mut dyn TileSource,
    ) -> Result<(Option<TileElevation>, TileKey), ()> {
        if !enabled {
            return Ok((None, key));
        }
        let mut probe = key;
        if let Ok(Some(elev)) = source.elevation(key) {
            return Ok((Some(elev), key));
        }
        if key.level > 15 {
            if let Some(p15) = Self::ancestor_at_level(key, 15) {
                probe = p15;
                match source.elevation(p15) {
                    Ok(Some(elev)) => return Ok((Some(elev), p15)),
                    Ok(None) => {}
                    Err(_) => return Err(()),
                }
            }
        }
        let mut anc = probe.parent();
        while let Some(a) = anc {
            if let Some(elev) = source.peek_cached_elevation(a) {
                return Ok((Some(elev), a));
            }
            if let Some((_, arc)) = self.cached_elev_arcs.get(&a) {
                return Ok((Some((**arc).clone()), a));
            }
            anc = a.parent();
        }
        if probe.level > 6 {
            let step_lvl = if probe.level > 11 {
                11
            } else {
                probe.level.saturating_sub(3).max(5)
            };
            if let Some(step) = Self::ancestor_at_level(probe, step_lvl) {
                if let Ok(Some(elev)) = source.elevation(step) {
                    return Ok((Some(elev), step));
                }
            }
        }
        Ok((None, key))
    }
}

#[derive(Clone)]
struct ResolvedLeafElev {
    elev_tile: TileKey,
    is_exact: bool,
    is_synthesized: bool,
    elev_err: bool,
    elev: Option<Arc<TileElevation>>,
    elev_hash: u64,
}

fn fill_partial_edge_samples(
    samples: &[Option<f32>],
    fallback: Option<&[f32]>,
) -> Option<Vec<f32>> {
    let n = samples.len();
    if n < 2 {
        return None;
    }
    let any_present = samples.iter().any(|s| s.is_some());
    if !any_present {
        return None;
    }
    if let Some(fb) = fallback {
        if fb.len() == n {
            return Some(
                samples
                    .iter()
                    .enumerate()
                    .map(|(i, s)| s.unwrap_or(fb[i]))
                    .collect(),
            );
        }
    }
    // Linearly bridge any missing sub-segments along a partially loaded neighbor edge.
    let mut out = vec![0.0f32; n];
    let mut first_valid: Option<usize> = None;
    for (i, s) in samples.iter().enumerate() {
        if let Some(v) = *s {
            out[i] = v;
            if first_valid.is_none() {
                first_valid = Some(i);
            }
        }
    }
    let first = first_valid?;
    for i in 0..first {
        out[i] = out[first];
    }
    let mut prev = first;
    for i in (first + 1)..n {
        if samples[i].is_some() {
            let span = (i - prev) as f32;
            if i > prev + 1 {
                let a = out[prev];
                let b = out[i];
                for j in (prev + 1)..i {
                    let t = (j - prev) as f32 / span;
                    out[j] = a + (b - a) * t;
                }
            }
            prev = i;
        }
    }
    for i in (prev + 1)..n {
        out[i] = out[prev];
    }
    Some(out)
}

#[allow(clippy::too_many_arguments)]
fn peek_source_or_cached_elev<'a>(
    key: TileKey,
    peek_cache: &'a mut HashMap<TileKey, Option<Arc<TileElevation>>>,
    cached_arcs: &HashMap<TileKey, (u64, Arc<TileElevation>)>,
    source: &dyn TileSource,
) -> Option<Arc<TileElevation>> {
    if let Some(entry) = peek_cache.get(&key) {
        return entry.clone();
    }
    let found = source
        .peek_cached_elevation(key)
        .map(Arc::new)
        .or_else(|| cached_arcs.get(&key).map(|(_, a)| Arc::clone(a)));
    peek_cache.insert(key, found.clone());
    found
}

#[allow(clippy::too_many_arguments)]
fn sample_child_hierarchy_uv(
    parent: TileKey,
    u: f32,
    v: f32,
    depth: u8,
    min_level: u8,
    allow_synthesized: bool,
    working: &HashMap<TileKey, ResolvedLeafElev>,
    peek_cache: &mut HashMap<TileKey, Option<Arc<TileElevation>>>,
    cached_arcs: &HashMap<TileKey, (u64, Arc<TileElevation>)>,
    source: &dyn TileSource,
) -> Option<f32> {
    let qx0: u8 = if u >= 0.5 { 1 } else { 0 };
    let qy0: u8 = if v >= 0.5 { 1 } else { 0 };
    let mut candidates = [(qx0, qy0); 4];
    let mut count = 1usize;
    let on_x_mid = (u - 0.5).abs() < 1e-4;
    let on_y_mid = (v - 0.5).abs() < 1e-4;
    if on_x_mid {
        candidates[count] = (1 - qx0, qy0);
        count += 1;
    }
    if on_y_mid {
        candidates[count] = (qx0, 1 - qy0);
        count += 1;
    }
    if on_x_mid && on_y_mid {
        candidates[count] = (1 - qx0, 1 - qy0);
        count += 1;
    }
    for &(qx, qy) in &candidates[..count] {
        let Some(child) = parent.child((qy << 1) | qx) else {
            continue;
        };
        let cu = (u * 2.0 - qx as f32).clamp(0.0, 1.0);
        let cv = (v * 2.0 - qy as f32).clamp(0.0, 1.0);
        if child.level >= min_level {
            if let Some(w) = working.get(&child) {
                if w.elev_tile == child && (allow_synthesized || !w.is_synthesized) {
                    if let Some(ref e) = w.elev {
                        return Some(e.sample_bilinear(cu, cv));
                    }
                }
            }
            if let Some(e) = peek_source_or_cached_elev(child, peek_cache, cached_arcs, source) {
                return Some(e.sample_bilinear(cu, cv));
            }
        }
        if depth > 0 {
            if let Some(h) = sample_child_hierarchy_uv(
                child,
                cu,
                cv,
                depth - 1,
                min_level,
                allow_synthesized,
                working,
                peek_cache,
                cached_arcs,
                source,
            ) {
                return Some(h);
            }
        }
    }
    None
}

/// Samples elevation for neighbor tile `nk` at normalized tile UV `(u, v)` from same-level,
/// finer subdivided child (`z+1..z+2`), or coarser ancestor (`>= min_level`) elevation data.
#[allow(clippy::too_many_arguments)]
fn sample_tile_uv_from_hierarchy(
    nk: TileKey,
    u: f32,
    v: f32,
    min_level: u8,
    allow_synthesized: bool,
    working: &HashMap<TileKey, ResolvedLeafElev>,
    peek_cache: &mut HashMap<TileKey, Option<Arc<TileElevation>>>,
    cached_arcs: &HashMap<TileKey, (u64, Arc<TileElevation>)>,
    source: &dyn TileSource,
) -> Option<f32> {
    // 1. Same-level tile (exact or working frame tile).
    if nk.level >= min_level {
        if let Some(w) = working.get(&nk) {
            if w.elev_tile.level >= min_level && (allow_synthesized || !w.is_synthesized) {
                if let Some(ref e) = w.elev {
                    if w.elev_tile == nk {
                        return Some(e.sample_bilinear(u, v));
                    }
                    let (u0, v0, u1, v1) = tile_sub_uv_in_ancestor(nk, w.elev_tile);
                    return Some(e.sample_bilinear(u0 + (u1 - u0) * u, v0 + (v1 - v0) * v));
                }
            }
        }
        if let Some(e) = peek_source_or_cached_elev(nk, peek_cache, cached_arcs, source) {
            return Some(e.sample_bilinear(u, v));
        }
    }

    // 2. Finer subdivided children (`z+1`, `z+2`) when adjacent LOD is deeper than `leaf`.
    if nk.level < 15 {
        if let Some(h) = sample_child_hierarchy_uv(
            nk,
            u,
            v,
            1,
            min_level,
            allow_synthesized,
            working,
            peek_cache,
            cached_arcs,
            source,
        ) {
            return Some(h);
        }
    }

    // 3. Coarser ancestors (`z-1` down to `min_level`) when adjacent LOD is shallower.
    let mut anc = nk.parent();
    while let Some(a) = anc {
        if a.level < min_level {
            break;
        }
        if let Some(w) = working.get(&a) {
            if w.elev_tile == a && (allow_synthesized || !w.is_synthesized) {
                if let Some(ref e) = w.elev {
                    let (u0, v0, u1, v1) = tile_sub_uv_in_ancestor(nk, a);
                    return Some(e.sample_bilinear(u0 + (u1 - u0) * u, v0 + (v1 - v0) * v));
                }
            }
        }
        if let Some(e) = peek_source_or_cached_elev(a, peek_cache, cached_arcs, source) {
            let (u0, v0, u1, v1) = tile_sub_uv_in_ancestor(nk, a);
            return Some(e.sample_bilinear(u0 + (u1 - u0) * u, v0 + (v1 - v0) * v));
        }
        anc = a.parent();
    }

    None
}

#[allow(clippy::too_many_arguments)]
fn sample_neighbor_edges_for_leaf(
    leaf: TileKey,
    res: u32,
    min_level: u8,
    allow_synthesized: bool,
    include_diagonals: bool,
    base_for_partial: Option<(&TileElevation, (f32, f32, f32, f32))>,
    working: &HashMap<TileKey, ResolvedLeafElev>,
    peek_cache: &mut HashMap<TileKey, Option<Arc<TileElevation>>>,
    cached_arcs: &HashMap<TileKey, (u64, Arc<TileElevation>)>,
    source: &dyn TileSource,
) -> NeighborElevationEdges {
    if leaf.level == 0 {
        return NeighborElevationEdges::default();
    }
    let n = res.clamp(2, 65) as usize;
    let denom = (n - 1) as f32;
    let per_axis = leaf.tiles_per_axis();
    let x_w = if leaf.x == 0 { per_axis - 1 } else { leaf.x - 1 };
    let x_e = if leaf.x + 1 >= per_axis { 0 } else { leaf.x + 1 };
    let y_n = leaf.y.checked_sub(1);
    let y_s = if leaf.y + 1 < per_axis {
        Some(leaf.y + 1)
    } else {
        None
    };

    let sample_base_edge = |fixed_u: Option<f32>, fixed_v: Option<f32>| -> Option<Vec<f32>> {
        let (base, (u0, v0, u1, v1)) = base_for_partial?;
        Some(
            (0..n)
                .map(|i| {
                    let t = i as f32 / denom;
                    let u = fixed_u.unwrap_or(t);
                    let v = fixed_v.unwrap_or(t);
                    base.sample_bilinear(u0 + (u1 - u0) * u, v0 + (v1 - v0) * v)
                })
                .collect(),
        )
    };

    let mut edges = NeighborElevationEdges::default();

    // West edge: target (u=0, v=0..1) meets West neighbor (u=1, v=0..1).
    let wk = TileKey::new(leaf.level, x_w, leaf.y);
    let w_raw: Vec<Option<f32>> = (0..n)
        .map(|i| {
            sample_tile_uv_from_hierarchy(
                wk,
                1.0,
                i as f32 / denom,
                min_level,
                allow_synthesized,
                working,
                peek_cache,
                cached_arcs,
                source,
            )
        })
        .collect();
    let w_fb = sample_base_edge(Some(0.0), None);
    edges.west = fill_partial_edge_samples(&w_raw, w_fb.as_deref());

    // East edge: target (u=1, v=0..1) meets East neighbor (u=0, v=0..1).
    let ek = TileKey::new(leaf.level, x_e, leaf.y);
    let e_raw: Vec<Option<f32>> = (0..n)
        .map(|i| {
            sample_tile_uv_from_hierarchy(
                ek,
                0.0,
                i as f32 / denom,
                min_level,
                allow_synthesized,
                working,
                peek_cache,
                cached_arcs,
                source,
            )
        })
        .collect();
    let e_fb = sample_base_edge(Some(1.0), None);
    edges.east = fill_partial_edge_samples(&e_raw, e_fb.as_deref());

    // North edge: target (u=0..1, v=0) meets North neighbor (u=0..1, v=1).
    if let Some(yn) = y_n {
        let nk = TileKey::new(leaf.level, leaf.x, yn);
        let n_raw: Vec<Option<f32>> = (0..n)
            .map(|i| {
                sample_tile_uv_from_hierarchy(
                    nk,
                    i as f32 / denom,
                    1.0,
                    min_level,
                    allow_synthesized,
                    working,
                    peek_cache,
                    cached_arcs,
                    source,
                )
            })
            .collect();
        let n_fb = sample_base_edge(None, Some(0.0));
        edges.north = fill_partial_edge_samples(&n_raw, n_fb.as_deref());
    }

    // South edge: target (u=0..1, v=1) meets South neighbor (u=0..1, v=0).
    if let Some(ys) = y_s {
        let sk = TileKey::new(leaf.level, leaf.x, ys);
        let s_raw: Vec<Option<f32>> = (0..n)
            .map(|i| {
                sample_tile_uv_from_hierarchy(
                    sk,
                    i as f32 / denom,
                    0.0,
                    min_level,
                    allow_synthesized,
                    working,
                    peek_cache,
                    cached_arcs,
                    source,
                )
            })
            .collect();
        let s_fb = sample_base_edge(None, Some(1.0));
        edges.south = fill_partial_edge_samples(&s_raw, s_fb.as_deref());
    }

    if include_diagonals {
        if edges.west.is_none() && edges.north.is_none() {
            if let Some(yn) = y_n {
                edges.nw = sample_tile_uv_from_hierarchy(
                    TileKey::new(leaf.level, x_w, yn),
                    1.0,
                    1.0,
                    min_level,
                    allow_synthesized,
                    working,
                    peek_cache,
                    cached_arcs,
                    source,
                );
            }
        }
        if edges.east.is_none() && edges.north.is_none() {
            if let Some(yn) = y_n {
                edges.ne = sample_tile_uv_from_hierarchy(
                    TileKey::new(leaf.level, x_e, yn),
                    0.0,
                    1.0,
                    min_level,
                    allow_synthesized,
                    working,
                    peek_cache,
                    cached_arcs,
                    source,
                );
            }
        }
        if edges.west.is_none() && edges.south.is_none() {
            if let Some(ys) = y_s {
                edges.sw = sample_tile_uv_from_hierarchy(
                    TileKey::new(leaf.level, x_w, ys),
                    1.0,
                    0.0,
                    min_level,
                    allow_synthesized,
                    working,
                    peek_cache,
                    cached_arcs,
                    source,
                );
            }
        }
        if edges.east.is_none() && edges.south.is_none() {
            if let Some(ys) = y_s {
                edges.se = sample_tile_uv_from_hierarchy(
                    TileKey::new(leaf.level, x_e, ys),
                    0.0,
                    0.0,
                    min_level,
                    allow_synthesized,
                    working,
                    peek_cache,
                    cached_arcs,
                    source,
                );
            }
        }
    }

    edges
}

impl RetainedTerrainBuilder {

    fn resolve_environment_assets(
        &mut self,
        input: &EarthInput,
        frame_ctx: &EarthPluginFrameContext<'_>,
    ) -> (Option<Arc<RetainedEnvironmentAssets>>, Arc<Vec<SatelliteElement>>) {
        let ll = input.low_level_state();
        let flat_t = ll.local_25d_blend.max(ll.map_2d_morph);
        let weather_active = self.plugins.is_active(WeatherPlugin::ID);
        let sat_active = self.plugins.is_active("satellites");
        let cel_active = self.plugins.is_active(crate::plugins::CelestialPlugin::ID);

        if weather_active
            && ll.map_2d_morph <= 0.05
            && input.weather.global_cloud_map
            && input.layers.clouds
            && std::env::var_os("STRATA_EARTH_OFFLINE").is_none()
        {
            self.cloud_store
                .ensure_fetched(input.weather.resolved_cloud_map_url());
        }
        if sat_active
            && flat_t < 0.85
            && input.space.satellites_visible
            && std::env::var_os("STRATA_EARTH_OFFLINE").is_none()
        {
            self.satellite_store.ensure_fetched();
        }
        if cel_active
            && ll.map_2d_morph <= 0.05
            && (flat_t < 0.85 || input.local_atmosphere.night_stars_sync)
            && input.space.stars_visible
            && std::env::var_os("STRATA_EARTH_OFFLINE").is_none()
        {
            self.skybox_store.ensure_loaded(
                input.space.skybox_survey,
                input.space.skybox_quality,
                &input.space.assets,
            );
        } else if cel_active && input.space.skybox_quality == crate::skybox::SpaceSkyboxQuality::Off
        {
            self.skybox_store.ensure_loaded(
                input.space.skybox_survey,
                crate::skybox::SpaceSkyboxQuality::Off,
                &input.space.assets,
            );
        }
        let (cloud_rev, cloud_tex) = self.cloud_store.snapshot();
        self.last_cloud_rev = cloud_rev;
        let (skybox_rev, skybox_cubemap) = self.skybox_store.snapshot();
        self.last_skybox_rev = skybox_rev;
        let (sat_rev, sat_elements) = self.satellite_store.snapshot();
        self.last_sat_rev = sat_rev;
        let env_rev = (cloud_rev, skybox_rev);
        let mut env = match &self.cached_env {
            Some((r, cached)) if *r == env_rev => Arc::clone(cached),
            _ => {
                let built_rev = cloud_rev.wrapping_mul(100_003).wrapping_add(skybox_rev);
                let built = build_environment_assets_with_skybox(
                    Some(cloud_tex),
                    skybox_cubemap,
                    built_rev.max(1),
                );
                self.cached_env = Some((env_rev, Arc::clone(&built)));
                built
            }
        };
        if self
            .plugins
            .active_hooks()
            .intersects(EarthPluginHooks::ENVIRONMENT)
        {
            let mut env_val = (*env).clone();
            self.plugins.run_prepare_environment(frame_ctx, &mut env_val);
            env = Arc::new(env_val);
        }
        (Some(env), sat_elements)
    }

    #[allow(clippy::too_many_arguments)]
    fn close_ground_leaf_grid(
        base_grid: u32,
        is_flat_2d: bool,
        elevation_enabled: bool,
        eye_alt_m: f64,
        eye_ecef: DVec3,
        cam_forward: DVec3,
        focus_lat_deg: f64,
        focus_lon_deg: f64,
        bounds: PatchBounds,
    ) -> u32 {
        if is_flat_2d {
            return base_grid;
        }
        let mut target_grid = base_grid;
        if eye_alt_m < 420.0 {
            let clamped_lat = focus_lat_deg.clamp(bounds.lat0, bounds.lat1);
            let clamped_lon = focus_lon_deg.clamp(bounds.lon0, bounds.lon1);
            let dlat_m = (clamped_lat - focus_lat_deg) * 111_320.0;
            let dlon_m = crate::geodesy::wrap_longitude(clamped_lon - focus_lon_deg)
                * 111_320.0
                * focus_lat_deg.to_radians().cos().abs().max(0.15);
            let dist_m = dlat_m.hypot(dlon_m);
            if eye_alt_m < 95.0 && dist_m < 380.0 {
                target_grid = target_grid.max(33);
            } else if dist_m < 780.0 {
                target_grid = target_grid.max(25);
            }
        }
        if elevation_enabled && (eye_alt_m < 420.0 || (base_grid >= 17 && eye_alt_m < 18_000.0)) {
            let mid_lat = 0.5 * (bounds.lat0 + bounds.lat1);
            let mid_lon = 0.5 * (bounds.lon0 + bounds.lon1);
            let tile_center = geodetic_to_ecef(Geodetic::new(mid_lat, mid_lon, 0.0));
            let to_tile = tile_center.sub(eye_ecef);
            let dist = to_tile.length().max(1.0);
            let surf_up = tile_center.scale(1.0 / tile_center.length().max(1.0));
            let cos_inc = (-to_tile.dot(surf_up) / dist).clamp(0.0, 1.0);
            let sin_inc = (1.0 - cos_inc * cos_inc).max(0.0).sqrt();
            let gaze_cos = (to_tile.dot(cam_forward) / dist).clamp(0.0, 1.0);
            // Oblique/grazing terrain silhouettes along the gaze corridor need dense DEM vertex
            // sampling to keep ridges and slopes crisp; steep nadir views stay at `base_grid`.
            let near_horizon_cap = (eye_alt_m * 12.0).clamp(2_400.0, 18_000.0);
            let mid_horizon_cap = (eye_alt_m * 20.0).clamp(3_500.0, 32_000.0);
            if sin_inc > 0.78 && gaze_cos > 0.85 && dist < near_horizon_cap {
                target_grid = target_grid.max(33);
            } else if sin_inc > 0.55 && gaze_cos > 0.72 && dist < mid_horizon_cap {
                target_grid = target_grid.max(25);
            }
        }
        target_grid
    }

    #[allow(clippy::too_many_arguments)]
    fn emit_tile_instance(
        &mut self,
        imagery_tile: TileKey,
        night_tile: TileKey,
        key_id: TileKeyId,
        mesh_revision: u64,
        center_ecef: [f64; 3],
        bounds: PatchBounds,
        want_night_tiles: bool,
        elev_enabled: bool,
        unlit: bool,
        fallback_tint: [f32; 4],
        source: &mut dyn TileSource,
        max_tile_level: u8,
        stats: &mut RetainedBuildStats,
        textures_out: &mut Vec<Arc<RetainedTileTexture>>,
        seen_tex_ids: &mut HashSet<u64>,
        loaded_imagery: &mut HashMap<TileKey, Arc<RetainedTileTexture>>,
        instances_out: &mut Vec<RetainedTileInstance>,
        emitted_meta: &mut Vec<(TileKey, TileKeyId, PatchBounds, bool, bool)>,
    ) {
        let (tex_id, tex_rev, uv_rect, tint) = if let Some((resolved_key, tid, trev, rect, tex)) =
            self.resolve_imagery_or_ancestor(imagery_tile, source, max_tile_level, stats)
        {
            loaded_imagery
                .entry(resolved_key)
                .or_insert_with(|| Arc::clone(&tex));
            if seen_tex_ids.insert(tid) {
                textures_out.push(tex);
            }
            (Some(tid), trev, rect, [1.0, 1.0, 1.0, 1.0])
        } else {
            (None, 0, [0.0, 0.0, 1.0, 1.0], fallback_tint)
        };
        let (night_texture_id, night_texture_revision, night_uv_rect) = self
            .resolve_night_for_tile(
                night_tile,
                want_night_tiles,
                source,
                stats,
                textures_out,
                seen_tex_ids,
            );
        instances_out.push(RetainedTileInstance {
            key: key_id,
            mesh_revision,
            center_ecef,
            texture_id: tex_id,
            texture_revision: tex_rev,
            uv_rect,
            tint,
            unlit,
            night_texture_id,
            night_texture_revision,
            night_uv_rect,
        });
        emitted_meta.push((
            imagery_tile,
            key_id,
            bounds,
            want_night_tiles,
            elev_enabled,
        ));
    }

    /// Synchronously builds an immediate Frame-0 [`RetainedTerrainFrame`] containing the
    /// procedural skybox + environment assets and the single `z=0` root tile (from non-blocking
    /// `source.peek_cached_imagery` or the compiled-in embedded base map) so first paint never stalls
    /// on background LOD traversal, DEMs, or `.ssky` cubemap disk/network fetches.
    pub fn build_frame_zero_bootstrap(
        input: &EarthInput,
        source: Option<&dyn TileSource>,
        plugins: &EarthPluginRegistry,
    ) -> (Arc<RetainedTerrainFrame>, RetainedBuildStats) {
        let ll = input.low_level_state();
        let flat_t = ll.local_25d_blend.max(ll.map_2d_morph);
        let morph = stabilized_morph_params(
            input.camera.focus,
            flat_t as f64,
            ll.map_2d_morph as f64,
            input.tile_projection,
        );
        let camera = Self::camera_from_input_with_plugins(
            input,
            plugins,
            input.weather.weather_time_s,
            1,
            false,
        );
        let frame_ctx = EarthPluginFrameContext {
            input,
            low_level: ll,
            altitude_m: input.camera.altitude_m(),
            ground_elevation_m: 0.0,
            time_s: input.weather.weather_time_s,
            overlay_rev: 1,
            is_dragging: false,
            weather_snapshot: None,
        };
        let mut env = build_environment_assets_with_skybox(
            Some(crate::astro::embedded_clouds_fallback()),
            None,
            1,
        );
        if plugins
            .active_hooks()
            .intersects(EarthPluginHooks::ENVIRONMENT)
        {
            plugins.run_prepare_environment(&frame_ctx, Arc::make_mut(&mut env));
        }

        let mut meshes = Vec::new();
        let mut instances = Vec::new();
        let mut textures = Vec::new();
        let mut stats = RetainedBuildStats::default();

        if input.layers.terrain {
            let root = TileKey::new(0, 0, 0);
            let key_id = TileKeyId::quadtree(0, 0, 0);
            let bounds = match input.tile_projection {
                TileProjection::WebMercator => tile_bounds_mercator(root),
                TileProjection::Geographic4326 => tile_bounds(root),
            };
            let grid = input.tile_grid.clamp(2, 17);
            let with_skirts = input
                .terrain_skirts
                .unwrap_or(ll.active_surface_mode == SurfaceMode::QuadtreeLod);
            let mut mesh = build_tile_local_rtc_mesh_morphed(
                root,
                bounds,
                grid,
                None,
                root,
                with_skirts,
                0.0,
                1,
                key_id.face,
                morph,
            );
            mesh.key = key_id;
            let center_ecef = mesh.center_ecef;
            meshes.push(Arc::new(mesh));
            stats.mesh_cache_misses = 1;

            let root_img = source
                .and_then(|s| s.peek_cached_imagery(root))
                .or_else(|| crate::basemap::embedded_base_tile(root));
            let (texture_id, texture_revision, tint) = if let Some(img) = root_img {
                let tid = 0xFFFF_E47F_0000_0001u64;
                textures.push(Arc::new(RetainedTileTexture {
                    id: tid,
                    revision: 1,
                    width: img.width,
                    height: img.height,
                    rgba8: Arc::new(img.pixels),
                }));
                stats.texture_cache_misses = 1;
                (Some(tid), 1, [1.0, 1.0, 1.0, 1.0])
            } else {
                (
                    None,
                    0,
                    [
                        input.fallback_tint.r(),
                        input.fallback_tint.g(),
                        input.fallback_tint.b(),
                        input.fallback_tint.a(),
                    ],
                )
            };

            let root_unlit =
                is_canonical_2d_morph(ll.map_2d_morph as f64) || ll.map_2d_morph >= 0.99;
            instances.push(RetainedTileInstance {
                key: key_id,
                mesh_revision: 1,
                center_ecef,
                texture_id,
                texture_revision,
                uv_rect: [0.0, 0.0, 1.0, 1.0],
                tint,
                unlit: root_unlit,
                night_texture_id: None,
                night_texture_revision: 0,
                night_uv_rect: [0.0, 0.0, 1.0, 1.0],
            });
            stats.visible_tiles = 1;
        }

        let frame = Arc::new(RetainedTerrainFrame {
            camera,
            geometry_revision: 1,
            geo_revision: 0,
            meshes,
            instances,
            textures,
            dynamic_meshes: Vec::new(),
            geo_mesh_assets: Arc::default(),
            geo_instances: Arc::default(),
            particle_layers: Arc::default(),
            particle_textures: Arc::default(),
            ocean: None,
            environment: Some(env),
        });
        (frame, stats)
    }

    /// Builds a [`RetainedTerrainFrame`] from cached tile-local RTC geometry & textures.
    pub fn build_frame(
        &mut self,
        input: &EarthInput,
        source: &mut dyn TileSource,
        geometry_revision: u64,
        allow_ancestor_mesh_fallback: bool,
    ) -> (Arc<RetainedTerrainFrame>, RetainedBuildStats) {
        self.build_frame_with_idle(
            input,
            source,
            geometry_revision,
            allow_ancestor_mesh_fallback,
            0.0,
        )
    }

    /// Builds a [`RetainedTerrainFrame`] with an optional progressive idle high-res refinement
    /// factor (`0.0` = normal moving LOD; `0.5..=1.0` = deeper gaze-centered idle refinement
    /// when `input.idle_high_res.enabled` is `true`).
    pub fn build_frame_with_idle(
        &mut self,
        input: &EarthInput,
        source: &mut dyn TileSource,
        geometry_revision: u64,
        allow_ancestor_mesh_fallback: bool,
        idle_factor: f32,
    ) -> (Arc<RetainedTerrainFrame>, RetainedBuildStats) {
        self.frame_counter = self.frame_counter.wrapping_add(1);
        self.last_plugin_rev = self.plugins.revision();
        self.last_ocean_waves = input.ocean_waves.clone();
        source.begin_frame(input.camera.focus, input.camera.altitude_m());
        let ll = input.low_level_state();
        let ground_elev_focus = if input.elevation_enabled {
            self.sample_ground_elevation_m(input.camera.focus.lat_deg, input.camera.focus.lon_deg)
        } else {
            0.0
        };
        let weather_snapshot = evaluate_base_weather_snapshot(
            input,
            ground_elev_focus,
            self.weather_provider.as_ref(),
            Some(&self.live_weather),
        );
        let frame_ctx = EarthPluginFrameContext {
            input,
            low_level: ll,
            altitude_m: input.camera.altitude_m(),
            ground_elevation_m: ground_elev_focus,
            time_s: input.weather.weather_time_s,
            overlay_rev: self.weather_rev.wrapping_add(1),
            is_dragging: false,
            weather_snapshot,
        };
        let camera = Self::camera_from_input_with_plugins_weather_and_ground(
            input,
            &self.plugins,
            input.weather.weather_time_s,
            self.weather_rev.wrapping_add(1),
            false,
            weather_snapshot,
            ground_elev_focus,
        );
        let (environment, sat_elements) = self.resolve_environment_assets(input, &frame_ctx);
        let mut stats = RetainedBuildStats::default();
        let mut meshes_out = Vec::new();
        let mut instances_out = Vec::new();
        let mut textures_out = Vec::new();
        let mut seen_tex_ids = HashSet::new();
        let mut seen_mesh_keys = HashSet::new();
        let mut frame_elevs: HashMap<TileKey, Arc<TileElevation>> = HashMap::new();
        let mut loaded_imagery: HashMap<TileKey, Arc<RetainedTileTexture>> = HashMap::new();
        let mut emitted_meta: Vec<(TileKey, TileKeyId, PatchBounds, bool, bool)> = Vec::new();
        let mut visible_leaves: Vec<TileKey> = Vec::new();

        let effective_mode = ll.active_surface_mode;
        let grid = input.tile_grid.max(2);
        let with_skirts = input
            .terrain_skirts
            .unwrap_or(effective_mode == SurfaceMode::QuadtreeLod);
        let fallback_tint = [
            input.fallback_tint.r(),
            input.fallback_tint.g(),
            input.fallback_tint.b(),
            input.fallback_tint.a(),
        ];
        let tile_hooks_active = self.plugins.has_tile_hooks();

        if input.layers.terrain {
            match effective_mode {
            SurfaceMode::QuadtreeLod => {
                let morph = stabilized_morph_params(
                    input.camera.focus,
                    (ll.local_25d_blend.max(ll.map_2d_morph)) as f64,
                    ll.map_2d_morph as f64,
                    input.tile_projection,
                );
                let morph_sig = mesh_morph_signature(morph);
                let horizon_cull = ll.map_2d_morph <= 1e-4;
                let up_override = if ll.map_2d_morph > 1e-4 {
                    None
                } else {
                    input.camera_up_ecef.map(|u| DVec3::new(u[0], u[1], u[2]))
                };
                let is_flat_2d = is_canonical_2d_morph(ll.map_2d_morph as f64)
                    && (ll.local_25d_blend.max(ll.map_2d_morph) as f64)
                        >= CANONICAL_2D_MORPH_THRESHOLD;
                let eff_elev_enabled =
                    input.elevation_enabled && !is_flat_2d && input.camera.altitude_m() <= 350_000.0;
                let idle_lod = input.idle_high_res.to_lod_params(idle_factor);
                let leaf_items: Vec<(TileKey, i32)> = if is_flat_2d {
                    let eff_sse = if idle_lod.is_active() {
                        (input.sse_threshold * idle_lod.sse_scale).max(0.5)
                    } else {
                        input.sse_threshold
                    };
                    let leaf_cap = MAX_QUADTREE_LEAVES
                        + if idle_lod.is_active() {
                            idle_lod.extra_leaves
                        } else {
                            0
                        };
                    traverse_quadtree_lod_2d_flat(
                        &input.camera,
                        input.viewport.width.max(1.0) as f64,
                        input.viewport.height.max(1.0) as f64,
                        eff_sse,
                        input.max_tile_level,
                        grid,
                        leaf_cap,
                        input.tile_projection,
                    )
                } else {
                    let src_ro: &dyn TileSource = &*source;
                    traverse_quadtree_lod_frustum6_with_relief(
                        &input.camera,
                        up_override,
                        input.viewport.width.max(1.0) as f64,
                        input.viewport.height.max(1.0) as f64,
                        input.sse_threshold,
                        input.max_tile_level,
                        grid,
                        horizon_cull,
                        MAX_QUADTREE_LEAVES,
                        input.tile_projection,
                        idle_lod,
                        eff_elev_enabled,
                        if eff_elev_enabled {
                            ground_elev_focus
                        } else {
                            0.0
                        },
                        |k| self.tile_average_altitude_m(k, eff_elev_enabled, src_ro),
                    )
                    .into_iter()
                    .map(|k| (k, 0))
                    .collect()
                };
                let cam_placement = input.camera.placement_with_up(up_override);
                let cam_eye_ecef = cam_placement.eye;
                let cam_forward_ecef = cam_placement.aim.sub(cam_placement.eye).normalize();
                let focus_east = LocalFrame::at(morph.focus.lat_deg, morph.focus.lon_deg).east;
                let world_circ = 2.0 * std::f64::consts::PI * wgs84::A;
                let mut leaf_flags: Vec<(bool, bool)> = vec![(false, eff_elev_enabled); leaf_items.len()];
                if tile_hooks_active && !leaf_items.is_empty() {
                    let mut ctxs: Vec<EarthTileBuildContext<'_>> = leaf_items
                        .iter()
                        .map(|&(leaf, _)| {
                            let bounds = match input.tile_projection {
                                TileProjection::WebMercator => tile_bounds_mercator(leaf),
                                TileProjection::Geographic4326 => tile_bounds(leaf),
                            };
                            EarthTileBuildContext {
                                input,
                                key: leaf,
                                key_id: TileKeyId::quadtree(leaf.level, leaf.x, leaf.y),
                                bounds,
                                want_night_tiles: false,
                                elevation_enabled: eff_elev_enabled,
                            }
                        })
                        .collect();
                    self.plugins.run_tile_begin_batch(&mut ctxs);
                    for (dst, c) in leaf_flags.iter_mut().zip(ctxs.iter()) {
                        *dst = (c.want_night_tiles, c.elevation_enabled);
                    }
                }
                // Pass 1: resolve native/child/ancestor elevations for every unique visible leaf.
                let mut working_elevs: HashMap<TileKey, ResolvedLeafElev> =
                    HashMap::with_capacity(leaf_items.len());
                let mut any_missing_elev = false;
                let mut any_coarse_elev = false;
                for (&(leaf, _), &(_, leaf_elev_enabled)) in
                    leaf_items.iter().zip(leaf_flags.iter())
                {
                    if working_elevs.contains_key(&leaf) {
                        continue;
                    }
                    if !leaf_elev_enabled {
                        working_elevs.insert(
                            leaf,
                            ResolvedLeafElev {
                                elev_tile: leaf,
                                is_exact: true,
                                is_synthesized: false,
                                elev_err: false,
                                elev: None,
                                elev_hash: 0,
                            },
                        );
                        continue;
                    }
                    let elev_lookup = self.resolve_elevation_or_ancestor(leaf, true, source);
                    let elev_err = elev_lookup.is_err();
                    let (mut elev_opt, mut elev_tile) = elev_lookup.unwrap_or((None, leaf));
                    let exact_level = leaf.level.min(15);
                    let mut is_exact = elev_opt.is_some() && elev_tile.level >= exact_level;
                    let mut is_synthesized = false;

                    // Zoom-out fast path: if exact DEM isn't loaded yet but all 4 child DEMs are
                    // already cached, downsample them directly instead of dropping to coarse/flat.
                    if !is_exact && !is_flat_2d {
                        if let Some(down) = Self::downsample_cached_children(leaf, grid, source) {
                            elev_opt = Some(down);
                            elev_tile = leaf;
                            is_exact = true;
                            is_synthesized = true;
                        }
                    }

                    let elev_hash = hash_elevation_64(elev_opt.as_ref());
                    let elev_arc = if let Some(ref e) = elev_opt {
                        if !is_synthesized {
                            record_frame_elevation(
                                &mut self.cached_elev_arcs,
                                &mut frame_elevs,
                                elev_tile,
                                elev_hash,
                                Some(e),
                            );
                            frame_elevs
                                .get(&elev_tile)
                                .cloned()
                                .or_else(|| Some(Arc::new(e.clone())))
                        } else {
                            let arc = Arc::new(e.clone());
                            frame_elevs.entry(leaf).or_insert_with(|| Arc::clone(&arc));
                            Some(arc)
                        }
                    } else {
                        None
                    };

                    if elev_arc.is_none() {
                        any_missing_elev = true;
                    } else if !is_exact && elev_tile.level < leaf.level {
                        any_coarse_elev = true;
                    }

                    working_elevs.insert(
                        leaf,
                        ResolvedLeafElev {
                            elev_tile,
                            is_exact,
                            is_synthesized,
                            elev_err,
                            elev: elev_arc,
                            elev_hash,
                        },
                    );
                }

                // Pass 2: seamlessly adapt unloaded tiles (and coarse-ancestor edges bordering
                // finer neighbors) to surrounding loaded heightmaps via Coons-patch interpolation,
                // making exposed cliff gaps / undermap holes impossible while DEMs stream in.
                if !is_flat_2d && (any_missing_elev || any_coarse_elev) {
                    let src_ro: &dyn TileSource = &*source;
                    let mut peek_cache: HashMap<TileKey, Option<Arc<TileElevation>>> =
                        HashMap::new();
                    let unique_leaves: Vec<TileKey> = working_elevs.keys().copied().collect();

                    // Pass 2A: up to 2 wave passes synthesize completely unloaded tiles from loaded
                    // cardinal edges & diagonal corners (wave 1 bridges multi-tile missing clusters).
                    if any_missing_elev {
                        for wave in 0..2u8 {
                            let allow_synth = wave > 0;
                            let mut updates: Vec<(TileKey, Arc<TileElevation>, u64)> = Vec::new();
                            for &leaf in &unique_leaves {
                                if leaf.level == 0 {
                                    continue;
                                }
                                let needs_synth = working_elevs
                                    .get(&leaf)
                                    .is_some_and(|w| !w.is_exact && w.elev.is_none());
                                if !needs_synth {
                                    continue;
                                }
                                let edges = sample_neighbor_edges_for_leaf(
                                    leaf,
                                    grid,
                                    0,
                                    allow_synth,
                                    true,
                                    None,
                                    &working_elevs,
                                    &mut peek_cache,
                                    &self.cached_elev_arcs,
                                    src_ro,
                                );
                                if let Some(synth) = synthesize_elevation_from_neighbors(grid, &edges)
                                {
                                    let h = hash_elevation_64(Some(&synth));
                                    updates.push((leaf, Arc::new(synth), h));
                                }
                            }
                            if updates.is_empty() {
                                break;
                            }
                            for (leaf, arc, h) in updates {
                                frame_elevs.entry(leaf).or_insert_with(|| Arc::clone(&arc));
                                if let Some(entry) = working_elevs.get_mut(&leaf) {
                                    entry.elev_tile = leaf;
                                    entry.is_synthesized = true;
                                    entry.elev = Some(arc);
                                    entry.elev_hash = h;
                                }
                            }
                        }

                        // Pass 2C safety net: if an isolated tile still has zero touching neighbors
                        // while other tiles in the same frame have elevation loaded (or focus ground
                        // height is known), lift it smoothly to the regional mean instead of 0.0 m.
                        let still_missing: Vec<TileKey> = unique_leaves
                            .iter()
                            .copied()
                            .filter(|k| {
                                working_elevs
                                    .get(k)
                                    .is_some_and(|w| !w.is_exact && w.elev.is_none())
                            })
                            .collect();
                        if !still_missing.is_empty() {
                            let mut sum_m = 0.0f64;
                            let mut cnt = 0u64;
                            for entry in working_elevs.values() {
                                if let Some(ref e) = entry.elev {
                                    let avg = e.average_elevation_m();
                                    if avg.is_finite() {
                                        sum_m += avg;
                                        cnt += 1;
                                    }
                                }
                            }
                            let fallback_h = if cnt > 0 {
                                Some((sum_m / cnt as f64) as f32)
                            } else if ground_elev_focus.is_finite() && ground_elev_focus.abs() > 1.0
                            {
                                Some(ground_elev_focus as f32)
                            } else {
                                None
                            };
                            if let Some(fh) = fallback_h {
                                let flat_elev = TileElevation {
                                    width: 2,
                                    height: 2,
                                    heights_m: vec![fh; 4],
                                };
                                let h = hash_elevation_64(Some(&flat_elev));
                                let arc = Arc::new(flat_elev);
                                for leaf in still_missing {
                                    frame_elevs.entry(leaf).or_insert_with(|| Arc::clone(&arc));
                                    if let Some(entry) = working_elevs.get_mut(&leaf) {
                                        entry.elev_tile = leaf;
                                        entry.is_synthesized = true;
                                        entry.elev = Some(Arc::clone(&arc));
                                        entry.elev_hash = h;
                                    }
                                }
                            }
                        }
                    }

                    // Pass 2B: stitch coarse-ancestor tiles along any edge shared with a strictly
                    // finer loaded neighbor so LOD/download borders stay 0.0 m watertight while
                    // keeping 100% of the ancestor's interior terrain relief.
                    if any_coarse_elev {
                        let mut stitched_updates: Vec<(TileKey, Arc<TileElevation>, u64)> =
                            Vec::new();
                        for &leaf in &unique_leaves {
                            if leaf.level == 0 {
                                continue;
                            }
                            let Some(entry) = working_elevs.get(&leaf) else {
                                continue;
                            };
                            if entry.is_exact || entry.elev_tile.level >= leaf.level {
                                continue;
                            }
                            let Some(ref base_arc) = entry.elev else {
                                continue;
                            };
                            let uv_rect = tile_sub_uv_in_ancestor(leaf, entry.elev_tile);
                            let min_finer = entry.elev_tile.level.saturating_add(1);
                            let edges = sample_neighbor_edges_for_leaf(
                                leaf,
                                grid,
                                min_finer,
                                false,
                                false,
                                Some((base_arc.as_ref(), uv_rect)),
                                &working_elevs,
                                &mut peek_cache,
                                &self.cached_elev_arcs,
                                src_ro,
                            );
                            if edges.has_any_edge() {
                                let stitched = stitch_elevation_with_neighbors(
                                    base_arc.as_ref(),
                                    uv_rect,
                                    grid,
                                    &edges,
                                );
                                let h = hash_elevation_64(Some(&stitched));
                                stitched_updates.push((leaf, Arc::new(stitched), h));
                            }
                        }
                        for (leaf, arc, h) in stitched_updates {
                            frame_elevs.insert(leaf, Arc::clone(&arc));
                            if let Some(entry) = working_elevs.get_mut(&leaf) {
                                entry.elev_tile = leaf;
                                entry.is_synthesized = true;
                                entry.elev = Some(arc);
                                entry.elev_hash = h;
                            }
                        }
                    }
                }

                let mut emitted_ancestors = HashSet::new();
                for ((leaf, wrap_x), (leaf_want_night, leaf_elev_enabled)) in
                    leaf_items.into_iter().zip(leaf_flags.into_iter())
                {
                    visible_leaves.push(leaf);
                    let resolved = working_elevs.get(&leaf);
                    let elev_err = resolved.is_some_and(|r| r.elev_err);
                    let elev_tile = resolved.map(|r| r.elev_tile).unwrap_or(leaf);
                    let elev = resolved.and_then(|r| r.elev.as_deref());
                    let elev_hash = resolved.map(|r| r.elev_hash).unwrap_or(0);
                    let key_id = TileKeyId::quadtree(leaf.level, leaf.x, leaf.y);

                    let use_ancestor_mesh = !is_flat_2d
                        && allow_ancestor_mesh_fallback
                        && elev_err
                        && elev.is_none()
                        && !self
                            .meshes
                            .get(&(key_id, grid, with_skirts))
                            .is_some_and(|c| {
                                c.elev_tile == elev_tile
                                    && c.elev_hash == elev_hash
                                    && c.morph_sig == morph_sig
                            })
                        && has_resident_ancestor(&self.meshes, leaf, grid, with_skirts, morph_sig);

                    if use_ancestor_mesh {
                        if let Some((anc_key, anc_mesh)) =
                            find_resident_ancestor(&self.meshes, leaf, grid, with_skirts, morph_sig)
                        {
                            stats.ancestor_fallbacks += 1;
                            let anc_id = anc_mesh.key;
                            if emitted_ancestors.insert(anc_id) {
                                if seen_mesh_keys.insert(anc_id) {
                                    meshes_out.push(Arc::clone(&anc_mesh));
                                }
                                let anc_bounds = match input.tile_projection {
                                    TileProjection::WebMercator => tile_bounds_mercator(anc_key),
                                    TileProjection::Geographic4326 => tile_bounds(anc_key),
                                };
                                self.emit_tile_instance(
                                    anc_key,
                                    anc_key,
                                    anc_id,
                                    anc_mesh.revision,
                                    anc_mesh.center_ecef,
                                    anc_bounds,
                                    leaf_want_night,
                                    leaf_elev_enabled,
                                    is_flat_2d || ll.map_2d_morph >= 0.99,
                                    fallback_tint,
                                    source,
                                    input.max_tile_level,
                                    &mut stats,
                                    &mut textures_out,
                                    &mut seen_tex_ids,
                                    &mut loaded_imagery,
                                    &mut instances_out,
                                    &mut emitted_meta,
                                );
                            }
                            continue;
                        }
                    }

                    let bounds = match input.tile_projection {
                        TileProjection::WebMercator => tile_bounds_mercator(leaf),
                        TileProjection::Geographic4326 => tile_bounds(leaf),
                    };
                    let is_tile_2d_unlit = is_flat_2d || ll.map_2d_morph >= 0.99;
                    let eye_alt_m =
                        (input.camera.altitude_m() - ground_elev_focus.max(0.0)).max(1.0);
                    let leaf_grid = Self::close_ground_leaf_grid(
                        grid,
                        is_flat_2d,
                        eff_elev_enabled,
                        eye_alt_m,
                        cam_eye_ecef,
                        cam_forward_ecef,
                        input.camera.focus.lat_deg,
                        input.camera.focus.lon_deg,
                        bounds,
                    );
                    let mesh = self.ensure_tile_mesh(
                        leaf,
                        key_id,
                        bounds,
                        leaf_grid,
                        with_skirts,
                        elev,
                        elev_tile,
                        elev_hash,
                        morph,
                        morph_sig,
                        &mut stats,
                    );

                    if seen_mesh_keys.insert(mesh.key) {
                        meshes_out.push(Arc::clone(&mesh));
                    }
                    let center_ecef = if wrap_x == 0 {
                        mesh.center_ecef
                    } else {
                        let shift_m = (wrap_x as f64) * world_circ;
                        [
                            mesh.center_ecef[0] + focus_east.x * shift_m,
                            mesh.center_ecef[1] + focus_east.y * shift_m,
                            mesh.center_ecef[2] + focus_east.z * shift_m,
                        ]
                    };
                    self.emit_tile_instance(
                        leaf,
                        leaf,
                        mesh.key,
                        mesh.revision,
                        center_ecef,
                        bounds,
                        leaf_want_night,
                        leaf_elev_enabled,
                        is_tile_2d_unlit,
                        fallback_tint,
                        source,
                        input.max_tile_level,
                        &mut stats,
                        &mut textures_out,
                        &mut seen_tex_ids,
                        &mut loaded_imagery,
                        &mut instances_out,
                        &mut emitted_meta,
                    );
                }
            }
            SurfaceMode::FlatPatches | SurfaceMode::SphericalGlobe => {
                let morph = stabilized_morph_params(
                    input.camera.focus,
                    ll.map_2d_morph as f64,
                    ll.map_2d_morph as f64,
                    TileProjection::Geographic4326,
                );
                let morph_sig = mesh_morph_signature(morph);
                let frustum = Frustum6::from_camera(
                    &input.camera,
                    input.viewport.width.max(1.0) as f64,
                    input.viewport.height.max(1.0) as f64,
                );
                let mut patch_specs: Vec<(TileKey, PatchBounds, TileKeyId)> = Vec::new();
                if effective_mode == SurfaceMode::SphericalGlobe {
                    let level = level_for_altitude(input.camera.altitude_m(), 2);
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
                            patch_specs.push((key, bounds, TileKeyId::new(level, x, y, 2)));
                        }
                    }
                } else {
                    let alt_m = input.camera.altitude_m();
                    let span_deg = tile_patch_span_deg(&input.camera, input.patch_scale);
                    let patches = surrounding_patches(input.camera.focus, span_deg);
                    let level = level_for_altitude(alt_m, input.max_tile_level);
                    for patch in patches {
                        let tile = requested_tile(&patch);
                        patch_specs.push((
                            tile,
                            patch.bounds,
                            flat_patch_key_id(patch.bounds, level),
                        ));
                    }
                }

                let mut patch_flags: Vec<(bool, bool)> =
                    vec![(false, input.elevation_enabled); patch_specs.len()];
                if tile_hooks_active && !patch_specs.is_empty() {
                    let mut ctxs: Vec<EarthTileBuildContext<'_>> = patch_specs
                        .iter()
                        .map(|&(tile, bounds, key_id)| EarthTileBuildContext {
                            input,
                            key: tile,
                            key_id,
                            bounds,
                            want_night_tiles: false,
                            elevation_enabled: input.elevation_enabled,
                        })
                        .collect();
                    self.plugins.run_tile_begin_batch(&mut ctxs);
                    for (dst, c) in patch_flags.iter_mut().zip(ctxs.iter()) {
                        *dst = (c.want_night_tiles, c.elevation_enabled);
                    }
                }

                for ((tile, bounds, key_id), (patch_want_night, patch_elev_enabled)) in
                    patch_specs.into_iter().zip(patch_flags.into_iter())
                {
                    let (_, radius) = tile_bounding_sphere(tile);
                    let mid_lat = (bounds.lat0 + bounds.lat1) * 0.5;
                    let mid_lon = (bounds.lon0 + bounds.lon1) * 0.5;
                    let patch_center: DVec3 = crate::geodesy::geodetic_to_ecef(
                        crate::geodesy::Geodetic::new(mid_lat.clamp(-89.9, 89.9), mid_lon, 0.0),
                    );
                    if ll.map_2d_morph <= 0.05
                        && !frustum.intersects_sphere(patch_center, radius.max(50_000.0) * 1.25)
                    {
                        continue;
                    }
                    visible_leaves.push(tile);

                    let (elev_opt, elev_tile) = self
                        .resolve_elevation_or_ancestor(tile, patch_elev_enabled, source)
                        .unwrap_or((None, tile));
                    let elev = elev_opt.as_ref();
                    let elev_hash = hash_elevation_64(elev);
                    record_frame_elevation(
                        &mut self.cached_elev_arcs,
                        &mut frame_elevs,
                        elev_tile,
                        elev_hash,
                        elev,
                    );

                    let eye_alt_m =
                        (input.camera.altitude_m() - ground_elev_focus.max(0.0)).max(1.0);
                    let patch_grid = if effective_mode == SurfaceMode::FlatPatches {
                        let up_ov = input
                            .camera_up_ecef
                            .map(|u| DVec3::new(u[0], u[1], u[2]));
                        let placement = input.camera.placement_with_up(up_ov);
                        let forward = placement.aim.sub(placement.eye).normalize();
                        Self::close_ground_leaf_grid(
                            grid,
                            ll.map_2d_morph >= 0.95,
                            patch_elev_enabled,
                            eye_alt_m,
                            placement.eye,
                            forward,
                            input.camera.focus.lat_deg,
                            input.camera.focus.lon_deg,
                            bounds,
                        )
                    } else {
                        grid
                    };
                    let mesh = self.ensure_tile_mesh(
                        tile,
                        key_id,
                        bounds,
                        patch_grid,
                        with_skirts,
                        elev,
                        elev_tile,
                        elev_hash,
                        morph,
                        morph_sig,
                        &mut stats,
                    );

                    if seen_mesh_keys.insert(mesh.key) {
                        meshes_out.push(Arc::clone(&mesh));
                    }
                    self.emit_tile_instance(
                        tile,
                        tile,
                        mesh.key,
                        mesh.revision,
                        mesh.center_ecef,
                        bounds,
                        patch_want_night,
                        patch_elev_enabled,
                        ll.map_2d_morph >= 0.99,
                        fallback_tint,
                        source,
                        input.max_tile_level,
                        &mut stats,
                        &mut textures_out,
                        &mut seen_tex_ids,
                        &mut loaded_imagery,
                        &mut instances_out,
                        &mut emitted_meta,
                    );
                }
            }
            }
        }

        source.end_frame();
        self.prune_lru_caches();

        // Mid-work Stage 5: per-tile instance & procedural texture injection.
        if tile_hooks_active && !instances_out.is_empty() {
            let ctxs: Vec<EarthTileBuildContext<'_>> = emitted_meta
                .iter()
                .map(|&(key, key_id, bounds, want_night_tiles, elevation_enabled)| {
                    EarthTileBuildContext {
                        input,
                        key,
                        key_id,
                        bounds,
                        want_night_tiles,
                        elevation_enabled,
                    }
                })
                .collect();
            let mut injector = EarthTextureInjector {
                textures_out: &mut textures_out,
                seen_tex_ids: &mut seen_tex_ids,
                cache: &mut self.plugin_textures,
                next_tex_id: &mut self.next_tex_id,
                frame_counter: self.frame_counter,
            };
            self.plugins
                .run_tile_instance_batch(&ctxs, &mut instances_out, &mut injector);
        }

        // Publish updated ground elevation snapshot prior to WORLD_BUILD and DYNAMIC_OVERLAYS
        // so both ocean shorelines and custom 3D placement plugins observe exact frame heights.
        let mut patches: Vec<(TileKey, Arc<TileElevation>)> = frame_elevs.into_iter().collect();
        patches.sort_by(|a, b| b.0.level.cmp(&a.0.level));
        if self.cached_elev_arcs.len() > 512 {
            let keep: HashSet<TileKey> = patches.iter().map(|(k, _)| *k).collect();
            self.cached_elev_arcs.retain(|k, _| keep.contains(k));
        }
        let ground_snap_val = GroundElevSnapshot {
            projection: input.tile_projection,
            patches,
        };
        if input.elevation_enabled {
            if let Some(ref snap_lock) = self.ground_snapshot {
                if let Ok(mut guard) = snap_lock.write() {
                    *guard = ground_snap_val.clone();
                }
            }
        }

        let mut dynamic_meshes = Vec::new();
        let hooks = self.plugins.active_hooks();
        if hooks.intersects(EarthPluginHooks::WORLD_BUILD) {
            let mut tiles: Vec<(TileKey, Arc<RetainedTileTexture>)> =
                loaded_imagery.into_iter().collect();
            tiles.sort_by(|a, b| b.0.level.cmp(&a.0.level));
            let imagery_snapshot = LoadedTileImagerySnapshot {
                projection: input.tile_projection,
                tiles,
            };
            let mut world_ctx = EarthWorldBuildContext {
                input,
                camera: &camera,
                low_level: ll,
                visible_leaves: &visible_leaves,
                meshes: &mut meshes_out,
                instances: &mut instances_out,
                textures: &mut textures_out,
                dynamic_meshes: &mut dynamic_meshes,
                ground_snapshot: Some(&ground_snap_val),
                imagery_snapshot: Some(&imagery_snapshot),
                tex_seen_ids: &mut seen_tex_ids,
                plugin_tex_cache: &mut self.plugin_textures,
                next_tex_id: &mut self.next_tex_id,
                frame_counter: self.frame_counter,
            };
            self.plugins.run_world_build(&mut world_ctx);
        }

        let flat_t = ll.local_25d_blend.max(ll.map_2d_morph);
        self.weather_rev = self.weather_rev.wrapping_add(1);
        if hooks.intersects(EarthPluginHooks::DYNAMIC_OVERLAYS) {
            let mut dyn_ctx = EarthDynamicOverlayContext {
                input,
                low_level: ll,
                flat_t,
                time_s: input.weather.weather_time_s,
                overlay_rev: self.weather_rev,
                satellite_elements: &sat_elements,
                weather_ids: &mut self.weather_ids,
                ground_snapshot: self.ground_snapshot.as_deref(),
            };
            self.plugins
                .run_dynamic_overlays(&mut dyn_ctx, &mut dynamic_meshes);
        }

        let mut particle_layers = Vec::new();
        let mut particle_textures = Vec::new();
        if self.include_particles && hooks.intersects(EarthPluginHooks::PARTICLES) {
            let mut p_ctx = EarthParticleContext {
                input,
                low_level: ll,
                time_s: input.weather.weather_time_s,
                weather_snapshot,
                ground_snapshot: self.ground_snapshot.as_deref(),
            };
            self.plugins
                .run_particle_layers(&mut p_ctx, &mut particle_layers, &mut particle_textures);
        }

        let ocean = if self.include_ocean
            && ll.map_2d_morph <= 0.05
            && hooks.intersects(EarthPluginHooks::OCEAN)
        {
            let mut o_ctx = EarthOceanContext {
                input,
                low_level: ll,
                time_s: input.weather.weather_time_s,
                overlay_rev: self.weather_rev,
                weather_snapshot,
                ground_snapshot: self.ground_snapshot.as_deref(),
            };
            self.plugins.run_ocean_surface(&mut o_ctx).map(Arc::new)
        } else {
            None
        };

        let morph = stabilized_morph_params(
            input.camera.focus,
            flat_t as f64,
            ll.map_2d_morph as f64,
            input.tile_projection,
        );
        let eye_dvec = DVec3::new(camera.eye_ecef[0], camera.eye_ecef[1], camera.eye_ecef[2]);
        if input.layers.buildings {
            let built_buildings = if let Some(ref mut bsrc) = self.building_source {
                build_buildings_dynamic_mesh(
                    self.dynamic_mesh_id_base + 1,
                    self.frame_counter,
                    input,
                    morph,
                    &emitted_meta,
                    &ground_snap_val,
                    bsrc.as_mut(),
                )
            } else {
                None
            };
            if let Some(ref slot) = self.last_buildings_mesh {
                if let Ok(mut guard) = slot.write() {
                    *guard = built_buildings.clone();
                }
            }
            if let Some(m) = built_buildings {
                dynamic_meshes.push(m);
            }
        } else if let Some(ref slot) = self.last_buildings_mesh {
            if let Ok(mut guard) = slot.write() {
                *guard = None;
            }
        }
        if input.layers.graticule && !self.plugins.list_ids().is_empty() {
            if let Some(grat) = build_graticule_dynamic_mesh(
                self.dynamic_mesh_id_base + 2,
                self.frame_counter,
                input,
                morph,
                eye_dvec,
            ) {
                dynamic_meshes.push(grat);
            }
        }
        let (geo_mesh_assets, geo_instances, geo_revision) = if let Ok(mut w) = self.world.write() {
            if w.is_empty() {
                (Arc::default(), Arc::default(), w.revision())
            } else {
                let (assets, insts) = w.compile_retained(
                    eye_dvec,
                    input.viewport.height.max(1.0) as f64,
                    input.camera.fov_y_deg,
                    input.elevation_enabled,
                    morph,
                    &|lat, lon| {
                        if input.elevation_enabled {
                            ground_snap_val.sample_elevation_m(lat, lon)
                        } else {
                            0.0
                        }
                    },
                );
                (assets, insts, w.revision())
            }
        } else {
            (Arc::default(), Arc::default(), 0)
        };

        stats.visible_tiles = instances_out.len();
        (
            Arc::new(RetainedTerrainFrame {
                camera,
                geometry_revision,
                geo_revision,
                meshes: meshes_out,
                instances: instances_out,
                textures: textures_out,
                dynamic_meshes,
                geo_mesh_assets,
                geo_instances,
                particle_layers: Arc::new(particle_layers),
                particle_textures: Arc::new(particle_textures),
                ocean,
                environment,
            }),
            stats,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn ensure_tile_mesh(
        &mut self,
        tile: TileKey,
        key_id: TileKeyId,
        bounds: PatchBounds,
        grid: u32,
        with_skirts: bool,
        elev: Option<&TileElevation>,
        elev_tile: TileKey,
        elev_hash: u64,
        morph: SurfaceMorphParams,
        morph_sig: u64,
        stats: &mut RetainedBuildStats,
    ) -> Arc<RetainedTileMesh> {
        let frame_now = self.frame_counter;
        if let Some(cached) = self.meshes.get_mut(&(key_id, grid, with_skirts)) {
            if cached.elev_tile == elev_tile
                && cached.elev_hash == elev_hash
                && cached.morph_sig == morph_sig
            {
                cached.last_used_frame = frame_now;
                stats.mesh_cache_hits += 1;
                return Arc::clone(&cached.mesh);
            }
        }
        stats.mesh_cache_misses += 1;
        let rev = self.next_revision;
        self.next_revision += 1;
        let min_skirt = if morph.flat_blend > 0.05 {
            600.0
        } else if elev.is_some() {
            350.0
        } else {
            0.0
        };
        let skirt_depth = compute_adaptive_skirt_depth(bounds, elevation_range(elev)).max(min_skirt);
        let mut built = build_tile_local_rtc_mesh_morphed(
            tile,
            bounds,
            grid,
            elev,
            elev_tile,
            with_skirts,
            skirt_depth,
            rev,
            key_id.face,
            morph,
        );
        built.key = key_id;
        let arc = Arc::new(built);
        self.meshes.insert(
            (key_id, grid, with_skirts),
            CachedMeshEntry {
                elev_tile,
                elev_hash,
                morph_sig,
                last_used_frame: frame_now,
                mesh: Arc::clone(&arc),
            },
        );
        arc
    }
}

fn ancestor_uv_point(descendant: TileKey, ancestor: TileKey, u: f32, v: f32) -> (f32, f32) {
    if descendant.level <= ancestor.level {
        return (u, v);
    }
    let dl = descendant.level - ancestor.level;
    let scale = (1u32 << dl.min(20)) as f32;
    let dx = (descendant.x - (ancestor.x << dl)) as f32;
    let dy = (descendant.y - (ancestor.y << dl)) as f32;
    ((dx + u) / scale, (dy + v) / scale)
}

fn has_resident_ancestor(
    meshes: &HashMap<(TileKeyId, u32, bool), CachedMeshEntry>,
    leaf: TileKey,
    grid: u32,
    skirts: bool,
    morph_sig: u64,
) -> bool {
    let mut cur = leaf.parent();
    while let Some(anc) = cur {
        if meshes
            .get(&(TileKeyId::quadtree(anc.level, anc.x, anc.y), grid, skirts))
            .is_some_and(|e| e.morph_sig == morph_sig)
        {
            return true;
        }
        cur = anc.parent();
    }
    false
}

fn find_resident_ancestor(
    meshes: &HashMap<(TileKeyId, u32, bool), CachedMeshEntry>,
    leaf: TileKey,
    grid: u32,
    skirts: bool,
    morph_sig: u64,
) -> Option<(TileKey, Arc<RetainedTileMesh>)> {
    let mut cur = leaf.parent();
    while let Some(anc) = cur {
        if let Some(entry) =
            meshes.get(&(TileKeyId::quadtree(anc.level, anc.x, anc.y), grid, skirts))
        {
            if entry.morph_sig == morph_sig {
                return Some((anc, Arc::clone(&entry.mesh)));
            }
        }
        cur = anc.parent();
    }
    None
}

fn push_dyn_quad(
    verts: &mut Vec<strata_render::scene3d::RteTileVertex>,
    inds: &mut Vec<u32>,
    a: DVec3,
    b: DVec3,
    c: DVec3,
    d: DVec3,
    normal: DVec3,
) {
    let base = verts.len() as u32;
    let n = [normal.x as f32, normal.y as f32, normal.z as f32];
    verts.push(strata_render::scene3d::RteTileVertex::new(
        [a.x as f32, a.y as f32, a.z as f32],
        n,
        [0.0, 0.0],
    ));
    verts.push(strata_render::scene3d::RteTileVertex::new(
        [b.x as f32, b.y as f32, b.z as f32],
        n,
        [1.0, 0.0],
    ));
    verts.push(strata_render::scene3d::RteTileVertex::new(
        [c.x as f32, c.y as f32, c.z as f32],
        n,
        [1.0, 1.0],
    ));
    verts.push(strata_render::scene3d::RteTileVertex::new(
        [d.x as f32, d.y as f32, d.z as f32],
        n,
        [0.0, 1.0],
    ));
    inds.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
}

#[allow(clippy::too_many_arguments)]
fn build_buildings_dynamic_mesh(
    id: u64,
    revision: u64,
    input: &EarthInput,
    morph: SurfaceMorphParams,
    emitted_meta: &[(TileKey, TileKeyId, PatchBounds, bool, bool)],
    ground_snap: &GroundElevSnapshot,
    source: &mut dyn crate::scene::BuildingSource,
) -> Option<RetainedDynamicMesh> {
    let (anchor_dvec, _) = crate::project_morphed_point_and_frame(
        input.camera.focus.lat_deg,
        input.camera.focus.lon_deg,
        0.0,
        morph,
    );
    let mut verts = Vec::new();
    let mut inds = Vec::new();
    let mut total_buildings = 0usize;
    'tiles: for &(tile, _, bounds, _, _) in emitted_meta {
        for fp in source.buildings(tile, bounds) {
            if total_buildings >= 4096 {
                break 'tiles;
            }
            total_buildings += 1;
            let ground_m = if input.elevation_enabled {
                ground_snap.sample_elevation_m(fp.lat, fp.lon)
            } else {
                0.0
            };
            let (base_pos, frame) =
                crate::project_morphed_point_and_frame(fp.lat, fp.lon, ground_m, morph);
            let (east, north, up) = (frame.east, frame.north, frame.up);
            let rel = base_pos.sub(anchor_dvec);
            let he = east.scale((fp.size_m.0 as f64).max(1.0) * 0.5);
            let hn = north.scale((fp.size_m.1 as f64).max(1.0) * 0.5);
            let h_vec = up.scale((fp.height_m as f64).max(1.0));
            let b00 = rel.sub(he).sub(hn);
            let b10 = rel.add(he).sub(hn);
            let b11 = rel.add(he).add(hn);
            let b01 = rel.sub(he).add(hn);
            let t00 = b00.add(h_vec);
            let t10 = b10.add(h_vec);
            let t11 = b11.add(h_vec);
            let t01 = b01.add(h_vec);
            push_dyn_quad(&mut verts, &mut inds, t00, t10, t11, t01, up);
            push_dyn_quad(&mut verts, &mut inds, b00, b10, t10, t00, north.scale(-1.0));
            push_dyn_quad(&mut verts, &mut inds, b10, b11, t11, t10, east);
            push_dyn_quad(&mut verts, &mut inds, b11, b01, t01, t11, north);
            push_dyn_quad(&mut verts, &mut inds, b01, b00, t00, t01, east.scale(-1.0));
        }
    }
    if verts.is_empty() {
        return None;
    }
    Some(RetainedDynamicMesh {
        id,
        revision,
        anchor_ecef: [anchor_dvec.x, anchor_dvec.y, anchor_dvec.z],
        vertices: Arc::new(verts),
        indices: Arc::new(inds),
        color: [0.82, 0.85, 0.90, 0.96],
        unlit: false,
    })
}

fn build_graticule_dynamic_mesh(
    id: u64,
    revision: u64,
    input: &EarthInput,
    morph: SurfaceMorphParams,
    eye_dvec: DVec3,
) -> Option<RetainedDynamicMesh> {
    let step_deg = input.graticule_step_deg;
    if !(step_deg > 0.0 && step_deg <= 90.0) {
        return None;
    }
    if morph.flat_blend >= 0.85 && morph.map_2d_morph < 0.5 {
        return None;
    }
    let (anchor_dvec, _) = crate::project_morphed_point_and_frame(
        input.camera.focus.lat_deg,
        input.camera.focus.lon_deg,
        0.0,
        morph,
    );
    let cam_dist = input.camera.eye_distance_m().max(200.0);
    let ribbon_half = (cam_dist * 0.0012).clamp(1.5, 18_000.0);
    let alt_lift = (cam_dist * 0.0005).clamp(2.0, 2_500.0);
    let cull_back = morph.flat_blend < 0.85;

    let mut verts = Vec::new();
    let mut inds = Vec::new();
    let mut emit_strip = |samples: &[(f64, f64)]| {
        let mut prev: Option<(DVec3, DVec3)> = None;
        for &(lat, lon) in samples {
            let (p_world, frame) =
                crate::project_morphed_point_and_frame(lat, lon, alt_lift, morph);
            let up = frame.up;
            if cull_back {
                let to_eye = eye_dvec.sub(p_world);
                if up.dot(to_eye) <= 0.0 {
                    prev = None;
                    continue;
                }
            }
            if let Some((p0, up0)) = prev {
                let seg = p_world.sub(p0);
                let seg_len = seg.length();
                if seg_len > 1.0 && seg_len < 3_500_000.0 {
                    let dir = seg.scale(1.0 / seg_len);
                    let side_raw = dir.cross(up0);
                    let side_len = side_raw.length();
                    if side_len > 1e-6 {
                        let side = side_raw.scale(ribbon_half / side_len);
                        let r0 = p0.sub(anchor_dvec);
                        let r1 = p_world.sub(anchor_dvec);
                        push_dyn_quad(
                            &mut verts,
                            &mut inds,
                            r0.sub(side),
                            r0.add(side),
                            r1.add(side),
                            r1.sub(side),
                            up,
                        );
                    }
                }
            }
            prev = Some((p_world, up));
        }
    };

    const MERIDIAN_SAMPLES: usize = 72;
    const PARALLEL_SAMPLES: usize = 96;
    let mut buf = Vec::with_capacity(PARALLEL_SAMPLES + 1);
    let mut emit_meridian = |lon: f64| {
        buf.clear();
        for i in 0..=MERIDIAN_SAMPLES {
            let t = i as f64 / MERIDIAN_SAMPLES as f64;
            let lat = -85.0 + t * 170.0;
            buf.push((lat, lon));
        }
        emit_strip(&buf);
    };
    emit_meridian(0.0);
    emit_meridian(180.0);
    let mut lon = step_deg;
    while lon < 180.0 - 1e-6 {
        emit_meridian(lon);
        emit_meridian(-lon);
        lon += step_deg;
    }

    let mut emit_parallel = |lat: f64| {
        buf.clear();
        for i in 0..=PARALLEL_SAMPLES {
            let t = i as f64 / PARALLEL_SAMPLES as f64;
            let lon = -180.0 + t * 360.0;
            buf.push((lat, lon));
        }
        emit_strip(&buf);
    };
    emit_parallel(0.0);
    let mut lat = step_deg;
    while lat < 90.0 {
        emit_parallel(lat);
        emit_parallel(-lat);
        lat += step_deg;
    }

    if verts.is_empty() {
        return None;
    }
    Some(RetainedDynamicMesh {
        id,
        revision,
        anchor_ecef: [anchor_dvec.x, anchor_dvec.y, anchor_dvec.z],
        vertices: Arc::new(verts),
        indices: Arc::new(inds),
        color: [0.58, 0.64, 0.72, 0.35],
        unlit: true,
    })
}

struct WorkerShared {
    latest_input: Option<(u64, EarthInput, f32)>,
    last_dispatched: Option<(u64, EarthInput, f32)>,
    pending_provider: Option<(TileProviderConfig, bool)>,
    pending_source: Option<Box<dyn TileSource + Send>>,
    pending_weather_provider: Option<Option<Arc<dyn WeatherProvider>>>,
    pending_building_source: Option<Option<Box<dyn crate::scene::BuildingSource + Send>>>,
    provider_gen: u64,
    aux_gen: u64,
    shutdown: bool,
    stats: EarthViewportStats,
}

/// Asynchronous statistics exposed by [`EarthTileWorker`] and [`EarthViewport`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct EarthViewportStats {
    pub camera_updates: u64,
    pub worker_builds: u64,
    pub mesh_cache_hits: u64,
    pub mesh_cache_misses: u64,
    pub last_visible_tiles: usize,
    pub ancestor_fallbacks: usize,
}

/// Background worker thread that processes coalesced camera snapshots into retained GPU frames.
pub struct EarthTileWorker {
    shared: Arc<(Mutex<WorkerShared>, Condvar)>,
    completed_seq: Arc<AtomicU64>,
    satellite_store: Arc<SatelliteCatalogStore>,
    ground_snapshot: Arc<RwLock<GroundElevSnapshot>>,
    last_buildings_mesh: Arc<RwLock<Option<RetainedDynamicMesh>>>,
    dynamic_mesh_id_base: u64,
    thread: Option<JoinHandle<()>>,
    next_seq: u64,
}

impl EarthTileWorker {
    pub fn spawn(
        retained: RetainedScene3D,
        source: Box<dyn TileSource + Send>,
        wakeup: Option<Arc<dyn Fn() + Send + Sync>>,
    ) -> Self {
        Self::spawn_with_plugins(retained, source, wakeup, EarthPluginRegistry::standard())
    }

    pub fn spawn_with_plugins(
        retained: RetainedScene3D,
        source: Box<dyn TileSource + Send>,
        wakeup: Option<Arc<dyn Fn() + Send + Sync>>,
        plugins: EarthPluginRegistry,
    ) -> Self {
        Self::spawn_with_plugins_and_world(
            retained,
            source,
            wakeup,
            plugins,
            Arc::new(RwLock::new(EarthWorld::new())),
        )
    }

    pub fn spawn_with_plugins_and_world(
        retained: RetainedScene3D,
        mut source: Box<dyn TileSource + Send>,
        wakeup: Option<Arc<dyn Fn() + Send + Sync>>,
        plugins: EarthPluginRegistry,
        world: Arc<RwLock<EarthWorld>>,
    ) -> Self {
        let shared = Arc::new((
            Mutex::new(WorkerShared {
                latest_input: None,
                last_dispatched: None,
                pending_provider: None,
                pending_source: None,
                pending_weather_provider: None,
                pending_building_source: None,
                provider_gen: 0,
                aux_gen: 0,
                shutdown: false,
                stats: EarthViewportStats::default(),
            }),
            Condvar::new(),
        ));
        let completed_seq = Arc::new(AtomicU64::new(0));
        let ground_snapshot = Arc::new(RwLock::new(GroundElevSnapshot::default()));
        let last_buildings_mesh = Arc::new(RwLock::new(None));
        let dynamic_mesh_id_base =
            NEXT_BUILDER_DYN_MESH_ID_BLOCK.fetch_add(16, Ordering::Relaxed);
        let shared_bg = Arc::clone(&shared);
        let shared_env = Arc::clone(&shared);
        let completed_bg = Arc::clone(&completed_seq);
        let ground_snapshot_bg = Arc::clone(&ground_snapshot);
        let last_buildings_bg = Arc::clone(&last_buildings_mesh);
        let world_bg = Arc::clone(&world);
        let env_wakeup_outer = wakeup.clone();
        let env_wakeup: Arc<dyn Fn() + Send + Sync> = Arc::new(move || {
            {
                let (lock, cv) = &*shared_env;
                if let Ok(mut guard) = lock.lock() {
                    if guard.latest_input.is_none() {
                        if let Some((seq, ref inp, idle_f)) = guard.last_dispatched {
                            guard.latest_input = Some((seq, inp.clone(), idle_f));
                            cv.notify_one();
                        }
                    }
                }
            }
            if let Some(ref w) = env_wakeup_outer {
                w();
            }
        });
        source.set_tile_ready_callback(Some(Arc::clone(&env_wakeup)));
        let tile_wakeup = Arc::clone(&env_wakeup);
        let cloud_store = Arc::new(GlobalCloudMapStore::new(None, Some(Arc::clone(&env_wakeup))));
        let satellite_store = Arc::new(SatelliteCatalogStore::new(None, Some(Arc::clone(&env_wakeup))));
        let skybox_store = Arc::new(SpaceSkyboxStore::new(Some(env_wakeup)));
        let sat_store_worker = Arc::clone(&satellite_store);

        let handle = std::thread::Builder::new()
            .name("strata-earth-worker".to_string())
            .spawn(move || {
                let mut builder = RetainedTerrainBuilder::with_all_stores(
                    cloud_store,
                    sat_store_worker,
                    skybox_store,
                );
                builder.set_plugins(plugins);
                // Main-thread EarthViewport emits camera-locked 144 Hz particle layers & GPU ocean uniforms synchronously.
                builder.set_include_particles(false);
                builder.set_include_ocean(false);
                builder.set_ground_snapshot(ground_snapshot_bg);
                builder.set_last_buildings_mesh(last_buildings_bg);
                builder.set_dynamic_mesh_id_base(dynamic_mesh_id_base);
                builder.set_world(world_bg);
                let mut geom_rev = 0u64;
                let mut seen_provider_gen = 0u64;
                let mut seen_aux_gen = 0u64;
                let mut last_canonical_2d: Option<bool> = None;
                loop {
                    let (
                        seq,
                        input,
                        idle_factor,
                        provider_update,
                        source_update,
                        weather_provider_update,
                        building_source_update,
                    ) = {
                        let (lock, cv) = &*shared_bg;
                        let mut guard = lock.lock().unwrap();
                        while guard.latest_input.is_none() && !guard.shutdown {
                            guard = cv.wait(guard).unwrap();
                        }
                        if guard.shutdown {
                            return;
                        }
                        let (prov_up, src_up) = if guard.provider_gen != seen_provider_gen {
                            seen_provider_gen = guard.provider_gen;
                            (guard.pending_provider.take(), guard.pending_source.take())
                        } else {
                            (None, None)
                        };
                        let (wp_up, bs_up) = if guard.aux_gen != seen_aux_gen {
                            seen_aux_gen = guard.aux_gen;
                            (
                                guard.pending_weather_provider.take(),
                                guard.pending_building_source.take(),
                            )
                        } else {
                            (None, None)
                        };
                        let triple = guard.latest_input.take().unwrap();
                        (triple.0, triple.1, triple.2, prov_up, src_up, wp_up, bs_up)
                    };

                    if let Some(mut new_src) = source_update {
                        new_src.set_tile_ready_callback(Some(Arc::clone(&tile_wakeup)));
                        source = new_src;
                        builder.clear_provider_caches(true);
                    } else if let Some((cfg, clear_elev)) = provider_update {
                        let _ = source.set_provider(&cfg);
                        builder.clear_provider_caches(clear_elev);
                    }
                    if let Some(wp) = weather_provider_update {
                        builder.set_weather_provider(wp);
                    }
                    if let Some(bs) = building_source_update {
                        builder.set_building_source(bs);
                    }

                    geom_rev += 1;
                    let start_geo_rev = retained.geo_revision();
                    let ll_now = input.low_level_state();
                    let is_canon_2d = is_canonical_2d_morph(ll_now.map_2d_morph as f64)
                        && (ll_now.local_25d_blend.max(ll_now.map_2d_morph) as f64)
                            >= CANONICAL_2D_MORPH_THRESHOLD;
                    let (mut frame, build_stats) = builder.build_frame_with_idle(
                        &input,
                        source.as_mut(),
                        geom_rev,
                        true,
                        idle_factor,
                    );
                    Arc::make_mut(&mut frame).geo_revision = start_geo_rev;
                    if last_canonical_2d.is_some_and(|prev| prev != is_canon_2d) {
                        retained.update_terrain_with_camera_preserving_overlays(frame);
                    } else {
                        retained.update_terrain_preserving_camera(frame);
                    }
                    last_canonical_2d = Some(is_canon_2d);

                    {
                        let (lock, cv) = &*shared_bg;
                        let mut guard = lock.lock().unwrap();
                        guard.stats.worker_builds += 1;
                        guard.stats.mesh_cache_hits += build_stats.mesh_cache_hits as u64;
                        guard.stats.mesh_cache_misses += build_stats.mesh_cache_misses as u64;
                        guard.stats.last_visible_tiles = build_stats.visible_tiles;
                        guard.stats.ancestor_fallbacks += build_stats.ancestor_fallbacks;
                        completed_bg.store(seq, Ordering::Release);
                        cv.notify_all();
                    }
                    if let Some(ref wake) = wakeup {
                        wake();
                    }
                }
            })
            .expect("spawn strata-earth-worker thread");

        Self {
            shared,
            completed_seq,
            satellite_store,
            ground_snapshot,
            last_buildings_mesh,
            dynamic_mesh_id_base,
            thread: Some(handle),
            next_seq: 0,
        }
    }

    #[inline]
    pub(crate) fn dynamic_mesh_id_base(&self) -> u64 {
        self.dynamic_mesh_id_base
    }

    pub(crate) fn last_buildings_mesh(&self) -> Option<RetainedDynamicMesh> {
        self.last_buildings_mesh
            .read()
            .ok()
            .and_then(|g| g.clone())
    }

    /// Propagates a [`WeatherProvider`] update to the background tile worker's [`RetainedTerrainBuilder`].
    pub fn set_weather_provider(&mut self, provider: Option<Arc<dyn WeatherProvider>>) {
        let (lock, cv) = &*self.shared;
        if let Ok(mut guard) = lock.lock() {
            guard.pending_weather_provider = Some(provider);
            guard.aux_gen = guard.aux_gen.wrapping_add(1);
            if guard.latest_input.is_none() {
                if let Some((seq, ref inp, idle_f)) = guard.last_dispatched {
                    guard.latest_input = Some((seq, inp.clone(), idle_f));
                }
            }
            cv.notify_one();
        }
    }

    /// Propagates a [`crate::scene::BuildingSource`] update to the background tile worker's [`RetainedTerrainBuilder`].
    pub fn set_building_source(
        &mut self,
        source: Option<Box<dyn crate::scene::BuildingSource + Send>>,
    ) {
        let (lock, cv) = &*self.shared;
        if let Ok(mut guard) = lock.lock() {
            guard.pending_building_source = Some(source);
            guard.aux_gen = guard.aux_gen.wrapping_add(1);
            if guard.latest_input.is_none() {
                if let Some((seq, ref inp, idle_f)) = guard.last_dispatched {
                    guard.latest_input = Some((seq, inp.clone(), idle_f));
                }
            }
            cv.notify_one();
        }
    }

    /// Samples terrain height in metres above WGS84 from the latest worker frame's elevation tiles.
    pub fn sample_ground_elevation_m(&self, lat_deg: f64, lon_deg: f64) -> f64 {
        self.ground_snapshot
            .read()
            .map(|s| s.sample_elevation_m(lat_deg, lon_deg))
            .unwrap_or(0.0)
    }

    /// Switches the active [`TileProviderConfig`] on the background tile source and clears stale GPU textures.
    pub fn set_tile_provider(&mut self, config: TileProviderConfig, clear_elevation: bool) {
        let (lock, cv) = &*self.shared;
        if let Ok(mut guard) = lock.lock() {
            guard.pending_provider = Some((config, clear_elevation));
            guard.pending_source = None;
            guard.provider_gen = guard.provider_gen.wrapping_add(1);
            if guard.latest_input.is_none() {
                if let Some((seq, ref inp, idle_f)) = guard.last_dispatched {
                    guard.latest_input = Some((seq, inp.clone(), idle_f));
                }
            }
            cv.notify_one();
        }
    }

    /// Replaces the background [`TileSource`] completely and clears all cached GPU meshes & textures.
    pub fn set_tile_source(&mut self, source: Box<dyn TileSource + Send>) {
        let (lock, cv) = &*self.shared;
        if let Ok(mut guard) = lock.lock() {
            guard.pending_source = Some(source);
            guard.pending_provider = None;
            guard.provider_gen = guard.provider_gen.wrapping_add(1);
            if guard.latest_input.is_none() {
                if let Some((seq, ref inp, idle_f)) = guard.last_dispatched {
                    guard.latest_input = Some((seq, inp.clone(), idle_f));
                }
            }
            cv.notify_one();
        }
    }

    /// Queues `input` for background LOD traversal + tile assembly with explicit `idle_factor` (`0.0..=1.0`).
    pub fn request_update_with_idle(&mut self, input: EarthInput, idle_factor: f32) -> u64 {
        self.next_seq += 1;
        let seq = self.next_seq;
        let clamped_idle = idle_factor.clamp(0.0, 1.0);
        let (lock, cv) = &*self.shared;
        if let Ok(mut guard) = lock.lock() {
            guard.last_dispatched = Some((seq, input.clone(), clamped_idle));
            guard.latest_input = Some((seq, input, clamped_idle));
            guard.stats.camera_updates += 1;
            cv.notify_one();
        }
        seq
    }

    /// Queues `input` for background LOD traversal + tile assembly, coalescing any unstarted request.
    pub fn request_update(&mut self, input: EarthInput) -> u64 {
        self.request_update_with_idle(input, 0.0)
    }

    pub fn flush(&self) {
        let target = self.next_seq;
        let (lock, cv) = &*self.shared;
        let mut guard = lock.lock().unwrap();
        while self.completed_seq.load(Ordering::Acquire) < target && !guard.shutdown {
            guard = cv.wait(guard).unwrap();
        }
    }

    pub fn stats(&self) -> EarthViewportStats {
        self.shared.0.lock().map(|g| g.stats).unwrap_or_default()
    }
}

impl Drop for EarthTileWorker {
    fn drop(&mut self) {
        {
            let (lock, cv) = &*self.shared;
            if let Ok(mut guard) = lock.lock() {
                guard.shutdown = true;
                cv.notify_all();
            }
        }
        if let Some(h) = self.thread.take() {
            let _ = h.join();
        }
    }
}

/// High-frequency retained 3D/2D globe controller pairing O(1) GPU camera updates with
/// background LOD streaming, seamless ground-proximity 2.5D flattening, and smooth 3D/2D morphs.
pub struct EarthViewport {
    retained: RetainedScene3D,
    worker: EarthTileWorker,
    world: Arc<RwLock<EarthWorld>>,
    input: EarthInput,
    plugins: EarthPluginRegistry,
    active_provider: TileProviderConfig,
    weather_provider: Option<Arc<dyn WeatherProvider>>,
    live_weather: OpenMeteoWeatherProvider,
    last_weather_snapshot: Option<WeatherSnapshot>,
    target_morph: f32,
    linear_morph: f32,
    transition_duration_s: f64,
    had_active_particles: bool,
    had_active_ocean: bool,
    weather_ids: HashMap<u64, u64>,
    overlay_rev: u64,
    last_overlay_flat_t: f32,
    last_geo_morph_sig: u64,
    last_geo_eye_dist_m: f64,
    last_geo_lon_sector: i32,
    idle_elapsed_s: f64,
    idle_stage: u8,
    last_camera_pose_sig: u64,
    stopped: Arc<AtomicBool>,
}

impl EarthViewport {
    #[inline]
    fn is_input_canonical_2d(input: &EarthInput) -> bool {
        let ll = input.low_level_state();
        is_canonical_2d_morph(ll.map_2d_morph as f64)
            && (ll.local_25d_blend.max(ll.map_2d_morph) as f64) >= CANONICAL_2D_MORPH_THRESHOLD
    }

    #[inline]
    fn camera_pose_signature(input: &EarthInput) -> u64 {
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        let mut mix = |v: u64| {
            h ^= v;
            h = h.wrapping_mul(0x100_0000_01b3);
        };
        let c = &input.camera;
        mix((c.focus.lat_deg * 1e6).round() as i64 as u64);
        mix((c.focus.lon_deg * 1e6).round() as i64 as u64);
        mix((c.focus.height_m * 100.0).round() as i64 as u64);
        for v in c.eye_offset_m {
            mix((v * 100.0).round() as i64 as u64);
        }
        for v in c.aim_offset_m {
            mix((v * 100.0).round() as i64 as u64);
        }
        mix((c.fov_y_deg * 1_000.0).round() as i64 as u64);
        mix((input.viewport.width as f64 * 10.0).round() as u64);
        mix((input.viewport.height as f64 * 10.0).round() as u64);
        mix((input.view_morph as f64 * 1_000.0).round() as u64);
        mix(match input.view_mode {
            EarthViewMode::Mode3D => 3,
            EarthViewMode::Mode2D => 2,
        });
        if let Some(up) = input.camera_up_ecef {
            for v in up {
                mix((v * 1e4).round() as i64 as u64);
            }
        }
        h
    }

    /// Creates an [`EarthViewport`] and immediately schedules the initial retained tile frame.
    pub fn new(input: &EarthInput, source: Box<dyn TileSource + Send>) -> Self {
        Self::with_wakeup(input, source, None)
    }

    pub fn with_wakeup(
        input: &EarthInput,
        source: Box<dyn TileSource + Send>,
        wakeup: Option<Arc<dyn Fn() + Send + Sync>>,
    ) -> Self {
        Self::with_plugins(input, source, wakeup, EarthPluginRegistry::standard())
    }

    /// Creates an [`EarthViewport`] backed by a custom [`EarthPluginRegistry`].
    pub fn with_plugins(
        input: &EarthInput,
        source: Box<dyn TileSource + Send>,
        wakeup: Option<Arc<dyn Fn() + Send + Sync>>,
        plugins: EarthPluginRegistry,
    ) -> Self {
        Self::with_plugins_and_bootstrap(
            input,
            source,
            wakeup,
            plugins,
            input.frame_zero_bootstrap,
        )
    }

    /// Creates an [`EarthViewport`] and synchronously installs a Frame-0 root tile plus
    /// procedural skybox when `bootstrap_frame_zero` is `true`, before the background worker completes.
    pub fn new_with_bootstrap(
        input: &EarthInput,
        source: Box<dyn TileSource + Send>,
        bootstrap_frame_zero: bool,
    ) -> Self {
        Self::with_wakeup_and_bootstrap(input, source, None, bootstrap_frame_zero)
    }

    /// Creates an [`EarthViewport`] with a wakeup callback and optional synchronous Frame-0 bootstrap.
    pub fn with_wakeup_and_bootstrap(
        input: &EarthInput,
        source: Box<dyn TileSource + Send>,
        wakeup: Option<Arc<dyn Fn() + Send + Sync>>,
        bootstrap_frame_zero: bool,
    ) -> Self {
        Self::with_plugins_and_bootstrap(
            input,
            source,
            wakeup,
            EarthPluginRegistry::standard(),
            bootstrap_frame_zero,
        )
    }

    /// Creates an [`EarthViewport`] backed by a custom [`EarthPluginRegistry`] with optional
    /// synchronous Frame-0 root-tile and procedural skybox bootstrap.
    pub fn with_plugins_and_bootstrap(
        input: &EarthInput,
        source: Box<dyn TileSource + Send>,
        wakeup: Option<Arc<dyn Fn() + Send + Sync>>,
        plugins: EarthPluginRegistry,
        bootstrap_frame_zero: bool,
    ) -> Self {
        let ll = input.low_level_state();
        let initial_morph = ll.map_2d_morph;
        let initial_flat_t = ll.local_25d_blend.max(ll.map_2d_morph);
        let cam_uniforms = RetainedTerrainBuilder::camera_from_input_with_plugins(
            input,
            &plugins,
            input.weather.weather_time_s,
            1,
            false,
        );
        let retained = RetainedScene3D::new(cam_uniforms);
        if bootstrap_frame_zero {
            let (frame, _) = RetainedTerrainBuilder::build_frame_zero_bootstrap(
                input,
                Some(&*source),
                &plugins,
            );
            retained.update_terrain_preserving_camera(frame);
        }
        let world = Arc::new(RwLock::new(EarthWorld::new()));
        let mut worker = EarthTileWorker::spawn_with_plugins_and_world(
            retained.clone(),
            source,
            wakeup,
            plugins.clone(),
            Arc::clone(&world),
        );
        worker.request_update(input.clone());
        let initial_pose_sig = Self::camera_pose_signature(input);
        let mut vp = Self {
            retained,
            worker,
            world,
            input: input.clone(),
            plugins,
            active_provider: TileProviderConfig::default(),
            weather_provider: None,
            live_weather: OpenMeteoWeatherProvider::new(),
            last_weather_snapshot: None,
            target_morph: initial_morph,
            linear_morph: initial_morph,
            transition_duration_s: 0.45,
            had_active_particles: false,
            had_active_ocean: false,
            weather_ids: HashMap::new(),
            overlay_rev: 1,
            last_overlay_flat_t: initial_flat_t,
            last_geo_morph_sig: 0,
            last_geo_eye_dist_m: input.camera.eye_distance_m(),
            last_geo_lon_sector: (crate::geodesy::wrap_longitude(input.camera.focus.lon_deg) / 30.0)
                .round() as i32,
            idle_elapsed_s: 0.0,
            idle_stage: 0,
            last_camera_pose_sig: initial_pose_sig,
            stopped: Arc::new(AtomicBool::new(false)),
        };
        if input.precipitation.enabled || input.ocean_waves.enabled || input.base_weather.enabled {
            vp.publish_camera_and_particles_and_ocean();
        }
        vp
    }

    /// Installs a synchronous Frame-0 root-tile + procedural skybox frame on demand if no terrain
    /// frame has been published yet. Returns `true` if a bootstrap frame was installed.
    pub fn bootstrap_frame_zero(&mut self) -> bool {
        if self.retained.snapshot_terrain().is_some() {
            return false;
        }
        let (frame, _) =
            RetainedTerrainBuilder::build_frame_zero_bootstrap(&self.input, None, &self.plugins);
        self.retained.update_terrain_preserving_camera(frame);
        if !self.stopped.load(Ordering::Relaxed) {
            let idle_f = self.idle_refinement_factor();
            self.worker
                .request_update_with_idle(self.input.clone(), idle_f);
        }
        true
    }

    #[inline]
    pub fn plugins(&self) -> &EarthPluginRegistry {
        &self.plugins
    }

    /// Registers (or replaces) a plugin in the live world-rendering pipeline and refreshes the scene.
    pub fn register_plugin<P: EarthPlugin>(&mut self, plugin: P) {
        self.plugins.register(plugin);
        self.refresh_after_plugin_change();
    }

    /// Inserts a plugin immediately before `target_id` in execution order.
    pub fn insert_plugin_before<P: EarthPlugin>(&mut self, target_id: &str, plugin: P) {
        self.plugins.insert_before(target_id, plugin);
        self.refresh_after_plugin_change();
    }

    /// Removes a plugin by `id`, returning `true` if one was unplugged.
    pub fn remove_plugin(&mut self, id: &str) -> bool {
        let removed = self.plugins.remove(id);
        if removed {
            if id == OceanWavesPlugin::ID {
                self.input.ocean_waves.enabled = false;
            } else if id == PrecipitationPlugin::ID {
                self.input.precipitation.enabled = false;
            }
            self.refresh_after_plugin_change();
        }
        removed
    }

    /// Removes all currently plugged-in plugins cleanly (`EarthPluginRegistry::empty()` parity).
    pub fn without_default_plugins(&mut self) {
        for (id, _) in self.plugins.list_ids() {
            let _ = self.plugins.remove(id);
        }
        self.input.ocean_waves.enabled = false;
        self.input.precipitation.enabled = false;
        self.refresh_after_plugin_change();
    }

    /// Enables or disables a registered plugin by `id`.
    pub fn set_plugin_enabled(&mut self, id: &str, enabled: bool) -> bool {
        let updated = self.plugins.set_enabled(id, enabled);
        if updated {
            if id == OceanWavesPlugin::ID {
                self.input.ocean_waves.enabled = enabled;
            } else if id == PrecipitationPlugin::ID {
                self.input.precipitation.enabled = enabled;
            }
            self.refresh_after_plugin_change();
        }
        updated
    }

    #[inline]
    pub fn is_plugin_enabled(&self, id: &str) -> Option<bool> {
        self.plugins.is_enabled(id)
    }

    fn refresh_after_plugin_change(&mut self) {
        self.publish_camera_and_overlays();
        if !self.stopped.load(Ordering::Relaxed) {
            self.worker.request_update(self.input.clone());
        }
    }

    /// Returns the active [`SurfaceMorphParams`] matching the current terrain tile projection & morph.
    #[inline]
    pub fn current_surface_morph(&self) -> SurfaceMorphParams {
        let ll = self.input.low_level_state();
        let flat_t = ll.local_25d_blend.max(ll.map_2d_morph);
        stabilized_morph_params(
            self.input.camera.focus,
            flat_t as f64,
            ll.map_2d_morph as f64,
            self.input.tile_projection,
        )
    }

    #[inline]
    fn camera_up_dvec(&self) -> Option<DVec3> {
        self.input
            .camera_up_ecef
            .map(|[x, y, z]| DVec3::new(x, y, z))
    }

    fn compile_world_retained(
        &mut self,
    ) -> (
        Arc<Vec<Arc<strata_render::scene3d::RetainedMeshAsset>>>,
        Arc<Vec<strata_render::scene3d::RetainedGeoInstance>>,
    ) {
        let morph = self.current_surface_morph();
        self.last_geo_morph_sig = mesh_morph_signature(morph);
        self.last_geo_eye_dist_m = self.input.camera.eye_distance_m();
        self.last_geo_lon_sector =
            (crate::geodesy::wrap_longitude(self.input.camera.focus.lon_deg) / 30.0).round() as i32;
        let placement = self
            .input
            .camera
            .placement_for_morph(morph, self.camera_up_dvec());
        let vp_h = self.input.viewport.height.max(1.0) as f64;
        let fov_y = self.input.camera.fov_y_deg;
        let elev_on = self.input.elevation_enabled;
        let worker_ref = &self.worker;
        if let Ok(mut w) = self.world.write() {
            if w.is_empty() {
                return (Arc::default(), Arc::default());
            }
            w.compile_retained(placement.eye, vp_h, fov_y, elev_on, morph, &|lat, lon| {
                if elev_on {
                    worker_ref.sample_ground_elevation_m(lat, lon)
                } else {
                    0.0
                }
            })
        } else {
            (Arc::default(), Arc::default())
        }
    }

    /// Immediately recompiles and publishes [`EarthWorld`] instances to the GPU retained scene
    /// in microseconds without rebuilding terrain tiles.
    pub fn refresh_world_objects(&mut self) {
        let (assets, instances) = self.compile_world_retained();
        self.retained.update_geo_objects(assets, instances);
    }

    fn build_dynamic_overlays(&mut self) -> Vec<RetainedDynamicMesh> {
        self.overlay_rev = self.overlay_rev.wrapping_add(1).max(1);
        let ll = self.input.low_level_state();
        let flat_t = ll.local_25d_blend.max(ll.map_2d_morph);
        self.last_overlay_flat_t = flat_t;
        let mut out = Vec::new();
        if self
            .plugins
            .active_hooks()
            .intersects(EarthPluginHooks::DYNAMIC_OVERLAYS)
        {
            let (_, elements) = self.worker.satellite_store.snapshot();
            let mut ctx = EarthDynamicOverlayContext {
                input: &self.input,
                low_level: ll,
                flat_t,
                time_s: self.input.weather.weather_time_s,
                overlay_rev: self.overlay_rev,
                satellite_elements: &elements,
                weather_ids: &mut self.weather_ids,
                ground_snapshot: Some(self.worker.ground_snapshot.as_ref()),
            };
            self.plugins.run_dynamic_overlays(&mut ctx, &mut out);
        }
        if self.input.layers.buildings {
            if let Some(bldg) = self.worker.last_buildings_mesh() {
                out.push(bldg);
            }
        }
        if self.input.layers.graticule {
            let morph = self.current_surface_morph();
            let placement = self
                .input
                .camera
                .placement_for_morph(morph, self.camera_up_dvec());
            if let Some(grat) = build_graticule_dynamic_mesh(
                self.worker.dynamic_mesh_id_base() + 2,
                self.overlay_rev,
                &self.input,
                morph,
                placement.eye,
            ) {
                out.push(grat);
            }
        }
        out
    }

    fn resolve_weather_snapshot(&mut self) -> Option<WeatherSnapshot> {
        let ground_elev = if self.input.elevation_enabled {
            self.worker.sample_ground_elevation_m(
                self.input.camera.focus.lat_deg,
                self.input.camera.focus.lon_deg,
            )
        } else {
            0.0
        };
        let snap = evaluate_base_weather_snapshot(
            &self.input,
            ground_elev,
            self.weather_provider.as_ref(),
            Some(&self.live_weather),
        );
        self.last_weather_snapshot = snap;
        snap
    }

    fn build_particle_layers(
        &mut self,
        weather_snapshot: Option<WeatherSnapshot>,
    ) -> (
        Arc<Vec<RetainedParticleLayer>>,
        Arc<Vec<Arc<RetainedTileTexture>>>,
    ) {
        let mut layers = Vec::new();
        let mut textures = Vec::new();
        if self
            .plugins
            .active_hooks()
            .intersects(EarthPluginHooks::PARTICLES)
        {
            let ll = self.input.low_level_state();
            let mut ctx = EarthParticleContext {
                input: &self.input,
                low_level: ll,
                time_s: self.input.weather.weather_time_s,
                weather_snapshot,
                ground_snapshot: Some(self.worker.ground_snapshot.as_ref()),
            };
            self.plugins
                .run_particle_layers(&mut ctx, &mut layers, &mut textures);
        }
        self.had_active_particles = !layers.is_empty();
        (Arc::new(layers), Arc::new(textures))
    }

    fn build_ocean_surface(
        &mut self,
        weather_snapshot: Option<WeatherSnapshot>,
    ) -> Option<Arc<RetainedOceanSurface>> {
        let surf = if self
            .plugins
            .active_hooks()
            .intersects(EarthPluginHooks::OCEAN)
        {
            let ll = self.input.low_level_state();
            let mut ctx = EarthOceanContext {
                input: &self.input,
                low_level: ll,
                time_s: self.input.weather.weather_time_s,
                overlay_rev: self.overlay_rev,
                weather_snapshot,
                ground_snapshot: Some(self.worker.ground_snapshot.as_ref()),
            };
            self.plugins.run_ocean_surface(&mut ctx).map(Arc::new)
        } else {
            None
        };
        self.had_active_ocean = surf.is_some();
        surf
    }

    fn camera_uniforms_with_weather(
        &self,
        weather_snapshot: Option<WeatherSnapshot>,
    ) -> RetainedCameraUniforms {
        let ground_elev = self.sample_ground_elevation_m(
            self.input.camera.focus.lat_deg,
            self.input.camera.focus.lon_deg,
        );
        RetainedTerrainBuilder::camera_from_input_with_plugins_weather_and_ground(
            &self.input,
            &self.plugins,
            self.input.weather.weather_time_s,
            self.overlay_rev,
            false,
            weather_snapshot,
            ground_elev,
        )
    }

    fn publish_camera_and_particles(&mut self) {
        let ws = self.resolve_weather_snapshot();
        let cam_u = self.camera_uniforms_with_weather(ws);
        let (p_layers, p_textures) = self.build_particle_layers(ws);
        self.retained
            .update_camera_and_particles(cam_u, p_layers, p_textures);
    }

    fn publish_camera_and_ocean(&mut self) {
        let ws = self.resolve_weather_snapshot();
        let cam_u = self.camera_uniforms_with_weather(ws);
        let ocean = self.build_ocean_surface(ws);
        self.retained.update_camera_and_ocean(cam_u, ocean);
    }

    fn publish_camera_and_particles_and_ocean(&mut self) {
        let ws = self.resolve_weather_snapshot();
        let cam_u = self.camera_uniforms_with_weather(ws);
        let (p_layers, p_textures) = self.build_particle_layers(ws);
        let ocean = self.build_ocean_surface(ws);
        self.retained
            .update_camera_and_particles_and_ocean(cam_u, p_layers, p_textures, ocean);
    }

    fn publish_camera_and_overlays(&mut self) {
        let ws = self.resolve_weather_snapshot();
        let cam_u = self.camera_uniforms_with_weather(ws);
        let dyn_meshes = self.build_dynamic_overlays();
        let (p_layers, p_textures) = self.build_particle_layers(ws);
        let ocean = self.build_ocean_surface(ws);
        let has_world = self.world.read().map(|w| !w.is_empty()).unwrap_or(false);
        if has_world {
            let (assets, instances) = self.compile_world_retained();
            self.retained
                .update_camera_and_world_and_particles_and_ocean(
                    cam_u, dyn_meshes, assets, instances, p_layers, p_textures, ocean,
                );
        } else {
            self.retained
                .update_camera_and_world_and_particles_and_ocean(
                    cam_u,
                    dyn_meshes,
                    Arc::default(),
                    Arc::default(),
                    p_layers,
                    p_textures,
                    ocean,
                );
        }
    }

    #[inline]
    pub fn retained_scene(&self) -> RetainedScene3D {
        self.retained.clone()
    }

    #[inline]
    pub fn input(&self) -> &EarthInput {
        &self.input
    }

    #[inline]
    pub fn elevation_enabled(&self) -> bool {
        self.input.elevation_enabled
    }

    #[inline]
    pub fn labels_enabled(&self) -> bool {
        self.input.layers.labels
    }

    /// Toggles geographic map labels in O(1) without rebuilding terrain geometry.
    pub fn set_labels_enabled(&mut self, enabled: bool) {
        self.input.layers.labels = enabled;
    }

    #[inline]
    pub fn buildings_enabled(&self) -> bool {
        self.input.layers.buildings
    }

    /// Toggles extruded 3D building rendering from the configured [`crate::scene::BuildingSource`].
    pub fn set_buildings_enabled(&mut self, enabled: bool) {
        if self.input.layers.buildings == enabled {
            return;
        }
        self.input.layers.buildings = enabled;
        self.publish_camera_and_overlays();
        if !self.stopped.load(Ordering::Relaxed) {
            let idle_f = self.idle_refinement_factor();
            self.worker
                .request_update_with_idle(self.input.clone(), idle_f);
        }
    }

    /// Installs or clears a [`crate::scene::BuildingSource`] on the background worker and refreshes the view.
    pub fn set_building_source(
        &mut self,
        source: Option<Box<dyn crate::scene::BuildingSource + Send>>,
    ) {
        self.worker.set_building_source(source);
        if !self.stopped.load(Ordering::Relaxed) {
            let idle_f = self.idle_refinement_factor();
            self.worker
                .request_update_with_idle(self.input.clone(), idle_f);
        }
    }

    #[inline]
    pub fn graticule_enabled(&self) -> bool {
        self.input.layers.graticule
    }

    /// Toggles the geodetic meridian/parallel graticule overlay in O(1).
    pub fn set_graticule_enabled(&mut self, enabled: bool) {
        if self.input.layers.graticule == enabled {
            return;
        }
        self.input.layers.graticule = enabled;
        self.publish_camera_and_overlays();
        if !self.stopped.load(Ordering::Relaxed) {
            let idle_f = self.idle_refinement_factor();
            self.worker
                .request_update_with_idle(self.input.clone(), idle_f);
        }
    }

    /// Toggles live heightmap (DEM) fetching and terrain relief displacement cleanly.
    ///
    /// Immediately re-evaluates all [`crate::geodesy::AltitudeMode::Terrain`] objects in
    /// [`EarthWorld`] (`DEM + altitude_m` when `true`; `0.0 + altitude_m` above sea level when `false`).
    pub fn set_elevation_enabled(&mut self, enabled: bool) {
        if self.input.elevation_enabled == enabled {
            return;
        }
        self.input.elevation_enabled = enabled;
        self.refresh_world_objects();
        if self.input.precipitation.enabled
            || self.had_active_particles
            || self.input.ocean_waves.enabled
            || self.had_active_ocean
        {
            self.publish_camera_and_particles_and_ocean();
        }
        if !self.stopped.load(Ordering::Relaxed) {
            self.worker.request_update(self.input.clone());
        }
    }

    #[inline]
    pub fn space_config(&self) -> &SpaceEnvironmentConfig {
        &self.input.space
    }

    /// Updates planetary lighting, starfield, Sun/Moon, satellites, and city-light configuration in O(1).
    pub fn set_space_config(&mut self, space: SpaceEnvironmentConfig) {
        let need_worker = self.input.space.city_lights != space.city_lights
            || self.input.space.satellites_visible != space.satellites_visible
            || self.input.space.stars_visible != space.stars_visible
            || self.input.space.skybox_survey != space.skybox_survey
            || self.input.space.skybox_quality != space.skybox_quality;
        self.input.space = space;
        self.publish_camera_and_overlays();
        if need_worker && !self.stopped.load(Ordering::Relaxed) {
            self.worker.request_update(self.input.clone());
        }
    }

    #[inline]
    pub fn local_atmosphere_config(&self) -> &LocalAtmosphereConfig {
        &self.input.local_atmosphere
    }

    /// Updates optional realistic 2.5D sky, Sun, night stars, terrain lighting, and mountain shadow config in O(1).
    pub fn set_local_atmosphere_config(&mut self, config: LocalAtmosphereConfig) {
        let need_night_tiles = (self.input.local_atmosphere.night_stars_sync
            && self.input.local_atmosphere.terrain_lighting)
            != (config.night_stars_sync && config.terrain_lighting);
        self.input.local_atmosphere = config;
        self.publish_camera_and_particles_and_ocean();
        if need_night_tiles && !self.stopped.load(Ordering::Relaxed) {
            self.worker.request_update(self.input.clone());
        }
    }

    #[inline]
    pub fn base_weather_config(&self) -> &BaseWeatherConfig {
        &self.input.base_weather
    }

    /// Updates the unified Base Weather API configuration in O(1), propagating coherently to
    /// 2.5D sky/haze, 2.5D/3D cloud darkening, precipitation, and GPU ocean wave trains.
    pub fn set_base_weather_config(&mut self, config: BaseWeatherConfig) {
        self.input.base_weather = config;
        self.publish_camera_and_particles_and_ocean();
    }

    /// Installs or clears a custom unified [`WeatherProvider`] driving precipitation, clouds,
    /// sky darkening, and ocean wind transitions coherently.
    pub fn set_weather_provider(&mut self, provider: Option<Arc<dyn WeatherProvider>>) {
        self.weather_provider = provider.clone();
        self.worker.set_weather_provider(provider);
        self.publish_camera_and_particles_and_ocean();
        if !self.stopped.load(Ordering::Relaxed) {
            let idle_f = self.idle_refinement_factor();
            self.worker
                .request_update_with_idle(self.input.clone(), idle_f);
        }
    }

    /// Returns the most recently evaluated unified [`WeatherSnapshot`], if `base_weather.enabled` is active.
    #[inline]
    pub fn last_weather_snapshot(&self) -> Option<WeatherSnapshot> {
        self.last_weather_snapshot
    }

    #[inline]
    pub fn weather_config(&self) -> &WeatherConfig {
        &self.input.weather
    }

    /// Updates global & local cloud-shell configuration in O(1) without terrain rebuilds.
    pub fn set_weather_config(&mut self, weather: WeatherConfig) {
        let need_worker = self.input.weather.global_cloud_map != weather.global_cloud_map
            || self.input.weather.cloud_resolution != weather.cloud_resolution;
        self.input.weather = weather;
        self.publish_camera_and_overlays();
        if need_worker && !self.stopped.load(Ordering::Relaxed) {
            self.worker.request_update(self.input.clone());
        }
    }

    #[inline]
    pub fn precipitation_config(&self) -> &PrecipitationConfig {
        &self.input.precipitation
    }

    /// Updates GPU-instanced 2D camera-local precipitation & atmospheric configuration in O(1).
    pub fn set_precipitation_config(&mut self, precipitation: PrecipitationConfig) {
        let enabled = precipitation.enabled;
        self.input.precipitation = precipitation;
        if enabled {
            if self.plugins.contains(PrecipitationPlugin::ID) {
                let _ = self.plugins.set_enabled(PrecipitationPlugin::ID, true);
            } else {
                self.plugins.register(PrecipitationPlugin::new());
            }
        }
        self.publish_camera_and_particles();
    }

    /// Convenience toggle enabling or disabling [`PrecipitationPlugin`] rendering in O(1).
    pub fn set_precipitation_enabled(&mut self, enabled: bool) {
        let mut cfg = self.input.precipitation.clone();
        cfg.enabled = enabled;
        self.set_precipitation_config(cfg);
    }

    /// Installs a custom functional [`PrecipitationEvaluator`] on the active [`PrecipitationPlugin`].
    pub fn set_precipitation_evaluator(&mut self, evaluator: PrecipitationEvaluator) {
        if !self.plugins.contains(PrecipitationPlugin::ID) {
            self.plugins.register(PrecipitationPlugin::new());
        }
        let _ = self
            .plugins
            .with_plugin_mut::<PrecipitationPlugin, _>(PrecipitationPlugin::ID, |p| {
                p.set_evaluator(evaluator);
            });
        self.publish_camera_and_particles();
    }

    /// Installs or clears a custom [`TemperatureProvider`] on the active [`PrecipitationPlugin`].
    pub fn set_temperature_provider(&mut self, provider: Option<Arc<dyn TemperatureProvider>>) {
        if !self.plugins.contains(PrecipitationPlugin::ID) {
            self.plugins.register(PrecipitationPlugin::new());
        }
        let _ = self
            .plugins
            .with_plugin_mut::<PrecipitationPlugin, _>(PrecipitationPlugin::ID, |p| {
                p.set_temperature_provider(provider);
            });
        self.publish_camera_and_particles();
    }

    /// Returns the most recently evaluated [`AtmosphereSample`] from [`PrecipitationPlugin`], if any.
    pub fn last_atmosphere_sample(&self) -> Option<AtmosphereSample> {
        self.plugins
            .with_plugin::<PrecipitationPlugin, _>(PrecipitationPlugin::ID, |p| {
                p.last_atmosphere_sample()
            })
            .flatten()
    }

    #[inline]
    pub fn ocean_waves_config(&self) -> &OceanWavesConfig {
        &self.input.ocean_waves
    }

    /// Updates optional GPU-native physical ocean waves & coastal surf configuration in O(1).
    pub fn set_ocean_waves_config(&mut self, ocean_waves: OceanWavesConfig) {
        let was_enabled = self.input.ocean_waves.enabled;
        let enabled = ocean_waves.enabled;
        let coastal_turned_on =
            !self.input.ocean_waves.coastal_shoaling && ocean_waves.coastal_shoaling;
        self.input.ocean_waves = ocean_waves;
        if enabled {
            if self.plugins.contains(OceanWavesPlugin::ID) {
                let _ = self.plugins.set_enabled(OceanWavesPlugin::ID, true);
            } else {
                self.plugins.register(OceanWavesPlugin::new());
            }
        }
        self.publish_camera_and_ocean();
        if ((!was_enabled && enabled) || coastal_turned_on) && !self.stopped.load(Ordering::Relaxed)
        {
            self.worker.request_update(self.input.clone());
        }
    }

    /// Convenience toggle enabling or disabling animated wavy ocean renders on non-land surfaces.
    pub fn set_ocean_waves_enabled(&mut self, enabled: bool) {
        let mut cfg = self.input.ocean_waves.clone();
        cfg.enabled = enabled;
        self.set_ocean_waves_config(cfg);
    }

    /// Updates simulation UNIX UTC timestamp (O(1) solar/lunar/sidereal/satellite uniform update).
    pub fn set_sim_time_unix_s(&mut self, unix_utc_s: f64) {
        self.input.sim_time_unix_s = unix_utc_s;
        self.publish_camera_and_overlays();
    }

    /// Returns `true` when global cloud spin, active precipitation, or unified weather precipitation need ticks.
    pub fn has_active_weather_animation(&self) -> bool {
        let ll = self.input.low_level_state();
        let flat_t = ll.local_25d_blend.max(ll.map_2d_morph);
        let w = &self.input.weather;
        let spin_3d = self.plugins.is_active(WeatherPlugin::ID)
            && w.global_cloud_map
            && self.input.layers.clouds
            && flat_t < 0.85
            && w.global_cloud_spin_rad_s.abs() > 1e-9;
        let precip = self.plugins.is_active(PrecipitationPlugin::ID)
            && ((self.input.precipitation.enabled && self.input.precipitation.intensity > 0.002)
                || (self.input.base_weather.enabled
                    && self.input.precipitation.sync_with_base_weather));
        spin_3d || precip
    }

    /// Advances active weather & precipitation animations by `dt_s` seconds synchronously in O(1).
    pub fn step_weather(&mut self, dt_s: f64) -> bool {
        if !self.has_active_weather_animation() {
            return false;
        }
        let ws = self.resolve_weather_snapshot();
        let ground_elevation_m = self.sample_ground_elevation_m(
            self.input.camera.focus.lat_deg,
            self.input.camera.focus.lon_deg,
        );
        let ctx = EarthPluginFrameContext {
            input: &self.input,
            low_level: self.input.low_level_state(),
            altitude_m: self.input.camera.altitude_m(),
            ground_elevation_m,
            time_s: self.input.weather.weather_time_s,
            overlay_rev: self.overlay_rev,
            is_dragging: false,
            weather_snapshot: ws,
        };
        let (other_dirty, particle_dirty, ocean_dirty) =
            self.plugins.run_tick_detailed(dt_s, &ctx);
        if other_dirty {
            self.publish_camera_and_overlays();
            true
        } else if particle_dirty && ocean_dirty {
            self.publish_camera_and_particles_and_ocean();
            true
        } else if particle_dirty {
            self.publish_camera_and_particles();
            true
        } else if ocean_dirty {
            self.publish_camera_and_ocean();
            true
        } else {
            false
        }
    }

    /// Advances all animated plugins (precipitation, GPU ocean waves, cloud spin, satellites, custom plugins) in O(1).
    pub fn step_dynamic_overlays(&mut self, dt_s: f64, tick_satellites: bool) -> bool {
        let ll = self.input.low_level_state();
        let ws = self.resolve_weather_snapshot();
        let ground_elevation_m = self.sample_ground_elevation_m(
            self.input.camera.focus.lat_deg,
            self.input.camera.focus.lon_deg,
        );
        let ctx = EarthPluginFrameContext {
            input: &self.input,
            low_level: ll,
            altitude_m: self.input.camera.altitude_m(),
            ground_elevation_m,
            time_s: self.input.weather.weather_time_s,
            overlay_rev: self.overlay_rev,
            is_dragging: !tick_satellites,
            weather_snapshot: ws,
        };
        if !self.plugins.any_wants_animation_tick(&ctx) {
            return false;
        }
        let (other_dirty, particle_dirty, ocean_dirty) =
            self.plugins.run_tick_detailed(dt_s, &ctx);
        if other_dirty {
            self.publish_camera_and_overlays();
            true
        } else if particle_dirty && ocean_dirty {
            self.publish_camera_and_particles_and_ocean();
            true
        } else if particle_dirty {
            self.publish_camera_and_particles();
            true
        } else if ocean_dirty {
            self.publish_camera_and_ocean();
            true
        } else {
            false
        }
    }

    /// Returns the high-level view mode (`Mode3D` or `Mode2D`).
    #[inline]
    pub fn view_mode(&self) -> EarthViewMode {
        self.input.view_mode
    }

    /// Returns the low-level internal surface regime and blend weights (`local_25d_blend`, `map_2d_morph`).
    #[inline]
    pub fn low_level_state(&self) -> EarthLowLevelState {
        self.input.low_level_state()
    }

    /// Programmatically initiates a smooth animated transition between `Mode3D` (with automatic
    /// seamless 2.5D ground transition) and `Mode2D`.
    pub fn set_view_mode(&mut self, mode: EarthViewMode) {
        self.input.view_mode = mode;
        self.target_morph = match mode {
            EarthViewMode::Mode3D => 0.0,
            EarthViewMode::Mode2D => 1.0,
        };
        // Avoid a 1-frame jump to map_2d_morph=1.0 when initiating 3D -> 2D animation
        // (since `low_level_state` falls back to view_mode when `view_morph == 0.0`).
        if mode == EarthViewMode::Mode2D
            && self.linear_morph < 1e-4
            && self.input.view_morph <= 0.0
        {
            self.input.view_morph = 1e-5;
        }
    }

    /// Programmatically switches between `Mode3D` and `Mode2D` immediately without animation.
    pub fn set_view_mode_immediate(&mut self, mode: EarthViewMode) {
        let prev_canon = Self::is_input_canonical_2d(&self.input);
        self.input.view_mode = mode;
        let val = match mode {
            EarthViewMode::Mode3D => 0.0,
            EarthViewMode::Mode2D => 1.0,
        };
        self.target_morph = val;
        self.linear_morph = val;
        self.input.view_morph = val;
        let next_canon = Self::is_input_canonical_2d(&self.input);
        if prev_canon != next_canon {
            if !self.stopped.load(Ordering::Relaxed) {
                self.worker.request_update(self.input.clone());
                self.worker.flush();
            }
            self.publish_camera_and_overlays();
        } else {
            self.publish_camera_and_overlays();
            if !self.stopped.load(Ordering::Relaxed) {
                self.worker.request_update(self.input.clone());
            }
        }
    }

    /// Returns whether automatic seamless 3D -> 2.5D ground transition is enabled.
    #[inline]
    pub fn seamless_25d(&self) -> bool {
        self.input.seamless_25d
    }

    /// Enables or disables automatic seamless 3D -> 2.5D ground flattening at runtime.
    pub fn set_seamless_25d(&mut self, enabled: bool) {
        if self.input.seamless_25d == enabled {
            return;
        }
        self.input.seamless_25d = enabled;
        if !self.stopped.load(Ordering::Relaxed) {
            self.worker.request_update(self.input.clone());
        }
        self.publish_camera_and_overlays();
    }

    /// Returns `true` while a programmatic 3D <-> 2D morph transition is still in progress.
    #[inline]
    pub fn is_transitioning(&self) -> bool {
        (self.linear_morph - self.target_morph).abs() > 1e-4
    }

    /// Advances any active programmatic 3D <-> 2D transition by `dt_s` seconds.
    ///
    /// Returns `true` if a frame update was produced.
    pub fn step_transition(&mut self, dt_s: f64) -> bool {
        let prev_canon = Self::is_input_canonical_2d(&self.input);
        let diff = self.target_morph - self.linear_morph;
        if diff.abs() <= 1e-4 {
            if (self.input.view_morph - self.target_morph).abs() > 1e-4 {
                self.linear_morph = self.target_morph;
                self.input.view_morph = self.target_morph;
                let next_canon = Self::is_input_canonical_2d(&self.input);
                if prev_canon != next_canon {
                    if !self.stopped.load(Ordering::Relaxed) {
                        self.worker.request_update(self.input.clone());
                        self.worker.flush();
                    }
                    self.publish_camera_and_overlays();
                } else {
                    self.publish_camera_and_overlays();
                    if !self.stopped.load(Ordering::Relaxed) {
                        self.worker.request_update(self.input.clone());
                    }
                }
                return true;
            }
            return false;
        }
        let step = (dt_s.max(0.001) / self.transition_duration_s.max(0.05)) as f32;
        if diff > 0.0 {
            self.linear_morph = (self.linear_morph + step).min(self.target_morph);
        } else {
            self.linear_morph = (self.linear_morph - step).max(self.target_morph);
        }
        let t = self.linear_morph.clamp(0.0, 1.0);
        // Cubic Hermite ease-in-out. When stepping from 2D -> 3D (`target_morph == 0.0`),
        // immediately drop below `CANONICAL_2D_MORPH_THRESHOLD` on the first step so the
        // worker synchronously flushes full polar-cap globe tiles instead of lingering in
        // canonical 2D tile culling for the first 1-2 frames.
        let mut eased = t * t * (3.0 - 2.0 * t);
        if self.target_morph < 0.5 && self.linear_morph < 1.0 - 1e-5 {
            eased = eased.min((CANONICAL_2D_MORPH_THRESHOLD - 1e-3) as f32);
        }
        self.input.view_morph = eased;
        let next_canon = Self::is_input_canonical_2d(&self.input);
        if prev_canon != next_canon {
            if !self.stopped.load(Ordering::Relaxed) {
                self.worker.request_update(self.input.clone());
                self.worker.flush();
            }
            self.publish_camera_and_overlays();
        } else {
            self.publish_camera_and_overlays();
            if !self.stopped.load(Ordering::Relaxed) {
                self.worker.request_update(self.input.clone());
            }
        }
        true
    }

    /// Updates camera uniforms in O(1) (`camera_generation += 1`) with zero mesh/DOM/layout rebuild,
    /// and notifies the background worker to update LOD tiles asynchronously.
    pub fn update_camera(&mut self, camera: &EarthCamera, viewport_w: f64, viewport_h: f64) {
        self.update_camera_with_up(camera, self.input.camera_up_ecef, viewport_w, viewport_h);
    }

    fn publish_camera_and_world(&mut self) {
        let ws = self.resolve_weather_snapshot();
        let cam_u = self.camera_uniforms_with_weather(ws);
        let (assets, instances) = self.compile_world_retained();
        self.retained
            .update_camera_and_geo_objects(cam_u, assets, instances);
    }

    /// Updates camera and optional explicit ECEF camera-up hint in O(1).
    pub fn update_camera_with_up(
        &mut self,
        camera: &EarthCamera,
        camera_up_ecef: Option<[f64; 3]>,
        viewport_w: f64,
        viewport_h: f64,
    ) {
        self.input.camera = *camera;
        self.input.camera_up_ecef = camera_up_ecef;
        self.input.viewport.width = viewport_w.max(1.0) as f32;
        self.input.viewport.height = viewport_h.max(1.0) as f32;
        let ll = self.input.low_level_state();
        let flat_t = ll.local_25d_blend.max(ll.map_2d_morph);
        let ocean_hooked = self
            .plugins
            .active_hooks()
            .intersects(EarthPluginHooks::OCEAN);
        let need_ocean =
            ocean_hooked && (self.input.ocean_waves.enabled || self.had_active_ocean);
        let particles_hooked = self
            .plugins
            .active_hooks()
            .intersects(EarthPluginHooks::PARTICLES);
        let need_particles = particles_hooked
            && (self.input.precipitation.enabled
                || self.had_active_particles
                || (self.input.base_weather.enabled
                    && self.input.precipitation.sync_with_base_weather));

        let (has_world, has_px_sized) = self
            .world
            .read()
            .map(|w| (!w.is_empty(), w.has_pixel_sized_objects()))
            .unwrap_or((false, false));
        let need_geo_refresh = if has_world {
            let cur_sig = mesh_morph_signature(self.current_surface_morph());
            let cur_dist = camera.eye_distance_m();
            let cur_lon_sector =
                (crate::geodesy::wrap_longitude(camera.focus.lon_deg) / 30.0).round() as i32;
            let wrap_sector_changed =
                Self::is_input_canonical_2d(&self.input) && cur_lon_sector != self.last_geo_lon_sector;
            cur_sig != self.last_geo_morph_sig
                || wrap_sector_changed
                || (has_px_sized
                    && (cur_dist - self.last_geo_eye_dist_m).abs()
                        > self.last_geo_eye_dist_m.max(1.0) * 0.025)
        } else {
            false
        };

        if (flat_t - self.last_overlay_flat_t).abs() > 0.015 {
            self.publish_camera_and_overlays();
        } else if need_geo_refresh && (need_particles || need_ocean) {
            let ws = self.resolve_weather_snapshot();
            let cam_u = self.camera_uniforms_with_weather(ws);
            let dyn_meshes = self.build_dynamic_overlays();
            let (assets, instances) = self.compile_world_retained();
            let (p_layers, p_textures) = self.build_particle_layers(ws);
            let ocean = self.build_ocean_surface(ws);
            self.retained
                .update_camera_and_world_and_particles_and_ocean(
                    cam_u, dyn_meshes, assets, instances, p_layers, p_textures, ocean,
                );
        } else if need_geo_refresh {
            self.publish_camera_and_world();
        } else if need_particles && need_ocean {
            self.publish_camera_and_particles_and_ocean();
        } else if need_particles {
            self.publish_camera_and_particles();
        } else if need_ocean {
            self.publish_camera_and_ocean();
        } else {
            let ws = self.resolve_weather_snapshot();
            let cam_u = self.camera_uniforms_with_weather(ws);
            self.retained.update_camera(cam_u);
        }
        let pose_sig = Self::camera_pose_signature(&self.input);
        if pose_sig != self.last_camera_pose_sig {
            self.last_camera_pose_sig = pose_sig;
            self.idle_elapsed_s = 0.0;
            self.idle_stage = 0;
        }
        if !self.stopped.load(Ordering::Relaxed) {
            let idle_f = self.idle_refinement_factor();
            self.worker
                .request_update_with_idle(self.input.clone(), idle_f);
        }
    }

    /// Returns the active [`IdleHighResConfig`] (`enabled: false` by default).
    #[inline]
    pub fn idle_high_res_config(&self) -> IdleHighResConfig {
        self.input.idle_high_res
    }

    /// Updates opt-in idle high-resolution view refinement settings.
    pub fn set_idle_high_res_config(&mut self, config: IdleHighResConfig) {
        let changed = self.input.idle_high_res != config;
        self.input.idle_high_res = config;
        if !config.enabled {
            let was_refined = self.idle_stage != 0;
            self.idle_elapsed_s = 0.0;
            self.idle_stage = 0;
            if was_refined && !self.stopped.load(Ordering::Relaxed) {
                self.worker.request_update_with_idle(self.input.clone(), 0.0);
            }
        } else if changed {
            self.idle_elapsed_s = 0.0;
            self.idle_stage = 0;
        }
    }

    /// Convenience toggle enabling or disabling optional idle high-resolution refinement.
    pub fn set_idle_high_res_enabled(&mut self, enabled: bool) {
        let mut cfg = self.input.idle_high_res;
        cfg.enabled = enabled;
        self.set_idle_high_res_config(cfg);
    }

    /// Returns the live unified [`crate::scene::EarthRenderQuality`] profile extracted from this viewport.
    #[inline]
    pub fn render_quality(&self) -> crate::scene::EarthRenderQuality {
        crate::scene::EarthRenderQuality::from_input(&self.input)
    }

    /// Returns the matching canonical [`crate::scene::EarthQualityPreset`] (or `Custom` if diverged).
    #[inline]
    pub fn quality_preset(&self) -> crate::scene::EarthQualityPreset {
        self.render_quality().matching_preset()
    }

    /// Applies a unified [`crate::scene::EarthRenderQuality`] profile live on this viewport,
    /// refreshing camera uniforms, shadows, ocean/sky state, and waking the background worker
    /// only when mesh/asset parameters changed.
    pub fn set_render_quality(&mut self, quality: crate::scene::EarthRenderQuality) {
        let prev = self.render_quality();
        let prev_night_tiles = self.input.local_atmosphere.night_stars_sync
            && self.input.local_atmosphere.terrain_lighting;
        let next_night_tiles = quality.realistic_sky_25d && quality.terrain_lighting_25d;
        let idle_changed = prev.idle_high_res != quality.idle_high_res
            || prev.idle_extra_leaf_budget != quality.idle_extra_leaf_budget;
        let was_refined = self.idle_stage != 0;

        let need_worker = prev.tile_grid != quality.tile_grid
            || (prev.sse_threshold - quality.sse_threshold).abs() > 1e-4
            || prev.max_tile_level != quality.max_tile_level
            || prev.skybox_quality != quality.skybox_quality
            || prev.cloud_resolution != quality.cloud_resolution
            || prev_night_tiles != next_night_tiles
            || (!prev.ocean_coastal_shoaling && quality.ocean_coastal_shoaling)
            || (!quality.idle_high_res && was_refined);

        quality.apply_to_input(&mut self.input);

        if idle_changed {
            self.idle_elapsed_s = 0.0;
            self.idle_stage = 0;
        }

        self.publish_camera_and_overlays();
        if need_worker && !self.stopped.load(Ordering::Relaxed) {
            let idle_f = self.idle_refinement_factor();
            self.worker
                .request_update_with_idle(self.input.clone(), idle_f);
        }
    }

    /// Applies one of the canonical [`crate::scene::EarthQualityPreset`] profiles live.
    pub fn set_quality_preset(&mut self, preset: crate::scene::EarthQualityPreset) {
        self.set_render_quality(crate::scene::EarthRenderQuality::from_preset(preset));
    }

    /// Returns the active terrain tile subdivision grid density.
    #[inline]
    pub fn tile_grid(&self) -> u32 {
        self.input.tile_grid
    }

    /// Updates terrain mesh tile subdivision density (`5..=65`) and triggers a background rebuild.
    pub fn set_tile_grid(&mut self, tile_grid: u32) {
        let clamped = tile_grid.clamp(5, 65);
        if self.input.tile_grid == clamped {
            return;
        }
        self.input.tile_grid = clamped;
        if !self.stopped.load(Ordering::Relaxed) {
            let idle_f = self.idle_refinement_factor();
            self.worker
                .request_update_with_idle(self.input.clone(), idle_f);
        }
    }

    /// Returns the active quadtree SSE threshold in pixels.
    #[inline]
    pub fn sse_threshold(&self) -> f64 {
        self.input.sse_threshold
    }

    /// Updates quadtree SSE threshold (`0.5..=12.0` px) and triggers a background LOD refresh.
    pub fn set_sse_threshold(&mut self, sse_threshold: f64) {
        let clamped = sse_threshold.clamp(0.5, 12.0);
        if (self.input.sse_threshold - clamped).abs() <= 1e-6 {
            return;
        }
        self.input.sse_threshold = clamped;
        if !self.stopped.load(Ordering::Relaxed) {
            let idle_f = self.idle_refinement_factor();
            self.worker
                .request_update_with_idle(self.input.clone(), idle_f);
        }
    }

    /// Returns the active maximum quadtree tile zoom level.
    #[inline]
    pub fn max_tile_level(&self) -> u8 {
        self.input.max_tile_level
    }

    /// Updates maximum quadtree tile zoom level (`4..=21`) and triggers a background LOD refresh.
    pub fn set_max_tile_level(&mut self, max_tile_level: u8) {
        let clamped = max_tile_level.clamp(4, 21);
        if self.input.max_tile_level == clamped {
            return;
        }
        self.input.max_tile_level = clamped;
        if !self.stopped.load(Ordering::Relaxed) {
            let idle_f = self.idle_refinement_factor();
            self.worker
                .request_update_with_idle(self.input.clone(), idle_f);
        }
    }

    /// Immediately centers the viewport camera focus over `(lat_deg, lon_deg)` at `range_m`
    /// (preserving the current camera range when `None`).
    pub fn center_on_coordinates(&mut self, lat_deg: f64, lon_deg: f64, range_m: Option<f64>) {
        if !lat_deg.is_finite() || !lon_deg.is_finite() {
            return;
        }
        let mut cam = self.input.camera;
        cam.focus = crate::geodesy::Geodetic::new(
            crate::geodesy::clamp_latitude(lat_deg),
            crate::geodesy::wrap_longitude(lon_deg),
            0.0,
        );
        if let Some(r) = range_m.filter(|v| v.is_finite() && *v > 0.0) {
            let target = r.clamp(
                crate::controller::MIN_RANGE_M,
                crate::controller::MAX_RANGE_M,
            );
            let [ex, ey, ez] = cam.eye_offset_m;
            let norm = (ex * ex + ey * ey + ez * ez).sqrt();
            if norm > 1e-3 {
                let s = target / norm;
                cam.eye_offset_m = [ex * s, ey * s, ez * s];
            } else {
                cam.eye_offset_m = [0.0, 0.0, target];
            }
        }
        let vw = self.input.viewport.width.max(1.0) as f64;
        let vh = self.input.viewport.height.max(1.0) as f64;
        self.update_camera_with_up(&cam, None, vw, vh);
    }

    /// Returns the current progressive idle refinement weight (`0.0` moving/off, `0.5` stage 1, `1.0` stage 2).
    #[inline]
    pub fn idle_refinement_factor(&self) -> f32 {
        if !self.input.idle_high_res.enabled {
            0.0
        } else {
            match self.idle_stage {
                0 => 0.0,
                1 => 0.5,
                _ => 1.0,
            }
        }
    }

    /// Returns `true` when idle high-res refinement is enabled and currently refining the stationary view.
    #[inline]
    pub fn is_idle_high_res_active(&self) -> bool {
        self.input.idle_high_res.enabled && self.idle_stage > 0
    }

    /// Advances stationary-camera idle high-resolution refinement by `dt_s` seconds.
    ///
    /// Does zero work when [`IdleHighResConfig::enabled`] is `false`. When enabled and
    /// `is_interacting == false`, promotes refinement in two smooth stages (`0.5` after
    /// `idle_delay_s`, `1.0` after `idle_delay_s * 2.2`), queueing a single background LOD
    /// pass per promotion so viewed tiles progressively sharpen without flooding the queue.
    pub fn step_idle_refinement(&mut self, dt_s: f64, is_interacting: bool) -> bool {
        if !self.input.idle_high_res.enabled {
            let was_refined = self.idle_stage != 0;
            self.idle_elapsed_s = 0.0;
            self.idle_stage = 0;
            if was_refined && !self.stopped.load(Ordering::Relaxed) {
                self.worker.request_update_with_idle(self.input.clone(), 0.0);
            }
            return was_refined;
        }
        if is_interacting {
            let was_refined = self.idle_stage != 0;
            self.idle_elapsed_s = 0.0;
            self.idle_stage = 0;
            if was_refined && !self.stopped.load(Ordering::Relaxed) {
                self.worker.request_update_with_idle(self.input.clone(), 0.0);
            }
            return was_refined;
        }
        let dt = dt_s.clamp(0.0, 0.25);
        self.idle_elapsed_s += dt;
        let delay1 = self.input.idle_high_res.idle_delay_s.clamp(0.05, 10.0);
        let delay2 = delay1 * 2.2;
        let next_stage = if self.idle_elapsed_s >= delay2 {
            2
        } else if self.idle_elapsed_s >= delay1 {
            1
        } else {
            0
        };
        if next_stage != self.idle_stage {
            self.idle_stage = next_stage;
            if !self.stopped.load(Ordering::Relaxed) {
                let idle_f = self.idle_refinement_factor();
                self.worker
                    .request_update_with_idle(self.input.clone(), idle_f);
            }
            return true;
        }
        false
    }

    /// Updates full [`EarthInput`] configuration (`layers`, `surface_mode`, `camera`, `viewport`, etc.).
    pub fn update_input(&mut self, input: &EarthInput) {
        let prev_canon = Self::is_input_canonical_2d(&self.input);
        self.input = input.clone();
        let target = self.input.low_level_state().map_2d_morph;
        self.target_morph = target;
        self.linear_morph = target;
        let pose_sig = Self::camera_pose_signature(&self.input);
        if pose_sig != self.last_camera_pose_sig || !self.input.idle_high_res.enabled {
            self.last_camera_pose_sig = pose_sig;
            self.idle_elapsed_s = 0.0;
            self.idle_stage = 0;
        }
        let next_canon = Self::is_input_canonical_2d(&self.input);
        if prev_canon != next_canon {
            if !self.stopped.load(Ordering::Relaxed) {
                let idle_f = self.idle_refinement_factor();
                self.worker
                    .request_update_with_idle(self.input.clone(), idle_f);
                self.worker.flush();
            }
            self.publish_camera_and_overlays();
        } else {
            self.publish_camera_and_overlays();
            if !self.stopped.load(Ordering::Relaxed) {
                let idle_f = self.idle_refinement_factor();
                self.worker
                    .request_update_with_idle(self.input.clone(), idle_f);
            }
        }
    }

    /// Returns the active [`TileProviderConfig`] configured on this viewport.
    pub fn tile_provider(&self) -> &TileProviderConfig {
        &self.active_provider
    }

    /// Switches the active Tile API provider at runtime, clearing stale GPU/network imagery
    /// caches while preserving elevation meshes when the elevation source is unchanged.
    pub fn set_tile_provider(&mut self, config: TileProviderConfig) {
        let elev_changed = self.active_provider.elevation != config.elevation
            || self.active_provider.api_key != config.api_key
            || self.active_provider.headers != config.headers;
        self.active_provider = config.clone();
        self.input.max_tile_level = config.max_imagery_level.clamp(4, 19);
        self.idle_elapsed_s = 0.0;
        self.idle_stage = 0;
        self.worker.set_tile_provider(config, elev_changed);
        if !self.stopped.load(Ordering::Relaxed) {
            self.worker.request_update_with_idle(self.input.clone(), 0.0);
        }
    }

    /// Switches the active map/satellite preset on this viewport cleanly.
    pub fn set_tile_preset(&mut self, preset: TileProviderPreset) {
        self.set_tile_provider(TileProviderConfig::preset(preset));
    }

    /// Replaces the underlying [`TileSource`] cleanly and rebuilds terrain textures & meshes.
    pub fn set_tile_source(&mut self, source: Box<dyn TileSource + Send>) {
        self.idle_elapsed_s = 0.0;
        self.idle_stage = 0;
        self.worker.set_tile_source(source);
        if !self.stopped.load(Ordering::Relaxed) {
            self.worker.request_update_with_idle(self.input.clone(), 0.0);
        }
    }

    /// Blocks until the most recently queued worker request has updated `RetainedScene3D`.
    pub fn sync_now(&self) {
        self.worker.flush();
    }

    pub fn stats(&self) -> EarthViewportStats {
        self.worker.stats()
    }

    /// Samples terrain elevation in metres above the WGS84 ellipsoid at `(lat_deg, lon_deg)`.
    ///
    /// Returns `0.0` when elevation displacement is disabled or no elevation patch covers the point.
    pub fn sample_ground_elevation_m(&self, lat_deg: f64, lon_deg: f64) -> f64 {
        if !self.input.elevation_enabled {
            return 0.0;
        }
        self.worker.sample_ground_elevation_m(lat_deg, lon_deg)
    }

    /// Borrows the geodetic [`EarthWorld`] for read inspection.
    pub fn world<R>(&self, f: impl FnOnce(&EarthWorld) -> R) -> R {
        let guard = self.world.read().expect("EarthWorld lock poisoned");
        f(&guard)
    }

    /// Mutates the geodetic [`EarthWorld`] and immediately publishes updated GPU object instances.
    pub fn world_mut<R>(&mut self, f: impl FnOnce(&mut EarthWorld) -> R) -> R {
        let out = {
            let mut guard = self.world.write().expect("EarthWorld lock poisoned");
            f(&mut guard)
        };
        self.refresh_world_objects();
        out
    }

    /// Spawns a geodetic [`EarthObject`] into the world and immediately publishes its GPU instance.
    pub fn spawn_object(&mut self, object: EarthObject) -> EarthObjectId {
        self.world_mut(|w| w.spawn(object))
    }

    /// Validated convenience wrapper spawning a glTF 2.0 binary (`.glb`) object and returning a `Result`.
    pub fn spawn_glb(
        &mut self,
        position: crate::geodesy::GeoPosition,
        bytes: &[u8],
        unit_scale_m: f64,
        anchor: crate::world::ObjectAnchor,
    ) -> Result<EarthObjectId, String> {
        self.world_mut(|w| w.spawn_glb(position, bytes, unit_scale_m, anchor))
    }

    /// Updates a [`crate::world::EarthGeometry::Polyline`] object's vertices in place without leaking CPU or GPU meshes.
    pub fn set_polyline_points(
        &mut self,
        id: EarthObjectId,
        points: Vec<crate::geodesy::GeoPosition>,
    ) -> bool {
        self.world_mut(|w| w.set_polyline_points(id, points))
    }

    /// Updates a [`crate::world::EarthGeometry::Polyline`] object's tube radius in place without leaking CPU or GPU meshes.
    pub fn set_polyline_radius(&mut self, id: EarthObjectId, radius_m: f64) -> bool {
        self.world_mut(|w| w.set_polyline_radius(id, radius_m))
    }

    /// Replaces `id`'s [`crate::world::EarthGeometry`] in place and evicts any unreferenced mesh buffers.
    pub fn replace_geometry(
        &mut self,
        id: EarthObjectId,
        geometry: crate::world::EarthGeometry,
    ) -> bool {
        self.world_mut(|w| w.replace_geometry(id, geometry))
    }

    /// Removes a geodetic [`EarthObject`] by ID, returning `true` if it existed.
    pub fn remove_object(&mut self, id: EarthObjectId) -> bool {
        self.world_mut(|w| w.remove(id))
    }

    /// Removes all geodetic [`EarthObject`]s from the world.
    pub fn clear_objects(&mut self) {
        self.world_mut(|w| w.clear());
    }

    /// Returns `true` if the world currently contains at least one visible object with a non-empty [`EarthObject::label`].
    pub fn has_labeled_objects(&self) -> bool {
        let Ok(guard) = self.world.read() else {
            return false;
        };
        let has = guard.iter().any(|(_, obj)| {
            obj.visible
                && obj
                    .label
                    .as_ref()
                    .is_some_and(|s| !s.trim().is_empty())
        });
        has
    }

    /// Projects all visible [`EarthObject::label`] annotations into screen coordinates for HUD overlay rendering.
    pub fn project_object_labels(&self) -> Vec<crate::labels::ProjectedEarthLabel> {
        let Ok(guard) = self.world.read() else {
            return Vec::new();
        };
        if guard.is_empty() {
            return Vec::new();
        }
        let morph = self.current_surface_morph();
        let placement = self
            .input
            .camera
            .placement_for_morph(morph, self.camera_up_dvec());
        let vp_h = self.input.viewport.height.max(1.0) as f64;
        let fov_y = self.input.camera.fov_y_deg;
        let elev_on = self.input.elevation_enabled;
        let worker_ref = &self.worker;
        let projector = crate::labels::EarthScreenProjector::from_input(&self.input);
        let vw = self.input.viewport.width.max(1.0);
        let vh = self.input.viewport.height.max(1.0);
        let mut out = Vec::new();
        for (id, obj) in guard.iter() {
            if !obj.visible {
                continue;
            }
            let Some(ref raw_label) = obj.label else {
                continue;
            };
            let text = raw_label.trim();
            if text.is_empty() {
                continue;
            }
            let Some(tf) = guard.resolve_object_transform(
                id,
                placement.eye,
                vp_h,
                fov_y,
                elev_on,
                morph,
                &|lat, lon| {
                    if elev_on {
                        worker_ref.sample_ground_elevation_m(lat, lon)
                    } else {
                        0.0
                    }
                },
            ) else {
                continue;
            };
            let z_col = tf.model_cols[2];
            let z_scale = ((z_col[0] as f64).powi(2)
                + (z_col[1] as f64).powi(2)
                + (z_col[2] as f64).powi(2))
            .sqrt();
            let top_lift_m = (tf.local_aabb_max[2].max(0.0) as f64) * z_scale;
            let label_alt_m = tf.resolved_altitude_m + top_lift_m;
            let Some((raw_x, raw_y, ndc_x, ndc_y, horizon_fade)) = projector.project_raw(
                obj.position.lat_deg,
                obj.position.lon_deg,
                label_alt_m,
            ) else {
                continue;
            };
            if horizon_fade <= 0.03 || ndc_x.abs() > 0.98 || ndc_y.abs() > 0.98 {
                continue;
            }
            let c = obj.material.color;
            let rgb = [
                ((c.r().clamp(0.45, 1.0)) * 255.0).round() as u8,
                ((c.g().clamp(0.45, 1.0)) * 255.0).round() as u8,
                ((c.b().clamp(0.45, 1.0)) * 255.0).round() as u8,
            ];
            out.push(crate::labels::ProjectedEarthLabel {
                id: 0xE0B7_0000_0000_0000u64 ^ id.0,
                text: text.to_string(),
                subtitle: None,
                kind: crate::labels::EarthLabelKind::City,
                icon: crate::labels::EarthLabelIcon::CityDot,
                screen_x: raw_x.clamp(16.0, (vw - 16.0).max(16.0)),
                screen_y: (raw_y - 10.0).clamp(14.0, (vh - 14.0).max(14.0)),
                alpha: horizon_fade.clamp(0.0, 1.0),
                rgb,
                font_px: 12.5,
            });
        }
        out
    }

    /// Evaluates the live world transform (including resolved `Terrain` vs `SeaLevel` altitude)
    /// for `id` under the current heightmap and view-morph state.
    pub fn resolve_object_transform(&self, id: EarthObjectId) -> Option<ResolvedObjectTransform> {
        let morph = self.current_surface_morph();
        let placement = self
            .input
            .camera
            .placement_for_morph(morph, self.camera_up_dvec());
        let vp_h = self.input.viewport.height.max(1.0) as f64;
        let fov_y = self.input.camera.fov_y_deg;
        let elev_on = self.input.elevation_enabled;
        let worker_ref = &self.worker;
        let guard = self.world.read().ok()?;
        guard.resolve_object_transform(
            id,
            placement.eye,
            vp_h,
            fov_y,
            elev_on,
            morph,
            &|lat, lon| {
                if elev_on {
                    worker_ref.sample_ground_elevation_m(lat, lon)
                } else {
                    0.0
                }
            },
        )
    }

    /// Raycasts viewport pixel `(screen_x, screen_y)` against pickable [`EarthWorld`] objects
    /// and the active planetary surface (terrain heightmap when `elevation_enabled`, sea level when off).
    pub fn raycast(&self, screen_x: f64, screen_y: f64) -> Option<EarthRayHit> {
        let morph = self.current_surface_morph();
        let elev_on = self.input.elevation_enabled;
        let worker_ref = &self.worker;
        let guard = self.world.read().ok()?;
        raycast_viewport(
            screen_x,
            screen_y,
            self.input.viewport.width as f64,
            self.input.viewport.height as f64,
            &self.input.camera,
            self.camera_up_dvec(),
            elev_on,
            morph,
            Some(&guard),
            &|lat, lon| {
                if elev_on {
                    worker_ref.sample_ground_elevation_m(lat, lon)
                } else {
                    0.0
                }
            },
        )
    }

    /// Raycasts viewport pixel `(screen_x, screen_y)` strictly against the planetary surface
    /// (ignoring pickable [`EarthObject`]s).
    pub fn raycast_terrain(&self, screen_x: f64, screen_y: f64) -> Option<EarthRayHit> {
        let morph = self.current_surface_morph();
        let elev_on = self.input.elevation_enabled;
        let worker_ref = &self.worker;
        let (ray_origin, ray_dir) = screen_to_ray_for_morph(
            screen_x,
            screen_y,
            self.input.viewport.width as f64,
            self.input.viewport.height as f64,
            &self.input.camera,
            morph,
            self.camera_up_dvec(),
        );
        raycast_surface_ray(ray_origin, ray_dir, elev_on, morph, &|lat, lon| {
            if elev_on {
                worker_ref.sample_ground_elevation_m(lat, lon)
            } else {
                0.0
            }
        })
    }
}
