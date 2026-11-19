//! Asynchronous tile streaming, frustum-culled LOD selection, and retained GPU mesh builder.

use crate::astro::{build_environment_assets_with_skybox, GlobalCloudMapStore};
use crate::geodesy::{
    build_tile_local_rtc_mesh_morphed, level_for_altitude, retained_camera_uniforms_with_up,
    stabilized_surface_morph, tile_bounding_sphere, tile_bounds, tile_bounds_mercator,
    traverse_quadtree_lod_2d_flat, traverse_quadtree_lod_frustum6_projected_with_up, wgs84,
    EarthCamera, Frustum6, Geodetic, LocalFrame, PatchBounds, SurfaceMorphParams, TileKey,
    TileProjection, MAX_QUADTREE_LEAVES,
};
use crate::gltf::{CustomWeatherMeshes, WeatherConfig};
use crate::plugins::{
    CachedPluginTexEntry, EarthDynamicOverlayContext, EarthPlugin, EarthPluginFrameContext,
    EarthPluginHooks, EarthPluginRegistry, EarthTextureInjector, EarthTileBuildContext,
    EarthWorldBuildContext, LoadedTileImagerySnapshot, OceanWavesPlugin, WeatherPlugin,
};
use crate::raycast::{raycast_surface_ray, raycast_viewport, EarthRayHit};
use crate::satellites::{SatelliteCatalogStore, SatelliteElement};
use crate::scene::{
    compute_adaptive_skirt_depth, requested_tile, sky_gradient, surrounding_patches,
    tile_patch_span_deg, EarthInput, EarthLowLevelState, EarthViewMode, OceanWavesConfig,
    SpaceEnvironmentConfig, SurfaceMode,
};
use crate::skybox::SpaceSkyboxStore;
use crate::tiles::{
    TileElevation, TileImage, TileProviderConfig, TileProviderPreset, TileSource,
};
use crate::world::{EarthObject, EarthObjectId, EarthWorld, ResolvedObjectTransform};
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, RwLock};
use std::thread::JoinHandle;
use strata_style::types::Color;
use strata_render::camera64::DVec3;
use strata_render::scene3d::{
    Background3D, RetainedCameraUniforms, RetainedDynamicMesh, RetainedEnvironmentAssets,
    RetainedScene3D, RetainedTerrainFrame, RetainedTileInstance, RetainedTileMesh,
    RetainedTileTexture, TileKeyId,
};

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

/// Persistent CPU cache converting visible LOD leaves into GPU-ready tile-local RTC meshes and textures.
pub struct RetainedTerrainBuilder {
    meshes: HashMap<(TileKeyId, u32, bool), CachedMeshEntry>,
    textures: HashMap<TileKey, CachedTexEntry>,
    night_textures: HashMap<TileKey, CachedTexEntry>,
    plugin_textures: HashMap<u64, CachedPluginTexEntry>,
    cached_elev_arcs: HashMap<TileKey, (u64, Arc<TileElevation>)>,
    ground_snapshot: Option<Arc<RwLock<GroundElevSnapshot>>>,
    world: Arc<RwLock<EarthWorld>>,
    next_tex_id: u64,
    next_revision: u64,
    frame_counter: u64,
    cloud_store: Arc<GlobalCloudMapStore>,
    satellite_store: Arc<SatelliteCatalogStore>,
    skybox_store: Arc<SpaceSkyboxStore>,
    plugins: EarthPluginRegistry,
    last_plugin_rev: u64,
    last_ocean_waves: OceanWavesConfig,
    custom_weather: Option<CustomWeatherMeshes>,
    weather_ids: HashMap<u64, u64>,
    weather_rev: u64,
    last_cloud_rev: u64,
    last_sat_rev: u64,
    last_skybox_rev: u64,
    last_built_weather: Option<(i64, i64, WeatherConfig)>,
    cached_env: Option<((u64, u64), Arc<RetainedEnvironmentAssets>)>,
}

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
        Self {
            meshes: HashMap::new(),
            textures: HashMap::new(),
            night_textures: HashMap::new(),
            plugin_textures: HashMap::new(),
            cached_elev_arcs: HashMap::new(),
            ground_snapshot: Some(Arc::new(RwLock::new(GroundElevSnapshot::default()))),
            world: Arc::new(RwLock::new(EarthWorld::new())),
            next_tex_id: 1,
            next_revision: 1,
            frame_counter: 0,
            cloud_store,
            satellite_store,
            skybox_store,
            plugins,
            last_plugin_rev,
            last_ocean_waves: OceanWavesConfig::default(),
            custom_weather: None,
            weather_ids: HashMap::new(),
            weather_rev: 0,
            last_cloud_rev: 0,
            last_sat_rev: 0,
            last_skybox_rev: 0,
            last_built_weather: None,
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

    pub fn set_custom_weather_meshes(&mut self, custom: Option<CustomWeatherMeshes>) {
        self.custom_weather = custom;
        self.last_built_weather = None;
    }

    /// Clears cached GPU tile textures (and optionally elevation meshes) when switching tile providers.
    pub fn clear_provider_caches(&mut self, clear_elevation: bool) {
        self.textures.clear();
        self.night_textures.clear();
        self.plugin_textures.clear();
        if clear_elevation {
            self.meshes.clear();
            self.cached_elev_arcs.clear();
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

    /// Returns `true` when neither background cloudmap/skybox/satellites/plugins nor 2.5D weather meshes require rebuilding.
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
        let ll = input.low_level_state();
        let active_25d = ll.local_25d_blend.max(ll.map_2d_morph) > 0.02
            && (input.weather.rain || input.weather.snow);
        if !active_25d {
            return self.last_built_weather.is_none();
        }
        let lat_q = (input.camera.focus.lat_deg * 1e5).round() as i64;
        let lon_q = (input.camera.focus.lon_deg * 1e5).round() as i64;
        self.last_built_weather
            .as_ref()
            .is_some_and(|(a, b, w)| *a == lat_q && *b == lon_q && w == &input.weather)
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
        let flat_t = ll.local_25d_blend.max(ll.map_2d_morph).clamp(0.0, 1.0);
        let in_2d_hide = map_t > 0.05;
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
        let up_override = if in_2d_hide {
            None
        } else {
            input.camera_up_ecef.map(|u| DVec3::new(u[0], u[1], u[2]))
        };
        let mut u = retained_camera_uniforms_with_up(
            &cam,
            up_override,
            input.viewport.width as f64,
            input.viewport.height as f64,
            [0.577_350_26, 0.577_350_26, 0.577_350_26],
            sun_color,
            ambient_color,
            bg,
        );
        u.local_25d_blend = flat_t;
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

    pub fn camera_from_input_with_plugins(
        input: &EarthInput,
        plugins: &EarthPluginRegistry,
        time_s: f64,
        overlay_rev: u64,
        is_dragging: bool,
    ) -> RetainedCameraUniforms {
        let mut u = Self::base_camera_from_input(input);
        let ctx = EarthPluginFrameContext {
            input,
            low_level: input.low_level_state(),
            altitude_m: input.camera.altitude_m(),
            time_s,
            overlay_rev,
            is_dragging,
        };
        plugins.run_prepare_camera(&ctx, &mut u);
        u
    }

    pub fn camera_from_input(input: &EarthInput) -> RetainedCameraUniforms {
        let mut u = Self::base_camera_from_input(input);
        let ctx = EarthPluginFrameContext {
            input,
            low_level: input.low_level_state(),
            altitude_m: input.camera.altitude_m(),
            time_s: input.weather.weather_time_s,
            overlay_rev: 0,
            is_dragging: false,
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
        const MAX_CACHED_MESHES: usize = 512;
        const MAX_CACHED_TEXTURES: usize = 768;
        const MAX_CACHED_NIGHT: usize = 256;

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
            anc = a.parent();
        }
        if probe.level > 11 {
            if let Some(p11) = Self::ancestor_at_level(probe, 11) {
                if let Ok(Some(elev)) = source.elevation(p11) {
                    return Ok((Some(elev), p11));
                }
            }
        }
        Ok((None, key))
    }

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
            && flat_t < 0.85
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
            && flat_t < 0.85
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
            unlit: false,
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

    /// Builds a [`RetainedTerrainFrame`] from cached tile-local RTC geometry & textures.
    pub fn build_frame(
        &mut self,
        input: &EarthInput,
        source: &mut dyn TileSource,
        geometry_revision: u64,
        allow_ancestor_mesh_fallback: bool,
    ) -> (Arc<RetainedTerrainFrame>, RetainedBuildStats) {
        self.frame_counter = self.frame_counter.wrapping_add(1);
        self.last_plugin_rev = self.plugins.revision();
        self.last_ocean_waves = input.ocean_waves.clone();
        source.begin_frame(input.camera.focus, input.camera.altitude_m());
        let ll = input.low_level_state();
        let frame_ctx = EarthPluginFrameContext {
            input,
            low_level: ll,
            altitude_m: input.camera.altitude_m(),
            time_s: input.weather.weather_time_s,
            overlay_rev: self.weather_rev.wrapping_add(1),
            is_dragging: false,
        };
        let camera = Self::camera_from_input_with_plugins(
            input,
            &self.plugins,
            input.weather.weather_time_s,
            self.weather_rev.wrapping_add(1),
            false,
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
                let horizon_cull = ll.map_2d_morph <= 0.05;
                let up_override = if ll.map_2d_morph > 0.05 {
                    None
                } else {
                    input.camera_up_ecef.map(|u| DVec3::new(u[0], u[1], u[2]))
                };
                let is_flat_2d = ll.map_2d_morph >= 0.99;
                let leaf_items: Vec<(TileKey, i32)> = if is_flat_2d {
                    traverse_quadtree_lod_2d_flat(
                        &input.camera,
                        input.viewport.width.max(1.0) as f64,
                        input.viewport.height.max(1.0) as f64,
                        input.sse_threshold,
                        input.max_tile_level,
                        grid,
                        MAX_QUADTREE_LEAVES,
                        input.tile_projection,
                    )
                } else {
                    traverse_quadtree_lod_frustum6_projected_with_up(
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
                    )
                    .into_iter()
                    .map(|k| (k, 0))
                    .collect()
                };
                let focus_east = LocalFrame::at(morph.focus.lat_deg, morph.focus.lon_deg).east;
                let world_circ = 2.0 * std::f64::consts::PI * wgs84::A;
                let eff_elev_enabled =
                    input.elevation_enabled && !is_flat_2d && input.camera.altitude_m() <= 350_000.0;
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
                let mut emitted_ancestors = HashSet::new();
                for ((leaf, wrap_x), (leaf_want_night, leaf_elev_enabled)) in
                    leaf_items.into_iter().zip(leaf_flags.into_iter())
                {
                    visible_leaves.push(leaf);
                    let elev_lookup =
                        self.resolve_elevation_or_ancestor(leaf, leaf_elev_enabled, source);
                    let elev_err = elev_lookup.is_err();
                    let (elev_opt, elev_tile) = elev_lookup.unwrap_or((None, leaf));
                    let elev = elev_opt.as_ref();
                    let elev_hash = hash_elevation_64(elev);
                    record_frame_elevation(
                        &mut self.cached_elev_arcs,
                        &mut frame_elevs,
                        elev_tile,
                        elev_hash,
                        elev,
                    );
                    let key_id = TileKeyId::quadtree(leaf.level, leaf.x, leaf.y);

                    let use_ancestor_mesh = !is_flat_2d
                        && allow_ancestor_mesh_fallback
                        && elev_err
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
                    let mesh = self.ensure_tile_mesh(
                        leaf,
                        key_id,
                        bounds,
                        grid,
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

                    let mesh = self.ensure_tile_mesh(
                        tile,
                        key_id,
                        bounds,
                        grid,
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
        let active_25d = flat_t > 0.02 && (input.weather.rain || input.weather.snow);
        self.last_built_weather = if active_25d {
            let lat_q = (input.camera.focus.lat_deg * 1e5).round() as i64;
            let lon_q = (input.camera.focus.lon_deg * 1e5).round() as i64;
            Some((lat_q, lon_q, input.weather.clone()))
        } else {
            None
        };
        self.weather_rev = self.weather_rev.wrapping_add(1);
        if hooks.intersects(EarthPluginHooks::DYNAMIC_OVERLAYS) {
            let mut dyn_ctx = EarthDynamicOverlayContext {
                input,
                low_level: ll,
                flat_t,
                time_s: input.weather.weather_time_s,
                overlay_rev: self.weather_rev,
                custom_weather: self.custom_weather.as_ref(),
                satellite_elements: &sat_elements,
                weather_ids: &mut self.weather_ids,
                ground_snapshot: self.ground_snapshot.as_deref(),
            };
            self.plugins
                .run_dynamic_overlays(&mut dyn_ctx, &mut dynamic_meshes);
        }

        let morph = stabilized_morph_params(
            input.camera.focus,
            flat_t as f64,
            ll.map_2d_morph as f64,
            input.tile_projection,
        );
        let eye_dvec = DVec3::new(camera.eye_ecef[0], camera.eye_ecef[1], camera.eye_ecef[2]);
        let (geo_mesh_assets, geo_instances) = if let Ok(mut w) = self.world.write() {
            if w.is_empty() {
                (Arc::default(), Arc::default())
            } else {
                w.compile_retained(
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
                )
            }
        } else {
            (Arc::default(), Arc::default())
        };

        stats.visible_tiles = instances_out.len();
        (
            Arc::new(RetainedTerrainFrame {
                camera,
                geometry_revision,
                meshes: meshes_out,
                instances: instances_out,
                textures: textures_out,
                dynamic_meshes,
                geo_mesh_assets,
                geo_instances,
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
        let min_skirt = if morph.flat_blend > 0.05 { 600.0 } else { 0.0 };
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

struct WorkerShared {
    latest_input: Option<(u64, EarthInput)>,
    last_dispatched: Option<(u64, EarthInput)>,
    custom_weather: Option<CustomWeatherMeshes>,
    custom_weather_gen: u64,
    pending_provider: Option<(TileProviderConfig, bool)>,
    pending_source: Option<Box<dyn TileSource + Send>>,
    provider_gen: u64,
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
                custom_weather: None,
                custom_weather_gen: 0,
                pending_provider: None,
                pending_source: None,
                provider_gen: 0,
                shutdown: false,
                stats: EarthViewportStats::default(),
            }),
            Condvar::new(),
        ));
        let completed_seq = Arc::new(AtomicU64::new(0));
        let ground_snapshot = Arc::new(RwLock::new(GroundElevSnapshot::default()));
        let shared_bg = Arc::clone(&shared);
        let shared_env = Arc::clone(&shared);
        let completed_bg = Arc::clone(&completed_seq);
        let ground_snapshot_bg = Arc::clone(&ground_snapshot);
        let world_bg = Arc::clone(&world);
        let env_wakeup_outer = wakeup.clone();
        let env_wakeup: Arc<dyn Fn() + Send + Sync> = Arc::new(move || {
            {
                let (lock, cv) = &*shared_env;
                if let Ok(mut guard) = lock.lock() {
                    if guard.latest_input.is_none() {
                        if let Some((seq, ref inp)) = guard.last_dispatched {
                            guard.latest_input = Some((seq, inp.clone()));
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
                builder.set_ground_snapshot(ground_snapshot_bg);
                builder.set_world(world_bg);
                let mut geom_rev = 0u64;
                let mut seen_custom_gen = 0u64;
                let mut seen_provider_gen = 0u64;
                loop {
                    let (seq, input, custom_update, provider_update, source_update) = {
                        let (lock, cv) = &*shared_bg;
                        let mut guard = lock.lock().unwrap();
                        while guard.latest_input.is_none() && !guard.shutdown {
                            guard = cv.wait(guard).unwrap();
                        }
                        if guard.shutdown {
                            return;
                        }
                        let cw = if guard.custom_weather_gen != seen_custom_gen {
                            seen_custom_gen = guard.custom_weather_gen;
                            Some(guard.custom_weather.clone())
                        } else {
                            None
                        };
                        let (prov_up, src_up) = if guard.provider_gen != seen_provider_gen {
                            seen_provider_gen = guard.provider_gen;
                            (guard.pending_provider.take(), guard.pending_source.take())
                        } else {
                            (None, None)
                        };
                        let pair = guard.latest_input.take().unwrap();
                        (pair.0, pair.1, cw, prov_up, src_up)
                    };

                    if let Some(cw) = custom_update {
                        builder.set_custom_weather_meshes(cw);
                    }
                    if let Some(mut new_src) = source_update {
                        new_src.set_tile_ready_callback(Some(Arc::clone(&tile_wakeup)));
                        source = new_src;
                        builder.clear_provider_caches(true);
                    } else if let Some((cfg, clear_elev)) = provider_update {
                        let _ = source.set_provider(&cfg);
                        builder.clear_provider_caches(clear_elev);
                    }

                    geom_rev += 1;
                    let (frame, build_stats) =
                        builder.build_frame(&input, source.as_mut(), geom_rev, true);
                    retained.update_terrain_preserving_camera(frame);

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
            thread: Some(handle),
            next_seq: 0,
        }
    }

    /// Samples terrain height in metres above WGS84 from the latest worker frame's elevation tiles.
    pub fn sample_ground_elevation_m(&self, lat_deg: f64, lon_deg: f64) -> f64 {
        self.ground_snapshot
            .read()
            .map(|s| s.sample_elevation_m(lat_deg, lon_deg))
            .unwrap_or(0.0)
    }

    /// Updates custom GLTF weather meshes on the background builder and schedules a refresh.
    pub fn set_custom_weather_meshes(&mut self, custom: Option<CustomWeatherMeshes>) {
        let (lock, cv) = &*self.shared;
        if let Ok(mut guard) = lock.lock() {
            guard.custom_weather = custom;
            guard.custom_weather_gen = guard.custom_weather_gen.wrapping_add(1);
            if guard.latest_input.is_none() {
                if let Some((seq, ref inp)) = guard.last_dispatched {
                    guard.latest_input = Some((seq, inp.clone()));
                }
            }
            cv.notify_one();
        }
    }

    /// Switches the active [`TileProviderConfig`] on the background tile source and clears stale GPU textures.
    pub fn set_tile_provider(&mut self, config: TileProviderConfig, clear_elevation: bool) {
        let (lock, cv) = &*self.shared;
        if let Ok(mut guard) = lock.lock() {
            guard.pending_provider = Some((config, clear_elevation));
            guard.pending_source = None;
            guard.provider_gen = guard.provider_gen.wrapping_add(1);
            if guard.latest_input.is_none() {
                if let Some((seq, ref inp)) = guard.last_dispatched {
                    guard.latest_input = Some((seq, inp.clone()));
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
                if let Some((seq, ref inp)) = guard.last_dispatched {
                    guard.latest_input = Some((seq, inp.clone()));
                }
            }
            cv.notify_one();
        }
    }

    /// Queues `input` for background LOD traversal + tile assembly, coalescing any unstarted request.
    pub fn request_update(&mut self, input: EarthInput) -> u64 {
        self.next_seq += 1;
        let seq = self.next_seq;
        let (lock, cv) = &*self.shared;
        if let Ok(mut guard) = lock.lock() {
            guard.last_dispatched = Some((seq, input.clone()));
            guard.latest_input = Some((seq, input));
            guard.stats.camera_updates += 1;
            cv.notify_one();
        }
        seq
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
    target_morph: f32,
    linear_morph: f32,
    transition_duration_s: f64,
    custom_weather: Option<CustomWeatherMeshes>,
    weather_ids: HashMap<u64, u64>,
    overlay_rev: u64,
    last_overlay_flat_t: f32,
    last_geo_morph_sig: u64,
    last_geo_eye_dist_m: f64,
    stopped: Arc<AtomicBool>,
}

impl EarthViewport {
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
        let world = Arc::new(RwLock::new(EarthWorld::new()));
        let mut worker = EarthTileWorker::spawn_with_plugins_and_world(
            retained.clone(),
            source,
            wakeup,
            plugins.clone(),
            Arc::clone(&world),
        );
        worker.request_update(input.clone());
        Self {
            retained,
            worker,
            world,
            input: input.clone(),
            plugins,
            active_provider: TileProviderConfig::default(),
            target_morph: initial_morph,
            linear_morph: initial_morph,
            transition_duration_s: 0.45,
            custom_weather: None,
            weather_ids: HashMap::new(),
            overlay_rev: 1,
            last_overlay_flat_t: initial_flat_t,
            last_geo_morph_sig: 0,
            last_geo_eye_dist_m: input.camera.eye_distance_m(),
            stopped: Arc::new(AtomicBool::new(false)),
        }
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
        self.refresh_after_plugin_change();
    }

    /// Enables or disables a registered plugin by `id`.
    pub fn set_plugin_enabled(&mut self, id: &str, enabled: bool) -> bool {
        let updated = self.plugins.set_enabled(id, enabled);
        if updated {
            if id == OceanWavesPlugin::ID {
                self.input.ocean_waves.enabled = enabled;
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
        let placement = self.input.camera.placement_with_up(self.camera_up_dvec());
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
                custom_weather: self.custom_weather.as_ref(),
                satellite_elements: &elements,
                weather_ids: &mut self.weather_ids,
                ground_snapshot: Some(self.worker.ground_snapshot.as_ref()),
            };
            self.plugins.run_dynamic_overlays(&mut ctx, &mut out);
        }
        out
    }

    fn publish_camera_and_overlays(&mut self) {
        let cam_u = RetainedTerrainBuilder::camera_from_input_with_plugins(
            &self.input,
            &self.plugins,
            self.input.weather.weather_time_s,
            self.overlay_rev,
            false,
        );
        let dyn_meshes = self.build_dynamic_overlays();
        let has_world = self.world.read().map(|w| !w.is_empty()).unwrap_or(false);
        if has_world {
            let (assets, instances) = self.compile_world_retained();
            self.retained
                .update_camera_and_world_overlays(cam_u, dyn_meshes, assets, instances);
        } else {
            self.retained
                .update_camera_and_dynamic_meshes(cam_u, dyn_meshes);
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
    pub fn weather_config(&self) -> &WeatherConfig {
        &self.input.weather
    }

    /// Updates global cloudmap & 2.5D clouds/rain/snow toggles in O(1) without terrain rebuilds.
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
    pub fn ocean_waves_config(&self) -> &OceanWavesConfig {
        &self.input.ocean_waves
    }

    /// Updates optional animated wavy ocean rendering configuration on non-land water surfaces.
    pub fn set_ocean_waves_config(&mut self, ocean_waves: OceanWavesConfig) {
        let enabled = ocean_waves.enabled;
        let changed = self.input.ocean_waves != ocean_waves;
        self.input.ocean_waves = ocean_waves;
        if enabled {
            if self.plugins.contains(OceanWavesPlugin::ID) {
                let _ = self.plugins.set_enabled(OceanWavesPlugin::ID, true);
            } else {
                self.plugins.register(OceanWavesPlugin::new());
            }
        }
        self.publish_camera_and_overlays();
        if changed && !self.stopped.load(Ordering::Relaxed) {
            self.worker.request_update(self.input.clone());
        }
    }

    /// Convenience toggle enabling or disabling animated wavy ocean renders on non-land surfaces.
    pub fn set_ocean_waves_enabled(&mut self, enabled: bool) {
        let mut cfg = self.input.ocean_waves.clone();
        cfg.enabled = enabled;
        self.set_ocean_waves_config(cfg);
    }

    /// Replaces the default embedded CC0 weather meshes with custom parsed GLTF meshes.
    pub fn set_custom_weather_meshes(&mut self, custom: Option<CustomWeatherMeshes>) {
        self.custom_weather = custom.clone();
        self.worker.set_custom_weather_meshes(custom);
        self.publish_camera_and_overlays();
    }

    /// Updates simulation UNIX UTC timestamp (O(1) solar/lunar/sidereal/satellite uniform update).
    pub fn set_sim_time_unix_s(&mut self, unix_utc_s: f64) {
        self.input.sim_time_unix_s = unix_utc_s;
        self.publish_camera_and_overlays();
    }

    /// Returns `true` when animated 2.5D precipitation/clouds or global cloud spin need ticking.
    pub fn has_active_weather_animation(&self) -> bool {
        if !self.plugins.is_active(WeatherPlugin::ID) {
            return false;
        }
        let ll = self.input.low_level_state();
        let flat_t = ll.local_25d_blend.max(ll.map_2d_morph);
        let w = &self.input.weather;
        let spin_3d = w.global_cloud_map
            && self.input.layers.clouds
            && flat_t < 0.85
            && w.global_cloud_spin_rad_s.abs() > 1e-9;
        let weather_25d = flat_t > 0.02
            && ((w.rain && w.rain_density > 0) || (w.snow && w.snow_density > 0));
        spin_3d || weather_25d
    }

    /// Advances active weather animations by `dt_s` seconds synchronously in O(1) without quadtree rebuilds.
    pub fn step_weather(&mut self, dt_s: f64) -> bool {
        if !self.has_active_weather_animation() {
            return false;
        }
        let ctx = EarthPluginFrameContext {
            input: &self.input,
            low_level: self.input.low_level_state(),
            altitude_m: self.input.camera.altitude_m(),
            time_s: self.input.weather.weather_time_s,
            overlay_rev: self.overlay_rev,
            is_dragging: false,
        };
        if self.plugins.run_tick(dt_s, &ctx) {
            self.publish_camera_and_overlays();
            return true;
        }
        false
    }

    /// Advances all animated plugins (ocean waves, 2.5D/global weather, orbital satellites, custom plugins) in O(1).
    pub fn step_dynamic_overlays(&mut self, dt_s: f64, tick_satellites: bool) -> bool {
        let ll = self.input.low_level_state();
        let ctx = EarthPluginFrameContext {
            input: &self.input,
            low_level: ll,
            altitude_m: self.input.camera.altitude_m(),
            time_s: self.input.weather.weather_time_s,
            overlay_rev: self.overlay_rev,
            is_dragging: !tick_satellites,
        };
        if !self.plugins.any_wants_animation_tick(&ctx) {
            return false;
        }
        let dirty = self.plugins.run_tick(dt_s, &ctx);
        if dirty {
            self.publish_camera_and_overlays();
        }
        dirty
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
    }

    /// Programmatically switches between `Mode3D` and `Mode2D` immediately without animation.
    pub fn set_view_mode_immediate(&mut self, mode: EarthViewMode) {
        self.input.view_mode = mode;
        let val = match mode {
            EarthViewMode::Mode3D => 0.0,
            EarthViewMode::Mode2D => 1.0,
        };
        self.target_morph = val;
        self.linear_morph = val;
        self.input.view_morph = val;
        self.publish_camera_and_overlays();
        if !self.stopped.load(Ordering::Relaxed) {
            self.worker.request_update(self.input.clone());
        }
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
        let diff = self.target_morph - self.linear_morph;
        if diff.abs() <= 1e-4 {
            if (self.input.view_morph - self.target_morph).abs() > 1e-4 {
                self.linear_morph = self.target_morph;
                self.input.view_morph = self.target_morph;
                self.publish_camera_and_overlays();
                if !self.stopped.load(Ordering::Relaxed) {
                    self.worker.request_update(self.input.clone());
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
        // Cubic Hermite ease-in-out
        self.input.view_morph = t * t * (3.0 - 2.0 * t);
        self.publish_camera_and_overlays();
        if !self.stopped.load(Ordering::Relaxed) {
            self.worker.request_update(self.input.clone());
        }
        true
    }

    /// Updates camera uniforms in O(1) (`camera_generation += 1`) with zero mesh/DOM/layout rebuild,
    /// and notifies the background worker to update LOD tiles asynchronously.
    pub fn update_camera(&mut self, camera: &EarthCamera, viewport_w: f64, viewport_h: f64) {
        self.update_camera_with_up(camera, self.input.camera_up_ecef, viewport_w, viewport_h);
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
        let ocean_active = self.input.ocean_waves.enabled
            && self.plugins.is_active(OceanWavesPlugin::ID)
            && camera.altitude_m() < 520_000.0;

        let has_world = self.world.read().map(|w| !w.is_empty()).unwrap_or(false);
        let need_geo_refresh = if has_world {
            let cur_sig = mesh_morph_signature(self.current_surface_morph());
            let cur_dist = camera.eye_distance_m();
            cur_sig != self.last_geo_morph_sig
                || (cur_dist - self.last_geo_eye_dist_m).abs()
                    > self.last_geo_eye_dist_m.max(1.0) * 0.006
        } else {
            false
        };

        if (flat_t - self.last_overlay_flat_t).abs() > 0.015 || ocean_active || need_geo_refresh {
            self.publish_camera_and_overlays();
        } else {
            let cam_u = RetainedTerrainBuilder::camera_from_input_with_plugins(
                &self.input,
                &self.plugins,
                self.input.weather.weather_time_s,
                self.overlay_rev,
                false,
            );
            self.retained.update_camera(cam_u);
        }
        if !self.stopped.load(Ordering::Relaxed) {
            self.worker.request_update(self.input.clone());
        }
    }

    /// Updates full [`EarthInput`] configuration (`layers`, `surface_mode`, `camera`, `viewport`, etc.).
    pub fn update_input(&mut self, input: &EarthInput) {
        self.input = input.clone();
        let target = self.input.low_level_state().map_2d_morph;
        self.target_morph = target;
        self.linear_morph = target;
        self.publish_camera_and_overlays();
        if !self.stopped.load(Ordering::Relaxed) {
            self.worker.request_update(self.input.clone());
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
        self.worker.set_tile_provider(config, elev_changed);
        if !self.stopped.load(Ordering::Relaxed) {
            self.worker.request_update(self.input.clone());
        }
    }

    /// Switches the active map/satellite preset on this viewport cleanly.
    pub fn set_tile_preset(&mut self, preset: TileProviderPreset) {
        self.set_tile_provider(TileProviderConfig::preset(preset));
    }

    /// Replaces the underlying [`TileSource`] cleanly and rebuilds terrain textures & meshes.
    pub fn set_tile_source(&mut self, source: Box<dyn TileSource + Send>) {
        self.worker.set_tile_source(source);
        if !self.stopped.load(Ordering::Relaxed) {
            self.worker.request_update(self.input.clone());
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

    /// Removes a geodetic [`EarthObject`] by ID, returning `true` if it existed.
    pub fn remove_object(&mut self, id: EarthObjectId) -> bool {
        self.world_mut(|w| w.remove(id))
    }

    /// Removes all geodetic [`EarthObject`]s from the world.
    pub fn clear_objects(&mut self) {
        self.world_mut(|w| w.clear());
    }

    /// Evaluates the live world transform (including resolved `Terrain` vs `SeaLevel` altitude)
    /// for `id` under the current heightmap and view-morph state.
    pub fn resolve_object_transform(&self, id: EarthObjectId) -> Option<ResolvedObjectTransform> {
        let morph = self.current_surface_morph();
        let placement = self.input.camera.placement_with_up(self.camera_up_dvec());
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
        let (ray_origin, ray_dir) = crate::geodesy::screen_to_ray_with_up(
            screen_x,
            screen_y,
            self.input.viewport.width as f64,
            self.input.viewport.height as f64,
            &self.input.camera,
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
