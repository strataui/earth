//! Modular world-pipeline plugins and mid-work injection hooks for `strata_earth`.
//!
//! Plugins intercept and extend every phase of Earth world construction without modifying
//! the underlying general-purpose `strata_render` engine:
//! - `on_tick`: advance animated simulation state in O(1) per frame.
//! - `on_prepare_camera`: adjust camera lighting/uniforms before draw.
//! - `on_prepare_environment`: inspect or replace global masks, cloud maps, stars, skyboxes.
//! - `on_tile_begin`: decide per-tile build options mid-traversal (e.g. night tiles, elevation).
//! - `on_tile_instance`: rewrite instances or inject custom tile/night textures in-flight.
//! - `on_world_build`: inject or mutate meshes, instances, textures, or dynamic geometry after traversal.
//! - `on_dynamic_overlays`: emit fast 60 FPS animated dynamic overlays without rebuilding terrain.

use std::any::Any;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use strata_render::scene3d::{
    RetainedCameraUniforms, RetainedDynamicMesh, RetainedEnvironmentAssets, RetainedOceanSurface,
    RetainedParticleLayer, RetainedTileInstance, RetainedTileMesh, RetainedTileTexture, TileKeyId,
};

use crate::astro::embedded_land_mask;
use crate::geodesy::{PatchBounds, TileKey, TileProjection};
pub use crate::ocean::{classify_non_land_water, OceanWavesPlugin};
use crate::precipitation::PrecipitationPlugin;
use crate::satellites::{build_3d_satellites_meshes, SatelliteElement};
use crate::scene::{EarthInput, EarthLowLevelState};
use crate::weather_api::WeatherSnapshot;
use crate::worker::{sample_tile_uv, GroundElevSnapshot};

/// Bitmask declaring which pipeline stages an [`EarthPlugin`] participates in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EarthPluginHooks(u32);

impl EarthPluginHooks {
    pub const EMPTY: Self = Self(0);
    pub const CAMERA: Self = Self(1 << 0);
    pub const ENVIRONMENT: Self = Self(1 << 1);
    pub const TILE: Self = Self(1 << 2);
    pub const WORLD_BUILD: Self = Self(1 << 3);
    pub const DYNAMIC_OVERLAYS: Self = Self(1 << 4);
    pub const TICK: Self = Self(1 << 5);
    pub const PARTICLES: Self = Self(1 << 6);
    pub const OCEAN: Self = Self(1 << 7);
    pub const ALL: Self = Self((1 << 8) - 1);

    #[inline]
    pub const fn empty() -> Self {
        Self::EMPTY
    }

    #[inline]
    pub const fn from_bits_truncate(bits: u32) -> Self {
        Self(bits & Self::ALL.0)
    }

    #[inline]
    pub const fn bits(self) -> u32 {
        self.0
    }

    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }

    #[inline]
    pub const fn intersects(self, other: Self) -> bool {
        (self.0 & other.0) != 0
    }

    #[inline]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
}

impl std::ops::BitOr for EarthPluginHooks {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

impl std::ops::BitOrAssign for EarthPluginHooks {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

#[inline]
fn smoothstep_f32(edge0: f32, edge1: f32, x: f32) -> f32 {
    if (edge1 - edge0).abs() < 1e-6 {
        return if x >= edge1 { 1.0 } else { 0.0 };
    }
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[inline]
fn hash_rgba_64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let step = (bytes.len() / 256).max(1);
    let mut i = 0;
    while i < bytes.len() {
        h ^= bytes[i] as u64;
        h = h.wrapping_mul(0x1000_0000_1b3);
        i += step;
    }
    h ^= bytes.len() as u64;
    h
}

/// Fast equirectangular sampler backed by the embedded Natural Earth land mask.
#[inline]
pub fn sample_land_mask_at(lat_deg: f64, lon_deg: f64) -> f32 {
    let mask = embedded_land_mask();
    let w = mask.width as usize;
    let h = mask.height as usize;
    if w == 0 || h == 0 || mask.rgba8.len() < w * h * 4 {
        return 0.0;
    }
    let lon_norm = (lon_deg + 180.0).rem_euclid(360.0);
    let u = (lon_norm / 360.0).clamp(0.0, 0.999_999);
    let v = (0.5 - lat_deg.clamp(-89.999, 89.999) / 180.0).clamp(0.0, 0.999_999);
    let fx = u * (w as f64 - 1.0);
    let fy = v * (h as f64 - 1.0);
    let x0 = fx.floor() as usize;
    let y0 = fy.floor() as usize;
    let x1 = (x0 + 1) % w;
    let y1 = (y0 + 1).min(h - 1);
    let tx = (fx - x0 as f64) as f32;
    let ty = (fy - y0 as f64) as f32;

    let r00 = mask.rgba8[(y0 * w + x0) * 4] as f32 / 255.0;
    let r10 = mask.rgba8[(y0 * w + x1) * 4] as f32 / 255.0;
    let r01 = mask.rgba8[(y1 * w + x0) * 4] as f32 / 255.0;
    let r11 = mask.rgba8[(y1 * w + x1) * 4] as f32 / 255.0;

    let top = r00 + (r10 - r00) * tx;
    let bot = r01 + (r11 - r01) * tx;
    (top + (bot - top) * ty).clamp(0.0, 1.0)
}

/// Zero-copy read-only view over cached tile imagery loaded in the latest world build.
#[derive(Clone)]
pub struct LoadedTileImagerySnapshot {
    pub projection: TileProjection,
    /// Sorted highest-zoom-level-first for finest coastal sampling.
    pub tiles: Vec<(TileKey, Arc<RetainedTileTexture>)>,
}

impl Default for LoadedTileImagerySnapshot {
    fn default() -> Self {
        Self {
            projection: TileProjection::WebMercator,
            tiles: Vec::new(),
        }
    }
}

impl LoadedTileImagerySnapshot {
    /// Samples the RGBA8 texel under `(lat_deg, lon_deg)` from the finest loaded tile, if any.
    pub fn sample_rgba(&self, lat_deg: f64, lon_deg: f64) -> Option<[u8; 4]> {
        let lon_norm = (lon_deg + 180.0).rem_euclid(360.0) - 180.0;
        for (tile, tex) in &self.tiles {
            if tex.width == 0 || tex.height == 0 {
                continue;
            }
            if let Some((u, v)) = sample_tile_uv(*tile, self.projection, lat_deg, lon_norm) {
                let px = ((u * (tex.width as f32 - 1.0)).round() as usize)
                    .min(tex.width as usize - 1);
                let py = ((v * (tex.height as f32 - 1.0)).round() as usize)
                    .min(tex.height as usize - 1);
                let idx = (py * tex.width as usize + px) * 4;
                if idx + 3 < tex.rgba8.len() {
                    return Some([
                        tex.rgba8[idx],
                        tex.rgba8[idx + 1],
                        tex.rgba8[idx + 2],
                        tex.rgba8[idx + 3],
                    ]);
                }
            }
        }
        None
    }
}

/// Cached entry for custom textures injected by plugins mid-work.
#[derive(Clone)]
pub(crate) struct CachedPluginTexEntry {
    pub(crate) id: u64,
    pub(crate) pixel_hash: u64,
    pub(crate) revision: u64,
    pub(crate) last_used_frame: u64,
    pub(crate) texture: Arc<RetainedTileTexture>,
}

/// Texture uploader allowing plugins to inject procedural or overlay textures per frame/tile.
pub struct EarthTextureInjector<'a> {
    pub(crate) textures_out: &'a mut Vec<Arc<RetainedTileTexture>>,
    pub(crate) seen_tex_ids: &'a mut HashSet<u64>,
    pub(crate) cache: &'a mut HashMap<u64, CachedPluginTexEntry>,
    pub(crate) next_tex_id: &'a mut u64,
    pub(crate) frame_counter: u64,
}

impl<'a> EarthTextureInjector<'a> {
    /// Registers or reuses an RGBA8 texture identified by caller `cache_key`, returning `(id, revision)`.
    pub fn inject_texture(
        &mut self,
        cache_key: u64,
        width: u32,
        height: u32,
        rgba8: Vec<u8>,
    ) -> (u64, u64) {
        let pixel_hash = hash_rgba_64(&rgba8);
        if let Some(entry) = self.cache.get_mut(&cache_key) {
            entry.last_used_frame = self.frame_counter;
            if entry.pixel_hash == pixel_hash
                && entry.texture.width == width
                && entry.texture.height == height
            {
                if self.seen_tex_ids.insert(entry.id) {
                    self.textures_out.push(Arc::clone(&entry.texture));
                }
                return (entry.id, entry.revision);
            }
            let rev = entry.revision + 1;
            let tex = Arc::new(RetainedTileTexture {
                id: entry.id,
                revision: rev,
                width,
                height,
                rgba8: Arc::new(rgba8),
            });
            let id = entry.id;
            entry.pixel_hash = pixel_hash;
            entry.revision = rev;
            entry.texture = Arc::clone(&tex);
            if self.seen_tex_ids.insert(id) {
                self.textures_out.push(tex);
            }
            return (id, rev);
        }

        let id = *self.next_tex_id;
        *self.next_tex_id += 1;
        let tex = Arc::new(RetainedTileTexture {
            id,
            revision: 1,
            width,
            height,
            rgba8: Arc::new(rgba8),
        });
        if self.seen_tex_ids.insert(id) {
            self.textures_out.push(Arc::clone(&tex));
        }
        self.cache.insert(
            cache_key,
            CachedPluginTexEntry {
                id,
                pixel_hash,
                revision: 1,
                last_used_frame: self.frame_counter,
                texture: tex,
            },
        );
        (id, 1)
    }
}

/// Lightweight per-frame metadata passed to camera/tick/environment plugin hooks.
pub struct EarthPluginFrameContext<'a> {
    pub input: &'a EarthInput,
    pub low_level: EarthLowLevelState,
    pub altitude_m: f64,
    /// Ground elevation (meters above WGS-84 sea level) beneath the camera/focus point.
    pub ground_elevation_m: f64,
    pub time_s: f64,
    pub overlay_rev: u64,
    pub is_dragging: bool,
    /// Single coherent weather snapshot evaluated from [`crate::weather_api::BaseWeatherConfig`], if enabled.
    pub weather_snapshot: Option<WeatherSnapshot>,
}

/// Mid-traversal per-tile decision & customization context inside [`RetainedTerrainBuilder::build_frame`].
pub struct EarthTileBuildContext<'a> {
    pub input: &'a EarthInput,
    pub key: TileKey,
    pub key_id: TileKeyId,
    pub bounds: PatchBounds,
    /// Plugins may enable or suppress night-light tile streaming/attachment per tile.
    pub want_night_tiles: bool,
    /// Plugins may enable or disable elevation displacement per tile.
    pub elevation_enabled: bool,
}

/// Mid-work world-assembly context after tile generation and before finalizing [`RetainedTerrainFrame`].
pub struct EarthWorldBuildContext<'a> {
    pub input: &'a EarthInput,
    pub camera: &'a RetainedCameraUniforms,
    pub low_level: EarthLowLevelState,
    pub visible_leaves: &'a [TileKey],
    pub meshes: &'a mut Vec<Arc<RetainedTileMesh>>,
    pub instances: &'a mut Vec<RetainedTileInstance>,
    pub textures: &'a mut Vec<Arc<RetainedTileTexture>>,
    pub dynamic_meshes: &'a mut Vec<RetainedDynamicMesh>,
    pub(crate) ground_snapshot: Option<&'a GroundElevSnapshot>,
    pub(crate) imagery_snapshot: Option<&'a LoadedTileImagerySnapshot>,
    pub(crate) tex_seen_ids: &'a mut HashSet<u64>,
    pub(crate) plugin_tex_cache: &'a mut HashMap<u64, CachedPluginTexEntry>,
    pub(crate) next_tex_id: &'a mut u64,
    pub(crate) frame_counter: u64,
}

impl<'a> EarthWorldBuildContext<'a> {
    #[inline]
    pub fn sample_ground_elevation_m(&self, lat_deg: f64, lon_deg: f64) -> f64 {
        self.ground_snapshot
            .map(|s| s.sample_elevation_m(lat_deg, lon_deg))
            .unwrap_or(0.0)
    }

    #[inline]
    pub fn sample_land_mask(&self, lat_deg: f64, lon_deg: f64) -> f32 {
        sample_land_mask_at(lat_deg, lon_deg)
    }

    #[inline]
    pub fn sample_imagery_rgba(&self, lat_deg: f64, lon_deg: f64) -> Option<[u8; 4]> {
        self.imagery_snapshot
            .and_then(|s| s.sample_rgba(lat_deg, lon_deg))
    }

    pub fn inject_texture(
        &mut self,
        cache_key: u64,
        width: u32,
        height: u32,
        rgba8: Vec<u8>,
    ) -> (u64, u64) {
        let mut inj = EarthTextureInjector {
            textures_out: self.textures,
            seen_tex_ids: self.tex_seen_ids,
            cache: self.plugin_tex_cache,
            next_tex_id: self.next_tex_id,
            frame_counter: self.frame_counter,
        };
        inj.inject_texture(cache_key, width, height, rgba8)
    }
}

/// Fast O(1) dynamic overlay emission context (called every interactive frame & rebuild).
pub struct EarthDynamicOverlayContext<'a> {
    pub input: &'a EarthInput,
    pub low_level: EarthLowLevelState,
    pub flat_t: f32,
    pub time_s: f64,
    pub overlay_rev: u64,
    pub satellite_elements: &'a [SatelliteElement],
    pub weather_ids: &'a mut HashMap<u64, u64>,
    pub(crate) ground_snapshot: Option<&'a RwLock<GroundElevSnapshot>>,
}

impl<'a> EarthDynamicOverlayContext<'a> {
    #[inline]
    pub fn sample_ground_elevation_m(&self, lat_deg: f64, lon_deg: f64) -> f64 {
        self.ground_snapshot
            .and_then(|lock| lock.read().ok().map(|s| s.sample_elevation_m(lat_deg, lon_deg)))
            .unwrap_or(0.0)
    }

    #[inline]
    pub fn sample_land_mask(&self, lat_deg: f64, lon_deg: f64) -> f32 {
        sample_land_mask_at(lat_deg, lon_deg)
    }
}

/// Fast O(1) GPU-instanced 2D camera-local particle volume emission context.
pub struct EarthParticleContext<'a> {
    pub input: &'a EarthInput,
    pub low_level: EarthLowLevelState,
    pub time_s: f64,
    /// Single coherent weather snapshot evaluated from [`crate::weather_api::BaseWeatherConfig`], if enabled.
    pub weather_snapshot: Option<WeatherSnapshot>,
    pub(crate) ground_snapshot: Option<&'a RwLock<GroundElevSnapshot>>,
}

impl<'a> EarthParticleContext<'a> {
    #[inline]
    pub fn sample_ground_elevation_m(&self, lat_deg: f64, lon_deg: f64) -> f64 {
        self.ground_snapshot
            .and_then(|lock| lock.read().ok().map(|s| s.sample_elevation_m(lat_deg, lon_deg)))
            .unwrap_or(0.0)
    }

    #[inline]
    pub fn sample_land_mask(&self, lat_deg: f64, lon_deg: f64) -> f32 {
        sample_land_mask_at(lat_deg, lon_deg)
    }
}

/// Fast O(1) GPU-native ocean wave spectrum emission context (zero CPU ocean polygon meshing).
pub struct EarthOceanContext<'a> {
    pub input: &'a EarthInput,
    pub low_level: EarthLowLevelState,
    pub time_s: f64,
    pub overlay_rev: u64,
    /// Single coherent weather snapshot evaluated from [`crate::weather_api::BaseWeatherConfig`], if enabled.
    pub weather_snapshot: Option<WeatherSnapshot>,
    pub(crate) ground_snapshot: Option<&'a RwLock<GroundElevSnapshot>>,
}

impl<'a> EarthOceanContext<'a> {
    #[inline]
    pub fn sample_ground_elevation_m(&self, lat_deg: f64, lon_deg: f64) -> f64 {
        self.ground_snapshot
            .and_then(|lock| lock.read().ok().map(|s| s.sample_elevation_m(lat_deg, lon_deg)))
            .unwrap_or(0.0)
    }

    #[inline]
    pub fn sample_land_mask(&self, lat_deg: f64, lon_deg: f64) -> f32 {
        sample_land_mask_at(lat_deg, lon_deg)
    }
}

/// Extensible plugin trait for `strata_earth` world rendering and mid-work injection.
pub trait EarthPlugin: Send + 'static {
    /// Stable unique identifier (e.g. `"city_lights"`, `"precipitation"`, `"ocean_waves"`).
    fn id(&self) -> &'static str;

    /// Declares which pipeline stages this plugin participates in (unused stages incur zero cost).
    fn hooks(&self) -> EarthPluginHooks {
        EarthPluginHooks::empty()
    }

    /// Runtime enable toggle. Disabled plugins are skipped during pipeline execution.
    fn is_enabled(&self) -> bool {
        true
    }

    /// Updates the plugin's runtime enabled state.
    fn set_enabled(&mut self, _enabled: bool) {}

    /// Returns `true` while the plugin needs continuous per-frame animation ticks.
    fn wants_animation_tick(&self, _ctx: &EarthPluginFrameContext<'_>) -> bool {
        false
    }

    /// Advances animated state by `dt_s` seconds; returning `true` schedules redraws.
    fn on_tick(&mut self, _dt_s: f64, _ctx: &EarthPluginFrameContext<'_>) -> bool {
        false
    }

    /// Modifies camera, sun/moon lighting, sky, or shader intensities before rendering.
    fn on_prepare_camera(
        &mut self,
        _ctx: &EarthPluginFrameContext<'_>,
        _camera: &mut RetainedCameraUniforms,
    ) {
    }

    /// Inspects or replaces planetary/space GPU environment assets (`land_mask`, clouds, stars, skybox).
    fn on_prepare_environment(
        &mut self,
        _ctx: &EarthPluginFrameContext<'_>,
        _env: &mut RetainedEnvironmentAssets,
    ) {
    }

    /// Mid-work per-tile interception prior to resolution/meshing.
    fn on_tile_begin(&mut self, _ctx: &mut EarthTileBuildContext<'_>) {}

    /// Mid-work per-tile interception right after `RetainedTileInstance` construction.
    fn on_tile_instance(
        &mut self,
        _ctx: &EarthTileBuildContext<'_>,
        _instance: &mut RetainedTileInstance,
        _textures: &mut EarthTextureInjector<'_>,
    ) {
    }

    /// Mid-work world-assembly hook after terrain leaves are built and before freezing the frame.
    fn on_world_build(&mut self, _ctx: &mut EarthWorldBuildContext<'_>) {}

    /// Fast O(1) dynamic mesh emission hook invoked during world builds and animations.
    fn on_dynamic_overlays(
        &mut self,
        _ctx: &mut EarthDynamicOverlayContext<'_>,
        _dynamic_meshes: &mut Vec<RetainedDynamicMesh>,
    ) {
    }

    /// Fast O(1) GPU-instanced 2D camera-local particle volume emission hook (zero CPU meshing).
    fn on_particle_layers(
        &mut self,
        _ctx: &mut EarthParticleContext<'_>,
        _layers: &mut Vec<RetainedParticleLayer>,
        _textures: &mut Vec<Arc<RetainedTileTexture>>,
    ) {
    }

    /// Fast O(1) GPU-native ocean wave spectrum emission hook (zero CPU ocean polygon meshing).
    fn on_ocean_surface(&mut self, _ctx: &mut EarthOceanContext<'_>) -> Option<RetainedOceanSurface> {
        None
    }

    /// Optional read-only downcast handle for host inspection.
    fn as_any(&self) -> Option<&dyn Any> {
        None
    }

    /// Optional mutable downcast handle for runtime plugin configuration.
    fn as_any_mut(&mut self) -> Option<&mut dyn Any> {
        None
    }
}

pub(crate) struct SharedPluginSlot {
    pub(crate) id: &'static str,
    pub(crate) hooks: AtomicU32,
    pub(crate) enabled: AtomicBool,
    pub(crate) inner: Mutex<Box<dyn EarthPlugin>>,
}

impl SharedPluginSlot {
    fn new(plugin: Box<dyn EarthPlugin>) -> Self {
        let id = plugin.id();
        let hooks = AtomicU32::new(plugin.hooks().bits());
        let enabled = AtomicBool::new(plugin.is_enabled());
        Self {
            id,
            hooks,
            enabled,
            inner: Mutex::new(plugin),
        }
    }
}

struct RegistryInner {
    slots: Vec<Arc<SharedPluginSlot>>,
    hooks_union: u32,
}

impl RegistryInner {
    fn recompute_union(&mut self) {
        let mut u = 0u32;
        for s in &self.slots {
            if s.enabled.load(Ordering::Relaxed) {
                u |= s.hooks.load(Ordering::Relaxed);
            }
        }
        self.hooks_union = u;
    }
}

/// Shareable, thread-safe ordered registry of [`EarthPlugin`] instances.
#[derive(Clone)]
pub struct EarthPluginRegistry {
    inner: Arc<RwLock<RegistryInner>>,
    active_hooks_union: Arc<AtomicU32>,
    revision: Arc<AtomicU64>,
}

impl Default for EarthPluginRegistry {
    fn default() -> Self {
        Self::standard()
    }
}

impl EarthPluginRegistry {
    /// Creates an empty plugin registry with no built-in plugins attached.
    pub fn empty() -> Self {
        Self {
            inner: Arc::new(RwLock::new(RegistryInner {
                slots: Vec::new(),
                hooks_union: 0,
            })),
            active_hooks_union: Arc::new(AtomicU32::new(0)),
            revision: Arc::new(AtomicU64::new(1)),
        }
    }

    /// Creates the default plugin stack matching built-in Earth rendering features
    /// (`CelestialPlugin`, `CityLightsPlugin`, `WeatherPlugin`, `PrecipitationPlugin`,
    /// `SatellitesPlugin`, `OceanWavesPlugin`) with zero visual or performance degradation.
    pub fn standard() -> Self {
        let reg = Self::empty();
        reg.register(CelestialPlugin::default());
        reg.register(CityLightsPlugin::default());
        reg.register(WeatherPlugin::default());
        reg.register(PrecipitationPlugin::default());
        reg.register(SatellitesPlugin::default());
        reg.register(OceanWavesPlugin::default());
        reg
    }

    /// Monotonically incremented whenever plugins are registered, removed, or toggled.
    #[inline]
    pub fn revision(&self) -> u64 {
        self.revision.load(Ordering::Acquire)
    }

    #[inline]
    pub fn active_hooks(&self) -> EarthPluginHooks {
        EarthPluginHooks::from_bits_truncate(self.active_hooks_union.load(Ordering::Acquire))
    }

    fn sync_header(&self, guard: &mut RegistryInner) {
        guard.recompute_union();
        self.active_hooks_union
            .store(guard.hooks_union, Ordering::Release);
        self.revision.fetch_add(1, Ordering::AcqRel);
    }

    /// Registers `plugin`, replacing any existing plugin with the same `id()` in-place or appending.
    pub fn register<P: EarthPlugin>(&self, plugin: P) {
        self.register_boxed(Box::new(plugin));
    }

    /// Registers a boxed plugin, replacing any existing plugin with the same `id()`.
    pub fn register_boxed(&self, plugin: Box<dyn EarthPlugin>) {
        let slot = Arc::new(SharedPluginSlot::new(plugin));
        if let Ok(mut guard) = self.inner.write() {
            if let Some(pos) = guard.slots.iter().position(|s| s.id == slot.id) {
                guard.slots[pos] = slot;
            } else {
                guard.slots.push(slot);
            }
            self.sync_header(&mut guard);
        }
    }

    /// Inserts `plugin` immediately before `target_id` (or appends if `target_id` is absent).
    pub fn insert_before<P: EarthPlugin>(&self, target_id: &str, plugin: P) {
        let slot = Arc::new(SharedPluginSlot::new(Box::new(plugin)));
        if let Ok(mut guard) = self.inner.write() {
            guard.slots.retain(|s| s.id != slot.id);
            if let Some(pos) = guard.slots.iter().position(|s| s.id == target_id) {
                guard.slots.insert(pos, slot);
            } else {
                guard.slots.push(slot);
            }
            self.sync_header(&mut guard);
        }
    }

    /// Removes the plugin with `id`, returning `true` if one was removed.
    pub fn remove(&self, id: &str) -> bool {
        if let Ok(mut guard) = self.inner.write() {
            let before = guard.slots.len();
            guard.slots.retain(|s| s.id != id);
            let removed = guard.slots.len() != before;
            if removed {
                self.sync_header(&mut guard);
            }
            return removed;
        }
        false
    }

    pub fn contains(&self, id: &str) -> bool {
        self.inner
            .read()
            .map(|g| g.slots.iter().any(|s| s.id == id))
            .unwrap_or(false)
    }

    pub fn is_enabled(&self, id: &str) -> Option<bool> {
        let guard = self.inner.read().ok()?;
        let slot = guard.slots.iter().find(|s| s.id == id)?;
        Some(slot.enabled.load(Ordering::Acquire))
    }

    pub fn is_active(&self, id: &str) -> bool {
        self.is_enabled(id).unwrap_or(false)
    }

    /// Enables or disables a registered plugin by `id`, returning `true` if found.
    pub fn set_enabled(&self, id: &str, enabled: bool) -> bool {
        let slot = match self.inner.read() {
            Ok(guard) => guard.slots.iter().find(|s| s.id == id).cloned(),
            Err(_) => None,
        };
        let Some(slot) = slot else {
            return false;
        };
        if let Ok(mut plugin) = slot.inner.lock() {
            plugin.set_enabled(enabled);
            slot.enabled.store(enabled, Ordering::Release);
        }
        if let Ok(mut guard) = self.inner.write() {
            self.sync_header(&mut guard);
        }
        true
    }

    pub fn list_ids(&self) -> Vec<(&'static str, bool)> {
        self.inner
            .read()
            .map(|g| {
                g.slots
                    .iter()
                    .map(|s| (s.id, s.enabled.load(Ordering::Acquire)))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Borrows a concrete plugin `T` by `id` for inspection.
    pub fn with_plugin<T: 'static, R>(&self, id: &str, f: impl FnOnce(&T) -> R) -> Option<R> {
        let slot = {
            let guard = self.inner.read().ok()?;
            guard.slots.iter().find(|s| s.id == id).cloned()?
        };
        let plugin = slot.inner.lock().ok()?;
        let typed = plugin.as_any()?.downcast_ref::<T>()?;
        Some(f(typed))
    }

    /// Borrows a concrete plugin `T` mutably by `id` and bumps registry revision on update.
    pub fn with_plugin_mut<T: 'static, R>(
        &self,
        id: &str,
        f: impl FnOnce(&mut T) -> R,
    ) -> Option<R> {
        let slot = {
            let guard = self.inner.read().ok()?;
            guard.slots.iter().find(|s| s.id == id).cloned()?
        };
        let out = {
            let mut plugin = slot.inner.lock().ok()?;
            let typed = plugin.as_any_mut()?.downcast_mut::<T>()?;
            let res = f(typed);
            slot.enabled.store(plugin.is_enabled(), Ordering::Release);
            slot.hooks.store(plugin.hooks().bits(), Ordering::Release);
            res
        };
        if let Ok(mut guard) = self.inner.write() {
            self.sync_header(&mut guard);
        }
        Some(out)
    }

    fn collect_matching_slots(&self, hook: EarthPluginHooks) -> Vec<Arc<SharedPluginSlot>> {
        if !self.active_hooks().intersects(hook) {
            return Vec::new();
        }
        let Ok(guard) = self.inner.read() else {
            return Vec::new();
        };
        guard
            .slots
            .iter()
            .filter(|s| {
                s.enabled.load(Ordering::Relaxed)
                    && (s.hooks.load(Ordering::Relaxed) & hook.bits()) != 0
            })
            .cloned()
            .collect()
    }

    pub(crate) fn any_wants_animation_tick(&self, ctx: &EarthPluginFrameContext<'_>) -> bool {
        let slots = self.collect_matching_slots(EarthPluginHooks::TICK);
        for slot in slots {
            if let Ok(plugin) = slot.inner.lock() {
                if plugin.wants_animation_tick(ctx) {
                    return true;
                }
            }
        }
        false
    }

    /// Runs `on_tick` across all tick-enabled plugins, distinguishing CPU dynamic-overlay/camera
    /// ticks from zero-mesh GPU particle and GPU ocean ticks so precipitation and ocean animation
    /// never trigger unnecessary satellite/overlay rebuilds.
    pub(crate) fn run_tick_detailed(
        &self,
        dt_s: f64,
        ctx: &EarthPluginFrameContext<'_>,
    ) -> (bool, bool, bool) {
        let slots = self.collect_matching_slots(EarthPluginHooks::TICK);
        let mut other_dirty = false;
        let mut particle_dirty = false;
        let mut ocean_dirty = false;
        for slot in slots {
            let hooks = EarthPluginHooks::from_bits_truncate(slot.hooks.load(Ordering::Relaxed));
            if let Ok(mut plugin) = slot.inner.lock() {
                if plugin.on_tick(dt_s, ctx) {
                    let touches_other = hooks
                        .intersects(EarthPluginHooks::DYNAMIC_OVERLAYS | EarthPluginHooks::CAMERA);
                    if touches_other {
                        other_dirty = true;
                    } else {
                        let is_p = hooks.intersects(EarthPluginHooks::PARTICLES);
                        let is_o = hooks.intersects(EarthPluginHooks::OCEAN);
                        if is_p {
                            particle_dirty = true;
                        }
                        if is_o {
                            ocean_dirty = true;
                        }
                        if !is_p && !is_o {
                            other_dirty = true;
                        }
                    }
                }
            }
        }
        (other_dirty, particle_dirty, ocean_dirty)
    }

    pub(crate) fn run_prepare_camera(
        &self,
        ctx: &EarthPluginFrameContext<'_>,
        camera: &mut RetainedCameraUniforms,
    ) {
        let slots = self.collect_matching_slots(EarthPluginHooks::CAMERA);
        for slot in slots {
            if let Ok(mut plugin) = slot.inner.lock() {
                plugin.on_prepare_camera(ctx, camera);
            }
        }
    }

    pub(crate) fn run_prepare_environment(
        &self,
        ctx: &EarthPluginFrameContext<'_>,
        env: &mut RetainedEnvironmentAssets,
    ) {
        let slots = self.collect_matching_slots(EarthPluginHooks::ENVIRONMENT);
        for slot in slots {
            if let Ok(mut plugin) = slot.inner.lock() {
                plugin.on_prepare_environment(ctx, env);
            }
        }
    }

    pub(crate) fn has_tile_hooks(&self) -> bool {
        self.active_hooks().intersects(EarthPluginHooks::TILE)
    }

    pub(crate) fn run_tile_begin_batch(&self, contexts: &mut [EarthTileBuildContext<'_>]) {
        let slots = self.collect_matching_slots(EarthPluginHooks::TILE);
        for slot in slots {
            if let Ok(mut plugin) = slot.inner.lock() {
                for ctx in contexts.iter_mut() {
                    plugin.on_tile_begin(ctx);
                }
            }
        }
    }

    pub(crate) fn run_tile_instance_batch(
        &self,
        contexts: &[EarthTileBuildContext<'_>],
        instances: &mut [RetainedTileInstance],
        textures: &mut EarthTextureInjector<'_>,
    ) {
        let slots = self.collect_matching_slots(EarthPluginHooks::TILE);
        for slot in slots {
            if let Ok(mut plugin) = slot.inner.lock() {
                for (ctx, inst) in contexts.iter().zip(instances.iter_mut()) {
                    plugin.on_tile_instance(ctx, inst, textures);
                }
            }
        }
    }

    pub(crate) fn run_world_build(&self, ctx: &mut EarthWorldBuildContext<'_>) {
        let slots = self.collect_matching_slots(EarthPluginHooks::WORLD_BUILD);
        for slot in slots {
            if let Ok(mut plugin) = slot.inner.lock() {
                plugin.on_world_build(ctx);
            }
        }
    }

    pub(crate) fn run_dynamic_overlays(
        &self,
        ctx: &mut EarthDynamicOverlayContext<'_>,
        dynamic_meshes: &mut Vec<RetainedDynamicMesh>,
    ) {
        let slots = self.collect_matching_slots(EarthPluginHooks::DYNAMIC_OVERLAYS);
        for slot in slots {
            if let Ok(mut plugin) = slot.inner.lock() {
                plugin.on_dynamic_overlays(ctx, dynamic_meshes);
            }
        }
    }

    pub(crate) fn run_particle_layers(
        &self,
        ctx: &mut EarthParticleContext<'_>,
        layers: &mut Vec<RetainedParticleLayer>,
        textures: &mut Vec<Arc<RetainedTileTexture>>,
    ) {
        let slots = self.collect_matching_slots(EarthPluginHooks::PARTICLES);
        for slot in slots {
            if let Ok(mut plugin) = slot.inner.lock() {
                plugin.on_particle_layers(ctx, layers, textures);
            }
        }
    }

    pub(crate) fn run_ocean_surface(
        &self,
        ctx: &mut EarthOceanContext<'_>,
    ) -> Option<RetainedOceanSurface> {
        let slots = self.collect_matching_slots(EarthPluginHooks::OCEAN);
        let mut out = None;
        for slot in slots {
            if let Ok(mut plugin) = slot.inner.lock() {
                if let Some(surf) = plugin.on_ocean_surface(ctx) {
                    out = Some(surf);
                }
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Built-in World Plugins (zero visual or runtime degradation)
// ---------------------------------------------------------------------------

fn resolve_weather_optics(ctx: &EarthPluginFrameContext<'_>) -> (f32, f32, f32) {
    let mut darkness = 0.0f32;
    let mut cloud_cover = ctx.input.cloud_cover.clamp(0.0, 1.0);
    let mut haze = 0.0f32;

    if let Some(ws) = ctx.weather_snapshot {
        darkness = darkness.max(ws.storm_darkness.clamp(0.0, 1.0));
        cloud_cover = cloud_cover.max(ws.cloud_cover.clamp(0.0, 1.0));
        haze = haze.max(ws.normalized_haze());
    }

    let precip = &ctx.input.precipitation;
    if precip.enabled && precip.intensity > 0.01 {
        let ceiling_m = precip
            .atmosphere
            .cloud_ceiling
            .resolve_msl_m(ctx.ground_elevation_m);
        let fade_band = precip.atmosphere.cloud_fade_band_m.max(150.0);
        let below_ceil = if precip.respect_cloud_ceiling {
            1.0 - smoothstep_f32(
                (ceiling_m - fade_band) as f32,
                ceiling_m as f32,
                ctx.altitude_m as f32,
            )
        } else {
            1.0
        };
        if below_ceil > 0.002 {
            let p_t = (precip.intensity / 1.15).clamp(0.0, 1.0);
            darkness = darkness.max((0.26 + 0.56 * p_t) * below_ceil);
            cloud_cover = cloud_cover.max((0.48 + 0.46 * p_t) * below_ceil);
            haze = haze.max((0.20 + 0.48 * p_t) * below_ceil);
        }
    }

    (
        darkness.clamp(0.0, 1.0),
        cloud_cover.clamp(0.0, 1.0),
        haze.clamp(0.0, 1.0),
    )
}

/// Built-in plugin controlling physical solar/lunar ephemeris, terminator shadows,
/// deep-space starfield, telescope skybox, and opt-in realistic 2.5D atmosphere/shadows.
#[derive(Debug, Clone)]
pub struct CelestialPlugin {
    pub enabled: bool,
    pub sun_intensity: f32,
}

impl Default for CelestialPlugin {
    fn default() -> Self {
        Self {
            enabled: true,
            sun_intensity: 1.0,
        }
    }
}

impl CelestialPlugin {
    pub const ID: &'static str = "celestial";

    pub fn new(enabled: bool) -> Self {
        Self {
            enabled,
            sun_intensity: 1.0,
        }
    }
}

impl EarthPlugin for CelestialPlugin {
    fn id(&self) -> &'static str {
        Self::ID
    }

    fn hooks(&self) -> EarthPluginHooks {
        EarthPluginHooks::CAMERA
    }

    fn is_enabled(&self) -> bool {
        self.enabled
    }

    fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    fn on_prepare_camera(
        &mut self,
        ctx: &EarthPluginFrameContext<'_>,
        camera: &mut RetainedCameraUniforms,
    ) {
        let sp = &ctx.input.space;
        let la = &ctx.input.local_atmosphere;
        let ll = ctx.low_level;
        let flat_t = ll.local_25d_blend.max(ll.map_2d_morph).clamp(0.0, 1.0);
        let in_2d_hide = ll.map_2d_morph > 0.05;
        let space_factor = 1.0 - smoothstep_f32(0.0, 0.85, flat_t);
        let (sun_dir, moon_dir, unix_utc_s) = ctx.input.resolved_celestial_state();
        camera.sun_dir = sun_dir;
        camera.moon_dir = if let Some(cm) = sp.custom_moon_dir {
            let len = (cm[0] * cm[0] + cm[1] * cm[1] + cm[2] * cm[2])
                .sqrt()
                .max(1e-6);
            [cm[0] / len, cm[1] / len, cm[2] / len]
        } else {
            moon_dir
        };
        camera.ecef_from_icrf = crate::astro::icrf_to_ecef_matrix(unix_utc_s);
        let si = self.sun_intensity.max(0.0);
        camera.sun_color = [si, 0.97 * si, 0.92 * si];
        camera.day_night_strength = if sp.day_night_lighting && !in_2d_hide {
            sp.shadow_contrast.clamp(0.0, 1.0) * space_factor
        } else {
            0.0
        };
        camera.sun_visible = ctx.input.layers.celestial && sp.sun_visible && !in_2d_hide;
        camera.sun_angular_radius_rad = sp.sun_angular_radius_rad.max(1e-4);
        camera.moon_visible = ctx.input.layers.celestial && sp.moon_visible && !in_2d_hide;
        camera.moon_angular_radius_rad = sp.moon_angular_radius_rad.max(1e-4);
        camera.stars_visible = sp.stars_visible && !in_2d_hide;
        camera.stars_intensity = if in_2d_hide {
            0.0
        } else {
            sp.stars_intensity.max(0.0)
        };
        camera.skybox_intensity = if in_2d_hide {
            0.0
        } else {
            sp.skybox_intensity.max(0.0)
        };

        // Optional realistic 2.5D atmosphere, night stars, directional terrain lighting & mountain shadows.
        camera.realistic_25d_sky = if la.realistic_sky && !in_2d_hide {
            1.0
        } else {
            0.0
        };
        camera.night_stars_25d = if la.night_stars_sync && sp.stars_visible && !in_2d_hide {
            1.0
        } else {
            0.0
        };
        camera.terrain_25d_lighting = if la.terrain_lighting && !in_2d_hide {
            1.0
        } else {
            0.0
        };
        camera.terrain_25d_shadows =
            if la.terrain_shadows && la.terrain_lighting && !in_2d_hide {
                la.shadow_strength.clamp(0.0, 1.0)
            } else {
                0.0
            };
        camera.shadow_map_size = la.shadow_map_size.clamp(512, 4096);

        if la.sync_with_weather && !in_2d_hide {
            let (dark, cov, haze) = resolve_weather_optics(ctx);
            camera.weather_darkness = dark;
            camera.weather_cloud_cover = cov;
            camera.weather_haze = haze;
        } else {
            camera.weather_darkness = 0.0;
            camera.weather_cloud_cover = 0.0;
            camera.weather_haze = 0.0;
        }
    }

    fn as_any(&self) -> Option<&dyn Any> {
        Some(self)
    }

    fn as_any_mut(&mut self) -> Option<&mut dyn Any> {
        Some(self)
    }
}

/// Built-in plugin controlling Black Marble night-side city lights & night tile streaming.
///
/// Leaves GPU fragment shading inside `strata_render` untouched so there is zero visual or FPS loss,
/// while letting hosts configure, disable, or unplug city lights and avoid night-tile fetches completely.
#[derive(Debug, Clone)]
pub struct CityLightsPlugin {
    pub enabled: bool,
    pub intensity_scale: f32,
}

impl Default for CityLightsPlugin {
    fn default() -> Self {
        Self {
            enabled: true,
            intensity_scale: 1.0,
        }
    }
}

impl CityLightsPlugin {
    pub const ID: &'static str = "city_lights";

    pub fn new(enabled: bool, intensity_scale: f32) -> Self {
        Self {
            enabled,
            intensity_scale: intensity_scale.max(0.0),
        }
    }

    fn effective_intensity(&self, ctx: &EarthPluginFrameContext<'_>) -> f32 {
        let sp = &ctx.input.space;
        let la = &ctx.input.local_atmosphere;
        let flat_t = ctx
            .low_level
            .local_25d_blend
            .max(ctx.low_level.map_2d_morph)
            .clamp(0.0, 1.0);
        let in_2d_hide = ctx.low_level.map_2d_morph > 0.05;
        if !self.enabled || !sp.day_night_lighting || !sp.city_lights || in_2d_hide {
            return 0.0;
        }
        let space_factor = 1.0 - smoothstep_f32(0.0, 0.85, flat_t);
        let allow_25d_night = if la.night_stars_sync && la.terrain_lighting {
            1.0
        } else {
            0.0
        };
        sp.city_lights_intensity.max(0.0)
            * self.intensity_scale.max(0.0)
            * space_factor.max(allow_25d_night)
    }
}

impl EarthPlugin for CityLightsPlugin {
    fn id(&self) -> &'static str {
        Self::ID
    }

    fn hooks(&self) -> EarthPluginHooks {
        EarthPluginHooks::CAMERA | EarthPluginHooks::TILE
    }

    fn is_enabled(&self) -> bool {
        self.enabled
    }

    fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    fn on_prepare_camera(
        &mut self,
        ctx: &EarthPluginFrameContext<'_>,
        camera: &mut RetainedCameraUniforms,
    ) {
        camera.city_lights_intensity = self.effective_intensity(ctx);
    }

    fn on_tile_begin(&mut self, ctx: &mut EarthTileBuildContext<'_>) {
        let ll = ctx.input.low_level_state();
        let la = &ctx.input.local_atmosphere;
        let allow_25d_night = la.night_stars_sync && la.terrain_lighting;
        let active = self.enabled
            && ctx.input.space.day_night_lighting
            && ctx.input.space.city_lights
            && ctx.input.layers.terrain
            && ctx.input.view_mode != crate::scene::EarthViewMode::Mode2D
            && ll.map_2d_morph <= 0.05
            && (ll.local_25d_blend < 0.70 || allow_25d_night);
        ctx.want_night_tiles = active;
    }

    fn as_any(&self) -> Option<&dyn Any> {
        Some(self)
    }

    fn as_any_mut(&mut self) -> Option<&mut dyn Any> {
        Some(self)
    }
}

/// Built-in plugin driving global and local 2.5D cloud-layer rotation, opacity, and
/// unified weather storm darkening/tinting uniforms.
///
/// Camera-local 2D precipitation (rain, sleet, mathematical 2D snowflakes, and custom textures)
/// is handled separately by [`crate::precipitation::PrecipitationPlugin`].
#[derive(Debug, Clone)]
pub struct WeatherPlugin {
    pub enabled: bool,
    pub clock_offset_s: f64,
}

impl Default for WeatherPlugin {
    fn default() -> Self {
        Self {
            enabled: true,
            clock_offset_s: 0.0,
        }
    }
}

impl WeatherPlugin {
    pub const ID: &'static str = "weather";

    fn is_animating(&self, ctx: &EarthPluginFrameContext<'_>) -> bool {
        if !self.enabled {
            return false;
        }
        let flat_t = ctx.low_level.local_25d_blend.max(ctx.low_level.map_2d_morph);
        let w = &ctx.input.weather;
        w.global_cloud_map
            && ctx.input.layers.clouds
            && flat_t < 0.85
            && w.global_cloud_spin_rad_s.abs() > 1e-9
    }
}

impl EarthPlugin for WeatherPlugin {
    fn id(&self) -> &'static str {
        Self::ID
    }

    fn hooks(&self) -> EarthPluginHooks {
        EarthPluginHooks::TICK | EarthPluginHooks::CAMERA
    }

    fn is_enabled(&self) -> bool {
        self.enabled
    }

    fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    fn wants_animation_tick(&self, ctx: &EarthPluginFrameContext<'_>) -> bool {
        self.is_animating(ctx)
    }

    fn on_tick(&mut self, dt_s: f64, ctx: &EarthPluginFrameContext<'_>) -> bool {
        if !self.is_animating(ctx) {
            return false;
        }
        self.clock_offset_s += dt_s.max(0.0);
        true
    }

    fn on_prepare_camera(
        &mut self,
        ctx: &EarthPluginFrameContext<'_>,
        camera: &mut RetainedCameraUniforms,
    ) {
        let in_2d_hide = ctx.input.view_mode == crate::scene::EarthViewMode::Mode2D
            || ctx.low_level.map_2d_morph > 0.05;
        let space_factor = (1.0 - ctx.low_level.local_25d_blend).clamp(0.0, 1.0);
        let cloud_alt_fade = smoothstep_f32(450_000.0, 2_000_000.0, ctx.altitude_m as f32);
        let w = &ctx.input.weather;
        let cloud_on = self.enabled && w.global_cloud_map && ctx.input.layers.clouds && !in_2d_hide;
        let base_cloud_op = if cloud_on {
            w.global_cloud_opacity.clamp(0.0, 1.0)
        } else {
            0.0
        };

        let (cloud_dark, cover_boost) = if in_2d_hide || !cloud_on {
            (0.0, 0.0)
        } else if let Some(ovr) = w.cloud_darkness_override {
            let d = ovr.clamp(0.0, 1.0);
            (d, (d * 0.85).clamp(0.0, 1.0))
        } else if w.sync_with_base_weather {
            let (opt_dark, opt_cov, _) = resolve_weather_optics(ctx);
            let boost = ((opt_cov - 0.25) / 0.75).clamp(0.0, 1.0);
            (opt_dark, boost)
        } else {
            (0.0, 0.0)
        };

        let eff_t = w.weather_time_s + self.clock_offset_s;
        let local_op = if cloud_on {
            (base_cloud_op + (0.94 - base_cloud_op).max(0.0) * cover_boost * 0.65).clamp(0.0, 1.0)
        } else {
            0.0
        };
        camera.cloud_layer_opacity = base_cloud_op * space_factor * cloud_alt_fade;
        camera.local_cloud_opacity = local_op * (1.0 - cloud_alt_fade);
        camera.cloud_lon_offset_rad = (w.global_cloud_spin_rad_s * eff_t) as f32;
        camera.time_s = eff_t as f32;

        // Tint clouds smoothly from sunlit white [1.0, 1.0, 1.0] to moody storm slate [0.42, 0.46, 0.54].
        camera.cloud_tint_rgb = [
            1.0 - 0.58 * cloud_dark,
            1.0 - 0.54 * cloud_dark,
            1.0 - 0.46 * cloud_dark,
        ];
        camera.cloud_25d_cover_boost = cover_boost;
        if w.cloud_darkness_override.is_some() && !in_2d_hide {
            camera.weather_darkness = camera.weather_darkness.max(cloud_dark);
        }
    }

    fn as_any(&self) -> Option<&dyn Any> {
        Some(self)
    }

    fn as_any_mut(&mut self) -> Option<&mut dyn Any> {
        Some(self)
    }
}

/// Built-in plugin rendering orbital spacecraft beacons & ground-track ribbons.
#[derive(Debug, Clone)]
pub struct SatellitesPlugin {
    pub enabled: bool,
}

impl Default for SatellitesPlugin {
    fn default() -> Self {
        Self { enabled: true }
    }
}

impl EarthPlugin for SatellitesPlugin {
    fn id(&self) -> &'static str {
        "satellites"
    }

    fn hooks(&self) -> EarthPluginHooks {
        EarthPluginHooks::TICK | EarthPluginHooks::DYNAMIC_OVERLAYS
    }

    fn is_enabled(&self) -> bool {
        self.enabled
    }

    fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    fn wants_animation_tick(&self, ctx: &EarthPluginFrameContext<'_>) -> bool {
        let flat_t = ctx.low_level.local_25d_blend.max(ctx.low_level.map_2d_morph);
        self.enabled && !ctx.is_dragging && ctx.input.space.satellites_visible && flat_t < 0.85
    }

    fn on_tick(&mut self, _dt_s: f64, ctx: &EarthPluginFrameContext<'_>) -> bool {
        self.wants_animation_tick(ctx)
    }

    fn on_dynamic_overlays(
        &mut self,
        ctx: &mut EarthDynamicOverlayContext<'_>,
        dynamic_meshes: &mut Vec<RetainedDynamicMesh>,
    ) {
        if !self.enabled || !ctx.input.space.satellites_visible || ctx.flat_t >= 0.85 {
            return;
        }
        let (sun_dir, _, unix_utc_s) = ctx.input.resolved_celestial_state();
        let sats = build_3d_satellites_meshes(
            unix_utc_s,
            sun_dir,
            ctx.input.camera.eye_distance_m(),
            ctx.flat_t,
            ctx.input.space.satellites_visible,
            ctx.input.space.satellite_orbits_visible,
            ctx.input.space.satellite_scale_m,
            ctx.satellite_elements,
            ctx.overlay_rev,
        );
        dynamic_meshes.extend(sats);
    }

    fn as_any(&self) -> Option<&dyn Any> {
        Some(self)
    }

    fn as_any_mut(&mut self) -> Option<&mut dyn Any> {
        Some(self)
    }
}
