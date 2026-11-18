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
use std::f64::consts::PI;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use strata_render::camera64::DVec3;
use strata_render::scene3d::{
    RetainedCameraUniforms, RetainedDynamicMesh, RetainedEnvironmentAssets, RetainedTileInstance,
    RetainedTileMesh, RetainedTileTexture, RteTileVertex, TileKeyId,
};

use crate::astro::embedded_land_mask;
use crate::geodesy::{
    geodetic_to_ecef, wgs84, wrap_longitude, Geodetic, LocalFrame, PatchBounds, TileKey,
    TileProjection,
};
use crate::gltf::{build_25d_weather_meshes, CustomWeatherMeshes};
use crate::satellites::{build_3d_satellites_meshes, SatelliteElement};
use crate::scene::{EarthInput, EarthLowLevelState, OceanWavesConfig};
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
    pub const ALL: Self = Self((1 << 6) - 1);

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
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[inline]
fn smoothstep_f64(edge0: f64, edge1: f64, x: f64) -> f64 {
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
    pub time_s: f64,
    pub overlay_rev: u64,
    pub is_dragging: bool,
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
    pub custom_weather: Option<&'a CustomWeatherMeshes>,
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

/// Extensible plugin trait for `strata_earth` world rendering and mid-work injection.
pub trait EarthPlugin: Send + 'static {
    /// Stable unique identifier (e.g. `"city_lights"`, `"ocean_waves"`, `"my_custom_layer"`).
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

    /// Advances animated state by `dt_s` seconds; returning `true` schedules dynamic overlay redraws.
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

    /// Fast O(1) dynamic mesh emission hook invoked both during world builds and 60 FPS animations.
    fn on_dynamic_overlays(
        &mut self,
        _ctx: &mut EarthDynamicOverlayContext<'_>,
        _dynamic_meshes: &mut Vec<RetainedDynamicMesh>,
    ) {
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
    /// (`CelestialPlugin`, `CityLightsPlugin`, `WeatherPlugin`, `SatellitesPlugin`, `OceanWavesPlugin`)
    /// with zero visual or performance degradation.
    pub fn standard() -> Self {
        let reg = Self::empty();
        reg.register(CelestialPlugin::default());
        reg.register(CityLightsPlugin::default());
        reg.register(WeatherPlugin::default());
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

    pub(crate) fn run_tick(&self, dt_s: f64, ctx: &EarthPluginFrameContext<'_>) -> bool {
        let slots = self.collect_matching_slots(EarthPluginHooks::TICK);
        let mut any_dirty = false;
        for slot in slots {
            if let Ok(mut plugin) = slot.inner.lock() {
                if plugin.on_tick(dt_s, ctx) {
                    any_dirty = true;
                }
            }
        }
        any_dirty
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
}

// ---------------------------------------------------------------------------
// Built-in World Plugins (zero visual or runtime degradation)
// ---------------------------------------------------------------------------

/// Built-in plugin controlling physical solar/lunar ephemeris, terminator shadows,
/// deep-space starfield, and telescope skybox intensities.
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
        let flat_t = ctx.low_level.local_25d_blend.max(ctx.low_level.map_2d_morph).clamp(0.0, 1.0);
        let in_2d_hide = ctx.low_level.map_2d_morph > 0.05;
        if !self.enabled || !sp.day_night_lighting || !sp.city_lights || in_2d_hide {
            return 0.0;
        }
        let space_factor = 1.0 - smoothstep_f32(0.0, 0.85, flat_t);
        sp.city_lights_intensity.max(0.0) * self.intensity_scale.max(0.0) * space_factor
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
        let active = self.enabled
            && ctx.input.space.day_night_lighting
            && ctx.input.space.city_lights
            && ctx.input.layers.terrain
            && ctx.input.view_mode != crate::scene::EarthViewMode::Mode2D
            && ll.map_2d_morph <= 0.05
            && ll.local_25d_blend < 0.70;
        ctx.want_night_tiles = active;
    }

    fn as_any(&self) -> Option<&dyn Any> {
        Some(self)
    }

    fn as_any_mut(&mut self) -> Option<&mut dyn Any> {
        Some(self)
    }
}

/// Built-in plugin driving global cloud rotation/opacity uniforms and 2.5D weather overlays
/// (CC0 clouds, rain streaks, snowflakes).
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
        let spin_3d = w.global_cloud_map
            && ctx.input.layers.clouds
            && flat_t < 0.85
            && w.global_cloud_spin_rad_s.abs() > 1e-9;
        let weather_25d = flat_t > 0.02
            && ((w.clouds_25d && w.cloud_count > 0)
                || (w.rain && w.rain_density > 0)
                || (w.snow && w.snow_density > 0));
        spin_3d || weather_25d
    }
}

impl EarthPlugin for WeatherPlugin {
    fn id(&self) -> &'static str {
        Self::ID
    }

    fn hooks(&self) -> EarthPluginHooks {
        EarthPluginHooks::TICK | EarthPluginHooks::CAMERA | EarthPluginHooks::DYNAMIC_OVERLAYS
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
        let cloud_on =
            self.enabled && ctx.input.weather.global_cloud_map && ctx.input.layers.clouds && !in_2d_hide;
        let base_cloud_op = if cloud_on {
            ctx.input.weather.global_cloud_opacity.clamp(0.0, 1.0)
        } else {
            0.0
        };
        let eff_t = ctx.input.weather.weather_time_s + self.clock_offset_s;
        camera.cloud_layer_opacity = base_cloud_op * space_factor * cloud_alt_fade;
        camera.local_cloud_opacity = base_cloud_op * (1.0 - cloud_alt_fade);
        camera.cloud_lon_offset_rad =
            (ctx.input.weather.global_cloud_spin_rad_s * eff_t) as f32;
        camera.time_s = eff_t as f32;
    }

    fn on_dynamic_overlays(
        &mut self,
        ctx: &mut EarthDynamicOverlayContext<'_>,
        dynamic_meshes: &mut Vec<RetainedDynamicMesh>,
    ) {
        if !self.enabled {
            return;
        }
        let mut cfg = ctx.input.weather.clone();
        cfg.weather_time_s += self.clock_offset_s;
        let meshes = build_25d_weather_meshes(
            ctx.input.camera.focus,
            ctx.altitude_m_or_cam(),
            ctx.flat_t,
            &cfg,
            ctx.custom_weather,
            ctx.overlay_rev,
            ctx.weather_ids,
        );
        dynamic_meshes.extend(meshes);
    }

    fn as_any(&self) -> Option<&dyn Any> {
        Some(self)
    }

    fn as_any_mut(&mut self) -> Option<&mut dyn Any> {
        Some(self)
    }
}

impl<'a> EarthDynamicOverlayContext<'a> {
    #[inline]
    fn altitude_m_or_cam(&self) -> f64 {
        self.input.camera.altitude_m()
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

// ---------------------------------------------------------------------------
// OceanWavesPlugin: Pure strata_earth Non-Land Wavy Ocean Surface Renderer
// ---------------------------------------------------------------------------

const OCEAN_BASE_MESH_ID: u64 = 0xE47A_0C01;
const OCEAN_GLINT_MESH_ID: u64 = 0xE47A_0C02;
const OCEAN_TROUGH_MESH_ID: u64 = 0xE47A_0C03;

#[derive(Clone, Copy)]
struct GerstnerOctave {
    dir_x: f64,
    dir_y: f64,
    wavelength_factor: f64,
    amp_weight: f64,
    steepness: f64,
    phase_offset: f64,
}

const OCEAN_OCTAVES: [GerstnerOctave; 4] = [
    GerstnerOctave {
        dir_x: 0.857,
        dir_y: 0.515,
        wavelength_factor: 1.00,
        amp_weight: 0.44,
        steepness: 0.52,
        phase_offset: 0.0,
    },
    GerstnerOctave {
        dir_x: -0.447,
        dir_y: 0.894,
        wavelength_factor: 0.62,
        amp_weight: 0.27,
        steepness: 0.46,
        phase_offset: 1.7,
    },
    GerstnerOctave {
        dir_x: 0.968,
        dir_y: -0.251,
        wavelength_factor: 0.37,
        amp_weight: 0.17,
        steepness: 0.40,
        phase_offset: 3.1,
    },
    GerstnerOctave {
        dir_x: -0.742,
        dir_y: -0.670,
        wavelength_factor: 0.21,
        amp_weight: 0.12,
        steepness: 0.32,
        phase_offset: 4.8,
    },
];

/// Determines non-land water weight in `[0.0 .. 1.0]` combining global landmass mask,
/// loaded terrain elevation, and coastal tile pixel classification.
#[inline]
pub fn classify_non_land_water(
    lat_deg: f64,
    lon_deg: f64,
    elev_m: f64,
    imagery: Option<&LoadedTileImagerySnapshot>,
) -> f32 {
    let land_r = sample_land_mask_at(lat_deg, lon_deg);
    if land_r >= 0.95 && elev_m > 1.2 {
        return 0.0;
    }
    if elev_m > 2.2 {
        return 0.0;
    }
    let elev_fade = 1.0 - smoothstep_f32(0.25, 1.9, elev_m as f32);
    if elev_fade <= 0.01 {
        return 0.0;
    }

    // Open deep ocean far from continents: trust the landmass mask so clouds baked into
    // offshore satellite photos don't punch dry holes in ocean waves.
    if land_r <= 0.07 {
        return elev_fade;
    }

    // Coastal transition & inland harbors/bays: refine sharply against loaded tile imagery.
    if let Some(snap) = imagery {
        if let Some([r, g, b, _a]) = snap.sample_rgba(lat_deg, lon_deg) {
            let max_rgb = r.max(g).max(b);
            let min_rgb = r.min(g).min(b);
            // Typical satellite marine water (dark blue/teal/navy) & map-style water fill (#aad3df, etc.)
            let cool_water = b >= r && b + 4 >= g && (b as i32 - r as i32 >= 3 || max_rgb < 110);
            let lagoon_water = g >= r + 6 && b >= r + 4 && r < 145;
            let sandy_or_concrete = (r as i32 - b as i32 >= 10) || min_rgb > 168;
            let green_land = (g as i32 > b as i32 + 9) && (g as i32 > r as i32 + 5) && max_rgb > 42;
            let px_water = if sandy_or_concrete || green_land {
                0.0
            } else if cool_water || lagoon_water {
                1.0
            } else {
                0.0
            };
            let mask_water = 1.0 - smoothstep_f32(0.18, 0.72, land_r);
            let blended = if land_r < 0.30 {
                px_water * 0.72 + mask_water * 0.28
            } else {
                px_water
            };
            return (blended * elev_fade).clamp(0.0, 1.0);
        }
    }

    let mask_water = 1.0 - smoothstep_f32(0.16, 0.62, land_r);
    (mask_water * elev_fade).clamp(0.0, 1.0)
}

#[inline]
fn morph_ocean_point(
    lat_deg: f64,
    lon_deg: f64,
    height_m: f64,
    flat_t: f64,
    map_2d: f64,
    focus: Geodetic,
    focus_ecef: DVec3,
    focus_frame: &LocalFrame,
    cos_scale: f64,
) -> (DVec3, DVec3) {
    let p_curved = geodetic_to_ecef(Geodetic::new(lat_deg, lon_deg, height_m));
    let up_curved = p_curved.normalize();
    if flat_t <= 1e-6 {
        return (p_curved, up_curved);
    }
    let dlon_deg = wrap_longitude(lon_deg - focus.lon_deg);
    let dlat_deg = lat_deg - focus.lat_deg;
    let east_m = dlon_deg.to_radians() * wgs84::A * cos_scale;
    let north_m = dlat_deg.to_radians() * wgs84::A;
    let up_h = height_m * (1.0 - map_2d * 0.85);
    let p_flat = focus_ecef
        .add(focus_frame.east.scale(east_m))
        .add(focus_frame.north.scale(north_m))
        .add(focus_frame.up.scale(up_h));
    let p = DVec3::new(
        p_curved.x + (p_flat.x - p_curved.x) * flat_t,
        p_curved.y + (p_flat.y - p_curved.y) * flat_t,
        p_curved.z + (p_flat.z - p_curved.z) * flat_t,
    );
    let up = DVec3::new(
        up_curved.x + (focus_frame.up.x - up_curved.x) * flat_t,
        up_curved.y + (focus_frame.up.y - up_curved.y) * flat_t,
        up_curved.z + (focus_frame.up.z - up_curved.z) * flat_t,
    )
    .normalize();
    (p, up)
}

/// Built-in plugin synthesizing animated 3D multi-octave Gerstner waves & specular sun-glint
/// crests exclusively over non-land water surfaces in `strata_earth`.
pub struct OceanWavesPlugin {
    pub enabled: bool,
    pub config_override: Option<OceanWavesConfig>,
    phase_time_s: f64,
    imagery_snapshot: Option<Arc<LoadedTileImagerySnapshot>>,
    scratch_sample_lat: Vec<f64>,
    scratch_sample_lon: Vec<f64>,
    scratch_sample_water: Vec<f32>,
    scratch_sample_r_norm: Vec<f32>,
    scratch_vert_map: Vec<i32>,
    scratch_base_verts: Vec<RteTileVertex>,
    scratch_base_idx: Vec<u32>,
    scratch_trough_verts: Vec<RteTileVertex>,
    scratch_trough_idx: Vec<u32>,
    scratch_glint_verts: Vec<RteTileVertex>,
    scratch_glint_idx: Vec<u32>,
    scratch_glint_score: Vec<f32>,
    scratch_crest_norm: Vec<f32>,
    cached_grid_sig: u64,
    cached_imagery_rev: u64,
    imagery_rev: u64,
}

impl Default for OceanWavesPlugin {
    fn default() -> Self {
        Self {
            enabled: true,
            config_override: None,
            phase_time_s: 0.0,
            imagery_snapshot: None,
            scratch_sample_lat: Vec::new(),
            scratch_sample_lon: Vec::new(),
            scratch_sample_water: Vec::new(),
            scratch_sample_r_norm: Vec::new(),
            scratch_vert_map: Vec::new(),
            scratch_base_verts: Vec::new(),
            scratch_base_idx: Vec::new(),
            scratch_trough_verts: Vec::new(),
            scratch_trough_idx: Vec::new(),
            scratch_glint_verts: Vec::new(),
            scratch_glint_idx: Vec::new(),
            scratch_glint_score: Vec::new(),
            scratch_crest_norm: Vec::new(),
            cached_grid_sig: 0,
            cached_imagery_rev: 0,
            imagery_rev: 1,
        }
    }
}

impl OceanWavesPlugin {
    pub const ID: &'static str = "ocean_waves";

    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_config(mut self, config: OceanWavesConfig) -> Self {
        self.config_override = Some(config);
        self
    }

    fn effective_config<'a>(&'a self, input: &'a EarthInput) -> &'a OceanWavesConfig {
        self.config_override.as_ref().unwrap_or(&input.ocean_waves)
    }

    fn altitude_visibility(&self, ctx: &EarthPluginFrameContext<'_>) -> f32 {
        let alt = ctx.altitude_m as f32;
        let low_fade = 1.0 - smoothstep_f32(280_000.0, 520_000.0, alt);
        if ctx.low_level.map_2d_morph > 0.05 {
            1.0 - smoothstep_f32(900_000.0, 2_200_000.0, alt)
        } else {
            low_fade
        }
    }

    fn is_currently_active(&self, ctx: &EarthPluginFrameContext<'_>) -> bool {
        let cfg = self.effective_config(ctx.input);
        self.enabled
            && cfg.enabled
            && ctx.input.layers.terrain
            && self.altitude_visibility(ctx) > 0.01
    }
}

impl EarthPlugin for OceanWavesPlugin {
    fn id(&self) -> &'static str {
        Self::ID
    }

    fn hooks(&self) -> EarthPluginHooks {
        EarthPluginHooks::TICK | EarthPluginHooks::WORLD_BUILD | EarthPluginHooks::DYNAMIC_OVERLAYS
    }

    fn is_enabled(&self) -> bool {
        self.enabled
    }

    fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    fn wants_animation_tick(&self, ctx: &EarthPluginFrameContext<'_>) -> bool {
        self.is_currently_active(ctx)
    }

    fn on_tick(&mut self, dt_s: f64, ctx: &EarthPluginFrameContext<'_>) -> bool {
        if !self.is_currently_active(ctx) {
            return false;
        }
        let speed = self.effective_config(ctx.input).speed.max(0.0) as f64;
        self.phase_time_s += dt_s.clamp(0.0, 0.2) * speed.max(0.05);
        true
    }

    fn on_world_build(&mut self, ctx: &mut EarthWorldBuildContext<'_>) {
        if !self.enabled || !self.effective_config(ctx.input).enabled {
            return;
        }
        if let Some(snap) = ctx.imagery_snapshot {
            self.imagery_snapshot = Some(Arc::new(snap.clone()));
            self.imagery_rev = self.imagery_rev.wrapping_add(1).max(1);
        }
    }

    fn on_dynamic_overlays(
        &mut self,
        ctx: &mut EarthDynamicOverlayContext<'_>,
        dynamic_meshes: &mut Vec<RetainedDynamicMesh>,
    ) {
        let cfg = self
            .config_override
            .as_ref()
            .unwrap_or(&ctx.input.ocean_waves)
            .clone();
        if !self.enabled || !cfg.enabled || !ctx.input.layers.terrain {
            return;
        }

        let alt_m = ctx.input.camera.altitude_m().max(2.0);
        let dist_to_focus = ctx.input.camera.eye_distance_m().max(alt_m);
        let alt_fade = if ctx.low_level.map_2d_morph > 0.05 {
            1.0 - smoothstep_f32(900_000.0, 2_200_000.0, alt_m as f32)
        } else {
            1.0 - smoothstep_f32(280_000.0, 520_000.0, alt_m as f32)
        };
        if alt_fade <= 0.01 {
            return;
        }

        const RINGS: usize = 32;
        const SECTORS: usize = 56;
        let focus = Geodetic::new(
            ctx.input.camera.focus.lat_deg.clamp(-84.5, 84.5),
            wrap_longitude(ctx.input.camera.focus.lon_deg),
            0.0,
        );
        let focus_ecef = geodetic_to_ecef(focus);
        let focus_frame = LocalFrame::at(focus.lat_deg, focus.lon_deg);
        let patch_radius_m = (dist_to_focus * 3.8).clamp(320.0, 280_000.0);
        let cos_lat = focus.lat_deg.to_radians().cos().abs().max(0.20);
        let flat_t = ctx.low_level.local_25d_blend.clamp(0.0, 1.0) as f64;
        let map_2d = ctx.low_level.map_2d_morph.clamp(0.0, 1.0) as f64;
        let eff_flat = flat_t.max(map_2d);
        let cos_scale = cos_lat + (1.0 - cos_lat) * map_2d;

        // Cache spatial water classification grid when focus/radius/imagery didn't move meaningfully.
        let sig = {
            let mut h = 0xcbf2_9ce4_8422_2325u64;
            for v in [
                (focus.lat_deg * 2_000.0).round() as i64 as u64,
                (focus.lon_deg * 2_000.0).round() as i64 as u64,
                (patch_radius_m.ln() * 250.0).round() as i64 as u64,
                self.imagery_rev,
            ] {
                h ^= v;
                h = h.wrapping_mul(0x1000_0000_1b3);
            }
            h
        };

        let sample_count = 1 + RINGS * SECTORS;
        if sig != self.cached_grid_sig
            || self.cached_imagery_rev != self.imagery_rev
            || self.scratch_sample_lat.len() != sample_count
        {
            self.cached_grid_sig = sig;
            self.cached_imagery_rev = self.imagery_rev;
            self.scratch_sample_lat.clear();
            self.scratch_sample_lon.clear();
            self.scratch_sample_water.clear();
            self.scratch_sample_r_norm.clear();

            let elev_guard = ctx.ground_snapshot.and_then(|l| l.read().ok());
            let elev_ref = elev_guard.as_deref();
            let img_ref = self.imagery_snapshot.as_deref();

            let push_sample = |e_m: f64,
                               n_m: f64,
                               r_norm: f32,
                               lats: &mut Vec<f64>,
                               lons: &mut Vec<f64>,
                               waters: &mut Vec<f32>,
                               r_norms: &mut Vec<f32>| {
                let dlat = (n_m / wgs84::A).to_degrees();
                let dlon = (e_m / (wgs84::A * cos_lat)).to_degrees();
                let lat = (focus.lat_deg + dlat).clamp(-84.9, 84.9);
                let lon = wrap_longitude(focus.lon_deg + dlon);
                let elev = elev_ref.map(|s| s.sample_elevation_m(lat, lon)).unwrap_or(0.0);
                let w = classify_non_land_water(lat, lon, elev, img_ref);
                lats.push(lat);
                lons.push(lon);
                waters.push(w);
                r_norms.push(r_norm);
            };

            push_sample(
                0.0,
                0.0,
                0.0,
                &mut self.scratch_sample_lat,
                &mut self.scratch_sample_lon,
                &mut self.scratch_sample_water,
                &mut self.scratch_sample_r_norm,
            );
            for ring in 1..=RINGS {
                let t = ring as f64 / RINGS as f64;
                let r_m = patch_radius_m * t.powf(1.65);
                for sec in 0..SECTORS {
                    let theta = (sec as f64 / SECTORS as f64) * 2.0 * PI;
                    let e_m = r_m * theta.cos();
                    let n_m = r_m * theta.sin();
                    push_sample(
                        e_m,
                        n_m,
                        t as f32,
                        &mut self.scratch_sample_lat,
                        &mut self.scratch_sample_lon,
                        &mut self.scratch_sample_water,
                        &mut self.scratch_sample_r_norm,
                    );
                }
            }
        }

        // Fast early-out if zero water exists anywhere inside this patch (deep inland!).
        if !self.scratch_sample_water.iter().any(|&w| w > 0.04) {
            return;
        }

        let t_anim = self.phase_time_s + ctx.input.weather.weather_time_s * 0.25;
        let wave_scale = cfg.wave_scale.clamp(0.2, 5.0) as f64;
        let strength = cfg.wave_strength.clamp(0.0, 4.0) as f64;
        let base_wavelength = (dist_to_focus * 0.022).clamp(8.0, 2_800.0) / wave_scale;
        let base_amp = (base_wavelength * 0.052).clamp(0.16, 42.0) * strength;
        let base_lift = (dist_to_focus * 0.00014).clamp(0.10, 22.0) + base_amp * 0.18;
        let g = 9.81_f64;

        let (sun_dir_f32, _, _) = ctx.input.resolved_celestial_state();
        let cam_pos = ctx.input.camera.placement().eye;

        self.scratch_vert_map.clear();
        self.scratch_vert_map.resize(sample_count, -1);
        self.scratch_base_verts.clear();
        self.scratch_base_idx.clear();
        self.scratch_trough_verts.clear();
        self.scratch_trough_idx.clear();
        self.scratch_glint_verts.clear();
        self.scratch_glint_idx.clear();
        self.scratch_glint_score.clear();
        self.scratch_crest_norm.clear();

        // Compute Gerstner wave height + analytic surface normals per water vertex.
        for idx in 0..sample_count {
            let water = self.scratch_sample_water[idx];
            if water <= 0.015 {
                self.scratch_glint_score.push(0.0);
                self.scratch_crest_norm.push(0.5);
                continue;
            }
            let r_norm = self.scratch_sample_r_norm[idx] as f64;
            let outer_fade = 1.0 - smoothstep_f64(0.68, 1.0, r_norm);
            let shore_fade = water as f64;
            let amp_mask = outer_fade * shore_fade;

            let lat = self.scratch_sample_lat[idx];
            let lon = self.scratch_sample_lon[idx];
            let dlon_m = wrap_longitude(lon - focus.lon_deg).to_radians() * wgs84::A * cos_lat;
            let dlat_m = (lat - focus.lat_deg).to_radians() * wgs84::A;

            let mut disp_e = 0.0_f64;
            let mut disp_n = 0.0_f64;
            let mut wave_h = 0.0_f64;
            let mut dh_de = 0.0_f64;
            let mut dh_dn = 0.0_f64;

            for oct in OCEAN_OCTAVES {
                let wl = base_wavelength * oct.wavelength_factor;
                let k = 2.0 * PI / wl.max(1.0);
                let omega = (g * k).sqrt();
                let amp = base_amp * oct.amp_weight * amp_mask;
                let phase = k * (oct.dir_x * dlon_m + oct.dir_y * dlat_m) - omega * t_anim
                    + oct.phase_offset;
                let (s_p, c_p) = phase.sin_cos();
                let q = oct.steepness * amp;
                disp_e += oct.dir_x * q * c_p;
                disp_n += oct.dir_y * q * c_p;
                wave_h += amp * s_p;
                let slope = k * amp * c_p;
                dh_de += oct.dir_x * slope;
                dh_dn += oct.dir_y * slope;
            }

            // Sink smoothly underneath terrain depth buffer near coastlines and circular perimeter
            // so ocean waves terminate cleanly with zero sharp polygons or rectangular seams.
            let sink_m = -((1.0 - shore_fade) * 0.45 + (1.0 - outer_fade) * 0.55)
                * (base_lift + base_amp * 0.85);
            let final_h = base_lift * amp_mask + wave_h + sink_m;
            let displaced_lat =
                (lat + (disp_n * amp_mask / wgs84::A).to_degrees()).clamp(-84.9, 84.9);
            let displaced_lon =
                wrap_longitude(lon + (disp_e * amp_mask / (wgs84::A * cos_lat)).to_degrees());

            let (p_world, local_up) = morph_ocean_point(
                displaced_lat,
                displaced_lon,
                final_h,
                eff_flat,
                map_2d,
                focus,
                focus_ecef,
                &focus_frame,
                cos_scale,
            );
            let rel = p_world.sub(focus_ecef);
            let nx = -dh_de * focus_frame.east.x - dh_dn * focus_frame.north.x + local_up.x;
            let ny = -dh_de * focus_frame.east.y - dh_dn * focus_frame.north.y + local_up.y;
            let nz = -dh_de * focus_frame.east.z - dh_dn * focus_frame.north.z + local_up.z;
            let normal = DVec3::new(nx, ny, nz).normalize();

            let out_v = self.scratch_base_verts.len() as i32;
            self.scratch_vert_map[idx] = out_v;
            self.scratch_base_verts.push(RteTileVertex {
                pos_u: [rel.x as f32, rel.y as f32, rel.z as f32, water],
                norm_v: [
                    normal.x as f32,
                    normal.y as f32,
                    normal.z as f32,
                    r_norm as f32,
                ],
            });

            // Normalized wave crest height `[0..1]` and specular half-vector glint score.
            let view_vec = cam_pos.sub(p_world).normalize();
            let sun_v =
                DVec3::new(sun_dir_f32[0] as f64, sun_dir_f32[1] as f64, sun_dir_f32[2] as f64)
                    .normalize();
            let half_v = view_vec.add(sun_v).normalize();
            let spec = normal.dot(half_v).max(0.0).powf(18.0);
            let crest = ((wave_h / base_amp.max(1e-4)) * 0.5 + 0.5).clamp(0.0, 1.0);
            let glint_score = ((crest * 0.52 + spec * 0.95) * amp_mask) as f32;
            self.scratch_glint_score.push(glint_score);
            self.scratch_crest_norm.push(crest as f32);
        }

        if self.scratch_base_verts.len() < 3 {
            return;
        }

        #[inline]
        fn push_offset_band_tri(
            base_verts: &[RteTileVertex],
            out_verts: &mut Vec<RteTileVertex>,
            out_idx: &mut Vec<u32>,
            va: usize,
            vb: usize,
            vc: usize,
            lift: f32,
        ) {
            let start = out_verts.len() as u32;
            for &v_i in &[va, vb, vc] {
                let mut gv = base_verts[v_i];
                gv.pos_u[0] += gv.norm_v[0] * lift;
                gv.pos_u[1] += gv.norm_v[1] * lift;
                gv.pos_u[2] += gv.norm_v[2] * lift;
                out_verts.push(gv);
            }
            out_idx.push(start);
            out_idx.push(start + 1);
            out_idx.push(start + 2);
        }

        let band_step = (base_amp * 0.04).clamp(0.03, 0.85) as f32;
        let glint_thresh = if cfg.glint_intensity > 0.02 { 0.36 } else { f32::MAX };

        let mut push_tri = |a: usize, b: usize, c: usize| {
            let va = self.scratch_vert_map[a];
            let vb = self.scratch_vert_map[b];
            let vc = self.scratch_vert_map[c];
            if va < 0 || vb < 0 || vc < 0 {
                return;
            }
            let max_w = self.scratch_sample_water[a]
                .max(self.scratch_sample_water[b])
                .max(self.scratch_sample_water[c]);
            if max_w < 0.08 {
                return;
            }
            let (ua, ub, uc) = (va as usize, vb as usize, vc as usize);
            self.scratch_base_idx.push(va as u32);
            self.scratch_base_idx.push(vb as u32);
            self.scratch_base_idx.push(vc as u32);

            let c_avg =
                (self.scratch_crest_norm[a] + self.scratch_crest_norm[b] + self.scratch_crest_norm[c])
                    / 3.0;
            let g_avg =
                (self.scratch_glint_score[a] + self.scratch_glint_score[b] + self.scratch_glint_score[c])
                    / 3.0;

            if c_avg < 0.38 && g_avg > 0.08 {
                push_offset_band_tri(
                    &self.scratch_base_verts,
                    &mut self.scratch_trough_verts,
                    &mut self.scratch_trough_idx,
                    ua,
                    ub,
                    uc,
                    band_step * 0.45,
                );
            } else if g_avg > glint_thresh {
                push_offset_band_tri(
                    &self.scratch_base_verts,
                    &mut self.scratch_glint_verts,
                    &mut self.scratch_glint_idx,
                    ua,
                    ub,
                    uc,
                    band_step * 1.15,
                );
            }
        };

        // Ring 0 -> Ring 1 fan
        for s in 0..SECTORS {
            let s_next = (s + 1) % SECTORS;
            push_tri(0, 1 + s, 1 + s_next);
        }
        // Concentric ring quads
        for ring in 1..RINGS {
            let r0 = 1 + (ring - 1) * SECTORS;
            let r1 = 1 + ring * SECTORS;
            for s in 0..SECTORS {
                let s_next = (s + 1) % SECTORS;
                push_tri(r0 + s, r1 + s, r1 + s_next);
                push_tri(r0 + s, r1 + s_next, r0 + s_next);
            }
        }

        if !self.scratch_base_idx.is_empty() {
            let alpha = (cfg.opacity.clamp(0.05, 0.85) * alt_fade).clamp(0.0, 0.85);
            dynamic_meshes.push(RetainedDynamicMesh {
                id: OCEAN_BASE_MESH_ID,
                revision: ctx.overlay_rev,
                anchor_ecef: [focus_ecef.x, focus_ecef.y, focus_ecef.z],
                vertices: Arc::new(self.scratch_base_verts.clone()),
                indices: Arc::new(self.scratch_base_idx.clone()),
                color: [0.05, 0.29, 0.54, alpha],
                unlit: false,
            });
            if !self.scratch_trough_idx.is_empty() {
                dynamic_meshes.push(RetainedDynamicMesh {
                    id: OCEAN_TROUGH_MESH_ID,
                    revision: ctx.overlay_rev,
                    anchor_ecef: [focus_ecef.x, focus_ecef.y, focus_ecef.z],
                    vertices: Arc::new(self.scratch_trough_verts.clone()),
                    indices: Arc::new(self.scratch_trough_idx.clone()),
                    color: [0.014, 0.095, 0.22, (alpha * 0.65).clamp(0.0, 0.55)],
                    unlit: false,
                });
            }
            if !self.scratch_glint_idx.is_empty() && cfg.glint_intensity > 0.02 {
                let g_alpha =
                    (0.32 * cfg.glint_intensity.clamp(0.0, 2.0) * alt_fade).clamp(0.0, 0.68);
                dynamic_meshes.push(RetainedDynamicMesh {
                    id: OCEAN_GLINT_MESH_ID,
                    revision: ctx.overlay_rev,
                    anchor_ecef: [focus_ecef.x, focus_ecef.y, focus_ecef.z],
                    vertices: Arc::new(self.scratch_glint_verts.clone()),
                    indices: Arc::new(self.scratch_glint_idx.clone()),
                    color: [0.85, 0.94, 1.0, g_alpha],
                    unlit: true,
                });
            }
        }
    }

    fn as_any(&self) -> Option<&dyn Any> {
        Some(self)
    }

    fn as_any_mut(&mut self) -> Option<&mut dyn Any> {
        Some(self)
    }
}
