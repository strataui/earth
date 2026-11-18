//! Tile imagery and elevation sourcing.
//!
//! The reference app couples tile loading to two app-specific things: an
//! `egui` texture upload and a `curl` subprocess. Neither belongs in the
//! engine. Here the two halves are separated:
//!
//! - [`TileSource`] is the decision layer: given a [`TileKey`], return the
//!   decoded pixels and where they came from. Implementations decide cache
//!   layout, transport, and format.
//! - [`TileRegistry`] is the upload layer: it owns the [`TextureStore`] and maps
//!   source identities to stable texture handles.
//!
//! Because the transport sits behind a trait, the plugin has no shell
//! dependency and tests can use an in-memory source.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::{
    mpsc::{channel, Receiver},
    Arc, Condvar, Mutex,
};

use crate::geodesy::{tile_bounds_mercator, Geodetic, TileKey};
use strata_render::texture3d::{
    AddressMode, FilterMode, SamplerDesc, TextureData, TextureDesc, TextureFormat, TextureId,
    TextureStore,
};

/// Decoded image data for one tile.
#[derive(Debug, Clone, PartialEq)]
pub struct TileImage {
    pub width: u32,
    pub height: u32,
    /// Tightly packed RGBA8, top row first.
    pub pixels: Vec<u8>,
}

impl TileImage {
    /// Returns the approximate memory footprint in bytes.
    pub fn byte_size(&self) -> usize {
        self.pixels.len() + std::mem::size_of::<Self>()
    }

    /// Builds a texture payload, validating the pixel count.
    pub fn to_texture_data(&self) -> Option<TextureData> {
        let expected = self.width as usize * self.height as usize * 4;
        if self.pixels.len() != expected {
            return None;
        }
        Some(TextureData {
            desc: TextureDesc {
                width: self.width,
                height: self.height,
                format: TextureFormat::Rgba8,
            },
            pixels: self.pixels.clone().into(),
        })
    }
}

/// Elevation sample for one tile.
#[derive(Debug, Clone, PartialEq)]
pub struct TileElevation {
    pub width: u32,
    pub height: u32,
    /// Height in metres, row-major, `width * height` entries. Row 0 is the
    /// north edge of the tile, matching the imagery orientation.
    pub heights_m: Vec<f32>,
}

impl TileElevation {
    /// Returns the approximate memory footprint in bytes.
    pub fn byte_size(&self) -> usize {
        self.heights_m.len() * std::mem::size_of::<f32>() + std::mem::size_of::<Self>()
    }

    /// Whether the grid is usable.
    pub fn is_valid(&self) -> bool {
        self.width > 0
            && self.height > 0
            && self.heights_m.len() == (self.width as usize * self.height as usize)
    }

    #[inline]
    fn sample_at(&self, x: usize, y: usize) -> f32 {
        self.heights_m[y * self.width as usize + x]
    }

    /// Bilinearly samples elevation at tile UV `(u, v)` in `[0, 1]^2`.
    ///
    /// This mirrors the reference `HeightmapTile::sample_bilinear`: the sample
    /// grid covers the tile corners inclusive, so a normalized coordinate maps
    /// onto `0..=width-1` and `0..=height-1`.
    pub fn sample_bilinear(&self, u: f32, v: f32) -> f32 {
        if !self.is_valid() {
            return 0.0;
        }
        let max_x = (self.width - 1) as f32;
        let max_y = (self.height - 1) as f32;
        let fx = u.clamp(0.0, 1.0) * max_x;
        let fy = v.clamp(0.0, 1.0) * max_y;

        let x0 = (fx.floor() as usize).min(self.width as usize - 1);
        let y0 = (fy.floor() as usize).min(self.height as usize - 1);
        let x1 = (x0 + 1).min(self.width as usize - 1);
        let y1 = (y0 + 1).min(self.height as usize - 1);
        let tx = fx - x0 as f32;
        let ty = fy - y0 as f32;

        let s00 = self.sample_at(x0, y0);
        let s10 = self.sample_at(x1, y0);
        let s01 = self.sample_at(x0, y1);
        let s11 = self.sample_at(x1, y1);
        let top = s00 + (s10 - s00) * tx;
        let bottom = s01 + (s11 - s01) * tx;
        top + (bottom - top) * ty
    }

    /// Samples elevation and its local UV partial derivatives.
    ///
    /// The derivatives are in metres per normalized UV unit, matching the
    /// reference `sample_bilinear_with_gradient` used for hillshading. They are
    /// central differences one sample cell wide, clamped at the tile edge.
    pub fn sample_bilinear_with_gradient(&self, u: f32, v: f32) -> (f32, f32, f32) {
        if !self.is_valid() {
            return (0.0, 0.0, 0.0);
        }
        let eps_u = 1.0 / self.width.max(2) as f32;
        let eps_v = 1.0 / self.height.max(2) as f32;
        let centre = self.sample_bilinear(u, v);
        let east = self.sample_bilinear((u + eps_u).min(1.0), v);
        let west = self.sample_bilinear((u - eps_u).max(0.0), v);
        let north = self.sample_bilinear(u, (v - eps_v).max(0.0));
        let south = self.sample_bilinear(u, (v + eps_v).min(1.0));
        let du = (east - west) / (2.0 * eps_u);
        let dv = (south - north) / (2.0 * eps_v);
        (centre, du, dv)
    }
}

/// Errors a tile source can report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TileError {
    /// The tile is not in the local cache and no transport is configured.
    NotCached,
    /// Transport failed; the message is for logs, not for display.
    Transport(String),
    /// The payload could not be decoded.
    Decode(String),
}

/// Where tile data comes from.
///
/// Implementations must be cheap to call per frame (hits should not block).
/// A miss may block, but callers should treat it as optional and keep drawing
/// the ancestor imagery instead.
pub trait TileSource {
    /// Signals a new frame with active camera focus and altitude so streaming sources
    /// can dynamically reprioritize in-flight download queues.
    fn begin_frame(&mut self, _focus: Geodetic, _altitude_m: f64) {}

    /// Signals that all visible tile requests for the active frame have been submitted,
    /// allowing streaming sources to cancel queued tasks that are no longer visible.
    fn end_frame(&mut self) {}

    /// Returns the imagery for a tile, if available right now.
    fn imagery(&mut self, key: TileKey) -> Result<Option<TileImage>, TileError>;

    /// Non-mutating lookup of imagery already resident in memory/embedded defaults
    /// (must never initiate network I/O). Used when resolving loaded ancestor fallbacks.
    fn peek_cached_imagery(&self, _key: TileKey) -> Option<TileImage> {
        None
    }

    /// Returns high-resolution night city-lights tile imagery if supported and cached/loaded.
    fn night_lights(&mut self, _key: TileKey) -> Result<Option<TileImage>, TileError> {
        Ok(None)
    }

    /// Non-mutating lookup of cached high-res night city-lights imagery.
    fn peek_cached_night_lights(&self, _key: TileKey) -> Option<TileImage> {
        None
    }

    /// Returns the elevation for a tile, if available right now.
    fn elevation(&mut self, key: TileKey) -> Result<Option<TileElevation>, TileError>;

    /// Non-mutating lookup of elevation already resident in memory (must never initiate network I/O).
    /// Used when resolving smooth ancestor elevation fallbacks while finer DEM tiles stream in.
    fn peek_cached_elevation(&self, _key: TileKey) -> Option<TileElevation> {
        None
    }

    /// Updates the active tile API provider configuration at runtime when supported,
    /// returning `true` if the source applied the switch cleanly.
    fn set_provider(&mut self, _config: &TileProviderConfig) -> bool {
        false
    }

    /// Registers an optional wakeup callback invoked on background fetch completion.
    fn set_tile_ready_callback(&mut self, _cb: Option<Arc<dyn Fn() + Send + Sync>>) {}
}

/// Identifies an uploaded tile image so repeated frames reuse one handle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TextureSlot {
    pub key: TileKey,
    /// Bumped when the source replaces an image for the same key.
    pub revision: u32,
}

/// Default sampler for satellite imagery: linear filtering, clamped edges.
pub fn tile_sampler() -> SamplerDesc {
    SamplerDesc {
        filter: FilterMode::Bilinear,
        address_u: AddressMode::ClampToEdge,
        address_v: AddressMode::ClampToEdge,
    }
}

/// Owns uploaded tile textures and hands out stable handles.
#[derive(Debug)]
pub struct TileRegistry {
    store: std::sync::Arc<TextureStore>,
    slots: std::collections::HashMap<TextureSlot, TextureId>,
    /// One-texel neutral texture drawn when a patch has no imagery.
    fallback: TextureId,
}

impl Default for TileRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl TileRegistry {
    pub fn new() -> Self {
        let mut store = TextureStore::new();
        let fallback = store
            .insert_rgba8(1, 1, vec![255u8, 255, 255, 255])
            .expect("a one-texel texture is always valid");
        Self {
            store: std::sync::Arc::new(store),
            slots: std::collections::HashMap::new(),
            fallback,
        }
    }

    /// Neutral texture used for patches that have no imagery yet.
    pub fn fallback_texture(&self) -> TextureId {
        self.fallback
    }

    /// Returns the store the renderer samples from.
    pub fn store(&self) -> &TextureStore {
        &self.store
    }

    /// Returns a shared handle to the store. Cloning is an atomic refcount bump;
    /// attach it to a scene with `Scene3D::set_textures` so every rasterizer
    /// resolves the same imagery.
    pub fn store_arc(&self) -> std::sync::Arc<TextureStore> {
        std::sync::Arc::clone(&self.store)
    }

    /// Uploads an image for a slot if it is not present yet.
    pub fn ensure(&mut self, slot: TextureSlot, image: &TileImage) -> Option<TextureId> {
        if let Some(existing) = self.slots.get(&slot) {
            return Some(*existing);
        }
        let data = image.to_texture_data()?;
        let id = std::sync::Arc::make_mut(&mut self.store).insert(data, tile_sampler())?;
        self.slots.insert(slot, id);
        Some(id)
    }

    /// Looks up a previously uploaded slot.
    pub fn get(&self, slot: TextureSlot) -> Option<TextureId> {
        self.slots.get(&slot).copied()
    }

    /// Drops slots that are not in `keep`, releasing their texture handles.
    ///
    /// Returns the number of evicted slots.
    pub fn retain(&mut self, keep: &std::collections::HashSet<TextureSlot>) -> usize {
        let before = self.slots.len();
        let evicted: Vec<TextureSlot> = self
            .slots
            .keys()
            .copied()
            .filter(|slot| !keep.contains(slot))
            .collect();
        for slot in &evicted {
            if let Some(id) = self.slots.remove(slot) {
                std::sync::Arc::make_mut(&mut self.store).remove(id);
            }
        }
        before - self.slots.len()
    }

    pub fn len(&self) -> usize {
        self.slots.len()
    }

    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }
}

/// A tile source backed by an in-memory map, useful for tests and for callers
/// that decode tiles themselves.
#[derive(Debug, Default)]
pub struct MemoryTileSource {
    imagery: std::collections::HashMap<TileKey, TileImage>,
    night_lights: std::collections::HashMap<TileKey, TileImage>,
    elevation: std::collections::HashMap<TileKey, TileElevation>,
    /// Optional revision counter per key, so a test can force a re-upload.
    revisions: std::collections::HashMap<TileKey, u32>,
}

impl MemoryTileSource {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert_imagery(&mut self, key: TileKey, image: TileImage) -> &mut Self {
        self.imagery.insert(key, image);
        self
    }

    pub fn insert_night_lights(&mut self, key: TileKey, image: TileImage) -> &mut Self {
        self.night_lights.insert(key, image);
        self
    }

    pub fn insert_elevation(&mut self, key: TileKey, elevation: TileElevation) -> &mut Self {
        self.elevation.insert(key, elevation);
        self
    }

    pub fn set_revision(&mut self, key: TileKey, revision: u32) -> &mut Self {
        self.revisions.insert(key, revision);
        self
    }

    pub fn revision(&self, key: TileKey) -> u32 {
        self.revisions.get(&key).copied().unwrap_or(0)
    }
}

impl TileSource for MemoryTileSource {
    fn imagery(&mut self, key: TileKey) -> Result<Option<TileImage>, TileError> {
        Ok(self.imagery.get(&key).cloned())
    }

    fn peek_cached_imagery(&self, key: TileKey) -> Option<TileImage> {
        self.imagery.get(&key).cloned()
    }

    fn night_lights(&mut self, key: TileKey) -> Result<Option<TileImage>, TileError> {
        Ok(self.night_lights.get(&key).cloned())
    }

    fn peek_cached_night_lights(&self, key: TileKey) -> Option<TileImage> {
        self.night_lights.get(&key).cloned()
    }

    fn elevation(&mut self, key: TileKey) -> Result<Option<TileElevation>, TileError> {
        Ok(self.elevation.get(&key).cloned())
    }

    fn peek_cached_elevation(&self, key: TileKey) -> Option<TileElevation> {
        self.elevation.get(&key).cloned()
    }
}

/// A tile source backed by a local filesystem directory containing tile files.
///
/// Looks up imagery at `{base_dir}/{z}/{x}/{y}.raw` or `{base_dir}/{z}_{x}_{y}.raw` (or `.rgba`).
/// Looks up elevation at `{base_dir}/{z}/{x}/{y}.elev` or `{base_dir}/{z}_{x}_{y}.elev` (or `.bin`).
#[derive(Debug, Clone)]
pub struct DirectoryTileSource {
    base_dir: std::path::PathBuf,
    default_tile_size: u32,
    max_disk_bytes: u64,
}

impl DirectoryTileSource {
    pub const DEFAULT_MAX_DISK_BYTES: u64 = 256 * 1024 * 1024; // 256 MB

    pub fn new(base_dir: impl Into<std::path::PathBuf>) -> Self {
        Self {
            base_dir: base_dir.into(),
            default_tile_size: 256,
            max_disk_bytes: Self::DEFAULT_MAX_DISK_BYTES,
        }
    }

    pub fn with_tile_size(mut self, size: u32) -> Self {
        self.default_tile_size = size;
        self
    }

    pub fn with_max_disk_bytes(mut self, max_bytes: u64) -> Self {
        self.max_disk_bytes = max_bytes;
        self
    }

    pub fn max_disk_bytes(&self) -> u64 {
        self.max_disk_bytes
    }

    pub fn base_dir(&self) -> &std::path::Path {
        &self.base_dir
    }

    /// Computes the total byte footprint of all tile files within `base_dir`.
    pub fn disk_usage_bytes(&self) -> std::io::Result<u64> {
        let mut total = 0u64;
        if !self.base_dir.exists() {
            return Ok(0);
        }
        for path in walk_files(&self.base_dir)? {
            if let Ok(meta) = path.metadata() {
                if meta.is_file() {
                    total += meta.len();
                }
            }
        }
        Ok(total)
    }

    /// Enforces `max_bytes` ceiling by evicting the oldest modified tile files on disk.
    /// Returns the number of files deleted.
    pub fn evict_disk_cache(&self, max_bytes: u64) -> std::io::Result<usize> {
        if !self.base_dir.exists() {
            return Ok(0);
        }
        let mut files: Vec<(std::path::PathBuf, u64, std::time::SystemTime)> = Vec::new();
        let mut total_bytes = 0u64;

        for path in walk_files(&self.base_dir)? {
            if let Ok(meta) = path.metadata() {
                if meta.is_file() {
                    let len = meta.len();
                    total_bytes += len;
                    let mtime = meta.modified().unwrap_or(std::time::SystemTime::UNIX_EPOCH);
                    files.push((path, len, mtime));
                }
            }
        }

        if total_bytes <= max_bytes {
            return Ok(0);
        }

        // Sort oldest first (LRU order based on file modification timestamp)
        files.sort_by_key(|(_, _, mtime)| *mtime);

        let mut evicted = 0;
        for (path, len, _) in files {
            if total_bytes <= max_bytes {
                break;
            }
            if std::fs::remove_file(&path).is_ok() {
                total_bytes = total_bytes.saturating_sub(len);
                evicted += 1;
            }
        }

        Ok(evicted)
    }

    /// Enforces the configured `max_disk_bytes` ceiling (default 256 MB).
    pub fn enforce_disk_ceiling(&self) -> std::io::Result<usize> {
        self.evict_disk_cache(self.max_disk_bytes)
    }

    /// Writes an image tile to disk at `{base_dir}/{level}/{x}/{y}.raw` with `[w, h]` header,
    /// then enforces the disk cache ceiling.
    pub fn write_imagery(
        &self,
        key: TileKey,
        image: &TileImage,
    ) -> std::io::Result<std::path::PathBuf> {
        let dir = self.base_dir.join(format!("{}/{}", key.level, key.x));
        std::fs::create_dir_all(&dir)?;
        let path = dir.join(format!("{}.raw", key.y));

        let mut data = Vec::with_capacity(8 + image.pixels.len());
        data.extend_from_slice(&image.width.to_le_bytes());
        data.extend_from_slice(&image.height.to_le_bytes());
        data.extend_from_slice(&image.pixels);

        std::fs::write(&path, &data)?;
        let _ = self.enforce_disk_ceiling();
        Ok(path)
    }

    /// Writes an elevation tile to disk at `{base_dir}/{level}/{x}/{y}.elev` with `[w, h]` header,
    /// then enforces the disk cache ceiling.
    pub fn write_elevation(
        &self,
        key: TileKey,
        elevation: &TileElevation,
    ) -> std::io::Result<std::path::PathBuf> {
        let dir = self.base_dir.join(format!("{}/{}", key.level, key.x));
        std::fs::create_dir_all(&dir)?;
        let path = dir.join(format!("{}.elev", key.y));

        let mut data = Vec::with_capacity(8 + elevation.heights_m.len() * 4);
        data.extend_from_slice(&elevation.width.to_le_bytes());
        data.extend_from_slice(&elevation.height.to_le_bytes());
        for h in &elevation.heights_m {
            data.extend_from_slice(&h.to_le_bytes());
        }

        std::fs::write(&path, &data)?;
        let _ = self.enforce_disk_ceiling();
        Ok(path)
    }
}

impl TileSource for DirectoryTileSource {
    fn imagery(&mut self, key: TileKey) -> Result<Option<TileImage>, TileError> {
        let candidates = [
            self.base_dir
                .join(format!("{}/{}/{}.raw", key.level, key.x, key.y)),
            self.base_dir
                .join(format!("{}/{}/{}.rgba", key.level, key.x, key.y)),
            self.base_dir
                .join(format!("{}_{}_{}.raw", key.level, key.x, key.y)),
            self.base_dir
                .join(format!("{}_{}_{}.rgba", key.level, key.x, key.y)),
        ];

        let path = match candidates.into_iter().find(|p| p.is_file()) {
            Some(p) => p,
            None => return Ok(None),
        };

        let bytes = std::fs::read(&path)
            .map_err(|e| TileError::Transport(format!("failed reading {}: {e}", path.display())))?;

        if bytes.is_empty() {
            return Ok(None);
        }

        // Check if prefixed with [u32 width, u32 height] (8 bytes)
        if bytes.len() >= 8 {
            let w = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
            let h = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
            if w > 0 && h > 0 && bytes.len() == 8 + (w as usize * h as usize * 4) {
                return Ok(Some(TileImage {
                    width: w,
                    height: h,
                    pixels: bytes[8..].to_vec(),
                }));
            }
        }

        // Check if raw RGBA byte buffer of default_tile_size
        let expected_default =
            self.default_tile_size as usize * self.default_tile_size as usize * 4;
        if bytes.len() == expected_default {
            return Ok(Some(TileImage {
                width: self.default_tile_size,
                height: self.default_tile_size,
                pixels: bytes,
            }));
        }

        // Infer square dimensions if bytes length is a multiple of 4
        if bytes.len().is_multiple_of(4) {
            let pixels = bytes.len() / 4;
            let side = (pixels as f64).sqrt().round() as usize;
            if side * side == pixels {
                return Ok(Some(TileImage {
                    width: side as u32,
                    height: side as u32,
                    pixels: bytes,
                }));
            }
        }

        Err(TileError::Decode(format!(
            "invalid image tile data length {} at {}",
            bytes.len(),
            path.display()
        )))
    }

    fn elevation(&mut self, key: TileKey) -> Result<Option<TileElevation>, TileError> {
        let candidates = [
            self.base_dir
                .join(format!("{}/{}/{}.elev", key.level, key.x, key.y)),
            self.base_dir
                .join(format!("{}/{}/{}.bin", key.level, key.x, key.y)),
            self.base_dir
                .join(format!("{}_{}_{}.elev", key.level, key.x, key.y)),
            self.base_dir
                .join(format!("{}_{}_{}.bin", key.level, key.x, key.y)),
        ];

        let path = match candidates.into_iter().find(|p| p.is_file()) {
            Some(p) => p,
            None => return Ok(None),
        };

        let bytes = std::fs::read(&path)
            .map_err(|e| TileError::Transport(format!("failed reading {}: {e}", path.display())))?;

        if bytes.is_empty() {
            return Ok(None);
        }

        let (w, h, data_bytes) = if bytes.len() >= 8 {
            let w = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
            let h = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
            if w > 0 && h > 0 && bytes.len() == 8 + (w as usize * h as usize * 4) {
                (w, h, &bytes[8..])
            } else {
                let count = bytes.len() / 4;
                let side = (count as f64).sqrt().round() as u32;
                if (side as usize * side as usize * 4) == bytes.len() {
                    (side, side, bytes.as_slice())
                } else {
                    return Err(TileError::Decode(format!(
                        "invalid elevation tile data length {} at {}",
                        bytes.len(),
                        path.display()
                    )));
                }
            }
        } else {
            let count = bytes.len() / 4;
            let side = (count as f64).sqrt().round() as u32;
            if side > 0 && (side as usize * side as usize * 4) == bytes.len() {
                (side, side, bytes.as_slice())
            } else {
                return Err(TileError::Decode(format!(
                    "elevation file length {} invalid at {}",
                    bytes.len(),
                    path.display()
                )));
            }
        };

        let heights_m: Vec<f32> = data_bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        Ok(Some(TileElevation {
            width: w,
            height: h,
            heights_m,
        }))
    }
}

fn walk_files(dir: &std::path::Path) -> std::io::Result<Vec<std::path::PathBuf>> {
    let mut files = Vec::new();
    if dir.is_dir() {
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                files.extend(walk_files(&path)?);
            } else if path.is_file() {
                files.push(path);
            }
        }
    }
    Ok(files)
}

/// Default in-memory cache byte budget: 256 MB.
pub const DEFAULT_MAX_CACHE_BYTES: usize = 256 * 1024 * 1024;

/// Key discriminator for cached tile data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TileDataKey {
    Imagery(TileKey),
    Elevation(TileKey),
    NightLights(TileKey),
}

/// In-memory cached tile item.
#[derive(Debug, Clone, PartialEq)]
pub enum CachedTileData {
    Imagery(TileImage),
    Elevation(TileElevation),
}

impl CachedTileData {
    pub fn byte_size(&self) -> usize {
        match self {
            Self::Imagery(img) => img.byte_size(),
            Self::Elevation(elev) => elev.byte_size(),
        }
    }
}

#[derive(Debug, Clone)]
struct LruNode {
    key: TileDataKey,
    value: CachedTileData,
    prev: Option<usize>,
    next: Option<usize>,
}

/// Bounded LRU in-memory cache for tile imagery and elevation.
///
/// Tracks cumulative memory consumption in bytes and evicts least recently
/// used entries when exceeding `max_bytes` (defaults to 256 MB).
#[derive(Debug, Clone)]
pub struct LruTileCache {
    max_bytes: usize,
    current_bytes: usize,
    nodes: Vec<Option<LruNode>>,
    free_slots: Vec<usize>,
    map: std::collections::HashMap<TileDataKey, usize>,
    head: Option<usize>,
    tail: Option<usize>,
}

impl Default for LruTileCache {
    fn default() -> Self {
        Self::default_256mb()
    }
}

impl LruTileCache {
    /// Creates a new bounded LRU cache with the specified byte budget.
    pub fn new(max_bytes: usize) -> Self {
        Self {
            max_bytes,
            current_bytes: 0,
            nodes: Vec::new(),
            free_slots: Vec::new(),
            map: std::collections::HashMap::new(),
            head: None,
            tail: None,
        }
    }

    /// Creates a new bounded LRU cache with default 256 MB capacity.
    pub fn default_256mb() -> Self {
        Self::new(DEFAULT_MAX_CACHE_BYTES)
    }

    /// Maximum capacity in bytes.
    pub fn max_bytes(&self) -> usize {
        self.max_bytes
    }

    /// Sets the maximum capacity in bytes, evicting entries if needed.
    pub fn set_max_bytes(&mut self, max_bytes: usize) {
        self.max_bytes = max_bytes;
        self.evict_if_needed();
    }

    /// Current approximate memory consumption in bytes.
    pub fn current_bytes(&self) -> usize {
        self.current_bytes
    }

    /// Number of cached tile entries.
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Whether the cache contains zero entries.
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Clears all entries from the cache.
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.free_slots.clear();
        self.map.clear();
        self.head = None;
        self.tail = None;
        self.current_bytes = 0;
    }

    /// Clears only cached imagery tiles, leaving elevation and night-lights intact.
    pub fn clear_imagery(&mut self) {
        let keys: Vec<TileDataKey> = self
            .map
            .keys()
            .copied()
            .filter(|k| matches!(k, TileDataKey::Imagery(_)))
            .collect();
        for k in keys {
            let _ = self.remove(&k);
        }
    }

    /// Clears only cached night-lights tiles.
    pub fn clear_night_lights(&mut self) {
        let keys: Vec<TileDataKey> = self
            .map
            .keys()
            .copied()
            .filter(|k| matches!(k, TileDataKey::NightLights(_)))
            .collect();
        for k in keys {
            let _ = self.remove(&k);
        }
    }

    /// Clears only cached elevation tiles.
    pub fn clear_elevation(&mut self) {
        let keys: Vec<TileDataKey> = self
            .map
            .keys()
            .copied()
            .filter(|k| matches!(k, TileDataKey::Elevation(_)))
            .collect();
        for k in keys {
            let _ = self.remove(&k);
        }
    }

    pub fn contains_imagery(&self, key: TileKey) -> bool {
        self.map.contains_key(&TileDataKey::Imagery(key))
    }

    pub fn contains_night_lights(&self, key: TileKey) -> bool {
        self.map.contains_key(&TileDataKey::NightLights(key))
    }

    pub fn contains_elevation(&self, key: TileKey) -> bool {
        self.map.contains_key(&TileDataKey::Elevation(key))
    }

    fn get_img(&mut self, data_key: TileDataKey) -> Option<TileImage> {
        match self.get(&data_key) {
            Some(CachedTileData::Imagery(img)) => Some(img),
            _ => None,
        }
    }

    fn peek_img(&self, data_key: TileDataKey) -> Option<TileImage> {
        let &idx = self.map.get(&data_key)?;
        match self.nodes.get(idx).and_then(|n| n.as_ref())?.value {
            CachedTileData::Imagery(ref img) => Some(img.clone()),
            CachedTileData::Elevation(_) => None,
        }
    }

    /// Fetches imagery for `key`, marking it as most recently used.
    pub fn get_imagery(&mut self, key: TileKey) -> Option<TileImage> {
        self.get_img(TileDataKey::Imagery(key))
    }

    /// Peeks cached imagery without updating LRU order or needing mutable access.
    pub fn peek_imagery(&self, key: TileKey) -> Option<TileImage> {
        self.peek_img(TileDataKey::Imagery(key))
    }

    /// Inserts imagery for `key`, evicting older entries if exceeding the byte budget.
    pub fn insert_imagery(&mut self, key: TileKey, image: TileImage) {
        self.insert(TileDataKey::Imagery(key), CachedTileData::Imagery(image));
    }

    /// Fetches cached high-res night lights imagery for `key`, marking it as most recently used.
    pub fn get_night_lights(&mut self, key: TileKey) -> Option<TileImage> {
        self.get_img(TileDataKey::NightLights(key))
    }

    /// Peeks cached high-res night lights imagery without mutating LRU order.
    pub fn peek_night_lights(&self, key: TileKey) -> Option<TileImage> {
        self.peek_img(TileDataKey::NightLights(key))
    }

    /// Inserts high-res night lights imagery for `key`.
    pub fn insert_night_lights(&mut self, key: TileKey, image: TileImage) {
        self.insert(
            TileDataKey::NightLights(key),
            CachedTileData::Imagery(image),
        );
    }

    /// Fetches elevation for `key`, marking it as most recently used.
    pub fn get_elevation(&mut self, key: TileKey) -> Option<TileElevation> {
        match self.get(&TileDataKey::Elevation(key)) {
            Some(CachedTileData::Elevation(elev)) => Some(elev),
            _ => None,
        }
    }

    /// Peeks cached elevation without updating LRU order or needing mutable access.
    pub fn peek_elevation(&self, key: TileKey) -> Option<TileElevation> {
        let &idx = self.map.get(&TileDataKey::Elevation(key))?;
        match self.nodes.get(idx).and_then(|n| n.as_ref())?.value {
            CachedTileData::Elevation(ref elev) => Some(elev.clone()),
            CachedTileData::Imagery(_) => None,
        }
    }

    /// Inserts elevation for `key`, evicting older entries if exceeding the byte budget.
    pub fn insert_elevation(&mut self, key: TileKey, elevation: TileElevation) {
        self.insert(
            TileDataKey::Elevation(key),
            CachedTileData::Elevation(elevation),
        );
    }

    /// Removes imagery for `key` if present.
    pub fn remove_imagery(&mut self, key: TileKey) -> Option<TileImage> {
        match self.remove(&TileDataKey::Imagery(key)) {
            Some(CachedTileData::Imagery(img)) => Some(img),
            _ => None,
        }
    }

    /// Removes elevation for `key` if present.
    pub fn remove_elevation(&mut self, key: TileKey) -> Option<TileElevation> {
        match self.remove(&TileDataKey::Elevation(key)) {
            Some(CachedTileData::Elevation(elev)) => Some(elev),
            _ => None,
        }
    }

    fn get(&mut self, key: &TileDataKey) -> Option<CachedTileData> {
        let idx = *self.map.get(key)?;
        self.detach(idx);
        self.push_head(idx);
        self.nodes[idx].as_ref().map(|n| n.value.clone())
    }

    fn insert(&mut self, key: TileDataKey, value: CachedTileData) {
        let item_bytes = value.byte_size();
        if let Some(&idx) = self.map.get(&key) {
            let old_bytes = self.nodes[idx].as_ref().unwrap().value.byte_size();
            self.current_bytes = self.current_bytes.saturating_sub(old_bytes);
            self.nodes[idx].as_mut().unwrap().value = value;
            self.current_bytes += item_bytes;
            self.detach(idx);
            self.push_head(idx);
        } else {
            let idx = if let Some(free_idx) = self.free_slots.pop() {
                self.nodes[free_idx] = Some(LruNode {
                    key,
                    value,
                    prev: None,
                    next: None,
                });
                free_idx
            } else {
                let new_idx = self.nodes.len();
                self.nodes.push(Some(LruNode {
                    key,
                    value,
                    prev: None,
                    next: None,
                }));
                new_idx
            };
            self.current_bytes += item_bytes;
            self.push_head(idx);
            self.map.insert(key, idx);
        }

        self.evict_if_needed();
    }

    fn remove(&mut self, key: &TileDataKey) -> Option<CachedTileData> {
        let idx = self.map.remove(key)?;
        self.detach(idx);
        let node = self.nodes[idx].take()?;
        self.current_bytes = self.current_bytes.saturating_sub(node.value.byte_size());
        self.free_slots.push(idx);
        Some(node.value)
    }

    fn detach(&mut self, idx: usize) {
        let node = self.nodes[idx].as_ref().unwrap();
        let prev = node.prev;
        let next = node.next;

        if let Some(p) = prev {
            self.nodes[p].as_mut().unwrap().next = next;
        } else {
            self.head = next;
        }

        if let Some(n) = next {
            self.nodes[n].as_mut().unwrap().prev = prev;
        } else {
            self.tail = prev;
        }

        let curr = self.nodes[idx].as_mut().unwrap();
        curr.prev = None;
        curr.next = None;
    }

    fn push_head(&mut self, idx: usize) {
        self.nodes[idx].as_mut().unwrap().prev = None;
        self.nodes[idx].as_mut().unwrap().next = self.head;

        if let Some(old_head) = self.head {
            self.nodes[old_head].as_mut().unwrap().prev = Some(idx);
        } else {
            self.tail = Some(idx);
        }

        self.head = Some(idx);
    }

    fn evict_if_needed(&mut self) {
        while self.current_bytes > self.max_bytes && self.tail.is_some() {
            let tail_idx = self.tail.unwrap();
            let tail_node = self.nodes[tail_idx].as_ref().unwrap();
            let key = tail_node.key;
            let bytes = tail_node.value.byte_size();

            self.map.remove(&key);
            self.detach(tail_idx);
            self.nodes[tail_idx] = None;
            self.free_slots.push(tail_idx);
            self.current_bytes = self.current_bytes.saturating_sub(bytes);
        }
    }
}

/// A decorating `TileSource` that provides bounded LRU in-memory caching
/// over an underlying tile source.
#[derive(Debug)]
pub struct LruTileCacheSource<S: TileSource> {
    inner: S,
    cache: LruTileCache,
    hits: u64,
    misses: u64,
}

impl<S: TileSource> LruTileCacheSource<S> {
    /// Wraps `inner` with an LRU cache bounded to `max_bytes`.
    pub fn new(inner: S, max_bytes: usize) -> Self {
        Self {
            inner,
            cache: LruTileCache::new(max_bytes),
            hits: 0,
            misses: 0,
        }
    }

    /// Wraps `inner` with an LRU cache with the default 256 MB ceiling.
    pub fn with_256mb(inner: S) -> Self {
        Self::new(inner, DEFAULT_MAX_CACHE_BYTES)
    }

    pub fn cache(&self) -> &LruTileCache {
        &self.cache
    }

    pub fn cache_mut(&mut self) -> &mut LruTileCache {
        &mut self.cache
    }

    pub fn inner(&self) -> &S {
        &self.inner
    }

    pub fn inner_mut(&mut self) -> &mut S {
        &mut self.inner
    }

    pub fn hits(&self) -> u64 {
        self.hits
    }

    pub fn misses(&self) -> u64 {
        self.misses
    }
}

impl<S: TileSource> TileSource for LruTileCacheSource<S> {
    fn begin_frame(&mut self, focus: Geodetic, altitude_m: f64) {
        self.inner.begin_frame(focus, altitude_m);
    }

    fn end_frame(&mut self) {
        self.inner.end_frame();
    }

    fn imagery(&mut self, key: TileKey) -> Result<Option<TileImage>, TileError> {
        if let Some(image) = self.cache.get_imagery(key) {
            self.hits += 1;
            return Ok(Some(image));
        }
        self.misses += 1;
        let result = self.inner.imagery(key)?;
        if let Some(image) = &result {
            self.cache.insert_imagery(key, image.clone());
        }
        Ok(result)
    }

    fn peek_cached_imagery(&self, key: TileKey) -> Option<TileImage> {
        self.cache
            .peek_imagery(key)
            .or_else(|| self.inner.peek_cached_imagery(key))
    }

    fn elevation(&mut self, key: TileKey) -> Result<Option<TileElevation>, TileError> {
        if let Some(elev) = self.cache.get_elevation(key) {
            self.hits += 1;
            return Ok(Some(elev));
        }
        self.misses += 1;
        let result = self.inner.elevation(key)?;
        if let Some(elev) = &result {
            self.cache.insert_elevation(key, elev.clone());
        }
        Ok(result)
    }

    fn peek_cached_elevation(&self, key: TileKey) -> Option<TileElevation> {
        self.cache
            .peek_elevation(key)
            .or_else(|| self.inner.peek_cached_elevation(key))
    }
}

/// Converts a slippy/quadtree [`TileKey`] into a Bing Maps QuadKey string (`'0'..='3'` digits).
pub fn tile_key_to_quadkey(key: TileKey) -> String {
    if key.level == 0 {
        return String::new();
    }
    let mut qk = String::with_capacity(key.level as usize);
    for i in (0..key.level).rev() {
        let mask = 1u32 << i;
        let mut digit = 0u8;
        if (key.x & mask) != 0 {
            digit |= 1;
        }
        if (key.y & mask) != 0 {
            digit |= 2;
        }
        qk.push((b'0' + digit) as char);
    }
    qk
}

/// Formats a Slippy Map XYZ, TMS (`{-y}`), WMTS, QuadKey (`{quadkey}`), or API-key template URL.
///
/// Supported tokens:
/// - `{z}` / `{TileMatrix}`: tile zoom level
/// - `{x}` / `{TileCol}`: tile column index
/// - `{y}` / `{TileRow}`: tile row index (XYZ top-to-bottom)
/// - `{-y}`: TMS inverted row index `(1 << z) - 1 - y`
/// - `{quadkey}` / `{q}`: Bing Maps QuadKey string
/// - `{s}`: subdomain round-robin
/// - `{api_key}` / `{apikey}` / `{access_token}` / `{token}`: API key substitution
pub fn format_tile_url(
    template: &str,
    key: TileKey,
    subdomains: &[String],
    subdomain_counter: &mut usize,
    api_key: Option<&str>,
) -> String {
    let per_axis = 1u32 << key.level.min(31);
    let tms_y = per_axis.saturating_sub(1).saturating_sub(key.y);
    let quadkey = if template.contains("{quadkey}") || template.contains("{q}") {
        tile_key_to_quadkey(key)
    } else {
        String::new()
    };

    let mut url = template
        .replace("{z}", &key.level.to_string())
        .replace("{x}", &key.x.to_string())
        .replace("{y}", &key.y.to_string())
        .replace("{-y}", &tms_y.to_string())
        .replace("{TileMatrix}", &key.level.to_string())
        .replace("{TileCol}", &key.x.to_string())
        .replace("{TileRow}", &key.y.to_string())
        .replace("{quadkey}", &quadkey)
        .replace("{q}", &quadkey);

    if let Some(k) = api_key {
        url = url
            .replace("{api_key}", k)
            .replace("{apikey}", k)
            .replace("{access_token}", k)
            .replace("{token}", k);
    }

    if url.contains("{s}") {
        if !subdomains.is_empty() {
            let sub = &subdomains[*subdomain_counter % subdomains.len()];
            *subdomain_counter += 1;
            url = url.replace("{s}", sub);
        } else {
            url = url.replace("{s}", "a");
        }
    }
    url
}

/// Formats a Slippy Map XYZ or WMTS URL template with a given TileKey.
pub fn format_slippy_url(
    template: &str,
    key: TileKey,
    subdomains: &[String],
    subdomain_counter: &mut usize,
) -> String {
    format_tile_url(template, key, subdomains, subdomain_counter, None)
}

/// Decodes standard 8-bit PNG byte streams into raw 32-bit RGBA pixels.
///
/// Implements RFC 2083 PNG decompression and reconstruction across Grayscale (0),
/// RGB (2), Indexed (3), Grayscale+Alpha (4), and RGBA (6) color types with standard
/// scanline filters (None, Sub, Up, Average, Paeth).
pub fn decode_png_to_rgba(bytes: &[u8]) -> Result<(u32, u32, Vec<u8>), TileError> {
    const PNG_SIG: &[u8; 8] = b"\x89PNG\r\n\x1a\n";
    if bytes.len() < 8 || &bytes[0..8] != PNG_SIG {
        return Err(TileError::Decode("invalid PNG signature".to_string()));
    }

    let mut pos = 8;
    let mut width = 0u32;
    let mut height = 0u32;
    let mut bit_depth = 8u8;
    let mut color_type = 6u8;
    let mut palette = Vec::new();
    let mut idat_data = Vec::new();

    while pos + 8 <= bytes.len() {
        let chunk_len =
            u32::from_be_bytes([bytes[pos], bytes[pos + 1], bytes[pos + 2], bytes[pos + 3]])
                as usize;
        let chunk_type = &bytes[pos + 4..pos + 8];
        pos += 8;
        if pos + chunk_len > bytes.len() {
            return Err(TileError::Decode("truncated PNG chunk".to_string()));
        }
        let data = &bytes[pos..pos + chunk_len];
        pos += chunk_len + 4; // Skip chunk data + 4-byte CRC

        match chunk_type {
            b"IHDR" => {
                if data.len() < 13 {
                    return Err(TileError::Decode("invalid IHDR chunk length".to_string()));
                }
                width = u32::from_be_bytes([data[0], data[1], data[2], data[3]]);
                height = u32::from_be_bytes([data[4], data[5], data[6], data[7]]);
                bit_depth = data[8];
                color_type = data[9];
                let compression = data[10];
                let filter = data[11];
                let interlace = data[12];
                if compression != 0 || filter != 0 || interlace != 0 {
                    return Err(TileError::Decode(
                        "unsupported PNG compression/filter/interlace mode".to_string(),
                    ));
                }
            }
            b"PLTE" => {
                palette.extend_from_slice(data);
            }
            b"IDAT" => {
                idat_data.extend_from_slice(data);
            }
            b"IEND" => break,
            _ => {}
        }
    }

    if width == 0 || height == 0 || idat_data.is_empty() {
        return Err(TileError::Decode(
            "missing IHDR or IDAT in PNG data".to_string(),
        ));
    }

    if bit_depth != 8 {
        return Err(TileError::Decode(format!(
            "unsupported bit depth {bit_depth}, only 8-bit PNGs are supported"
        )));
    }

    let channels: usize = match color_type {
        0 => 1, // Grayscale
        2 => 3, // RGB
        3 => 1, // Indexed
        4 => 2, // Grayscale + Alpha
        6 => 4, // RGBA
        _ => {
            return Err(TileError::Decode(format!(
                "unsupported PNG color type {color_type}"
            )))
        }
    };

    use std::io::Read;
    let mut decoder = flate2::read::ZlibDecoder::new(&idat_data[..]);
    let mut decompressed = Vec::new();
    decoder
        .read_to_end(&mut decompressed)
        .map_err(|e| TileError::Decode(format!("zlib decompression failed: {e}")))?;

    let stride = width as usize * channels;
    let expected_len = height as usize * (stride + 1);
    if decompressed.len() < expected_len {
        return Err(TileError::Decode(format!(
            "incomplete scanline data: expected {expected_len}, got {}",
            decompressed.len()
        )));
    }

    let mut raw_scanlines = vec![0u8; height as usize * stride];
    let mut prev_row = vec![0u8; stride];

    for y in 0..height as usize {
        let src_row_start = y * (stride + 1);
        let filter_type = decompressed[src_row_start];
        let src_row = &decompressed[src_row_start + 1..src_row_start + 1 + stride];
        let dst_row_start = y * stride;

        for x in 0..stride {
            let a = if x >= channels {
                raw_scanlines[dst_row_start + x - channels]
            } else {
                0
            };
            let b = prev_row[x];
            let c = if x >= channels {
                prev_row[x - channels]
            } else {
                0
            };

            let filtered = src_row[x];
            let reconstructed = match filter_type {
                0 => filtered,
                1 => filtered.wrapping_add(a),
                2 => filtered.wrapping_add(b),
                3 => filtered.wrapping_add(((a as u16 + b as u16) / 2) as u8),
                4 => {
                    let p = a as i16 + b as i16 - c as i16;
                    let pa = (p - a as i16).abs();
                    let pb = (p - b as i16).abs();
                    let pc = (p - c as i16).abs();
                    let pr = if pa <= pb && pa <= pc {
                        a
                    } else if pb <= pc {
                        b
                    } else {
                        c
                    };
                    filtered.wrapping_add(pr)
                }
                _ => {
                    return Err(TileError::Decode(format!(
                        "unknown PNG filter {filter_type}"
                    )))
                }
            };
            raw_scanlines[dst_row_start + x] = reconstructed;
        }
        prev_row.copy_from_slice(&raw_scanlines[dst_row_start..dst_row_start + stride]);
    }

    let pixel_count = width as usize * height as usize;
    let mut rgba = vec![255u8; pixel_count * 4];

    match color_type {
        0 => {
            // Grayscale
            for i in 0..pixel_count {
                let g = raw_scanlines[i];
                rgba[i * 4] = g;
                rgba[i * 4 + 1] = g;
                rgba[i * 4 + 2] = g;
                rgba[i * 4 + 3] = 255;
            }
        }
        2 => {
            // RGB
            for i in 0..pixel_count {
                rgba[i * 4] = raw_scanlines[i * 3];
                rgba[i * 4 + 1] = raw_scanlines[i * 3 + 1];
                rgba[i * 4 + 2] = raw_scanlines[i * 3 + 2];
                rgba[i * 4 + 3] = 255;
            }
        }
        3 => {
            // Indexed
            for i in 0..pixel_count {
                let idx = raw_scanlines[i] as usize;
                if idx * 3 + 2 < palette.len() {
                    rgba[i * 4] = palette[idx * 3];
                    rgba[i * 4 + 1] = palette[idx * 3 + 1];
                    rgba[i * 4 + 2] = palette[idx * 3 + 2];
                }
                rgba[i * 4 + 3] = 255;
            }
        }
        4 => {
            // Grayscale + Alpha
            for i in 0..pixel_count {
                let g = raw_scanlines[i * 2];
                let a = raw_scanlines[i * 2 + 1];
                rgba[i * 4] = g;
                rgba[i * 4 + 1] = g;
                rgba[i * 4 + 2] = g;
                rgba[i * 4 + 3] = a;
            }
        }
        6 => {
            // RGBA
            rgba.copy_from_slice(&raw_scanlines[..pixel_count * 4]);
        }
        _ => unreachable!(),
    }

    Ok((width, height, rgba))
}

/// Decodes Mapbox Terrain-RGB pixels into metric elevations.
///
/// Uses the standard Terrain-RGB equation:
/// `height = -10000.0 + (R * 65536.0 + G * 256.0 + B) * 0.1`
pub fn decode_terrain_rgb(width: u32, height: u32, rgba_pixels: &[u8]) -> TileElevation {
    let count = width as usize * height as usize;
    let mut heights_m = Vec::with_capacity(count);
    for i in 0..count {
        let r = rgba_pixels[i * 4] as f32;
        let g = rgba_pixels[i * 4 + 1] as f32;
        let b = rgba_pixels[i * 4 + 2] as f32;
        let h = -10000.0 + (r * 65536.0 + g * 256.0 + b) * 0.1;
        heights_m.push(h);
    }
    TileElevation {
        width,
        height,
        heights_m,
    }
}

/// Decodes Mapzen/AWS Terrarium PNG pixels into metric elevations.
///
/// Uses the standard Terrarium equation:
/// `height = (R * 256.0 + G + B / 256.0) - 32768.0`
pub fn decode_terrarium(width: u32, height: u32, rgba_pixels: &[u8]) -> TileElevation {
    let count = width as usize * height as usize;
    let mut heights_m = Vec::with_capacity(count);
    for i in 0..count {
        let r = rgba_pixels[i * 4] as f32;
        let g = rgba_pixels[i * 4 + 1] as f32;
        let b = rgba_pixels[i * 4 + 2] as f32;
        let h = (r * 256.0 + g + b / 256.0) - 32768.0;
        heights_m.push(h);
    }
    TileElevation {
        width,
        height,
        heights_m,
    }
}

/// Pluggable HTTP transport interface for tile fetching.
pub trait HttpTransport: Send + Sync + 'static {
    fn fetch(&self, url: &str) -> Result<Vec<u8>, TileError>;
}

/// Standard production HTTP/HTTPS transport using pure-Rust `ureq` + `rustls`.
#[derive(Debug, Clone)]
pub struct StandardHttpTransport {
    user_agent: String,
    timeout_ms: u64,
    headers: Vec<(String, String)>,
    agent: ureq::Agent,
}

impl Default for StandardHttpTransport {
    fn default() -> Self {
        let timeout_ms = 10_000;
        let user_agent =
            "StrataEarth/0.1 (SlippyTileStreaming; https://github.com/strataui)".to_string();
        let agent = ureq::AgentBuilder::new()
            .timeout(std::time::Duration::from_millis(timeout_ms))
            .user_agent(&user_agent)
            .build();
        Self {
            user_agent,
            timeout_ms,
            headers: Vec::new(),
            agent,
        }
    }
}

impl StandardHttpTransport {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_user_agent(mut self, ua: impl Into<String>) -> Self {
        self.user_agent = ua.into();
        self.agent = ureq::AgentBuilder::new()
            .timeout(std::time::Duration::from_millis(self.timeout_ms))
            .user_agent(&self.user_agent)
            .build();
        self
    }

    pub fn with_timeout_ms(mut self, timeout_ms: u64) -> Self {
        self.timeout_ms = timeout_ms;
        self.agent = ureq::AgentBuilder::new()
            .timeout(std::time::Duration::from_millis(self.timeout_ms))
            .user_agent(&self.user_agent)
            .build();
        self
    }

    /// Adds a custom HTTP header sent on every tile request (e.g., `Authorization`, `x-api-key`, `Referer`).
    pub fn with_header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.push((name.into(), value.into()));
        self
    }

    /// Replaces or sets custom HTTP headers sent on every tile request.
    pub fn with_headers<K, V>(mut self, headers: impl IntoIterator<Item = (K, V)>) -> Self
    where
        K: Into<String>,
        V: Into<String>,
    {
        self.headers = headers
            .into_iter()
            .map(|(k, v)| (k.into(), v.into()))
            .collect();
        self
    }
}

impl HttpTransport for StandardHttpTransport {
    fn fetch(&self, url: &str) -> Result<Vec<u8>, TileError> {
        let mut req = self.agent.get(url).set(
            "Accept",
            "image/png,image/jpeg,image/webp,application/json,*/*",
        );
        for (name, val) in &self.headers {
            req = req.set(name, val);
        }
        let resp = req
            .call()
            .map_err(|e| TileError::Transport(format!("HTTP/HTTPS fetch failed for {url}: {e}")))?;

        use std::io::Read;
        let mut bytes = Vec::with_capacity(32 * 1024);
        resp.into_reader()
            .take(32 * 1024 * 1024)
            .read_to_end(&mut bytes)
            .map_err(|e| TileError::Transport(format!("HTTP read body failed for {url}: {e}")))?;
        Ok(bytes)
    }
}

/// Memory-backed mock HTTP transport for fast, deterministic unit testing.
#[allow(clippy::type_complexity)]
#[derive(Debug, Clone, Default)]
pub struct MockHttpTransport {
    responses: Arc<Mutex<HashMap<String, Result<Vec<u8>, String>>>>,
    requests: Arc<Mutex<Vec<String>>>,
}

impl MockHttpTransport {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set_response(&self, url: impl Into<String>, bytes: Vec<u8>) {
        self.responses.lock().unwrap().insert(url.into(), Ok(bytes));
    }

    pub fn set_error(&self, url: impl Into<String>, error: impl Into<String>) {
        self.responses
            .lock()
            .unwrap()
            .insert(url.into(), Err(error.into()));
    }

    pub fn requested_urls(&self) -> Vec<String> {
        self.requests.lock().unwrap().clone()
    }
}

impl HttpTransport for MockHttpTransport {
    fn fetch(&self, url: &str) -> Result<Vec<u8>, TileError> {
        self.requests.lock().unwrap().push(url.to_string());
        let map = self.responses.lock().unwrap();
        match map.get(url) {
            Some(Ok(bytes)) => Ok(bytes.clone()),
            Some(Err(err)) => Err(TileError::Transport(err.clone())),
            None => Err(TileError::Transport(format!("404 Not Found: {url}"))),
        }
    }
}

/// Statistics for network tile streaming operations.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NetworkTileStats {
    pub network_requests: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub disk_cache_hits: u64,
    pub decode_errors: u64,
    pub cancelled_requests: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElevationEncoding {
    RawF32,
    TerrainRgb,
    Terrarium,
}

#[inline]
fn fnv1a_64_str(acc: &mut u64, s: &str) {
    for &b in s.as_bytes() {
        *acc ^= b as u64;
        *acc = acc.wrapping_mul(0x100_0000_01b3);
    }
    *acc ^= 0xff;
    *acc = acc.wrapping_mul(0x100_0000_01b3);
}

#[inline]
fn sanitize_slug(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len().min(32));
    for ch in raw.chars() {
        if out.len() >= 32 {
            break;
        }
        if ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' {
            out.push(ch.to_ascii_lowercase());
        } else if !out.ends_with('_') {
            out.push('_');
        }
    }
    let trimmed = out.trim_matches('_');
    if trimmed.is_empty() {
        "custom".to_string()
    } else {
        trimmed.to_string()
    }
}

/// Configured elevation DEM source paired with a [`TileProviderConfig`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TileElevationPreset {
    /// No elevation heightmap streaming.
    None,
    /// Public-domain AWS / Mapzen Terrarium elevation PNGs (`z=0..=15`).
    AwsTerrarium,
    /// Mapbox Terrain-RGB elevation tiles (`z=0..=15`, uses `{api_key}`).
    MapboxTerrainRgb,
    /// Custom XYZ / WMTS elevation endpoint.
    Custom {
        template: String,
        encoding: ElevationEncoding,
        max_level: u8,
    },
}

impl Default for TileElevationPreset {
    fn default() -> Self {
        Self::AwsTerrarium
    }
}

impl TileElevationPreset {
    fn resolve(&self) -> Option<(String, ElevationEncoding, u8)> {
        match self {
            Self::None => None,
            Self::AwsTerrarium => Some((
                "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
                    .to_string(),
                ElevationEncoding::Terrarium,
                15,
            )),
            Self::MapboxTerrainRgb => Some((
                "https://api.mapbox.com/v4/mapbox.terrain-rgb/{z}/{x}/{y}.pngraw?access_token={api_key}"
                    .to_string(),
                ElevationEncoding::TerrainRgb,
                15,
            )),
            Self::Custom {
                template,
                encoding,
                max_level,
            } => Some((template.clone(), *encoding, *max_level)),
        }
    }
}

/// Built-in world imagery & map tile API presets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum TileProviderPreset {
    /// Esri ArcGIS World Imagery satellite mosaic (`z=0..=18`).
    #[default]
    ArcGisWorldImagery,
    /// Esri ArcGIS World Topographic map (`z=0..=18`).
    ArcGisWorldTopo,
    /// Esri ArcGIS World Street map (`z=0..=18`).
    ArcGisWorldStreet,
    /// Standard OpenStreetMap Slippy XYZ tiles (`z=0..=19`).
    OpenStreetMap,
    /// OpenTopoMap topographic contour tiles (`z=0..=17`).
    OpenTopoMap,
    /// Carto Voyager light street/terrain style (`z=0..=19`).
    CartoVoyager,
    /// Carto Dark Matter dark basemap (`z=0..=19`).
    CartoDarkMatter,
    /// Google Maps Satellite imagery (`z=0..=19`).
    GoogleSatellite,
    /// Google Maps Hybrid satellite + roads (`z=0..=19`).
    GoogleHybrid,
    /// Google Maps standard roadmap (`z=0..=19`).
    GoogleRoadmap,
    /// Microsoft Bing Aerial Maps via `{quadkey}` (`z=0..=19`).
    BingAerial,
    /// Mapbox Satellite imagery (requires `api_key`, `z=0..=19`).
    MapboxSatellite,
    /// Mapbox Streets raster tiles (requires `api_key`, `z=0..=19`).
    MapboxStreets,
}

impl TileProviderPreset {
    /// Stable filesystem-safe identifier used in pack filenames and CLI flags.
    pub const fn id(self) -> &'static str {
        match self {
            Self::ArcGisWorldImagery => "arcgis_world_imagery",
            Self::ArcGisWorldTopo => "arcgis_world_topo",
            Self::ArcGisWorldStreet => "arcgis_world_street",
            Self::OpenStreetMap => "osm",
            Self::OpenTopoMap => "opentopomap",
            Self::CartoVoyager => "carto_voyager",
            Self::CartoDarkMatter => "carto_dark",
            Self::GoogleSatellite => "google_satellite",
            Self::GoogleHybrid => "google_hybrid",
            Self::GoogleRoadmap => "google_roadmap",
            Self::BingAerial => "bing_aerial",
            Self::MapboxSatellite => "mapbox_satellite",
            Self::MapboxStreets => "mapbox_streets",
        }
    }

    /// Human-readable short display name for UI switchers.
    pub const fn display_name(self) -> &'static str {
        match self {
            Self::ArcGisWorldImagery => "Satellite",
            Self::ArcGisWorldTopo => "Esri Topo",
            Self::ArcGisWorldStreet => "Esri Street",
            Self::OpenStreetMap => "OSM Street",
            Self::OpenTopoMap => "OpenTopo",
            Self::CartoVoyager => "Voyager",
            Self::CartoDarkMatter => "Dark Map",
            Self::GoogleSatellite => "Google Sat",
            Self::GoogleHybrid => "Google Hybrid",
            Self::GoogleRoadmap => "Google Road",
            Self::BingAerial => "Bing Aerial",
            Self::MapboxSatellite => "Mapbox Sat",
            Self::MapboxStreets => "Mapbox Road",
        }
    }

    /// Standard pre-baked `z=0..=4` pack filename matching this preset.
    pub fn pack_filename(self) -> String {
        format!("base_world_map_{}.pack", self.id())
    }

    /// Parses a preset from CLI / config string aliases (case-insensitive).
    pub fn from_id(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "arcgis_world_imagery" | "arcgis" | "satellite" | "esri_imagery" | "world_imagery" => {
                Some(Self::ArcGisWorldImagery)
            }
            "arcgis_world_topo" | "esri_topo" | "topo" | "world_topo" => Some(Self::ArcGisWorldTopo),
            "arcgis_world_street" | "esri_street" | "world_street" => {
                Some(Self::ArcGisWorldStreet)
            }
            "osm" | "openstreetmap" | "street" => Some(Self::OpenStreetMap),
            "opentopomap" | "otm" => Some(Self::OpenTopoMap),
            "carto_voyager" | "voyager" => Some(Self::CartoVoyager),
            "carto_dark" | "dark" | "dark_matter" => Some(Self::CartoDarkMatter),
            "google_satellite" | "google_sat" | "google" => Some(Self::GoogleSatellite),
            "google_hybrid" => Some(Self::GoogleHybrid),
            "google_roadmap" | "google_road" => Some(Self::GoogleRoadmap),
            "bing_aerial" | "bing" => Some(Self::BingAerial),
            "mapbox_satellite" | "mapbox" => Some(Self::MapboxSatellite),
            "mapbox_streets" | "mapbox_road" => Some(Self::MapboxStreets),
            _ => None,
        }
    }

    /// Ordered list of public keyless presets usable without an external API token.
    pub const fn public_presets() -> &'static [Self] {
        &[
            Self::ArcGisWorldImagery,
            Self::OpenStreetMap,
            Self::ArcGisWorldTopo,
            Self::ArcGisWorldStreet,
            Self::OpenTopoMap,
            Self::CartoVoyager,
            Self::CartoDarkMatter,
            Self::GoogleSatellite,
            Self::GoogleHybrid,
            Self::GoogleRoadmap,
            Self::BingAerial,
        ]
    }

    /// Cycles to the next public keyless preset.
    pub fn next_public(self) -> Self {
        let list = Self::public_presets();
        let idx = list.iter().position(|&p| p == self).unwrap_or(0);
        list[(idx + 1) % list.len()]
    }

    /// Converts this preset into a full [`TileProviderConfig`].
    pub fn to_config(self) -> TileProviderConfig {
        TileProviderConfig::preset(self)
    }
}

/// Complete configuration for an imagery/elevation/night tile API endpoint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TileProviderConfig {
    pub id: String,
    pub display_name: String,
    pub imagery_url_template: String,
    pub subdomains: Vec<String>,
    pub max_imagery_level: u8,
    pub api_key: Option<String>,
    pub headers: Vec<(String, String)>,
    pub elevation: TileElevationPreset,
    pub night_lights_url_template: Option<String>,
    pub night_lights_max_level: u8,
    pub base_pack_id: Option<String>,
}

impl Default for TileProviderConfig {
    fn default() -> Self {
        Self::preset(TileProviderPreset::ArcGisWorldImagery)
    }
}

impl TileProviderConfig {
    /// Builds a configuration from a built-in [`TileProviderPreset`].
    pub fn preset(preset: TileProviderPreset) -> Self {
        let (tpl, subs, max_z, night) = match preset {
            TileProviderPreset::ArcGisWorldImagery => (
                "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                &[][..],
                18,
                Some(DEFAULT_NASA_BLACK_MARBLE_URL.to_string()),
            ),
            TileProviderPreset::ArcGisWorldTopo => (
                "https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
                &[][..],
                18,
                None,
            ),
            TileProviderPreset::ArcGisWorldStreet => (
                "https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
                &[][..],
                18,
                None,
            ),
            TileProviderPreset::OpenStreetMap => (
                "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
                &[][..],
                19,
                None,
            ),
            TileProviderPreset::OpenTopoMap => (
                "https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
                &["a", "b", "c"][..],
                17,
                None,
            ),
            TileProviderPreset::CartoVoyager => (
                "https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png",
                &["a", "b", "c", "d"][..],
                19,
                None,
            ),
            TileProviderPreset::CartoDarkMatter => (
                "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png",
                &["a", "b", "c", "d"][..],
                19,
                None,
            ),
            TileProviderPreset::GoogleSatellite => (
                "https://mt{s}.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
                &["0", "1", "2", "3"][..],
                19,
                Some(DEFAULT_NASA_BLACK_MARBLE_URL.to_string()),
            ),
            TileProviderPreset::GoogleHybrid => (
                "https://mt{s}.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
                &["0", "1", "2", "3"][..],
                19,
                None,
            ),
            TileProviderPreset::GoogleRoadmap => (
                "https://mt{s}.google.com/vt/lyrs=m&x={x}&y={y}&z={z}",
                &["0", "1", "2", "3"][..],
                19,
                None,
            ),
            TileProviderPreset::BingAerial => (
                "https://ecn.t{s}.tiles.virtualearth.net/tiles/a{quadkey}.jpeg?g=1",
                &["0", "1", "2", "3"][..],
                19,
                Some(DEFAULT_NASA_BLACK_MARBLE_URL.to_string()),
            ),
            TileProviderPreset::MapboxSatellite => (
                "https://api.mapbox.com/v4/mapbox.satellite/{z}/{x}/{y}@2x.png?access_token={api_key}",
                &[][..],
                19,
                Some(DEFAULT_NASA_BLACK_MARBLE_URL.to_string()),
            ),
            TileProviderPreset::MapboxStreets => (
                "https://api.mapbox.com/styles/v1/mapbox/streets-v12/tiles/{z}/{x}/{y}?access_token={api_key}",
                &[][..],
                19,
                None,
            ),
        };
        Self {
            id: preset.id().to_string(),
            display_name: preset.display_name().to_string(),
            imagery_url_template: tpl.to_string(),
            subdomains: subs.iter().map(|s| (*s).to_string()).collect(),
            max_imagery_level: max_z,
            api_key: None,
            headers: Vec::new(),
            elevation: TileElevationPreset::AwsTerrarium,
            night_lights_url_template: night,
            night_lights_max_level: 8,
            base_pack_id: Some(preset.id().to_string()),
        }
    }

    /// Creates a custom provider configuration targeting any Slippy XYZ, TMS (`{-y}`),
    /// WMTS, or QuadKey (`{quadkey}`) URL template.
    pub fn custom(id: impl Into<String>, imagery_url_template: impl Into<String>) -> Self {
        let id_str = id.into();
        let slug = sanitize_slug(&id_str);
        Self {
            id: slug,
            display_name: id_str,
            imagery_url_template: imagery_url_template.into(),
            subdomains: vec!["a".to_string(), "b".to_string(), "c".to_string()],
            max_imagery_level: 19,
            api_key: None,
            headers: Vec::new(),
            elevation: TileElevationPreset::None,
            night_lights_url_template: None,
            night_lights_max_level: 8,
            base_pack_id: None,
        }
    }

    pub fn with_display_name(mut self, name: impl Into<String>) -> Self {
        self.display_name = name.into();
        self
    }

    pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    pub fn with_header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.push((name.into(), value.into()));
        self
    }

    pub fn with_headers<K, V>(mut self, headers: impl IntoIterator<Item = (K, V)>) -> Self
    where
        K: Into<String>,
        V: Into<String>,
    {
        self.headers = headers
            .into_iter()
            .map(|(k, v)| (k.into(), v.into()))
            .collect();
        self
    }

    pub fn with_subdomains(mut self, subdomains: &[&str]) -> Self {
        self.subdomains = subdomains.iter().map(|s| (*s).to_string()).collect();
        self
    }

    pub fn with_max_imagery_level(mut self, max_level: u8) -> Self {
        self.max_imagery_level = max_level;
        self
    }

    pub fn with_elevation(mut self, elevation: TileElevationPreset) -> Self {
        self.elevation = elevation;
        self
    }

    pub fn with_terrarium_elevation(mut self) -> Self {
        self.elevation = TileElevationPreset::AwsTerrarium;
        self
    }

    pub fn with_mapbox_elevation(mut self, token: impl Into<String>) -> Self {
        self.api_key = Some(token.into());
        self.elevation = TileElevationPreset::MapboxTerrainRgb;
        self
    }

    pub fn with_nasa_black_marble(mut self) -> Self {
        self.night_lights_url_template = Some(DEFAULT_NASA_BLACK_MARBLE_URL.to_string());
        self.night_lights_max_level = 8;
        self
    }

    pub fn with_base_pack_id(mut self, pack_id: Option<impl Into<String>>) -> Self {
        self.base_pack_id = pack_id.map(Into::into);
        self
    }

    /// Computes a collision-free cache namespace directory name keyed by provider ID + template + API key hash.
    pub fn imagery_cache_namespace(&self) -> String {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        fnv1a_64_str(&mut h, &self.id);
        fnv1a_64_str(&mut h, &self.imagery_url_template);
        if let Some(ref key) = self.api_key {
            fnv1a_64_str(&mut h, key);
        }
        format!("{}_{:016x}", sanitize_slug(&self.id), h)
    }

    /// Computes a stable cache namespace for the active elevation source, or `None` when disabled.
    pub fn elevation_cache_namespace(&self) -> Option<String> {
        let (tpl, enc, max_z) = self.elevation.resolve()?;
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        fnv1a_64_str(&mut h, &tpl);
        let enc_tag = match enc {
            ElevationEncoding::RawF32 => "raw",
            ElevationEncoding::TerrainRgb => "trgb",
            ElevationEncoding::Terrarium => "terrarium",
        };
        fnv1a_64_str(&mut h, enc_tag);
        h ^= max_z as u64;
        if let Some(ref key) = self.api_key {
            fnv1a_64_str(&mut h, key);
        }
        Some(format!("{enc_tag}_{h:016x}"))
    }

    /// Computes a stable cache namespace for the night lights layer.
    pub fn night_cache_namespace(&self) -> Option<String> {
        let tpl = self.night_lights_url_template.as_deref()?;
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        fnv1a_64_str(&mut h, tpl);
        Some(format!("night_{h:016x}"))
    }
}

struct FetchTask {
    key: TileDataKey,
    tile: TileKey,
    kind_weight: f64,
    generation: u64,
    provider_gen: u64,
    url: String,
    is_elevation: bool,
    elevation_encoding: ElevationEncoding,
    disk_cache_path: Option<PathBuf>,
}

struct DynamicFetchQueue {
    tasks: HashMap<TileDataKey, FetchTask>,
    in_flight: HashSet<(TileDataKey, u64)>,
    not_found_keys: HashSet<(TileDataKey, u64)>,
    generation: u64,
    imagery_provider_gen: u64,
    elev_provider_gen: u64,
    focus_lat: f64,
    focus_lon: f64,
    focus_alt_m: f64,
    shutdown: bool,
}

impl DynamicFetchQueue {
    fn new() -> Self {
        Self {
            tasks: HashMap::new(),
            in_flight: HashSet::new(),
            not_found_keys: HashSet::new(),
            generation: 1,
            imagery_provider_gen: 1,
            elev_provider_gen: 1,
            focus_lat: 0.0,
            focus_lon: 0.0,
            focus_alt_m: 1_000_000.0,
            shutdown: false,
        }
    }

    fn task_score(&self, task: &FetchTask) -> f64 {
        let gen_score = if task.generation == self.generation {
            1_000_000.0
        } else {
            -100_000.0 * ((self.generation.saturating_sub(task.generation)) as f64).min(10.0)
        };
        let b = tile_bounds_mercator(task.tile);
        let mid_lat = (b.lat0 + b.lat1) * 0.5;
        let mid_lon = (b.lon0 + b.lon1) * 0.5;
        let dlat = (mid_lat - self.focus_lat).abs();
        let mut dlon = (mid_lon - self.focus_lon).abs();
        if dlon > 180.0 {
            dlon = 360.0 - dlon;
        }
        let lon_scale = self.focus_lat.to_radians().cos().abs().max(0.2);
        let ang_dist = dlat.hypot(dlon * lon_scale);
        let view_deg = (self.focus_alt_m.max(100.0) / 6_371_000.0)
            .to_degrees()
            .max(0.02);
        let norm_dist = (ang_dist / view_deg).min(24.0);
        let zoom_bias = -(task.tile.level as f64) * 3_000.0;
        gen_score + task.kind_weight + zoom_bias - norm_dist * 400.0
    }

    fn pop_best(&mut self) -> Option<FetchTask> {
        let mut best_key: Option<TileDataKey> = None;
        let mut best_score = f64::NEG_INFINITY;
        for (k, t) in &self.tasks {
            let s = self.task_score(t);
            if s > best_score {
                best_score = s;
                best_key = Some(*k);
            }
        }
        let k = best_key?;
        let task = self.tasks.remove(&k)?;
        self.in_flight.insert((k, task.provider_gen));
        Some(task)
    }
}

type TileReadyCallback = Arc<dyn Fn() + Send + Sync>;
type TileReadySlot = Arc<Mutex<Option<TileReadyCallback>>>;
type WorkerResultMsg = (TileDataKey, u64, Result<CachedTileData, TileError>);

pub const DEFAULT_NASA_BLACK_MARBLE_URL: &str =
    "https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/VIIRS_CityLights_2012/default/GoogleMapsCompatible_Level8/{z}/{y}/{x}.jpg";

/// Non-blocking asynchronous network tile streaming source supporting Slippy XYZ, TMS (`{-y}`),
/// WMTS, and Bing QuadKey (`{quadkey}`) protocols with provider-keyed LRU/disk caches.
pub struct NetworkTileSource {
    provider: TileProviderConfig,
    imagery_cache_ns: String,
    elev_cache_ns: Option<String>,
    night_cache_ns: Option<String>,
    elevation_url_template: Option<String>,
    elevation_max_level: u8,
    elevation_encoding: ElevationEncoding,
    subdomain_counter: usize,
    cache: LruTileCache,
    disk_cache_dir: Option<PathBuf>,
    transport: Arc<dyn HttpTransport>,
    uses_default_transport: bool,
    queue: Arc<(Mutex<DynamicFetchQueue>, Condvar)>,
    results_tx: std::sync::mpsc::Sender<WorkerResultMsg>,
    results_rx: Receiver<WorkerResultMsg>,
    on_tile_ready: TileReadySlot,
    stats: NetworkTileStats,
    default_tile_size: u32,
}

impl NetworkTileSource {
    /// Creates a new network tile source targeting the specified imagery URL template.
    pub fn new(imagery_url_template: impl Into<String>) -> Self {
        let tpl: String = imagery_url_template.into();
        let cfg = TileProviderConfig::custom("custom", tpl);
        Self::from_provider(cfg)
    }

    /// Creates a [`NetworkTileSource`] from a [`TileProviderConfig`], applying templates,
    /// subdomains, API keys, custom HTTP headers, and elevation configuration.
    pub fn from_provider(provider: TileProviderConfig) -> Self {
        let mut transport = StandardHttpTransport::new();
        if !provider.headers.is_empty() {
            transport = transport.with_headers(provider.headers.clone());
        }
        Self::from_provider_with_transport(
            provider,
            DEFAULT_MAX_CACHE_BYTES,
            None,
            Arc::new(transport),
            true,
        )
    }

    /// Creates a [`NetworkTileSource`] from a built-in [`TileProviderPreset`].
    pub fn from_preset(preset: TileProviderPreset) -> Self {
        Self::from_provider(TileProviderConfig::preset(preset))
    }

    fn from_provider_with_transport(
        provider: TileProviderConfig,
        max_cache_bytes: usize,
        disk_cache_dir: Option<PathBuf>,
        transport: Arc<dyn HttpTransport>,
        uses_default_transport: bool,
    ) -> Self {
        let queue = Arc::new((Mutex::new(DynamicFetchQueue::new()), Condvar::new()));
        let (results_tx, results_rx) = channel::<WorkerResultMsg>();
        let on_tile_ready: TileReadySlot = Arc::new(Mutex::new(None));

        Self::spawn_fetch_workers(
            8,
            Arc::clone(&queue),
            results_tx.clone(),
            Arc::clone(&transport),
            Arc::clone(&on_tile_ready),
        );

        let imagery_cache_ns = provider.imagery_cache_namespace();
        let elev_cache_ns = provider.elevation_cache_namespace();
        let night_cache_ns = provider.night_cache_namespace();
        let (elevation_url_template, elevation_encoding, elevation_max_level) =
            match provider.elevation.resolve() {
                Some((t, e, z)) => (Some(t), e, z),
                None => (None, ElevationEncoding::RawF32, 20),
            };

        Self {
            provider,
            imagery_cache_ns,
            elev_cache_ns,
            night_cache_ns,
            elevation_url_template,
            elevation_max_level,
            elevation_encoding,
            subdomain_counter: 0,
            cache: LruTileCache::new(max_cache_bytes),
            disk_cache_dir,
            transport,
            uses_default_transport,
            queue,
            results_tx,
            results_rx,
            on_tile_ready,
            stats: NetworkTileStats::default(),
            default_tile_size: 256,
        }
    }

    /// Convenience constructor targeting standard OpenStreetMap tiles.
    pub fn osm() -> Self {
        Self::from_preset(TileProviderPreset::OpenStreetMap).with_elevation_preset(TileElevationPreset::None)
    }

    /// Convenience constructor targeting open satellite world imagery (ArcGIS World Imagery)
    /// paired with global AWS Terrarium DEM heightmaps and streamed NASA Black Marble (`z=0..=8`).
    pub fn world_satellite_with_terrain() -> Self {
        Self::from_preset(TileProviderPreset::ArcGisWorldImagery)
    }

    /// Convenience constructor targeting Mapbox satellite imagery and Terrain-RGB elevation.
    pub fn mapbox(access_token: &str) -> Self {
        Self::from_provider(
            TileProviderConfig::preset(TileProviderPreset::MapboxSatellite)
                .with_mapbox_elevation(access_token)
                .with_nasa_black_marble(),
        )
    }

    fn spawn_fetch_workers(
        worker_count: usize,
        queue: Arc<(Mutex<DynamicFetchQueue>, Condvar)>,
        results_tx: std::sync::mpsc::Sender<WorkerResultMsg>,
        transport: Arc<dyn HttpTransport>,
        on_tile_ready: TileReadySlot,
    ) {
        for idx in 0..worker_count.max(1) {
            let q = Arc::clone(&queue);
            let tx = results_tx.clone();
            let worker_transport = Arc::clone(&transport);
            let worker_ready = Arc::clone(&on_tile_ready);
            let _ = std::thread::Builder::new()
                .name(format!("strata-tile-fetcher-{idx}"))
                .spawn(move || loop {
                    let task = {
                        let (lock, cv) = &*q;
                        let mut guard = match lock.lock() {
                            Ok(g) => g,
                            Err(_) => break,
                        };
                        loop {
                            if guard.shutdown {
                                return;
                            }
                            if let Some(t) = guard.pop_best() {
                                break t;
                            }
                            guard = match cv.wait(guard) {
                                Ok(g) => g,
                                Err(_) => return,
                            };
                        }
                    };
                    let result = Self::process_fetch_task(&worker_transport, &task);
                    let ok = result.is_ok();
                    let is_permanent_miss = matches!(
                        &result,
                        Err(TileError::Transport(msg))
                            if msg.contains("404") || msg.contains("400") || msg.contains("410")
                    );
                    if let Ok(mut guard) = q.0.lock() {
                        guard.in_flight.remove(&(task.key, task.provider_gen));
                        if is_permanent_miss {
                            if guard.not_found_keys.len() > 2048 {
                                guard.not_found_keys.clear();
                            }
                            guard.not_found_keys.insert((task.key, task.provider_gen));
                        }
                    }
                    let _ = tx.send((task.key, task.provider_gen, result));
                    if ok || is_permanent_miss {
                        if let Ok(guard) = worker_ready.lock() {
                            if let Some(ref cb) = *guard {
                                cb();
                            }
                        }
                    }
                });
        }
    }

    fn respawn_workers(&mut self, transport: Arc<dyn HttpTransport>) {
        if let Ok(mut guard) = self.queue.0.lock() {
            guard.shutdown = true;
            self.queue.1.notify_all();
        }
        let queue = Arc::new((Mutex::new(DynamicFetchQueue::new()), Condvar::new()));
        let (results_tx, results_rx) = channel::<WorkerResultMsg>();
        Self::spawn_fetch_workers(
            8,
            Arc::clone(&queue),
            results_tx.clone(),
            Arc::clone(&transport),
            Arc::clone(&self.on_tile_ready),
        );
        self.transport = transport;
        self.queue = queue;
        self.results_tx = results_tx;
        self.results_rx = results_rx;
    }

    /// Fully customizable builder constructor.
    pub fn with_options(
        imagery_url_template: impl Into<String>,
        elevation_url_template: Option<String>,
        elevation_is_terrain_rgb: bool,
        max_cache_bytes: usize,
        disk_cache_dir: Option<PathBuf>,
        transport: Arc<dyn HttpTransport>,
    ) -> Self {
        let elev = match elevation_url_template {
            Some(template) => TileElevationPreset::Custom {
                template,
                encoding: if elevation_is_terrain_rgb {
                    ElevationEncoding::TerrainRgb
                } else {
                    ElevationEncoding::RawF32
                },
                max_level: 20,
            },
            None => TileElevationPreset::None,
        };
        let mut cfg = TileProviderConfig::custom("custom", imagery_url_template);
        cfg.elevation = elev;
        Self::from_provider_with_transport(cfg, max_cache_bytes, disk_cache_dir, transport, false)
    }

    /// Returns the active [`TileProviderConfig`].
    pub fn provider_config(&self) -> &TileProviderConfig {
        &self.provider
    }

    /// Returns the active disk-cache namespace derived from the current imagery provider URL/key.
    pub fn imagery_cache_namespace(&self) -> &str {
        &self.imagery_cache_ns
    }

    /// Switches the active tile provider at runtime, clearing stale imagery/night caches and
    /// isolating disk storage per provider hash while preserving elevation caches when unchanged.
    pub fn apply_provider(&mut self, config: &TileProviderConfig) {
        let imagery_changed = self.provider.imagery_url_template != config.imagery_url_template
            || self.provider.api_key != config.api_key
            || self.provider.subdomains != config.subdomains
            || self.provider.headers != config.headers
            || self.provider.night_lights_url_template != config.night_lights_url_template
            || self.provider.id != config.id;
        let elev_changed = self.provider.elevation != config.elevation
            || self.provider.api_key != config.api_key
            || self.provider.headers != config.headers;

        self.provider = config.clone();
        self.imagery_cache_ns = config.imagery_cache_namespace();
        self.elev_cache_ns = config.elevation_cache_namespace();
        self.night_cache_ns = config.night_cache_namespace();

        let (el_tpl, el_enc, el_max) = match config.elevation.resolve() {
            Some((t, e, z)) => (Some(t), e, z),
            None => (None, ElevationEncoding::RawF32, 20),
        };
        self.elevation_url_template = el_tpl;
        self.elevation_encoding = el_enc;
        self.elevation_max_level = el_max;

        if imagery_changed {
            self.cache.clear_imagery();
            self.cache.clear_night_lights();
        }
        if elev_changed {
            self.cache.clear_elevation();
        }

        if let Ok(mut guard) = self.queue.0.lock() {
            if imagery_changed {
                guard.imagery_provider_gen = guard.imagery_provider_gen.wrapping_add(1).max(1);
                guard
                    .tasks
                    .retain(|k, _| matches!(k, TileDataKey::Elevation(_)));
                guard
                    .not_found_keys
                    .retain(|(k, _)| matches!(k, TileDataKey::Elevation(_)));
            }
            if elev_changed {
                guard.elev_provider_gen = guard.elev_provider_gen.wrapping_add(1).max(1);
                guard
                    .tasks
                    .retain(|k, _| !matches!(k, TileDataKey::Elevation(_)));
                guard
                    .not_found_keys
                    .retain(|(k, _)| !matches!(k, TileDataKey::Elevation(_)));
            }
        }

        if !config.headers.is_empty() && self.uses_default_transport {
            let t = Arc::new(StandardHttpTransport::new().with_headers(config.headers.clone()));
            self.respawn_workers(t);
        } else if self.uses_default_transport && imagery_changed {
            let t = Arc::new(StandardHttpTransport::new());
            self.respawn_workers(t);
        }
        self.poll_completed();
    }

    /// Registers a wakeup callback invoked whenever a downloaded tile finishes decoding.
    pub fn set_on_tile_ready(&mut self, cb: Option<Arc<dyn Fn() + Send + Sync>>) {
        if let Ok(mut guard) = self.on_tile_ready.lock() {
            *guard = cb;
        }
    }

    /// Replaces the active HTTP transport (useful for mock injection in tests).
    pub fn with_transport(mut self, transport: impl HttpTransport) -> Self {
        self.uses_default_transport = false;
        let t: Arc<dyn HttpTransport> = Arc::new(transport);
        self.respawn_workers(t);
        self
    }

    /// Sets the API key / token used in URL template substitutions (`{api_key}`, `{access_token}`).
    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.provider.api_key = Some(api_key.into());
        self.imagery_cache_ns = self.provider.imagery_cache_namespace();
        self.elev_cache_ns = self.provider.elevation_cache_namespace();
        self
    }

    /// Adds a custom HTTP header on outgoing requests (uses `StandardHttpTransport`).
    pub fn with_header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.provider.headers.push((name.into(), value.into()));
        if self.uses_default_transport {
            let t = Arc::new(StandardHttpTransport::new().with_headers(self.provider.headers.clone()));
            self.respawn_workers(t);
        }
        self
    }

    /// Sets the on-disk cache directory (automatically namespaced per provider template/key hash).
    pub fn with_disk_cache(mut self, path: impl Into<PathBuf>) -> Self {
        self.disk_cache_dir = Some(path.into());
        self
    }

    /// Configures subdomains for load balancing.
    pub fn with_subdomains(mut self, subdomains: &[&str]) -> Self {
        self.provider.subdomains = subdomains.iter().map(|s| s.to_string()).collect();
        self
    }

    /// Sets the elevation preset cleanly.
    pub fn with_elevation_preset(mut self, preset: TileElevationPreset) -> Self {
        self.provider.elevation = preset;
        let (el_tpl, el_enc, el_max) = match self.provider.elevation.resolve() {
            Some((t, e, z)) => (Some(t), e, z),
            None => (None, ElevationEncoding::RawF32, 20),
        };
        self.elevation_url_template = el_tpl;
        self.elevation_encoding = el_enc;
        self.elevation_max_level = el_max;
        self.elev_cache_ns = self.provider.elevation_cache_namespace();
        self
    }

    /// Sets the elevation URL template.
    pub fn with_elevation(self, template: impl Into<String>, is_terrain_rgb: bool) -> Self {
        let tpl = template.into();
        let enc = if is_terrain_rgb {
            ElevationEncoding::TerrainRgb
        } else {
            ElevationEncoding::RawF32
        };
        self.with_elevation_preset(TileElevationPreset::Custom {
            template: tpl,
            encoding: enc,
            max_level: 20,
        })
    }

    /// Enables public-domain global AWS Terrarium DEM heightmaps (`z=0..15`).
    pub fn with_terrarium_elevation(self) -> Self {
        self.with_elevation_preset(TileElevationPreset::AwsTerrarium)
    }

    /// Configures a streamed multi-resolution night city-lights WMTS/XYZ layer up to `max_level`.
    pub fn with_night_lights(mut self, template: impl Into<String>, max_level: u8) -> Self {
        self.provider.night_lights_url_template = Some(template.into());
        self.provider.night_lights_max_level = max_level;
        self.night_cache_ns = self.provider.night_cache_namespace();
        self
    }

    /// Enables public-domain NASA GIBS VIIRS Black Marble streaming (`z=0..=8`, ~611 m/texel).
    pub fn with_nasa_black_marble(self) -> Self {
        self.with_night_lights(DEFAULT_NASA_BLACK_MARBLE_URL, 8)
    }

    /// Inspects the current network tile statistics.
    pub fn stats(&self) -> NetworkTileStats {
        self.stats
    }

    /// Returns the number of queued tile fetch tasks waiting to start.
    pub fn pending_queue_len(&self) -> usize {
        self.queue.0.lock().map(|g| g.tasks.len()).unwrap_or(0)
    }

    /// Accesses the underlying in-memory LRU cache.
    pub fn cache(&self) -> &LruTileCache {
        &self.cache
    }

    #[inline]
    fn imagery_disk_path(&self, key: TileKey) -> Option<PathBuf> {
        self.disk_cache_dir.as_ref().map(|d| {
            d.join("providers")
                .join(&self.imagery_cache_ns)
                .join(format!("{}/{}/{}.png", key.level, key.x, key.y))
        })
    }

    #[inline]
    fn night_disk_path(&self, key: TileKey) -> Option<PathBuf> {
        let ns = self.night_cache_ns.as_deref()?;
        self.disk_cache_dir.as_ref().map(|d| {
            d.join("night")
                .join(ns)
                .join(format!("{}/{}/{}.jpg", key.level, key.x, key.y))
        })
    }

    #[inline]
    fn elevation_disk_path(&self, key: TileKey) -> Option<PathBuf> {
        let ns = self.elev_cache_ns.as_deref()?;
        self.disk_cache_dir.as_ref().map(|d| {
            d.join("elev")
                .join(ns)
                .join(format!("{}/{}/{}.elev", key.level, key.x, key.y))
        })
    }

    /// Drains completed background downloads into the in-memory LRU cache, discarding
    /// stale responses from superseded providers.
    pub fn poll_completed(&mut self) {
        let (exp_img_gen, exp_elev_gen) = self
            .queue
            .0
            .lock()
            .map(|g| (g.imagery_provider_gen, g.elev_provider_gen))
            .unwrap_or((1, 1));
        while let Ok((key, pgen, result)) = self.results_rx.try_recv() {
            let expected = match key {
                TileDataKey::Elevation(_) => exp_elev_gen,
                TileDataKey::Imagery(_) | TileDataKey::NightLights(_) => exp_img_gen,
            };
            if pgen != expected {
                continue;
            }
            match result {
                Ok(data) => match data {
                    CachedTileData::Imagery(img) => match key {
                        TileDataKey::Imagery(k) => self.cache.insert_imagery(k, img),
                        TileDataKey::NightLights(k) => self.cache.insert_night_lights(k, img),
                        TileDataKey::Elevation(_) => {}
                    },
                    CachedTileData::Elevation(elev) => {
                        if let TileDataKey::Elevation(k) = key {
                            self.cache.insert_elevation(k, elev);
                        }
                    }
                },
                Err(e) => {
                    let is_placeholder_or_404 = matches!(
                        &e,
                        TileError::Transport(msg)
                            if msg.contains("404") || msg.contains("400") || msg.contains("410")
                    );
                    if !is_placeholder_or_404 {
                        self.stats.decode_errors += 1;
                    }
                }
            }
        }
    }

    /// Synchronously fetches and caches an imagery tile, blocking until complete.
    pub fn fetch_sync(&mut self, key: TileKey) -> Result<Option<TileImage>, TileError> {
        let url = format_tile_url(
            &self.provider.imagery_url_template,
            key,
            &self.provider.subdomains,
            &mut self.subdomain_counter,
            self.provider.api_key.as_deref(),
        );
        let bytes = self.transport.fetch(&url)?;
        let image = match Self::decode_imagery_bytes(&bytes, self.default_tile_size) {
            Ok(img) => img,
            Err(TileError::Transport(ref msg)) if msg.contains("unavailable placeholder") => {
                if let Ok(mut guard) = self.queue.0.lock() {
                    let pgen = guard.imagery_provider_gen;
                    guard
                        .not_found_keys
                        .insert((TileDataKey::Imagery(key), pgen));
                }
                return Ok(None);
            }
            Err(e) => return Err(e),
        };
        if let Some(path) = self.imagery_disk_path(key) {
            if let Some(parent) = path.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            let _ = std::fs::write(path, &bytes);
        }
        self.cache.insert_imagery(key, image.clone());
        Ok(Some(image))
    }

    /// Synchronously fetches and caches an elevation tile, blocking until complete.
    pub fn fetch_elevation_sync(
        &mut self,
        key: TileKey,
    ) -> Result<Option<TileElevation>, TileError> {
        let template = match &self.elevation_url_template {
            Some(t) => t.clone(),
            None => return Ok(None),
        };
        let url = format_tile_url(
            &template,
            key,
            &self.provider.subdomains,
            &mut self.subdomain_counter,
            self.provider.api_key.as_deref(),
        );
        let bytes = self.transport.fetch(&url)?;
        let elev = Self::decode_elevation_bytes(&bytes, self.elevation_encoding)?;
        if let Some(path) = self.elevation_disk_path(key) {
            if let Some(parent) = path.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            let _ = std::fs::write(path, &bytes);
        }
        self.cache.insert_elevation(key, elev.clone());
        Ok(Some(elev))
    }

    fn process_fetch_task(
        transport: &Arc<dyn HttpTransport>,
        task: &FetchTask,
    ) -> Result<CachedTileData, TileError> {
        let bytes = transport.fetch(&task.url)?;
        let decoded = if task.is_elevation {
            let elev = Self::decode_elevation_bytes(&bytes, task.elevation_encoding)?;
            CachedTileData::Elevation(elev)
        } else {
            let img = Self::decode_imagery_bytes(&bytes, 256)?;
            CachedTileData::Imagery(img)
        };
        if let Some(path) = &task.disk_cache_path {
            if let Some(parent) = path.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            let _ = std::fs::write(path, &bytes);
        }
        Ok(decoded)
    }

    fn validate_decoded_imagery(img: TileImage) -> Result<TileImage, TileError> {
        if is_unavailable_placeholder_tile(img.width, img.height, &img.pixels) {
            return Err(TileError::Transport(
                "404 unavailable placeholder tile".to_string(),
            ));
        }
        Ok(img)
    }

    fn decode_imagery_bytes(bytes: &[u8], default_size: u32) -> Result<TileImage, TileError> {
        if bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
            if let Ok((w, h, rgba)) = decode_png_to_rgba(bytes) {
                return Self::validate_decoded_imagery(TileImage {
                    width: w,
                    height: h,
                    pixels: rgba,
                });
            }
        }
        if let Ok(dyn_img) = image::load_from_memory(bytes) {
            let rgba = dyn_img.to_rgba8();
            return Self::validate_decoded_imagery(TileImage {
                width: rgba.width(),
                height: rgba.height(),
                pixels: rgba.into_raw(),
            });
        }
        if bytes.len() >= 8 {
            let w = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
            let h = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
            if w > 0 && h > 0 && bytes.len() == 8 + (w as usize * h as usize * 4) {
                return Self::validate_decoded_imagery(TileImage {
                    width: w,
                    height: h,
                    pixels: bytes[8..].to_vec(),
                });
            }
        }
        if bytes.len() == (default_size as usize * default_size as usize * 4) {
            return Self::validate_decoded_imagery(TileImage {
                width: default_size,
                height: default_size,
                pixels: bytes.to_vec(),
            });
        }
        if bytes.len().is_multiple_of(4) {
            let pixels = bytes.len() / 4;
            let side = (pixels as f64).sqrt().round() as usize;
            if side * side == pixels {
                return Self::validate_decoded_imagery(TileImage {
                    width: side as u32,
                    height: side as u32,
                    pixels: bytes.to_vec(),
                });
            }
        }
        Err(TileError::Decode(format!(
            "unrecognized image data length {}",
            bytes.len()
        )))
    }

    fn decode_elevation_bytes(
        bytes: &[u8],
        encoding: ElevationEncoding,
    ) -> Result<TileElevation, TileError> {
        if bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
            let (w, h, rgba) = decode_png_to_rgba(bytes).or_else(|_| {
                image::load_from_memory(bytes)
                    .map(|img| {
                        let r = img.to_rgba8();
                        (r.width(), r.height(), r.into_raw())
                    })
                    .map_err(|e| TileError::Decode(format!("elevation png decode failed: {e}")))
            })?;
            return Ok(match encoding {
                ElevationEncoding::Terrarium => decode_terrarium(w, h, &rgba),
                ElevationEncoding::TerrainRgb | ElevationEncoding::RawF32 => {
                    decode_terrain_rgb(w, h, &rgba)
                }
            });
        }
        if bytes.len() >= 8 {
            let w = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
            let h = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
            if w > 0 && h > 0 && bytes.len() == 8 + (w as usize * h as usize * 4) {
                let heights_m: Vec<f32> = bytes[8..]
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();
                return Ok(TileElevation {
                    width: w,
                    height: h,
                    heights_m,
                });
            }
        }
        Err(TileError::Decode(format!(
            "unrecognized elevation data length {}",
            bytes.len()
        )))
    }

    fn enqueue_fetch(
        &mut self,
        data_key: TileDataKey,
        tile: TileKey,
        kind_weight: f64,
        template: &str,
        is_elevation: bool,
        disk_cache_path: Option<PathBuf>,
    ) {
        let (lock, cv) = &*self.queue;
        let Ok(mut guard) = lock.lock() else {
            return;
        };
        let gen = guard.generation;
        let pgen = if is_elevation {
            guard.elev_provider_gen
        } else {
            guard.imagery_provider_gen
        };
        if guard.in_flight.contains(&(data_key, pgen))
            || guard.not_found_keys.contains(&(data_key, pgen))
        {
            return;
        }
        if let Some(existing) = guard.tasks.get_mut(&data_key) {
            existing.generation = gen;
            existing.provider_gen = pgen;
            existing.kind_weight = kind_weight.max(existing.kind_weight);
            cv.notify_one();
            return;
        }
        let url = format_tile_url(
            template,
            tile,
            &self.provider.subdomains,
            &mut self.subdomain_counter,
            self.provider.api_key.as_deref(),
        );
        guard.tasks.insert(
            data_key,
            FetchTask {
                key: data_key,
                tile,
                kind_weight,
                generation: gen,
                provider_gen: pgen,
                url,
                is_elevation,
                elevation_encoding: self.elevation_encoding,
                disk_cache_path,
            },
        );
        cv.notify_one();
        self.stats.network_requests += 1;
    }

    fn is_key_not_found(&self, data_key: TileDataKey) -> bool {
        self.queue
            .0
            .lock()
            .map(|g| g.not_found_keys.contains(&(data_key, g.imagery_provider_gen)))
            .unwrap_or(false)
    }

    fn mark_key_not_found(&self, data_key: TileDataKey) {
        if let Ok(mut guard) = self.queue.0.lock() {
            let pgen = guard.imagery_provider_gen;
            if guard.not_found_keys.len() > 2048 {
                guard.not_found_keys.clear();
            }
            guard.not_found_keys.insert((data_key, pgen));
        }
    }

    fn warm_or_enqueue_available_ancestor(
        &mut self,
        leaf: TileKey,
        is_night: bool,
        template: &str,
        weight: f64,
    ) {
        let mut anc = leaf.parent();
        while let Some(parent_key) = anc {
            let already_cached = if is_night {
                self.cache.peek_night_lights(parent_key).is_some()
            } else {
                self.cache.peek_imagery(parent_key).is_some()
            };
            if already_cached {
                break;
            }
            let parent_data_key = if is_night {
                TileDataKey::NightLights(parent_key)
            } else {
                TileDataKey::Imagery(parent_key)
            };
            let parent_disk = if is_night {
                self.night_disk_path(parent_key)
            } else {
                self.imagery_disk_path(parent_key)
            };
            if let Some(ref path) = parent_disk {
                if path.is_file() {
                    if let Ok(bytes) = std::fs::read(path) {
                        match Self::decode_imagery_bytes(&bytes, self.default_tile_size) {
                            Ok(img) => {
                                if is_night {
                                    self.cache.insert_night_lights(parent_key, img);
                                } else {
                                    self.cache.insert_imagery(parent_key, img);
                                }
                                self.stats.disk_cache_hits += 1;
                                break;
                            }
                            Err(_) => {
                                let _ = std::fs::remove_file(path);
                                self.mark_key_not_found(parent_data_key);
                            }
                        }
                    }
                }
            }
            if self.is_key_not_found(parent_data_key) {
                anc = parent_key.parent();
                continue;
            }
            self.enqueue_fetch(
                parent_data_key,
                parent_key,
                weight,
                template,
                false,
                parent_disk,
            );
            break;
        }
    }

    fn fetch_image_layer(
        &mut self,
        key: TileKey,
        is_night: bool,
    ) -> Result<Option<TileImage>, TileError> {
        self.poll_completed();

        let cached = if is_night {
            self.cache.get_night_lights(key)
        } else {
            self.cache.get_imagery(key)
        };
        if let Some(img) = cached {
            self.stats.cache_hits += 1;
            return Ok(Some(img));
        }

        let (template, disk_path) = if is_night {
            let Some(ref tpl) = self.provider.night_lights_url_template else {
                return Ok(None);
            };
            if key.level > self.provider.night_lights_max_level {
                return Ok(None);
            }
            (tpl.clone(), self.night_disk_path(key))
        } else {
            if key.level > self.provider.max_imagery_level {
                return Ok(None);
            }
            (
                self.provider.imagery_url_template.clone(),
                self.imagery_disk_path(key),
            )
        };

        let data_key = if is_night {
            TileDataKey::NightLights(key)
        } else {
            TileDataKey::Imagery(key)
        };
        let weight = if is_night { 50_000.0 } else { 200_000.0 };

        if let Some(ref path) = disk_path {
            if path.is_file() {
                if let Ok(bytes) = std::fs::read(path) {
                    match Self::decode_imagery_bytes(&bytes, self.default_tile_size) {
                        Ok(img) => {
                            if is_night {
                                self.cache.insert_night_lights(key, img.clone());
                            } else {
                                self.cache.insert_imagery(key, img.clone());
                            }
                            self.stats.disk_cache_hits += 1;
                            return Ok(Some(img));
                        }
                        Err(_) => {
                            let _ = std::fs::remove_file(path);
                            self.mark_key_not_found(data_key);
                        }
                    }
                }
            }
        }

        if self.is_key_not_found(data_key) {
            self.warm_or_enqueue_available_ancestor(key, is_night, &template, weight);
            return Ok(None);
        }

        self.stats.cache_misses += 1;
        self.enqueue_fetch(data_key, key, weight, &template, false, disk_path);
        Ok(None)
    }
}

impl Drop for NetworkTileSource {
    fn drop(&mut self) {
        if let Ok(mut guard) = self.queue.0.lock() {
            guard.shutdown = true;
            self.queue.1.notify_all();
        }
    }
}

impl TileSource for NetworkTileSource {
    fn begin_frame(&mut self, focus: Geodetic, altitude_m: f64) {
        if let Ok(mut guard) = self.queue.0.lock() {
            guard.generation = guard.generation.wrapping_add(1).max(1);
            guard.focus_lat = focus.lat_deg;
            guard.focus_lon = focus.lon_deg;
            guard.focus_alt_m = altitude_m.max(100.0);
            if guard.tasks.len() > 2048 {
                let min_keep = guard.generation.saturating_sub(1);
                guard.tasks.retain(|_, t| t.generation >= min_keep);
            }
        }
    }

    fn end_frame(&mut self) {
        let cancelled = if let Ok(mut guard) = self.queue.0.lock() {
            let gen = guard.generation;
            let before = guard.tasks.len();
            guard.tasks.retain(|_, t| t.generation >= gen);
            before.saturating_sub(guard.tasks.len())
        } else {
            0
        };
        self.stats.cancelled_requests += cancelled as u64;
    }

    fn imagery(&mut self, key: TileKey) -> Result<Option<TileImage>, TileError> {
        self.fetch_image_layer(key, false)
    }

    fn peek_cached_imagery(&self, key: TileKey) -> Option<TileImage> {
        self.cache.peek_imagery(key)
    }

    fn night_lights(&mut self, key: TileKey) -> Result<Option<TileImage>, TileError> {
        self.fetch_image_layer(key, true)
    }

    fn peek_cached_night_lights(&self, key: TileKey) -> Option<TileImage> {
        self.cache.peek_night_lights(key)
    }

    fn elevation(&mut self, key: TileKey) -> Result<Option<TileElevation>, TileError> {
        self.poll_completed();

        let template = match &self.elevation_url_template {
            Some(t) => t.clone(),
            None => return Ok(None),
        };
        if key.level > self.elevation_max_level {
            return Ok(None);
        }

        if let Some(elev) = self.cache.get_elevation(key) {
            self.stats.cache_hits += 1;
            return Ok(Some(elev));
        }

        let disk_path = self.elevation_disk_path(key);
        if let Some(ref path) = disk_path {
            if path.is_file() {
                if let Ok(bytes) = std::fs::read(path) {
                    if let Ok(elev) = Self::decode_elevation_bytes(&bytes, self.elevation_encoding)
                    {
                        self.cache.insert_elevation(key, elev.clone());
                        self.stats.disk_cache_hits += 1;
                        return Ok(Some(elev));
                    }
                }
            }
        }

        self.stats.cache_misses += 1;
        let elev_weight = 100_000.0;
        self.enqueue_fetch(
            TileDataKey::Elevation(key),
            key,
            elev_weight,
            &template,
            true,
            disk_path,
        );
        Ok(None)
    }

    fn peek_cached_elevation(&self, key: TileKey) -> Option<TileElevation> {
        self.cache.peek_elevation(key)
    }

    fn set_provider(&mut self, config: &TileProviderConfig) -> bool {
        self.apply_provider(config);
        true
    }

    fn set_tile_ready_callback(&mut self, cb: Option<Arc<dyn Fn() + Send + Sync>>) {
        self.set_on_tile_ready(cb);
    }
}

/// Detects provider placeholder tiles such as ArcGIS `"Map data not yet available"` /
/// `"Map data not available"` responses (returned with HTTP 200 and a uniform neutral-grey
/// background with optional centered greyscale text) so they are treated as missing
/// tiles and cleanly fall back to lower-zoom ancestor imagery.
pub fn is_unavailable_placeholder_tile(width: u32, height: u32, rgba: &[u8]) -> bool {
    let w = width as usize;
    let h = height as usize;
    if w < 64 || h < 64 || rgba.len() < w * h * 4 {
        return false;
    }

    let border_rows = (h / 5).max(8);
    let mut border_sum = 0u64;
    let mut border_count = 0u64;
    let mut border_min = 255u8;
    let mut border_max = 0u8;

    for y in (0..border_rows).chain((h - border_rows)..h) {
        let row_off = y * w * 4;
        for x in (0..w).step_by(2) {
            let idx = row_off + x * 4;
            let r = rgba[idx];
            let g = rgba[idx + 1];
            let b = rgba[idx + 2];
            let a = rgba[idx + 3];
            if a < 240 {
                return false;
            }
            let cmax = r.max(g).max(b);
            let cmin = r.min(g).min(b);
            if cmax - cmin > 4 {
                return false;
            }
            let luma = ((r as u16 + g as u16 + b as u16) / 3) as u8;
            border_sum += luma as u64;
            border_count += 1;
            border_min = border_min.min(luma);
            border_max = border_max.max(luma);
        }
    }

    if border_count == 0 {
        return false;
    }
    let bg_mean = (border_sum as f32) / (border_count as f32);
    if !(125.0..=246.0).contains(&bg_mean) {
        return false;
    }
    if border_max.saturating_sub(border_min) > 6 {
        return false;
    }

    let mut border_abs_diff = 0.0f32;
    let mut mad_count = 0u32;
    for y in (0..border_rows).chain((h - border_rows)..h) {
        let row_off = y * w * 4;
        for x in (0..w).step_by(4) {
            let idx = row_off + x * 4;
            let luma =
                (rgba[idx] as f32 + rgba[idx + 1] as f32 + rgba[idx + 2] as f32) * (1.0 / 3.0);
            border_abs_diff += (luma - bg_mean).abs();
            mad_count += 1;
        }
    }
    let border_mad = border_abs_diff / (mad_count.max(1) as f32);
    if border_mad > 1.0 {
        return false;
    }

    let mut total_samples = 0u32;
    let mut achromatic_samples = 0u32;
    let mut bg_matching_samples = 0u32;
    let mut text_contrast_samples = 0u32;

    let mid_y0 = h / 4;
    let mid_y1 = h - h / 4;
    for y in (0..h).step_by(2) {
        let row_off = y * w * 4;
        let in_mid = y >= mid_y0 && y < mid_y1;
        for x in (0..w).step_by(2) {
            let idx = row_off + x * 4;
            let r = rgba[idx];
            let g = rgba[idx + 1];
            let b = rgba[idx + 2];
            let cmax = r.max(g).max(b);
            let cmin = r.min(g).min(b);
            total_samples += 1;
            if cmax - cmin <= 6 {
                achromatic_samples += 1;
            }
            let luma = (r as f32 + g as f32 + b as f32) * (1.0 / 3.0);
            let diff = (luma - bg_mean).abs();
            if diff <= 3.5 {
                bg_matching_samples += 1;
            } else if in_mid && diff >= 10.0 {
                text_contrast_samples += 1;
            }
        }
    }

    if total_samples == 0 {
        return false;
    }
    let achromatic_ratio = (achromatic_samples as f32) / (total_samples as f32);
    let bg_ratio = (bg_matching_samples as f32) / (total_samples as f32);

    achromatic_ratio >= 0.975
        && bg_ratio >= 0.86
        && (text_contrast_samples >= 8 || bg_ratio >= 0.995)
}

