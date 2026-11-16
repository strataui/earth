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
    mpsc::{channel, Receiver, Sender},
    Arc, Mutex,
};

use crate::geodesy::TileKey;
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
    /// Returns the imagery for a tile, if available right now.
    fn imagery(&mut self, key: TileKey) -> Result<Option<TileImage>, TileError>;

    /// Non-mutating lookup of imagery already resident in memory/embedded defaults
    /// (must never initiate network I/O). Used when resolving loaded ancestor fallbacks.
    fn peek_cached_imagery(&self, _key: TileKey) -> Option<TileImage> {
        None
    }

    /// Returns the elevation for a tile, if available right now.
    fn elevation(&mut self, key: TileKey) -> Result<Option<TileElevation>, TileError>;
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

    fn elevation(&mut self, key: TileKey) -> Result<Option<TileElevation>, TileError> {
        Ok(self.elevation.get(&key).cloned())
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

    pub fn contains_imagery(&self, key: TileKey) -> bool {
        self.map.contains_key(&TileDataKey::Imagery(key))
    }

    pub fn contains_elevation(&self, key: TileKey) -> bool {
        self.map.contains_key(&TileDataKey::Elevation(key))
    }

    /// Fetches imagery for `key`, marking it as most recently used.
    pub fn get_imagery(&mut self, key: TileKey) -> Option<TileImage> {
        match self.get(&TileDataKey::Imagery(key)) {
            Some(CachedTileData::Imagery(img)) => Some(img),
            _ => None,
        }
    }

    /// Peeks cached imagery without updating LRU order or needing mutable access.
    pub fn peek_imagery(&self, key: TileKey) -> Option<TileImage> {
        let &idx = self.map.get(&TileDataKey::Imagery(key))?;
        match self.nodes.get(idx).and_then(|n| n.as_ref())?.value {
            CachedTileData::Imagery(ref img) => Some(img.clone()),
            CachedTileData::Elevation(_) => None,
        }
    }

    /// Inserts imagery for `key`, evicting older entries if exceeding the byte budget.
    pub fn insert_imagery(&mut self, key: TileKey, image: TileImage) {
        self.insert(TileDataKey::Imagery(key), CachedTileData::Imagery(image));
    }

    /// Fetches elevation for `key`, marking it as most recently used.
    pub fn get_elevation(&mut self, key: TileKey) -> Option<TileElevation> {
        match self.get(&TileDataKey::Elevation(key)) {
            Some(CachedTileData::Elevation(elev)) => Some(elev),
            _ => None,
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
}

/// Formats a Slippy Map XYZ or WMTS URL template with a given TileKey.
///
/// Supported tokens:
/// - `{z}`: tile zoom level
/// - `{x}`: tile column index
/// - `{y}`: tile row index
/// - `{s}`: subdomain round-robin
/// - `{TileMatrix}`: WMTS tile matrix (level)
/// - `{TileCol}`: WMTS column
/// - `{TileRow}`: WMTS row
pub fn format_slippy_url(
    template: &str,
    key: TileKey,
    subdomains: &[String],
    subdomain_counter: &mut usize,
) -> String {
    let mut url = template
        .replace("{z}", &key.level.to_string())
        .replace("{x}", &key.x.to_string())
        .replace("{y}", &key.y.to_string())
        .replace("{TileMatrix}", &key.level.to_string())
        .replace("{TileCol}", &key.x.to_string())
        .replace("{TileRow}", &key.y.to_string());

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
}

impl HttpTransport for StandardHttpTransport {
    fn fetch(&self, url: &str) -> Result<Vec<u8>, TileError> {
        let resp = self
            .agent
            .get(url)
            .set(
                "Accept",
                "image/png,image/jpeg,image/webp,application/json,*/*",
            )
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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElevationEncoding {
    RawF32,
    TerrainRgb,
    Terrarium,
}

struct FetchTask {
    key: TileDataKey,
    url: String,
    is_elevation: bool,
    elevation_encoding: ElevationEncoding,
    disk_cache_path: Option<PathBuf>,
}

type TileReadyCallback = Arc<dyn Fn() + Send + Sync>;
type TileReadySlot = Arc<Mutex<Option<TileReadyCallback>>>;

/// Non-blocking asynchronous network tile streaming source supporting Slippy XYZ
/// and WMTS raster tile protocols with memory and disk LRU caching.
pub struct NetworkTileSource {
    imagery_url_template: String,
    elevation_url_template: Option<String>,
    elevation_encoding: ElevationEncoding,
    subdomains: Vec<String>,
    subdomain_counter: usize,
    cache: LruTileCache,
    disk_cache_dir: Option<PathBuf>,
    transport: Arc<dyn HttpTransport>,
    pending: Arc<Mutex<HashSet<TileDataKey>>>,
    results_rx: Receiver<(TileDataKey, Result<CachedTileData, TileError>)>,
    task_tx: Sender<FetchTask>,
    on_tile_ready: TileReadySlot,
    stats: NetworkTileStats,
    default_tile_size: u32,
}

impl NetworkTileSource {
    /// Creates a new network tile source targeting the specified imagery URL template.
    pub fn new(imagery_url_template: impl Into<String>) -> Self {
        Self::with_options(
            imagery_url_template,
            None,
            false,
            DEFAULT_MAX_CACHE_BYTES,
            None,
            Arc::new(StandardHttpTransport::new()),
        )
    }

    /// Convenience constructor targeting standard OpenStreetMap tiles (`https://tile.openstreetmap.org/{z}/{x}/{y}.png`).
    pub fn osm() -> Self {
        Self::new("https://tile.openstreetmap.org/{z}/{x}/{y}.png")
    }

    /// Convenience constructor targeting open satellite world imagery (ArcGIS World Imagery)
    /// paired with global AWS Terrarium DEM heightmaps.
    pub fn world_satellite_with_terrain() -> Self {
        Self::new("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}")
            .with_terrarium_elevation()
    }

    /// Convenience constructor targeting Mapbox satellite imagery and Terrain-RGB elevation.
    pub fn mapbox(access_token: &str) -> Self {
        let img = format!(
            "https://api.mapbox.com/v4/mapbox.satellite/{{z}}/{{x}}/{{y}}@2x.png?access_token={access_token}"
        );
        let elev = format!(
            "https://api.mapbox.com/v4/mapbox.terrain-rgb/{{z}}/{{x}}/{{y}}.pngraw?access_token={access_token}"
        );
        Self::with_options(
            img,
            Some(elev),
            true,
            DEFAULT_MAX_CACHE_BYTES,
            None,
            Arc::new(StandardHttpTransport::new()),
        )
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
        let (task_tx, task_rx) = channel::<FetchTask>();
        let (results_tx, results_rx) =
            channel::<(TileDataKey, Result<CachedTileData, TileError>)>();
        let on_tile_ready: TileReadySlot = Arc::new(Mutex::new(None));

        let worker_transport = Arc::clone(&transport);
        let worker_ready = Arc::clone(&on_tile_ready);
        std::thread::Builder::new()
            .name("strata-tile-fetcher".to_string())
            .spawn(move || {
                while let Ok(task) = task_rx.recv() {
                    let result = Self::process_fetch_task(&worker_transport, &task);
                    let ok = result.is_ok();
                    let _ = results_tx.send((task.key, result));
                    if ok {
                        if let Ok(guard) = worker_ready.lock() {
                            if let Some(ref cb) = *guard {
                                cb();
                            }
                        }
                    }
                }
            })
            .expect("failed to spawn strata-tile-fetcher thread");

        Self {
            imagery_url_template: imagery_url_template.into(),
            elevation_url_template,
            elevation_encoding: if elevation_is_terrain_rgb {
                ElevationEncoding::TerrainRgb
            } else {
                ElevationEncoding::RawF32
            },
            subdomains: vec!["a".to_string(), "b".to_string(), "c".to_string()],
            subdomain_counter: 0,
            cache: LruTileCache::new(max_cache_bytes),
            disk_cache_dir,
            transport,
            pending: Arc::new(Mutex::new(HashSet::new())),
            results_rx,
            task_tx,
            on_tile_ready,
            stats: NetworkTileStats::default(),
            default_tile_size: 256,
        }
    }

    /// Registers a wakeup callback invoked whenever a downloaded tile finishes decoding.
    pub fn set_on_tile_ready(&mut self, cb: Option<Arc<dyn Fn() + Send + Sync>>) {
        if let Ok(mut guard) = self.on_tile_ready.lock() {
            *guard = cb;
        }
    }

    /// Replaces the active HTTP transport (useful for mock injection in tests).
    pub fn with_transport(mut self, transport: impl HttpTransport) -> Self {
        let transport: Arc<dyn HttpTransport> = Arc::new(transport);
        self.transport = Arc::clone(&transport);
        // Restart worker with new transport
        let (task_tx, task_rx) = channel::<FetchTask>();
        let (results_tx, results_rx) =
            channel::<(TileDataKey, Result<CachedTileData, TileError>)>();
        let worker_transport = Arc::clone(&transport);
        let worker_ready = Arc::clone(&self.on_tile_ready);
        std::thread::Builder::new()
            .name("strata-tile-fetcher".to_string())
            .spawn(move || {
                while let Ok(task) = task_rx.recv() {
                    let result = Self::process_fetch_task(&worker_transport, &task);
                    let ok = result.is_ok();
                    let _ = results_tx.send((task.key, result));
                    if ok {
                        if let Ok(guard) = worker_ready.lock() {
                            if let Some(ref cb) = *guard {
                                cb();
                            }
                        }
                    }
                }
            })
            .expect("failed to spawn strata-tile-fetcher thread");

        self.task_tx = task_tx;
        self.results_rx = results_rx;
        self
    }

    /// Sets the on-disk cache directory.
    pub fn with_disk_cache(mut self, path: impl Into<PathBuf>) -> Self {
        self.disk_cache_dir = Some(path.into());
        self
    }

    /// Configures subdomains for load balancing.
    pub fn with_subdomains(mut self, subdomains: &[&str]) -> Self {
        self.subdomains = subdomains.iter().map(|s| s.to_string()).collect();
        self
    }

    /// Sets the elevation URL template.
    pub fn with_elevation(mut self, template: impl Into<String>, is_terrain_rgb: bool) -> Self {
        self.elevation_url_template = Some(template.into());
        self.elevation_encoding = if is_terrain_rgb {
            ElevationEncoding::TerrainRgb
        } else {
            ElevationEncoding::RawF32
        };
        self
    }

    /// Enables public-domain global AWS Terrarium DEM heightmaps (`z=0..15`).
    pub fn with_terrarium_elevation(mut self) -> Self {
        self.elevation_url_template = Some(
            "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png".to_string(),
        );
        self.elevation_encoding = ElevationEncoding::Terrarium;
        self
    }

    /// Inspects the current network tile statistics.
    pub fn stats(&self) -> NetworkTileStats {
        self.stats
    }

    /// Accesses the underlying in-memory LRU cache.
    pub fn cache(&self) -> &LruTileCache {
        &self.cache
    }

    /// Drains completed background downloads into the in-memory LRU cache.
    pub fn poll_completed(&mut self) {
        while let Ok((key, result)) = self.results_rx.try_recv() {
            if let Ok(mut pending) = self.pending.lock() {
                pending.remove(&key);
            }
            match result {
                Ok(data) => match data {
                    CachedTileData::Imagery(img) => {
                        if let TileDataKey::Imagery(k) = key {
                            self.cache.insert_imagery(k, img);
                        }
                    }
                    CachedTileData::Elevation(elev) => {
                        if let TileDataKey::Elevation(k) = key {
                            self.cache.insert_elevation(k, elev);
                        }
                    }
                },
                Err(_e) => {
                    self.stats.decode_errors += 1;
                }
            }
        }
    }

    /// Synchronously fetches and caches an imagery tile, blocking until complete.
    pub fn fetch_sync(&mut self, key: TileKey) -> Result<Option<TileImage>, TileError> {
        let url = format_slippy_url(
            &self.imagery_url_template,
            key,
            &self.subdomains,
            &mut self.subdomain_counter,
        );
        let bytes = self.transport.fetch(&url)?;
        if let Some(dir) = &self.disk_cache_dir {
            let path = dir.join(format!("{}/{}/{}.png", key.level, key.x, key.y));
            if let Some(parent) = path.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            let _ = std::fs::write(path, &bytes);
        }
        let image = Self::decode_imagery_bytes(&bytes, self.default_tile_size)?;
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
        let url = format_slippy_url(
            &template,
            key,
            &self.subdomains,
            &mut self.subdomain_counter,
        );
        let bytes = self.transport.fetch(&url)?;
        if let Some(dir) = &self.disk_cache_dir {
            let path = dir.join(format!("{}/{}/{}.elev", key.level, key.x, key.y));
            if let Some(parent) = path.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            let _ = std::fs::write(path, &bytes);
        }
        let elev = Self::decode_elevation_bytes(&bytes, self.elevation_encoding)?;
        self.cache.insert_elevation(key, elev.clone());
        Ok(Some(elev))
    }

    fn process_fetch_task(
        transport: &Arc<dyn HttpTransport>,
        task: &FetchTask,
    ) -> Result<CachedTileData, TileError> {
        let bytes = transport.fetch(&task.url)?;
        if let Some(path) = &task.disk_cache_path {
            if let Some(parent) = path.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            let _ = std::fs::write(path, &bytes);
        }
        if task.is_elevation {
            let elev = Self::decode_elevation_bytes(&bytes, task.elevation_encoding)?;
            Ok(CachedTileData::Elevation(elev))
        } else {
            let img = Self::decode_imagery_bytes(&bytes, 256)?;
            Ok(CachedTileData::Imagery(img))
        }
    }

    fn decode_imagery_bytes(bytes: &[u8], default_size: u32) -> Result<TileImage, TileError> {
        if bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
            if let Ok((w, h, rgba)) = decode_png_to_rgba(bytes) {
                return Ok(TileImage {
                    width: w,
                    height: h,
                    pixels: rgba,
                });
            }
        }
        if let Ok(dyn_img) = image::load_from_memory(bytes) {
            let rgba = dyn_img.to_rgba8();
            return Ok(TileImage {
                width: rgba.width(),
                height: rgba.height(),
                pixels: rgba.into_raw(),
            });
        }
        if bytes.len() >= 8 {
            let w = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
            let h = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
            if w > 0 && h > 0 && bytes.len() == 8 + (w as usize * h as usize * 4) {
                return Ok(TileImage {
                    width: w,
                    height: h,
                    pixels: bytes[8..].to_vec(),
                });
            }
        }
        if bytes.len() == (default_size as usize * default_size as usize * 4) {
            return Ok(TileImage {
                width: default_size,
                height: default_size,
                pixels: bytes.to_vec(),
            });
        }
        if bytes.len().is_multiple_of(4) {
            let pixels = bytes.len() / 4;
            let side = (pixels as f64).sqrt().round() as usize;
            if side * side == pixels {
                return Ok(TileImage {
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
}

impl TileSource for NetworkTileSource {
    fn imagery(&mut self, key: TileKey) -> Result<Option<TileImage>, TileError> {
        self.poll_completed();

        // 1. Check in-memory LRU cache
        if let Some(img) = self.cache.get_imagery(key) {
            self.stats.cache_hits += 1;
            return Ok(Some(img));
        }

        // 2. Check on-disk cache if configured
        if let Some(dir) = &self.disk_cache_dir {
            let path = dir.join(format!("{}/{}/{}.png", key.level, key.x, key.y));
            if path.is_file() {
                if let Ok(bytes) = std::fs::read(&path) {
                    if let Ok(img) = Self::decode_imagery_bytes(&bytes, self.default_tile_size) {
                        self.cache.insert_imagery(key, img.clone());
                        self.stats.disk_cache_hits += 1;
                        return Ok(Some(img));
                    }
                }
            }
        }

        self.stats.cache_misses += 1;
        let data_key = TileDataKey::Imagery(key);

        // 3. Enqueue fetch if not already pending
        let mut is_new = false;
        if let Ok(mut pending) = self.pending.lock() {
            if !pending.contains(&data_key) {
                pending.insert(data_key);
                is_new = true;
            }
        }

        if is_new {
            let url = format_slippy_url(
                &self.imagery_url_template,
                key,
                &self.subdomains,
                &mut self.subdomain_counter,
            );
            let disk_cache_path = self
                .disk_cache_dir
                .as_ref()
                .map(|dir| dir.join(format!("{}/{}/{}.png", key.level, key.x, key.y)));
            let _ = self.task_tx.send(FetchTask {
                key: data_key,
                url,
                is_elevation: false,
                elevation_encoding: ElevationEncoding::RawF32,
                disk_cache_path,
            });
            self.stats.network_requests += 1;
        }

        // Return None non-blockingly so the scene renderer falls back to an ancestor tile
        Ok(None)
    }

    fn peek_cached_imagery(&self, key: TileKey) -> Option<TileImage> {
        self.cache.peek_imagery(key)
    }

    fn elevation(&mut self, key: TileKey) -> Result<Option<TileElevation>, TileError> {
        self.poll_completed();

        let template = match &self.elevation_url_template {
            Some(t) => t.clone(),
            None => return Ok(None),
        };

        if let Some(elev) = self.cache.get_elevation(key) {
            self.stats.cache_hits += 1;
            return Ok(Some(elev));
        }

        if let Some(dir) = &self.disk_cache_dir {
            let path = dir.join(format!("{}/{}/{}.elev", key.level, key.x, key.y));
            if path.is_file() {
                if let Ok(bytes) = std::fs::read(&path) {
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
        let data_key = TileDataKey::Elevation(key);

        let mut is_new = false;
        if let Ok(mut pending) = self.pending.lock() {
            if !pending.contains(&data_key) {
                pending.insert(data_key);
                is_new = true;
            }
        }

        if is_new {
            let url = format_slippy_url(
                &template,
                key,
                &self.subdomains,
                &mut self.subdomain_counter,
            );
            let disk_cache_path = self
                .disk_cache_dir
                .as_ref()
                .map(|dir| dir.join(format!("{}/{}/{}.elev", key.level, key.x, key.y)));
            let _ = self.task_tx.send(FetchTask {
                key: data_key,
                url,
                is_elevation: true,
                elevation_encoding: self.elevation_encoding,
                disk_cache_path,
            });
            self.stats.network_requests += 1;
        }

        Ok(None)
    }
}
