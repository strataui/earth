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
    store: TextureStore,
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
            store,
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

    /// Uploads an image for a slot if it is not present yet.
    pub fn ensure(&mut self, slot: TextureSlot, image: &TileImage) -> Option<TextureId> {
        if let Some(existing) = self.slots.get(&slot) {
            return Some(*existing);
        }
        let data = image.to_texture_data()?;
        let id = self.store.insert(data, tile_sampler())?;
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
                self.store.remove(id);
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
}

impl DirectoryTileSource {
    pub fn new(base_dir: impl Into<std::path::PathBuf>) -> Self {
        Self {
            base_dir: base_dir.into(),
            default_tile_size: 256,
        }
    }

    pub fn with_tile_size(mut self, size: u32) -> Self {
        self.default_tile_size = size;
        self
    }

    pub fn base_dir(&self) -> &std::path::Path {
        &self.base_dir
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
        if bytes.len() % 4 == 0 {
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
            .as_chunks::<4>().0.iter()
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        Ok(Some(TileElevation {
            width: w,
            height: h,
            heights_m,
        }))
    }
}
