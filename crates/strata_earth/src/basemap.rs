//! Embedded compressed default base world map and seamless hybrid tile streaming.
//!
//! Embeds public-domain NASA Blue Marble (`z=0..=4`, `4096x4096` EPSG:3857) JPEG tiles inside the
//! `strata_earth` crate so every Earth view immediately renders crisp global continents,
//! oceans, bathymetry, and mountain shading on frame 0—even offline—and smoothly upgrades
//! chunk-by-chunk as higher-zoom network tiles arrive.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use crate::geodesy::TileKey;
use crate::tiles::{NetworkTileSource, TileElevation, TileError, TileImage, TileSource};

/// Highest zoom level baked inside `assets/base_world_map.pack`.
pub const MAX_EMBEDDED_BASE_LEVEL: u8 = 4;

const EMBEDDED_PACK: &[u8] = include_bytes!("../assets/base_world_map.pack");
const PACK_MAGIC: &[u8; 8] = b"SEBM0001";

static DECODED_BASE_TILES: OnceLock<Mutex<HashMap<TileKey, TileImage>>> = OnceLock::new();
static RAW_BASE_ENTRIES: OnceLock<HashMap<TileKey, &'static [u8]>> = OnceLock::new();

fn raw_entries() -> &'static HashMap<TileKey, &'static [u8]> {
    RAW_BASE_ENTRIES.get_or_init(|| {
        let mut map = HashMap::new();
        if EMBEDDED_PACK.len() < 12 || &EMBEDDED_PACK[..8] != PACK_MAGIC {
            return map;
        }
        let count = u32::from_le_bytes([
            EMBEDDED_PACK[8],
            EMBEDDED_PACK[9],
            EMBEDDED_PACK[10],
            EMBEDDED_PACK[11],
        ]) as usize;
        let mut offset = 12usize;
        for _ in 0..count {
            if offset + 13 > EMBEDDED_PACK.len() {
                break;
            }
            let level = EMBEDDED_PACK[offset];
            let x = u32::from_le_bytes([
                EMBEDDED_PACK[offset + 1],
                EMBEDDED_PACK[offset + 2],
                EMBEDDED_PACK[offset + 3],
                EMBEDDED_PACK[offset + 4],
            ]);
            let y = u32::from_le_bytes([
                EMBEDDED_PACK[offset + 5],
                EMBEDDED_PACK[offset + 6],
                EMBEDDED_PACK[offset + 7],
                EMBEDDED_PACK[offset + 8],
            ]);
            let len = u32::from_le_bytes([
                EMBEDDED_PACK[offset + 9],
                EMBEDDED_PACK[offset + 10],
                EMBEDDED_PACK[offset + 11],
                EMBEDDED_PACK[offset + 12],
            ]) as usize;
            offset += 13;
            if offset + len > EMBEDDED_PACK.len() {
                break;
            }
            map.insert(
                TileKey::new(level, x, y),
                &EMBEDDED_PACK[offset..offset + len],
            );
            offset += len;
        }
        map
    })
}

/// Decodes and returns a cached RGBA8 base world map tile (`z=0..=4`), if present in the embedded pack.
pub fn embedded_base_tile(key: TileKey) -> Option<TileImage> {
    if key.level > MAX_EMBEDDED_BASE_LEVEL {
        return None;
    }
    let cache = DECODED_BASE_TILES.get_or_init(|| Mutex::new(HashMap::new()));
    if let Ok(guard) = cache.lock() {
        if let Some(img) = guard.get(&key) {
            return Some(img.clone());
        }
    }
    let bytes = *raw_entries().get(&key)?;
    let decoded = image::load_from_memory(bytes).ok()?;
    let rgba = decoded.to_rgba8();
    let img = TileImage {
        width: rgba.width(),
        height: rgba.height(),
        pixels: rgba.into_raw(),
    };
    if let Ok(mut guard) = cache.lock() {
        guard.insert(key, img.clone());
    }
    Some(img)
}

/// Synchronous [`TileSource`] serving only the embedded compressed base world map (`z=0..=4`).
#[derive(Debug, Default, Clone, Copy)]
pub struct EmbeddedBaseMapSource;

impl EmbeddedBaseMapSource {
    pub fn new() -> Self {
        Self
    }
}

impl TileSource for EmbeddedBaseMapSource {
    fn imagery(&mut self, key: TileKey) -> Result<Option<TileImage>, TileError> {
        Ok(embedded_base_tile(key))
    }

    fn peek_cached_imagery(&self, key: TileKey) -> Option<TileImage> {
        embedded_base_tile(key)
    }

    fn elevation(&mut self, _key: TileKey) -> Result<Option<TileElevation>, TileError> {
        Ok(None)
    }
}

/// Default production Earth tile source:
/// - Immediately provides the embedded compressed NASA Blue Marble base map (`z=0..=4`, `4096x4096`).
/// - Streams higher-resolution satellite/street imagery and Terrarium DEM heightmaps asynchronously
///   from network endpoints (with in-memory + optional disk LRU caching).
/// - Always serves lower-resolution built-in or cached ancestors while higher-zoom chunks stream.
pub struct EarthDefaultTileSource {
    network: Option<NetworkTileSource>,
    max_elevation_level: u8,
}

impl EarthDefaultTileSource {
    /// Creates an offline-safe source backed solely by the embedded compressed base world map.
    pub fn offline_basemap() -> Self {
        Self {
            network: None,
            max_elevation_level: 15,
        }
    }

    /// Creates the default CesiumJS-style world imagery + global Terrarium elevation streaming source.
    pub fn world_satellite() -> Self {
        let net = NetworkTileSource::world_satellite_with_terrain();
        Self {
            network: Some(net),
            max_elevation_level: 15,
        }
    }

    /// Creates an OpenStreetMap road/label map source + global Terrarium elevation streaming.
    pub fn world_street() -> Self {
        let net = NetworkTileSource::osm().with_terrarium_elevation();
        Self {
            network: Some(net),
            max_elevation_level: 15,
        }
    }

    /// Wraps a custom [`NetworkTileSource`] with the embedded default base world map fallback.
    pub fn with_network(network: NetworkTileSource) -> Self {
        Self {
            network: Some(network),
            max_elevation_level: 15,
        }
    }

    /// Enables persistent disk caching for streamed higher-zoom tiles.
    pub fn with_disk_cache(mut self, dir: impl Into<std::path::PathBuf>) -> Self {
        if let Some(net) = self.network.take() {
            self.network = Some(net.with_disk_cache(dir));
        }
        self
    }

    /// Registers a wakeup callback invoked on background threads when any downloaded tile arrives.
    pub fn set_on_tile_ready(&mut self, cb: Option<Arc<dyn Fn() + Send + Sync>>) {
        if let Some(ref mut net) = self.network {
            net.set_on_tile_ready(cb);
        }
    }

    /// Mutable access to the underlying [`NetworkTileSource`], if enabled.
    pub fn network_mut(&mut self) -> Option<&mut NetworkTileSource> {
        self.network.as_mut()
    }
}

impl TileSource for EarthDefaultTileSource {
    fn imagery(&mut self, key: TileKey) -> Result<Option<TileImage>, TileError> {
        if key.level <= MAX_EMBEDDED_BASE_LEVEL {
            return Ok(embedded_base_tile(key));
        }
        if let Some(ref mut net) = self.network {
            if let Some(img) = net.imagery(key)? {
                return Ok(Some(img));
            }
        }
        Ok(None)
    }

    fn peek_cached_imagery(&self, key: TileKey) -> Option<TileImage> {
        if let Some(ref net) = self.network {
            if let Some(img) = net.peek_cached_imagery(key) {
                return Some(img);
            }
        }
        embedded_base_tile(key)
    }

    fn elevation(&mut self, key: TileKey) -> Result<Option<TileElevation>, TileError> {
        let mut eff = key;
        while eff.level > self.max_elevation_level {
            eff = match eff.parent() {
                Some(p) => p,
                None => break,
            };
        }
        if let Some(ref mut net) = self.network {
            return net.elevation(eff);
        }
        Ok(None)
    }

    fn peek_cached_elevation(&self, key: TileKey) -> Option<TileElevation> {
        let mut eff = key;
        while eff.level > self.max_elevation_level {
            eff = match eff.parent() {
                Some(p) => p,
                None => break,
            };
        }
        if let Some(ref net) = self.network {
            return net.peek_cached_elevation(eff);
        }
        None
    }
}
