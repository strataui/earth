//! Embedded and per-provider `z=0..=4` base world map packs (`SEBM0001`) and hybrid streaming.
//!
//! Embeds a compact `SEBM0001` base world map pack (`z=0..=4`, `4096x4096` world resolution)
//! inside `strata_earth` for instant frame-0 offline/placeholder rendering, and supports loading
//! matching per-provider `z=0..=4` packs baked from each Tile API (`base_world_map_<id>.pack`)
//! or streaming `z=0..=4` asynchronously from the active [`NetworkTileSource`] whenever no
//! matching pack is loaded.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use crate::assets_cdn::AssetSourceConfig;
use crate::geodesy::TileKey;
use crate::tiles::{
    NetworkTileSource, TileElevation, TileError, TileImage, TileProviderConfig,
    TileProviderPreset, TileSource,
};

/// Highest zoom level baked inside `assets/base_world_map.pack`.
pub const MAX_EMBEDDED_BASE_LEVEL: u8 = 4;

const EMBEDDED_PACK: &[u8] = include_bytes!("../assets/base_world_map.pack");
const PACK_MAGIC: &[u8; 8] = b"SEBM0001";

#[derive(Clone)]
enum PackBacking {
    Static(&'static [u8]),
    Owned(Arc<[u8]>),
}

impl PackBacking {
    #[inline]
    fn as_bytes(&self) -> &[u8] {
        match self {
            Self::Static(s) => s,
            Self::Owned(b) => b.as_ref(),
        }
    }
}

/// Encodes `(TileKey, compressed_image_bytes)` entries into an `SEBM0001` base world map pack.
///
/// Format:
/// - `[0..8]`: `b"SEBM0001"`
/// - `[8..12]`: `u32` little-endian entry count
/// - Repeated entry: `[level: u8, x: u32 LE, y: u32 LE, len: u32 LE, payload[0..len]]`
pub fn encode_sebm_pack(entries: &[(TileKey, &[u8])]) -> Vec<u8> {
    let payload_sum: usize = entries.iter().map(|(_, b)| 13 + b.len()).sum();
    let mut out = Vec::with_capacity(12 + payload_sum);
    out.extend_from_slice(PACK_MAGIC);
    out.extend_from_slice(&(entries.len() as u32).to_le_bytes());
    for &(key, bytes) in entries {
        out.push(key.level);
        out.extend_from_slice(&key.x.to_le_bytes());
        out.extend_from_slice(&key.y.to_le_bytes());
        out.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
        out.extend_from_slice(bytes);
    }
    out
}

/// Parsed `SEBM0001` compressed base world map pack with lazy on-demand RGBA8 decoding cache.
pub struct BaseWorldMapPack {
    backing: PackBacking,
    entries: HashMap<TileKey, (usize, usize)>,
    decoded: Mutex<HashMap<TileKey, TileImage>>,
    max_level: u8,
}

impl std::fmt::Debug for BaseWorldMapPack {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BaseWorldMapPack")
            .field("entries", &self.entries.len())
            .field("max_level", &self.max_level)
            .finish()
    }
}

impl BaseWorldMapPack {
    fn parse_with_backing(backing: PackBacking) -> Result<Self, TileError> {
        let bytes = backing.as_bytes();
        if bytes.len() < 12 || &bytes[..8] != PACK_MAGIC {
            return Err(TileError::Decode(
                "invalid SEBM0001 base pack header".to_string(),
            ));
        }
        let count = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
        let mut entries = HashMap::with_capacity(count);
        let mut max_level = 0u8;
        let mut offset = 12usize;
        for _ in 0..count {
            if offset + 13 > bytes.len() {
                break;
            }
            let level = bytes[offset];
            let x = u32::from_le_bytes([
                bytes[offset + 1],
                bytes[offset + 2],
                bytes[offset + 3],
                bytes[offset + 4],
            ]);
            let y = u32::from_le_bytes([
                bytes[offset + 5],
                bytes[offset + 6],
                bytes[offset + 7],
                bytes[offset + 8],
            ]);
            let len = u32::from_le_bytes([
                bytes[offset + 9],
                bytes[offset + 10],
                bytes[offset + 11],
                bytes[offset + 12],
            ]) as usize;
            offset += 13;
            if offset + len > bytes.len() {
                return Err(TileError::Decode(
                    "truncated SEBM0001 base pack entry".to_string(),
                ));
            }
            entries.insert(TileKey::new(level, x, y), (offset, offset + len));
            if level > max_level {
                max_level = level;
            }
            offset += len;
        }
        Ok(Self {
            backing,
            entries,
            decoded: Mutex::new(HashMap::new()),
            max_level,
        })
    }

    /// Parses an owned `SEBM0001` pack byte buffer.
    pub fn from_bytes(bytes: impl Into<Arc<[u8]>>) -> Result<Self, TileError> {
        Self::parse_with_backing(PackBacking::Owned(bytes.into()))
    }

    /// Loads and parses an `SEBM0001` pack file from disk.
    pub fn from_file(path: &Path) -> Result<Self, TileError> {
        let bytes = std::fs::read(path)
            .map_err(|e| TileError::Transport(format!("read base pack {}: {e}", path.display())))?;
        Self::from_bytes(bytes)
    }

    /// Returns the shared compiled-in default base world map pack (`assets/base_world_map.pack`).
    pub fn embedded_default() -> Arc<Self> {
        static EMBEDDED: OnceLock<Arc<BaseWorldMapPack>> = OnceLock::new();
        Arc::clone(EMBEDDED.get_or_init(|| {
            Arc::new(
                Self::parse_with_backing(PackBacking::Static(EMBEDDED_PACK)).unwrap_or_else(|_| {
                    Self {
                        backing: PackBacking::Static(&[]),
                        entries: HashMap::new(),
                        decoded: Mutex::new(HashMap::new()),
                        max_level: 0,
                    }
                }),
            )
        }))
    }

    /// Highest tile zoom level present in this pack.
    #[inline]
    pub fn max_level(&self) -> u8 {
        self.max_level
    }

    /// Number of tiles baked into this pack.
    #[inline]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Returns `true` if `key` exists in this pack.
    #[inline]
    pub fn contains(&self, key: TileKey) -> bool {
        self.entries.contains_key(&key)
    }

    /// Decodes and returns a cached RGBA8 tile from this pack, if present.
    pub fn tile(&self, key: TileKey) -> Option<TileImage> {
        if key.level > self.max_level {
            return None;
        }
        if let Ok(guard) = self.decoded.lock() {
            if let Some(img) = guard.get(&key) {
                return Some(img.clone());
            }
        }
        let &(start, end) = self.entries.get(&key)?;
        let raw = &self.backing.as_bytes()[start..end];
        let decoded = image::load_from_memory(raw).ok()?;
        let rgba = decoded.to_rgba8();
        let img = TileImage {
            width: rgba.width(),
            height: rgba.height(),
            pixels: rgba.into_raw(),
        };
        if let Ok(mut guard) = self.decoded.lock() {
            guard.insert(key, img.clone());
        }
        Some(img)
    }
}

/// Decodes and returns a cached RGBA8 tile (`z=0..=4`) from the compiled-in default pack.
pub fn embedded_base_tile(key: TileKey) -> Option<TileImage> {
    if key.level > MAX_EMBEDDED_BASE_LEVEL {
        return None;
    }
    BaseWorldMapPack::embedded_default().tile(key)
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

/// Production Earth tile source supporting multiple Tile API providers and matching `z=0..=4` packs:
/// - When a matching baked `SEBM0001` pack is loaded for the active provider, `z=0..=4` renders
///   instantaneously with exact provider colors and zero `z=4 -> z=5` brightness/color pop.
/// - When an arbitrary/custom or unpacked provider is active, the compiled-in pack acts ONLY as an
///   instant offline/frame-0 placeholder (`peek_cached_imagery`) while the active [`NetworkTileSource`]
///   downloads and upgrades `z=0..=4` asynchronously from the target Tile API.
/// - Switching providers cleanly clears/namespaces caches so tiles never bleed across APIs.
pub struct EarthDefaultTileSource {
    network: Option<NetworkTileSource>,
    provider: TileProviderConfig,
    active_pack: Option<Arc<BaseWorldMapPack>>,
    fallback_pack: Arc<BaseWorldMapPack>,
    pack_search_dirs: Vec<PathBuf>,
    max_elevation_level: u8,
    stream_base_from_network: bool,
}

impl EarthDefaultTileSource {
    fn default_pack_dirs() -> Vec<PathBuf> {
        let mut dirs = Vec::new();
        dirs.push(AssetSourceConfig::default().resolved_cache_dir());
        dirs.push(PathBuf::from("dist/assets"));
        dirs.push(PathBuf::from("crates/strata_earth/assets"));
        dirs
    }

    fn find_pack_for_provider(
        config: &TileProviderConfig,
        dirs: &[PathBuf],
        allow_embedded_satellite: bool,
    ) -> Option<Arc<BaseWorldMapPack>> {
        let pack_id = config.base_pack_id.as_deref()?;
        let filename = format!("base_world_map_{pack_id}.pack");
        for dir in dirs {
            for candidate in [dir.join(&filename), dir.join("packs").join(&filename)] {
                if candidate.is_file() {
                    if let Ok(pack) = BaseWorldMapPack::from_file(&candidate) {
                        return Some(Arc::new(pack));
                    }
                }
            }
        }
        let default_sat_tpl =
            TileProviderConfig::preset(TileProviderPreset::ArcGisWorldImagery).imagery_url_template;
        if allow_embedded_satellite
            && pack_id == TileProviderPreset::ArcGisWorldImagery.id()
            && config.imagery_url_template == default_sat_tpl
        {
            return Some(BaseWorldMapPack::embedded_default());
        }
        None
    }

    /// Creates an offline-safe source backed solely by the embedded compressed base world map.
    pub fn offline_basemap() -> Self {
        let fallback_pack = BaseWorldMapPack::embedded_default();
        Self {
            network: None,
            provider: TileProviderConfig::default(),
            active_pack: Some(Arc::clone(&fallback_pack)),
            fallback_pack,
            pack_search_dirs: Self::default_pack_dirs(),
            max_elevation_level: 15,
            stream_base_from_network: false,
        }
    }

    /// Creates a hybrid streaming + base-pack source for any [`TileProviderConfig`].
    pub fn from_provider(config: TileProviderConfig) -> Self {
        let dirs = Self::default_pack_dirs();
        let active_pack = Self::find_pack_for_provider(&config, &dirs, true);
        let net = NetworkTileSource::from_provider(config.clone());
        Self {
            network: Some(net),
            provider: config,
            active_pack,
            fallback_pack: BaseWorldMapPack::embedded_default(),
            pack_search_dirs: dirs,
            max_elevation_level: 15,
            stream_base_from_network: false,
        }
    }

    /// Creates a hybrid streaming + base-pack source for any built-in [`TileProviderPreset`].
    pub fn from_preset(preset: TileProviderPreset) -> Self {
        Self::from_provider(TileProviderConfig::preset(preset))
    }

    /// Creates the default ArcGIS World Imagery satellite + Terrarium elevation + Black Marble source.
    pub fn world_satellite() -> Self {
        Self::from_preset(TileProviderPreset::ArcGisWorldImagery)
    }

    /// Creates an OpenStreetMap street/label map source + Terrarium elevation.
    ///
    /// Automatically loads `base_world_map_osm.pack` if present; otherwise streams `z=0..=4`
    /// dynamically from OpenStreetMap while using the embedded pack only as frame-0 placeholder.
    pub fn world_street() -> Self {
        Self::from_preset(TileProviderPreset::OpenStreetMap)
    }

    /// Wraps a custom [`NetworkTileSource`], only pinning `z <= 4` when a matching pack is found.
    pub fn with_network(network: NetworkTileSource) -> Self {
        let config = network.provider_config().clone();
        let dirs = Self::default_pack_dirs();
        let active_pack = Self::find_pack_for_provider(&config, &dirs, true);
        Self {
            network: Some(network),
            provider: config,
            active_pack,
            fallback_pack: BaseWorldMapPack::embedded_default(),
            pack_search_dirs: dirs,
            max_elevation_level: 15,
            stream_base_from_network: false,
        }
    }

    /// Enables persistent provider-namespaced disk caching and adds `dir` to base-pack search paths.
    pub fn with_disk_cache(mut self, dir: impl Into<std::path::PathBuf>) -> Self {
        let p: PathBuf = dir.into();
        self.pack_search_dirs.insert(0, p.clone());
        if let Some(net) = self.network.take() {
            self.network = Some(net.with_disk_cache(&p));
        }
        if self.active_pack.is_none() && !self.stream_base_from_network {
            self.active_pack = Self::find_pack_for_provider(&self.provider, &self.pack_search_dirs, true);
        }
        self
    }

    /// Adds a directory searched for `base_world_map_<id>.pack` files.
    pub fn with_pack_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.pack_search_dirs.insert(0, dir.into());
        if !self.stream_base_from_network {
            self.active_pack = Self::find_pack_for_provider(&self.provider, &self.pack_search_dirs, true);
        }
        self
    }

    /// Explicitly binds a pre-loaded [`BaseWorldMapPack`] as the active provider's `z=0..=4` pack.
    pub fn with_base_pack(mut self, pack: Arc<BaseWorldMapPack>) -> Self {
        self.active_pack = Some(pack);
        self
    }

    /// When `true`, forces `z=0..=4` tiles to stream directly from `NetworkTileSource`
    /// even if an embedded/matching pack exists (falling back to the pack only until arrival).
    pub fn with_always_stream_base(mut self, stream: bool) -> Self {
        self.stream_base_from_network = stream;
        if stream {
            self.active_pack = None;
        }
        self
    }

    /// Returns the active [`TileProviderConfig`].
    pub fn provider_config(&self) -> &TileProviderConfig {
        &self.provider
    }

    /// Returns `true` when an exact matching `z=0..=4` pack is bound for the active provider.
    pub fn has_matching_base_pack(&self) -> bool {
        self.active_pack.is_some()
    }

    /// Switches the active Tile API provider, clearing stale network caches and reloading
    /// any matching `base_world_map_<id>.pack` (or falling back to live `z=0..=4` streaming).
    pub fn switch_provider(&mut self, config: TileProviderConfig) {
        self.provider = config.clone();
        if let Some(ref mut net) = self.network {
            net.apply_provider(&config);
        }
        self.active_pack = if self.network.is_none() {
            Some(Arc::clone(&self.fallback_pack))
        } else if self.stream_base_from_network {
            None
        } else {
            Self::find_pack_for_provider(&config, &self.pack_search_dirs, true)
        };
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
    fn begin_frame(&mut self, focus: crate::geodesy::Geodetic, altitude_m: f64) {
        if let Some(ref mut net) = self.network {
            net.begin_frame(focus, altitude_m);
        }
    }

    fn end_frame(&mut self) {
        if let Some(ref mut net) = self.network {
            net.end_frame();
        }
    }

    fn imagery(&mut self, key: TileKey) -> Result<Option<TileImage>, TileError> {
        if let Some(ref pack) = self.active_pack {
            if key.level <= pack.max_level() {
                if let Some(img) = pack.tile(key) {
                    return Ok(Some(img));
                }
            }
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
        if let Some(ref pack) = self.active_pack {
            if let Some(img) = pack.tile(key) {
                return Some(img);
            }
        }
        self.fallback_pack.tile(key)
    }

    fn night_lights(&mut self, key: TileKey) -> Result<Option<TileImage>, TileError> {
        if key.level <= MAX_EMBEDDED_BASE_LEVEL {
            return Ok(None);
        }
        if let Some(ref mut net) = self.network {
            if let Some(img) = net.night_lights(key)? {
                return Ok(Some(img));
            }
        }
        Ok(None)
    }

    fn peek_cached_night_lights(&self, key: TileKey) -> Option<TileImage> {
        if key.level <= MAX_EMBEDDED_BASE_LEVEL {
            return None;
        }
        if let Some(ref net) = self.network {
            return net.peek_cached_night_lights(key);
        }
        None
    }

    fn elevation(&mut self, key: TileKey) -> Result<Option<TileElevation>, TileError> {
        if key.level > self.max_elevation_level {
            return Ok(None);
        }
        if let Some(ref mut net) = self.network {
            return net.elevation(key);
        }
        Ok(None)
    }

    fn peek_cached_elevation(&self, key: TileKey) -> Option<TileElevation> {
        if key.level > self.max_elevation_level {
            return None;
        }
        if let Some(ref net) = self.network {
            return net.peek_cached_elevation(key);
        }
        None
    }

    fn set_provider(&mut self, config: &TileProviderConfig) -> bool {
        self.switch_provider(config.clone());
        true
    }

    fn set_tile_ready_callback(&mut self, cb: Option<Arc<dyn Fn() + Send + Sync>>) {
        self.set_on_tile_ready(cb);
    }
}
