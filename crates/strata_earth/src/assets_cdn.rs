//! Remote asset delivery, SHA-256 verification, and persistent disk caching.
//!
//! Supports permanent bottom-pinned GitHub Releases (`assets` for latest unversioned
//! flat filenames + `manifest.json`, and `assets-legacy` for superseded `-v1`/`-v2`
//! files) fronted by an optional CDN base URL (defaulting to
//! `https://cdn.strata.larvance.com/earth`).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crate::json::Json;

/// Default public CDN base URL fronting the rolling `assets` and `assets-legacy` releases.
pub const DEFAULT_ASSET_CDN_BASE: &str = "https://cdn.strata.larvance.com/earth";

/// Default GitHub repository slug (`owner/repo`) hosting `assets` and `assets-legacy`.
pub const DEFAULT_ASSET_GITHUB_REPO: &str = "strataui/earth";

/// Rolling release tag holding the latest unversioned asset files and `manifest.json`.
pub const DEFAULT_ASSETS_RELEASE_TAG: &str = "assets";

/// Rolling release tag holding archived older versions with `-vN` suffixes.
pub const DEFAULT_ASSETS_LEGACY_TAG: &str = "assets-legacy";

/// Configuration controlling remote asset resolution, mirrors, and local disk caching.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AssetSourceConfig {
    /// Primary CDN base URL (`None` disables custom CDN and uses GitHub Releases directly).
    pub primary_cdn_base: Option<String>,
    /// GitHub `owner/repo` fallback host for rolling `assets` and `assets-legacy` releases.
    pub github_repo: String,
    /// Release tag for latest assets (defaults to `"assets"`).
    pub assets_tag: String,
    /// Release tag for archived `-vN` assets (defaults to `"assets-legacy"`).
    pub legacy_tag: String,
    /// Optional explicit cache directory; `None` resolves to the OS user cache directory.
    pub cache_dir: Option<PathBuf>,
    /// Allows automatic background downloads when cache misses (`true` by default).
    pub auto_download: bool,
    /// Falls back directly to NASA/IPAC IRSA and ESA HiPS if CDN/Release packs are not published yet.
    pub allow_direct_hips_fallback: bool,
    /// Disables all network fetches (`true` when `STRATA_EARTH_OFFLINE` is set).
    pub offline: bool,
}

impl Default for AssetSourceConfig {
    fn default() -> Self {
        let env_cdn = std::env::var("STRATA_EARTH_CDN_URL")
            .ok()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty());
        Self {
            primary_cdn_base: env_cdn.or_else(|| Some(DEFAULT_ASSET_CDN_BASE.to_string())),
            github_repo: DEFAULT_ASSET_GITHUB_REPO.to_string(),
            assets_tag: DEFAULT_ASSETS_RELEASE_TAG.to_string(),
            legacy_tag: DEFAULT_ASSETS_LEGACY_TAG.to_string(),
            cache_dir: None,
            auto_download: true,
            allow_direct_hips_fallback: true,
            offline: std::env::var_os("STRATA_EARTH_OFFLINE").is_some(),
        }
    }
}

impl AssetSourceConfig {
    /// Overrides the primary CDN URL base (passing `None` uses GitHub Releases directly).
    pub fn with_cdn_base(mut self, cdn_base: Option<impl Into<String>>) -> Self {
        self.primary_cdn_base = cdn_base.map(Into::into);
        self
    }

    /// Overrides the local disk cache directory.
    pub fn with_cache_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.cache_dir = Some(dir.into());
        self
    }

    /// Resolves the active disk cache directory.
    pub fn resolved_cache_dir(&self) -> PathBuf {
        if let Some(ref dir) = self.cache_dir {
            return dir.clone();
        }
        if let Some(xdg) = std::env::var_os("XDG_CACHE_HOME") {
            if !xdg.is_empty() {
                return PathBuf::from(xdg).join("strata_earth").join("assets");
            }
        }
        if let Some(home) = std::env::var_os("HOME") {
            if !home.is_empty() {
                return PathBuf::from(home)
                    .join(".cache")
                    .join("strata_earth")
                    .join("assets");
            }
        }
        std::env::temp_dir().join("strata_earth").join("assets")
    }

    /// Returns ordered candidate URLs for `filename`, including CDN and GitHub rolling releases.
    pub fn candidate_urls_for(&self, filename: &str, legacy_filename: Option<&str>) -> Vec<String> {
        let mut urls = Vec::new();
        let file = filename.trim_start_matches('/');
        if let Some(ref base) = self.primary_cdn_base {
            let trimmed = base.trim_end_matches('/');
            if !trimmed.is_empty() {
                if trimmed.ends_with(&format!("/{}", self.assets_tag)) {
                    urls.push(format!("{trimmed}/{file}"));
                } else {
                    urls.push(format!("{trimmed}/{}/{file}", self.assets_tag));
                }
            }
        }
        let repo = self.github_repo.trim_matches('/');
        if !repo.is_empty() {
            urls.push(format!(
                "https://github.com/{repo}/releases/download/{}/{file}",
                self.assets_tag
            ));
        }

        if let Some(legacy) = legacy_filename {
            let legacy = legacy.trim_start_matches('/');
            if let Some(ref base) = self.primary_cdn_base {
                let trimmed = base.trim_end_matches('/');
                if !trimmed.is_empty() {
                    let root = trimmed
                        .strip_suffix(&format!("/{}", self.assets_tag))
                        .unwrap_or(trimmed);
                    urls.push(format!("{root}/{}/{legacy}", self.legacy_tag));
                }
            }
            if !repo.is_empty() {
                urls.push(format!(
                    "https://github.com/{repo}/releases/download/{}/{legacy}",
                    self.legacy_tag
                ));
            }
        }
        urls
    }
}

/// One asset record inside `manifest.json`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AssetManifestEntry {
    pub key: String,
    pub filename: String,
    pub version: u32,
    pub sha256: Option<String>,
    pub bytes: u64,
    pub legacy_filename: Option<String>,
    pub license: Option<String>,
}

/// Parsed `manifest.json` describing all assets hosted in the rolling `assets` release.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AssetManifest {
    pub schema_version: u32,
    pub entries: HashMap<String, AssetManifestEntry>,
}

impl AssetManifest {
    /// Parses a JSON `manifest.json` payload into [`AssetManifest`].
    pub fn from_json_str(raw: &str) -> Result<Self, String> {
        let root = Json::parse(raw)?;
        let Json::Object(ref root_entries) = root else {
            return Err("manifest root must be a JSON object".to_string());
        };
        let schema_version = root
            .get("schema_version")
            .and_then(Json::as_f64)
            .unwrap_or(1.0) as u32;
        let mut entries = HashMap::new();
        let _ = root_entries;
        if let Some(Json::Object(assets_entries)) = root.get("assets") {
            for (key, val) in assets_entries {
                if !matches!(val, Json::Object(_)) {
                    continue;
                }
                let filename = val
                    .get("filename")
                    .and_then(Json::as_str)
                    .unwrap_or(key.as_str())
                    .to_string();
                let version = val
                    .get("version")
                    .and_then(Json::as_f64)
                    .unwrap_or(1.0)
                    .max(1.0) as u32;
                let sha256 = val
                    .get("sha256")
                    .and_then(Json::as_str)
                    .map(|s| s.to_ascii_lowercase());
                let bytes = val.get("bytes").and_then(Json::as_f64).unwrap_or(0.0).max(0.0) as u64;
                let legacy_filename = val
                    .get("legacy_filename")
                    .and_then(Json::as_str)
                    .map(ToOwned::to_owned);
                let license = val
                    .get("license")
                    .and_then(Json::as_str)
                    .map(ToOwned::to_owned);
                entries.insert(
                    key.clone(),
                    AssetManifestEntry {
                        key: key.clone(),
                        filename,
                        version,
                        sha256,
                        bytes,
                        legacy_filename,
                        license,
                    },
                );
            }
        }
        Ok(Self {
            schema_version,
            entries,
        })
    }

    /// Serializes the manifest cleanly into formatted JSON.
    pub fn to_json_string(&self) -> String {
        let mut keys: Vec<&String> = self.entries.keys().collect();
        keys.sort();
        let mut out = format!("{{\n  \"schema_version\": {},\n  \"assets\": {{", self.schema_version.max(1));
        for (idx, k) in keys.iter().enumerate() {
            let e = &self.entries[*k];
            if idx > 0 {
                out.push(',');
            }
            out.push_str(&format!("\n    \"{}\": {{", escape_json_str(&e.key)));
            out.push_str(&format!("\n      \"filename\": \"{}\",", escape_json_str(&e.filename)));
            out.push_str(&format!("\n      \"version\": {},", e.version));
            out.push_str(&format!("\n      \"bytes\": {}", e.bytes));
            if let Some(ref sha) = e.sha256 {
                out.push_str(&format!(",\n      \"sha256\": \"{}\"", escape_json_str(sha)));
            }
            if let Some(ref leg) = e.legacy_filename {
                out.push_str(&format!(",\n      \"legacy_filename\": \"{}\"", escape_json_str(leg)));
            }
            if let Some(ref lic) = e.license {
                out.push_str(&format!(",\n      \"license\": \"{}\"", escape_json_str(lic)));
            }
            out.push_str("\n    }");
        }
        if !keys.is_empty() {
            out.push('\n');
        }
        out.push_str("  }\n}\n");
        out
    }
}

fn escape_json_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c => out.push(c),
        }
    }
    out
}

/// Inserts `-v{version}` right before the file extension (e.g. `foo.ssky` -> `foo-v1.ssky`).
pub fn versioned_legacy_filename(filename: &str, version: u32) -> String {
    match filename.rsplit_once('.') {
        Some((stem, ext)) => format!("{stem}-v{version}.{ext}"),
        None => format!("{filename}-v{version}"),
    }
}

/// Computes lowercase hexadecimal SHA-256 digest of `data` in pure Rust.
pub fn sha256_hex(data: &[u8]) -> String {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
    ];
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
    ];
    let bit_len = (data.len() as u64).wrapping_mul(8);
    let mut padded = Vec::with_capacity(((data.len() + 9 + 63) / 64) * 64);
    padded.extend_from_slice(data);
    padded.push(0x80);
    while (padded.len() % 64) != 56 {
        padded.push(0);
    }
    padded.extend_from_slice(&bit_len.to_be_bytes());

    for chunk in padded.chunks_exact(64) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([
                chunk[i * 4],
                chunk[i * 4 + 1],
                chunk[i * 4 + 2],
                chunk[i * 4 + 3],
            ]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = h;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);

            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }
        let next = [a, b, c, d, e, f, g, hh];
        for (dst, src) in h.iter_mut().zip(next.iter()) {
            *dst = dst.wrapping_add(*src);
        }
    }

    let mut hex = String::with_capacity(64);
    for word in h {
        hex.push_str(&format!("{word:08x}"));
    }
    hex
}

/// Atomically writes `bytes` to `target_path` using a sibling temporary file + rename.
pub fn write_file_atomic(target_path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    if let Some(parent) = target_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let file_name = target_path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("asset");
    let tmp_path = target_path.with_file_name(format!(".{file_name}.{stamp}.tmp"));
    std::fs::write(&tmp_path, bytes)?;
    std::fs::rename(&tmp_path, target_path)
}

/// Performs an HTTP GET with redirect support and clean body size bounds.
pub fn http_get_bytes(url: &str, timeout: Duration, max_bytes: usize) -> Result<Vec<u8>, String> {
    let agent = ureq::AgentBuilder::new()
        .timeout_connect(Duration::from_secs(8))
        .timeout_read(timeout)
        .redirects(5)
        .build();
    let resp = agent
        .get(url)
        .set("User-Agent", "StrataEarth/0.1 (+https://github.com/strataui/earth)")
        .call()
        .map_err(|e| format!("GET {url}: {e}"))?;
    let mut reader = resp.into_reader();
    let mut buf = Vec::new();
    let mut chunk = [0u8; 32 * 1024];
    loop {
        let n = std::io::Read::read(&mut reader, &mut chunk)
            .map_err(|e| format!("read {url}: {e}"))?;
        if n == 0 {
            break;
        }
        if buf.len() + n > max_bytes {
            return Err(format!("asset {url} exceeded limit ({max_bytes} bytes)"));
        }
        buf.extend_from_slice(&chunk[..n]);
    }
    Ok(buf)
}

/// Persistent asset client caching downloaded packs on disk with CDN/Release fallback.
#[derive(Debug, Default)]
pub struct AssetManager {
    manifest_cache: Mutex<Option<(SystemTime, Arc<AssetManifest>)>>,
}

impl AssetManager {
    pub fn new() -> Self {
        Self::default()
    }

    /// Reads an asset from disk cache or downloads it from `cfg` candidate URLs.
    pub fn fetch_or_cache(
        &self,
        cfg: &AssetSourceConfig,
        filename: &str,
        legacy_filename: Option<&str>,
        expected_sha256: Option<&str>,
        timeout: Duration,
        max_bytes: usize,
    ) -> Result<Vec<u8>, String> {
        let cache_path = cfg.resolved_cache_dir().join(filename);
        if let Ok(bytes) = std::fs::read(&cache_path) {
            if !bytes.is_empty() && sha_matches(&bytes, expected_sha256) {
                return Ok(bytes);
            }
        }
        if cfg.offline || !cfg.auto_download {
            return Err(format!("asset {filename} not in local cache (offline/auto_download=false)"));
        }

        let urls = cfg.candidate_urls_for(filename, legacy_filename);
        let mut last_err = String::from("no candidate asset URLs configured");
        for url in urls {
            match http_get_bytes(&url, timeout, max_bytes) {
                Ok(bytes) if !bytes.is_empty() => {
                    if !sha_matches(&bytes, expected_sha256) {
                        last_err = format!("sha256 mismatch at {url}");
                        continue;
                    }
                    let _ = write_file_atomic(&cache_path, &bytes);
                    return Ok(bytes);
                }
                Ok(_) => last_err = format!("empty response at {url}"),
                Err(e) => last_err = e,
            }
        }
        Err(last_err)
    }

    /// Loads `manifest.json` from cache/network (short TTL) if available.
    pub fn fetch_manifest(&self, cfg: &AssetSourceConfig) -> Option<Arc<AssetManifest>> {
        if let Ok(guard) = self.manifest_cache.lock() {
            if let Some((ts, ref m)) = *guard {
                if ts.elapsed().unwrap_or(Duration::from_secs(9999)) < Duration::from_secs(1800) {
                    return Some(Arc::clone(m));
                }
            }
        }
        let manifest_path = cfg.resolved_cache_dir().join("manifest.json");
        if let Ok(raw) = std::fs::read_to_string(&manifest_path) {
            if let Ok(meta) = std::fs::metadata(&manifest_path) {
                if let Ok(modified) = meta.modified() {
                    if modified.elapsed().unwrap_or(Duration::from_secs(9999))
                        < Duration::from_secs(3600)
                    {
                        if let Ok(parsed) = AssetManifest::from_json_str(&raw) {
                            let arc = Arc::new(parsed);
                            if let Ok(mut g) = self.manifest_cache.lock() {
                                *g = Some((SystemTime::now(), Arc::clone(&arc)));
                            }
                            return Some(arc);
                        }
                    }
                }
            }
        }
        if cfg.offline || !cfg.auto_download {
            return std::fs::read_to_string(&manifest_path)
                .ok()
                .and_then(|s| AssetManifest::from_json_str(&s).ok())
                .map(Arc::new);
        }
        for url in cfg.candidate_urls_for("manifest.json", None) {
            if let Ok(bytes) = http_get_bytes(&url, Duration::from_secs(8), 2 * 1024 * 1024) {
                if let Ok(raw) = std::str::from_utf8(&bytes) {
                    if let Ok(parsed) = AssetManifest::from_json_str(raw) {
                        let _ = write_file_atomic(&manifest_path, &bytes);
                        let arc = Arc::new(parsed);
                        if let Ok(mut g) = self.manifest_cache.lock() {
                            *g = Some((SystemTime::now(), Arc::clone(&arc)));
                        }
                        return Some(arc);
                    }
                }
            }
        }
        None
    }
}

fn sha_matches(bytes: &[u8], expected: Option<&str>) -> bool {
    match expected {
        Some(exp) if !exp.is_empty() => sha256_hex(bytes) == exp.to_ascii_lowercase(),
        _ => true,
    }
}
