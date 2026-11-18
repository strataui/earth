//! Offline/CI baker generating:
//! 1. Multi-resolution `.ssky` skybox cubemap packs (`manifest.json` for rolling `assets` releases).
//! 2. `SEBM0001` `z=0..=4` base world map packs (`base_world_map_<id>.pack`) baked directly from
//!    any configured Tile API provider preset or custom XYZ/TMS/WMTS/QuadKey URL so `z <= 4`
//!    matches `z >= 5` seamlessly with zero color/brightness jump.
//!
//! Examples:
//!   # Bake skybox cubemaps + ArcGIS/OSM/Topo z=0..=4 world map packs:
//!   cargo run --release -p strata-earth --example bake_assets -- \
//!     --out dist/assets --tile-packs arcgis_world_imagery,osm,arcgis_world_topo,arcgis_world_street
//!
//!   # Re-bake bundled default assets/base_world_map.pack from live ArcGIS World_Imagery:
//!   cargo run --release -p strata-earth --example bake_assets -- \
//!     --skip-skybox --tile-packs arcgis_world_imagery --write-default-basemap
//!
//!   # Bake a custom Tile API endpoint:
//!   cargo run --release -p strata-earth --example bake_assets -- \
//!     --skip-skybox --tile-id my_map --tile-url "https://tiles.example.com/{z}/{x}/{y}.png?key={api_key}" \
//!     --api-key SECRET --header "Referer:https://example.com"

use std::collections::HashMap;
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use image::codecs::jpeg::JpegEncoder;
use image::{DynamicImage, ImageFormat, RgbaImage};
use strata_earth::assets_cdn::{http_get_bytes, write_file_atomic};
use strata_earth::{
    blend_composite_cubemap, encode_sebm_pack, encode_ssky_jpeg, format_tile_url,
    project_hips_allsky_to_cubemap, project_hips_tiles_to_cubemap, sha256_hex, AssetManifest,
    AssetManifestEntry, BaseWorldMapPack, HipsFrame, HttpTransport, SpaceSkyboxQuality,
    SpaceSkyboxSurvey, StandardHttpTransport, TileKey, TileProviderConfig, TileProviderPreset,
};

struct BakeArgs {
    out_dir: PathBuf,
    tiles_cache: PathBuf,
    surveys: Vec<SpaceSkyboxSurvey>,
    qualities: Vec<SpaceSkyboxQuality>,
    skip_skybox: bool,
    tile_providers: Vec<TileProviderConfig>,
    tile_max_level: u8,
    tile_jpeg_quality: u8,
    write_default_basemap: bool,
    default_basemap_path: PathBuf,
}

fn parse_args() -> BakeArgs {
    let mut out_dir = PathBuf::from("dist/assets");
    let mut tiles_cache = PathBuf::from("target/hips_bake_cache");
    let mut surveys = vec![
        SpaceSkyboxSurvey::CompositeOptical,
        SpaceSkyboxSurvey::GaiaDr3Optical,
        SpaceSkyboxSurvey::AllWiseInfrared,
    ];
    let mut qualities = vec![
        SpaceSkyboxQuality::Low,
        SpaceSkyboxQuality::Medium,
        SpaceSkyboxQuality::High,
        SpaceSkyboxQuality::Ultra,
    ];
    let mut skip_skybox = false;
    let mut preset_ids: Vec<String> = Vec::new();
    let mut custom_url: Option<String> = None;
    let mut custom_id = "custom".to_string();
    let mut custom_subdomains: Vec<String> = Vec::new();
    let mut api_key: Option<String> = None;
    let mut headers: Vec<(String, String)> = Vec::new();
    let mut tile_max_level: u8 = 4;
    let mut tile_jpeg_quality: u8 = 88;
    let mut write_default_basemap = false;
    let mut default_basemap_path = PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/assets/base_world_map.pack"
    ));

    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "--out" if i + 1 < args.len() => {
                out_dir = PathBuf::from(&args[i + 1]);
                i += 2;
            }
            "--tiles-cache" if i + 1 < args.len() => {
                tiles_cache = PathBuf::from(&args[i + 1]);
                i += 2;
            }
            "--surveys" if i + 1 < args.len() => {
                surveys = args[i + 1]
                    .split(',')
                    .filter_map(|s| match s.trim().to_ascii_lowercase().as_str() {
                        "composite" => Some(SpaceSkyboxSurvey::CompositeOptical),
                        "dss2" => Some(SpaceSkyboxSurvey::Dss2Optical),
                        "allwise" => Some(SpaceSkyboxSurvey::AllWiseInfrared),
                        "gaia" => Some(SpaceSkyboxSurvey::GaiaDr3Optical),
                        _ => None,
                    })
                    .collect();
                i += 2;
            }
            "--qualities" if i + 1 < args.len() => {
                qualities = args[i + 1]
                    .split(',')
                    .filter_map(|s| match s.trim().to_ascii_lowercase().as_str() {
                        "low" | "512" => Some(SpaceSkyboxQuality::Low),
                        "medium" | "med" | "1k" => Some(SpaceSkyboxQuality::Medium),
                        "high" | "2k" => Some(SpaceSkyboxQuality::High),
                        "ultra" | "4k" => Some(SpaceSkyboxQuality::Ultra),
                        _ => None,
                    })
                    .collect();
                i += 2;
            }
            "--include-dss2" => {
                if !surveys.contains(&SpaceSkyboxSurvey::Dss2Optical) {
                    surveys.push(SpaceSkyboxSurvey::Dss2Optical);
                }
                i += 1;
            }
            "--skip-skybox" => {
                skip_skybox = true;
                i += 1;
            }
            "--tile-packs" if i + 1 < args.len() => {
                for tok in args[i + 1].split(',') {
                    let t = tok.trim();
                    if !t.is_empty() {
                        preset_ids.push(t.to_string());
                    }
                }
                i += 2;
            }
            "--tile-url" if i + 1 < args.len() => {
                custom_url = Some(args[i + 1].clone());
                i += 2;
            }
            "--tile-id" if i + 1 < args.len() => {
                custom_id = args[i + 1].clone();
                i += 2;
            }
            "--subdomains" if i + 1 < args.len() => {
                custom_subdomains = args[i + 1]
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect();
                i += 2;
            }
            "--api-key" if i + 1 < args.len() => {
                api_key = Some(args[i + 1].clone());
                i += 2;
            }
            "--header" if i + 1 < args.len() => {
                if let Some((k, v)) = args[i + 1].split_once(':') {
                    headers.push((k.trim().to_string(), v.trim().to_string()));
                }
                i += 2;
            }
            "--tile-max-level" if i + 1 < args.len() => {
                tile_max_level = args[i + 1].parse::<u8>().unwrap_or(4).min(6);
                i += 2;
            }
            "--tile-jpeg-quality" if i + 1 < args.len() => {
                tile_jpeg_quality = args[i + 1].parse::<u8>().unwrap_or(88).clamp(40, 100);
                i += 2;
            }
            "--write-default-basemap" => {
                write_default_basemap = true;
                i += 1;
            }
            "--default-basemap-path" if i + 1 < args.len() => {
                write_default_basemap = true;
                default_basemap_path = PathBuf::from(&args[i + 1]);
                i += 2;
            }
            _ => i += 1,
        }
    }

    let mut tile_providers: Vec<TileProviderConfig> = Vec::new();
    for id in preset_ids {
        let lower = id.to_ascii_lowercase();
        if lower == "all" || lower == "public" {
            for &p in TileProviderPreset::public_presets() {
                let mut cfg = TileProviderConfig::preset(p);
                cfg.api_key = api_key.clone();
                cfg.headers = headers.clone();
                tile_providers.push(cfg);
            }
        } else if lower == "default" {
            let mut cfg = TileProviderConfig::preset(TileProviderPreset::ArcGisWorldImagery);
            cfg.api_key = api_key.clone();
            cfg.headers = headers.clone();
            tile_providers.push(cfg);
        } else if let Some(preset) = TileProviderPreset::from_id(&id) {
            let mut cfg = TileProviderConfig::preset(preset);
            cfg.api_key = api_key.clone();
            cfg.headers = headers.clone();
            tile_providers.push(cfg);
        } else {
            panic!("Unknown --tile-packs preset '{id}'. Use preset IDs (e.g. arcgis_world_imagery, osm, arcgis_world_topo, arcgis_world_street) or 'public'.");
        }
    }
    if let Some(tpl) = custom_url {
        let mut cfg = TileProviderConfig::custom(custom_id, tpl);
        if !custom_subdomains.is_empty() {
            cfg.subdomains = custom_subdomains;
        }
        if let Some(ref k) = api_key {
            cfg.api_key = Some(k.clone());
        }
        cfg.headers = headers;
        tile_providers.push(cfg);
    }

    BakeArgs {
        out_dir,
        tiles_cache,
        surveys,
        qualities,
        skip_skybox,
        tile_providers,
        tile_max_level,
        tile_jpeg_quality,
        write_default_basemap,
        default_basemap_path,
    }
}

fn fetch_cached_url(cache_file: &Path, url: &str) -> Result<Vec<u8>, String> {
    if let Ok(b) = std::fs::read(cache_file) {
        if !b.is_empty() {
            return Ok(b);
        }
    }
    let bytes = http_get_bytes(url, Duration::from_secs(30), 30 * 1024 * 1024)?;
    let _ = write_file_atomic(cache_file, &bytes);
    Ok(bytes)
}

fn load_allsky(cache_dir: &Path, id: &str, url: &str) -> Result<DynamicImage, String> {
    let ext = if url.ends_with(".png") { "png" } else { "jpg" };
    let path = cache_dir.join(format!("{id}_allsky.{ext}"));
    let bytes = fetch_cached_url(&path, url)?;
    image::load_from_memory(&bytes).map_err(|e| format!("decode allsky {id}: {e}"))
}

fn load_hips_order_tiles(
    cache_dir: &Path,
    id: &str,
    base_url: &str,
    ext: &str,
    order: u8,
) -> Result<HashMap<u64, RgbaImage>, String> {
    let n_tiles = 12usize * (1usize << (2 * usize::from(order)));
    let next_idx = Arc::new(Mutex::new(0usize));
    let out_tiles = Arc::new(Mutex::new(HashMap::with_capacity(n_tiles)));
    let workers = 12usize;

    std::thread::scope(|s| {
        for _ in 0..workers {
            let next_ref = Arc::clone(&next_idx);
            let out_ref = Arc::clone(&out_tiles);
            s.spawn(move || loop {
                let ipix = {
                    let mut g = next_ref.lock().unwrap();
                    let cur = *g;
                    if cur >= n_tiles {
                        break;
                    }
                    *g += 1;
                    cur
                };
                let dir = (ipix / 10_000) * 10_000;
                let url = format!("{base_url}/Norder{order}/Dir{dir}/Npix{ipix}.{ext}");
                let local = cache_dir
                    .join(id)
                    .join(format!("Norder{order}"))
                    .join(format!("Npix{ipix}.{ext}"));
                if let Ok(bytes) = fetch_cached_url(&local, &url) {
                    if let Ok(img) = image::load_from_memory(&bytes) {
                        out_ref.lock().unwrap().insert(ipix as u64, img.to_rgba8());
                    }
                }
            });
        }
    });

    Ok(Arc::try_unwrap(out_tiles).unwrap().into_inner().unwrap())
}

fn build_faces_for(
    survey: SpaceSkyboxSurvey,
    quality: SpaceSkyboxQuality,
    cache_dir: &Path,
    include_dss2_in_composite: bool,
) -> Result<[Arc<Vec<u8>>; 6], String> {
    let face_size = quality.face_size();
    let order_needed = match quality {
        SpaceSkyboxQuality::Off => return Err("cannot bake Off".into()),
        SpaceSkyboxQuality::Low | SpaceSkyboxQuality::Medium => None,
        SpaceSkyboxQuality::High => Some(2u8),
        SpaceSkyboxQuality::Ultra => Some(3u8),
    };

    let build_single =
        |id: &str, base: &str, ext: &str, frame: HipsFrame| -> Result<[Arc<Vec<u8>>; 6], String> {
            if let Some(order) = order_needed {
                let tiles = load_hips_order_tiles(cache_dir, id, base, ext, order)?;
                Ok(project_hips_tiles_to_cubemap(
                    order, frame, face_size, &tiles,
                ))
            } else {
                let allsky = load_allsky(cache_dir, id, &format!("{base}/Norder3/Allsky.{ext}"))?;
                Ok(project_hips_allsky_to_cubemap(&allsky, frame, face_size))
            }
        };

    match survey {
        SpaceSkyboxSurvey::GaiaDr3Optical => build_single(
            "gaia_dr3",
            "https://skies.esac.esa.int/Gaia/dr3/RGB",
            "png",
            HipsFrame::Galactic,
        ),
        SpaceSkyboxSurvey::AllWiseInfrared => build_single(
            "allwise",
            "https://irsa.ipac.caltech.edu/data/hips/CDS/AllWISE/RGB-W4-W2-W1",
            "jpg",
            HipsFrame::EquatorialIcrs,
        ),
        SpaceSkyboxSurvey::Dss2Optical => build_single(
            "dss2",
            "https://irsa.ipac.caltech.edu/data/hips/CDS/DSS2/color",
            "jpg",
            HipsFrame::EquatorialIcrs,
        ),
        SpaceSkyboxSurvey::CompositeOptical => {
            let gaia = build_single(
                "gaia_dr3",
                "https://skies.esac.esa.int/Gaia/dr3/RGB",
                "png",
                HipsFrame::Galactic,
            )?;
            let wise = build_single(
                "allwise",
                "https://irsa.ipac.caltech.edu/data/hips/CDS/AllWISE/RGB-W4-W2-W1",
                "jpg",
                HipsFrame::EquatorialIcrs,
            )
            .ok();
            let dss2 = if include_dss2_in_composite {
                build_single(
                    "dss2",
                    "https://irsa.ipac.caltech.edu/data/hips/CDS/DSS2/color",
                    "jpg",
                    HipsFrame::EquatorialIcrs,
                )
                .ok()
            } else {
                None
            };
            Ok(blend_composite_cubemap(&gaia, dss2.as_ref(), wise.as_ref()))
        }
    }
}

fn license_for(survey: SpaceSkyboxSurvey) -> &'static str {
    match survey {
        SpaceSkyboxSurvey::GaiaDr3Optical => "ESA/Gaia/DPAC (attribution)",
        SpaceSkyboxSurvey::AllWiseInfrared => "IPAC/NASA WISE (acknowledgment)",
        SpaceSkyboxSurvey::Dss2Optical => "STScI/NASA DSS2 & CDS (non-commercial/academic)",
        SpaceSkyboxSurvey::CompositeOptical => "ESA/Gaia/DPAC + IPAC/NASA WISE",
    }
}

fn bake_tile_pack_for_provider(
    provider: &TileProviderConfig,
    max_level: u8,
    jpeg_quality: u8,
    cache_dir: &Path,
) -> Result<Vec<u8>, String> {
    let mut keys = Vec::new();
    for z in 0..=max_level {
        let n = 1u32 << z;
        for y in 0..n {
            for x in 0..n {
                keys.push(TileKey::new(z, x, y));
            }
        }
    }

    let transport = Arc::new(
        StandardHttpTransport::new()
            .with_timeout_ms(20_000)
            .with_headers(provider.headers.clone()),
    );
    let ns = provider.imagery_cache_namespace();
    let next_idx = Arc::new(Mutex::new(0usize));
    let baked_entries: Arc<Mutex<Vec<(TileKey, Vec<u8>)>>> =
        Arc::new(Mutex::new(Vec::with_capacity(keys.len())));
    let errors: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let workers = 10usize;

    std::thread::scope(|s| {
        let ns_ref = ns.as_str();
        for worker_id in 0..workers {
            let next_ref = Arc::clone(&next_idx);
            let out_ref = Arc::clone(&baked_entries);
            let err_ref = Arc::clone(&errors);
            let tr = Arc::clone(&transport);
            let keys_ref = &keys;
            s.spawn(move || {
                let mut sub_counter = worker_id;
                loop {
                    let idx = {
                        let mut g = next_ref.lock().unwrap();
                        let cur = *g;
                        if cur >= keys_ref.len() {
                            break;
                        }
                        *g += 1;
                        cur
                    };
                    let key = keys_ref[idx];
                    let url = format_tile_url(
                        &provider.imagery_url_template,
                        key,
                        &provider.subdomains,
                        &mut sub_counter,
                        provider.api_key.as_deref(),
                    );
                    let cache_path = cache_dir
                        .join("tile_packs")
                        .join(ns_ref)
                        .join(format!("{}/{}/{}.raw", key.level, key.x, key.y));

                    let raw_bytes = match std::fs::read(&cache_path) {
                        Ok(b) if !b.is_empty() => Ok(b),
                        _ => tr
                            .fetch(&url)
                            .map_err(|e| format!("{key:?} ({url}): {e:?}"))
                            .inspect(|b| {
                                let _ = write_file_atomic(&cache_path, b);
                            }),
                    };

                    let raw = match raw_bytes {
                        Ok(b) => b,
                        Err(err) => {
                            err_ref.lock().unwrap().push(err);
                            continue;
                        }
                    };

                    let dyn_img = match image::load_from_memory(&raw) {
                        Ok(d) => d,
                        Err(e) => {
                            err_ref
                                .lock()
                                .unwrap()
                                .push(format!("{key:?} decode: {e}"));
                            continue;
                        }
                    };
                    let rgba = dyn_img.to_rgba8();
                    let has_alpha = rgba.pixels().any(|p| p[3] < 255);
                    let mut encoded = Vec::new();
                    if has_alpha {
                        if let Err(e) = DynamicImage::ImageRgba8(rgba)
                            .write_to(&mut Cursor::new(&mut encoded), ImageFormat::Png)
                        {
                            err_ref
                                .lock()
                                .unwrap()
                                .push(format!("{key:?} encode png: {e}"));
                            continue;
                        }
                    } else {
                        let rgb = dyn_img.to_rgb8();
                        let mut enc = JpegEncoder::new_with_quality(&mut encoded, jpeg_quality);
                        if let Err(e) = enc.encode_image(&rgb) {
                            err_ref
                                .lock()
                                .unwrap()
                                .push(format!("{key:?} encode jpeg: {e}"));
                            continue;
                        }
                    }
                    out_ref.lock().unwrap().push((key, encoded));
                }
            });
        }
    });

    let err_list = Arc::try_unwrap(errors).unwrap().into_inner().unwrap();
    if !err_list.is_empty() {
        return Err(format!(
            "failed fetching {} tiles for {} (first error: {})",
            err_list.len(),
            provider.id,
            err_list[0]
        ));
    }

    let mut pairs = Arc::try_unwrap(baked_entries)
        .unwrap()
        .into_inner()
        .unwrap();
    pairs.sort_by_key(|(k, _)| (k.level, k.y, k.x));
    let refs: Vec<(TileKey, &[u8])> = pairs.iter().map(|(k, b)| (*k, b.as_slice())).collect();
    let packed = encode_sebm_pack(&refs);
    BaseWorldMapPack::from_bytes(packed.clone())
        .map_err(|e| format!("self-verify baked pack {}: {e:?}", provider.id))?;
    Ok(packed)
}

fn main() {
    let args = parse_args();
    std::fs::create_dir_all(&args.out_dir).expect("create out_dir");
    std::fs::create_dir_all(&args.tiles_cache).expect("create tiles_cache");
    let include_dss2 = args.surveys.contains(&SpaceSkyboxSurvey::Dss2Optical);

    let mut manifest = AssetManifest {
        schema_version: 1,
        entries: HashMap::new(),
    };

    if !args.skip_skybox {
        for &survey in &args.surveys {
            for &quality in &args.qualities {
                if quality == SpaceSkyboxQuality::Off {
                    continue;
                }
                let filename = survey.cdn_filename(quality);
                println!(
                    "[bake_assets] building {filename} ({}x{} per face)...",
                    quality.face_size(),
                    quality.face_size()
                );
                let faces = build_faces_for(survey, quality, &args.tiles_cache, include_dss2)
                    .expect("bake cubemap faces");
                let packed = encode_ssky_jpeg(quality.face_size(), survey, quality, &faces, 92)
                    .expect("encode ssky");
                let sha = sha256_hex(&packed);
                let bytes_len = packed.len() as u64;
                write_file_atomic(&args.out_dir.join(&filename), &packed).expect("write ssky");
                println!("  -> wrote {filename} ({bytes_len} bytes, sha256={sha})");
                manifest.entries.insert(
                    filename.clone(),
                    AssetManifestEntry {
                        key: filename.clone(),
                        filename,
                        version: 1,
                        sha256: Some(sha),
                        bytes: bytes_len,
                        legacy_filename: None,
                        license: Some(license_for(survey).to_string()),
                    },
                );
            }
        }
    }

    let mut first_baked_pack: Option<Vec<u8>> = None;
    for provider in &args.tile_providers {
        let filename = format!("base_world_map_{}.pack", provider.id);
        println!(
            "[bake_assets] baking {filename} (z=0..={}, template={})...",
            args.tile_max_level, provider.imagery_url_template
        );
        let packed = bake_tile_pack_for_provider(
            provider,
            args.tile_max_level,
            args.tile_jpeg_quality,
            &args.tiles_cache,
        )
        .expect("bake tile base pack");
        let sha = sha256_hex(&packed);
        let bytes_len = packed.len() as u64;
        write_file_atomic(&args.out_dir.join(&filename), &packed).expect("write base pack");
        println!("  -> wrote {filename} ({bytes_len} bytes, sha256={sha})");
        if provider.id == TileProviderPreset::ArcGisWorldImagery.id() || first_baked_pack.is_none()
        {
            first_baked_pack = Some(packed);
        }
        manifest.entries.insert(
            filename.clone(),
            AssetManifestEntry {
                key: filename.clone(),
                filename,
                version: 1,
                sha256: Some(sha),
                bytes: bytes_len,
                legacy_filename: None,
                license: Some(format!("Tile API pack: {}", provider.display_name)),
            },
        );
    }

    if args.write_default_basemap {
        let packed = first_baked_pack.unwrap_or_else(|| {
            let def = TileProviderConfig::preset(TileProviderPreset::ArcGisWorldImagery);
            bake_tile_pack_for_provider(
                &def,
                args.tile_max_level,
                args.tile_jpeg_quality,
                &args.tiles_cache,
            )
            .expect("bake default ArcGIS basemap pack")
        });
        write_file_atomic(&args.default_basemap_path, &packed)
            .expect("write default base_world_map.pack");
        println!(
            "[bake_assets] updated embedded default pack -> {} ({} bytes)",
            args.default_basemap_path.display(),
            packed.len()
        );
    }

    let manifest_json = manifest.to_json_string();
    write_file_atomic(&args.out_dir.join("manifest.json"), manifest_json.as_bytes())
        .expect("write manifest.json");
    println!(
        "[bake_assets] complete -> {}",
        args.out_dir.join("manifest.json").display()
    );
}
