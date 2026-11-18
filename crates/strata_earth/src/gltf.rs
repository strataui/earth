//! Minimal pure-Rust glTF 2.0 `.glb` loader and 2.5D weather particle batch builder.
//!
//! Loads CC0 3D models (`weather_cloud.glb`, `weather_rain.glb`, `weather_snow.glb`)
//! sourced cleanly from Poly Pizza (CC0 Storm cloud), OpenGameArt (CC0 Drops), and
//! Kenney Holiday Kit (CC0 `snowflake-a.glb`), plus custom user-supplied `.glb` bytes.

use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use strata_render::scene3d::{RetainedDynamicMesh, RteTileVertex};

use crate::geodesy::{Geodetic, LocalFrame};
use crate::json::Json;

const CC0_CLOUD_GLB: &[u8] = include_bytes!("../assets/weather_cloud.glb");
const CC0_RAIN_GLB: &[u8] = include_bytes!("../assets/weather_rain.glb");
const CC0_SNOW_GLB: &[u8] = include_bytes!("../assets/weather_snow.glb");

static DEFAULT_CLOUD_MESH: OnceLock<WeatherMesh> = OnceLock::new();
static DEFAULT_RAIN_MESH: OnceLock<WeatherMesh> = OnceLock::new();
static DEFAULT_SNOW_MESH: OnceLock<WeatherMesh> = OnceLock::new();

/// Normalized unit-scale triangle mesh extracted from a glTF 2.0 `.glb` asset.
#[derive(Debug, Clone, PartialEq)]
pub struct WeatherMesh {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub indices: Vec<u32>,
}

impl WeatherMesh {
    fn fallback_octahedron() -> Self {
        let p = [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
        ];
        Self {
            positions: p.to_vec(),
            normals: p.to_vec(),
            indices: vec![
                0, 1, 2, 0, 2, 3, 0, 3, 4, 0, 4, 1, 5, 2, 1, 5, 3, 2, 5, 4, 3, 5, 1, 4,
            ],
        }
    }
}

fn j_field<'a>(obj: &'a [(String, Json)], key: &str) -> Option<&'a Json> {
    obj.iter().find(|(k, _)| k == key).map(|(_, v)| v)
}

fn j_usize(val: Option<&Json>) -> Option<usize> {
    match val? {
        Json::Number(n) if *n >= 0.0 => Some(*n as usize),
        _ => None,
    }
}

/// Parses the first triangle mesh primitive from a glTF 2.0 binary `.glb` buffer.
pub fn parse_glb_mesh(bytes: &[u8]) -> Result<WeatherMesh, String> {
    if bytes.len() < 20 || &bytes[0..4] != b"glTF" {
        return Err("invalid GLB header".to_string());
    }
    let version = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
    if version != 2 {
        return Err(format!("unsupported GLB version {version}"));
    }

    let mut offset = 12usize;
    let mut json_bytes: Option<&[u8]> = None;
    let mut bin_bytes: Option<&[u8]> = None;

    while offset + 8 <= bytes.len() {
        let chunk_len = u32::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ]) as usize;
        let chunk_type = &bytes[offset + 4..offset + 8];
        offset += 8;
        if offset + chunk_len > bytes.len() {
            return Err("truncated GLB chunk".to_string());
        }
        let payload = &bytes[offset..offset + chunk_len];
        if chunk_type == b"JSON" && json_bytes.is_none() {
            json_bytes = Some(payload);
        } else if chunk_type == b"BIN\0" && bin_bytes.is_none() {
            bin_bytes = Some(payload);
        }
        offset += chunk_len;
    }

    let json_text = std::str::from_utf8(json_bytes.ok_or("missing JSON chunk")?)
        .map_err(|e| format!("invalid glTF UTF-8: {e}"))?;
    let bin = bin_bytes.ok_or("missing BIN chunk")?;
    let Json::Object(root) = Json::parse(json_text).map_err(|e| format!("glTF JSON parse: {e}"))?
    else {
        return Err("glTF root not an object".to_string());
    };

    let Some(Json::Array(meshes)) = j_field(&root, "meshes") else {
        return Err("no meshes in glTF".to_string());
    };
    let Some(Json::Array(accessors)) = j_field(&root, "accessors") else {
        return Err("no accessors in glTF".to_string());
    };
    let Some(Json::Array(buffer_views)) = j_field(&root, "bufferViews") else {
        return Err("no bufferViews in glTF".to_string());
    };

    let mesh_obj = match meshes.first() {
        Some(Json::Object(o)) => o,
        _ => return Err("empty meshes array".to_string()),
    };
    let Some(Json::Array(primitives)) = j_field(mesh_obj, "primitives") else {
        return Err("no primitives in mesh".to_string());
    };
    let prim = match primitives.first() {
        Some(Json::Object(o)) => o,
        _ => return Err("empty primitives".to_string()),
    };
    let Some(Json::Object(attrs)) = j_field(prim, "attributes") else {
        return Err("no attributes in primitive".to_string());
    };

    let read_accessor = |acc_idx: usize| -> Result<(&[u8], usize, usize, usize, usize), String> {
        let Some(Json::Object(acc)) = accessors.get(acc_idx) else {
            return Err("invalid accessor index".to_string());
        };
        let bv_idx = j_usize(j_field(acc, "bufferView")).ok_or("accessor missing bufferView")?;
        let Some(Json::Object(bv)) = buffer_views.get(bv_idx) else {
            return Err("invalid bufferView index".to_string());
        };
        let bv_offset = j_usize(j_field(bv, "byteOffset")).unwrap_or(0);
        let acc_offset = j_usize(j_field(acc, "byteOffset")).unwrap_or(0);
        let base = bv_offset + acc_offset;
        let count = j_usize(j_field(acc, "count")).ok_or("accessor missing count")?;
        let comp = j_usize(j_field(acc, "componentType")).ok_or("missing componentType")?;
        let stride = j_usize(j_field(bv, "byteStride")).unwrap_or(0);
        Ok((bin, base, count, comp, stride))
    };

    let pos_idx = j_usize(j_field(attrs, "POSITION")).ok_or("primitive missing POSITION")?;
    let (pbin, pbase, pcount, pcomp, pstride) = read_accessor(pos_idx)?;
    if pcomp != 5126 {
        return Err("POSITION componentType must be f32".to_string());
    }
    let pstep = pstride.max(12);
    let mut positions = Vec::with_capacity(pcount);
    for i in 0..pcount {
        let o = pbase + i * pstep;
        if o + 12 > pbin.len() {
            return Err("POSITION out of bounds".to_string());
        }
        let x = f32::from_le_bytes([pbin[o], pbin[o + 1], pbin[o + 2], pbin[o + 3]]);
        let y = f32::from_le_bytes([pbin[o + 4], pbin[o + 5], pbin[o + 6], pbin[o + 7]]);
        let z = f32::from_le_bytes([pbin[o + 8], pbin[o + 9], pbin[o + 10], pbin[o + 11]]);
        positions.push([x, y, z]);
    }

    let mut normals = if let Some(n_idx) = j_usize(j_field(attrs, "NORMAL")) {
        let (nbin, nbase, ncount, ncomp, nstride) = read_accessor(n_idx)?;
        if ncomp == 5126 && ncount == pcount {
            let nstep = nstride.max(12);
            let mut out = Vec::with_capacity(ncount);
            for i in 0..ncount {
                let o = nbase + i * nstep;
                if o + 12 > nbin.len() {
                    break;
                }
                let x = f32::from_le_bytes([nbin[o], nbin[o + 1], nbin[o + 2], nbin[o + 3]]);
                let y = f32::from_le_bytes([nbin[o + 4], nbin[o + 5], nbin[o + 6], nbin[o + 7]]);
                let z = f32::from_le_bytes([nbin[o + 8], nbin[o + 9], nbin[o + 10], nbin[o + 11]]);
                out.push([x, y, z]);
            }
            out
        } else {
            vec![[0.0, 0.0, 1.0]; pcount]
        }
    } else {
        vec![[0.0, 0.0, 1.0]; pcount]
    };

    let indices = if let Some(i_idx) = j_usize(j_field(prim, "indices")) {
        let (ibin, ibase, icount, icomp, istride) = read_accessor(i_idx)?;
        let mut out = Vec::with_capacity(icount);
        match icomp {
            5123 => {
                let step = istride.max(2);
                for i in 0..icount {
                    let o = ibase + i * step;
                    if o + 2 > ibin.len() {
                        break;
                    }
                    out.push(u16::from_le_bytes([ibin[o], ibin[o + 1]]) as u32);
                }
            }
            5125 => {
                let step = istride.max(4);
                for i in 0..icount {
                    let o = ibase + i * step;
                    if o + 4 > ibin.len() {
                        break;
                    }
                    out.push(u32::from_le_bytes([
                        ibin[o],
                        ibin[o + 1],
                        ibin[o + 2],
                        ibin[o + 3],
                    ]));
                }
            }
            5121 => {
                let step = istride.max(1);
                for i in 0..icount {
                    let o = ibase + i * step;
                    if o >= ibin.len() {
                        break;
                    }
                    out.push(ibin[o] as u32);
                }
            }
            _ => return Err("unsupported index componentType".to_string()),
        }
        out
    } else {
        (0..positions.len() as u32).collect()
    };

    if normals.len() != positions.len() {
        normals.resize(positions.len(), [0.0, 0.0, 1.0]);
    }
    Ok(WeatherMesh {
        positions,
        normals,
        indices,
    })
}

/// Returns the CC0 3D cloud mesh (`weather_cloud.glb`).
pub fn cc0_cloud_mesh() -> &'static WeatherMesh {
    DEFAULT_CLOUD_MESH.get_or_init(|| {
        parse_glb_mesh(CC0_CLOUD_GLB).unwrap_or_else(|_| WeatherMesh::fallback_octahedron())
    })
}

/// Returns the CC0 3D raindrop mesh (`weather_rain.glb`).
pub fn cc0_rain_mesh() -> &'static WeatherMesh {
    DEFAULT_RAIN_MESH.get_or_init(|| {
        parse_glb_mesh(CC0_RAIN_GLB).unwrap_or_else(|_| WeatherMesh::fallback_octahedron())
    })
}

/// Returns the CC0 3D snowflake mesh (`weather_snow.glb`).
pub fn cc0_snow_mesh() -> &'static WeatherMesh {
    DEFAULT_SNOW_MESH.get_or_init(|| {
        parse_glb_mesh(CC0_SNOW_GLB).unwrap_or_else(|_| WeatherMesh::fallback_octahedron())
    })
}

/// Customizable 3D weather configuration (3D orbital cloud map + 2.5D-only clouds, rain, snow).
#[derive(Debug, Clone, PartialEq)]
pub struct WeatherConfig {
    /// Render live equirectangular global cloud map in 3D space (fades out into 2.5D).
    pub global_cloud_map: bool,
    /// Opacity of the 3D global cloud map (`0.0..=1.0`).
    pub global_cloud_opacity: f32,
    /// Resolution tier for the live global cloud map (`1K`, `2K`, `4K`, `8K`).
    pub cloud_resolution: crate::astro::CloudMapResolution,
    /// Custom or default live global cloud-map URL (`https://clouds.matteason.co.uk/...`).
    pub cloud_map_url: String,
    /// Global cloud drift offset speed in radians/second (defaults to `0.0`).
    pub global_cloud_spin_rad_s: f64,

    /// Render 3D CC0 cloud models overhead (ONLY visible when Earth transitions into 2.5D).
    pub clouds_25d: bool,
    /// Number of 2.5D cloud instances around focus.
    pub cloud_count: usize,
    /// Mean cloud base altitude in metres above focus ground.
    pub cloud_altitude_m: f32,
    /// Scale of 2.5D cloud meshes in metres.
    pub cloud_scale_m: f32,
    /// Horizontal wind drift vector `[east_mps, north_mps]` for 2.5D clouds.
    pub cloud_wind_mps: [f32; 2],
    /// Cloud tint RGBA.
    pub cloud_color: [f32; 4],

    /// Render 3D CC0 raindrops (ONLY visible in 2.5D).
    pub rain: bool,
    /// Number of raindrop instances in the local volume.
    pub rain_density: usize,
    /// Raindrop fall speed in m/s.
    pub rain_speed_mps: f32,
    /// Raindrop half-size in metres.
    pub rain_scale_m: f32,
    /// Raindrop RGBA tint.
    pub rain_color: [f32; 4],

    /// Render 3D CC0 snowflakes (ONLY visible in 2.5D).
    pub snow: bool,
    /// Number of snowflake instances in the local volume.
    pub snow_density: usize,
    /// Snowflake fall speed in m/s.
    pub snow_speed_mps: f32,
    /// Snowflake half-size in metres.
    pub snow_scale_m: f32,
    /// Snowflake RGBA tint.
    pub snow_color: [f32; 4],

    /// Elapsed weather animation seconds (drives wind drift and precipitation fall).
    pub weather_time_s: f64,
}

impl Default for WeatherConfig {
    fn default() -> Self {
        let cloud_resolution = crate::astro::CloudMapResolution::default();
        Self {
            global_cloud_map: true,
            global_cloud_opacity: 0.72,
            cloud_resolution,
            cloud_map_url: cloud_resolution.url().to_string(),
            global_cloud_spin_rad_s: 0.0,
            clouds_25d: false,
            cloud_count: 0,
            cloud_altitude_m: 1_800.0,
            cloud_scale_m: 1_200.0,
            cloud_wind_mps: [22.0, 8.0],
            cloud_color: [0.97, 0.98, 1.0, 0.90],
            rain: false,
            rain_density: 260,
            rain_speed_mps: 32.0,
            rain_scale_m: 1.8,
            rain_color: [0.72, 0.86, 1.0, 0.76],
            snow: false,
            snow_density: 220,
            snow_speed_mps: 5.5,
            snow_scale_m: 2.6,
            snow_color: [0.96, 0.98, 1.0, 0.94],
            weather_time_s: 0.0,
        }
    }
}

impl WeatherConfig {
    /// Selects a cloud map resolution preset and updates [`Self::cloud_map_url`].
    pub fn with_cloud_resolution(mut self, res: crate::astro::CloudMapResolution) -> Self {
        self.cloud_resolution = res;
        if self.cloud_map_url.is_empty()
            || crate::astro::CloudMapResolution::is_preset_url(&self.cloud_map_url)
        {
            self.cloud_map_url = res.url().to_string();
        }
        self
    }

    /// Returns a custom URL if set by caller, otherwise the URL for [`Self::cloud_resolution`].
    pub fn resolved_cloud_map_url(&self) -> &str {
        if !self.cloud_map_url.trim().is_empty()
            && !crate::astro::CloudMapResolution::is_preset_url(&self.cloud_map_url)
        {
            &self.cloud_map_url
        } else {
            self.cloud_resolution.url()
        }
    }
}

#[inline]
fn det_hash(seed: u32) -> f64 {
    let mut x = seed.wrapping_mul(747_796_405).wrapping_add(2_891_336_453);
    x = ((x >> ((x >> 28).wrapping_add(4))) ^ x).wrapping_mul(277_803_737);
    let out = (x >> 22) ^ x;
    (out as f64) / (u32::MAX as f64)
}

#[inline]
fn batch_instance_into_local_enu(
    frame: &LocalFrame,
    mesh: &WeatherMesh,
    offset_enu: [f64; 3],
    scale: [f64; 3],
    yaw_rad: f64,
    out_verts: &mut Vec<RteTileVertex>,
    out_inds: &mut Vec<u32>,
) {
    let base_v = out_verts.len() as u32;
    let (sy, cy) = yaw_rad.sin_cos();
    let e = frame.east;
    let n = frame.north;
    let u = frame.up;

    for (p, nrm) in mesh.positions.iter().zip(mesh.normals.iter()) {
        // Model local X/Y horizontal, Z up -> rotate yaw in EN plane
        let lx = p[0] as f64 * scale[0];
        let ly = p[1] as f64 * scale[1];
        let lz = p[2] as f64 * scale[2];
        let rx = lx * cy - ly * sy + offset_enu[0];
        let ry = lx * sy + ly * cy + offset_enu[1];
        let rz = lz + offset_enu[2];

        let world_off = [
            (e.x * rx + n.x * ry + u.x * rz) as f32,
            (e.y * rx + n.y * ry + u.y * rz) as f32,
            (e.z * rx + n.z * ry + u.z * rz) as f32,
        ];

        let nx = nrm[0] as f64 * cy - nrm[1] as f64 * sy;
        let ny = nrm[0] as f64 * sy + nrm[1] as f64 * cy;
        let nz = nrm[2] as f64;
        let wn = [
            (e.x * nx + n.x * ny + u.x * nz) as f32,
            (e.y * nx + n.y * ny + u.y * nz) as f32,
            (e.z * nx + n.z * ny + u.z * nz) as f32,
        ];
        out_verts.push(RteTileVertex::new(world_off, wn, [0.0, 0.0]));
    }
    for &idx in &mesh.indices {
        out_inds.push(base_v + idx);
    }
}

/// Optional custom `.glb` mesh overrides for 2.5D weather.
#[derive(Debug, Clone, Default)]
pub struct CustomWeatherMeshes {
    pub cloud: Option<Arc<WeatherMesh>>,
    pub rain: Option<Arc<WeatherMesh>>,
    pub snow: Option<Arc<WeatherMesh>>,
}

/// Generates batched [`RetainedDynamicMesh`] layers for 2.5D clouds, rain, and snow.
///
/// Returns an empty `Vec` whenever `local_25d_blend <= 0.02` so 3D orbital view stays purely
/// covered by the global satellite cloudmap.
pub fn build_25d_weather_meshes(
    focus: Geodetic,
    camera_range_m: f64,
    local_25d_blend: f32,
    weather: &WeatherConfig,
    custom: Option<&CustomWeatherMeshes>,
    revision: u64,
    _cached_ids: &mut HashMap<u64, u64>,
) -> Vec<RetainedDynamicMesh> {
    if local_25d_blend <= 0.02 {
        return Vec::new();
    }
    let alpha_fade = ((local_25d_blend - 0.02) / 0.48).clamp(0.0, 1.0);
    let frame = LocalFrame::at(focus.lat_deg, focus.lon_deg);
    let anchor_ecef = crate::geodesy::geodetic_to_ecef(focus);
    let anchor = [anchor_ecef.x, anchor_ecef.y, anchor_ecef.z];
    let t = weather.weather_time_s;
    let mut out = Vec::with_capacity(2);

    #[inline]
    fn pick_mesh<'a>(
        custom: Option<&'a Arc<WeatherMesh>>,
        default: &'a WeatherMesh,
    ) -> &'a WeatherMesh {
        custom.map(Arc::as_ref).unwrap_or(default)
    }

    // 1. 2.5D 3D CC0 Raindrops (clouds use the continuous cloud texture layer in fs_cloud, never 3D cloud models)
    if weather.rain && weather.rain_density > 0 {
        let mesh = pick_mesh(custom.and_then(|c| c.rain.as_ref()), cc0_rain_mesh());
        let count = weather.rain_density.min(1_000);
        let mut verts = Vec::with_capacity(count * mesh.positions.len());
        let mut inds = Vec::with_capacity(count * mesh.indices.len());
        let vol_r = (camera_range_m * 0.55).clamp(180.0, 12_000.0);
        let vol_h = (camera_range_m * 0.70).clamp(240.0, 6_000.0);
        let p_scale = (weather.rain_scale_m as f64) * (camera_range_m / 1_500.0).clamp(0.4, 8.0);
        for i in 0..count as u32 {
            let h0 = det_hash(i * 29 + 101);
            let h1 = det_hash(i * 29 + 102);
            let h2 = det_hash(i * 29 + 103);
            let east = (h0 - 0.5) * 2.0 * vol_r;
            let north = (h1 - 0.5) * 2.0 * vol_r;
            let up = ((h2 * vol_h) - (weather.rain_speed_mps as f64 * t)).rem_euclid(vol_h) + 8.0;
            batch_instance_into_local_enu(
                &frame,
                mesh,
                [east, north, up],
                [p_scale * 0.45, p_scale * 0.45, p_scale * 2.2],
                0.0,
                &mut verts,
                &mut inds,
            );
        }
        let mut col = weather.rain_color;
        col[3] *= alpha_fade;
        out.push(RetainedDynamicMesh {
            id: 0x9E47_0002,
            revision,
            anchor_ecef: anchor,
            vertices: Arc::new(verts),
            indices: Arc::new(inds),
            color: col,
            unlit: false,
        });
    }

    // 3. 2.5D 3D CC0 Snowflakes
    if weather.snow && weather.snow_density > 0 {
        let mesh = pick_mesh(custom.and_then(|c| c.snow.as_ref()), cc0_snow_mesh());
        let count = weather.snow_density.min(900);
        let mut verts = Vec::with_capacity(count * mesh.positions.len());
        let mut inds = Vec::with_capacity(count * mesh.indices.len());
        let vol_r = (camera_range_m * 0.55).clamp(180.0, 12_000.0);
        let vol_h = (camera_range_m * 0.70).clamp(240.0, 6_000.0);
        let p_scale = (weather.snow_scale_m as f64) * (camera_range_m / 1_500.0).clamp(0.4, 8.0);
        for i in 0..count as u32 {
            let h0 = det_hash(i * 37 + 211);
            let h1 = det_hash(i * 37 + 212);
            let h2 = det_hash(i * 37 + 213);
            let h3 = det_hash(i * 37 + 214);
            let flutter = (t * (1.2 + h3) + h0 * std::f64::consts::TAU).sin() * (p_scale * 1.4);
            let east = (h0 - 0.5) * 2.0 * vol_r + flutter;
            let north = (h1 - 0.5) * 2.0 * vol_r + flutter * 0.6;
            let up = ((h2 * vol_h) - (weather.snow_speed_mps as f64 * t)).rem_euclid(vol_h) + 8.0;
            batch_instance_into_local_enu(
                &frame,
                mesh,
                [east, north, up],
                [p_scale, p_scale, p_scale],
                t * (0.8 + h3) + h0 * std::f64::consts::TAU,
                &mut verts,
                &mut inds,
            );
        }
        let mut col = weather.snow_color;
        col[3] *= alpha_fade;
        out.push(RetainedDynamicMesh {
            id: 0x9E47_0003,
            revision,
            anchor_ecef: anchor,
            vertices: Arc::new(verts),
            indices: Arc::new(inds),
            color: col,
            unlit: false,
        });
    }

    out
}
