//! Minimal pure-Rust glTF 2.0 `.glb` binary mesh loader and global cloud-layer [`WeatherConfig`].
//!
//! Used by [`crate::world::EarthGeometry::glb`] for custom user-supplied 3D `.glb` assets, while
//! atmospheric precipitation is handled by the GPU-instanced 2D [`crate::precipitation::PrecipitationPlugin`].

use crate::json::Json;

/// Normalized unit-scale triangle mesh extracted from a glTF 2.0 `.glb` asset.
#[derive(Debug, Clone, PartialEq)]
pub struct WeatherMesh {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub indices: Vec<u32>,
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

/// Global and local atmospheric cloud-shell configuration (precipitation is configured separately
/// via [`crate::precipitation::PrecipitationConfig`] and [`crate::precipitation::PrecipitationPlugin`]).
#[derive(Debug, Clone, PartialEq)]
pub struct WeatherConfig {
    /// Render live equirectangular global cloud map in 3D space and overhead in 2.5D.
    pub global_cloud_map: bool,
    /// Opacity of the cloud layer (`0.0..=1.0`).
    pub global_cloud_opacity: f32,
    /// Resolution tier for the live global cloud map (`1K`, `2K`, `4K`, `8K`).
    pub cloud_resolution: crate::astro::CloudMapResolution,
    /// Custom or default live global cloud-map URL (`https://clouds.matteason.co.uk/...`).
    pub cloud_map_url: String,
    /// Global cloud drift offset speed in radians/second (defaults to `0.0`).
    pub global_cloud_spin_rad_s: f64,
    /// Elapsed weather animation seconds (drives global cloud drift).
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
