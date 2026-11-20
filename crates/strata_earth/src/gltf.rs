//! Minimal pure-Rust glTF 2.0 `.glb` binary mesh loader and global cloud-layer [`WeatherConfig`].
//!
//! Used by [`crate::world::EarthGeometry::glb`] for custom user-supplied 3D `.glb` assets, while
//! atmospheric precipitation is handled by the GPU-instanced 2D [`crate::precipitation::PrecipitationPlugin`].

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use strata_render::scene3d::RetainedTileTexture;

use crate::json::Json;

static NEXT_GLB_TEXTURE_ID: AtomicU64 = AtomicU64::new(0x6B47_1000);

/// Allocates a process-wide unique GPU texture ID for a glTF or material albedo texture.
pub fn alloc_earth_material_texture_id() -> u64 {
    NEXT_GLB_TEXTURE_ID.fetch_add(1, Ordering::Relaxed)
}

/// Complete triangle mesh and material attributes extracted from a glTF 2.0 `.glb` asset.
///
/// Vertex coordinates and normals are automatically converted from the glTF 2.0 right-handed
/// Y-up standard (`+X` Right, `+Y` Up, `-Z` Forward) into local ENU Z-up model space
/// (`+X` East, `+Y` North, `+Z` Up) via `(x, y, z)_gltf -> (x, -z, y)_enu` (`det = +1`).
#[derive(Debug, Clone, PartialEq)]
pub struct WeatherMesh {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub uvs: Vec<[f32; 2]>,
    pub vertex_colors: Option<Vec<[f32; 4]>>,
    pub indices: Vec<u32>,
    pub albedo_texture: Option<Arc<RetainedTileTexture>>,
    pub base_color_factor: [f32; 4],
    pub metallic_factor: f32,
    pub roughness_factor: f32,
    pub emissive_factor: [f32; 3],
}

impl Default for WeatherMesh {
    fn default() -> Self {
        Self {
            positions: Vec::new(),
            normals: Vec::new(),
            uvs: Vec::new(),
            vertex_colors: None,
            indices: Vec::new(),
            albedo_texture: None,
            base_color_factor: [1.0, 1.0, 1.0, 1.0],
            metallic_factor: 0.15,
            roughness_factor: 0.45,
            emissive_factor: [0.0, 0.0, 0.0],
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

fn j_f32(val: Option<&Json>) -> Option<f32> {
    match val? {
        Json::Number(n) if n.is_finite() => Some(*n as f32),
        _ => None,
    }
}

fn j_str(val: Option<&Json>) -> Option<&str> {
    match val? {
        Json::String(s) => Some(s.as_str()),
        _ => None,
    }
}

fn j_f32_array<const N: usize>(val: Option<&Json>) -> Option<[f32; N]> {
    let Json::Array(arr) = val? else {
        return None;
    };
    if arr.len() < N {
        return None;
    }
    let mut out = [0.0f32; N];
    for i in 0..N {
        out[i] = j_f32(arr.get(i))?;
    }
    Some(out)
}

const MAT4_IDENTITY: [f32; 16] = [
    1.0, 0.0, 0.0, 0.0, // col 0
    0.0, 1.0, 0.0, 0.0, // col 1
    0.0, 0.0, 1.0, 0.0, // col 2
    0.0, 0.0, 0.0, 1.0, // col 3
];

fn mat4_mul(a: [f32; 16], b: [f32; 16]) -> [f32; 16] {
    let mut out = [0.0f32; 16];
    for c in 0..4 {
        for r in 0..4 {
            out[c * 4 + r] = a[r] * b[c * 4]
                + a[4 + r] * b[c * 4 + 1]
                + a[8 + r] * b[c * 4 + 2]
                + a[12 + r] * b[c * 4 + 3];
        }
    }
    out
}

fn mat4_from_trs(t: [f32; 3], q: [f32; 4], s: [f32; 3]) -> [f32; 16] {
    let (qx, qy, qz, qw) = {
        let len = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
        if len > 1e-8 {
            (q[0] / len, q[1] / len, q[2] / len, q[3] / len)
        } else {
            (0.0, 0.0, 0.0, 1.0)
        }
    };
    let xx = qx * qx;
    let yy = qy * qy;
    let zz = qz * qz;
    let xy = qx * qy;
    let xz = qx * qz;
    let yz = qy * qz;
    let wx = qw * qx;
    let wy = qw * qy;
    let wz = qw * qz;

    let r00 = (1.0 - 2.0 * (yy + zz)) * s[0];
    let r10 = (2.0 * (xy + wz)) * s[0];
    let r20 = (2.0 * (xz - wy)) * s[0];

    let r01 = (2.0 * (xy - wz)) * s[1];
    let r11 = (1.0 - 2.0 * (xx + zz)) * s[1];
    let r21 = (2.0 * (yz + wx)) * s[1];

    let r02 = (2.0 * (xz + wy)) * s[2];
    let r12 = (2.0 * (yz - wx)) * s[2];
    let r22 = (1.0 - 2.0 * (xx + yy)) * s[2];

    [
        r00, r10, r20, 0.0, r01, r11, r21, 0.0, r02, r12, r22, 0.0, t[0], t[1], t[2], 1.0,
    ]
}

fn parse_node_local_matrix(node_obj: &[(String, Json)]) -> [f32; 16] {
    if let Some(m) = j_f32_array::<16>(j_field(node_obj, "matrix")) {
        return m;
    }
    let t = j_f32_array::<3>(j_field(node_obj, "translation")).unwrap_or([0.0, 0.0, 0.0]);
    let r = j_f32_array::<4>(j_field(node_obj, "rotation")).unwrap_or([0.0, 0.0, 0.0, 1.0]);
    let s = j_f32_array::<3>(j_field(node_obj, "scale")).unwrap_or([1.0, 1.0, 1.0]);
    mat4_from_trs(t, r, s)
}

#[inline]
fn transform_point_gltf_to_enu(m: &[f32; 16], p: [f32; 3]) -> [f32; 3] {
    let gx = m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12];
    let gy = m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13];
    let gz = m[2] * p[0] + m[6] * p[1] + m[10] * p[2] + m[14];
    // glTF (+X right, +Y up, -Z forward) -> ENU (+X East, +Y North, +Z Up)
    [gx, -gz, gy]
}

#[inline]
fn transform_normal_gltf_to_enu(m: &[f32; 16], n: [f32; 3]) -> [f32; 3] {
    let c0 = [m[0], m[1], m[2]];
    let c1 = [m[4], m[5], m[6]];
    let c2 = [m[8], m[9], m[10]];
    let cross = |a: [f32; 3], b: [f32; 3]| -> [f32; 3] {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    };
    let adj0 = cross(c1, c2);
    let adj1 = cross(c2, c0);
    let adj2 = cross(c0, c1);
    let mut gx = adj0[0] * n[0] + adj1[0] * n[1] + adj2[0] * n[2];
    let mut gy = adj0[1] * n[0] + adj1[1] * n[1] + adj2[1] * n[2];
    let mut gz = adj0[2] * n[0] + adj1[2] * n[1] + adj2[2] * n[2];
    let len2 = gx * gx + gy * gy + gz * gz;
    if len2 < 1e-12 {
        gx = c0[0] * n[0] + c1[0] * n[1] + c2[0] * n[2];
        gy = c0[1] * n[0] + c1[1] * n[1] + c2[1] * n[2];
        gz = c0[2] * n[0] + c1[2] * n[1] + c2[2] * n[2];
    }
    let ex = gx;
    let ey = -gz;
    let ez = gy;
    let len = (ex * ex + ey * ey + ez * ez).sqrt();
    if len > 1e-6 {
        [ex / len, ey / len, ez / len]
    } else {
        [0.0, 0.0, 1.0]
    }
}

fn decode_base64_bytes(b64: &str) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(b64.len() * 3 / 4);
    let mut buf = 0u32;
    let mut bits = 0u32;
    for byte in b64.bytes() {
        let val = match byte {
            b'A'..=b'Z' => byte - b'A',
            b'a'..=b'z' => byte - b'a' + 26,
            b'0'..=b'9' => byte - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            b'=' => break,
            b' ' | b'\n' | b'\r' | b'\t' => continue,
            _ => return None,
        };
        buf = (buf << 6) | (val as u32);
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push(((buf >> bits) & 0xFF) as u8);
        }
    }
    Some(out)
}

#[derive(Clone)]
struct ParsedGlbMaterial {
    base_color_factor: [f32; 4],
    metallic_factor: f32,
    roughness_factor: f32,
    emissive_factor: [f32; 3],
    base_color_tex_idx: Option<usize>,
}

impl Default for ParsedGlbMaterial {
    fn default() -> Self {
        Self {
            base_color_factor: [1.0, 1.0, 1.0, 1.0],
            metallic_factor: 0.15,
            roughness_factor: 0.45,
            emissive_factor: [0.0, 0.0, 0.0],
            base_color_tex_idx: None,
        }
    }
}

/// Parses all triangle mesh primitives, node transforms, UV coordinates, vertex colors,
/// PBR materials, and embedded base-color textures from a glTF 2.0 binary `.glb` buffer.
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
    let bin = bin_bytes.unwrap_or(&[]);
    let Json::Object(root) = Json::parse(json_text).map_err(|e| format!("glTF JSON parse: {e}"))?
    else {
        return Err("glTF root not an object".to_string());
    };

    let Some(Json::Array(meshes)) = j_field(&root, "meshes") else {
        return Err("no meshes in glTF".to_string());
    };
    if meshes.is_empty() {
        return Err("empty meshes array".to_string());
    }
    let Some(Json::Array(accessors)) = j_field(&root, "accessors") else {
        return Err("no accessors in glTF".to_string());
    };
    let Some(Json::Array(buffer_views)) = j_field(&root, "bufferViews") else {
        return Err("no bufferViews in glTF".to_string());
    };

    // 1. Parse PBR materials
    let mut parsed_materials: Vec<ParsedGlbMaterial> = Vec::new();
    if let Some(Json::Array(mats)) = j_field(&root, "materials") {
        for mat_val in mats {
            let mut m = ParsedGlbMaterial::default();
            if let Json::Object(m_obj) = mat_val {
                if let Some(ef) = j_f32_array::<3>(j_field(m_obj, "emissiveFactor")) {
                    m.emissive_factor = ef;
                }
                if let Some(Json::Object(pbr)) = j_field(m_obj, "pbrMetallicRoughness") {
                    if let Some(bcf) = j_f32_array::<4>(j_field(pbr, "baseColorFactor")) {
                        m.base_color_factor = bcf;
                    }
                    if let Some(mf) = j_f32(j_field(pbr, "metallicFactor")) {
                        m.metallic_factor = mf.clamp(0.0, 1.0);
                    }
                    if let Some(rf) = j_f32(j_field(pbr, "roughnessFactor")) {
                        m.roughness_factor = rf.clamp(0.04, 1.0);
                    }
                    if let Some(Json::Object(bct)) = j_field(pbr, "baseColorTexture") {
                        m.base_color_tex_idx = j_usize(j_field(bct, "index"));
                    }
                }
            }
            parsed_materials.push(m);
        }
    }

    // Helper to decode a glTF texture index (`textures[tex_idx].source -> images[img_idx]`)
    let decode_gltf_texture = |tex_idx: usize| -> Option<Arc<RetainedTileTexture>> {
        let Json::Array(textures) = j_field(&root, "textures")? else {
            return None;
        };
        let Json::Object(tex_obj) = textures.get(tex_idx)? else {
            return None;
        };
        let img_idx = j_usize(j_field(tex_obj, "source"))?;
        let Json::Array(images) = j_field(&root, "images")? else {
            return None;
        };
        let Json::Object(img_obj) = images.get(img_idx)? else {
            return None;
        };
        let owned_bytes: Option<Vec<u8>>;
        let raw_bytes: &[u8] = if let Some(bv_idx) = j_usize(j_field(img_obj, "bufferView")) {
            let Json::Object(bv) = buffer_views.get(bv_idx)? else {
                return None;
            };
            let bv_off = j_usize(j_field(bv, "byteOffset")).unwrap_or(0);
            let bv_len = j_usize(j_field(bv, "byteLength"))?;
            if bv_off + bv_len > bin.len() {
                return None;
            }
            &bin[bv_off..bv_off + bv_len]
        } else if let Some(uri) = j_str(j_field(img_obj, "uri")) {
            let (_, b64_part) = uri.split_once(";base64,")?;
            owned_bytes = decode_base64_bytes(b64_part);
            owned_bytes.as_deref()?
        } else {
            return None;
        };

        let dyn_img = image::load_from_memory(raw_bytes).ok()?;
        let rgba = dyn_img.to_rgba8();
        let (w, h) = rgba.dimensions();
        if w == 0 || h == 0 {
            return None;
        }
        Some(Arc::new(RetainedTileTexture {
            id: alloc_earth_material_texture_id(),
            revision: 1,
            width: w,
            height: h,
            rgba8: Arc::new(rgba.into_raw()),
        }))
    };

    // 2. Collect (mesh_index, world_matrix) instances by traversing scene/nodes hierarchy
    let mut mesh_instances: Vec<(usize, [f32; 16])> = Vec::new();
    if let Some(Json::Array(nodes)) = j_field(&root, "nodes") {
        let mut root_node_indices: Vec<usize> = Vec::new();
        if let Some(Json::Array(scenes)) = j_field(&root, "scenes") {
            let scene_idx = j_usize(j_field(&root, "scene")).unwrap_or(0);
            if let Some(Json::Object(sc_obj)) = scenes.get(scene_idx) {
                if let Some(Json::Array(sc_nodes)) = j_field(sc_obj, "nodes") {
                    for sn in sc_nodes {
                        if let Some(idx) = j_usize(Some(sn)) {
                            root_node_indices.push(idx);
                        }
                    }
                }
            }
        }
        if root_node_indices.is_empty() {
            let mut is_child = vec![false; nodes.len()];
            for node_val in nodes {
                if let Json::Object(n_obj) = node_val {
                    if let Some(Json::Array(ch)) = j_field(n_obj, "children") {
                        for c in ch {
                            if let Some(ci) = j_usize(Some(c)) {
                                if ci < is_child.len() {
                                    is_child[ci] = true;
                                }
                            }
                        }
                    }
                }
            }
            for (ni, &child) in is_child.iter().enumerate() {
                if !child {
                    root_node_indices.push(ni);
                }
            }
        }

        let mut stack: Vec<(usize, [f32; 16], usize)> = root_node_indices
            .into_iter()
            .rev()
            .map(|ni| (ni, MAT4_IDENTITY, 0))
            .collect();
        while let Some((node_idx, parent_mat, depth)) = stack.pop() {
            if depth > 64 {
                continue;
            }
            let Some(Json::Object(node_obj)) = nodes.get(node_idx) else {
                continue;
            };
            let world_mat = mat4_mul(parent_mat, parse_node_local_matrix(node_obj));
            if let Some(mesh_idx) = j_usize(j_field(node_obj, "mesh")) {
                if mesh_idx < meshes.len() {
                    mesh_instances.push((mesh_idx, world_mat));
                }
            }
            if let Some(Json::Array(children)) = j_field(node_obj, "children") {
                for c in children.iter().rev() {
                    if let Some(ci) = j_usize(Some(c)) {
                        stack.push((ci, world_mat, depth + 1));
                    }
                }
            }
        }
    }
    if mesh_instances.is_empty() {
        for mi in 0..meshes.len() {
            mesh_instances.push((mi, MAT4_IDENTITY));
        }
    }

    let read_accessor =
        |acc_idx: usize| -> Result<(&[u8], usize, usize, usize, usize, &str), String> {
            let Some(Json::Object(acc)) = accessors.get(acc_idx) else {
                return Err("invalid accessor index".to_string());
            };
            let bv_idx =
                j_usize(j_field(acc, "bufferView")).ok_or("accessor missing bufferView")?;
            let Some(Json::Object(bv)) = buffer_views.get(bv_idx) else {
                return Err("invalid bufferView index".to_string());
            };
            let bv_offset = j_usize(j_field(bv, "byteOffset")).unwrap_or(0);
            let acc_offset = j_usize(j_field(acc, "byteOffset")).unwrap_or(0);
            let base = bv_offset + acc_offset;
            let count = j_usize(j_field(acc, "count")).ok_or("accessor missing count")?;
            let comp = j_usize(j_field(acc, "componentType")).ok_or("missing componentType")?;
            let stride = j_usize(j_field(bv, "byteStride")).unwrap_or(0);
            let acc_type = j_str(j_field(acc, "type")).unwrap_or("SCALAR");
            Ok((bin, base, count, comp, stride, acc_type))
        };

    let mut all_positions: Vec<[f32; 3]> = Vec::new();
    let mut all_normals: Vec<[f32; 3]> = Vec::new();
    let mut all_uvs: Vec<[f32; 2]> = Vec::new();
    let mut all_colors: Vec<[f32; 4]> = Vec::new();
    let mut has_any_custom_color = false;
    let mut all_indices: Vec<u32> = Vec::new();
    let mut primary_mat: Option<ParsedGlbMaterial> = None;
    let mut primary_tex: Option<Arc<RetainedTileTexture>> = None;

    for (mesh_idx, world_mat) in mesh_instances {
        let Some(Json::Object(mesh_obj)) = meshes.get(mesh_idx) else {
            continue;
        };
        let Some(Json::Array(primitives)) = j_field(mesh_obj, "primitives") else {
            return Err("no primitives in mesh".to_string());
        };
        if primitives.is_empty() {
            return Err("empty primitives".to_string());
        }

        for prim_val in primitives {
            let Json::Object(prim) = prim_val else {
                continue;
            };
            let mode = j_usize(j_field(prim, "mode")).unwrap_or(4);
            if !matches!(mode, 4..=6) {
                continue;
            }
            let Some(Json::Object(attrs)) = j_field(prim, "attributes") else {
                return Err("no attributes in primitive".to_string());
            };

            let pos_idx =
                j_usize(j_field(attrs, "POSITION")).ok_or("primitive missing POSITION")?;
            let (pbin, pbase, pcount, pcomp, pstride, _) = read_accessor(pos_idx)?;
            if pcomp != 5126 {
                return Err("POSITION componentType must be f32".to_string());
            }
            if pcount == 0 {
                continue;
            }
            let pstep = pstride.max(12);
            let mut prim_positions = Vec::with_capacity(pcount);
            for i in 0..pcount {
                let o = pbase + i * pstep;
                if o + 12 > pbin.len() {
                    return Err("POSITION out of bounds".to_string());
                }
                let x = f32::from_le_bytes([pbin[o], pbin[o + 1], pbin[o + 2], pbin[o + 3]]);
                let y = f32::from_le_bytes([pbin[o + 4], pbin[o + 5], pbin[o + 6], pbin[o + 7]]);
                let z = f32::from_le_bytes([pbin[o + 8], pbin[o + 9], pbin[o + 10], pbin[o + 11]]);
                prim_positions.push(transform_point_gltf_to_enu(&world_mat, [x, y, z]));
            }

            // Indices (validate `< pcount` and convert TRIANGLE_STRIP / TRIANGLE_FAN to triangle list)
            let raw_indices: Vec<u32> = if let Some(i_idx) = j_usize(j_field(prim, "indices")) {
                let (ibin, ibase, icount, icomp, istride, _) = read_accessor(i_idx)?;
                let mut out = Vec::with_capacity(icount);
                match icomp {
                    5123 => {
                        let step = istride.max(2);
                        for i in 0..icount {
                            let o = ibase + i * step;
                            if o + 2 > ibin.len() {
                                return Err("index buffer out of bounds".to_string());
                            }
                            out.push(u16::from_le_bytes([ibin[o], ibin[o + 1]]) as u32);
                        }
                    }
                    5125 => {
                        let step = istride.max(4);
                        for i in 0..icount {
                            let o = ibase + i * step;
                            if o + 4 > ibin.len() {
                                return Err("index buffer out of bounds".to_string());
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
                                return Err("index buffer out of bounds".to_string());
                            }
                            out.push(ibin[o] as u32);
                        }
                    }
                    _ => return Err("unsupported index componentType".to_string()),
                }
                out
            } else {
                (0..pcount as u32).collect()
            };

            for &idx in &raw_indices {
                if (idx as usize) >= pcount {
                    return Err(format!(
                        "primitive triangle index {idx} out of bounds for {pcount} vertices"
                    ));
                }
            }

            let prim_tri_indices: Vec<u32> = match mode {
                4 => raw_indices,
                5 => {
                    let mut tris = Vec::new();
                    if raw_indices.len() >= 3 {
                        for i in 0..(raw_indices.len() - 2) {
                            if i % 2 == 0 {
                                tris.extend_from_slice(&[
                                    raw_indices[i],
                                    raw_indices[i + 1],
                                    raw_indices[i + 2],
                                ]);
                            } else {
                                tris.extend_from_slice(&[
                                    raw_indices[i + 1],
                                    raw_indices[i],
                                    raw_indices[i + 2],
                                ]);
                            }
                        }
                    }
                    tris
                }
                6 => {
                    let mut tris = Vec::new();
                    if raw_indices.len() >= 3 {
                        let i0 = raw_indices[0];
                        for i in 1..(raw_indices.len() - 1) {
                            tris.extend_from_slice(&[i0, raw_indices[i], raw_indices[i + 1]]);
                        }
                    }
                    tris
                }
                _ => Vec::new(),
            };

            // Normals (transformed to ENU, or computed from triangle faces if absent)
            let mut prim_normals: Vec<[f32; 3]> =
                if let Some(n_idx) = j_usize(j_field(attrs, "NORMAL")) {
                    let (nbin, nbase, ncount, ncomp, nstride, _) = read_accessor(n_idx)?;
                    if ncomp == 5126 && ncount == pcount {
                        let nstep = nstride.max(12);
                        let mut out = Vec::with_capacity(ncount);
                        for i in 0..ncount {
                            let o = nbase + i * nstep;
                            if o + 12 > nbin.len() {
                                break;
                            }
                            let x = f32::from_le_bytes([
                                nbin[o],
                                nbin[o + 1],
                                nbin[o + 2],
                                nbin[o + 3],
                            ]);
                            let y = f32::from_le_bytes([
                                nbin[o + 4],
                                nbin[o + 5],
                                nbin[o + 6],
                                nbin[o + 7],
                            ]);
                            let z = f32::from_le_bytes([
                                nbin[o + 8],
                                nbin[o + 9],
                                nbin[o + 10],
                                nbin[o + 11],
                            ]);
                            out.push(transform_normal_gltf_to_enu(&world_mat, [x, y, z]));
                        }
                        out
                    } else {
                        Vec::new()
                    }
                } else {
                    Vec::new()
                };
            if prim_normals.len() != pcount {
                let mut acc = vec![[0.0f32; 3]; pcount];
                for tri in prim_tri_indices.chunks_exact(3) {
                    let i0 = tri[0] as usize;
                    let i1 = tri[1] as usize;
                    let i2 = tri[2] as usize;
                    let p0 = prim_positions[i0];
                    let p1 = prim_positions[i1];
                    let p2 = prim_positions[i2];
                    let e1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
                    let e2 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
                    let fn_v = [
                        e1[1] * e2[2] - e1[2] * e2[1],
                        e1[2] * e2[0] - e1[0] * e2[2],
                        e1[0] * e2[1] - e1[1] * e2[0],
                    ];
                    for &vi in &[i0, i1, i2] {
                        acc[vi][0] += fn_v[0];
                        acc[vi][1] += fn_v[1];
                        acc[vi][2] += fn_v[2];
                    }
                }
                prim_normals = acc
                    .into_iter()
                    .map(|n| {
                        let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
                        if l > 1e-6 {
                            [n[0] / l, n[1] / l, n[2] / l]
                        } else {
                            [0.0, 0.0, 1.0]
                        }
                    })
                    .collect();
            }

            // TEXCOORD_0
            let mut prim_uvs: Vec<[f32; 2]> =
                if let Some(uv_idx) = j_usize(j_field(attrs, "TEXCOORD_0")) {
                    let (ubin, ubase, ucount, ucomp, ustride, _) = read_accessor(uv_idx)?;
                    let mut out = Vec::with_capacity(ucount.min(pcount));
                    if ucount == pcount {
                        match ucomp {
                            5126 => {
                                let step = ustride.max(8);
                                for i in 0..pcount {
                                    let o = ubase + i * step;
                                    if o + 8 > ubin.len() {
                                        break;
                                    }
                                    let u = f32::from_le_bytes([
                                        ubin[o],
                                        ubin[o + 1],
                                        ubin[o + 2],
                                        ubin[o + 3],
                                    ]);
                                    let v = f32::from_le_bytes([
                                        ubin[o + 4],
                                        ubin[o + 5],
                                        ubin[o + 6],
                                        ubin[o + 7],
                                    ]);
                                    out.push([u, v]);
                                }
                            }
                            5121 => {
                                let step = ustride.max(2);
                                for i in 0..pcount {
                                    let o = ubase + i * step;
                                    if o + 2 > ubin.len() {
                                        break;
                                    }
                                    out.push([
                                        ubin[o] as f32 / 255.0,
                                        ubin[o + 1] as f32 / 255.0,
                                    ]);
                                }
                            }
                            5123 => {
                                let step = ustride.max(4);
                                for i in 0..pcount {
                                    let o = ubase + i * step;
                                    if o + 4 > ubin.len() {
                                        break;
                                    }
                                    let u = u16::from_le_bytes([ubin[o], ubin[o + 1]]) as f32
                                        / 65535.0;
                                    let v = u16::from_le_bytes([ubin[o + 2], ubin[o + 3]]) as f32
                                        / 65535.0;
                                    out.push([u, v]);
                                }
                            }
                            _ => {}
                        }
                    }
                    out
                } else {
                    Vec::new()
                };
            if prim_uvs.len() != pcount {
                prim_uvs.resize(pcount, [0.0, 0.0]);
            }

            // Material & COLOR_0
            let mat = j_usize(j_field(prim, "material"))
                .and_then(|mi| parsed_materials.get(mi).cloned())
                .unwrap_or_default();
            if primary_mat.is_none() {
                primary_mat = Some(mat.clone());
            }
            if primary_tex.is_none() {
                if let Some(ti) = mat.base_color_tex_idx {
                    primary_tex = decode_gltf_texture(ti);
                }
            }
            let bcf = mat.base_color_factor;
            if (bcf[0] - 1.0).abs() > 1e-4
                || (bcf[1] - 1.0).abs() > 1e-4
                || (bcf[2] - 1.0).abs() > 1e-4
                || (bcf[3] - 1.0).abs() > 1e-4
            {
                has_any_custom_color = true;
            }

            let mut prim_colors: Vec<[f32; 4]> =
                if let Some(c_idx) = j_usize(j_field(attrs, "COLOR_0")) {
                    let (cbin, cbase, ccount, ccomp, cstride, ctype) = read_accessor(c_idx)?;
                    let is_vec4 = ctype == "VEC4";
                    let mut out = Vec::with_capacity(ccount.min(pcount));
                    if ccount == pcount {
                        has_any_custom_color = true;
                        match ccomp {
                            5126 => {
                                let min_s = if is_vec4 { 16 } else { 12 };
                                let step = cstride.max(min_s);
                                for i in 0..pcount {
                                    let o = cbase + i * step;
                                    if o + min_s > cbin.len() {
                                        break;
                                    }
                                    let r = f32::from_le_bytes([
                                        cbin[o],
                                        cbin[o + 1],
                                        cbin[o + 2],
                                        cbin[o + 3],
                                    ]);
                                    let g = f32::from_le_bytes([
                                        cbin[o + 4],
                                        cbin[o + 5],
                                        cbin[o + 6],
                                        cbin[o + 7],
                                    ]);
                                    let b = f32::from_le_bytes([
                                        cbin[o + 8],
                                        cbin[o + 9],
                                        cbin[o + 10],
                                        cbin[o + 11],
                                    ]);
                                    let a = if is_vec4 {
                                        f32::from_le_bytes([
                                            cbin[o + 12],
                                            cbin[o + 13],
                                            cbin[o + 14],
                                            cbin[o + 15],
                                        ])
                                    } else {
                                        1.0
                                    };
                                    out.push([r * bcf[0], g * bcf[1], b * bcf[2], a * bcf[3]]);
                                }
                            }
                            5121 => {
                                let min_s = if is_vec4 { 4 } else { 3 };
                                let step = cstride.max(min_s);
                                for i in 0..pcount {
                                    let o = cbase + i * step;
                                    if o + min_s > cbin.len() {
                                        break;
                                    }
                                    let r = (cbin[o] as f32 / 255.0) * bcf[0];
                                    let g = (cbin[o + 1] as f32 / 255.0) * bcf[1];
                                    let b = (cbin[o + 2] as f32 / 255.0) * bcf[2];
                                    let a = if is_vec4 {
                                        (cbin[o + 3] as f32 / 255.0) * bcf[3]
                                    } else {
                                        bcf[3]
                                    };
                                    out.push([r, g, b, a]);
                                }
                            }
                            5123 => {
                                let min_s = if is_vec4 { 8 } else { 6 };
                                let step = cstride.max(min_s);
                                for i in 0..pcount {
                                    let o = cbase + i * step;
                                    if o + min_s > cbin.len() {
                                        break;
                                    }
                                    let r = (u16::from_le_bytes([cbin[o], cbin[o + 1]]) as f32
                                        / 65535.0)
                                        * bcf[0];
                                    let g = (u16::from_le_bytes([cbin[o + 2], cbin[o + 3]]) as f32
                                        / 65535.0)
                                        * bcf[1];
                                    let b = (u16::from_le_bytes([cbin[o + 4], cbin[o + 5]]) as f32
                                        / 65535.0)
                                        * bcf[2];
                                    let a = if is_vec4 {
                                        (u16::from_le_bytes([cbin[o + 6], cbin[o + 7]]) as f32
                                            / 65535.0)
                                            * bcf[3]
                                    } else {
                                        bcf[3]
                                    };
                                    out.push([r, g, b, a]);
                                }
                            }
                            _ => {}
                        }
                    }
                    out
                } else {
                    Vec::new()
                };
            if prim_colors.len() != pcount {
                prim_colors.resize(pcount, bcf);
            }

            let base_v = all_positions.len() as u32;
            all_positions.extend_from_slice(&prim_positions);
            all_normals.extend_from_slice(&prim_normals);
            all_uvs.extend_from_slice(&prim_uvs);
            all_colors.extend_from_slice(&prim_colors);
            all_indices.extend(prim_tri_indices.into_iter().map(|i| base_v + i));
        }
    }

    if all_positions.is_empty() || all_indices.is_empty() {
        return Err("glTF contained no valid triangle primitives".to_string());
    }

    let pmat = primary_mat.unwrap_or_default();
    Ok(WeatherMesh {
        positions: all_positions,
        normals: all_normals,
        uvs: all_uvs,
        vertex_colors: if has_any_custom_color {
            Some(all_colors)
        } else {
            None
        },
        indices: all_indices,
        albedo_texture: primary_tex,
        base_color_factor: pmat.base_color_factor,
        metallic_factor: pmat.metallic_factor,
        roughness_factor: pmat.roughness_factor,
        emissive_factor: pmat.emissive_factor,
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
    /// Synchronize 2.5D/3D cloud storm darkening, slate tint, and overcast coverage with
    /// [`crate::scene::EarthInput::base_weather`] and active precipitation (`true` default).
    pub sync_with_base_weather: bool,
    /// Optional explicit cloud storm darkening override `[0.0..=1.0]` (`None` = derive automatically).
    pub cloud_darkness_override: Option<f32>,
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
            sync_with_base_weather: true,
            cloud_darkness_override: None,
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

    pub fn with_sync_with_base_weather(mut self, sync: bool) -> Self {
        self.sync_with_base_weather = sync;
        self
    }

    pub fn with_cloud_darkness_override(mut self, darkness: Option<f32>) -> Self {
        self.cloud_darkness_override = darkness.map(|d| d.clamp(0.0, 1.0));
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
