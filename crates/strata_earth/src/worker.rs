//! Asynchronous tile streaming, frustum-culled LOD selection, and retained GPU mesh builder.

use crate::geodesy::{
    build_tile_local_rtc_mesh_morphed, level_for_altitude, retained_camera_uniforms,
    tile_bounding_sphere, tile_bounds, tile_bounds_mercator,
    traverse_quadtree_lod_frustum6_with_options, EarthCamera, Frustum6, PatchBounds,
    SurfaceMorphParams, TileKey, TileProjection, MAX_QUADTREE_LEAVES,
};
use crate::scene::{
    compute_adaptive_skirt_depth, requested_tile, sky_gradient, surrounding_patches,
    tile_patch_span_deg, EarthInput, EarthLowLevelState, EarthViewMode, SurfaceMode,
};
use crate::tiles::{TileElevation, TileImage, TileSource};
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use strata_render::camera64::DVec3;
use strata_render::scene3d::{
    RetainedCameraUniforms, RetainedScene3D, RetainedTerrainFrame, RetainedTileInstance,
    RetainedTileMesh, RetainedTileTexture, TileKeyId,
};

#[inline]
fn hash_bytes_64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    // Sample stride keeps hashing O(1) (~256 samples max) on large tiles while catching edits.
    let step = (bytes.len() / 256).max(1);
    let mut i = 0;
    while i < bytes.len() {
        h ^= bytes[i] as u64;
        h = h.wrapping_mul(0x100000001b3);
        i += step;
    }
    h ^= bytes.len() as u64;
    h
}

#[inline]
fn hash_elevation_64(elev: Option<&TileElevation>) -> u64 {
    let Some(e) = elev else {
        return 0;
    };
    let mut h: u64 = 0xcbf29ce484222325 ^ ((e.width as u64) << 32) ^ (e.height as u64);
    let step = (e.heights_m.len() / 64).max(1);
    let mut i = 0;
    while i < e.heights_m.len() {
        h ^= e.heights_m[i].to_bits() as u64;
        h = h.wrapping_mul(0x100000001b3);
        i += step;
    }
    h
}

#[inline]
fn flat_patch_key_id(bounds: PatchBounds, level: u8) -> TileKeyId {
    let bits = [
        bounds.lat0.to_bits(),
        bounds.lat1.to_bits(),
        bounds.lon0.to_bits(),
        bounds.lon1.to_bits(),
    ];
    let mut h: u64 = 0xcbf29ce484222325;
    for b in bits {
        h ^= b;
        h = h.wrapping_mul(0x100000001b3);
    }
    TileKeyId::new(level, (h >> 32) as u32, h as u32, 1)
}

#[inline]
fn elevation_range(elev: Option<&TileElevation>) -> f64 {
    let Some(grid) = elev else {
        return 0.0;
    };
    let mut min_h = f32::MAX;
    let mut max_h = f32::MIN;
    for &h in &grid.heights_m {
        if h < min_h {
            min_h = h;
        }
        if h > max_h {
            max_h = h;
        }
    }
    if max_h >= min_h {
        (max_h - min_h) as f64
    } else {
        0.0
    }
}

/// Statistics from a single retained terrain build pass.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RetainedBuildStats {
    pub visible_tiles: usize,
    pub mesh_cache_hits: usize,
    pub mesh_cache_misses: usize,
    pub texture_cache_hits: usize,
    pub texture_cache_misses: usize,
    pub ancestor_fallbacks: usize,
}

#[inline]
fn mesh_morph_signature(morph: SurfaceMorphParams) -> u64 {
    let proj_bit = match morph.projection {
        TileProjection::WebMercator => 1u64,
        TileProjection::Geographic4326 => 2u64,
    };
    if morph.flat_blend <= 1e-6 {
        return proj_bit;
    }
    let flat_q = (morph.flat_blend.clamp(0.0, 1.0) * 1000.0).round() as u64;
    let map_q = (morph.map_2d_morph.clamp(0.0, 1.0) * 1000.0).round() as u64;
    let lat_q = ((morph.focus.lat_deg + 90.0) * 10_000.0).round() as u64;
    let lon_q = ((morph.focus.lon_deg + 180.0) * 10_000.0).round() as u64;
    let mut h = 0xcbf29ce484222325u64 ^ proj_bit;
    for v in [flat_q, map_q, lat_q, lon_q] {
        h ^= v;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

struct CachedMeshEntry {
    elev_hash: u64,
    morph_sig: u64,
    revision: u64,
    mesh: Arc<RetainedTileMesh>,
}

struct CachedTexEntry {
    id: u64,
    pixel_hash: u64,
    revision: u64,
    texture: Arc<RetainedTileTexture>,
}

/// Persistent CPU cache converting visible LOD leaves into GPU-ready tile-local RTC meshes and textures.
pub struct RetainedTerrainBuilder {
    meshes: HashMap<(TileKeyId, u32, bool), CachedMeshEntry>,
    textures: HashMap<TileKey, CachedTexEntry>,
    next_tex_id: u64,
    next_revision: u64,
}

impl std::fmt::Debug for RetainedTerrainBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RetainedTerrainBuilder")
            .field("cached_meshes", &self.meshes.len())
            .field("cached_textures", &self.textures.len())
            .field("next_revision", &self.next_revision)
            .finish()
    }
}

impl Default for RetainedTerrainBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl RetainedTerrainBuilder {
    pub fn new() -> Self {
        Self {
            meshes: HashMap::new(),
            textures: HashMap::new(),
            next_tex_id: 1,
            next_revision: 1,
        }
    }

    pub fn camera_from_input(input: &EarthInput) -> RetainedCameraUniforms {
        let ll = input.low_level_state();
        let map_t = ll.map_2d_morph as f64;
        let mut cam = input.camera;
        if map_t > 1e-4 {
            let [ex, ey, ez] = cam.eye_offset_m;
            let dist = (ex * ex + ey * ey + ez * ez).sqrt().max(cam.altitude_m());
            cam.eye_offset_m = [
                ex * (1.0 - map_t),
                ey * (1.0 - map_t),
                ez + (dist - ez) * map_t,
            ];
            cam.aim_offset_m = [
                cam.aim_offset_m[0] * (1.0 - map_t),
                cam.aim_offset_m[1] * (1.0 - map_t),
                cam.aim_offset_m[2] * (1.0 - map_t),
            ];
        }
        let placement = cam.placement();
        let key_dir = placement.eye.sub(placement.aim).normalize();
        let sun_dir = {
            let v = [
                key_dir.x as f32 + 0.35,
                key_dir.y as f32 + 0.35,
                key_dir.z as f32 + 0.25,
            ];
            let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-6);
            [v[0] / len, v[1] / len, v[2] / len]
        };
        let dl = input.daylight.clamp(0.0, 1.0);
        let sun_color = [0.15 + 0.2 * dl, 0.15 + 0.2 * dl, 0.14 + 0.18 * dl];
        let amb_s = 0.25 + 0.2 * dl;
        let ambient_color = [
            (150.0 / 255.0) * amb_s,
            (170.0 / 255.0) * amb_s,
            (200.0 / 255.0) * amb_s,
        ];
        retained_camera_uniforms(
            &cam,
            input.viewport.width as f64,
            input.viewport.height as f64,
            sun_dir,
            sun_color,
            ambient_color,
            sky_gradient(input.daylight),
        )
    }

    fn resolve_or_cache_texture(
        &mut self,
        key: TileKey,
        img: &TileImage,
        stats: &mut RetainedBuildStats,
    ) -> (u64, u64, Arc<RetainedTileTexture>) {
        let pixel_hash = hash_bytes_64(&img.pixels);
        if let Some(entry) = self.textures.get(&key) {
            if entry.pixel_hash == pixel_hash
                && entry.texture.width == img.width
                && entry.texture.height == img.height
            {
                stats.texture_cache_hits += 1;
                return (entry.id, entry.revision, Arc::clone(&entry.texture));
            }
            let rev = entry.revision + 1;
            let tex = Arc::new(RetainedTileTexture {
                id: entry.id,
                revision: rev,
                width: img.width,
                height: img.height,
                rgba8: Arc::new(img.pixels.clone()),
            });
            let id = entry.id;
            self.textures.insert(
                key,
                CachedTexEntry {
                    id,
                    pixel_hash,
                    revision: rev,
                    texture: Arc::clone(&tex),
                },
            );
            stats.texture_cache_misses += 1;
            return (id, rev, tex);
        }

        let id = self.next_tex_id;
        self.next_tex_id += 1;
        let tex = Arc::new(RetainedTileTexture {
            id,
            revision: 1,
            width: img.width,
            height: img.height,
            rgba8: Arc::new(img.pixels.clone()),
        });
        self.textures.insert(
            key,
            CachedTexEntry {
                id,
                pixel_hash,
                revision: 1,
                texture: Arc::clone(&tex),
            },
        );
        stats.texture_cache_misses += 1;
        (id, 1, tex)
    }

    fn resolve_imagery_or_ancestor(
        &mut self,
        key: TileKey,
        source: &mut dyn TileSource,
        max_level: u8,
        stats: &mut RetainedBuildStats,
    ) -> Option<(u64, u64, [f32; 4], Arc<RetainedTileTexture>)> {
        let mut probe = key;
        while probe.level > max_level {
            probe = probe.parent()?;
        }

        // Request exact leaf once (queues async network download if not cached).
        if let Ok(Some(img)) = source.imagery(probe) {
            if !img.pixels.is_empty() {
                let (id, rev, tex) = self.resolve_or_cache_texture(probe, &img, stats);
                let uv_rect = if probe == key {
                    [0.0, 0.0, 1.0, 1.0]
                } else {
                    stats.ancestor_fallbacks += 1;
                    let (u0, v0) = ancestor_uv_point(key, probe, 0.0, 0.0);
                    let (u1, v1) = ancestor_uv_point(key, probe, 1.0, 1.0);
                    [u0, v0, u1, v1]
                };
                return Some((id, rev, uv_rect, tex));
            }
        }

        // Walk ancestors without issuing spurious network fetches for intermediate missing levels.
        let mut anc = probe.parent();
        while let Some(a) = anc {
            if let Some(img) = source.peek_cached_imagery(a) {
                if !img.pixels.is_empty() {
                    let (id, rev, tex) = self.resolve_or_cache_texture(a, &img, stats);
                    stats.ancestor_fallbacks += 1;
                    let (u0, v0) = ancestor_uv_point(key, a, 0.0, 0.0);
                    let (u1, v1) = ancestor_uv_point(key, a, 1.0, 1.0);
                    return Some((id, rev, [u0, v0, u1, v1], tex));
                }
            }
            if let Some(entry) = self.textures.get(&a) {
                stats.ancestor_fallbacks += 1;
                let (u0, v0) = ancestor_uv_point(key, a, 0.0, 0.0);
                let (u1, v1) = ancestor_uv_point(key, a, 1.0, 1.0);
                return Some((
                    entry.id,
                    entry.revision,
                    [u0, v0, u1, v1],
                    Arc::clone(&entry.texture),
                ));
            }
            if a.level <= 2 {
                if let Ok(Some(img)) = source.imagery(a) {
                    if !img.pixels.is_empty() {
                        let (id, rev, tex) = self.resolve_or_cache_texture(a, &img, stats);
                        stats.ancestor_fallbacks += 1;
                        let (u0, v0) = ancestor_uv_point(key, a, 0.0, 0.0);
                        let (u1, v1) = ancestor_uv_point(key, a, 1.0, 1.0);
                        return Some((id, rev, [u0, v0, u1, v1], tex));
                    }
                }
            }
            anc = a.parent();
        }
        None
    }

    /// Builds a [`RetainedTerrainFrame`] from cached tile-local RTC geometry & textures.
    pub fn build_frame(
        &mut self,
        input: &EarthInput,
        source: &mut dyn TileSource,
        geometry_revision: u64,
        allow_ancestor_mesh_fallback: bool,
    ) -> (Arc<RetainedTerrainFrame>, RetainedBuildStats) {
        let camera = Self::camera_from_input(input);
        let mut stats = RetainedBuildStats::default();
        let mut meshes_out = Vec::new();
        let mut instances_out = Vec::new();
        let mut textures_out = Vec::new();
        let mut seen_tex_ids = HashSet::new();
        let mut seen_mesh_keys = HashSet::new();

        if !input.layers.terrain {
            return (
                Arc::new(RetainedTerrainFrame {
                    camera,
                    geometry_revision,
                    meshes: meshes_out,
                    instances: instances_out,
                    textures: textures_out,
                    dynamic_meshes: Vec::new(),
                }),
                stats,
            );
        }

        let ll = input.low_level_state();
        let effective_mode = ll.active_surface_mode;
        let grid = input.tile_grid.max(2);
        let with_skirts = input
            .terrain_skirts
            .unwrap_or(effective_mode == SurfaceMode::QuadtreeLod);
        let fallback_tint = [
            input.fallback_tint.r(),
            input.fallback_tint.g(),
            input.fallback_tint.b(),
            input.fallback_tint.a(),
        ];

        match effective_mode {
            SurfaceMode::QuadtreeLod => {
                let morph = SurfaceMorphParams {
                    focus: input.camera.focus,
                    flat_blend: (ll.local_25d_blend.max(ll.map_2d_morph)) as f64,
                    map_2d_morph: ll.map_2d_morph as f64,
                    projection: input.tile_projection,
                };
                let morph_sig = mesh_morph_signature(morph);
                let horizon_cull = ll.map_2d_morph <= 0.05;
                let leaves = traverse_quadtree_lod_frustum6_with_options(
                    &input.camera,
                    input.viewport.width.max(1.0) as f64,
                    input.viewport.height.max(1.0) as f64,
                    input.sse_threshold,
                    input.max_tile_level,
                    grid,
                    horizon_cull,
                    MAX_QUADTREE_LEAVES,
                );
                let mut emitted_ancestors = HashSet::new();
                for leaf in leaves {
                    let elev_res = source.elevation(leaf);
                    let elev = elev_res.as_ref().ok().and_then(|o| o.as_ref());
                    let elev_hash = hash_elevation_64(elev);
                    let key_id = TileKeyId::quadtree(leaf.level, leaf.x, leaf.y);

                    let use_ancestor_mesh = allow_ancestor_mesh_fallback
                        && elev_res.is_err()
                        && !self
                            .meshes
                            .get(&(key_id, grid, with_skirts))
                            .is_some_and(|c| c.elev_hash == elev_hash && c.morph_sig == morph_sig)
                        && has_resident_ancestor(&self.meshes, leaf, grid, with_skirts);

                    if use_ancestor_mesh {
                        if let Some((anc_key, anc_mesh)) =
                            find_resident_ancestor(&self.meshes, leaf, grid, with_skirts)
                        {
                            stats.ancestor_fallbacks += 1;
                            let anc_id = anc_mesh.key;
                            if emitted_ancestors.insert(anc_id) {
                                if seen_mesh_keys.insert(anc_id) {
                                    meshes_out.push(Arc::clone(&anc_mesh));
                                }
                                let (tex_id, tex_rev, uv_rect, tint) =
                                    if let Some((tid, trev, rect, tex)) = self
                                        .resolve_imagery_or_ancestor(
                                            anc_key,
                                            source,
                                            input.max_tile_level,
                                            &mut stats,
                                        )
                                    {
                                        if seen_tex_ids.insert(tid) {
                                            textures_out.push(tex);
                                        }
                                        (Some(tid), trev, rect, [1.0, 1.0, 1.0, 1.0])
                                    } else {
                                        (None, 0, [0.0, 0.0, 1.0, 1.0], fallback_tint)
                                    };
                                instances_out.push(RetainedTileInstance {
                                    key: anc_id,
                                    mesh_revision: anc_mesh.revision,
                                    center_ecef: anc_mesh.center_ecef,
                                    texture_id: tex_id,
                                    texture_revision: tex_rev,
                                    uv_rect,
                                    tint,
                                    unlit: false,
                                });
                            }
                            continue;
                        }
                    }

                    let bounds = match input.tile_projection {
                        TileProjection::WebMercator => tile_bounds_mercator(leaf),
                        TileProjection::Geographic4326 => tile_bounds(leaf),
                    };
                    let mesh = self.ensure_tile_mesh(
                        leaf,
                        key_id,
                        bounds,
                        grid,
                        with_skirts,
                        elev,
                        elev_hash,
                        morph,
                        morph_sig,
                        &mut stats,
                    );

                    if seen_mesh_keys.insert(mesh.key) {
                        meshes_out.push(Arc::clone(&mesh));
                    }
                    let (tex_id, tex_rev, uv_rect, tint) = if let Some((tid, trev, rect, tex)) =
                        self.resolve_imagery_or_ancestor(
                            leaf,
                            source,
                            input.max_tile_level,
                            &mut stats,
                        ) {
                        if seen_tex_ids.insert(tid) {
                            textures_out.push(tex);
                        }
                        (Some(tid), trev, rect, [1.0, 1.0, 1.0, 1.0])
                    } else {
                        (None, 0, [0.0, 0.0, 1.0, 1.0], fallback_tint)
                    };
                    instances_out.push(RetainedTileInstance {
                        key: mesh.key,
                        mesh_revision: mesh.revision,
                        center_ecef: mesh.center_ecef,
                        texture_id: tex_id,
                        texture_revision: tex_rev,
                        uv_rect,
                        tint,
                        unlit: false,
                    });
                }
            }
            SurfaceMode::FlatPatches | SurfaceMode::SphericalGlobe => {
                let morph = SurfaceMorphParams {
                    focus: input.camera.focus,
                    flat_blend: ll.map_2d_morph as f64,
                    map_2d_morph: ll.map_2d_morph as f64,
                    projection: TileProjection::Geographic4326,
                };
                let morph_sig = mesh_morph_signature(morph);
                let frustum = Frustum6::from_camera(
                    &input.camera,
                    input.viewport.width.max(1.0) as f64,
                    input.viewport.height.max(1.0) as f64,
                );
                let mut patch_specs: Vec<(TileKey, PatchBounds, TileKeyId)> = Vec::new();
                if effective_mode == SurfaceMode::SphericalGlobe {
                    let level = level_for_altitude(input.camera.altitude_m(), 2);
                    let per_axis = 1u32 << level;
                    let lon_span = 360.0 / per_axis as f64;
                    let lat_span = 180.0 / per_axis as f64;
                    for y in 0..per_axis {
                        let lat1 = 90.0 - y as f64 * lat_span;
                        let lat0 = lat1 - lat_span;
                        for x in 0..per_axis {
                            let lon0 = -180.0 + x as f64 * lon_span;
                            let lon1 = lon0 + lon_span;
                            let key = TileKey::new(level, x, y);
                            let bounds = PatchBounds {
                                lon0,
                                lon1,
                                lat0,
                                lat1,
                            };
                            patch_specs.push((key, bounds, TileKeyId::new(level, x, y, 2)));
                        }
                    }
                } else {
                    let alt_m = input.camera.altitude_m();
                    let span_deg = tile_patch_span_deg(&input.camera, input.patch_scale);
                    let patches = surrounding_patches(input.camera.focus, span_deg);
                    let level = level_for_altitude(alt_m, input.max_tile_level);
                    for patch in patches {
                        let tile = requested_tile(&patch);
                        patch_specs.push((
                            tile,
                            patch.bounds,
                            flat_patch_key_id(patch.bounds, level),
                        ));
                    }
                }

                for (tile, bounds, key_id) in patch_specs {
                    let (_, radius) = tile_bounding_sphere(tile);
                    let mid_lat = (bounds.lat0 + bounds.lat1) * 0.5;
                    let mid_lon = (bounds.lon0 + bounds.lon1) * 0.5;
                    let patch_center: DVec3 = crate::geodesy::geodetic_to_ecef(
                        crate::geodesy::Geodetic::new(mid_lat.clamp(-89.9, 89.9), mid_lon, 0.0),
                    );
                    if ll.map_2d_morph <= 0.05
                        && !frustum.intersects_sphere(patch_center, radius.max(50_000.0) * 1.25)
                    {
                        continue;
                    }

                    let elev_res = source.elevation(tile);
                    let elev = elev_res.as_ref().ok().and_then(|o| o.as_ref());
                    let elev_hash = hash_elevation_64(elev);

                    let mesh = self.ensure_tile_mesh(
                        tile,
                        key_id,
                        bounds,
                        grid,
                        with_skirts,
                        elev,
                        elev_hash,
                        morph,
                        morph_sig,
                        &mut stats,
                    );

                    if seen_mesh_keys.insert(mesh.key) {
                        meshes_out.push(Arc::clone(&mesh));
                    }
                    let (tex_id, tex_rev, uv_rect, tint) = if let Some((tid, trev, rect, tex)) =
                        self.resolve_imagery_or_ancestor(
                            tile,
                            source,
                            input.max_tile_level,
                            &mut stats,
                        ) {
                        if seen_tex_ids.insert(tid) {
                            textures_out.push(tex);
                        }
                        (Some(tid), trev, rect, [1.0, 1.0, 1.0, 1.0])
                    } else {
                        (None, 0, [0.0, 0.0, 1.0, 1.0], fallback_tint)
                    };
                    instances_out.push(RetainedTileInstance {
                        key: mesh.key,
                        mesh_revision: mesh.revision,
                        center_ecef: mesh.center_ecef,
                        texture_id: tex_id,
                        texture_revision: tex_rev,
                        uv_rect,
                        tint,
                        unlit: false,
                    });
                }
            }
        }

        stats.visible_tiles = instances_out.len();
        (
            Arc::new(RetainedTerrainFrame {
                camera,
                geometry_revision,
                meshes: meshes_out,
                instances: instances_out,
                textures: textures_out,
                dynamic_meshes: Vec::new(),
            }),
            stats,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn ensure_tile_mesh(
        &mut self,
        tile: TileKey,
        key_id: TileKeyId,
        bounds: PatchBounds,
        grid: u32,
        with_skirts: bool,
        elev: Option<&TileElevation>,
        elev_hash: u64,
        morph: SurfaceMorphParams,
        morph_sig: u64,
        stats: &mut RetainedBuildStats,
    ) -> Arc<RetainedTileMesh> {
        if let Some(cached) = self.meshes.get(&(key_id, grid, with_skirts)) {
            if cached.elev_hash == elev_hash && cached.morph_sig == morph_sig {
                stats.mesh_cache_hits += 1;
                return Arc::clone(&cached.mesh);
            }
        }
        stats.mesh_cache_misses += 1;
        let rev = match self.meshes.get(&(key_id, grid, with_skirts)) {
            Some(c) => c.revision + 1,
            None => {
                let r = self.next_revision;
                self.next_revision += 1;
                r
            }
        };
        let skirt_depth = compute_adaptive_skirt_depth(bounds, elevation_range(elev));
        let mut built = build_tile_local_rtc_mesh_morphed(
            tile,
            bounds,
            grid,
            elev,
            tile,
            with_skirts,
            skirt_depth,
            rev,
            key_id.face,
            morph,
        );
        built.key = key_id;
        let arc = Arc::new(built);
        self.meshes.insert(
            (key_id, grid, with_skirts),
            CachedMeshEntry {
                elev_hash,
                morph_sig,
                revision: rev,
                mesh: Arc::clone(&arc),
            },
        );
        arc
    }
}

fn ancestor_uv_point(descendant: TileKey, ancestor: TileKey, u: f32, v: f32) -> (f32, f32) {
    if descendant.level <= ancestor.level {
        return (u, v);
    }
    let dl = descendant.level - ancestor.level;
    let scale = (1u32 << dl.min(20)) as f32;
    let dx = (descendant.x - (ancestor.x << dl)) as f32;
    let dy = (descendant.y - (ancestor.y << dl)) as f32;
    ((dx + u) / scale, (dy + v) / scale)
}

fn has_resident_ancestor(
    meshes: &HashMap<(TileKeyId, u32, bool), CachedMeshEntry>,
    leaf: TileKey,
    grid: u32,
    skirts: bool,
) -> bool {
    let mut cur = leaf.parent();
    while let Some(anc) = cur {
        if meshes.contains_key(&(TileKeyId::quadtree(anc.level, anc.x, anc.y), grid, skirts)) {
            return true;
        }
        cur = anc.parent();
    }
    false
}

fn find_resident_ancestor(
    meshes: &HashMap<(TileKeyId, u32, bool), CachedMeshEntry>,
    leaf: TileKey,
    grid: u32,
    skirts: bool,
) -> Option<(TileKey, Arc<RetainedTileMesh>)> {
    let mut cur = leaf.parent();
    while let Some(anc) = cur {
        if let Some(entry) =
            meshes.get(&(TileKeyId::quadtree(anc.level, anc.x, anc.y), grid, skirts))
        {
            return Some((anc, Arc::clone(&entry.mesh)));
        }
        cur = anc.parent();
    }
    None
}

struct WorkerShared {
    latest_input: Option<(u64, EarthInput)>,
    shutdown: bool,
    stats: EarthViewportStats,
}

/// Asynchronous statistics exposed by [`EarthTileWorker`] and [`EarthViewport`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct EarthViewportStats {
    pub camera_updates: u64,
    pub worker_builds: u64,
    pub mesh_cache_hits: u64,
    pub mesh_cache_misses: u64,
    pub last_visible_tiles: usize,
    pub ancestor_fallbacks: usize,
}

/// Background worker thread that processes coalesced camera snapshots into retained GPU frames.
pub struct EarthTileWorker {
    shared: Arc<(Mutex<WorkerShared>, Condvar)>,
    completed_seq: Arc<AtomicU64>,
    thread: Option<JoinHandle<()>>,
    next_seq: u64,
}

impl EarthTileWorker {
    pub fn spawn(
        retained: RetainedScene3D,
        mut source: Box<dyn TileSource + Send>,
        wakeup: Option<Arc<dyn Fn() + Send + Sync>>,
    ) -> Self {
        let shared = Arc::new((
            Mutex::new(WorkerShared {
                latest_input: None,
                shutdown: false,
                stats: EarthViewportStats::default(),
            }),
            Condvar::new(),
        ));
        let completed_seq = Arc::new(AtomicU64::new(0));
        let shared_bg = Arc::clone(&shared);
        let completed_bg = Arc::clone(&completed_seq);

        let handle = std::thread::Builder::new()
            .name("strata-earth-worker".to_string())
            .spawn(move || {
                let mut builder = RetainedTerrainBuilder::new();
                let mut geom_rev = 0u64;
                loop {
                    let (seq, input) = {
                        let (lock, cv) = &*shared_bg;
                        let mut guard = lock.lock().unwrap();
                        while guard.latest_input.is_none() && !guard.shutdown {
                            guard = cv.wait(guard).unwrap();
                        }
                        if guard.shutdown {
                            return;
                        }
                        guard.latest_input.take().unwrap()
                    };

                    geom_rev += 1;
                    let (frame, build_stats) =
                        builder.build_frame(&input, source.as_mut(), geom_rev, true);
                    retained.update_terrain_preserving_camera(frame);

                    {
                        let (lock, cv) = &*shared_bg;
                        let mut guard = lock.lock().unwrap();
                        guard.stats.worker_builds += 1;
                        guard.stats.mesh_cache_hits += build_stats.mesh_cache_hits as u64;
                        guard.stats.mesh_cache_misses += build_stats.mesh_cache_misses as u64;
                        guard.stats.last_visible_tiles = build_stats.visible_tiles;
                        guard.stats.ancestor_fallbacks += build_stats.ancestor_fallbacks;
                        completed_bg.store(seq, Ordering::Release);
                        cv.notify_all();
                    }
                    if let Some(ref wake) = wakeup {
                        wake();
                    }
                }
            })
            .expect("spawn strata-earth-worker thread");

        Self {
            shared,
            completed_seq,
            thread: Some(handle),
            next_seq: 0,
        }
    }

    /// Queues `input` for background LOD traversal + tile assembly, coalescing any unstarted request.
    pub fn request_update(&mut self, input: EarthInput) -> u64 {
        self.next_seq += 1;
        let seq = self.next_seq;
        let (lock, cv) = &*self.shared;
        if let Ok(mut guard) = lock.lock() {
            guard.latest_input = Some((seq, input));
            guard.stats.camera_updates += 1;
            cv.notify_one();
        }
        seq
    }

    pub fn flush(&self) {
        let target = self.next_seq;
        let (lock, cv) = &*self.shared;
        let mut guard = lock.lock().unwrap();
        while self.completed_seq.load(Ordering::Acquire) < target && !guard.shutdown {
            guard = cv.wait(guard).unwrap();
        }
    }

    pub fn stats(&self) -> EarthViewportStats {
        self.shared.0.lock().map(|g| g.stats).unwrap_or_default()
    }
}

impl Drop for EarthTileWorker {
    fn drop(&mut self) {
        {
            let (lock, cv) = &*self.shared;
            if let Ok(mut guard) = lock.lock() {
                guard.shutdown = true;
                cv.notify_all();
            }
        }
        if let Some(h) = self.thread.take() {
            let _ = h.join();
        }
    }
}

/// High-frequency retained 3D/2D globe controller pairing O(1) GPU camera updates with
/// background LOD streaming, seamless ground-proximity 2.5D flattening, and smooth 3D/2D morphs.
pub struct EarthViewport {
    retained: RetainedScene3D,
    worker: EarthTileWorker,
    input: EarthInput,
    target_morph: f32,
    linear_morph: f32,
    transition_duration_s: f64,
    stopped: Arc<AtomicBool>,
}

impl EarthViewport {
    /// Creates an [`EarthViewport`] and immediately schedules the initial retained tile frame.
    pub fn new(input: &EarthInput, source: Box<dyn TileSource + Send>) -> Self {
        Self::with_wakeup(input, source, None)
    }

    pub fn with_wakeup(
        input: &EarthInput,
        source: Box<dyn TileSource + Send>,
        wakeup: Option<Arc<dyn Fn() + Send + Sync>>,
    ) -> Self {
        let initial_morph = input.low_level_state().map_2d_morph;
        let cam_uniforms = RetainedTerrainBuilder::camera_from_input(input);
        let retained = RetainedScene3D::new(cam_uniforms);
        let mut worker = EarthTileWorker::spawn(retained.clone(), source, wakeup);
        worker.request_update(input.clone());
        Self {
            retained,
            worker,
            input: input.clone(),
            target_morph: initial_morph,
            linear_morph: initial_morph,
            transition_duration_s: 0.45,
            stopped: Arc::new(AtomicBool::new(false)),
        }
    }

    #[inline]
    pub fn retained_scene(&self) -> RetainedScene3D {
        self.retained.clone()
    }

    /// Returns the high-level view mode (`Mode3D` or `Mode2D`).
    #[inline]
    pub fn view_mode(&self) -> EarthViewMode {
        self.input.view_mode
    }

    /// Returns the low-level internal surface regime and blend weights (`local_25d_blend`, `map_2d_morph`).
    #[inline]
    pub fn low_level_state(&self) -> EarthLowLevelState {
        self.input.low_level_state()
    }

    /// Programmatically initiates a smooth animated transition between `Mode3D` (with automatic
    /// seamless 2.5D ground transition) and `Mode2D`.
    pub fn set_view_mode(&mut self, mode: EarthViewMode) {
        self.input.view_mode = mode;
        self.target_morph = match mode {
            EarthViewMode::Mode3D => 0.0,
            EarthViewMode::Mode2D => 1.0,
        };
    }

    /// Programmatically switches between `Mode3D` and `Mode2D` immediately without animation.
    pub fn set_view_mode_immediate(&mut self, mode: EarthViewMode) {
        self.input.view_mode = mode;
        let val = match mode {
            EarthViewMode::Mode3D => 0.0,
            EarthViewMode::Mode2D => 1.0,
        };
        self.target_morph = val;
        self.linear_morph = val;
        self.input.view_morph = val;
        let cam_u = RetainedTerrainBuilder::camera_from_input(&self.input);
        self.retained.update_camera(cam_u);
        if !self.stopped.load(Ordering::Relaxed) {
            self.worker.request_update(self.input.clone());
        }
    }

    /// Returns `true` while a programmatic 3D <-> 2D morph transition is still in progress.
    #[inline]
    pub fn is_transitioning(&self) -> bool {
        (self.linear_morph - self.target_morph).abs() > 1e-4
    }

    /// Advances any active programmatic 3D <-> 2D transition by `dt_s` seconds.
    ///
    /// Returns `true` if a frame update was produced.
    pub fn step_transition(&mut self, dt_s: f64) -> bool {
        let diff = self.target_morph - self.linear_morph;
        if diff.abs() <= 1e-4 {
            if (self.input.view_morph - self.target_morph).abs() > 1e-4 {
                self.linear_morph = self.target_morph;
                self.input.view_morph = self.target_morph;
                let cam_u = RetainedTerrainBuilder::camera_from_input(&self.input);
                self.retained.update_camera(cam_u);
                if !self.stopped.load(Ordering::Relaxed) {
                    self.worker.request_update(self.input.clone());
                }
                return true;
            }
            return false;
        }
        let step = (dt_s.max(0.001) / self.transition_duration_s.max(0.05)) as f32;
        if diff > 0.0 {
            self.linear_morph = (self.linear_morph + step).min(self.target_morph);
        } else {
            self.linear_morph = (self.linear_morph - step).max(self.target_morph);
        }
        let t = self.linear_morph.clamp(0.0, 1.0);
        // Cubic Hermite ease-in-out
        self.input.view_morph = t * t * (3.0 - 2.0 * t);
        let cam_u = RetainedTerrainBuilder::camera_from_input(&self.input);
        self.retained.update_camera(cam_u);
        if !self.stopped.load(Ordering::Relaxed) {
            self.worker.request_update(self.input.clone());
        }
        true
    }

    /// Updates camera uniforms in O(1) (`camera_generation += 1`) with zero mesh/DOM/layout rebuild,
    /// and notifies the background worker to update LOD tiles asynchronously.
    pub fn update_camera(&mut self, camera: &EarthCamera, viewport_w: f64, viewport_h: f64) {
        self.input.camera = *camera;
        self.input.viewport.width = viewport_w.max(1.0) as f32;
        self.input.viewport.height = viewport_h.max(1.0) as f32;
        let cam_u = RetainedTerrainBuilder::camera_from_input(&self.input);
        self.retained.update_camera(cam_u);
        if !self.stopped.load(Ordering::Relaxed) {
            self.worker.request_update(self.input.clone());
        }
    }

    /// Updates full [`EarthInput`] configuration (`layers`, `surface_mode`, `camera`, `viewport`, etc.).
    pub fn update_input(&mut self, input: &EarthInput) {
        self.input = input.clone();
        let target = self.input.low_level_state().map_2d_morph;
        self.target_morph = target;
        self.linear_morph = target;
        let cam_u = RetainedTerrainBuilder::camera_from_input(&self.input);
        self.retained.update_camera(cam_u);
        if !self.stopped.load(Ordering::Relaxed) {
            self.worker.request_update(self.input.clone());
        }
    }

    /// Blocks until the most recently queued worker request has updated `RetainedScene3D`.
    pub fn sync_now(&self) {
        self.worker.flush();
    }

    pub fn stats(&self) -> EarthViewportStats {
        self.worker.stats()
    }
}
