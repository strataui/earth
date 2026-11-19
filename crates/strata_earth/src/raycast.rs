//! Terrain-, mode-, and object-aware 3D world raycasting for `strata_earth`.
//!
//! Accurately intersects screen rays across **3D Curved Globe**, **2.5D Morphed Tangent Plane**,
//! and **2D Flat Map**, respecting live `elevation_enabled` state (`DEM` heightmap bisection when
//! enabled; closed-form smooth `0.0 m` sea-level surface when disabled) as well as pickable
//! [`crate::world::EarthObject`]s in [`crate::world::EarthWorld`].

use strata_render::camera64::DVec3;

use crate::geodesy::{
    ecef_to_geodetic, geodetic_to_ecef, project_morphed_point_and_frame, raycast_ellipsoid,
    screen_to_ray_with_up, unproject_morphed_surface_lat_lon, wgs84, wrap_longitude, EarthCamera,
    Geodetic, LocalFrame, SurfaceMorphParams,
};
use crate::world::{CompiledMeshEntry, EarthObjectId, EarthWorld, ResolvedObjectTransform};

const MAX_TERRAIN_CEILING_M: f64 = 9_200.0;
const MIN_TERRAIN_FLOOR_M: f64 = -600.0;

/// Result of raycasting a screen coordinate or 3D ray against the planetary surface and [`EarthWorld`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EarthRayHit {
    /// Geodetic latitude in degrees (`[-90, 90]`).
    pub lat_deg: f64,
    /// Geodetic longitude in degrees (`[-180, 180]`).
    pub lon_deg: f64,
    /// World altitude in metres above WGS84 sea level at the intersection point.
    ///
    /// On ground hits: equals `terrain_elevation_m` when `elevation_enabled == true`, and `0.0`
    /// when `elevation_enabled == false`.
    pub altitude_m: f64,
    /// Sampled DEM terrain elevation at `(lat_deg, lon_deg)` (`0.0` when `elevation_enabled == false`).
    pub terrain_elevation_m: f64,
    /// Intersection position in world ECEF / morphed coordinates (metres).
    pub world_ecef: [f64; 3],
    /// Outward unit surface normal at the intersection in world ECEF / morphed coordinates.
    pub surface_normal_ecef: [f64; 3],
    /// Distance from camera eye along the ray in metres.
    pub distance_m: f64,
    /// `Some(id)` if the ray struck a pickable [`crate::world::EarthObject`] closer than the terrain.
    pub object_id: Option<EarthObjectId>,
}

/// Raycasts a viewport pixel `(screen_x, screen_y)` against both pickable [`EarthWorld`] objects
/// and the planetary surface (terrain heightmap when `elevation_enabled`, sea level when disabled).
#[allow(clippy::too_many_arguments)]
pub fn raycast_viewport(
    screen_x: f64,
    screen_y: f64,
    viewport_width: f64,
    viewport_height: f64,
    camera: &EarthCamera,
    up_override: Option<DVec3>,
    elevation_enabled: bool,
    morph: SurfaceMorphParams,
    world: Option<&EarthWorld>,
    dem_sampler: &dyn Fn(f64, f64) -> f64,
) -> Option<EarthRayHit> {
    if viewport_width <= 0.0 || viewport_height <= 0.0 {
        return None;
    }
    let (ray_origin, ray_dir) = screen_to_ray_with_up(
        screen_x,
        screen_y,
        viewport_width,
        viewport_height,
        camera,
        up_override,
    );
    raycast_ray(
        ray_origin,
        ray_dir,
        viewport_height,
        camera.fov_y_deg,
        elevation_enabled,
        morph,
        world,
        dem_sampler,
    )
}

/// Raycasts a world-space ray `(ray_origin, ray_dir)` against pickable [`EarthWorld`] objects and
/// the planetary surface.
#[allow(clippy::too_many_arguments)]
pub fn raycast_ray(
    ray_origin: DVec3,
    ray_dir: DVec3,
    viewport_height: f64,
    fov_y_deg: f64,
    elevation_enabled: bool,
    morph: SurfaceMorphParams,
    world: Option<&EarthWorld>,
    dem_sampler: &dyn Fn(f64, f64) -> f64,
) -> Option<EarthRayHit> {
    let dir = ray_dir.normalize();
    if dir.length() < 1e-6 {
        return None;
    }

    let terrain_hit = raycast_surface_ray(ray_origin, dir, elevation_enabled, morph, dem_sampler);
    let max_t = terrain_hit.map(|h| h.distance_m).unwrap_or(f64::INFINITY);

    if let Some(w) = world {
        if !w.is_empty() {
            if let Some(obj_hit) = raycast_world_objects(
                ray_origin,
                dir,
                max_t,
                viewport_height,
                fov_y_deg,
                elevation_enabled,
                morph,
                w,
                dem_sampler,
            ) {
                return Some(obj_hit);
            }
        }
    }

    terrain_hit
}

/// Raycasts strictly against the planetary surface (ignoring [`EarthWorld`] objects).
pub fn raycast_surface_ray(
    ray_origin: DVec3,
    ray_dir: DVec3,
    elevation_enabled: bool,
    morph: SurfaceMorphParams,
    dem_sampler: &dyn Fn(f64, f64) -> f64,
) -> Option<EarthRayHit> {
    let dir = ray_dir.normalize();
    if dir.length() < 1e-6 {
        return None;
    }

    let flat_t = morph.flat_blend.clamp(0.0, 1.0);
    let map_2d = morph.map_2d_morph.clamp(0.0, 1.0);

    // 1. Pure 2D flat map mode (`map_2d_morph ~ 1.0`, `flat_blend ~ 1.0`).
    if map_2d >= 0.999 && flat_t >= 0.999 {
        let focus_surface = Geodetic::new(morph.focus.lat_deg, morph.focus.lon_deg, 0.0);
        let focus_frame = LocalFrame::at(focus_surface.lat_deg, focus_surface.lon_deg);
        let focus_ecef = geodetic_to_ecef(focus_surface);
        let denom = dir.dot(focus_frame.up);
        if denom.abs() < 1e-8 {
            return None;
        }
        let t = focus_ecef.sub(ray_origin).dot(focus_frame.up) / denom;
        if t <= 1e-4 {
            return None;
        }
        let hit_pt = ray_origin.add(dir.scale(t));
        let (lat_deg, lon_deg) = unproject_morphed_surface_lat_lon(hit_pt, 0.0, morph);
        if !(-89.95..=89.95).contains(&lat_deg) {
            return None;
        }
        let terrain_elev = if elevation_enabled {
            dem_sampler(lat_deg, lon_deg)
        } else {
            0.0
        };
        return Some(EarthRayHit {
            lat_deg,
            lon_deg,
            altitude_m: terrain_elev,
            terrain_elevation_m: terrain_elev,
            world_ecef: [hit_pt.x, hit_pt.y, hit_pt.z],
            surface_normal_ecef: [focus_frame.up.x, focus_frame.up.y, focus_frame.up.z],
            distance_m: t,
            object_id: None,
        });
    }

    // 2. Smooth 0.0m sea-level surface when heightmap is disabled.
    if !elevation_enabled {
        if flat_t <= 1e-6 {
            let hit_pt = raycast_ellipsoid(ray_origin, dir)?;
            let t = hit_pt.sub(ray_origin).dot(dir);
            if t <= 1e-4 {
                return None;
            }
            let g = ecef_to_geodetic(hit_pt);
            let lon_deg = wrap_longitude(g.lon_deg);
            let frame = LocalFrame::at(g.lat_deg, lon_deg);
            return Some(EarthRayHit {
                lat_deg: g.lat_deg,
                lon_deg,
                altitude_m: 0.0,
                terrain_elevation_m: 0.0,
                world_ecef: [hit_pt.x, hit_pt.y, hit_pt.z],
                surface_normal_ecef: [frame.up.x, frame.up.y, frame.up.z],
                distance_m: t,
                object_id: None,
            });
        }

        if flat_t >= 1.0 - 1e-6 {
            let focus_surface = Geodetic::new(morph.focus.lat_deg, morph.focus.lon_deg, 0.0);
            let focus_frame = LocalFrame::at(focus_surface.lat_deg, focus_surface.lon_deg);
            let focus_ecef = geodetic_to_ecef(focus_surface);
            let denom = dir.dot(focus_frame.up);
            if denom.abs() < 1e-8 {
                return None;
            }
            let t = focus_ecef.sub(ray_origin).dot(focus_frame.up) / denom;
            if t <= 1e-4 {
                return None;
            }
            let hit_pt = ray_origin.add(dir.scale(t));
            let (lat_deg, lon_deg) = unproject_morphed_surface_lat_lon(hit_pt, 0.0, morph);
            if !(-89.95..=89.95).contains(&lat_deg) {
                return None;
            }
            return Some(EarthRayHit {
                lat_deg,
                lon_deg,
                altitude_m: 0.0,
                terrain_elevation_m: 0.0,
                world_ecef: [hit_pt.x, hit_pt.y, hit_pt.z],
                surface_normal_ecef: [focus_frame.up.x, focus_frame.up.y, focus_frame.up.z],
                distance_m: t,
                object_id: None,
            });
        }
    }

    // 3. General terrain-aware (and intermediate-morph) bracketed ray-march + bisection.
    let ceil_m = if elevation_enabled {
        MAX_TERRAIN_CEILING_M
    } else {
        50.0
    };
    let floor_m = if elevation_enabled {
        MIN_TERRAIN_FLOOR_M
    } else {
        -50.0
    };

    let (t_start, t_end) = compute_ray_search_bracket(ray_origin, dir, morph, ceil_m, floor_m)?;
    if t_end <= t_start {
        return None;
    }

    let eval_point = |t: f64| -> (f64, f64, f64, f64, DVec3) {
        let p = ray_origin.add(dir.scale(t));
        if flat_t <= 1e-6 {
            let g = ecef_to_geodetic(p);
            let lon = wrap_longitude(g.lon_deg);
            let terr = if elevation_enabled {
                dem_sampler(g.lat_deg, lon)
            } else {
                0.0
            };
            let up = LocalFrame::at(g.lat_deg, lon).up;
            return (g.height_m - terr, g.lat_deg, lon, terr, up);
        }
        if flat_t >= 1.0 - 1e-6 {
            let focus_surface = Geodetic::new(morph.focus.lat_deg, morph.focus.lon_deg, 0.0);
            let focus_frame = LocalFrame::at(focus_surface.lat_deg, focus_surface.lon_deg);
            let focus_ecef = geodetic_to_ecef(focus_surface);
            let plane_up = p.sub(focus_ecef).dot(focus_frame.up);
            let relief_scale = (1.0 - map_2d).max(1e-4);
            let ray_h = plane_up / relief_scale;
            let (lat, lon) = unproject_morphed_surface_lat_lon(p, ray_h, morph);
            let terr = if elevation_enabled {
                dem_sampler(lat, lon)
            } else {
                0.0
            };
            return (ray_h - terr, lat, lon, terr, focus_frame.up);
        }

        let (lat0, lon0) = unproject_morphed_surface_lat_lon(p, 0.0, morph);
        let (p0, frame0) = project_morphed_point_and_frame(lat0, lon0, 0.0, morph);
        let approx_h = p.sub(p0).dot(frame0.up) / (1.0 - map_2d * flat_t).max(0.05);
        let (lat, lon) = unproject_morphed_surface_lat_lon(p, approx_h, morph);
        let terr = if elevation_enabled {
            dem_sampler(lat, lon)
        } else {
            0.0
        };
        let (p_surf, surf_frame) = project_morphed_point_and_frame(lat, lon, terr, morph);
        let res = p.sub(p_surf).dot(surf_frame.up);
        (res, lat, lon, terr, surf_frame.up)
    };

    const MARCH_STEPS: usize = 96;
    let span = t_end - t_start;
    let sample_t = |i: usize| -> f64 {
        let u = i as f64 / MARCH_STEPS as f64;
        // Quadratic bias near t_start when span > 50 km for sharp foreground ridge capture.
        let s = if span > 50_000.0 {
            0.35 * u + 0.65 * u * u
        } else {
            u
        };
        t_start + span * s
    };

    let mut prev_t = t_start;
    let (mut prev_res, _, _, _, _) = eval_point(prev_t);
    let mut bracket: Option<(f64, f64, f64, f64)> = None;

    for step in 1..=MARCH_STEPS {
        let curr_t = sample_t(step);
        let (curr_res, _, _, _, _) = eval_point(curr_t);
        if prev_res >= 0.0 && curr_res <= 0.0 {
            bracket = Some((prev_t, curr_t, prev_res, curr_res));
            break;
        }
        prev_t = curr_t;
        prev_res = curr_res;
    }

    let (mut lo_t, mut hi_t, mut lo_res, mut hi_res) = bracket?;
    for _ in 0..14 {
        let mid_t = 0.5 * (lo_t + hi_t);
        let (mid_res, _, _, _, _) = eval_point(mid_t);
        if mid_res > 0.0 {
            lo_t = mid_t;
            lo_res = mid_res;
        } else {
            hi_t = mid_t;
            hi_res = mid_res;
        }
    }

    let denom = lo_res - hi_res;
    let hit_t = if denom.abs() > 1e-9 {
        (lo_t + (hi_t - lo_t) * (lo_res / denom)).clamp(lo_t, hi_t)
    } else {
        0.5 * (lo_t + hi_t)
    };

    let (_res, lat_deg, lon_deg, terrain_elev, surf_up) = eval_point(hit_t);
    if !(-89.95..=89.95).contains(&lat_deg) {
        return None;
    }

    let (exact_world, exact_frame) =
        project_morphed_point_and_frame(lat_deg, lon_deg, terrain_elev, morph);
    let normal = if elevation_enabled {
        estimate_terrain_normal(lat_deg, lon_deg, terrain_elev, morph, dem_sampler)
            .unwrap_or(exact_frame.up)
    } else {
        surf_up
    };

    Some(EarthRayHit {
        lat_deg,
        lon_deg,
        altitude_m: terrain_elev,
        terrain_elevation_m: terrain_elev,
        world_ecef: [exact_world.x, exact_world.y, exact_world.z],
        surface_normal_ecef: [normal.x, normal.y, normal.z],
        distance_m: hit_t,
        object_id: None,
    })
}

fn estimate_terrain_normal(
    lat_deg: f64,
    lon_deg: f64,
    center_elev_m: f64,
    morph: SurfaceMorphParams,
    dem_sampler: &dyn Fn(f64, f64) -> f64,
) -> Option<DVec3> {
    let eps_deg = 0.00025; // ~28m finite-difference baseline
    let lat_n = (lat_deg + eps_deg).clamp(-89.94, 89.94);
    let lon_e = wrap_longitude(lon_deg + eps_deg);
    let h_n = dem_sampler(lat_n, lon_deg);
    let h_e = dem_sampler(lat_deg, lon_e);
    let (p0, _) = project_morphed_point_and_frame(lat_deg, lon_deg, center_elev_m, morph);
    let (pe, _) = project_morphed_point_and_frame(lat_deg, lon_e, h_e, morph);
    let (pn, _) = project_morphed_point_and_frame(lat_n, lon_deg, h_n, morph);
    let n = pe.sub(p0).cross(pn.sub(p0));
    if n.length() > 1e-6 {
        Some(n.normalize())
    } else {
        None
    }
}

fn ray_ellipsoid_interval(
    ray_origin: DVec3,
    dir: DVec3,
    inflate_m: f64,
) -> Option<(f64, f64)> {
    let inv_a = 1.0 / (wgs84::A + inflate_m);
    let inv_b = 1.0 / (wgs84::B + inflate_m);
    let p0 = DVec3::new(
        ray_origin.x * inv_a,
        ray_origin.y * inv_a,
        ray_origin.z * inv_b,
    );
    let d = DVec3::new(dir.x * inv_a, dir.y * inv_a, dir.z * inv_b);
    let a = d.x * d.x + d.y * d.y + d.z * d.z;
    let b = 2.0 * (p0.x * d.x + p0.y * d.y + p0.z * d.z);
    let c = p0.x * p0.x + p0.y * p0.y + p0.z * p0.z - 1.0;
    let disc = b * b - 4.0 * a * c;
    if disc < 0.0 {
        return None;
    }
    let s = disc.sqrt();
    let t0 = (-b - s) / (2.0 * a);
    let t1 = (-b + s) / (2.0 * a);
    if t1 <= 1e-4 {
        return None;
    }
    Some((t0, t1))
}

fn ray_flat_band_interval(
    ray_origin: DVec3,
    dir: DVec3,
    morph: SurfaceMorphParams,
    ceil_m: f64,
    floor_m: f64,
) -> Option<(f64, f64)> {
    let focus_surface = Geodetic::new(morph.focus.lat_deg, morph.focus.lon_deg, 0.0);
    let focus_frame = LocalFrame::at(focus_surface.lat_deg, focus_surface.lon_deg);
    let focus_ecef = geodetic_to_ecef(focus_surface);
    let relief_scale = (1.0 - morph.map_2d_morph.clamp(0.0, 1.0)).max(1e-4);
    let z_ceil = ceil_m * relief_scale;
    let z_floor = floor_m * relief_scale;
    let u0 = ray_origin.sub(focus_ecef).dot(focus_frame.up);
    let du = dir.dot(focus_frame.up);

    if du < -1e-7 {
        let t_enter = if u0 > z_ceil {
            (z_ceil - u0) / du
        } else {
            0.01
        };
        let t_exit = (z_floor - u0) / du;
        if t_exit <= 1e-4 {
            return None;
        }
        Some((t_enter.max(0.01), t_exit.max(t_enter.max(0.01) + 1.0)))
    } else if du > 1e-7 {
        if u0 > z_ceil {
            return None;
        }
        let t_enter = if u0 < z_floor {
            ((z_floor - u0) / du).max(0.01)
        } else {
            0.01
        };
        let t_exit = ((z_ceil - u0) / du).min(200_000.0);
        if t_exit <= t_enter {
            return None;
        }
        Some((t_enter, t_exit))
    } else if u0 >= z_floor && u0 <= z_ceil {
        Some((0.01, 150_000.0))
    } else {
        None
    }
}

fn compute_ray_search_bracket(
    ray_origin: DVec3,
    dir: DVec3,
    morph: SurfaceMorphParams,
    ceil_m: f64,
    floor_m: f64,
) -> Option<(f64, f64)> {
    let flat_t = morph.flat_blend.clamp(0.0, 1.0);
    if flat_t >= 1.0 - 1e-6 {
        return ray_flat_band_interval(ray_origin, dir, morph, ceil_m, floor_m);
    }

    let curved = ray_ellipsoid_interval(ray_origin, dir, ceil_m).map(|(outer_t0, outer_t1)| {
        let t_start = outer_t0.max(0.01);
        let t_end = match ray_ellipsoid_interval(ray_origin, dir, floor_m) {
            Some((inner_t0, _)) if inner_t0 > t_start => inner_t0,
            _ => outer_t1,
        };
        (t_start, t_end)
    });

    if flat_t <= 1e-6 {
        return curved;
    }

    let flat = ray_flat_band_interval(ray_origin, dir, morph, ceil_m + 5_000.0, floor_m - 5_000.0);
    match (curved, flat) {
        (Some((c0, c1)), Some((f0, f1))) => Some((c0.min(f0).max(0.01), c1.max(f1))),
        (Some(c), None) => Some(c),
        (None, Some(f)) => Some(f),
        (None, None) => None,
    }
}

#[allow(clippy::too_many_arguments)]
fn raycast_world_objects(
    ray_origin: DVec3,
    dir: DVec3,
    max_t: f64,
    viewport_height: f64,
    fov_y_deg: f64,
    elevation_enabled: bool,
    morph: SurfaceMorphParams,
    world: &EarthWorld,
    dem_sampler: &dyn Fn(f64, f64) -> f64,
) -> Option<EarthRayHit> {
    let mut best_t = max_t;
    let mut best_hit: Option<(ResolvedObjectTransform, DVec3)> = None;

    for (tf, mesh) in world.iter_pickable_resolved(
        ray_origin,
        viewport_height,
        fov_y_deg,
        elevation_enabled,
        morph,
        dem_sampler,
    ) {
        let anchor = DVec3::new(tf.anchor_ecef[0], tf.anchor_ecef[1], tf.anchor_ecef[2]);
        let oc = anchor.sub(ray_origin);
        let proj = oc.dot(dir);
        let r = tf.world_bounding_radius_m.max(0.1);
        if proj + r < 0.0 || proj - r > best_t {
            continue;
        }
        let perp_sq = oc.dot(oc) - proj * proj;
        if perp_sq > r * r * 1.15 {
            continue;
        }

        if let Some((t_hit, world_normal)) =
            intersect_transformed_mesh(ray_origin, dir, &tf, mesh, best_t)
        {
            if t_hit < best_t {
                best_t = t_hit;
                best_hit = Some((tf, world_normal));
            }
        }
    }

    let (tf, normal) = best_hit?;
    let hit_pt = ray_origin.add(dir.scale(best_t));
    let (lat_deg, lon_deg) =
        unproject_morphed_surface_lat_lon(hit_pt, tf.resolved_altitude_m, morph);
    let terrain_elev = if elevation_enabled {
        dem_sampler(lat_deg, lon_deg)
    } else {
        0.0
    };
    let (p_zero, frame_zero) = project_morphed_point_and_frame(lat_deg, lon_deg, 0.0, morph);
    let relief_scale = (1.0 - morph.map_2d_morph.clamp(0.0, 1.0)).max(0.05);
    let altitude_m = if morph.map_2d_morph >= 0.999 {
        tf.resolved_altitude_m
    } else {
        hit_pt.sub(p_zero).dot(frame_zero.up) / relief_scale
    };

    Some(EarthRayHit {
        lat_deg,
        lon_deg,
        altitude_m,
        terrain_elevation_m: terrain_elev,
        world_ecef: [hit_pt.x, hit_pt.y, hit_pt.z],
        surface_normal_ecef: [normal.x, normal.y, normal.z],
        distance_m: best_t,
        object_id: Some(tf.object_id),
    })
}

fn intersect_transformed_mesh(
    ray_origin: DVec3,
    dir: DVec3,
    tf: &ResolvedObjectTransform,
    mesh: &CompiledMeshEntry,
    max_t: f64,
) -> Option<(f64, DVec3)> {
    let c0 = DVec3::new(tf.model_cols[0][0], tf.model_cols[0][1], tf.model_cols[0][2]);
    let c1 = DVec3::new(tf.model_cols[1][0], tf.model_cols[1][1], tf.model_cols[1][2]);
    let c2 = DVec3::new(tf.model_cols[2][0], tf.model_cols[2][1], tf.model_cols[2][2]);

    let r0 = c1.cross(c2);
    let det = c0.dot(r0);
    if det.abs() < 1e-12 {
        return None;
    }
    let inv_det = 1.0 / det;
    let r1 = c2.cross(c0);
    let r2 = c0.cross(c1);

    let anchor = DVec3::new(tf.anchor_ecef[0], tf.anchor_ecef[1], tf.anchor_ecef[2]);
    let rel_o = ray_origin.sub(anchor);
    let lo = [
        r0.dot(rel_o) * inv_det,
        r1.dot(rel_o) * inv_det,
        r2.dot(rel_o) * inv_det,
    ];
    let ld = [
        r0.dot(dir) * inv_det,
        r1.dot(dir) * inv_det,
        r2.dot(dir) * inv_det,
    ];

    if !ray_intersects_aabb(lo, ld, tf.local_aabb_min, tf.local_aabb_max, max_t) {
        return None;
    }

    let verts = &mesh.asset.vertices;
    let inds = &mesh.asset.indices;
    let mut best_t = max_t;
    let mut best_local_n: Option<[f64; 3]> = None;

    for tri in inds.chunks_exact(3) {
        let p0 = verts[tri[0] as usize].position();
        let p1 = verts[tri[1] as usize].position();
        let p2 = verts[tri[2] as usize].position();
        if let Some((t, ln)) = ray_triangle_mt(lo, ld, p0, p1, p2) {
            if t > 1e-3 && t < best_t {
                best_t = t;
                best_local_n = Some(ln);
            }
        }
    }

    let ln = best_local_n?;
    let wn_raw = r0
        .scale(ln[0] * det.signum())
        .add(r1.scale(ln[1] * det.signum()))
        .add(r2.scale(ln[2] * det.signum()));
    let wn = if wn_raw.length() > 1e-9 {
        let mut n = wn_raw.normalize();
        if n.dot(dir) > 0.0 {
            n = n.scale(-1.0);
        }
        n
    } else {
        dir.scale(-1.0)
    };
    Some((best_t, wn))
}

#[inline]
fn ray_intersects_aabb(
    lo: [f64; 3],
    ld: [f64; 3],
    min_b: [f32; 3],
    max_b: [f32; 3],
    max_t: f64,
) -> bool {
    let mut t_min = 0.0_f64;
    let mut t_max = max_t;
    for k in 0..3 {
        let b0 = min_b[k] as f64 - 1e-4;
        let b1 = max_b[k] as f64 + 1e-4;
        if ld[k].abs() < 1e-12 {
            if lo[k] < b0 || lo[k] > b1 {
                return false;
            }
        } else {
            let inv = 1.0 / ld[k];
            let mut t1 = (b0 - lo[k]) * inv;
            let mut t2 = (b1 - lo[k]) * inv;
            if t1 > t2 {
                std::mem::swap(&mut t1, &mut t2);
            }
            t_min = t_min.max(t1);
            t_max = t_max.min(t2);
            if t_min > t_max {
                return false;
            }
        }
    }
    t_max > 1e-4
}

#[inline]
fn ray_triangle_mt(
    lo: [f64; 3],
    ld: [f64; 3],
    p0: [f32; 3],
    p1: [f32; 3],
    p2: [f32; 3],
) -> Option<(f64, [f64; 3])> {
    let v0 = [p0[0] as f64, p0[1] as f64, p0[2] as f64];
    let e1 = [
        p1[0] as f64 - v0[0],
        p1[1] as f64 - v0[1],
        p1[2] as f64 - v0[2],
    ];
    let e2 = [
        p2[0] as f64 - v0[0],
        p2[1] as f64 - v0[1],
        p2[2] as f64 - v0[2],
    ];
    let h = [
        ld[1] * e2[2] - ld[2] * e2[1],
        ld[2] * e2[0] - ld[0] * e2[2],
        ld[0] * e2[1] - ld[1] * e2[0],
    ];
    let a = e1[0] * h[0] + e1[1] * h[1] + e1[2] * h[2];
    if a.abs() < 1e-12 {
        return None;
    }
    let f = 1.0 / a;
    let s = [lo[0] - v0[0], lo[1] - v0[1], lo[2] - v0[2]];
    let u = f * (s[0] * h[0] + s[1] * h[1] + s[2] * h[2]);
    if !(-1e-5..=1.00001).contains(&u) {
        return None;
    }
    let q = [
        s[1] * e1[2] - s[2] * e1[1],
        s[2] * e1[0] - s[0] * e1[2],
        s[0] * e1[1] - s[1] * e1[0],
    ];
    let v = f * (ld[0] * q[0] + ld[1] * q[1] + ld[2] * q[2]);
    if v < -1e-5 || u + v > 1.00001 {
        return None;
    }
    let t = f * (e2[0] * q[0] + e2[1] * q[1] + e2[2] * q[2]);
    let n = [
        e1[1] * e2[2] - e1[2] * e2[1],
        e1[2] * e2[0] - e1[0] * e2[2],
        e1[0] * e2[1] - e1[1] * e2[0],
    ];
    Some((t, n))
}
