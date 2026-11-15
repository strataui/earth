//! Geodesy, camera control, and tile quadtree for the Earth scene plugin.
//!
//! Coordinates are WGS84-ish ECEF metres in f64. Nothing here narrows to f32;
//! the render hand-off is [`crate::EarthScene::build`].

use strata_render::camera64::{DMat4, DVec3, EARTH_RADIUS_M};

/// WGS84 ellipsoid constants.
pub mod wgs84 {
    pub const A: f64 = 6_378_137.0;
    pub const F: f64 = 1.0 / 298.257_223_563;
    pub const B: f64 = A * (1.0 - F);
    pub const E2: f64 = F * (2.0 - F);
}

/// WGS84 geodetic position.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Geodetic {
    /// Latitude in degrees, positive north.
    pub lat_deg: f64,
    /// Longitude in degrees, positive east.
    pub lon_deg: f64,
    /// Height above the ellipsoid in metres.
    pub height_m: f64,
}

impl Geodetic {
    pub const fn new(lat_deg: f64, lon_deg: f64, height_m: f64) -> Self {
        Self {
            lat_deg,
            lon_deg,
            height_m,
        }
    }
}

/// Converts a geodetic position to ECEF metres.
pub fn geodetic_to_ecef(position: Geodetic) -> DVec3 {
    let lat = position.lat_deg.to_radians();
    let lon = position.lon_deg.to_radians();
    let (sin_lat, cos_lat) = lat.sin_cos();
    let (sin_lon, cos_lon) = lon.sin_cos();

    // Radius of curvature in the prime vertical.
    let n = wgs84::A / (1.0 - wgs84::E2 * sin_lat * sin_lat).sqrt();
    let x = (n + position.height_m) * cos_lat * cos_lon;
    let y = (n + position.height_m) * cos_lat * sin_lon;
    let z = (n * (1.0 - wgs84::E2) + position.height_m) * sin_lat;
    DVec3::new(x, y, z)
}

/// Converts an ECEF Cartesian position in metres to WGS84 geodetic coordinates using
/// Bowring's vector inversion algorithm.
///
/// Accurate to sub-millimetre precision worldwide across all terrestrial and orbital
/// altitudes.
pub fn ecef_to_geodetic(ecef: DVec3) -> Geodetic {
    let p = (ecef.x * ecef.x + ecef.y * ecef.y).sqrt();
    if p < 1e-6 {
        let lat = if ecef.z >= 0.0 { 90.0 } else { -90.0 };
        let height = ecef.z.abs() - wgs84::B;
        return Geodetic::new(lat, 0.0, height);
    }

    let a = wgs84::A;
    let b = wgs84::B;
    let e2 = wgs84::E2;
    let ep2 = (a * a - b * b) / (b * b);

    let theta = (ecef.z * a).atan2(p * b);
    let (sin_theta, cos_theta) = theta.sin_cos();

    let mut lat = (ecef.z + ep2 * b * sin_theta.powi(3)).atan2(p - e2 * a * cos_theta.powi(3));
    for _ in 0..2 {
        let sin_l = lat.sin();
        let n_cur = a / (1.0 - e2 * sin_l * sin_l).sqrt();
        lat = (ecef.z + e2 * n_cur * sin_l).atan2(p);
    }
    let lon = ecef.y.atan2(ecef.x);

    let (sin_lat, cos_lat) = lat.sin_cos();
    let n = a / (1.0 - e2 * sin_lat * sin_lat).sqrt();
    let height = if cos_lat.abs() > 1e-6 {
        p / cos_lat - n
    } else {
        ecef.z.abs() / sin_lat.abs() - n * (1.0 - e2)
    };

    Geodetic::new(lat.to_degrees(), lon.to_degrees(), height)
}

/// Intersects a 3D ray `(ray_origin, ray_dir)` with the WGS84 reference ellipsoid.
///
/// Returns the closest forward intersection point in ECEF coordinates, or `None`
/// if the ray misses the planet.
pub fn raycast_ellipsoid(ray_origin: DVec3, ray_dir: DVec3) -> Option<DVec3> {
    let dir = ray_dir.normalize();
    if dir.length() < 1e-6 {
        return None;
    }

    let inv_a = 1.0 / wgs84::A;
    let inv_b = 1.0 / wgs84::B;

    let p0 = DVec3::new(
        ray_origin.x * inv_a,
        ray_origin.y * inv_a,
        ray_origin.z * inv_b,
    );
    let d = DVec3::new(dir.x * inv_a, dir.y * inv_a, dir.z * inv_b);

    let a_coeff = d.x * d.x + d.y * d.y + d.z * d.z;
    let b_coeff = 2.0 * (p0.x * d.x + p0.y * d.y + p0.z * d.z);
    let c_coeff = p0.x * p0.x + p0.y * p0.y + p0.z * p0.z - 1.0;

    let discriminant = b_coeff * b_coeff - 4.0 * a_coeff * c_coeff;
    if discriminant < 0.0 {
        return None;
    }

    let sqrt_disc = discriminant.sqrt();
    let t1 = (-b_coeff - sqrt_disc) / (2.0 * a_coeff);
    let t2 = (-b_coeff + sqrt_disc) / (2.0 * a_coeff);

    let t = if t1 > 1e-4 {
        t1
    } else if t2 > 1e-4 {
        t2
    } else {
        return None;
    };

    Some(ray_origin.add(dir.scale(t)))
}

/// The local east/north/up basis at a geodetic position, in ECEF.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LocalFrame {
    pub east: DVec3,
    pub north: DVec3,
    pub up: DVec3,
}

impl LocalFrame {
    /// Computes the ENU basis at the given latitude and longitude.
    pub fn at(lat_deg: f64, lon_deg: f64) -> Self {
        let lat = lat_deg.to_radians();
        let lon = lon_deg.to_radians();
        let (sin_lat, cos_lat) = lat.sin_cos();
        let (sin_lon, cos_lon) = lon.sin_cos();
        Self {
            east: DVec3::new(-sin_lon, cos_lon, 0.0),
            north: DVec3::new(-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat),
            up: DVec3::new(cos_lat * cos_lon, cos_lat * sin_lon, sin_lat),
        }
    }
}

/// Camera control for the globe, expressed as a ground focus point plus a
/// geodesic eye offset in the local frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EarthCamera {
    /// Point on the globe the camera looks at.
    pub focus: Geodetic,
    /// Aim point offset, metres east, north, and up from `focus`.
    pub aim_offset_m: [f64; 3],
    /// Eye offset from `focus`, metres east, north, and up.
    ///
    /// The default points north along the tangent, so the camera hovers
    /// `altitude` above the ground with the local up as its screen up.
    pub eye_offset_m: [f64; 3],
    /// Vertical field of view in degrees.
    pub fov_y_deg: f64,
    /// Near plane in metres.
    pub near_m: f64,
}

impl Default for EarthCamera {
    fn default() -> Self {
        Self {
            focus: Geodetic::new(0.0, 0.0, 0.0),
            aim_offset_m: [0.0, 0.0, 0.0],
            eye_offset_m: [0.0, 0.0, 3.0 * EARTH_RADIUS_M],
            fov_y_deg: 60.0,
            near_m: 1.0,
        }
    }
}

/// Altitude at or above which the camera is treated as a globe view.
pub const GLOBE_ALTITUDE_M: f64 = 2_400_000.0;

/// Distance from the eye to the horizon of a sphere of Earth radius.
///
/// `sqrt(2 R h + h^2)` is exact for an eye `h` above the surface; it is zero at
/// sea level and grows as the eye climbs.
pub fn horizon_distance_m(altitude_m: f64) -> f64 {
    let h = altitude_m.max(0.0);
    (2.0 * EARTH_RADIUS_M * h + h * h).sqrt()
}

/// Picks the near and far planes for an altitude.
///
/// Kept separate from [`EarthCamera::placement`] so the policy can be tested
/// directly. `requested_near` is honoured only when it is larger than the
/// altitude-derived value.
pub fn depth_range(altitude_m: f64, horizon_m: f64, requested_near: f64) -> (f64, f64) {
    let altitude = altitude_m.max(0.0);
    let near = (altitude / 100.0).max(requested_near).clamp(0.05, 1.0e6);
    let far = if altitude >= GLOBE_ALTITUDE_M {
        horizon_m * 2.0 + 100_000.0
    } else {
        horizon_m * 1.5 + 50_000.0
    };
    (near, far.max(near * 100.0))
}

/// A resolved eye/aim pair in ECEF with a depth range matched to its altitude.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CameraPlacement {
    pub eye: DVec3,
    pub aim: DVec3,
    pub up: DVec3,
    pub fov_y_deg: f64,
    pub near_m: f64,
    pub far_m: f64,
}

impl EarthCamera {
    /// Distance from the focus point to the eye in metres.
    pub fn eye_distance_m(&self) -> f64 {
        let [e, n, u] = self.eye_offset_m;
        (e * e + n * n + u * u).sqrt()
    }

    /// Altitude of the eye above the ellipsoid in metres.
    ///
    /// The up component is the height over the focus point; a horizontal
    /// offset is a chord along the tangent plane and lifts the eye by
    /// `horizontal^2 / 2R` above that plane.
    pub fn altitude_m(&self) -> f64 {
        let [e, n, u] = self.eye_offset_m;
        let horizontal_sq = e * e + n * n;
        let radius = EARTH_RADIUS_M + self.focus.height_m.max(0.0);
        (self.focus.height_m + u + horizontal_sq / (2.0 * radius)).max(0.0)
    }

    /// Ground width covered by the vertical field of view, in metres.
    ///
    /// The texture rasterizer keeps only the frame around each triangle, so
    /// terrain patches are sized from this so they stay partly on screen. The
    /// eye-to-aim distance is used rather than altitude: an oblique camera sees
    /// a wider footprint at the same altitude, and an altitude-only span leaves
    /// the patch sheet too small to cover the view.
    pub fn ground_span_m(&self) -> f64 {
        let distance = self.eye_distance_m().max(1e-3);
        2.0 * distance * (self.fov_y_deg.to_radians() * 0.5).tan()
    }

    /// Resolves the camera to ECEF positions and a depth range suited to the
    /// altitude.
    ///
    /// Depth precision is governed by the near-to-far ratio, not by the far
    /// plane alone. Two regimes are used:
    ///
    /// - Below [`GLOBE_ALTITUDE_M`] the camera is local, so the far plane only
    ///   has to reach the local horizon (plus the cloud shell and a margin).
    /// - At or above it the camera is a globe view, so the far plane reaches
    ///   the limb twice over.
    ///
    /// In both cases the near plane is `altitude / 100`, so the closest
    /// geometry the camera can see sits at least a hundred near planes away.
    /// That keeps a one-metre offset resolvable at terrain altitudes while
    /// never clipping the surface below the eye.
    pub fn placement(&self) -> CameraPlacement {
        let anchor = geodetic_to_ecef(self.focus);
        let frame = LocalFrame::at(self.focus.lat_deg, self.focus.lon_deg);
        let offset = |[e, n, u]: [f64; 3]| {
            frame
                .east
                .scale(e)
                .add(frame.north.scale(n))
                .add(frame.up.scale(u))
        };
        let eye = anchor.add(offset(self.eye_offset_m));
        let aim = anchor.add(offset(self.aim_offset_m));
        // The camera's up vector is the local up, so the globe's north stays
        // up in the frame.
        let up = frame.up;
        let altitude = self.altitude_m();
        let horizon = horizon_distance_m(altitude);
        let (near, far) = depth_range(altitude, horizon, self.near_m);
        CameraPlacement {
            eye,
            aim,
            up,
            fov_y_deg: self.fov_y_deg,
            near_m: near,
            far_m: far,
        }
    }
}

/// Unprojects a 2D screen coordinate in the viewport to a 3D ray in ECEF world space.
///
/// Returns `(ray_origin, ray_direction)` in ECEF metres.
pub fn screen_to_ray(
    screen_x: f64,
    screen_y: f64,
    viewport_width: f64,
    viewport_height: f64,
    camera: &EarthCamera,
) -> (DVec3, DVec3) {
    let placement = camera.placement();
    let eye = placement.eye;
    let aim = placement.aim;

    let forward = aim.sub(eye).normalize();
    let cam_up = if forward.cross(placement.up).length() < 1e-6 {
        let frame = LocalFrame::at(camera.focus.lat_deg, camera.focus.lon_deg);
        if forward.cross(frame.north).length() > 1e-6 {
            frame.north
        } else if forward.cross(frame.east).length() > 1e-6 {
            frame.east
        } else if forward.cross(DVec3::new(0.0, 0.0, 1.0)).length() > 1e-6 {
            DVec3::new(0.0, 0.0, 1.0)
        } else {
            DVec3::new(0.0, 1.0, 0.0)
        }
    } else {
        placement.up
    };
    let right = forward.cross(cam_up).normalize();
    let up = right.cross(forward).normalize();

    let aspect = if viewport_height.abs() > 1e-6 {
        viewport_width / viewport_height
    } else {
        1.0
    };

    let fov_y = placement.fov_y_deg.to_radians();
    let tan_half_y = (fov_y * 0.5).tan();
    let tan_half_x = tan_half_y * aspect;

    let ndc_x = (screen_x / viewport_width.max(1.0)) * 2.0 - 1.0;
    let ndc_y = 1.0 - (screen_y / viewport_height.max(1.0)) * 2.0;

    let dir = right
        .scale(ndc_x * tan_half_x)
        .add(up.scale(ndc_y * tan_half_y))
        .add(forward)
        .normalize();

    (eye, dir)
}

/// Picks a geographic coordinate on the WGS84 ellipsoid under a screen position.
///
/// Returns `Some(Geodetic)` with surface height if the ray intersects the planet,
/// or `None` if the cursor is pointed at space.
pub fn pick_geodetic(
    screen_x: f64,
    screen_y: f64,
    viewport_width: f64,
    viewport_height: f64,
    camera: &EarthCamera,
) -> Option<Geodetic> {
    let (ray_origin, ray_dir) =
        screen_to_ray(screen_x, screen_y, viewport_width, viewport_height, camera);
    let hit_ecef = raycast_ellipsoid(ray_origin, ray_dir)?;
    Some(ecef_to_geodetic(hit_ecef))
}

/// A node in the tile quadtree.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TileKey {
    /// Zoom level, 0 = whole world.
    pub level: u8,
    pub x: u32,
    pub y: u32,
}

impl TileKey {
    pub const fn new(level: u8, x: u32, y: u32) -> Self {
        Self { level, x, y }
    }

    /// Number of tiles per axis at this level.
    pub fn tiles_per_axis(&self) -> u32 {
        1u32 << self.level
    }

    /// The parent tile, or `None` at level 0.
    pub fn parent(&self) -> Option<Self> {
        if self.level == 0 {
            None
        } else {
            Some(Self::new(self.level - 1, self.x / 2, self.y / 2))
        }
    }

    /// The child quadrant containing this tile's `(u, v)` position.
    pub fn child(&self, quadrant: u8) -> Option<Self> {
        if self.level >= 30 {
            return None;
        }
        let (dx, dy) = (
            ((quadrant & 1) != 0) as u32,
            (((quadrant >> 1) & 1) != 0) as u32,
        );
        let base = Self::new(self.level + 1, self.x * 2, self.y * 2);
        let per_axis = base.tiles_per_axis();
        let x = base.x + dx;
        let y = base.y + dy;
        if x < per_axis && y < per_axis {
            Some(Self::new(base.level, x, y))
        } else {
            None
        }
    }

    /// Maps a normalized `(u, v)` inside this tile into the parent's space.
    pub fn uv_in_parent(&self, u: f64, v: f64) -> (f64, f64) {
        let x_parity = self.x & 1;
        let y_parity = self.y & 1;
        ((u + x_parity as f64) * 0.5, (v + y_parity as f64) * 0.5)
    }
}

/// Resolves the sub-rectangle of a loaded ancestor image that a tile shows.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TileTextureResolution {
    /// Tile whose image should be sampled.
    pub source: TileKey,
    /// Sub-rectangle of the source image in `0..1` UV space:
    /// `(u_min, v_min, u_max, v_max)`.
    pub uv_rect: (f64, f64, f64, f64),
    /// True when the requested tile itself was available.
    pub exact: bool,
}

/// Resolves which loaded tile supplies the imagery for `target`.
///
/// Walks ancestors nearest-first and accumulates the UV sub-rectangle in the
/// ancestor's space. `is_loaded` decides availability; the function is pure so
/// it can be tested without a cache or a renderer.
pub fn resolve_tile_imagery(
    target: TileKey,
    mut is_loaded: impl FnMut(TileKey) -> bool,
) -> Option<TileTextureResolution> {
    if is_loaded(target) {
        return Some(TileTextureResolution {
            source: target,
            uv_rect: (0.0, 0.0, 1.0, 1.0),
            exact: true,
        });
    }

    let (mut u_min, mut v_min, mut u_max, mut v_max) = (0.0f64, 0.0f64, 1.0f64, 1.0f64);
    let mut current = target;
    while let Some(parent) = current.parent() {
        let (pu_min, pv_min) = current.uv_in_parent(u_min, v_min);
        let (pu_max, pv_max) = current.uv_in_parent(u_max, v_max);
        u_min = pu_min;
        v_min = pv_min;
        u_max = pu_max;
        v_max = pv_max;
        current = parent;
        if is_loaded(current) {
            return Some(TileTextureResolution {
                source: current,
                uv_rect: (u_min, v_min, u_max, v_max),
                exact: false,
            });
        }
    }
    None
}

/// Picks the tile level for a camera altitude: one level per factor-of-two
/// reduction in eye height above the ground, clamped to `0..=max_level`.
pub fn level_for_altitude(altitude_m: f64, max_level: u8) -> u8 {
    if altitude_m <= 0.0 {
        return max_level;
    }
    // A whole-earth view sits around 20 000 km up; halving from there adds a
    // level each time.
    let reference = 20_000_000.0;
    let ratio = (reference / altitude_m.max(1.0)).log2();
    (ratio.max(0.0).round() as u8).min(max_level)
}

/// Wraps a longitude into `-180..180`.
pub fn wrap_longitude(lon_deg: f64) -> f64 {
    (lon_deg + 180.0).rem_euclid(360.0) - 180.0
}

/// Clamps a latitude to the polar limit.
pub fn clamp_latitude(lat_deg: f64) -> f64 {
    lat_deg.clamp(-90.0, 90.0)
}

/// Geographic bounds of one terrain patch or tile.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PatchBounds {
    /// West edge longitude in degrees.
    pub lon0: f64,
    /// East edge longitude in degrees.
    pub lon1: f64,
    /// South edge latitude in degrees.
    pub lat0: f64,
    /// North edge latitude in degrees.
    pub lat1: f64,
}

/// Geographic bounds of a tile at its own level.
pub fn tile_bounds(key: TileKey) -> PatchBounds {
    let per_axis = key.tiles_per_axis() as f64;
    let lon_span = 360.0 / per_axis;
    let lat_span = 180.0 / per_axis;
    let lon0 = key.x as f64 * lon_span - 180.0;
    let lat1 = 90.0 - key.y as f64 * lat_span;
    PatchBounds {
        lon0,
        lon1: lon0 + lon_span,
        lat0: lat1 - lat_span,
        lat1,
    }
}

/// Computes the 3D bounding sphere center (ECEF `DVec3`) and radius (metres)
/// for a `TileKey` on the WGS84 ellipsoid.
pub fn tile_bounding_sphere(key: TileKey) -> (DVec3, f64) {
    if key.level == 0 {
        return (DVec3::new(0.0, 0.0, 0.0), wgs84::A * 1.08);
    }
    let bounds = tile_bounds(key);
    let mid_lon = (bounds.lon0 + bounds.lon1) * 0.5;
    let mid_lat = (bounds.lat0 + bounds.lat1) * 0.5;
    let center = geodetic_to_ecef(Geodetic::new(mid_lat, mid_lon, 0.0));

    let sample_points = [
        Geodetic::new(bounds.lat0, bounds.lon0, 0.0),
        Geodetic::new(bounds.lat0, bounds.lon1, 0.0),
        Geodetic::new(bounds.lat1, bounds.lon0, 0.0),
        Geodetic::new(bounds.lat1, bounds.lon1, 0.0),
        Geodetic::new(bounds.lat0, mid_lon, 0.0),
        Geodetic::new(bounds.lat1, mid_lon, 0.0),
        Geodetic::new(mid_lat, bounds.lon0, 0.0),
        Geodetic::new(mid_lat, bounds.lon1, 0.0),
    ];

    let mut max_dist_sq = 0.0f64;
    for pt in &sample_points {
        let ecef = geodetic_to_ecef(*pt);
        let diff = ecef.sub(center);
        let dist_sq = diff.dot(diff);
        if dist_sq > max_dist_sq {
            max_dist_sq = dist_sq;
        }
    }

    // Include 5000 metres margin for topography relief and curvature
    let radius = max_dist_sq.sqrt() + 5000.0;
    (center, radius)
}

/// Tests whether a tile bounding sphere is occluded behind the WGS84 ellipsoid horizon.
pub fn is_tile_horizon_culled(center: DVec3, radius: f64, eye: DVec3) -> bool {
    let eye_dist = eye.length();
    let b = wgs84::B;
    if eye_dist <= b {
        return false;
    }
    let eye_dir = eye.scale(1.0 / eye_dist);
    let horizon_plane_dist = (b * b) / eye_dist;
    let proj = center.dot(eye_dir);
    (proj + radius) < horizon_plane_dist
}

/// Tests whether a tile bounding sphere intersects the camera view frustum cone.
pub fn is_tile_in_frustum(
    center: DVec3,
    radius: f64,
    eye: DVec3,
    aim: DVec3,
    fov_y_deg: f64,
    aspect: f64,
) -> bool {
    let to_aim = aim.sub(eye);
    if to_aim.length() < 1e-6 {
        return true;
    }
    let forward = to_aim.normalize();
    let to_center = center.sub(eye);
    let center_dist = to_center.length();

    if center_dist <= radius {
        return true;
    }

    let forward_dist = to_center.dot(forward);
    if forward_dist + radius < 0.0 {
        return false;
    }

    let half_fov_y = (fov_y_deg.to_radians() * 0.5).tan();
    let half_fov_x = half_fov_y * aspect.max(1e-6);
    let frustum_half_angle = (half_fov_x * half_fov_x + half_fov_y * half_fov_y)
        .sqrt()
        .atan();
    let max_angle = frustum_half_angle * 1.35;

    let center_dir = to_center.scale(1.0 / center_dist);
    let cos_angle = center_dir.dot(forward).clamp(-1.0, 1.0);
    let angle = cos_angle.acos();
    let angular_radius = (radius / center_dist).min(1.0).asin();

    angle - angular_radius <= max_angle
}

/// Computes the Screen-Space Error (SSE) in pixels for a tile viewed from distance `dist_to_eye`.
pub fn compute_tile_sse(
    tile_span_m: f64,
    grid_resolution: u32,
    dist_to_eye: f64,
    viewport_height: f64,
    fov_y_deg: f64,
) -> f64 {
    let grid = grid_resolution.max(2) as f64;
    let geometric_error_m = tile_span_m / (grid - 1.0);
    let fov_rad = fov_y_deg.to_radians();
    let tan_half_fov = (fov_rad * 0.5).tan().max(1e-6);
    let dist = dist_to_eye.max(1.0);

    (geometric_error_m * viewport_height) / (2.0 * dist * tan_half_fov)
}

/// Hierarchical quadtree LOD traversal over the WGS84 planetary ellipsoid.
///
/// Subdivides nodes from Level 0 down to `max_level` based on horizon culling,
/// frustum culling, and Screen-Space Error (SSE) thresholds. Emits all visible
/// leaf tiles to `on_leaf_tile`.
#[allow(clippy::too_many_arguments)]
pub fn traverse_quadtree_lod(
    eye: DVec3,
    aim: DVec3,
    fov_y_deg: f64,
    aspect: f64,
    viewport_height: f64,
    sse_threshold: f64,
    max_level: u8,
    tile_grid: u32,
    mut on_leaf_tile: impl FnMut(TileKey, PatchBounds),
) {
    let mut stack = Vec::with_capacity(64);
    stack.push(TileKey::new(0, 0, 0));

    while let Some(key) = stack.pop() {
        let (center, radius) = tile_bounding_sphere(key);

        // 1. Horizon culling
        if is_tile_horizon_culled(center, radius, eye) {
            continue;
        }

        // 2. Frustum culling
        if !is_tile_in_frustum(center, radius, eye, aim, fov_y_deg, aspect) {
            continue;
        }

        // 3. Screen-Space Error calculation
        let bounds = tile_bounds(key);
        let dist_to_eye = (center.sub(eye).length() - radius).max(1.0);
        let per_axis = key.tiles_per_axis() as f64;
        let ground_span_m = (2.0 * std::f64::consts::PI * wgs84::A) / per_axis;
        let sse = compute_tile_sse(
            ground_span_m,
            tile_grid,
            dist_to_eye,
            viewport_height,
            fov_y_deg,
        );

        // 4. Subdivide if SSE > threshold and not at leaf limit
        if sse > sse_threshold && key.level < max_level {
            for q in 0..4 {
                if let Some(child) = key.child(q) {
                    stack.push(child);
                }
            }
        } else {
            on_leaf_tile(key, bounds);
        }
    }
}

// =========================================================================
// Double-Precision (f64) Double-Single / Float-Float RTE GPU Math
// =========================================================================

/// Splits a double-precision (f64) value into high and low single-precision (f32) floats.
///
/// This provides 48 bits of effective significand precision (24 bits high + 24 bits low),
/// preserving sub-millimeter geometric accuracy across the entire planetary scale
/// (~6.378e6 m) on standard GPU hardware without requiring native f64 shaders.
#[inline(always)]
pub fn split_f64_to_high_low(val: f64) -> (f32, f32) {
    let high = val as f32;
    let low = (val - high as f64) as f32;
    (high, low)
}

/// Splits a double-precision 3D vector (ECEF coordinate) into high and low `[f32; 3]` components.
#[inline(always)]
pub fn split_dvec3_to_high_low(val: DVec3) -> ([f32; 3], [f32; 3]) {
    val.split_high_low()
}

/// Computes Relative-To-Eye (RTE) relative position in f32 from f64 inputs directly.
#[inline(always)]
pub fn rte_relative_position_f32(pos: DVec3, eye: DVec3) -> [f32; 3] {
    let rel = pos.sub(eye);
    [rel.x as f32, rel.y as f32, rel.z as f32]
}

/// Evaluates the GPU vertex shader Relative-To-Eye (RTE) formula using split high and low floats:
/// `(pos_high - eye_high) + (pos_low - eye_low)`.
///
/// This yields sub-millimeter precision agreement with true f64 difference at planetary scale.
#[inline(always)]
pub fn rte_gpu_relative_position(
    pos_high: [f32; 3],
    pos_low: [f32; 3],
    eye_high: [f32; 3],
    eye_low: [f32; 3],
) -> [f32; 3] {
    [
        (pos_high[0] - eye_high[0]) + (pos_low[0] - eye_low[0]),
        (pos_high[1] - eye_high[1]) + (pos_low[1] - eye_low[1]),
        (pos_high[2] - eye_high[2]) + (pos_low[2] - eye_low[2]),
    ]
}

/// Uniforms for Relative-To-Eye (RTE) GPU rendering.
///
/// Encapsulates the split camera eye position and the translation-free view-projection matrix.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RteEyeUniforms {
    /// High 32 bits of camera eye in ECEF.
    pub eye_high: [f32; 3],
    pub _pad0: f32,
    /// Low 32 bits of camera eye in ECEF.
    pub eye_low: [f32; 3],
    pub _pad1: f32,
    /// Combined view-projection matrix with camera translation removed (column-major).
    pub view_proj_relative: [[f32; 4]; 4],
}

impl RteEyeUniforms {
    pub fn new(
        eye: DVec3,
        aim: DVec3,
        up: DVec3,
        fov_y_deg: f64,
        aspect: f64,
        near: f64,
        far: f64,
    ) -> Self {
        let (eye_high, eye_low) = eye.split_high_low();
        let view_rel = DMat4::view_rotation_only(eye, aim, up);
        let f = 1.0 / (fov_y_deg.to_radians() * 0.5).tan();
        let range_inv = 1.0 / (near - far);
        let proj = DMat4 {
            m: [
                f / aspect,
                0.0,
                0.0,
                0.0,
                0.0,
                f,
                0.0,
                0.0,
                0.0,
                0.0,
                far * range_inv,
                -1.0,
                0.0,
                0.0,
                near * far * range_inv,
                0.0,
            ],
        };
        let vp = proj.multiply(&view_rel);

        let mut vp_cols = [[0.0f32; 4]; 4];
        for (col, col_slice) in vp_cols.iter_mut().enumerate() {
            for (row, cell) in col_slice.iter_mut().enumerate() {
                *cell = vp.m[col * 4 + row] as f32;
            }
        }

        Self {
            eye_high,
            _pad0: 0.0,
            eye_low,
            _pad1: 0.0,
            view_proj_relative: vp_cols,
        }
    }

    pub fn from_camera(camera: &EarthCamera, viewport_width: f64, viewport_height: f64) -> Self {
        let placement = camera.placement();
        let aspect = (viewport_width / viewport_height.max(1.0)).max(0.01);
        Self::new(
            placement.eye,
            placement.aim,
            placement.up,
            placement.fov_y_deg,
            aspect,
            placement.near_m,
            placement.far_m,
        )
    }

    /// Transforms a split RTE vertex position to clip-space coordinates `[x, y, z, w]`.
    pub fn transform_point_clip(&self, pos_high: [f32; 3], pos_low: [f32; 3]) -> [f32; 4] {
        let rel = rte_gpu_relative_position(pos_high, pos_low, self.eye_high, self.eye_low);
        let m = &self.view_proj_relative;
        [
            m[0][0] * rel[0] + m[1][0] * rel[1] + m[2][0] * rel[2] + m[3][0],
            m[0][1] * rel[0] + m[1][1] * rel[1] + m[2][1] * rel[2] + m[3][1],
            m[0][2] * rel[0] + m[1][2] * rel[1] + m[2][2] * rel[2] + m[3][2],
            m[0][3] * rel[0] + m[1][3] * rel[1] + m[2][3] * rel[2] + m[3][3],
        ]
    }
}

/// Six normalized view-frustum clip planes in RTE `(point - eye)` coordinates (`f64`).
///
/// Each plane satisfies `a * rel.x + b * rel.y + c * rel.z + d >= 0` inside the frustum.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Frustum6 {
    pub planes: [[f64; 4]; 6],
    pub eye: DVec3,
}

impl Frustum6 {
    /// Extracts the 6 exact clip planes from `camera` and viewport size in `f64`.
    pub fn from_camera(camera: &EarthCamera, viewport_w: f64, viewport_h: f64) -> Self {
        let placement = camera.placement();
        let aspect = (viewport_w.max(1.0) / viewport_h.max(1.0)).max(0.01);
        Self::new(
            placement.eye,
            placement.aim,
            placement.up,
            placement.fov_y_deg,
            aspect,
            placement.near_m,
            placement.far_m,
        )
    }

    pub fn new(
        eye: DVec3,
        aim: DVec3,
        up: DVec3,
        fov_y_deg: f64,
        aspect: f64,
        near: f64,
        far: f64,
    ) -> Self {
        let view_rel = DMat4::view_rotation_only(eye, aim, up);
        let f = 1.0 / (fov_y_deg.to_radians() * 0.5).tan();
        let range_inv = 1.0 / (near - far);
        let proj = DMat4 {
            m: [
                f / aspect,
                0.0,
                0.0,
                0.0,
                0.0,
                f,
                0.0,
                0.0,
                0.0,
                0.0,
                far * range_inv,
                -1.0,
                0.0,
                0.0,
                near * far * range_inv,
                0.0,
            ],
        };
        let vp = proj.multiply(&view_rel);
        let m = &vp.m;
        let row = |r: usize| [m[r], m[4 + r], m[8 + r], m[12 + r]];
        let r0 = row(0);
        let r1 = row(1);
        let r2 = row(2);
        let r3 = row(3);
        let norm = |p: [f64; 4]| -> [f64; 4] {
            let len = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt().max(1e-12);
            [p[0] / len, p[1] / len, p[2] / len, p[3] / len]
        };
        let add = |a: [f64; 4], b: [f64; 4]| [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]];
        let sub = |a: [f64; 4], b: [f64; 4]| [a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]];
        Self {
            planes: [
                norm(add(r3, r0)), // Left
                norm(sub(r3, r0)), // Right
                norm(add(r3, r1)), // Bottom
                norm(sub(r3, r1)), // Top
                norm(r2),          // Near
                norm(sub(r3, r2)), // Far
            ],
            eye,
        }
    }

    #[inline]
    pub fn intersects_sphere(&self, center: DVec3, radius: f64) -> bool {
        let rx = center.x - self.eye.x;
        let ry = center.y - self.eye.y;
        let rz = center.z - self.eye.z;
        for p in &self.planes {
            if p[0] * rx + p[1] * ry + p[2] * rz + p[3] < -radius {
                return false;
            }
        }
        true
    }
}

/// Traverses the planetary quadtree using exact 6-plane frustum culling, horizon occlusion,
/// and SSE threshold subdivision.
pub fn traverse_quadtree_lod_frustum6(
    camera: &EarthCamera,
    viewport_width_px: f64,
    viewport_height_px: f64,
    sse_threshold_px: f64,
    max_level: u8,
    tile_grid: u32,
) -> Vec<TileKey> {
    let placement = camera.placement();
    let frustum = Frustum6::from_camera(camera, viewport_width_px, viewport_height_px);
    let mut leaves = Vec::new();
    let mut stack = vec![TileKey::new(0, 0, 0)];
    let threshold = sse_threshold_px.max(0.5);

    while let Some(key) = stack.pop() {
        let (center, radius) = tile_bounding_sphere(key);
        if is_tile_horizon_culled(center, radius, placement.eye) {
            continue;
        }
        if !frustum.intersects_sphere(center, radius * 1.15) {
            continue;
        }
        if key.level < max_level {
            let dist_to_eye = (center.sub(placement.eye).length() - radius).max(1.0);
            let per_axis = key.tiles_per_axis() as f64;
            let ground_span_m = (2.0 * std::f64::consts::PI * wgs84::A) / per_axis;
            let sse = compute_tile_sse(
                ground_span_m,
                tile_grid,
                dist_to_eye,
                viewport_height_px,
                placement.fov_y_deg,
            );
            if sse > threshold {
                for q in 0..4 {
                    if let Some(child) = key.child(q) {
                        stack.push(child);
                    }
                }
                continue;
            }
        }
        leaves.push(key);
    }

    leaves.sort_by(|a, b| {
        a.level
            .cmp(&b.level)
            .then_with(|| a.x.cmp(&b.x))
            .then_with(|| a.y.cmp(&b.y))
    });
    leaves
}

/// Builds a [`strata_render::scene3d::RetainedCameraUniforms`] packet from an [`EarthCamera`].
pub fn retained_camera_uniforms(
    camera: &EarthCamera,
    viewport_w: f64,
    viewport_h: f64,
    sun_dir: [f32; 3],
    sun_color: [f32; 3],
    ambient_color: [f32; 3],
    background: strata_render::scene3d::Background3D,
) -> strata_render::scene3d::RetainedCameraUniforms {
    let placement = camera.placement();
    let aspect = (viewport_w.max(1.0) / viewport_h.max(1.0)).max(0.01);
    let view_rel = DMat4::view_rotation_only(placement.eye, placement.aim, placement.up);
    let f = 1.0 / (placement.fov_y_deg.to_radians() * 0.5).tan();
    let range_inv = 1.0 / (placement.near_m - placement.far_m);
    let proj_d = DMat4 {
        m: [
            f / aspect,
            0.0,
            0.0,
            0.0,
            0.0,
            f,
            0.0,
            0.0,
            0.0,
            0.0,
            placement.far_m * range_inv,
            -1.0,
            0.0,
            0.0,
            placement.near_m * placement.far_m * range_inv,
            0.0,
        ],
    };
    let vp_d = proj_d.multiply(&view_rel);
    let to_cols = |d: &DMat4| -> [[f32; 4]; 4] {
        let mut out = [[0.0f32; 4]; 4];
        for (c, col) in out.iter_mut().enumerate() {
            for (r, cell) in col.iter_mut().enumerate() {
                *cell = d.m[c * 4 + r] as f32;
            }
        }
        out
    };
    strata_render::scene3d::RetainedCameraUniforms {
        proj: to_cols(&proj_d),
        view_rot: to_cols(&view_rel),
        view_proj_rte: to_cols(&vp_d),
        eye_ecef: [placement.eye.x, placement.eye.y, placement.eye.z],
        sun_dir,
        sun_color,
        ambient_color,
        background,
        near: placement.near_m as f32,
        far: placement.far_m as f32,
    }
}

/// Builds a tile-local RTC [`strata_render::scene3d::RetainedTileMesh`] relative to the tile's
/// `center_ecef` once per `(TileKey, dem_revision)`.
#[allow(clippy::too_many_arguments)]
pub fn build_tile_local_rtc_mesh(
    key: TileKey,
    bounds: PatchBounds,
    grid: u32,
    elevation: Option<&crate::tiles::TileElevation>,
    elevation_tile: TileKey,
    generate_skirts: bool,
    skirt_depth_m: f64,
    revision: u64,
    face: u8,
) -> strata_render::scene3d::RetainedTileMesh {
    let n = grid.max(2) as usize;
    let mid_lat = (bounds.lat0 + bounds.lat1) * 0.5;
    let mid_lon = (bounds.lon0 + bounds.lon1) * 0.5;
    let center_dvec = geodetic_to_ecef(Geodetic::new(mid_lat.clamp(-89.9, 89.9), mid_lon, 0.0));

    let mut positions = Vec::with_capacity(n * n);
    let mut radials = Vec::with_capacity(n * n);
    let mut uvs = Vec::with_capacity(n * n);

    for row in 0..n {
        let v = row as f64 / (n - 1) as f64;
        for col in 0..n {
            let u = col as f64 / (n - 1) as f64;
            let lon = bounds.lon0 + (bounds.lon1 - bounds.lon0) * u;
            let lat = (bounds.lat0 + (bounds.lat1 - bounds.lat0) * v).clamp(-89.9, 89.9);
            let height = elevation
                .map(|grid| {
                    let (tu, tv) = crate::scene::uv_in_tile(elevation_tile, lat, lon);
                    grid.sample_bilinear(tu, tv) as f64
                })
                .unwrap_or(0.0);
            let p = geodetic_to_ecef(Geodetic::new(lat, lon, height));
            positions.push(p);
            radials.push(p.normalize());
            uvs.push([u as f32, v as f32]);
        }
    }

    let mut normals = Vec::with_capacity(n * n);
    for row in 0..n {
        for col in 0..n {
            let idx = row * n + col;
            let radial = radials[idx];
            let east = positions[row * n + (col + 1).min(n - 1)]
                .sub(positions[row * n + col.saturating_sub(1)]);
            let north = positions[(row + 1).min(n - 1) * n + col]
                .sub(positions[row.saturating_sub(1) * n + col]);
            let cross = east.cross(north);
            let normal = if cross.length() < 1e-9 {
                radial
            } else {
                let nrm = cross.normalize();
                if nrm.dot(radial) < 0.0 {
                    nrm.scale(-1.0)
                } else {
                    nrm
                }
            };
            normals.push(normal.as_f32());
        }
    }

    let skirt_verts = if generate_skirts { 4 * n } else { 0 };
    let mut vertices = Vec::with_capacity(n * n + skirt_verts);
    let mut max_r2 = 0.0f64;

    for i in 0..(n * n) {
        let rel = positions[i].sub(center_dvec);
        max_r2 = max_r2.max(rel.x * rel.x + rel.y * rel.y + rel.z * rel.z);
        vertices.push(strata_render::scene3d::RteTileVertex::new(
            rel.as_f32(),
            normals[i],
            uvs[i],
        ));
    }

    if generate_skirts {
        let push_skirt_pt = |vertices: &mut Vec<strata_render::scene3d::RteTileVertex>,
                             idx: usize| {
            let dropped = positions[idx].sub(radials[idx].scale(skirt_depth_m));
            let rel = dropped.sub(center_dvec);
            vertices.push(strata_render::scene3d::RteTileVertex::new(
                rel.as_f32(),
                normals[idx],
                uvs[idx],
            ));
        };
        // South (row = 0, col = 0..n)
        for col in 0..n {
            push_skirt_pt(&mut vertices, col);
        }
        // East (col = n-1, row = 0..n)
        for row in 0..n {
            push_skirt_pt(&mut vertices, row * n + (n - 1));
        }
        // North (row = n-1, col = 0..n)
        for col in 0..n {
            push_skirt_pt(&mut vertices, (n - 1) * n + col);
        }
        // West (col = 0, row = 0..n)
        for row in 0..n {
            push_skirt_pt(&mut vertices, row * n);
        }
    }

    strata_render::scene3d::RetainedTileMesh {
        key: strata_render::scene3d::TileKeyId::new(key.level, key.x, key.y, face),
        revision,
        center_ecef: [center_dvec.x, center_dvec.y, center_dvec.z],
        grid_res: n as u32,
        has_skirts: generate_skirts,
        vertices: std::sync::Arc::new(vertices),
        bounding_radius: max_r2.sqrt() as f32,
    }
}
