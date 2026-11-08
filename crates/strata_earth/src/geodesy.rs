//! Geodesy, camera control, and tile quadtree for the Earth scene plugin.
//!
//! Coordinates are WGS84-ish ECEF metres in f64. Nothing here narrows to f32;
//! the render hand-off is [`crate::EarthScene::build`].

use strata_render::camera64::{DVec3, EARTH_RADIUS_M};

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

    let p0 = DVec3::new(ray_origin.x * inv_a, ray_origin.y * inv_a, ray_origin.z * inv_b);
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
    let (ray_origin, ray_dir) = screen_to_ray(
        screen_x,
        screen_y,
        viewport_width,
        viewport_height,
        camera,
    );
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
