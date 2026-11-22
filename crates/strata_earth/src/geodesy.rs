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
        self.placement_with_up(None)
    }

    /// Resolves ECEF eye, aim, depth range, and optional custom up vector for arcball orientation.
    pub fn placement_with_up(&self, up_override: Option<DVec3>) -> CameraPlacement {
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
        let up = match up_override {
            Some(u) if u.length() > 1e-6 => u.normalize(),
            _ => frame.up,
        };
        let altitude = self.altitude_m();
        let horizon = horizon_distance_m(altitude);
        let clearance = altitude.min(self.eye_distance_m().max(2.0));
        let (near, _) = depth_range(clearance, horizon, self.near_m);
        let (_, far) = depth_range(altitude, horizon, self.near_m);
        CameraPlacement {
            eye,
            aim,
            up,
            fov_y_deg: self.fov_y_deg,
            near_m: near,
            far_m: far,
        }
    }

    /// Resolves camera placement matching `morph` across 3D curved globe, 2.5D local tangent
    /// plane, and steady-state canonical 2D flat map (`map_2d_morph >= 0.999`).
    ///
    /// In canonical 2D mode, the camera is positioned strictly nadir over the fixed equatorial
    /// plane (`LocalFrame::at(morph.focus.lat_deg, morph.focus.lon_deg)` with `morph.focus = (0, 0)`),
    /// keeping `view_rot` constant across arbitrary fast panning so the flat map never tilts or
    /// rotates from spherical meridian convergence.
    pub fn placement_for_morph(
        &self,
        morph: SurfaceMorphParams,
        up_override: Option<DVec3>,
    ) -> CameraPlacement {
        if is_canonical_2d_morph(morph.map_2d_morph)
            && morph.flat_blend >= CANONICAL_2D_MORPH_THRESHOLD
        {
            let focus_surface = Geodetic::new(morph.focus.lat_deg, morph.focus.lon_deg, 0.0);
            let frame = LocalFrame::at(focus_surface.lat_deg, focus_surface.lon_deg);
            let origin = geodetic_to_ecef(focus_surface);
            let focus_lat = self.focus.lat_deg.clamp(-85.051129, 85.051129);
            let focus_lon = wrap_longitude(self.focus.lon_deg);
            let dlon_deg = wrap_longitude(focus_lon - morph.focus.lon_deg);
            let dlat_deg = focus_lat - morph.focus.lat_deg;
            let east_m = dlon_deg.to_radians() * wgs84::A;
            let north_m = dlat_deg.to_radians() * wgs84::A;
            let dist = self.eye_distance_m().max(self.altitude_m()).max(1.0);
            let aim = origin
                .add(frame.east.scale(east_m))
                .add(frame.north.scale(north_m));
            let eye = aim.add(frame.up.scale(dist));
            let horizon = horizon_distance_m(dist);
            let (near, far) = depth_range(dist, horizon, self.near_m);
            return CameraPlacement {
                eye,
                aim,
                up: frame.north,
                fov_y_deg: self.fov_y_deg,
                near_m: near,
                far_m: far.max(dist * 4.0 + 100_000.0),
            };
        }
        self.placement_with_up(up_override)
    }
}

fn screen_to_ray_from_placement(
    screen_x: f64,
    screen_y: f64,
    viewport_width: f64,
    viewport_height: f64,
    placement: CameraPlacement,
    fallback_lat_deg: f64,
    fallback_lon_deg: f64,
) -> (DVec3, DVec3) {
    let eye = placement.eye;
    let aim = placement.aim;

    let forward = aim.sub(eye).normalize();
    let cam_up = if forward.cross(placement.up).length() < 1e-6 {
        let frame = LocalFrame::at(fallback_lat_deg, fallback_lon_deg);
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
    screen_to_ray_with_up(
        screen_x,
        screen_y,
        viewport_width,
        viewport_height,
        camera,
        None,
    )
}

/// Same as [`screen_to_ray`] with an optional explicit ECEF camera-up hint.
pub fn screen_to_ray_with_up(
    screen_x: f64,
    screen_y: f64,
    viewport_width: f64,
    viewport_height: f64,
    camera: &EarthCamera,
    up_override: Option<DVec3>,
) -> (DVec3, DVec3) {
    let placement = camera.placement_with_up(up_override);
    screen_to_ray_from_placement(
        screen_x,
        screen_y,
        viewport_width,
        viewport_height,
        placement,
        camera.focus.lat_deg,
        camera.focus.lon_deg,
    )
}

/// Same as [`screen_to_ray_with_up`], honoring the active [`SurfaceMorphParams`] coordinate frame
/// (including the fixed canonical 2D map plane when `map_2d_morph >= 0.999`).
pub fn screen_to_ray_for_morph(
    screen_x: f64,
    screen_y: f64,
    viewport_width: f64,
    viewport_height: f64,
    camera: &EarthCamera,
    morph: SurfaceMorphParams,
    up_override: Option<DVec3>,
) -> (DVec3, DVec3) {
    let placement = camera.placement_for_morph(morph, up_override);
    screen_to_ray_from_placement(
        screen_x,
        screen_y,
        viewport_width,
        viewport_height,
        placement,
        camera.focus.lat_deg,
        camera.focus.lon_deg,
    )
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
    pick_geodetic_with_up(
        screen_x,
        screen_y,
        viewport_width,
        viewport_height,
        camera,
        None,
    )
}

/// Same as [`pick_geodetic`] with an optional explicit ECEF camera-up hint.
pub fn pick_geodetic_with_up(
    screen_x: f64,
    screen_y: f64,
    viewport_width: f64,
    viewport_height: f64,
    camera: &EarthCamera,
    up_override: Option<DVec3>,
) -> Option<Geodetic> {
    let (ray_origin, ray_dir) = screen_to_ray_with_up(
        screen_x,
        screen_y,
        viewport_width,
        viewport_height,
        camera,
        up_override,
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
/// Tile latitude projection scheme for quadtree tiles.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TileProjection {
    /// Standard Slippy/XYZ Web Mercator (`EPSG:3857`) used by OSM, ArcGIS, Mapbox, NASA GIBS 3857, and Terrarium DEM.
    #[default]
    WebMercator,
    /// Equirectangular geographic degrees (`EPSG:4326`).
    Geographic4326,
}

/// Converts normalized global Web Mercator Y (`0.0` north .. `1.0` south) into latitude in degrees.
///
/// Polar world edges (`gy <= 0` or `gy >= 1`) close smoothly to `±89.9°` so the WGS84 globe has
/// no open polar holes while all interior rows follow exact inverse Web Mercator latitude.
#[inline]
pub fn mercator_gy_to_lat_deg(gy: f64) -> f64 {
    if gy <= 0.0 {
        return 89.9;
    }
    if gy >= 1.0 {
        return -89.9;
    }
    (std::f64::consts::PI * (1.0 - 2.0 * gy))
        .sinh()
        .atan()
        .to_degrees()
        .clamp(-89.9, 89.9)
}

/// Returns the geographic bounds of `key` under Plate Carree (`EPSG:4326`) subdivision.
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

/// Returns the geographic bounds of `key` under standard Web Mercator (`EPSG:3857`) subdivision.
pub fn tile_bounds_mercator(key: TileKey) -> PatchBounds {
    let per_axis = key.tiles_per_axis() as f64;
    let lon_span = 360.0 / per_axis;
    let lon0 = key.x as f64 * lon_span - 180.0;
    let gy0 = (key.y as f64) / per_axis;
    let gy1 = ((key.y + 1) as f64) / per_axis;
    PatchBounds {
        lon0,
        lon1: lon0 + lon_span,
        lat0: mercator_gy_to_lat_deg(gy1),
        lat1: mercator_gy_to_lat_deg(gy0),
    }
}

/// Computes the seamless 3D-globe -> 2.5D-ground-plane blend (`0.0..=1.0`) from camera altitude.
///
/// Above 250 km: pure curved WGS84 globe (`0.0`).
/// Between 250 km and 20 km: smooth C1 Hermite ramp.
/// Below 20 km: local 2.5D tangent plane (`1.0`).
#[inline]
pub fn compute_seamless_25d_blend(altitude_m: f64) -> f32 {
    const START_M: f64 = 250_000.0;
    const END_M: f64 = 20_000.0;
    let u = ((START_M - altitude_m) / (START_M - END_M)).clamp(0.0, 1.0);
    (u * u * (3.0 - 2.0 * u)) as f32
}

/// Altitude reference frame for geodetic objects in [`crate::world::EarthWorld`].
///
/// Stored live on each [`GeoPosition`] (never eagerly baked into absolute altitude) so toggling
/// terrain elevation dynamically transitions [`AltitudeMode::Terrain`] objects between ground-relative
/// and sea-level-relative height without mutating user positions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum AltitudeMode {
    /// Metres relative to the WGS84 sea-level surface (`0.0 m`). Independent of heightmap state.
    SeaLevel,
    /// Metres relative to active ground (`sampled_dem + altitude_m` when heightmap is enabled;
    /// `0.0 + altitude_m` when heightmap is disabled).
    #[default]
    Terrain,
}

/// Native geodetic 3D coordinate in `strata_earth` (`f64` degrees and metres).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GeoPosition {
    pub lat_deg: f64,
    pub lon_deg: f64,
    pub altitude_m: f64,
    pub altitude_mode: AltitudeMode,
}

impl Default for GeoPosition {
    fn default() -> Self {
        Self {
            lat_deg: 0.0,
            lon_deg: 0.0,
            altitude_m: 0.0,
            altitude_mode: AltitudeMode::Terrain,
        }
    }
}

impl GeoPosition {
    /// Creates a geodetic coordinate with an explicit [`AltitudeMode`].
    #[inline]
    pub const fn new(
        lat_deg: f64,
        lon_deg: f64,
        altitude_m: f64,
        altitude_mode: AltitudeMode,
    ) -> Self {
        Self {
            lat_deg,
            lon_deg,
            altitude_m,
            altitude_mode,
        }
    }

    /// Creates a terrain-relative coordinate (`altitude_m` above ground when heightmap is on,
    /// or `altitude_m` above sea level when heightmap is off).
    #[inline]
    pub const fn terrain(lat_deg: f64, lon_deg: f64, altitude_m: f64) -> Self {
        Self::new(lat_deg, lon_deg, altitude_m, AltitudeMode::Terrain)
    }

    /// Creates a sea-level-relative coordinate (`altitude_m` relative to `0.0 m` WGS84 ellipsoid).
    #[inline]
    pub const fn sea_level(lat_deg: f64, lon_deg: f64, altitude_m: f64) -> Self {
        Self::new(lat_deg, lon_deg, altitude_m, AltitudeMode::SeaLevel)
    }

    /// Evaluates the live world height above the WGS84 ellipsoid given heightmap state and DEM sample.
    #[inline]
    pub fn resolve_height_m(&self, elevation_enabled: bool, sampled_terrain_m: f64) -> f64 {
        let base_m = match self.altitude_mode {
            AltitudeMode::SeaLevel => 0.0,
            AltitudeMode::Terrain => {
                if elevation_enabled {
                    sampled_terrain_m
                } else {
                    0.0
                }
            }
        };
        base_m + self.altitude_m
    }
}

/// Parameters controlling continuous vertex morphing across 3D globe, local 2.5D ground plane,
/// and flattened 2D map projection.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SurfaceMorphParams {
    pub focus: Geodetic,
    /// `0.0` = curved WGS84 ellipsoid, `1.0` = local/map flat plane.
    pub flat_blend: f64,
    /// `0.0` = preserve 3D relief & local cosine metric (2.5D), `1.0` = flat 2D map plane.
    pub map_2d_morph: f64,
    pub projection: TileProjection,
}

impl Default for SurfaceMorphParams {
    fn default() -> Self {
        Self {
            focus: Geodetic::new(0.0, 0.0, 0.0),
            flat_blend: 0.0,
            map_2d_morph: 0.0,
            projection: TileProjection::WebMercator,
        }
    }
}

/// Threshold on [`SurfaceMorphParams::map_2d_morph`] at or above which the flat 2D map uses
/// the fixed canonical equatorial plane (`(0°, 0°)`), preventing tangent-frame rotation lag
/// and keeping 2D tile meshes 100% cache-stable across arbitrary fast panning.
pub const CANONICAL_2D_MORPH_THRESHOLD: f64 = 0.999;

/// Returns `true` when `map_2d_morph` represents steady-state flat 2D map mode.
#[inline]
pub fn is_canonical_2d_morph(map_2d_morph: f64) -> bool {
    map_2d_morph >= CANONICAL_2D_MORPH_THRESHOLD
}

/// Computes the canonical quantized [`SurfaceMorphParams`] shared identically across terrain tile
/// mesh generation, [`crate::world::EarthWorld`] 3D object placement, and world raycasting.
#[inline]
pub fn stabilized_surface_morph(
    focus: Geodetic,
    flat_blend: f64,
    map_2d_morph: f64,
    projection: TileProjection,
) -> SurfaceMorphParams {
    let anchor = if is_canonical_2d_morph(map_2d_morph)
        && flat_blend >= CANONICAL_2D_MORPH_THRESHOLD
    {
        Geodetic::new(0.0, 0.0, 0.0)
    } else if flat_blend >= 0.99 && map_2d_morph <= 1e-4 {
        Geodetic::new(
            (focus.lat_deg * 50.0).round() / 50.0,
            (focus.lon_deg * 50.0).round() / 50.0,
            0.0,
        )
    } else if flat_blend > 1e-6 {
        Geodetic::new(
            (focus.lat_deg * 200.0).round() / 200.0,
            (focus.lon_deg * 200.0).round() / 200.0,
            0.0,
        )
    } else {
        focus
    };
    SurfaceMorphParams {
        focus: anchor,
        flat_blend,
        map_2d_morph,
        projection,
    }
}

/// Projects `(lat_deg, lon_deg, height_m)` into world ECEF/morphed coordinates and computes the
/// matching orthonormal local East-North-Up basis under `morph`.
///
/// Matches [`build_tile_local_rtc_mesh_morphed`] identically across 3D globe, 2.5D tangent plane,
/// and 2D flat map views so 3D objects sit flush on terrain with zero drift.
#[inline]
pub fn project_morphed_point_and_frame(
    lat_deg: f64,
    lon_deg: f64,
    height_m: f64,
    morph: SurfaceMorphParams,
) -> (DVec3, LocalFrame) {
    let flat_t = morph.flat_blend.clamp(0.0, 1.0);
    let map_2d = morph.map_2d_morph.clamp(0.0, 1.0);
    let curved_frame = LocalFrame::at(lat_deg, lon_deg);
    let p_curved = geodetic_to_ecef(Geodetic::new(lat_deg, lon_deg, height_m));
    if flat_t <= 1e-6 {
        return (p_curved, curved_frame);
    }

    let focus_surface = Geodetic::new(morph.focus.lat_deg, morph.focus.lon_deg, 0.0);
    let focus_frame = LocalFrame::at(focus_surface.lat_deg, focus_surface.lon_deg);
    let focus_ecef = geodetic_to_ecef(focus_surface);
    let cos_focus = focus_surface.lat_deg.to_radians().cos().abs().max(0.25);
    let cos_scale = cos_focus + (1.0 - cos_focus) * map_2d;

    let dlon_deg = wrap_longitude(lon_deg - morph.focus.lon_deg);
    let dlat_deg = lat_deg - morph.focus.lat_deg;
    let east_m = dlon_deg.to_radians() * wgs84::A * cos_scale;
    let north_m = dlat_deg.to_radians() * wgs84::A;
    let up_h = height_m * (1.0 - map_2d);
    let p_flat = focus_ecef
        .add(focus_frame.east.scale(east_m))
        .add(focus_frame.north.scale(north_m))
        .add(focus_frame.up.scale(up_h));

    let p_morphed = DVec3::new(
        p_curved.x + (p_flat.x - p_curved.x) * flat_t,
        p_curved.y + (p_flat.y - p_curved.y) * flat_t,
        p_curved.z + (p_flat.z - p_curved.z) * flat_t,
    );

    let up = DVec3::new(
        curved_frame.up.x + (focus_frame.up.x - curved_frame.up.x) * flat_t,
        curved_frame.up.y + (focus_frame.up.y - curved_frame.up.y) * flat_t,
        curved_frame.up.z + (focus_frame.up.z - curved_frame.up.z) * flat_t,
    )
    .normalize();
    let east_raw = DVec3::new(
        curved_frame.east.x + (focus_frame.east.x - curved_frame.east.x) * flat_t,
        curved_frame.east.y + (focus_frame.east.y - curved_frame.east.y) * flat_t,
        curved_frame.east.z + (focus_frame.east.z - curved_frame.east.z) * flat_t,
    );
    let east_ortho = east_raw.sub(up.scale(east_raw.dot(up)));
    let east = if east_ortho.length() > 1e-8 {
        east_ortho.normalize()
    } else {
        focus_frame.east
    };
    let north = up.cross(east).normalize();

    (p_morphed, LocalFrame { east, north, up })
}

/// Inverts [`project_morphed_point_and_frame`] for a world point at known `height_m`, returning
/// `(lat_deg, lon_deg)`. Exact in pure 3D (`flat_blend == 0`) and pure flat (`flat_blend == 1`),
/// and sub-millimetre across intermediate morph transitions.
pub fn unproject_morphed_surface_lat_lon(
    point_ecef: DVec3,
    height_m: f64,
    morph: SurfaceMorphParams,
) -> (f64, f64) {
    let flat_t = morph.flat_blend.clamp(0.0, 1.0);
    if flat_t <= 1e-6 {
        let g = ecef_to_geodetic(point_ecef);
        return (g.lat_deg, wrap_longitude(g.lon_deg));
    }

    let map_2d = morph.map_2d_morph.clamp(0.0, 1.0);
    let focus_surface = Geodetic::new(morph.focus.lat_deg, morph.focus.lon_deg, 0.0);
    let focus_frame = LocalFrame::at(focus_surface.lat_deg, focus_surface.lon_deg);
    let focus_ecef = geodetic_to_ecef(focus_surface);
    let cos_focus = focus_surface.lat_deg.to_radians().cos().abs().max(0.25);
    let cos_scale = cos_focus + (1.0 - cos_focus) * map_2d;

    if flat_t >= 1.0 - 1e-6 {
        let rel = point_ecef.sub(focus_ecef);
        let east_m = rel.dot(focus_frame.east);
        let north_m = rel.dot(focus_frame.north);
        let lat_deg = morph.focus.lat_deg + (north_m / wgs84::A).to_degrees();
        let lon_deg = wrap_longitude(
            morph.focus.lon_deg + (east_m / (wgs84::A * cos_scale.max(1e-6))).to_degrees(),
        );
        return (lat_deg, lon_deg);
    }

    let init = ecef_to_geodetic(point_ecef);
    let mut lat = init.lat_deg.clamp(-89.95, 89.95);
    let mut lon = wrap_longitude(init.lon_deg);
    for _ in 0..4 {
        let (p_est, frame) = project_morphed_point_and_frame(lat, lon, height_m, morph);
        let err = point_ecef.sub(p_est);
        let de = err.dot(frame.east);
        let dn = err.dot(frame.north);
        if de * de + dn * dn < 1e-8 {
            break;
        }
        let cos_lat = lat.to_radians().cos().abs().max(0.05);
        let m_east = wgs84::A * ((1.0 - flat_t) * cos_lat + flat_t * cos_scale);
        lat = (lat + (dn / wgs84::A).to_degrees()).clamp(-89.95, 89.95);
        lon = wrap_longitude(lon + (de / m_east.max(1.0)).to_degrees());
    }
    (lat, lon)
}

/// Computes the 3D bounding sphere center (ECEF `DVec3`) and radius (metres)
/// for a `TileKey` on the WGS84 ellipsoid at `avg_altitude_m` (use `0.0` for sea level
/// when heightmaps are disabled) using the requested tile projection.
pub fn tile_bounding_sphere_projected_with_altitude(
    key: TileKey,
    projection: TileProjection,
    avg_altitude_m: f64,
) -> (DVec3, f64) {
    if key.level == 0 {
        return (DVec3::new(0.0, 0.0, 0.0), wgs84::A * 1.08);
    }
    let bounds = match projection {
        TileProjection::WebMercator => tile_bounds_mercator(key),
        TileProjection::Geographic4326 => tile_bounds(key),
    };
    let h = if avg_altitude_m.is_finite() {
        avg_altitude_m.clamp(-12_000.0, 12_000.0)
    } else {
        0.0
    };
    let mid_lon = (bounds.lon0 + bounds.lon1) * 0.5;
    let mid_lat = (bounds.lat0 + bounds.lat1) * 0.5;
    let center = geodetic_to_ecef(Geodetic::new(mid_lat, mid_lon, h));

    let sample_points = [
        Geodetic::new(bounds.lat0, bounds.lon0, h),
        Geodetic::new(bounds.lat0, bounds.lon1, h),
        Geodetic::new(bounds.lat1, bounds.lon0, h),
        Geodetic::new(bounds.lat1, bounds.lon1, h),
        Geodetic::new(bounds.lat0, mid_lon, h),
        Geodetic::new(bounds.lat1, mid_lon, h),
        Geodetic::new(mid_lat, bounds.lon0, h),
        Geodetic::new(mid_lat, bounds.lon1, h),
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

/// Computes the 3D bounding sphere center (ECEF `DVec3`) and radius (metres)
/// for a `TileKey` on the WGS84 ellipsoid at sea level (`0.0 m`) using the requested tile projection.
pub fn tile_bounding_sphere_projected(key: TileKey, projection: TileProjection) -> (DVec3, f64) {
    tile_bounding_sphere_projected_with_altitude(key, projection, 0.0)
}

/// Computes the 3D bounding sphere center (ECEF `DVec3`) and radius (metres)
/// for a Geographic4326 `TileKey` on the WGS84 ellipsoid.
pub fn tile_bounding_sphere(key: TileKey) -> (DVec3, f64) {
    tile_bounding_sphere_projected(key, TileProjection::Geographic4326)
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
        Self::from_camera_with_up(camera, None, viewport_w, viewport_h)
    }

    /// Extracts the 6 exact clip planes from `camera` with an optional custom up vector.
    pub fn from_camera_with_up(
        camera: &EarthCamera,
        up_override: Option<DVec3>,
        viewport_w: f64,
        viewport_h: f64,
    ) -> Self {
        let placement = camera.placement_with_up(up_override);
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

/// Default hard ceiling on visible quadtree leaves emitted per frame so GPU texture atlas capacity is never exceeded.
pub const MAX_QUADTREE_LEAVES: usize = 256;

#[derive(Clone, Copy)]
struct LodCandidate {
    sse: f64,
    key: TileKey,
}

impl PartialEq for LodCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key
    }
}

impl Eq for LodCandidate {}

impl Ord for LodCandidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.sse
            .total_cmp(&other.sse)
            .then_with(|| other.key.level.cmp(&self.key.level))
    }
}

impl PartialOrd for LodCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Traverses the planetary quadtree using exact 6-plane frustum culling, horizon occlusion,
/// and SSE-prioritized subdivision using `projection` bounds (`WebMercator` or `Geographic4326`).
#[allow(clippy::too_many_arguments)]
pub fn traverse_quadtree_lod_frustum6_projected(
    camera: &EarthCamera,
    viewport_width_px: f64,
    viewport_height_px: f64,
    sse_threshold_px: f64,
    max_level: u8,
    tile_grid: u32,
    horizon_cull: bool,
    max_leaves: usize,
    projection: TileProjection,
) -> Vec<TileKey> {
    traverse_quadtree_lod_frustum6_projected_with_up(
        camera,
        None,
        viewport_width_px,
        viewport_height_px,
        sse_threshold_px,
        max_level,
        tile_grid,
        horizon_cull,
        max_leaves,
        projection,
    )
}

/// Runtime LOD refinement parameters applied while the camera is idle and
/// [`crate::scene::IdleHighResConfig`] is enabled.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IdleLodParams {
    /// Effective SSE threshold scale (`1.0` = normal moving LOD; `< 1.0` = finer detail).
    pub sse_scale: f64,
    /// Gaze-center priority weight in `[0.0..=1.0]` favoring tiles directly in front of the camera.
    pub view_center_bias: f64,
    /// Additional leaf tiles allowed beyond `max_leaves` while idle.
    pub extra_leaves: usize,
    /// Maximum extra zoom levels permitted above normal moving LOD (`0..=4`).
    pub max_extra_levels: u8,
}

impl Default for IdleLodParams {
    fn default() -> Self {
        Self::inactive()
    }
}

impl IdleLodParams {
    /// Disabled / moving-camera LOD parameters (zero overhead).
    #[inline]
    pub const fn inactive() -> Self {
        Self {
            sse_scale: 1.0,
            view_center_bias: 0.0,
            extra_leaves: 0,
            max_extra_levels: 0,
        }
    }

    #[inline]
    pub fn is_active(&self) -> bool {
        self.sse_scale < 0.999 && self.max_extra_levels > 0
    }
}

/// Same as [`traverse_quadtree_lod_frustum6_projected`] with an optional explicit camera up hint.
#[allow(clippy::too_many_arguments)]
pub fn traverse_quadtree_lod_frustum6_projected_with_up(
    camera: &EarthCamera,
    up_override: Option<DVec3>,
    viewport_width_px: f64,
    viewport_height_px: f64,
    sse_threshold_px: f64,
    max_level: u8,
    tile_grid: u32,
    horizon_cull: bool,
    max_leaves: usize,
    projection: TileProjection,
) -> Vec<TileKey> {
    traverse_quadtree_lod_frustum6_with_elevation(
        camera,
        up_override,
        viewport_width_px,
        viewport_height_px,
        sse_threshold_px,
        max_level,
        tile_grid,
        horizon_cull,
        max_leaves,
        projection,
        IdleLodParams::inactive(),
        0.0,
        |_| 0.0,
    )
}

/// Traverses the planetary quadtree using exact 6-plane frustum culling, horizon occlusion,
/// heightmap average-altitude tile proximity (`tile_avg_altitude_m`, returning `0.0` for sea level
/// when heightmaps are off), and optional gaze-centered idle high-resolution refinement (`idle_lod`).
#[allow(clippy::too_many_arguments)]
pub fn traverse_quadtree_lod_frustum6_with_elevation<F>(
    camera: &EarthCamera,
    up_override: Option<DVec3>,
    viewport_width_px: f64,
    viewport_height_px: f64,
    sse_threshold_px: f64,
    max_level: u8,
    tile_grid: u32,
    horizon_cull: bool,
    max_leaves: usize,
    projection: TileProjection,
    idle_lod: IdleLodParams,
    focus_elevation_m: f64,
    mut tile_avg_altitude_m: F,
) -> Vec<TileKey>
where
    F: FnMut(TileKey) -> f64,
{
    let idle_active = idle_lod.is_active();
    let cap = max_leaves
        .saturating_add(if idle_active { idle_lod.extra_leaves } else { 0 })
        .max(4);
    let placement = camera.placement_with_up(up_override);
    let eye = placement.eye;
    let cam_forward = placement.aim.sub(eye).normalize();
    let frustum = Frustum6::from_camera_with_up(
        camera,
        up_override,
        viewport_width_px,
        viewport_height_px,
    );
    let mut heap = std::collections::BinaryHeap::<LodCandidate>::with_capacity(cap * 2);
    let mut leaves = Vec::with_capacity(cap);

    let lod_grid = tile_grid.max(65);
    let root = TileKey::new(0, 0, 0);
    let root_span_m = 2.0 * std::f64::consts::PI * wgs84::A;
    let root_sse = compute_tile_sse(
        root_span_m,
        lod_grid,
        eye.length().max(1.0),
        viewport_height_px,
        camera.fov_y_deg,
    );
    heap.push(LodCandidate {
        sse: root_sse,
        key: root,
    });

    let focus_h = if focus_elevation_m.is_finite() {
        focus_elevation_m.max(0.0)
    } else {
        0.0
    };
    let eye_alt = (eye.length() - (wgs84::A + focus_h)).max(1.0);
    let max_idle_boost = if idle_active {
        (1.0 / idle_lod.sse_scale.clamp(0.15, 1.0))
            .min(2.0f64.powi(idle_lod.max_extra_levels.min(4) as i32))
    } else {
        1.0
    };
    let center_bias = idle_lod.view_center_bias.clamp(0.0, 1.0);

    while let Some(node) = heap.pop() {
        let key = node.key;
        if key.level < max_level && node.sse > sse_threshold_px {
            let mut visible_children = [None, None, None, None];
            let mut vis_count = 0usize;
            for q in 0..4u8 {
                let Some(child) = key.child(q) else {
                    continue;
                };
                let avg_alt_m = tile_avg_altitude_m(child);
                let (center, radius) =
                    tile_bounding_sphere_projected_with_altitude(child, projection, avg_alt_m);
                if horizon_cull && is_tile_horizon_culled(center, radius, eye) {
                    continue;
                }
                if horizon_cull && !frustum.intersects_sphere(center, radius * 1.15) {
                    continue;
                }
                let surface_r = (radius - 5_000.0).max(0.0);
                let to_center = center.sub(eye);
                let center_dist = to_center.length().max(1.0);
                let dist_to_eye = (center_dist - surface_r).max(1.0);
                let per_axis = child.tiles_per_axis() as f64;
                let lat_cos = if projection == TileProjection::WebMercator {
                    let horiz_r = (center.x * center.x + center.y * center.y).sqrt();
                    (horiz_r / center.length().max(1.0)).clamp(0.25, 1.0)
                } else {
                    1.0
                };
                let ground_span_m =
                    ((2.0 * std::f64::consts::PI * wgs84::A) / per_axis) * lat_cos;
                let mut child_sse = compute_tile_sse(
                    ground_span_m,
                    lod_grid,
                    dist_to_eye,
                    viewport_height_px,
                    camera.fov_y_deg,
                );
                if eye_alt < 450_000.0 {
                    let surf_up = center.scale(1.0 / center.length().max(1.0));
                    let cos_inc = (-to_center.dot(surf_up) / center_dist).clamp(0.16, 1.0);
                    let mut foreshorten = cos_inc.sqrt().clamp(0.42, 1.0);
                    let dist_ratio = dist_to_eye / eye_alt;
                    let mut dist_falloff =
                        (1.0 / (1.0 + 0.24 * (dist_ratio - 2.0).max(0.0))).max(0.22);
                    if idle_active {
                        // While idle, relax grazing foreshortening & distant-valley falloff on tiles
                        // directly in front of the view ray so mountains & vistas sharpen visibly.
                        let gaze_cos = (to_center.dot(cam_forward) / center_dist).clamp(0.0, 1.0);
                        let gaze_w = gaze_cos.powf(3.0);
                        let relax = 0.40 + 0.45 * gaze_w;
                        foreshorten += (1.0 - foreshorten) * relax;
                        dist_falloff += (1.0 - dist_falloff) * (relax * 0.72);
                    }
                    child_sse *= foreshorten * dist_falloff;
                }
                if eye_alt < 180.0 {
                    let close_w = (1.0 - eye_alt / 180.0).clamp(0.0, 1.0);
                    let prox_w = (1.0 - dist_to_eye / 280.0).clamp(0.0, 1.0);
                    child_sse *= 1.0 + 1.75 * close_w * prox_w;
                }
                if idle_active {
                    let gaze_cos = (to_center.dot(cam_forward) / center_dist).clamp(0.0, 1.0);
                    let gaze_focus = gaze_cos.powf(4.0);
                    // Prioritize tiles directly near the camera gaze center before peripheral edges.
                    let boost_blend = (1.0 - center_bias) + center_bias * (0.28 + 0.72 * gaze_focus);
                    let eff_boost = 1.0 + (max_idle_boost - 1.0) * boost_blend;
                    child_sse *= eff_boost;
                }
                visible_children[q as usize] = Some(LodCandidate {
                    sse: child_sse,
                    key: child,
                });
                vis_count += 1;
            }
            if vis_count == 0 {
                continue;
            }
            if leaves.len() + heap.len() + vis_count <= cap {
                for ch in visible_children.into_iter().flatten() {
                    heap.push(ch);
                }
            } else {
                leaves.push(key);
                break;
            }
        } else {
            leaves.push(key);
            if leaves.len() + heap.len() >= cap {
                break;
            }
        }
    }
    while let Some(rem) = heap.pop() {
        if leaves.len() < cap {
            leaves.push(rem.key);
        } else {
            break;
        }
    }
    leaves.sort();
    leaves
}

/// Traverses the planetary quadtree for a flat 2D horizontal map with seamless East/West
/// world-wrapping copies (`wrap_x`) and strict North/South pole bounds.
#[allow(clippy::too_many_arguments)]
pub fn traverse_quadtree_lod_2d_flat(
    camera: &EarthCamera,
    viewport_width_px: f64,
    viewport_height_px: f64,
    sse_threshold_px: f64,
    max_level: u8,
    tile_grid: u32,
    max_leaves: usize,
    projection: TileProjection,
) -> Vec<(TileKey, i32)> {
    let cap = max_leaves.max(16);
    let lod_grid = tile_grid.max(65);
    let alt_m = camera.altitude_m().max(45.0);
    let focus_lat = camera.focus.lat_deg.clamp(-85.051129, 85.051129);
    let focus_lon = wrap_longitude(camera.focus.lon_deg);
    let aspect = (viewport_width_px.max(1.0) / viewport_height_px.max(1.0)).max(0.01);
    let tan_half_y = (camera.fov_y_deg.to_radians() * 0.5).tan().max(1e-4);
    let half_h_m = alt_m * tan_half_y * 1.15;
    let half_w_m = half_h_m * aspect * 1.15;
    let m_per_deg = (std::f64::consts::PI * wgs84::A) / 180.0;
    let half_lat_deg = half_h_m / m_per_deg;
    let half_lon_deg = half_w_m / m_per_deg;
    let max_wrap = ((half_lon_deg / 360.0).ceil() as i32 + 2).clamp(1, 4);
    let mirror_shift: Option<i32> = if focus_lon > 135.0 {
        Some(-1)
    } else if focus_lon < -135.0 {
        Some(1)
    } else {
        None
    };

    let tile_eval = |key: TileKey| -> Option<(f64, Vec<i32>)> {
        let bounds = match projection {
            TileProjection::WebMercator => tile_bounds_mercator(key),
            TileProjection::Geographic4326 => tile_bounds(key),
        };
        if bounds.lat1 < focus_lat - half_lat_deg || bounds.lat0 > focus_lat + half_lat_deg {
            return None;
        }
        let tile_mid_lon = (bounds.lon0 + bounds.lon1) * 0.5;
        let half_span_lon = (bounds.lon1 - bounds.lon0) * 0.5;
        // Base 2D tile meshes live in the fixed canonical (0°, 0°) plane at `tile_mid_lon ∈ [-180°, +180°]`.
        // Copy `w` shifts the tile by `w * 360°`, placing it at `tile_mid_lon + w*360 - focus_lon` relative to camera.
        let raw_dlon = tile_mid_lon - focus_lon;
        let mut wraps = Vec::new();
        let mut min_dlon_abs = f64::INFINITY;
        for w in -max_wrap..=max_wrap {
            let dlon = raw_dlon + (w as f64) * 360.0;
            if dlon + half_span_lon >= -half_lon_deg && dlon - half_span_lon <= half_lon_deg {
                wraps.push(w);
                let edge_gap_lon = (dlon.abs() - half_span_lon).max(0.0);
                min_dlon_abs = min_dlon_abs.min(edge_gap_lon);
            }
        }
        if wraps.is_empty() {
            return None;
        }
        let mid_lat = (bounds.lat0 + bounds.lat1) * 0.5;
        let half_span_lat = (bounds.lat1 - bounds.lat0) * 0.5;
        let dlat_gap = ((mid_lat - focus_lat).abs() - half_span_lat).max(0.0);
        let dx_m = min_dlon_abs * m_per_deg;
        let dy_m = dlat_gap * m_per_deg;
        let dist_to_eye = (alt_m * alt_m + dx_m * dx_m + dy_m * dy_m).sqrt();
        let per_axis = key.tiles_per_axis() as f64;
        let lat_cos = if projection == TileProjection::WebMercator {
            mid_lat.to_radians().cos().abs().clamp(0.25, 1.0)
        } else {
            1.0
        };
        let ground_span_m = ((2.0 * std::f64::consts::PI * wgs84::A) / per_axis) * lat_cos;
        let dist_ratio = dist_to_eye / alt_m;
        let dist_falloff = (1.0 / (1.0 + 0.20 * (dist_ratio - 1.5).max(0.0))).max(0.32);
        let sse = compute_tile_sse(
            ground_span_m,
            lod_grid,
            dist_to_eye,
            viewport_height_px,
            camera.fov_y_deg,
        ) * dist_falloff;
        Some((sse, wraps))
    };

    let mut heap = std::collections::BinaryHeap::<LodCandidate>::with_capacity(cap * 2);
    let mut leaves: Vec<TileKey> = Vec::with_capacity(cap);
    let root = TileKey::new(0, 0, 0);
    let root_span_m = 2.0 * std::f64::consts::PI * wgs84::A;
    let root_sse = compute_tile_sse(
        root_span_m,
        lod_grid,
        alt_m,
        viewport_height_px,
        camera.fov_y_deg,
    );
    heap.push(LodCandidate {
        sse: root_sse,
        key: root,
    });

    while let Some(node) = heap.pop() {
        let key = node.key;
        if key.level < max_level && (key.level == 0 || node.sse > sse_threshold_px) {
            let mut children = [None, None, None, None];
            let mut vis_count = 0usize;
            for q in 0..4u8 {
                let Some(child) = key.child(q) else {
                    continue;
                };
                if let Some((child_sse, _)) = tile_eval(child) {
                    children[q as usize] = Some(LodCandidate {
                        sse: child_sse,
                        key: child,
                    });
                    vis_count += 1;
                }
            }
            if vis_count == 0 {
                continue;
            }
            if leaves.len() + heap.len() + vis_count <= cap {
                for ch in children.into_iter().flatten() {
                    heap.push(ch);
                }
            } else {
                leaves.push(key);
                break;
            }
        } else {
            leaves.push(key);
            if leaves.len() + heap.len() >= cap {
                break;
            }
        }
    }
    while let Some(rem) = heap.pop() {
        if leaves.len() < cap {
            leaves.push(rem.key);
        } else {
            break;
        }
    }
    leaves.sort();

    let mut out = Vec::with_capacity(leaves.len() * 2);
    for key in leaves {
        if let Some((_, mut wraps)) = tile_eval(key) {
            if let Some(shift) = mirror_shift {
                let base_len = wraps.len();
                for i in 0..base_len {
                    let mw = wraps[i] + shift;
                    if !wraps.contains(&mw) {
                        wraps.push(mw);
                    }
                }
            }
            for w in wraps {
                out.push((key, w));
            }
        }
    }
    out
}

/// Traverses the planetary quadtree using exact 6-plane frustum culling, horizon occlusion,
/// and SSE-prioritized subdivision capped safely to [`MAX_QUADTREE_LEAVES`].
pub fn traverse_quadtree_lod_frustum6(
    camera: &EarthCamera,
    viewport_width_px: f64,
    viewport_height_px: f64,
    sse_threshold_px: f64,
    max_level: u8,
    tile_grid: u32,
) -> Vec<TileKey> {
    traverse_quadtree_lod_frustum6_with_options(
        camera,
        viewport_width_px,
        viewport_height_px,
        sse_threshold_px,
        max_level,
        tile_grid,
        true,
        MAX_QUADTREE_LEAVES,
    )
}

/// Extended quadtree traversal with explicit horizon-cull toggle (for 2D map morph) and leaf budget.
#[allow(clippy::too_many_arguments)]
pub fn traverse_quadtree_lod_frustum6_with_options(
    camera: &EarthCamera,
    viewport_width_px: f64,
    viewport_height_px: f64,
    sse_threshold_px: f64,
    max_level: u8,
    tile_grid: u32,
    horizon_cull: bool,
    max_leaves: usize,
) -> Vec<TileKey> {
    let placement = camera.placement();
    let frustum = Frustum6::from_camera(camera, viewport_width_px, viewport_height_px);
    let threshold = sse_threshold_px.max(0.5);
    let leaf_cap = max_leaves.max(16);

    let eval_candidate = |key: TileKey| -> Option<LodCandidate> {
        let (center, radius) = tile_bounding_sphere(key);
        if horizon_cull && is_tile_horizon_culled(center, radius, placement.eye) {
            return None;
        }
        if horizon_cull && !frustum.intersects_sphere(center, radius * 1.15) {
            return None;
        }
        let dist_to_eye = (center.sub(placement.eye).length() - radius).max(1.0);
        let per_axis = key.tiles_per_axis() as f64;
        let ground_span_m = (2.0 * std::f64::consts::PI * wgs84::A) / per_axis;
        let sse = if key.level < max_level {
            compute_tile_sse(
                ground_span_m,
                tile_grid,
                dist_to_eye,
                viewport_height_px,
                placement.fov_y_deg,
            )
        } else {
            0.0
        };
        Some(LodCandidate { sse, key })
    };

    let mut heap = std::collections::BinaryHeap::new();
    let mut leaves = Vec::new();
    if let Some(root) = eval_candidate(TileKey::new(0, 0, 0)) {
        heap.push(root);
    }

    while let Some(top) = heap.pop() {
        let can_subdivide = top.key.level < max_level
            && top.sse > threshold
            && (leaves.len() + heap.len() + 4 <= leaf_cap || top.key.level < 2);
        if can_subdivide {
            let mut pushed = 0usize;
            for q in 0..4 {
                if let Some(child) = top.key.child(q) {
                    if let Some(cand) = eval_candidate(child) {
                        heap.push(cand);
                        pushed += 1;
                    }
                }
            }
            if pushed == 0 && !horizon_cull {
                leaves.push(top.key);
            }
        } else {
            leaves.push(top.key);
        }
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
    retained_camera_uniforms_with_up(
        camera,
        None,
        viewport_w,
        viewport_h,
        sun_dir,
        sun_color,
        ambient_color,
        background,
    )
}

#[allow(clippy::too_many_arguments)]
fn retained_camera_uniforms_from_placement(
    placement: CameraPlacement,
    viewport_w: f64,
    viewport_h: f64,
    sun_dir: [f32; 3],
    sun_color: [f32; 3],
    ambient_color: [f32; 3],
    background: strata_render::scene3d::Background3D,
) -> strata_render::scene3d::RetainedCameraUniforms {
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
        ..Default::default()
    }
}

/// Same as [`retained_camera_uniforms`] with an optional explicit ECEF camera up vector.
#[allow(clippy::too_many_arguments)]
pub fn retained_camera_uniforms_with_up(
    camera: &EarthCamera,
    up_override: Option<DVec3>,
    viewport_w: f64,
    viewport_h: f64,
    sun_dir: [f32; 3],
    sun_color: [f32; 3],
    ambient_color: [f32; 3],
    background: strata_render::scene3d::Background3D,
) -> strata_render::scene3d::RetainedCameraUniforms {
    let placement = camera.placement_with_up(up_override);
    retained_camera_uniforms_from_placement(
        placement,
        viewport_w,
        viewport_h,
        sun_dir,
        sun_color,
        ambient_color,
        background,
    )
}

/// Same as [`retained_camera_uniforms_with_up`], honoring the active [`SurfaceMorphParams`] frame
/// (including the fixed canonical 2D map plane when `map_2d_morph >= 0.999`).
#[allow(clippy::too_many_arguments)]
pub fn retained_camera_uniforms_for_morph(
    camera: &EarthCamera,
    morph: SurfaceMorphParams,
    up_override: Option<DVec3>,
    viewport_w: f64,
    viewport_h: f64,
    sun_dir: [f32; 3],
    sun_color: [f32; 3],
    ambient_color: [f32; 3],
    background: strata_render::scene3d::Background3D,
) -> strata_render::scene3d::RetainedCameraUniforms {
    let placement = camera.placement_for_morph(morph, up_override);
    retained_camera_uniforms_from_placement(
        placement,
        viewport_w,
        viewport_h,
        sun_dir,
        sun_color,
        ambient_color,
        background,
    )
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
    build_tile_local_rtc_mesh_morphed(
        key,
        bounds,
        grid,
        elevation,
        elevation_tile,
        generate_skirts,
        skirt_depth_m,
        revision,
        face,
        SurfaceMorphParams {
            projection: TileProjection::Geographic4326,
            ..SurfaceMorphParams::default()
        },
    )
}

/// Builds a tile-local RTC mesh with configurable tile projection (`WebMercator` or `Geographic4326`)
/// and continuous vertex morphing across 3D globe, local 2.5D ground plane, and flattened 2D map.
#[allow(clippy::too_many_arguments)]
pub fn build_tile_local_rtc_mesh_morphed(
    key: TileKey,
    bounds: PatchBounds,
    grid: u32,
    elevation: Option<&crate::tiles::TileElevation>,
    elevation_tile: TileKey,
    generate_skirts: bool,
    skirt_depth_m: f64,
    revision: u64,
    face: u8,
    morph: SurfaceMorphParams,
) -> strata_render::scene3d::RetainedTileMesh {
    let n = grid.max(2) as usize;
    let flat_t = morph.flat_blend.clamp(0.0, 1.0);
    let map_2d = morph.map_2d_morph.clamp(0.0, 1.0);
    let per_axis = key.tiles_per_axis() as f64;
    let tile_mid_lon = (bounds.lon0 + bounds.lon1) * 0.5;
    let focus_surface = Geodetic::new(morph.focus.lat_deg, morph.focus.lon_deg, 0.0);
    let focus_frame = LocalFrame::at(focus_surface.lat_deg, focus_surface.lon_deg);
    let focus_ecef = geodetic_to_ecef(focus_surface);
    let cos_focus = focus_surface.lat_deg.to_radians().cos().abs().max(0.25);
    let cos_scale = cos_focus + (1.0 - cos_focus) * map_2d;

    let project_point = |lat: f64, lon: f64, height: f64| -> (DVec3, DVec3) {
        let p_curved = geodetic_to_ecef(Geodetic::new(lat, lon, height));
        let radial_curved = p_curved.normalize();
        if flat_t <= 1e-6 {
            return (p_curved, radial_curved);
        }
        let dlon_deg = wrap_longitude(tile_mid_lon - morph.focus.lon_deg) + (lon - tile_mid_lon);
        let dlat_deg = lat - morph.focus.lat_deg;
        let east_m = dlon_deg.to_radians() * wgs84::A * cos_scale;
        let north_m = dlat_deg.to_radians() * wgs84::A;
        let up_h = height * (1.0 - map_2d);
        let p_flat = focus_ecef
            .add(focus_frame.east.scale(east_m))
            .add(focus_frame.north.scale(north_m))
            .add(focus_frame.up.scale(up_h));
        let p_morphed = DVec3::new(
            p_curved.x + (p_flat.x - p_curved.x) * flat_t,
            p_curved.y + (p_flat.y - p_curved.y) * flat_t,
            p_curved.z + (p_flat.z - p_curved.z) * flat_t,
        );
        let up_morphed = DVec3::new(
            radial_curved.x + (focus_frame.up.x - radial_curved.x) * flat_t,
            radial_curved.y + (focus_frame.up.y - radial_curved.y) * flat_t,
            radial_curved.z + (focus_frame.up.z - radial_curved.z) * flat_t,
        )
        .normalize();
        (p_morphed, up_morphed)
    };

    let mid_lat = (bounds.lat0 + bounds.lat1) * 0.5;
    let (center_dvec, _) = project_point(mid_lat.clamp(-89.9, 89.9), tile_mid_lon, 0.0);

    let mut positions = Vec::with_capacity(n * n);
    let mut radials = Vec::with_capacity(n * n);
    let mut heights = Vec::with_capacity(n * n);
    let mut lat_rows = Vec::with_capacity(n);
    let mut uvs = Vec::with_capacity(n * n);

    for row in 0..n {
        let v = row as f64 / (n - 1) as f64;
        let v_tex = 1.0 - v;
        let lat_row = match morph.projection {
            TileProjection::WebMercator if face == 0 => {
                let gy_top = (key.y as f64 + v_tex) / per_axis;
                mercator_gy_to_lat_deg(gy_top)
            }
            _ => (bounds.lat0 + (bounds.lat1 - bounds.lat0) * v).clamp(-89.9, 89.9),
        };
        lat_rows.push(lat_row);
        for col in 0..n {
            let u = col as f64 / (n - 1) as f64;
            let lon = bounds.lon0 + (bounds.lon1 - bounds.lon0) * u;
            let height = elevation
                .map(|elev_grid| {
                    let (tu, tv) = if morph.projection == TileProjection::WebMercator
                        && face == 0
                        && key.level >= elevation_tile.level
                    {
                        let dl = key.level - elevation_tile.level;
                        let scale = (1u32 << dl.min(20)) as f64;
                        let dx = (key.x - (elevation_tile.x << dl)) as f64;
                        let dy = (key.y - (elevation_tile.y << dl)) as f64;
                        (((dx + u) / scale) as f32, ((dy + v_tex) / scale) as f32)
                    } else {
                        crate::scene::uv_in_tile(elevation_tile, lat_row, lon)
                    };
                    elev_grid.sample_bilinear(tu, tv) as f64
                })
                .unwrap_or(0.0);
            let (p, radial) = project_point(lat_row, lon, height);
            positions.push(p);
            radials.push(radial);
            heights.push(height * (1.0 - map_2d));
            // Texture row 0 is North (lat1 -> v = 1.0 in mesh), row H-1 is South (lat0 -> v = 0.0).
            uvs.push([u as f32, v_tex as f32]);
        }
    }

    let mut normals = Vec::with_capacity(n * n);
    if elevation.is_none() || map_2d >= 0.999 {
        for radial in &radials {
            normals.push(radial.as_f32());
        }
    } else {
        let lon_span_rad = (bounds.lon1 - bounds.lon0).to_radians().abs().max(1e-9);
        for row in 0..n {
            let lat_row = lat_rows[row];
            let cos_lat = lat_row.to_radians().cos().abs().max(0.1);
            let r0 = row.saturating_sub(1);
            let r1 = (row + 1).min(n - 1);
            let dn_m = ((lat_rows[r1] - lat_rows[r0]).to_radians().abs() * wgs84::A).max(1.0);
            for col in 0..n {
                let idx = row * n + col;
                let radial = radials[idx];
                let c0 = col.saturating_sub(1);
                let c1 = (col + 1).min(n - 1);
                let de_m = (((c1 - c0) as f64 / (n - 1) as f64) * lon_span_rad * wgs84::A * cos_lat)
                    .max(1.0);
                let dh_e = heights[row * n + c1] - heights[row * n + c0];
                let dh_n = heights[r1 * n + col] - heights[r0 * n + col];
                if dh_e.abs() < 1e-4 && dh_n.abs() < 1e-4 {
                    normals.push(radial.as_f32());
                } else {
                    let u = col as f64 / (n - 1) as f64;
                    let lon = bounds.lon0 + (bounds.lon1 - bounds.lon0) * u;
                    let lf = LocalFrame::at(lat_row, lon);
                    let east_dir = DVec3::new(
                        lf.east.x + (focus_frame.east.x - lf.east.x) * flat_t,
                        lf.east.y + (focus_frame.east.y - lf.east.y) * flat_t,
                        lf.east.z + (focus_frame.east.z - lf.east.z) * flat_t,
                    )
                    .normalize();
                    let north_dir = DVec3::new(
                        lf.north.x + (focus_frame.north.x - lf.north.x) * flat_t,
                        lf.north.y + (focus_frame.north.y - lf.north.y) * flat_t,
                        lf.north.z + (focus_frame.north.z - lf.north.z) * flat_t,
                    )
                    .normalize();
                    let nrm = radial
                        .sub(east_dir.scale(dh_e / de_m))
                        .sub(north_dir.scale(dh_n / dn_m))
                        .normalize();
                    normals.push(nrm.as_f32());
                }
            }
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
