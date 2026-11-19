//! Interactive 3D globe, 2.5D terrain orbit, and 2D flat-map camera controller.
//!
//! Pure mathematical state machine with zero windowing or widget dependencies: feeds cleanly
//! from Strata component pointer/wheel events (or headless/benchmark drivers) and produces
//! [`EarthCamera`] + ECEF screen-up hints guaranteed to remain clear of terrain elevation.

use crate::geodesy::{
    ecef_to_geodetic, geodetic_to_ecef, raycast_ellipsoid, screen_to_ray_with_up, wgs84,
    EarthCamera, Geodetic, LocalFrame,
};
use strata_render::camera64::DVec3;

/// Closest permitted camera range above the surface focus point (metres).
pub const MIN_RANGE_M: f64 = 45.0;
/// Farthest permitted camera range for outer orbital views (metres).
pub const MAX_RANGE_M: f64 = 24_000_000.0;
/// Minimum vertical clearance maintained between camera eye and sampled terrain elevation (metres).
pub const GROUND_CLEARANCE_M: f64 = 25.0;
/// Altitude band below which 3D globe arcball seamlessly transitions into 2.5D tangent-plane orbit.
pub const LOCAL_TRANSITION_RANGE_M: f64 = 180_000.0;
/// Default vertical field of view in degrees.
pub const DEFAULT_FOV_Y_DEG: f64 = 52.0;

#[inline]
fn smoothstep(edge0: f64, edge1: f64, x: f64) -> f64 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[inline]
fn normalize_deg(deg: f64) -> f64 {
    deg.rem_euclid(360.0)
}

#[inline]
fn normalize3(v: [f64; 3]) -> [f64; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len > 1e-12 {
        [v[0] / len, v[1] / len, v[2] / len]
    } else {
        [0.0, 0.0, 1.0]
    }
}

#[inline]
fn rotate_vec_axis_angle(v: DVec3, axis: DVec3, angle_rad: f64) -> DVec3 {
    let axis_len = axis.length();
    if axis_len < 1e-12 || angle_rad.abs() < 1e-12 {
        return v;
    }
    let k = axis.scale(1.0 / axis_len);
    let (s, c) = angle_rad.sin_cos();
    let k_dot_v = k.dot(v);
    let k_cross_v = k.cross(v);
    v.scale(c)
        .add(k_cross_v.scale(s))
        .add(k.scale(k_dot_v * (1.0 - c)))
}

#[inline]
fn unit_dir_to_lat_lon(dir: [f64; 3]) -> (f64, f64) {
    let d = normalize3(dir);
    let lat = d[2].clamp(-1.0, 1.0).asin().to_degrees().clamp(-89.95, 89.95);
    let lon = d[1].atan2(d[0]).to_degrees();
    (lat, lon)
}

#[inline]
fn lat_lon_to_unit_dir(lat_deg: f64, lon_deg: f64) -> [f64; 3] {
    let lat = lat_deg.clamp(-89.95, 89.95).to_radians();
    let lon = lon_deg.to_radians();
    let (slat, clat) = lat.sin_cos();
    let (slon, clon) = lon.sin_cos();
    [clat * clon, clat * slon, slat]
}

#[inline]
fn up_dir_from_heading(lat_deg: f64, lon_deg: f64, heading_deg: f64) -> [f64; 3] {
    let frame = LocalFrame::at(lat_deg, lon_deg);
    let h = heading_deg.to_radians();
    let dir = frame.north.scale(h.cos()).add(frame.east.scale(h.sin())).normalize();
    [dir.x, dir.y, dir.z]
}

use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// Default destination altitude above surface (`75 m`, within `50..=100 m`) when `altitude_m` is `None`.
pub const DEFAULT_FLY_TO_ALTITUDE_M: f64 = 75.0;
/// Full-world overview range (`18,500 km`) reached during the 3D pull-back phase of [`EarthOrbitController::fly_to`].
pub const GLOBAL_OVERVIEW_RANGE_M: f64 = 18_500_000.0;

const DRAG_VELOCITY_WINDOW: Duration = Duration::from_millis(90);

#[inline]
fn smootherstep01(u: f64) -> f64 {
    let t = u.clamp(0.0, 1.0);
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

#[inline]
fn shortest_angle_delta_deg(from_deg: f64, to_deg: f64) -> f64 {
    ((to_deg - from_deg) + 180.0).rem_euclid(360.0) - 180.0
}

/// Converts a camera altitude in metres into a continuous 2D Web Mercator zoom value (`0.0..=22.0`).
#[inline]
pub fn zoom_2d_for_altitude(altitude_m: f64) -> f64 {
    let world_span_m = 2.0 * std::f64::consts::PI * wgs84::A;
    let tan_half_fov = (DEFAULT_FOV_Y_DEG.to_radians() * 0.5).tan();
    let visible_m = 2.0 * altitude_m.max(1.0) * tan_half_fov;
    let ratio = (world_span_m / visible_m.max(1e-6)).max(1e-6);
    (ratio.log2() + 1.0).clamp(0.0, 22.0)
}

/// Converts a continuous 2D Web Mercator zoom value (`0.0..=22.0`) into the corresponding camera altitude in metres.
#[inline]
pub fn altitude_for_zoom_2d(zoom: f64) -> f64 {
    let world_span_m = 2.0 * std::f64::consts::PI * wgs84::A;
    let tan_half_fov = (DEFAULT_FOV_Y_DEG.to_radians() * 0.5).tan();
    let z = zoom.clamp(0.0, 22.0);
    let visible_m = world_span_m / 2.0f64.powf(z - 1.0);
    (visible_m / (2.0 * tan_half_fov)).clamp(MIN_RANGE_M, MAX_RANGE_M)
}

/// Initiates a smooth three-stage global transit on `controller`: pulls back to a whole-world
/// overview, orbits/pans to `(lat_deg, lon_deg)`, and zooms in to `altitude_m` (defaulting to
/// [`DEFAULT_FLY_TO_ALTITUDE_M`] = `75 m`, converted to 2D zoom in 2D map mode).
pub fn fly_to(
    controller: &mut EarthOrbitController,
    lat_deg: f64,
    lon_deg: f64,
    altitude_m: Option<f64>,
) {
    controller.fly_to(lat_deg, lon_deg, altitude_m);
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DragAction {
    PanOrArcball,
    OrbitTilt,
}

#[derive(Debug, Clone, Copy)]
struct DragVelSample {
    t: Instant,
    dx: f64,
    dy: f64,
}

#[derive(Debug, Clone, Copy)]
struct OrbitalTransit {
    start_lat_deg: f64,
    start_lon_deg: f64,
    target_lat_deg: f64,
    target_lon_deg: f64,
    start_dir: [f64; 3],
    target_dir: [f64; 3],
    central_angle_rad: f64,
    antipodal_axis: [f64; 3],
    start_range_m: f64,
    apex_range_m: f64,
    end_range_3d_m: f64,
    end_zoom_2d: f64,
    start_heading_deg: f64,
    start_tilt_deg: f64,
    ascent_s: f64,
    slew_start_s: f64,
    slew_s: f64,
    descent_start_s: f64,
    descent_s: f64,
    total_s: f64,
    elapsed_s: f64,
}

/// Interactive camera controller supporting 3D globe arcball, 2.5D close terrain orbit,
/// and 2D flat-map panning/zooming with smooth momentum damping and strict ground collision clamping.
#[derive(Debug, Clone)]
pub struct EarthOrbitController {
    pub lat_deg: f64,
    pub lon_deg: f64,
    pub focus_ground_m: f64,
    pub eye_dir: [f64; 3],
    pub up_dir: [f64; 3],
    pub range_m: f64,
    pub target_range_m: f64,
    pub heading_deg: f64,
    pub tilt_deg: f64,
    pub viewport_w: f64,
    pub viewport_h: f64,
    drag_action: Option<DragAction>,
    last_pointer: Option<(f64, f64)>,
    zoom_cursor_px: Option<(f64, f64)>,
    spin_axis: [f64; 3],
    spin_vel_rad_s: f64,
    /// Surface pan/orbit angular velocity in `(east_great_circle_deg_s, north_deg_s)`.
    pan_vel_gc_deg_s: (f64, f64),
    orbit_vel_deg_s: (f64, f64),
    drag_samples: VecDeque<DragVelSample>,
    drag_start_instant: Option<Instant>,
    carry_pan_vel: (f64, f64),
    carry_orbit_vel: (f64, f64),
    active_transit: Option<OrbitalTransit>,
    is_2d_hint: bool,
    ground_needs_snap: bool,
    last_middle_click: Option<(Instant, f64, f64)>,
    middle_press_origin: Option<(f64, f64)>,
}

impl Default for EarthOrbitController {
    fn default() -> Self {
        let mut ctrl = Self {
            lat_deg: 26.0,
            lon_deg: 16.0,
            focus_ground_m: 0.0,
            eye_dir: [0.0; 3],
            up_dir: [0.0; 3],
            range_m: 13_500_000.0,
            target_range_m: 13_500_000.0,
            heading_deg: 0.0,
            tilt_deg: 0.0,
            viewport_w: 1440.0,
            viewport_h: 900.0,
            drag_action: None,
            last_pointer: None,
            zoom_cursor_px: None,
            spin_axis: [0.0, 0.0, 1.0],
            spin_vel_rad_s: 0.0,
            pan_vel_gc_deg_s: (0.0, 0.0),
            orbit_vel_deg_s: (0.0, 0.0),
            drag_samples: VecDeque::with_capacity(32),
            drag_start_instant: None,
            carry_pan_vel: (0.0, 0.0),
            carry_orbit_vel: (0.0, 0.0),
            active_transit: None,
            is_2d_hint: false,
            ground_needs_snap: true,
            last_middle_click: None,
            middle_press_origin: None,
        };
        ctrl.sync_dirs_from_geo();
        ctrl
    }
}

impl EarthOrbitController {
    /// Creates a new controller initialized to a global hero framing.
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    fn max_tilt_for_range(range_m: f64) -> f64 {
        84.0 * (1.0 - smoothstep(180_000.0, 1_500_000.0, range_m) * 0.4)
    }

    #[inline]
    fn sync_dirs_from_geo(&mut self) {
        self.eye_dir = lat_lon_to_unit_dir(self.lat_deg, self.lon_deg);
        self.up_dir = up_dir_from_heading(self.lat_deg, self.lon_deg, self.heading_deg);
    }

    #[inline]
    fn sync_geo_from_dirs(&mut self) {
        let (lat, lon) = unit_dir_to_lat_lon(self.eye_dir);
        self.lat_deg = lat;
        self.lon_deg = lon;
        self.up_dir = up_dir_from_heading(lat, lon, self.heading_deg);
    }

    #[inline]
    pub fn is_dragging(&self) -> bool {
        self.drag_action.is_some()
    }

    /// Returns `true` while a three-stage [`Self::fly_to`] global transit animation is in progress.
    #[inline]
    pub fn is_flying(&self) -> bool {
        self.active_transit.is_some()
    }

    /// Cancels any active [`Self::fly_to`] transit animation at the current camera pose.
    #[inline]
    pub fn cancel_fly_to(&mut self) {
        self.active_transit = None;
    }

    /// Applies an exponential natural-log range step (`delta_ln < 0.0` zooms in closer;
    /// `delta_ln > 0.0` zooms out farther) centered on the current focus point.
    #[inline]
    pub fn zoom_by_ln(&mut self, delta_ln: f64) {
        if !delta_ln.is_finite() || delta_ln.abs() < 1e-9 {
            return;
        }
        self.active_transit = None;
        self.zoom_cursor_px = None;
        self.target_range_m =
            (self.target_range_m * delta_ln.exp()).clamp(MIN_RANGE_M, MAX_RANGE_M);
    }

    /// Multiplies target camera range by `factor` (`< 1.0` zooms in; `> 1.0` zooms out).
    #[inline]
    pub fn zoom_by_factor(&mut self, factor: f64) {
        if factor.is_finite() && factor > 0.0 {
            self.zoom_by_ln(factor.ln());
        }
    }

    /// Applies an exponential natural-log range step anchored toward viewport pixel `(x, y)`.
    #[inline]
    pub fn zoom_at_screen_px(&mut self, delta_ln: f64, x: f64, y: f64) {
        if !delta_ln.is_finite() || delta_ln.abs() < 1e-9 {
            return;
        }
        self.active_transit = None;
        self.target_range_m =
            (self.target_range_m * delta_ln.exp()).clamp(MIN_RANGE_M, MAX_RANGE_M);
        self.zoom_cursor_px = Some((x, y));
    }

    /// Shifts camera focus by `(d_lat_deg, d_lon_deg)` degrees.
    pub fn pan_by_deg(&mut self, d_lat_deg: f64, d_lon_deg: f64) {
        self.active_transit = None;
        let lat_lim = if self.is_2d_hint { 85.05112878 } else { 89.95 };
        self.lat_deg = (self.lat_deg + d_lat_deg).clamp(-lat_lim, lat_lim);
        self.lon_deg = ((self.lon_deg + d_lon_deg) + 180.0).rem_euclid(360.0) - 180.0;
        self.sync_dirs_from_geo();
    }

    /// Rotates camera heading and pitch (`tilt_deg`) in degrees around the focus point.
    pub fn orbit_by_deg(&mut self, d_heading_deg: f64, d_tilt_deg: f64) {
        self.active_transit = None;
        if self.is_2d_hint {
            self.heading_deg = 0.0;
            self.tilt_deg = 0.0;
        } else {
            let max_tilt = Self::max_tilt_for_range(self.range_m);
            self.heading_deg = normalize_deg(self.heading_deg + d_heading_deg);
            self.tilt_deg = (self.tilt_deg + d_tilt_deg).clamp(0.0, max_tilt);
        }
        self.sync_dirs_from_geo();
    }

    #[inline]
    pub fn set_viewport_size(&mut self, w: f64, h: f64) {
        if w > 1.0 {
            self.viewport_w = w;
        }
        if h > 1.0 {
            self.viewport_h = h;
        }
    }

    #[inline]
    pub fn set_is_2d_hint(&mut self, is_2d: bool) {
        self.is_2d_hint = is_2d;
        if is_2d {
            self.tilt_deg = 0.0;
            self.spin_vel_rad_s = 0.0;
        }
    }

    /// Immediately jumps camera focus, distance, heading, and tilt, canceling active momentum.
    pub fn jump_to(
        &mut self,
        lat_deg: f64,
        lon_deg: f64,
        range_m: f64,
        heading_deg: f64,
        tilt_deg: f64,
    ) {
        self.lat_deg = lat_deg.clamp(-89.95, 89.95);
        self.lon_deg = (lon_deg + 180.0).rem_euclid(360.0) - 180.0;
        self.range_m = range_m.clamp(MIN_RANGE_M, MAX_RANGE_M);
        self.target_range_m = self.range_m;
        self.heading_deg = normalize_deg(heading_deg);
        self.tilt_deg = tilt_deg.clamp(0.0, 84.0);
        self.focus_ground_m = 0.0;
        self.ground_needs_snap = true;
        self.spin_vel_rad_s = 0.0;
        self.pan_vel_gc_deg_s = (0.0, 0.0);
        self.orbit_vel_deg_s = (0.0, 0.0);
        self.drag_samples.clear();
        self.drag_start_instant = None;
        self.carry_pan_vel = (0.0, 0.0);
        self.carry_orbit_vel = (0.0, 0.0);
        self.active_transit = None;
        self.zoom_cursor_px = None;
        self.sync_dirs_from_geo();
    }

    /// Smoothly transits to any `(lat_deg, lon_deg)` point on Earth via a three-stage global
    /// trajectory:
    /// 1. Pulls all the way back to [`GLOBAL_OVERVIEW_RANGE_M`] (`16,500 km`) so the whole world is visible.
    /// 2. Orbits/pans across the globe along the shortest path to `(lat_deg, lon_deg)`.
    /// 3. Zooms smoothly down to `altitude_m` (defaulting to [`DEFAULT_FLY_TO_ALTITUDE_M`] = `75 m`
    ///    when `None`; in 2D map mode, `altitude_m` is converted to the equivalent 2D zoom level
    ///    via [`zoom_2d_for_altitude`]).
    pub fn fly_to(&mut self, lat_deg: f64, lon_deg: f64, altitude_m: Option<f64>) {
        let target_lat = lat_deg.clamp(-89.95, 89.95);
        let target_lon = (lon_deg + 180.0).rem_euclid(360.0) - 180.0;
        let requested_alt = altitude_m.unwrap_or(DEFAULT_FLY_TO_ALTITUDE_M);
        let end_range_3d_m = requested_alt.clamp(MIN_RANGE_M, MAX_RANGE_M);
        let end_zoom_2d = zoom_2d_for_altitude(requested_alt);

        let start_lat = self.lat_deg.clamp(-89.95, 89.95);
        let start_lon = (self.lon_deg + 180.0).rem_euclid(360.0) - 180.0;
        let start_dir = lat_lon_to_unit_dir(start_lat, start_lon);
        let target_dir = lat_lon_to_unit_dir(target_lat, target_lon);

        let u0 = DVec3::new(start_dir[0], start_dir[1], start_dir[2]);
        let u1 = DVec3::new(target_dir[0], target_dir[1], target_dir[2]);
        let dot = u0.dot(u1).clamp(-1.0, 1.0);
        let central_angle_rad = dot.acos();

        let z_axis = DVec3::new(0.0, 0.0, 1.0);
        let mut anti = z_axis.cross(u0);
        if anti.length() < 1e-4 {
            anti = DVec3::new(1.0, 0.0, 0.0).cross(u0);
        }
        let anti = anti.normalize();

        let start_range_m = self.range_m.clamp(MIN_RANGE_M, MAX_RANGE_M);
        let apex_range_m = GLOBAL_OVERVIEW_RANGE_M
            .max(start_range_m)
            .max(end_range_3d_m)
            .min(MAX_RANGE_M);

        let climb_ln = (apex_range_m / start_range_m.max(MIN_RANGE_M)).max(1.0).ln();
        let ascent_s = (0.38 + climb_ln * 0.085).clamp(0.40, 1.45);

        let angular_deg = central_angle_rad.to_degrees();
        let slew_s = (0.35 + (angular_deg / 180.0) * 0.95).clamp(0.35, 1.30);

        let drop_ln = (apex_range_m / end_range_3d_m.max(MIN_RANGE_M)).max(1.0).ln();
        let descent_s = (0.38 + drop_ln * 0.095).clamp(0.40, 1.55);

        let slew_start_s = ascent_s * 0.85;
        let slew_end_s = slew_start_s + slew_s;
        let descent_start_s = (slew_end_s - slew_s * 0.15).max(ascent_s);
        let total_s = descent_start_s + descent_s;

        self.drag_action = None;
        self.drag_samples.clear();
        self.drag_start_instant = None;
        self.spin_vel_rad_s = 0.0;
        self.pan_vel_gc_deg_s = (0.0, 0.0);
        self.orbit_vel_deg_s = (0.0, 0.0);
        self.carry_pan_vel = (0.0, 0.0);
        self.carry_orbit_vel = (0.0, 0.0);
        self.zoom_cursor_px = None;
        self.ground_needs_snap = true;

        self.active_transit = Some(OrbitalTransit {
            start_lat_deg: start_lat,
            start_lon_deg: start_lon,
            target_lat_deg: target_lat,
            target_lon_deg: target_lon,
            start_dir,
            target_dir,
            central_angle_rad,
            antipodal_axis: [anti.x, anti.y, anti.z],
            start_range_m,
            apex_range_m,
            end_range_3d_m,
            end_zoom_2d,
            start_heading_deg: self.heading_deg,
            start_tilt_deg: self.tilt_deg,
            ascent_s,
            slew_start_s,
            slew_s,
            descent_start_s,
            descent_s,
            total_s,
            elapsed_s: 0.0,
        });
    }

    /// Computes active ground focus weight (`0.0` at globe altitudes / 2D, `1.0` near terrain).
    #[inline]
    pub fn ground_focus_weight(&self, is_2d: bool) -> f64 {
        if is_2d {
            0.0
        } else {
            1.0 - smoothstep(120_000.0, 260_000.0, self.range_m)
        }
    }

    #[inline]
    fn effective_heading_deg(&self, is_2d: bool) -> f64 {
        if is_2d {
            0.0
        } else {
            let local_w = (1.0 - smoothstep(40_000.0, LOCAL_TRANSITION_RANGE_M, self.range_m))
                * smoothstep(8.0, 28.0, self.tilt_deg);
            let mut h = normalize_deg(self.heading_deg);
            if h > 180.0 {
                h -= 360.0;
            }
            normalize_deg(h * local_w)
        }
    }

    #[inline]
    fn raw_camera_and_up(&self, is_2d: bool, ground_elev_m: f64) -> (EarthCamera, Option<[f64; 3]>) {
        let gw = self.ground_focus_weight(is_2d);
        let focus_h = ground_elev_m * gw;
        let lat = if is_2d {
            self.lat_deg.clamp(-85.05112878, 85.05112878)
        } else {
            self.lat_deg.clamp(-89.95, 89.95)
        };
        let eff_tilt = if is_2d {
            0.0
        } else {
            // Smoothly relax extreme tilt as range climbs above transition band to keep whole globe centered.
            let max_t = 84.0 * (1.0 - smoothstep(180_000.0, 1_500_000.0, self.range_m) * 0.4);
            self.tilt_deg.clamp(0.0, max_t)
        };
        let tilt_rad = eff_tilt.to_radians();
        let eff_heading = self.effective_heading_deg(is_2d);
        let h_rad = eff_heading.to_radians();

        let horiz = self.range_m * tilt_rad.sin();
        let up_offset = self.range_m * tilt_rad.cos();
        let eye_offset_m = [
            -horiz * h_rad.sin(),
            -horiz * h_rad.cos(),
            up_offset,
        ];

        let near_m = (self.range_m * 0.002).clamp(0.5, 20_000.0);
        let cam = EarthCamera {
            focus: Geodetic::new(lat, self.lon_deg, focus_h),
            aim_offset_m: [0.0, 0.0, 0.0],
            eye_offset_m,
            fov_y_deg: DEFAULT_FOV_Y_DEG,
            near_m,
        };

        let up_opt = if is_2d {
            None
        } else {
            Some(up_dir_from_heading(lat, self.lon_deg, eff_heading))
        };
        (cam, up_opt)
    }

    /// Produces the resolved [`EarthCamera`] and optional ECEF screen-up vector using `ground_elev_m`
    /// as the surface anchor altitude at focus (scaled smoothly to zero at globe altitude / 2D).
    ///
    /// Also enforces `eye_offset_m[2] + focus.height_m >= ground_elev_m + GROUND_CLEARANCE_M`
    /// (unless `is_2d` where camera stays nadir above ellipsoid).
    pub fn to_camera_and_up(
        &self,
        is_2d: bool,
        ground_elev_m: f64,
    ) -> (EarthCamera, Option<[f64; 3]>) {
        let (mut cam, up_opt) = self.raw_camera_and_up(is_2d, ground_elev_m);
        if !is_2d {
            Self::clamp_camera_against_floor(&mut cam, ground_elev_m.max(0.0) + GROUND_CLEARANCE_M);
        }
        (cam, up_opt)
    }

    /// Produces a terrain-clamped [`EarthCamera`] and ECEF screen-up vector, sampling terrain under the
    /// actual resolved eye `(eye_lat, eye_lon)` so oblique angles near ridges never clip underground.
    pub fn to_camera_and_up_clamped(
        &self,
        is_2d: bool,
        ground_sampler: impl Fn(f64, f64) -> f64,
    ) -> (EarthCamera, Option<[f64; 3]>) {
        let focus_ground = if is_2d {
            0.0
        } else {
            ground_sampler(self.lat_deg, self.lon_deg).max(0.0)
        };
        let (mut cam, up_opt) = self.raw_camera_and_up(is_2d, focus_ground);
        if !is_2d {
            let up_vec = up_opt.map(|u| DVec3::new(u[0], u[1], u[2]));
            Self::clamp_camera_with_sampler(&mut cam, up_vec, &ground_sampler);
        }
        (cam, up_opt)
    }

    fn clamp_camera_against_floor(cam: &mut EarthCamera, floor_m: f64) {
        let cur_h = cam.altitude_m();
        if cur_h < floor_m {
            let lift = floor_m - cur_h;
            let dist = cam.eye_distance_m().max(MIN_RANGE_M);
            let new_u = cam.eye_offset_m[2] + lift;
            if new_u < dist {
                let target_horiz = (dist * dist - new_u * new_u).max(0.0).sqrt();
                let cur_horiz =
                    (cam.eye_offset_m[0] * cam.eye_offset_m[0] + cam.eye_offset_m[1] * cam.eye_offset_m[1])
                        .sqrt();
                if cur_horiz > 1e-6 {
                    let s = target_horiz / cur_horiz;
                    cam.eye_offset_m[0] *= s;
                    cam.eye_offset_m[1] *= s;
                }
            }
            cam.eye_offset_m[2] = new_u;
        }
    }

    fn clamp_camera_with_sampler<F: Fn(f64, f64) -> f64 + ?Sized>(
        cam: &mut EarthCamera,
        up_vec: Option<DVec3>,
        ground_sampler: &F,
    ) -> bool {
        let mut adjusted = false;
        for _ in 0..2 {
            let placement = cam.placement_with_up(up_vec);
            let eye_geo = ecef_to_geodetic(placement.eye);
            let terrain_h = ground_sampler(eye_geo.lat_deg, eye_geo.lon_deg).max(0.0);
            let min_eye_h = terrain_h + GROUND_CLEARANCE_M;
            if eye_geo.height_m < min_eye_h {
                let lift = min_eye_h - eye_geo.height_m;
                let dist = cam.eye_distance_m().max(MIN_RANGE_M);
                let new_u = cam.eye_offset_m[2] + lift;
                if new_u < dist {
                    let target_horiz = (dist * dist - new_u * new_u).max(0.0).sqrt();
                    let cur_horiz = (cam.eye_offset_m[0] * cam.eye_offset_m[0]
                        + cam.eye_offset_m[1] * cam.eye_offset_m[1])
                        .sqrt();
                    if cur_horiz > 1e-6 {
                        let s = target_horiz / cur_horiz;
                        cam.eye_offset_m[0] *= s;
                        cam.eye_offset_m[1] *= s;
                    }
                }
                cam.eye_offset_m[2] = new_u;
                adjusted = true;
            } else {
                break;
            }
        }
        adjusted
    }

    /// Returns the continuous 2D Web Mercator zoom value (`0.0..=22.0`) corresponding to `self.range_m`.
    #[inline]
    pub fn zoom_2d(&self) -> f64 {
        zoom_2d_for_altitude(self.range_m)
    }

    /// Approximates active Web Mercator / slippy tile zoom level (`0..=19`) in 2D mode.
    pub fn level_for_2d(&self) -> u8 {
        self.zoom_2d().round().clamp(0.0, 19.0) as u8
    }

    /// Resets camera pitch (`tilt_deg`) back to `0.0` (nadir), returning `true` if pitch changed.
    pub fn reset_pitch(&mut self) -> bool {
        let changed = self.tilt_deg.abs() > 1e-6
            || self.orbit_vel_deg_s.1.abs() > 1e-6
            || self.active_transit.is_some();
        self.tilt_deg = 0.0;
        self.orbit_vel_deg_s = (0.0, 0.0);
        self.carry_orbit_vel = (0.0, 0.0);
        self.active_transit = None;
        self.drag_samples.clear();
        self.drag_start_instant = None;
        self.drag_action = None;
        self.last_pointer = None;
        self.middle_press_origin = None;
        self.sync_dirs_from_geo();
        changed
    }

    #[inline]
    fn record_drag_sample_at(&mut self, now: Instant, dx: f64, dy: f64) {
        self.drag_samples.push_back(DragVelSample { t: now, dx, dy });
        while self
            .drag_samples
            .front()
            .is_some_and(|s| now.saturating_duration_since(s.t) > DRAG_VELOCITY_WINDOW)
        {
            self.drag_samples.pop_front();
        }
    }

    fn commit_release_momentum_at(&mut self, action: DragAction, now: Instant) {
        while self
            .drag_samples
            .front()
            .is_some_and(|s| now.saturating_duration_since(s.t) > DRAG_VELOCITY_WINDOW)
        {
            self.drag_samples.pop_front();
        }
        let Some(oldest) = self.drag_samples.front().copied() else {
            return;
        };

        let mut sum_x = 0.0f64;
        let mut sum_y = 0.0f64;
        for s in &self.drag_samples {
            sum_x += s.dx;
            sum_y += s.dy;
        }

        let raw_span_s = now.saturating_duration_since(oldest.t).as_secs_f64();
        let (dt_window, release_gain) = if self.drag_samples.len() <= 1 && raw_span_s < 0.005 {
            (0.050, 1.0)
        } else {
            (raw_span_s.max(0.028), 0.45)
        };
        let mut vx = (sum_x / dt_window) * release_gain;
        let mut vy = (sum_y / dt_window) * release_gain;

        let stroke_s = self
            .drag_start_instant
            .map(|t0| now.saturating_duration_since(t0).as_secs_f64())
            .unwrap_or(0.0);

        let carry = match action {
            DragAction::PanOrArcball => self.carry_pan_vel,
            DragAction::OrbitTilt => self.carry_orbit_vel,
        };
        if stroke_s < 0.38 {
            let carry_decay = (-1.5 * stroke_s).exp();
            let cx = carry.0 * carry_decay;
            let cy = carry.1 * carry_decay;
            let dot = vx * cx + vy * cy;
            let v_mag_sq = vx * vx + vy * vy;
            if dot > 0.0 && v_mag_sq > 1e-9 {
                let proj_scale = (dot / v_mag_sq).clamp(0.0, 5.0) * 0.90;
                vx += vx * proj_scale;
                vy += vy * proj_scale;
            }
        }

        match action {
            DragAction::PanOrArcball => {
                let max_gc_deg_s =
                    ((self.range_m / wgs84::A).to_degrees() * 11.0).clamp(0.002, 620.0);
                let mag = vx.hypot(vy);
                if mag > max_gc_deg_s && mag > 1e-12 {
                    let s = max_gc_deg_s / mag;
                    vx *= s;
                    vy *= s;
                }
                self.pan_vel_gc_deg_s = (vx, vy);
            }
            DragAction::OrbitTilt => {
                self.orbit_vel_deg_s = (vx.clamp(-620.0, 620.0), vy.clamp(-240.0, 240.0));
            }
        }
    }

    /// Starts a pointer interaction (`0` = Left pan/arcball, `1`/`2` = Middle/Right heading+tilt orbit).
    /// Double middle-clicking (`button == 1`) in 3D mode resets camera pitch (`tilt_deg`) to `0.0`.
    pub fn handle_mouse_down(&mut self, button: u32, x: f64, y: f64) -> bool {
        self.handle_mouse_down_at(button, x, y, Instant::now())
    }

    fn handle_mouse_down_at(&mut self, button: u32, x: f64, y: f64, now: Instant) -> bool {
        self.last_pointer = Some((x, y));
        self.active_transit = None;
        self.carry_pan_vel = self.pan_vel_gc_deg_s;
        self.carry_orbit_vel = self.orbit_vel_deg_s;
        self.drag_samples.clear();
        self.drag_start_instant = Some(now);
        self.spin_vel_rad_s = 0.0;
        self.pan_vel_gc_deg_s = (0.0, 0.0);
        self.orbit_vel_deg_s = (0.0, 0.0);
        self.zoom_cursor_px = None;

        if button == 1 && !self.is_2d_hint {
            if let Some((prev_time, px, py)) = self.last_middle_click {
                if now.saturating_duration_since(prev_time) <= Duration::from_millis(450)
                    && (x - px).hypot(y - py) <= 24.0
                {
                    self.last_middle_click = None;
                    self.reset_pitch();
                    return true;
                }
            }
            self.last_middle_click = Some((now, x, y));
            self.middle_press_origin = Some((x, y));
        } else {
            self.last_middle_click = None;
            self.middle_press_origin = None;
        }

        self.drag_action = match button {
            0 => Some(DragAction::PanOrArcball),
            1 | 2 => Some(DragAction::OrbitTilt),
            _ => None,
        };
        self.drag_action.is_some()
    }

    /// Releases an active pointer drag interaction, transferring recent flick velocity into smooth momentum.
    pub fn handle_mouse_up(&mut self, button: u32) -> bool {
        self.handle_mouse_up_at(button, Instant::now())
    }

    fn handle_mouse_up_at(&mut self, _button: u32, now: Instant) -> bool {
        let prev_action = self.drag_action.take();
        if let Some(action) = prev_action {
            self.commit_release_momentum_at(action, now);
        }
        self.drag_samples.clear();
        self.drag_start_instant = None;
        self.carry_pan_vel = (0.0, 0.0);
        self.carry_orbit_vel = (0.0, 0.0);
        self.last_pointer = None;
        self.middle_press_origin = None;
        prev_action.is_some()
    }

    /// Cancels pointer tracking if the cursor leaves without pointer capture.
    pub fn handle_cursor_left(&mut self) {
        self.drag_action = None;
        self.drag_samples.clear();
        self.drag_start_instant = None;
        self.carry_pan_vel = (0.0, 0.0);
        self.carry_orbit_vel = (0.0, 0.0);
        self.last_pointer = None;
        self.middle_press_origin = None;
    }

    /// Applies pointer drag motion using component-local `(x, y)` coordinates.
    pub fn handle_mouse_move(&mut self, x: f64, y: f64) -> bool {
        self.handle_mouse_move_at(x, y, Instant::now())
    }

    fn handle_mouse_move_at(&mut self, x: f64, y: f64, now: Instant) -> bool {
        let Some(action) = self.drag_action else {
            self.last_pointer = Some((x, y));
            return false;
        };
        let Some((px, py)) = self.last_pointer else {
            self.last_pointer = Some((x, y));
            return false;
        };
        if let Some((ox, oy)) = self.middle_press_origin {
            if (x - ox).hypot(y - oy) > 6.0 {
                self.last_middle_click = None;
                self.middle_press_origin = None;
            }
        }
        self.last_pointer = Some((x, y));
        let dx = x - px;
        let dy = y - py;
        if dx.abs() < 1e-6 && dy.abs() < 1e-6 {
            return false;
        }

        match action {
            DragAction::OrbitTilt => {
                self.orbit_25d_around_pivot(dx, dy, now);
            }
            DragAction::PanOrArcball => {
                if self.is_2d_hint {
                    self.pan_2d(dx, dy, now);
                } else if self.range_m > LOCAL_TRANSITION_RANGE_M && self.tilt_deg < 4.0 {
                    self.drag_globe_arcball(px, py, x, y, now);
                } else {
                    self.drag_local_tangent_plane(px, py, x, y, now);
                }
            }
        }
        true
    }

    fn orbit_25d_around_pivot(&mut self, dx: f64, dy: f64, now: Instant) {
        let max_t = if self.is_2d_hint {
            0.0
        } else {
            84.0 * (1.0 - smoothstep(180_000.0, 1_500_000.0, self.range_m) * 0.4)
        };
        let d_tilt = -dy * 0.24;
        let prev_tilt = self.tilt_deg;
        self.tilt_deg = (self.tilt_deg + d_tilt).clamp(0.0, max_t);
        let applied_tilt = self.tilt_deg - prev_tilt;
        let local_w = if self.is_2d_hint {
            0.0
        } else {
            (1.0 - smoothstep(40_000.0, LOCAL_TRANSITION_RANGE_M, self.range_m))
                * smoothstep(8.0, 28.0, self.tilt_deg)
        };
        let d_heading = dx * 0.28 * local_w;
        self.heading_deg = normalize_deg(self.heading_deg + d_heading);
        self.record_drag_sample_at(now, d_heading, applied_tilt);
        self.sync_dirs_from_geo();
    }

    fn pan_2d(&mut self, dx: f64, dy: f64, now: Instant) {
        let visible_m = 2.0 * self.range_m * (DEFAULT_FOV_Y_DEG.to_radians() * 0.5).tan();
        let m_per_px = visible_m / self.viewport_h.max(1.0);
        let cos_lat = self.lat_deg.to_radians().cos().abs().max(0.15);
        let d_lon = -(dx * m_per_px) / (wgs84::A * cos_lat).to_radians();
        let d_lat = (dy * m_per_px) / wgs84::A.to_radians();
        self.lon_deg = ((self.lon_deg + d_lon) + 180.0).rem_euclid(360.0) - 180.0;
        self.lat_deg = (self.lat_deg + d_lat).clamp(-85.05112878, 85.05112878);
        self.record_drag_sample_at(now, d_lon * cos_lat, d_lat);
        self.sync_dirs_from_geo();
    }

    fn drag_globe_arcball(&mut self, px: f64, py: f64, x: f64, y: f64, now: Instant) {
        let dx = x - px;
        let dy = y - py;
        // Unified roll-free angular rotation calibrated at the sub-camera focus point so
        // dragging inside the sphere, across the horizon limb, and outside the Earth boundary
        // moves smoothly along the exact same screen axes with zero diagonal deflection or roll.
        let fov_rad = DEFAULT_FOV_Y_DEG.to_radians();
        let visible_arc_m = 2.0 * (fov_rad * 0.5).tan() * self.range_m.min(wgs84::A * 2.2);
        let deg_per_px = (visible_arc_m / wgs84::A).to_degrees() / self.viewport_h.max(1.0);
        let h_rad = self.effective_heading_deg(false).to_radians();
        let step_east = -dx * h_rad.cos() + dy * h_rad.sin();
        let step_north = dx * h_rad.sin() + dy * h_rad.cos();
        let d_lat = step_north * deg_per_px;
        let cos_lat = self.lat_deg.to_radians().cos().abs().max(0.25);
        let d_east_gc_deg = step_east * deg_per_px;
        let d_lon = d_east_gc_deg / cos_lat;
        self.lat_deg = (self.lat_deg + d_lat).clamp(-89.95, 89.95);
        self.lon_deg = ((self.lon_deg + d_lon) + 180.0).rem_euclid(360.0) - 180.0;
        self.record_drag_sample_at(now, d_east_gc_deg, d_lat);
        self.spin_vel_rad_s = 0.0;
        self.sync_dirs_from_geo();
    }

    fn drag_local_tangent_plane(&mut self, px: f64, py: f64, x: f64, y: f64, now: Instant) {
        let gw = self.ground_focus_weight(false);
        let plane_h = self.focus_ground_m * gw;
        let (cam, up_opt) = self.raw_camera_and_up(false, self.focus_ground_m);
        let up_v = up_opt.map(|u| DVec3::new(u[0], u[1], u[2]));
        let (o1, d1) = screen_to_ray_with_up(px, py, self.viewport_w, self.viewport_h, &cam, up_v);
        let (o2, d2) = screen_to_ray_with_up(x, y, self.viewport_w, self.viewport_h, &cam, up_v);

        let frame = LocalFrame::at(self.lat_deg, self.lon_deg);
        let origin = geodetic_to_ecef(Geodetic::new(self.lat_deg, self.lon_deg, plane_h));
        let denom1 = d1.dot(frame.up);
        let denom2 = d2.dot(frame.up);

        let (d_east_m, d_north_m) = if denom1 < -0.04 && denom2 < -0.04 {
            let t1 = origin.sub(o1).dot(frame.up) / denom1;
            let t2 = origin.sub(o2).dot(frame.up) / denom2;
            let max_t = self.range_m * 14.0 + 10_000.0;
            if t1 > 0.0 && t2 > 0.0 && t1 < max_t && t2 < max_t {
                let p1 = o1.add(d1.scale(t1));
                let p2 = o2.add(d2.scale(t2));
                let delta = p1.sub(p2);
                (delta.dot(frame.east), delta.dot(frame.north))
            } else {
                self.fallback_tangent_delta_m(x - px, y - py)
            }
        } else {
            self.fallback_tangent_delta_m(x - px, y - py)
        };

        let m_per_deg_lat = wgs84::A.to_radians();
        let cos_lat = self.lat_deg.to_radians().cos().abs().max(0.15);
        let m_per_deg_lon = m_per_deg_lat * cos_lat;
        let max_step_deg = (self.range_m / m_per_deg_lat).max(1e-4) * 3.0;
        let d_lon = (d_east_m / m_per_deg_lon).clamp(-max_step_deg, max_step_deg);
        let d_lat = (d_north_m / m_per_deg_lat).clamp(-max_step_deg, max_step_deg);

        self.lat_deg = (self.lat_deg + d_lat).clamp(-89.95, 89.95);
        self.lon_deg = ((self.lon_deg + d_lon) + 180.0).rem_euclid(360.0) - 180.0;
        self.record_drag_sample_at(now, d_lon * cos_lat, d_lat);
        self.sync_dirs_from_geo();
    }

    fn fallback_tangent_delta_m(&self, dx: f64, dy: f64) -> (f64, f64) {
        let visible_m = 2.0 * self.range_m * (DEFAULT_FOV_Y_DEG.to_radians() * 0.5).tan();
        let m_per_px = visible_m / self.viewport_h.max(1.0);
        let h = self.heading_deg.to_radians();
        // Dragging right (dx > 0) shifts camera left along -right; dragging down (dy > 0) shifts forward.
        let right_step = -dx * m_per_px;
        let fwd_step = dy * m_per_px;
        let d_east = right_step * h.cos() + fwd_step * h.sin();
        let d_north = -right_step * h.sin() + fwd_step * h.cos();
        (d_east, d_north)
    }

    /// Processes scroll-wheel zoom (`delta_y < 0` zooms closer, `delta_y > 0` zooms farther)
    /// anchored toward component-local cursor `(x, y)`.
    pub fn handle_scroll(&mut self, delta_y: f64, x: f64, y: f64) {
        if delta_y.abs() < 1e-6 {
            return;
        }
        self.active_transit = None;
        let steps = (delta_y / 110.0).clamp(-3.5, 3.5);
        let factor = (steps * 0.18).exp();
        self.target_range_m = (self.target_range_m * factor).clamp(MIN_RANGE_M, MAX_RANGE_M);
        self.zoom_cursor_px = Some((x, y));
    }

    fn step_orbital_transit(&mut self, dt: f64, is_2d: bool) -> bool {
        let Some(mut tr) = self.active_transit else {
            return false;
        };
        tr.elapsed_s = (tr.elapsed_s + dt).min(tr.total_s);
        let t = tr.elapsed_s;

        let end_range_m = if is_2d {
            altitude_for_zoom_2d(tr.end_zoom_2d)
        } else {
            tr.end_range_3d_m
        };
        let apex_range_m = if is_2d {
            MAX_RANGE_M
        } else {
            tr.apex_range_m.max(end_range_m)
        };

        // Stage 1 (pull back to whole-world view) & Stage 3 (zoom in to destination altitude)
        if t <= tr.ascent_s {
            let u1 = smootherstep01(t / tr.ascent_s.max(1e-6));
            let ln_r = tr.start_range_m.ln() + (apex_range_m.ln() - tr.start_range_m.ln()) * u1;
            self.range_m = ln_r.exp().clamp(MIN_RANGE_M, MAX_RANGE_M);
            self.tilt_deg = if is_2d {
                0.0
            } else {
                tr.start_tilt_deg * (1.0 - u1)
            };
            let dh = shortest_angle_delta_deg(tr.start_heading_deg, 0.0);
            self.heading_deg = if is_2d {
                0.0
            } else {
                normalize_deg(tr.start_heading_deg + dh * u1)
            };
        } else if t < tr.descent_start_s {
            self.range_m = apex_range_m;
            self.tilt_deg = 0.0;
            self.heading_deg = 0.0;
        } else {
            let u3 = smootherstep01((t - tr.descent_start_s) / tr.descent_s.max(1e-6));
            let ln_r = apex_range_m.ln() + (end_range_m.ln() - apex_range_m.ln()) * u3;
            self.range_m = ln_r.exp().clamp(MIN_RANGE_M, MAX_RANGE_M);
            self.tilt_deg = 0.0;
            self.heading_deg = 0.0;
        }
        self.target_range_m = self.range_m;

        // Stage 2 (global orbit / pan from origin to destination)
        let w = smootherstep01((t - tr.slew_start_s) / tr.slew_s.max(1e-6));
        if is_2d {
            let s_lat = tr.start_lat_deg.clamp(-85.05112878, 85.05112878);
            let e_lat = tr.target_lat_deg.clamp(-85.05112878, 85.05112878);
            let d_lon = shortest_angle_delta_deg(tr.start_lon_deg, tr.target_lon_deg);
            self.lat_deg = (s_lat + (e_lat - s_lat) * w).clamp(-85.05112878, 85.05112878);
            self.lon_deg = ((tr.start_lon_deg + d_lon * w) + 180.0).rem_euclid(360.0) - 180.0;
        } else {
            let theta = tr.central_angle_rad;
            let dir = if theta < 1e-6 {
                tr.target_dir
            } else if theta > std::f64::consts::PI - 1e-4 {
                let u0 = DVec3::new(tr.start_dir[0], tr.start_dir[1], tr.start_dir[2]);
                let axis = DVec3::new(
                    tr.antipodal_axis[0],
                    tr.antipodal_axis[1],
                    tr.antipodal_axis[2],
                );
                let v = rotate_vec_axis_angle(u0, axis, w * theta).normalize();
                [v.x, v.y, v.z]
            } else {
                let sin_theta = theta.sin();
                let a = ((1.0 - w) * theta).sin() / sin_theta;
                let b = (w * theta).sin() / sin_theta;
                normalize3([
                    a * tr.start_dir[0] + b * tr.target_dir[0],
                    a * tr.start_dir[1] + b * tr.target_dir[1],
                    a * tr.start_dir[2] + b * tr.target_dir[2],
                ])
            };
            let (lat, lon) = unit_dir_to_lat_lon(dir);
            self.lat_deg = lat;
            self.lon_deg = lon;
        }

        if tr.elapsed_s >= tr.total_s - 1e-9 {
            let lat_lim = if is_2d { 85.05112878 } else { 89.95 };
            self.lat_deg = tr.target_lat_deg.clamp(-lat_lim, lat_lim);
            self.lon_deg = tr.target_lon_deg;
            self.range_m = end_range_m;
            self.target_range_m = end_range_m;
            self.tilt_deg = 0.0;
            self.heading_deg = 0.0;
            self.active_transit = None;
        } else {
            self.active_transit = Some(tr);
        }
        self.sync_dirs_from_geo();
        true
    }

    /// Advances smooth zoom easing, global [`Self::fly_to`] transits, inertia momentum,
    /// and ground-elevation collision clamping.
    ///
    /// Returns `true` if any camera state changed during this step.
    pub fn step_damping(
        &mut self,
        dt_s: f64,
        is_2d: bool,
        ground_sampler: impl Fn(f64, f64) -> f64,
    ) -> bool {
        let dt = dt_s.clamp(0.0, 0.1);
        if dt <= 0.0 {
            return false;
        }
        self.is_2d_hint = is_2d;
        let mut changed = false;

        let transit_active = self.step_orbital_transit(dt, is_2d);
        if transit_active {
            changed = true;
        }

        // 1. Update smoothed terrain height under focus point
        let sampled_focus_ground = if is_2d {
            0.0
        } else {
            ground_sampler(self.lat_deg, self.lon_deg).max(0.0)
        };
        if self.ground_needs_snap {
            if is_2d || sampled_focus_ground > 0.0 || self.focus_ground_m == 0.0 {
                if (sampled_focus_ground - self.focus_ground_m).abs() > 0.05 {
                    changed = true;
                }
                self.focus_ground_m = sampled_focus_ground;
                if sampled_focus_ground > 0.0 || is_2d {
                    self.ground_needs_snap = false;
                }
            }
        } else {
            let ground_alpha = 1.0 - (-12.0 * dt).exp();
            if (sampled_focus_ground - self.focus_ground_m).abs() > 0.05 {
                self.focus_ground_m += (sampled_focus_ground - self.focus_ground_m) * ground_alpha;
                changed = true;
            } else {
                self.focus_ground_m = sampled_focus_ground;
            }
        }

        // 2. Momentum damping when pointer is released
        if self.drag_action.is_none() {
            let decay = (-2.6 * dt).exp();
            if self.spin_vel_rad_s.abs() > 1e-4 {
                let d_angle = self.spin_vel_rad_s * dt;
                let axis = DVec3::new(self.spin_axis[0], self.spin_axis[1], self.spin_axis[2]);
                let cur_eye = DVec3::new(self.eye_dir[0], self.eye_dir[1], self.eye_dir[2]);
                let next_eye = rotate_vec_axis_angle(cur_eye, axis, d_angle).normalize();
                self.eye_dir = [next_eye.x, next_eye.y, next_eye.z];
                self.sync_geo_from_dirs();
                self.spin_vel_rad_s *= decay;
                if self.spin_vel_rad_s.abs() <= 1e-4 {
                    self.spin_vel_rad_s = 0.0;
                }
                changed = true;
            }

            if self.pan_vel_gc_deg_s.0.abs() > 1e-6 || self.pan_vel_gc_deg_s.1.abs() > 1e-6 {
                let lat_lim = if is_2d { 85.05112878 } else { 89.95 };
                let min_cos =
                    if is_2d || self.range_m <= LOCAL_TRANSITION_RANGE_M || self.tilt_deg >= 4.0 {
                        0.15
                    } else {
                        0.25
                    };
                let cos_lat = self.lat_deg.to_radians().cos().abs().max(min_cos);
                let d_lon = (self.pan_vel_gc_deg_s.0 / cos_lat) * dt;
                let d_lat = self.pan_vel_gc_deg_s.1 * dt;
                self.lon_deg = ((self.lon_deg + d_lon) + 180.0).rem_euclid(360.0) - 180.0;
                let unclamped_lat = self.lat_deg + d_lat;
                self.lat_deg = unclamped_lat.clamp(-lat_lim, lat_lim);
                if (unclamped_lat - self.lat_deg).abs() > 1e-9 {
                    self.pan_vel_gc_deg_s.1 = 0.0;
                }
                self.pan_vel_gc_deg_s.0 *= decay;
                self.pan_vel_gc_deg_s.1 *= decay;
                if self.pan_vel_gc_deg_s.0.abs() <= 1e-6 {
                    self.pan_vel_gc_deg_s.0 = 0.0;
                }
                if self.pan_vel_gc_deg_s.1.abs() <= 1e-6 {
                    self.pan_vel_gc_deg_s.1 = 0.0;
                }
                self.sync_dirs_from_geo();
                changed = true;
            }

            if !is_2d && (self.orbit_vel_deg_s.0.abs() > 1e-3 || self.orbit_vel_deg_s.1.abs() > 1e-3)
            {
                let max_t =
                    84.0 * (1.0 - smoothstep(180_000.0, 1_500_000.0, self.range_m) * 0.4);
                self.heading_deg = normalize_deg(self.heading_deg + self.orbit_vel_deg_s.0 * dt);
                let unclamped_t = self.tilt_deg + self.orbit_vel_deg_s.1 * dt;
                self.tilt_deg = unclamped_t.clamp(0.0, max_t);
                if (unclamped_t - self.tilt_deg).abs() > 1e-9 {
                    self.orbit_vel_deg_s.1 = 0.0;
                }
                self.orbit_vel_deg_s.0 *= decay;
                self.orbit_vel_deg_s.1 *= decay;
                if self.orbit_vel_deg_s.0.abs() <= 1e-3 {
                    self.orbit_vel_deg_s.0 = 0.0;
                }
                if self.orbit_vel_deg_s.1.abs() <= 1e-3 {
                    self.orbit_vel_deg_s.1 = 0.0;
                }
                self.sync_dirs_from_geo();
                changed = true;
            }
        }

        // 3. Smooth log-range zoom easing + cursor anchor shift when zooming inward
        let log_diff = (self.target_range_m.ln() - self.range_m.ln()).abs();
        if log_diff > 5e-4 {
            let zoom_alpha = 1.0 - (-12.5 * dt).exp();
            let old_range = self.range_m;
            let new_log = old_range.ln() + (self.target_range_m.ln() - old_range.ln()) * zoom_alpha;
            let new_range = new_log.exp().clamp(MIN_RANGE_M, MAX_RANGE_M);

            if let Some((cx, cy)) = self.zoom_cursor_px {
                if self.drag_action.is_none() && new_range < old_range {
                    self.apply_cursor_zoom_anchor(cx, cy, old_range, new_range, is_2d);
                }
            }
            self.range_m = new_range;
            changed = true;
        } else if (self.range_m - self.target_range_m).abs() > 1e-3 {
            self.range_m = self.target_range_m.clamp(MIN_RANGE_M, MAX_RANGE_M);
            self.zoom_cursor_px = None;
            changed = true;
        }

        // 4. Enforce tilt ceiling & strict ground collision safety in 3D/2.5D
        if !is_2d {
            let max_t = 84.0 * (1.0 - smoothstep(180_000.0, 1_500_000.0, self.range_m) * 0.4);
            if self.tilt_deg > max_t {
                self.tilt_deg = max_t;
                self.sync_dirs_from_geo();
                changed = true;
            }
            let (mut cam, up_opt) = self.raw_camera_and_up(false, self.focus_ground_m);
            let up_vec = up_opt.map(|u| DVec3::new(u[0], u[1], u[2]));
            if Self::clamp_camera_with_sampler(&mut cam, up_vec, &ground_sampler) {
                let dist = cam.eye_distance_m().max(MIN_RANGE_M);
                let cos_t = (cam.eye_offset_m[2] / dist).clamp(-1.0, 1.0);
                let safe_tilt = cos_t.acos().to_degrees().clamp(0.0, max_t);
                if safe_tilt < self.tilt_deg - 1e-3 {
                    self.tilt_deg = safe_tilt;
                    self.orbit_vel_deg_s.1 = 0.0;
                    self.sync_dirs_from_geo();
                    changed = true;
                }
            }
        }

        changed
    }

    fn apply_cursor_zoom_anchor(
        &mut self,
        cx: f64,
        cy: f64,
        old_range: f64,
        new_range: f64,
        is_2d: bool,
    ) {
        if is_2d {
            let ratio = new_range / old_range.max(1e-3);
            let dx = cx - self.viewport_w * 0.5;
            let dy = self.viewport_h * 0.5 - cy;
            let visible_m = 2.0 * old_range * (DEFAULT_FOV_Y_DEG.to_radians() * 0.5).tan();
            let m_per_px = visible_m / self.viewport_h.max(1.0);
            let shift_scale = (1.0 - ratio).clamp(0.0, 0.5);
            let cos_lat = self.lat_deg.to_radians().cos().abs().max(0.15);
            let d_lon = (dx * m_per_px * shift_scale) / (wgs84::A * cos_lat).to_radians();
            let d_lat = (dy * m_per_px * shift_scale) / wgs84::A.to_radians();
            self.lon_deg = ((self.lon_deg + d_lon) + 180.0).rem_euclid(360.0) - 180.0;
            self.lat_deg = (self.lat_deg + d_lat).clamp(-85.05112878, 85.05112878);
            self.sync_dirs_from_geo();
            return;
        }

        let (cam_before, up_opt) = self.raw_camera_and_up(false, self.focus_ground_m);
        let up_v = up_opt.map(|u| DVec3::new(u[0], u[1], u[2]));
        let (o_before, d_before) =
            screen_to_ray_with_up(cx, cy, self.viewport_w, self.viewport_h, &cam_before, up_v);
        let Some(hit_before) = raycast_ellipsoid(o_before, d_before) else {
            return;
        };

        self.range_m = new_range;
        let (cam_after, up_after_opt) = self.raw_camera_and_up(false, self.focus_ground_m);
        let up_after_v = up_after_opt.map(|u| DVec3::new(u[0], u[1], u[2]));
        let (o_after, d_after) =
            screen_to_ray_with_up(cx, cy, self.viewport_w, self.viewport_h, &cam_after, up_after_v);
        let Some(hit_after) = raycast_ellipsoid(o_after, d_after) else {
            self.range_m = old_range;
            return;
        };

        let n_before = hit_before.normalize();
        let n_after = hit_after.normalize();
        let cross = n_after.cross(n_before);
        let sin_a = cross.length();
        let cos_a = n_after.dot(n_before).clamp(-1.0, 1.0);
        let angle = sin_a.atan2(cos_a).clamp(-0.25, 0.25);
        if sin_a > 1e-8 && angle.abs() > 1e-7 {
            let axis = cross.scale(1.0 / sin_a);
            let cur_eye = DVec3::new(self.eye_dir[0], self.eye_dir[1], self.eye_dir[2]);
            let next_eye = rotate_vec_axis_angle(cur_eye, axis, angle).normalize();
            self.eye_dir = [next_eye.x, next_eye.y, next_eye.z];
            self.sync_geo_from_dirs();
        }
        self.range_m = old_range;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn closer_zoom_reaches_min_range_45m_and_keeps_eye_above_terrain() {
        let mut ctrl = EarthOrbitController::new();
        ctrl.jump_to(46.0, 7.6, 500.0, 25.0, 72.0);
        ctrl.set_viewport_size(640.0, 400.0);

        // Zoom repeatedly toward focus
        for _ in 0..40 {
            ctrl.handle_scroll(-110.0, 320.0, 200.0);
        }
        let mountain_sampler = |lat: f64, lon: f64| -> f64 {
            // 4,500 m Matterhorn-scale ridge around 46.0N, 7.6E
            let dlat = (lat - 46.0) * 111_000.0;
            let dlon = (lon - 7.6) * 77_000.0;
            4_500.0 - 0.1 * (dlat.abs() + dlon.abs()).min(500.0)
        };

        for _ in 0..120 {
            ctrl.step_damping(1.0 / 60.0, false, mountain_sampler);
        }

        assert!(
            ctrl.range_m <= 60.0 && ctrl.range_m >= MIN_RANGE_M - 1e-6,
            "expected close range near MIN_RANGE_M=45.0, got {}",
            ctrl.range_m
        );

        let (cam, up_opt) = ctrl.to_camera_and_up_clamped(false, mountain_sampler);
        let up_v = up_opt.map(|u| DVec3::new(u[0], u[1], u[2]));
        let placement = cam.placement_with_up(up_v);
        let eye_geo = ecef_to_geodetic(placement.eye);
        let terrain_h = mountain_sampler(eye_geo.lat_deg, eye_geo.lon_deg);

        assert!(
            eye_geo.height_m >= terrain_h + GROUND_CLEARANCE_M - 0.5,
            "eye height ({}) went below terrain + clearance ({} + {})",
            eye_geo.height_m,
            terrain_h,
            GROUND_CLEARANCE_M
        );
        assert!(
            placement.near_m <= 2.5,
            "expected tight near plane for close ground view, got {}",
            placement.near_m
        );
    }

    #[test]
    fn steep_valley_wall_prevents_underground_penetration() {
        let mut ctrl = EarthOrbitController::new();
        ctrl.jump_to(45.0, 6.0, 180.0, 0.0, 82.0);
        // Focus is in a valley at 1,000 m, while surrounding ridge at eye offset rises to 1,500 m.
        let valley_and_cliff = |lat: f64, _lon: f64| -> f64 {
            if lat < 44.999 {
                1_500.0
            } else {
                1_000.0
            }
        };

        for _ in 0..30 {
            ctrl.step_damping(1.0 / 60.0, false, valley_and_cliff);
        }
        let (cam, up_opt) = ctrl.to_camera_and_up_clamped(false, valley_and_cliff);
        let placement = cam.placement_with_up(up_opt.map(|u| DVec3::new(u[0], u[1], u[2])));
        let eye_geo = ecef_to_geodetic(placement.eye);
        let terrain_h = valley_and_cliff(eye_geo.lat_deg, eye_geo.lon_deg);

        assert!(
            eye_geo.height_m >= terrain_h + GROUND_CLEARANCE_M - 0.5,
            "eye height ({}) breached steep cliff floor ({})",
            eye_geo.height_m,
            terrain_h + GROUND_CLEARANCE_M
        );
    }

    #[test]
    fn non_fullscreen_component_viewport_pans_and_orbits_cleanly() {
        let mut ctrl = EarthOrbitController::new();
        ctrl.jump_to(37.7749, -122.4194, 4_000.0, 0.0, 35.0);
        ctrl.set_viewport_size(640.0, 400.0);

        let before_lat = ctrl.lat_deg;
        let before_lon = ctrl.lon_deg;
        assert!(ctrl.handle_mouse_down(0, 320.0, 200.0));
        assert!(ctrl.handle_mouse_move(240.0, 150.0));
        assert!(ctrl.handle_mouse_up(0));

        assert!(
            (ctrl.lat_deg - before_lat).abs() > 1e-5 || (ctrl.lon_deg - before_lon).abs() > 1e-5,
            "left drag inside 640x400 component should pan focus"
        );

        let before_heading = ctrl.heading_deg;
        let before_tilt = ctrl.tilt_deg;
        assert!(ctrl.handle_mouse_down(2, 320.0, 200.0));
        assert!(ctrl.handle_mouse_move(400.0, 160.0));
        assert!(ctrl.handle_mouse_up(2));

        assert!((ctrl.heading_deg - before_heading).abs() > 1.0);
        assert!((ctrl.tilt_deg - before_tilt).abs() > 1.0);
    }

    #[test]
    fn globe_left_drag_across_limb_keeps_vertical_axis_and_zero_roll() {
        let mut ctrl = EarthOrbitController::new();
        ctrl.set_viewport_size(1440.0, 900.0);
        ctrl.jump_to(26.0, 16.0, 13_500_000.0, 0.0, 0.0);

        let start_lat = ctrl.lat_deg;
        assert!(ctrl.handle_mouse_down(0, 720.0, 450.0));
        // Drag straight left from screen center all the way across the limb into deep space (x=720 -> 80)
        let mut x = 720.0;
        let mut prev_lon = ctrl.lon_deg;
        while x > 80.0 {
            x -= 20.0;
            assert!(ctrl.handle_mouse_move(x, 450.0));
            assert!(
                (ctrl.lat_deg - start_lat).abs() < 1e-9,
                "pure horizontal drag must never deflect diagonally in latitude, got lat={}",
                ctrl.lat_deg
            );
            assert!(
                ctrl.heading_deg.abs() < 1e-9,
                "3D globe drag must never roll camera heading, got heading={}",
                ctrl.heading_deg
            );
            assert!(ctrl.lon_deg != prev_lon);
            prev_lon = ctrl.lon_deg;
        }
        assert!(ctrl.handle_mouse_up(0));
    }

    #[test]
    fn double_middle_click_resets_pitch_in_3d_mode() {
        let mut ctrl = EarthOrbitController::new();
        ctrl.set_viewport_size(1440.0, 900.0);
        ctrl.jump_to(26.0, 16.0, 13_500_000.0, 0.0, 0.0);

        // Middle-click and drag upward to increase pitch (tilt_deg).
        assert!(ctrl.handle_mouse_down(1, 720.0, 450.0));
        assert!(ctrl.handle_mouse_move(720.0, 350.0));
        assert!(ctrl.handle_mouse_up(1));
        assert!(ctrl.tilt_deg > 10.0, "expected positive pitch after middle drag");

        // Double middle-click at the same position resets pitch to 0.0.
        assert!(ctrl.handle_mouse_down(1, 720.0, 350.0));
        assert!(ctrl.handle_mouse_up(1));
        assert!(ctrl.tilt_deg > 10.0, "single middle click must not reset pitch yet");

        assert!(ctrl.handle_mouse_down(1, 722.0, 351.0));
        let _ = ctrl.handle_mouse_up(1);
        assert!(
            ctrl.tilt_deg.abs() < 1e-9,
            "double middle click in 3D mode must reset pitch to 0.0, got {}",
            ctrl.tilt_deg
        );
    }

    #[test]
    fn external_zoom_by_ln_pan_and_orbit_primitives_drive_camera_smoothly() {
        let mut ctrl = EarthOrbitController::new();
        ctrl.set_viewport_size(1440.0, 900.0);
        ctrl.jump_to(38.0, 24.0, 10_000_000.0, 0.0, 0.0);

        // External caller driving zoom_by_ln(-1.95 * dt) for 1.0 second at 144 Hz -> zooms in ~7x closer.
        let dt = 1.0 / 144.0;
        for _ in 0..144 {
            ctrl.zoom_by_ln(-1.95 * dt);
            assert!(ctrl.step_damping(dt, false, |_, _| 0.0));
        }
        assert!(
            ctrl.range_m < 2_200_000.0 && ctrl.range_m > 500_000.0,
            "driving zoom_by_ln(-1.95 * dt) for 1s should zoom smoothly from 10,000 km into ~1,400-1,800 km, got {}",
            ctrl.range_m
        );

        for _ in 0..30 {
            ctrl.step_damping(dt, false, |_, _| 0.0);
        }
        let zoomed_in_range = ctrl.range_m;

        // External caller driving zoom_by_ln(+1.95 * dt) for 1.0 second at 144 Hz -> zooms back out.
        for _ in 0..144 {
            ctrl.zoom_by_ln(1.95 * dt);
            assert!(ctrl.step_damping(dt, false, |_, _| 0.0));
        }
        assert!(
            ctrl.range_m > zoomed_in_range * 4.5,
            "driving zoom_by_ln(+1.95 * dt) for 1s should zoom back out substantially, got {} vs {}",
            ctrl.range_m,
            zoomed_in_range
        );

        ctrl.pan_by_deg(2.5, -4.0);
        assert!((ctrl.lat_deg - 40.5).abs() < 1e-6);
        assert!((ctrl.lon_deg - 20.0).abs() < 1e-6);

        ctrl.jump_to(40.5, 20.0, 4_000.0, 0.0, 10.0);
        ctrl.orbit_by_deg(35.0, 15.0);
        assert!((ctrl.heading_deg - 35.0).abs() < 1e-6);
        assert!((ctrl.tilt_deg - 25.0).abs() < 1e-6);
    }

    #[test]
    fn mouse_flick_coasts_with_momentum_and_repeated_flicks_gain_speed() {
        let mut ctrl = EarthOrbitController::new();
        ctrl.set_viewport_size(1440.0, 900.0);
        ctrl.jump_to(20.0, 0.0, 13_500_000.0, 0.0, 0.0);

        // Simulate a realistic 500 Hz mouse flick (30 packets over 60 ms, plus a tiny 0.5 px finger-lift packet).
        let t0 = Instant::now();
        ctrl.handle_mouse_down_at(0, 720.0, 450.0, t0);
        let mut x = 720.0;
        for i in 1..=30 {
            x -= 8.0;
            ctrl.handle_mouse_move_at(x, 450.0, t0 + Duration::from_millis(i * 2));
        }
        // Tiny slow packet right before button release must NOT wipe out flick velocity.
        ctrl.handle_mouse_move_at(x - 0.5, 450.0, t0 + Duration::from_millis(66));
        ctrl.handle_mouse_up_at(0, t0 + Duration::from_millis(68));

        let vel_after_first_flick = ctrl.pan_vel_gc_deg_s.0;
        assert!(
            vel_after_first_flick > 40.0,
            "expected brisk eastward angular momentum after 500Hz flick, got {vel_after_first_flick}"
        );

        // Coast briefly (80 ms), then flick AGAIN in the same direction -> momentum must build higher!
        let lon_after_release = ctrl.lon_deg;
        for _ in 0..12 {
            assert!(ctrl.step_damping(1.0 / 144.0, false, |_, _| 0.0));
        }
        assert!(
            ctrl.lon_deg > lon_after_release + 2.0,
            "globe should coast smoothly after mouse release"
        );

        let t1 = t0 + Duration::from_millis(150);
        ctrl.handle_mouse_down_at(0, 720.0, 450.0, t1);
        let mut x2 = 720.0;
        for i in 1..=30 {
            x2 -= 8.0;
            ctrl.handle_mouse_move_at(x2, 450.0, t1 + Duration::from_millis(i * 2));
        }
        ctrl.handle_mouse_up_at(0, t1 + Duration::from_millis(65));

        let vel_after_second_flick = ctrl.pan_vel_gc_deg_s.0;
        assert!(
            vel_after_second_flick > vel_after_first_flick * 1.35,
            "consecutive flicks in same direction must gain momentum: first={vel_after_first_flick}, second={vel_after_second_flick}"
        );
    }

    #[test]
    fn pausing_pointer_before_release_stops_without_unwanted_drift() {
        let mut ctrl = EarthOrbitController::new();
        ctrl.set_viewport_size(1440.0, 900.0);
        ctrl.jump_to(20.0, 0.0, 13_500_000.0, 0.0, 0.0);

        let t0 = Instant::now();
        ctrl.handle_mouse_down_at(0, 720.0, 450.0, t0);
        ctrl.handle_mouse_move_at(560.0, 450.0, t0 + Duration::from_millis(25));
        // Hold mouse stationary for 140 ms (> 90 ms window) before releasing button.
        ctrl.handle_mouse_up_at(0, t0 + Duration::from_millis(165));

        assert_eq!(
            ctrl.pan_vel_gc_deg_s,
            (0.0, 0.0),
            "holding pointer still before mouseup must release cleanly with zero drift"
        );
    }

    #[test]
    fn fly_to_default_altitude_pulls_back_to_whole_world_orbits_and_zooms_to_75m() {
        let mut ctrl = EarthOrbitController::new();
        ctrl.set_viewport_size(1440.0, 900.0);
        // Start zoomed close to San Francisco with tilt and heading.
        ctrl.jump_to(37.7749, -122.4194, 420.0, 35.0, 52.0);

        // Transit across the Pacific Date Line to Tokyo with `altitude_m = None` (defaults to 75m).
        fly_to(&mut ctrl, 35.6762, 139.6503, None);
        assert!(ctrl.is_flying());

        let mut peak_range_m = ctrl.range_m;
        let mut steps = 0usize;
        while ctrl.is_flying() && steps < 1000 {
            assert!(ctrl.step_damping(1.0 / 120.0, false, |_, _| 0.0));
            peak_range_m = peak_range_m.max(ctrl.range_m);
            steps += 1;
        }
        assert!(
            !ctrl.is_flying(),
            "fly_to should complete within reasonable duration (steps={steps})"
        );
        assert!(
            peak_range_m >= GLOBAL_OVERVIEW_RANGE_M * 0.99,
            "fly_to must pull all the way back to see the whole world ({}), peak was {}",
            GLOBAL_OVERVIEW_RANGE_M,
            peak_range_m
        );
        assert!(
            (ctrl.lat_deg - 35.6762).abs() < 1e-4 && (ctrl.lon_deg - 139.6503).abs() < 1e-4,
            "expected final destination at Tokyo, got ({}, {})",
            ctrl.lat_deg,
            ctrl.lon_deg
        );
        assert!(
            (ctrl.range_m - DEFAULT_FLY_TO_ALTITUDE_M).abs() < 1e-4
                && (50.0..=100.0).contains(&ctrl.range_m),
            "omitting altitude must land at 50-100m default ({}), got {}",
            DEFAULT_FLY_TO_ALTITUDE_M,
            ctrl.range_m
        );
    }

    #[test]
    fn fly_to_in_2d_mode_converts_altitude_to_zoom_and_roundtrips() {
        for alt in [75.0, 500.0, 4_000.0, 120_000.0, 5_000_000.0, 16_500_000.0] {
            let z = zoom_2d_for_altitude(alt);
            let roundtrip = altitude_for_zoom_2d(z);
            assert!(
                ((roundtrip - alt) / alt).abs() < 1e-6,
                "2D zoom conversion must invert altitude accurately: alt={alt}, z={z}, rt={roundtrip}"
            );
        }

        let mut ctrl = EarthOrbitController::new();
        ctrl.set_viewport_size(1440.0, 900.0);
        ctrl.set_is_2d_hint(true);
        ctrl.jump_to(51.5074, -0.1278, 2_000.0, 0.0, 0.0);

        let custom_alt_m = 1_200.0;
        let expected_zoom = zoom_2d_for_altitude(custom_alt_m);
        ctrl.fly_to(-33.8688, 151.2093, Some(custom_alt_m));

        let mut min_zoom_during_pullback = ctrl.zoom_2d();
        while ctrl.is_flying() {
            ctrl.step_damping(1.0 / 120.0, true, |_, _| 0.0);
            min_zoom_during_pullback = min_zoom_during_pullback.min(ctrl.zoom_2d());
        }
        assert!(
            min_zoom_during_pullback < 2.0,
            "2D fly_to must pull out to whole-world zoom (<2.0), got min zoom {min_zoom_during_pullback}"
        );
        assert!(
            (ctrl.zoom_2d() - expected_zoom).abs() < 1e-4,
            "2D fly_to must land at exact converted zoom {expected_zoom}, got {}",
            ctrl.zoom_2d()
        );
        assert!((ctrl.lat_deg - (-33.8688)).abs() < 1e-4);
        assert!((ctrl.lon_deg - 151.2093).abs() < 1e-4);
    }

    #[test]
    fn fly_to_antipodal_and_manual_interrupt_work_cleanly() {
        let mut ctrl = EarthOrbitController::new();
        ctrl.jump_to(0.0, 0.0, 1_000.0, 0.0, 0.0);
        ctrl.fly_to(0.0, 180.0, Some(90.0));
        for _ in 0..90 {
            ctrl.step_damping(1.0 / 60.0, false, |_, _| 0.0);
            assert!(ctrl.lat_deg.is_finite() && ctrl.lon_deg.is_finite() && ctrl.range_m.is_finite());
        }
        assert!(ctrl.is_flying());
        // Manual mouse press cancels fly_to immediately so user retains instant control.
        ctrl.handle_mouse_down(0, 720.0, 450.0);
        assert!(!ctrl.is_flying());
    }
}


