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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DragAction {
    PanOrArcball,
    OrbitTilt,
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
    pan_vel_lon_lat_s: (f64, f64),
    orbit_vel_deg_s: (f64, f64),
    is_2d_hint: bool,
    ground_needs_snap: bool,
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
            pan_vel_lon_lat_s: (0.0, 0.0),
            orbit_vel_deg_s: (0.0, 0.0),
            is_2d_hint: false,
            ground_needs_snap: true,
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
        self.pan_vel_lon_lat_s = (0.0, 0.0);
        self.orbit_vel_deg_s = (0.0, 0.0);
        self.zoom_cursor_px = None;
        self.sync_dirs_from_geo();
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

    /// Approximates active Web Mercator / slippy tile zoom level (`0..=19`) in 2D mode.
    pub fn level_for_2d(&self) -> u8 {
        let world_span_m = 2.0 * std::f64::consts::PI * wgs84::A;
        let visible_m = 2.0 * self.range_m * (DEFAULT_FOV_Y_DEG.to_radians() * 0.5).tan();
        let ratio = (world_span_m / visible_m.max(1.0)).max(1.0);
        (ratio.log2() + 1.0).round().clamp(0.0, 19.0) as u8
    }

    /// Starts a pointer interaction (`0` = Left pan/arcball, `1`/`2` = Middle/Right heading+tilt orbit).
    pub fn handle_mouse_down(&mut self, button: u32, x: f64, y: f64) -> bool {
        self.last_pointer = Some((x, y));
        self.spin_vel_rad_s = 0.0;
        self.pan_vel_lon_lat_s = (0.0, 0.0);
        self.orbit_vel_deg_s = (0.0, 0.0);
        self.zoom_cursor_px = None;
        self.drag_action = match button {
            0 => Some(DragAction::PanOrArcball),
            1 | 2 => Some(DragAction::OrbitTilt),
            _ => None,
        };
        self.drag_action.is_some()
    }

    /// Releases an active pointer drag interaction.
    pub fn handle_mouse_up(&mut self, _button: u32) -> bool {
        let was_dragging = self.drag_action.is_some();
        self.drag_action = None;
        self.last_pointer = None;
        was_dragging
    }

    /// Cancels pointer tracking if the cursor leaves without pointer capture.
    pub fn handle_cursor_left(&mut self) {
        self.drag_action = None;
        self.last_pointer = None;
    }

    /// Applies pointer drag motion using component-local `(x, y)` coordinates.
    pub fn handle_mouse_move(&mut self, x: f64, y: f64) -> bool {
        let Some(action) = self.drag_action else {
            self.last_pointer = Some((x, y));
            return false;
        };
        let Some((px, py)) = self.last_pointer else {
            self.last_pointer = Some((x, y));
            return false;
        };
        self.last_pointer = Some((x, y));
        let dx = x - px;
        let dy = y - py;
        if dx.abs() < 1e-6 && dy.abs() < 1e-6 {
            return false;
        }

        match action {
            DragAction::OrbitTilt => {
                self.orbit_25d_around_pivot(dx, dy);
            }
            DragAction::PanOrArcball => {
                if self.is_2d_hint {
                    self.pan_2d(dx, dy);
                } else if self.range_m > LOCAL_TRANSITION_RANGE_M && self.tilt_deg < 4.0 {
                    self.drag_globe_arcball(px, py, x, y);
                } else {
                    self.drag_local_tangent_plane(px, py, x, y);
                }
            }
        }
        true
    }

    fn orbit_25d_around_pivot(&mut self, dx: f64, dy: f64) {
        let max_t = if self.is_2d_hint {
            0.0
        } else {
            84.0 * (1.0 - smoothstep(180_000.0, 1_500_000.0, self.range_m) * 0.4)
        };
        let d_tilt = -dy * 0.24;
        self.tilt_deg = (self.tilt_deg + d_tilt).clamp(0.0, max_t);
        let local_w = if self.is_2d_hint {
            0.0
        } else {
            (1.0 - smoothstep(40_000.0, LOCAL_TRANSITION_RANGE_M, self.range_m))
                * smoothstep(8.0, 28.0, self.tilt_deg)
        };
        let d_heading = dx * 0.28 * local_w;
        self.heading_deg = normalize_deg(self.heading_deg + d_heading);
        self.orbit_vel_deg_s = (d_heading * 18.0, d_tilt * 18.0);
        self.sync_dirs_from_geo();
    }

    fn pan_2d(&mut self, dx: f64, dy: f64) {
        let visible_m = 2.0 * self.range_m * (DEFAULT_FOV_Y_DEG.to_radians() * 0.5).tan();
        let m_per_px = visible_m / self.viewport_h.max(1.0);
        let cos_lat = self.lat_deg.to_radians().cos().abs().max(0.15);
        let d_lon = -(dx * m_per_px) / (wgs84::A * cos_lat).to_radians();
        let d_lat = (dy * m_per_px) / wgs84::A.to_radians();
        self.lon_deg = ((self.lon_deg + d_lon) + 180.0).rem_euclid(360.0) - 180.0;
        self.lat_deg = (self.lat_deg + d_lat).clamp(-85.05112878, 85.05112878);
        self.pan_vel_lon_lat_s = (d_lon * 18.0, d_lat * 18.0);
        self.sync_dirs_from_geo();
    }

    fn drag_globe_arcball(&mut self, px: f64, py: f64, x: f64, y: f64) {
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
        let d_lon = step_east * deg_per_px / cos_lat;
        self.lat_deg = (self.lat_deg + d_lat).clamp(-89.95, 89.95);
        self.lon_deg = ((self.lon_deg + d_lon) + 180.0).rem_euclid(360.0) - 180.0;
        self.pan_vel_lon_lat_s = (d_lon * 16.0, d_lat * 16.0);
        self.spin_vel_rad_s = 0.0;
        self.sync_dirs_from_geo();
    }

    fn drag_local_tangent_plane(&mut self, px: f64, py: f64, x: f64, y: f64) {
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
        self.pan_vel_lon_lat_s = (d_lon * 16.0, d_lat * 16.0);
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
        let steps = (delta_y / 110.0).clamp(-3.5, 3.5);
        let factor = (steps * 0.18).exp();
        self.target_range_m = (self.target_range_m * factor).clamp(MIN_RANGE_M, MAX_RANGE_M);
        self.zoom_cursor_px = Some((x, y));
    }

    /// Advances smooth zoom easing, inertia momentum, and ground-elevation collision clamping.
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
            let decay = (-8.5 * dt).exp();
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

            if self.pan_vel_lon_lat_s.0.abs() > 1e-6 || self.pan_vel_lon_lat_s.1.abs() > 1e-6 {
                let lat_lim = if is_2d { 85.05112878 } else { 89.95 };
                self.lon_deg = ((self.lon_deg + self.pan_vel_lon_lat_s.0 * dt) + 180.0)
                    .rem_euclid(360.0)
                    - 180.0;
                self.lat_deg = (self.lat_deg + self.pan_vel_lon_lat_s.1 * dt).clamp(-lat_lim, lat_lim);
                self.pan_vel_lon_lat_s.0 *= decay;
                self.pan_vel_lon_lat_s.1 *= decay;
                if self.pan_vel_lon_lat_s.0.abs() <= 1e-6 {
                    self.pan_vel_lon_lat_s.0 = 0.0;
                }
                if self.pan_vel_lon_lat_s.1.abs() <= 1e-6 {
                    self.pan_vel_lon_lat_s.1 = 0.0;
                }
                self.sync_dirs_from_geo();
                changed = true;
            }

            if !is_2d && (self.orbit_vel_deg_s.0.abs() > 1e-3 || self.orbit_vel_deg_s.1.abs() > 1e-3)
            {
                let max_t =
                    84.0 * (1.0 - smoothstep(180_000.0, 1_500_000.0, self.range_m) * 0.4);
                self.heading_deg = normalize_deg(self.heading_deg + self.orbit_vel_deg_s.0 * dt);
                self.tilt_deg = (self.tilt_deg + self.orbit_vel_deg_s.1 * dt).clamp(0.0, max_t);
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
            let zoom_alpha = 1.0 - (-14.0 * dt).exp();
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
}
