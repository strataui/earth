//! Embeddable Strata `<EarthView>` UI component built on [`EarthViewport`], [`EarthWorld`],
//! [`EarthOrbitController`], and first-class external interaction events.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use strata::platform::{KeyCode, PlatformEvent};
use strata::prelude::*;
use strata::render::scene3d::Canvas3D;
use strata::FrameOutcome;

use crate::controller::EarthOrbitController;
use crate::events::{
    parse_dom_key_to_keycode, EarthCameraState, EarthControlsConfig, EarthKeyEvent,
    EarthModifiers, EarthMouseButton, EarthPointerEvent, EarthWheelEvent,
};
use crate::labels::{
    builtin_world_labels, project_earth_labels, EarthLabelIcon, EarthLabelKind, EarthMapLabel,
    EarthScreenProjector, LiveMapLabelStore, ProjectedEarthLabel,
};
use crate::plugins::{EarthPlugin, EarthPluginRegistry};
use crate::precipitation::{
    AtmosphereSample, PrecipitationConfig, PrecipitationEvaluator, TemperatureProvider,
};
use crate::raycast::EarthRayHit;
use crate::scene::{
    EarthInput, EarthViewMode, IdleHighResConfig, LocalAtmosphereConfig, OceanWavesConfig,
};
use crate::tiles::TileSource;
use crate::weather_api::{BaseWeatherConfig, WeatherProvider, WeatherSnapshot};
use crate::worker::EarthViewport;
use crate::world::{EarthObject, EarthObjectId, EarthWorld, ResolvedObjectTransform};

/// Maximum pointer excursion (in logical pixels) between press and release still classified as a click.
pub const CLICK_MAX_TRAVEL_PX: f64 = 5.0;

#[derive(Debug, Clone, Copy)]
struct ActivePressCandidate {
    button: EarthMouseButton,
    start_x: f64,
    start_y: f64,
    max_travel_px: f64,
    camera_active: bool,
}

#[derive(Default)]
struct EarthEventListeners {
    on_click: Option<Rc<dyn Fn(&EarthPointerEvent)>>,
    on_double_click: Option<Rc<dyn Fn(&EarthPointerEvent)>>,
    on_context_menu: Option<Rc<dyn Fn(&EarthPointerEvent)>>,
    on_pointer_down: Option<Rc<dyn Fn(&EarthPointerEvent)>>,
    on_pointer_move: Option<Rc<dyn Fn(&EarthPointerEvent)>>,
    on_pointer_up: Option<Rc<dyn Fn(&EarthPointerEvent)>>,
    on_wheel: Option<Rc<dyn Fn(&EarthWheelEvent)>>,
    on_key_down: Option<Rc<dyn Fn(&EarthKeyEvent)>>,
    on_key_up: Option<Rc<dyn Fn(&EarthKeyEvent)>>,
    on_camera_change: Option<Rc<dyn Fn(&EarthCameraState)>>,
}

/// Context provided to UI [`EarthOverlayPlugin`] components rendered inside [`EarthView`].
pub struct EarthOverlayContext<'a> {
    pub input: &'a EarthInput,
    pub viewport_width: f32,
    pub viewport_height: f32,
    pub altitude_m: f64,
    pub focus_lat_deg: f64,
    pub focus_lon_deg: f64,
    projector: EarthScreenProjector,
}

impl<'a> EarthOverlayContext<'a> {
    /// Projects `(lat_deg, lon_deg, elevation_m)` into viewport pixel coordinates `(x, y)`.
    /// Returns `None` if occluded behind the Earth horizon or outside clip bounds.
    pub fn project_geodetic(
        &self,
        lat_deg: f64,
        lon_deg: f64,
        elevation_m: f64,
    ) -> Option<(f32, f32)> {
        self.projector.project_point(lat_deg, lon_deg, elevation_m)
    }
}

/// Custom UI/2D overlay plugin attached directly to [`EarthView`].
pub trait EarthOverlayPlugin: 'static {
    /// Stable identifier for toggling or replacing this overlay plugin.
    fn id(&self) -> &'static str;

    fn is_enabled(&self) -> bool {
        true
    }

    fn set_enabled(&mut self, _enabled: bool) {}

    /// Builds a declarative Strata [`View`] overlay positioned over the globe viewport.
    fn render(&self, ctx: &EarthOverlayContext<'_>) -> Option<View>;
}

/// Mutable backing state for an embedded [`EarthView`] component.
pub struct EarthViewState {
    pub viewport: EarthViewport,
    pub orbit: EarthOrbitController,
    pub controls: EarthControlsConfig,
    pub builtin_labels: Vec<EarthMapLabel>,
    pub live_labels: LiveMapLabelStore,
    pub is_grabbing: Signal<bool>,
    pub ui_rev: Signal<u64>,
    overlay_plugins: Vec<Box<dyn EarthOverlayPlugin>>,
    overlay_plugin_rev: u64,
    cached_labels: Vec<ProjectedEarthLabel>,
    cached_nearby_sector: Option<((i32, i32), u64, Vec<EarthMapLabel>)>,
    last_label_sig: u64,
    last_label_refresh: Instant,
    last_labels_enabled: bool,
    last_camera_sync: Option<Instant>,
    camera_sync_pending: bool,
    tile_ready: Arc<AtomicBool>,
    label_ready: Arc<AtomicBool>,
    uncapped: bool,
    last_cursor_px: Option<(f64, f64)>,
    press_candidate: Option<ActivePressCandidate>,
    last_left_click: Option<(Instant, f64, f64)>,
    live_modifiers: EarthModifiers,
    held_keys: HashMap<KeyCode, Instant>,
    last_emitted_cam: Option<EarthCameraState>,
}

impl EarthViewState {
    fn new(
        input: &EarthInput,
        source: Box<dyn TileSource + Send>,
        orbit: EarthOrbitController,
    ) -> Self {
        Self::new_with_plugins(input, source, orbit, EarthPluginRegistry::standard())
    }

    fn new_with_plugins(
        input: &EarthInput,
        source: Box<dyn TileSource + Send>,
        mut orbit: EarthOrbitController,
        plugins: EarthPluginRegistry,
    ) -> Self {
        let tile_ready = Arc::new(AtomicBool::new(false));
        let tile_flag = Arc::clone(&tile_ready);
        let tile_wakeup: Arc<dyn Fn() + Send + Sync> = Arc::new(move || {
            tile_flag.store(true, Ordering::Release);
        });

        let label_ready = Arc::new(AtomicBool::new(false));
        let label_flag = Arc::clone(&label_ready);
        let label_wakeup: Arc<dyn Fn() + Send + Sync> = Arc::new(move || {
            label_flag.store(true, Ordering::Release);
        });

        orbit.set_viewport_size(input.viewport.width as f64, input.viewport.height as f64);
        orbit.set_is_2d_hint(input.view_mode == EarthViewMode::Mode2D);

        let mut effective_input = input.clone();
        if effective_input.low_level_surface_override.is_none()
            && effective_input.surface_mode == crate::scene::SurfaceMode::FlatPatches
        {
            effective_input.surface_mode = crate::scene::SurfaceMode::QuadtreeLod;
        }
        if effective_input.surface_mode == crate::scene::SurfaceMode::QuadtreeLod
            && effective_input.tile_grid <= 5
        {
            effective_input.tile_grid = 17;
        }
        let viewport =
            EarthViewport::with_plugins(&effective_input, source, Some(tile_wakeup), plugins);
        let builtin_labels = builtin_world_labels();
        let live_labels = LiveMapLabelStore::new(Some(label_wakeup));

        let mut st = Self {
            viewport,
            orbit,
            controls: EarthControlsConfig::default(),
            builtin_labels,
            live_labels,
            is_grabbing: Signal::new(false),
            ui_rev: Signal::new(0),
            overlay_plugins: Vec::new(),
            overlay_plugin_rev: 0,
            cached_labels: Vec::new(),
            cached_nearby_sector: None,
            last_label_sig: 0,
            last_label_refresh: Instant::now() - Duration::from_secs(1),
            last_labels_enabled: effective_input.layers.labels,
            last_camera_sync: None,
            camera_sync_pending: false,
            tile_ready,
            label_ready,
            uncapped: false,
            last_cursor_px: None,
            press_candidate: None,
            last_left_click: None,
            live_modifiers: EarthModifiers::default(),
            held_keys: HashMap::new(),
            last_emitted_cam: None,
        };
        st.sync_camera();
        st.last_camera_sync = None;
        let (labels, sig) = st.project_and_hash_labels();
        st.cached_labels = labels;
        st.last_label_sig = sig;
        st
    }

    #[inline]
    pub fn camera_state(&self) -> EarthCameraState {
        EarthCameraState {
            lat_deg: self.orbit.lat_deg,
            lon_deg: self.orbit.lon_deg,
            range_m: self.orbit.range_m,
            heading_deg: self.orbit.heading_deg,
            tilt_deg: self.orbit.tilt_deg,
            view_mode: self.viewport.view_mode(),
            zoom_2d: self.orbit.zoom_2d(),
        }
    }

    fn update_modifier_from_dom_key(&mut self, dom_key: &str, pressed: bool) {
        crate::events::update_modifiers_from_dom_key(&mut self.live_modifiers, dom_key, pressed);
    }

    /// Synchronizes `viewport` camera uniforms and worker LOD target from `orbit` with strict
    /// terrain elevation collision clamping.
    pub fn sync_camera(&mut self) {
        self.camera_sync_pending = false;
        self.last_camera_sync = Some(Instant::now());
        let is_2d = self.viewport.view_mode() == EarthViewMode::Mode2D;
        let vw = self.orbit.viewport_w;
        let vh = self.orbit.viewport_h;
        let (cam, up_opt) = self
            .orbit
            .to_camera_and_up_clamped(is_2d, |lat, lon| self.viewport.sample_ground_elevation_m(lat, lon));
        self.viewport.update_camera_with_up(&cam, up_opt, vw, vh);
    }

    fn sync_camera_throttled(&mut self) {
        let now = Instant::now();
        if self
            .last_camera_sync
            .is_some_and(|t| now.duration_since(t) < Duration::from_millis(5))
        {
            self.camera_sync_pending = true;
        } else {
            self.sync_camera();
        }
    }

    /// Bumps the reactive UI revision signal so label/HUD overlays re-render on the next paint.
    pub fn notify_ui_changed(&self) {
        self.ui_rev.update(|v| *v = v.wrapping_add(1));
    }

    /// Switches view mode (`Mode3D` <-> `Mode2D`) with smooth morph transition.
    pub fn set_view_mode(&mut self, mode: EarthViewMode) {
        self.viewport.set_view_mode(mode);
        self.orbit.set_is_2d_hint(mode == EarthViewMode::Mode2D);
        self.sync_camera();
        self.notify_ui_changed();
    }

    /// Switches view mode (`Mode3D` <-> `Mode2D`) immediately without animation.
    pub fn set_view_mode_immediate(&mut self, mode: EarthViewMode) {
        self.viewport.set_view_mode_immediate(mode);
        self.orbit.set_is_2d_hint(mode == EarthViewMode::Mode2D);
        self.sync_camera();
        let (labels, sig) = self.project_and_hash_labels();
        self.cached_labels = labels;
        self.last_label_sig = sig;
        self.notify_ui_changed();
    }

    /// Jumps camera target immediately to a geographic location & framing.
    pub fn jump_to(
        &mut self,
        lat_deg: f64,
        lon_deg: f64,
        range_m: f64,
        heading_deg: f64,
        tilt_deg: f64,
    ) {
        self.orbit
            .jump_to(lat_deg, lon_deg, range_m, heading_deg, tilt_deg);
        self.sync_camera();
        let (labels, sig) = self.project_and_hash_labels();
        self.cached_labels = labels;
        self.last_label_sig = sig;
        self.notify_ui_changed();
    }

    /// Initiates a smooth three-stage global transit: pulls all the way back to a whole-world
    /// overview, orbits/pans to `(lat_deg, lon_deg)`, and zooms in to `altitude_m` (defaulting to
    /// [`crate::controller::DEFAULT_FLY_TO_ALTITUDE_M`] = `75 m`, or the equivalent 2D zoom value
    /// in 2D map mode).
    pub fn fly_to(&mut self, lat_deg: f64, lon_deg: f64, altitude_m: Option<f64>) {
        self.orbit
            .set_is_2d_hint(self.viewport.view_mode() == EarthViewMode::Mode2D);
        self.orbit.fly_to(lat_deg, lon_deg, altitude_m);
        self.notify_ui_changed();
    }

    fn has_active_overlay_plugins(&self) -> bool {
        self.overlay_plugins.iter().any(|p| p.is_enabled())
    }

    /// Registers or replaces a custom UI overlay plugin on this Earth view state.
    pub fn register_overlay_plugin<P: EarthOverlayPlugin>(&mut self, plugin: P) {
        let id = plugin.id();
        let boxed: Box<dyn EarthOverlayPlugin> = Box::new(plugin);
        if let Some(pos) = self.overlay_plugins.iter().position(|p| p.id() == id) {
            self.overlay_plugins[pos] = boxed;
        } else {
            self.overlay_plugins.push(boxed);
        }
        self.overlay_plugin_rev = self.overlay_plugin_rev.wrapping_add(1);
        self.notify_ui_changed();
    }

    /// Removes a UI overlay plugin by `id`, returning `true` if removed.
    pub fn remove_overlay_plugin(&mut self, id: &str) -> bool {
        let before = self.overlay_plugins.len();
        self.overlay_plugins.retain(|p| p.id() != id);
        let removed = self.overlay_plugins.len() != before;
        if removed {
            self.overlay_plugin_rev = self.overlay_plugin_rev.wrapping_add(1);
            self.notify_ui_changed();
        }
        removed
    }

    /// Enables or disables a UI overlay plugin by `id`.
    pub fn set_overlay_plugin_enabled(&mut self, id: &str, enabled: bool) -> bool {
        if let Some(p) = self.overlay_plugins.iter_mut().find(|p| p.id() == id) {
            p.set_enabled(enabled);
            self.overlay_plugin_rev = self.overlay_plugin_rev.wrapping_add(1);
            self.notify_ui_changed();
            return true;
        }
        false
    }

    /// Projects visible labels and hashes ONLY their screen-visible half-pixel positions & alpha,
    /// so sub-pixel zoom/momentum tails and zero-label street views trigger zero DOM/Cairo rebuilds.
    fn project_and_hash_labels(&mut self) -> (Vec<ProjectedEarthLabel>, u64) {
        let labels_on = self.viewport.input().layers.labels;
        let plugins_on = self.has_active_overlay_plugins();
        let mut projected = if !labels_on {
            Vec::new()
        } else if self.orbit.range_m < 250_000.0 {
            let sector = (
                (self.orbit.lat_deg * 2.0).floor() as i32,
                (self.orbit.lon_deg * 2.0).floor() as i32,
            );
            let store_rev = self.live_labels.revision();
            let need_reload = match &self.cached_nearby_sector {
                Some((s, r, _)) => *s != sector || *r != store_rev,
                None => true,
            };
            if need_reload {
                let nearby = self
                    .live_labels
                    .collect_nearby(self.orbit.lat_deg, self.orbit.lon_deg);
                self.cached_nearby_sector = Some((sector, store_rev, nearby));
            }
            let nearby_slice = self
                .cached_nearby_sector
                .as_ref()
                .map(|(_, _, v)| v.as_slice())
                .unwrap_or(&[]);
            let inp = self.viewport.input();
            if nearby_slice.is_empty() {
                project_earth_labels(inp, &self.builtin_labels, 36)
            } else {
                let mut combined = Vec::with_capacity(self.builtin_labels.len() + nearby_slice.len());
                combined.extend_from_slice(&self.builtin_labels);
                combined.extend_from_slice(nearby_slice);
                project_earth_labels(inp, &combined, 38)
            }
        } else {
            project_earth_labels(self.viewport.input(), &self.builtin_labels, 36)
        };
        let obj_labels = self.viewport.project_object_labels();
        if !obj_labels.is_empty() {
            projected.extend(obj_labels);
        }

        let mut h = 0xcbf2_9ce4_8422_2325u64;
        let mut mix = |v: u64| {
            h ^= v;
            h = h.wrapping_mul(0x100_0000_01b3);
        };
        mix(labels_on as u64);
        mix(self.overlay_plugin_rev);
        mix(plugins_on as u64);
        mix(projected.len() as u64);
        for lbl in &projected {
            mix(lbl.id);
            for b in lbl.text.as_bytes() {
                mix(*b as u64);
            }
            mix((lbl.screen_x * 2.0).round() as i32 as u64);
            mix((lbl.screen_y * 2.0).round() as i32 as u64);
            mix((lbl.alpha * 40.0).round() as i32 as u64);
        }
        if plugins_on {
            let inp = self.viewport.input();
            mix((self.orbit.lat_deg * 1e5).round() as i64 as u64);
            mix((self.orbit.lon_deg * 1e5).round() as i64 as u64);
            mix((self.orbit.heading_deg * 100.0).round() as i64 as u64);
            mix((self.orbit.tilt_deg * 100.0).round() as i64 as u64);
            mix((self.orbit.range_m.ln() * 2_000.0).round() as u64);
            mix((inp.view_morph * 1_000.0).round() as u64);
        }
        (projected, h)
    }

    #[cfg(test)]
    fn compute_projected_labels(&self) -> Vec<ProjectedEarthLabel> {
        let inp = self.viewport.input();
        let mut out = if !inp.layers.labels {
            Vec::new()
        } else {
            let nearby = if self.orbit.range_m < 250_000.0 {
                self.live_labels
                    .collect_nearby(self.orbit.lat_deg, self.orbit.lon_deg)
            } else {
                Vec::new()
            };
            if nearby.is_empty() {
                project_earth_labels(inp, &self.builtin_labels, 36)
            } else {
                let mut combined = Vec::with_capacity(self.builtin_labels.len() + nearby.len());
                combined.extend_from_slice(&self.builtin_labels);
                combined.extend(nearby);
                project_earth_labels(inp, &combined, 38)
            }
        };
        out.extend(self.viewport.project_object_labels());
        out
    }

    fn compute_plugin_overlay_nodes(&self) -> Vec<strata_dom::NodeBlueprint> {
        if !self.has_active_overlay_plugins() {
            return Vec::new();
        }
        let inp = self.viewport.input();
        let projector = EarthScreenProjector::from_input(inp);
        let ctx = EarthOverlayContext {
            input: inp,
            viewport_width: inp.viewport.width.max(1.0),
            viewport_height: inp.viewport.height.max(1.0),
            altitude_m: inp.camera.altitude_m(),
            focus_lat_deg: inp.camera.focus.lat_deg,
            focus_lon_deg: inp.camera.focus.lon_deg,
            projector,
        };
        self.overlay_plugins
            .iter()
            .filter(|p| p.is_enabled())
            .filter_map(|p| p.render(&ctx).map(|v| v.into_node_blueprint()))
            .collect()
    }

    /// Advances smooth zoom/inertia damping, transitions, ocean/weather/satellite animations,
    /// and label/plugin overlay projection updates for one frame tick.
    pub fn step_frame(&mut self, dt_s: f64) -> FrameOutcome {
        let _tile_dirty = self.tile_ready.swap(false, Ordering::AcqRel);
        let label_dirty = self.label_ready.swap(false, Ordering::AcqRel);
        let trans_dirty = self.viewport.step_transition(dt_s);
        let _weather_dirty = self
            .viewport
            .step_dynamic_overlays(dt_s, !self.orbit.is_dragging());

        let is_2d = self.viewport.view_mode() == EarthViewMode::Mode2D;
        let vp_ref = &self.viewport;
        let orbit_dirty = self
            .orbit
            .step_damping(dt_s, is_2d, |lat, lon| vp_ref.sample_ground_elevation_m(lat, lon));

        let camera_moving = self.camera_sync_pending || orbit_dirty || trans_dirty;
        if camera_moving {
            self.sync_camera();
        }
        let _idle_refined = self.viewport.step_idle_refinement(
            dt_s,
            camera_moving || self.orbit.is_dragging() || self.orbit.is_flying(),
        );

        if self.viewport.labels_enabled()
            && self.orbit.range_m < 180_000.0
            && std::env::var_os("STRATA_EARTH_OFFLINE").is_none()
        {
            self.live_labels.ensure_local_sector(
                self.orbit.lat_deg,
                self.orbit.lon_deg,
                self.orbit.range_m,
            );
        }

        let labels_enabled = self.viewport.labels_enabled();
        let toggled_labels = labels_enabled != self.last_labels_enabled;
        self.last_labels_enabled = labels_enabled;
        let overlay_plugins_active = self.has_active_overlay_plugins();
        let obj_labels_active = self.viewport.has_labeled_objects();

        if !toggled_labels
            && !label_dirty
            && !labels_enabled
            && !overlay_plugins_active
            && !obj_labels_active
        {
            return FrameOutcome::Unchanged;
        }

        let (projected, new_sig) = self.project_and_hash_labels();
        let sig_changed = new_sig != self.last_label_sig;
        if sig_changed {
            self.cached_labels = projected;
            self.last_label_sig = new_sig;
        }

        if toggled_labels || label_dirty || sig_changed {
            self.last_label_refresh = Instant::now();
            self.notify_ui_changed();
            FrameOutcome::Dirty
        } else {
            FrameOutcome::Unchanged
        }
    }
}

/// Shareable, cloneable handle to an interactive embeddable Strata Earth component.
#[derive(Clone)]
pub struct EarthView {
    state: Rc<RefCell<EarthViewState>>,
    listeners: Rc<RefCell<EarthEventListeners>>,
}

impl EarthView {
    /// Creates a new interactive Earth view component backed by `input` and `source`.
    pub fn new(input: &EarthInput, source: Box<dyn TileSource + Send>) -> Self {
        Self::with_orbit(input, source, EarthOrbitController::new())
    }

    /// Creates a new Earth view initialized to a specific [`EarthOrbitController`] camera pose.
    pub fn with_orbit(
        input: &EarthInput,
        source: Box<dyn TileSource + Send>,
        orbit: EarthOrbitController,
    ) -> Self {
        Self {
            state: Rc::new(RefCell::new(EarthViewState::new(input, source, orbit))),
            listeners: Rc::new(RefCell::new(EarthEventListeners::default())),
        }
    }

    /// Creates a new Earth view initialized with a specific [`EarthPluginRegistry`].
    pub fn with_plugins(
        input: &EarthInput,
        source: Box<dyn TileSource + Send>,
        orbit: EarthOrbitController,
        plugins: EarthPluginRegistry,
    ) -> Self {
        Self {
            state: Rc::new(RefCell::new(EarthViewState::new_with_plugins(
                input, source, orbit, plugins,
            ))),
            listeners: Rc::new(RefCell::new(EarthEventListeners::default())),
        }
    }

    /// Removes default built-in plugins (`Celestial`, `CityLights`, `Weather`, `Satellites`, `OceanWaves`)
    /// so hosts can compose custom plugin stacks from scratch.
    pub fn without_default_plugins(self) -> Self {
        self.state.borrow_mut().viewport.without_default_plugins();
        self
    }

    /// Registers (or replaces) a world-pipeline plugin via builder style.
    pub fn with_plugin<P: EarthPlugin>(self, plugin: P) -> Self {
        self.register_plugin(plugin);
        self
    }

    /// Registers (or replaces) a custom 2D/UI overlay plugin via builder style.
    pub fn with_overlay_plugin<P: EarthOverlayPlugin>(self, plugin: P) -> Self {
        self.register_overlay_plugin(plugin);
        self
    }

    /// Configures optional animated wavy ocean renders on non-land water surfaces via builder style.
    pub fn with_ocean_waves(self, config: OceanWavesConfig) -> Self {
        self.state
            .borrow_mut()
            .viewport
            .set_ocean_waves_config(config);
        self
    }

    /// Configures GPU-instanced 2D camera-local precipitation and atmospheric physics via builder style.
    pub fn with_precipitation(self, config: PrecipitationConfig) -> Self {
        self.state
            .borrow_mut()
            .viewport
            .set_precipitation_config(config);
        self
    }

    /// Configures whether the underlying `<Canvas3D>` runs uncapped animation ticks.
    pub fn with_uncapped(self, uncapped: bool) -> Self {
        self.state.borrow_mut().uncapped = uncapped;
        self
    }

    /// Configures built-in mouse/wheel navigation permissions via builder style.
    pub fn with_controls_config(self, config: EarthControlsConfig) -> Self {
        self.set_controls_config(config);
        self
    }

    pub fn set_controls_config(&self, config: EarthControlsConfig) {
        self.state.borrow_mut().controls = config;
    }

    pub fn controls_config(&self) -> EarthControlsConfig {
        self.state.borrow().controls
    }

    // -------------------------------------------------------------------------
    // External Event Listeners (Builder & Runtime)
    // -------------------------------------------------------------------------

    /// Registers a primary-button click listener. Fires strictly on mouse release when
    /// press-to-release pointer travel `<= 5.0 px`, so orbiting or panning never triggers false clicks.
    pub fn on_click(&self, handler: impl Fn(&EarthPointerEvent) + 'static) {
        self.listeners.borrow_mut().on_click = Some(Rc::new(handler));
    }

    pub fn with_on_click(self, handler: impl Fn(&EarthPointerEvent) + 'static) -> Self {
        self.on_click(handler);
        self
    }

    /// Registers a primary-button double-click listener.
    pub fn on_double_click(&self, handler: impl Fn(&EarthPointerEvent) + 'static) {
        self.listeners.borrow_mut().on_double_click = Some(Rc::new(handler));
    }

    pub fn with_on_double_click(self, handler: impl Fn(&EarthPointerEvent) + 'static) -> Self {
        self.on_double_click(handler);
        self
    }

    /// Registers a secondary-button (right-click) listener (`<= 5.0 px` travel).
    pub fn on_context_menu(&self, handler: impl Fn(&EarthPointerEvent) + 'static) {
        self.listeners.borrow_mut().on_context_menu = Some(Rc::new(handler));
    }

    pub fn with_on_context_menu(self, handler: impl Fn(&EarthPointerEvent) + 'static) -> Self {
        self.on_context_menu(handler);
        self
    }

    pub fn on_pointer_down(&self, handler: impl Fn(&EarthPointerEvent) + 'static) {
        self.listeners.borrow_mut().on_pointer_down = Some(Rc::new(handler));
    }

    pub fn with_on_pointer_down(self, handler: impl Fn(&EarthPointerEvent) + 'static) -> Self {
        self.on_pointer_down(handler);
        self
    }

    pub fn on_pointer_move(&self, handler: impl Fn(&EarthPointerEvent) + 'static) {
        self.listeners.borrow_mut().on_pointer_move = Some(Rc::new(handler));
    }

    pub fn with_on_pointer_move(self, handler: impl Fn(&EarthPointerEvent) + 'static) -> Self {
        self.on_pointer_move(handler);
        self
    }

    pub fn on_pointer_up(&self, handler: impl Fn(&EarthPointerEvent) + 'static) {
        self.listeners.borrow_mut().on_pointer_up = Some(Rc::new(handler));
    }

    pub fn with_on_pointer_up(self, handler: impl Fn(&EarthPointerEvent) + 'static) -> Self {
        self.on_pointer_up(handler);
        self
    }

    pub fn on_wheel(&self, handler: impl Fn(&EarthWheelEvent) + 'static) {
        self.listeners.borrow_mut().on_wheel = Some(Rc::new(handler));
    }

    pub fn with_on_wheel(self, handler: impl Fn(&EarthWheelEvent) + 'static) -> Self {
        self.on_wheel(handler);
        self
    }

    pub fn on_key_down(&self, handler: impl Fn(&EarthKeyEvent) + 'static) {
        self.listeners.borrow_mut().on_key_down = Some(Rc::new(handler));
    }

    pub fn with_on_key_down(self, handler: impl Fn(&EarthKeyEvent) + 'static) -> Self {
        self.on_key_down(handler);
        self
    }

    pub fn on_key_up(&self, handler: impl Fn(&EarthKeyEvent) + 'static) {
        self.listeners.borrow_mut().on_key_up = Some(Rc::new(handler));
    }

    pub fn with_on_key_up(self, handler: impl Fn(&EarthKeyEvent) + 'static) -> Self {
        self.on_key_up(handler);
        self
    }

    pub fn on_camera_change(&self, handler: impl Fn(&EarthCameraState) + 'static) {
        self.listeners.borrow_mut().on_camera_change = Some(Rc::new(handler));
    }

    pub fn with_on_camera_change(self, handler: impl Fn(&EarthCameraState) + 'static) -> Self {
        self.on_camera_change(handler);
        self
    }

    /// Forwards a window-level [`PlatformEvent`] (such as `KeyDown`, `KeyUp`, or `WindowFocused(false)`)
    /// into [`EarthView`]'s external key listeners and live modifier tracker when no text input is focused.
    ///
    /// Automatically deduplicates against DOM `onkeydown`/`onkeyup` and synthesizes clean `on_key_up`
    /// releases for any held keys if the window loses focus (`WindowFocused(false)`).
    pub fn dispatch_platform_event(&self, event: &PlatformEvent) {
        match event {
            PlatformEvent::KeyDown { key, modifiers } => {
                let now = Instant::now();
                let mods = EarthModifiers::from_platform(*modifiers);
                let (skip_dup, repeat) = {
                    let mut st = self.state.borrow_mut();
                    st.live_modifiers = mods;
                    if *key == KeyCode::Unknown {
                        (false, false)
                    } else {
                        match st.held_keys.get(key).copied() {
                            Some(prev) if now.duration_since(prev) < Duration::from_millis(8) => {
                                (true, false)
                            }
                            Some(_) => {
                                st.held_keys.insert(*key, now);
                                (false, true)
                            }
                            None => {
                                st.held_keys.insert(*key, now);
                                (false, false)
                            }
                        }
                    }
                };
                if !skip_dup {
                    let cb = self.listeners.borrow().on_key_down.clone();
                    if let Some(cb) = cb {
                        let ev = EarthKeyEvent::new(*key, format!("{key:?}"), mods, repeat);
                        cb(&ev);
                    }
                }
            }
            PlatformEvent::KeyUp { key, modifiers } => {
                let mods = EarthModifiers::from_platform(*modifiers);
                let was_held = {
                    let mut st = self.state.borrow_mut();
                    st.live_modifiers = mods;
                    if *key == KeyCode::Unknown {
                        true
                    } else {
                        st.held_keys.remove(key).is_some()
                    }
                };
                if was_held {
                    let cb = self.listeners.borrow().on_key_up.clone();
                    if let Some(cb) = cb {
                        let ev = EarthKeyEvent::new(*key, format!("{key:?}"), mods, false);
                        cb(&ev);
                    }
                }
            }
            PlatformEvent::WindowFocused(false) => {
                let released: Vec<KeyCode> = {
                    let mut st = self.state.borrow_mut();
                    st.live_modifiers = EarthModifiers::default();
                    st.held_keys.drain().map(|(k, _)| k).collect()
                };
                if !released.is_empty() {
                    let cb = self.listeners.borrow().on_key_up.clone();
                    if let Some(cb) = cb {
                        for key in released {
                            let ev = EarthKeyEvent::new(
                                key,
                                format!("{key:?}"),
                                EarthModifiers::default(),
                                false,
                            );
                            cb(&ev);
                        }
                    }
                }
            }
            _ => {}
        }
    }

    fn emit_camera_change_if_needed(&self) {
        let cb = self.listeners.borrow().on_camera_change.clone();
        let Some(cb) = cb else {
            return;
        };
        let state_opt = {
            let mut st = self.state.borrow_mut();
            let cur = st.camera_state();
            if st.last_emitted_cam == Some(cur) {
                None
            } else {
                st.last_emitted_cam = Some(cur);
                Some(cur)
            }
        };
        if let Some(cam) = state_opt {
            cb(&cam);
        }
    }

    // -------------------------------------------------------------------------
    // Native Geodetic World & Terrain/Object Raycasting APIs
    // -------------------------------------------------------------------------

    /// Raycasts component-local pixel `(x, y)` against pickable [`EarthWorld`] 3D objects and
    /// the planetary surface (terrain heightmap when `elevation_enabled`, sea level when off).
    pub fn raycast(&self, x: f64, y: f64) -> Option<EarthRayHit> {
        self.state.borrow().viewport.raycast(x, y)
    }

    /// Raycasts component-local pixel `(x, y)` strictly against the planetary surface.
    pub fn raycast_terrain(&self, x: f64, y: f64) -> Option<EarthRayHit> {
        self.state.borrow().viewport.raycast_terrain(x, y)
    }

    /// Borrows the geodetic [`EarthWorld`] read-only.
    pub fn world<R>(&self, f: impl FnOnce(&EarthWorld) -> R) -> R {
        self.state.borrow().viewport.world(f)
    }

    /// Mutates the geodetic [`EarthWorld`] and immediately publishes updated GPU instances.
    pub fn world_mut<R>(&self, f: impl FnOnce(&mut EarthWorld) -> R) -> R {
        let mut st = self.state.borrow_mut();
        let had_labeled = st.viewport.has_labeled_objects();
        let out = st.viewport.world_mut(f);
        if had_labeled || st.viewport.has_labeled_objects() {
            let (projected, new_sig) = st.project_and_hash_labels();
            if new_sig != st.last_label_sig {
                st.cached_labels = projected;
                st.last_label_sig = new_sig;
                st.notify_ui_changed();
            }
        }
        out
    }

    /// Spawns a geodetic [`EarthObject`] into the world and returns its [`EarthObjectId`].
    pub fn spawn_object(&self, object: EarthObject) -> EarthObjectId {
        self.world_mut(|w| w.spawn(object))
    }

    /// Validated convenience wrapper spawning a glTF 2.0 binary (`.glb`) object and returning a `Result`.
    pub fn spawn_glb(
        &self,
        position: crate::geodesy::GeoPosition,
        bytes: &[u8],
        unit_scale_m: f64,
        anchor: crate::world::ObjectAnchor,
    ) -> Result<EarthObjectId, String> {
        self.world_mut(|w| w.spawn_glb(position, bytes, unit_scale_m, anchor))
    }

    /// Updates a [`crate::world::EarthGeometry::Polyline`] object's vertices in place without leaking CPU or GPU meshes.
    pub fn set_polyline_points(
        &self,
        id: EarthObjectId,
        points: Vec<crate::geodesy::GeoPosition>,
    ) -> bool {
        self.world_mut(|w| w.set_polyline_points(id, points))
    }

    /// Updates a [`crate::world::EarthGeometry::Polyline`] object's tube radius in place without leaking CPU or GPU meshes.
    pub fn set_polyline_radius(&self, id: EarthObjectId, radius_m: f64) -> bool {
        self.world_mut(|w| w.set_polyline_radius(id, radius_m))
    }

    /// Replaces `id`'s [`crate::world::EarthGeometry`] in place and evicts any unreferenced mesh buffers.
    pub fn replace_geometry(
        &self,
        id: EarthObjectId,
        geometry: crate::world::EarthGeometry,
    ) -> bool {
        self.world_mut(|w| w.replace_geometry(id, geometry))
    }

    /// Removes an [`EarthObject`] by ID, returning `true` if it existed.
    pub fn remove_object(&self, id: EarthObjectId) -> bool {
        self.world_mut(|w| w.remove(id))
    }

    /// Removes all [`EarthObject`]s from the world.
    pub fn clear_objects(&self) {
        self.world_mut(|w| w.clear());
    }

    /// Returns the number of [`EarthObject`]s currently in the world.
    pub fn object_count(&self) -> usize {
        self.world(|w| w.len())
    }

    /// Evaluates the live world transform (including resolved `Terrain` vs `SeaLevel` altitude)
    /// for `id` under the current heightmap and view-morph state.
    pub fn resolve_object_transform(&self, id: EarthObjectId) -> Option<ResolvedObjectTransform> {
        self.state.borrow().viewport.resolve_object_transform(id)
    }

    /// Enables or disables heightmap (DEM) relief and immediately updates all
    /// [`crate::geodesy::AltitudeMode::Terrain`] objects (`DEM + alt` when `true`, `0.0 + alt` when `false`).
    pub fn set_elevation_enabled(&self, enabled: bool) {
        let mut st = self.state.borrow_mut();
        st.viewport.set_elevation_enabled(enabled);
        st.sync_camera();
    }

    pub fn elevation_enabled(&self) -> bool {
        self.state.borrow().viewport.elevation_enabled()
    }

    /// Samples terrain elevation in metres above the WGS84 ellipsoid at `(lat_deg, lon_deg)`.
    pub fn sample_ground_elevation_m(&self, lat_deg: f64, lon_deg: f64) -> f64 {
        self.state
            .borrow()
            .viewport
            .sample_ground_elevation_m(lat_deg, lon_deg)
    }

    /// Toggles geographic map labels in O(1) without rebuilding terrain geometry.
    pub fn set_labels_enabled(&self, enabled: bool) {
        let mut st = self.state.borrow_mut();
        st.viewport.set_labels_enabled(enabled);
        let (projected, new_sig) = st.project_and_hash_labels();
        st.cached_labels = projected;
        st.last_label_sig = new_sig;
        st.last_labels_enabled = enabled;
        st.notify_ui_changed();
    }

    pub fn labels_enabled(&self) -> bool {
        self.state.borrow().viewport.labels_enabled()
    }

    /// Toggles extruded 3D building rendering from the configured [`crate::scene::BuildingSource`].
    pub fn set_buildings_enabled(&self, enabled: bool) {
        let mut st = self.state.borrow_mut();
        st.viewport.set_buildings_enabled(enabled);
        st.notify_ui_changed();
    }

    pub fn buildings_enabled(&self) -> bool {
        self.state.borrow().viewport.buildings_enabled()
    }

    /// Installs or clears a [`crate::scene::BuildingSource`] on the background tile worker.
    pub fn set_building_source(
        &self,
        source: Option<Box<dyn crate::scene::BuildingSource + Send>>,
    ) {
        let mut st = self.state.borrow_mut();
        st.viewport.set_building_source(source);
        st.notify_ui_changed();
    }

    /// Toggles the geodetic meridian/parallel graticule overlay in O(1).
    pub fn set_graticule_enabled(&self, enabled: bool) {
        let mut st = self.state.borrow_mut();
        st.viewport.set_graticule_enabled(enabled);
        st.notify_ui_changed();
    }

    pub fn graticule_enabled(&self) -> bool {
        self.state.borrow().viewport.graticule_enabled()
    }

    /// Returns a snapshot of the active [`SpaceEnvironmentConfig`].
    pub fn space_config(&self) -> crate::scene::SpaceEnvironmentConfig {
        self.state.borrow().viewport.space_config().clone()
    }

    /// Updates planetary lighting, starfield, Sun/Moon, satellites, and city-light configuration in O(1).
    pub fn set_space_config(&self, config: crate::scene::SpaceEnvironmentConfig) {
        let mut st = self.state.borrow_mut();
        st.viewport.set_space_config(config);
        st.notify_ui_changed();
    }

    /// Returns a snapshot of the active [`crate::gltf::WeatherConfig`].
    pub fn weather_config(&self) -> crate::gltf::WeatherConfig {
        self.state.borrow().viewport.weather_config().clone()
    }

    /// Updates global & local cloud-shell configuration in O(1) without terrain rebuilds.
    pub fn set_weather_config(&self, config: crate::gltf::WeatherConfig) {
        let mut st = self.state.borrow_mut();
        st.viewport.set_weather_config(config);
        st.notify_ui_changed();
    }

    /// Returns a snapshot of the active [`TileProviderConfig`].
    pub fn tile_provider(&self) -> crate::tiles::TileProviderConfig {
        self.state.borrow().viewport.tile_provider().clone()
    }

    /// Switches the active [`TileProviderConfig`] at runtime.
    pub fn set_tile_provider(&self, config: crate::tiles::TileProviderConfig) {
        let mut st = self.state.borrow_mut();
        st.viewport.set_tile_provider(config);
        st.notify_ui_changed();
    }

    /// Switches the active map/satellite preset cleanly.
    pub fn set_tile_preset(&self, preset: crate::tiles::TileProviderPreset) {
        let mut st = self.state.borrow_mut();
        st.viewport.set_tile_preset(preset);
        st.notify_ui_changed();
    }

    /// Replaces the underlying [`TileSource`] cleanly and rebuilds terrain textures & meshes.
    pub fn set_tile_source(&self, source: Box<dyn crate::tiles::TileSource + Send>) {
        let mut st = self.state.borrow_mut();
        st.viewport.set_tile_source(source);
        st.notify_ui_changed();
    }

    /// Returns the high-level view mode (`Mode3D` or `Mode2D`).
    pub fn view_mode(&self) -> EarthViewMode {
        self.state.borrow().viewport.view_mode()
    }

    /// Programmatically initiates a smooth animated transition between `Mode3D` and `Mode2D`.
    pub fn set_view_mode(&self, mode: EarthViewMode) {
        let mut st = self.state.borrow_mut();
        st.set_view_mode(mode);
        st.notify_ui_changed();
    }

    /// Programmatically switches between `Mode3D` and `Mode2D` immediately without animation.
    pub fn set_view_mode_immediate(&self, mode: EarthViewMode) {
        let mut st = self.state.borrow_mut();
        st.set_view_mode_immediate(mode);
        st.notify_ui_changed();
    }

    /// Returns the simulation UNIX UTC timestamp configured on this view.
    pub fn sim_time_unix_s(&self) -> f64 {
        self.state.borrow().viewport.input().sim_time_unix_s
    }

    /// Updates the simulation UNIX UTC timestamp in O(1) (solar/lunar/sidereal/satellite uniforms).
    pub fn set_sim_time_unix_s(&self, unix_utc_s: f64) {
        let mut st = self.state.borrow_mut();
        st.viewport.set_sim_time_unix_s(unix_utc_s);
        st.notify_ui_changed();
    }

    /// Returns a snapshot of the live camera state.
    pub fn camera_state(&self) -> EarthCameraState {
        self.state.borrow().camera_state()
    }

    // -------------------------------------------------------------------------
    // Plugin & Camera Control APIs
    // -------------------------------------------------------------------------

    /// Registers (or replaces) a world-pipeline [`EarthPlugin`] on this component.
    pub fn register_plugin<P: EarthPlugin>(&self, plugin: P) {
        self.state.borrow_mut().viewport.register_plugin(plugin);
    }

    /// Removes a world-pipeline plugin by `id`, returning `true` if removed.
    pub fn remove_plugin(&self, id: &str) -> bool {
        self.state.borrow_mut().viewport.remove_plugin(id)
    }

    /// Enables or disables a world-pipeline plugin by `id`.
    pub fn set_plugin_enabled(&self, id: &str, enabled: bool) -> bool {
        self.state
            .borrow_mut()
            .viewport
            .set_plugin_enabled(id, enabled)
    }

    #[inline]
    pub fn is_plugin_enabled(&self, id: &str) -> Option<bool> {
        self.state.borrow().viewport.is_plugin_enabled(id)
    }

    /// Returns a snapshot of the active [`OceanWavesConfig`].
    pub fn ocean_waves_config(&self) -> OceanWavesConfig {
        self.state.borrow().viewport.ocean_waves_config().clone()
    }

    /// Updates GPU-native physical ocean waves and optional coastal surf settings in O(1).
    pub fn set_ocean_waves_config(&self, config: OceanWavesConfig) {
        let mut st = self.state.borrow_mut();
        st.viewport.set_ocean_waves_config(config);
        st.notify_ui_changed();
    }

    /// Enables or disables animated wavy ocean renders on non-land water areas.
    pub fn set_ocean_waves_enabled(&self, enabled: bool) {
        let mut st = self.state.borrow_mut();
        st.viewport.set_ocean_waves_enabled(enabled);
        st.notify_ui_changed();
    }

    /// Returns a snapshot of the active [`LocalAtmosphereConfig`].
    pub fn local_atmosphere_config(&self) -> LocalAtmosphereConfig {
        self.state
            .borrow()
            .viewport
            .local_atmosphere_config()
            .clone()
    }

    /// Updates optional realistic 2.5D sky, visible Sun, horizon-synced night stars, terrain
    /// lighting, and real-time GPU mountain shadows in O(1).
    pub fn set_local_atmosphere_config(&self, config: LocalAtmosphereConfig) {
        let mut st = self.state.borrow_mut();
        st.viewport.set_local_atmosphere_config(config);
        st.notify_ui_changed();
    }

    /// Returns a snapshot of the active unified [`BaseWeatherConfig`].
    pub fn base_weather_config(&self) -> BaseWeatherConfig {
        self.state.borrow().viewport.base_weather_config().clone()
    }

    /// Updates the unified Base Weather API configuration in O(1), driving precipitation,
    /// cloud storm darkening, 2.5D sky/haze, and ocean wave wind transitions coherently.
    pub fn set_base_weather_config(&self, config: BaseWeatherConfig) {
        let mut st = self.state.borrow_mut();
        st.viewport.set_base_weather_config(config);
        st.notify_ui_changed();
    }

    /// Installs or clears a custom unified [`WeatherProvider`] driving all atmospheric & marine plugins.
    pub fn set_weather_provider(&self, provider: Option<Arc<dyn WeatherProvider>>) {
        let mut st = self.state.borrow_mut();
        st.viewport.set_weather_provider(provider);
        st.notify_ui_changed();
    }

    /// Returns the most recently evaluated unified [`WeatherSnapshot`], if `base_weather.enabled` is active.
    pub fn last_weather_snapshot(&self) -> Option<WeatherSnapshot> {
        self.state.borrow().viewport.last_weather_snapshot()
    }

    /// Returns a snapshot of the active [`PrecipitationConfig`].
    pub fn precipitation_config(&self) -> PrecipitationConfig {
        self.state.borrow().viewport.precipitation_config().clone()
    }

    /// Updates GPU-instanced 2D camera-local precipitation and atmospheric settings in O(1).
    pub fn set_precipitation_config(&self, config: PrecipitationConfig) {
        let mut st = self.state.borrow_mut();
        st.viewport.set_precipitation_config(config);
        st.notify_ui_changed();
    }

    /// Convenience toggle enabling or disabling [`crate::precipitation::PrecipitationPlugin`].
    pub fn set_precipitation_enabled(&self, enabled: bool) {
        let mut st = self.state.borrow_mut();
        st.viewport.set_precipitation_enabled(enabled);
        st.notify_ui_changed();
    }

    /// Configures opt-in idle high-resolution view refinement via builder style (`enabled: false` by default).
    pub fn with_idle_high_res(self, config: IdleHighResConfig) -> Self {
        self.set_idle_high_res_config(config);
        self
    }

    /// Returns the active [`IdleHighResConfig`].
    pub fn idle_high_res_config(&self) -> IdleHighResConfig {
        self.state.borrow().viewport.idle_high_res_config()
    }

    /// Updates opt-in idle high-resolution view refinement settings.
    pub fn set_idle_high_res_config(&self, config: IdleHighResConfig) {
        let mut st = self.state.borrow_mut();
        st.viewport.set_idle_high_res_config(config);
        st.notify_ui_changed();
    }

    /// Convenience toggle enabling or disabling optional idle high-resolution refinement.
    pub fn set_idle_high_res_enabled(&self, enabled: bool) {
        let mut st = self.state.borrow_mut();
        st.viewport.set_idle_high_res_enabled(enabled);
        st.notify_ui_changed();
    }

    /// Returns `true` when idle high-res refinement is enabled and currently active on a stationary view.
    pub fn is_idle_high_res_active(&self) -> bool {
        self.state.borrow().viewport.is_idle_high_res_active()
    }

    /// Installs a custom functional [`PrecipitationEvaluator`] controlling which 2D layers render.
    pub fn set_precipitation_evaluator(&self, evaluator: PrecipitationEvaluator) {
        self.state
            .borrow_mut()
            .viewport
            .set_precipitation_evaluator(evaluator);
    }

    /// Installs or clears a custom [`TemperatureProvider`] (e.g., live sounding fetcher or closure).
    pub fn set_temperature_provider(&self, provider: Option<Arc<dyn TemperatureProvider>>) {
        self.state
            .borrow_mut()
            .viewport
            .set_temperature_provider(provider);
    }

    /// Returns the most recently evaluated [`AtmosphereSample`] at the camera's exact altitude.
    pub fn last_atmosphere_sample(&self) -> Option<AtmosphereSample> {
        self.state.borrow().viewport.last_atmosphere_sample()
    }

    /// Registers or replaces a UI overlay plugin on this [`EarthView`] component.
    pub fn register_overlay_plugin<P: EarthOverlayPlugin>(&self, plugin: P) {
        self.state.borrow_mut().register_overlay_plugin(plugin);
    }

    /// Removes a UI overlay plugin by `id`.
    pub fn remove_overlay_plugin(&self, id: &str) -> bool {
        self.state.borrow_mut().remove_overlay_plugin(id)
    }

    /// Enables or disables a UI overlay plugin by `id`.
    pub fn set_overlay_plugin_enabled(&self, id: &str, enabled: bool) -> bool {
        self.state
            .borrow_mut()
            .set_overlay_plugin_enabled(id, enabled)
    }

    /// Borrows inner state read-only.
    pub fn with_state<R>(&self, f: impl FnOnce(&EarthViewState) -> R) -> R {
        f(&self.state.borrow())
    }

    /// Borrows inner state mutably.
    pub fn with_state_mut<R>(&self, f: impl FnOnce(&mut EarthViewState) -> R) -> R {
        f(&mut self.state.borrow_mut())
    }

    /// Smoothly transits to any `(lat_deg, lon_deg)` point on Earth via a three-stage global
    /// pull-back, orbital/2D slew, and descent to `altitude_m` (defaulting to `75 m` when `None`,
    /// or the equivalent 2D zoom value in 2D map mode).
    pub fn fly_to(&self, lat_deg: f64, lon_deg: f64, altitude_m: Option<f64>) {
        self.state.borrow_mut().fly_to(lat_deg, lon_deg, altitude_m);
        self.emit_camera_change_if_needed();
    }

    /// Immediately jumps camera focus, range, heading, and pitch.
    pub fn jump_to(
        &self,
        lat_deg: f64,
        lon_deg: f64,
        range_m: f64,
        heading_deg: f64,
        tilt_deg: f64,
    ) {
        self.state
            .borrow_mut()
            .jump_to(lat_deg, lon_deg, range_m, heading_deg, tilt_deg);
        self.emit_camera_change_if_needed();
    }

    /// Applies a smooth exponential natural-log zoom step (`delta_ln < 0.0` zooms in;
    /// `delta_ln > 0.0` zooms out).
    pub fn zoom_by_ln(&self, delta_ln: f64) {
        self.state.borrow_mut().orbit.zoom_by_ln(delta_ln);
    }

    /// Applies a smooth exponential natural-log zoom step anchored toward component pixel `(x, y)`.
    pub fn zoom_at_screen_px(&self, delta_ln: f64, x: f64, y: f64) {
        self.state
            .borrow_mut()
            .orbit
            .zoom_at_screen_px(delta_ln, x, y);
    }

    pub fn pan_by_deg(&self, d_lat_deg: f64, d_lon_deg: f64) {
        {
            let mut st = self.state.borrow_mut();
            st.orbit.pan_by_deg(d_lat_deg, d_lon_deg);
            st.sync_camera();
        }
        self.emit_camera_change_if_needed();
    }

    pub fn orbit_by_deg(&self, d_heading_deg: f64, d_tilt_deg: f64) {
        {
            let mut st = self.state.borrow_mut();
            st.orbit.orbit_by_deg(d_heading_deg, d_tilt_deg);
            st.sync_camera();
        }
        self.emit_camera_change_if_needed();
    }

    /// Advances smooth damping, transitions, weather, ocean waves, and label updates by `dt_s` seconds.
    pub fn step_frame(&self, dt_s: f64) -> FrameOutcome {
        let outcome = self.state.borrow_mut().step_frame(dt_s);
        self.emit_camera_change_if_needed();
        outcome
    }

    /// Dispatches a component-local pointer press (`(x, y)` in logical viewport pixels) through
    /// [`Self::on_pointer_down`], click-travel tracking, and built-in camera pan/orbit controls.
    pub fn dispatch_pointer_down(
        &self,
        button: EarthMouseButton,
        x: f64,
        y: f64,
        dom_mods: EarthModifiers,
    ) {
        let (vw, vh, dx, dy, mods) = {
            let mut st = self.state.borrow_mut();
            let (dx, dy) = st
                .last_cursor_px
                .map(|(lx, ly)| (x - lx, y - ly))
                .unwrap_or((0.0, 0.0));
            st.last_cursor_px = Some((x, y));
            let mods = st.live_modifiers.merge(dom_mods);
            st.press_candidate = Some(ActivePressCandidate {
                button,
                start_x: x,
                start_y: y,
                max_travel_px: 0.0,
                camera_active: false,
            });
            (st.orbit.viewport_w, st.orbit.viewport_h, dx, dy, mods)
        };

        let ptr_ev = EarthPointerEvent::new(x, y, dx, dy, vw, vh, button, mods);
        let cb_down = self.listeners.borrow().on_pointer_down.clone();
        if let Some(cb) = cb_down {
            cb(&ptr_ev);
        }

        if !ptr_ev.is_default_camera_prevented() {
            let cam_changed = {
                let mut st = self.state.borrow_mut();
                let allowed = match button {
                    EarthMouseButton::Left => st.controls.pan_enabled,
                    EarthMouseButton::Middle | EarthMouseButton::Right => st.controls.orbit_enabled,
                    EarthMouseButton::Other(_) => false,
                };
                if allowed {
                    let btn_idx = button.controller_button_index();
                    if st.orbit.handle_mouse_down(btn_idx, x, y) {
                        st.sync_camera();
                        if let Some(ref mut cand) = st.press_candidate {
                            cand.camera_active = true;
                        }
                        if !st.is_grabbing.untracked_get() {
                            st.is_grabbing.set(true);
                        }
                        true
                    } else {
                        false
                    }
                } else {
                    false
                }
            };
            if cam_changed {
                self.emit_camera_change_if_needed();
            }
        }
    }

    /// Dispatches a component-local pointer move (`(x, y)` in logical viewport pixels) through
    /// [`Self::on_pointer_move`], click-travel tracking, and built-in camera drag controls.
    pub fn dispatch_pointer_move(&self, x: f64, y: f64, dom_mods: EarthModifiers) {
        let (vw, vh, dx, dy, button, mods) = {
            let mut st = self.state.borrow_mut();
            let (dx, dy) = st
                .last_cursor_px
                .map(|(lx, ly)| (x - lx, y - ly))
                .unwrap_or((0.0, 0.0));
            st.last_cursor_px = Some((x, y));
            let mut set_grabbing = false;
            let btn = if let Some(ref mut cand) = st.press_candidate {
                let dist = (x - cand.start_x).hypot(y - cand.start_y);
                cand.max_travel_px = cand.max_travel_px.max(dist);
                if cand.camera_active && cand.max_travel_px > CLICK_MAX_TRAVEL_PX {
                    set_grabbing = true;
                }
                cand.button
            } else {
                EarthMouseButton::Left
            };
            if set_grabbing && !st.is_grabbing.untracked_get() {
                st.is_grabbing.set(true);
            }
            (
                st.orbit.viewport_w,
                st.orbit.viewport_h,
                dx,
                dy,
                btn,
                st.live_modifiers.merge(dom_mods),
            )
        };

        let ptr_ev = EarthPointerEvent::new(x, y, dx, dy, vw, vh, button, mods);
        let cb_move = self.listeners.borrow().on_pointer_move.clone();
        if let Some(cb) = cb_move {
            cb(&ptr_ev);
        }

        if !ptr_ev.is_default_camera_prevented() {
            let cam_changed = {
                let mut st = self.state.borrow_mut();
                if st.orbit.handle_mouse_move(x, y) {
                    st.sync_camera_throttled();
                    true
                } else {
                    false
                }
            };
            if cam_changed {
                self.emit_camera_change_if_needed();
            }
        }
    }

    /// Dispatches a component-local pointer release (`(x, y)` in logical viewport pixels) through
    /// camera release momentum, [`Self::on_pointer_up`], and travel-discriminated [`Self::on_click`],
    /// [`Self::on_double_click`], and [`Self::on_context_menu`].
    pub fn dispatch_pointer_up(
        &self,
        button: EarthMouseButton,
        x: f64,
        y: f64,
        dom_mods: EarthModifiers,
    ) {
        let now = Instant::now();
        let (vw, vh, dx, dy, mods, is_clean_click, is_double_click, dbl_zoom_enabled, cam_changed) =
            {
                let mut st = self.state.borrow_mut();
                let (dx, dy) = st
                    .last_cursor_px
                    .map(|(lx, ly)| (x - lx, y - ly))
                    .unwrap_or((0.0, 0.0));
                st.last_cursor_px = Some((x, y));
                let mods = st.live_modifiers.merge(dom_mods);

                let btn_idx = button.controller_button_index();
                let cam_changed = if st.orbit.handle_mouse_up(btn_idx) {
                    st.sync_camera();
                    true
                } else {
                    false
                };
                if st.is_grabbing.untracked_get() {
                    st.is_grabbing.set(false);
                }

                let mut clean_click = false;
                if let Some(cand) = st.press_candidate.take() {
                    let end_dist = (x - cand.start_x).hypot(y - cand.start_y);
                    let travel = cand.max_travel_px.max(end_dist);
                    if cand.button == button && travel <= CLICK_MAX_TRAVEL_PX {
                        clean_click = true;
                    }
                }

                let mut dbl_click = false;
                if clean_click && button == EarthMouseButton::Left {
                    if let Some((prev_t, px, py)) = st.last_left_click {
                        if now.duration_since(prev_t) <= Duration::from_millis(350)
                            && (x - px).hypot(y - py) <= 12.0
                        {
                            dbl_click = true;
                            st.last_left_click = None;
                        } else {
                            st.last_left_click = Some((now, x, y));
                        }
                    } else {
                        st.last_left_click = Some((now, x, y));
                    }
                }

                (
                    st.orbit.viewport_w,
                    st.orbit.viewport_h,
                    dx,
                    dy,
                    mods,
                    clean_click,
                    dbl_click,
                    st.controls.double_click_zoom_enabled && st.controls.zoom_enabled,
                    cam_changed,
                )
            };

        if cam_changed {
            self.emit_camera_change_if_needed();
        }

        let ptr_ev = EarthPointerEvent::new(x, y, dx, dy, vw, vh, button, mods);
        let cb_up = self.listeners.borrow().on_pointer_up.clone();
        if let Some(cb) = cb_up {
            cb(&ptr_ev);
        }

        if is_clean_click {
            match button {
                EarthMouseButton::Left => {
                    let click_ev = EarthPointerEvent::new(x, y, 0.0, 0.0, vw, vh, button, mods);
                    let cb_click = self.listeners.borrow().on_click.clone();
                    if let Some(cb) = cb_click {
                        cb(&click_ev);
                    }
                    if is_double_click {
                        let dbl_ev = EarthPointerEvent::new(x, y, 0.0, 0.0, vw, vh, button, mods);
                        let cb_dbl = self.listeners.borrow().on_double_click.clone();
                        if let Some(cb) = cb_dbl {
                            cb(&dbl_ev);
                        }
                        if dbl_zoom_enabled
                            && !click_ev.is_default_camera_prevented()
                            && !dbl_ev.is_default_camera_prevented()
                        {
                            self.zoom_at_screen_px(-std::f64::consts::LN_2, x, y);
                        }
                    }
                }
                EarthMouseButton::Right => {
                    let cb_ctx = self.listeners.borrow().on_context_menu.clone();
                    if let Some(cb) = cb_ctx {
                        let ctx_ev = EarthPointerEvent::new(x, y, 0.0, 0.0, vw, vh, button, mods);
                        cb(&ctx_ev);
                    }
                }
                _ => {}
            }
        }
    }

    /// Dispatches a component-local scroll-wheel tick through [`Self::on_wheel`] and built-in zoom.
    pub fn dispatch_wheel(
        &self,
        delta_x: f64,
        delta_y: f64,
        x: f64,
        y: f64,
        dom_mods: EarthModifiers,
    ) {
        let (vw, vh, mods, zoom_enabled) = {
            let st = self.state.borrow();
            (
                st.orbit.viewport_w,
                st.orbit.viewport_h,
                st.live_modifiers.merge(dom_mods),
                st.controls.zoom_enabled,
            )
        };

        let wheel_ev = EarthWheelEvent::new(x, y, delta_x, delta_y, vw, vh, mods);
        let cb_wheel = self.listeners.borrow().on_wheel.clone();
        if let Some(cb) = cb_wheel {
            cb(&wheel_ev);
        }

        if zoom_enabled && !wheel_ev.is_default_camera_prevented() {
            {
                let mut st = self.state.borrow_mut();
                st.orbit.handle_scroll(delta_y, x, y);
                st.sync_camera_throttled();
            }
            self.emit_camera_change_if_needed();
        }
    }

    /// Returns a self-contained Strata [`View`] (`w-full h-full overflow-hidden`) binding all
    /// pointer drag, right/middle orbit, scroll wheel, click discrimination, keyboard hooks,
    /// pointer capture, and reactive cursor states directly via Strata's component APIs.
    pub fn view(&self) -> View {
        let (retained, grabbing, labels, plugin_overlays, uncapped) = {
            let st = self.state.borrow();
            let grabbing = st.is_grabbing.get();
            let _ = st.ui_rev.get();
            (
                st.viewport.retained_scene(),
                grabbing,
                st.cached_labels.clone(),
                st.compute_plugin_overlay_nodes(),
                st.uncapped,
            )
        };

        let cursor_cls = if grabbing {
            "absolute inset-0 w-full h-full z-0 cursor-grabbing"
        } else {
            "absolute inset-0 w-full h-full z-0 cursor-grab"
        };

        let self_down = self.clone();
        let self_move = self.clone();
        let self_up = self.clone();
        let self_wheel = self.clone();
        let self_resize = self.clone();
        let self_key_down = self.clone();
        let self_key_up = self.clone();

        let canvas = Canvas3D::builder()
            .retained(retained)
            .uncapped(uncapped)
            .class(cursor_cls)
            .onresize(move |ev: strata_dom::Event| {
                if ev.width > 0.0 && ev.height > 0.0 {
                    let changed = {
                        let mut st = self_resize.state.borrow_mut();
                        if (st.orbit.viewport_w - ev.width).abs() > 0.25
                            || (st.orbit.viewport_h - ev.height).abs() > 0.25
                        {
                            st.orbit.set_viewport_size(ev.width, ev.height);
                            st.sync_camera();
                            true
                        } else {
                            false
                        }
                    };
                    if changed {
                        self_resize.emit_camera_change_if_needed();
                    }
                }
            })
            .onmousedown(move |ev: strata_dom::Event| {
                ev.stop_propagation();
                ev.prevent_default();
                ev.set_pointer_capture(ev.pointer_id);
                self_down
                    .state
                    .borrow_mut()
                    .orbit
                    .set_viewport_size(ev.width, ev.height);
                self_down.dispatch_pointer_down(
                    EarthMouseButton::from_dom_button(ev.button),
                    ev.offset_x,
                    ev.offset_y,
                    EarthModifiers {
                        shift: ev.shift_key,
                        ctrl: ev.ctrl_key,
                        alt: ev.alt_key,
                        meta: ev.meta_key,
                    },
                );
            })
            .onmousemove(move |ev: strata_dom::Event| {
                ev.stop_propagation();
                self_move
                    .state
                    .borrow_mut()
                    .orbit
                    .set_viewport_size(ev.width, ev.height);
                self_move.dispatch_pointer_move(
                    ev.offset_x,
                    ev.offset_y,
                    EarthModifiers {
                        shift: ev.shift_key,
                        ctrl: ev.ctrl_key,
                        alt: ev.alt_key,
                        meta: ev.meta_key,
                    },
                );
            })
            .onmouseup(move |ev: strata_dom::Event| {
                ev.stop_propagation();
                ev.release_pointer_capture(ev.pointer_id);
                self_up
                    .state
                    .borrow_mut()
                    .orbit
                    .set_viewport_size(ev.width, ev.height);
                self_up.dispatch_pointer_up(
                    EarthMouseButton::from_dom_button(ev.button),
                    ev.offset_x,
                    ev.offset_y,
                    EarthModifiers {
                        shift: ev.shift_key,
                        ctrl: ev.ctrl_key,
                        alt: ev.alt_key,
                        meta: ev.meta_key,
                    },
                );
            })
            .onwheel(move |ev: strata_dom::Event| {
                ev.stop_propagation();
                ev.prevent_default();
                self_wheel
                    .state
                    .borrow_mut()
                    .orbit
                    .set_viewport_size(ev.width, ev.height);
                self_wheel.dispatch_wheel(
                    ev.delta_x,
                    ev.delta_y,
                    ev.offset_x,
                    ev.offset_y,
                    EarthModifiers {
                        shift: ev.shift_key,
                        ctrl: ev.ctrl_key,
                        alt: ev.alt_key,
                        meta: ev.meta_key,
                    },
                );
            })
            .build();

        let label_nodes: Vec<strata_dom::NodeBlueprint> = labels
            .iter()
            .map(|lbl| render_earth_label(lbl).into_node_blueprint())
            .collect();
        let labels_fragment = strata_dom::NodeBlueprint::fragment(label_nodes);
        let plugins_fragment = strata_dom::NodeBlueprint::fragment(plugin_overlays);

        rsx! {
            <div
                tabindex="0"
                class="relative w-full h-full bg-black overflow-hidden outline-none"
                onkeydown={move |ev: strata_dom::Event| {
                    let key = parse_dom_key_to_keycode(&ev.key);
                    let dom_mods = EarthModifiers {
                        shift: ev.shift_key,
                        ctrl: ev.ctrl_key,
                        alt: ev.alt_key,
                        meta: ev.meta_key,
                    };
                    let now = Instant::now();
                    let (skip_dup, repeat, mods) = {
                        let mut st = self_key_down.state.borrow_mut();
                        st.live_modifiers = st.live_modifiers.merge(dom_mods);
                        st.update_modifier_from_dom_key(&ev.key, true);
                        let mods = st.live_modifiers;
                        if key == KeyCode::Unknown {
                            (false, ev.repeat, mods)
                        } else {
                            match st.held_keys.get(&key).copied() {
                                Some(prev) if now.duration_since(prev) < Duration::from_millis(8) => (true, false, mods),
                                Some(_) => {
                                    st.held_keys.insert(key, now);
                                    (false, true, mods)
                                }
                                None => {
                                    st.held_keys.insert(key, now);
                                    (false, ev.repeat, mods)
                                }
                            }
                        }
                    };
                    if !skip_dup {
                        let cb = self_key_down.listeners.borrow().on_key_down.clone();
                        if let Some(cb) = cb {
                            let kev = EarthKeyEvent::new(key, ev.key.clone(), mods, repeat);
                            cb(&kev);
                            if kev.is_default_prevented() {
                                ev.prevent_default();
                            }
                        }
                    }
                }}
                onkeyup={move |ev: strata_dom::Event| {
                    let key = parse_dom_key_to_keycode(&ev.key);
                    let dom_mods = EarthModifiers {
                        shift: ev.shift_key,
                        ctrl: ev.ctrl_key,
                        alt: ev.alt_key,
                        meta: ev.meta_key,
                    };
                    let (was_held, mods) = {
                        let mut st = self_key_up.state.borrow_mut();
                        st.live_modifiers = dom_mods;
                        st.update_modifier_from_dom_key(&ev.key, false);
                        let mods = st.live_modifiers;
                        let held = if key == KeyCode::Unknown {
                            true
                        } else {
                            st.held_keys.remove(&key).is_some()
                        };
                        (held, mods)
                    };
                    if was_held {
                        let cb = self_key_up.listeners.borrow().on_key_up.clone();
                        if let Some(cb) = cb {
                            let kev = EarthKeyEvent::new(key, ev.key.clone(), mods, false);
                            cb(&kev);
                            if kev.is_default_prevented() {
                                ev.prevent_default();
                            }
                        }
                    }
                }}
            >
                {canvas}
                <div class="absolute inset-0 z-10 pointer-events-none overflow-hidden">
                    {labels_fragment}
                    {plugins_fragment}
                </div>
            </div>
        }
    }
}

fn render_earth_label(lbl: &ProjectedEarthLabel) -> View {
    let x = lbl.screen_x;
    let y = lbl.screen_y;
    let [r, g, b] = lbl.rgb;
    let alpha = lbl.alpha.clamp(0.0, 1.0);
    let op = format!("{:.2}", alpha);
    let bg_op = format!("{:.2}", alpha * 0.35);
    let bg25_op = format!("{:.2}", alpha * 0.25);

    let key_id = lbl.id.to_string();
    match lbl.icon {
        EarthLabelIcon::HighwayShield => {
            let shield_cls = if lbl.text.starts_with('I') {
                "bg-blue-900/90 border-blue-400/80 text-white"
            } else if lbl.text.starts_with("US") || lbl.text.starts_with("SR") {
                "bg-neutral-900/90 border-neutral-300/85 text-white"
            } else {
                "bg-emerald-950/90 border-emerald-400/80 text-emerald-100"
            };
            rsx! {
                <div
                    key={key_id}
                    class={format!("absolute px-1.5 py-0.5 rounded border text-[10px] font-bold tracking-wide select-none shadow-md {}", shield_cls)}
                    style={format!("left: {:.2}px; top: {:.2}px; opacity: {};", x - 18.0, y - 9.0, op)}
                >
                    <span>{lbl.text.clone()}</span>
                </div>
            }
        }
        EarthLabelIcon::MountainRight => {
            rsx! {
                <div
                    key={key_id}
                    class="absolute flex items-center gap-1 px-1.5 py-0.5 rounded select-none whitespace-nowrap"
                    style={format!("left: {:.2}px; top: {:.2}px; background-color: rgba(0,0,0,{}); color: rgba({},{},{},{}); font-size: {:.1}px;", x - 42.0, y - 10.0, bg_op, r, g, b, op, lbl.font_px)}
                >
                    <span class="font-medium">{lbl.text.clone()}</span>
                    <svg class="w-3.5 h-3.5 shrink-0" viewBox="0 0 24 24" fill="currentColor">
                        <path d="M14 6l-3.75 5 2.85 3.8-1.6 1.2C9.81 13.75 7 10 7 10l-6 8h22L14 6z" />
                    </svg>
                </div>
            }
        }
        EarthLabelIcon::ParkLeafRight => {
            rsx! {
                <div
                    key={key_id}
                    class="absolute flex items-center gap-1 px-1.5 py-0.5 rounded select-none whitespace-nowrap"
                    style={format!("left: {:.2}px; top: {:.2}px; background-color: rgba(0,0,0,{}); color: rgba({},{},{},{}); font-size: {:.1}px;", x - 44.0, y - 10.0, bg_op, r, g, b, op, lbl.font_px)}
                >
                    <span class="font-medium">{lbl.text.clone()}</span>
                    <svg class="w-3 h-3 shrink-0" viewBox="0 0 24 24" fill="currentColor">
                        <path d="M17 8C8 10 5.9 16.17 3.82 21.34l1.89.66l.95-2.3c.48.17.98.3 1.34.3C19 20 22 3 22 3c-1 2-8 2.25-13 3.25S2 11.5 2 13.5s1.75 3.75 1.75 3.75C7 8 17 8 17 8z" />
                    </svg>
                </div>
            }
        }
        EarthLabelIcon::CityDot => {
            rsx! {
                <div
                    key={key_id}
                    class="absolute flex items-center gap-1.5 px-1.5 py-0.5 rounded select-none whitespace-nowrap"
                    style={format!("left: {:.2}px; top: {:.2}px; background-color: rgba(0,0,0,{}); color: rgba({},{},{},{}); font-size: {:.1}px;", x - 26.0, y - 9.0, bg_op, r, g, b, op, lbl.font_px)}
                >
                    <div class="w-1.5 h-1.5 rounded-full bg-white/95" />
                    <span class="font-medium">{lbl.text.clone()}</span>
                </div>
            }
        }
        EarthLabelIcon::None => {
            let extra_cls = match lbl.kind {
                EarthLabelKind::Continent => "font-semibold tracking-widest",
                EarthLabelKind::Ocean => "italic tracking-wider",
                EarthLabelKind::Country => "font-medium tracking-wide",
                _ => "font-normal",
            };
            rsx! {
                <div
                    key={key_id}
                    class={format!("absolute px-1 py-0.5 rounded select-none whitespace-nowrap {}", extra_cls)}
                    style={format!("left: {:.2}px; top: {:.2}px; background-color: rgba(0,0,0,{}); color: rgba({},{},{},{}); font-size: {:.1}px;", x - 34.0, y - 9.0, bg25_op, r, g, b, op, lbl.font_px)}
                >
                    <span>{lbl.text.clone()}</span>
                </div>
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tiles::MemoryTileSource;
    use strata::platform::{
        MouseButton, PlatformEvent, SoftwareSurfacePresenter, SwapchainCompositor,
        SwapchainFormat,
    };
    use strata::runtime::platforms::dispatch_native_platform_event;

    #[test]
    fn embedded_earth_view_in_subrect_handles_pan_orbit_zoom_and_cursor() {
        let mut input = EarthInput::default();
        input.elevation_enabled = false;
        input.layers.labels = false;
        let source = Box::new(MemoryTileSource::default());
        let earth = EarthView::new(&input, source);
        earth.with_state_mut(|st| {
            assert_eq!(
                st.viewport.input().effective_surface_mode(),
                crate::scene::SurfaceMode::QuadtreeLod
            );
            st.orbit.jump_to(37.7749, -122.4194, 5_000.0, 0.0, 30.0);
            st.sync_camera();
        });
        assert_eq!(
            earth.step_frame(1.0 / 60.0),
            FrameOutcome::Unchanged,
            "with labels and overlay plugins off, pure-3D jump does not require 2D HUD rebuilds"
        );
        earth.with_state_mut(|st| st.viewport.set_labels_enabled(true));
        assert_eq!(
            earth.step_frame(1.0 / 60.0),
            FrameOutcome::Dirty,
            "enabling labels triggers a single 2D HUD overlay rebuild"
        );
        assert_eq!(
            earth.step_frame(1.0 / 60.0),
            FrameOutcome::Unchanged,
            "steady-state weather/orbit tick without label changes must not trigger 2D HUD rebuilds"
        );
        earth.with_state_mut(|st| st.viewport.set_labels_enabled(false));
        assert_eq!(earth.step_frame(1.0 / 60.0), FrameOutcome::Dirty);

        let earth_view = earth.clone();
        let build_root = move || {
            rsx! {
                <div class="w-full h-full flex flex-col bg-slate-950">
                    <div class="w-full h-[120px]">
                        <button class="w-32 h-10">"Header"</button>
                    </div>
                    <div class="flex flex-row w-full h-[400px]">
                        <div class="w-[300px] h-full" />
                        <div class="w-[640px] h-[400px]">
                            {earth_view.view()}
                        </div>
                    </div>
                </div>
            }
        };

        let mut runtime = AppRuntime::new(build_root(), 1280.0, 800.0, 1.0);
        runtime.tick_frame();
        let mut compositor = SwapchainCompositor::new(1280, 800, 1.0);
        let mut presenter =
            SoftwareSurfacePresenter::new(1280, 800, SwapchainFormat::Bgra8Unorm);
        let mut no_global_hook = |_: &PlatformEvent, _: &mut AppRuntime| false;

        // Moving inside embedded 640x400 subrect (at x=300..940, y=120..520) yields Grab cursor.
        dispatch_native_platform_event(
            &PlatformEvent::MouseMove { x: 600.0, y: 300.0 },
            &mut runtime,
            &mut compositor,
            &mut presenter,
            &mut no_global_hook,
        );
        assert_eq!(runtime.current_cursor, strata::platform::CursorIcon::Grab);

        // Click on outside header should NOT trigger Earth drag or change camera.
        let before_outside = earth.with_state(|st| (st.orbit.lat_deg, st.orbit.lon_deg));
        dispatch_native_platform_event(
            &PlatformEvent::MouseDown {
                button: MouseButton::Left,
                x: 40.0,
                y: 20.0,
            },
            &mut runtime,
            &mut compositor,
            &mut presenter,
            &mut no_global_hook,
        );
        dispatch_native_platform_event(
            &PlatformEvent::MouseMove { x: 100.0, y: 30.0 },
            &mut runtime,
            &mut compositor,
            &mut presenter,
            &mut no_global_hook,
        );
        dispatch_native_platform_event(
            &PlatformEvent::MouseUp {
                button: MouseButton::Left,
                x: 100.0,
                y: 30.0,
            },
            &mut runtime,
            &mut compositor,
            &mut presenter,
            &mut no_global_hook,
        );
        let after_outside = earth.with_state(|st| (st.orbit.lat_deg, st.orbit.lon_deg));
        assert_eq!(before_outside, after_outside);

        // Left drag inside embedded component pans focus & switches cursor to Grabbing then Grab.
        dispatch_native_platform_event(
            &PlatformEvent::MouseDown {
                button: MouseButton::Left,
                x: 620.0,
                y: 320.0,
            },
            &mut runtime,
            &mut compositor,
            &mut presenter,
            &mut no_global_hook,
        );
        assert!(earth.with_state(|st| st.is_grabbing.untracked_get()));
        dispatch_native_platform_event(
            &PlatformEvent::MouseMove { x: 520.0, y: 260.0 },
            &mut runtime,
            &mut compositor,
            &mut presenter,
            &mut no_global_hook,
        );
        dispatch_native_platform_event(
            &PlatformEvent::MouseUp {
                button: MouseButton::Left,
                x: 520.0,
                y: 260.0,
            },
            &mut runtime,
            &mut compositor,
            &mut presenter,
            &mut no_global_hook,
        );
        assert!(!earth.with_state(|st| st.is_grabbing.untracked_get()));
        let (panned_lat, panned_lon, vw, vh) = earth.with_state(|st| {
            (
                st.orbit.lat_deg,
                st.orbit.lon_deg,
                st.orbit.viewport_w,
                st.orbit.viewport_h,
            )
        });
        assert!((vw - 640.0).abs() < 1e-3 && (vh - 400.0).abs() < 1e-3);
        assert!(
            (panned_lat - before_outside.0).abs() > 1e-5
                || (panned_lon - before_outside.1).abs() > 1e-5
        );

        // Right drag orbits heading & tilt inside embedded Canvas3D without any global hook.
        let (h0, t0) = earth.with_state(|st| (st.orbit.heading_deg, st.orbit.tilt_deg));
        dispatch_native_platform_event(
            &PlatformEvent::MouseDown {
                button: MouseButton::Right,
                x: 620.0,
                y: 320.0,
            },
            &mut runtime,
            &mut compositor,
            &mut presenter,
            &mut no_global_hook,
        );
        dispatch_native_platform_event(
            &PlatformEvent::MouseMove { x: 700.0, y: 270.0 },
            &mut runtime,
            &mut compositor,
            &mut presenter,
            &mut no_global_hook,
        );
        dispatch_native_platform_event(
            &PlatformEvent::MouseUp {
                button: MouseButton::Right,
                x: 700.0,
                y: 270.0,
            },
            &mut runtime,
            &mut compositor,
            &mut presenter,
            &mut no_global_hook,
        );
        let (h1, t1) = earth.with_state(|st| (st.orbit.heading_deg, st.orbit.tilt_deg));
        assert!((h1 - h0).abs() > 1.0 && (t1 - t0).abs() > 1.0);

        // Scroll inside component zooms closer.
        let r0 = earth.with_state(|st| st.orbit.target_range_m);
        dispatch_native_platform_event(
            &PlatformEvent::Scroll {
                delta_x: 0.0,
                delta_y: -220.0,
                is_momentum: false,
            },
            &mut runtime,
            &mut compositor,
            &mut presenter,
            &mut no_global_hook,
        );
        let r1 = earth.with_state(|st| st.orbit.target_range_m);
        assert!(r1 < r0, "wheel up inside EarthView must zoom closer ({r1} < {r0})");
    }

    #[test]
    fn earth_view_detects_window_resize_and_updates_camera_without_mouse_events() {
        let mut input = EarthInput::default();
        input.elevation_enabled = false;
        let source = Box::new(MemoryTileSource::default());
        let earth = EarthView::new(&input, source);

        let earth_view = earth.clone();
        let build_root = move || {
            rsx! {
                <div class="w-full h-full">
                    {earth_view.view()}
                </div>
            }
        };

        let mut runtime = AppRuntime::new(build_root(), 1200.0, 750.0, 1.0);
        // Initial mount fires Canvas3D onresize with (1200, 750) automatically.
        let (init_w, init_h, init_vp_w, init_vp_h) = earth.with_state(|st| {
            let vp = st.viewport.input().viewport;
            (st.orbit.viewport_w, st.orbit.viewport_h, vp.width, vp.height)
        });
        assert!((init_w - 1200.0).abs() < 0.5 && (init_h - 750.0).abs() < 0.5);
        assert!((init_vp_w - 1200.0).abs() < 0.5 && (init_vp_h - 750.0).abs() < 0.5);

        let mut compositor = SwapchainCompositor::new(1200, 750, 1.0);
        let mut presenter =
            SoftwareSurfacePresenter::new(1200, 750, SwapchainFormat::Bgra8Unorm);
        let mut no_global_hook = |_: &PlatformEvent, _: &mut AppRuntime| false;

        // Resize window to 1600x920 with zero mouse events: Canvas3D onresize must fire
        // and mark both rebuild and redraw true.
        let (needs_rebuild, needs_redraw, _) = dispatch_native_platform_event(
            &PlatformEvent::WindowResized {
                width: 1600,
                height: 920,
            },
            &mut runtime,
            &mut compositor,
            &mut presenter,
            &mut no_global_hook,
        );
        assert!(needs_rebuild, "Canvas3D onresize must request view rebuild on window resize");
        assert!(needs_redraw, "WindowResized must request immediate redraw");

        let (new_w, new_h, new_vp_w, new_vp_h) = earth.with_state(|st| {
            let vp = st.viewport.input().viewport;
            (st.orbit.viewport_w, st.orbit.viewport_h, vp.width, vp.height)
        });
        assert!((new_w - 1600.0).abs() < 0.5 && (new_h - 920.0).abs() < 0.5);
        assert!((new_vp_w - 1600.0).abs() < 0.5 && (new_vp_h - 920.0).abs() < 0.5);
    }

    #[test]
    fn bench_earth_view_end_to_end_cpu_and_dom_reconcile() {
        let mut input = EarthInput::globe();
        input.elevation_enabled = false;
        input.layers.labels = true;
        let source = Box::new(MemoryTileSource::default());
        let earth = EarthView::new(&input, source);

        let earth_view = earth.clone();
        let build_root = move || {
            rsx! {
                <div class="relative w-full h-full bg-black overflow-hidden">
                    {earth_view.view()}
                    <div class="absolute top-3 right-3 z-20 flex flex-wrap justify-end gap-1.5 max-w-[440px] p-2 rounded-xl bg-neutral-950/75 border border-neutral-800/80">
                        <button class="px-2.5 py-1 rounded-lg border text-[11px] font-medium">"Mode: 3D"</button>
                        <button class="px-2.5 py-1 rounded-lg border text-[11px] font-medium">"Map: Satellite"</button>
                        <button class="px-2.5 py-1 rounded-lg border text-[11px] font-medium">"View: Space"</button>
                        <button class="px-2.5 py-1 rounded-lg border text-[11px] font-medium">"Sun Light: On"</button>
                        <button class="px-2.5 py-1 rounded-lg border text-[11px] font-medium">"City Lights"</button>
                        <button class="px-2.5 py-1 rounded-lg border text-[11px] font-medium">"Ocean Waves"</button>
                        <button class="px-2.5 py-1 rounded-lg border text-[11px] font-medium">"Sun: OS Clock"</button>
                        <button class="px-2.5 py-1 rounded-lg border text-[11px] font-medium">"Stars/Moon"</button>
                        <button class="px-2.5 py-1 rounded-lg border text-[11px] font-medium">"Sky: Composite"</button>
                        <button class="px-2.5 py-1 rounded-lg border text-[11px] font-medium">"Res: 2K"</button>
                        <button class="px-2.5 py-1 rounded-lg border text-[11px] font-medium">"Satellites"</button>
                        <button class="px-2.5 py-1 rounded-lg border text-[11px] font-medium">"Heightmap"</button>
                        <button class="px-2.5 py-1 rounded-lg border text-[11px] font-medium">"Labels"</button>
                        <button class="px-2.5 py-1 rounded-lg border text-[11px] font-medium">"Clouds"</button>
                        <button class="px-2.5 py-1 rounded-lg border text-[11px] font-medium">"Clouds: 2K"</button>
                        <button class="px-2.5 py-1 rounded-lg border text-[11px] font-medium">"Rain"</button>
                        <button class="px-2.5 py-1 rounded-lg border text-[11px] font-medium">"Snow"</button>
                    </div>
                </div>
            }
        };

        let mut runtime = AppRuntime::new(build_root(), 1440.0, 900.0, 1.0);
        // Warm up 5 frames
        for _ in 0..5 {
            let _ = earth.step_frame(1.0 / 60.0);
            runtime.update_view(build_root());
        }

        // 1. Benchmark 500 pure 3D camera drag/step frames
        let t_step0 = Instant::now();
        earth.with_state_mut(|st| {
            st.orbit.handle_mouse_down(0, 720.0, 450.0);
        });
        for i in 0..500 {
            let x = 720.0 + ((i as f64) * 0.1).sin() * 80.0;
            let y = 450.0 + ((i as f64) * 0.1).cos() * 40.0;
            earth.with_state_mut(|st| {
                if st.orbit.handle_mouse_move(x, y) {
                    st.sync_camera();
                }
            });
            let _ = earth.step_frame(1.0 / 60.0);
        }
        earth.with_state_mut(|st| {
            st.orbit.handle_mouse_up(0);
        });
        let step_ms = t_step0.elapsed().as_secs_f64() * 1000.0;

        // 2. Benchmark 200 full 2D HUD + 36 projected label DOM update_view passes with incremental reconciliation
        let t_dom0 = Instant::now();
        for i in 0..200 {
            earth.with_state_mut(|st| {
                st.orbit.lon_deg += 0.25;
                st.orbit.lat_deg = ((i as f64) * 0.05).sin() * 20.0;
                st.sync_camera();
            });
            runtime.update_view(build_root());
        }
        let dom_ms = t_dom0.elapsed().as_secs_f64() * 1000.0;

        eprintln!(
            "[bench EarthView 1440x900] 500 3D drag+step_frame ticks: {:.2} ms ({:.3} ms/frame, {:.0} FPS) | 200 2D HUD+36-label DOM reconcile passes: {:.2} ms ({:.3} ms/pass)",
            step_ms,
            step_ms / 500.0,
            500_000.0 / step_ms,
            dom_ms,
            dom_ms / 200.0,
        );
    }

    #[test]
    fn orbit_camera_motion_updates_labels_every_frame_smoothly_without_66ms_stepping() {
        let mut orbit = EarthOrbitController::new();
        orbit.jump_to(40.1, 25.5, 580_000.0, 0.0, 0.0);
        orbit.set_viewport_size(1920.0, 1080.0);
        let (cam, up) = orbit.to_camera_and_up(false, 0.0);
        let mut input = EarthInput::globe();
        input.camera = cam;
        input.camera_up_ecef = up;
        input.viewport = strata::layout::Rect {
            x: 0.0,
            y: 0.0,
            width: 1920.0,
            height: 1080.0,
        };
        input.elevation_enabled = false;
        input.layers.labels = true;
        let earth = EarthView::with_orbit(&input, Box::new(MemoryTileSource::default()), orbit);

        // Initial projection populates visible labels.
        let _ = earth.view();
        let tracked_text = earth.with_state(|st| {
            st.compute_projected_labels()
                .first()
                .expect("expected visible labels over Aegean at 580 km")
                .text
                .clone()
        });

        let mut xs = Vec::new();
        for _ in 0..16 {
            earth.with_state_mut(|st| {
                let lat = st.orbit.lat_deg;
                let lon = st.orbit.lon_deg + 0.012;
                let rng = st.orbit.range_m;
                st.orbit.jump_to(lat, lon, rng, 0.0, 0.0);
                st.sync_camera();
            });
            let outcome = earth.step_frame(1.0 / 144.0);
            assert_eq!(
                outcome,
                strata::FrameOutcome::Dirty,
                "144Hz orbit increments must mark overlay Dirty every frame without 66ms throttling"
            );
            let x = earth.with_state(|st| {
                st.compute_projected_labels()
                    .into_iter()
                    .find(|l| l.text == tracked_text)
                    .expect("tracked label should remain deterministically visible across small orbit")
                    .screen_x
            });
            xs.push(x);
        }

        for pair in xs.windows(2) {
            let dx = pair[1] - pair[0];
            assert!(
                dx < -0.2 && dx > -12.0,
                "expected smooth subpixel westward motion per 144Hz frame, got dx={dx} ({pair:?})"
            );
        }
    }

    #[test]
    fn orbit_and_zoom_reconciliation_never_renders_black_label_text_or_icons() {
        use strata_render::compositor::DrawCommand;

        let mut input = EarthInput::default();
        input.elevation_enabled = false;
        input.layers.labels = true;
        let source = Box::new(MemoryTileSource::default());
        let earth = EarthView::new(&input, source);

        let mut runtime = strata::runtime::AppRuntime::new(earth.view(), 1920.0, 1080.0, 1.0);

        // Sweep across global oceans/continents, Mediterranean countries/seas,
        // Aegean cities/mountains, and US highways/parks so every EarthLabelIcon
        // variant and horizon-alpha transition exercises DOM reconciliation.
        let camera_keyframes = [
            (20.0, 0.0, 18_000_000.0, 0.0, 0.0),
            (25.0, 15.0, 12_000_000.0, 0.0, 10.0),
            (38.0, 24.0, 3_200_000.0, 12.0, 20.0),
            (38.2, 24.5, 620_000.0, -8.0, 32.0),
            (38.25, 24.8, 580_000.0, 0.0, 35.0),
            (37.7749, -122.4194, 420_000.0, 15.0, 40.0),
            (37.7749, -122.4194, 95_000.0, 0.0, 45.0),
            (38.0, 23.7, 750_000.0, 0.0, 25.0),
        ];

        let mut total_checked_labels = 0usize;
        for &(lat, lon, range_m, heading, tilt) in &camera_keyframes {
            earth.with_state_mut(|st| {
                st.orbit.jump_to(lat, lon, range_m, heading, tilt);
                st.sync_camera();
            });
            let _ = earth.step_frame(1.0 / 144.0);
            runtime.update_view(earth.view());
            assert!(
                runtime.last_update_reconciled(),
                "EarthView HUD updates should reconcile in-place"
            );

            let projected = earth.with_state(|st| st.compute_projected_labels());
            assert!(
                !projected.is_empty(),
                "expected projected labels at ({lat}, {lon}, range={range_m})"
            );

            for lbl in &projected {
                let mut matched = false;
                for cmd in runtime.display_list.commands() {
                    match cmd {
                        DrawCommand::DrawText { text, color, .. } if text == &lbl.text => {
                            matched = true;
                            assert_ne!(
                                (color.r, color.g, color.b),
                                (0, 0, 0),
                                "label {:?} rendered with black DrawText color {color:?}",
                                lbl.text
                            );
                        }
                        DrawCommand::DrawTextFragments { fragments } => {
                            for frag in fragments {
                                if frag.text == lbl.text {
                                    matched = true;
                                    let c = frag.style.color;
                                    assert_ne!(
                                        (c.r, c.g, c.b),
                                        (0, 0, 0),
                                        "label {:?} rendered with black fragment color {c:?}",
                                        lbl.text
                                    );
                                    if lbl.alpha > 0.5 && lbl.icon != crate::labels::EarthLabelIcon::HighwayShield {
                                        assert!(
                                            c.a >= 100,
                                            "bright label {:?} (alpha={:.2}) stuck at faint alpha {c:?}",
                                            lbl.text,
                                            lbl.alpha
                                        );
                                    }
                                }
                            }
                        }
                        _ => {}
                    }
                }
                assert!(
                    matched,
                    "projected label {:?} missing from compiled DisplayList",
                    lbl.text
                );
                total_checked_labels += 1;
            }

            for cmd in runtime.display_list.commands() {
                if let DrawCommand::DrawSvgPath {
                    fill_color: Some(fill),
                    stroke_color: None,
                    ..
                } = cmd
                {
                    assert_ne!(
                        (fill.r, fill.g, fill.b),
                        (0, 0, 0),
                        "SVG fill='currentColor' icon lost inherited color and turned black: {fill:?}"
                    );
                }
            }
        }
        assert!(total_checked_labels >= 20);
    }
}


