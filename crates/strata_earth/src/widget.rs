//! Embeddable Strata `<EarthView>` UI component built on [`EarthViewport`] and [`EarthOrbitController`].
//!
//! Uses only standard Strata component APIs ([`Canvas3D`], [`Signal`], `ev.offset_x/y`,
//! `ev.width/height`, `ev.button`, `ev.delta_y`, `ev.set_pointer_capture`, `ev.prevent_default`,
//! and `ev.stop_propagation`) with zero global window event hooks, allowing drop-in placement inside
//! any layout container of any size.

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use strata::prelude::*;
use strata::render::scene3d::Canvas3D;
use strata::FrameOutcome;

use crate::controller::EarthOrbitController;
use crate::labels::{
    builtin_world_labels, project_earth_labels, EarthLabelIcon, EarthLabelKind, EarthMapLabel,
    EarthScreenProjector, LiveMapLabelStore, ProjectedEarthLabel,
};
use crate::plugins::{EarthPlugin, EarthPluginRegistry};
use crate::scene::{EarthInput, EarthViewMode, OceanWavesConfig};
use crate::tiles::TileSource;
use crate::worker::EarthViewport;

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
    pub builtin_labels: Vec<EarthMapLabel>,
    pub live_labels: LiveMapLabelStore,
    pub is_grabbing: Signal<bool>,
    pub ui_rev: Signal<u64>,
    overlay_plugins: Vec<Box<dyn EarthOverlayPlugin>>,
    overlay_plugin_rev: u64,
    last_label_sig: u64,
    last_label_refresh: Instant,
    last_labels_enabled: bool,
    tile_ready: Arc<AtomicBool>,
    label_ready: Arc<AtomicBool>,
    uncapped: bool,
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
        let viewport = EarthViewport::with_plugins(&effective_input, source, Some(tile_wakeup), plugins);
        let builtin_labels = builtin_world_labels();
        let live_labels = LiveMapLabelStore::new(Some(label_wakeup));

        let mut st = Self {
            viewport,
            orbit,
            builtin_labels,
            live_labels,
            is_grabbing: Signal::new(false),
            ui_rev: Signal::new(0),
            overlay_plugins: Vec::new(),
            overlay_plugin_rev: 0,
            last_label_sig: 0,
            last_label_refresh: Instant::now(),
            last_labels_enabled: effective_input.layers.labels,
            tile_ready,
            label_ready,
            uncapped: false,
        };
        st.sync_camera();
        st.last_label_sig = st.label_state_signature();
        st
    }

    /// Synchronizes `viewport` camera uniforms and worker LOD target from `orbit` with strict
    /// terrain elevation collision clamping.
    pub fn sync_camera(&mut self) {
        let is_2d = self.viewport.view_mode() == EarthViewMode::Mode2D;
        let vw = self.orbit.viewport_w;
        let vh = self.orbit.viewport_h;
        let (cam, up_opt) = self
            .orbit
            .to_camera_and_up_clamped(is_2d, |lat, lon| self.viewport.sample_ground_elevation_m(lat, lon));
        self.viewport.update_camera_with_up(&cam, up_opt, vw, vh);
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

    /// Jumps camera target smoothly to a geographic location & framing.
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

    fn label_state_signature(&self) -> u64 {
        let inp = self.viewport.input();
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        let nearby_count = if inp.layers.labels && self.orbit.range_m < 250_000.0 {
            self.live_labels
                .collect_nearby(self.orbit.lat_deg, self.orbit.lon_deg)
                .len() as u64
        } else {
            0
        };
        let parts = [
            inp.layers.labels as u64,
            self.overlay_plugin_rev,
            self.has_active_overlay_plugins() as u64,
            (self.orbit.eye_dir[0] * 1_000_000.0).round() as i64 as u64,
            (self.orbit.eye_dir[1] * 1_000_000.0).round() as i64 as u64,
            (self.orbit.eye_dir[2] * 1_000_000.0).round() as i64 as u64,
            (self.orbit.up_dir[0] * 1_000_000.0).round() as i64 as u64,
            (self.orbit.up_dir[1] * 1_000_000.0).round() as i64 as u64,
            (self.orbit.up_dir[2] * 1_000_000.0).round() as i64 as u64,
            (self.orbit.range_m.ln() * 4_000.0).round() as u64,
            (self.orbit.tilt_deg * 100.0).round() as u64,
            self.orbit.viewport_w.round() as u64,
            self.orbit.viewport_h.round() as u64,
            (inp.view_morph * 1_000.0).round() as u64,
            nearby_count,
        ];
        for p in parts {
            h ^= p;
            h = h.wrapping_mul(0x100_0000_01b3);
        }
        h
    }

    fn compute_projected_labels(&self) -> Vec<ProjectedEarthLabel> {
        let inp = self.viewport.input();
        if !inp.layers.labels {
            return Vec::new();
        }
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
    ///
    /// Returns [`FrameOutcome::Dirty`] only when the 2D HUD / projected label overlay tree changed
    /// and needs a DOM rebuild. Pure-3D updates (orbit damping, clouds, ocean waves, terrain tiles)
    /// bump [`RetainedScene3D`] generations in place so Strata's `poll_retained_3d_dirty()`
    /// presents them with zero 2D HUD rebuilds.
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

        if orbit_dirty || trans_dirty {
            self.sync_camera();
        }

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

        let new_sig = self.label_state_signature();
        let labels_changed = new_sig != self.last_label_sig;
        let now = Instant::now();
        let refresh_due = toggled_labels
            || label_dirty
            || (labels_changed
                && (labels_enabled || overlay_plugins_active)
                && (!self.orbit.is_dragging()
                    || now.duration_since(self.last_label_refresh) >= Duration::from_millis(50)));

        if refresh_due {
            self.last_label_sig = new_sig;
            self.last_label_refresh = now;
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

    /// Configures whether the underlying `<Canvas3D>` runs uncapped animation ticks.
    pub fn with_uncapped(self, uncapped: bool) -> Self {
        self.state.borrow_mut().uncapped = uncapped;
        self
    }

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

    /// Enables or disables animated wavy ocean renders on non-land water areas.
    pub fn set_ocean_waves_enabled(&self, enabled: bool) {
        let mut st = self.state.borrow_mut();
        st.viewport.set_ocean_waves_enabled(enabled);
        st.notify_ui_changed();
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

    /// Advances smooth damping, transitions, weather, ocean waves, and label updates by `dt_s` seconds.
    pub fn step_frame(&self, dt_s: f64) -> FrameOutcome {
        self.state.borrow_mut().step_frame(dt_s)
    }

    /// Returns a self-contained Strata [`View`] (`w-full h-full overflow-hidden`) binding all
    /// pointer drag, right/middle orbit, scroll wheel, pointer capture, and reactive cursor states
    /// directly via Strata's `<Canvas3D>` component APIs.
    pub fn view(&self) -> View {
        let (retained, grabbing, labels, plugin_overlays, uncapped) = {
            let st = self.state.borrow();
            let grabbing = st.is_grabbing.get();
            let _ = st.ui_rev.get();
            (
                st.viewport.retained_scene(),
                grabbing,
                st.compute_projected_labels(),
                st.compute_plugin_overlay_nodes(),
                st.uncapped,
            )
        };

        let cursor_cls = if grabbing {
            "absolute inset-0 w-full h-full z-0 cursor-grabbing"
        } else {
            "absolute inset-0 w-full h-full z-0 cursor-grab"
        };

        let st_down = Rc::clone(&self.state);
        let st_move = Rc::clone(&self.state);
        let st_up = Rc::clone(&self.state);
        let st_wheel = Rc::clone(&self.state);

        let canvas = Canvas3D::builder()
            .retained(retained)
            .uncapped(uncapped)
            .class(cursor_cls)
            .onmousedown(move |ev: strata_dom::Event| {
                ev.stop_propagation();
                ev.prevent_default();
                ev.set_pointer_capture(ev.pointer_id);
                let mut st = st_down.borrow_mut();
                st.orbit.set_viewport_size(ev.width, ev.height);
                let btn = ev.button.max(0) as u32;
                if st.orbit.handle_mouse_down(btn, ev.offset_x, ev.offset_y) {
                    st.sync_camera();
                    if !st.is_grabbing.untracked_get() {
                        st.is_grabbing.set(true);
                    }
                }
            })
            .onmousemove(move |ev: strata_dom::Event| {
                ev.stop_propagation();
                let mut st = st_move.borrow_mut();
                st.orbit.set_viewport_size(ev.width, ev.height);
                if st.orbit.handle_mouse_move(ev.offset_x, ev.offset_y) {
                    st.sync_camera();
                }
            })
            .onmouseup(move |ev: strata_dom::Event| {
                ev.stop_propagation();
                ev.release_pointer_capture(ev.pointer_id);
                let mut st = st_up.borrow_mut();
                st.orbit.set_viewport_size(ev.width, ev.height);
                let btn = ev.button.max(0) as u32;
                if st.orbit.handle_mouse_up(btn) {
                    st.sync_camera();
                }
                if st.is_grabbing.untracked_get() {
                    st.is_grabbing.set(false);
                }
            })
            .onwheel(move |ev: strata_dom::Event| {
                ev.stop_propagation();
                ev.prevent_default();
                let mut st = st_wheel.borrow_mut();
                st.orbit.set_viewport_size(ev.width, ev.height);
                st.orbit
                    .handle_scroll(ev.delta_y, ev.offset_x, ev.offset_y);
                st.sync_camera();
            })
            .build();

        let label_nodes: Vec<strata_dom::NodeBlueprint> =
            labels.iter().map(|lbl| render_earth_label(lbl).into_node_blueprint()).collect();
        let labels_fragment = strata_dom::NodeBlueprint::fragment(label_nodes);
        let plugins_fragment = strata_dom::NodeBlueprint::fragment(plugin_overlays);

        rsx! {
            <div class="relative w-full h-full bg-black overflow-hidden">
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
    let op = format!("{:.2}", lbl.alpha.clamp(0.0, 1.0));

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
                    class={format!("absolute px-1.5 py-0.5 rounded border text-[10px] font-bold tracking-wide select-none shadow-md {}", shield_cls)}
                    style={format!("left: {:.1}px; top: {:.1}px; opacity: {};", x - 18.0, y - 9.0, op)}
                >
                    <span>{lbl.text.clone()}</span>
                </div>
            }
        }
        EarthLabelIcon::MountainRight => {
            rsx! {
                <div
                    class="absolute flex items-center gap-1 px-1.5 py-0.5 rounded bg-black/35 select-none whitespace-nowrap"
                    style={format!("left: {:.1}px; top: {:.1}px; opacity: {}; color: rgb({},{},{}); font-size: {:.1}px;", x - 42.0, y - 10.0, op, r, g, b, lbl.font_px)}
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
                    class="absolute flex items-center gap-1 px-1.5 py-0.5 rounded bg-black/35 select-none whitespace-nowrap"
                    style={format!("left: {:.1}px; top: {:.1}px; opacity: {}; color: rgb({},{},{}); font-size: {:.1}px;", x - 44.0, y - 10.0, op, r, g, b, lbl.font_px)}
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
                    class="absolute flex items-center gap-1.5 px-1.5 py-0.5 rounded bg-black/35 select-none whitespace-nowrap"
                    style={format!("left: {:.1}px; top: {:.1}px; opacity: {}; color: rgb({},{},{}); font-size: {:.1}px;", x - 26.0, y - 9.0, op, r, g, b, lbl.font_px)}
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
                    class={format!("absolute px-1 py-0.5 rounded bg-black/25 select-none whitespace-nowrap {}", extra_cls)}
                    style={format!("left: {:.1}px; top: {:.1}px; opacity: {}; color: rgb({},{},{}); font-size: {:.1}px;", x - 34.0, y - 9.0, op, r, g, b, lbl.font_px)}
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
}

