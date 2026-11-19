//! External interaction event payloads and camera control configuration for [`crate::EarthView`].
//!
//! Enables applications to attach rich pointer, click, wheel, keyboard, and camera-change listeners
//! directly to an [`crate::EarthView`] (for example raycasting click-to-place 3D objects, custom
//! gizmo dragging with [`EarthPointerEvent::prevent_default_camera`], or application-defined
//! keyboard shortcuts) with zero keybinds hardcoded inside `strata_earth`.

use std::cell::Cell;

use strata::platform::{KeyCode, Modifiers, MouseButton};

use crate::scene::EarthViewMode;

/// Mouse button associated with an [`EarthPointerEvent`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum EarthMouseButton {
    #[default]
    Left,
    Middle,
    Right,
    Other(u16),
}

impl EarthMouseButton {
    #[inline]
    pub fn from_dom_button(btn: i16) -> Self {
        match btn {
            0 => Self::Left,
            1 => Self::Middle,
            2 => Self::Right,
            other => Self::Other(other.max(0) as u16),
        }
    }

    #[inline]
    pub fn from_platform_button(btn: MouseButton) -> Self {
        match btn {
            MouseButton::Left => Self::Left,
            MouseButton::Middle => Self::Middle,
            MouseButton::Right => Self::Right,
            MouseButton::Other(v) => Self::Other(v),
        }
    }

    #[inline]
    pub fn controller_button_index(self) -> u32 {
        match self {
            Self::Left => 0,
            Self::Middle => 1,
            Self::Right => 2,
            Self::Other(v) => v as u32,
        }
    }
}

/// Active keyboard modifier flags accompanying pointer, wheel, and keyboard events.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct EarthModifiers {
    pub shift: bool,
    pub ctrl: bool,
    pub alt: bool,
    pub meta: bool,
}

impl EarthModifiers {
    #[inline]
    pub const fn from_platform(m: Modifiers) -> Self {
        Self {
            shift: m.shift,
            ctrl: m.ctrl,
            alt: m.alt,
            meta: m.meta,
        }
    }

    #[inline]
    pub fn merge(self, other: Self) -> Self {
        Self {
            shift: self.shift || other.shift,
            ctrl: self.ctrl || other.ctrl,
            alt: self.alt || other.alt,
            meta: self.meta || other.meta,
        }
    }
}

/// Pointer / click event delivered to [`crate::EarthView`] callbacks (`on_click`, `on_double_click`,
/// `on_context_menu`, `on_pointer_down`, `on_pointer_move`, `on_pointer_up`).
#[derive(Debug, Clone)]
pub struct EarthPointerEvent {
    /// Cursor X position in [`crate::EarthView`]-local logical pixels (`0.0 .. viewport_width`).
    pub x: f64,
    /// Cursor Y position in [`crate::EarthView`]-local logical pixels (`0.0 .. viewport_height`).
    pub y: f64,
    /// Incremental cursor motion since the previous pointer event (pixels).
    pub delta_x: f64,
    /// Incremental cursor motion since the previous pointer event (pixels).
    pub delta_y: f64,
    /// Current width of the [`crate::EarthView`] viewport in logical pixels.
    pub viewport_width: f64,
    /// Current height of the [`crate::EarthView`] viewport in logical pixels.
    pub viewport_height: f64,
    /// Mouse button associated with this event.
    pub button: EarthMouseButton,
    /// Active keyboard modifiers (`shift`, `ctrl`, `alt`, `meta`).
    pub modifiers: EarthModifiers,
    camera_prevented: Cell<bool>,
}

impl EarthPointerEvent {
    #[inline]
    pub fn new(
        x: f64,
        y: f64,
        delta_x: f64,
        delta_y: f64,
        viewport_width: f64,
        viewport_height: f64,
        button: EarthMouseButton,
        modifiers: EarthModifiers,
    ) -> Self {
        Self {
            x,
            y,
            delta_x,
            delta_y,
            viewport_width,
            viewport_height,
            button,
            modifiers,
            camera_prevented: Cell::new(false),
        }
    }

    /// Suppresses built-in [`crate::EarthOrbitController`] camera pan/orbit handling for this gesture.
    #[inline]
    pub fn prevent_default_camera(&self) {
        self.camera_prevented.set(true);
    }

    #[inline]
    pub fn is_default_camera_prevented(&self) -> bool {
        self.camera_prevented.get()
    }
}

/// Scroll / trackpad wheel event delivered to [`crate::EarthView::on_wheel`].
#[derive(Debug, Clone)]
pub struct EarthWheelEvent {
    pub x: f64,
    pub y: f64,
    pub delta_x: f64,
    pub delta_y: f64,
    pub viewport_width: f64,
    pub viewport_height: f64,
    pub modifiers: EarthModifiers,
    camera_prevented: Cell<bool>,
}

impl EarthWheelEvent {
    #[inline]
    pub fn new(
        x: f64,
        y: f64,
        delta_x: f64,
        delta_y: f64,
        viewport_width: f64,
        viewport_height: f64,
        modifiers: EarthModifiers,
    ) -> Self {
        Self {
            x,
            y,
            delta_x,
            delta_y,
            viewport_width,
            viewport_height,
            modifiers,
            camera_prevented: Cell::new(false),
        }
    }

    /// Suppresses built-in scroll-wheel zoom handling for this wheel tick.
    #[inline]
    pub fn prevent_default_camera(&self) {
        self.camera_prevented.set(true);
    }

    #[inline]
    pub fn is_default_camera_prevented(&self) -> bool {
        self.camera_prevented.get()
    }
}

/// Keyboard event delivered to [`crate::EarthView::on_key_down`] and [`crate::EarthView::on_key_up`].
#[derive(Debug, Clone)]
pub struct EarthKeyEvent {
    /// Normalized platform key code (layout-aware).
    pub key: KeyCode,
    /// Printable or W3C key string (`"k"`, `"l"`, `"ArrowUp"`, etc.).
    pub key_str: String,
    pub modifiers: EarthModifiers,
    pub repeat: bool,
    default_prevented: Cell<bool>,
}

impl EarthKeyEvent {
    #[inline]
    pub fn new(
        key: KeyCode,
        key_str: impl Into<String>,
        modifiers: EarthModifiers,
        repeat: bool,
    ) -> Self {
        Self {
            key,
            key_str: key_str.into(),
            modifiers,
            repeat,
            default_prevented: Cell::new(false),
        }
    }

    #[inline]
    pub fn prevent_default(&self) {
        self.default_prevented.set(true);
    }

    #[inline]
    pub fn is_default_prevented(&self) -> bool {
        self.default_prevented.get()
    }
}

/// Snapshot of camera telemetry emitted to [`crate::EarthView::on_camera_change`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EarthCameraState {
    pub lat_deg: f64,
    pub lon_deg: f64,
    pub range_m: f64,
    pub heading_deg: f64,
    pub tilt_deg: f64,
    pub view_mode: EarthViewMode,
    pub zoom_2d: f64,
}

/// Fine-grained enable/disable toggles for built-in mouse/wheel camera navigation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EarthControlsConfig {
    pub pan_enabled: bool,
    pub orbit_enabled: bool,
    pub zoom_enabled: bool,
    pub double_click_zoom_enabled: bool,
}

impl Default for EarthControlsConfig {
    fn default() -> Self {
        Self {
            pan_enabled: true,
            orbit_enabled: true,
            zoom_enabled: true,
            double_click_zoom_enabled: true,
        }
    }
}

/// Updates live modifier state from a W3C DOM `KeyboardEvent.key` string when routed via DOM listeners.
pub(crate) fn update_modifiers_from_dom_key(mods: &mut EarthModifiers, dom_key: &str, pressed: bool) {
    match dom_key {
        "Shift" | "ShiftLeft" | "ShiftRight" => mods.shift = pressed,
        "Control" | "ControlLeft" | "ControlRight" => mods.ctrl = pressed,
        "Alt" | "AltLeft" | "AltRight" => mods.alt = pressed,
        "Meta" | "MetaLeft" | "MetaRight" | "Super" | "SuperLeft" | "SuperRight" => {
            mods.meta = pressed
        }
        _ => {}
    }
}

/// Parses a W3C DOM `KeyboardEvent.key` string back into [`KeyCode`] when routed via DOM listeners.
pub(crate) fn parse_dom_key_to_keycode(dom_key: &str) -> KeyCode {
    if dom_key.len() == 1 {
        let ch = dom_key.chars().next().unwrap().to_ascii_lowercase();
        return match ch {
            'a' => KeyCode::KeyA,
            'b' => KeyCode::KeyB,
            'c' => KeyCode::KeyC,
            'd' => KeyCode::KeyD,
            'e' => KeyCode::KeyE,
            'f' => KeyCode::KeyF,
            'g' => KeyCode::KeyG,
            'h' => KeyCode::KeyH,
            'i' => KeyCode::KeyI,
            'j' => KeyCode::KeyJ,
            'k' => KeyCode::KeyK,
            'l' => KeyCode::KeyL,
            'm' => KeyCode::KeyM,
            'n' => KeyCode::KeyN,
            'o' => KeyCode::KeyO,
            'p' => KeyCode::KeyP,
            'q' => KeyCode::KeyQ,
            'r' => KeyCode::KeyR,
            's' => KeyCode::KeyS,
            't' => KeyCode::KeyT,
            'u' => KeyCode::KeyU,
            'v' => KeyCode::KeyV,
            'w' => KeyCode::KeyW,
            'x' => KeyCode::KeyX,
            'y' => KeyCode::KeyY,
            'z' => KeyCode::KeyZ,
            '0' => KeyCode::Digit0,
            '1' => KeyCode::Digit1,
            '2' => KeyCode::Digit2,
            '3' => KeyCode::Digit3,
            '4' => KeyCode::Digit4,
            '5' => KeyCode::Digit5,
            '6' => KeyCode::Digit6,
            '7' => KeyCode::Digit7,
            '8' => KeyCode::Digit8,
            '9' => KeyCode::Digit9,
            ' ' => KeyCode::Space,
            _ => KeyCode::Unknown,
        };
    }
    match dom_key {
        "ArrowUp" => KeyCode::ArrowUp,
        "ArrowDown" => KeyCode::ArrowDown,
        "ArrowLeft" => KeyCode::ArrowLeft,
        "ArrowRight" => KeyCode::ArrowRight,
        "Enter" => KeyCode::Enter,
        "Escape" => KeyCode::Escape,
        "Backspace" => KeyCode::Backspace,
        "Delete" => KeyCode::Delete,
        "Tab" => KeyCode::Tab,
        "Home" => KeyCode::Home,
        "End" => KeyCode::End,
        "PageUp" => KeyCode::PageUp,
        "PageDown" => KeyCode::PageDown,
        "F1" => KeyCode::F1,
        "F2" => KeyCode::F2,
        "F3" => KeyCode::F3,
        "F4" => KeyCode::F4,
        "F5" => KeyCode::F5,
        "F6" => KeyCode::F6,
        "F7" => KeyCode::F7,
        "F8" => KeyCode::F8,
        "F9" => KeyCode::F9,
        "F10" => KeyCode::F10,
        "F11" => KeyCode::F11,
        "F12" => KeyCode::F12,
        _ => KeyCode::Unknown,
    }
}
