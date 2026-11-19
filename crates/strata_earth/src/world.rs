//! Native geodetic 3D scene graph (`EarthWorld` & `EarthObject`) for `strata_earth`.
//!
//! Stores every object natively in double-precision geodetic coordinates (`lat_deg`, `lon_deg`,
//! `altitude_m`, [`AltitudeMode`]) rather than Cartesian `[x, y, z]`. Canonical local-space meshes
//! (`+X` East, `+Y` North, `+Z` Up) are deduplicated and uploaded to GPU VRAM once per unique
//! [`EarthMeshId`], while per-object placement, orientation, scale, live terrain clamping, and
//! 3D/2.5D/2D surface morphing compile in `O(N)` into hardware-instanced [`RetainedGeoInstance`]s.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use strata_render::camera64::DVec3;
use strata_render::scene3d::{RetainedGeoInstance, RetainedMeshAsset, RteTileVertex};
use strata_render::terrain3d::MeshBuilder;
use strata_style::Color;

use crate::geodesy::{
    project_morphed_point_and_frame, AltitudeMode, GeoPosition, LocalFrame, SurfaceMorphParams,
};

/// Stable handle to an [`EarthObject`] spawned in an [`EarthWorld`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct EarthObjectId(pub u64);

/// Stable handle to a reusable canonical local-space 3D mesh asset in an [`EarthWorld`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct EarthMeshId(pub u64);

/// Vertical origin convention for built-in procedural 3D primitives.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ObjectAnchor {
    /// Local origin `(0, 0, 0)` is at the geometric center of the primitive.
    #[default]
    Center,
    /// Local origin `(0, 0, 0)` sits at the bottom base of the primitive (`z >= 0`),
    /// so `GeoPosition::terrain(lat, lon, 0.0)` rests flush on the surface.
    Base,
}

/// Intrinsic local tangent-frame orientation in degrees (`+Y` North, `+X` East, `+Z` Up).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct GeoOrientation {
    /// Compass heading in degrees clockwise from true North (`0°` = North, `90°` = East).
    pub heading_deg: f64,
    /// Nose-up pitch in degrees (`+90°` = straight up).
    pub pitch_deg: f64,
    /// Bank/roll clockwise around forward axis in degrees.
    pub roll_deg: f64,
}

impl GeoOrientation {
    #[inline]
    pub const fn new(heading_deg: f64, pitch_deg: f64, roll_deg: f64) -> Self {
        Self {
            heading_deg,
            pitch_deg,
            roll_deg,
        }
    }

    #[inline]
    pub const fn heading(heading_deg: f64) -> Self {
        Self::new(heading_deg, 0.0, 0.0)
    }

    /// Computes the right-handed orthonormal local ENU rotation columns `(right_x, fwd_y, up_z)`.
    #[inline]
    pub fn local_basis_enu(&self) -> ([f64; 3], [f64; 3], [f64; 3]) {
        let (sh, ch) = self.heading_deg.to_radians().sin_cos();
        let (sp, cp) = self.pitch_deg.to_radians().sin_cos();
        let (sr, cr) = self.roll_deg.to_radians().sin_cos();

        // Heading clockwise from +Y (North) around +Z (Up):
        let right0 = [ch, -sh, 0.0];
        let fwd0 = [sh, ch, 0.0];
        let up0 = [0.0, 0.0, 1.0];

        // Pitch around right0:
        let fwd_p = [
            fwd0[0] * cp + up0[0] * sp,
            fwd0[1] * cp + up0[1] * sp,
            fwd0[2] * cp + up0[2] * sp,
        ];
        let up_p = [
            -fwd0[0] * sp + up0[0] * cp,
            -fwd0[1] * sp + up0[1] * cp,
            -fwd0[2] * sp + up0[2] * cp,
        ];

        // Roll around fwd_p:
        let right_pr = [
            right0[0] * cr - up_p[0] * sr,
            right0[1] * cr - up_p[1] * sr,
            right0[2] * cr - up_p[2] * sr,
        ];
        let up_pr = [
            right0[0] * sr + up_p[0] * cr,
            right0[1] * sr + up_p[1] * cr,
            right0[2] * sr + up_p[2] * cr,
        ];

        (right_pr, fwd_p, up_pr)
    }
}

/// Canonical local-space vertex (`+X` East, `+Y` North, `+Z` Up, in metres).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EarthVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
}

impl EarthVertex {
    #[inline]
    pub const fn new(position: [f32; 3], normal: [f32; 3], uv: [f32; 2]) -> Self {
        Self {
            position,
            normal,
            uv,
        }
    }
}

/// Built-in and custom 3D geometries authored in local tangent East-North-Up metres.
#[derive(Debug, Clone, PartialEq)]
pub enum EarthGeometry {
    /// Axis-aligned box (`width_m` East, `depth_m` North, `height_m` Up).
    Box {
        width_m: f64,
        depth_m: f64,
        height_m: f64,
        anchor: ObjectAnchor,
    },
    /// UV sphere with configurable longitudinal `segments` and latitudinal `rings`.
    Sphere {
        radius_m: f64,
        segments: u32,
        rings: u32,
        anchor: ObjectAnchor,
    },
    /// Cylinder or truncated frustum along local `+Z` (Up).
    Cylinder {
        radius_top_m: f64,
        radius_bottom_m: f64,
        height_m: f64,
        segments: u32,
        anchor: ObjectAnchor,
    },
    /// Cone pointing toward local `+Z` (Up).
    Cone {
        radius_m: f64,
        height_m: f64,
        segments: u32,
        anchor: ObjectAnchor,
    },
    /// Vertical capsule (total height = `length_m + 2 * radius_m`).
    Capsule {
        radius_m: f64,
        length_m: f64,
        segments: u32,
        anchor: ObjectAnchor,
    },
    /// Surveyor / waypoint 3D pin whose tip touches `(0, 0, 0)` and rises to `height_m`.
    BeaconPin { height_m: f64, radius_m: f64 },
    /// 3D tubular polyline connecting geodetic waypoints (`points` evaluated live against terrain/sea-level).
    Polyline {
        points: Vec<GeoPosition>,
        radius_m: f64,
    },
    /// 2D local `(east_m, north_m)` footprint extruded vertically from `z = 0` to `z = height_m`.
    ExtrudedPolygon {
        footprint_enu_m: Vec<[f64; 2]>,
        height_m: f64,
    },
    /// Arbitrary indexed triangle mesh in local ENU metres.
    CustomMesh {
        vertices: Vec<EarthVertex>,
        indices: Vec<u32>,
    },
    /// Binary glTF 2.0 `.glb` model scaled by `unit_scale_m`.
    Glb {
        bytes: Arc<[u8]>,
        unit_scale_m: f64,
        anchor: ObjectAnchor,
    },
}

impl EarthGeometry {
    #[inline]
    pub const fn box_m(width_m: f64, depth_m: f64, height_m: f64) -> Self {
        Self::Box {
            width_m,
            depth_m,
            height_m,
            anchor: ObjectAnchor::Center,
        }
    }

    #[inline]
    pub const fn cube(size_m: f64) -> Self {
        Self::box_m(size_m, size_m, size_m)
    }

    #[inline]
    pub const fn sphere(radius_m: f64) -> Self {
        Self::Sphere {
            radius_m,
            segments: 24,
            rings: 16,
            anchor: ObjectAnchor::Center,
        }
    }

    #[inline]
    pub const fn cylinder(radius_m: f64, height_m: f64) -> Self {
        Self::Cylinder {
            radius_top_m: radius_m,
            radius_bottom_m: radius_m,
            height_m,
            segments: 24,
            anchor: ObjectAnchor::Center,
        }
    }

    #[inline]
    pub const fn cone(radius_m: f64, height_m: f64) -> Self {
        Self::Cone {
            radius_m,
            height_m,
            segments: 24,
            anchor: ObjectAnchor::Base,
        }
    }

    #[inline]
    pub const fn capsule(radius_m: f64, length_m: f64) -> Self {
        Self::Capsule {
            radius_m,
            length_m,
            segments: 20,
            anchor: ObjectAnchor::Center,
        }
    }

    #[inline]
    pub const fn beacon_pin(height_m: f64, radius_m: f64) -> Self {
        Self::BeaconPin { height_m, radius_m }
    }

    /// Sets the vertical [`ObjectAnchor`] (`Base` vs `Center`) on anchorable geometries.
    #[inline]
    pub fn with_anchor(self, anchor: ObjectAnchor) -> Self {
        match self {
            Self::Box {
                width_m,
                depth_m,
                height_m,
                ..
            } => Self::Box {
                width_m,
                depth_m,
                height_m,
                anchor,
            },
            Self::Sphere {
                radius_m,
                segments,
                rings,
                ..
            } => Self::Sphere {
                radius_m,
                segments,
                rings,
                anchor,
            },
            Self::Cylinder {
                radius_top_m,
                radius_bottom_m,
                height_m,
                segments,
                ..
            } => Self::Cylinder {
                radius_top_m,
                radius_bottom_m,
                height_m,
                segments,
                anchor,
            },
            Self::Cone {
                radius_m,
                height_m,
                segments,
                ..
            } => Self::Cone {
                radius_m,
                height_m,
                segments,
                anchor,
            },
            Self::Capsule {
                radius_m,
                length_m,
                segments,
                ..
            } => Self::Capsule {
                radius_m,
                length_m,
                segments,
                anchor,
            },
            Self::Glb {
                bytes,
                unit_scale_m,
                ..
            } => Self::Glb {
                bytes,
                unit_scale_m,
                anchor,
            },
            other => other,
        }
    }

    #[inline]
    pub fn glb(bytes: impl Into<Arc<[u8]>>, unit_scale_m: f64) -> Self {
        Self::Glb {
            bytes: bytes.into(),
            unit_scale_m,
            anchor: ObjectAnchor::Base,
        }
    }

    #[inline]
    fn default_accent_mix(&self) -> f32 {
        match self {
            EarthGeometry::BeaconPin { .. } => 0.88,
            _ => 0.0,
        }
    }
}

/// Surface shading parameters for an [`EarthObject`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EarthMaterial {
    pub color: Color,
    pub roughness: f32,
    pub metalness: f32,
    pub emissive: Color,
    pub unlit: bool,
}

impl Default for EarthMaterial {
    fn default() -> Self {
        Self::standard(Color::rgb(56, 189, 248))
    }
}

impl EarthMaterial {
    #[inline]
    pub const fn standard(color: Color) -> Self {
        Self {
            color,
            roughness: 0.38,
            metalness: 0.15,
            emissive: Color::rgba(0, 0, 0, 0),
            unlit: false,
        }
    }

    #[inline]
    pub const fn metallic(color: Color, roughness: f32) -> Self {
        Self {
            color,
            roughness,
            metalness: 0.78,
            emissive: Color::rgba(0, 0, 0, 0),
            unlit: false,
        }
    }

    #[inline]
    pub const fn emissive(color: Color, emissive: Color) -> Self {
        Self {
            color,
            roughness: 0.35,
            metalness: 0.12,
            emissive,
            unlit: false,
        }
    }

    #[inline]
    pub const fn unlit(color: Color) -> Self {
        Self {
            color,
            roughness: 1.0,
            metalness: 0.0,
            emissive: Color::rgba(0, 0, 0, 0),
            unlit: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
enum ObjectMeshBinding {
    Inline(EarthGeometry),
    Shared(EarthMeshId),
}

/// Geodetic 3D object placed inside an [`EarthWorld`].
#[derive(Debug, Clone, PartialEq)]
pub struct EarthObject {
    pub position: GeoPosition,
    pub orientation: GeoOrientation,
    pub scale: [f64; 3],
    pub material: EarthMaterial,
    pub visible: bool,
    pub pickable: bool,
    /// Minimum apparent screen diameter in pixels (`0.0` = strictly physical scale only).
    /// When `> 0.0`, keeps small physical objects legible from orbital distances while rendering
    /// at exact `1:1` physical metre scale when zoomed in close.
    pub min_pixel_size: f32,
    pub label: Option<String>,
    mesh_binding: ObjectMeshBinding,
}

impl EarthObject {
    /// Creates a new 3D object at `position` with `geometry`.
    pub fn new(position: GeoPosition, geometry: EarthGeometry) -> Self {
        Self {
            position,
            orientation: GeoOrientation::default(),
            scale: [1.0, 1.0, 1.0],
            material: EarthMaterial::default(),
            visible: true,
            pickable: true,
            min_pixel_size: 0.0,
            label: None,
            mesh_binding: ObjectMeshBinding::Inline(geometry),
        }
    }

    /// Creates an instanced 3D object referencing a pre-registered [`EarthMeshId`].
    pub fn instanced(position: GeoPosition, mesh_id: EarthMeshId) -> Self {
        Self {
            position,
            orientation: GeoOrientation::default(),
            scale: [1.0, 1.0, 1.0],
            material: EarthMaterial::default(),
            visible: true,
            pickable: true,
            min_pixel_size: 0.0,
            label: None,
            mesh_binding: ObjectMeshBinding::Shared(mesh_id),
        }
    }

    #[inline]
    pub fn with_orientation(mut self, orientation: GeoOrientation) -> Self {
        self.orientation = orientation;
        self
    }

    #[inline]
    pub fn with_heading_deg(mut self, heading_deg: f64) -> Self {
        self.orientation.heading_deg = heading_deg;
        self
    }

    #[inline]
    pub fn with_scale(mut self, sx: f64, sy: f64, sz: f64) -> Self {
        self.scale = [sx, sy, sz];
        self
    }

    #[inline]
    pub fn with_uniform_scale(mut self, s: f64) -> Self {
        self.scale = [s, s, s];
        self
    }

    #[inline]
    pub fn with_material(mut self, material: EarthMaterial) -> Self {
        self.material = material;
        self
    }

    #[inline]
    pub fn with_color(mut self, color: Color) -> Self {
        self.material.color = color;
        self
    }

    #[inline]
    pub fn with_visible(mut self, visible: bool) -> Self {
        self.visible = visible;
        self
    }

    #[inline]
    pub fn with_pickable(mut self, pickable: bool) -> Self {
        self.pickable = pickable;
        self
    }

    #[inline]
    pub fn with_min_pixel_size(mut self, min_px: f32) -> Self {
        self.min_pixel_size = min_px.max(0.0);
        self
    }

    #[inline]
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Returns the inline [`EarthGeometry`] if this object was not created from a shared mesh ID.
    pub fn geometry(&self) -> Option<&EarthGeometry> {
        match &self.mesh_binding {
            ObjectMeshBinding::Inline(g) => Some(g),
            ObjectMeshBinding::Shared(_) => None,
        }
    }
}

/// Resolved world-space transform and local bounds for an [`EarthObject`] at a given camera/morph state.
#[derive(Debug, Clone)]
pub struct ResolvedObjectTransform {
    pub object_id: EarthObjectId,
    pub mesh_id: EarthMeshId,
    pub position: GeoPosition,
    /// Resolved world altitude above the WGS84 sea-level surface (`DEM + alt` when `Terrain` &
    /// `elevation_enabled`, else `alt`).
    pub resolved_altitude_m: f64,
    /// Sampled terrain elevation (`0.0` when `elevation_enabled == false`).
    pub terrain_elevation_m: f64,
    pub anchor_ecef: [f64; 3],
    /// Column-major 3x3 world transform `[col_x, col_y, col_z]` mapping local model metres to world metres.
    pub model_cols: [[f64; 3]; 3],
    pub local_aabb_min: [f32; 3],
    pub local_aabb_max: [f32; 3],
    pub world_bounding_radius_m: f64,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum PrimitiveCacheKey {
    Box {
        w: u64,
        d: u64,
        h: u64,
        anchor: ObjectAnchor,
    },
    Sphere {
        r: u64,
        seg: u32,
        rings: u32,
        anchor: ObjectAnchor,
    },
    Cylinder {
        rt: u64,
        rb: u64,
        h: u64,
        seg: u32,
        anchor: ObjectAnchor,
    },
    Cone {
        r: u64,
        h: u64,
        seg: u32,
        anchor: ObjectAnchor,
    },
    Capsule {
        r: u64,
        l: u64,
        seg: u32,
        anchor: ObjectAnchor,
    },
    BeaconPin {
        h: u64,
        r: u64,
    },
}

impl PrimitiveCacheKey {
    fn from_geometry(geom: &EarthGeometry) -> Option<Self> {
        match *geom {
            EarthGeometry::Box {
                width_m,
                depth_m,
                height_m,
                anchor,
            } => Some(Self::Box {
                w: width_m.to_bits(),
                d: depth_m.to_bits(),
                h: height_m.to_bits(),
                anchor,
            }),
            EarthGeometry::Sphere {
                radius_m,
                segments,
                rings,
                anchor,
            } => Some(Self::Sphere {
                r: radius_m.to_bits(),
                seg: segments,
                rings,
                anchor,
            }),
            EarthGeometry::Cylinder {
                radius_top_m,
                radius_bottom_m,
                height_m,
                segments,
                anchor,
            } => Some(Self::Cylinder {
                rt: radius_top_m.to_bits(),
                rb: radius_bottom_m.to_bits(),
                h: height_m.to_bits(),
                seg: segments,
                anchor,
            }),
            EarthGeometry::Cone {
                radius_m,
                height_m,
                segments,
                anchor,
            } => Some(Self::Cone {
                r: radius_m.to_bits(),
                h: height_m.to_bits(),
                seg: segments,
                anchor,
            }),
            EarthGeometry::Capsule {
                radius_m,
                length_m,
                segments,
                anchor,
            } => Some(Self::Capsule {
                r: radius_m.to_bits(),
                l: length_m.to_bits(),
                seg: segments,
                anchor,
            }),
            EarthGeometry::BeaconPin { height_m, radius_m } => Some(Self::BeaconPin {
                h: height_m.to_bits(),
                r: radius_m.to_bits(),
            }),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct CompiledMeshEntry {
    pub asset: Arc<RetainedMeshAsset>,
    pub local_aabb_min: [f32; 3],
    pub local_aabb_max: [f32; 3],
    pub local_radius_m: f64,
    pub accent_mix: f32,
}

#[derive(Debug, Clone)]
struct StoredObject {
    object: EarthObject,
    mesh_id: EarthMeshId,
    dynamic_polyline: Option<(Vec<GeoPosition>, f64, u64)>,
}

/// Self-contained planetary 3D object scene graph for `strata_earth`.
#[derive(Debug, Clone)]
pub struct EarthWorld {
    next_object_id: u64,
    next_mesh_id: u64,
    revision: u64,
    meshes: BTreeMap<EarthMeshId, CompiledMeshEntry>,
    primitive_cache: HashMap<PrimitiveCacheKey, EarthMeshId>,
    objects: BTreeMap<EarthObjectId, StoredObject>,
    cached_asset_list: Option<(u64, Arc<Vec<Arc<RetainedMeshAsset>>>)>,
    meshes_revision: u64,
}

impl Default for EarthWorld {
    fn default() -> Self {
        Self::new()
    }
}

impl EarthWorld {
    /// Creates an empty [`EarthWorld`].
    pub fn new() -> Self {
        Self {
            next_object_id: 1,
            next_mesh_id: 1,
            revision: 1,
            meshes: BTreeMap::new(),
            primitive_cache: HashMap::new(),
            objects: BTreeMap::new(),
            cached_asset_list: None,
            meshes_revision: 1,
        }
    }

    /// Monotonically increasing world mutation revision.
    #[inline]
    pub fn revision(&self) -> u64 {
        self.revision
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.objects.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.objects.is_empty()
    }

    /// Registers a reusable local-space [`EarthGeometry`] and returns its [`EarthMeshId`].
    pub fn register_mesh(&mut self, geometry: EarthGeometry) -> EarthMeshId {
        if let Some(key) = PrimitiveCacheKey::from_geometry(&geometry) {
            if let Some(&existing) = self.primitive_cache.get(&key) {
                return existing;
            }
            let id = self.insert_compiled_mesh(&geometry);
            self.primitive_cache.insert(key, id);
            return id;
        }
        self.insert_compiled_mesh(&geometry)
    }

    fn insert_compiled_mesh(&mut self, geometry: &EarthGeometry) -> EarthMeshId {
        let id = EarthMeshId(self.next_mesh_id);
        self.next_mesh_id += 1;
        let entry = build_mesh_entry(id.0, 1, geometry);
        self.meshes.insert(id, entry);
        self.meshes_revision = self.meshes_revision.wrapping_add(1);
        self.cached_asset_list = None;
        id
    }

    /// Spawns an [`EarthObject`] into the world and returns its stable [`EarthObjectId`].
    pub fn spawn(&mut self, mut object: EarthObject) -> EarthObjectId {
        let id = EarthObjectId(self.next_object_id);
        self.next_object_id += 1;

        let mut dynamic_polyline = None;
        let mesh_id = match &object.mesh_binding {
            ObjectMeshBinding::Shared(mid) => *mid,
            ObjectMeshBinding::Inline(geom) => {
                if let EarthGeometry::Polyline { points, radius_m } = geom {
                    if let Some(first) = points.first() {
                        object.position = *first;
                    }
                    dynamic_polyline = Some((points.clone(), *radius_m, 0));
                }
                self.register_mesh(geom.clone())
            }
        };

        self.objects.insert(
            id,
            StoredObject {
                object,
                mesh_id,
                dynamic_polyline,
            },
        );
        self.revision = self.revision.wrapping_add(1);
        id
    }

    /// Spawns an instanced object referencing a pre-registered [`EarthMeshId`].
    pub fn spawn_instanced(&mut self, mesh_id: EarthMeshId, position: GeoPosition) -> EarthObjectId {
        self.spawn(EarthObject::instanced(position, mesh_id))
    }

    pub fn get(&self, id: EarthObjectId) -> Option<&EarthObject> {
        self.objects.get(&id).map(|s| &s.object)
    }

    /// Mutates an [`EarthObject`] in place and increments the world revision.
    pub fn get_mut(&mut self, id: EarthObjectId) -> Option<&mut EarthObject> {
        self.revision = self.revision.wrapping_add(1);
        self.objects.get_mut(&id).map(|s| &mut s.object)
    }

    pub fn set_position(&mut self, id: EarthObjectId, position: GeoPosition) -> bool {
        if let Some(s) = self.objects.get_mut(&id) {
            s.object.position = position;
            self.revision = self.revision.wrapping_add(1);
            true
        } else {
            false
        }
    }

    pub fn set_orientation(&mut self, id: EarthObjectId, orientation: GeoOrientation) -> bool {
        if let Some(s) = self.objects.get_mut(&id) {
            s.object.orientation = orientation;
            self.revision = self.revision.wrapping_add(1);
            true
        } else {
            false
        }
    }

    pub fn set_scale(&mut self, id: EarthObjectId, scale: [f64; 3]) -> bool {
        if let Some(s) = self.objects.get_mut(&id) {
            s.object.scale = scale;
            self.revision = self.revision.wrapping_add(1);
            true
        } else {
            false
        }
    }

    pub fn set_material(&mut self, id: EarthObjectId, material: EarthMaterial) -> bool {
        if let Some(s) = self.objects.get_mut(&id) {
            s.object.material = material;
            self.revision = self.revision.wrapping_add(1);
            true
        } else {
            false
        }
    }

    pub fn set_visible(&mut self, id: EarthObjectId, visible: bool) -> bool {
        if let Some(s) = self.objects.get_mut(&id) {
            s.object.visible = visible;
            self.revision = self.revision.wrapping_add(1);
            true
        } else {
            false
        }
    }

    pub fn remove(&mut self, id: EarthObjectId) -> bool {
        let removed = self.objects.remove(&id).is_some();
        if removed {
            self.revision = self.revision.wrapping_add(1);
        }
        removed
    }

    pub fn clear(&mut self) {
        if !self.objects.is_empty() {
            self.objects.clear();
            self.revision = self.revision.wrapping_add(1);
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = (EarthObjectId, &EarthObject)> {
        self.objects.iter().map(|(&id, s)| (id, &s.object))
    }

    /// Returns `true` if any visible object uses `min_pixel_size > 0.0` (requiring distance-dependent scale).
    pub fn has_pixel_sized_objects(&self) -> bool {
        self.objects
            .values()
            .any(|s| s.object.visible && s.object.min_pixel_size > 0.0)
    }

    /// Evaluates the exact world transform for a single [`EarthObjectId`] under the given
    /// camera, heightmap toggle, surface morph, and live DEM sampler.
    pub fn resolve_object_transform(
        &self,
        id: EarthObjectId,
        camera_eye_ecef: DVec3,
        viewport_h: f64,
        fov_y_deg: f64,
        elevation_enabled: bool,
        morph: SurfaceMorphParams,
        dem_sampler: &dyn Fn(f64, f64) -> f64,
    ) -> Option<ResolvedObjectTransform> {
        let stored = self.objects.get(&id)?;
        let mesh = self.meshes.get(&stored.mesh_id)?;
        Some(compute_object_transform(
            id,
            stored,
            mesh,
            camera_eye_ecef,
            viewport_h,
            fov_y_deg,
            elevation_enabled,
            morph,
            dem_sampler,
        ))
    }

    /// Iterates all visible, pickable objects with their live world transforms and local meshes.
    pub(crate) fn iter_pickable_resolved<'a>(
        &'a self,
        camera_eye_ecef: DVec3,
        viewport_h: f64,
        fov_y_deg: f64,
        elevation_enabled: bool,
        morph: SurfaceMorphParams,
        dem_sampler: &'a dyn Fn(f64, f64) -> f64,
    ) -> impl Iterator<Item = (ResolvedObjectTransform, &'a CompiledMeshEntry)> + 'a {
        self.objects.iter().filter_map(move |(&id, stored)| {
            if !stored.object.visible || !stored.object.pickable {
                return None;
            }
            let mesh = self.meshes.get(&stored.mesh_id)?;
            let tf = compute_object_transform(
                id,
                stored,
                mesh,
                camera_eye_ecef,
                viewport_h,
                fov_y_deg,
                elevation_enabled,
                morph,
                dem_sampler,
            );
            Some((tf, mesh))
        })
    }

    /// Compiles all visible objects into GPU-ready shared [`RetainedMeshAsset`]s and hardware
    /// [`RetainedGeoInstance`]s.
    ///
    /// - Rigid meshes upload once and are never rebuilt on camera motion, morphing, or heightmap toggles.
    /// - [`AltitudeMode::Terrain`] evaluates live against `dem_sampler` when `elevation_enabled == true`,
    ///   and immediately resolves to `0.0 + altitude_m` (exact sea-level offset) when `elevation_enabled == false`.
    pub fn compile_retained(
        &mut self,
        camera_eye_ecef: DVec3,
        viewport_h: f64,
        fov_y_deg: f64,
        elevation_enabled: bool,
        morph: SurfaceMorphParams,
        dem_sampler: &dyn Fn(f64, f64) -> f64,
    ) -> (
        Arc<Vec<Arc<RetainedMeshAsset>>>,
        Arc<Vec<RetainedGeoInstance>>,
    ) {
        if self.objects.is_empty() {
            return (Arc::default(), Arc::default());
        }

        // Rebuild dynamic multi-point polylines only when their resolved waypoint positions change.
        let mut polylines_dirty = false;
        let poly_updates: Vec<(EarthMeshId, Vec<GeoPosition>, f64, u64)> = self
            .objects
            .values_mut()
            .filter_map(|stored| {
                if !stored.object.visible {
                    return None;
                }
                let (pts, radius_m, last_sig) = stored.dynamic_polyline.as_mut()?;
                let sig = polyline_state_hash(pts, *radius_m, elevation_enabled, morph, dem_sampler);
                if *last_sig == sig {
                    return None;
                }
                *last_sig = sig;
                Some((stored.mesh_id, pts.clone(), *radius_m, sig))
            })
            .collect();

        for (mesh_id, pts, radius_m, _sig) in poly_updates {
            let prev_rev = self
                .meshes
                .get(&mesh_id)
                .map(|m| m.asset.revision)
                .unwrap_or(1);
            let entry = build_polyline_world_mesh(
                mesh_id.0,
                prev_rev.wrapping_add(1),
                &pts,
                radius_m,
                elevation_enabled,
                morph,
                dem_sampler,
            );
            self.meshes.insert(mesh_id, entry);
            polylines_dirty = true;
        }
        if polylines_dirty {
            self.meshes_revision = self.meshes_revision.wrapping_add(1);
            self.cached_asset_list = None;
        }

        let assets = match &self.cached_asset_list {
            Some((rev, arc)) if *rev == self.meshes_revision => Arc::clone(arc),
            _ => {
                let list: Vec<Arc<RetainedMeshAsset>> =
                    self.meshes.values().map(|m| Arc::clone(&m.asset)).collect();
                let arc = Arc::new(list);
                self.cached_asset_list = Some((self.meshes_revision, Arc::clone(&arc)));
                arc
            }
        };

        let mut instances = Vec::with_capacity(self.objects.len());
        for (&id, stored) in &self.objects {
            if !stored.object.visible {
                continue;
            }
            let Some(mesh) = self.meshes.get(&stored.mesh_id) else {
                continue;
            };
            let tf = compute_object_transform(
                id,
                stored,
                mesh,
                camera_eye_ecef,
                viewport_h,
                fov_y_deg,
                elevation_enabled,
                morph,
                dem_sampler,
            );
            let mat = &stored.object.material;
            let base_color = [mat.color.r(), mat.color.g(), mat.color.b(), mat.color.a()];
            let emissive_scale = mat.emissive.a();
            let emissive_rgb = [
                mat.emissive.r() * emissive_scale,
                mat.emissive.g() * emissive_scale,
                mat.emissive.b() * emissive_scale,
            ];
            let accent_mix = stored
                .object
                .geometry()
                .map(|g| g.default_accent_mix())
                .unwrap_or(mesh.accent_mix);

            instances.push(RetainedGeoInstance {
                mesh_id: stored.mesh_id.0,
                anchor_ecef: tf.anchor_ecef,
                model_cols: [
                    [
                        tf.model_cols[0][0] as f32,
                        tf.model_cols[0][1] as f32,
                        tf.model_cols[0][2] as f32,
                    ],
                    [
                        tf.model_cols[1][0] as f32,
                        tf.model_cols[1][1] as f32,
                        tf.model_cols[1][2] as f32,
                    ],
                    [
                        tf.model_cols[2][0] as f32,
                        tf.model_cols[2][1] as f32,
                        tf.model_cols[2][2] as f32,
                    ],
                ],
                base_color,
                emissive_rgb,
                roughness: mat.roughness,
                metalness: mat.metalness,
                unlit: mat.unlit,
                accent_mix,
            });
        }

        (assets, Arc::new(instances))
    }

    /// Emits software-rasterizable RTC triangles for [`crate::EarthScene::render_to_framebuffer`].
    pub(crate) fn emit_cpu_meshes(
        &mut self,
        rtc_origin: DVec3,
        camera_eye_ecef: DVec3,
        viewport_h: f64,
        fov_y_deg: f64,
        elevation_enabled: bool,
        morph: SurfaceMorphParams,
        dem_sampler: &dyn Fn(f64, f64) -> f64,
    ) -> Vec<(strata_render::scene3d::Geometry3D, EarthMaterial)> {
        let _ = self.compile_retained(
            camera_eye_ecef,
            viewport_h,
            fov_y_deg,
            elevation_enabled,
            morph,
            dem_sampler,
        );
        let mut out = Vec::new();
        for (&id, stored) in &self.objects {
            if !stored.object.visible {
                continue;
            }
            let Some(mesh) = self.meshes.get(&stored.mesh_id) else {
                continue;
            };
            let tf = compute_object_transform(
                id,
                stored,
                mesh,
                camera_eye_ecef,
                viewport_h,
                fov_y_deg,
                elevation_enabled,
                morph,
                dem_sampler,
            );
            let mut builder = MeshBuilder::new();
            builder.set_origin(rtc_origin);
            let verts = &mesh.asset.vertices;
            let inds = &mesh.asset.indices;

            let c0 = DVec3::new(tf.model_cols[0][0], tf.model_cols[0][1], tf.model_cols[0][2]);
            let c1 = DVec3::new(tf.model_cols[1][0], tf.model_cols[1][1], tf.model_cols[1][2]);
            let c2 = DVec3::new(tf.model_cols[2][0], tf.model_cols[2][1], tf.model_cols[2][2]);
            let adj0 = c1.cross(c2);
            let adj1 = c2.cross(c0);
            let adj2 = c0.cross(c1);
            let anchor = DVec3::new(tf.anchor_ecef[0], tf.anchor_ecef[1], tf.anchor_ecef[2]);

            let base_v = builder.vertex_count() as u32;
            for v in verts.iter() {
                let p = v.position();
                let n = v.normal();
                let wp = anchor
                    .add(c0.scale(p[0] as f64))
                    .add(c1.scale(p[1] as f64))
                    .add(c2.scale(p[2] as f64));
                let wn_raw = adj0
                    .scale(n[0] as f64)
                    .add(adj1.scale(n[1] as f64))
                    .add(adj2.scale(n[2] as f64));
                let wn = if wn_raw.length() > 1e-9 {
                    wn_raw.normalize()
                } else {
                    DVec3::new(0.0, 0.0, 1.0)
                };
                builder.push_vertex(wp, [wn.x as f32, wn.y as f32, wn.z as f32], v.uv().into());
            }
            for tri in inds.chunks_exact(3) {
                builder.push_triangle(base_v + tri[0], base_v + tri[1], base_v + tri[2]);
            }
            if let Some(geom) = builder.finish() {
                out.push((geom, stored.object.material));
            }
        }
        out
    }
}

#[allow(clippy::too_many_arguments)]
fn compute_object_transform(
    id: EarthObjectId,
    stored: &StoredObject,
    mesh: &CompiledMeshEntry,
    camera_eye_ecef: DVec3,
    viewport_h: f64,
    fov_y_deg: f64,
    elevation_enabled: bool,
    morph: SurfaceMorphParams,
    dem_sampler: &dyn Fn(f64, f64) -> f64,
) -> ResolvedObjectTransform {
    let obj = &stored.object;
    let sampled_terrain_m = match obj.position.altitude_mode {
        AltitudeMode::Terrain if elevation_enabled => {
            dem_sampler(obj.position.lat_deg, obj.position.lon_deg)
        }
        _ => 0.0,
    };
    let resolved_altitude_m = obj
        .position
        .resolve_height_m(elevation_enabled, sampled_terrain_m);

    let (mut anchor_dvec, frame) = project_morphed_point_and_frame(
        obj.position.lat_deg,
        obj.position.lon_deg,
        resolved_altitude_m,
        morph,
    );

    let map_2d = morph.map_2d_morph.clamp(0.0, 1.0);
    if map_2d > 1e-5 {
        // Keep 3D objects cleanly above 2D map tiles with deterministic altitude ordering.
        let lift_m = (2.0 + resolved_altitude_m.max(0.0) * 0.02) * map_2d;
        anchor_dvec = anchor_dvec.add(frame.up.scale(lift_m));
    }

    if stored.dynamic_polyline.is_some() {
        return ResolvedObjectTransform {
            object_id: id,
            mesh_id: stored.mesh_id,
            position: obj.position,
            resolved_altitude_m,
            terrain_elevation_m: sampled_terrain_m,
            anchor_ecef: [anchor_dvec.x, anchor_dvec.y, anchor_dvec.z],
            model_cols: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            local_aabb_min: mesh.local_aabb_min,
            local_aabb_max: mesh.local_aabb_max,
            world_bounding_radius_m: mesh.local_radius_m,
        };
    }

    let max_user_scale = obj.scale[0]
        .abs()
        .max(obj.scale[1].abs())
        .max(obj.scale[2].abs())
        .max(1e-6);
    let phys_radius_m = (mesh.local_radius_m * max_user_scale).max(0.25);

    let mut pixel_boost = 1.0_f64;
    if obj.min_pixel_size > 0.0 && viewport_h > 1.0 {
        let dist_m = anchor_dvec.sub(camera_eye_ecef).length().max(1.0);
        let meters_per_px =
            (2.0 * dist_m * (fov_y_deg.to_radians() * 0.5).tan()) / viewport_h.max(1.0);
        let min_diameter_m = (obj.min_pixel_size as f64) * meters_per_px;
        let phys_diameter_m = (phys_radius_m * 2.0).max(1e-4);
        if min_diameter_m > phys_diameter_m {
            pixel_boost = min_diameter_m / phys_diameter_m;
        }
    }

    let sx = obj.scale[0] * pixel_boost;
    let sy = obj.scale[1] * pixel_boost;
    let sz = obj.scale[2] * pixel_boost * (1.0 - 0.72 * map_2d);

    let (rx, fy, uz) = obj.orientation.local_basis_enu();
    let enu_to_world = |local_enu: [f64; 3], s: f64| -> [f64; 3] {
        let v = frame
            .east
            .scale(local_enu[0] * s)
            .add(frame.north.scale(local_enu[1] * s))
            .add(frame.up.scale(local_enu[2] * s));
        [v.x, v.y, v.z]
    };

    let col0 = enu_to_world(rx, sx);
    let col1 = enu_to_world(fy, sy);
    let col2 = enu_to_world(uz, sz);

    ResolvedObjectTransform {
        object_id: id,
        mesh_id: stored.mesh_id,
        position: obj.position,
        resolved_altitude_m,
        terrain_elevation_m: sampled_terrain_m,
        anchor_ecef: [anchor_dvec.x, anchor_dvec.y, anchor_dvec.z],
        model_cols: [col0, col1, col2],
        local_aabb_min: mesh.local_aabb_min,
        local_aabb_max: mesh.local_aabb_max,
        world_bounding_radius_m: phys_radius_m * pixel_boost,
    }
}

fn polyline_state_hash(
    points: &[GeoPosition],
    radius_m: f64,
    elevation_enabled: bool,
    morph: SurfaceMorphParams,
    dem_sampler: &dyn Fn(f64, f64) -> f64,
) -> u64 {
    let mut h = 0xcbf29ce484222325_u64 ^ radius_m.to_bits() ^ u64::from(elevation_enabled);
    h = h.wrapping_mul(0x100000001b3) ^ morph.flat_blend.to_bits();
    h = h.wrapping_mul(0x100000001b3) ^ morph.map_2d_morph.to_bits();
    h = h.wrapping_mul(0x100000001b3) ^ morph.focus.lat_deg.to_bits();
    h = h.wrapping_mul(0x100000001b3) ^ morph.focus.lon_deg.to_bits();
    for p in points {
        let terr = match p.altitude_mode {
            AltitudeMode::Terrain if elevation_enabled => dem_sampler(p.lat_deg, p.lon_deg),
            _ => 0.0,
        };
        let resolved_q = (p.resolve_height_m(elevation_enabled, terr) * 100.0).round() as i64;
        h = h.wrapping_mul(0x100000001b3) ^ p.lat_deg.to_bits();
        h = h.wrapping_mul(0x100000001b3) ^ p.lon_deg.to_bits();
        h = h.wrapping_mul(0x100000001b3) ^ (resolved_q as u64);
    }
    h
}

fn finalize_mesh_entry(
    id: u64,
    revision: u64,
    vertices: Vec<RteTileVertex>,
    indices: Vec<u32>,
    accent_mix: f32,
) -> CompiledMeshEntry {
    let mut min_b = [f32::INFINITY; 3];
    let mut max_b = [f32::NEG_INFINITY; 3];
    let mut max_r2 = 0.0_f64;
    for v in &vertices {
        let p = v.position();
        for k in 0..3 {
            min_b[k] = min_b[k].min(p[k]);
            max_b[k] = max_b[k].max(p[k]);
        }
        let r2 = (p[0] as f64) * (p[0] as f64)
            + (p[1] as f64) * (p[1] as f64)
            + (p[2] as f64) * (p[2] as f64);
        if r2 > max_r2 {
            max_r2 = r2;
        }
    }
    if !min_b[0].is_finite() {
        min_b = [0.0; 3];
        max_b = [0.0; 3];
    }
    CompiledMeshEntry {
        asset: Arc::new(RetainedMeshAsset {
            id,
            revision,
            vertices: Arc::new(vertices),
            indices: Arc::new(indices),
        }),
        local_aabb_min: min_b,
        local_aabb_max: max_b,
        local_radius_m: max_r2.sqrt().max(0.05),
        accent_mix,
    }
}

fn build_mesh_entry(id: u64, revision: u64, geometry: &EarthGeometry) -> CompiledMeshEntry {
    match geometry {
        EarthGeometry::Box {
            width_m,
            depth_m,
            height_m,
            anchor,
        } => {
            let hx = (width_m.max(0.01) * 0.5) as f32;
            let hy = (depth_m.max(0.01) * 0.5) as f32;
            let h = height_m.max(0.01) as f32;
            let (z0, z1) = match anchor {
                ObjectAnchor::Center => (-h * 0.5, h * 0.5),
                ObjectAnchor::Base => (0.0, h),
            };
            let mut verts = Vec::with_capacity(24);
            let mut inds = Vec::with_capacity(36);
            let mut push_quad = |p: [[f32; 3]; 4], n: [f32; 3]| {
                let base = verts.len() as u32;
                verts.push(RteTileVertex::new(p[0], n, [0.0, 0.0]));
                verts.push(RteTileVertex::new(p[1], n, [1.0, 0.0]));
                verts.push(RteTileVertex::new(p[2], n, [1.0, 1.0]));
                verts.push(RteTileVertex::new(p[3], n, [0.0, 1.0]));
                inds.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
            };
            // +Z Top
            push_quad(
                [[-hx, -hy, z1], [hx, -hy, z1], [hx, hy, z1], [-hx, hy, z1]],
                [0.0, 0.0, 1.0],
            );
            // -Z Bottom
            push_quad(
                [[-hx, hy, z0], [hx, hy, z0], [hx, -hy, z0], [-hx, -hy, z0]],
                [0.0, 0.0, -1.0],
            );
            // +X East
            push_quad(
                [[hx, -hy, z0], [hx, hy, z0], [hx, hy, z1], [hx, -hy, z1]],
                [1.0, 0.0, 0.0],
            );
            // -X West
            push_quad(
                [[-hx, hy, z0], [-hx, -hy, z0], [-hx, -hy, z1], [-hx, hy, z1]],
                [-1.0, 0.0, 0.0],
            );
            // +Y North
            push_quad(
                [[hx, hy, z0], [-hx, hy, z0], [-hx, hy, z1], [hx, hy, z1]],
                [0.0, 1.0, 0.0],
            );
            // -Y South
            push_quad(
                [[-hx, -hy, z0], [hx, -hy, z0], [hx, -hy, z1], [-hx, -hy, z1]],
                [0.0, -1.0, 0.0],
            );
            finalize_mesh_entry(id, revision, verts, inds, 0.0)
        }
        EarthGeometry::Sphere {
            radius_m,
            segments,
            rings,
            anchor,
        } => {
            let r = radius_m.max(0.01) as f32;
            let seg = (*segments).max(8);
            let rng = (*rings).max(6);
            let z_off = match anchor {
                ObjectAnchor::Center => 0.0,
                ObjectAnchor::Base => r,
            };
            let mut verts = Vec::with_capacity(((seg + 1) * (rng + 1)) as usize);
            let mut inds = Vec::with_capacity((seg * rng * 6) as usize);
            for iy in 0..=rng {
                let v = iy as f32 / rng as f32;
                let phi = v * std::f32::consts::PI;
                let (sp, cp) = phi.sin_cos();
                for ix in 0..=seg {
                    let u = ix as f32 / seg as f32;
                    let theta = u * std::f32::consts::TAU;
                    let (st, ct) = theta.sin_cos();
                    let nx = sp * ct;
                    let ny = sp * st;
                    let nz = cp;
                    verts.push(RteTileVertex::new(
                        [nx * r, ny * r, nz * r + z_off],
                        [nx, ny, nz],
                        [u, v],
                    ));
                }
            }
            let stride = seg + 1;
            for iy in 0..rng {
                for ix in 0..seg {
                    let i00 = iy * stride + ix;
                    let i10 = i00 + 1;
                    let i01 = (iy + 1) * stride + ix;
                    let i11 = i01 + 1;
                    inds.extend_from_slice(&[i00, i01, i11, i00, i11, i10]);
                }
            }
            finalize_mesh_entry(id, revision, verts, inds, 0.0)
        }
        EarthGeometry::Cylinder {
            radius_top_m,
            radius_bottom_m,
            height_m,
            segments,
            anchor,
        } => build_frustum_mesh(
            id,
            revision,
            radius_top_m.max(0.0) as f32,
            radius_bottom_m.max(0.0) as f32,
            height_m.max(0.01) as f32,
            (*segments).max(6),
            *anchor,
        ),
        EarthGeometry::Cone {
            radius_m,
            height_m,
            segments,
            anchor,
        } => build_frustum_mesh(
            id,
            revision,
            0.0,
            radius_m.max(0.01) as f32,
            height_m.max(0.01) as f32,
            (*segments).max(6),
            *anchor,
        ),
        EarthGeometry::Capsule {
            radius_m,
            length_m,
            segments,
            anchor,
        } => {
            let r = radius_m.max(0.01) as f32;
            let half_l = (length_m.max(0.0) * 0.5) as f32;
            let seg = (*segments).max(8);
            let hemi_rings = 6u32;
            let total_rings = hemi_rings * 2 + 1;
            let z_shift = match anchor {
                ObjectAnchor::Center => 0.0,
                ObjectAnchor::Base => half_l + r,
            };
            let mut verts = Vec::with_capacity(((seg + 1) * (total_rings + 1)) as usize);
            let mut inds = Vec::with_capacity((seg * total_rings * 6) as usize);
            for ir in 0..=total_rings {
                let (phi, cz) = if ir <= hemi_rings {
                    let t = ir as f32 / hemi_rings as f32;
                    (t * std::f32::consts::FRAC_PI_2, half_l)
                } else {
                    let t = (ir - hemi_rings - 1) as f32 / hemi_rings as f32;
                    (
                        std::f32::consts::FRAC_PI_2 + t * std::f32::consts::FRAC_PI_2,
                        -half_l,
                    )
                };
                let (sp, cp) = phi.sin_cos();
                let v = ir as f32 / total_rings as f32;
                for ix in 0..=seg {
                    let u = ix as f32 / seg as f32;
                    let (st, ct) = (u * std::f32::consts::TAU).sin_cos();
                    let nx = sp * ct;
                    let ny = sp * st;
                    let nz = cp;
                    verts.push(RteTileVertex::new(
                        [nx * r, ny * r, cz + nz * r + z_shift],
                        [nx, ny, nz],
                        [u, v],
                    ));
                }
            }
            let stride = seg + 1;
            for ir in 0..total_rings {
                for ix in 0..seg {
                    let i00 = ir * stride + ix;
                    let i10 = i00 + 1;
                    let i01 = (ir + 1) * stride + ix;
                    let i11 = i01 + 1;
                    inds.extend_from_slice(&[i00, i01, i11, i00, i11, i10]);
                }
            }
            finalize_mesh_entry(id, revision, verts, inds, 0.0)
        }
        EarthGeometry::BeaconPin { height_m, radius_m } => {
            build_beacon_pin_mesh(id, revision, height_m.max(0.1) as f32, radius_m.max(0.05) as f32)
        }
        EarthGeometry::Polyline { points, radius_m } => build_polyline_world_mesh(
            id,
            revision,
            points,
            *radius_m,
            true,
            SurfaceMorphParams::default(),
            &|_, _| 0.0,
        ),
        EarthGeometry::ExtrudedPolygon {
            footprint_enu_m,
            height_m,
        } => build_extruded_polygon_mesh(id, revision, footprint_enu_m, height_m.max(0.01) as f32),
        EarthGeometry::CustomMesh { vertices, indices } => {
            let verts = vertices
                .iter()
                .map(|v| RteTileVertex::new(v.position, v.normal, v.uv))
                .collect();
            finalize_mesh_entry(id, revision, verts, indices.clone(), 0.0)
        }
        EarthGeometry::Glb {
            bytes,
            unit_scale_m,
            anchor,
        } => {
            let parsed = crate::gltf::parse_glb_mesh(bytes).unwrap_or_else(|_| crate::gltf::WeatherMesh {
                positions: vec![[0.0, 0.0, 0.5], [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5]],
                normals: vec![[0.0, 0.0, 1.0]; 3],
                indices: vec![0, 1, 2],
            });
            let scale = unit_scale_m.max(1e-4) as f32;
            let min_z = parsed
                .positions
                .iter()
                .map(|p| p[2])
                .fold(f32::INFINITY, f32::min);
            let z_shift = match anchor {
                ObjectAnchor::Center => 0.0,
                ObjectAnchor::Base => {
                    if min_z.is_finite() {
                        -min_z * scale
                    } else {
                        0.0
                    }
                }
            };
            let verts: Vec<RteTileVertex> = parsed
                .positions
                .iter()
                .enumerate()
                .map(|(i, p)| {
                    let n = parsed.normals.get(i).copied().unwrap_or([0.0, 0.0, 1.0]);
                    RteTileVertex::new(
                        [p[0] * scale, p[1] * scale, p[2] * scale + z_shift],
                        n,
                        [0.0, 0.0],
                    )
                })
                .collect();
            finalize_mesh_entry(id, revision, verts, parsed.indices, 0.0)
        }
    }
}

fn build_frustum_mesh(
    id: u64,
    revision: u64,
    r_top: f32,
    r_bot: f32,
    height: f32,
    seg: u32,
    anchor: ObjectAnchor,
) -> CompiledMeshEntry {
    let (z0, z1) = match anchor {
        ObjectAnchor::Center => (-height * 0.5, height * 0.5),
        ObjectAnchor::Base => (0.0, height),
    };
    let mut verts = Vec::new();
    let mut inds = Vec::new();
    let slope_z = (r_bot - r_top) / height.max(1e-5);

    let side_base = 0u32;
    for ix in 0..=seg {
        let u = ix as f32 / seg as f32;
        let (st, ct) = (u * std::f32::consts::TAU).sin_cos();
        let n_len = (1.0 + slope_z * slope_z).sqrt();
        let n = [ct / n_len, st / n_len, slope_z / n_len];
        verts.push(RteTileVertex::new([ct * r_bot, st * r_bot, z0], n, [u, 0.0]));
        verts.push(RteTileVertex::new([ct * r_top, st * r_top, z1], n, [u, 1.0]));
    }
    for ix in 0..seg {
        let b0 = side_base + ix * 2;
        let t0 = b0 + 1;
        let b1 = b0 + 2;
        let t1 = b0 + 3;
        inds.extend_from_slice(&[b0, b1, t1, b0, t1, t0]);
    }

    if r_top > 1e-5 {
        let center_idx = verts.len() as u32;
        verts.push(RteTileVertex::new([0.0, 0.0, z1], [0.0, 0.0, 1.0], [0.5, 0.5]));
        for ix in 0..=seg {
            let (st, ct) = ((ix as f32 / seg as f32) * std::f32::consts::TAU).sin_cos();
            verts.push(RteTileVertex::new(
                [ct * r_top, st * r_top, z1],
                [0.0, 0.0, 1.0],
                [0.5 + 0.5 * ct, 0.5 + 0.5 * st],
            ));
        }
        for ix in 0..seg {
            inds.extend_from_slice(&[center_idx, center_idx + 1 + ix, center_idx + 2 + ix]);
        }
    }

    if r_bot > 1e-5 {
        let center_idx = verts.len() as u32;
        verts.push(RteTileVertex::new(
            [0.0, 0.0, z0],
            [0.0, 0.0, -1.0],
            [0.5, 0.5],
        ));
        for ix in 0..=seg {
            let (st, ct) = ((ix as f32 / seg as f32) * std::f32::consts::TAU).sin_cos();
            verts.push(RteTileVertex::new(
                [ct * r_bot, st * r_bot, z0],
                [0.0, 0.0, -1.0],
                [0.5 + 0.5 * ct, 0.5 + 0.5 * st],
            ));
        }
        for ix in 0..seg {
            inds.extend_from_slice(&[center_idx, center_idx + 2 + ix, center_idx + 1 + ix]);
        }
    }

    finalize_mesh_entry(id, revision, verts, inds, 0.0)
}

/// Builds a precision 3D surveyor beacon pin whose tip touches `(0, 0, 0)` and rises to `height`.
/// Includes a bright collar band (`uv.y = 1.0`) for crisp legibility against both bright snow and dark ocean.
fn build_beacon_pin_mesh(id: u64, revision: u64, height: f32, radius: f32) -> CompiledMeshEntry {
    let seg = 20u32;
    // Profile rings `(z_frac, r_frac, accent_v)` from sharp ground tip `(0.0)` up to crown `(1.0)`:
    let profile: &[(f32, f32, f32)] = &[
        (0.00, 0.00, 0.0),
        (0.42, 0.34, 0.0),
        (0.56, 0.72, 0.0),
        (0.63, 0.88, 1.0), // crisp white surveyor ring
        (0.71, 0.98, 1.0), // crisp white surveyor ring
        (0.78, 1.00, 0.0),
        (0.88, 0.86, 0.0),
        (0.95, 0.54, 0.25),
        (1.00, 0.00, 0.45),
    ];

    let rings = profile.len();
    let mut verts = Vec::with_capacity(rings * (seg as usize + 1));
    let mut inds = Vec::with_capacity((rings - 1) * seg as usize * 6);

    for (ir, &(zf, rf, accent_v)) in profile.iter().enumerate() {
        let z = zf * height;
        let r = rf * radius;
        let prev = profile[ir.saturating_sub(1)];
        let next = profile[(ir + 1).min(rings - 1)];
        let dz = ((next.0 - prev.0) * height).max(1e-4);
        let dr = (prev.1 - next.1) * radius;
        let n_len = (dz * dz + dr * dr).sqrt().max(1e-5);
        let nr = dz / n_len;
        let nz = dr / n_len;

        for ix in 0..=seg {
            let u = ix as f32 / seg as f32;
            let (st, ct) = (u * std::f32::consts::TAU).sin_cos();
            let normal = if ir + 1 == rings {
                [0.0, 0.0, 1.0]
            } else {
                [ct * nr, st * nr, nz]
            };
            verts.push(RteTileVertex::new([ct * r, st * r, z], normal, [u, accent_v]));
        }
    }

    let stride = seg + 1;
    for ir in 0..(rings as u32 - 1) {
        for ix in 0..seg {
            let i00 = ir * stride + ix;
            let i10 = i00 + 1;
            let i01 = (ir + 1) * stride + ix;
            let i11 = i01 + 1;
            inds.extend_from_slice(&[i00, i10, i11, i00, i11, i01]);
        }
    }

    finalize_mesh_entry(id, revision, verts, inds, 0.88)
}

fn build_extruded_polygon_mesh(
    id: u64,
    revision: u64,
    footprint: &[[f64; 2]],
    height: f32,
) -> CompiledMeshEntry {
    if footprint.len() < 3 {
        return finalize_mesh_entry(id, revision, Vec::new(), Vec::new(), 0.0);
    }
    let pts: Vec<[f32; 2]> = footprint
        .iter()
        .map(|p| [p[0] as f32, p[1] as f32])
        .collect();

    // Ensure counter-clockwise winding in XY.
    let mut signed_area2 = 0.0f32;
    for i in 0..pts.len() {
        let a = pts[i];
        let b = pts[(i + 1) % pts.len()];
        signed_area2 += a[0] * b[1] - b[0] * a[1];
    }
    let ccw_pts: Vec<[f32; 2]> = if signed_area2 >= 0.0 {
        pts
    } else {
        pts.into_iter().rev().collect()
    };

    let mut verts = Vec::new();
    let mut inds = Vec::new();

    // Perimeter walls
    for i in 0..ccw_pts.len() {
        let a = ccw_pts[i];
        let b = ccw_pts[(i + 1) % ccw_pts.len()];
        let dx = b[0] - a[0];
        let dy = b[1] - a[1];
        let len = (dx * dx + dy * dy).sqrt().max(1e-5);
        let n = [dy / len, -dx / len, 0.0];
        let base = verts.len() as u32;
        verts.push(RteTileVertex::new([a[0], a[1], 0.0], n, [0.0, 0.0]));
        verts.push(RteTileVertex::new([b[0], b[1], 0.0], n, [1.0, 0.0]));
        verts.push(RteTileVertex::new([b[0], b[1], height], n, [1.0, 1.0]));
        verts.push(RteTileVertex::new([a[0], a[1], height], n, [0.0, 1.0]));
        inds.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }

    // Ear-clipping triangulation for roof (+Z) and floor (-Z)
    let roof_base = verts.len() as u32;
    for p in &ccw_pts {
        verts.push(RteTileVertex::new([p[0], p[1], height], [0.0, 0.0, 1.0], [0.0, 0.0]));
    }
    let floor_base = verts.len() as u32;
    for p in &ccw_pts {
        verts.push(RteTileVertex::new([p[0], p[1], 0.0], [0.0, 0.0, -1.0], [0.0, 0.0]));
    }

    let mut rem: Vec<usize> = (0..ccw_pts.len()).collect();
    let mut guard = 0usize;
    while rem.len() >= 3 && guard < ccw_pts.len() * ccw_pts.len() + 8 {
        guard += 1;
        let mut clipped = false;
        for ri in 0..rem.len() {
            let i_prev = rem[(ri + rem.len() - 1) % rem.len()];
            let i_curr = rem[ri];
            let i_next = rem[(ri + 1) % rem.len()];
            let a = ccw_pts[i_prev];
            let b = ccw_pts[i_curr];
            let c = ccw_pts[i_next];
            let cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
            if cross <= 1e-6 {
                continue;
            }
            let any_inside = rem.iter().any(|&k| {
                k != i_prev && k != i_curr && k != i_next && point_in_tri_2d(ccw_pts[k], a, b, c)
            });
            if !any_inside {
                inds.extend_from_slice(&[
                    roof_base + i_prev as u32,
                    roof_base + i_curr as u32,
                    roof_base + i_next as u32,
                ]);
                inds.extend_from_slice(&[
                    floor_base + i_prev as u32,
                    floor_base + i_next as u32,
                    floor_base + i_curr as u32,
                ]);
                rem.remove(ri);
                clipped = true;
                break;
            }
        }
        if !clipped {
            let i0 = rem[0] as u32;
            for k in 1..(rem.len() - 1) {
                let i1 = rem[k] as u32;
                let i2 = rem[k + 1] as u32;
                inds.extend_from_slice(&[roof_base + i0, roof_base + i1, roof_base + i2]);
                inds.extend_from_slice(&[floor_base + i0, floor_base + i2, floor_base + i1]);
            }
            break;
        }
    }

    finalize_mesh_entry(id, revision, verts, inds, 0.0)
}

#[inline]
fn point_in_tri_2d(p: [f32; 2], a: [f32; 2], b: [f32; 2], c: [f32; 2]) -> bool {
    let e0 = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]);
    let e1 = (c[0] - b[0]) * (p[1] - b[1]) - (c[1] - b[1]) * (p[0] - b[0]);
    let e2 = (a[0] - c[0]) * (p[1] - c[1]) - (a[1] - c[1]) * (p[0] - c[0]);
    e0 >= -1e-5 && e1 >= -1e-5 && e2 >= -1e-5
}

fn build_polyline_world_mesh(
    id: u64,
    revision: u64,
    points: &[GeoPosition],
    radius_m: f64,
    elevation_enabled: bool,
    morph: SurfaceMorphParams,
    dem_sampler: &dyn Fn(f64, f64) -> f64,
) -> CompiledMeshEntry {
    if points.len() < 2 {
        return finalize_mesh_entry(id, revision, Vec::new(), Vec::new(), 0.0);
    }
    let resolve_pt = |gp: &GeoPosition| -> (DVec3, LocalFrame) {
        let terr = match gp.altitude_mode {
            AltitudeMode::Terrain if elevation_enabled => dem_sampler(gp.lat_deg, gp.lon_deg),
            _ => 0.0,
        };
        let h = gp.resolve_height_m(elevation_enabled, terr);
        project_morphed_point_and_frame(gp.lat_deg, gp.lon_deg, h, morph)
    };

    let (anchor, _) = resolve_pt(&points[0]);
    let r = radius_m.max(0.05);
    let sides = 8usize;
    let mut verts = Vec::with_capacity(points.len() * sides);
    let mut inds = Vec::with_capacity((points.len() - 1) * sides * 6);

    let world_pts: Vec<(DVec3, LocalFrame)> = points.iter().map(resolve_pt).collect();

    for (i, &(pt, frame)) in world_pts.iter().enumerate() {
        let tangent = if i == 0 {
            world_pts[1].0.sub(pt)
        } else if i + 1 == world_pts.len() {
            pt.sub(world_pts[i - 1].0)
        } else {
            world_pts[i + 1].0.sub(world_pts[i - 1].0)
        };
        let t_dir = if tangent.length() > 1e-6 {
            tangent.normalize()
        } else {
            frame.north
        };
        let side_a_raw = t_dir.cross(frame.up);
        let side_a = if side_a_raw.length() > 1e-6 {
            side_a_raw.normalize()
        } else {
            frame.east
        };
        let side_b = side_a.cross(t_dir).normalize();
        let rel_center = pt.sub(anchor);

        for s in 0..sides {
            let u = s as f32 / sides as f32;
            let (sn, cs) = (u * std::f32::consts::TAU).sin_cos();
            let n = side_a.scale(cs as f64).add(side_b.scale(sn as f64));
            let pos = rel_center.add(n.scale(r));
            verts.push(RteTileVertex::new(
                [pos.x as f32, pos.y as f32, pos.z as f32],
                [n.x as f32, n.y as f32, n.z as f32],
                [u, i as f32 / (world_pts.len() - 1) as f32],
            ));
        }
    }

    let stride = sides as u32;
    for i in 0..(world_pts.len() as u32 - 1) {
        for s in 0..stride {
            let s_next = (s + 1) % stride;
            let i00 = i * stride + s;
            let i10 = i * stride + s_next;
            let i01 = (i + 1) * stride + s;
            let i11 = (i + 1) * stride + s_next;
            inds.extend_from_slice(&[i00, i10, i11, i00, i11, i01]);
        }
    }

    finalize_mesh_entry(id, revision, verts, inds, 0.0)
}
