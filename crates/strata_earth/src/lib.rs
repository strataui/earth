//! Earth scene plugin.
//!
//! Assembles the renderer state for one Earth frame: camera, background,
//! terrain tiles, relief walls, building prisms, and the 2D overlay pass. The
//! crate is additive: it builds a [`strata_render::scene3d::Scene3D`] using only
//! the public engine surface, so existing 3D users are unaffected.
//!
//! Design rules enforced by tests:
//!
//! - No `egui`, widget, or shell-command dependency; the plugin is pure Rust
//!   over the engine.
//! - Tile decisions are separate from texture upload ([`tiles::TileSource`] vs
//!   [`tiles::TileRegistry`]).
//! - World coordinates stay f64 until the render hand-off.
//! - Mesh rebuilds reuse buffers; steady-state frames do not allocate.

pub mod acceptance;
pub mod descriptor;
pub mod geodesy;
pub mod json;
pub mod scene;
pub mod tiles;

pub use acceptance::{
    evaluate as evaluate_acceptance, AcceptanceCriteria, AcceptanceReport, PixelDelta,
};
pub use descriptor::{
    compare, replay, CameraDelta, DeltaSummary, DescriptorCamera, DescriptorCelestial,
    DescriptorClouds, DescriptorDelta, DescriptorMarker, DescriptorPatch, DescriptorRing,
    DescriptorSphere, DescriptorTile, FrameDescriptor, PatchDelta, PatchMatch, TileDelta,
    TileSupply, DESCRIPTOR_FORMAT, DESCRIPTOR_VERSION, DESCRIPTOR_VERSION_V1,
};
pub use geodesy::{
    clamp_latitude, depth_range, ecef_to_geodetic, geodetic_to_ecef, horizon_distance_m,
    level_for_altitude, pick_geodetic, raycast_ellipsoid, resolve_tile_imagery, screen_to_ray,
    wrap_longitude, CameraPlacement, EarthCamera, Geodetic, LocalFrame, TileKey,
    TileTextureResolution, GLOBE_ALTITUDE_M,
};
pub use scene::{
    building_triangle_count, celestial_bodies, requested_tile, sky_gradient, surrounding_patches,
    surrounding_tiles, terrain_triangle_count, tile_patch_span_deg, EarthInput, EarthScene,
    EntityMarker, FrameStats, LayerToggles, PatchBounds, PatchFrame, SurfaceMode,
};
pub use scene::{horizontal_fov_deg, vertical_fov_deg};
pub use tiles::{
    tile_sampler, DirectoryTileSource, MemoryTileSource, TextureSlot, TileElevation, TileError,
    TileImage, TileRegistry, TileSource,
};
