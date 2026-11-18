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
pub mod assets_cdn;
pub mod astro;
pub mod basemap;
pub mod controller;
pub mod descriptor;
pub mod geodesy;
pub mod gltf;
pub mod json;
pub mod labels;
pub mod plugins;
pub mod satellites;
pub mod scene;
pub mod skybox;
pub mod tiles;
#[cfg(feature = "widget")]
pub mod widget;
pub mod worker;

pub use assets_cdn::{
    sha256_hex, versioned_legacy_filename, AssetManager, AssetManifest, AssetManifestEntry,
    AssetSourceConfig, DEFAULT_ASSETS_LEGACY_TAG, DEFAULT_ASSETS_RELEASE_TAG,
    DEFAULT_ASSET_CDN_BASE, DEFAULT_ASSET_GITHUB_REPO,
};
pub use labels::{
    builtin_world_labels, label_style, parse_overpass_labels_json, project_earth_labels,
    EarthLabelIcon, EarthLabelKind, EarthMapLabel, EarthScreenProjector, LiveMapLabelStore,
    ProjectedEarthLabel,
};

pub use acceptance::{
    evaluate as evaluate_acceptance, AcceptanceCriteria, AcceptanceReport, PixelDelta,
};
pub use astro::{
    build_environment_assets, build_environment_assets_with_skybox, embedded_bsc5_stars,
    embedded_clouds_fallback, embedded_land_mask, embedded_moon_texture, equatorial_to_ecef,
    greenwich_mean_sidereal_time_rad, icrf_to_ecef_matrix, moon_direction_from_unix_utc,
    sun_direction_from_local_time, sun_direction_from_subsolar_lat_lon,
    sun_direction_from_system_clock, sun_direction_from_unix_utc, system_clock_time,
    unix_to_julian_date, CloudMapResolution, GlobalCloudMapStore, LocalClockTime,
    DEFAULT_CLOUD_MAP_URL,
};
pub use skybox::{
    blend_composite_cubemap, cubemap_face_dir, decode_ssky, encode_ssky_jpeg, encode_ssky_png,
    healpix_nest_index, icrs_to_galactic, project_hips_allsky_to_cubemap,
    project_hips_tiles_to_cubemap, vec_to_healpix_face_uv, HipsFrame, SpaceSkyboxQuality,
    SpaceSkyboxStore, SpaceSkyboxSurvey,
};
pub use basemap::{
    embedded_base_tile, encode_sebm_pack, BaseWorldMapPack, EarthDefaultTileSource,
    EmbeddedBaseMapSource, MAX_EMBEDDED_BASE_LEVEL,
};
pub use descriptor::{
    compare, replay, CameraDelta, DeltaSummary, DescriptorCamera, DescriptorCelestial,
    DescriptorClouds, DescriptorDelta, DescriptorMarker, DescriptorPatch, DescriptorRing,
    DescriptorSphere, DescriptorTile, FrameDescriptor, PatchDelta, PatchMatch, TileDelta,
    TileSupply, DESCRIPTOR_FORMAT, DESCRIPTOR_VERSION, DESCRIPTOR_VERSION_V1,
};
pub use geodesy::{
    build_tile_local_rtc_mesh, build_tile_local_rtc_mesh_morphed, clamp_latitude,
    compute_seamless_25d_blend, compute_tile_sse, depth_range, ecef_to_geodetic, geodetic_to_ecef,
    horizon_distance_m, is_tile_horizon_culled, is_tile_in_frustum, level_for_altitude,
    mercator_gy_to_lat_deg, pick_geodetic, pick_geodetic_with_up, raycast_ellipsoid,
    resolve_tile_imagery, retained_camera_uniforms, retained_camera_uniforms_with_up,
    rte_gpu_relative_position, rte_relative_position_f32, screen_to_ray, screen_to_ray_with_up,
    split_dvec3_to_high_low, split_f64_to_high_low, tile_bounding_sphere,
    tile_bounding_sphere_projected, tile_bounds, tile_bounds_mercator, traverse_quadtree_lod,
    traverse_quadtree_lod_2d_flat, traverse_quadtree_lod_frustum6,
    traverse_quadtree_lod_frustum6_projected, traverse_quadtree_lod_frustum6_projected_with_up,
    wrap_longitude, CameraPlacement, EarthCamera, Frustum6, Geodetic, LocalFrame, RteEyeUniforms,
    SurfaceMorphParams, TileKey, TileProjection, TileTextureResolution, GLOBE_ALTITUDE_M,
    MAX_QUADTREE_LEAVES,
};
pub use gltf::{
    build_25d_weather_meshes, cc0_cloud_mesh, cc0_rain_mesh, cc0_snow_mesh, parse_glb_mesh,
    CustomWeatherMeshes, WeatherConfig, WeatherMesh,
};
pub use plugins::{
    classify_non_land_water, sample_land_mask_at, CelestialPlugin, CityLightsPlugin,
    EarthDynamicOverlayContext, EarthPlugin, EarthPluginFrameContext, EarthPluginHooks,
    EarthPluginRegistry, EarthTextureInjector, EarthTileBuildContext, EarthWorldBuildContext,
    LoadedTileImagerySnapshot, OceanWavesPlugin, SatellitesPlugin, WeatherPlugin,
};
pub use satellites::{
    build_3d_satellites_meshes, embedded_satellite_elements, parse_celestrak_gp_json,
    propagate_satellite_ecef, PropagatedSatellite, SatelliteCatalogStore, SatelliteElement,
    DEFAULT_CELESTRAK_STATIONS_URL, DEFAULT_CELESTRAK_VISUAL_URL,
};
pub use scene::{
    building_triangle_count, celestial_bodies, compute_adaptive_skirt_depth,
    parse_overpass_buildings, requested_tile, sky_gradient, surrounding_patches, surrounding_tiles,
    terrain_skirt_triangle_count, terrain_triangle_count, terrain_triangle_count_with_skirts,
    tile_patch_span_deg, BuildingFootprint, BuildingSource, EarthInput, EarthLowLevelState,
    EarthScene, EarthViewMode, EntityMarker, FrameStats, LayerToggles, MemoryBuildingSource,
    NetworkBuildingSource, OceanWavesConfig, PatchBounds, PatchFrame, SpaceEnvironmentConfig,
    SunDirectionMode, SurfaceMode,
};
pub use scene::{horizontal_fov_deg, vertical_fov_deg};
pub use tiles::{
    decode_png_to_rgba, decode_terrain_rgb, decode_terrarium, format_slippy_url, format_tile_url,
    is_unavailable_placeholder_tile, tile_key_to_quadkey, tile_sampler, DirectoryTileSource,
    ElevationEncoding, HttpTransport, LruTileCache, LruTileCacheSource, MemoryTileSource,
    MockHttpTransport, NetworkTileSource, NetworkTileStats, StandardHttpTransport, TextureSlot,
    TileElevation, TileElevationPreset, TileError, TileImage, TileProviderConfig,
    TileProviderPreset, TileRegistry, TileSource, DEFAULT_MAX_CACHE_BYTES,
};
pub use controller::{
    EarthOrbitController, DEFAULT_FOV_Y_DEG, GROUND_CLEARANCE_M, LOCAL_TRANSITION_RANGE_M,
    MAX_RANGE_M, MIN_RANGE_M,
};
#[cfg(feature = "widget")]
pub use widget::{EarthOverlayContext, EarthOverlayPlugin, EarthView, EarthViewState};
pub use worker::{
    EarthTileWorker, EarthViewport, EarthViewportStats, RetainedBuildStats, RetainedTerrainBuilder,
};
