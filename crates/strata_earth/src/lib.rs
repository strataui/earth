//! Strata Earth planetary 3D engine: native `f64` geodetic scene graph ([`EarthWorld`]),
//! hardware-instanced double-single RTE rendering, terrain/mode-aware raycasting, and `<EarthView>`.

pub mod acceptance;
pub mod assets_cdn;
pub mod astro;
pub mod basemap;
pub mod controller;
pub mod descriptor;
#[cfg(feature = "widget")]
pub mod events;
pub mod geodesy;
pub mod gltf;
pub mod json;
pub mod labels;
pub mod plugins;
pub mod precipitation;
pub mod raycast;
pub mod satellites;
pub mod scene;
pub mod skybox;
pub mod tiles;
#[cfg(feature = "widget")]
pub mod widget;
pub mod worker;
pub mod world;

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
    DescriptorClouds, DescriptorDelta, DescriptorObject, DescriptorPatch, DescriptorRing,
    DescriptorSphere, DescriptorTile, FrameDescriptor, PatchDelta, PatchMatch, TileDelta,
    TileSupply, DESCRIPTOR_FORMAT, DESCRIPTOR_VERSION, DESCRIPTOR_VERSION_V1,
};
#[cfg(feature = "widget")]
pub use events::{
    EarthCameraState, EarthControlsConfig, EarthKeyEvent, EarthModifiers, EarthMouseButton,
    EarthPointerEvent, EarthWheelEvent,
};
pub use geodesy::{
    build_tile_local_rtc_mesh, build_tile_local_rtc_mesh_morphed, clamp_latitude,
    compute_seamless_25d_blend, compute_tile_sse, depth_range, ecef_to_geodetic, geodetic_to_ecef,
    horizon_distance_m, is_tile_horizon_culled, is_tile_in_frustum, level_for_altitude,
    mercator_gy_to_lat_deg, pick_geodetic, pick_geodetic_with_up, project_morphed_point_and_frame,
    raycast_ellipsoid, resolve_tile_imagery, retained_camera_uniforms,
    retained_camera_uniforms_with_up, rte_gpu_relative_position, rte_relative_position_f32,
    screen_to_ray, screen_to_ray_with_up, split_dvec3_to_high_low, split_f64_to_high_low,
    stabilized_surface_morph, tile_bounding_sphere, tile_bounding_sphere_projected, tile_bounds,
    tile_bounds_mercator, traverse_quadtree_lod, traverse_quadtree_lod_2d_flat,
    traverse_quadtree_lod_frustum6, traverse_quadtree_lod_frustum6_projected,
    traverse_quadtree_lod_frustum6_projected_with_up, unproject_morphed_surface_lat_lon,
    wrap_longitude, AltitudeMode, CameraPlacement, EarthCamera, Frustum6, GeoPosition, Geodetic,
    LocalFrame, RteEyeUniforms, SurfaceMorphParams, TileKey, TileProjection,
    TileTextureResolution, GLOBE_ALTITUDE_M, MAX_QUADTREE_LEAVES,
};
pub use gltf::{parse_glb_mesh, WeatherConfig, WeatherMesh};
pub use plugins::{
    classify_non_land_water, sample_land_mask_at, CelestialPlugin, CityLightsPlugin,
    EarthDynamicOverlayContext, EarthParticleContext, EarthPlugin, EarthPluginFrameContext,
    EarthPluginHooks, EarthPluginRegistry, EarthTextureInjector, EarthTileBuildContext,
    EarthWorldBuildContext, LoadedTileImagerySnapshot, OceanWavesPlugin, SatellitesPlugin,
    WeatherPlugin,
};
pub use precipitation::{
    climatological_sea_level_temp_c, compute_precipitation_phase, isa_pressure_hpa,
    moist_adiabatic_lapse_rate_k_per_km, parse_open_meteo_sounding_json,
    physical_precipitation_evaluator, saturation_vapor_pressure_pa, wet_bulb_temp_c,
    AtmosphereConfig, AtmosphereQuery, AtmosphereSample, CloudCeiling, FnTemperatureProvider,
    LapseRateModel, OpenMeteoSounding, OpenMeteoTemperatureProvider, ParticleBillboard,
    ParticleSprite, PrecipitationConfig, PrecipitationEvalContext, PrecipitationEvaluator,
    PrecipitationLayer, PrecipitationMode, PrecipitationPhase, PrecipitationPlugin,
    StaticAtmosphereProvider, TemperatureAnchor, TemperatureProvider, CLOUD_SHELL_ALTITUDE_M,
    DEFAULT_CLOUD_CEILING_MSL_M, DEFAULT_CLOUD_FADE_BAND_M, ISA_LAPSE_RATE_K_PER_KM,
    ISA_SEA_LEVEL_PRESSURE_HPA, ISA_TROPOPAUSE_ALT_M,
};
pub use raycast::{raycast_ray, raycast_surface_ray, raycast_viewport, EarthRayHit};
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
    EarthScene, EarthViewMode, FrameStats, LayerToggles, MemoryBuildingSource,
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
    altitude_for_zoom_2d, fly_to, zoom_2d_for_altitude, EarthOrbitController,
    DEFAULT_FLY_TO_ALTITUDE_M, DEFAULT_FOV_Y_DEG, FULL_25D_PITCH_TAPER_START_M,
    FULL_3D_PITCH_CUTOFF_M, GLOBAL_OVERVIEW_RANGE_M, GROUND_CLEARANCE_M, LOCAL_TRANSITION_RANGE_M,
    MAX_RANGE_M, MIN_RANGE_M,
};
#[cfg(feature = "widget")]
pub use widget::{
    EarthOverlayContext, EarthOverlayPlugin, EarthView, EarthViewState, CLICK_MAX_TRAVEL_PX,
};
pub use worker::{
    EarthTileWorker, EarthViewport, EarthViewportStats, RetainedBuildStats, RetainedTerrainBuilder,
};
pub use strata_style::types::Color;
pub use world::{
    EarthGeometry, EarthMaterial, EarthMeshId, EarthObject, EarthObjectId, EarthVertex,
    EarthWorld, GeoOrientation, ObjectAnchor, ResolvedObjectTransform,
};
