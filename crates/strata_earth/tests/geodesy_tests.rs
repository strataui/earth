//! Geodesy, camera control, and tile resolution.
//!
//! The Earth plugin must work at Earth scale in f64, resolve missing tiles to
//! their nearest loaded ancestor, and keep the camera contract the reference
//! defines.

use strata_earth::{
    clamp_latitude, depth_range, geodetic_to_ecef, horizon_distance_m, level_for_altitude,
    resolve_tile_imagery, wrap_longitude, EarthCamera, Geodetic, LocalFrame, TileKey,
};

const R: f64 = 6_378_137.0;

#[test]
fn equator_prime_meridian_is_on_the_x_axis() {
    let p = geodetic_to_ecef(Geodetic::new(0.0, 0.0, 0.0));
    assert!((p.x - R).abs() < 1.0, "got {p:?}");
    assert!(p.y.abs() < 1e-6);
    assert!(p.z.abs() < 1e-6);
}

#[test]
fn north_pole_is_on_the_z_axis() {
    let p = geodetic_to_ecef(Geodetic::new(90.0, 0.0, 0.0));
    assert!(p.x.abs() < 1e-6);
    assert!(p.y.abs() < 1e-6);
    assert!((p.z - 6_356_752.3).abs() < 1.0, "polar radius was {}", p.z);
}

#[test]
fn altitude_lifts_the_point_radially() {
    let ground = geodetic_to_ecef(Geodetic::new(45.0, 10.0, 0.0));
    let high = geodetic_to_ecef(Geodetic::new(45.0, 10.0, 1_000.0));
    let delta = high.sub(ground).length();
    assert!((delta - 1_000.0).abs() < 1e-3, "altitude delta was {delta}");
}

#[test]
fn local_frame_is_orthonormal_and_points_up() {
    let frame = LocalFrame::at(51.4778, -0.0015);
    assert!((frame.up.length() - 1.0).abs() < 1e-12);
    assert!((frame.east.length() - 1.0).abs() < 1e-12);
    assert!(frame.east.dot(frame.up).abs() < 1e-12);
    assert!(frame.north.dot(frame.up).abs() < 1e-12);
    assert!((frame.east.cross(frame.north).sub(frame.up)).length() < 1e-12);
}

#[test]
fn camera_placement_puts_the_eye_above_the_focus() {
    let camera = EarthCamera {
        focus: Geodetic::new(0.0, 0.0, 0.0),
        ..EarthCamera::default()
    };
    let placement = camera.placement();
    let ground = geodetic_to_ecef(camera.focus);
    // The eye sits three Earth radii above the focus, straight up.
    let chord = 3.0 * 6_378_137.0;
    let height = placement.eye.sub(ground).length();
    assert!((height - chord).abs() < 1e-3, "eye distance was {height}");
    assert!(
        (camera.altitude_m() - chord).abs() < 1e-3,
        "a purely vertical offset is the altitude, got {}",
        camera.altitude_m()
    );
    // Over the focus point a purely vertical offset is exactly the height.
    let overhead = EarthCamera {
        focus: Geodetic::new(0.0, 0.0, 0.0),
        eye_offset_m: [0.0, 0.0, 10_000.0],
        ..EarthCamera::default()
    };
    assert!(
        (overhead.altitude_m() - 10_000.0).abs() < 1e-6,
        "a vertical offset of 10 km was {}, expected 10000",
        overhead.altitude_m()
    );
    let lifted = EarthCamera {
        focus: Geodetic::new(0.0, 0.0, 0.0),
        eye_offset_m: [30_000_000.0, 0.0, 10_000.0],
        ..EarthCamera::default()
    };
    let lift = 30_000_000.0f64 * 30_000_000.0 / (2.0 * R);
    assert!(
        (lifted.altitude_m() - (lift + 10_000.0)).abs() < 50.0,
        "a horizontal chord of 30 000 km plus a 10 km lift was {}, expected about {}",
        lifted.altitude_m(),
        lift + 10_000.0
    );
}

#[test]
fn tile_key_parent_and_child_round_trip() {
    let key = TileKey::new(5, 13, 9);
    let parent = key.parent().expect("parent");
    assert_eq!(parent, TileKey::new(4, 6, 4));
    let child = parent
        .child(((key.y & 1) << 1 | (key.x & 1)) as u8)
        .expect("child");
    assert_eq!(child, key);
    assert!(TileKey::new(0, 0, 0).parent().is_none());
}

#[test]
fn exact_tile_resolution_uses_the_whole_image() {
    let target = TileKey::new(8, 100, 60);
    let resolution = resolve_tile_imagery(target, |k| k == target).expect("resolution");
    assert!(resolution.exact);
    assert_eq!(resolution.source, target);
    assert_eq!(resolution.uv_rect, (0.0, 0.0, 1.0, 1.0));
}

#[test]
fn missing_tile_resolves_to_a_quadrant_of_its_ancestor() {
    // Zoom 5 tile (17, 9): its level-4 parent is (8, 4). An odd x selects the
    // right half and an odd y the bottom half of the parent image.
    let target = TileKey::new(5, 17, 9);
    let resolution =
        resolve_tile_imagery(target, |k| k == TileKey::new(4, 8, 4)).expect("resolution");
    assert!(!resolution.exact);
    assert_eq!(resolution.source, TileKey::new(4, 8, 4));
    assert_eq!(resolution.uv_rect, (0.5, 0.5, 1.0, 1.0));
}

#[test]
fn resolution_walks_multiple_levels() {
    // Each level strips one low bit, so level 9 to level 4 is five steps down.
    let target = TileKey::new(9, 0b1010_10101, 0b0110_10011);
    let ancestor = target
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .parent()
        .unwrap();
    assert_eq!(ancestor.level, 4);
    let resolution = resolve_tile_imagery(target, |k| k == ancestor).expect("resolution");
    assert_eq!(resolution.source, ancestor);
    let (u0, v0, u1, v1) = resolution.uv_rect;
    assert!(u1 > u0 && v1 > v0);
    assert!(
        (u1 - u0 - 1.0 / 32.0).abs() < 1e-9,
        "5 levels deep is a 1/32 sub-rect"
    );
    assert!((v1 - v0 - 1.0 / 32.0).abs() < 1e-9);
}

#[test]
fn resolution_is_none_when_nothing_is_loaded() {
    assert!(resolve_tile_imagery(TileKey::new(3, 1, 1), |_| false).is_none());
}

#[test]
fn zoom_level_follows_altitude() {
    assert_eq!(level_for_altitude(3.0 * R, 18), 0);
    assert_eq!(level_for_altitude(1.5 * R, 18), 1);
    assert!(level_for_altitude(1_000.0, 18) > level_for_altitude(100_000.0, 18));
    assert_eq!(level_for_altitude(1.0, 6), 6, "the max level clamps");
}

#[test]
fn longitude_and_latitude_clamp() {
    assert!((wrap_longitude(190.0) + 170.0).abs() < 1e-9);
    assert!((wrap_longitude(-190.0) - 170.0).abs() < 1e-9);
    assert_eq!(clamp_latitude(95.0), 90.0);
    assert_eq!(clamp_latitude(-95.0), -90.0);
}

#[test]
fn depth_policy_keeps_a_workable_near_far_ratio() {
    // Terrain altitude: 4 km up, horizon about 226 km.
    let (near, far) = depth_range(4_000.0, horizon_distance_m(4_000.0), 1.0);
    assert!((near - 40.0).abs() < 1e-6, "near was {near}");
    assert!(far > 200_000.0 && far < 500_000.0, "far was {far}");
    assert!(
        far / near < 20_000.0,
        "a local view must not carry a planet-scale ratio, got {}",
        far / near
    );

    // Globe altitude: the far plane still reaches the limb twice over.
    let horizon = horizon_distance_m(12_000_000.0);
    let (near, far) = depth_range(12_000_000.0, horizon, 1.0);
    assert!((near - 120_000.0).abs() < 1e-6, "near was {near}");
    assert!(
        far >= horizon * 2.0,
        "the globe far plane must reach the limb"
    );
}

#[test]
fn camera_placement_uses_the_altitude_depth_policy() {
    let terrain = EarthCamera {
        focus: Geodetic::new(37.0, -122.0, 0.0),
        eye_offset_m: [0.0, 0.0, 4_000.0],
        ..EarthCamera::default()
    };
    let placement = terrain.placement();
    assert!((placement.near_m - 40.0).abs() < 1e-6);
    assert!(
        placement.far_m < 1_000_000.0,
        "a terrain camera must not keep a 51 000 km far plane, got {}",
        placement.far_m
    );

    let globe = EarthCamera {
        focus: Geodetic::new(37.0, -122.0, 0.0),
        eye_offset_m: [0.0, 0.0, 12_000_000.0],
        ..EarthCamera::default()
    };
    let placement = globe.placement();
    assert!(placement.far_m > 20_000_000.0);
}

#[test]
fn ecef_geodetic_round_trip_accuracy() {
    let test_points = [
        Geodetic::new(0.0, 0.0, 0.0),
        Geodetic::new(0.0, 90.0, 500.0),
        Geodetic::new(90.0, 0.0, 0.0),
        Geodetic::new(-90.0, 0.0, 100.0),
        Geodetic::new(37.7749, -122.4194, 150.0),
        Geodetic::new(51.5074, -0.1278, -30.0),
        Geodetic::new(-33.8688, 151.2093, 2_500.0),
        Geodetic::new(45.0, 45.0, 800_000.0),
    ];

    for point in test_points {
        let ecef = strata_earth::geodetic_to_ecef(point);
        let inverted = strata_earth::ecef_to_geodetic(ecef);

        let lat_err = (inverted.lat_deg - point.lat_deg).abs();
        let lon_err = if point.lat_deg.abs() > 89.99 {
            0.0 // Longitude is degenerate at the exact poles
        } else {
            (inverted.lon_deg - point.lon_deg).abs()
        };
        let height_err = (inverted.height_m - point.height_m).abs();

        assert!(
            lat_err < 1e-7,
            "latitude error {lat_err} for point {point:?}"
        );
        assert!(
            lon_err < 1e-7,
            "longitude error {lon_err} for point {point:?}"
        );
        assert!(
            height_err < 1e-3,
            "height error {height_err} m for point {point:?}"
        );
    }
}

#[test]
fn raycast_ellipsoid_hits_and_misses() {
    use strata_render::camera64::DVec3;

    // Ray from space looking directly down at the prime meridian equator
    let origin = DVec3::new(10_000_000.0, 0.0, 0.0);
    let dir = DVec3::new(-1.0, 0.0, 0.0);
    let hit = strata_earth::raycast_ellipsoid(origin, dir).expect("nadir ray must hit");
    assert!((hit.x - 6_378_137.0).abs() < 1e-3);
    assert!(hit.y.abs() < 1e-3);
    assert!(hit.z.abs() < 1e-3);

    // Ray pointing away into space
    let away_dir = DVec3::new(1.0, 0.0, 0.0);
    assert!(strata_earth::raycast_ellipsoid(origin, away_dir).is_none());

    // Ray tangential outside the limb
    let miss_origin = DVec3::new(10_000_000.0, 7_000_000.0, 0.0);
    let miss_dir = DVec3::new(-1.0, 0.0, 0.0);
    assert!(strata_earth::raycast_ellipsoid(miss_origin, miss_dir).is_none());
}

#[test]
fn screen_to_ray_and_pick_geodetic_center_accuracy() {
    let camera = EarthCamera {
        focus: Geodetic::new(37.7749, -122.4194, 0.0),
        eye_offset_m: [0.0, 0.0, 50_000.0],
        fov_y_deg: 45.0,
        ..Default::default()
    };
    let w = 1440.0;
    let h = 900.0;

    // Center of screen (720, 450) must pick the focus point
    let picked = strata_earth::pick_geodetic(w * 0.5, h * 0.5, w, h, &camera)
        .expect("center of screen must pick the focus");

    assert!((picked.lat_deg - camera.focus.lat_deg).abs() < 1e-4);
    assert!((picked.lon_deg - camera.focus.lon_deg).abs() < 1e-4);

    // Extreme corner off-screen when high or looking at space
    let space_camera = EarthCamera {
        focus: Geodetic::new(0.0, 0.0, 0.0),
        eye_offset_m: [0.0, 0.0, 20_000_000.0],
        fov_y_deg: 90.0,
        ..Default::default()
    };
    // Ray at top-left corner (0, 0) misses the planet
    let corner_pick = strata_earth::pick_geodetic(0.0, 0.0, w, h, &space_camera);
    assert!(
        corner_pick.is_none(),
        "far corner of wide FOV must miss the globe"
    );
}

#[test]
fn test_wgs84_ellipsoid_horizon_culling() {
    use strata_render::camera64::DVec3;

    // Eye high above the North Pole (altitude ~20,000 km)
    let eye = DVec3::new(0.0, 0.0, 26_378_137.0);

    // North pole tile at level 2: in clear line of sight
    let north_tile = TileKey::new(2, 0, 0); // Northernmost quadrant
    let (north_center, north_radius) = strata_earth::tile_bounding_sphere(north_tile);
    assert!(
        !strata_earth::is_tile_horizon_culled(north_center, north_radius, eye),
        "north hemisphere tile must be visible from above north pole"
    );

    // South pole tile at level 2: completely on the far side of the planet
    let south_tile = TileKey::new(2, 0, 3); // Southernmost quadrant
    let (south_center, south_radius) = strata_earth::tile_bounding_sphere(south_tile);
    assert!(
        strata_earth::is_tile_horizon_culled(south_center, south_radius, eye),
        "south pole tile must be horizon-culled from above north pole"
    );

    // Low altitude eye: 2,000 metres above equator (lat 0, lon 0)
    let eye_local = strata_earth::geodetic_to_ecef(Geodetic::new(0.0, 0.0, 2_000.0));

    // Local tile directly below the eye
    let local_tile = TileKey::new(10, 512, 512);
    let (local_center, local_radius) = strata_earth::tile_bounding_sphere(local_tile);
    assert!(
        !strata_earth::is_tile_horizon_culled(local_center, local_radius, eye_local),
        "nadir tile directly beneath eye must not be culled"
    );

    // Opposite side of the planet (lat 0, lon 180)
    let antipode_tile = TileKey::new(10, 0, 512);
    let (antipode_center, antipode_radius) = strata_earth::tile_bounding_sphere(antipode_tile);
    assert!(
        strata_earth::is_tile_horizon_culled(antipode_center, antipode_radius, eye_local),
        "antipodal tile must be horizon-culled"
    );
}

#[test]
fn test_quadtree_lod_sse_hierarchical_subdivision() {
    use strata_render::camera64::DVec3;

    // 1. Orbital Camera at 20,000 km altitude looking at center of the Earth
    let eye_orbital = DVec3::new(0.0, 0.0, 26_378_137.0);
    let aim = DVec3::ZERO;
    let mut orbital_tiles = Vec::new();

    strata_earth::traverse_quadtree_lod(
        eye_orbital,
        aim,
        45.0,
        1440.0 / 900.0,
        900.0,
        4.0,
        18,
        5,
        |key, _| {
            orbital_tiles.push(key);
        },
    );

    assert!(
        !orbital_tiles.is_empty(),
        "orbital quadtree traversal must yield visible tiles"
    );
    // In orbit, coarse tiles (level <= 6) must dominate, no sub-meter tiles
    for tile in &orbital_tiles {
        assert!(
            tile.level <= 8,
            "orbital distance should produce coarse tiles, got level {}",
            tile.level
        );
    }

    // 2. Low-altitude camera at 1,000 m over San Francisco looking down
    let eye_low = strata_earth::geodetic_to_ecef(Geodetic::new(37.7749, -122.4194, 1_000.0));
    let aim_low = strata_earth::geodetic_to_ecef(Geodetic::new(37.7749, -122.4194, 0.0));
    let mut low_tiles = Vec::new();

    strata_earth::traverse_quadtree_lod(
        eye_low,
        aim_low,
        45.0,
        1440.0 / 900.0,
        900.0,
        4.0,
        18,
        5,
        |key, _| {
            low_tiles.push(key);
        },
    );

    assert!(
        !low_tiles.is_empty(),
        "low-altitude traversal must yield visible tiles"
    );
    let max_subdiv_level = low_tiles.iter().map(|t| t.level).max().unwrap_or(0);
    assert!(
        max_subdiv_level >= 12,
        "1,000m altitude directly over focus must subdivide to high LOD (level >= 12), reached {}",
        max_subdiv_level
    );
}

#[test]
fn test_f64_double_single_split_accuracy() {
    let base = R; // 6,378,137.0 m
    let sub_meter_offsets = [0.25, 0.05, 0.005, 0.001];

    for &offset in &sub_meter_offsets {
        let original = base + offset;

        // Naive f32 cast completely collapses sub-meter details
        let f32_base = base as f32;
        let f32_orig = original as f32;
        if offset < 0.5 {
            assert_eq!(
                f32_base, f32_orig,
                "naive f32 at Earth radius must quantise away sub-half-meter offset {offset}"
            );
        }

        // Double-single split preserves sub-millimeter precision
        let (high, low) = strata_earth::split_f64_to_high_low(original);
        let reconstructed = high as f64 + low as f64;
        let err = (reconstructed - original).abs();
        assert!(
            err < 1e-6,
            "double-single split reconstruction error {} exceeds 1 micrometer for offset {}",
            err,
            offset
        );
    }
}

#[test]
fn test_gpu_rte_relative_position_sub_millimeter_fidelity() {
    let eye = strata_render::camera64::DVec3::new(R + 5000.0, 100.0, 50.0);
    let pos = strata_render::camera64::DVec3::new(R + 5000.25, 100.10, 50.05);

    // Exact f64 relative position
    let exact_rel = pos.sub(eye);

    // Naive f32 relative subtraction
    let eye_f32 = eye.as_f32();
    let pos_f32 = pos.as_f32();
    let naive_rel_x = pos_f32[0] - eye_f32[0];
    let naive_err = (naive_rel_x as f64 - exact_rel.x).abs();
    assert!(
        naive_err > 0.1,
        "naive f32 subtraction at Earth scale should have noticeable error, got {}",
        naive_err
    );

    // GPU RTE split subtraction: (pos_high - eye_high) + (pos_low - eye_low)
    let (pos_h, pos_l) = strata_earth::split_dvec3_to_high_low(pos);
    let (eye_h, eye_l) = strata_earth::split_dvec3_to_high_low(eye);
    let gpu_rel = strata_earth::rte_gpu_relative_position(pos_h, pos_l, eye_h, eye_l);

    let err_x = (gpu_rel[0] as f64 - exact_rel.x).abs();
    let err_y = (gpu_rel[1] as f64 - exact_rel.y).abs();
    let err_z = (gpu_rel[2] as f64 - exact_rel.z).abs();

    assert!(
        err_x < 1e-6,
        "GPU RTE X error {} exceeds sub-millimeter tolerance",
        err_x
    );
    assert!(
        err_y < 1e-6,
        "GPU RTE Y error {} exceeds sub-millimeter tolerance",
        err_y
    );
    assert!(
        err_z < 1e-6,
        "GPU RTE Z error {} exceeds sub-millimeter tolerance",
        err_z
    );
}

#[test]
fn test_rte_eye_uniforms_projection_identity() {
    let camera = EarthCamera {
        focus: Geodetic::new(0.0, 0.0, 0.0),
        eye_offset_m: [0.0, 0.0, 10_000.0],
        ..EarthCamera::default()
    };

    let uniforms = strata_earth::RteEyeUniforms::from_camera(&camera, 1440.0, 900.0);
    let ground = strata_earth::geodetic_to_ecef(camera.focus);
    let (pos_h, pos_l) = strata_earth::split_dvec3_to_high_low(ground);

    let clip = uniforms.transform_point_clip(pos_h, pos_l);
    assert!(
        clip[3] > 0.0,
        "w must be positive in front of camera, got {}",
        clip[3]
    );
    let ndc_z = clip[2] / clip[3];
    assert!(
        (0.0..=1.0).contains(&ndc_z),
        "NDC depth must be inside [0, 1], got {}",
        ndc_z
    );
}
