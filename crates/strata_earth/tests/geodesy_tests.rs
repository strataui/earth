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

#[test]
fn test_camera_up_override_prevents_polar_gimbal_lock_and_orients_view_rot() {
    use strata_render::camera64::DVec3;
    use strata_render::scene3d::Background3D;

    // Directly above the North Pole looking straight down at Earth's center.
    let camera = EarthCamera {
        focus: Geodetic::new(90.0, 0.0, 0.0),
        aim_offset_m: [0.0, 0.0, -R],
        eye_offset_m: [0.0, 0.0, 8_000_000.0],
        ..EarthCamera::default()
    };

    let up_y = DVec3::new(0.0, 1.0, 0.0);
    let up_neg_x = DVec3::new(-1.0, 0.0, 0.0);

    let u1 = strata_earth::retained_camera_uniforms_with_up(
        &camera,
        Some(up_y),
        1440.0,
        900.0,
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [0.1, 0.1, 0.1],
        Background3D::Transparent,
    );
    let u2 = strata_earth::retained_camera_uniforms_with_up(
        &camera,
        Some(up_neg_x),
        1440.0,
        900.0,
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [0.1, 0.1, 0.1],
        Background3D::Transparent,
    );

    // Row 1 of column-major view_rot is the camera's local +Y (up) axis in world space.
    let cam_up_1 = [u1.view_rot[0][1], u1.view_rot[1][1], u1.view_rot[2][1]];
    let cam_up_2 = [u2.view_rot[0][1], u2.view_rot[1][1], u2.view_rot[2][1]];
    assert!(
        cam_up_1[1] > 0.999 && cam_up_1[0].abs() < 1e-3 && cam_up_1[2].abs() < 1e-3,
        "expected camera +Y to align with +Y override at North Pole, got {:?}",
        cam_up_1
    );
    assert!(
        cam_up_2[0] < -0.999 && cam_up_2[1].abs() < 1e-3 && cam_up_2[2].abs() < 1e-3,
        "expected camera +Y to align with -X override at North Pole, got {:?}",
        cam_up_2
    );

    let leaves = strata_earth::traverse_quadtree_lod_frustum6_projected_with_up(
        &camera,
        Some(up_y),
        1440.0,
        900.0,
        2.0,
        16,
        9,
        true,
        strata_earth::MAX_QUADTREE_LEAVES,
        strata_earth::TileProjection::WebMercator,
    );
    assert_eq!(strata_earth::MAX_QUADTREE_LEAVES, 256);
    assert!(!leaves.is_empty());
    assert!(
        leaves.iter().all(|k| k.level >= 1),
        "close 3D orbit should subdivide past level 0, got {:?}",
        leaves
    );
}

#[test]
fn test_2d_flat_lod_wraps_horizontally_and_stops_at_poles() {
    let camera = EarthCamera {
        focus: Geodetic::new(15.0, 175.0, 0.0),
        aim_offset_m: [0.0, 0.0, 0.0],
        eye_offset_m: [0.0, 0.0, 25_000_000.0],
        fov_y_deg: 55.0,
        near_m: 1_000.0,
    };
    let items = strata_earth::traverse_quadtree_lod_2d_flat(
        &camera,
        1280.0,
        800.0,
        2.0,
        16,
        9,
        strata_earth::MAX_QUADTREE_LEAVES,
        strata_earth::TileProjection::WebMercator,
    );
    assert!(!items.is_empty());
    assert!(
        items.iter().all(|(k, _)| k.level >= 1),
        "2D flat traversal must subdivide past level 0 for clean wrapping"
    );
    let has_left_wrap = items.iter().any(|(_, w)| *w < 0);
    let has_center = items.iter().any(|(_, w)| *w == 0);
    let has_right_wrap = items.iter().any(|(_, w)| *w > 0);
    assert!(
        has_left_wrap && has_center && has_right_wrap,
        "wide 2D view across antimeridian should emit wrapped horizontal copies (-1, 0, +1)"
    );
    for (k, _) in &items {
        let b = strata_earth::tile_bounds_mercator(*k);
        assert!(b.lat1 <= 90.0 && b.lat0 >= -90.0);
    }

    // Near the North Pole at medium zoom: must not wrap vertically around to southern tiles.
    let north_cam = EarthCamera {
        focus: Geodetic::new(85.0, 0.0, 0.0),
        aim_offset_m: [0.0, 0.0, 0.0],
        eye_offset_m: [0.0, 0.0, 500_000.0],
        fov_y_deg: 55.0,
        near_m: 1_000.0,
    };
    let north_items = strata_earth::traverse_quadtree_lod_2d_flat(
        &north_cam,
        1280.0,
        800.0,
        2.0,
        16,
        9,
        strata_earth::MAX_QUADTREE_LEAVES,
        strata_earth::TileProjection::WebMercator,
    );
    assert!(!north_items.is_empty());
    for (k, _) in &north_items {
        let b = strata_earth::tile_bounds_mercator(*k);
        assert!(
            b.lat1 > 60.0,
            "near-North-Pole 2D view must end cleanly at pole without vertical wrap: {b:?}"
        );
    }
}

#[test]
fn test_elevation_above_max_zoom_samples_ancestor_quadrant_seamlessly() {
    use strata_earth::{
        EarthCamera, EarthDefaultTileSource, EarthInput, EarthViewMode, NetworkTileSource,
        RetainedTerrainBuilder, SurfaceMode, TileProjection,
    };

    let tmp_dir = std::env::temp_dir().join(format!(
        "strata_earth_elev_subtile_test_{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&tmp_dir);
    let net = NetworkTileSource::new("http://127.0.0.1:0/{z}/{x}/{y}.png")
        .with_elevation("http://127.0.0.1:0/{z}/{x}/{y}.png", false)
        .with_disk_cache(&tmp_dir);
    let elev_ns = net
        .provider_config()
        .elevation_cache_namespace()
        .expect("elevation namespace");
    let tile_dir = tmp_dir.join("elev").join(elev_ns).join("15/18738");
    std::fs::create_dir_all(&tile_dir).unwrap();

    // Write a deterministic 16x16 sloped z=15 heightmap (0 m at NW corner to 600 m at SE corner)
    // in NetworkTileSource's raw elevation format (w_le, h_le, f32_le heights).
    let w = 16u32;
    let h = 16u32;
    let mut raw = Vec::with_capacity(8 + (w * h * 4) as usize);
    raw.extend_from_slice(&w.to_le_bytes());
    raw.extend_from_slice(&h.to_le_bytes());
    for y in 0..h {
        for x in 0..w {
            let elev_m = (x as f32 / (w - 1) as f32) * 350.0 + (y as f32 / (h - 1) as f32) * 250.0;
            raw.extend_from_slice(&elev_m.to_le_bytes());
        }
    }
    std::fs::write(tile_dir.join("12385.elev"), &raw).unwrap();

    let mut source = EarthDefaultTileSource::with_network(net);

    // EarthDefaultTileSource must NOT report ancestor z=15 data as if it belonged to z=16+.
    use strata_earth::TileSource;
    let child_nw = TileKey::new(16, 37476, 24770);
    let parent_15 = TileKey::new(15, 18738, 12385);
    assert!(source.elevation(parent_15).unwrap().is_some());
    assert!(
        source.elevation(child_nw).unwrap().is_none(),
        "z=16 request must return None above max_elevation_level=15 so caller retains z=15 ancestor key"
    );
    assert!(source.peek_cached_elevation(child_nw).is_none());

    // Center camera inside (15, 18738, 12385) close enough to subdivide into z=16+ leaves.
    let bounds_15 = strata_earth::tile_bounds_mercator(parent_15);
    let mid_lat = (bounds_15.lat0 + bounds_15.lat1) * 0.5;
    let mid_lon = (bounds_15.lon0 + bounds_15.lon1) * 0.5;

    let mut input = EarthInput {
        camera: EarthCamera {
            focus: Geodetic::new(mid_lat, mid_lon, 0.0),
            aim_offset_m: [0.0, 0.0, 0.0],
            eye_offset_m: [0.0, -400.0, 450.0],
            fov_y_deg: 55.0,
            near_m: 2.0,
        },
        view_mode: EarthViewMode::Mode3D,
        seamless_25d: true,
        tile_projection: TileProjection::WebMercator,
        surface_mode: SurfaceMode::QuadtreeLod,
        elevation_enabled: true,
        tile_grid: 9,
        max_tile_level: 17,
        sse_threshold: 2.0,
        ..EarthInput::default()
    };
    input.viewport.width = 1440.0;
    input.viewport.height = 900.0;

    let mut builder = RetainedTerrainBuilder::new();
    let (frame, _) = builder.build_frame(&input, &mut source, 1, true);

    use std::collections::HashMap;
    let n = 9usize;
    let by_key: HashMap<(u8, u32, u32), &_> = frame
        .meshes
        .iter()
        .map(|m| ((m.key.level, m.key.col, m.key.row), m))
        .collect();

    let mut checked_subtile_pairs = 0usize;
    for (&(lvl, col, row), m) in &by_key {
        if lvl <= 15 {
            continue;
        }
        let s = lvl - 15;
        let px = col >> s;
        let py = row >> s;
        if px != 18738 || py != 12385 {
            continue;
        }
        if let Some(right) = by_key.get(&(lvl, col + 1, row)) {
            if (col + 1) >> s == px {
                checked_subtile_pairs += 1;
                for r in 0..n {
                    let a = m.vertices[r * n + (n - 1)].pos_u;
                    let b = right.vertices[r * n].pos_u;
                    let d = ((m.center_ecef[0] + a[0] as f64 - (right.center_ecef[0] + b[0] as f64)).powi(2)
                        + (m.center_ecef[1] + a[1] as f64 - (right.center_ecef[1] + b[1] as f64)).powi(2)
                        + (m.center_ecef[2] + a[2] as f64 - (right.center_ecef[2] + b[2] as f64)).powi(2))
                    .sqrt();
                    assert!(d < 0.05, "east-west seam gap within z=15 parent at ({lvl},{col},{row}): {d}m");
                }
            }
        }
        if let Some(south) = by_key.get(&(lvl, col, row + 1)) {
            if (row + 1) >> s == py {
                checked_subtile_pairs += 1;
                for c in 0..n {
                    let a = m.vertices[c].pos_u;
                    let b = south.vertices[(n - 1) * n + c].pos_u;
                    let d = ((m.center_ecef[0] + a[0] as f64 - (south.center_ecef[0] + b[0] as f64)).powi(2)
                        + (m.center_ecef[1] + a[1] as f64 - (south.center_ecef[1] + b[1] as f64)).powi(2)
                        + (m.center_ecef[2] + a[2] as f64 - (south.center_ecef[2] + b[2] as f64)).powi(2))
                    .sqrt();
                    assert!(d < 0.05, "north-south seam gap within z=15 parent at ({lvl},{col},{row}): {d}m");
                }
            }
        }
    }
    assert!(
        checked_subtile_pairs > 0,
        "expected z>=16 child tiles inside (15, 18738, 12385) to verify seam continuity"
    );

    let _ = std::fs::remove_dir_all(&tmp_dir);
}

#[test]
fn test_tilted_lod_reduces_resolution_for_far_tiles() {
    use strata_earth::{
        EarthInput, EarthViewMode, RetainedTerrainBuilder, SurfaceMode, TileElevation, TileError,
        TileImage, TileProjection, TileSource,
    };

    struct OfflineDummyTileSource;
    impl TileSource for OfflineDummyTileSource {
        fn imagery(&mut self, _key: TileKey) -> Result<Option<TileImage>, TileError> {
            Ok(None)
        }
        fn elevation(&mut self, _key: TileKey) -> Result<Option<TileElevation>, TileError> {
            Ok(None)
        }
    }

    let focus = Geodetic::new(40.7128, -74.0060, 0.0);
    // Low, strongly tilted 2.5D camera looking north over a deep foreground-to-horizon swath.
    let mut input = EarthInput {
        camera: EarthCamera {
            focus,
            aim_offset_m: [0.0, 40_000.0, 0.0],
            eye_offset_m: [0.0, -45_000.0, 22_000.0],
            fov_y_deg: 55.0,
            near_m: 100.0,
        },
        view_mode: EarthViewMode::Mode3D,
        seamless_25d: true,
        tile_projection: TileProjection::WebMercator,
        surface_mode: SurfaceMode::QuadtreeLod,
        elevation_enabled: false,
        tile_grid: 9,
        max_tile_level: 16,
        sse_threshold: 2.0,
        ..EarthInput::default()
    };
    input.viewport.width = 1440.0;
    input.viewport.height = 900.0;

    let mut source = OfflineDummyTileSource;
    let mut builder = RetainedTerrainBuilder::new();
    let (frame, _) = builder.build_frame(&input, &mut source, 1, true);
    assert!(
        frame.meshes.len() > 20,
        "expected substantial leaf set across tilted frustum, got {}",
        frame.meshes.len()
    );

    let origin = geodetic_to_ecef(focus);
    let basis = LocalFrame::at(focus.lat_deg, focus.lon_deg);
    let [oe, on, ou] = input.camera.eye_offset_m;
    let eye_x = origin.x + basis.east.x * oe + basis.north.x * on + basis.up.x * ou;
    let eye_y = origin.y + basis.east.y * oe + basis.north.y * on + basis.up.y * ou;
    let eye_z = origin.z + basis.east.z * oe + basis.north.z * on + basis.up.z * ou;
    let mut by_dist: Vec<(f64, u8)> = frame
        .meshes
        .iter()
        .map(|m| {
            let dx = m.center_ecef[0] - eye_x;
            let dy = m.center_ecef[1] - eye_y;
            let dz = m.center_ecef[2] - eye_z;
            ((dx * dx + dy * dy + dz * dz).sqrt(), m.key.level)
        })
        .collect();
    by_dist.sort_by(|a, b| a.0.total_cmp(&b.0));

    let q = (by_dist.len() / 4).max(4);
    let near_avg: f64 = by_dist[..q].iter().map(|(_, l)| *l as f64).sum::<f64>() / q as f64;
    let far_avg: f64 = by_dist[by_dist.len() - q..]
        .iter()
        .map(|(_, l)| *l as f64)
        .sum::<f64>()
        / q as f64;
    let near_max = by_dist[..q].iter().map(|(_, l)| *l).max().unwrap_or(0);
    let far_max = by_dist[by_dist.len() - q..]
        .iter()
        .map(|(_, l)| *l)
        .max()
        .unwrap_or(0);

    assert!(
        far_avg + 0.5 < near_avg && far_max < near_max,
        "expected distant tilted horizon tiles to receive strictly coarser LODs than foreground: near_avg={near_avg:.2} near_max={near_max}, far_avg={far_avg:.2} far_max={far_max}"
    );
}

#[test]
fn full_3d_globe_view_covers_entire_visible_hemisphere() {
    use strata_earth::{
        EarthDefaultTileSource, EarthInput, EarthOrbitController, EarthViewMode,
        RetainedTerrainBuilder, MAX_QUADTREE_LEAVES,
    };

    let mut orbit = EarthOrbitController::new();
    orbit.jump_to(26.0, 16.0, 13_500_000.0, 0.0, 0.0);
    orbit.set_viewport_size(1440.0, 900.0);
    let (cam, up_opt) = orbit.to_camera_and_up(false, 0.0);
    let mut input = EarthInput::globe();
    input.camera = cam;
    input.camera_up_ecef = up_opt;
    input.view_mode = EarthViewMode::Mode3D;
    input.viewport.width = 1440.0;
    input.viewport.height = 900.0;

    let up_v = up_opt.map(|u| strata_render::camera64::DVec3::new(u[0], u[1], u[2]));
    let leaves = strata_earth::traverse_quadtree_lod_frustum6_projected_with_up(
        &input.camera,
        up_v,
        1440.0,
        900.0,
        input.sse_threshold,
        input.max_tile_level,
        input.tile_grid.max(2),
        true,
        MAX_QUADTREE_LEAVES,
        input.tile_projection,
    );
    let mut min_lat = 90.0f64;
    let mut max_lat = -90.0f64;
    let mut min_lon = 180.0f64;
    let mut max_lon = -180.0f64;
    for &k in &leaves {
        let b = strata_earth::tile_bounds_mercator(k);
        min_lat = min_lat.min(b.lat0);
        max_lat = max_lat.max(b.lat1);
        min_lon = min_lon.min(b.lon0);
        max_lon = max_lon.max(b.lon1);
    }

    let mut source = EarthDefaultTileSource::offline_basemap();
    let mut builder = RetainedTerrainBuilder::new();
    let (frame, stats) = builder.build_frame(&input, &mut source, 1, true);
    assert!(
        frame.meshes.len() >= 24 && frame.instances.len() >= 24 && stats.visible_tiles >= 24,
        "full 3D globe build must emit all visible hemisphere tiles, got meshes={} instances={}",
        frame.meshes.len(),
        frame.instances.len(),
    );
    assert!(
        max_lon - min_lon > 120.0 && max_lat - min_lat > 100.0,
        "full 3D view must cover visible hemisphere, got lon=[{min_lon}..{max_lon}] lat=[{min_lat}..{max_lat}]"
    );
}

#[test]
fn close_ground_oblique_view_selects_high_detail_without_idle_hires() {
    use strata_earth::{EarthInput, EarthOrbitController, EarthViewMode, MAX_QUADTREE_LEAVES};

    let mut orbit = EarthOrbitController::new();
    orbit.jump_to(37.7680, -122.5180, 330.0, 250.0, 68.0);
    orbit.set_viewport_size(1440.0, 900.0);
    let (cam, up_opt) = orbit.to_camera_and_up(false, 0.0);
    let mut input = EarthInput::globe();
    input.camera = cam;
    input.camera_up_ecef = up_opt;
    input.view_mode = EarthViewMode::Mode3D;
    input.viewport.width = 1440.0;
    input.viewport.height = 900.0;
    assert!(!input.idle_high_res.enabled);

    let up_v = up_opt.map(|u| strata_render::camera64::DVec3::new(u[0], u[1], u[2]));
    let leaves = strata_earth::traverse_quadtree_lod_frustum6_projected_with_up(
        &input.camera,
        up_v,
        1440.0,
        900.0,
        input.sse_threshold,
        input.max_tile_level,
        input.tile_grid.max(2),
        true,
        MAX_QUADTREE_LEAVES,
        input.tile_projection,
    );
    let max_lvl = leaves.iter().map(|k| k.level).max().unwrap_or(0);
    assert!(
        max_lvl >= 17,
        "EarthInput::globe at 330m oblique view must reach z>=17 even with Idle Hi-Res Off (got max z={max_lvl})"
    );
}

#[test]
fn polar_tiles_close_cleanly_at_exact_poles_without_notch_or_nan_normals() {
    use strata_earth::{
        build_tile_local_rtc_mesh_morphed, mercator_gy_to_lat_deg, tile_bounds_mercator,
        SurfaceMorphParams, TileElevation, TileKey, TileProjection,
    };

    assert_eq!(mercator_gy_to_lat_deg(0.0), 90.0);
    assert_eq!(mercator_gy_to_lat_deg(1.0), -90.0);

    let key = TileKey::new(1, 0, 0);
    let bounds = tile_bounds_mercator(key);
    assert!((bounds.lat1 - 90.0).abs() < 1e-9);

    let elev = TileElevation {
        width: 4,
        height: 4,
        heights_m: (0..16).map(|i| 100.0 + (i as f32) * 5.0).collect(),
    };
    let grid = 9u32;
    let mesh = build_tile_local_rtc_mesh_morphed(
        key,
        bounds,
        grid,
        Some(&elev),
        key,
        true,
        500.0,
        1,
        0,
        SurfaceMorphParams::default(),
    );

    let n = grid as usize;
    let top_row_start = (n - 1) * n;
    let first_pole_pos = mesh.vertices[top_row_start].pos_u;
    for col in 0..n {
        let v = mesh.vertices[top_row_start + col];
        for k in 0..3 {
            assert!(
                (v.pos_u[k] - first_pole_pos[k]).abs() < 1e-2,
                "North Pole row vertices must converge to single apex without hole/notch"
            );
            assert!(v.norm_v[k].is_finite(), "North Pole normal must be finite");
        }
        assert!(
            v.norm_v[2] > 0.999,
            "North Pole apex normal must point along +Z, got {:?}",
            v.norm_v
        );
    }
    let approx_world_z = mesh.center_ecef[2] + first_pole_pos[2] as f64;
    let expected_polar_r = 6_356_752.314_2 + 107.5;
    assert!(
        (approx_world_z - expected_polar_r).abs() < 25.0,
        "expected closed polar apex near WGS84 polar radius + pole elev, got z={approx_world_z}"
    );
    let _ = TileProjection::WebMercator;
}

#[test]
fn morph_2d_to_3d_transition_preserves_closed_north_pole_cap_and_polar_tiles() {
    use strata_earth::{
        build_tile_local_rtc_mesh_morphed, stabilized_surface_morph, tile_bounds_mercator,
        traverse_quadtree_lod_frustum6_with_elevation, EarthCamera, EarthInput, EarthViewMode,
        EarthViewport, Geodetic, IdleLodParams, MemoryTileSource, TileKey, TileProjection,
        CANONICAL_2D_MORPH_THRESHOLD, MAX_QUADTREE_LEAVES,
    };

    // Mid-latitude high-altitude globe camera where North Pole sits on the visible upper limb
    let cam = EarthCamera {
        focus: Geodetic::new(35.0, 10.0, 0.0),
        eye_offset_m: [0.0, 0.0, 12_000_000.0],
        aim_offset_m: [0.0, 0.0, 0.0],
        ..EarthCamera::default()
    };

    // 1. Across every morph step from 2D -> 3D (0.995 down to 0.0), y == 0 polar row tiles must
    //    fully span [-180°, +180°] with lat1 == 90.0° and zero dropped wedges.
    for &morph_t in &[0.995f64, 0.85, 0.55, 0.35, 0.12, 0.04, 0.01, 0.0] {
        let horizon_cull = morph_t <= 1e-4;
        let leaves = traverse_quadtree_lod_frustum6_with_elevation(
            &cam,
            None,
            1280.0,
            720.0,
            2.0,
            6,
            17,
            horizon_cull,
            MAX_QUADTREE_LEAVES,
            TileProjection::WebMercator,
            IdleLodParams::inactive(),
            0.0,
            |_| 0.0,
        );
        let mut polar_intervals: Vec<(f64, f64)> = leaves
            .iter()
            .filter(|k| k.y == 0)
            .map(|&k| {
                let b = tile_bounds_mercator(k);
                assert!(
                    (b.lat1 - 90.0).abs() < 1e-9,
                    "polar tile {:?} at morph={} must close to 90°",
                    k,
                    morph_t
                );
                (b.lon0, b.lon1)
            })
            .collect();
        polar_intervals.sort_by(|a, b| a.0.total_cmp(&b.0));
        assert!(
            !polar_intervals.is_empty(),
            "morph={morph_t} must emit y == 0 polar cap tiles"
        );
        assert!(
            (polar_intervals[0].0 - (-180.0)).abs() < 1e-6,
            "morph={morph_t} north polar ring must start at -180°, got {:?}",
            polar_intervals
        );
        let mut covered_lon = polar_intervals[0].1;
        for &(lon0, lon1) in &polar_intervals[1..] {
            assert!(
                lon0 <= covered_lon + 1e-6,
                "morph={morph_t} north polar ring has longitude gap between {covered_lon} and {lon0}"
            );
            covered_lon = covered_lon.max(lon1);
        }
        assert!(
            (covered_lon - 180.0).abs() < 1e-6,
            "morph={morph_t} north polar ring must cover all the way to +180°, got {covered_lon}"
        );
    }

    // 2. Across the curved globe morph range (0.0..=0.85, including the ~180ms mid-morph ~0.78),
    //    every y == 0 tile's north edge must converge to a single watertight north-pole apex
    //    across all longitudes (no curved horizontal collar or cap hole).
    for &morph_t in &[0.0f64, 0.02, 0.15, 0.35, 0.55, 0.70, 0.78, 0.85] {
        let morph = stabilized_surface_morph(
            cam.focus,
            morph_t,
            morph_t,
            TileProjection::WebMercator,
        );
        let grid = 9usize;
        let mut ref_apex: Option<[f64; 3]> = None;
        for x in 0..4u32 {
            let key = TileKey::new(2, x, 0);
            let bounds = tile_bounds_mercator(key);
            let mesh = build_tile_local_rtc_mesh_morphed(
                key,
                bounds,
                grid as u32,
                None,
                key,
                true,
                500.0,
                1,
                0,
                morph,
            );
            let top_row = (grid - 1) * grid;
            for col in 0..grid {
                let v = mesh.vertices[top_row + col];
                let world = [
                    mesh.center_ecef[0] + v.pos_u[0] as f64,
                    mesh.center_ecef[1] + v.pos_u[1] as f64,
                    mesh.center_ecef[2] + v.pos_u[2] as f64,
                ];
                if let Some(apex) = ref_apex {
                    let dx = world[0] - apex[0];
                    let dy = world[1] - apex[1];
                    let dz = world[2] - apex[2];
                    let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                    assert!(
                        dist < 2.0,
                        "morph={morph_t} tile {:?} col={col} polar apex diverged by {dist}m (open cap cutout)",
                        key
                    );
                } else {
                    ref_apex = Some(world);
                }
            }
        }
    }

    // Once nearly flat (0.95), the polar row smoothly begins opening toward the 2D top edge.
    {
        let morph = stabilized_surface_morph(cam.focus, 0.95, 0.95, TileProjection::WebMercator);
        let key = TileKey::new(2, 0, 0);
        let bounds = tile_bounds_mercator(key);
        let grid = 9usize;
        let mesh = build_tile_local_rtc_mesh_morphed(
            key,
            bounds,
            grid as u32,
            None,
            key,
            true,
            500.0,
            1,
            0,
            morph,
        );
        let top_row = (grid - 1) * grid;
        let v0 = mesh.vertices[top_row];
        let v1 = mesh.vertices[top_row + grid - 1];
        let span = ((v1.pos_u[0] - v0.pos_u[0]) as f64)
            .hypot((v1.pos_u[1] - v0.pos_u[1]) as f64)
            .hypot((v1.pos_u[2] - v0.pos_u[2]) as f64);
        assert!(
            span > 1_000.0,
            "morph=0.95 polar top edge must smoothly begin opening toward 2D map width, got {span}m"
        );
    }

    // 2b. High-latitude rows south of the exact pole (e.g. 85°N) must taper smoothly by cos(lat)
    //     rather than flaring sideways to full equatorial/focus width at ~100ms (flat_t ≈ 0.874)
    //     or mid-morph (0.78, 0.85), and must match project_morphed_point_and_frame exactly.
    use strata_earth::{project_morphed_point_and_frame, PatchBounds};
    use strata_render::camera64::DVec3;
    let eq_span_90deg = 90.0_f64.to_radians() * 6_378_137.0;
    for &(morph_t, max_ratio) in &[(0.78f64, 0.12f64), (0.85f64, 0.12f64), (0.874f64, 0.22f64)] {
        let morph = stabilized_surface_morph(
            Geodetic::new(20.0, 0.0, 0.0),
            morph_t,
            morph_t,
            TileProjection::Geographic4326,
        );
        let (p_west, _) = project_morphed_point_and_frame(85.0, -45.0, 0.0, morph);
        let (p_east, _) = project_morphed_point_and_frame(85.0, 45.0, 0.0, morph);
        let span_85n = p_east.sub(p_west).length();
        let ratio = span_85n / eq_span_90deg;
        assert!(
            ratio < max_ratio,
            "morph={morph_t} 85°N 90°-lon span flared to {ratio:.3}x equatorial width (max {max_ratio:.3}x)"
        );

        // Mesh vertices on a Geographic4326 tile with lat0=80..lat1=90 must match project_morphed_point_and_frame.
        let key = TileKey::new(2, 2, 0);
        let bounds = PatchBounds {
            lat0: 80.0,
            lat1: 90.0,
            lon0: 0.0,
            lon1: 90.0,
        };
        let grid = 5usize; // row 2 -> lat = 85.0, col 4 -> lon = 90.0
        let mesh = build_tile_local_rtc_mesh_morphed(
            key,
            bounds,
            grid as u32,
            None,
            key,
            false,
            0.0,
            1,
            0,
            morph,
        );
        let v = mesh.vertices[2 * grid + 4];
        let mesh_world = DVec3::new(
            mesh.center_ecef[0] + v.pos_u[0] as f64,
            mesh.center_ecef[1] + v.pos_u[1] as f64,
            mesh.center_ecef[2] + v.pos_u[2] as f64,
        );
        let (proj_world, _) = project_morphed_point_and_frame(85.0, 90.0, 0.0, morph);
        let err_m = mesh_world.sub(proj_world).length();
        assert!(
            err_m < 1.0,
            "morph={morph_t} mesh vs project_morphed_point_and_frame drifted by {err_m}m at 85°N"
        );
    }

    // 3. Switching EarthViewport from Mode2D back to Mode3D must drop view_morph below
    //    CANONICAL_2D_MORPH_THRESHOLD on the very first 120Hz transition step so the worker
    //    immediately switches off 2D-culled tiles and emits full polar globe tiles.
    let mut input = EarthInput::globe();
    input.viewport.width = 1280.0;
    input.viewport.height = 720.0;
    input.view_mode = EarthViewMode::Mode2D;
    input.view_morph = 1.0;
    let mut vp = EarthViewport::new(&input, Box::new(MemoryTileSource::default()));
    vp.set_view_mode(EarthViewMode::Mode3D);
    let stepped = vp.step_transition(1.0 / 120.0);
    assert!(stepped);
    assert!(
        (vp.input().view_morph as f64) < CANONICAL_2D_MORPH_THRESHOLD,
        "first 2D -> 3D transition step must immediately cross CANONICAL_2D_MORPH_THRESHOLD, got {}",
        vp.input().view_morph
    );
}

#[test]
fn grazing_pitch_preserves_terrain_relief_lod_and_oblique_mesh_fidelity() {
    use strata_earth::{
        traverse_quadtree_lod_frustum6_with_relief, EarthInput, EarthOrbitController, EarthViewMode,
        IdleLodParams, MemoryTileSource, RetainedTerrainBuilder, TileProjection,
        MAX_QUADTREE_LEAVES,
    };
    use strata_render::camera64::DVec3;

    // Near-ground grazing/horizontal view (~480 m altitude, 82° pitch looking toward mountain horizon).
    let mut orbit = EarthOrbitController::new();
    orbit.jump_to(46.5197, 7.9821, 3_500.0, 45.0, 82.0);
    orbit.set_viewport_size(1440.0, 900.0);
    let (cam, up_opt) = orbit.to_camera_and_up(false, 0.0);
    let up_v = up_opt.map(|u| DVec3::new(u[0], u[1], u[2]));
    let placement = cam.placement_with_up(up_v);
    let eye = placement.eye;
    let cam_fwd = placement.aim.sub(eye).normalize();

    let flat_leaves = traverse_quadtree_lod_frustum6_with_relief(
        &cam,
        up_v,
        1440.0,
        900.0,
        2.3,
        18,
        17,
        true,
        MAX_QUADTREE_LEAVES,
        TileProjection::WebMercator,
        IdleLodParams::inactive(),
        false,
        0.0,
        |_| 0.0,
    );
    let relief_leaves = traverse_quadtree_lod_frustum6_with_relief(
        &cam,
        up_v,
        1440.0,
        900.0,
        2.3,
        18,
        17,
        true,
        MAX_QUADTREE_LEAVES,
        TileProjection::WebMercator,
        IdleLodParams::inactive(),
        true,
        0.0,
        |_| 0.0,
    );

    let gaze_corridor_avg_lvl = |leaves: &[TileKey]| -> f64 {
        let mut sum = 0.0;
        let mut count = 0usize;
        for &k in leaves {
            let b = strata_earth::tile_bounds_mercator(k);
            let c = geodetic_to_ecef(Geodetic::new(
                0.5 * (b.lat0 + b.lat1),
                0.5 * (b.lon0 + b.lon1),
                0.0,
            ));
            let to_c = c.sub(eye);
            let dist = to_c.length().max(1.0);
            let gaze = to_c.dot(cam_fwd) / dist;
            if dist > 2_000.0 && gaze > 0.85 {
                sum += k.level as f64;
                count += 1;
            }
        }
        assert!(count > 0, "expected grazing gaze-corridor tiles beyond 2km");
        sum / count as f64
    };
    let max_lvl = |leaves: &[TileKey]| -> u8 { leaves.iter().map(|k| k.level).max().unwrap_or(0) };

    let flat_gaze_avg = gaze_corridor_avg_lvl(&flat_leaves);
    let relief_gaze_avg = gaze_corridor_avg_lvl(&relief_leaves);
    let flat_max = max_lvl(&flat_leaves);
    let relief_max = max_lvl(&relief_leaves);
    assert!(
        relief_gaze_avg > flat_gaze_avg + 0.8 && relief_max >= flat_max,
        "grazing-pitch vertical terrain-relief SSE must select significantly higher LOD along the gaze corridor than flat-imagery foreshortening (flat_gaze_avg={flat_gaze_avg:.2}, relief_gaze_avg={relief_gaze_avg:.2}, flat_max={flat_max}, relief_max={relief_max})"
    );

    // Pitch-aware mesh grid resolution: oblique gaze-corridor tiles with elevation enabled should
    // subdivide up to 33x33 even above 420m altitude, whereas a nadir view at the same altitude
    // stays at base_grid.
    let mut oblique_input = EarthInput::globe();
    oblique_input.camera = cam.clone();
    oblique_input.camera_up_ecef = up_opt;
    oblique_input.view_mode = EarthViewMode::Mode3D;
    oblique_input.elevation_enabled = true;
    oblique_input.tile_grid = 17;
    let mut source = MemoryTileSource::default();
    let mut builder = RetainedTerrainBuilder::new();
    let (oblique_frame, _) = builder.build_frame(&oblique_input, &mut source, 1, true);
    let oblique_max_verts = oblique_frame
        .meshes
        .iter()
        .map(|m| m.vertices.len())
        .max()
        .unwrap_or(0);
    assert!(
        oblique_max_verts >= 33 * 33,
        "grazing-pitch terrain silhouette tiles along gaze corridor must upgrade mesh grid >= 33x33, got max {oblique_max_verts} verts"
    );

    let mut nadir_orbit = EarthOrbitController::new();
    nadir_orbit.jump_to(46.5197, 7.9821, 2_000.0, 0.0, 0.0);
    nadir_orbit.set_viewport_size(1440.0, 900.0);
    let (nadir_cam, nadir_up) = nadir_orbit.to_camera_and_up(false, 0.0);
    let mut nadir_input = EarthInput::globe();
    nadir_input.camera = nadir_cam;
    nadir_input.camera_up_ecef = nadir_up;
    nadir_input.view_mode = EarthViewMode::Mode3D;
    nadir_input.elevation_enabled = true;
    nadir_input.tile_grid = 17;
    let mut nadir_builder = RetainedTerrainBuilder::new();
    let (nadir_frame, _) = nadir_builder.build_frame(&nadir_input, &mut source, 1, true);
    let nadir_max_verts = nadir_frame
        .meshes
        .iter()
        .map(|m| m.vertices.len())
        .max()
        .unwrap_or(0);
    assert!(
        nadir_max_verts < 33 * 33,
        "nadir view above 420m should stay at base_grid without oblique 33x33 expansion, got max {nadir_max_verts} verts"
    );
}







