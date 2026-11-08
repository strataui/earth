//! Golden-image capture for the four Earth camera regimes.
//!
//! This is both a reviewer's tool and the acceptance instrument for the plugin:
//! it renders each regime, writes a PNG plus a sidecar of metrics, and asserts
//! that two consecutive runs are bit-identical. It never downloads tiles, never
//! shells out, and never touches `egui`.
//!
//! Usage:
//!
//! ```text
//! cargo run --release -p strata_earth_golden -- <output-dir>
//! ```
//!
//! With no argument the goldens are written to `target/earth-goldens`.

#![allow(clippy::chunks_exact_to_as_chunks)]

use std::path::PathBuf;

use strata_earth::{
    requested_tile, surrounding_patches, tile_patch_span_deg, EarthCamera, EarthInput, EarthScene,
    EntityMarker, FrameDescriptor, Geodetic, MemoryTileSource, TileElevation, TileImage, TileKey,
};
use strata_layout::Rect;
use strata_render::rasterizer::FrameBuffer;
use strata_render::rasterizer3d::render_scene_3d_with_textures;
use strata_style::types::Color;

const W: usize = 1440;
const H: usize = 900;

/// One acceptance regime from the handoff: name, altitude, sim time, clouds.
struct Regime {
    name: &'static str,
    altitude_m: f64,
    heading_deg: f64,
    look_down_deg: f64,
    daylight: f32,
    tile_grid: u32,
    patch_scale: f64,
}

const REGIMES: &[Regime] = &[
    // Cloud cover is zero for the parity captures: the software rasterizer has
    // no cloud pass yet, and the first comparison is on geometry, UVs, and the
    // sky ramp. Raise these values once the pass lands.
    //
    // The acceptance pair is the two ground frames. The globe frames are
    // informational until a sphere surface lands.
    Regime {
        name: "terrain_z18",
        altitude_m: 4_000.0,
        heading_deg: 0.0,
        look_down_deg: 0.0,
        daylight: 0.60,
        tile_grid: 9,
        patch_scale: 1.0,
    },
    Regime {
        name: "terrain_oblique",
        altitude_m: 1_500.0,
        heading_deg: 135.0,
        look_down_deg: 25.0,
        daylight: 0.75,
        tile_grid: 9,
        patch_scale: 1.0,
    },
    Regime {
        name: "full_globe",
        altitude_m: 12_000_000.0,
        heading_deg: 0.0,
        look_down_deg: 0.0,
        daylight: 0.85,
        tile_grid: 5,
        patch_scale: 1.0,
    },
    Regime {
        name: "mid_morph",
        altitude_m: 1_200_000.0,
        heading_deg: 0.0,
        look_down_deg: 0.0,
        daylight: 0.70,
        tile_grid: 5,
        patch_scale: 1.0,
    },
];

/// Eye offset in the local ENU basis for a heading and look-down angle.
///
/// A nadir camera sits straight above the focus. An oblique camera sits
/// opposite the look direction, far enough back that the view is `look_down`
/// degrees below horizontal, with the up component reduced by the spherical
/// lift of the horizontal chord so the reported altitude matches the scenario.
fn eye_offset_m(altitude_m: f64, heading_deg: f64, look_down_deg: f64) -> [f64; 3] {
    if look_down_deg <= 0.0 {
        return [0.0, 0.0, altitude_m];
    }
    const EARTH_RADIUS_M: f64 = 6_378_137.0;
    let distance = altitude_m / look_down_deg.to_radians().tan();
    let up = (altitude_m - distance * distance / (2.0 * EARTH_RADIUS_M)).max(0.0);
    let (sin, cos) = heading_deg.to_radians().sin_cos();
    [-sin * distance, -cos * distance, up]
}

fn procedural_tile(seed: u8) -> TileImage {
    const SIZE: u32 = 64;
    let mut pixels = Vec::with_capacity((SIZE * SIZE * 4) as usize);
    for y in 0..SIZE {
        for x in 0..SIZE {
            let grid = ((x % 8 == 0) || (y % 8 == 0)) as u8 * 40;
            let base = 60u8.saturating_add(seed % 40);
            pixels.push(base.saturating_add(grid));
            pixels.push(base.saturating_add(30).saturating_add(grid));
            pixels.push((base / 2).saturating_add(80).saturating_add(grid));
            pixels.push(255);
        }
    }
    TileImage {
        width: SIZE,
        height: SIZE,
        pixels,
    }
}

fn input_for(regime: &Regime, focus: Geodetic) -> EarthInput {
    EarthInput {
        camera: EarthCamera {
            focus,
            eye_offset_m: eye_offset_m(regime.altitude_m, regime.heading_deg, regime.look_down_deg),
            fov_y_deg: 55.52,
            ..Default::default()
        },
        viewport: Rect::new(0.0, 0.0, W as f32, H as f32),
        sim_time_unix_s: 1_767_225_600.0,
        cloud_cover: 0.0,
        daylight: regime.daylight,
        tile_grid: regime.tile_grid,
        patch_scale: regime.patch_scale,
        graticule_step_deg: 30.0,
        max_tile_level: 18,
        markers: vec![
            EntityMarker {
                position: Geodetic::new(37.7749, -122.4194, 0.0),
                label: "SF".to_string(),
                color: Color::rgba(56, 189, 248, 255),
            },
            EntityMarker {
                position: Geodetic::new(37.3382, -121.8863, 0.0),
                label: "SJC".to_string(),
                color: Color::rgba(248, 113, 113, 255),
            },
        ],
        ..Default::default()
    }
}

fn capture(
    input: &EarthInput,
) -> (
    FrameBuffer,
    FrameDescriptor,
    strata_earth::FrameStats,
    u8,
    usize,
) {
    let level = strata_earth::level_for_altitude(input.camera.altitude_m(), input.max_tile_level);
    let span = tile_patch_span_deg(&input.camera, input.patch_scale);
    let patches = surrounding_patches(input.camera.focus, span);
    let requested: Vec<TileKey> = patches.iter().map(requested_tile).collect();

    let mut source = MemoryTileSource::new();
    for (index, key) in requested.iter().enumerate() {
        source.insert_imagery(*key, procedural_tile(index as u8));
        source.insert_elevation(
            *key,
            TileElevation {
                width: 2,
                height: 2,
                heights_m: vec![120.0, 180.0, 90.0, 150.0],
            },
        );
    }
    let mut scene = EarthScene::new();
    scene.build(input, &mut source);
    let descriptor = FrameDescriptor::capture(&scene, input);
    let stats = scene.stats();
    let mut fb = FrameBuffer::new(W, H);
    render_scene_3d_with_textures(
        &mut fb,
        input.viewport,
        scene.scene(),
        None,
        Some(scene.tile_registry().store()),
    );

    // Terrain coverage is structural: render the same frame with the terrain
    // layer disabled and count the pixels that change. A bright-colour proxy
    // reports zero for a dim but fully covered frame.
    let mut without_terrain = input.clone();
    without_terrain.layers.terrain = false;
    let mut bare_scene = EarthScene::new();
    bare_scene.build(&without_terrain, &mut source);
    let mut bare = FrameBuffer::new(W, H);
    render_scene_3d_with_textures(
        &mut bare,
        without_terrain.viewport,
        bare_scene.scene(),
        None,
        Some(bare_scene.tile_registry().store()),
    );
    let cover = fb
        .pixels
        .as_chunks::<4>().0.iter()
        .zip(bare.pixels.chunks_exact(4))
        .filter(|(a, b)| a != b)
        .count();

    (fb, descriptor, stats, level, cover)
}

/// Minimal PNG writer (RGBA8, no filtering) so goldens need no image crate.
fn write_png(path: &std::path::Path, fb: &FrameBuffer) -> std::io::Result<()> {
    let mut raw = Vec::with_capacity(fb.width * fb.height * 4 + fb.height);
    for row in 0..fb.height {
        raw.push(0u8);
        let start = row * fb.width * 4;
        raw.extend_from_slice(&fb.pixels[start..start + fb.width * 4]);
    }

    fn crc32(data: &[u8]) -> u32 {
        let mut table = [0u32; 256];
        for (i, entry) in table.iter_mut().enumerate() {
            let mut c = i as u32;
            for _ in 0..8 {
                c = if c & 1 != 0 {
                    0xEDB8_8320 ^ (c >> 1)
                } else {
                    c >> 1
                };
            }
            *entry = c;
        }
        let mut crc = 0xFFFF_FFFFu32;
        for byte in data {
            crc = table[((crc ^ *byte as u32) & 0xFF) as usize] ^ (crc >> 8);
        }
        crc ^ 0xFFFF_FFFF
    }

    fn adler32(data: &[u8]) -> u32 {
        let (mut a, mut b) = (1u32, 0u32);
        for byte in data {
            a = (a + *byte as u32) % 65521;
            b = (b + a) % 65521;
        }
        (b << 16) | a
    }

    fn chunk(out: &mut Vec<u8>, kind: &[u8; 4], payload: &[u8]) {
        out.extend_from_slice(&(payload.len() as u32).to_be_bytes());
        let mut body = Vec::with_capacity(4 + payload.len());
        body.extend_from_slice(kind);
        body.extend_from_slice(payload);
        out.extend_from_slice(&body);
        out.extend_from_slice(&crc32(&body).to_be_bytes());
    }

    // Stored (uncompressed) deflate blocks.
    let mut zlib = vec![0x78, 0x01];
    let total = raw.len();
    for (index, block) in raw.chunks(65535).enumerate() {
        let last = (index + 1) * 65535 >= total;
        zlib.push(if last { 1 } else { 0 });
        let len = block.len() as u16;
        zlib.extend_from_slice(&len.to_le_bytes());
        zlib.extend_from_slice(&(!len).to_le_bytes());
        zlib.extend_from_slice(block);
    }
    zlib.extend_from_slice(&adler32(&raw).to_be_bytes());

    let mut png = Vec::new();
    png.extend_from_slice(&[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]);
    let mut ihdr = Vec::with_capacity(13);
    ihdr.extend_from_slice(&(fb.width as u32).to_be_bytes());
    ihdr.extend_from_slice(&(fb.height as u32).to_be_bytes());
    ihdr.extend_from_slice(&[8, 6, 0, 0, 0]);
    chunk(&mut png, b"IHDR", &ihdr);
    chunk(&mut png, b"IDAT", &zlib);
    chunk(&mut png, b"IEND", &[]);
    std::fs::write(path, png)
}

fn main() -> std::io::Result<()> {
    let out_dir = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("target/earth-goldens"));
    std::fs::create_dir_all(&out_dir)?;

    let focus = Geodetic::new(37.6, -122.0, 0.0);
    let mut failures = 0usize;

    println!("Earth goldens at {W}x{H} -> {}", out_dir.display());
    for regime in REGIMES {
        let input = input_for(regime, focus);
        let (first, descriptor, stats, level, cover) = capture(&input);
        let (second, second_descriptor, _, _, second_cover) = capture(&input);

        let identical = first.pixels == second.pixels;
        let descriptor_identical =
            descriptor.to_json_string() == second_descriptor.to_json_string();
        if !identical || !descriptor_identical || cover != second_cover {
            failures += 1;
        }
        // A ground frame is mostly terrain; if the layer silently stops
        // drawing, this must fail instead of writing a false golden.
        if regime.name.starts_with("terrain") && cover < W * H / 2 {
            eprintln!(
                "  {} terrain layer covers only {} of {} pixels",
                regime.name,
                cover,
                W * H
            );
            failures += 1;
        }
        let summary = format!(
            "tiles={} exact={} ancestor={} terrain_tris={} building_tris={} overlay_prims={} draws={} level={}",
            stats.tiles_requested,
            stats.tiles_exact,
            stats.tiles_from_ancestor,
            stats.terrain_triangles,
            stats.building_triangles,
            stats.overlay_primitives,
            stats.draw_calls,
            level
        );
        let png_path = out_dir.join(format!("{}.png", regime.name));
        write_png(&png_path, &first)?;
        std::fs::write(
            out_dir.join(format!("{}.json", regime.name)),
            descriptor.to_json_string(),
        )?;
        std::fs::write(
            out_dir.join(format!("{}.txt", regime.name)),
            format!(
                "regime={} altitude_m={} sim_time={} cloud={} daylight={} {}\nterrain_layer_cover_px={}\ndeterministic={} descriptor_deterministic={} coverage_deterministic={}\n",
                regime.name,
                regime.altitude_m,
                1_767_225_600.0,
                0.0,
                regime.daylight,
                summary,
                cover,
                identical,
                descriptor_identical,
                cover == second_cover
            ),
        )?;
        println!(
            "  {:<16} deterministic={:<5} terrain_cover={:<7} {}",
            regime.name, identical, cover, summary
        );
    }

    if failures > 0 {
        eprintln!("{failures} regime(s) were not deterministic");
        std::process::exit(1);
    }
    Ok(())
}
