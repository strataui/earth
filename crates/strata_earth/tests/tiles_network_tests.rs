//! Unit and integration tests for NetworkTileSource, PNG decoding, Terrain-RGB,
//! and Slippy XYZ / WMTS map streaming protocols.

use std::time::Duration;
use strata_earth::{
    decode_png_to_rgba, decode_terrain_rgb, format_slippy_url,
    MockHttpTransport, NetworkTileSource, TileKey, TileSource,
};

fn crc32_simple(data: &[u8]) -> u32 {
    let mut crc = 0xFFFFFFFFu32;
    for &b in data {
        crc ^= b as u32;
        for _ in 0..8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ 0xEDB88320;
            } else {
                crc >>= 1;
            }
        }
    }
    !crc
}

fn write_chunk(out: &mut Vec<u8>, chunk_type: &[u8; 4], data: &[u8]) {
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    out.extend_from_slice(chunk_type);
    out.extend_from_slice(data);
    let mut crc_buf = Vec::new();
    crc_buf.extend_from_slice(chunk_type);
    crc_buf.extend_from_slice(data);
    let crc = crc32_simple(&crc_buf);
    out.extend_from_slice(&crc.to_be_bytes());
}

fn encode_test_png(width: u32, height: u32, rgba_pixels: &[u8]) -> Vec<u8> {
    use flate2::write::ZlibEncoder;
    use flate2::Compression;
    use std::io::Write;

    let mut out = Vec::new();
    out.extend_from_slice(b"\x89PNG\r\n\x1a\n");

    // IHDR
    let mut ihdr = Vec::new();
    ihdr.extend_from_slice(&width.to_be_bytes());
    ihdr.extend_from_slice(&height.to_be_bytes());
    ihdr.push(8); // 8-bit depth
    ihdr.push(6); // RGBA
    ihdr.push(0); // deflate
    ihdr.push(0); // filter
    ihdr.push(0); // non-interlaced
    write_chunk(&mut out, b"IHDR", &ihdr);

    // IDAT
    let stride = width as usize * 4;
    let mut raw_scanlines = Vec::new();
    for y in 0..height as usize {
        raw_scanlines.push(0); // filter 0 (None)
        raw_scanlines.extend_from_slice(&rgba_pixels[y * stride..(y + 1) * stride]);
    }
    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(&raw_scanlines).unwrap();
    let compressed = encoder.finish().unwrap();
    write_chunk(&mut out, b"IDAT", &compressed);

    // IEND
    write_chunk(&mut out, b"IEND", &[]);
    out
}

#[test]
fn test_format_slippy_url_interpolation() {
    let key = TileKey::new(14, 2621, 6332);
    let subdomains = vec!["a".to_string(), "b".to_string(), "c".to_string()];
    let mut counter = 0;

    // Standard OpenStreetMap XYZ template
    let osm_template = "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png";
    let url1 = format_slippy_url(osm_template, key, &subdomains, &mut counter);
    assert_eq!(url1, "https://a.tile.openstreetmap.org/14/2621/6332.png");

    let url2 = format_slippy_url(osm_template, key, &subdomains, &mut counter);
    assert_eq!(url2, "https://b.tile.openstreetmap.org/14/2621/6332.png");

    let url3 = format_slippy_url(osm_template, key, &subdomains, &mut counter);
    assert_eq!(url3, "https://c.tile.openstreetmap.org/14/2621/6332.png");

    // Standard WMTS query parameters template
    let wmts_template = "https://server.com/wmts?TileMatrix={TileMatrix}&TileCol={TileCol}&TileRow={TileRow}";
    let mut wmts_counter = 0;
    let wmts_url = format_slippy_url(wmts_template, key, &subdomains, &mut wmts_counter);
    assert_eq!(wmts_url, "https://server.com/wmts?TileMatrix=14&TileCol=2621&TileRow=6332");
}

#[test]
fn test_decode_png_to_rgba_round_trip() {
    let width = 2;
    let height = 2;
    let expected_pixels = vec![
        255, 0, 0, 255,   // Red
        0, 255, 0, 255,   // Green
        0, 0, 255, 255,   // Blue
        255, 255, 0, 255, // Yellow
    ];

    let png_bytes = encode_test_png(width, height, &expected_pixels);
    let (decoded_w, decoded_h, decoded_rgba) = decode_png_to_rgba(&png_bytes).expect("PNG decoding must succeed");

    assert_eq!(decoded_w, width);
    assert_eq!(decoded_h, height);
    assert_eq!(decoded_rgba, expected_pixels);
}

#[test]
fn test_decode_terrain_rgb_elevation() {
    // Test known elevations:
    // 1. Sea level (0m):
    //    height = -10000 + (R*65536 + G*256 + B)*0.1 = 0
    //    (R*65536 + G*256 + B) = 100000 -> R=1, G=134, B=160
    // 2. 1000m:
    //    110000 -> 110000 / 65536 = 1, rem 44464; 44464 / 256 = 173, rem 176
    //    R=1, G=173, B=176
    let pixels = vec![
        1, 134, 160, 255, // 0.0 m
        1, 173, 176, 255, // 1000.0 m
    ];
    let elevation = decode_terrain_rgb(2, 1, &pixels);
    assert_eq!(elevation.width, 2);
    assert_eq!(elevation.height, 1);
    assert!((elevation.heights_m[0] - 0.0).abs() < 0.1, "expected 0m, got {}", elevation.heights_m[0]);
    assert!((elevation.heights_m[1] - 1000.0).abs() < 0.1, "expected 1000m, got {}", elevation.heights_m[1]);
}

#[test]
fn test_network_tile_source_non_blocking_streaming() {
    let key = TileKey::new(5, 10, 10);
    let width = 2;
    let height = 2;
    let test_pixels = vec![
        40, 90, 60, 255,
        50, 100, 70, 255,
        60, 110, 80, 255,
        70, 120, 90, 255,
    ];
    let png_bytes = encode_test_png(width, height, &test_pixels);

    let mock_transport = MockHttpTransport::new();
    let expected_url = "https://tile.openstreetmap.org/5/10/10.png";
    mock_transport.set_response(expected_url, png_bytes);

    let mut source = NetworkTileSource::osm().with_transport(mock_transport.clone());

    // 1. First call: cache miss, enqueues background worker, returns Ok(None) non-blockingly
    let first_result = source.imagery(key).expect("call must succeed without blocking");
    assert!(first_result.is_none(), "first call must return None immediately for smooth fallback");
    assert_eq!(source.stats().network_requests, 1);
    assert_eq!(source.stats().cache_misses, 1);
    assert_eq!(source.stats().cache_hits, 0);

    // 2. Allow background thread to process the task
    std::thread::sleep(Duration::from_millis(60));

    // 3. Second call: poll_completed drains response into LRU cache, returns Ok(Some(image))
    let second_result = source.imagery(key).expect("call must succeed");
    assert!(second_result.is_some(), "second call must return the downloaded tile from cache");
    let image = second_result.unwrap();
    assert_eq!(image.width, width);
    assert_eq!(image.height, height);
    assert_eq!(image.pixels, test_pixels);
    assert_eq!(source.stats().cache_hits, 1);

    // 4. Third call: immediate in-memory cache hit
    let third_result = source.imagery(key).expect("call must succeed");
    assert!(third_result.is_some());
    assert_eq!(source.stats().cache_hits, 2);
    assert_eq!(source.stats().network_requests, 1, "no extra network requests should be issued");
}

#[test]
fn test_network_tile_source_sync_fetch_and_disk_caching() {
    let key = TileKey::new(7, 20, 30);
    let test_pixels = [100, 150, 200, 255].repeat(4);
    let png_bytes = encode_test_png(2, 2, &test_pixels);

    let temp_dir = std::env::temp_dir().join(format!("strata_tile_disk_cache_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&temp_dir);

    let mock_transport = MockHttpTransport::new();
    let expected_url = "https://tile.openstreetmap.org/7/20/30.png";
    mock_transport.set_response(expected_url, png_bytes);

    let mut source = NetworkTileSource::osm()
        .with_transport(mock_transport)
        .with_disk_cache(&temp_dir);

    // Synchronous fetch
    let sync_img = source.fetch_sync(key).expect("fetch_sync must succeed").expect("image exists");
    assert_eq!(sync_img.width, 2);
    assert_eq!(sync_img.height, 2);
    assert_eq!(sync_img.pixels, test_pixels);

    // Verify written to disk cache
    let cached_file = temp_dir.join("7/20/30.png");
    assert!(cached_file.is_file(), "disk cache file must exist at {}", cached_file.display());

    // Create a new source pointing at the same disk cache
    let empty_mock = MockHttpTransport::new(); // has no responses configured
    let mut disk_source = NetworkTileSource::osm()
        .with_transport(empty_mock)
        .with_disk_cache(&temp_dir);

    // Should hit disk cache immediately without network
    let disk_img = disk_source.imagery(key).expect("call must succeed").expect("hit from disk cache");
    assert_eq!(disk_img.width, 2);
    assert_eq!(disk_img.height, 2);
    assert_eq!(disk_source.stats().disk_cache_hits, 1);

    let _ = std::fs::remove_dir_all(&temp_dir);
}
