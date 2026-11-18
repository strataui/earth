//! Unit and integration tests for NetworkTileSource, PNG decoding, Terrain-RGB,
//! and Slippy XYZ / WMTS map streaming protocols.

use std::time::Duration;
use strata_earth::{
    decode_png_to_rgba, decode_terrain_rgb, format_slippy_url, MockHttpTransport,
    NetworkTileSource, TileKey, TileProviderConfig, TileSource,
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
    let wmts_template =
        "https://server.com/wmts?TileMatrix={TileMatrix}&TileCol={TileCol}&TileRow={TileRow}";
    let mut wmts_counter = 0;
    let wmts_url = format_slippy_url(wmts_template, key, &subdomains, &mut wmts_counter);
    assert_eq!(
        wmts_url,
        "https://server.com/wmts?TileMatrix=14&TileCol=2621&TileRow=6332"
    );
}

#[test]
fn test_decode_png_to_rgba_round_trip() {
    let width = 2;
    let height = 2;
    let expected_pixels = vec![
        255, 0, 0, 255, // Red
        0, 255, 0, 255, // Green
        0, 0, 255, 255, // Blue
        255, 255, 0, 255, // Yellow
    ];

    let png_bytes = encode_test_png(width, height, &expected_pixels);
    let (decoded_w, decoded_h, decoded_rgba) =
        decode_png_to_rgba(&png_bytes).expect("PNG decoding must succeed");

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
    assert!(
        (elevation.heights_m[0] - 0.0).abs() < 0.1,
        "expected 0m, got {}",
        elevation.heights_m[0]
    );
    assert!(
        (elevation.heights_m[1] - 1000.0).abs() < 0.1,
        "expected 1000m, got {}",
        elevation.heights_m[1]
    );
}

#[test]
fn test_network_tile_source_non_blocking_streaming() {
    let key = TileKey::new(5, 10, 10);
    let width = 2;
    let height = 2;
    let test_pixels = vec![
        40, 90, 60, 255, 50, 100, 70, 255, 60, 110, 80, 255, 70, 120, 90, 255,
    ];
    let png_bytes = encode_test_png(width, height, &test_pixels);

    let mock_transport = MockHttpTransport::new();
    let expected_url = "https://tile.openstreetmap.org/5/10/10.png";
    mock_transport.set_response(expected_url, png_bytes);

    let mut source = NetworkTileSource::osm().with_transport(mock_transport.clone());

    // 1. First call: cache miss, enqueues background worker, returns Ok(None) non-blockingly
    let first_result = source
        .imagery(key)
        .expect("call must succeed without blocking");
    assert!(
        first_result.is_none(),
        "first call must return None immediately for smooth fallback"
    );
    assert_eq!(source.stats().network_requests, 1);
    assert_eq!(source.stats().cache_misses, 1);
    assert_eq!(source.stats().cache_hits, 0);

    // 2. Allow background thread to process the task
    std::thread::sleep(Duration::from_millis(60));

    // 3. Second call: poll_completed drains response into LRU cache, returns Ok(Some(image))
    let second_result = source.imagery(key).expect("call must succeed");
    assert!(
        second_result.is_some(),
        "second call must return the downloaded tile from cache"
    );
    let image = second_result.unwrap();
    assert_eq!(image.width, width);
    assert_eq!(image.height, height);
    assert_eq!(image.pixels, test_pixels);
    assert_eq!(source.stats().cache_hits, 1);

    // 4. Third call: immediate in-memory cache hit
    let third_result = source.imagery(key).expect("call must succeed");
    assert!(third_result.is_some());
    assert_eq!(source.stats().cache_hits, 2);
    assert_eq!(
        source.stats().network_requests,
        1,
        "no extra network requests should be issued"
    );
}

#[test]
fn test_network_tile_source_sync_fetch_and_disk_caching() {
    let key = TileKey::new(7, 20, 30);
    let test_pixels = [100, 150, 200, 255].repeat(4);
    let png_bytes = encode_test_png(2, 2, &test_pixels);

    let temp_dir =
        std::env::temp_dir().join(format!("strata_tile_disk_cache_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&temp_dir);

    let mock_transport = MockHttpTransport::new();
    let expected_url = "https://tile.openstreetmap.org/7/20/30.png";
    mock_transport.set_response(expected_url, png_bytes);

    let mut source = NetworkTileSource::osm()
        .with_transport(mock_transport)
        .with_disk_cache(&temp_dir);

    // Synchronous fetch
    let sync_img = source
        .fetch_sync(key)
        .expect("fetch_sync must succeed")
        .expect("image exists");
    assert_eq!(sync_img.width, 2);
    assert_eq!(sync_img.height, 2);
    assert_eq!(sync_img.pixels, test_pixels);

    // Verify written to provider-namespaced disk cache
    let cached_file = temp_dir
        .join("providers")
        .join(source.imagery_cache_namespace())
        .join("7/20/30.png");
    assert!(
        cached_file.is_file(),
        "disk cache file must exist at {}",
        cached_file.display()
    );

    // Create a new source pointing at the same disk cache
    let empty_mock = MockHttpTransport::new(); // has no responses configured
    let mut disk_source = NetworkTileSource::osm()
        .with_transport(empty_mock)
        .with_disk_cache(&temp_dir);

    // Should hit disk cache immediately without network
    let disk_img = disk_source
        .imagery(key)
        .expect("call must succeed")
        .expect("hit from disk cache");
    assert_eq!(disk_img.width, 2);
    assert_eq!(disk_img.height, 2);
    assert_eq!(disk_source.stats().disk_cache_hits, 1);

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_dynamic_priority_and_night_lights_streaming() {
    use strata_earth::Geodetic;

    let far_key = TileKey::new(8, 10, 10);
    let focus_lat = 40.7128;
    let focus_lon = -74.0060;
    // Approx NYC tile at z=8
    let near_key = TileKey::new(8, 75, 96);
    let night_key = TileKey::new(7, 37, 48);

    let px_near = [200, 180, 140, 255].repeat(4);
    let px_far = [10, 20, 30, 255].repeat(4);
    let px_night = [255, 210, 110, 255].repeat(4);

    let transport = MockHttpTransport::new();
    transport.set_response(
        "https://tile.openstreetmap.org/8/10/10.png",
        encode_test_png(2, 2, &px_far),
    );
    transport.set_response(
        "https://tile.openstreetmap.org/8/75/96.png",
        encode_test_png(2, 2, &px_near),
    );
    transport.set_response(
        "https://night.example.test/7/48/37.png",
        encode_test_png(2, 2, &px_night),
    );

    let temp_dir = std::env::temp_dir().join(format!(
        "strata_dynamic_queue_night_{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&temp_dir);

    let mut source = NetworkTileSource::osm()
        .with_night_lights("https://night.example.test/{z}/{y}/{x}.png", 8)
        .with_transport(transport)
        .with_disk_cache(&temp_dir);

    // Generation 1 far request, then move focus to NYC in generation 2 and enqueue near + night tile.
    source.begin_frame(Geodetic::new(0.0, 0.0, 0.0), 2_000_000.0);
    assert!(source.imagery(far_key).unwrap().is_none());

    source.begin_frame(Geodetic::new(focus_lat, focus_lon, 0.0), 400_000.0);
    assert!(source.imagery(near_key).unwrap().is_none());
    assert!(source.night_lights(night_key).unwrap().is_none());

    std::thread::sleep(Duration::from_millis(80));

    let near_img = source
        .imagery(near_key)
        .unwrap()
        .expect("focused near imagery tile must complete");
    assert_eq!(near_img.pixels, px_near);

    let night_img = source
        .night_lights(night_key)
        .unwrap()
        .expect("night lights tile must stream and cache");
    assert_eq!(night_img.pixels, px_night);

    // Night disk cache lives under `{dir}/night/{ns}/{z}/{x}/{y}.jpg` without colliding with daytime tiles.
    let night_ns = source
        .provider_config()
        .night_cache_namespace()
        .expect("night namespace");
    assert!(temp_dir.join("night").join(night_ns).join("7/37/48.jpg").is_file());

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_end_frame_cancels_stale_queued_tasks_and_prioritizes_center_at_low_alt() {
    use std::sync::{Arc, Condvar, Mutex};
    use strata_earth::{Geodetic, HttpTransport, TileError};

    struct GatedMockTransport {
        gate_one: Arc<(Mutex<bool>, Condvar)>,
        gate_rest: Arc<(Mutex<bool>, Condvar)>,
        entered_gate: Arc<(Mutex<usize>, Condvar)>,
        requested: Arc<Mutex<Vec<String>>>,
        png_bytes: Vec<u8>,
    }

    impl HttpTransport for GatedMockTransport {
        fn fetch(&self, url: &str) -> Result<Vec<u8>, TileError> {
            if url.contains("/6/") {
                {
                    let (lk, cv) = &*self.entered_gate;
                    let mut count = lk.lock().unwrap();
                    *count += 1;
                    cv.notify_all();
                }
                let gate = if url.ends_with("/6/0/0.png") {
                    &self.gate_one
                } else {
                    &self.gate_rest
                };
                let (lk, cv) = &**gate;
                let mut released = lk.lock().unwrap();
                while !*released {
                    released = cv.wait(released).unwrap();
                }
                return Ok(self.png_bytes.clone());
            }
            self.requested.lock().unwrap().push(url.to_string());
            Ok(self.png_bytes.clone())
        }
    }

    let png_bytes = encode_test_png(2, 2, &[120, 140, 160, 255].repeat(4));
    let gate_one = Arc::new((Mutex::new(false), Condvar::new()));
    let gate_rest = Arc::new((Mutex::new(false), Condvar::new()));
    let entered_gate = Arc::new((Mutex::new(0usize), Condvar::new()));
    let requested = Arc::new(Mutex::new(Vec::new()));

    let transport = GatedMockTransport {
        gate_one: gate_one.clone(),
        gate_rest: gate_rest.clone(),
        entered_gate: entered_gate.clone(),
        requested: requested.clone(),
        png_bytes,
    };

    let mut source = NetworkTileSource::osm()
        .with_elevation("https://elev.example.test/{z}/{x}/{y}.png", true)
        .with_transport(transport);

    let nyc = Geodetic::new(40.7128, -74.0060, 0.0);
    // Frame 1: occupy all 8 worker threads on gated level-6 tiles so subsequent queued tasks
    // stay in DynamicFetchQueue long enough to verify cancellation and priority ordering.
    source.begin_frame(nyc, 5_000.0);
    for x in 0..8 {
        assert!(source.imagery(TileKey::new(6, x, 0)).unwrap().is_none());
    }
    {
        let (lk, cv) = &*entered_gate;
        let mut count = lk.lock().unwrap();
        let deadline = std::time::Instant::now() + Duration::from_secs(2);
        while *count < 8 {
            let rem = deadline.saturating_duration_since(std::time::Instant::now());
            assert!(!rem.is_zero(), "expected 8 worker threads to reach gate");
            let (g, _) = cv.wait_timeout(count, rem).unwrap();
            count = g;
        }
    }

    // Still in frame 1: enqueue 4 stale tiles (< old 128 cap) that will disappear in frame 2.
    for x in 10..14 {
        assert!(source.imagery(TileKey::new(8, x, 10)).unwrap().is_none());
    }
    source.end_frame();
    assert_eq!(source.stats().cancelled_requests, 0);
    assert_eq!(source.pending_queue_len(), 4);

    // Frame 2 at low altitude (5km) centered on NYC (z=14 tile (4823, 6160)):
    // drop all 4 z=8 tiles, and enqueue a far (~1.5 deg away) elevation tile + center imagery tile.
    let center_img = TileKey::new(14, 4823, 6160);
    let far_elev = TileKey::new(14, 4900, 6160);
    source.begin_frame(nyc, 5_000.0);
    assert!(source.elevation(far_elev).unwrap().is_none());
    assert!(source.imagery(center_img).unwrap().is_none());
    source.end_frame();

    assert_eq!(
        source.stats().cancelled_requests,
        4,
        "all 4 unrequested frame-1 queued tasks must be cancelled immediately at end_frame"
    );
    assert_eq!(source.pending_queue_len(), 2);

    // Release 1 worker first so it pops both surviving tasks strictly sequentially in score order.
    {
        let (lk, cv) = &*gate_one;
        *lk.lock().unwrap() = true;
        cv.notify_all();
    }
    let deadline = std::time::Instant::now() + Duration::from_secs(2);
    loop {
        if requested.lock().unwrap().len() >= 2 {
            break;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "timed out waiting for remaining queued tiles"
        );
        std::thread::sleep(Duration::from_millis(10));
    }
    {
        let (lk, cv) = &*gate_rest;
        *lk.lock().unwrap() = true;
        cv.notify_all();
    }

    let urls = requested.lock().unwrap().clone();
    assert!(
        !urls.iter().any(|u| u.contains("/8/")),
        "cancelled z=8 tiles must never hit HTTP transport: {urls:?}"
    );
    assert_eq!(
        urls.len(),
        2,
        "only the 2 surviving frame-2 tasks should run: {urls:?}"
    );
    assert!(
        urls[0].contains("tile.openstreetmap.org/14/4823/6160.png"),
        "center imagery must outrank far periphery elevation at 5km altitude, got order: {urls:?}"
    );
    assert!(
        urls[1].contains("elev.example.test/14/4900/6160.png"),
        "expected far elevation second, got order: {urls:?}"
    );
}

#[test]
fn test_coarse_chunks_and_maps_prioritized_over_fine_and_heightmaps() {
    use std::sync::{Arc, Condvar, Mutex};
    use strata_earth::{Geodetic, HttpTransport, TileError};

    struct GatedOrderTransport {
        gate_one: Arc<(Mutex<bool>, Condvar)>,
        gate_rest: Arc<(Mutex<bool>, Condvar)>,
        entered_gate: Arc<(Mutex<usize>, Condvar)>,
        requested: Arc<Mutex<Vec<String>>>,
        png_bytes: Vec<u8>,
    }

    impl HttpTransport for GatedOrderTransport {
        fn fetch(&self, url: &str) -> Result<Vec<u8>, TileError> {
            if url.contains("/5/") {
                {
                    let (lk, cv) = &*self.entered_gate;
                    let mut count = lk.lock().unwrap();
                    *count += 1;
                    cv.notify_all();
                }
                let gate = if url.ends_with("/5/0/0.png") {
                    &self.gate_one
                } else {
                    &self.gate_rest
                };
                let (lk, cv) = &**gate;
                let mut released = lk.lock().unwrap();
                while !*released {
                    released = cv.wait(released).unwrap();
                }
                return Ok(self.png_bytes.clone());
            }
            self.requested.lock().unwrap().push(url.to_string());
            Ok(self.png_bytes.clone())
        }
    }

    let png_bytes = encode_test_png(2, 2, &[150, 170, 190, 255].repeat(4));
    let gate_one = Arc::new((Mutex::new(false), Condvar::new()));
    let gate_rest = Arc::new((Mutex::new(false), Condvar::new()));
    let entered_gate = Arc::new((Mutex::new(0usize), Condvar::new()));
    let requested = Arc::new(Mutex::new(Vec::new()));

    let mut source = NetworkTileSource::osm()
        .with_elevation("https://elev.example.test/{z}/{x}/{y}.png", true)
        .with_transport(GatedOrderTransport {
            gate_one: gate_one.clone(),
            gate_rest: gate_rest.clone(),
            entered_gate: entered_gate.clone(),
            requested: requested.clone(),
            png_bytes,
        });

    let nyc = Geodetic::new(40.7128, -74.0060, 0.0);
    source.begin_frame(nyc, 10_000.0);
    for x in 0..8 {
        assert!(source.imagery(TileKey::new(5, x, 0)).unwrap().is_none());
    }
    {
        let (lk, cv) = &*entered_gate;
        let mut count = lk.lock().unwrap();
        let deadline = std::time::Instant::now() + Duration::from_secs(2);
        while *count < 8 {
            let rem = deadline.saturating_duration_since(std::time::Instant::now());
            assert!(!rem.is_zero());
            let (g, _) = cv.wait_timeout(count, rem).unwrap();
            count = g;
        }
    }
    source.end_frame();

    // Same center spot (NYC): enqueue heightmap FIRST, fine periphery map SECOND, fine center map THIRD,
    // and coarse z=8 parent chunk LAST. Queue priority order must be:
    // 1. Coarse z=8 big chunk map
    // 2. Fine z=14 center map
    // 3. Fine z=14 periphery map
    // 4. Coarse z=8 center heightmap
    let center_14 = TileKey::new(14, 4823, 6160);
    let far_14 = TileKey::new(14, 4860, 6160);
    let big_8 = TileKey::new(8, 75, 96);
    let big_elev_8 = TileKey::new(8, 75, 96);

    source.begin_frame(nyc, 10_000.0);
    assert!(source.elevation(big_elev_8).unwrap().is_none());
    assert!(source.imagery(far_14).unwrap().is_none());
    assert!(source.imagery(center_14).unwrap().is_none());
    assert!(source.imagery(big_8).unwrap().is_none());
    source.end_frame();

    {
        let (lk, cv) = &*gate_one;
        *lk.lock().unwrap() = true;
        cv.notify_all();
    }
    let deadline = std::time::Instant::now() + Duration::from_secs(2);
    loop {
        if requested.lock().unwrap().len() >= 4 {
            break;
        }
        assert!(std::time::Instant::now() < deadline);
        std::thread::sleep(Duration::from_millis(10));
    }
    {
        let (lk, cv) = &*gate_rest;
        *lk.lock().unwrap() = true;
        cv.notify_all();
    }

    let urls = requested.lock().unwrap().clone();
    assert_eq!(urls.len(), 4);
    assert!(
        urls[0].contains("tile.openstreetmap.org/8/75/96.png"),
        "expected big z=8 imagery chunk first, got: {urls:?}"
    );
    assert!(
        urls[1].contains("tile.openstreetmap.org/14/4823/6160.png"),
        "expected center z=14 imagery second, got: {urls:?}"
    );
    assert!(
        urls[2].contains("tile.openstreetmap.org/14/4860/6160.png"),
        "expected periphery z=14 imagery third, got: {urls:?}"
    );
    assert!(
        urls[3].contains("elev.example.test/8/75/96.png"),
        "expected heightmap last behind maps, got: {urls:?}"
    );
}

#[test]
fn test_tile_url_quadkey_tms_and_api_key_templates() {
    use strata_earth::{format_tile_url, tile_key_to_quadkey, TileProviderConfig, TileProviderPreset};

    // Level 3 (x=3, y=5): x bits = 011, y bits = 101 -> interleaved quadkey = "213"
    let key = TileKey::new(3, 3, 5);
    assert_eq!(tile_key_to_quadkey(key), "213");
    assert_eq!(tile_key_to_quadkey(TileKey::new(0, 0, 0)), "");

    let subs = vec!["0".to_string(), "1".to_string()];
    let mut c = 0;
    let bing_tpl = TileProviderConfig::preset(TileProviderPreset::BingAerial).imagery_url_template;
    let bing_url = format_tile_url(&bing_tpl, key, &subs, &mut c, None);
    assert_eq!(
        bing_url,
        "https://ecn.t0.tiles.virtualearth.net/tiles/a213.jpeg?g=1"
    );

    let tms_url = format_tile_url(
        "https://tms.example.test/{z}/{x}/{-y}.png?key={api_key}",
        key,
        &subs,
        &mut c,
        Some("SECRET_TOKEN"),
    );
    // z=3 -> 8 rows; -y = 8 - 1 - 5 = 2
    assert_eq!(tms_url, "https://tms.example.test/3/3/2.png?key=SECRET_TOKEN");

    // Verify all public presets produce distinct valid imagery cache namespaces
    let mut seen_ns = std::collections::HashSet::new();
    for &p in TileProviderPreset::public_presets() {
        let ns = TileProviderConfig::preset(p).imagery_cache_namespace();
        assert!(seen_ns.insert(ns), "duplicate cache namespace for preset {p:?}");
    }
}

#[test]
fn test_provider_switching_isolates_caches_and_preserves_shared_elevation() {
    let temp_dir = std::env::temp_dir().join(format!(
        "strata_provider_isolation_{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&temp_dir);

    let key = TileKey::new(6, 12, 24);
    let px_a = [240, 40, 40, 255].repeat(4);
    let px_b = [40, 80, 240, 255].repeat(4);

    let mock = MockHttpTransport::new();
    mock.set_response(
        "https://provider-a.example.test/6/12/24.png",
        encode_test_png(2, 2, &px_a),
    );
    mock.set_response(
        "https://provider-b.example.test/6/12/24.png",
        encode_test_png(2, 2, &px_b),
    );

    let mut cfg_a = TileProviderConfig::custom(
        "prov_a",
        "https://provider-a.example.test/{z}/{x}/{y}.png",
    )
    .with_terrarium_elevation();
    cfg_a.subdomains = vec![];
    let mut cfg_b = TileProviderConfig::custom(
        "prov_b",
        "https://provider-b.example.test/{z}/{x}/{y}.png",
    )
    .with_terrarium_elevation();
    cfg_b.subdomains = vec![];

    let mut src = NetworkTileSource::from_provider(cfg_a)
        .with_transport(mock)
        .with_disk_cache(&temp_dir);

    let img_a = src.fetch_sync(key).unwrap().unwrap();
    assert_eq!(img_a.pixels, px_a);
    assert!(src.cache().peek_imagery(key).is_some());

    // Switch provider sharing the SAME disk_cache_dir and unchanged AWS Terrarium elevation
    src.apply_provider(&cfg_b);
    // Previous imagery in memory must be cleared and disk must NOT read prov_a tile
    assert!(src.cache().peek_imagery(key).is_none());
    assert!(src.imagery(key).unwrap().is_none());

    std::thread::sleep(Duration::from_millis(80));
    let img_b = src
        .imagery(key)
        .unwrap()
        .expect("provider B tile must arrive cleanly");
    assert_eq!(img_b.pixels, px_b, "provider B must not bleed provider A tiles");

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_z0_to_z4_unpacked_provider_streams_and_upgrades_over_embedded_placeholder() {
    use strata_earth::{EarthDefaultTileSource, TileSource};

    let z2 = TileKey::new(2, 1, 1);
    let custom_px = [12, 220, 180, 255].repeat(4);

    let mock = MockHttpTransport::new();
    mock.set_response(
        "https://custom-z2.example.test/2/1/1.png",
        encode_test_png(2, 2, &custom_px),
    );

    let cfg = TileProviderConfig::custom("custom_no_pack", "https://custom-z2.example.test/{z}/{x}/{y}.png");
    let net = NetworkTileSource::from_provider(cfg).with_transport(mock);
    let mut src = EarthDefaultTileSource::with_network(net);
    assert!(!src.has_matching_base_pack());

    // Frame 0: placeholder is immediately available via peek_cached_imagery,
    // while imagery(z=2) triggers an async fetch instead of permanently pinning z<=4.
    assert!(src.peek_cached_imagery(z2).is_some());
    assert!(src.imagery(z2).unwrap().is_none());

    let mut upgraded = None;
    for _ in 0..80 {
        if let Some(img) = src.imagery(z2).unwrap() {
            upgraded = Some(img);
            break;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    let tile = upgraded.expect("z=2 tile must upgrade asynchronously from custom Tile API");
    assert_eq!(tile.pixels, custom_px);
    assert_eq!(src.peek_cached_imagery(z2).unwrap().pixels, custom_px);
}

#[test]
fn test_sebm0001_pack_roundtrip_matches_provider_at_z0_to_z2() {
    use strata_earth::{
        encode_sebm_pack, BaseWorldMapPack, EarthDefaultTileSource, TileSource,
    };

    let temp_dir = std::env::temp_dir().join(format!(
        "strata_sebm_pack_test_{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(&temp_dir).unwrap();

    let custom_px = [210, 140, 45, 255].repeat(4);
    let png = encode_test_png(2, 2, &custom_px);
    let k0 = TileKey::new(0, 0, 0);
    let k2 = TileKey::new(2, 3, 1);
    let packed = encode_sebm_pack(&[(k0, &png), (k2, &png)]);

    let pack_path = temp_dir.join("base_world_map_custom_pack.pack");
    std::fs::write(&pack_path, &packed).unwrap();

    let parsed = BaseWorldMapPack::from_file(&pack_path).expect("parse written SEBM0001 pack");
    assert_eq!(parsed.max_level(), 2);
    assert_eq!(parsed.len(), 2);
    assert_eq!(parsed.tile(k2).unwrap().pixels, custom_px);

    let cfg = TileProviderConfig::custom(
        "custom_pack",
        "https://packed.example.test/{z}/{x}/{y}.png",
    )
    .with_base_pack_id(Some("custom_pack"));
    let mock = MockHttpTransport::new();
    let net = NetworkTileSource::from_provider(cfg).with_transport(mock.clone());
    let mut src = EarthDefaultTileSource::with_network(net).with_pack_dir(&temp_dir);

    assert!(src.has_matching_base_pack());
    // z=0 and z=2 resolve instantaneously from the matching baked pack with zero HTTP fetches
    assert_eq!(src.imagery(k0).unwrap().unwrap().pixels, custom_px);
    assert_eq!(src.imagery(k2).unwrap().unwrap().pixels, custom_px);
    assert!(mock.requested_urls().is_empty());

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_unavailable_placeholder_tile_ignored_and_falls_back_to_ancestor() {
    use strata_earth::is_unavailable_placeholder_tile;

    let w = 256u32;
    let h = 256u32;
    let mut placeholder_rgba = vec![204u8; (w * h * 4) as usize];
    for px in placeholder_rgba.chunks_exact_mut(4) {
        px[3] = 255;
    }
    // Paint simulated "Map data not yet available" dark-grey text strokes in the center band.
    for y in 114..142 {
        for x in (48..208).step_by(3) {
            let idx = ((y * w + x) * 4) as usize;
            placeholder_rgba[idx] = 112;
            placeholder_rgba[idx + 1] = 112;
            placeholder_rgba[idx + 2] = 112;
        }
    }
    assert!(
        is_unavailable_placeholder_tile(w, h, &placeholder_rgba),
        "must detect 'Map data not yet available' grey placeholder tile"
    );

    // Real satellite tile with color/texture variations must NOT be flagged.
    let mut real_rgba = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            real_rgba.push(((x * 3 + y) & 0xFF) as u8);
            real_rgba.push(((x + y * 5 + 60) & 0xFF) as u8);
            real_rgba.push(((x * 2 + 120) & 0xFF) as u8);
            real_rgba.push(255);
        }
    }
    assert!(
        !is_unavailable_placeholder_tile(w, h, &real_rgba),
        "real satellite imagery must not be flagged as placeholder"
    );

    let temp_dir = std::env::temp_dir().join(format!(
        "strata_placeholder_fallback_test_{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&temp_dir);

    let parent_key = TileKey::new(13, 4100, 3050);
    let child_key = TileKey::new(14, 8200, 6100);
    let placeholder_png = encode_test_png(w, h, &placeholder_rgba);
    let real_parent_png = encode_test_png(w, h, &real_rgba);

    let mock = MockHttpTransport::new();
    mock.set_response(
        "https://tile.openstreetmap.org/14/8200/6100.png",
        placeholder_png.clone(),
    );
    mock.set_response(
        "https://tile.openstreetmap.org/13/4100/3050.png",
        real_parent_png,
    );

    let mut source = NetworkTileSource::osm()
        .with_transport(mock)
        .with_disk_cache(&temp_dir);

    // Pre-seed a stale placeholder file on disk to verify it gets purged on read.
    let stale_disk_path = temp_dir
        .join("providers")
        .join(source.imagery_cache_namespace())
        .join("14/8200/6100.png");
    std::fs::create_dir_all(stale_disk_path.parent().unwrap()).unwrap();
    std::fs::write(&stale_disk_path, &placeholder_png).unwrap();
    assert!(stale_disk_path.is_file());

    // Requesting child_key (z=14) must reject the placeholder, delete the stale disk file,
    // and automatically enqueue the lower-res ancestor (z=13).
    assert!(
        source.imagery(child_key).unwrap().is_none(),
        "placeholder child tile must be ignored"
    );
    assert!(
        !stale_disk_path.is_file(),
        "stale placeholder file on disk must be deleted"
    );

    let mut parent_loaded = None;
    for _ in 0..80 {
        let _ = source.imagery(child_key);
        if let Some(img) = source.peek_cached_imagery(parent_key) {
            parent_loaded = Some(img);
            break;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    let parent_img = parent_loaded.expect("lower-res parent tile z=13 must be automatically loaded");
    assert_eq!(parent_img.pixels, real_rgba);

    let _ = std::fs::remove_dir_all(&temp_dir);
}
