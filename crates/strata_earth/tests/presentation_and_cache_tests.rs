//! Integration tests for Earth 3D native presentation and bounded LRU tile caching.

use strata_earth::{
    DirectoryTileSource, EarthInput, EarthScene, Geodetic, LruTileCache, LruTileCacheSource,
    MemoryTileSource, SurfaceMode, TileElevation, TileImage, TileKey, TileSource,
    DEFAULT_MAX_CACHE_BYTES,
};
use strata_layout::Rect;
use strata_render::compositor::{DisplayList, DrawCommand};
use strata_render::gpu_context::GpuSurfacePresenter;
use strata_render::swapchain::{SoftwareSurfacePresenter, SurfacePresenter, SwapchainFormat};

fn solid_tile(r: u8, g: u8, b: u8, size: u32) -> TileImage {
    let mut pixels = Vec::with_capacity((size * size * 4) as usize);
    for _ in 0..(size * size) {
        pixels.extend_from_slice(&[r, g, b, 255]);
    }
    TileImage {
        width: size,
        height: size,
        pixels,
    }
}

fn solid_elevation(val: f32, size: u32) -> TileElevation {
    TileElevation {
        width: size,
        height: size,
        heights_m: vec![val; (size * size) as usize],
    }
}

#[test]
fn test_earth_scene_to_draw_command_and_display_list() {
    let mut input = EarthInput::default();
    input.camera.focus = Geodetic::new(37.7749, -122.4194, 0.0);
    input.viewport = Rect::new(0.0, 0.0, 800.0, 600.0);

    let mut source = MemoryTileSource::new();
    let mut scene = EarthScene::new();
    scene.build(&input, &mut source);

    let viewport_rect = Rect::new(0.0, 0.0, 800.0, 600.0);
    let cmd = scene.to_draw_command(viewport_rect);

    match cmd {
        DrawCommand::DrawViewport3D { bounds, scene: s } => {
            assert_eq!(bounds, viewport_rect);
            assert_eq!(s.camera, scene.scene().camera);
        }
        _ => panic!("expected DrawViewport3D command"),
    }

    let mut list = DisplayList::new();
    scene.append_to_display_list(&mut list, viewport_rect);
    assert_eq!(list.commands().len(), 1);
}

#[test]
fn test_earth_scene_render_to_framebuffer() {
    let mut input = EarthInput::default();
    input.camera.focus = Geodetic::new(0.0, 0.0, 0.0);
    input.viewport = Rect::new(0.0, 0.0, 160.0, 120.0);

    let mut source = MemoryTileSource::new();
    let mut scene = EarthScene::new();
    scene.build(&input, &mut source);

    let fb = scene.render_to_framebuffer(160, 120);
    assert_eq!(fb.width, 160);
    assert_eq!(fb.height, 120);
    assert_eq!(fb.pixels.len(), 160 * 120 * 4);

    // Ensure pixels are not all transparent black (the sky ramp renders into background)
    let non_zero_pixels = fb.pixels.iter().filter(|&&byte| byte > 0).count();
    assert!(
        non_zero_pixels > 0,
        "rendered Earth framebuffer must contain non-zero pixel data"
    );
}

#[test]
fn test_earth_scene_present_to_software_surface() {
    let mut input = EarthInput::default();
    input.camera.focus = Geodetic::new(51.5074, -0.1278, 0.0);
    input.viewport = Rect::new(0.0, 0.0, 128.0, 128.0);

    let mut source = MemoryTileSource::new();
    let mut scene = EarthScene::new();
    scene.build(&input, &mut source);

    let mut presenter = SoftwareSurfacePresenter::new(128, 128, SwapchainFormat::Rgba8Unorm);
    assert_eq!(presenter.surface_size(), (128, 128));
    assert_eq!(presenter.presented_frames, 0);

    let res = scene.present_to_surface(&mut presenter);
    assert!(res.is_ok(), "presentation must succeed");
    assert_eq!(presenter.presented_frames, 1);
    assert_eq!(presenter.damage_clipped_frames, 0);

    // Backing buffer must have received the rendered pixels
    let non_zero = presenter
        .backing_buffer
        .iter()
        .filter(|&&byte| byte > 0)
        .count();
    assert!(
        non_zero > 0,
        "software presenter backing buffer must contain frame pixels"
    );
}

#[test]
fn test_earth_scene_present_to_software_surface_with_damage() {
    let mut input = EarthInput::default();
    input.camera.focus = Geodetic::new(48.8566, 2.3522, 0.0);
    input.viewport = Rect::new(0.0, 0.0, 100.0, 100.0);

    let mut source = MemoryTileSource::new();
    let mut scene = EarthScene::new();
    scene.build(&input, &mut source);

    let mut presenter = SoftwareSurfacePresenter::new(100, 100, SwapchainFormat::Rgba8Unorm);
    let damage = Some(Rect::new(10.0, 10.0, 40.0, 40.0));

    let res = scene.present_to_surface_with_damage(&mut presenter, damage);
    assert!(res.is_ok(), "presentation with damage must succeed");
    assert_eq!(presenter.presented_frames, 1);
    assert_eq!(presenter.damage_clipped_frames, 1);
}

#[test]
fn test_earth_scene_present_to_gpu_surface_presenter() {
    let input = EarthInput {
        surface_mode: SurfaceMode::SphericalGlobe,
        viewport: Rect::new(0.0, 0.0, 120.0, 90.0),
        ..Default::default()
    };

    let mut source = MemoryTileSource::new();
    let mut scene = EarthScene::new();
    scene.build(&input, &mut source);

    let mut presenter = GpuSurfacePresenter::new(120, 90, SwapchainFormat::Rgba8Unorm);
    let res = scene.present_to_surface(&mut presenter);
    assert!(res.is_ok(), "gpu surface presentation must succeed");
    assert_eq!(presenter.presented_frames, 1);
}

#[test]
fn test_lru_tile_cache_byte_bounding_and_eviction() {
    // Each 32x32 image is 32*32*4 = 4096 bytes + struct overhead ~ 4128 bytes.
    let tile_img1 = solid_tile(255, 0, 0, 32);
    let size_per_img = tile_img1.byte_size();

    // Set budget to hold exactly 2 tile images
    let budget = size_per_img * 2 + 50;
    let mut cache = LruTileCache::new(budget);
    assert_eq!(cache.max_bytes(), budget);
    assert_eq!(cache.current_bytes(), 0);
    assert_eq!(cache.len(), 0);

    let key1 = TileKey::new(1, 0, 0);
    let key2 = TileKey::new(1, 1, 0);
    let key3 = TileKey::new(1, 0, 1);

    cache.insert_imagery(key1, tile_img1);
    assert_eq!(cache.len(), 1);
    assert!(cache.current_bytes() <= budget);
    assert!(cache.contains_imagery(key1));

    let tile_img2 = solid_tile(0, 255, 0, 32);
    cache.insert_imagery(key2, tile_img2);
    assert_eq!(cache.len(), 2);
    assert!(cache.current_bytes() <= budget);
    assert!(cache.contains_imagery(key1));
    assert!(cache.contains_imagery(key2));

    // Third tile exceeds budget, should evict key1 (the least recently used)
    let tile_img3 = solid_tile(0, 0, 255, 32);
    cache.insert_imagery(key3, tile_img3);

    assert_eq!(cache.len(), 2);
    assert!(cache.current_bytes() <= budget);
    assert!(!cache.contains_imagery(key1), "key1 must be evicted");
    assert!(cache.contains_imagery(key2));
    assert!(cache.contains_imagery(key3));
}

#[test]
fn test_lru_tile_cache_access_refreshes_order() {
    let tile_img = solid_tile(100, 100, 100, 32);
    let size_per_img = tile_img.byte_size();
    let budget = size_per_img * 2 + 50;
    let mut cache = LruTileCache::new(budget);

    let key1 = TileKey::new(2, 0, 0);
    let key2 = TileKey::new(2, 1, 0);
    let key3 = TileKey::new(2, 2, 0);

    cache.insert_imagery(key1, tile_img.clone());
    cache.insert_imagery(key2, tile_img.clone());

    // Access key1 so it becomes the Most Recently Used
    let fetched = cache.get_imagery(key1);
    assert!(fetched.is_some());

    // Now insert key3; since key1 was accessed, key2 is now the LRU and must be evicted
    cache.insert_imagery(key3, tile_img);

    assert!(
        cache.contains_imagery(key1),
        "recently accessed key1 must be retained"
    );
    assert!(
        !cache.contains_imagery(key2),
        "untouched key2 must be evicted"
    );
    assert!(
        cache.contains_imagery(key3),
        "newly inserted key3 must be present"
    );
}

#[test]
fn test_lru_tile_cache_elevation_support() {
    let elev = solid_elevation(120.0, 16);
    let size = elev.byte_size();
    let budget = size * 2 + 50;
    let mut cache = LruTileCache::new(budget);

    let key1 = TileKey::new(3, 0, 0);
    let key2 = TileKey::new(3, 0, 1);
    let key3 = TileKey::new(3, 0, 2);

    cache.insert_elevation(key1, elev.clone());
    cache.insert_elevation(key2, elev.clone());
    assert_eq!(cache.len(), 2);

    let fetched = cache.get_elevation(key1);
    assert!(fetched.is_some());
    assert_eq!(fetched.unwrap().sample_bilinear(0.5, 0.5), 120.0);

    cache.insert_elevation(key3, elev);
    assert!(cache.contains_elevation(key1));
    assert!(!cache.contains_elevation(key2), "key2 must be evicted");
    assert!(cache.contains_elevation(key3));
}

#[test]
fn test_lru_tile_cache_source_decorator() {
    let mut mem_source = MemoryTileSource::new();
    let key1 = TileKey::new(4, 1, 1);
    let key2 = TileKey::new(4, 1, 2);

    mem_source.insert_imagery(key1, solid_tile(10, 20, 30, 16));
    mem_source.insert_elevation(key1, solid_elevation(50.0, 8));

    let mut cached_source = LruTileCacheSource::with_256mb(mem_source);
    assert_eq!(cached_source.cache().max_bytes(), DEFAULT_MAX_CACHE_BYTES);
    assert_eq!(cached_source.hits(), 0);
    assert_eq!(cached_source.misses(), 0);

    // 1. First fetch: cache miss, delegate to inner
    let img1 = cached_source
        .imagery(key1)
        .expect("read ok")
        .expect("found");
    assert_eq!(img1.width, 16);
    assert_eq!(cached_source.hits(), 0);
    assert_eq!(cached_source.misses(), 1);

    // 2. Second fetch: cache hit
    let img2 = cached_source
        .imagery(key1)
        .expect("read ok")
        .expect("found");
    assert_eq!(img2.width, 16);
    assert_eq!(cached_source.hits(), 1);
    assert_eq!(cached_source.misses(), 1);

    // 3. Elevation fetch: cache miss then hit
    let elev1 = cached_source
        .elevation(key1)
        .expect("read ok")
        .expect("found");
    assert_eq!(elev1.width, 8);
    assert_eq!(cached_source.hits(), 1);
    assert_eq!(cached_source.misses(), 2);

    let elev2 = cached_source
        .elevation(key1)
        .expect("read ok")
        .expect("found");
    assert_eq!(elev2.width, 8);
    assert_eq!(cached_source.hits(), 2);
    assert_eq!(cached_source.misses(), 2);

    // 4. Missing key in underlying source
    let missing = cached_source.imagery(key2).expect("read ok");
    assert!(missing.is_none());
    assert_eq!(cached_source.hits(), 2);
    assert_eq!(cached_source.misses(), 3);
}

#[test]
fn test_directory_tile_source_disk_ceiling_and_eviction() {
    let dir = std::env::temp_dir().join(format!("strata_lru_disk_test_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create test dir");

    // 32x32 image with 8-byte header = 8 + 32*32*4 = 4104 bytes on disk.
    let img1 = solid_tile(255, 0, 0, 32);
    let img2 = solid_tile(0, 255, 0, 32);
    let img3 = solid_tile(0, 0, 255, 32);

    // Set max disk budget to 9000 bytes (holds 2 tiles of 4104 bytes, but not 3)
    let source = DirectoryTileSource::new(&dir).with_max_disk_bytes(9000);
    assert_eq!(source.max_disk_bytes(), 9000);

    let key1 = TileKey::new(1, 0, 0);
    let key2 = TileKey::new(1, 0, 1);
    let key3 = TileKey::new(1, 0, 2);

    let path1 = source.write_imagery(key1, &img1).expect("write tile 1");
    std::thread::sleep(std::time::Duration::from_millis(15));
    let path2 = source.write_imagery(key2, &img2).expect("write tile 2");
    std::thread::sleep(std::time::Duration::from_millis(15));

    let usage_2 = source.disk_usage_bytes().expect("disk usage");
    assert!(usage_2 > 8000 && usage_2 <= 9000);
    assert!(path1.exists());
    assert!(path2.exists());

    // Writing third tile will exceed 9000 bytes and trigger eviction of oldest file (path1)
    let path3 = source.write_imagery(key3, &img3).expect("write tile 3");

    let usage_after = source.disk_usage_bytes().expect("disk usage");
    assert!(
        usage_after <= 9000,
        "disk usage must remain <= 9000 bytes, got {usage_after}"
    );
    assert!(
        !path1.exists(),
        "oldest file path1 must have been evicted from disk"
    );
    assert!(path2.exists(), "newer file path2 must be retained");
    assert!(path3.exists(), "newest file path3 must be retained");

    let _ = std::fs::remove_dir_all(&dir);
}
