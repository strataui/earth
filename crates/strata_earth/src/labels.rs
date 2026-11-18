//! Hierarchical Google Earth-style geographic labels, highway route shields,
//! light-green mountain peak markers, smooth zoom-range fade transitions,
//! horizon occlusion, and screen-space collision decluttering.
//!
//! Combines a built-in global public-domain gazetteer (oceans, continents,
//! countries, regions/states, cities, mountain peaks, parks, and major highway
//! shields) with optional non-blocking background OpenStreetMap Overpass queries
//! when zooming into any local region (< 150 km).

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use strata_render::camera64::{DVec3, EARTH_RADIUS_M};

use crate::geodesy::{geodetic_to_ecef, Geodetic, LocalFrame};
use crate::scene::EarthInput;

/// Semantic category determining label color, typography, badge, and priority.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum EarthLabelKind {
    Continent,
    Ocean,
    Country,
    Region,
    City,
    Mountain,
    Park,
    Highway,
}

/// Visual icon glyph optionally drawn alongside a geographic label.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EarthLabelIcon {
    None,
    CityDot,
    /// Rendered as a mountain silhouette on the right side of light-green mountain labels.
    MountainRight,
    ParkLeafRight,
    HighwayShield,
}

/// Geographic feature candidate in WGS84 coordinates with altitude visibility window.
#[derive(Debug, Clone, PartialEq)]
pub struct EarthMapLabel {
    pub id: u64,
    pub text: String,
    pub subtitle: Option<String>,
    pub kind: EarthLabelKind,
    pub lat_deg: f64,
    pub lon_deg: f64,
    pub elevation_m: f64,
    pub min_alt_m: f64,
    pub max_alt_m: f64,
    pub priority: u16,
}

/// Screen-space projected & collision-culled label ready for crisp UI overlay rendering.
#[derive(Debug, Clone, PartialEq)]
pub struct ProjectedEarthLabel {
    pub id: u64,
    pub text: String,
    pub subtitle: Option<String>,
    pub kind: EarthLabelKind,
    pub icon: EarthLabelIcon,
    pub screen_x: f32,
    pub screen_y: f32,
    pub alpha: f32,
    /// RGB `[0..=255]` text / badge accent color matching Google Earth categories.
    pub rgb: [u8; 3],
    pub font_px: f32,
}

#[inline]
fn smoothstep_f64(edge0: f64, edge1: f64, x: f64) -> f64 {
    if (edge1 - edge0).abs() < 1e-9 {
        return if x >= edge1 { 1.0 } else { 0.0 };
    }
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[inline]
fn label_id_hash(kind: EarthLabelKind, text: &str, lat: f64, lon: f64) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64 ^ (kind as u64);
    for b in text.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    h ^= (lat * 100.0).round() as i64 as u64;
    h = h.wrapping_mul(0x100_0000_01b3);
    h ^= (lon * 100.0).round() as i64 as u64;
    h.wrapping_mul(0x100_0000_01b3)
}

type LabelTuple = (
    &'static str,
    Option<&'static str>,
    EarthLabelKind,
    f64,
    f64,
    f64,
    f64,
    f64,
    u16,
);

/// Built-in public-domain world gazetteer spanning oceans, continents, countries,
/// states/provinces, major cities, mountains/parks (with light-green mountain markers),
/// and recognizable regional highway route codes (e.g. `CT05`, `US-101`, `I-80`, `E40`).
const BUILTIN_LABELS: &[LabelTuple] = &[
    // --- Oceans & Seas ---
    (
        "PACIFIC OCEAN",
        None,
        EarthLabelKind::Ocean,
        8.0,
        -155.0,
        0.0,
        1_800_000.0,
        26_000_000.0,
        95,
    ),
    (
        "ATLANTIC OCEAN",
        None,
        EarthLabelKind::Ocean,
        22.0,
        -40.0,
        0.0,
        1_600_000.0,
        26_000_000.0,
        95,
    ),
    (
        "INDIAN OCEAN",
        None,
        EarthLabelKind::Ocean,
        -14.0,
        76.0,
        0.0,
        1_600_000.0,
        26_000_000.0,
        94,
    ),
    (
        "ARCTIC OCEAN",
        None,
        EarthLabelKind::Ocean,
        78.0,
        5.0,
        0.0,
        2_200_000.0,
        26_000_000.0,
        88,
    ),
    (
        "SOUTHERN OCEAN",
        None,
        EarthLabelKind::Ocean,
        -64.0,
        20.0,
        0.0,
        2_200_000.0,
        26_000_000.0,
        86,
    ),
    (
        "Mediterranean Sea",
        None,
        EarthLabelKind::Ocean,
        35.5,
        18.0,
        0.0,
        400_000.0,
        9_500_000.0,
        78,
    ),
    (
        "Black Sea",
        None,
        EarthLabelKind::Ocean,
        43.3,
        34.0,
        0.0,
        220_000.0,
        6_000_000.0,
        70,
    ),
    (
        "Red Sea",
        None,
        EarthLabelKind::Ocean,
        20.5,
        38.5,
        0.0,
        300_000.0,
        7_500_000.0,
        72,
    ),
    (
        "Caribbean Sea",
        None,
        EarthLabelKind::Ocean,
        15.0,
        -75.0,
        0.0,
        400_000.0,
        8_500_000.0,
        74,
    ),
    (
        "Gulf of Mexico",
        None,
        EarthLabelKind::Ocean,
        25.0,
        -90.0,
        0.0,
        400_000.0,
        8_500_000.0,
        74,
    ),
    (
        "South China Sea",
        None,
        EarthLabelKind::Ocean,
        14.0,
        114.0,
        0.0,
        450_000.0,
        8_500_000.0,
        74,
    ),
    (
        "North Sea",
        None,
        EarthLabelKind::Ocean,
        56.5,
        3.5,
        0.0,
        200_000.0,
        5_500_000.0,
        68,
    ),
    (
        "Arabian Sea",
        None,
        EarthLabelKind::Ocean,
        15.0,
        64.0,
        0.0,
        450_000.0,
        9_000_000.0,
        72,
    ),
    (
        "Bay of Bengal",
        None,
        EarthLabelKind::Ocean,
        14.0,
        89.0,
        0.0,
        450_000.0,
        9_000_000.0,
        72,
    ),
    // --- Continents ---
    (
        "NORTH AMERICA",
        None,
        EarthLabelKind::Continent,
        45.0,
        -101.0,
        0.0,
        3_600_000.0,
        25_000_000.0,
        98,
    ),
    (
        "SOUTH AMERICA",
        None,
        EarthLabelKind::Continent,
        -14.0,
        -59.0,
        0.0,
        3_600_000.0,
        25_000_000.0,
        98,
    ),
    (
        "EUROPE",
        None,
        EarthLabelKind::Continent,
        50.5,
        14.0,
        0.0,
        3_200_000.0,
        25_000_000.0,
        98,
    ),
    (
        "AFRICA",
        None,
        EarthLabelKind::Continent,
        4.0,
        20.0,
        0.0,
        3_600_000.0,
        25_000_000.0,
        99,
    ),
    (
        "ASIA",
        None,
        EarthLabelKind::Continent,
        44.0,
        90.0,
        0.0,
        3_800_000.0,
        25_000_000.0,
        99,
    ),
    (
        "AUSTRALIA",
        None,
        EarthLabelKind::Continent,
        -25.0,
        134.0,
        0.0,
        3_400_000.0,
        25_000_000.0,
        96,
    ),
    (
        "ANTARCTICA",
        None,
        EarthLabelKind::Continent,
        -81.0,
        45.0,
        0.0,
        4_000_000.0,
        25_000_000.0,
        92,
    ),
    // --- Countries ---
    (
        "United States",
        None,
        EarthLabelKind::Country,
        39.8,
        -98.6,
        0.0,
        500_000.0,
        14_500_000.0,
        90,
    ),
    (
        "Canada",
        None,
        EarthLabelKind::Country,
        56.1,
        -106.3,
        0.0,
        600_000.0,
        14_500_000.0,
        88,
    ),
    (
        "Mexico",
        None,
        EarthLabelKind::Country,
        23.6,
        -102.5,
        0.0,
        450_000.0,
        11_000_000.0,
        84,
    ),
    (
        "Brazil",
        None,
        EarthLabelKind::Country,
        -10.8,
        -52.9,
        0.0,
        500_000.0,
        14_500_000.0,
        89,
    ),
    (
        "Argentina",
        None,
        EarthLabelKind::Country,
        -35.4,
        -64.2,
        0.0,
        450_000.0,
        11_500_000.0,
        82,
    ),
    (
        "Colombia",
        None,
        EarthLabelKind::Country,
        4.6,
        -74.1,
        0.0,
        350_000.0,
        8_500_000.0,
        77,
    ),
    (
        "Chile",
        None,
        EarthLabelKind::Country,
        -33.4,
        -70.6,
        0.0,
        350_000.0,
        8_500_000.0,
        76,
    ),
    (
        "Peru",
        None,
        EarthLabelKind::Country,
        -9.2,
        -75.0,
        0.0,
        350_000.0,
        8_500_000.0,
        76,
    ),
    (
        "United Kingdom",
        None,
        EarthLabelKind::Country,
        54.0,
        -2.5,
        0.0,
        220_000.0,
        8_500_000.0,
        84,
    ),
    (
        "France",
        None,
        EarthLabelKind::Country,
        46.6,
        2.4,
        0.0,
        220_000.0,
        9_000_000.0,
        85,
    ),
    (
        "Germany",
        None,
        EarthLabelKind::Country,
        51.1,
        10.4,
        0.0,
        220_000.0,
        9_000_000.0,
        85,
    ),
    (
        "Spain",
        None,
        EarthLabelKind::Country,
        40.4,
        -3.7,
        0.0,
        220_000.0,
        8_500_000.0,
        82,
    ),
    (
        "Italy",
        None,
        EarthLabelKind::Country,
        42.8,
        12.6,
        0.0,
        220_000.0,
        8_500_000.0,
        82,
    ),
    (
        "Poland",
        None,
        EarthLabelKind::Country,
        52.0,
        19.4,
        0.0,
        220_000.0,
        7_000_000.0,
        76,
    ),
    (
        "Ukraine",
        None,
        EarthLabelKind::Country,
        49.0,
        31.4,
        0.0,
        260_000.0,
        8_000_000.0,
        78,
    ),
    (
        "Sweden",
        None,
        EarthLabelKind::Country,
        62.0,
        15.0,
        0.0,
        250_000.0,
        7_500_000.0,
        74,
    ),
    (
        "Norway",
        None,
        EarthLabelKind::Country,
        61.5,
        8.8,
        0.0,
        250_000.0,
        7_500_000.0,
        74,
    ),
    (
        "Turkey",
        None,
        EarthLabelKind::Country,
        39.0,
        35.2,
        0.0,
        240_000.0,
        9_500_000.0,
        83,
    ),
    (
        "Greece",
        None,
        EarthLabelKind::Country,
        39.1,
        22.0,
        0.0,
        180_000.0,
        6_500_000.0,
        74,
    ),
    (
        "Egypt",
        None,
        EarthLabelKind::Country,
        26.8,
        30.8,
        0.0,
        300_000.0,
        10_500_000.0,
        84,
    ),
    (
        "Morocco",
        None,
        EarthLabelKind::Country,
        31.8,
        -7.1,
        0.0,
        280_000.0,
        9_000_000.0,
        78,
    ),
    (
        "Algeria",
        None,
        EarthLabelKind::Country,
        28.0,
        2.6,
        0.0,
        380_000.0,
        10_500_000.0,
        79,
    ),
    (
        "Nigeria",
        None,
        EarthLabelKind::Country,
        9.1,
        8.7,
        0.0,
        320_000.0,
        9_500_000.0,
        80,
    ),
    (
        "Kenya",
        None,
        EarthLabelKind::Country,
        0.2,
        37.9,
        0.0,
        280_000.0,
        8_500_000.0,
        77,
    ),
    (
        "Ethiopia",
        None,
        EarthLabelKind::Country,
        9.1,
        40.5,
        0.0,
        320_000.0,
        8_500_000.0,
        77,
    ),
    (
        "South Africa",
        None,
        EarthLabelKind::Country,
        -29.0,
        24.0,
        0.0,
        350_000.0,
        10_500_000.0,
        83,
    ),
    (
        "Saudi Arabia",
        None,
        EarthLabelKind::Country,
        24.0,
        45.0,
        0.0,
        350_000.0,
        10_500_000.0,
        82,
    ),
    (
        "Iran",
        None,
        EarthLabelKind::Country,
        32.4,
        53.7,
        0.0,
        350_000.0,
        9_500_000.0,
        79,
    ),
    (
        "Russia",
        None,
        EarthLabelKind::Country,
        61.5,
        95.0,
        0.0,
        700_000.0,
        15_000_000.0,
        89,
    ),
    (
        "Kazakhstan",
        None,
        EarthLabelKind::Country,
        48.0,
        67.0,
        0.0,
        450_000.0,
        9_500_000.0,
        76,
    ),
    (
        "India",
        None,
        EarthLabelKind::Country,
        21.0,
        78.9,
        0.0,
        450_000.0,
        14_500_000.0,
        91,
    ),
    (
        "China",
        None,
        EarthLabelKind::Country,
        35.8,
        104.2,
        0.0,
        500_000.0,
        14_500_000.0,
        92,
    ),
    (
        "Japan",
        None,
        EarthLabelKind::Country,
        36.2,
        138.2,
        0.0,
        280_000.0,
        11_000_000.0,
        87,
    ),
    (
        "South Korea",
        None,
        EarthLabelKind::Country,
        36.4,
        127.8,
        0.0,
        180_000.0,
        8_000_000.0,
        80,
    ),
    (
        "Thailand",
        None,
        EarthLabelKind::Country,
        15.8,
        101.0,
        0.0,
        250_000.0,
        8_500_000.0,
        77,
    ),
    (
        "Vietnam",
        None,
        EarthLabelKind::Country,
        16.0,
        107.8,
        0.0,
        250_000.0,
        8_500_000.0,
        77,
    ),
    (
        "Indonesia",
        None,
        EarthLabelKind::Country,
        -2.5,
        117.5,
        0.0,
        400_000.0,
        11_000_000.0,
        82,
    ),
    (
        "New Zealand",
        None,
        EarthLabelKind::Country,
        -41.5,
        172.5,
        0.0,
        240_000.0,
        9_000_000.0,
        77,
    ),
    // --- Land Regions / States / Provinces ---
    (
        "California",
        None,
        EarthLabelKind::Region,
        37.2,
        -119.5,
        0.0,
        65_000.0,
        3_600_000.0,
        75,
    ),
    (
        "Nevada",
        None,
        EarthLabelKind::Region,
        39.3,
        -116.6,
        0.0,
        80_000.0,
        3_400_000.0,
        68,
    ),
    (
        "Oregon",
        None,
        EarthLabelKind::Region,
        44.0,
        -120.5,
        0.0,
        80_000.0,
        3_400_000.0,
        68,
    ),
    (
        "Washington",
        None,
        EarthLabelKind::Region,
        47.4,
        -120.7,
        0.0,
        80_000.0,
        3_400_000.0,
        68,
    ),
    (
        "Texas",
        None,
        EarthLabelKind::Region,
        31.5,
        -99.3,
        0.0,
        90_000.0,
        3_800_000.0,
        73,
    ),
    (
        "Colorado",
        None,
        EarthLabelKind::Region,
        39.0,
        -105.5,
        0.0,
        80_000.0,
        3_200_000.0,
        69,
    ),
    (
        "New York",
        None,
        EarthLabelKind::Region,
        42.9,
        -75.5,
        0.0,
        75_000.0,
        3_000_000.0,
        70,
    ),
    (
        "Florida",
        None,
        EarthLabelKind::Region,
        28.0,
        -81.7,
        0.0,
        75_000.0,
        3_200_000.0,
        70,
    ),
    (
        "Alaska",
        None,
        EarthLabelKind::Region,
        64.2,
        -152.0,
        0.0,
        140_000.0,
        4_500_000.0,
        70,
    ),
    (
        "Sahara",
        None,
        EarthLabelKind::Region,
        23.0,
        12.0,
        0.0,
        750_000.0,
        11_000_000.0,
        80,
    ),
    (
        "Anatolia",
        None,
        EarthLabelKind::Region,
        39.1,
        33.5,
        0.0,
        140_000.0,
        3_400_000.0,
        70,
    ),
    (
        "Bavaria",
        None,
        EarthLabelKind::Region,
        48.9,
        11.5,
        0.0,
        75_000.0,
        2_200_000.0,
        68,
    ),
    (
        "Tuscany",
        None,
        EarthLabelKind::Region,
        43.4,
        11.1,
        0.0,
        60_000.0,
        1_900_000.0,
        66,
    ),
    (
        "Andalusia",
        None,
        EarthLabelKind::Region,
        37.5,
        -4.6,
        0.0,
        75_000.0,
        2_200_000.0,
        66,
    ),
    (
        "Catalonia",
        None,
        EarthLabelKind::Region,
        41.8,
        1.6,
        0.0,
        65_000.0,
        2_000_000.0,
        66,
    ),
    (
        "Scotland",
        None,
        EarthLabelKind::Region,
        56.8,
        -4.2,
        0.0,
        75_000.0,
        2_600_000.0,
        68,
    ),
    (
        "Siberia",
        None,
        EarthLabelKind::Region,
        61.0,
        102.0,
        0.0,
        900_000.0,
        12_000_000.0,
        76,
    ),
    (
        "Tibet",
        None,
        EarthLabelKind::Region,
        31.5,
        88.5,
        0.0,
        450_000.0,
        7_500_000.0,
        74,
    ),
    (
        "Hokkaido",
        None,
        EarthLabelKind::Region,
        43.4,
        142.8,
        0.0,
        85_000.0,
        2_600_000.0,
        66,
    ),
    (
        "San Francisco Bay Area",
        None,
        EarthLabelKind::Region,
        37.65,
        -122.25,
        0.0,
        28_000.0,
        420_000.0,
        72,
    ),
    // --- Major & Regional Cities ---
    (
        "San Francisco",
        None,
        EarthLabelKind::City,
        37.7749,
        -122.4194,
        0.0,
        2_000.0,
        5_500_000.0,
        92,
    ),
    (
        "Oakland",
        None,
        EarthLabelKind::City,
        37.8044,
        -122.2712,
        0.0,
        1_500.0,
        240_000.0,
        72,
    ),
    (
        "San Jose",
        None,
        EarthLabelKind::City,
        37.3382,
        -121.8863,
        0.0,
        1_500.0,
        450_000.0,
        76,
    ),
    (
        "Berkeley",
        None,
        EarthLabelKind::City,
        37.8715,
        -122.2730,
        0.0,
        1_500.0,
        110_000.0,
        64,
    ),
    (
        "Marin",
        None,
        EarthLabelKind::City,
        37.9735,
        -122.5311,
        0.0,
        1_500.0,
        95_000.0,
        60,
    ),
    (
        "Los Angeles",
        None,
        EarthLabelKind::City,
        34.0522,
        -118.2437,
        0.0,
        2_000.0,
        6_500_000.0,
        89,
    ),
    (
        "Seattle",
        None,
        EarthLabelKind::City,
        47.6062,
        -122.3321,
        0.0,
        2_000.0,
        4_500_000.0,
        84,
    ),
    (
        "New York",
        None,
        EarthLabelKind::City,
        40.7128,
        -74.0060,
        0.0,
        2_000.0,
        8_500_000.0,
        94,
    ),
    (
        "Chicago",
        None,
        EarthLabelKind::City,
        41.8781,
        -87.6298,
        0.0,
        2_000.0,
        5_500_000.0,
        86,
    ),
    (
        "Washington, D.C.",
        None,
        EarthLabelKind::City,
        38.9072,
        -77.0369,
        0.0,
        2_000.0,
        5_000_000.0,
        85,
    ),
    (
        "Toronto",
        None,
        EarthLabelKind::City,
        43.6532,
        -79.3832,
        0.0,
        2_000.0,
        4_800_000.0,
        83,
    ),
    (
        "Mexico City",
        None,
        EarthLabelKind::City,
        19.4326,
        -99.1332,
        0.0,
        2_000.0,
        6_500_000.0,
        88,
    ),
    (
        "São Paulo",
        None,
        EarthLabelKind::City,
        -23.5505,
        -46.6333,
        0.0,
        2_000.0,
        7_500_000.0,
        89,
    ),
    (
        "Rio de Janeiro",
        None,
        EarthLabelKind::City,
        -22.9068,
        -43.1729,
        0.0,
        2_000.0,
        6_000_000.0,
        85,
    ),
    (
        "Buenos Aires",
        None,
        EarthLabelKind::City,
        -34.6037,
        -58.3816,
        0.0,
        2_000.0,
        6_500_000.0,
        86,
    ),
    (
        "London",
        None,
        EarthLabelKind::City,
        51.5072,
        -0.1276,
        0.0,
        2_000.0,
        8_500_000.0,
        95,
    ),
    (
        "Paris",
        None,
        EarthLabelKind::City,
        48.8566,
        2.3522,
        0.0,
        2_000.0,
        8_500_000.0,
        94,
    ),
    (
        "Berlin",
        None,
        EarthLabelKind::City,
        52.5200,
        13.4050,
        0.0,
        2_000.0,
        6_500_000.0,
        88,
    ),
    (
        "Madrid",
        None,
        EarthLabelKind::City,
        40.4168,
        -3.7038,
        0.0,
        2_000.0,
        6_500_000.0,
        87,
    ),
    (
        "Rome",
        None,
        EarthLabelKind::City,
        41.9028,
        12.4964,
        0.0,
        2_000.0,
        6_500_000.0,
        88,
    ),
    (
        "Istanbul",
        None,
        EarthLabelKind::City,
        41.0082,
        28.9784,
        0.0,
        2_000.0,
        7_500_000.0,
        91,
    ),
    (
        "Ankara",
        None,
        EarthLabelKind::City,
        39.9334,
        32.8597,
        0.0,
        2_000.0,
        4_200_000.0,
        80,
    ),
    (
        "Cairo",
        None,
        EarthLabelKind::City,
        30.0444,
        31.2357,
        0.0,
        2_000.0,
        8_000_000.0,
        92,
    ),
    (
        "Casablanca",
        None,
        EarthLabelKind::City,
        33.5731,
        -7.5898,
        0.0,
        2_000.0,
        4_500_000.0,
        80,
    ),
    (
        "Lagos",
        None,
        EarthLabelKind::City,
        6.5244,
        3.3792,
        0.0,
        2_000.0,
        6_500_000.0,
        86,
    ),
    (
        "Nairobi",
        None,
        EarthLabelKind::City,
        -1.2921,
        36.8219,
        0.0,
        2_000.0,
        5_500_000.0,
        83,
    ),
    (
        "Johannesburg",
        None,
        EarthLabelKind::City,
        -26.2041,
        28.0473,
        0.0,
        2_000.0,
        6_000_000.0,
        84,
    ),
    (
        "Cape Town",
        None,
        EarthLabelKind::City,
        -33.9249,
        18.4241,
        0.0,
        2_000.0,
        5_500_000.0,
        83,
    ),
    (
        "Dubai",
        None,
        EarthLabelKind::City,
        25.2048,
        55.2708,
        0.0,
        2_000.0,
        6_500_000.0,
        88,
    ),
    (
        "Riyadh",
        None,
        EarthLabelKind::City,
        24.7136,
        46.6753,
        0.0,
        2_000.0,
        5_500_000.0,
        82,
    ),
    (
        "Moscow",
        None,
        EarthLabelKind::City,
        55.7558,
        37.6173,
        0.0,
        2_000.0,
        8_000_000.0,
        90,
    ),
    (
        "Mumbai",
        None,
        EarthLabelKind::City,
        19.0760,
        72.8777,
        0.0,
        2_000.0,
        7_500_000.0,
        90,
    ),
    (
        "New Delhi",
        None,
        EarthLabelKind::City,
        28.6139,
        77.2090,
        0.0,
        2_000.0,
        7_500_000.0,
        89,
    ),
    (
        "Beijing",
        None,
        EarthLabelKind::City,
        39.9042,
        116.4074,
        0.0,
        2_000.0,
        8_500_000.0,
        92,
    ),
    (
        "Shanghai",
        None,
        EarthLabelKind::City,
        31.2304,
        121.4737,
        0.0,
        2_000.0,
        8_000_000.0,
        90,
    ),
    (
        "Tokyo",
        None,
        EarthLabelKind::City,
        35.6762,
        139.6503,
        0.0,
        2_000.0,
        9_000_000.0,
        95,
    ),
    (
        "Seoul",
        None,
        EarthLabelKind::City,
        37.5665,
        126.9780,
        0.0,
        2_000.0,
        7_000_000.0,
        88,
    ),
    (
        "Singapore",
        None,
        EarthLabelKind::City,
        1.3521,
        103.8198,
        0.0,
        2_000.0,
        7_500_000.0,
        89,
    ),
    (
        "Sydney",
        None,
        EarthLabelKind::City,
        -33.8688,
        151.2093,
        0.0,
        2_000.0,
        8_000_000.0,
        89,
    ),
    // --- Mountains & Peaks (light green `#8EE59E` + mountain icon on right!) ---
    (
        "Mount Everest",
        Some("8,848 m"),
        EarthLabelKind::Mountain,
        27.9881,
        86.9250,
        8848.0,
        4_000.0,
        4_500_000.0,
        88,
    ),
    (
        "K2",
        Some("8,611 m"),
        EarthLabelKind::Mountain,
        35.8808,
        76.5155,
        8611.0,
        4_000.0,
        3_600_000.0,
        82,
    ),
    (
        "Mont Blanc",
        Some("4,808 m"),
        EarthLabelKind::Mountain,
        45.8326,
        6.8652,
        4808.0,
        3_000.0,
        3_400_000.0,
        84,
    ),
    (
        "Matterhorn",
        Some("4,478 m"),
        EarthLabelKind::Mountain,
        45.9763,
        7.6586,
        4478.0,
        3_000.0,
        2_200_000.0,
        78,
    ),
    (
        "Mount Kilimanjaro",
        Some("5,895 m"),
        EarthLabelKind::Mountain,
        -3.0674,
        37.3556,
        5895.0,
        3_500.0,
        5_200_000.0,
        86,
    ),
    (
        "Mount Fuji",
        Some("3,776 m"),
        EarthLabelKind::Mountain,
        35.3606,
        138.7274,
        3776.0,
        3_000.0,
        3_400_000.0,
        85,
    ),
    (
        "Denali",
        Some("6,190 m"),
        EarthLabelKind::Mountain,
        63.0692,
        -151.0070,
        6190.0,
        4_000.0,
        4_500_000.0,
        83,
    ),
    (
        "Aconcagua",
        Some("6,961 m"),
        EarthLabelKind::Mountain,
        -32.6532,
        -70.0109,
        6961.0,
        4_000.0,
        4_200_000.0,
        82,
    ),
    (
        "Mount Rainier",
        Some("4,392 m"),
        EarthLabelKind::Mountain,
        46.8523,
        -121.7603,
        4392.0,
        3_000.0,
        2_400_000.0,
        80,
    ),
    (
        "Mount Whitney",
        Some("4,421 m"),
        EarthLabelKind::Mountain,
        36.5785,
        -118.2923,
        4421.0,
        3_000.0,
        2_200_000.0,
        78,
    ),
    (
        "Mount Tamalpais",
        Some("784 m"),
        EarthLabelKind::Mountain,
        37.9235,
        -122.5965,
        784.0,
        1_500.0,
        280_000.0,
        80,
    ),
    (
        "Mount Diablo",
        Some("1,173 m"),
        EarthLabelKind::Mountain,
        37.8816,
        -121.9142,
        1173.0,
        1_500.0,
        320_000.0,
        79,
    ),
    (
        "Twin Peaks",
        Some("281 m"),
        EarthLabelKind::Mountain,
        37.7544,
        -122.4477,
        281.0,
        1_200.0,
        55_000.0,
        77,
    ),
    (
        "Mount Olympus",
        Some("2,917 m"),
        EarthLabelKind::Mountain,
        40.0859,
        22.3586,
        2917.0,
        3_000.0,
        2_400_000.0,
        76,
    ),
    (
        "Mount Ararat",
        Some("5,137 m"),
        EarthLabelKind::Mountain,
        39.7019,
        44.2991,
        5137.0,
        3_000.0,
        2_800_000.0,
        78,
    ),
    (
        "Toubkal",
        Some("4,167 m"),
        EarthLabelKind::Mountain,
        31.0592,
        -7.9158,
        4167.0,
        3_000.0,
        3_400_000.0,
        78,
    ),
    (
        "Atlas Mountains",
        None,
        EarthLabelKind::Mountain,
        32.0,
        -5.0,
        2200.0,
        180_000.0,
        6_500_000.0,
        75,
    ),
    (
        "Alps",
        None,
        EarthLabelKind::Mountain,
        46.5,
        9.8,
        2500.0,
        200_000.0,
        6_500_000.0,
        76,
    ),
    (
        "Himalayas",
        None,
        EarthLabelKind::Mountain,
        28.8,
        84.5,
        4500.0,
        300_000.0,
        8_500_000.0,
        80,
    ),
    (
        "Andes",
        None,
        EarthLabelKind::Mountain,
        -18.0,
        -68.5,
        3800.0,
        350_000.0,
        9_500_000.0,
        78,
    ),
    (
        "Rocky Mountains",
        None,
        EarthLabelKind::Mountain,
        43.5,
        -109.5,
        2900.0,
        300_000.0,
        7_500_000.0,
        76,
    ),
    // --- Parks ---
    (
        "Golden Gate Park",
        None,
        EarthLabelKind::Park,
        37.7694,
        -122.4862,
        40.0,
        1_200.0,
        48_000.0,
        78,
    ),
    (
        "Yosemite National Park",
        None,
        EarthLabelKind::Park,
        37.8651,
        -119.5383,
        1800.0,
        4_000.0,
        1_400_000.0,
        75,
    ),
    (
        "Yellowstone National Park",
        None,
        EarthLabelKind::Park,
        44.4280,
        -110.5885,
        2200.0,
        6_000.0,
        1_800_000.0,
        76,
    ),
    (
        "Serengeti National Park",
        None,
        EarthLabelKind::Park,
        -2.3333,
        34.8333,
        1400.0,
        12_000.0,
        2_200_000.0,
        74,
    ),
    // --- Highway Shields (`CT05`, `US-101`, `I-80`, `I-280`, `CA-1`, `E40`, `A1`, `M25`, etc.) ---
    (
        "US-101",
        None,
        EarthLabelKind::Highway,
        37.7340,
        -122.4050,
        15.0,
        1_200.0,
        260_000.0,
        74,
    ),
    (
        "I-80",
        None,
        EarthLabelKind::Highway,
        37.8190,
        -122.3450,
        25.0,
        1_200.0,
        480_000.0,
        76,
    ),
    (
        "I-280",
        None,
        EarthLabelKind::Highway,
        37.6800,
        -122.4550,
        30.0,
        1_200.0,
        220_000.0,
        73,
    ),
    (
        "CA-1",
        None,
        EarthLabelKind::Highway,
        37.7420,
        -122.5070,
        15.0,
        1_200.0,
        240_000.0,
        71,
    ),
    (
        "I-5",
        None,
        EarthLabelKind::Highway,
        38.5700,
        -121.5000,
        10.0,
        8_000.0,
        850_000.0,
        70,
    ),
    (
        "I-95",
        None,
        EarthLabelKind::Highway,
        40.7600,
        -73.9600,
        15.0,
        4_000.0,
        750_000.0,
        72,
    ),
    (
        "I-10",
        None,
        EarthLabelKind::Highway,
        33.9800,
        -118.1800,
        40.0,
        4_000.0,
        750_000.0,
        70,
    ),
    (
        "CT05",
        None,
        EarthLabelKind::Highway,
        40.7850,
        -72.9200,
        20.0,
        1_200.0,
        380_000.0,
        72,
    ),
    (
        "M25",
        None,
        EarthLabelKind::Highway,
        51.4950,
        -0.4950,
        25.0,
        2_000.0,
        420_000.0,
        72,
    ),
    (
        "A1",
        None,
        EarthLabelKind::Highway,
        48.9400,
        2.3800,
        40.0,
        2_000.0,
        420_000.0,
        72,
    ),
    (
        "E40",
        None,
        EarthLabelKind::Highway,
        50.8500,
        4.3500,
        35.0,
        3_000.0,
        620_000.0,
        71,
    ),
    (
        "E75",
        None,
        EarthLabelKind::Highway,
        44.8100,
        20.4600,
        85.0,
        3_000.0,
        620_000.0,
        71,
    ),
    (
        "O-3",
        None,
        EarthLabelKind::Highway,
        41.0500,
        28.9900,
        70.0,
        2_000.0,
        420_000.0,
        72,
    ),
    (
        "Tomei",
        None,
        EarthLabelKind::Highway,
        35.5800,
        139.5200,
        45.0,
        2_000.0,
        420_000.0,
        71,
    ),
];

/// Returns the built-in public-domain world label gazetteer.
pub fn builtin_world_labels() -> Vec<EarthMapLabel> {
    BUILTIN_LABELS
        .iter()
        .map(
            |&(
                text,
                subtitle,
                kind,
                lat_deg,
                lon_deg,
                elevation_m,
                min_alt_m,
                max_alt_m,
                priority,
            )| {
                EarthMapLabel {
                    id: label_id_hash(kind, text, lat_deg, lon_deg),
                    text: text.to_string(),
                    subtitle: subtitle.map(ToString::to_string),
                    kind,
                    lat_deg,
                    lon_deg,
                    elevation_m,
                    min_alt_m,
                    max_alt_m,
                    priority,
                }
            },
        )
        .collect()
}

/// Optional live OpenStreetMap Overpass enricher for zoomed local peaks (`natural=peak`),
/// town/suburb names (`place`), and highway route codes (`highway` `ref`, e.g. `CT05`, `US-101`).
pub struct LiveMapLabelStore {
    inner: Arc<Mutex<LiveLabelState>>,
    wakeup: Option<Arc<dyn Fn() + Send + Sync>>,
}

struct LiveLabelState {
    sectors: HashMap<(i32, i32), Vec<EarthMapLabel>>,
    in_flight: HashMap<(i32, i32), bool>,
}

impl LiveMapLabelStore {
    pub fn new(wakeup: Option<Arc<dyn Fn() + Send + Sync>>) -> Self {
        Self {
            inner: Arc::new(Mutex::new(LiveLabelState {
                sectors: HashMap::new(),
                in_flight: HashMap::new(),
            })),
            wakeup,
        }
    }

    /// Triggers a background Overpass fetch for the `0.5 deg` sector around `(lat_deg, lon_deg)`
    /// when altitude `< 180 km` and offline mode is not enabled. Never blocks.
    pub fn ensure_local_sector(&self, lat_deg: f64, lon_deg: f64, altitude_m: f64) {
        if altitude_m > 180_000.0 || std::env::var_os("STRATA_EARTH_OFFLINE").is_some() {
            return;
        }
        let sector = (
            (lat_deg * 2.0).floor() as i32,
            (lon_deg * 2.0).floor() as i32,
        );
        {
            let Ok(mut guard) = self.inner.lock() else {
                return;
            };
            if guard.sectors.contains_key(&sector) || guard.in_flight.contains_key(&sector) {
                return;
            }
            guard.in_flight.insert(sector, true);
        }

        let s_lat = sector.0 as f64 * 0.5;
        let w_lon = sector.1 as f64 * 0.5;
        let n_lat = s_lat + 0.5;
        let e_lon = w_lon + 0.5;
        let query = format!(
            "[out:json][timeout:8];(\
            node[\"natural\"=\"peak\"][\"name\"]({s_lat},{w_lon},{n_lat},{e_lon});\
            node[\"place\"~\"city|town|suburb\"][\"name\"]({s_lat},{w_lon},{n_lat},{e_lon});\
            way[\"highway\"~\"motorway|trunk|primary\"][\"ref\"]({s_lat},{w_lon},{n_lat},{e_lon});\
            );out center 40;"
        );
        let url = format!(
            "https://overpass-api.de/api/interpreter?data={}",
            url_encode_minimal(&query)
        );
        let inner = Arc::clone(&self.inner);
        let wake = self.wakeup.clone();
        std::thread::Builder::new()
            .name("strata-earth-labels".to_string())
            .spawn(move || {
                let mut labels = Vec::new();
                if let Ok(resp) = ureq::get(&url)
                    .timeout(std::time::Duration::from_secs(10))
                    .set("User-Agent", "strata-earth/0.1")
                    .call()
                {
                    if let Ok(text) = resp.into_string() {
                        labels = parse_overpass_labels_json(&text);
                    }
                }
                if let Ok(mut guard) = inner.lock() {
                    guard.in_flight.remove(&sector);
                    guard.sectors.insert(sector, labels);
                }
                if let Some(w) = wake {
                    w();
                }
            })
            .ok();
    }

    pub fn collect_nearby(&self, lat_deg: f64, lon_deg: f64) -> Vec<EarthMapLabel> {
        let sector = (
            (lat_deg * 2.0).floor() as i32,
            (lon_deg * 2.0).floor() as i32,
        );
        let Ok(guard) = self.inner.lock() else {
            return Vec::new();
        };
        let mut out = Vec::new();
        for dy in -1..=1 {
            for dx in -1..=1 {
                if let Some(list) = guard.sectors.get(&(sector.0 + dy, sector.1 + dx)) {
                    out.extend(list.iter().cloned());
                }
            }
        }
        out
    }
}

fn url_encode_minimal(s: &str) -> String {
    let mut out = String::with_capacity(s.len() * 2);
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char);
            }
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

fn find_json_str(obj: &str, key: &str) -> Option<String> {
    let pat = format!("\"{key}\"");
    let pos = obj.find(&pat)?;
    let rest = obj[pos + pat.len()..].trim_start();
    let after = rest.strip_prefix(':')?.trim_start();
    let inner = after.strip_prefix('"')?;
    let end = inner.find('"')?;
    Some(inner[..end].to_string())
}

fn find_json_f64(obj: &str, key: &str) -> Option<f64> {
    let pat = format!("\"{key}\"");
    let pos = obj.find(&pat)?;
    let rest = obj[pos + pat.len()..].trim_start();
    let after = rest.strip_prefix(':')?.trim_start();
    let end = after
        .find(|c: char| !(c.is_ascii_digit() || c == '.' || c == '-' || c == '+'))
        .unwrap_or(after.len());
    after[..end].parse().ok()
}

/// Parses Overpass JSON nodes/ways into peak (`Mountain`), `place` (`City`/`Region`), and `ref` (`Highway`) labels.
pub fn parse_overpass_labels_json(json: &str) -> Vec<EarthMapLabel> {
    let mut out = Vec::new();
    let mut seen_names = HashMap::new();
    let Some(el_pos) = json.find("\"elements\"") else {
        return out;
    };
    let slice = &json[el_pos..];
    let mut depth = 0i32;
    let mut obj_start = None;
    let mut in_str = false;
    let mut esc = false;
    for (idx, ch) in slice.char_indices() {
        if esc {
            esc = false;
            continue;
        }
        if in_str {
            if ch == '\\' {
                esc = true;
            } else if ch == '"' {
                in_str = false;
            }
            continue;
        }
        match ch {
            '"' => in_str = true,
            '{' => {
                if depth == 0 {
                    obj_start = Some(idx);
                }
                depth += 1;
            }
            '}' => {
                depth -= 1;
                if depth == 0 {
                    if let Some(st) = obj_start.take() {
                        let obj = &slice[st..=idx];
                        let lat = find_json_f64(obj, "lat");
                        let lon = find_json_f64(obj, "lon");
                        if let (Some(lat_deg), Some(lon_deg)) = (lat, lon) {
                            if let Some(ref_code) = find_json_str(obj, "ref") {
                                if find_json_str(obj, "highway").is_some()
                                    && ref_code.len() <= 10
                                    && !seen_names.contains_key(&ref_code)
                                {
                                    seen_names.insert(ref_code.clone(), true);
                                    out.push(EarthMapLabel {
                                        id: label_id_hash(
                                            EarthLabelKind::Highway,
                                            &ref_code,
                                            lat_deg,
                                            lon_deg,
                                        ),
                                        text: ref_code,
                                        subtitle: None,
                                        kind: EarthLabelKind::Highway,
                                        lat_deg,
                                        lon_deg,
                                        elevation_m: 10.0,
                                        min_alt_m: 800.0,
                                        max_alt_m: 220_000.0,
                                        priority: 69,
                                    });
                                    continue;
                                }
                            }
                            if let Some(name) = find_json_str(obj, "name") {
                                let natural = find_json_str(obj, "natural");
                                if natural.as_deref() == Some("peak")
                                    && !seen_names.contains_key(&name)
                                {
                                    seen_names.insert(name.clone(), true);
                                    let ele_m = find_json_str(obj, "ele")
                                        .and_then(|e| e.parse::<f64>().ok())
                                        .unwrap_or(400.0);
                                    out.push(EarthMapLabel {
                                        id: label_id_hash(
                                            EarthLabelKind::Mountain,
                                            &name,
                                            lat_deg,
                                            lon_deg,
                                        ),
                                        text: name,
                                        subtitle: Some(format!("{} m", ele_m.round() as i32)),
                                        kind: EarthLabelKind::Mountain,
                                        lat_deg,
                                        lon_deg,
                                        elevation_m: ele_m,
                                        min_alt_m: 800.0,
                                        max_alt_m: 180_000.0,
                                        priority: 72,
                                    });
                                } else if find_json_str(obj, "place").is_some()
                                    && !seen_names.contains_key(&name)
                                {
                                    seen_names.insert(name.clone(), true);
                                    out.push(EarthMapLabel {
                                        id: label_id_hash(
                                            EarthLabelKind::City,
                                            &name,
                                            lat_deg,
                                            lon_deg,
                                        ),
                                        text: name,
                                        subtitle: None,
                                        kind: EarthLabelKind::City,
                                        lat_deg,
                                        lon_deg,
                                        elevation_m: 0.0,
                                        min_alt_m: 800.0,
                                        max_alt_m: 160_000.0,
                                        priority: 66,
                                    });
                                }
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }
    out
}

#[inline]
pub fn label_style(kind: EarthLabelKind) -> ([u8; 3], EarthLabelIcon, f32) {
    match kind {
        EarthLabelKind::Continent => ([232, 239, 248], EarthLabelIcon::None, 12.5),
        EarthLabelKind::Ocean => ([136, 182, 224], EarthLabelIcon::None, 11.5),
        EarthLabelKind::Country => ([224, 232, 244], EarthLabelIcon::None, 12.0),
        EarthLabelKind::Region => ([188, 203, 222], EarthLabelIcon::None, 11.0),
        EarthLabelKind::City => ([248, 250, 252], EarthLabelIcon::CityDot, 11.5),
        // Light green with mountain peak symbol on the right, matching Google Earth landmarks
        EarthLabelKind::Mountain => ([142, 234, 156], EarthLabelIcon::MountainRight, 11.0),
        EarthLabelKind::Park => ([122, 224, 146], EarthLabelIcon::ParkLeafRight, 10.5),
        EarthLabelKind::Highway => ([240, 253, 244], EarthLabelIcon::HighwayShield, 10.5),
    }
}

/// Reusable geodetic-to-screen projector honoring curved 3D, 2.5D tangent flatten, and 2D map morphs.
#[derive(Clone, Copy)]
pub struct EarthScreenProjector {
    eye: DVec3,
    vp: [[f32; 4]; 4],
    focus_lat: f64,
    focus_lon: f64,
    focus_frame: LocalFrame,
    focus_ecef: DVec3,
    cos_scale: f64,
    flat_t: f64,
    map_2d: f64,
    vw: f32,
    vh: f32,
}

impl EarthScreenProjector {
    /// Builds a screen projector matching `input`'s exact camera & surface morph matrices.
    pub fn from_input(input: &EarthInput) -> Self {
        let vw = input.viewport.width.max(1.0);
        let vh = input.viewport.height.max(1.0);
        let cam = &input.camera;
        let ll = input.low_level_state();
        let flat_t = ll.local_25d_blend.max(ll.map_2d_morph) as f64;
        let map_2d = ll.map_2d_morph as f64;
        let cam_u = crate::worker::RetainedTerrainBuilder::camera_from_input(input);
        let eye = DVec3::new(cam_u.eye_ecef[0], cam_u.eye_ecef[1], cam_u.eye_ecef[2]);
        let focus_surface = Geodetic::new(cam.focus.lat_deg, cam.focus.lon_deg, 0.0);
        let focus_frame = LocalFrame::at(focus_surface.lat_deg, focus_surface.lon_deg);
        let focus_ecef = geodetic_to_ecef(focus_surface);
        let cos_focus = focus_surface.lat_deg.to_radians().cos().abs().max(0.25);
        let cos_scale = cos_focus + (1.0 - cos_focus) * map_2d;
        Self {
            eye,
            vp: cam_u.view_proj_rte,
            focus_lat: cam.focus.lat_deg,
            focus_lon: cam.focus.lon_deg,
            focus_frame,
            focus_ecef,
            cos_scale,
            flat_t,
            map_2d,
            vw,
            vh,
        }
    }

    /// Projects `(lat_deg, lon_deg, elev_m)` to `(screen_x, screen_y, ndc_x, ndc_y, horizon_fade)`.
    pub fn project_raw(
        &self,
        lat_deg: f64,
        lon_deg: f64,
        elev_m: f64,
    ) -> Option<(f32, f32, f32, f32, f32)> {
        let p_curved = geodetic_to_ecef(Geodetic::new(lat_deg, lon_deg, elev_m));
        let p_world = if self.flat_t <= 1e-6 {
            p_curved
        } else {
            let dlon_deg = ((lon_deg - self.focus_lon + 180.0).rem_euclid(360.0)) - 180.0;
            let dlat_deg = lat_deg - self.focus_lat;
            let east_m = dlon_deg.to_radians() * EARTH_RADIUS_M * self.cos_scale;
            let north_m = dlat_deg.to_radians() * EARTH_RADIUS_M;
            let up_h = elev_m * (1.0 - self.map_2d);
            let p_flat = self
                .focus_ecef
                .add(self.focus_frame.east.scale(east_m))
                .add(self.focus_frame.north.scale(north_m))
                .add(self.focus_frame.up.scale(up_h));
            DVec3::new(
                p_curved.x + (p_flat.x - p_curved.x) * self.flat_t,
                p_curved.y + (p_flat.y - p_curved.y) * self.flat_t,
                p_curved.z + (p_flat.z - p_curved.z) * self.flat_t,
            )
        };

        let to_eye = self.eye.sub(p_world);
        let dist = to_eye.length().max(1.0);
        let normal = if self.flat_t < 0.5 {
            p_curved.normalize()
        } else {
            self.focus_frame.up
        };
        let cos_horizon = normal.dot(to_eye.scale(1.0 / dist));
        let horizon_fade = if self.map_2d > 0.35 {
            1.0
        } else {
            smoothstep_f64(0.06, 0.24, cos_horizon) as f32
        };
        if horizon_fade <= 0.01 {
            return None;
        }

        let rel = p_world.sub(self.eye);
        let rx = rel.x as f32;
        let ry = rel.y as f32;
        let rz = rel.z as f32;
        let clip_x =
            self.vp[0][0] * rx + self.vp[1][0] * ry + self.vp[2][0] * rz + self.vp[3][0];
        let clip_y =
            self.vp[0][1] * rx + self.vp[1][1] * ry + self.vp[2][1] * rz + self.vp[3][1];
        let clip_z =
            self.vp[0][2] * rx + self.vp[1][2] * ry + self.vp[2][2] * rz + self.vp[3][2];
        let clip_w =
            self.vp[0][3] * rx + self.vp[1][3] * ry + self.vp[2][3] * rz + self.vp[3][3];
        if clip_w <= 0.001 {
            return None;
        }
        let ndc_x = clip_x / clip_w;
        let ndc_y = clip_y / clip_w;
        let ndc_z = clip_z / clip_w;
        if !(0.0..=1.0).contains(&ndc_z) {
            return None;
        }
        let screen_x = (ndc_x + 1.0) * 0.5 * self.vw;
        let screen_y = (1.0 - ndc_y) * 0.5 * self.vh;
        Some((screen_x, screen_y, ndc_x, ndc_y, horizon_fade))
    }

    /// Projects a geodetic coordinate into pixel coordinates `(x, y)` inside `[0..vw, 0..vh]`.
    pub fn project_point(&self, lat_deg: f64, lon_deg: f64, elev_m: f64) -> Option<(f32, f32)> {
        let (sx, sy, ndc_x, ndc_y, _) = self.project_raw(lat_deg, lon_deg, elev_m)?;
        if ndc_x.abs() > 1.0 || ndc_y.abs() > 1.0 {
            return None;
        }
        Some((sx, sy))
    }
}

/// Projects `labels` into screen coordinates using `input`'s exact camera & seamless 3D/2.5D/2D
/// morph state, applies smooth altitude/horizon fades, and declutters overlapping screen boxes.
pub fn project_earth_labels(
    input: &EarthInput,
    labels: &[EarthMapLabel],
    max_visible: usize,
) -> Vec<ProjectedEarthLabel> {
    if !input.layers.labels || labels.is_empty() {
        return Vec::new();
    }
    let projector = EarthScreenProjector::from_input(input);
    let alt_m = input.camera.altitude_m();
    let vw = projector.vw;
    let vh = projector.vh;

    let mut candidates: Vec<(u16, f32, ProjectedEarthLabel)> = Vec::new();
    for lbl in labels {
        let fade_in = smoothstep_f64(lbl.min_alt_m * 0.70, lbl.min_alt_m * 1.45, alt_m);
        let fade_out = 1.0 - smoothstep_f64(lbl.max_alt_m * 0.68, lbl.max_alt_m * 1.25, alt_m);
        let alt_alpha = fade_in * fade_out;
        if alt_alpha <= 0.02 {
            continue;
        }

        let Some((raw_x, raw_y, ndc_x, ndc_y, horizon_fade)) =
            projector.project_raw(lbl.lat_deg, lbl.lon_deg, lbl.elevation_m)
        else {
            continue;
        };
        let alpha = (alt_alpha as f32) * horizon_fade;
        if alpha <= 0.03 {
            continue;
        }
        if ndc_x.abs() > 0.96 || ndc_y.abs() > 0.94 {
            continue;
        }

        let screen_x = raw_x.clamp(20.0, vw - 20.0);
        let screen_y = raw_y.clamp(18.0, vh - 18.0);
        // Keep clear of top-right floating control card
        if screen_x > vw - 455.0 && screen_y < 145.0 {
            continue;
        }

        let (rgb, icon, font_px) = label_style(lbl.kind);
        candidates.push((
            lbl.priority,
            alpha,
            ProjectedEarthLabel {
                id: lbl.id,
                text: lbl.text.clone(),
                subtitle: lbl.subtitle.clone(),
                kind: lbl.kind,
                icon,
                screen_x,
                screen_y,
                alpha,
                rgb,
                font_px,
            },
        ));
    }

    candidates.sort_by(|a, b| {
        b.0.cmp(&a.0)
            .then_with(|| b.1.total_cmp(&a.1))
            .then_with(|| a.2.id.cmp(&b.2.id))
    });

    // Screen-space collision decluttering so labels never stack noisily
    let mut accepted: Vec<ProjectedEarthLabel> =
        Vec::with_capacity(max_visible.min(candidates.len()));
    let mut boxes: Vec<[f32; 4]> = Vec::new();
    for (_, _, item) in candidates {
        if accepted.len() >= max_visible {
            break;
        }
        let char_w = item.font_px * 0.60;
        let extra = match item.icon {
            EarthLabelIcon::MountainRight | EarthLabelIcon::ParkLeafRight => 24.0,
            EarthLabelIcon::CityDot => 14.0,
            EarthLabelIcon::HighwayShield => 18.0,
            EarthLabelIcon::None => 8.0,
        };
        let box_w = (item.text.chars().count() as f32 * char_w + extra).max(36.0);
        let box_h = item.font_px + 12.0;
        let x0 = item.screen_x - box_w * 0.5 - 6.0;
        let x1 = item.screen_x + box_w * 0.5 + 6.0;
        let y0 = item.screen_y - box_h * 0.5 - 4.0;
        let y1 = item.screen_y + box_h * 0.5 + 4.0;

        let overlaps = boxes
            .iter()
            .any(|b| x0 < b[2] && x1 > b[0] && y0 < b[3] && y1 > b[1]);
        if !overlaps {
            boxes.push([x0, y0, x1, y1]);
            accepted.push(item);
        }
    }
    accepted
}
