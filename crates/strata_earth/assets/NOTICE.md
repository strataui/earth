# Public Domain & CC0 Asset Provenance (`strata_earth/assets/`)

All bundled assets in `crates/strata_earth/assets/` except `base_world_map.pack` are Public Domain (US Government / CC0 1.0) with no copyright restrictions:

- `base_world_map.pack`: `SEBM0001` (`EPSG:3857`, `z=0..=4`, `4096x4096` world resolution) baked via `bake_assets` from ArcGIS `World_Imagery` (Esri / Maxar / Earthstar Geographics; matches live `z >= 5` imagery without `z=4 -> z=5` pop).
- `land_mask.png`: Rasterized from Natural Earth `ne_110m_land.geojson`, Public Domain (1.0).
- `night_lights.jpg`: Derived from NASA GIBS `VIIRS_CityLights_2012` Earth at Night (`Black Marble`), NASA Public Domain (17 U.S.C. § 105).
- `moon_albedo.jpg`: Derived from NASA Scientific Visualization Studio CGI Moon Kit (`lroc_color_poles_1k.jpg`, SVS #4720), NASA Public Domain.
- `bsc5_stars.bin`: Compiled from Yale Bright Star Catalog 5th Edition (`brettonw/YaleBrightStarCatalog`, 8,404 naked-eye stars `V <= 6.5` in J2000 ICRF), Public Domain / MIT data.
- `clouds_fallback.jpg`: Derived from EUMETSAT/NOAA global cloud composite (`matteason/live-cloud-maps`), MIT License.
- `weather_cloud.glb`: Cloud mesh extracted from "Storm" by Armory_3D via Poly Pizza, **CC0 1.0 Universal (Public Domain)**.
- `weather_rain.glb`: Raindrop mesh from "Drops" by Marcel Plagmann via OpenGameArt, **CC0 1.0 Universal (Public Domain)**.
- `weather_snow.glb`: `snowflake-a.glb` from Kenney Holiday Kit (`shorepine/kenney`), **CC0 1.0 Universal (Public Domain)**.
- Orbital spacecraft GP elements (`satellites.rs` / CelesTrak GP JSON): US Government / USSPACECOM public-domain orbital elements.
- Space skybox (`skybox.rs`, fetched on demand via CDN/HiPS):
  - **ESA Gaia DR3 Colour Flux Map** (`ESAVO/P/GAIA/DR3-RGB`): Copyright ESA/Gaia/DPAC (DOI `10.5270/esa-qa4lep3`), scientific attribution.
  - **NASA/IPAC AllWISE Color** (`CDS/P/allWISE/color`): IPAC/NASA JPL/Caltech & UCLA acknowledgment.
  - **DSS2 Color** (`CDS/P/DSS2/color`, optional direct IRSA stream): Digitized Sky Survey — STScI/NASA Grant `NAG W-2166`, colored by CDS Strasbourg (`archive.stsci.edu/publishing/data-use`).
