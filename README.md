# Earth

High-precision f64 3D geospatial geodesy, quadtree tile LOD streaming & digital twin engine for Strata.

## Overview

Earth is a sovereign 3D planetary rendering and geospatial geodesy engine designed to render continental, regional, and sub-meter scale digital twins with double-precision (f64) mathematical fidelity.

Traditional 32-bit floating point graphics pipelines suffer catastrophic jitter and precision loss when camera systems transition between orbital views and street-level terrain. Earth solves this with relative-to-eye (RTE) coordinate transformations, WGS84 ellipsoidal geodesy, and Morton-indexed quadtree pyramid LOD management.

## Architectural Capabilities

- **High-Precision f64 Geodesy:** Accurate WGS84 planetary ellipsoid mathematics, geodetic latitude/longitude/altitude to Earth-Centered Earth-Fixed (ECEF) and East-North-Up (ENU) coordinate frame transformations.
- **Quadtree Pyramid LOD:** Dynamic tile hierarchy, quadtree neighbor resolution, horizon culling, and camera frustum intersection.
- **Adaptive Terrain Meshing:** Skirt generation to eliminate crack artifacts along tile seams of differing LOD levels.
- **Golden Camera Regimes:** Deterministic validation across orbital, regional, high-altitude, and ground-level perspective regimes.
- **Structural Frame Descriptors:** Scene-graph inspection, tile coverage analysis, and automated frame regression testing.

## Repository Workspace

- `crates/strata_earth`: Core geodesy engine, terrain generation, slippy map tile pipeline, and camera models.
- `examples/strata_earth_golden`: Deterministic image and frame capture harness for all Earth camera regimes.

## Verification & Testing

```bash
cargo check --workspace --all-targets
cargo test --workspace
```

## License

Licensed under the MIT License. See [LICENSE](file:///home/usr/Projects/strataui/earth/LICENSE) for details.
