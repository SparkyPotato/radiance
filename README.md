# radiance

A very WIP game engine, currently focusing on the renderer.

## Features

### Virtual Geometry
A virtual geometry system similar to Unreal Engine's _Nanite_, able to render scenes with trillions of triangles in 1-2 ms. Does not implement mesh streaming yet.

### Ground-truth Path Tracing
HW-accelerated path tracer for ground-truth, to be used to generate references for the real-time path.
Currently very simple with only MIS and NEE.

### Hillaire Sky
An approximation of [_A Scalable and Production Ready Sky and Atmosphere Rendering Technique_](https://sebh.github.io/publications/egsr2020.pdf) by Sébastien Hillaire.

### Auto-exposure and tonemapping
Histogram-based auto-exposure with several tonemapping operators:
- ACES
- AgX (default, 'filmic', and punchy 'looks')
- [Tony McMapface](https://github.com/h3r2tic/tony-mc-mapface)

## Requirements
Shrimply `git clone https://github.com/SparkyPotato/radiance/ --recursive` then `cargo run`.

