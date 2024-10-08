# ldr_wgpu
![image](https://github.com/ScanMountGoat/ldr_wgpu/assets/23301691/d103781d-b0fa-4cf0-bc42-8b13ab42a459)

ldr_wgpu is an experimental LDraw renderer targeting modern desktop GPU hardware with an emphasis on performance.

The goal of this project is to research and implement suitable techniques for rendering large LDraw scenes in real time. This project is not designed to match the quality of offline renderers like Blender Cycles. The code is intended to serve as a reference for others wanting to build their own renderers and contains numerous comments and links to blog posts, technical docs, papers, etc. This project may be reworked to be useable as a library for future Rust projects at some point.

## Design
High level design decisions are outlined below. See [ARCHITECTURE](https://github.com/ScanMountGoat/ldr_wgpu/blob/main/ARCHITECTURE.md) for details.
If you have any comments or suggestions, feel free to open an [issue](https://github.com/ScanMountGoat/ldr_wgpu/issues) to discuss it.

- GPU driven rendering for greatly reduced driver overhead
- Compute based occlusion and frustum culling
- Instancing for reduced memory usage
- Heavy usage of multithreading and caching for fast scene load times

## Compatibility
The code is built using WGPU and targets modern GPU hardware for newer versions of Windows, Linux, and MacOS. The renderer takes advantage of modern features not available on older devices and requires DX12, Vulkan, or Metal support. This includes most GPUs and devices manufactured after around the year 2010.

## Building
With a newer version of the [Rust toolchain](https://www.rust-lang.org/tools/install) installed, run `cargo build --release` from the main repository directory. Don't forget the --release since debug builds in Rust will run slowly. The executable will be located in `target/release`. Run the program as `cargo run --release -p ldr_viewer <ldraw library path> <ldraw file path>` or from the executable directory as `ldr_viewer <ldraw library path> <ldraw file path>`.

## Copyrights
LDraw™ is a trademark owned and licensed by the Jessiman Estate, which does not sponsor, endorse, or authorize this project.  
LEGO® is a registered trademark of the LEGO Group, which does not sponsor, endorse, or authorize this project.
