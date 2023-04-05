# ldr_wgpu
![image](https://user-images.githubusercontent.com/23301691/229853622-1ec21101-1b3d-4ee3-bc37-593f025fe935.png)

ldr_wgpu is an experimental LDraw renderer targeting modern desktop GPU hardware.

The goal of this project is to research and implement suitable techniques for rendering large LDraw scenes in real time. This project is not designed to match the quality of offline renderers like Blender Cycles. The code is intended to serve as a reference for others wanting to build their own renderers and contains numerous comments and links to blog posts, technical docs, papers, etc. This project may be reworked to be useable as a library for future Rust projects at some point.

## Design
Major optimizations and design decisions are outlined in the sections below. This highlights major techniques without focusing on specific implementation details. See the comments in the source code for implementation decisions. If you have any comments or suggestions, feel free to open an [issue](https://github.com/ScanMountGoat/ldr_wgpu/issues) to discuss it.

### Loading
Files are loaded using the code provided by the [ldr_tools](https://github.com/ScanMountGoat/ldr_tools_blender) library used by the ldr_tools_blender addon. This handles parsing LDraw files as well as most of the geometry processing. 

Processing time for LDraw files is often dominated by the vertex count. Avoiding unecessary work is key for good performance. The main model geometry is loaded using standard resolution primitives and studs without logos. This may change in the future. 

Duplicate vertices are welded to reduce processing time later. ldr_tools accomplishes this using an [R-tree](https://en.wikipedia.org/wiki/R-tree) and a distance threshold. This outperforms a naive nested for loop for removing duplicates and avoids issues with hashing floating point numbers.

The geometry for each part is combined to reduce per object overhead. Fewer, larger draw calls tend to perform better than many small draw calls. This also makes it easy to cache geometry by the part name and color. Once the parts for the file are collected, each part can be converted to geometry in parallel to boost performance.

### Draw Calls
Modern GPUs are highly effective at processing large amounts of data in parallel. Careful organization of work into draw calls is very important to fully utilize the hardware. Issuing draw calls for each subpart or LDraw primitive creates significant overhead and leads to poor GPU utilization and performance. Merging the scene into a single draw call can be more efficient but makes it difficult to instance parts and modify parts later. 

ldr_wgpu combines geometry for each each part into a single draw. Draws are batched together and issued as a single call using [indirect rendering](https://www.khronos.org/opengl/wiki/Vertex_Rendering#Indirect_rendering). Indirect rendering places draw call parameters in a buffer. This allows culling of individual parts to be calculated entirely on the GPU using compute shaders. All draws are indexed since this greatly reduces the amount of unique vertices that need to be processed by the GPU.

The indirect buffers are often very sparse due to the high number of occluded objects each frame. These empty draw calls still have a performance cost and can lead to significant overhead on scenes with many objects. Removing these empty draw calls each frame using a process known as stream compaction gives a noticeable performance improvement. Stream compaction is handled using a parallel scan algorithm that counts the number of visible draw calls to determine the order of the final compacted stream. On supported hardware, this process happens entirely on the GPU without needing to synchronize with or copy data to the CPU. It's possible to implement this approach on the CPU, but this negates many of the performance benefits.

### Instancing
LDraw scenes are highly repetitive since the same part can be used many times within the same model. Instancing provides a way to reduce memory usage and processing time by reducing duplicate geometry. The geometry for each part can be stored exactly once with a separate instance buffer containing the transforms for each instace of the part. Parts appearing in multiple colors can also share geometry. This currently isn't implemented since it makes it more difficult to combine all parts into a single indirect draw call.

### Object Culling
LDraw files are designed to model physical objects similar to CAD models. This results in lots of internal geometric detail that may never be visibile while rendering. The ideal renderer would only render exactly the polygons that are visible. Perfect culling can be expensive, so a suitable culling algorithm should cull most of the hidden geometry without incurring too much additional processing time. Culling should also be *conservative*, meaning that no geometry that should be visible is accidentally culled.

#### Frustum Culling
Frustum culling is conceptually simple and easy to implement. Any object that lies completely outside the viewing frustum of the camera won't be visible on screen and can be culled. Note that frustum culling only improves performance when zoomed in enough for part of the model to not be contained on screen. Object bounding spheres simplify the implementation and make the check very efficient.

#### Occlusion Culling
The biggest performance improvement comes from occlusion culling. This culls objects that are hidden behind other objects. Numerous techniques exist in the literature and in game engines with tradeoffs between accuracy, compatibility, and performance. 

ldr_wgpu uses a form of hierarchical-z map occlusion culling presented in the paper "Patch-Based Occlusion Culling for Hardware Tessellation". This runs every frame and doesn't require any kind of expensive preprocessing of the model. The basic technique is to render some of the objects in the scene and produce a mipmapped depth map. This depth map is then used to efficiently determine which objects are visible. Each object can be efficiently and conservatively occlusion culled by computing just the position of the corners of its axis-aligned bounding box and checking a 2x2 pixel region from a mip of the occluder depth map. 

Accurate occlusion culling in this way requires accurate depth information. Rendering the same scene twice defeats the point of culling. This is accomplished by using the previous frame's visibility as an estimate for the current frame's visibility. Previously visible objects are used for the occluder pass to determine what objects are newly visible in this frame and update visibility estimates for the next frame. This avoids the need for separate geometry for occluders or inaccurate depth reprojection from the previous frame. See the source code for details.

### Depth Buffer Precision
The standard configuration for depth testing uses a floating point depth format and a depth test using less than or less than or equal. This results in most of the precision being concentrated near the near plane. ldr_wgpu uses the common [reversed-z](https://developer.nvidia.com/content/depth-precision-visualized) trick to more evenly distribute the depth precision. The increased precision far away is critical for occlusion culling to work properly on large models. This also allows the far plane to be positioned at infinity, resulting in infinite draw distance with minimal precision issues.

## Compatibility
The code is built using WGPU and targets modern GPU hardware for newer versions of Windows, Linux, and MacOS. The renderer takes advantage of modern features not available on older devices and requires DX12, Vulkan, or Metal support. This includes most GPUs and devices manufactured after around the year 2010.

## Building
With a newer version of the [Rust toolchain](https://www.rust-lang.org/tools/install) installed, run `cargo build --release` from the main repository directory. Don't forget the --release since debug builds in Rust will run slowly. The executable will be located in `target/release`. Run the program as `cargo run --release <ldraw library path> <ldraw file path>` or `ldr_wgpu <ldraw library path> <ldraw file path>`.

## Copyrights
LDraw™ is a trademark owned and licensed by the Jessiman Estate, which does not sponsor, endorse, or authorize this project.

LEGO® is a registered trademark of the LEGO Group, which does not sponsor, endorse, or authorize this project.
