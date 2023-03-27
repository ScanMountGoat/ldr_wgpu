# ldr_wgpu
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
Modern GPUs can render very efficiently. Careful organization of work into draw calls is very important to fully utilize the hardware. Issuing draw calls for each subpart or LDraw primitive creates significant overhead and leads to poor GPU utilization and performance. Merging the scene into a single draw call can be more efficient but makes it difficult to instance parts and modify parts later. 

ldr_wgpu combines geometry for each each part into a single draw. Draws are batched together and issued as a single call using [indirect rendering](https://www.khronos.org/opengl/wiki/Vertex_Rendering#Indirect_rendering). Indirect rendering places draw call parameters in a buffer. This allows culling of individual parts to be calculated entirely on the GPU using compute shaders. All draws are indexed since this greatly reduces the amount of unique vertices that need to be processed by the GPU.

### Object Culling
LDraw files are designed to model physical objects similar to CAD models. This results in lots of internal geometric detail that may never be visibile while rendering. The ideal renderer would only render exactly the polygons that are visible. Perfect culling can be expensive, so a suitable culling algorithm should cull most of the hidden geometry without incurring too much additional processing time. Culling should also be *conservative*, meaning that no geometry that should be visible is accidentally culled.

#### Frustum Culling
Frustum culling is conceptually simple and easy to implement. Any object that lies completely outside the viewing frustum of the camera won't be visible on screen and can be culled. Note that frustum culling only improves performance when zoomed in enough for part of the model to not be contained on screen. Object bounding spheres simplify the implementation and make the check very efficient. Frustum culling is applied to all geometry, including the lower detail occluder pass defined later. 

#### Occlusion Culling
The biggest performance improvement comes from occlusion culling. This culls objects that are hidden behind other objects. Numerous techniques exist in the literature and in game engines with tradeoffs between accuracy, compatibility, and performance. 

ldr_wgpu uses a form of [hierarchical-z map occlusion culling](https://www.rastergrid.com/blog/2010/10/hierarchical-z-map-based-occlusion-culling/). This runs every frame and doesn't require any kind of expensive preprocessing of the model. The basic technique starts by rendering an occluder pass. This produces a mipmapped depth map. This depth map is then used to efficiently determine which objects in the high detailed main pass are occluded by other objects. Each object can be efficiently and conservatively occlusion culled by computing just the position of the corners of its axis-aligned bounding box and checking a 2x2 pixel region from a mip of the occluder depth map. 

Accurate occlusion culling in this way requires accurate depth information. Using a separate depth only occluder pass avoids issues with reusing the previous frame's depth like depth reprojection. Rendering the same scene twice defeats the point of culling. The cost of the occluder pass plus the occluded main pass should be less than the non occluded main pass. This is accomplished by using a very low detailed model for the occluder pass. Parts are rendered with the lowest possible fidelity with any color information removed. While it may be tempting to use bounding boxes, modifying the object silhouttes will result in inaccurate occlusion tests. The final render time ends up being significantly faster in many cases than just drawing the main pass without any occlusion.

## Compatibility
The code is built using WGPU and targets modern GPU hardware for newer versions of Windows, Linux, and MacOS. The renderer takes advantage of modern features not available on older devices and requires DX12, Vulkan, or Metal support. This includes most GPUs and devices manufactured after 2010.

## Building
With a newer version of the Rust toolchain installed, run cargo build --release. Don't forget the --release since debug builds in Rust will run slowly. Run the program as `cargo run --release <ldraw library path> <ldraw file path>`.

## Copyrights
LDraw™ is a trademark owned and licensed by the Jessiman Estate, which does not sponsor, endorse, or authorize this project.

LEGO® is a registered trademark of the LEGO Group, which does not sponsor, endorse, or authorize this project.