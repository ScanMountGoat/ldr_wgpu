
# Architecture
Major optimizations and design decisions are outlined in the sections below. This highlights major techniques without focusing on specific implementation details. See the comments in the source code for implementation decisions.

## Loading
Files are loaded using the code provided by the [ldr_tools](https://github.com/ScanMountGoat/ldr_tools_blender) library used by the [ldr_tools_blender](https://github.com/ScanMountGoat/ldr_tools_blender) addon. This handles parsing LDraw files as well as most of the geometry processing. Parsing of the LDraw files themselves is handled by a parsing implementation in ldr_tools based on work originally done for [weldr](https://github.com/djeedai/weldr).

Processing time for LDraw files is often dominated by the vertex count. Avoiding unecessary work is key for good performance. The main model geometry is loaded using standard resolution primitives and studs without logos. This may change in the future. 

Duplicate vertices are welded to reduce processing time later. ldr_tools accomplishes this using an [R-tree](https://en.wikipedia.org/wiki/R-tree) and a distance threshold. This outperforms a naive nested for loop for removing duplicates and avoids issues with hashing floating point numbers.

The geometry for each part is combined to reduce per object overhead. Fewer, larger draw calls tend to perform better than many small draw calls. This also makes it easy to cache geometry by the part name and color. Once the parts for the file are collected, each part can be converted to geometry in parallel to boost performance.

### Caching and Instancing
Scene Hierarchy:
- Part
  - Color
    - Instance transforms and bounds

Processing:
1. Recursively load, parse, and collect data from LDraw files using a single thread
2. Process all part geometries in parallel
3. Process all color information for each part in parallel
4. Process all instances for each part and color combination in parallel

The repetitive nature of LDraw models makes caching a great way to reduce loading times. The parsing library weldr caches parsed .ldr and .dat files by filename to avoid processing files from disk more than once. The scene representation returned by ldr_tools is carefully chosen to minimize the amount of processing. The initial step parses the files and collects all the part names, colors, and transforms. This initial step is hard to parallelize since the data is collected into combined lists for the entire scene. Each of the remaining processing stages has no dependencies between elements and can be done in parallel to reduce loading times. 

A scene consists of a list of unique part data containing the part name and geometry. This is used for processing bounding information and normals. Subparts and primitives are not cached. Caching subpart geometry would not actually reduce the amount of processing because merging subparts requires applying the subpart transform to each of its vertices. 

The actual parts placed in the scene are represented using a list of entries where each entry contains the geometry name, the color, and a list of instance transforms. This ensures the color processing is done only once for each unique part and color. Processing for each instance is very cheap and only needs to transform the part's bounding info by the instance transform and calculate offsets into shared geometry buffers. In general, loading times scale more with the number of unique parts and colors in the scene rather than the number of part instances since instances require less processing.

## Rendering

### Raytracing
Hardware accelerated raytracing is only supported on newer GPUs but can dramatically outperform traditional rasterization on complex scenes. Raytracing performance scales primarily with screen resolution since rays are traced from the perspective of the camera through each pixel. Scenes with hundreds of thousands of objects have similar render times to scenes with only a few objects since tracing into the bounding volume hierarchy (BVH) scales very well with high object counts and polygon counts. Raytracing also does not require any form of level of detail (LOD) or occlusion culling to efficiently render complex scenes. The top level acceleration structure (TLAS) supplied to the driver supports object instancing, so the memory overhead of raytracing is very low. 

Raytracing also enables more accurate rendering of certain visual effects. The renderer supports accurate [order-independent transparency](https://interplayoflight.wordpress.com/2023/07/15/raytraced-order-independent-transparency/) even for many layers of transparent pieces. Rendering transparent objects has a performance cost compared to opaque objects due to potentially tracing many rays per pixel for correct blending.