// frustum based culling adapted from here.
// https://vkguide.dev/docs/gpudriven/compute_culling/
struct Camera {
    z_near: f32,
    z_far: f32,
    p00: f32,
    p11: f32,
    frustum: vec4<f32>,
    view: mat4x4<f32>,
    view_projection: mat4x4<f32>,
    pyramid_dimensions: vec4<f32>,
}

@group(0) @binding(0)
var<uniform> camera: Camera;

// Mipmapped version of the depth map.
@group(0) @binding(1)
var depth_pyramid: texture_2d<f32>;

@group(0) @binding(2)
var depth_pyramid_sampler: sampler;

struct DrawIndirect {
    vertex_count: u32,
    instance_count: u32,
    base_vertex: u32,
    base_instance: u32,
}

@group(1) @binding(0)
var<storage, read_write> draws: array<DrawIndirect>;

struct InstanceBounds {
    sphere: vec4<f32>,
    min_xyz: vec4<f32>,
    max_xyz: vec4<f32>,
}

@group(1) @binding(1)
var<storage, read> instance_bounds: array<InstanceBounds>;

fn is_within_view_frustum(center: vec3<f32>, radius: f32) -> bool {
	// Cull objects completely outside the viewing frustum.
	if (center.z * camera.frustum.y - abs(center.x) * camera.frustum.x < -radius) {
        return false;
    }
    if (center.z * camera.frustum.w - abs(center.y) * camera.frustum.z < -radius) {
        return false;
    }

	// Cull objects completely outside the near and far planes.
    // TODO: Why do we need to negate center.z?
    if (-center.z + radius < camera.z_near || -center.z - radius > camera.z_far) {
        return false;
    }

	return true;
}

// Adapted from the code from niagara:
// https://github.com/zeux/niagara/blob/master/src/shaders/math.h
// 2D Polyhedral Bounds of a Clipped, Perspective-Projected 3D Sphere. Michael Mara, Morgan McGuire. 2013
// Original paper link: https://jcgt.org/published/0002/02/05/
fn project_sphere(c: vec3<f32>, r: f32, z_near: f32, p00: f32, p11: f32) -> vec4<f32> {
	let cr = c * r;
	let czr2 = c.z * c.z - r * r;

	let vx = sqrt(c.x * c.x + czr2);
	let minx = (vx * c.x - cr.z) / (vx * c.z + cr.x);
	let maxx = (vx * c.x + cr.z) / (vx * c.z - cr.x);

	let vy = sqrt(c.y * c.y + czr2);
	let miny = (vy * c.y - cr.z) / (vy * c.z + cr.y);
	let maxy = (vy * c.y + cr.z) / (vy * c.z - cr.y);

	var aabb = vec4(minx * p00, miny * p11, maxx * p00, maxy * p11);
	aabb = aabb.xwzy * vec4(0.5f, -0.5f, 0.5f, -0.5f) + vec4(0.5f); // clip space -> uv space

	return aabb;
}

fn world_to_ndc(position: vec3<f32>) -> vec3<f32> {
    var clip_pos = camera.view_projection * vec4(position, 1.0);
    clip_pos.z = max(clip_pos.z, 0.0);

    let ndc_pos = clip_pos.xyz / clip_pos.w;
    var ndc_pos_xy = clamp(ndc_pos.xy, vec2(-1.0), vec2(1.0));
    ndc_pos_xy = ndc_pos_xy * vec2(0.5, -0.5) + vec2(0.5, 0.5);

    return vec3(ndc_pos_xy, ndc_pos.z);
}

// https://interplayoflight.wordpress.com/2017/11/15/experiments-in-gpu-based-occlusion-culling/
fn is_occluded(min_xyz: vec3<f32>, max_xyz: vec3<f32>) -> bool {
    // Occlusion based culling using axis aligned bounding boxes.
    // Transform the corners to the same space as the depth map.
    let aabb_corners = array<vec3<f32>, 8>(
        world_to_ndc(min_xyz),
        world_to_ndc(vec3(max_xyz.x, min_xyz.yz)),
        world_to_ndc(vec3(min_xyz.x, max_xyz.y, min_xyz.z)),
        world_to_ndc(vec3(min_xyz.xy, max_xyz.z)),
        world_to_ndc(vec3(max_xyz.xy, min_xyz.z)),
        world_to_ndc(vec3(min_xyz.x, max_xyz.yz)),
        world_to_ndc(vec3(max_xyz.x, min_xyz.y, max_xyz.z)),
        world_to_ndc(max_xyz),
    );

    var min_xyz = min(aabb_corners[0], aabb_corners[1]);
    min_xyz = min(min_xyz, aabb_corners[2]);
    min_xyz = min(min_xyz, aabb_corners[3]);
    min_xyz = min(min_xyz, aabb_corners[4]);
    min_xyz = min(min_xyz, aabb_corners[5]);
    min_xyz = min(min_xyz, aabb_corners[6]);
    min_xyz = min(min_xyz, aabb_corners[7]);

    var max_xyz = max(aabb_corners[0], aabb_corners[1]);
    max_xyz = max(max_xyz, aabb_corners[2]);
    max_xyz = max(max_xyz, aabb_corners[3]);
    max_xyz = max(max_xyz, aabb_corners[4]);
    max_xyz = max(max_xyz, aabb_corners[5]);
    max_xyz = max(max_xyz, aabb_corners[6]);
    max_xyz = max(max_xyz, aabb_corners[7]);

    let aabb = vec4(min_xyz.xy, max_xyz.xy);

    // Calculate the covered area in pixels for the base mip level.
    let dimensions = (max_xyz.xy - min_xyz.xy) * camera.pyramid_dimensions.xy;

    // Calculate the mip level that will be covered by 2x2 pixels.
    // 4x4 pixels on the base level should use mip level 1.
    let level = ceil(log2(max(dimensions.x, dimensions.y) / 2.0));

    // Compute the max depth of the 2x2 texels for the AABB.
    // The depth pyramid also uses max for reduction.
    // This helps make the occlusion conservative.
    let depth00 = textureSampleLevel(depth_pyramid, depth_pyramid_sampler, aabb.xy, level).x;
    let depth01 = textureSampleLevel(depth_pyramid, depth_pyramid_sampler, aabb.zy, level).x;
    let depth10 = textureSampleLevel(depth_pyramid, depth_pyramid_sampler, aabb.xw, level).x;
    let depth11 = textureSampleLevel(depth_pyramid, depth_pyramid_sampler, aabb.zw, level).x;
    let max_occluder_depth = max(max(depth00, depth01), max(depth10, depth11));

    // Check if the minimum depth of the object exceeds the max occluder depth.
    // This means the object is definitely occluded. 
    return min_xyz.z > max_occluder_depth;
}

fn is_visible(index: u32) -> bool {
    // Bounding spheres for frustum culling.
	let bounding_sphere = instance_bounds[index].sphere;
    let center_view = (camera.view * vec4(bounding_sphere.xyz, 1.0)).xyz;
    let radius = bounding_sphere.w;

    if (!is_within_view_frustum(center_view, radius)) {
        return false;
    }

    // Axis-aligned bounding box for occlusion culling.
    let min_xyz = instance_bounds[index].min_xyz.xyz;
    let max_xyz = instance_bounds[index].max_xyz.xyz;

    if (is_occluded(min_xyz, max_xyz)) {
        return false;
    }

    return true;
}

@compute
@workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Start with every object visible.
    let i = global_id.x;
    draws[i].instance_count = 1u;

    if (!is_visible(i)) {
        draws[i].instance_count = 0u;
    }
}