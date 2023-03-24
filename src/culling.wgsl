// frustum based culling adapted from here.
// https://vkguide.dev/docs/gpudriven/compute_culling/
struct Camera {
    z_near: f32,
    z_far: f32,
    p00: f32,
    p11: f32,
    frustum: vec4<f32>,
    model_view_matrix: mat4x4<f32>,
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

@group(1) @binding(1)
var<storage, read> bounding_spheres: array<vec4<f32>>;

fn is_within_view_frustum(index: u32) -> bool {
	//grab sphere cull data from the object buffer
	let bounding_sphere = bounding_spheres[index];

	let center = (camera.model_view_matrix * vec4(bounding_sphere.xyz, 1.0)).xyz;
	let radius = bounding_sphere.w;

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

// https://vkguide.dev/docs/gpudriven/compute_culling/
fn is_occluded(index: u32) -> bool {
    // Occlusion based culling using axis aligned bounding boxes.
    let bounding_sphere = bounding_spheres[index];
    let center = bounding_sphere.xyz;
    let radius = bounding_sphere.w;

    // Project the cull sphere into screenspace coordinates.
    let aabb = project_sphere(center, radius, camera.z_near, camera.p00, camera.p11);

    let width = (aabb.z - aabb.x) * camera.pyramid_dimensions.x;
    let height = (aabb.w - aabb.y) * camera.pyramid_dimensions.y;

    let level = log2(max(width, height));

    // Compute the max depth of the 2x2 texels for the AABB.
    // The depth pyramid also uses max for reduction.
    // This helps make the occlusion conservative.
    // https://interplayoflight.wordpress.com/2017/11/15/experiments-in-gpu-based-occlusion-culling/
    let depth00 = textureSampleLevel(depth_pyramid, depth_pyramid_sampler, aabb.xy, level).x;
    let depth01 = textureSampleLevel(depth_pyramid, depth_pyramid_sampler, aabb.zy, level).x;
    let depth10 = textureSampleLevel(depth_pyramid, depth_pyramid_sampler, aabb.xw, level).x;
    let depth11 = textureSampleLevel(depth_pyramid, depth_pyramid_sampler, aabb.zw, level).x;
    let depth = max(max(depth00, depth01), max(depth10, depth11));

    // Check if the minimum depth of the object exceeds the max occluder depth.
    // This means the object is definitely occluded. 
    let min_depth = camera.z_near / (center.z - radius);
    return min_depth >= depth;
}

fn is_visible(index: u32) -> bool {
    if (!is_within_view_frustum(index)) {
        return false;
    }

    if (is_occluded(index)) {
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