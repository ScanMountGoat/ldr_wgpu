struct DrawIndirect {
    vertex_count: u32,
    instance_count: u32,
    base_vertex: u32,
    base_instance: u32,
}

@group(0) @binding(0)
var<storage, read_write> draws: array<DrawIndirect>;

@group(0) @binding(1)
var<storage, read> bounding_spheres: array<vec4<f32>>;

// frustum based culling adapted from here.
// https://vkguide.dev/docs/gpudriven/compute_culling/
struct Camera {
    z_near: f32,
    z_far: f32,
    _pad1: f32,
    _pad2: f32,
    frustum: vec4<f32>,
    model_view_matrix: mat4x4<f32>,
}

@group(0) @binding(2)
var<uniform> camera: Camera;

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

@compute
@workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Start with every object visible.
    let i = global_id.x;
    draws[i].instance_count = 1u;

    if (!is_within_view_frustum(i)) {
        draws[i].instance_count = 0u;
    }
}