// Frustum based culling adapted from here:
// https://vkguide.dev/docs/gpudriven/compute_culling/
struct Camera {
    z_near: f32,
    z_far: f32,
    p00: f32,
    p11: f32,
    frustum: vec4<f32>,
    view: mat4x4<f32>,
    view_projection: mat4x4<f32>,
}

@group(0) @binding(0)
var<uniform> camera: Camera;

// Mipmapped version of the depth map.
@group(0) @binding(1)
var depth_pyramid: texture_2d<f32>;

// Assume this is a point sampler.
// We don't want to interpolate depth values.
@group(0) @binding(2)
var depth_sampler: sampler;

struct InstanceBounds {
    sphere: vec4<f32>,
    min_xyz: vec4<f32>,
    max_xyz: vec4<f32>,
}

@group(1) @binding(0)
var<storage, read> instance_bounds: array<InstanceBounds>;

@group(1) @binding(1)
var<storage, read_write> visibility: array<u32>;

@group(1) @binding(2)
var<storage, read_write> new_visibility: array<u32>;

@group(1) @binding(3)
var<storage, read> transparent: array<u32>;

fn is_within_view_frustum(center: vec3<f32>, radius: f32) -> bool {
	// Cull objects completely outside the viewing frustum.
    if center.z * camera.frustum.y - abs(center.x) * camera.frustum.x < -radius {
        return false;
    }
    if center.z * camera.frustum.w - abs(center.y) * camera.frustum.z < -radius {
        return false;
    }

    return true;
}

fn world_to_coords(position: vec3<f32>) -> vec3<f32> {
    var clip_pos = camera.view_projection * vec4(position, 1.0);

    let ndc_pos = clamp(clip_pos.xyz / clip_pos.w, vec3(-1.0), vec3(1.0));
    // Convert to UV coordinates.
    var ndc_pos_xy = ndc_pos.xy * vec2(0.5, -0.5) + vec2(0.5, 0.5);
    return vec3(ndc_pos_xy, ndc_pos.z);
}

fn is_occluded(min_xyz: vec3<f32>, max_xyz: vec3<f32>) -> bool {
    // Occlusion based culling using axis aligned bounding boxes.
    // Transform the corners to the same space as the depth map.
    // https://interplayoflight.wordpress.com/2017/11/15/experiments-in-gpu-based-occlusion-culling/
    // TODO: Faster to use the calculation that transforms the AABB directly?
    let aabb_corners = array<vec3<f32>, 8>(
        world_to_coords(min_xyz),
        world_to_coords(vec3(max_xyz.x, min_xyz.yz)),
        world_to_coords(vec3(min_xyz.x, max_xyz.y, min_xyz.z)),
        world_to_coords(vec3(min_xyz.xy, max_xyz.z)),
        world_to_coords(vec3(max_xyz.xy, min_xyz.z)),
        world_to_coords(vec3(min_xyz.x, max_xyz.yz)),
        world_to_coords(vec3(max_xyz.x, min_xyz.y, max_xyz.z)),
        world_to_coords(max_xyz),
    );

    var min_xyz_final = min(aabb_corners[0], aabb_corners[1]);
    min_xyz_final = min(min_xyz_final, aabb_corners[2]);
    min_xyz_final = min(min_xyz_final, aabb_corners[3]);
    min_xyz_final = min(min_xyz_final, aabb_corners[4]);
    min_xyz_final = min(min_xyz_final, aabb_corners[5]);
    min_xyz_final = min(min_xyz_final, aabb_corners[6]);
    min_xyz_final = min(min_xyz_final, aabb_corners[7]);

    var max_xyz_final = max(aabb_corners[0], aabb_corners[1]);
    max_xyz_final = max(max_xyz_final, aabb_corners[2]);
    max_xyz_final = max(max_xyz_final, aabb_corners[3]);
    max_xyz_final = max(max_xyz_final, aabb_corners[4]);
    max_xyz_final = max(max_xyz_final, aabb_corners[5]);
    max_xyz_final = max(max_xyz_final, aabb_corners[6]);
    max_xyz_final = max(max_xyz_final, aabb_corners[7]);

    // An axis-aligned bounding box in screen space.
    let aabb = vec4(min_xyz_final.xy, max_xyz_final.xy);

    // Calculate the covered area in pixels for the base mip level.
    let aabb_size = max_xyz_final.xy - min_xyz_final.xy;
    let aabb_size_base_level = aabb_size * vec2<f32>(textureDimensions(depth_pyramid, 0));

    // Calculate the mip level that will be covered by at most 2x2 pixels.
    // 4x4 pixels on the base level should use mip level 1.
    // 3x3 pixels should also use level 1.
    let level = ceil(log2(max(aabb_size_base_level.x, aabb_size_base_level.y))) - 1.0;

    // Compute the min depth of the 2x2 texels for the AABB.
    // The depth pyramid also uses min for reduction.
    // This helps make the occlusion conservative.
    // The comparisons are reversed since we use a reversed-z buffer.
    // The AABB coordinates are between 0.0 and 1.0.
    let depth00 = textureSampleLevel(depth_pyramid, depth_sampler, aabb.xy, level).x;
    let depth01 = textureSampleLevel(depth_pyramid, depth_sampler, aabb.zy, level).x;
    let depth10 = textureSampleLevel(depth_pyramid, depth_sampler, aabb.xw, level).x;
    let depth11 = textureSampleLevel(depth_pyramid, depth_sampler, aabb.zw, level).x;
    let farthest_occluder_depth = min(min(depth00, depth01), min(depth10, depth11));

    // Check if the closest depth of the object exceeds the farthest occluder depth.
    // This means the object is definitely occluded.
    // The comparisons are reversed since we use a reversed-z buffer.
    let closest_depth = max_xyz_final.z;
    return closest_depth < farthest_occluder_depth;
}

fn is_visible(index: u32) -> bool {
    // Bounding spheres for frustum culling.
    let bounding_sphere = instance_bounds[index].sphere;
    let center_view = (camera.view * vec4(bounding_sphere.xyz, 1.0)).xyz;
    // Assume no scaling in the view matrix.
    let radius = bounding_sphere.w;

    if !is_within_view_frustum(center_view, radius) {
        return false;
    }

    // Axis-aligned bounding box for occlusion culling.
    let min_xyz = instance_bounds[index].min_xyz.xyz;
    let max_xyz = instance_bounds[index].max_xyz.xyz;

    // Use the existing state for visibility.
    if is_occluded(min_xyz, max_xyz) {
        return false;
    }

    return true;
}

@compute
@workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Assume all the arrays have the same length.
    let index = global_id.x;
    if index >= arrayLength(&visibility) {
        return;
    }

    // Set visibility for all objects based on culling.
    // This serves as a visibility estimate for next frame.
    let previously_visible = visibility[index] != 0u;
    let visible = is_visible(index);
    // Transparent objects should never be in the previously visible pass.
    // This prevents transparent objects occluding other objects.
    let is_transparent = transparent[index] != 0u;
    if visible && !is_transparent {
        visibility[index] = 1u;
    } else {
        visibility[index] = 0u;
    }

    // Also track objects visible this frame but not last frame
    // Also drawing these objects makes the culling conservative.
    if visible && !previously_visible {
        new_visibility[index] = 1u;
    } else {
        new_visibility[index] = 0u;
    }
}
