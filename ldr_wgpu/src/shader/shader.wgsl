struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var result: VertexOutput;
    let x = i32(vertex_index) / 2;
    let y = i32(vertex_index) & 1;
    let tc = vec2<f32>(
        f32(x) * 2.0,
        f32(y) * 2.0
    );
    result.position = vec4<f32>(
        tc.x * 2.0 - 1.0,
        1.0 - tc.y * 2.0,
        0.0, 1.0
    );
    result.tex_coords = tc;
    return result;
}

struct Camera {
    view: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
};

struct Vertex {
    pos: vec3<f32>,
    normal: u32,
};

struct Face {
    color_code: u32,
    texture_index: i32,
};

struct Material {
    roughness_exponent: f32,
    metalness: f32,
    specularity: f32,
    albedo: vec3<f32>
}

struct Geometry {
    vertex_start_index: u32,
    index_start_index: u32,
};

struct LDrawColor {
    rgba: vec4<f32>
}

@group(0) @binding(0)
var<uniform> camera: Camera;

@group(0) @binding(1)
var<storage, read> colors: array<LDrawColor>;

@group(0) @binding(2)
var color_sampler: sampler;

@group(1) @binding(0)
var<storage, read> vertices: array<Vertex>;

@group(1) @binding(1)
var<storage, read> indices: array<u32>;

@group(1) @binding(2)
var<storage, read> faces: array<Face>;

@group(1) @binding(3)
var<storage, read> uvs: array<vec4<f32>>;

@group(1) @binding(4)
var<storage, read> geometries: array<Geometry>;

const TEXTURE_COUNT: u32 = 4u;

@group(1) @binding(5)
var textures: binding_array<texture_2d<f32>, TEXTURE_COUNT>;

@group(1) @binding(6)
var acc_struct: acceleration_structure;

fn interpolate_bary(v0: vec3<f32>, v1: vec3<f32>, v2: vec3<f32>, bary: vec3<f32>) -> vec3<f32> {
    return v0 * bary.x + v1 * bary.y + v2 * bary.z;
}

fn calculate_color(intersection: RayIntersection) -> vec4<f32> {
    // The blas geometry index is stored in the custom data field.
    let geometry = geometries[intersection.instance_custom_data];

    let index_start = geometry.index_start_index;
    let vertex_start = geometry.vertex_start_index;

    let first_index_index = intersection.primitive_index * 3u + index_start;

    let i0 = vertex_start + indices[first_index_index + 0u];
    let i1 = vertex_start + indices[first_index_index + 1u];
    let i2 = vertex_start + indices[first_index_index + 2u];
    let v0 = vertices[i0];
    let v1 = vertices[i1];
    let v2 = vertices[i2];

    let bary = vec3<f32>(1.0 - intersection.barycentrics.x - intersection.barycentrics.y, intersection.barycentrics);

    let pos = interpolate_bary(v0.pos.xyz, v1.pos.xyz, v2.pos.xyz, bary);

    let n0 = unpack4x8unorm(v0.normal).xyz * 2.0 - 1.0;
    let n1 = unpack4x8unorm(v1.normal).xyz * 2.0 - 1.0;
    let n2 = unpack4x8unorm(v2.normal).xyz * 2.0 - 1.0;

    let normal_raw = interpolate_bary(n0, n1, n2, bary);

    var world_normal = intersection.object_to_world * vec4(normal_raw, 0.0);
    let view_normal = camera.view * vec4(world_normal.xyz, 0.0);
    let normal = normalize(view_normal.xyz);

    // Normals are now in view space, so the view vector is simple.
    let view = vec3(0.0, 0.0, 1.0);

    let n_dot_v = clamp(dot(normal, view), 0.0, 1.0);

    // Colors are defined per face to avoid interpolation.
    let face_index = intersection.primitive_index + index_start / 3;
    let color_code = faces[face_index].color_code;
    let ldraw_color = colors[color_code].rgba;

    var color_rgb = ldraw_color.rgb;

    let texture_index = faces[face_index].texture_index;
    if texture_index >= 0 && texture_index < i32(TEXTURE_COUNT) {
        // UVs are defined per vertex in each face.
        var uv = vec2(0.0);
        if first_index_index + 2 < arrayLength(&uvs) {
            let uv0 = uvs[first_index_index + 0].xyz;
            let uv1 = uvs[first_index_index + 1].xyz;
            let uv2 = uvs[first_index_index + 2].xyz;
            uv = interpolate_bary(uv0, uv1, uv2, bary).xy;
        }

        let texture_color = textureSample(textures[texture_index], color_sampler, uv);
        color_rgb = mix(color_rgb, texture_color.rgb, texture_color.a);
    }

    // TODO: Color tints at certain angles?
    return vec4(color_rgb * n_dot_v, ldraw_color.a);
}

@fragment
fn fs_main(vertex: VertexOutput) -> @location(0) vec4<f32> {
    var color = vec3(0.0);
    var transmission = 1.0;

    let d = vertex.tex_coords * 2.0 - 1.0;

    var origin = (camera.view_inv * vec4(0.0, 0.0, 0.0, 1.0)).xyz;
    let temp = camera.proj_inv * vec4(d.x, d.y, 1.0, 1.0);
    let direction = (camera.view_inv * vec4(normalize(temp.xyz), 0.0)).xyz;

    var rq: ray_query;
    
    // TODO: trace_ray function?
    // TODO: max draw distance?
    let flags = 0x10u; // cull back facing
    rayQueryInitialize(&rq, acc_struct, RayDesc(flags, 0xFFu, 0.1, 100000.0, origin, direction));
    rayQueryProceed(&rq);
    var intersection = rayQueryGetCommittedIntersection(&rq);

    while intersection.kind != RAY_QUERY_INTERSECTION_NONE {
        // Blend with the "under" operator since we trace front to back.
        // https://interplayoflight.wordpress.com/2023/07/15/raytraced-order-independent-transparency/
        let new_color = calculate_color(intersection);
        color += transmission * new_color.a * new_color.rgb;
        transmission *= (1.0 - new_color.a);

        if transmission > 0.01 {
            origin += intersection.t * direction;
            rayQueryInitialize(&rq, acc_struct, RayDesc(flags, 0xFFu, 0.1, 100000.0, origin, direction));
            rayQueryProceed(&rq);
            intersection = rayQueryGetCommittedIntersection(&rq);
        } else {
            break;
        }
    }

    return vec4(color, 1.0);
}
