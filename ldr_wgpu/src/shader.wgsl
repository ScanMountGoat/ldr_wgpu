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
    pos: vec4<f32>,
    normal: vec3<f32>,
    color_code: u32
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
    _pad1: u32,
    _pad2: u32
};

struct LDrawColor {
    rgba: vec4<f32>
}

@group(0) @binding(0)
var<uniform> camera: Camera;

@group(0) @binding(1)
var<storage, read> colors: array<LDrawColor>;

@group(1) @binding(0)
var<storage, read> vertices: array<Vertex>;

@group(1) @binding(1)
var<storage, read> indices: array<u32>;

@group(1) @binding(2)
var<storage, read> geometries: array<Geometry>;

@group(1) @binding(3)
var acc_struct: acceleration_structure;

fn interpolate_bary(v0: vec3<f32>, v1: vec3<f32>, v2: vec3<f32>, bary: vec3<f32>) -> vec3<f32> {
    return v0 * bary.x + v1 * bary.y + v2 * bary.z;
}

@fragment
fn fs_main(vertex: VertexOutput) -> @location(0) vec4<f32> {
    var color = vec4(0.0);

    let d = vertex.tex_coords * 2.0 - 1.0;

    let origin = (camera.view_inv * vec4(0.0, 0.0, 0.0, 1.0)).xyz;
    let temp = camera.proj_inv * vec4(d.x, d.y, 1.0, 1.0);
    let direction = (camera.view_inv * vec4(normalize(temp.xyz), 0.0)).xyz;

    // TODO: max draw distance?
    var rq: ray_query;
    rayQueryInitialize(&rq, acc_struct, RayDesc(0u, 0xFFu, 0.1, 100000.0, origin, direction));
    rayQueryProceed(&rq);

    // TODO: Multiple samples per pixel for less aliasing.
    let intersection = rayQueryGetCommittedIntersection(&rq);
    if intersection.kind != RAY_QUERY_INTERSECTION_NONE {
        // The blas geometry index is stored in the custom data field.
        let geometry = geometries[intersection.instance_custom_data];

        let index_start = geometry.index_start_index;
        let vertex_start = geometry.vertex_start_index;

        let first_index_index = intersection.primitive_index * 3u + index_start;

        let v0 = vertices[vertex_start + indices[first_index_index + 0u] ];
        let v1 = vertices[vertex_start + indices[first_index_index + 1u] ];
        let v2 = vertices[vertex_start + indices[first_index_index + 2u] ];

        let bary = vec3<f32>(1.0 - intersection.barycentrics.x - intersection.barycentrics.y, intersection.barycentrics);

        let pos = interpolate_bary(v0.pos.xyz, v1.pos.xyz, v2.pos.xyz, bary);
        let normal_raw = interpolate_bary(v0.normal.xyz, v1.normal.xyz, v2.normal.xyz, bary);

        let world_normal = intersection.object_to_world * vec4(normal_raw, 0.0);
        let view_normal = camera.view * vec4(world_normal.xyz, 0.0);
        let normal = normalize(view_normal.xyz);

        // Normals are now in view space, so the view vector is simple.
        let view = vec3(0.0, 0.0, 1.0);

        let n_dot_v = max(dot(normal, view), 0.0);

        // TODO: How to handle attributes that don't interpolate?
        // TODO: store color codes per face instead of per vertex?
        let color0 = colors[v0.color_code].rgba;
        let color1 = colors[v1.color_code].rgba;
        let color2 = colors[v2.color_code].rgba;
        let ldraw_color = interpolate_bary(color0.rgb, color1.rgb, color2.rgb, bary);

        let color_rgb = ldraw_color.rgb * n_dot_v;
        color = vec4(color_rgb, 1.0);
    }

    return color;
}
