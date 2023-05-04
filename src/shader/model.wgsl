struct Camera {
    view_projection: mat4x4<f32>,
    position: vec4<f32>
}

@group(0) @binding(0)
var<uniform> camera: Camera;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: u32,
    @location(2) normal: vec4<f32>
}

struct InstanceInput {
    @location(3) model_matrix_0: vec4<f32>,
    @location(4) model_matrix_1: vec4<f32>,
    @location(5) model_matrix_2: vec4<f32>,
    @location(6) model_matrix_3: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec4<f32>
}

fn unpack_color(color: u32) -> vec4<f32> {
    // wgpu doesn't support unpack4x8unorm for DX12.
    let r = f32(color & 0xFFu);
    let g = f32((color >> 8u) & 0xFFu);
    let b = f32((color >> 16u) & 0xFFu);
    let a = f32((color >> 24u) & 0xFFu);
    return vec4(r, g, b, a) / 255.0;
}

@vertex
fn vs_main(
    model: VertexInput,
    instance: InstanceInput,
) -> VertexOutput {
    let model_matrix = mat4x4<f32>(
        instance.model_matrix_0,
        instance.model_matrix_1,
        instance.model_matrix_2,
        instance.model_matrix_3,
    );
    var out: VertexOutput;
    let position = model.position.xyz;
    out.clip_position = camera.view_projection * model_matrix * vec4<f32>(model.position.xyz, 1.0);
    out.color = unpack_color(model.color);
    // TODO: is this always correct?
    out.normal = (model_matrix * vec4(model.normal.xyz, 0.0)).xyz;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // TODO: avoid normalization?
    // Calculate the lighting relative to the camera.
    let viewVector = normalize(camera.position.xyz - in.position.xyz);
    let lighting = dot(in.normal.xyz, viewVector) * 0.75 + 0.25;
    var color = in.color.rgb * lighting;
    // Premultiplied alpha.
    return vec4(color * in.color.a, in.color.a);
}

// TODO: Is it better to use colors from a separate vertex buffer?
@fragment
fn fs_edge_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4(0.0, 0.0, 0.0, 1.0);
}