struct Camera {
    mvp_matrix: mat4x4<f32>,
}

@group(0) @binding(0)
var<uniform> camera: Camera;

struct VertexInput {
    @location(0) position: vec4<f32>,
}

struct WorldTransform {
    transform: mat4x4<f32>,
}

@group(1) @binding(0)
var<uniform> world_transform: WorldTransform;

// struct InstanceInput {
//     @location(1) model_matrix_0: vec4<f32>,
//     @location(2) model_matrix_1: vec4<f32>,
//     @location(3) model_matrix_2: vec4<f32>,
//     @location(4) model_matrix_3: vec4<f32>,
// }

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
}

@vertex
fn vs_main(
    model: VertexInput,
    // instance: InstanceInput,
) -> VertexOutput {
    let model_matrix = world_transform.transform;
    // let model_matrix = mat4x4<f32>(
    //     instance.model_matrix_0,
    //     instance.model_matrix_1,
    //     instance.model_matrix_2,
    //     instance.model_matrix_3,
    // );
    var out: VertexOutput;
    out.clip_position = camera.mvp_matrix * model_matrix * vec4<f32>(model.position.xyz, 1.0);
    // out.clip_position = camera.mvp_matrix * vec4<f32>(model.position.xyz, 1.0);

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4(1.0);
}