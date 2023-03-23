@group(0) @binding(0)
var input: texture_2d<f32>;

@group(0) @binding(1)
var output: texture_storage_2d<r32float, write>;

@compute
@workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // The input should have twice the resolution of the output.
    // TODO: Gather a 2x2 pixel region and take the min instead.
    let coords = vec2(i32(global_id.x), i32(global_id.y));
    let value = textureLoad(input, coords * 2, 0i);
    textureStore(output, coords, value);
}