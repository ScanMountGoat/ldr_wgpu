@group(0) @binding(0)
var input: texture_depth_2d;

@group(0) @binding(1)
var output: texture_storage_2d<r32float, write>;

@compute
@workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let coords = vec2<i32>(global_id.xy);
    let value = textureLoad(input, coords, 0i);
    textureStore(output, coords, vec4(value));
}