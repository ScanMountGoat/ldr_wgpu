@group(0) @binding(0)
var input: texture_storage_2d<r32float, read>;

@group(0) @binding(1)
var output: texture_storage_2d<r32float, write>;

@compute
@workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
   // TODO: downsample the image.
    let coords = vec2(0i, 0i);
    let value = textureLoad(input, coords);
    // textureStore(output, coords, value);
}