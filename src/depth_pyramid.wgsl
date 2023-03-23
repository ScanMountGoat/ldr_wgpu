@group(0) @binding(0)
var input: texture_storage_2d<r32float, read>;

@group(0) @binding(1)
var output: texture_storage_2d<r32float, write>;

@compute
@workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // The input should have twice the resolution of the output.
    let x = i32(global_id.x) * 2;
    let y = i32(global_id.y) * 2;
    // Gather a 2x2 pixel region and take the min depth.
    let value00 = textureLoad(input, vec2(x, y));
    let value01 = textureLoad(input, vec2(x+1, y));
    let value10 = textureLoad(input, vec2(x, y+1));
    let value11 = textureLoad(input, vec2(x+1, y+1));
    let min_x = min(value00, value01);
    let min_y = min(value10, value11);

    let coords = vec2(i32(global_id.x), i32(global_id.y));
    let value = min(min_x, min_y);
    textureStore(output, coords, value);
}