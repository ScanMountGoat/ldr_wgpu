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
    // This avoids occluders appearing larger in smaller mips.
    // This uses the opposite comparison as the post below since we use reversed-z.
    // https://interplayoflight.wordpress.com/2017/11/15/experiments-in-gpu-based-occlusion-culling/
    let value00 = textureLoad(input, vec2(x, y));
    let value01 = textureLoad(input, vec2(x+1, y));
    let value10 = textureLoad(input, vec2(x, y+1));
    let value11 = textureLoad(input, vec2(x+1, y+1));
    var value = min(min(value00, value01), min(value10, value11));

    // Odd texture sizes require sampling an extra row and column of pixels.
    // https://www.rastergrid.com/blog/2010/10/hierarchical-z-map-based-occlusion-culling/
    let input_dimensions = textureDimensions(input);
    let odd_width = (input_dimensions.x & 1) != 0;
    let odd_height = (input_dimensions.y & 1) != 0;

    if (odd_width && x == input_dimensions.x - 3) {
        value = min(value, textureLoad(input, vec2(x+2, y)));
        value = min(value, textureLoad(input, vec2(x+2, y+1)));

        if (odd_height && y == input_dimensions.y - 3) {
            value = min(value, textureLoad(input, vec2(x+2, y+2)));
        }
    } else if (odd_height && y == input_dimensions.y - 3) {
        value = min(value, textureLoad(input, vec2(x, y+2)));
        value = min(value, textureLoad(input, vec2(x+1, y+2)));
    }

    let coords = vec2(i32(global_id.x), i32(global_id.y));
    textureStore(output, coords, value);
}