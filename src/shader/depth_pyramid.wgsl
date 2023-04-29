@group(0) @binding(0)
var input: texture_storage_2d<r32float, read>;

@group(0) @binding(1)
var output: texture_storage_2d<r32float, write>;

@compute
@workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let output_dimensions = textureDimensions(input);
    let output_coords = vec2<i32>(global_id.xy);
    if (output_coords.x >= output_dimensions.x || output_coords.y >= output_dimensions.y) {
        return;
    }

    // The input should have twice the resolution of the output.
    let x = i32(global_id.x) * 2;
    let y = i32(global_id.y) * 2;

    // Gather a 2x2 pixel region and take the min depth (farthest depth in reversed-z).
    // This avoids occluders appearing larger in smaller mips.
    let value00 = textureLoad(input, vec2(x, y));
    let value01 = textureLoad(input, vec2(x+1, y));
    let value10 = textureLoad(input, vec2(x, y+1));
    let value11 = textureLoad(input, vec2(x+1, y+1));
    var value = min(min(value00, value01), min(value10, value11));

    // Handle the case where the previous mipmap has odd dimensions.
    // We may need to reduce 3 pixels instead of just two pixels in each dimension.
    // https://miketuritzin.com/post/hierarchical-depth-buffers/
    let dimensions = textureDimensions(input);
    let include_extra_column = (dimensions.x & 1) != 0;
    let include_extra_row = (dimensions.y & 1) != 0;

    if (include_extra_column) {
        value = min(value, textureLoad(input, vec2(x+2, y)));
        value = min(value, textureLoad(input, vec2(x+2, y+1)));

        if (include_extra_row) {
            value = min(value, textureLoad(input, vec2(x+2, y+2)));
        }
    } 
    if (include_extra_row) {
        value = min(value, textureLoad(input, vec2(x, y+2)));
        value = min(value, textureLoad(input, vec2(x+1, y+2)));
    }

    textureStore(output, output_coords, value);
}