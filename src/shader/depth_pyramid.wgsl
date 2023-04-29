@group(0) @binding(0)
var input: texture_storage_2d<r32float, read>;

@group(0) @binding(1)
var output: texture_storage_2d<r32float, write>;

fn loadDepth(x: i32, y: i32, dimensions: vec2<i32>) -> f32 {
    // Avoid platform specific out of bounds access behavior.
    // Sampling edge texels multiple times is fine.
    let max_x = dimensions.x - 1;
    let max_y = dimensions.y - 1;
    let coords = vec2(clamp(x, 0, max_x), clamp(y, 0, max_y));
    return textureLoad(input, coords).x;
}

@compute
@workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let output_dimensions = textureDimensions(input);
    let output_coords = vec2<i32>(global_id.xy);
    if (output_coords.x >= output_dimensions.x || output_coords.y >= output_dimensions.y) {
        return;
    }

    let dimensions = textureDimensions(input);

    // The input should have twice the resolution of the output.
    let x = i32(global_id.x) * 2;
    let y = i32(global_id.y) * 2;

    // Gather a 2x2 pixel region and take the min depth (farthest depth in reversed-z).
    // This avoids occluders appearing larger in smaller mips.
    let value00 = loadDepth(x, y, dimensions);
    let value01 = loadDepth(x+1, y, dimensions);
    let value10 = loadDepth(x, y+1, dimensions);
    let value11 = loadDepth(x+1, y+1, dimensions);
    var value = min(min(value00, value01), min(value10, value11));

    // Handle the case where the previous mipmap has odd dimensions.
    // We may need to reduce 3 pixels instead of just two pixels in each dimension.
    // https://miketuritzin.com/post/hierarchical-depth-buffers/
    let include_extra_column = (dimensions.x & 1) != 0;
    let include_extra_row = (dimensions.y & 1) != 0;

    if (include_extra_column) {
        value = min(value, loadDepth(x+2, y, dimensions));
        value = min(value, loadDepth(x+2, y+1, dimensions));

        if (include_extra_row) {
            value = min(value, loadDepth(x+2, y+2, dimensions));
        }
    } 
    if (include_extra_row) {
        value = min(value, loadDepth(x, y+2, dimensions));
        value = min(value, loadDepth(x+1, y+2, dimensions));
    }

    textureStore(output, output_coords, vec4(value));
}