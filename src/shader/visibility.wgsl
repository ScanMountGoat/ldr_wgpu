struct DrawIndexedIndirect {
    vertex_count: u32,
    instance_count: u32,
    base_index: u32,
    vertex_offset: i32,
    base_instance: u32,
}

// TODO: Make these write only?
@group(0) @binding(0)
var<storage, read_write> draws: array<DrawIndexedIndirect>;

@group(0) @binding(1)
var<storage, read_write> edge_draws: array<DrawIndexedIndirect>;

// Set to visible or newly visible buffer depending on the render pass.
@group(0) @binding(2)
var<storage, read_write> visibility: array<u32>;

@compute
@workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Assume all the arrays have the same length.
    let index = global_id.x;
    if (index >= arrayLength(&draws)) {
        return;
    }

    if (visibility[index] != 0u) {
        draws[index].instance_count = 1u;
        edge_draws[index].instance_count = 1u;
    } else {
        draws[index].instance_count = 0u;
        edge_draws[index].instance_count = 0u;
    }
}
