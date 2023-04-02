struct DrawIndexedIndirect {
    vertex_count: u32,
    instance_count: u32,
    base_index: u32,
    vertex_offset: i32,
    base_instance: u32,
}

@group(0) @binding(0)
var<storage, read_write> draws: array<DrawIndexedIndirect>;

@group(0) @binding(1)
var<storage, read_write> edge_draws: array<DrawIndexedIndirect>;

@group(0) @binding(2)
var<storage, read_write> compacted_draws: array<DrawIndexedIndirect>;

@group(0) @binding(3)
var<storage, read_write> compacted_edge_draws: array<DrawIndexedIndirect>;

@group(0) @binding(4)
var<storage, read> visibility: array<u32>;

// Inclusive scan to enable stream compaction.
@group(0) @binding(5)
var<storage, read> scanned_visibility: array<u32>;

// Use a buffer for storing the final compacted count.
// The multi draw indirect count feature can read this buffer directly.
// Unsupported devices will need to copy the count to the CPU.
@group(0) @binding(6)
var<storage, read_write> compacted_draw_count: array<u32>;

@compute
@workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Assume all the arrays have the same length.
    let index = global_id.x;
    let len = arrayLength(&draws);
    if (index >= len) {
        return;
    }

    if (visibility[index] != 0u) {
        // Move each visible draw based on the number of previous visible draws.
        // This has the effect of removing empty draws and compacting the buffer.
        // For example, scanning 1 0 1 1 gives 1 1 2 3 and compacts to 1 1 1 0.
        // This requires updating the draw count to render properly.
        var compacted_index = 0u;
        if (index > 0u) {
            compacted_index = scanned_visibility[index - 1u];
        }
        compacted_draws[compacted_index] = draws[index];
        compacted_edge_draws[compacted_index] = edge_draws[index];
    }

    if (index == 0u) {
        // This only needs to be written once.
        compacted_draw_count[0] = scanned_visibility[len - 1u];
    }
}
