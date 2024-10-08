use std::collections::HashMap;

use glam::{Mat4, Vec4Swizzles};
use ldr_tools::{LDrawColor, LDrawSceneInstanced};
use log::info;
use meshopt::optimize_vertex_cache;
use rayon::prelude::*;
use wgpu::util::DeviceExt;

use crate::geometry::IndexedVertexData;

/// Combined data for every part in the scene.
/// Renderable with a single multidraw indirect call.
pub struct IndirectSceneData {
    pub instance_transforms_buffer: wgpu::Buffer,
    pub instance_bounds_buffer: wgpu::Buffer,
    pub visibility_buffer: wgpu::Buffer,
    pub new_visibility_buffer: wgpu::Buffer,
    pub scanned_new_visibility_buffer: wgpu::Buffer,
    pub scanned_visibility_buffer: wgpu::Buffer,
    pub transparent_buffer: wgpu::Buffer,
    pub compacted_count_buffer: wgpu::Buffer,
    pub compacted_count_staging_buffer: wgpu::Buffer,
    pub vertex_buffer: wgpu::Buffer,
    pub solid: IndirectData,
    pub edges: IndirectData,
}

pub struct IndirectData {
    pub index_buffer: wgpu::Buffer,
    pub indirect_buffer: wgpu::Buffer,
    pub compacted_indirect_buffer: wgpu::Buffer,
    pub draw_count: u32,
    pub compacted_draw_count: u32,
}

// wgpu already provides this type.
// Make our own so we can derive bytemuck.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DrawIndexedIndirect {
    vertex_count: u32,
    instance_count: u32,
    base_index: u32,
    vertex_offset: i32,
    base_instance: u32,
}

pub fn load_render_data(
    device: &wgpu::Device,
    scene: &LDrawSceneInstanced,
    color_table: &HashMap<u32, LDrawColor>,
) -> IndirectSceneData {
    // Combine all data into a single multidraw indirect call.
    let mut combined_vertices = Vec::new();
    let mut combined_indices = Vec::new();
    let mut combined_transforms = Vec::new();
    let mut indirect_draws = Vec::new();
    let mut instance_bounds = Vec::new();
    let mut is_part_transparent = Vec::new();

    let mut combined_edge_indices = Vec::new();
    let mut edge_indirect_draws = Vec::new();

    // Sort so that transparent draws happen last for proper blending.
    // Opaque objects evaluate to false and appear first when sorted.
    // This is simpler than drawing separate opaque and transparent passes.
    let mut alpha_sorted: Vec<_> = scene.geometry_world_transforms.iter().collect();
    alpha_sorted.sort_by_key(|((_, color), _)| is_transparent(color_table, color));

    // Geometry for parts appearing in multiple colors should be calculated only once.
    // Use multiple threads to improve performance since parts are independent.
    let part_vertex_data: HashMap<_, _> = scene
        .geometry_cache
        .par_iter()
        .map(|(name, geometry)| (name.clone(), IndexedVertexData::from_geometry(geometry)))
        .collect();

    // TODO: perform these conversions in parallel?
    // TODO: Parallelizing this will require scanning the sizes to calculate buffer offsets.
    for ((name, color), transforms) in alpha_sorted {
        let base_index = combined_indices.len() as u32;
        let base_edge_index = combined_edge_indices.len() as u32;
        let vertex_offset = combined_vertices.len() as i32;

        // Create separate vertex data if a part has multiple colors.
        // This is necessary since we store face colors per vertex.
        // Copy the vertex data so that we can replace the color.
        let mut vertex_data = part_vertex_data[name].clone();
        vertex_data.replace_colors(*color, color_table);

        // Modern GPUs reuse indices in small batches.
        // This also helps slightly on Apple M1.
        // https://arbook.icg.tugraz.at/schmalstieg/Schmalstieg_351.pdf
        let vertex_indices =
            optimize_vertex_cache(&vertex_data.vertex_indices, vertex_data.vertices.len());

        combined_vertices.extend_from_slice(&vertex_data.vertices);
        combined_indices.extend_from_slice(&vertex_indices);
        combined_edge_indices.extend_from_slice(&vertex_data.edge_indices);

        let is_transparent = color_table
            .get(color)
            .map(|c| c.rgba_linear[3] < 1.0)
            .unwrap_or_default();

        // Each draw specifies the part mesh using an offset and count.
        // The base instance steps through the transforms buffer.
        // Each draw uses a single instance to allow culling individual draws.
        for transform in transforms {
            // TODO: Is this the best way to share culling information with edges?
            let edge_indirect_draw = DrawIndexedIndirect {
                vertex_count: combined_edge_indices.len() as u32 - base_edge_index,
                instance_count: 1,
                base_index: base_edge_index,
                vertex_offset,
                base_instance: combined_transforms.len() as u32,
            };
            edge_indirect_draws.push(edge_indirect_draw);

            let draw = DrawIndexedIndirect {
                vertex_count: combined_indices.len() as u32 - base_index,
                instance_count: 1,
                base_index,
                vertex_offset,
                base_instance: combined_transforms.len() as u32,
            };
            indirect_draws.push(draw);

            // Transform the bounds from the cached geometry.
            // This avoids looping over the points again and improves performance.
            // TODO: Find an efficient way to potentially update this each frame.
            let bounds = transform_bounds(vertex_data.bounds, *transform);
            instance_bounds.push(bounds);

            combined_transforms.push(*transform);

            is_part_transparent.push(is_transparent as u32);
        }
    }

    info!(
        "vertices: {}, indices: {}",
        combined_vertices.len(),
        combined_indices.len()
    );

    // TODO: Create buffer creation helper functions
    // vertex_buffer, index_buffer, indirect_buffer, etc
    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("vertex buffer"),
        contents: bytemuck::cast_slice(&combined_vertices),
        usage: wgpu::BufferUsages::VERTEX,
    });

    let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("index buffer"),
        contents: bytemuck::cast_slice(&combined_indices),
        usage: wgpu::BufferUsages::INDEX,
    });

    let edge_index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("edge index buffer"),
        contents: bytemuck::cast_slice(&combined_edge_indices),
        usage: wgpu::BufferUsages::INDEX,
    });

    // TODO: the non compacted buffer could just be storage?
    let indirect_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("indirect buffer"),
        contents: bytemuck::cast_slice(&indirect_draws),
        usage: wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::STORAGE,
    });
    let compacted_indirect_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("compacted indirect buffer"),
        contents: bytemuck::cast_slice(&indirect_draws),
        usage: wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::STORAGE,
    });

    let edge_indirect_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("edge indirect buffer"),
        contents: bytemuck::cast_slice(&edge_indirect_draws),
        usage: wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::STORAGE,
    });
    let compacted_edge_indirect_buffer =
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("compacted edge indirect buffer"),
            contents: bytemuck::cast_slice(&edge_indirect_draws),
            usage: wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::STORAGE,
        });

    let instance_transforms_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("instance transforms buffer"),
        contents: bytemuck::cast_slice(&combined_transforms),
        usage: wgpu::BufferUsages::VERTEX,
    });

    let instance_bounds_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("instance bounds buffer"),
        contents: bytemuck::cast_slice(&instance_bounds),
        usage: wgpu::BufferUsages::STORAGE,
    });

    // Start with all objects visible.
    // This should only negatively impact performance on the first frame.
    let visibility_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("visibility buffer"),
        contents: bytemuck::cast_slice(&vec![1u32; indirect_draws.len()]),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let new_visibility_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("new visibility buffer"),
        contents: bytemuck::cast_slice(&vec![0u32; indirect_draws.len()]),
        usage: wgpu::BufferUsages::STORAGE,
    });

    // Used to prevent transparent objects occluding other objects.
    let transparent_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("transparent buffer"),
        contents: bytemuck::cast_slice(&is_part_transparent),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let compacted_count_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("compacted draw count buffer"),
        contents: bytemuck::cast_slice(&[0u32]),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::INDIRECT,
    });

    let compacted_count_staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("compacted count staging buffer"),
        size: compacted_count_buffer.size(),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let scanned_visibility_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("scanned visibility buffer"),
        size: visibility_buffer.size(),
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let scanned_new_visibility_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("scanned visibility buffer"),
        size: visibility_buffer.size(),
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    IndirectSceneData {
        vertex_buffer,
        visibility_buffer,
        new_visibility_buffer,
        instance_transforms_buffer,
        instance_bounds_buffer,
        compacted_count_buffer,
        compacted_count_staging_buffer,
        scanned_visibility_buffer,
        scanned_new_visibility_buffer,
        transparent_buffer,
        solid: IndirectData {
            index_buffer,
            indirect_buffer,
            draw_count: indirect_draws.len() as u32,
            compacted_draw_count: indirect_draws.len() as u32,
            compacted_indirect_buffer,
        },
        edges: IndirectData {
            index_buffer: edge_index_buffer,
            indirect_buffer: edge_indirect_buffer,
            draw_count: edge_indirect_draws.len() as u32,
            compacted_draw_count: edge_indirect_draws.len() as u32,
            compacted_indirect_buffer: compacted_edge_indirect_buffer,
        },
    }
}

fn transform_bounds(
    bounds: crate::shader::culling::InstanceBounds,
    transform: Mat4,
) -> crate::shader::culling::InstanceBounds {
    // More efficient than transforming each corner of the AABB.
    // https://stackoverflow.com/questions/6053522/how-to-recalculate-axis-aligned-bounding-box-after-translate-rotate
    let mut min_xyz = transform.col(3).xyz();
    let mut max_xyz = transform.col(3).xyz();
    for i in 0..3 {
        for j in 0..3 {
            let a = transform.row(i)[j] * bounds.min_xyz[j];
            let b = transform.row(i)[j] * bounds.max_xyz[j];
            min_xyz[i] += if a < b { a } else { b };
            max_xyz[i] += if a < b { b } else { a };
        }
    }

    crate::shader::culling::InstanceBounds {
        // Assume no scaling for now to simplify the math.
        sphere: transform
            .transform_point3(bounds.sphere.xyz())
            .extend(bounds.sphere.w),
        min_xyz: min_xyz.extend(0.0),
        max_xyz: max_xyz.extend(0.0),
    }
}

fn is_transparent(color_table: &HashMap<u32, LDrawColor>, color: &u32) -> bool {
    color_table
        .get(color)
        .map(|c| c.rgba_linear[3] < 1.0)
        .unwrap_or_default()
}

pub fn draw_indirect<'a>(
    render_pass: &mut wgpu::RenderPass<'a>,
    scene: &'a IndirectSceneData,
    data: &'a IndirectData,
    supports_indirect_count: bool,
) {
    // Draw the instances of each unique part and color.
    // This allows reusing most of the rendering state for better performance.
    render_pass.set_index_buffer(data.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
    render_pass.set_vertex_buffer(0, scene.vertex_buffer.slice(..));
    render_pass.set_vertex_buffer(1, scene.instance_transforms_buffer.slice(..));

    // Draw each instance with a different transform.
    if supports_indirect_count {
        render_pass.multi_draw_indexed_indirect_count(
            &data.compacted_indirect_buffer,
            0,
            &scene.compacted_count_buffer,
            0,
            data.draw_count,
        );
    } else {
        render_pass.multi_draw_indexed_indirect(
            &data.compacted_indirect_buffer,
            0,
            data.compacted_draw_count,
        );
    }
}

#[cfg(test)]
mod tests {
    use glam::{vec3, vec4};

    use crate::shader::culling::InstanceBounds;

    use super::*;

    #[test]
    fn transform_bounds_identity() {
        assert_eq!(
            InstanceBounds {
                sphere: vec4(0.0, 0.0, 0.0, 1.0),
                min_xyz: vec4(-1.0, -1.0, -1.0, 0.0),
                max_xyz: vec4(1.0, 1.0, 1.0, 0.0),
            },
            transform_bounds(
                InstanceBounds {
                    sphere: vec4(0.0, 0.0, 0.0, 1.0),
                    min_xyz: vec4(-1.0, -1.0, -1.0, 0.0),
                    max_xyz: vec4(1.0, 1.0, 1.0, 0.0),
                },
                Mat4::IDENTITY
            )
        );
    }

    #[test]
    fn transform_bounds_translation() {
        assert_eq!(
            InstanceBounds {
                sphere: vec4(1.0, 2.0, 3.0, 1.0),
                min_xyz: vec4(0.0, 1.0, 2.0, 0.0),
                max_xyz: vec4(2.0, 3.0, 4.0, 0.0),
            },
            transform_bounds(
                InstanceBounds {
                    sphere: vec4(0.0, 0.0, 0.0, 1.0),
                    min_xyz: vec4(-1.0, -1.0, -1.0, 0.0),
                    max_xyz: vec4(1.0, 1.0, 1.0, 0.0),
                },
                Mat4::from_translation(vec3(1.0, 2.0, 3.0))
            )
        );
    }

    #[test]
    fn transform_bounds_translation_rotation() {
        assert_eq!(
            InstanceBounds {
                sphere: vec4(1.0, 2.0, 3.0, 1.0),
                min_xyz: vec4(0.0, 1.0, 2.0, 0.0),
                max_xyz: vec4(2.0, 3.0, 4.0, 0.0),
            },
            transform_bounds(
                InstanceBounds {
                    sphere: vec4(0.0, 0.0, 0.0, 1.0),
                    min_xyz: vec4(-1.0, -1.0, -1.0, 0.0),
                    max_xyz: vec4(1.0, 1.0, 1.0, 0.0),
                },
                // rotate x -180 degrees -> translate 1,2,3
                // constructed manually to avoid precision issues
                Mat4::from_cols_array_2d(&[
                    [1.0, 0.0, 0.0, 0.0,],
                    [0.0, -1.0, 0.0, 0.0,],
                    [0.0, 0.0, -1.0, 0.0,],
                    [1.0, 2.0, 3.0, 1.0,],
                ])
            )
        );
    }
}
