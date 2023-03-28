use crate::{depth_stencil_reversed, shader};

pub fn create_pipeline(
    device: &wgpu::Device,
    surface_format: wgpu::TextureFormat,
    edges: bool,
) -> wgpu::RenderPipeline {
    let shader = shader::model::create_shader_module(device);
    let render_pipeline_layout = shader::model::create_pipeline_layout(device);

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Render Pipeline"),
        layout: Some(&render_pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: "vs_main",
            buffers: &[
                shader::model::VertexInput::vertex_buffer_layout(wgpu::VertexStepMode::Vertex),
                shader::model::InstanceInput::vertex_buffer_layout(wgpu::VertexStepMode::Instance),
            ],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: if edges { "fs_edge_main" } else { "fs_main" },
            targets: &[Some(wgpu::ColorTargetState {
                format: surface_format,
                // Premultiplied alpha.
                blend: Some(wgpu::BlendState {
                    color: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::One,
                        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                        operation: wgpu::BlendOperation::Add,
                    },
                    alpha: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::One,
                        dst_factor: wgpu::BlendFactor::One,
                        operation: wgpu::BlendOperation::Add,
                    },
                }),
                write_mask: wgpu::ColorWrites::all(),
            })],
        }),
        primitive: if edges {
            wgpu::PrimitiveState {
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Line,
                topology: wgpu::PrimitiveTopology::LineList,
                ..Default::default()
            }
        } else {
            wgpu::PrimitiveState {
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            }
        },
        depth_stencil: Some(depth_stencil_reversed()),
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
    })
}

pub fn create_occluder_pipeline(device: &wgpu::Device) -> wgpu::RenderPipeline {
    let shader = shader::model::create_shader_module(device);
    let render_pipeline_layout = shader::model::create_pipeline_layout(device);

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Occluder Pipeline"),
        layout: Some(&render_pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: "vs_main",
            buffers: &[
                shader::model::VertexInput::vertex_buffer_layout(wgpu::VertexStepMode::Vertex),
                shader::model::InstanceInput::vertex_buffer_layout(wgpu::VertexStepMode::Instance),
            ],
        },
        fragment: None,
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: Some(depth_stencil_reversed()),
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
    })
}

pub fn create_culling_pipeline(device: &wgpu::Device, entry_point: &str) -> wgpu::ComputePipeline {
    let shader = shader::culling::create_shader_module(device);
    let render_pipeline_layout = shader::culling::create_pipeline_layout(device);

    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Culling Pipeline"),
        layout: Some(&render_pipeline_layout),
        module: &shader,
        entry_point,
    })
}

pub fn create_depth_pyramid_pipeline(device: &wgpu::Device) -> wgpu::ComputePipeline {
    let shader = shader::depth_pyramid::create_shader_module(device);
    let render_pipeline_layout = shader::depth_pyramid::create_pipeline_layout(device);

    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Depth Pyramid Pipeline"),
        layout: Some(&render_pipeline_layout),
        module: &shader,
        entry_point: "main",
    })
}

pub fn create_blit_depth_pipeline(device: &wgpu::Device) -> wgpu::ComputePipeline {
    let shader = shader::blit_depth::create_shader_module(device);
    let render_pipeline_layout = shader::blit_depth::create_pipeline_layout(device);

    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Blit Depth Pipeline"),
        layout: Some(&render_pipeline_layout),
        module: &shader,
        entry_point: "main",
    })
}
