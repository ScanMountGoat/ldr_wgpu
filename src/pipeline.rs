use crate::{depth_stencil_reversed, shader, MSAA_SAMPLES};

pub fn create_pipeline(
    device: &wgpu::Device,
    surface_format: wgpu::TextureFormat,
    edges: bool,
) -> wgpu::RenderPipeline {
    let module = shader::model::create_shader_module(device);
    let render_pipeline_layout = shader::model::create_pipeline_layout(device);

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Render Pipeline"),
        layout: Some(&render_pipeline_layout),
        vertex: shader::model::vertex_state(
            &module,
            &shader::model::vs_main_entry(
                wgpu::VertexStepMode::Vertex,
                wgpu::VertexStepMode::Instance,
            ),
        ),
        fragment: Some(wgpu::FragmentState {
            module: &module,
            entry_point: if edges {
                shader::model::ENTRY_FS_EDGE_MAIN
            } else {
                shader::model::ENTRY_FS_MAIN
            },
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
        multisample: wgpu::MultisampleState {
            count: MSAA_SAMPLES,
            ..Default::default()
        },
        multiview: None,
    })
}
