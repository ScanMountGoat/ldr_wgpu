use glam::{Mat4, Vec4};
use wgpu::util::DeviceExt;

use crate::Scene;

pub struct Renderer {
    camera_buf: wgpu::Buffer,

    pipeline: wgpu::RenderPipeline,
    bind_group0: crate::shader::shader::bind_groups::BindGroup0,

    color_texture: wgpu::TextureView,
    blit_pipeline: wgpu::RenderPipeline,
    blit_bind_group0: crate::shader::blit::bind_groups::BindGroup0,
}

impl Renderer {
    pub fn new(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        surface_format: wgpu::TextureFormat,
        ldraw_path: &str,
    ) -> Self {
        let camera = {
            crate::shader::shader::Camera {
                view: Mat4::IDENTITY,
                view_inv: Mat4::IDENTITY,
                proj_inv: Mat4::IDENTITY,
            }
        };
        let camera_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Buffer"),
            contents: bytemuck::cast_slice(&[camera]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let pipeline = create_shader_pipeline(device, wgpu::TextureFormat::Rgba8Unorm);

        let color_table = ldr_tools::load_color_table(ldraw_path);

        // TODO: How large will this be?
        // TODO: Reindex colors to only include used colors?
        let mut colors = vec![crate::shader::shader::LDrawColor { rgba: Vec4::ZERO }; 1024];
        for (code, color) in color_table {
            if let Some(scene_color) = colors.get_mut(code as usize) {
                *scene_color = crate::shader::shader::LDrawColor {
                    rgba: color.rgba_linear.into(),
                };
            }
        }
        let colors = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("LDraw Colors Buffer"),
            contents: bytemuck::cast_slice(&colors),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let color_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let bind_group0 = crate::shader::shader::bind_groups::BindGroup0::from_bindings(
            device,
            crate::shader::shader::bind_groups::BindGroupLayout0 {
                camera: camera_buf.as_entire_buffer_binding(),
                colors: colors.as_entire_buffer_binding(),
                color_sampler: &color_sampler,
            },
        );

        let color_texture = create_color_texture(device, width, height);

        let blit_bind_group0 = crate::shader::blit::bind_groups::BindGroup0::from_bindings(
            device,
            crate::shader::blit::bind_groups::BindGroupLayout0 {
                color: &color_texture,
                color_sampler: &color_sampler,
            },
        );

        let blit_pipeline = blit_pipeline(device, surface_format);

        Renderer {
            camera_buf,
            pipeline,
            bind_group0,
            color_texture,
            blit_bind_group0,
            blit_pipeline,
        }
    }

    pub fn update_camera(&self, queue: &wgpu::Queue, camera_data: crate::shader::shader::Camera) {
        queue.write_buffer(&self.camera_buf, 0, bytemuck::cast_slice(&[camera_data]));
    }

    pub fn render(
        &mut self,
        view: &wgpu::TextureView,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        scene: &Scene,
    ) {
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        self.model_pass(&mut encoder, scene);
        self.blit_pass(&mut encoder, view);

        queue.submit(Some(encoder.finish()));
    }

    fn model_pass(&mut self, encoder: &mut wgpu::CommandEncoder, scene: &Scene) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Model Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.color_texture,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::GREEN),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        pass.set_pipeline(&self.pipeline);
        self.bind_group0.set(&mut pass);

        scene.bind_group1.set(&mut pass);

        pass.draw(0..3, 0..1);
    }

    fn blit_pass(&mut self, encoder: &mut wgpu::CommandEncoder, view: &wgpu::TextureView) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Blit Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::GREEN),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        pass.set_pipeline(&self.blit_pipeline);
        self.blit_bind_group0.set(&mut pass);

        pass.draw(0..3, 0..1);
    }

    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        self.color_texture = create_color_texture(device, width, height);

        let color_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        self.blit_bind_group0 = crate::shader::blit::bind_groups::BindGroup0::from_bindings(
            device,
            crate::shader::blit::bind_groups::BindGroupLayout0 {
                color: &self.color_texture,
                color_sampler: &color_sampler,
            },
        );
    }
}

fn create_shader_pipeline(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    let module = crate::shader::shader::create_shader_module(device);
    let layout = crate::shader::shader::create_pipeline_layout(device);
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: None,
        layout: Some(&layout),
        vertex: crate::shader::shader::vertex_state(
            &module,
            &crate::shader::shader::vs_main_entry(),
        ),
        fragment: Some(crate::shader::shader::fragment_state(
            &module,
            &crate::shader::shader::fs_main_entry([Some(format.into())]),
        )),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    })
}

fn blit_pipeline(
    device: &wgpu::Device,
    surface_format: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    let module = crate::shader::blit::create_shader_module(device);
    let layout = crate::shader::blit::create_pipeline_layout(device);
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Blit Pipeline"),
        layout: Some(&layout),
        vertex: crate::shader::blit::vertex_state(&module, &crate::shader::blit::vs_main_entry()),
        fragment: Some(wgpu::FragmentState {
            module: &module,
            entry_point: Some("fs_main"),
            compilation_options: Default::default(),
            targets: &[Some(surface_format.into())],
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    })
}

fn create_color_texture(device: &wgpu::Device, width: u32, height: u32) -> wgpu::TextureView {
    // Use 2x for width and height to apply basic supersampling.
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("color texture"),
        size: wgpu::Extent3d {
            width: width * 2,
            height: height * 2,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });

    texture.create_view(&Default::default())
}
