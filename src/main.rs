use std::{collections::HashMap, num::NonZeroU32};

use futures::executor::block_on;
use glam::{vec3, vec4, Mat4, Vec3, Vec4};
use ldr_tools::{FaceColor, GeometrySettings, LDrawColor, LDrawSceneInstanced, StudType};
use wgpu::util::DeviceExt;
use winit::{
    dpi::PhysicalPosition,
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

use crate::pipeline::*;

mod pipeline;
mod shader;

const MSAA_SAMPLES: u32 = 4;
const COLOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Bgra8UnormSrgb;
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

// The far plane can be infinity since we use reversed-z.
const Z_NEAR: f32 = 0.1;
const Z_FAR: f32 = f32::INFINITY;

fn depth_stencil_reversed() -> wgpu::DepthStencilState {
    wgpu::DepthStencilState {
        // Reversed-z
        format: DEPTH_FORMAT,
        depth_write_enabled: true,
        depth_compare: wgpu::CompareFunction::GreaterEqual,
        stencil: Default::default(),
        bias: Default::default(),
    }
}

fn depth_op_reversed() -> wgpu::Operations<f32> {
    wgpu::Operations {
        // Clear to 0 for reversed z.
        load: wgpu::LoadOp::Clear(0.0),
        store: true,
    }
}

struct CameraData {
    view: Mat4,
    view_projection: Mat4,
    // https://vkguide.dev/docs/gpudriven/compute_culling/
    frustum: Vec4,
    p00: f32,
    p11: f32,
}

// wgpu already provides this type.
// Make our own so we can derive bytemuck.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DrawIndexedIndirect {
    vertex_count: u32,
    instance_count: u32,
    base_index: u32,
    vertex_offset: i32,
    base_instance: u32,
}

/// Combined data for every part in the scene.
/// Renderable with a single multidraw indirect call.
struct IndirectSceneData {
    instance_transforms_buffer: wgpu::Buffer,
    instance_bounds_buffer: wgpu::Buffer,
    visibility_buffer: wgpu::Buffer,
    new_visibility_buffer: wgpu::Buffer,
    scanned_new_visibility_buffer: wgpu::Buffer,
    scanned_visibility_buffer: wgpu::Buffer,
    transparent_buffer: wgpu::Buffer,
    compacted_count_buffer: wgpu::Buffer,
    compacted_count_staging_buffer: wgpu::Buffer,
    vertex_buffer: wgpu::Buffer,
    solid: IndirectData,
    edges: IndirectData,
}

struct IndirectData {
    index_buffer: wgpu::Buffer,
    indirect_buffer: wgpu::Buffer,
    compacted_indirect_buffer: wgpu::Buffer,
    draw_count: u32,
    compacted_draw_count: u32,
}

struct ScanBindGroups {
    scan: shader::scan::bind_groups::BindGroup0,
    scan_sums: Option<Box<ScanBindGroups>>,
    add_sums: shader::scan_add::bind_groups::BindGroup0,
}

struct State {
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    size: winit::dpi::PhysicalSize<u32>,
    config: wgpu::SurfaceConfiguration,

    translation: Vec3,
    rotation_xyz: Vec3,
    camera_buffer: wgpu::Buffer,

    output_view_msaa: wgpu::TextureView,
    depth_texture: wgpu::Texture,
    depth_view: wgpu::TextureView,

    // Store the texture separately since depth attachments can't have mipmaps.
    depth_pyramid_pipeline: wgpu::ComputePipeline,
    blit_depth_pipeline: wgpu::ComputePipeline,
    depth_pyramid: DepthPyramid,

    // Render State
    // TODO: Organize the data better.
    bind_group0: shader::model::bind_groups::BindGroup0,
    model_pipeline: wgpu::RenderPipeline,
    model_edges_pipeline: wgpu::RenderPipeline,

    visibility_pipeline: wgpu::ComputePipeline,
    visible_bind_group: shader::visibility::bind_groups::BindGroup0,
    newly_visible_bind_group: shader::visibility::bind_groups::BindGroup0,

    camera_culling_buffer: wgpu::Buffer,
    culling_bind_group0: shader::culling::bind_groups::BindGroup0,
    culling_bind_group1: shader::culling::bind_groups::BindGroup1,
    culling_pipeline: wgpu::ComputePipeline,

    scan_pipeline: wgpu::ComputePipeline,
    scan_add_pipeline: wgpu::ComputePipeline,
    scan_visible: ScanBindGroups,
    scan_newly_visible: ScanBindGroups,

    render_data: IndirectSceneData,

    supports_indirect_count: bool,

    input_state: InputState,
}

struct DepthPyramid {
    width: u32,
    height: u32,
    all_mips: wgpu::TextureView,
    base_bind_group: shader::blit_depth::bind_groups::BindGroup0,
    mip_bind_groups: Vec<shader::depth_pyramid::bind_groups::BindGroup0>,
}

#[derive(Default)]
struct InputState {
    is_mouse_left_clicked: bool,
    is_mouse_right_clicked: bool,
    previous_cursor_position: PhysicalPosition<f64>,
}

impl State {
    async fn new(
        window: &Window,
        scene: &LDrawSceneInstanced,
        color_table: &HashMap<u32, LDrawColor>,
    ) -> Self {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        let surface = unsafe { instance.create_surface(window).unwrap() };
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();
        println!("{:#?}", adapter.get_info());

        let mut features = wgpu::Features::MULTI_DRAW_INDIRECT
            | wgpu::Features::INDIRECT_FIRST_INSTANCE
            | wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES
            | wgpu::Features::POLYGON_MODE_LINE;

        let supports_indirect_count = adapter
            .features()
            .contains(wgpu::Features::MULTI_DRAW_INDIRECT_COUNT);
        println!("{:?}", adapter.features());

        if supports_indirect_count {
            features |= wgpu::Features::MULTI_DRAW_INDIRECT_COUNT;
        }

        // TODO: Add an optimized code path if MULTI_DRAW_INDIRECT_COUNT is available.
        // This avoids synchronization with the CPU.
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    features,
                    limits: wgpu::Limits::default(),
                },
                None,
            )
            .await
            .unwrap();

        let size = window.inner_size();
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: COLOR_FORMAT,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: Vec::new(),
        };
        surface.configure(&device, &config);

        let model_pipeline = create_pipeline(&device, COLOR_FORMAT, false);
        let model_edges_pipeline = create_pipeline(&device, COLOR_FORMAT, true);

        let visibility_pipeline = shader::visibility::compute::create_main_pipeline(&device);
        let culling_pipeline = shader::culling::compute::create_main_pipeline(&device);
        let scan_pipeline = shader::scan::compute::create_main_pipeline(&device);
        let scan_add_pipeline = shader::scan_add::compute::create_main_pipeline(&device);
        let depth_pyramid_pipeline = shader::depth_pyramid::compute::create_main_pipeline(&device);
        let blit_depth_pipeline = shader::blit_depth::compute::create_main_pipeline(&device);

        let translation = vec3(0.0, -0.5, -20.0);
        let rotation_xyz = Vec3::ZERO;
        let camera_data = calculate_camera_data(size, translation, rotation_xyz);

        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("camera buffer"),
            contents: bytemuck::cast_slice(&[shader::model::Camera {
                view_projection: camera_data.view_projection,
            }]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group0 = shader::model::bind_groups::BindGroup0::from_bindings(
            &device,
            shader::model::bind_groups::BindGroupLayout0 {
                camera: camera_buffer.as_entire_buffer_binding(),
            },
        );

        let start = std::time::Instant::now();
        let render_data = load_render_data(&device, scene, color_table);
        println!(
            "Load {} parts and {} unique parts: {:?}",
            render_data.solid.draw_count,
            scene.geometry_cache.len(),
            start.elapsed()
        );

        // TODO: just use encase for this to avoid manually handling padding?
        let camera_culling_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("camera culling buffer"),
            contents: bytemuck::cast_slice(&[shader::culling::Camera {
                z_near: Z_NEAR,
                z_far: Z_FAR,
                p00: camera_data.p00,
                p11: camera_data.p11,
                frustum: camera_data.frustum,
                view_projection: camera_data.view_projection,
                view: camera_data.view,
            }]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let (depth_texture, depth_view) = create_depth_texture(&device, size.width, size.height);

        let depth_pyramid = create_depth_pyramid(&device, size, &depth_view);

        let culling_bind_group0 = shader::culling::bind_groups::BindGroup0::from_bindings(
            &device,
            shader::culling::bind_groups::BindGroupLayout0 {
                camera: camera_culling_buffer.as_entire_buffer_binding(),
                depth_pyramid: &depth_pyramid.all_mips,
            },
        );

        let culling_bind_group1 = shader::culling::bind_groups::BindGroup1::from_bindings(
            &device,
            shader::culling::bind_groups::BindGroupLayout1 {
                instance_bounds: render_data
                    .instance_bounds_buffer
                    .as_entire_buffer_binding(),
                visibility: render_data.visibility_buffer.as_entire_buffer_binding(),
                new_visibility: render_data.new_visibility_buffer.as_entire_buffer_binding(),
                transparent: render_data.transparent_buffer.as_entire_buffer_binding(),
            },
        );

        let visible_bind_group = shader::visibility::bind_groups::BindGroup0::from_bindings(
            &device,
            shader::visibility::bind_groups::BindGroupLayout0 {
                draws: render_data.solid.indirect_buffer.as_entire_buffer_binding(),
                edge_draws: render_data.edges.indirect_buffer.as_entire_buffer_binding(),
                visibility: render_data.visibility_buffer.as_entire_buffer_binding(),
                scanned_visibility: render_data
                    .scanned_visibility_buffer
                    .as_entire_buffer_binding(),
                compacted_draws: render_data
                    .solid
                    .compacted_indirect_buffer
                    .as_entire_buffer_binding(),
                compacted_edge_draws: render_data
                    .edges
                    .compacted_indirect_buffer
                    .as_entire_buffer_binding(),
                compacted_draw_count: render_data
                    .compacted_count_buffer
                    .as_entire_buffer_binding(),
            },
        );

        let newly_visible_bind_group = shader::visibility::bind_groups::BindGroup0::from_bindings(
            &device,
            shader::visibility::bind_groups::BindGroupLayout0 {
                draws: render_data.solid.indirect_buffer.as_entire_buffer_binding(),
                edge_draws: render_data.edges.indirect_buffer.as_entire_buffer_binding(),
                visibility: render_data.new_visibility_buffer.as_entire_buffer_binding(),
                scanned_visibility: render_data
                    .scanned_new_visibility_buffer
                    .as_entire_buffer_binding(),
                compacted_draws: render_data
                    .solid
                    .compacted_indirect_buffer
                    .as_entire_buffer_binding(),
                compacted_edge_draws: render_data
                    .edges
                    .compacted_indirect_buffer
                    .as_entire_buffer_binding(),
                compacted_draw_count: render_data
                    .compacted_count_buffer
                    .as_entire_buffer_binding(),
            },
        );

        // Create separate inputs for both visible and newly visible passes.
        // Most of the output buffers can be reused.
        let scan_visible = create_scan_bind_groups(
            &device,
            &render_data.visibility_buffer,
            &render_data.scanned_visibility_buffer,
        );

        let scan_newly_visible = create_scan_bind_groups(
            &device,
            &render_data.new_visibility_buffer,
            &render_data.scanned_new_visibility_buffer,
        );

        let output_view_msaa = create_output_msaa_view(&device, size.width, size.height);

        Self {
            surface,
            device,
            queue,
            size,
            config,
            model_pipeline,
            model_edges_pipeline,
            visibility_pipeline,
            culling_pipeline,
            culling_bind_group0,
            culling_bind_group1,
            bind_group0,
            render_data,
            translation,
            rotation_xyz,
            camera_buffer,
            depth_texture,
            depth_view,
            output_view_msaa,
            camera_culling_buffer,
            depth_pyramid,
            depth_pyramid_pipeline,
            blit_depth_pipeline,
            visible_bind_group,
            newly_visible_bind_group,
            scan_pipeline,
            scan_add_pipeline,
            scan_visible,
            scan_newly_visible,
            supports_indirect_count,
            input_state: Default::default(),
        }
    }

    fn update_camera(&self, size: winit::dpi::PhysicalSize<u32>) {
        let camera_data = calculate_camera_data(size, self.translation, self.rotation_xyz);
        self.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&[shader::model::Camera {
                view_projection: camera_data.view_projection,
            }]),
        );
        self.queue.write_buffer(
            &self.camera_culling_buffer,
            0,
            bytemuck::cast_slice(&[shader::culling::Camera {
                z_near: Z_NEAR,
                z_far: Z_FAR,
                p00: camera_data.p00,
                p11: camera_data.p11,
                frustum: camera_data.frustum,
                view_projection: camera_data.view_projection,
                view: camera_data.view,
            }]),
        );
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            // Update each resource that depends on window size.
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);

            let (depth_texture, depth_view) =
                create_depth_texture(&self.device, new_size.width, new_size.height);
            self.depth_texture = depth_texture;
            self.depth_view = depth_view;

            self.depth_pyramid = create_depth_pyramid(&self.device, new_size, &self.depth_view);

            self.output_view_msaa =
                create_output_msaa_view(&self.device, new_size.width, new_size.height);

            // The textures were updated, so use views pointing to the new textures.
            self.culling_bind_group0 = shader::culling::bind_groups::BindGroup0::from_bindings(
                &self.device,
                shader::culling::bind_groups::BindGroupLayout0 {
                    camera: self.camera_culling_buffer.as_entire_buffer_binding(),
                    depth_pyramid: &self.depth_pyramid.all_mips,
                },
            );
        }
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let output_view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        // Use a two pass conservative culling scheme introduced in the following paper:
        // "Patch-Based Occlusion Culling for Hardware Tessellation"
        // http://www.graphics.stanford.edu/~niessner/papers/2012/2occlusion/niessner2012patch.pdf
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        self.set_visibility_pass(&mut encoder, false);

        // The synchronization and copies aren't necessary if indirect count is supported.
        if !self.supports_indirect_count {
            encoder.copy_buffer_to_buffer(
                &self.render_data.compacted_count_buffer,
                0,
                &self.render_data.compacted_count_staging_buffer,
                0,
                self.render_data.compacted_count_staging_buffer.size(),
            );
            // Submit to make sure the copy finishes.
            self.queue.submit(std::iter::once(encoder.finish()));
            self.update_compacted_draw_count();

            encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Render Encoder 2"),
                });
        }

        // Draw everything that was visible last frame.
        self.previously_visible_pass(&mut encoder, &output_view);

        // Apply culling to set visibility and enable newly visible objects.
        self.depth_pyramid_pass(&mut encoder);
        self.occlusion_culling_pass(&mut encoder);
        self.set_visibility_pass(&mut encoder, true);

        if !self.supports_indirect_count {
            // Make sure the staging buffer is set up for the next compaction operation.
            encoder.copy_buffer_to_buffer(
                &self.render_data.compacted_count_buffer,
                0,
                &self.render_data.compacted_count_staging_buffer,
                0,
                self.render_data.compacted_count_staging_buffer.size(),
            );
            // Submit to make sure the copy completes.
            self.queue.submit(std::iter::once(encoder.finish()));
            self.update_compacted_draw_count();

            encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Render Encoder 3"),
                });
        }

        // Draw everything that is newly visible in this frame.
        self.newly_visible_pass(&mut encoder, &output_view);

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();
        Ok(())
    }

    fn previously_visible_pass(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        output_view: &wgpu::TextureView,
    ) {
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Previously Visible Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.output_view_msaa,
                resolve_target: Some(&output_view),
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: true,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.depth_view,
                depth_ops: Some(depth_op_reversed()),
                stencil_ops: None,
            }),
        });

        shader::model::bind_groups::set_bind_groups(
            &mut render_pass,
            shader::model::bind_groups::BindGroups {
                bind_group0: &self.bind_group0,
            },
        );

        // TODO: make this a method?
        render_pass.set_pipeline(&self.model_pipeline);
        draw_indirect(
            &mut render_pass,
            &self.render_data,
            &self.render_data.solid,
            self.supports_indirect_count,
        );

        render_pass.set_pipeline(&self.model_edges_pipeline);
        draw_indirect(
            &mut render_pass,
            &self.render_data,
            &self.render_data.edges,
            self.supports_indirect_count,
        );
    }

    fn newly_visible_pass(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        output_view: &wgpu::TextureView,
    ) {
        // Load the data from the first model pass.
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Newly Visible Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.output_view_msaa,
                resolve_target: Some(&output_view),
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: true,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: true,
                }),
                stencil_ops: None,
            }),
        });

        shader::model::bind_groups::set_bind_groups(
            &mut render_pass,
            shader::model::bind_groups::BindGroups {
                bind_group0: &self.bind_group0,
            },
        );

        // TODO: make this a method?
        render_pass.set_pipeline(&self.model_pipeline);
        draw_indirect(
            &mut render_pass,
            &self.render_data,
            &self.render_data.solid,
            self.supports_indirect_count,
        );

        render_pass.set_pipeline(&self.model_edges_pipeline);
        draw_indirect(
            &mut render_pass,
            &self.render_data,
            &self.render_data.edges,
            self.supports_indirect_count,
        );
    }

    fn update_compacted_draw_count(&mut self) {
        // TODO: return a value instead?
        let buffer_slice = self.render_data.compacted_count_staging_buffer.slice(..);

        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

        self.device.poll(wgpu::Maintain::Wait);

        if let Some(Ok(())) = block_on(receiver.receive()) {
            let data = buffer_slice.get_mapped_range();
            let counts: &[u32] = bytemuck::cast_slice(&data);

            let draw_count = counts[0];
            self.render_data.solid.compacted_draw_count = draw_count;
            self.render_data.edges.compacted_draw_count = draw_count;

            drop(data);
            self.render_data.compacted_count_staging_buffer.unmap();
        }
    }

    fn occlusion_culling_pass(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Occlusion Culling Pass"),
        });

        compute_pass.set_pipeline(&self.culling_pipeline);
        shader::culling::bind_groups::set_bind_groups(
            &mut compute_pass,
            shader::culling::bind_groups::BindGroups {
                bind_group0: &self.culling_bind_group0,
                bind_group1: &self.culling_bind_group1,
            },
        );

        // Assume the workgroup is 1D.
        let [size_x, _, _] = shader::culling::compute::MAIN_WORKGROUP_SIZE;
        let count = div_round_up(self.render_data.solid.draw_count, size_x);
        compute_pass.dispatch_workgroups(count, 1, 1);
    }

    fn set_visibility_pass(&self, encoder: &mut wgpu::CommandEncoder, newly_visible: bool) {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Set Visibility Pass"),
        });

        if newly_visible {
            self.scan_recursive(&mut compute_pass, &self.scan_newly_visible);
            self.set_visibility(&mut compute_pass, &self.newly_visible_bind_group);
        } else {
            self.scan_recursive(&mut compute_pass, &self.scan_visible);
            self.set_visibility(&mut compute_pass, &self.visible_bind_group);
        }
    }

    fn scan_recursive<'a>(
        &'a self,
        compute_pass: &mut wgpu::ComputePass<'a>,
        bind_groups: &'a ScanBindGroups,
    ) {
        // Recursively scan to support scanning arrays much larger than a single workgroup.
        self.scan(compute_pass, &bind_groups.scan);
        if let Some(scan_sums) = &bind_groups.scan_sums {
            self.scan_recursive(compute_pass, scan_sums);
        }
        self.scan_add(compute_pass, &bind_groups.add_sums);
    }

    fn set_visibility<'a>(
        &'a self,
        compute_pass: &mut wgpu::ComputePass<'a>,
        bind_group0: &'a shader::visibility::bind_groups::BindGroup0,
    ) {
        compute_pass.set_pipeline(&self.visibility_pipeline);
        shader::visibility::bind_groups::set_bind_groups(
            compute_pass,
            shader::visibility::bind_groups::BindGroups { bind_group0 },
        );

        // Assume the workgroup is 1D.
        let [size_x, _, _] = shader::visibility::compute::MAIN_WORKGROUP_SIZE;
        let count = div_round_up(self.render_data.solid.draw_count, size_x);
        compute_pass.dispatch_workgroups(count, 1, 1);
    }

    fn scan_add<'a>(
        &'a self,
        compute_pass: &mut wgpu::ComputePass<'a>,
        bind_group0: &'a shader::scan_add::bind_groups::BindGroup0,
    ) {
        compute_pass.set_pipeline(&self.scan_add_pipeline);
        shader::scan_add::bind_groups::set_bind_groups(
            compute_pass,
            shader::scan_add::bind_groups::BindGroups { bind_group0 },
        );

        // Assume the workgroup is 1D and processes 2 elements per thread.
        let [size_x, _, _] = shader::scan_add::compute::MAIN_WORKGROUP_SIZE;
        let count = div_round_up(self.render_data.solid.draw_count, size_x * 2);
        compute_pass.dispatch_workgroups(count, 1, 1);
    }

    fn scan<'a>(
        &'a self,
        compute_pass: &mut wgpu::ComputePass<'a>,
        bind_group0: &'a shader::scan::bind_groups::BindGroup0,
    ) {
        compute_pass.set_pipeline(&self.scan_pipeline);
        shader::scan::bind_groups::set_bind_groups(
            compute_pass,
            shader::scan::bind_groups::BindGroups { bind_group0 },
        );

        // Assume the workgroup is 1D and processes 2 elements per thread.
        let [size_x, _, _] = shader::scan::compute::MAIN_WORKGROUP_SIZE;
        let count = div_round_up(self.render_data.solid.draw_count, size_x * 2);
        compute_pass.dispatch_workgroups(count, 1, 1);
    }

    fn depth_pyramid_pass(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Depth Pyramid Pass"),
        });

        // Copy the base level.
        compute_pass.set_pipeline(&self.blit_depth_pipeline);
        shader::blit_depth::bind_groups::set_bind_groups(
            &mut compute_pass,
            shader::blit_depth::bind_groups::BindGroups {
                bind_group0: &self.depth_pyramid.base_bind_group,
            },
        );

        // Assume the workgroup is 2D.
        let [size_x, size_y, _] = shader::blit_depth::compute::MAIN_WORKGROUP_SIZE;
        let count_x = div_round_up(self.depth_pyramid.width, size_x);
        let count_y = div_round_up(self.depth_pyramid.height, size_y);

        compute_pass.dispatch_workgroups(count_x, count_y, 1);

        // Make the depth pyramid for the next frame using the current depth.
        // Each dispatch generates one mip level of the pyramid.
        compute_pass.set_pipeline(&self.depth_pyramid_pipeline);
        for (i, bind_group0) in self.depth_pyramid.mip_bind_groups.iter().enumerate() {
            // The first level is copied separately from the depth texture.
            let mip = i + 1;
            let mip_width = (self.depth_pyramid.width >> mip).max(1);
            let mip_height = (self.depth_pyramid.height >> mip).max(1);

            shader::depth_pyramid::bind_groups::set_bind_groups(
                &mut compute_pass,
                shader::depth_pyramid::bind_groups::BindGroups { bind_group0 },
            );

            // Assume the workgroup is 2D.
            let [size_x, size_y, _] = shader::depth_pyramid::compute::MAIN_WORKGROUP_SIZE;
            let count_x = div_round_up(mip_width, size_x);
            let count_y = div_round_up(mip_height, size_y);

            compute_pass.dispatch_workgroups(count_x, count_y, 1);
        }
    }

    // Make this a reusable library that only requires glam?
    fn handle_input(&mut self, event: &WindowEvent) {
        match event {
            WindowEvent::KeyboardInput { input, .. } => {
                // Basic camera controls using arrow keys.
                if let Some(keycode) = input.virtual_keycode {
                    match keycode {
                        VirtualKeyCode::Left => self.translation.x += 1.0,
                        VirtualKeyCode::Right => self.translation.x -= 1.0,
                        VirtualKeyCode::Up => self.translation.y -= 1.0,
                        VirtualKeyCode::Down => self.translation.y += 1.0,
                        _ => (),
                    }
                }
            }
            WindowEvent::MouseInput { button, state, .. } => {
                // Track mouse clicks to only rotate when dragging while clicked.
                match (button, state) {
                    (MouseButton::Left, ElementState::Pressed) => {
                        self.input_state.is_mouse_left_clicked = true
                    }
                    (MouseButton::Left, ElementState::Released) => {
                        self.input_state.is_mouse_left_clicked = false
                    }
                    (MouseButton::Right, ElementState::Pressed) => {
                        self.input_state.is_mouse_right_clicked = true
                    }
                    (MouseButton::Right, ElementState::Released) => {
                        self.input_state.is_mouse_right_clicked = false
                    }
                    _ => (),
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                if self.input_state.is_mouse_left_clicked {
                    let delta_x = position.x - self.input_state.previous_cursor_position.x;
                    let delta_y = position.y - self.input_state.previous_cursor_position.y;

                    // Swap XY so that dragging left/right rotates left/right.
                    self.rotation_xyz.x += (delta_y * 0.01) as f32;
                    self.rotation_xyz.y += (delta_x * 0.01) as f32;
                }
                // Always update the position to avoid jumps when moving between clicks.
                self.input_state.previous_cursor_position = *position;
            }
            WindowEvent::MouseWheel { delta, .. } => {
                // TODO: Add tests for handling scroll events properly?
                // Scale zoom speed with distance to make it easier to zoom out large scenes.
                let delta_z = match delta {
                    MouseScrollDelta::LineDelta(_x, y) => *y * self.translation.z.abs() * 0.1,
                    MouseScrollDelta::PixelDelta(p) => {
                        p.y as f32 * self.translation.z.abs() * 0.005
                    }
                };

                // Clamp to prevent the user from zooming through the origin.
                self.translation.z = (self.translation.z + delta_z).min(-1.0);
            }
            _ => (),
        }
    }
}

fn create_scan_bind_groups(
    device: &wgpu::Device,
    input: &wgpu::Buffer,
    output: &wgpu::Buffer,
) -> ScanBindGroups {
    // Each workgroup processes 512 elements.
    // This means we only need N / 512 workgroup sums.
    let element_count = input.size() as u32 / std::mem::size_of::<u32>() as u32;
    let elements_per_workgroup = crate::shader::scan::compute::MAIN_WORKGROUP_SIZE[0] * 2;

    let sum_count = div_round_up(element_count, elements_per_workgroup);

    let sums_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("scan workgroup sums buffer"),
        contents: bytemuck::cast_slice(&vec![0u32; sum_count as usize]),
        usage: wgpu::BufferUsages::STORAGE,
    });

    // The workgroup sums themselves may need to be scanned.
    let scanned_sums_buffer = if sum_count > 1 {
        Some(
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("scan workgroup sums buffer"),
                contents: bytemuck::cast_slice(&vec![0u32; sum_count as usize]),
                usage: wgpu::BufferUsages::STORAGE,
            }),
        )
    } else {
        None
    };

    // Recursively scan the workgroup sums before adding them.
    ScanBindGroups {
        scan: shader::scan::bind_groups::BindGroup0::from_bindings(
            device,
            shader::scan::bind_groups::BindGroupLayout0 {
                input: input.as_entire_buffer_binding(),
                output: output.as_entire_buffer_binding(),
                workgroup_sums: sums_buffer.as_entire_buffer_binding(),
            },
        ),
        scan_sums: scanned_sums_buffer.as_ref().map(|scanned_sums| {
            Box::new(create_scan_bind_groups(device, &sums_buffer, scanned_sums))
        }),
        add_sums: shader::scan_add::bind_groups::BindGroup0::from_bindings(
            device,
            shader::scan_add::bind_groups::BindGroupLayout0 {
                output: output.as_entire_buffer_binding(),
                // Use the recursively scanned sums.
                workgroup_sums: scanned_sums_buffer
                    .as_ref()
                    .unwrap_or(&sums_buffer)
                    .as_entire_buffer_binding(),
            },
        ),
    }
}

fn draw_indirect<'a>(
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

fn create_depth_pyramid(
    device: &wgpu::Device,
    window_size: winit::dpi::PhysicalSize<u32>,
    base_depth_view: &wgpu::TextureView,
) -> DepthPyramid {
    // Always use power of two dimensions to avoid edge cases when downsampling.
    let width = window_size.width;
    let height = window_size.height;

    let (pyramid, pyramid_mips) = create_depth_pyramid_texture(device, width, height);
    let pyramid_bind_groups = depth_pyramid_bind_groups(device, &pyramid_mips);

    let pyramid_view = pyramid.create_view(&wgpu::TextureViewDescriptor::default());

    let base_bind_group = shader::blit_depth::bind_groups::BindGroup0::from_bindings(
        device,
        shader::blit_depth::bind_groups::BindGroupLayout0 {
            input: base_depth_view,
            output: &pyramid_mips[0],
        },
    );

    DepthPyramid {
        width,
        height,
        all_mips: pyramid_view,
        base_bind_group,
        mip_bind_groups: pyramid_bind_groups,
    }
}

fn depth_pyramid_bind_groups(
    device: &wgpu::Device,
    mips: &[wgpu::TextureView],
) -> Vec<shader::depth_pyramid::bind_groups::BindGroup0> {
    // The base level is handled separately.
    (1..mips.len())
        .map(|i| {
            shader::depth_pyramid::bind_groups::BindGroup0::from_bindings(
                device,
                shader::depth_pyramid::bind_groups::BindGroupLayout0 {
                    input: &mips[i - 1],
                    output: &mips[i],
                },
            )
        })
        .collect()
}

const fn div_round_up(x: u32, d: u32) -> u32 {
    (x + d - 1) / d
}

fn create_output_msaa_view(device: &wgpu::Device, width: u32, height: u32) -> wgpu::TextureView {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("output msaa texture"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: MSAA_SAMPLES,
        dimension: wgpu::TextureDimension::D2,
        format: COLOR_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });

    texture.create_view(&Default::default())
}

fn create_depth_texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
) -> (wgpu::Texture, wgpu::TextureView) {
    let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("depth texture"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: MSAA_SAMPLES,
        dimension: wgpu::TextureDimension::D2,
        format: DEPTH_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });

    let depth_view = depth_texture.create_view(&Default::default());

    (depth_texture, depth_view)
}

fn create_depth_pyramid_texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
) -> (wgpu::Texture, Vec<wgpu::TextureView>) {
    let size = wgpu::Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
    };
    let mip_level_count = size.max_mips(wgpu::TextureDimension::D2);
    let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("depth texture"),
        size,
        mip_level_count,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R32Float,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });

    let mip_views = (0..mip_level_count)
        .map(|mip| {
            depth_texture.create_view(&wgpu::TextureViewDescriptor {
                base_mip_level: mip,
                mip_level_count: Some(NonZeroU32::new(1).unwrap()),
                ..Default::default()
            })
        })
        .collect();

    (depth_texture, mip_views)
}

fn load_render_data(
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

    for ((name, color), transforms) in alpha_sorted {
        // Create separate vertex data if a part has multiple colors.
        // This is necessary since we store face colors per vertex.
        let geometry = &scene.geometry_cache[name];

        let base_index = combined_indices.len() as u32;
        let base_edge_index = combined_edge_indices.len() as u32;
        let vertex_offset = combined_vertices.len() as i32;

        append_geometry(
            &mut combined_vertices,
            &mut combined_indices,
            &mut combined_edge_indices,
            geometry,
            *color,
            color_table,
        );

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
                vertex_count: geometry.vertex_indices.len() as u32,
                instance_count: 1,
                base_index,
                vertex_offset,
                base_instance: combined_transforms.len() as u32,
            };
            indirect_draws.push(draw);

            let bounds = calculate_instance_bounds(geometry, transform);
            instance_bounds.push(bounds);
            combined_transforms.push(*transform);

            is_part_transparent.push(is_transparent as u32);
        }
    }

    println!(
        "Vertices: {}, Indices: {}",
        combined_vertices.len(),
        combined_indices.len()
    );

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
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let new_visibility_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("new visibility buffer"),
        contents: bytemuck::cast_slice(&vec![0u32; indirect_draws.len()]),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
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

fn is_transparent(color_table: &HashMap<u32, LDrawColor>, color: &u32) -> bool {
    color_table
        .get(color)
        .map(|c| c.rgba_linear[3] < 1.0)
        .unwrap_or_default()
}

fn calculate_instance_bounds(
    geometry: &ldr_tools::LDrawGeometry,
    transform: &Mat4,
) -> shader::culling::InstanceBounds {
    // TODO: Find an efficient way to potentially update this each frame.
    let points_world: Vec<_> = geometry
        .vertices
        .iter()
        .map(|v| transform.transform_point3(*v))
        .collect();

    let sphere_center = points_world.iter().sum::<Vec3>() / points_world.len().max(1) as f32;
    let sphere_radius = points_world
        .iter()
        .map(|v| v.distance(sphere_center))
        .reduce(f32::max)
        .unwrap_or_default();

    let min_xyz = points_world
        .iter()
        .copied()
        .reduce(Vec3::min)
        .unwrap_or_default();
    let max_xyz = points_world
        .iter()
        .copied()
        .reduce(Vec3::max)
        .unwrap_or_default();

    shader::culling::InstanceBounds {
        sphere: sphere_center.extend(sphere_radius),
        min_xyz: min_xyz.extend(0.0),
        max_xyz: max_xyz.extend(0.0),
    }
}

fn append_geometry(
    vertices: &mut Vec<shader::model::VertexInput>,
    vertex_indices: &mut Vec<u32>,
    edge_indices: &mut Vec<u32>,
    geometry: &ldr_tools::LDrawGeometry,
    color_code: u32,
    color_table: &HashMap<u32, LDrawColor>,
) {
    // TODO: Edge colors?
    // TODO: Don't calculate grainy faces to save geometry?

    // TODO: missing color codes?
    // TODO: publicly expose color handling logic in ldr_tools.
    // TODO: handle the case where the face color list is empty?
    match geometry.face_colors.as_slice() {
        [face_color] => {
            // TODO: Is it worth handling this case separately for performance?
            // Each face and also vertex has the same color.
            // The welded vertex indices can be used as is.
            let color = rgba_color(face_color, color_code, color_table);
            vertices.extend(
                geometry
                    .vertices
                    .iter()
                    .map(|v| shader::model::VertexInput {
                        position: *v,
                        color,
                    }),
            );
            vertex_indices.extend_from_slice(&geometry.vertex_indices);

            // Sharp edges define the outlines rendered in instructions or editing applications.
            edge_indices.extend(
                geometry
                    .edges
                    .iter()
                    .zip(geometry.is_edge_sharp.iter())
                    .filter(|(_, sharp)| **sharp)
                    .flat_map(|(e, _)| *e),
            );
        }
        face_colors => {
            let mut index_by_face_vertex = HashMap::new();

            for (i, vertex_index) in geometry.vertex_indices.iter().enumerate() {
                // Assume faces are already triangulated.
                // This means every 3 indices defines a new face.
                let face_color = face_colors.get(i / 3).unwrap_or(&face_colors[0]);
                let new_index = insert_vertex(
                    face_color,
                    color_code,
                    color_table,
                    vertex_index,
                    &mut index_by_face_vertex,
                    geometry,
                    vertices,
                );
                vertex_indices.push(new_index);
            }

            // The vertices have been reindexed, so use the mapping from earlier.
            // This ensures the sharp edge indices reference the correct vertices.
            // TODO: This color specific indexing probably won't work for normals.
            edge_indices.extend(
                geometry
                    .edges
                    .iter()
                    .zip(geometry.is_edge_sharp.iter())
                    .filter(|(_, sharp)| **sharp)
                    .flat_map(|([v0, v1], _)| {
                        // Assume all black edges for now.
                        let i0 = insert_vertex(
                            &FaceColor {
                                color: 0,
                                is_grainy_slope: false,
                            },
                            color_code,
                            color_table,
                            v0,
                            &mut index_by_face_vertex,
                            geometry,
                            vertices,
                        );
                        let i1 = insert_vertex(
                            &FaceColor {
                                color: 0,
                                is_grainy_slope: false,
                            },
                            color_code,
                            color_table,
                            v1,
                            &mut index_by_face_vertex,
                            geometry,
                            vertices,
                        );

                        [i0, i1]
                    }),
            );
        }
    }
}

// TODO: Simplify this and move to ldr_tools with tests?
fn insert_vertex(
    face_color: &ldr_tools::FaceColor,
    color_code: u32,
    color_table: &HashMap<u32, LDrawColor>,
    vertex_index: &u32,
    index_by_face_vertex: &mut HashMap<(u32, u32), u32>,
    geometry: &ldr_tools::LDrawGeometry,
    vertices: &mut Vec<shader::model::VertexInput>,
) -> u32 {
    let color = rgba_color(face_color, color_code, color_table);

    // Store separate indices for position and color.
    // TODO: Create a struct for this?
    // TODO: always ignore grainy slope information?
    let face_vertex_key = (*vertex_index, face_color.color);

    // A vertex is unique if its position and color are unique.
    // This allows attributes like color to be indexed by face.
    // Only the necessary vertices will be duplicated when reindexing.
    if let Some(cached_index) = index_by_face_vertex.get(&face_vertex_key) {
        *cached_index
    } else {
        let new_vertex = shader::model::VertexInput {
            position: geometry.vertices[*vertex_index as usize],
            color,
        };
        let new_index = index_by_face_vertex.len() as u32;
        index_by_face_vertex.insert(face_vertex_key, new_index);

        vertices.push(new_vertex);
        new_index
    }
}

fn rgba_color(
    face_color: &ldr_tools::FaceColor,
    color_code: u32,
    color_table: &HashMap<u32, LDrawColor>,
) -> u32 {
    let replaced_color = if face_color.color == 16 {
        color_code
    } else {
        face_color.color
    };

    color_table
        .get(&replaced_color)
        .map(|c| {
            // TODO: What is the GPU endianness?
            u32::from_le_bytes(c.rgba_linear.map(|f| (f * 255.0) as u8))
        })
        .unwrap_or(0xFFFFFFFF)
}

fn calculate_camera_data(
    size: winit::dpi::PhysicalSize<u32>,
    translation: glam::Vec3,
    rotation: glam::Vec3,
) -> CameraData {
    let aspect = size.width as f32 / size.height as f32;

    // wgpu and LDraw have different coordinate systems.
    let axis_correction = Mat4::from_rotation_x(180.0f32.to_radians());

    let view = glam::Mat4::from_translation(translation)
        * glam::Mat4::from_rotation_x(rotation.x)
        * glam::Mat4::from_rotation_y(rotation.y)
        * axis_correction;

    let projection = glam::Mat4::perspective_infinite_reverse_rh(0.5, aspect, Z_NEAR);

    let view_projection = projection * view;

    // Calculate camera frustum data for culling.
    // https://github.com/zeux/niagara/blob/3fafe000ba8fe6e309b41e915b81242b4ca3db28/src/niagara.cpp#L836-L852
    let perspective_t = projection.transpose();
    // x + w < 0
    let frustum_x = (perspective_t.col(3) + perspective_t.col(0)).normalize();
    // y + w < 0
    let frustum_y = (perspective_t.col(3) + perspective_t.col(1)).normalize();
    let frustum = vec4(frustum_x.x, frustum_x.z, frustum_y.y, frustum_y.z);

    // Used for occlusion based culling.
    let p00 = projection.col(0).x;
    let p11 = projection.col(1).y;

    CameraData {
        view,
        view_projection,
        frustum,
        p00,
        p11,
    }
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    let ldraw_path = &args[1];
    let path = &args[2];

    // Weld vertices to take advantage of vertex caching/batching on the GPU.
    // https://www.khronos.org/opengl/wiki/Post_Transform_Cache
    // https://arbook.icg.tugraz.at/schmalstieg/Schmalstieg_351.pdf
    // TODO: Is it worth applying a more optimal vertex ordering?
    let start = std::time::Instant::now();
    let settings = GeometrySettings {
        triangulate: true,
        weld_vertices: true,
        stud_type: StudType::HighContrast,
        ..Default::default()
    };
    let scene = ldr_tools::load_file_instanced(path, ldraw_path, &settings);
    println!("Load scene: {:?}", start.elapsed());

    let color_table = ldr_tools::load_color_table(ldraw_path);

    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title(concat!("ldr_wgpu ", env!("CARGO_PKG_VERSION")))
        .build(&event_loop)
        .unwrap();

    let mut state = block_on(State::new(&window, &scene, &color_table));
    event_loop.run(move |event, _, control_flow| match event {
        Event::WindowEvent {
            ref event,
            window_id,
        } if window_id == window.id() => match event {
            WindowEvent::CloseRequested
            | WindowEvent::KeyboardInput {
                input:
                    KeyboardInput {
                        state: ElementState::Pressed,
                        virtual_keycode: Some(VirtualKeyCode::Escape),
                        ..
                    },
                ..
            } => *control_flow = ControlFlow::Exit,
            WindowEvent::Resized(physical_size) => {
                state.resize(*physical_size);
                state.update_camera(*physical_size);
            }
            WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                state.resize(**new_inner_size);
            }
            _ => {
                state.handle_input(event);
                state.update_camera(window.inner_size());
            }
        },
        Event::RedrawRequested(_) => match state.render() {
            Ok(_) => {}
            Err(wgpu::SurfaceError::Lost) => state.resize(state.size),
            Err(wgpu::SurfaceError::OutOfMemory) => *control_flow = ControlFlow::Exit,
            Err(e) => eprintln!("{e:?}"),
        },
        Event::MainEventsCleared => {
            window.request_redraw();
        }
        _ => (),
    });
}
