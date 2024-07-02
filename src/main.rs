use std::collections::HashMap;

use futures::executor::block_on;
use glam::{vec3, vec4, Mat4, Vec3, Vec4};
use ldr_tools::{GeometrySettings, LDrawColor, LDrawSceneInstanced, StudType};
use log::{debug, error, info};
use scene::{draw_indirect, IndirectSceneData};
use texture::create_depth_pyramid_texture;
use wgpu::util::DeviceExt;
use winit::{
    dpi::PhysicalPosition,
    event::*,
    event_loop::EventLoop,
    window::{Window, WindowBuilder},
};

use crate::{
    pipeline::*,
    scene::load_render_data,
    texture::{create_depth_texture, create_output_msaa_view},
};

mod geometry;
mod normal;
mod pipeline;
mod scene;
mod shader;
mod texture;

const MSAA_SAMPLES: u32 = 4;
const COLOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Bgra8UnormSrgb;
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

const FOV_Y: f32 = 0.5;
const Z_NEAR: f32 = 0.1;
// The far plane can be infinity since we use reversed-z.
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
        store: wgpu::StoreOp::Store,
    }
}

struct CameraData {
    view: Mat4,
    view_projection: Mat4,
    // https://vkguide.dev/docs/gpudriven/compute_culling/
    frustum: Vec4,
    p00: f32,
    p11: f32,
    position: Vec4,
}

struct ScanBindGroups {
    scan: shader::scan::bind_groups::BindGroup0,
    scan_sums: Option<Box<ScanBindGroups>>,
    add_sums: shader::scan_add::bind_groups::BindGroup0,
}

struct State<'w> {
    surface: wgpu::Surface<'w>,
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

impl<'w> State<'w> {
    async fn new(
        window: &'w Window,
        scene: &LDrawSceneInstanced,
        color_table: &HashMap<u32, LDrawColor>,
    ) -> Self {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        let surface = instance.create_surface(window).unwrap();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();
        debug!("{:#?}", adapter.get_info());

        let mut required_features = wgpu::Features::MULTI_DRAW_INDIRECT
            | wgpu::Features::INDIRECT_FIRST_INSTANCE
            | wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES
            | wgpu::Features::POLYGON_MODE_LINE
            | wgpu::Features::FLOAT32_FILTERABLE;

        debug!("{:?}", adapter.features());

        // Indirect count isn't supported on metal, so check first.
        let supports_indirect_count = adapter
            .features()
            .contains(wgpu::Features::MULTI_DRAW_INDIRECT_COUNT);
        if supports_indirect_count {
            required_features |= wgpu::Features::MULTI_DRAW_INDIRECT_COUNT;
        }

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features,
                    required_limits: wgpu::Limits::default(),
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
            desired_maximum_frame_latency: 2,
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

        let translation = vec3(0.0, -0.5, -200.0);
        let rotation_xyz = Vec3::ZERO;
        let camera_data = calculate_camera_data(size, translation, rotation_xyz);

        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("camera buffer"),
            contents: bytemuck::cast_slice(&[shader::model::Camera {
                view_projection: camera_data.view_projection,
                position: camera_data.position,
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
        info!(
            "Load {} parts, {} unique colored parts, and {} unique parts: {:?}",
            render_data.solid.draw_count,
            scene.geometry_world_transforms.len(),
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

        let depth_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            min_filter: wgpu::FilterMode::Nearest,
            mag_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let culling_bind_group0 = shader::culling::bind_groups::BindGroup0::from_bindings(
            &device,
            shader::culling::bind_groups::BindGroupLayout0 {
                camera: camera_culling_buffer.as_entire_buffer_binding(),
                depth_pyramid: &depth_pyramid.all_mips,
                depth_sampler: &depth_sampler,
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
                position: camera_data.position,
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

            let depth_sampler = self.device.create_sampler(&wgpu::SamplerDescriptor {
                min_filter: wgpu::FilterMode::Nearest,
                mag_filter: wgpu::FilterMode::Nearest,
                mipmap_filter: wgpu::FilterMode::Nearest,
                ..Default::default()
            });

            // The textures were updated, so use views pointing to the new textures.
            self.culling_bind_group0 = shader::culling::bind_groups::BindGroup0::from_bindings(
                &self.device,
                shader::culling::bind_groups::BindGroupLayout0 {
                    camera: self.camera_culling_buffer.as_entire_buffer_binding(),
                    depth_pyramid: &self.depth_pyramid.all_mips,
                    depth_sampler: &depth_sampler,
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

        // TODO: Draw transparent twice with front faces and then back faces culled?
        // TODO: Fix high contrast studs (manually add stud files to ldr_tools)
        // TODO: Port right click pan from ssbh_wgpu
        // Draw everything that was visible last frame.
        self.model_pass(&mut encoder, &output_view, true);

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
        self.model_pass(&mut encoder, &output_view, false);

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();
        Ok(())
    }

    fn model_pass(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        output_view: &wgpu::TextureView,
        first_pass: bool,
    ) {
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some(if first_pass {
                "Visible Pass"
            } else {
                "Previously Visible Pass"
            }),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.output_view_msaa,
                resolve_target: Some(output_view),
                ops: wgpu::Operations {
                    load: if first_pass {
                        wgpu::LoadOp::Clear(wgpu::Color::BLACK)
                    } else {
                        wgpu::LoadOp::Load
                    },
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.depth_view,
                depth_ops: Some(if first_pass {
                    depth_op_reversed()
                } else {
                    wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    }
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        shader::model::set_bind_groups(&mut render_pass, &self.bind_group0);

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
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(&self.culling_pipeline);
        shader::culling::set_bind_groups(
            &mut compute_pass,
            &self.culling_bind_group0,
            &self.culling_bind_group1,
        );

        // Assume the workgroup is 1D.
        let [size_x, _, _] = shader::culling::compute::MAIN_WORKGROUP_SIZE;
        let count = div_round_up(self.render_data.solid.draw_count, size_x);
        compute_pass.dispatch_workgroups(count, 1, 1);
    }

    fn set_visibility_pass(&self, encoder: &mut wgpu::CommandEncoder, newly_visible: bool) {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Set Visibility Pass"),
            timestamp_writes: None,
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
        shader::visibility::set_bind_groups(compute_pass, bind_group0);

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
        shader::scan_add::set_bind_groups(compute_pass, bind_group0);

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
        shader::scan::set_bind_groups(compute_pass, bind_group0);

        // Assume the workgroup is 1D and processes 2 elements per thread.
        let [size_x, _, _] = shader::scan::compute::MAIN_WORKGROUP_SIZE;
        let count = div_round_up(self.render_data.solid.draw_count, size_x * 2);
        compute_pass.dispatch_workgroups(count, 1, 1);
    }

    fn depth_pyramid_pass(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Depth Pyramid Pass"),
            timestamp_writes: None,
        });

        // Copy the base level.
        compute_pass.set_pipeline(&self.blit_depth_pipeline);
        shader::blit_depth::set_bind_groups(&mut compute_pass, &self.depth_pyramid.base_bind_group);

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

            shader::depth_pyramid::set_bind_groups(&mut compute_pass, bind_group0);

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
            WindowEvent::KeyboardInput { .. } => {}
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
                } else if self.input_state.is_mouse_right_clicked {
                    let delta_x = position.x - self.input_state.previous_cursor_position.x;
                    let delta_y = position.y - self.input_state.previous_cursor_position.y;

                    // Translate an equivalent distance in screen space based on the camera.
                    // The viewport height and vertical field of view define the conversion.
                    let fac = FOV_Y.sin() * self.translation.z.abs() / self.size.height as f32;

                    // Negate y so that dragging up "drags" the model up.
                    self.translation.x += delta_x as f32 * fac;
                    self.translation.y -= delta_y as f32 * fac;
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

fn create_depth_pyramid(
    device: &wgpu::Device,
    window_size: winit::dpi::PhysicalSize<u32>,
    base_depth_view: &wgpu::TextureView,
) -> DepthPyramid {
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

    let projection = glam::Mat4::perspective_infinite_reverse_rh(FOV_Y, aspect, Z_NEAR);

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

    let position = view.inverse().col(3);

    CameraData {
        view,
        view_projection,
        frustum,
        p00,
        p11,
        position,
    }
}

fn main() {
    // Ignore most wgpu logs to avoid flooding the console.
    simple_logger::SimpleLogger::new()
        .with_level(log::LevelFilter::Warn)
        .with_module_level("ldr_wgpu", log::LevelFilter::Info)
        .init()
        .unwrap();

    let args: Vec<_> = std::env::args().collect();
    let ldraw_path = &args[1];
    let path = &args[2];

    // Weld vertices to take advantage of vertex caching/batching on the GPU.
    let start = std::time::Instant::now();
    let settings = GeometrySettings {
        triangulate: true,
        weld_vertices: true,
        stud_type: StudType::HighContrast,
        ..Default::default()
    };
    let scene = ldr_tools::load_file_instanced(path, ldraw_path, &[], &settings);
    info!("Load scene: {:?}", start.elapsed());

    let color_table = ldr_tools::load_color_table(ldraw_path);

    let event_loop = EventLoop::new().unwrap();
    let window = WindowBuilder::new()
        .with_title(concat!("ldr_wgpu ", env!("CARGO_PKG_VERSION")))
        .build(&event_loop)
        .unwrap();

    let mut state = block_on(State::new(&window, &scene, &color_table));
    event_loop
        .run(|event, target| match event {
            Event::WindowEvent {
                ref event,
                window_id,
            } if window_id == window.id() => match event {
                WindowEvent::CloseRequested => target.exit(),
                WindowEvent::Resized(physical_size) => {
                    state.resize(*physical_size);
                    state.update_camera(*physical_size);
                    window.request_redraw();
                }
                WindowEvent::ScaleFactorChanged { .. } => {}
                WindowEvent::RedrawRequested => {
                    match state.render() {
                        Ok(_) => {}
                        Err(wgpu::SurfaceError::Lost) => state.resize(state.size),
                        Err(wgpu::SurfaceError::OutOfMemory) => target.exit(),
                        Err(e) => error!("{e:?}"),
                    }
                    window.request_redraw();
                }
                _ => {
                    state.handle_input(event);
                    state.update_camera(window.inner_size());
                    window.request_redraw();
                }
            },
            _ => (),
        })
        .unwrap();
}
