use std::{collections::HashMap, num::NonZeroU32};

use futures::executor::block_on;
use glam::{vec3, vec4, Mat4, Vec3, Vec4};
use ldr_tools::{GeometrySettings, LDrawColor, LDrawSceneInstanced, PrimitiveResolution, StudType};
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

// The far plane can be infinity since we use reversed-z.
const Z_NEAR: f32 = 0.1;
const Z_FAR: f32 = f32::INFINITY;

fn depth_stencil_reversed() -> wgpu::DepthStencilState {
    wgpu::DepthStencilState {
        // Reversed-z
        format: wgpu::TextureFormat::Depth32Float,
        depth_write_enabled: true,
        depth_compare: wgpu::CompareFunction::Greater,
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
    draw_count: u32,
    // TODO: Duplicate this data for edges as well.
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    indirect_buffer: wgpu::Buffer,
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

    depth_texture: wgpu::Texture,
    depth_view: wgpu::TextureView,

    // Store the texture separately since depth attachments can't have mipmaps.
    depth_pyramid_pipeline: wgpu::ComputePipeline,
    blit_depth_pipeline: wgpu::ComputePipeline,
    depth_pyramid: DepthPyramid,
    depth_pyramid_sampler: wgpu::Sampler,

    // Render State
    bind_group0: shader::model::bind_groups::BindGroup0,
    model_pipeline: wgpu::RenderPipeline,
    occluder_pipeline: wgpu::RenderPipeline,

    camera_culling_buffer: wgpu::Buffer,
    culling_bind_group0: shader::culling::bind_groups::BindGroup0,
    culling_bind_group1: shader::culling::bind_groups::BindGroup1,
    culling_bind_group1_occluder: shader::culling::bind_groups::BindGroup1,
    frustum_culling_pipeline: wgpu::ComputePipeline,
    occlusion_culling_pipeline: wgpu::ComputePipeline,

    render_data: IndirectSceneData,
    // TODO: Avoid overlap between these two versions of the scene.
    render_data_occluder: IndirectSceneData,

    input_state: InputState,
}

struct DepthPyramid {
    pyramid_view: wgpu::TextureView,
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
        scene_occluder: &LDrawSceneInstanced,
        color_table: &HashMap<u32, LDrawColor>,
    ) -> Self {
        // TODO: Investigate why DX12 is so slow.
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN | wgpu::Backends::METAL,
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

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    features: wgpu::Features::MULTI_DRAW_INDIRECT
                        | wgpu::Features::INDIRECT_FIRST_INSTANCE
                        | wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES,
                    limits: wgpu::Limits::default(),
                },
                None,
            )
            .await
            .unwrap();

        let size = window.inner_size();
        let surface_format = wgpu::TextureFormat::Bgra8UnormSrgb;
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: Vec::new(),
        };
        surface.configure(&device, &config);

        let model_pipeline = create_pipeline(&device, surface_format);
        let occluder_pipeline = create_occluder_pipeline(&device);

        let frustum_culling_pipeline = create_culling_pipeline(&device, "frustum_main");
        let occlusion_culling_pipeline = create_culling_pipeline(&device, "occlusion_main");

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
            render_data.draw_count,
            scene.geometry_cache.len(),
            start.elapsed()
        );

        // TODO: Avoid duplicating culling information?
        // TODO: Don't include color information since this is only used for depth.
        let start = std::time::Instant::now();
        let render_data_occluder = load_render_data(&device, scene_occluder, color_table);
        println!(
            "Load {} parts and {} unique parts: {:?}",
            render_data.draw_count,
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

        let (depth_texture, depth_view) = create_depth_texture(&device, size);

        let depth_pyramid_pipeline = create_depth_pyramid_pipeline(&device);
        let blit_depth_pipeline = create_blit_depth_pipeline(&device);

        let depth_pyramid = create_depth_pyramid(&device, size, &depth_view);

        let depth_pyramid_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("depth pyramid sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let culling_bind_group0 = shader::culling::bind_groups::BindGroup0::from_bindings(
            &device,
            shader::culling::bind_groups::BindGroupLayout0 {
                camera: camera_culling_buffer.as_entire_buffer_binding(),
                depth_pyramid: &depth_pyramid.pyramid_view,
                depth_pyramid_sampler: &depth_pyramid_sampler,
            },
        );

        let culling_bind_group1 = shader::culling::bind_groups::BindGroup1::from_bindings(
            &device,
            shader::culling::bind_groups::BindGroupLayout1 {
                draws: render_data.indirect_buffer.as_entire_buffer_binding(),
                instance_bounds: render_data
                    .instance_bounds_buffer
                    .as_entire_buffer_binding(),
            },
        );

        let culling_bind_group1_occluder = shader::culling::bind_groups::BindGroup1::from_bindings(
            &device,
            shader::culling::bind_groups::BindGroupLayout1 {
                draws: render_data_occluder
                    .indirect_buffer
                    .as_entire_buffer_binding(),
                instance_bounds: render_data_occluder
                    .instance_bounds_buffer
                    .as_entire_buffer_binding(),
            },
        );

        Self {
            surface,
            device,
            queue,
            size,
            config,
            model_pipeline,
            occluder_pipeline,
            frustum_culling_pipeline,
            occlusion_culling_pipeline,
            culling_bind_group0,
            culling_bind_group1,
            culling_bind_group1_occluder,
            bind_group0,
            render_data,
            render_data_occluder,
            translation,
            rotation_xyz,
            camera_buffer,
            depth_texture,
            depth_view,
            camera_culling_buffer,
            depth_pyramid,
            depth_pyramid_pipeline,
            blit_depth_pipeline,
            depth_pyramid_sampler,
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

            let (depth_texture, depth_view) = create_depth_texture(&self.device, new_size);
            self.depth_texture = depth_texture;
            self.depth_view = depth_view;

            self.depth_pyramid = create_depth_pyramid(&self.device, new_size, &self.depth_view);

            // The textures were updated, so use views pointing to the new textures.
            self.culling_bind_group0 = shader::culling::bind_groups::BindGroup0::from_bindings(
                &self.device,
                shader::culling::bind_groups::BindGroupLayout0 {
                    camera: self.camera_culling_buffer.as_entire_buffer_binding(),
                    depth_pyramid: &self.depth_pyramid.pyramid_view,
                    depth_pyramid_sampler: &self.depth_pyramid_sampler,
                },
            );
        }
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let output_view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        // Apply cheap and accurate culling methods first.
        self.frustum_culling_pass(&mut encoder, &self.culling_bind_group1);
        self.frustum_culling_pass(&mut encoder, &self.culling_bind_group1_occluder);

        // Use a lower detailed scene for generating the occlusion depth map.
        self.occluder_pass(&mut encoder);
        self.depth_pyramid_compute_pass(&mut encoder);

        // Occlude the high detailed draws using the occlusion depth map.
        // This saves time if low detail + high detail occluded <  high detail.
        self.occlusion_culling_pass(&mut encoder);

        self.model_pass(&mut encoder, &output_view);

        self.queue.submit(std::iter::once(encoder.finish()));

        // Actually draw the frame.
        output.present();

        Ok(())
    }

    fn occluder_pass(&mut self, encoder: &mut wgpu::CommandEncoder) {
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Occluder Pass"),
            color_attachments: &[],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.depth_view,
                depth_ops: Some(depth_op_reversed()),
                stencil_ops: None,
            }),
        });

        render_pass.set_pipeline(&self.occluder_pipeline);

        shader::model::bind_groups::set_bind_groups(
            &mut render_pass,
            shader::model::bind_groups::BindGroups {
                bind_group0: &self.bind_group0,
            },
        );

        // TODO: make this a method?
        draw_indirect(&mut render_pass, &self.render_data_occluder);
    }

    fn model_pass(&mut self, encoder: &mut wgpu::CommandEncoder, output_view: &wgpu::TextureView) {
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Model Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: output_view,
                resolve_target: None,
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

        render_pass.set_pipeline(&self.model_pipeline);

        shader::model::bind_groups::set_bind_groups(
            &mut render_pass,
            shader::model::bind_groups::BindGroups {
                bind_group0: &self.bind_group0,
            },
        );

        // TODO: make this a method?
        draw_indirect(&mut render_pass, &self.render_data);
    }

    fn frustum_culling_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bind_group1: &shader::culling::bind_groups::BindGroup1,
    ) {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Frustum Culling Pass"),
        });

        compute_pass.set_pipeline(&self.frustum_culling_pipeline);
        shader::culling::bind_groups::set_bind_groups(
            &mut compute_pass,
            shader::culling::bind_groups::BindGroups {
                bind_group0: &self.culling_bind_group0,
                bind_group1,
            },
        );

        // Assume the workgroup is 1D.
        let [size_x, _, _] = shader::culling::compute::FRUSTUM_MAIN_WORKGROUP_SIZE;
        let count = div_round_up(self.render_data.draw_count, size_x);
        compute_pass.dispatch_workgroups(count, 1, 1);
    }

    fn occlusion_culling_pass(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Occlusion Culling Pass"),
        });

        compute_pass.set_pipeline(&self.occlusion_culling_pipeline);
        shader::culling::bind_groups::set_bind_groups(
            &mut compute_pass,
            shader::culling::bind_groups::BindGroups {
                bind_group0: &self.culling_bind_group0,
                bind_group1: &self.culling_bind_group1,
            },
        );

        // Assume the workgroup is 1D.
        let [size_x, _, _] = shader::culling::compute::OCCLUSION_MAIN_WORKGROUP_SIZE;
        let count = div_round_up(self.render_data.draw_count, size_x);
        compute_pass.dispatch_workgroups(count, 1, 1);
    }

    fn depth_pyramid_compute_pass(&self, encoder: &mut wgpu::CommandEncoder) {
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
        let count_x = div_round_up(self.size.width, size_x);
        let count_y = div_round_up(self.size.height, size_y);

        compute_pass.dispatch_workgroups(count_x, count_y, 1);

        // Make the depth pyramid for the next frame using the current depth.
        // Each dispatch generates one mip level of the pyramid.
        compute_pass.set_pipeline(&self.depth_pyramid_pipeline);
        for (i, bind_group0) in self.depth_pyramid.mip_bind_groups.iter().enumerate() {
            // The first level is copied separately from the depth texture.
            let mip = i + 1;
            let mip_width = (self.size.width >> mip).max(1);
            let mip_height = (self.size.height >> mip).max(1);

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

fn draw_indirect<'a>(render_pass: &mut wgpu::RenderPass<'a>, indirect_data: &'a IndirectSceneData) {
    // Draw the instances of each unique part and color.
    // This allows reusing most of the rendering state for better performance.
    render_pass.set_index_buffer(
        indirect_data.index_buffer.slice(..),
        wgpu::IndexFormat::Uint32,
    );
    render_pass.set_vertex_buffer(0, indirect_data.vertex_buffer.slice(..));
    render_pass.set_vertex_buffer(1, indirect_data.instance_transforms_buffer.slice(..));

    // Draw each instance with a different transform.
    // TODO: Use an indirect buffer that's always visible or just frustum culled.
    render_pass.multi_draw_indexed_indirect(
        &indirect_data.indirect_buffer,
        0,
        indirect_data.draw_count,
    );
}

fn create_depth_pyramid(
    device: &wgpu::Device,
    size: winit::dpi::PhysicalSize<u32>,
    base_depth: &wgpu::TextureView,
) -> DepthPyramid {
    let (pyramid, pyramid_mips) = create_depth_pyramid_texture(device, size);
    let pyramid_bind_groups = depth_pyramid_bind_groups(device, &pyramid_mips);
    let base_bind_group = shader::blit_depth::bind_groups::BindGroup0::from_bindings(
        device,
        shader::blit_depth::bind_groups::BindGroupLayout0 {
            input: base_depth,
            output: &pyramid_mips[0],
        },
    );

    let pyramid_view = pyramid.create_view(&wgpu::TextureViewDescriptor::default());

    DepthPyramid {
        pyramid_view,
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

fn create_depth_texture(
    device: &wgpu::Device,
    size: winit::dpi::PhysicalSize<u32>,
) -> (wgpu::Texture, wgpu::TextureView) {
    let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("depth texture"),
        size: wgpu::Extent3d {
            width: size.width,
            height: size.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });

    let depth_view = depth_texture.create_view(&Default::default());

    (depth_texture, depth_view)
}

fn create_depth_pyramid_texture(
    device: &wgpu::Device,
    size: winit::dpi::PhysicalSize<u32>,
) -> (wgpu::Texture, Vec<wgpu::TextureView>) {
    let size = wgpu::Extent3d {
        width: size.width,
        height: size.height,
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

    for ((name, color), transforms) in &scene.geometry_world_transforms {
        // Create separate vertex data if a part has multiple colors.
        // This is necessary since we store face colors per vertex.
        let geometry = &scene.geometry_cache[name];

        let base_index = combined_indices.len() as u32;
        let vertex_offset = combined_vertices.len() as i32;

        append_geometry(
            &mut combined_vertices,
            &mut combined_indices,
            geometry,
            *color,
            color_table,
        );

        // Each draw specifies the part mesh using an offset and count.
        // The base instance steps through the transforms buffer.
        // Each draw uses a single instance to allow culling individual draws.
        for transform in transforms {
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

    let indirect_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("indirect buffer"),
        contents: bytemuck::cast_slice(&indirect_draws),
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

    let draw_count = indirect_draws.len() as u32;

    IndirectSceneData {
        vertex_buffer,
        index_buffer,
        instance_transforms_buffer,
        instance_bounds_buffer,
        indirect_buffer,
        draw_count,
    }
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
    geometry: &ldr_tools::LDrawGeometry,
    color_code: u32,
    color_table: &HashMap<u32, LDrawColor>,
) {
    // TODO: missing color codes?
    // TODO: publicly expose color handling logic in ldr_tools.
    // TODO: handle the case where the face color list is empty?
    match geometry.face_colors.as_slice() {
        [face_color] => {
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
        }
        face_colors => {
            // Assume faces are already triangulated.
            // Make each vertex unique to convert per face to vertex coloring.
            // This means every 3 vertices defines a new face.
            for (i, vertex_index) in geometry.vertex_indices.iter().enumerate() {
                let face_color = face_colors.get(i / 3).unwrap_or(&face_colors[0]);
                let color = rgba_color(face_color, color_code, color_table);

                let new_vertex = shader::model::VertexInput {
                    position: geometry.vertices[*vertex_index as usize],
                    color,
                };
                vertices.push(new_vertex);
                vertex_indices.push(i as u32);
            }
        }
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

    // Use the lowest possible quality for the occluder pass.
    // The occluder pass itself isn't occlusion culled,
    // so reducing the vertex count is critical for good performance.
    // Weld vertices to take advantage of vertex caching on the GPU.
    // https://www.khronos.org/opengl/wiki/Post_Transform_Cache
    let start = std::time::Instant::now();
    let settings = GeometrySettings {
        triangulate: true,
        weld_vertices: true,
        stud_type: StudType::Disabled,
        primitive_resolution: PrimitiveResolution::Low,
        ..Default::default()
    };
    // TODO: Don't include any transparent parts since they shouldn't occlude anything.
    // TODO: Ignore stickers and replace patterned with non patterned versions.
    let scene_occluder = ldr_tools::load_file_instanced(path, ldraw_path, &settings);
    println!("Load scene occluder: {:?}", start.elapsed());

    let start = std::time::Instant::now();
    let settings = GeometrySettings {
        triangulate: true,
        weld_vertices: true,
        ..Default::default()
    };
    let scene = ldr_tools::load_file_instanced(path, ldraw_path, &settings);
    println!("Load scene: {:?}", start.elapsed());

    let color_table = ldr_tools::load_color_table(ldraw_path);

    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title(concat!("ldr_wgpu ", env!("CARGO_PKG_VERSION")))
        // .with_inner_size(winit::dpi::PhysicalSize { width: 1024.0, height: 1024.0 })
        .build(&event_loop)
        .unwrap();

    let mut state = block_on(State::new(&window, &scene, &scene_occluder, &color_table));
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
