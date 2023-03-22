use std::collections::HashMap;

use futures::executor::block_on;
use glam::{vec3, vec4, Mat4, Vec3, Vec4};
use ldr_tools::{GeometrySettings, LDrawColor, LDrawSceneInstanced};
use wgpu::util::DeviceExt;
use winit::{
    dpi::PhysicalPosition,
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

mod culling;
mod shader;

const Z_NEAR: f32 = 0.1;
const Z_FAR: f32 = 1000.0;

struct CameraData {
    mvp_matrix: Mat4,
    model_view_matrix: Mat4,
    // https://vkguide.dev/docs/gpudriven/compute_culling/
    frustum: Vec4,
}

// wgpu already provides this type.
// Make our own so we can derive bytemuck.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DrawIndirect {
    pub vertex_count: u32,
    pub instance_count: u32,
    pub base_vertex: u32,
    pub base_instance: u32,
}

/// Combined data for every part in the scene.
/// Renderable with a single multidraw indirect call.
struct IndirectSceneData {
    vertex_buffer: wgpu::Buffer,
    instance_transforms_buffer: wgpu::Buffer,
    instance_bounds_buffer: wgpu::Buffer,
    indirect_buffer: wgpu::Buffer,
    draw_count: u32,
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

    // Render State
    bind_group0: crate::shader::bind_groups::BindGroup0,
    pipeline: wgpu::RenderPipeline,

    camera_culling_buffer: wgpu::Buffer,
    culling_bind_group0: crate::culling::bind_groups::BindGroup0,
    culling_pipeline: wgpu::ComputePipeline,

    render_data: IndirectSceneData,

    input_state: InputState,
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

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    features: wgpu::Features::MULTI_DRAW_INDIRECT
                        | wgpu::Features::INDIRECT_FIRST_INSTANCE,
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

        let pipeline = create_pipeline(&device, surface_format);
        let culling_pipeline = create_culling_pipeline(&device);

        let translation = vec3(0.0, -0.5, -20.0);
        let rotation_xyz = Vec3::ZERO;
        let camera_data = calculate_camera_data(size, translation, rotation_xyz);

        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("camera buffer"),
            contents: bytemuck::cast_slice(&[crate::shader::Camera {
                mvp_matrix: camera_data.mvp_matrix,
            }]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group0 = crate::shader::bind_groups::BindGroup0::from_bindings(
            &device,
            crate::shader::bind_groups::BindGroupLayout0 {
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

        // TODO: just use encase for this to avoid manually handling padding?
        let camera_culling_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("camera culling buffer"),
            contents: bytemuck::cast_slice(&[crate::culling::Camera {
                z_near: Z_NEAR,
                z_far: Z_FAR,
                _pad1: 0.0,
                _pad2: 0.0,
                frustum: camera_data.frustum,
                model_view_matrix: camera_data.model_view_matrix,
            }]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let culling_bind_group0 = crate::culling::bind_groups::BindGroup0::from_bindings(
            &device,
            crate::culling::bind_groups::BindGroupLayout0 {
                draws: render_data.indirect_buffer.as_entire_buffer_binding(),
                bounding_spheres: render_data
                    .instance_bounds_buffer
                    .as_entire_buffer_binding(),
                camera: camera_culling_buffer.as_entire_buffer_binding(),
            },
        );

        let (depth_texture, depth_view) = create_depth_texture(&device, size);

        Self {
            surface,
            device,
            queue,
            size,
            config,
            pipeline,
            culling_pipeline,
            culling_bind_group0,
            bind_group0,
            render_data,
            translation,
            rotation_xyz,
            camera_buffer,
            depth_texture,
            depth_view,
            camera_culling_buffer,
            input_state: Default::default(),
        }
    }

    fn update_camera(&self, size: winit::dpi::PhysicalSize<u32>) {
        let camera_data = calculate_camera_data(size, self.translation, self.rotation_xyz);
        self.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&[crate::shader::Camera {
                mvp_matrix: camera_data.mvp_matrix,
            }]),
        );
        self.queue.write_buffer(
            &self.camera_culling_buffer,
            0,
            bytemuck::cast_slice(&[crate::culling::Camera {
                z_near: Z_NEAR,
                z_far: Z_FAR,
                _pad1: 0.0,
                _pad2: 0.0,
                frustum: camera_data.frustum,
                model_view_matrix: camera_data.model_view_matrix,
            }]),
        );
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);

            let (depth_texture, depth_view) = create_depth_texture(&self.device, new_size);
            self.depth_texture = depth_texture;
            self.depth_view = depth_view;
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

        self.culling_compute_pass(&mut encoder);
        self.model_pass(&mut encoder, output_view);

        self.queue.submit(std::iter::once(encoder.finish()));

        // Actually draw the frame.
        output.present();

        Ok(())
    }

    fn model_pass(&mut self, encoder: &mut wgpu::CommandEncoder, output_view: wgpu::TextureView) {
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Model Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &output_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: true,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: true,
                }),
                stencil_ops: None,
            }),
        });

        render_pass.set_pipeline(&self.pipeline);

        crate::shader::bind_groups::set_bind_groups(
            &mut render_pass,
            crate::shader::bind_groups::BindGroups {
                bind_group0: &self.bind_group0,
            },
        );

        // Draw the instances of each unique part and color.
        // This allows reusing most of the rendering state for better performance.
        render_pass.set_vertex_buffer(0, self.render_data.vertex_buffer.slice(..));
        render_pass.set_vertex_buffer(1, self.render_data.instance_transforms_buffer.slice(..));

        // Draw each instance with a different transform.
        render_pass.multi_draw_indirect(
            &self.render_data.indirect_buffer,
            0,
            self.render_data.draw_count,
        );
    }

    fn culling_compute_pass(&mut self, encoder: &mut wgpu::CommandEncoder) {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Culling Pass"),
        });

        compute_pass.set_pipeline(&self.culling_pipeline);
        crate::culling::bind_groups::set_bind_groups(
            &mut compute_pass,
            crate::culling::bind_groups::BindGroups {
                bind_group0: &self.culling_bind_group0,
            },
        );

        // Assume the workgroup is 1D.
        let [size_x, _, _] = crate::culling::compute::MAIN_WORKGROUP_SIZE;
        let count = div_round_up(self.render_data.draw_count, size_x);
        compute_pass.dispatch_workgroups(count, 1, 1);
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

const fn div_round_up(x: u32, d: u32) -> u32 {
    (x + d - 1) / d
}

fn create_culling_pipeline(device: &wgpu::Device) -> wgpu::ComputePipeline {
    let shader = crate::culling::create_shader_module(device);
    let render_pipeline_layout = crate::culling::create_pipeline_layout(device);

    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Render Pipeline"),
        layout: Some(&render_pipeline_layout),
        module: &shader,
        entry_point: "main",
    })
}

fn create_pipeline(
    device: &wgpu::Device,
    surface_format: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    let shader = crate::shader::create_shader_module(device);
    let render_pipeline_layout = crate::shader::create_pipeline_layout(device);

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Render Pipeline"),
        layout: Some(&render_pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: "vs_main",
            buffers: &[
                crate::shader::VertexInput::vertex_buffer_layout(wgpu::VertexStepMode::Vertex),
                crate::shader::InstanceInput::vertex_buffer_layout(wgpu::VertexStepMode::Instance),
            ],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: "fs_main",
            targets: &[Some(surface_format.into())],
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::LessEqual,
            stencil: Default::default(),
            bias: Default::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
    })
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
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });

    let depth_view = depth_texture.create_view(&Default::default());

    (depth_texture, depth_view)
}

fn load_render_data(
    device: &wgpu::Device,
    scene: &LDrawSceneInstanced,
    color_table: &HashMap<u32, LDrawColor>,
) -> IndirectSceneData {
    // Combine all data into a single multidraw indirect call.
    let mut combined_vertices = Vec::new();
    let mut combined_transforms = Vec::new();
    let mut indirect_draws = Vec::new();
    let mut instance_bounds = Vec::new();

    for ((name, color), transforms) in &scene.geometry_world_transforms {
        // Create separate vertex data if a part has multiple colors.
        // This is necessary since we store face colors per vertex.
        let geometry = &scene.geometry_cache[name];

        let base_vertex = combined_vertices.len() as u32;
        append_vertices(&mut combined_vertices, geometry, *color, color_table);
        let vertex_count = combined_vertices.len() as u32 - base_vertex;

        // Each draw specifies the part mesh using an offset and count.
        // The base instance steps through the transforms buffer.
        // Each draw uses a single instance to allow culling individual draws.
        for transform in transforms {
            let draw = DrawIndirect {
                vertex_count,
                instance_count: 1,
                base_instance: combined_transforms.len() as u32,
                base_vertex,
            };
            indirect_draws.push(draw);

            // TODO: Find an efficient way to potentially update this each frame.
            // TODO: Create a struct for the bounding data.
            let points_world: Vec<_> = geometry
                .vertices
                .iter()
                .map(|v| transform.transform_point3(*v))
                .collect();
            let sphere_center =
                points_world.iter().sum::<Vec3>() / points_world.len().max(1) as f32;
            let sphere_radius = points_world
                .iter()
                .map(|v| v.distance(sphere_center))
                .reduce(f32::max)
                .unwrap_or_default();
            instance_bounds.push(sphere_center.extend(sphere_radius));

            combined_transforms.push(*transform);
        }
    }

    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("vertex buffer"),
        contents: bytemuck::cast_slice(&combined_vertices),
        usage: wgpu::BufferUsages::VERTEX,
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
        instance_transforms_buffer,
        instance_bounds_buffer,
        indirect_buffer,
        draw_count,
    }
}

fn append_vertices(
    vertices: &mut Vec<shader::VertexInput>,
    geometry: &ldr_tools::LDrawGeometry,
    color_code: u32,
    color_table: &HashMap<u32, LDrawColor>,
) {
    // TODO: wgsl_to_wgpu should support non strict offset checking for vertex buffer structs.
    // This allows desktop applications to use vec3 for positions.
    for (i, v) in geometry.vertices.iter().enumerate() {
        // Assume faces are already triangulated and not welded.
        // This means every 3 vertices defines a new face.
        // TODO: missing color codes?
        // TODO: publicly expose color handling logic in ldr_tools.
        // TODO: handle the case where the face color list is empty?
        let face_color = geometry
            .face_colors
            .get(i / 3)
            .unwrap_or(&geometry.face_colors[0]);
        let replaced_color = if face_color.color == 16 {
            color_code
        } else {
            face_color.color
        };

        let color = color_table
            .get(&replaced_color)
            .map(|c| Vec4::from(c.rgba_linear))
            .unwrap_or(vec4(1.0, 0.0, 1.0, 1.0));

        let new_vertex = crate::shader::VertexInput {
            position: vec4(v.x, v.y, v.z, 1.0),
            color,
        };
        vertices.push(new_vertex);
    }
}

fn calculate_camera_data(
    size: winit::dpi::PhysicalSize<u32>,
    translation: glam::Vec3,
    rotation: glam::Vec3,
) -> CameraData {
    let aspect = size.width as f32 / size.height as f32;

    // wgpu and LDraw have different coordinate systems.
    let axis_correction = Mat4::from_rotation_x(180.0f32.to_radians());

    let model_view_matrix = glam::Mat4::from_translation(translation)
        * glam::Mat4::from_rotation_x(rotation.x)
        * glam::Mat4::from_rotation_y(rotation.y)
        * axis_correction;
    let perspective_matrix = glam::Mat4::perspective_rh(0.5, aspect, Z_NEAR, Z_FAR);

    let mvp_matrix = perspective_matrix * model_view_matrix;

    // Calculate camera frustum data for culling.
    // https://github.com/zeux/niagara/blob/3fafe000ba8fe6e309b41e915b81242b4ca3db28/src/niagara.cpp#L836-L852
    let perspective_t = perspective_matrix.transpose();
    // x + w < 0
    let frustum_x = (perspective_t.col(3) + perspective_t.col(0)).normalize();
    // y + w < 0
    let frustum_y = (perspective_t.col(3) + perspective_t.col(1)).normalize();
    let frustum = vec4(frustum_x.x, frustum_x.z, frustum_y.y, frustum_y.z);

    CameraData {
        mvp_matrix,
        frustum,
        model_view_matrix,
    }
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    let ldraw_path = &args[1];
    let path = &args[2];

    let start = std::time::Instant::now();
    let settings = GeometrySettings {
        triangulate: true,
        weld_vertices: false,
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
