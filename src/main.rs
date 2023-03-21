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

mod shader;

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

struct InstancedMesh {
    vertex_buffer: wgpu::Buffer,
    instance_transforms_buffer: wgpu::Buffer,
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

    // Render State
    bind_group0: crate::shader::bind_groups::BindGroup0,
    pipeline: wgpu::RenderPipeline,

    instanced_meshes: Vec<InstancedMesh>,

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
                    features: wgpu::Features::default(),
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

        let shader = crate::shader::create_shader_module(&device);
        let render_pipeline_layout = crate::shader::create_pipeline_layout(&device);

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[
                    crate::shader::VertexInput::vertex_buffer_layout(wgpu::VertexStepMode::Vertex),
                    crate::shader::InstanceInput::vertex_buffer_layout(
                        wgpu::VertexStepMode::Instance,
                    ),
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
        });

        let translation = vec3(0.0, -0.5, -20.0);
        let rotation_xyz = Vec3::ZERO;
        let mvp_matrix = mvp_matrix(size, translation, rotation_xyz);

        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("uniforms"),
            contents: bytemuck::cast_slice(&[crate::shader::Camera { mvp_matrix }]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group0 = crate::shader::bind_groups::BindGroup0::from_bindings(
            &device,
            crate::shader::bind_groups::BindGroupLayout0 {
                camera: camera_buffer.as_entire_buffer_binding(),
            },
        );

        let start = std::time::Instant::now();
        let instanced_meshes = load_instances(&device, scene, color_table);
        println!(
            "Load {:?} parts and {:?} unique parts: {:?}",
            instanced_meshes.len(),
            scene.geometry_cache.len(),
            start.elapsed()
        );

        let (depth_texture, depth_view) = create_depth_texture(&device, size);

        Self {
            surface,
            device,
            queue,
            size,
            config,
            pipeline,
            bind_group0,
            instanced_meshes,
            translation,
            rotation_xyz,
            camera_buffer,
            depth_texture,
            depth_view,
            input_state: Default::default(),
        }
    }

    fn update_camera(&self, size: winit::dpi::PhysicalSize<u32>) {
        let mvp_matrix = mvp_matrix(size, self.translation, self.rotation_xyz);
        self.queue
            .write_buffer(&self.camera_buffer, 0, bytemuck::cast_slice(&[mvp_matrix]));
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

        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Render Pass"),
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
        self.bind_group0.set(&mut render_pass);

        // Draw the instances of each unique part and color.
        // This allows reusing most of the rendering state for better performance.
        for instanced_mesh in &self.instanced_meshes {
            render_pass.set_vertex_buffer(0, instanced_mesh.vertex_buffer.slice(..));
            render_pass.set_vertex_buffer(1, instanced_mesh.instance_transforms_buffer.slice(..));
            // render_pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);

            // Draw each instance with a different transform.
            render_pass.draw_indirect(&instanced_mesh.indirect_buffer, 0)
        }

        drop(render_pass);

        self.queue.submit(std::iter::once(encoder.finish()));

        // Actually draw the frame.
        output.present();

        Ok(())
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

fn load_instances(
    device: &wgpu::Device,
    scene: &LDrawSceneInstanced,
    color_table: &HashMap<u32, LDrawColor>,
) -> Vec<InstancedMesh> {
    scene
        .geometry_world_transforms
        .iter()
        .map(|((name, color), transforms)| {
            let geometry = &scene.geometry_cache[name];

            let vertex_buffer = create_vertex_buffer(device, geometry, *color, color_table);
            let instance_transforms_buffer =
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("instance transforms buffer"),
                    contents: bytemuck::cast_slice(&transforms),
                    usage: wgpu::BufferUsages::VERTEX,
                });

            let instance_count = transforms.len() as u32;

            // TODO: multidraw indirect for culling support?
            // TODO: benchmark on metal since it's emulated.
            let indirect_draw = DrawIndirect {
                vertex_count: geometry.vertex_indices.len() as u32,
                instance_count,
                base_instance: 0,
                base_vertex: 0,
            };
            let indirect_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("indirect buffer"),
                contents: bytemuck::cast_slice(&[indirect_draw]),
                usage: wgpu::BufferUsages::INDIRECT,
            });

            InstancedMesh {
                vertex_buffer,
                instance_transforms_buffer,
                indirect_buffer,
            }
        })
        .collect()
}

fn create_vertex_buffer(
    device: &wgpu::Device,
    geometry: &ldr_tools::LDrawGeometry,
    color_code: u32,
    color_table: &HashMap<u32, LDrawColor>,
) -> wgpu::Buffer {
    // TODO: wgsl_to_wgpu should support non strict offset checking for vertex buffer structs.
    // This allows desktop applications to use vec3 for positions.
    let vertices: Vec<_> = geometry
        .vertices
        .iter()
        .enumerate()
        .map(|(i, v)| {
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

            crate::shader::VertexInput {
                position: vec4(v.x, v.y, v.z, 1.0),
                color,
            }
        })
        .collect();

    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("vertex buffer"),
        contents: bytemuck::cast_slice(&vertices),
        usage: wgpu::BufferUsages::VERTEX,
    })
}

fn mvp_matrix(
    size: winit::dpi::PhysicalSize<u32>,
    translation: glam::Vec3,
    rotation: glam::Vec3,
) -> glam::Mat4 {
    let aspect = size.width as f32 / size.height as f32;

    // wgpu and LDraw have different coordinate systems.
    let axis_correction = Mat4::from_rotation_x(180.0f32.to_radians());

    let model_view_matrix = glam::Mat4::from_translation(translation)
        * glam::Mat4::from_rotation_x(rotation.x)
        * glam::Mat4::from_rotation_y(rotation.y)
        * axis_correction;
    let perspective_matrix = glam::Mat4::perspective_rh(0.5, aspect, 0.1, 1000.0);

    perspective_matrix * model_view_matrix
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
