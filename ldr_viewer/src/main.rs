use futures::executor::block_on;
use glam::{vec3, Vec3};
use ldr_wgpu::{calculate_camera_data, Renderer, Scene, FOV_Y};
use log::{debug, error};
use winit::{
    dpi::PhysicalPosition,
    event::*,
    event_loop::EventLoop,
    window::{Window, WindowBuilder},
};

struct State<'a> {
    surface: wgpu::Surface<'a>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,

    renderer: Renderer,
    scene: Scene,
}

#[derive(Default)]
struct InputState {
    translation: Vec3,
    rotation_xyz: Vec3,
    is_mouse_left_clicked: bool,
    is_mouse_right_clicked: bool,
    previous_cursor_position: PhysicalPosition<f64>,
}

impl<'a> State<'a> {
    async fn new(
        window: &'a Window,
        format: wgpu::TextureFormat,
        path: &str,
        ldraw_path: &str,
    ) -> Self {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
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

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features: ldr_wgpu::REQUIRED_FEATURES,
                required_limits: wgpu::Limits {
                    max_binding_array_elements_per_shader_stage: 4,
                    ..Default::default()
                },
                ..Default::default()
            })
            .await
            .unwrap();

        let size = window.inner_size();
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: Vec::new(),
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let example = Renderer::new(&device, size.width, size.height, config.format, ldraw_path);
        let scene = Scene::new(&device, &queue, path, ldraw_path);

        Self {
            surface,
            device,
            queue,
            config,
            renderer: example,
            scene,
        }
    }
}

impl InputState {
    pub fn handle_input(&mut self, event: &WindowEvent, size: winit::dpi::PhysicalSize<u32>) {
        match event {
            WindowEvent::KeyboardInput { .. } => {}
            WindowEvent::MouseInput { button, state, .. } => {
                // Track mouse clicks to only rotate when dragging while clicked.
                match (button, state) {
                    (MouseButton::Left, ElementState::Pressed) => self.is_mouse_left_clicked = true,
                    (MouseButton::Left, ElementState::Released) => {
                        self.is_mouse_left_clicked = false
                    }
                    (MouseButton::Right, ElementState::Pressed) => {
                        self.is_mouse_right_clicked = true
                    }
                    (MouseButton::Right, ElementState::Released) => {
                        self.is_mouse_right_clicked = false
                    }
                    _ => (),
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                if self.is_mouse_left_clicked {
                    let delta_x = position.x - self.previous_cursor_position.x;
                    let delta_y = position.y - self.previous_cursor_position.y;

                    // Swap XY so that dragging left/right rotates left/right.
                    self.rotation_xyz.x += (delta_y * 0.01) as f32;
                    self.rotation_xyz.y += (delta_x * 0.01) as f32;
                } else if self.is_mouse_right_clicked {
                    let delta_x = position.x - self.previous_cursor_position.x;
                    let delta_y = position.y - self.previous_cursor_position.y;

                    // Translate an equivalent distance in screen space based on the camera.
                    // The viewport height and vertical field of view define the conversion.
                    let fac = FOV_Y.sin() * self.translation.z.abs() / size.height as f32;

                    // Negate y so that dragging up "drags" the model up.
                    self.translation.x += delta_x as f32 * fac;
                    self.translation.y -= delta_y as f32 * fac;
                }
                // Always update the position to avoid jumps when moving between clicks.
                self.previous_cursor_position = *position;
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

fn main() {
    let args: Vec<_> = std::env::args().collect();

    let ldraw_path = &args[1];
    let path = &args[2];

    // Ignore most wgpu logs to avoid flooding the console.
    simple_logger::SimpleLogger::new()
        .with_level(log::LevelFilter::Warn)
        .with_module_level("ldr_wgpu", log::LevelFilter::Debug)
        .init()
        .unwrap();

    let event_loop = EventLoop::new().unwrap();
    let window = WindowBuilder::new()
        .with_title(concat!("ldr_wgpu ", env!("CARGO_PKG_VERSION")))
        .build(&event_loop)
        .unwrap();

    // Choose a format that's guaranteed to be supported.
    let format = wgpu::TextureFormat::Bgra8UnormSrgb;

    let mut state = block_on(State::new(&window, format, path, ldraw_path));

    let mut input_state = InputState {
        translation: vec3(0.0, -0.5, -200.0),
        rotation_xyz: Vec3::ZERO,
        ..Default::default()
    };

    let size = window.inner_size();

    let camera_data = calculate_camera_data(
        size.width,
        size.height,
        input_state.translation,
        input_state.rotation_xyz,
    );
    state.renderer.update_camera(&state.queue, camera_data);

    event_loop
        .run(|event, target| match event {
            Event::WindowEvent {
                ref event,
                window_id,
            } if window_id == window.id() => match event {
                WindowEvent::CloseRequested => target.exit(),
                WindowEvent::Resized(size) => {
                    state.config.width = size.width;
                    state.config.height = size.height;
                    state.surface.configure(&state.device, &state.config);

                    let camera_data = calculate_camera_data(
                        size.width,
                        size.height,
                        input_state.translation,
                        input_state.rotation_xyz,
                    );
                    state.renderer.update_camera(&state.queue, camera_data);

                    state
                        .renderer
                        .resize(&state.device, size.width, size.height);

                    window.request_redraw();
                }
                WindowEvent::ScaleFactorChanged { .. } => {}
                WindowEvent::RedrawRequested => {
                    match state.surface.get_current_texture() {
                        Ok(output) => {
                            let output_view = output
                                .texture
                                .create_view(&wgpu::TextureViewDescriptor::default());

                            state.renderer.render(
                                &output_view,
                                &state.device,
                                &state.queue,
                                &state.scene,
                            );

                            output.present();
                        }
                        Err(wgpu::SurfaceError::Lost) => {}
                        Err(wgpu::SurfaceError::OutOfMemory) => target.exit(),
                        Err(e) => error!("{e:?}"),
                    }
                    window.request_redraw();
                }
                _ => {
                    let size = window.inner_size();
                    input_state.handle_input(event, size);

                    let camera_data = calculate_camera_data(
                        size.width,
                        size.height,
                        input_state.translation,
                        input_state.rotation_xyz,
                    );
                    state.renderer.update_camera(&state.queue, camera_data);

                    window.request_redraw();
                }
            },
            _ => (),
        })
        .unwrap();
}
