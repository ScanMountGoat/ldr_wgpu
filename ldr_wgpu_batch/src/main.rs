use std::path::Path;

use futures::executor::block_on;
use image::ImageBuffer;
use ldr_tools::glam::{vec3, Vec3};
use ldr_wgpu::calculate_camera_data;
use log::info;

const WIDTH: u32 = 512;
const HEIGHT: u32 = 512;

fn main() {
    let args: Vec<_> = std::env::args().collect();
    let ldraw_path = &args[1];
    let input_folder = &args[2];
    let output_folder = &args[3];

    // Ignore most logs to avoid flooding the console.
    simple_logger::SimpleLogger::new()
        .with_level(log::LevelFilter::Warn)
        .init()
        .unwrap();

    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });

    let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .unwrap();

    let (device, queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: None,
        required_features: ldr_wgpu::REQUIRED_FEATURES,
        required_limits: wgpu::Limits {
            max_binding_array_elements_per_shader_stage: 4,
            ..Default::default()
        },
        ..Default::default()
    }))
    .unwrap();

    let format = wgpu::TextureFormat::Rgba8UnormSrgb;

    let size = wgpu::Extent3d {
        width: WIDTH,
        height: HEIGHT,
        depth_or_array_layers: 1,
    };
    let texture_desc = wgpu::TextureDescriptor {
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::RENDER_ATTACHMENT,
        label: None,
        view_formats: &[],
    };
    let output = device.create_texture(&texture_desc);
    let output_view = output.create_view(&Default::default());

    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        size: WIDTH as u64 * HEIGHT as u64 * 4,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        label: None,
        mapped_at_creation: false,
    });

    let translation = vec3(0.0, -0.5, -200.0);
    let rotation_xyz = Vec3::ZERO;
    let camera_data = calculate_camera_data(WIDTH, HEIGHT, translation, rotation_xyz);

    let mut renderer = ldr_wgpu::Renderer::new(&device, WIDTH, HEIGHT, format, ldraw_path);
    renderer.update_camera(&queue, camera_data);

    let start = std::time::Instant::now();

    globwalk::GlobWalkerBuilder::from_patterns(input_folder, &["*.{dat}"])
        .max_depth(1)
        .build()
        .unwrap()
        .for_each(|entry| {
            let path = entry.as_ref().unwrap().path();
            let path_str = path.to_str().unwrap();
            println!("{path:?}");

            let start = std::time::Instant::now();
            let scene = ldr_wgpu::Scene::new(&device, &queue, ldraw_path, path_str);
            info!("Load scene: {:?}", start.elapsed());

            renderer.render(&output_view, &device, &queue, &scene);

            let file_name = path.with_extension("png");
            let file_name = file_name.file_name().unwrap();
            let output_path = Path::new(output_folder).join(file_name);

            let encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("PNG Render Encoder"),
            });

            save_screenshot(
                &device,
                &queue,
                encoder,
                &output,
                &output_buffer,
                size,
                output_path,
            );

            // Clean up resources.
            queue.submit(std::iter::empty());
            device.poll(wgpu::PollType::Wait).unwrap();
        });

    println!("{:?}", start.elapsed());
}

fn save_screenshot(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    mut encoder: wgpu::CommandEncoder,
    output: &wgpu::Texture,
    output_buffer: &wgpu::Buffer,
    size: wgpu::Extent3d,
    output_path: std::path::PathBuf,
) {
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            aspect: wgpu::TextureAspect::All,
            texture: output,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: output_buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(WIDTH * 4),
                rows_per_image: Some(HEIGHT),
            },
        },
        size,
    );
    queue.submit([encoder.finish()]);

    // Save the output texture.
    // Adapted from WGPU Example https://github.com/gfx-rs/wgpu/tree/master/wgpu/examples/capture
    {
        // TODO: Find ways to optimize this?
        let buffer_slice = output_buffer.slice(..);

        // TODO: Reuse the channel?
        let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        device.poll(wgpu::PollType::Wait).unwrap();
        block_on(rx.receive()).unwrap().unwrap();

        let data = buffer_slice.get_mapped_range();
        let buffer =
            ImageBuffer::<image::Rgba<u8>, _>::from_raw(WIDTH, HEIGHT, data.to_owned()).unwrap();
        buffer.save(output_path).unwrap();
    }
    output_buffer.unmap();
}
