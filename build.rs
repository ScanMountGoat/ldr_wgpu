use std::fmt::Write;

use wgsl_to_wgpu::{create_shader_module, MatrixVectorTypes, WriteOptions};

fn main() {
    write_shader("src/shader/model.wgsl", "model.wgsl", "src/shader/model.rs");
    write_shader(
        "src/shader/culling.wgsl",
        "culling.wgsl",
        "src/shader/culling.rs",
    );
    write_shader(
        "src/shader/depth_pyramid.wgsl",
        "depth_pyramid.wgsl",
        "src/shader/depth_pyramid.rs",
    );
    write_shader(
        "src/shader/blit_depth.wgsl",
        "blit_depth.wgsl",
        "src/shader/blit_depth.rs",
    );
    write_shader(
        "src/shader/visibility.wgsl",
        "visibility.wgsl",
        "src/shader/visibility.rs",
    );
    write_shader(
        "src/shader/scan.wgsl",
        "scan.wgsl",
        "src/shader/scan.rs",
    );
}

fn write_shader(wgsl_path: &str, include_path: &str, output_path: &str) {
    println!("cargo:rerun-if-changed={wgsl_path}");

    let wgsl_source = std::fs::read_to_string(wgsl_path).unwrap();

    // Generate the Rust bindings and write to a file.
    let mut text = String::new();
    writeln!(&mut text, "// File automatically generated by build.rs.").unwrap();
    writeln!(&mut text, "// Changes made to this file will not be saved.").unwrap();
    text += &create_shader_module(
        &wgsl_source,
        include_path,
        WriteOptions {
            derive_bytemuck: true,
            matrix_vector_types: MatrixVectorTypes::Glam,
            ..Default::default()
        },
    )
    .unwrap();

    std::fs::write(output_path, text.as_bytes()).unwrap();
}
