// File automatically generated by build.rs.
// Changes made to this file will not be saved.
pub mod bind_groups {
    pub struct BindGroup0(wgpu::BindGroup);
    pub struct BindGroupLayout0<'a> {
        pub input: &'a wgpu::TextureView,
        pub output: &'a wgpu::TextureView,
    }
    const LAYOUT_DESCRIPTOR0: wgpu::BindGroupLayoutDescriptor = wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Depth,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::R32Float,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
        ],
    };
    impl BindGroup0 {
        pub fn get_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
            device.create_bind_group_layout(&LAYOUT_DESCRIPTOR0)
        }
        pub fn from_bindings(device: &wgpu::Device, bindings: BindGroupLayout0) -> Self {
            let bind_group_layout = device.create_bind_group_layout(&LAYOUT_DESCRIPTOR0);
            let bind_group = device
                .create_bind_group(
                    &wgpu::BindGroupDescriptor {
                        layout: &bind_group_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(bindings.input),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::TextureView(
                                    bindings.output,
                                ),
                            },
                        ],
                        label: None,
                    },
                );
            Self(bind_group)
        }
        pub fn set<'a>(&'a self, render_pass: &mut wgpu::ComputePass<'a>) {
            render_pass.set_bind_group(0, &self.0, &[]);
        }
    }
    pub struct BindGroups<'a> {
        pub bind_group0: &'a BindGroup0,
    }
    pub fn set_bind_groups<'a>(
        pass: &mut wgpu::ComputePass<'a>,
        bind_groups: BindGroups<'a>,
    ) {
        bind_groups.bind_group0.set(pass);
    }
}
pub mod compute {
    pub const MAIN_WORKGROUP_SIZE: [u32; 3] = [16, 16, 1];
    pub fn create_main_pipeline(device: &wgpu::Device) -> wgpu::ComputePipeline {
        let module = super::create_shader_module(device);
        let layout = super::create_pipeline_layout(device);
        device
            .create_compute_pipeline(
                &wgpu::ComputePipelineDescriptor {
                    label: Some("Compute Pipeline main"),
                    layout: Some(&layout),
                    module: &module,
                    entry_point: "main",
                },
            )
    }
}
pub fn create_shader_module(device: &wgpu::Device) -> wgpu::ShaderModule {
    let source = std::borrow::Cow::Borrowed(include_str!("blit_depth.wgsl"));
    device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(source),
        })
}
pub fn create_pipeline_layout(device: &wgpu::Device) -> wgpu::PipelineLayout {
    device
        .create_pipeline_layout(
            &wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[
                    &bind_groups::BindGroup0::get_bind_group_layout(device),
                ],
                push_constant_ranges: &[],
            },
        )
}
