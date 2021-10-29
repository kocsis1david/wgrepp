//! Simple TAA implementation
//!
//! # Effect and resouces separation
//!
//! [`TaaEffect`] is intended to be only created once, but there can be multple
//! [`TaaResources`] for each render target, e.g. one for the main window and others for offscreen
//! render targets.
//!
//! Example:
//!
//! ```
//! // Initialization
//! let effect = TaaEffect::new(
//!     &device,
//!     &queue,
//!     TextureFormat::Rg16Float,
//!     true
//! );
//!
//! let resources = TaaResources::new(
//!     &effect,
//!     &device,
//!     &queue,
//!     &depth_texture_view,
//!     &velocity_texture_view,
//!     [window_size.x, window_size.y],
//!     32,
//! )
//!
//! // Draw at every frame
//! resources.write_uniforms(
//!     &queue,
//!     &proj_matrix.into(),
//!     1.0,
//!     0.01,
//!     [0, 0],
//! );
//!
//! resources.draw(&effect, &mut encoder, true);
//! ```
//!
//! The implementation is based on <https://learnopengl.com/Advanced-Lighting/TAA> and some ideas
//! from [Stable TAA in Battlefield 3 with Selective Temporal
//! Filtering](https://developer.nvidia.com/sites/default/files/akamai/gamedev/files/gdc12/GDC12_Bavoil_Stable_TAA_In_BF3_With_STF.pdf)
//!

use num::integer::div_ceil;
use shaderc::ShaderKind;

use crate::helper::{compile_glsl_shader, glsl_storage_format};

const TAA_WORK_GROUP_SIZE: [u32; 2] = [8, 8];
pub const TAA_NOISE_TEXTURE_SIZE: u32 = 64;
pub const TAA_OUTPUT_TEXTURE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// Contains pipelines and resources that only need to exist once
pub struct TaaEffect {
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
}

impl TaaEffect {
    /// Creates an [`TaaEffect`]
    ///
    /// # Arguments
    ///
    /// - `queue` - Needed to issue write commands
    /// - `normal_format` - The format of the normal input texture.
    /// - `blur` - Set to true to create pipelines for blurring
    pub fn new(device: &wgpu::Device, normal_format: wgpu::TextureFormat) -> TaaEffect {
        let mut macros = Vec::new();
        if !check_normal_format(normal_format, &mut macros) {
            panic!("Unsupported normal format");
        }

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        view_dimension: wgpu::TextureViewDimension::D2,
                        format: TAA_OUTPUT_TEXTURE_FORMAT,
                        access: wgpu::StorageTextureAccess::WriteOnly,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        view_dimension: wgpu::TextureViewDimension::D2,
                        format: normal_format,
                        access: wgpu::StorageTextureAccess::ReadOnly,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let compute_shader = compile_glsl_shader(
            device,
            include_str!("taa.comp"),
            ShaderKind::Compute,
            &macros,
        )
        .unwrap();

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            module: &compute_shader,
            entry_point: "main",
        });

        TaaEffect {
            bind_group_layout,
            pipeline,
        }
    }
}

fn check_normal_format(
    normal_format: wgpu::TextureFormat,
    macros: &mut Vec<(&str, Option<&str>)>,
) -> bool {
    let desc = normal_format.describe();

    match desc.components {
        2 => {
            macros.push(("NORMAL_TWO_COMPONENT_FORMAT", None));
        }
        3 | 4 => {}
        _ => return false,
    }

    match desc.sample_type {
        wgpu::TextureSampleType::Float { .. } => {}
        _ => return false,
    }

    if let Some(normal_format) = glsl_storage_format(normal_format) {
        macros.push(("NORMAL_FORMAT", Some(normal_format)));
        true
    } else {
        false
    }
}

/// Resources per render target
pub struct TaaResources {
    textures: TaaTextures,
    bind_group: wgpu::BindGroup,
}

impl TaaResources {
    /// # Arguments
    ///
    /// - `velocity_texture_view` - Texture with normals in view space
    /// - `texture_size` - Initial size of the taa texture, same as the window size
    /// - `sample_count` - Number of samples per pixel, 16 or 32 is recommended
    pub fn new(
        effect: &TaaEffect,
        device: &wgpu::Device,
        depth_texture_view: &wgpu::TextureView,
        velocity_texture_view: &wgpu::TextureView,
        texture_size: [u32; 2],
    ) -> TaaResources {
        let textures = TaaTextures::new(device, texture_size);

        let bind_group = create_bind_group(
            device,
            effect,
            &textures,
            depth_texture_view,
            velocity_texture_view,
        );

        TaaResources {
            textures,
            bind_group,
        }
    }

    /// Should be called when the window is resized
    ///
    /// `depth_texture_view` and `velocity_texture_view` must have the `texture_size` size.
    pub fn resize(
        &mut self,
        effect: &TaaEffect,
        device: &wgpu::Device,
        depth_texture_view: &wgpu::TextureView,
        velocity_texture_view: &wgpu::TextureView,
        texture_size: [u32; 2],
    ) {
        self.textures = TaaTextures::new(device, texture_size);

        let bind_group = create_bind_group(
            device,
            effect,
            &self.textures,
            depth_texture_view,
            velocity_texture_view,
        );

        self.bind_group = bind_group;
    }

    /// Contains the output TAA values in `TAA_TEXTURE_FORMAT` format.
    ///
    /// After `resize` is called, the textures are recreated and the previous output texture view
    /// becomes invalid.
    pub fn output_texture_view(&self, frame: u64) -> &wgpu::TextureView {
        &self.textures.output[(frame % 2) as usize]
    }

    /// Draws the TAA to the `output_texture_view` texture.
    pub fn draw(&self, effect: &TaaEffect, encoder: &mut wgpu::CommandEncoder) {
        let size = &self.textures.size;
        let work_group_count_x = div_ceil(size[0], TAA_WORK_GROUP_SIZE[0]);
        let work_group_count_y = div_ceil(size[1], TAA_WORK_GROUP_SIZE[1]);

        let mut cpass =
            encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("taa") });

        cpass.set_pipeline(&effect.pipeline);
        cpass.set_bind_group(0, &self.bind_group, &[]);
        cpass.dispatch(work_group_count_x, work_group_count_y, 1);
    }
}

fn create_bind_group(
    device: &wgpu::Device,
    effect: &TaaEffect,
    textures: &TaaTextures,
    depth_texture_view: &wgpu::TextureView,
    velocity_texture_view: &wgpu::TextureView,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &effect.bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(&textures.output[0]),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(depth_texture_view),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::TextureView(velocity_texture_view),
            },
        ],
    })
}

struct TaaTextures {
    output: [wgpu::TextureView; 2],
    size: [u32; 2],
}

impl TaaTextures {
    pub fn new(device: &wgpu::Device, size: [u32; 2]) -> TaaTextures {
        TaaTextures {
            output: [
                create_output_texture(device, size),
                create_output_texture(device, size),
            ],
            size,
        }
    }
}

fn create_output_texture(device: &wgpu::Device, size: [u32; 2]) -> wgpu::TextureView {
    device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("taa_output_texture"),
            size: wgpu::Extent3d {
                width: size[0],
                height: size[1],
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: TAA_OUTPUT_TEXTURE_FORMAT,
            usage: wgpu::TextureUsages::STORAGE_BINDING,
        })
        .create_view(&wgpu::TextureViewDescriptor::default())
}
