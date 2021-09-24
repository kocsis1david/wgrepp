//! Simple SSAO implementation
//!
//! # Effect and resouces separation
//!
//! [`SsaoEffect`] is intended to be only created once, but there can be multple
//! [`SsaoResources`] for each render target, e.g. one for the main window and others for offscreen
//! render targets.
//!
//! Example:
//!
//! ```
//! // Initialization
//! let effect = SsaoEffect::new(
//!     &device,
//!     &queue,
//!     TextureFormat::Rg16Float,
//!     true
//! );
//!
//! let resources = SsaoResources::new(
//!     &effect,
//!     &device,
//!     &queue,
//!     &depth_texture_view,
//!     &normal_texture_view,
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
//! The implementation is based on <https://learnopengl.com/Advanced-Lighting/SSAO> and some ideas
//! from [Stable SSAO in Battlefield 3 with Selective Temporal
//! Filtering](https://developer.nvidia.com/sites/default/files/akamai/gamedev/files/gdc12/GDC12_Bavoil_Stable_SSAO_In_BF3_With_STF.pdf)
//!

use std::num::NonZeroU32;

use bytemuck::{bytes_of, cast_slice, Pod, Zeroable};
use num::integer::div_ceil;
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;
use shaderc::ShaderKind;

use crate::helper::{align_to, compile_glsl_shader};

const NOISE_TEXTURE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg32Float;
const SSAO_WORK_GROUP_SIZE: [u32; 2] = [8, 8];
pub const SSAO_NOISE_TEXTURE_SIZE: u32 = 64;
/// Output texture format
pub const SSAO_TEXTURE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R8Unorm;

/// Contains pipelines and resources that only need to exist once
pub struct SsaoEffect {
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
    noise_texture_view: wgpu::TextureView,
    blur: Option<SsaoBlurEffect>,
}

impl SsaoEffect {
    /// Creates an [`SsaoEffect`]
    ///
    /// # Arguments
    ///
    /// - `queue` - Needed to issue write commands
    /// - `normal_format` - The format of the normal input texture. Currently only rg16f is
    ///   supported.
    /// - `blur` - Set to true to create pipelines for blurring
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        normal_format: wgpu::TextureFormat,
        blur: bool,
    ) -> SsaoEffect {
        assert!(normal_format == wgpu::TextureFormat::Rg16Float);

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(
                            std::mem::size_of::<SsaoUniforms>() as u64,
                        ),
                    },
                    count: None,
                },
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
                        format: SSAO_TEXTURE_FORMAT,
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
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        view_dimension: wgpu::TextureViewDimension::D2,
                        format: NOISE_TEXTURE_FORMAT,
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

        let compute_shader =
            compile_glsl_shader(device, include_str!("ssao.comp"), ShaderKind::Compute, &[])
                .unwrap();

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            module: &compute_shader,
            entry_point: "main",
        });

        let mut random = Xoshiro256PlusPlus::seed_from_u64(0);
        let noise_texture_view = create_noise_texture(device, queue, &mut random)
            .create_view(&wgpu::TextureViewDescriptor::default());

        let blur = if blur {
            Some(SsaoBlurEffect::new(device))
        } else {
            None
        };

        SsaoEffect {
            bind_group_layout,
            pipeline,
            noise_texture_view,
            blur,
        }
    }
}

#[repr(C, align(16))]
#[derive(Copy, Clone)]
struct SsaoUniforms {
    projection: [[f32; 4]; 4],
    uv_to_view_space_add: [f32; 2],
    uv_to_view_space_mul: [f32; 2],
    depth_add: f32,
    depth_mul: f32,
    noise_offset: [u32; 2],
    sample_count: u32,
    radius: f32,
    bias: f32,
}

unsafe impl Pod for SsaoUniforms {}
unsafe impl Zeroable for SsaoUniforms {}

struct SsaoBlurEffect {
    bind_group_layout: wgpu::BindGroupLayout,
    blur_x_pipeline: wgpu::ComputePipeline,
    blur_y_pipeline: wgpu::ComputePipeline,
}

impl SsaoBlurEffect {
    pub(crate) fn new(device: &wgpu::Device) -> SsaoBlurEffect {
        let blur_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(
                                std::mem::size_of::<SsaoUniforms>() as u64,
                            ),
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            view_dimension: wgpu::TextureViewDimension::D2,
                            format: SSAO_TEXTURE_FORMAT,
                            access: wgpu::StorageTextureAccess::WriteOnly,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            view_dimension: wgpu::TextureViewDimension::D2,
                            format: SSAO_TEXTURE_FORMAT,
                            access: wgpu::StorageTextureAccess::ReadOnly,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            multisampled: false,
                        },
                        count: None,
                    },
                ],
            });

        let blur_x_compute_shader = compile_glsl_shader(
            device,
            include_str!("blur.comp"),
            ShaderKind::Compute,
            &[("BLUR_X_PASS", None)],
        )
        .unwrap();

        let blur_y_compute_shader = compile_glsl_shader(
            device,
            include_str!("blur.comp"),
            ShaderKind::Compute,
            &[("BLUR_Y_PASS", None)],
        )
        .unwrap();

        let blur_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&blur_bind_group_layout],
            push_constant_ranges: &[],
        });

        let blur_x_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&blur_pipeline_layout),
            module: &blur_x_compute_shader,
            entry_point: "main",
        });

        let blur_y_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&blur_pipeline_layout),
            module: &blur_y_compute_shader,
            entry_point: "main",
        });

        SsaoBlurEffect {
            bind_group_layout: blur_bind_group_layout,
            blur_x_pipeline,
            blur_y_pipeline,
        }
    }
}

/// Resources per render target
pub struct SsaoResources {
    textures: SsaoTextures,
    samples: SsaoSamplesBuffer,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    blur_bind_groups: Option<(wgpu::BindGroup, wgpu::BindGroup)>,
}

impl SsaoResources {
    /// # Arguments
    ///
    /// - `normal_texture_view` - Texture with normals in view space
    /// - `texture_size` - Initial size of the ssao texture, same as the window size
    /// - `sample_count` - Number of samples per pixel, 16 or 32 is recommended
    pub fn new(
        effect: &SsaoEffect,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        depth_texture_view: &wgpu::TextureView,
        normal_texture_view: &wgpu::TextureView,
        texture_size: [u32; 2],
        sample_count: u32,
    ) -> SsaoResources {
        let textures = SsaoTextures::new(effect, device, texture_size);

        let mut samples = SsaoSamplesBuffer::new(device, sample_count);
        samples.write(device, queue, sample_count);

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ssao_uniforms"),
            size: std::mem::size_of::<SsaoUniforms>() as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM,
            mapped_at_creation: false,
        });

        let (bind_group, blur_bind_groups) = create_bind_groups(
            device,
            &uniform_buffer,
            effect,
            &samples,
            &textures,
            depth_texture_view,
            normal_texture_view,
        );

        SsaoResources {
            textures,
            samples,
            uniform_buffer,
            bind_group,
            blur_bind_groups,
        }
    }

    /// Should be called when the window is resized
    pub fn resize(
        &mut self,
        effect: &SsaoEffect,
        device: &wgpu::Device,
        depth_texture_view: &wgpu::TextureView,
        normal_texture_view: &wgpu::TextureView,
        texture_size: [u32; 2],
    ) {
        self.textures = SsaoTextures::new(effect, device, texture_size);

        let (bind_group, blur_bind_group) = create_bind_groups(
            device,
            &self.uniform_buffer,
            effect,
            &self.samples,
            &self.textures,
            depth_texture_view,
            normal_texture_view,
        );

        self.bind_group = bind_group;
        self.blur_bind_groups = blur_bind_group;
    }

    /// Contains the output SSAO values in `SSAO_TEXTURE_FORMAT` format
    pub fn output_texture_view(&self) -> &wgpu::TextureView {
        &self.textures.output
    }

    /// Writes uniforms used by the SSAO shader
    ///
    /// Should be called at least once before draw
    ///
    /// # Arguments
    ///
    /// - `proj` - The projection matrix, only perspective projection is supported for now
    /// - `radius` - Radius of the sphere in which the samples are taken
    /// - `bias` - Minimum difference of linear Z to contribute to the occlusion value
    /// - `noise_offset` - Set to [0, 0] if TAA is not used, can be set to a random value to in the
    ///   range `0..SSAO_NOISE_TEXTURE_SIZE` and change it every frame if TAA is used
    pub fn write_uniforms(
        &self,
        queue: &wgpu::Queue,
        proj: &[[f32; 4]; 4],
        radius: f32,
        bias: f32,
        noise_offset: [u32; 2],
    ) {
        let tan_half_fov_x = 1.0 / proj[0][0];
        let tan_half_fov_y = 1.0 / proj[1][1];

        queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytes_of(&SsaoUniforms {
                projection: *proj,
                uv_to_view_space_mul: [tan_half_fov_x * 2.0, tan_half_fov_y * -2.0],
                uv_to_view_space_add: [tan_half_fov_x * -1.0, tan_half_fov_y],
                depth_add: -proj[2][2],
                depth_mul: proj[3][2],
                noise_offset,
                sample_count: self.samples.count,
                radius,
                bias,
            }),
        );
    }

    /// Draws the SSAO to the `output_texture_view` texture.
    ///
    /// # Arguments
    /// - `blur` - Set to true to blur. Needs to be set to true at [`SsaoEffect::new`] as well.
    pub fn draw(&self, effect: &SsaoEffect, encoder: &mut wgpu::CommandEncoder, blur: bool) {
        let size = &self.textures.size;
        let work_group_count_x = div_ceil(size[0], SSAO_WORK_GROUP_SIZE[0]);
        let work_group_count_y = div_ceil(size[1], SSAO_WORK_GROUP_SIZE[1]);

        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("ssao"),
        });

        cpass.set_pipeline(&effect.pipeline);
        cpass.set_bind_group(0, &self.bind_group, &[]);
        cpass.dispatch(work_group_count_x, work_group_count_y, 1);

        if blur {
            let (blur_effect, (blur_x_bind_group, blur_y_bind_group)) = effect
                .blur
                .as_ref()
                .zip(self.blur_bind_groups.as_ref())
                .unwrap();

            cpass.set_pipeline(&blur_effect.blur_x_pipeline);
            cpass.set_bind_group(0, blur_x_bind_group, &[]);
            cpass.dispatch(work_group_count_x, work_group_count_y, 1);

            cpass.set_pipeline(&blur_effect.blur_y_pipeline);
            cpass.set_bind_group(0, blur_y_bind_group, &[]);
            cpass.dispatch(work_group_count_x, work_group_count_y, 1);
        }
    }
}

fn create_bind_groups(
    device: &wgpu::Device,
    uniform_buffer: &wgpu::Buffer,
    effect: &SsaoEffect,
    samples: &SsaoSamplesBuffer,
    textures: &SsaoTextures,
    depth_texture_view: &wgpu::TextureView,
    normal_texture_view: &wgpu::TextureView,
) -> (wgpu::BindGroup, Option<(wgpu::BindGroup, wgpu::BindGroup)>) {
    let uniforms_entry = wgpu::BindGroupEntry {
        binding: 0,
        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: uniform_buffer,
            offset: 0,
            size: wgpu::BufferSize::new(std::mem::size_of::<SsaoUniforms>() as u64),
        }),
    };

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &effect.bind_group_layout,
        entries: &[
            uniforms_entry.clone(),
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &samples.buffer,
                    offset: 0,
                    size: None,
                }),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(&textures.output),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(depth_texture_view),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::TextureView(normal_texture_view),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: wgpu::BindingResource::TextureView(&effect.noise_texture_view),
            },
        ],
    });

    let blur_bind_group = effect.blur.as_ref().map(|blur_effect| {
        let depth_entry = wgpu::BindGroupEntry {
            binding: 3,
            resource: wgpu::BindingResource::TextureView(depth_texture_view),
        };

        (
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &blur_effect.bind_group_layout,
                entries: &[
                    uniforms_entry.clone(),
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(
                            textures.blur.as_ref().unwrap(),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&textures.output),
                    },
                    depth_entry.clone(),
                ],
            }),
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &blur_effect.bind_group_layout,
                entries: &[
                    uniforms_entry,
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&textures.output),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(
                            textures.blur.as_ref().unwrap(),
                        ),
                    },
                    depth_entry,
                ],
            }),
        )
    });

    (bind_group, blur_bind_group)
}

struct SsaoTextures {
    output: wgpu::TextureView,
    blur: Option<wgpu::TextureView>,
    size: [u32; 2],
}

impl SsaoTextures {
    pub fn new(effect: &SsaoEffect, device: &wgpu::Device, size: [u32; 2]) -> SsaoTextures {
        let output = create_simple_2d_texture(
            device,
            size,
            SSAO_TEXTURE_FORMAT,
            wgpu::TextureUsages::STORAGE_BINDING,
        )
        .create_view(&wgpu::TextureViewDescriptor::default());

        let blur = effect.blur.as_ref().map(|_| {
            create_simple_2d_texture(
                device,
                size,
                SSAO_TEXTURE_FORMAT,
                wgpu::TextureUsages::STORAGE_BINDING,
            )
            .create_view(&wgpu::TextureViewDescriptor::default())
        });

        SsaoTextures { output, blur, size }
    }
}

struct SsaoSamplesBuffer {
    buffer: wgpu::Buffer,
    capacity: u32,
    count: u32,
}

impl SsaoSamplesBuffer {
    fn new(device: &wgpu::Device, capacity: u32) -> Self {
        let size = capacity as wgpu::BufferAddress
            * std::mem::size_of::<[f32; 4]>() as wgpu::BufferAddress;

        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            buffer,
            capacity,
            count: 0,
        }
    }

    fn write(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, count: u32) {
        if self.capacity > count {
            *self = SsaoSamplesBuffer::new(device, count.next_power_of_two());
        }

        let mut random = Xoshiro256PlusPlus::seed_from_u64(0);
        let samples = get_sample_vectors(&mut random, count);
        queue.write_buffer(&self.buffer, 0, cast_slice(&samples[..]));
        self.count = count;
    }
}

// https://www.pbr-book.org/3ed-2018/Monte_Carlo_Integration/2D_Sampling_with_Multidimensional_Transformations#UniformlySamplingaHemisphere
// fn get_sample_vectors(random: &mut Xoshiro256PlusPlus, count: u32) -> Vec<Vector4<f32>> {
//     assert!(count.is_power_of_two());

//     let cx = count.sqrt().next_power_of_two();
//     let cy = count / cx;
//     assert_eq!(cx * cy, count);

//     let mut ret = Vec::with_capacity(count as usize);
//     for ty in 0..cy {
//         for tx in 0..cx {
//             let u = (tx as f32 + random.gen_range(0.0..1.0f32)) / cx as f32;
//             let v = (ty as f32 + random.gen_range(0.0..1.0f32)) / cy as f32;

//             let z = u;
//             let r = (1.0 - z * z).sqrt();
//             let phi = 2.0 * std::f32::consts::PI * v;

//             ret.push(Vector4::new(r * phi.cos(), r * phi.sin(), z, 0.0));
//         }
//     }

//     ret
// }

fn get_sample_vectors(random: &mut Xoshiro256PlusPlus, count: u32) -> Vec<[f32; 4]> {
    (0..count)
        .map(|i| {
            use std::f32::consts::PI;
            let lng = random.gen_range(0.0..PI * 2.0);
            let lat = random.gen_range(0.0..1.0f32).acos();
            let r = ((i + 1) as f32 / count as f32).sqrt();

            [
                r * lng.cos() * lat.sin(),
                r * lng.sin() * lat.sin(),
                r * lat.cos(),
                0.0,
            ]
        })
        .collect()
}

fn create_simple_2d_texture(
    device: &wgpu::Device,
    size: [u32; 2],
    format: wgpu::TextureFormat,
    usage: wgpu::TextureUsages,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            width: size[0],
            height: size[1],
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage,
    })
}

fn create_noise_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    random: &mut Xoshiro256PlusPlus,
) -> wgpu::Texture {
    let size = wgpu::Extent3d {
        width: SSAO_NOISE_TEXTURE_SIZE,
        height: SSAO_NOISE_TEXTURE_SIZE,
        depth_or_array_layers: 1,
    };

    let mut noise_data = Vec::new();
    let bytes_per_row = align_to(
        size.width * std::mem::size_of::<[f32; 2]>() as u32,
        wgpu::COPY_BYTES_PER_ROW_ALIGNMENT,
    );

    for y in 0..size.height {
        for _ in 0..size.width {
            noise_data.extend_from_slice(cast_slice(&[
                random.gen_range(-1.0f32..=1.0),
                random.gen_range(-1.0f32..=1.0),
            ]));
        }

        noise_data.resize(((y + 1) * bytes_per_row) as usize, 0);
    }

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: NOISE_TEXTURE_FORMAT,
        usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::STORAGE_BINDING,
    });

    queue.write_texture(
        wgpu::ImageCopyTexture {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d { x: 0, y: 0, z: 0 },
            aspect: wgpu::TextureAspect::All,
        },
        &noise_data,
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(NonZeroU32::new(bytes_per_row).unwrap()),
            rows_per_image: Some(NonZeroU32::new(size.height).unwrap()),
        },
        size,
    );

    texture
}
