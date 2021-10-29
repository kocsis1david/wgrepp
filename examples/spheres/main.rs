use std::{
    num::NonZeroU64,
    ops::{Range, Rem, Sub},
};

use bytemuck::{bytes_of, cast_slice, Pod, Zeroable};
use cgmath::{Matrix, SquareMatrix, Zero};
use rand::Rng;
use wgpu::{include_wgsl, util::DeviceExt};
use wgrepp::ssao::{SsaoEffect, SsaoResources, SSAO_TEXTURE_FORMAT};
use winit::{
    dpi::PhysicalSize,
    event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::Window,
};

const SPHERE_COUNT: u32 = 1000;
const SSAO_RADIUS: f32 = 1.0;
const SSAO_BIAS: f32 = 0.01;

const SURFACE_TEXTURE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Bgra8UnormSrgb;
const DEPTH_TEXTURE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
const COLOR_TEXTURE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const NORMAL_TEXTURE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg11b10Float;

#[rustfmt::skip]
#[allow(unused)]
pub const OPENGL_TO_WGPU_MATRIX: cgmath::Matrix4<f32> = cgmath::Matrix4::new(
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.0,
    0.0, 0.0, 0.5, 1.0,
);

#[repr(C)]
#[derive(Clone, Copy)]
struct Temporal<T> {
    previous: T,
    current: T,
}

impl<T: Copy> From<T> for Temporal<T> {
    fn from(value: T) -> Temporal<T> {
        Temporal {
            previous: value,
            current: value,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct MeshVsUniforms {
    view_proj_matrix: Temporal<cgmath::Matrix4<f32>>,
}

unsafe impl Pod for MeshVsUniforms {}
unsafe impl Zeroable for MeshVsUniforms {}

#[repr(C)]
#[derive(Clone, Copy)]
struct BlendFsUniforms {
    ssao_enabled: u32,
}

unsafe impl Pod for BlendFsUniforms {}
unsafe impl Zeroable for BlendFsUniforms {}

#[repr(C)]
#[derive(Clone, Copy)]
struct Instance {
    model_matrix: Temporal<cgmath::Matrix4<f32>>,
    normal_matrix: cgmath::Matrix4<f32>,
    color: [f32; 4],
}

unsafe impl Pod for Instance {}
unsafe impl Zeroable for Instance {}

#[repr(C)]
#[derive(Clone, Copy)]
struct Vertex {
    position: [f32; 3],
    normal: [f32; 3],
}

unsafe impl Pod for Vertex {}
unsafe impl Zeroable for Vertex {}

struct Mesh {
    buffer: wgpu::Buffer,
    indices_offset: u64,
    index_count: u32,
}

impl Mesh {
    fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        vertices: &[Vertex],
        indices: &[u16],
    ) -> Mesh {
        let indices_offset = align_to(
            (vertices.len() as u32) as u64 * std::mem::size_of::<Vertex>() as u64,
            wgpu::COPY_BUFFER_ALIGNMENT,
        );

        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sphere_buffer"),
            size: indices_offset + indices.len() as u64 * 2,
            usage: wgpu::BufferUsages::VERTEX
                | wgpu::BufferUsages::INDEX
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        queue.write_buffer(&buffer, 0, cast_slice(&vertices));
        queue.write_buffer(&buffer, indices_offset, cast_slice(&indices));

        Mesh {
            buffer,
            index_count: indices.len() as u32,
            indices_offset,
        }
    }

    fn new_sphere(device: &wgpu::Device, queue: &wgpu::Queue, segments: u32, rings: u32) -> Self {
        use std::f32::consts::PI;

        let mut vertices = Vec::new();
        for r in 0..=rings {
            let latitude = r as f32 / rings as f32 * PI - PI / 2.0;
            let y = latitude.sin();

            for s in 0..segments {
                let longitude = s as f32 / segments as f32 * PI * 2.0;
                let r = (1.0 - y * y).sqrt();

                let p = [longitude.cos() * r, y, -longitude.sin() * r];
                vertices.push(Vertex {
                    position: p,
                    normal: p,
                });
            }
        }

        let mut indices = Vec::new();
        for r in 0..rings {
            for s in 0..segments {
                let bl = (r * segments + s) as u16;
                let br = (r * segments + (s + 1) % segments) as u16;
                let tl = ((r + 1) * segments + s) as u16;
                let tr = ((r + 1) * segments + (s + 1) % segments) as u16;
                indices.extend_from_slice(&[bl, br, tr, bl, tr, tl]);
            }
        }

        Self::new(device, queue, &vertices, &indices)
    }

    fn new_plane(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let normal = [0.0, 1.0, 0.0];
        let v = |position: [f32; 3]| Vertex { position, normal };

        Self::new(
            device,
            queue,
            &[
                v([-1.0, 0.0, 1.0]),
                v([1.0, 0.0, 1.0]),
                v([1.0, 0.0, -1.0]),
                v([-1.0, 0.0, -1.0]),
            ],
            &[0, 1, 2, 0, 2, 3],
        )
    }

    fn draw<'a>(&'a self, rpass: &mut wgpu::RenderPass<'a>, instances: Range<u32>) {
        rpass.set_vertex_buffer(0, self.buffer.slice(0..self.indices_offset));
        rpass.set_index_buffer(
            self.buffer.slice(self.indices_offset..),
            wgpu::IndexFormat::Uint16,
        );
        rpass.draw_indexed(0..self.index_count, 0, instances);
    }
}

struct SpheresExample {
    color_texture_view: wgpu::TextureView,
    depth_texture_view: wgpu::TextureView,
    normal_texture_view: wgpu::TextureView,
    uniform_buffer: wgpu::Buffer,
    mesh_pipeline: wgpu::RenderPipeline,
    mesh_bind_group: wgpu::BindGroup,
    sphere_mesh: Mesh,
    plane_mesh: Mesh,
    ssao_effect: SsaoEffect,
    ssao_resources: SsaoResources,
    ssao_enabled: bool,
    blend_pipeline: wgpu::RenderPipeline,
    blend_uniform_buffer: wgpu::Buffer,
    blend_bind_group_layout: wgpu::BindGroupLayout,
    blend_bind_group: wgpu::BindGroup,
}

impl SpheresExample {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        size: PhysicalSize<u32>,
    ) -> SpheresExample {
        let color_texture_view = create_color_texture(device, size);
        let depth_texture_view = create_depth_texture(device, size);
        let normal_texture_view = create_normal_texture(device, size);

        let mesh_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("mesh_bind_group_layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: Some(
                                NonZeroU64::new(std::mem::size_of::<MeshVsUniforms>() as u64)
                                    .unwrap(),
                            ),
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: Some(
                                NonZeroU64::new(std::mem::size_of::<Instance>() as u64).unwrap(),
                            ),
                        },
                        count: None,
                    },
                ],
            });

        let mesh_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mesh_pipeline_layout"),
            bind_group_layouts: &[&mesh_bind_group_layout],
            push_constant_ranges: &[],
        });

        let mesh_shader_module = device.create_shader_module(&include_wgsl!("mesh.wgsl"));

        let mesh_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("mesh_pipeline"),
            layout: Some(&mesh_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &mesh_shader_module,
                entry_point: "vs_main",
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Vertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x3,
                            offset: 0,
                            shader_location: 0,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x3,
                            offset: 12,
                            shader_location: 1,
                        },
                    ],
                }],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                clamp_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_TEXTURE_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &mesh_shader_module,
                entry_point: "fs_main",
                targets: &[
                    wgpu::ColorTargetState {
                        format: COLOR_TEXTURE_FORMAT,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    },
                    wgpu::ColorTargetState {
                        format: NORMAL_TEXTURE_FORMAT,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    },
                ],
            }),
        });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("uniforms_buffer"),
            size: std::mem::size_of::<MeshVsUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let proj_matrix = proj_matrix(size);
        let view_matrix = view_matrix();
        write_uniforms(queue, &uniform_buffer, &view_matrix, &proj_matrix);

        let plane_matrix = cgmath::Matrix4::from_scale(1000.0);
        let mut instances = vec![Instance {
            model_matrix: Temporal::from(plane_matrix),
            normal_matrix: (view_matrix * plane_matrix).invert().unwrap().transpose(),
            color: [1.0, 1.0, 1.0, 1.0],
        }];

        let mut rng = rand::thread_rng();
        for _ in 0..SPHERE_COUNT {
            let sphere_matrix = cgmath::Matrix4::from_translation(cgmath::Vector3::new(
                rng.gen_range(-100.0..100.0),
                1.0,
                rng.gen_range(-170.0..30.0),
            ));
            instances.push(Instance {
                model_matrix: Temporal::from(sphere_matrix),
                normal_matrix: (view_matrix * sphere_matrix).invert().unwrap().transpose(),
                color: [
                    rng.gen_range(0.0..1.0),
                    rng.gen_range(0.0..1.0),
                    rng.gen_range(0.0..1.0),
                    1.0,
                ],
            });
        }

        let instances_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("instances_buffer"),
            contents: cast_slice(&instances),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let mesh_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("spheres_bind_group"),
            layout: &mesh_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &uniform_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &instances_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
            ],
        });

        let ssao_effect = SsaoEffect::new(device, queue, NORMAL_TEXTURE_FORMAT, true);
        let ssao_resources = SsaoResources::new(
            &ssao_effect,
            device,
            queue,
            &depth_texture_view,
            &normal_texture_view,
            [size.width, size.height],
            32,
        );

        ssao_resources.write_uniforms(&queue, &proj_matrix.into(), SSAO_RADIUS, SSAO_BIAS, [0, 0]);

        let blend_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("blend_uniform_buffer"),
            size: std::mem::size_of::<BlendFsUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        queue.write_buffer(
            &blend_uniform_buffer,
            0,
            bytes_of(&BlendFsUniforms { ssao_enabled: 1 }),
        );

        let blend_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("blend_bind_group_layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: Some(
                                NonZeroU64::new(std::mem::size_of::<BlendFsUniforms>() as u64)
                                    .unwrap(),
                            ),
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::ReadOnly,
                            format: COLOR_TEXTURE_FORMAT,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::ReadOnly,
                            format: SSAO_TEXTURE_FORMAT,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                ],
            });

        let blend_bind_group = create_blend_bind_group(
            device,
            &blend_bind_group_layout,
            &blend_uniform_buffer,
            &color_texture_view,
            ssao_resources.output_texture_view(),
        );

        let blend_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("blend_pipeline_layout"),
                bind_group_layouts: &[&blend_bind_group_layout],
                push_constant_ranges: &[],
            });

        let blend_shader_module = device.create_shader_module(&include_wgsl!("blend.wgsl"));

        let blend_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("blend_pipeline"),
            layout: Some(&blend_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &blend_shader_module,
                entry_point: "vs_main",
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &blend_shader_module,
                entry_point: "fs_main",
                targets: &[wgpu::ColorTargetState {
                    format: SURFACE_TEXTURE_FORMAT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                }],
            }),
        });

        SpheresExample {
            color_texture_view,
            depth_texture_view,
            normal_texture_view,
            uniform_buffer,
            mesh_pipeline,
            mesh_bind_group,
            sphere_mesh: Mesh::new_sphere(device, queue, 32, 16),
            plane_mesh: Mesh::new_plane(device, queue),
            ssao_effect,
            ssao_resources,
            ssao_enabled: true,
            blend_pipeline,
            blend_uniform_buffer,
            blend_bind_group_layout,
            blend_bind_group,
        }
    }

    pub fn resize(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, size: PhysicalSize<u32>) {
        self.depth_texture_view = create_depth_texture(device, size);
        self.color_texture_view = create_color_texture(device, size);
        self.normal_texture_view = create_normal_texture(device, size);
        self.ssao_resources.resize(
            &self.ssao_effect,
            device,
            &self.depth_texture_view,
            &self.normal_texture_view,
            [size.width, size.height],
        );
        self.blend_bind_group = create_blend_bind_group(
            device,
            &self.blend_bind_group_layout,
            &self.blend_uniform_buffer,
            &self.color_texture_view,
            &self.ssao_resources.output_texture_view(),
        );

        let proj_matrix = proj_matrix(size);
        let view_matrix = view_matrix();
        write_uniforms(queue, &self.uniform_buffer, &view_matrix, &proj_matrix);

        self.ssao_resources.write_uniforms(
            queue,
            &proj_matrix.into(),
            SSAO_RADIUS,
            SSAO_BIAS,
            [0, 0],
        );
    }

    pub fn render(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        frame_output_texture_view: &wgpu::TextureView,
    ) {
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[
                wgpu::RenderPassColorAttachment {
                    view: &self.color_texture_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.2,
                            g: 0.3,
                            b: 0.4,
                            a: 1.0,
                        }),
                        store: true,
                    },
                },
                wgpu::RenderPassColorAttachment {
                    view: &self.normal_texture_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: true,
                    },
                },
            ],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.depth_texture_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: true,
                }),
                stencil_ops: None,
            }),
        });

        rpass.set_pipeline(&self.mesh_pipeline);
        rpass.set_bind_group(0, &self.mesh_bind_group, &[]);
        self.plane_mesh.draw(&mut rpass, 0..1);
        self.sphere_mesh.draw(&mut rpass, 1..(SPHERE_COUNT + 1));
        std::mem::drop(rpass);

        if self.ssao_enabled {
            self.ssao_resources
                .draw(&self.ssao_effect, &mut encoder, true);
        }

        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[wgpu::RenderPassColorAttachment {
                view: frame_output_texture_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.2,
                        g: 0.3,
                        b: 0.4,
                        a: 1.0,
                    }),
                    store: true,
                },
            }],
            depth_stencil_attachment: None,
        });
        rpass.set_pipeline(&self.blend_pipeline);
        rpass.set_bind_group(0, &self.blend_bind_group, &[]);
        rpass.draw(0..3, 0..1);
        std::mem::drop(rpass);

        queue.submit(Some(encoder.finish()));
    }

    fn set_ssao_enabled(&mut self, queue: &wgpu::Queue, ssao_enabled: bool) {
        self.ssao_enabled = ssao_enabled;
        queue.write_buffer(
            &self.blend_uniform_buffer,
            0,
            bytes_of(&BlendFsUniforms {
                ssao_enabled: if ssao_enabled { 1 } else { 0 },
            }),
        );
    }
}

fn create_blend_bind_group(
    device: &wgpu::Device,
    blend_bind_group_layout: &wgpu::BindGroupLayout,
    blend_uniform_buffer: &wgpu::Buffer,
    color_texture_view: &wgpu::TextureView,
    ssao_texture_view: &wgpu::TextureView,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("blend_bind_group"),
        layout: blend_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &blend_uniform_buffer,
                    offset: 0,
                    size: None,
                }),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(color_texture_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(ssao_texture_view),
            },
        ],
    })
}

fn write_uniforms(
    queue: &wgpu::Queue,
    uniform_buffer: &wgpu::Buffer,
    view_matrix: &cgmath::Matrix4<f32>,
    proj_matrix: &cgmath::Matrix4<f32>,
) {
    let view_proj_matrix = *proj_matrix * *view_matrix;

    let uniforms = MeshVsUniforms {
        view_proj_matrix: Temporal {
            previous: view_proj_matrix,
            current: view_proj_matrix,
        },
    };

    queue.write_buffer(uniform_buffer, 0, bytes_of(&uniforms));
}

fn proj_matrix(size: PhysicalSize<u32>) -> cgmath::Matrix4<f32> {
    let aspect_ratio = size.width as f32 / size.height as f32;
    OPENGL_TO_WGPU_MATRIX * cgmath::perspective(cgmath::Deg(45f32), aspect_ratio, 1.0, 1000.0)
}

fn view_matrix() -> cgmath::Matrix4<f32> {
    cgmath::Matrix4::look_at_rh(
        cgmath::Point3::new(0f32, 4.0, 32.0),
        cgmath::Point3::new(0f32, 0.0, 0.0),
        cgmath::Vector3::unit_y(),
    )
}

fn create_color_texture(device: &wgpu::Device, size: PhysicalSize<u32>) -> wgpu::TextureView {
    device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("color_texture"),
            size: wgpu::Extent3d {
                width: size.width,
                height: size.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: COLOR_TEXTURE_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::STORAGE_BINDING,
        })
        .create_view(&wgpu::TextureViewDescriptor::default())
}

fn create_depth_texture(device: &wgpu::Device, size: PhysicalSize<u32>) -> wgpu::TextureView {
    device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("depth_texture"),
            size: wgpu::Extent3d {
                width: size.width,
                height: size.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_TEXTURE_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        })
        .create_view(&wgpu::TextureViewDescriptor::default())
}

fn create_normal_texture(device: &wgpu::Device, size: PhysicalSize<u32>) -> wgpu::TextureView {
    device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("normal_texture"),
            size: wgpu::Extent3d {
                width: size.width,
                height: size.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: NORMAL_TEXTURE_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::STORAGE_BINDING,
        })
        .create_view(&wgpu::TextureViewDescriptor::default())
}

#[tokio::main]
async fn main() {
    env_logger::init();

    let event_loop = EventLoop::new();
    let window = Window::new(&event_loop).unwrap();

    let instance = wgpu::Instance::new(wgpu::Backends::VULKAN);
    let surface = unsafe { instance.create_surface(&window) };

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        })
        .await
        .unwrap();

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                features: wgpu::Features::SPIRV_SHADER_PASSTHROUGH
                    | wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES,
                limits: wgpu::Limits {
                    max_push_constant_size: 128,
                    ..wgpu::Limits::default()
                },
            },
            None,
        )
        .await
        .unwrap();

    let size = window.inner_size();
    configure_surface(&device, &surface, size);

    let mut example = SpheresExample::new(&device, &queue, size);

    event_loop.run(move |event, _, control_flow| match &event {
        Event::WindowEvent { event, .. } => match event {
            WindowEvent::CloseRequested => {
                *control_flow = ControlFlow::Exit;
            }
            WindowEvent::Resized(s) => {
                configure_surface(&device, &surface, *s);
                example.resize(&device, &queue, *s);
            }
            WindowEvent::KeyboardInput {
                input:
                    KeyboardInput {
                        virtual_keycode: Some(VirtualKeyCode::Key1),
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => {
                example.set_ssao_enabled(&queue, !example.ssao_enabled);
            }
            _ => {}
        },
        Event::MainEventsCleared => {
            window.request_redraw();
        }
        Event::RedrawRequested(_) => {
            let frame = match surface.get_current_texture() {
                Ok(frame) => frame,
                Err(_) => {
                    configure_surface(&device, &surface, window.inner_size());
                    surface
                        .get_current_texture()
                        .expect("Failed to acquire next surface texture!")
                }
            };

            let frame_output_texture_view = frame
                .texture
                .create_view(&wgpu::TextureViewDescriptor::default());

            example.render(&device, &queue, &frame_output_texture_view);
            frame.present();
        }
        _ => {}
    });
}

fn configure_surface(device: &wgpu::Device, surface: &wgpu::Surface, size: PhysicalSize<u32>) {
    surface.configure(
        device,
        &wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: SURFACE_TEXTURE_FORMAT,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
        },
    )
}

fn align_to<S>(x: S, alignment: S) -> S
where
    S: Copy + Sub<Output = S> + Rem<Output = S> + PartialEq + Zero,
{
    let r = x % alignment;
    if r != S::zero() {
        x - r
    } else {
        x
    }
}
