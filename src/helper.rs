use std::{
    borrow::Cow,
    ops::{Rem, Sub},
};

use num::Zero;
use shaderc::ShaderKind;

pub fn align_to<S>(x: S, alignment: S) -> S
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

pub fn compile_glsl_shader(
    device: &wgpu::Device,
    source: &str,
    shader_kind: ShaderKind,
    macros: &[(&str, Option<&str>)],
) -> Result<wgpu::ShaderModule, shaderc::Error> {
    let mut compiler = shaderc::Compiler::new().unwrap();

    let mut options = shaderc::CompileOptions::new().unwrap();
    options.set_forced_version_profile(460, shaderc::GlslProfile::None);

    for (name, value) in macros.iter().copied() {
        options.add_macro_definition(name, value);
    }

    let spirv =
        compiler.compile_into_spirv(&source, shader_kind, "shader.glsl", "main", Some(&options))?;

    Ok(unsafe {
        device.create_shader_module_spirv(&wgpu::ShaderModuleDescriptorSpirV {
            label: None,
            source: Cow::Borrowed(spirv.as_binary()),
        })
    })
}

// https://github.com/gfx-rs/naga/blob/943235cd5e91df9a1d41c60f525d26734bc0d261/src/back/glsl/mod.rs#L2904
pub fn glsl_storage_format(format: wgpu::TextureFormat) -> Option<&'static str> {
    use wgpu::TextureFormat as Tf;

    match format {
        Tf::R8Unorm => Some("r8"),
        Tf::R8Snorm => Some("r8_snorm"),
        Tf::R8Uint => Some("r8ui"),
        Tf::R8Sint => Some("r8i"),
        Tf::R16Uint => Some("r16ui"),
        Tf::R16Sint => Some("r16i"),
        Tf::R16Float => Some("r16f"),
        Tf::Rg8Unorm => Some("rg8"),
        Tf::Rg8Snorm => Some("rg8_snorm"),
        Tf::Rg8Uint => Some("rg8ui"),
        Tf::Rg8Sint => Some("rg8i"),
        Tf::R32Uint => Some("r32ui"),
        Tf::R32Sint => Some("r32i"),
        Tf::R32Float => Some("r32f"),
        Tf::Rg16Uint => Some("rg16ui"),
        Tf::Rg16Sint => Some("rg16i"),
        Tf::Rg16Float => Some("rg16f"),
        Tf::Rgba8Unorm => Some("rgba8ui"),
        Tf::Rgba8Snorm => Some("rgba8_snorm"),
        Tf::Rgba8Uint => Some("rgba8ui"),
        Tf::Rgba8Sint => Some("rgba8i"),
        Tf::Rgb10a2Unorm => Some("rgb10_a2ui"),
        Tf::Rg11b10Float => Some("r11f_g11f_b10f"),
        Tf::Rg32Uint => Some("rg32ui"),
        Tf::Rg32Sint => Some("rg32i"),
        Tf::Rg32Float => Some("rg32f"),
        Tf::Rgba16Uint => Some("rgba16ui"),
        Tf::Rgba16Sint => Some("rgba16i"),
        Tf::Rgba16Float => Some("rgba16f"),
        Tf::Rgba32Uint => Some("rgba32ui"),
        Tf::Rgba32Sint => Some("rgba32i"),
        Tf::Rgba32Float => Some("rgba32f"),
        _ => None,
    }
}
