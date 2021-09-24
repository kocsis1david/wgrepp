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
