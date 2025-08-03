use std::{error::Error, path::PathBuf};

use ash::{util::read_spv, vk};
use bytemuck::cast_slice;
use rspirv::{
	binary::Assemble,
	dr::{load_words, Builder, Operand},
	spirv::{AddressingModel, Capability, ExecutionModel, MemoryModel},
};
use slang::{
	Blob,
	CompileTarget,
	CompilerOptionEntry,
	CompilerOptionName,
	FileSystem,
	GlobalSession,
	MatrixLayoutMode,
	Module,
	Session,
	SessionDescBuilder,
	TargetDescBuilder,
};

#[derive(Clone)]
struct CacheFs {
	source: PathBuf,
	cache: PathBuf,
}

impl FileSystem for CacheFs {
	fn load_file(&mut self, path: &str) -> Result<Blob, slang::Error> {
		let source = self.source.join(path);
		// let cache = self.cache.join(path);
		// if cache.exists() {
		// 	std::fs::read(cache)
		//} else {
		std::fs::read(source)
		//}
		.map(|x| x.into())
		.map_err(|x| slang::Error::Blob(x.to_string().into()))
	}
}

pub struct ShaderBuilder {
	slang: GlobalSession,
	fs: CacheFs,
	sesh: Session,
}

unsafe impl Send for ShaderBuilder {}

impl ShaderBuilder {
	fn make_session(slang: &GlobalSession, fs: CacheFs) -> Result<Session, slang::Error> {
		slang.create_session(
			SessionDescBuilder::default()
				.targets(&[TargetDescBuilder::default()
					.format(CompileTarget::SPIRV)
					.profile(slang.find_profile("sm_6_6"))
					.force_glsl_scalar_buffer_layout(true)])
				.compiler_option_entries(&mut [
					CompilerOptionEntry::new(CompilerOptionName::EMIT_SPIRV_DIRECTLY, 1),
					// CompilerOptionEntry::new(CompilerOptionName::DEBUG_INFORMATION, 2),
					CompilerOptionEntry::new(CompilerOptionName::OPTIMIZATION, 2),
				])
				.file_system(fs)
				.search_paths(&[c"./".as_ptr() as _])
				.default_matrix_layout_mode(MatrixLayoutMode::COLUMN_MAJOR),
		)
	}

	pub fn new(source: PathBuf, cache: PathBuf) -> Result<Self, Box<dyn Error>> {
		let fs = CacheFs { source, cache };
		let slang = GlobalSession::new()?;
		let sesh = Self::make_session(&slang, fs.clone())?;

		Ok(Self { slang, fs, sesh })
	}

	fn load_raw(&mut self, name: &str) -> Result<Module, String> {
		let path = name.replace('.', "/");
		let mut module = self.sesh.load_module(&path).map_err(fmt_error)?;
		let path = self.fs.cache.join(path).with_extension("slang-module");
		let _ = std::fs::create_dir_all(path.parent().unwrap());
		module
			.write_to_file(path.as_os_str().to_str().unwrap())
			.map_err(|e| format!("error in `{name}`: {}", fmt_error(e)))?;
		Ok(module)
	}

	pub fn reload(&mut self) -> Result<(), String> {
		self.sesh = Self::make_session(&self.slang, self.fs.clone()).map_err(fmt_error)?;
		Ok(())
	}

	pub fn load_module(
		&mut self, name: &str, entry: &str, spec: &[&str],
	) -> Result<(Vec<u32>, vk::ShaderStageFlags), String> {
		let mut module = self.load_raw(name)?;
		let mut sentry = module
			.find_entry_point_by_name(entry)
			.map_err(|e| format!("error in `{name}.{entry}`: {}", fmt_error(e)))?;
		let prog = if spec.is_empty() {
			sentry.link()
		} else {
			let mut others = Vec::with_capacity(spec.len());
			for n in spec {
				let m = self.load_raw(n)?;
				others.push((*m).clone());
			}
			others.push((*sentry).clone());
			let mut speced = self.sesh.create_composite_component_type(&others).map_err(fmt_error)?;
			speced.link()
		}
		.map_err(fmt_error)?;
		let blob = prog.get_entry_point_code(0, 0).map_err(fmt_error)?;

		let spirv = read_spv(&mut std::io::Cursor::new(blob.as_slice())).expect("failed to read spirv");
		let mut builder = Builder::new_from_module(
			load_words(&spirv).map_err(|e| format!("invalid spirv in {name}.{entry} {{{spec:?}}}: {e:?}",))?,
		);
		builder.extension("SPV_KHR_vulkan_memory_model");
		builder.capability(Capability::VulkanMemoryModel);
		builder.extension("SPV_KHR_physical_storage_buffer");
		builder.capability(Capability::PhysicalStorageBufferAddresses);
		builder.memory_model(AddressingModel::PhysicalStorageBuffer64, MemoryModel::Vulkan);
		let module = builder.module();

		let stage = module.entry_points[0]
			.operands
			.iter()
			.find_map(|x| match x {
				Operand::ExecutionModel(m) => Some(match m {
					ExecutionModel::Vertex => vk::ShaderStageFlags::VERTEX,
					ExecutionModel::TessellationControl => vk::ShaderStageFlags::TESSELLATION_CONTROL,
					ExecutionModel::TessellationEvaluation => vk::ShaderStageFlags::TESSELLATION_EVALUATION,
					ExecutionModel::Geometry => vk::ShaderStageFlags::GEOMETRY,
					ExecutionModel::Fragment => vk::ShaderStageFlags::FRAGMENT,
					ExecutionModel::GLCompute => vk::ShaderStageFlags::COMPUTE,
					ExecutionModel::Kernel => panic!("why do you have an opencl shader"),
					ExecutionModel::TaskNV => vk::ShaderStageFlags::TASK_NV,
					ExecutionModel::MeshNV => vk::ShaderStageFlags::MESH_NV,
					ExecutionModel::RayGenerationKHR => vk::ShaderStageFlags::RAYGEN_KHR,
					ExecutionModel::IntersectionKHR => vk::ShaderStageFlags::INTERSECTION_KHR,
					ExecutionModel::AnyHitKHR => vk::ShaderStageFlags::ANY_HIT_KHR,
					ExecutionModel::ClosestHitKHR => vk::ShaderStageFlags::CLOSEST_HIT_KHR,
					ExecutionModel::MissKHR => vk::ShaderStageFlags::MISS_KHR,
					ExecutionModel::CallableKHR => vk::ShaderStageFlags::CALLABLE_KHR,
					ExecutionModel::TaskEXT => vk::ShaderStageFlags::TASK_EXT,
					ExecutionModel::MeshEXT => vk::ShaderStageFlags::MESH_EXT,
				}),
				_ => None,
			})
			.unwrap();

		let ret = module.assemble();

		let path = if spec.is_empty() {
			format!("{name}.{entry}.spv")
		} else {
			format!("{}.{}+{}.spv", name, entry, spec.join(","))
		};
		let f = self.fs.cache.join("spirv/");
		let _ = std::fs::create_dir_all(&f);
		let _ = std::fs::write(f.join(path), cast_slice(&ret));

		Ok((ret, stage))
	}
}

fn fmt_error(err: slang::Error) -> String {
	match err {
		slang::Error::Result(x) => format!("slang error: {x}"),
		slang::Error::Blob(d) => d.as_str().unwrap().to_string(),
	}
}
