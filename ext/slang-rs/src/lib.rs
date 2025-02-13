mod impls;
mod utils;

pub mod sys {
    pub use slang_sys::*;
}

use std::{
    ffi::{c_char, c_void, CStr, CString},
    fmt::{Debug, Formatter},
    marker::PhantomData,
    mem,
    ops::{Deref, DerefMut},
    ptr, slice,
    str::Utf8Error,
};

use bitflags::bitflags;

pub use crate::utils::{Error, IntoError};
use crate::{
    impls::{FileSystemImpl, OwnedBlobImpl, StaticBlobImpl},
    sys::{vtable_call, Interface},
    utils::{assert_size_and_align, define_interface},
};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct CompileTarget(i32);

impl CompileTarget {
    pub const UNKNOWN: Self = Self(sys::SlangCompileTarget_SLANG_TARGET_UNKNOWN as _);
    pub const NONE: Self = Self(sys::SlangCompileTarget_SLANG_TARGET_NONE as _);
    pub const GLSL: Self = Self(sys::SlangCompileTarget_SLANG_GLSL as _);
    pub const HLSL: Self = Self(sys::SlangCompileTarget_SLANG_HLSL as _);
    pub const SPIRV: Self = Self(sys::SlangCompileTarget_SLANG_SPIRV as _);
    pub const SPIRV_ASM: Self = Self(sys::SlangCompileTarget_SLANG_SPIRV_ASM as _);
    pub const DXBC: Self = Self(sys::SlangCompileTarget_SLANG_DXBC as _);
    pub const DXBC_ASM: Self = Self(sys::SlangCompileTarget_SLANG_DXBC_ASM as _);
    pub const DXIL: Self = Self(sys::SlangCompileTarget_SLANG_DXIL as _);
    pub const DXIL_ASM: Self = Self(sys::SlangCompileTarget_SLANG_DXIL_ASM as _);
    pub const C_SOURCE: Self = Self(sys::SlangCompileTarget_SLANG_C_SOURCE as _);
    pub const CPP_SOURCE: Self = Self(sys::SlangCompileTarget_SLANG_CPP_SOURCE as _);
    pub const HOST_EXECUTABLE: Self = Self(sys::SlangCompileTarget_SLANG_HOST_EXECUTABLE as _);
    pub const SHADER_SHARED_LIBRARY: Self =
        Self(sys::SlangCompileTarget_SLANG_SHADER_SHARED_LIBRARY as _);
    pub const SHADER_HOST_CALLABLE: Self =
        Self(sys::SlangCompileTarget_SLANG_SHADER_HOST_CALLABLE as _);
    pub const CUDA_SOURCE: Self = Self(sys::SlangCompileTarget_SLANG_CUDA_SOURCE as _);
    pub const PTX: Self = Self(sys::SlangCompileTarget_SLANG_PTX as _);
    pub const CUDA_OBJECT_CODE: Self = Self(sys::SlangCompileTarget_SLANG_CUDA_OBJECT_CODE as _);
    pub const OBJECT_CODE: Self = Self(sys::SlangCompileTarget_SLANG_OBJECT_CODE as _);
    pub const HOST_CPP_SOURCE: Self = Self(sys::SlangCompileTarget_SLANG_HOST_CPP_SOURCE as _);
    pub const HOST_HOST_CALLABLE: Self =
        Self(sys::SlangCompileTarget_SLANG_HOST_HOST_CALLABLE as _);
    pub const CPP_PYTORCH_BINDING: Self =
        Self(sys::SlangCompileTarget_SLANG_CPP_PYTORCH_BINDING as _);
    pub const TARGET_COUNT_OF: Self = Self(sys::SlangCompileTarget_SLANG_TARGET_COUNT_OF as _);
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct ProfileID(u32);

impl ProfileID {
    pub const UNKNOWN: Self = Self(sys::SlangProfileID_SLANG_PROFILE_UNKNOWN as _);
}

bitflags! {
    #[repr(transparent)]
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
    pub struct TargetFlags : u32 {
        const PARAMETER_BLOCKS_USE_REGISTER_SPACES = sys::SLANG_TARGET_FLAG_PARAMETER_BLOCKS_USE_REGISTER_SPACES as _;
        const GENERATE_WHOLE_PROGRAM = sys::SLANG_TARGET_FLAG_GENERATE_WHOLE_PROGRAM as _;
        const DUMP_IR = sys::SLANG_TARGET_FLAG_DUMP_IR as _;
        const GENERATE_SPIRV_DIRECTLY = sys::SLANG_TARGET_FLAG_GENERATE_SPIRV_DIRECTLY as _;
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct FloatingPointMode(u32);

impl FloatingPointMode {
    pub const DEFAULT: Self =
        Self(sys::SlangFloatingPointMode_SLANG_FLOATING_POINT_MODE_DEFAULT as _);
    pub const FAST: Self = Self(sys::SlangFloatingPointMode_SLANG_FLOATING_POINT_MODE_FAST as _);
    pub const PRECISE: Self =
        Self(sys::SlangFloatingPointMode_SLANG_FLOATING_POINT_MODE_PRECISE as _);
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct LineDirectiveMode(u32);

impl LineDirectiveMode {
    pub const DEFAULT: Self =
        Self(sys::SlangLineDirectiveMode_SLANG_LINE_DIRECTIVE_MODE_DEFAULT as _);
    pub const NONE: Self = Self(sys::SlangLineDirectiveMode_SLANG_LINE_DIRECTIVE_MODE_NONE as _);
    pub const STANDARD: Self =
        Self(sys::SlangLineDirectiveMode_SLANG_LINE_DIRECTIVE_MODE_STANDARD as _);
    pub const GLSL: Self = Self(sys::SlangLineDirectiveMode_SLANG_LINE_DIRECTIVE_MODE_GLSL as _);
    pub const SOURCE_MAP: Self =
        Self(sys::SlangLineDirectiveMode_SLANG_LINE_DIRECTIVE_MODE_SOURCE_MAP as _);
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct CompilerOptionName(i32);

impl CompilerOptionName {
    pub const MACRO_DEFINE: Self = Self(sys::slang_CompilerOptionName_MacroDefine as _);
    pub const DEP_FILE: Self = Self(sys::slang_CompilerOptionName_DepFile as _);
    pub const ENTRY_POINT_NAME: Self = Self(sys::slang_CompilerOptionName_EntryPointName as _);
    pub const SPECIALIZE: Self = Self(sys::slang_CompilerOptionName_Specialize as _);
    pub const HELP: Self = Self(sys::slang_CompilerOptionName_Help as _);
    pub const HELP_STYLE: Self = Self(sys::slang_CompilerOptionName_HelpStyle as _);
    pub const INCLUDE: Self = Self(sys::slang_CompilerOptionName_Include as _);
    pub const LANGUAGE: Self = Self(sys::slang_CompilerOptionName_Language as _);
    pub const MATRIX_LAYOUT_COLUMN: Self =
        Self(sys::slang_CompilerOptionName_MatrixLayoutColumn as _);
    pub const MATRIX_LAYOUT_ROW: Self = Self(sys::slang_CompilerOptionName_MatrixLayoutRow as _);
    pub const ZERO_INITIALIZE: Self = Self(sys::slang_CompilerOptionName_ZeroInitialize as _);
    pub const IGNORE_CAPABILITIES: Self =
        Self(sys::slang_CompilerOptionName_IgnoreCapabilities as _);
    pub const RESTRICTIVE_CAPABILITY_CHECK: Self =
        Self(sys::slang_CompilerOptionName_RestrictiveCapabilityCheck as _);
    pub const MODULE_NAME: Self = Self(sys::slang_CompilerOptionName_ModuleName as _);
    pub const OUTPUT: Self = Self(sys::slang_CompilerOptionName_Output as _);
    pub const PROFILE: Self = Self(sys::slang_CompilerOptionName_Profile as _);
    pub const STAGE: Self = Self(sys::slang_CompilerOptionName_Stage as _);
    pub const TARGET: Self = Self(sys::slang_CompilerOptionName_Target as _);
    pub const VERSION: Self = Self(sys::slang_CompilerOptionName_Version as _);
    pub const WARNINGS_AS_ERRORS: Self = Self(sys::slang_CompilerOptionName_WarningsAsErrors as _);
    pub const DISABLE_WARNINGS: Self = Self(sys::slang_CompilerOptionName_DisableWarnings as _);
    pub const ENABLE_WARNING: Self = Self(sys::slang_CompilerOptionName_EnableWarning as _);
    pub const DISABLE_WARNING: Self = Self(sys::slang_CompilerOptionName_DisableWarning as _);
    pub const DUMP_WARNING_DIAGNOSTICS: Self =
        Self(sys::slang_CompilerOptionName_DumpWarningDiagnostics as _);
    pub const INPUT_FILES_REMAIN: Self = Self(sys::slang_CompilerOptionName_InputFilesRemain as _);
    pub const EMIT_IR: Self = Self(sys::slang_CompilerOptionName_EmitIr as _);
    pub const REPORT_DOWNSTREAM_TIME: Self =
        Self(sys::slang_CompilerOptionName_ReportDownstreamTime as _);
    pub const REPORT_PERF_BENCHMARK: Self =
        Self(sys::slang_CompilerOptionName_ReportPerfBenchmark as _);
    pub const SKIP_SPIRV_VALIDATION: Self =
        Self(sys::slang_CompilerOptionName_SkipSPIRVValidation as _);
    pub const SOURCE_EMBED_STYLE: Self = Self(sys::slang_CompilerOptionName_SourceEmbedStyle as _);
    pub const SOURCE_EMBED_NAME: Self = Self(sys::slang_CompilerOptionName_SourceEmbedName as _);
    pub const SOURCE_EMBED_LANGUAGE: Self =
        Self(sys::slang_CompilerOptionName_SourceEmbedLanguage as _);
    pub const DISABLE_SHORT_CIRCUIT: Self =
        Self(sys::slang_CompilerOptionName_DisableShortCircuit as _);
    pub const MINIMUM_SLANG_OPTIMIZATION: Self =
        Self(sys::slang_CompilerOptionName_MinimumSlangOptimization as _);
    pub const DISABLE_NON_ESSENTIAL_VALIDATIONS: Self =
        Self(sys::slang_CompilerOptionName_DisableNonEssentialValidations as _);
    pub const DISABLE_SOURCE_MAP: Self = Self(sys::slang_CompilerOptionName_DisableSourceMap as _);
    pub const UNSCOPED_ENUM: Self = Self(sys::slang_CompilerOptionName_UnscopedEnum as _);
    pub const PRESERVE_PARAMETERS: Self =
        Self(sys::slang_CompilerOptionName_PreserveParameters as _);
    pub const CAPABILITY: Self = Self(sys::slang_CompilerOptionName_Capability as _);
    pub const DEFAULT_IMAGE_FORMAT_UNKNOWN: Self =
        Self(sys::slang_CompilerOptionName_DefaultImageFormatUnknown as _);
    pub const DISABLE_DYNAMIC_DISPATCH: Self =
        Self(sys::slang_CompilerOptionName_DisableDynamicDispatch as _);
    pub const DISABLE_SPECIALIZATION: Self =
        Self(sys::slang_CompilerOptionName_DisableSpecialization as _);
    pub const FLOATING_POINT_MODE: Self =
        Self(sys::slang_CompilerOptionName_FloatingPointMode as _);
    pub const DEBUG_INFORMATION: Self = Self(sys::slang_CompilerOptionName_DebugInformation as _);
    pub const LINE_DIRECTIVE_MODE: Self =
        Self(sys::slang_CompilerOptionName_LineDirectiveMode as _);
    pub const OPTIMIZATION: Self = Self(sys::slang_CompilerOptionName_Optimization as _);
    pub const OBFUSCATE: Self = Self(sys::slang_CompilerOptionName_Obfuscate as _);
    pub const VULKAN_BIND_SHIFT: Self = Self(sys::slang_CompilerOptionName_VulkanBindShift as _);
    pub const VULKAN_BIND_GLOBALS: Self =
        Self(sys::slang_CompilerOptionName_VulkanBindGlobals as _);
    pub const VULKAN_INVERT_Y: Self = Self(sys::slang_CompilerOptionName_VulkanInvertY as _);
    pub const VULKAN_USE_DX_POSITION_W: Self =
        Self(sys::slang_CompilerOptionName_VulkanUseDxPositionW as _);
    pub const VULKAN_USE_ENTRY_POINT_NAME: Self =
        Self(sys::slang_CompilerOptionName_VulkanUseEntryPointName as _);
    pub const VULKAN_USE_GL_LAYOUT: Self =
        Self(sys::slang_CompilerOptionName_VulkanUseGLLayout as _);
    pub const VULKAN_EMIT_REFLECTION: Self =
        Self(sys::slang_CompilerOptionName_VulkanEmitReflection as _);
    pub const GLSL_FORCE_SCALAR_LAYOUT: Self =
        Self(sys::slang_CompilerOptionName_GLSLForceScalarLayout as _);
    pub const ENABLE_EFFECT_ANNOTATIONS: Self =
        Self(sys::slang_CompilerOptionName_EnableEffectAnnotations as _);
    pub const EMIT_SPIRV_VIA_GLSL: Self = Self(sys::slang_CompilerOptionName_EmitSpirvViaGLSL as _);
    pub const EMIT_SPIRV_DIRECTLY: Self =
        Self(sys::slang_CompilerOptionName_EmitSpirvDirectly as _);
    pub const SPIRV_CORE_GRAMMAR_JSON: Self =
        Self(sys::slang_CompilerOptionName_SPIRVCoreGrammarJSON as _);
    pub const INCOMPLETE_LIBRARY: Self = Self(sys::slang_CompilerOptionName_IncompleteLibrary as _);
    pub const COMPILER_PATH: Self = Self(sys::slang_CompilerOptionName_CompilerPath as _);
    pub const DEFAULT_DOWNSTREAM_COMPILER: Self =
        Self(sys::slang_CompilerOptionName_DefaultDownstreamCompiler as _);
    pub const DOWNSTREAM_ARGS: Self = Self(sys::slang_CompilerOptionName_DownstreamArgs as _);
    pub const PASS_THROUGH: Self = Self(sys::slang_CompilerOptionName_PassThrough as _);
    pub const DUMP_REPRO: Self = Self(sys::slang_CompilerOptionName_DumpRepro as _);
    pub const DUMP_REPRO_ON_ERROR: Self = Self(sys::slang_CompilerOptionName_DumpReproOnError as _);
    pub const EXTRACT_REPRO: Self = Self(sys::slang_CompilerOptionName_ExtractRepro as _);
    pub const LOAD_REPRO: Self = Self(sys::slang_CompilerOptionName_LoadRepro as _);
    pub const LOAD_REPRO_DIRECTORY: Self =
        Self(sys::slang_CompilerOptionName_LoadReproDirectory as _);
    pub const REPRO_FALLBACK_DIRECTORY: Self =
        Self(sys::slang_CompilerOptionName_ReproFallbackDirectory as _);
    pub const DUMP_AST: Self = Self(sys::slang_CompilerOptionName_DumpAst as _);
    pub const DUMP_INTERMEDIATE_PREFIX: Self =
        Self(sys::slang_CompilerOptionName_DumpIntermediatePrefix as _);
    pub const DUMP_INTERMEDIATES: Self = Self(sys::slang_CompilerOptionName_DumpIntermediates as _);
    pub const DUMP_IR: Self = Self(sys::slang_CompilerOptionName_DumpIr as _);
    pub const DUMP_IR_IDS: Self = Self(sys::slang_CompilerOptionName_DumpIrIds as _);
    pub const PREPROCESSOR_OUTPUT: Self =
        Self(sys::slang_CompilerOptionName_PreprocessorOutput as _);
    pub const OUTPUT_INCLUDES: Self = Self(sys::slang_CompilerOptionName_OutputIncludes as _);
    pub const REPRO_FILE_SYSTEM: Self = Self(sys::slang_CompilerOptionName_ReproFileSystem as _);
    pub const SERIAL_IR: Self = Self(sys::slang_CompilerOptionName_SerialIr as _);
    pub const SKIP_CODE_GEN: Self = Self(sys::slang_CompilerOptionName_SkipCodeGen as _);
    pub const VALIDATE_IR: Self = Self(sys::slang_CompilerOptionName_ValidateIr as _);
    pub const VERBOSE_PATHS: Self = Self(sys::slang_CompilerOptionName_VerbosePaths as _);
    pub const VERIFY_DEBUG_SERIAL_IR: Self =
        Self(sys::slang_CompilerOptionName_VerifyDebugSerialIr as _);
    pub const NO_CODE_GEN: Self = Self(sys::slang_CompilerOptionName_NoCodeGen as _);
    pub const FILE_SYSTEM: Self = Self(sys::slang_CompilerOptionName_FileSystem as _);
    pub const HETEROGENEOUS: Self = Self(sys::slang_CompilerOptionName_Heterogeneous as _);
    pub const NO_MANGLE: Self = Self(sys::slang_CompilerOptionName_NoMangle as _);
    pub const NO_HLSL_BINDING: Self = Self(sys::slang_CompilerOptionName_NoHLSLBinding as _);
    pub const NO_HLSL_PACK_CONSTANT_BUFFER_ELEMENTS: Self =
        Self(sys::slang_CompilerOptionName_NoHLSLPackConstantBufferElements as _);
    pub const VALIDATE_UNIFORMITY: Self =
        Self(sys::slang_CompilerOptionName_ValidateUniformity as _);
    pub const ALLOW_GLSL: Self = Self(sys::slang_CompilerOptionName_AllowGLSL as _);
    pub const ARCHIVE_TYPE: Self = Self(sys::slang_CompilerOptionName_ArchiveType as _);
    pub const COMPILE_CORE_MODULE: Self =
        Self(sys::slang_CompilerOptionName_CompileCoreModule as _);
    pub const DOC: Self = Self(sys::slang_CompilerOptionName_Doc as _);
    pub const IR_COMPRESSION: Self = Self(sys::slang_CompilerOptionName_IrCompression as _);
    pub const LOAD_CORE_MODULE: Self = Self(sys::slang_CompilerOptionName_LoadCoreModule as _);
    pub const REFERENCE_MODULE: Self = Self(sys::slang_CompilerOptionName_ReferenceModule as _);
    pub const SAVE_CORE_MODULE: Self = Self(sys::slang_CompilerOptionName_SaveCoreModule as _);
    pub const SAVE_CORE_MODULE_BIN_SOURCE: Self =
        Self(sys::slang_CompilerOptionName_SaveCoreModuleBinSource as _);
    pub const TRACK_LIVENESS: Self = Self(sys::slang_CompilerOptionName_TrackLiveness as _);
    pub const LOOP_INVERSION: Self = Self(sys::slang_CompilerOptionName_LoopInversion as _);
    pub const PARAMETER_BLOCKS_USE_REGISTER_SPACES: Self =
        Self(sys::slang_CompilerOptionName_ParameterBlocksUseRegisterSpaces as _);

    pub const COUNT_OF_PARSABLE_OPTIONS: Self =
        Self(sys::slang_CompilerOptionName_CountOfParsableOptions as _);

    pub const DEBUG_INFORMATION_FORMAT: Self =
        Self(sys::slang_CompilerOptionName_DebugInformationFormat as _);
    pub const VULKAN_BIND_SHIFT_ALL: Self =
        Self(sys::slang_CompilerOptionName_VulkanBindShiftAll as _);
    pub const GENERATE_WHOLE_PROGRAM: Self =
        Self(sys::slang_CompilerOptionName_GenerateWholeProgram as _);
    pub const USE_UP_TO_DATE_BINARY_MODULE: Self =
        Self(sys::slang_CompilerOptionName_UseUpToDateBinaryModule as _);
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct CompilerOptionValueKind(i32);

impl CompilerOptionValueKind {
    pub const INT: Self = Self(sys::slang_CompilerOptionValueKind_Int as _);
    pub const STRING: Self = Self(sys::slang_CompilerOptionValueKind_String as _);
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CompilerOptionValue {
    pub kind: CompilerOptionValueKind,
    pub int_value0: i32,
    pub int_value1: i32,
    pub string_value0: *const c_char,
    pub string_value1: *const c_char,
}

impl Default for CompilerOptionValue {
    #[inline]
    fn default() -> Self {
        unsafe { mem::zeroed() }
    }
}

impl From<i32> for CompilerOptionValue {
    #[inline]
    fn from(value: i32) -> Self {
        Self {
            kind: CompilerOptionValueKind::INT,
            int_value0: value,
            ..Default::default()
        }
    }
}

impl From<(i32, i32)> for CompilerOptionValue {
    #[inline]
    fn from((first, second): (i32, i32)) -> Self {
        Self {
            kind: CompilerOptionValueKind::INT,
            int_value0: first,
            int_value1: second,
            ..Default::default()
        }
    }
}

impl From<*const c_char> for CompilerOptionValue {
    #[inline]
    fn from(value: *const c_char) -> Self {
        Self {
            kind: CompilerOptionValueKind::STRING,
            string_value0: value,
            ..Default::default()
        }
    }
}

impl From<(*const c_char, *const c_char)> for CompilerOptionValue {
    #[inline]
    fn from((first, second): (*const c_char, *const c_char)) -> Self {
        Self {
            kind: CompilerOptionValueKind::STRING,
            string_value0: first,
            string_value1: second,
            ..Default::default()
        }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CompilerOptionEntry {
    pub name: CompilerOptionName,
    pub value: CompilerOptionValue,
}

impl CompilerOptionEntry {
    #[inline]
    pub fn new(name: CompilerOptionName, value: impl Into<CompilerOptionValue>) -> Self {
        Self {
            name,
            value: value.into(),
        }
    }
}

impl Default for CompilerOptionEntry {
    #[inline]
    fn default() -> Self {
        unsafe { mem::zeroed() }
    }
}

assert_size_and_align!(CompilerOptionEntry, sys::slang_CompilerOptionEntry);

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct TargetDesc {
    pub structure_size: usize,
    pub format: CompileTarget,
    pub profile: ProfileID,
    pub flags: TargetFlags,
    pub floating_point_mode: FloatingPointMode,
    pub line_directive_mode: LineDirectiveMode,
    pub force_glsl_scalar_buffer_layout: bool,
    pub compiler_option_entries: *mut CompilerOptionEntry,
    pub compiler_option_entry_count: u32,
}

impl Default for TargetDesc {
    #[inline]
    fn default() -> Self {
        unsafe { mem::zeroed() }
    }
}

assert_size_and_align!(TargetDesc, sys::slang_TargetDesc);

#[repr(transparent)]
pub struct TargetDescBuilder<'a>(TargetDesc, PhantomData<&'a ()>);

impl<'a> TargetDescBuilder<'a> {
    #[inline]
    pub fn format(mut self, format: CompileTarget) -> Self {
        self.0.format = format;
        self
    }

    #[inline]
    pub fn profile(mut self, profile: ProfileID) -> Self {
        self.0.profile = profile;
        self
    }

    #[inline]
    pub fn flags(mut self, flags: TargetFlags) -> Self {
        self.0.flags = flags;
        self
    }

    #[inline]
    pub fn floating_point_mode(mut self, floating_point_mode: FloatingPointMode) -> Self {
        self.0.floating_point_mode = floating_point_mode;
        self
    }

    #[inline]
    pub fn line_directive_mode(mut self, line_directive_mode: LineDirectiveMode) -> Self {
        self.0.line_directive_mode = line_directive_mode;
        self
    }

    #[inline]
    pub fn force_glsl_scalar_buffer_layout(
        mut self,
        force_glsl_scalar_buffer_layout: bool,
    ) -> Self {
        self.0.force_glsl_scalar_buffer_layout = force_glsl_scalar_buffer_layout;
        self
    }

    #[inline]
    pub fn compiler_option_entries(
        mut self,
        compiler_option_entries: &'a mut [CompilerOptionEntry],
    ) -> Self {
        self.0.compiler_option_entries = compiler_option_entries.as_mut_ptr();
        self.0.compiler_option_entry_count = compiler_option_entries.len() as _;
        self
    }
}

impl Default for TargetDescBuilder<'_> {
    #[inline]
    fn default() -> Self {
        Self(
            TargetDesc {
                structure_size: mem::size_of::<TargetDesc>(),
                ..Default::default()
            },
            PhantomData,
        )
    }
}

assert_size_and_align!(TargetDesc, sys::slang_TargetDesc);

bitflags! {
    #[repr(transparent)]
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
    pub struct SessionFlags : u32 {
        const NONE = sys::kSessionFlags_None as _;
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct MatrixLayoutMode(u32);

impl MatrixLayoutMode {
    pub const UNKNOWN: Self =
        Self(sys::SlangMatrixLayoutMode_SLANG_MATRIX_LAYOUT_MODE_UNKNOWN as _);
    pub const ROW_MAJOR: Self = Self(sys::SlangMatrixLayoutMode_SLANG_MATRIX_LAYOUT_ROW_MAJOR as _);
    pub const COLUMN_MAJOR: Self =
        Self(sys::SlangMatrixLayoutMode_SLANG_MATRIX_LAYOUT_COLUMN_MAJOR as _);
}

pub trait FileSystem {
    fn load_file(&mut self, path: &str) -> utils::Result<Blob>;
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct PreprocessorMacroDesc {
    pub name: *const c_char,
    pub value: *const c_char,
}

impl Default for PreprocessorMacroDesc {
    #[inline]
    fn default() -> Self {
        unsafe { mem::zeroed() }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct SessionDesc {
    pub structure_size: usize,
    pub targets: *const TargetDesc,
    pub target_count: i64,
    pub flags: SessionFlags,
    pub default_matrix_layout_mode: MatrixLayoutMode,
    pub search_paths: *const *const c_char,
    pub search_path_count: i64,
    pub preprocessor_macros: *const PreprocessorMacroDesc,
    pub preprocessor_macro_count: i64,
    pub file_system: *mut sys::ISlangFileSystem,
    pub enable_effect_annotations: bool,
    pub allow_glsl_syntax: bool,
    pub compiler_option_entries: *mut CompilerOptionEntry,
    pub compiler_option_entry_count: u32,
}

impl Default for SessionDesc {
    #[inline]
    fn default() -> Self {
        unsafe { mem::zeroed() }
    }
}

assert_size_and_align!(SessionDesc, sys::slang_SessionDesc);

pub struct SessionDescBuilder<'a> {
    desc: SessionDesc,
    file_system: Option<Box<dyn FileSystem>>,
    phantom: PhantomData<&'a ()>,
}

impl<'a> SessionDescBuilder<'a> {
    #[inline]
    pub fn targets(mut self, targets: &'a [TargetDescBuilder]) -> Self {
        self.desc.targets = targets.as_ptr().cast();
        self.desc.target_count = targets.len() as _;
        self
    }

    #[inline]
    pub fn search_paths(mut self, search_paths: &'a [*const c_char]) -> Self {
        self.desc.search_paths = search_paths.as_ptr();
        self.desc.search_path_count = search_paths.len() as _;
        self
    }

    #[inline]
    pub fn flags(mut self, flags: SessionFlags) -> Self {
        self.desc.flags = flags;
        self
    }

    #[inline]
    pub fn default_matrix_layout_mode(
        mut self,
        default_matrix_layout_mode: MatrixLayoutMode,
    ) -> Self {
        self.desc.default_matrix_layout_mode = default_matrix_layout_mode;
        self
    }

    #[inline]
    pub fn enable_effect_annotations(mut self, enable_effect_annotations: bool) -> Self {
        self.desc.enable_effect_annotations = enable_effect_annotations;
        self
    }

    #[inline]
    pub fn file_system(mut self, file_system: impl FileSystem + 'static) -> Self {
        self.file_system = Some(Box::new(file_system));
        self
    }

    #[inline]
    pub fn allow_glsl_syntax(mut self, allow_glsl_syntax: bool) -> Self {
        self.desc.allow_glsl_syntax = allow_glsl_syntax;
        self
    }

    #[inline]
    pub fn compiler_option_entries(
        mut self,
        compiler_option_entries: &'a mut [CompilerOptionEntry],
    ) -> Self {
        self.desc.compiler_option_entries = compiler_option_entries.as_mut_ptr();
        self.desc.compiler_option_entry_count = compiler_option_entries.len() as _;
        self
    }
}

impl Default for SessionDescBuilder<'_> {
    #[inline]
    fn default() -> Self {
        Self {
            desc: SessionDesc {
                structure_size: mem::size_of::<SessionDesc>(),
                ..Default::default()
            },
            file_system: None,
            phantom: PhantomData,
        }
    }
}

assert_size_and_align!(SessionDesc, sys::slang_SessionDesc);

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Stage(u32);

impl Stage {
    pub const NONE: Self = Self(sys::SlangStage_SLANG_STAGE_NONE);
    pub const VERTEX: Self = Self(sys::SlangStage_SLANG_STAGE_VERTEX);
    pub const HULL: Self = Self(sys::SlangStage_SLANG_STAGE_HULL);
    pub const DOMAIN: Self = Self(sys::SlangStage_SLANG_STAGE_DOMAIN);
    pub const GEOMETRY: Self = Self(sys::SlangStage_SLANG_STAGE_GEOMETRY);
    pub const FRAGMENT: Self = Self(sys::SlangStage_SLANG_STAGE_FRAGMENT);
    pub const COMPUTE: Self = Self(sys::SlangStage_SLANG_STAGE_COMPUTE);
    pub const RAY_GENERATION: Self = Self(sys::SlangStage_SLANG_STAGE_RAY_GENERATION);
    pub const INTERSECTION: Self = Self(sys::SlangStage_SLANG_STAGE_INTERSECTION);
    pub const ANY_HIT: Self = Self(sys::SlangStage_SLANG_STAGE_ANY_HIT);
    pub const CLOSEST_HIT: Self = Self(sys::SlangStage_SLANG_STAGE_CLOSEST_HIT);
    pub const MISS: Self = Self(sys::SlangStage_SLANG_STAGE_MISS);
    pub const CALLABLE: Self = Self(sys::SlangStage_SLANG_STAGE_CALLABLE);
    pub const MESH: Self = Self(sys::SlangStage_SLANG_STAGE_MESH);
    pub const AMPLIFICATION: Self = Self(sys::SlangStage_SLANG_STAGE_AMPLIFICATION);
    pub const PIXEL: Self = Self(sys::SlangStage_SLANG_STAGE_PIXEL);
}

define_interface!(EntryPoint, sys::slang_IEntryPoint, ComponentType);

impl EntryPoint {}

define_interface!(Module, sys::slang_IModule, ComponentType);

impl Module {
    #[inline]
    pub fn find_entry_point_by_name(&mut self, name: &str) -> utils::Result<EntryPoint> {
        let name = CString::new(name).unwrap();
        let mut entry_point = ptr::null_mut();
        utils::result_from_ffi(unsafe {
            vtable_call!(
                self.0,
                findEntryPointByName(name.as_ptr(), &mut entry_point)
            )
        })?;
        Ok(EntryPoint(entry_point))
    }

    #[inline]
    pub fn get_defined_entry_point_count(&self) -> i32 {
        unsafe { vtable_call!(self.0, getDefinedEntryPointCount()) }
    }

    #[inline]
    pub fn get_defined_entry_point(&mut self, index: i32) -> utils::Result<EntryPoint> {
        let mut entry_point = ptr::null_mut();
        utils::result_from_ffi(unsafe {
            vtable_call!(self.0, getDefinedEntryPoint(index, &mut entry_point))
        })?;
        Ok(EntryPoint(entry_point))
    }

    #[inline]
    pub fn serialize(&mut self) -> utils::Result<Blob> {
        let mut blob = ptr::null_mut();
        utils::result_from_ffi(unsafe { vtable_call!(self.0, serialize(&mut blob)) })?;
        Ok(Blob(blob))
    }

    #[inline]
    pub fn write_to_file(&mut self, file_name: &str) -> utils::Result<()> {
        let file_name = CString::new(file_name).unwrap();
        utils::result_from_ffi(unsafe { vtable_call!(self.0, writeToFile(file_name.as_ptr())) })
    }

    #[inline]
    pub fn get_name(&mut self) -> &str {
        let c_str = unsafe { CStr::from_ptr(vtable_call!(self.0, getName())) };
        c_str.to_str().unwrap()
    }

    #[inline]
    pub fn get_file_path(&mut self) -> &str {
        let c_str = unsafe { CStr::from_ptr(vtable_call!(self.0, getFilePath())) };
        c_str.to_str().unwrap()
    }

    #[inline]
    pub fn get_unique_identity(&mut self) -> &str {
        let c_str = unsafe { CStr::from_ptr(vtable_call!(self.0, getUniqueIdentity())) };
        c_str.to_str().unwrap()
    }

    #[inline]
    pub fn find_and_check_entry_point(
        &mut self,
        name: &str,
        stage: Stage,
    ) -> utils::Result<EntryPoint> {
        let name = CString::new(name).unwrap();
        let mut entry_point = ptr::null_mut();
        let mut diagnostics = ptr::null_mut();

        utils::result_from_blob(
            unsafe {
                vtable_call!(
                    self.0,
                    findAndCheckEntryPoint(
                        name.as_ptr(),
                        stage.0,
                        &mut entry_point,
                        &mut diagnostics
                    )
                )
            },
            diagnostics,
        )?;
        Ok(EntryPoint(entry_point))
    }

    #[inline]
    pub fn get_dependency_files(&mut self) -> Vec<&str> {
        unsafe {
            let count = vtable_call!(self.0, getDependencyFileCount());
            let mut files = vec![""; count as usize];
            for i in 0..count {
                files[i as usize] = CStr::from_ptr(vtable_call!(self.0, getDependencyFilePath(i)))
                    .to_str()
                    .unwrap();
            }
            files
        }
    }
}

define_interface!(Blob, sys::slang_IBlob);

impl Blob {
    #[inline]
    pub fn get_buffer_pointer(&self) -> *const c_void {
        unsafe { vtable_call!(self.0, getBufferPointer()) }
    }

    #[inline]
    pub fn get_buffer_size(&self) -> usize {
        unsafe { vtable_call!(self.0, getBufferSize()) }
    }

    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        unsafe { slice::from_raw_parts(self.get_buffer_pointer().cast(), self.get_buffer_size()) }
    }

    #[inline]
    pub fn as_str(&self) -> Result<&str, Utf8Error> {
        std::str::from_utf8(self.as_slice())
    }
}

impl Debug for Blob {
    #[inline]
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self.as_str() {
            Ok(text) => write!(f, "{text}"),
            Err(_) => self.0.fmt(f),
        }
    }
}

impl From<&'static [u8]> for Blob {
    #[inline]
    fn from(value: &'static [u8]) -> Self {
        let blob = Box::leak(Box::new(StaticBlobImpl::new(value)));

        Blob(blob as *mut _ as *mut _)
    }
}

impl From<&'static str> for Blob {
    #[inline]
    fn from(value: &'static str) -> Self {
        Self::from(value.as_bytes())
    }
}

impl From<Vec<u8>> for Blob {
    #[inline]
    fn from(value: Vec<u8>) -> Self {
        let blob = Box::leak(Box::new(OwnedBlobImpl::new(value)));

        Blob(blob as *mut _ as *mut _)
    }
}

impl From<String> for Blob {
    #[inline]
    fn from(value: String) -> Self {
        Self::from(value.into_bytes())
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct SpecializationArgKind(i32);

impl SpecializationArgKind {
    pub const UNKNOWN: Self = Self(sys::slang_SpecializationArg_Kind_Unknown);
    pub const TYPE: Self = Self(sys::slang_SpecializationArg_Kind_Type);
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct SpecializationArg {
    pub kind: SpecializationArgKind,
    pub ty: *mut c_void, //TODO:
}

impl Default for SpecializationArg {
    #[inline]
    fn default() -> Self {
        unsafe { mem::zeroed() }
    }
}

assert_size_and_align!(SpecializationArg, sys::slang_SpecializationArg);

define_interface!(ProgramLayout, sys::slang_ProgramLayout, Debug);

define_interface!(SharedLibrary, sys::ISlangSharedLibrary, Debug);

impl SharedLibrary {
    #[inline]
    pub fn find_symbol_address_by_name(&self, name: &str) -> *mut c_void {
        let name = CString::new(name).unwrap();

        unsafe { vtable_call!(self.0, findSymbolAddressByName(name.as_ptr())) }
    }
}

define_interface!(ComponentType, sys::slang_IComponentType, Debug);

impl ComponentType {
    #[inline]
    pub fn get_session(&mut self) -> Session {
        Session(unsafe { vtable_call!(self.0, getSession()) })
    }

    #[inline]
    pub fn get_layout(&mut self, target_index: i64) -> Result<ProgramLayout, Error> {
        let mut diagnostics = ptr::null_mut();

        let program_layout =
            unsafe { vtable_call!(self.0, getLayout(target_index, &mut diagnostics)) };

        if program_layout.is_null() {
            Err(Error::Blob(Blob(diagnostics)))
        } else {
            Ok(ProgramLayout(program_layout))
        }
    }

    #[inline]
    pub fn get_specialization_param_count(&self) -> i64 {
        unsafe { vtable_call!(self.0, getSpecializationParamCount()) }
    }

    #[inline]
    pub fn get_entry_point_code(
        &self,
        entry_point_index: i64,
        target_index: i64,
    ) -> utils::Result<Blob> {
        let mut code = ptr::null_mut();
        let mut diagnostics = ptr::null_mut();

        utils::result_from_blob(
            unsafe {
                vtable_call!(
                    self.0,
                    getEntryPointCode(entry_point_index, target_index, &mut code, &mut diagnostics)
                )
            },
            diagnostics,
        )?;

        Ok(Blob(code))
    }

    //TODO: getResultAsFileSystem

    #[inline]
    pub fn get_entry_point_hash(&self, entry_point_index: i64, target_index: i64) -> Blob {
        let mut hash = ptr::null_mut();

        unsafe {
            vtable_call!(
                self.0,
                getEntryPointHash(entry_point_index, target_index, &mut hash)
            )
        }

        Blob(hash)
    }

    #[inline]
    pub fn specialize(
        &mut self,
        specialization_args: &[SpecializationArg],
    ) -> utils::Result<ComponentType> {
        let mut specialized_component_type = ptr::null_mut();
        let mut diagnostics = ptr::null_mut();

        utils::result_from_blob(
            unsafe {
                vtable_call!(
                    self.0,
                    specialize(
                        specialization_args.as_ptr().cast(),
                        specialization_args.len() as _,
                        &mut specialized_component_type,
                        &mut diagnostics
                    )
                )
            },
            diagnostics,
        )?;

        Ok(ComponentType(specialized_component_type))
    }

    #[inline]
    pub fn link(&mut self) -> utils::Result<ComponentType> {
        let mut linked_component_type = ptr::null_mut();
        let mut diagnostics = ptr::null_mut();

        utils::result_from_blob(
            unsafe { vtable_call!(self.0, link(&mut linked_component_type, &mut diagnostics)) },
            diagnostics,
        )?;

        Ok(ComponentType(linked_component_type))
    }

    #[inline]
    pub fn get_entry_point_host_callable(
        &mut self,
        entry_point_index: i64,
        target_index: i64,
    ) -> utils::Result<SharedLibrary> {
        let mut shared_library = ptr::null_mut();
        let mut diagnostics = ptr::null_mut();

        utils::result_from_blob(
            unsafe {
                vtable_call!(
                    self.0,
                    getEntryPointHostCallable(
                        entry_point_index,
                        target_index,
                        &mut shared_library,
                        &mut diagnostics
                    )
                )
            },
            diagnostics,
        )?;

        Ok(SharedLibrary(shared_library))
    }

    #[inline]
    pub fn rename_entry_point(&mut self, new_name: &str) -> utils::Result<ComponentType> {
        let mut entry_point = ptr::null_mut();

        let new_name = CString::new(new_name).unwrap();

        utils::result_from_ffi(unsafe {
            vtable_call!(
                self.0,
                renameEntryPoint(new_name.as_ptr(), &mut entry_point)
            )
        })?;

        Ok(ComponentType(entry_point))
    }

    #[inline]
    pub fn link_with_options(
        &mut self,
        compiler_option_entries: &[CompilerOptionEntry],
    ) -> utils::Result<ComponentType> {
        let mut linked_component_type = ptr::null_mut();
        let mut diagnostics = ptr::null_mut();

        utils::result_from_blob(
            unsafe {
                vtable_call!(
                    self.0,
                    linkWithOptions(
                        &mut linked_component_type,
                        compiler_option_entries.len() as _,
                        compiler_option_entries
                            .as_ptr()
                            .cast::<sys::slang_CompilerOptionEntry>()
                            as *mut _,
                        &mut diagnostics
                    )
                )
            },
            diagnostics,
        )?;

        Ok(ComponentType(linked_component_type))
    }
}

define_interface!(Session, sys::slang_ISession, Debug);

impl Session {
    #[inline]
    pub fn get_global_session(&mut self) -> GlobalSession {
        GlobalSession(unsafe { vtable_call!(self.0, getGlobalSession()) })
    }

    #[inline]
    pub fn load_module(&mut self, module_name: &str) -> utils::Result<Module> {
        let module_name = CString::new(module_name).unwrap();
        let mut diagnostics = ptr::null_mut();

        let module =
            unsafe { vtable_call!(self.0, loadModule(module_name.as_ptr(), &mut diagnostics)) };

        if !diagnostics.is_null() {
            Err(Error::Blob(Blob(diagnostics)))
        } else {
            Ok(Module(module))
        }
    }

    #[inline]
    pub fn load_module_from_source(
        &mut self,
        module_name: &str,
        path: &str,
        source: &Blob,
    ) -> utils::Result<Module> {
        let module_name = CString::new(module_name).unwrap();
        let path = CString::new(path).unwrap();
        let mut diagnostics = ptr::null_mut();

        let module = unsafe {
            vtable_call!(
                self.0,
                loadModuleFromSource(
                    module_name.as_ptr(),
                    path.as_ptr(),
                    source.0,
                    &mut diagnostics
                )
            )
        };

        if module.is_null() {
            Err(Error::Blob(Blob(diagnostics)))
        } else {
            Ok(Module(module))
        }
    }

    #[inline]
    pub fn create_composite_component_type(
        &mut self,
        component_types: &[ComponentType],
    ) -> utils::Result<ComponentType> {
        let mut composite_component_type = ptr::null_mut();
        let mut diagnostics = ptr::null_mut();

        utils::result_from_blob(
            unsafe {
                vtable_call!(
                    self.0,
                    createCompositeComponentType(
                        component_types.as_ptr().cast(),
                        component_types.len() as _,
                        &mut composite_component_type,
                        &mut diagnostics
                    )
                )
            },
            diagnostics,
        )?;
        Ok(ComponentType(composite_component_type))
    }

    //TODO: specializeType

    //TODO: getTypeLayout

    //TODO: getContainerType

    //TODO: getDynamicType

    //TODO: getTypeRTTIMangledName

    //TODO: getTypeConformanceWitnessMangledName

    //TODO: getTypeConformanceWitnessSequentialID
}

define_interface!(GlobalSession, sys::slang_IGlobalSession, Debug);

impl GlobalSession {
    #[inline]
    pub fn new() -> utils::Result<Self> {
        let mut session = ptr::null_mut();
        utils::result_from_ffi(unsafe {
            sys::slang_createGlobalSession(sys::SLANG_API_VERSION as _, &mut session)
        })?;

        Ok(Self(session))
    }

    #[inline]
    pub fn new_without_stdlib() -> utils::Result<Self> {
        let mut session = ptr::null_mut();
        utils::result_from_ffi(unsafe {
            sys::slang_createGlobalSessionWithoutCoreModule(
                sys::SLANG_API_VERSION as _,
                &mut session,
            )
        })?;

        Ok(Self(session))
    }

    #[inline]
    pub fn create_session(&self, builder: SessionDescBuilder) -> utils::Result<Session> {
        let mut desc = builder.desc;
        if let Some(file_system) = builder.file_system {
            let file_system = Box::leak(Box::new(FileSystemImpl::new(file_system)));

            desc.file_system = file_system as *mut _ as *mut _;
        }

        let mut session = ptr::null_mut();
        utils::result_from_ffi(unsafe {
            vtable_call!(
                self.0,
                createSession(&desc as *const SessionDesc as *const _, &mut session)
            )
        })?;
        Ok(Session(session))
    }

    #[inline]
    pub fn find_profile(&self, name: &str) -> ProfileID {
        let name = CString::new(name).unwrap();
        ProfileID(unsafe { vtable_call!(self.0, findProfile(name.as_ptr())) })
    }
}
