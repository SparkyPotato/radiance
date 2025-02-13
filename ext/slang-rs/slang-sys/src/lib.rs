#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals)]

use std::ffi::{c_char, c_int, c_void};

pub use interface::*;

mod interface;

include!("../gen/bindings.rs");

interface!(slang_ProgramLayout, [0xc140b5fd, 0x0c78, 0x452e, {0xba, 0x7c, 0x1a, 0x1e, 0x70, 0xc7, 0xf7, 0x1c}], {

});

interface!(slang_IGlobalSession, [0xc140b5fd, 0x0c78, 0x452e, {0xba, 0x7c, 0x1a, 0x1e, 0x70, 0xc7, 0xf7, 0x1c}], {
    createSession: unsafe extern "C" fn(*mut slang_IGlobalSession, desc: *const slang_SessionDesc, outSession: *mut *mut slang_ISession) -> SlangResult,
    findProfile: unsafe extern "C" fn(*mut slang_IGlobalSession, name: *const c_char) -> SlangProfileID,
    setDownstreamCompilerPath: unsafe extern "C" fn(*mut slang_IGlobalSession, passThrough: SlangPassThrough, path: *const c_char),
    setDownstreamCompilerPrelude: unsafe extern "C" fn(*mut slang_IGlobalSession, passThrough: SlangPassThrough, preludeText: *const c_char),
    getDownstreamCompilerPrelude: unsafe extern "C" fn(*mut slang_IGlobalSession, passThrough: SlangPassThrough, outPrelude: *mut *mut ISlangBlob),
    getBuildTagString: unsafe extern "C" fn(*mut slang_IGlobalSession) -> *const c_char,
    setDefaultDownstreamCompiler: unsafe extern "C" fn(*mut slang_IGlobalSession, sourceLanguage: SlangSourceLanguage, defaultCompiler: SlangPassThrough) -> SlangResult,
    getDefaultDownstreamCompiler: unsafe extern "C" fn(*mut slang_IGlobalSession, sourceLanguage: SlangSourceLanguage) -> SlangPassThrough,
    setLanguagePrelude: unsafe extern "C" fn(*mut slang_IGlobalSession, sourceLanguage: SlangSourceLanguage, preludeText: *const c_char),
    getLanguagePrelude: unsafe extern "C" fn(*mut slang_IGlobalSession, sourceLanguage: SlangSourceLanguage, outPrelude: *mut *mut ISlangBlob),
    createCompileRequest: unsafe extern "C" fn(*mut slang_IGlobalSession, *mut *mut slang_ICompileRequest) -> SlangResult,
    addBuiltins: unsafe extern "C" fn(*mut slang_IGlobalSession, sourcePath: *const c_char, sourceString: *const c_char),
    setSharedLibraryLoader: unsafe extern "C" fn(*mut slang_IGlobalSession, loader: *mut ISlangSharedLibraryLoader),
    getSharedLibraryLoader: unsafe extern "C" fn(*mut slang_IGlobalSession) -> *mut ISlangSharedLibraryLoader,
    checkCompileTargetSupport: unsafe extern "C" fn(*mut slang_IGlobalSession, target: SlangCompileTarget) -> SlangResult,
    checkPassThroughSupport: unsafe extern "C" fn(*mut slang_IGlobalSession, passThrough: SlangPassThrough) -> SlangResult,
    compileStdLib: unsafe extern "C" fn(*mut slang_IGlobalSession, flags: slang_CompileCoreModuleFlags) -> SlangResult,
    loadStdLib: unsafe extern "C" fn(*mut slang_IGlobalSession, stdLib: *const c_void, stdLibSizeInBytes: usize) -> SlangResult,
    saveStdLib: unsafe extern "C" fn(*mut slang_IGlobalSession, archiveType: SlangArchiveType, outBlob: *mut *mut ISlangBlob) -> SlangResult,
    findCapability: unsafe extern "C" fn(*mut slang_IGlobalSession, name: *const c_char) -> SlangCapabilityID,
    setDownstreamCompilerForTransition: unsafe extern "C" fn(*mut slang_IGlobalSession, source: SlangCompileTarget, target: SlangCompileTarget, compiler: SlangPassThrough),
    getDownstreamCompilerForTransition: unsafe extern "C" fn(*mut slang_IGlobalSession, source: SlangCompileTarget, target: SlangCompileTarget) -> SlangPassThrough,
    getCompilerElapsedTime: unsafe extern "C" fn(*mut slang_IGlobalSession, outTotalTime: *mut f64, outDownstreamTime: *mut f64),
    setSPIRVCoreGrammar: unsafe extern "C" fn(*mut slang_IGlobalSession, jsonPath: *const c_char) -> SlangResult,
    parseCommandLineArguments: unsafe extern "C" fn(*mut slang_IGlobalSession, argc: c_int, argv: *const *const c_char, outSessionDesc: *mut slang_SessionDesc, outAuxAllocation: *mut *mut ISlangUnknown) -> SlangResult,
    getSessionDescDigest: unsafe extern "C" fn(*mut slang_IGlobalSession, sessionDesc: *const slang_SessionDesc, outBlob: *mut *mut ISlangBlob) -> SlangResult,
});

interface!(slang_ISession, [0x67618701, 0xd116, 0x468f, {0xab, 0x3b, 0x47, 0x4b, 0xed, 0xce, 0xe, 0x3d}], {
    getGlobalSession: unsafe extern "C" fn(*mut slang_ISession) -> *mut slang_IGlobalSession,
    loadModule: unsafe extern "C" fn(*mut slang_ISession, moduleName: *const c_char, outDiagnostics: *mut *mut slang_IBlob) -> *mut slang_IModule,
    loadModuleFromSource: unsafe extern "C" fn(*mut slang_ISession, moduleName: *const c_char, path: *const c_char, source: *mut slang_IBlob, outDiagnostics: *mut *mut slang_IBlob) -> *mut slang_IModule,
    createCompositeComponentType: unsafe extern "C" fn(*mut slang_ISession, componentTypes: *const *const slang_IComponentType, componentTypeCount: SlangInt, outCompositeComponentType: *mut *mut slang_IComponentType, outDiagnostics: *mut *mut ISlangBlob) -> SlangResult,
    specializeType: unsafe extern "C" fn(*mut slang_ISession, type_: *mut slang_TypeReflection, specializationArgs: *const slang_SpecializationArg, specializationArgCount: SlangInt, outDiagnostics: *mut *mut ISlangBlob) -> *mut slang_TypeReflection,
    getTypeLayout: unsafe extern "C" fn(*mut slang_ISession, type_: *mut slang_TypeReflection, targetIndex: SlangInt, rules: slang_LayoutRules, outDiagnostics: *mut *mut ISlangBlob) -> *mut slang_TypeLayoutReflection,
    getDynamicType: unsafe extern "C" fn(*mut slang_ISession) -> *mut slang_TypeReflection,
    getTypeRTTIMangledName: unsafe extern "C" fn(*mut slang_ISession, type_: *mut slang_TypeReflection, outNameBlob: *mut *mut ISlangBlob) -> SlangResult,
    getTypeConformanceWitnessMangledName: unsafe extern "C" fn(*mut slang_ISession, type_: *mut slang_TypeReflection, interfaceType: *mut slang_TypeReflection, outNameBlob: *mut *mut ISlangBlob) -> SlangResult,
    getTypeConformanceWitnessSequentialID: unsafe extern "C" fn(*mut slang_ISession, type_: *mut slang_TypeReflection, interfaceType: *mut slang_TypeReflection, outId: *mut u32) -> SlangResult,
    createCompileRequest: unsafe extern "C" fn(*mut slang_ISession, outCompileRequest: *mut *mut SlangCompileRequest) -> SlangResult,
    createTypeConformanceComponentType: unsafe extern "C" fn(*mut slang_ISession, type_: *mut slang_TypeReflection, interfaceType: *mut slang_TypeReflection, outConformance: *mut *mut slang_ITypeConformance, conformanceIdOverride: SlangInt, outDiagnostics: *mut *mut ISlangBlob) -> SlangResult,
    loadModuleFromIRBlob: unsafe extern "C" fn(*mut slang_ISession, moduleName: *const c_char, path: *const c_char, source: *mut slang_IBlob, outDiagnostics: *mut *mut slang_IBlob) -> *mut slang_IModule,
    getLoadedModuleCount: unsafe extern "C" fn(*mut slang_ISession) -> SlangInt,
    getLoadedModule: unsafe extern "C" fn(*mut slang_ISession, index: SlangInt) -> *mut slang_IModule,
    isBinaryModuleUpToDate: unsafe extern "C" fn(*mut slang_ISession, modulePath: *const c_char, binaryModuleBlob: *mut slang_IBlob) -> bool,
    loadModuleFromSourceString: unsafe extern "C" fn(*mut slang_ISession, moduleName: *const c_char, path: *const c_char, string: *const c_char, outDiagnostics: *mut *mut slang_IBlob) -> *mut slang_IModule,
});

interface!(slang_IBlob, [0x8BA5FB08, 0x5195, 0x40e2, {0xAC, 0x58, 0x0D, 0x98, 0x9C, 0x3A, 0x01, 0x02}], {
    getBufferPointer: unsafe extern "C" fn(*mut slang_IBlob) -> *const c_void,
    getBufferSize: unsafe extern "C" fn(*mut slang_IBlob) -> usize,
});

interface!(ISlangCastable, [0x87ede0e1, 0x4852, 0x44b0, {0x8b, 0xf2, 0xcb, 0x31, 0x87, 0x4d, 0xe2, 0x39}], {
    castAs: unsafe extern "C" fn(*mut ISlangCastable, guid: &SlangUUID) -> *mut c_void,
});

interface!(ISlangFileSystem, [0x003A09FC, 0x3A4D, 0x4BA0, {0xAD, 0x60, 0x1F, 0xD8, 0x63, 0xA9, 0x15, 0xAB}]: ISlangCastable, {
    loadFile: unsafe extern "C" fn(*mut ISlangFileSystem, path: *const c_char, outBlob: *mut *mut slang_IBlob) -> SlangResult,
});

interface!(ISlangFileSystemExt, [0x5fb632d2, 0x979d, 0x4481, {0x9f, 0xee, 0x66, 0x3c, 0x3f, 0x14, 0x49, 0xe1}]: ISlangFileSystem, {
    getFileUniqueIdentity: unsafe extern "C" fn(*mut ISlangFileSystemExt, path: *const c_char, outUniqueIdentity: *mut *mut slang_IBlob) -> SlangResult,
    calcCombinedPath: unsafe extern "C" fn(*mut ISlangFileSystemExt, fromPathType: SlangPathType, fromPath: *const c_char, path: *const c_char, path_out: *mut *mut slang_IBlob) -> SlangResult,
    getPathType: unsafe extern "C" fn(*mut ISlangFileSystemExt, path: *const char, pathTypeOut: SlangPathType) -> SlangResult,
    getPath: unsafe extern "C" fn(*mut ISlangFileSystemExt, kind: PathKind, path: *const char, outPath: *mut *mut slang_IBlob) -> SlangResult,
    clearCache: unsafe extern "C" fn(*mut ISlangFileSystemExt),
    enumeratePathContents: unsafe extern "C" fn(*mut ISlangFileSystemExt, callback: FileSystemContentsCallBack, userData: *mut c_void) -> SlangResult,
});

interface!(ISlangMutableFileSystem, [0xa058675c, 0x1d65, 0x452a, { 0x84, 0x58, 0xcc, 0xde, 0xd1, 0x42, 0x71, 0x5 }]: ISlangFileSystemExt, {
    saveFile: unsafe extern "C" fn(*mut ISlangMutableFileSystem, path: *const c_char, data: *const c_void, size: usize) -> SlangResult,
    saveFileBlob: unsafe extern "C" fn(*mut ISlangMutableFileSystem, path: *const c_char, dataBlob: *mut slang_IBlob) -> SlangResult,
    remove: unsafe extern "C" fn(*mut ISlangMutableFileSystem, path: *const c_char) -> SlangResult,
    createDirectory: unsafe extern "C" fn(*mut ISlangMutableFileSystem, path: *const c_char) -> SlangResult,
});

interface!(ISlangSharedLibrary, [0x70dbc7c4, 0xdc3b, 0x4a07, {0xae, 0x7e, 0x75, 0x2a, 0xf6, 0xa8, 0x15, 0x55}]: ISlangCastable, {
    findSymbolAddressByName: unsafe extern "C" fn(*mut ISlangSharedLibrary, name: *const c_char) -> *mut c_void,
});

interface!(slang_IComponentType, [0x5bc42be8, 0x5c50, 0x4929, {0x9e, 0x5e, 0xd1, 0x5e, 0x7c, 0x24, 0x1, 0x5f}], {
    getSession: unsafe extern "C" fn(*mut slang_IComponentType) -> *mut slang_ISession,
    getLayout: unsafe extern "C" fn(*mut slang_IComponentType, target_index: SlangInt, out_diagnostics: *mut *mut slang_IBlob) -> *mut slang_ProgramLayout,
    getSpecializationParamCount: unsafe extern "C" fn(*mut slang_IComponentType) -> SlangInt,
    getEntryPointCode: unsafe extern "C" fn(*mut slang_IComponentType, entry_point_index: SlangInt, target_index: SlangInt, out_code: *mut *mut slang_IBlob, out_diagnostics: *mut *mut slang_IBlob) -> SlangResult,
    getResultAsFileSystem: unsafe extern "C" fn(*mut slang_IComponentType, entry_point_index: SlangInt, target_index: SlangInt, out_file_system: *mut *mut ISlangMutableFileSystem) -> SlangResult,
    getEntryPointHash: unsafe extern "C" fn(*mut slang_IComponentType, entry_point_index: SlangInt, target_index: SlangInt, out_hash: *mut *mut slang_IBlob),
    specialize: unsafe extern "C" fn(*mut slang_IComponentType, specialization_args: *const slang_SpecializationArg, specialization_arg_count: SlangInt, out_specialized_component_type: *mut *mut slang_IComponentType, out_diagnostics: *mut *mut ISlangBlob) -> SlangResult,
    link: unsafe extern "C" fn(*mut slang_IComponentType, out_linked_component_type: *mut *mut slang_IComponentType, out_diagnostics: *mut *mut ISlangBlob) -> SlangResult,
    getEntryPointHostCallable: unsafe extern "C" fn(*mut slang_IComponentType, entry_point_index: SlangInt, target_index: SlangInt, out_shared_library: *mut *mut ISlangSharedLibrary, out_diagnostics: *mut *mut slang_IBlob) -> SlangResult,
    renameEntryPoint: unsafe extern "C" fn(*mut slang_IComponentType, new_name: *const c_char, out_entry_point: *mut *mut slang_IComponentType) -> SlangResult,
    linkWithOptions: unsafe extern "C" fn(*mut slang_IComponentType, out_linked_component_type: *mut *mut slang_IComponentType, compiler_option_entry_count: u32, compiler_option_entries: *mut slang_CompilerOptionEntry, out_diagnostics: *mut *mut ISlangBlob) -> SlangResult,
    getTargetCode: unsafe extern "C" fn(*mut slang_IComponentType, target_index: SlangInt, out_code: *mut *mut ISlangBlob, out_diagnostics: *mut *mut ISlangBlob) -> SlangResult,
    getTargetMetadata: unsafe extern "C" fn(*mut slang_IComponentType, target_index: SlangInt, out_medata: *mut *mut slang_IMetadata, out_diagnosticts: *mut *mut ISlangBlob) -> SlangResult,
    getEntryPointMetadata: unsafe extern "C" fn(*mut slang_IComponentType, entry_point_index: SlangInt, target_index: SlangInt, out_metadata: *mut *mut slang_IMetadata, out_diagnostics: *mut *mut ISlangBlob) -> SlangResult,
});

interface!(slang_IModule, [0xc720e64, 0x8722, 0x4d31, {0x89, 0x90, 0x63, 0x8a, 0x98, 0xb1, 0xc2, 0x79}]: slang_IComponentType, {
    findEntryPointByName: unsafe extern "C" fn(*mut slang_IModule, name: *const c_char, outEntryPoint: *mut *mut slang_IEntryPoint) -> SlangResult,
    getDefinedEntryPointCount: unsafe extern "C" fn(*mut slang_IModule) -> SlangInt32,
    getDefinedEntryPoint: unsafe extern "C" fn(*mut slang_IModule, index: SlangInt32, outEntryPoint: *mut *mut slang_IEntryPoint) -> SlangResult,
    serialize: unsafe extern "C" fn(*mut slang_IModule, outSerializedBlob: *mut *mut ISlangBlob) -> SlangResult,
    writeToFile: unsafe extern "C" fn(*mut slang_IModule, fileName: *const c_char) -> SlangResult,
    getName: unsafe extern "C" fn(*mut slang_IModule) -> *const c_char,
    getFilePath: unsafe extern "C" fn(*mut slang_IModule) -> *const c_char,
    getUniqueIdentity: unsafe extern "C" fn(*mut slang_IModule) -> *const c_char,
    findAndCheckEntryPoint: unsafe extern "C" fn(*mut slang_IModule, name: *const c_char, stage: SlangStage, outEntryPoint: *mut *mut slang_IEntryPoint, outDiagnostics: *mut *mut ISlangBlob) -> SlangResult,
    getDependencyFileCount: unsafe extern "C" fn(*mut slang_IModule) -> SlangInt32,
    getDependencyFilePath: unsafe extern "C" fn(*mut slang_IModule, index: SlangInt32) -> *const c_char,
});

interface!(slang_IEntryPoint, [0x8f241361, 0xf5bd, 0x4ca0, {0xa3, 0xac, 0x2, 0xf7, 0xfa, 0x24, 0x2, 0xb8}]: slang_IComponentType, {

});
