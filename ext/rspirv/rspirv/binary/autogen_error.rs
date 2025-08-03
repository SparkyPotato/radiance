// AUTOMATICALLY GENERATED from the SPIR-V JSON grammar:
//   external/spirv.core.grammar.json.
// DO NOT MODIFY!

use std::{error, fmt};
#[doc = "Decoder Error"]
#[derive(Debug, PartialEq, Eq)]
#[allow(clippy::upper_case_acronyms)]
pub enum Error {
    StreamExpected(usize),
    LimitReached(usize),
    ImageOperandsUnknown(usize, spirv::Word),
    FPFastMathModeUnknown(usize, spirv::Word),
    SelectionControlUnknown(usize, spirv::Word),
    LoopControlUnknown(usize, spirv::Word),
    FunctionControlUnknown(usize, spirv::Word),
    MemorySemanticsUnknown(usize, spirv::Word),
    MemoryAccessUnknown(usize, spirv::Word),
    KernelProfilingInfoUnknown(usize, spirv::Word),
    RayFlagsUnknown(usize, spirv::Word),
    FragmentShadingRateUnknown(usize, spirv::Word),
    RawAccessChainOperandsUnknown(usize, spirv::Word),
    SourceLanguageUnknown(usize, spirv::Word),
    ExecutionModelUnknown(usize, spirv::Word),
    AddressingModelUnknown(usize, spirv::Word),
    MemoryModelUnknown(usize, spirv::Word),
    ExecutionModeUnknown(usize, spirv::Word),
    StorageClassUnknown(usize, spirv::Word),
    DimUnknown(usize, spirv::Word),
    SamplerAddressingModeUnknown(usize, spirv::Word),
    SamplerFilterModeUnknown(usize, spirv::Word),
    ImageFormatUnknown(usize, spirv::Word),
    ImageChannelOrderUnknown(usize, spirv::Word),
    ImageChannelDataTypeUnknown(usize, spirv::Word),
    FPRoundingModeUnknown(usize, spirv::Word),
    FPDenormModeUnknown(usize, spirv::Word),
    QuantizationModesUnknown(usize, spirv::Word),
    FPOperationModeUnknown(usize, spirv::Word),
    OverflowModesUnknown(usize, spirv::Word),
    LinkageTypeUnknown(usize, spirv::Word),
    AccessQualifierUnknown(usize, spirv::Word),
    HostAccessQualifierUnknown(usize, spirv::Word),
    FunctionParameterAttributeUnknown(usize, spirv::Word),
    DecorationUnknown(usize, spirv::Word),
    BuiltInUnknown(usize, spirv::Word),
    ScopeUnknown(usize, spirv::Word),
    GroupOperationUnknown(usize, spirv::Word),
    KernelEnqueueFlagsUnknown(usize, spirv::Word),
    CapabilityUnknown(usize, spirv::Word),
    RayQueryIntersectionUnknown(usize, spirv::Word),
    RayQueryCommittedIntersectionTypeUnknown(usize, spirv::Word),
    RayQueryCandidateIntersectionTypeUnknown(usize, spirv::Word),
    PackedVectorFormatUnknown(usize, spirv::Word),
    CooperativeMatrixOperandsUnknown(usize, spirv::Word),
    CooperativeMatrixLayoutUnknown(usize, spirv::Word),
    CooperativeMatrixUseUnknown(usize, spirv::Word),
    CooperativeMatrixReduceUnknown(usize, spirv::Word),
    TensorClampModeUnknown(usize, spirv::Word),
    TensorAddressingOperandsUnknown(usize, spirv::Word),
    InitializationModeQualifierUnknown(usize, spirv::Word),
    LoadCacheControlUnknown(usize, spirv::Word),
    StoreCacheControlUnknown(usize, spirv::Word),
    NamedMaximumNumberOfRegistersUnknown(usize, spirv::Word),
    MatrixMultiplyAccumulateOperandsUnknown(usize, spirv::Word),
    FPEncodingUnknown(usize, spirv::Word),
    CooperativeVectorMatrixLayoutUnknown(usize, spirv::Word),
    ComponentTypeUnknown(usize, spirv::Word),
    #[doc = r"Failed to decode a string."]
    #[doc = r""]
    #[doc = r"For structured error handling, the second element could be"]
    #[doc = r"`string::FromUtf8Error`, but the will prohibit the compiler"]
    #[doc = r"from generating `PartialEq` for this enum."]
    DecodeStringFailed(usize, String),
}
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Error::StreamExpected(index) => {
                write!(f, "expected more bytes in the stream at index {index}")
            }
            Error::LimitReached(index) => write!(f, "reached word limit at index {index}"),
            Error::ImageOperandsUnknown(index, word) => write!(
                f,
                "unknown value {word} for operand kind ImageOperands at index {index}"
            ),
            Error::FPFastMathModeUnknown(index, word) => write!(
                f,
                "unknown value {word} for operand kind FPFastMathMode at index {index}"
            ),
            Error::SelectionControlUnknown(index, word) => write!(
                f,
                "unknown value {word} for operand kind SelectionControl at index {index}"
            ),
            Error::LoopControlUnknown(index, word) => write!(
                f,
                "unknown value {word} for operand kind LoopControl at index {index}"
            ),
            Error::FunctionControlUnknown(index, word) => write!(
                f,
                "unknown value {word} for operand kind FunctionControl at index {index}"
            ),
            Error::MemorySemanticsUnknown(index, word) => write!(
                f,
                "unknown value {word} for operand kind MemorySemantics at index {index}"
            ),
            Error::MemoryAccessUnknown(index, word) => write!(
                f,
                "unknown value {word} for operand kind MemoryAccess at index {index}"
            ),
            Error::KernelProfilingInfoUnknown(index, word) => write!(
                f,
                "unknown value {word} for operand kind KernelProfilingInfo at index {index}"
            ),
            Error::RayFlagsUnknown(index, word) => write!(
                f,
                "unknown value {word} for operand kind RayFlags at index {index}"
            ),
            Error::FragmentShadingRateUnknown(index, word) => write!(
                f,
                "unknown value {word} for operand kind FragmentShadingRate at index {index}"
            ),
            Error::RawAccessChainOperandsUnknown(index, word) => write!(
                f,
                "unknown value {word} for operand kind RawAccessChainOperands at index {index}"
            ),
            Error::SourceLanguageUnknown(index, word) => write!(
                f,
                "unknown value {word} for operand kind SourceLanguage at index {index}"
            ),
            Error::ExecutionModelUnknown(index, word) => write!(
                f,
                "unknown value {word} for operand kind ExecutionModel at index {index}"
            ),
            Error::AddressingModelUnknown(index, word) => write!(
                f,
                "unknown value {word} for operand kind AddressingModel at index {index}"
            ),
            Error::MemoryModelUnknown(index, word) => write!(
                f,
                "unknown value {word} for operand kind MemoryModel at index {index}"
            ),
            Error::ExecutionModeUnknown(index, word) => write!(
                f,
                "unknown value {word} for operand kind ExecutionMode at index {index}"
            ),
            Error::StorageClassUnknown(index, word) => write!(
                f,
                "unknown value {word} for operand kind StorageClass at index {index}"
            ),
            Error::DimUnknown(index, word) => write!(
                f,
                "unknown value {word} for operand kind Dim at index {index}"
            ),
            Error::SamplerAddressingModeUnknown(index, word) => write!(
                f,
                "unknown value {word} for operand kind SamplerAddressingMode at index {index}"
            ),
            Error::SamplerFilterModeUnknown(index, word) => write!(
                f,
                "unknown value {word} for operand kind SamplerFilterMode at index {index}"
            ),
            Error::ImageFormatUnknown(index, word) => write!(
                f,
                "unknown value {word} for operand kind ImageFormat at index {index}"
            ),
            Error::ImageChannelOrderUnknown(index, word) => write!(
                f,
                "unknown value {word} for operand kind ImageChannelOrder at index {index}"
            ),
            Error::ImageChannelDataTypeUnknown(index, word) => write!(
                f,
                "unknown value {word} for operand kind ImageChannelDataType at index {index}"
            ),
            Error::FPRoundingModeUnknown(index, word) => write!(
                f,
                "unknown value {word} for operand kind FPRoundingMode at index {index}"
            ),
            Error::FPDenormModeUnknown(index, word) => write!(
                f,
                "unknown value {word} for operand kind FPDenormMode at index {index}"
            ),
            Error::QuantizationModesUnknown(index, word) => write!(
                f,
                "unknown value {word} for operand kind QuantizationModes at index {index}"
            ),
            Error::FPOperationModeUnknown(index, word) => write!(
                f,
                "unknown value {word} for operand kind FPOperationMode at index {index}"
            ),
            Error::OverflowModesUnknown(index, word) => write!(
                f,
                "unknown value {word} for operand kind OverflowModes at index {index}"
            ),
            Error::LinkageTypeUnknown(index, word) => write!(
                f,
                "unknown value {word} for operand kind LinkageType at index {index}"
            ),
            Error::AccessQualifierUnknown(index, word) => write!(
                f,
                "unknown value {word} for operand kind AccessQualifier at index {index}"
            ),
            Error::HostAccessQualifierUnknown(index, word) => write!(
                f,
                "unknown value {word} for operand kind HostAccessQualifier at index {index}"
            ),
            Error::FunctionParameterAttributeUnknown(index, word) => write!(
                f,
                "unknown value {word} for operand kind FunctionParameterAttribute at index {index}"
            ),
            Error::DecorationUnknown(index, word) => write!(
                f,
                "unknown value {word} for operand kind Decoration at index {index}"
            ),
            Error::BuiltInUnknown(index, word) => write!(
                f,
                "unknown value {word} for operand kind BuiltIn at index {index}"
            ),
            Error::ScopeUnknown(index, word) => write!(
                f,
                "unknown value {word} for operand kind Scope at index {index}"
            ),
            Error::GroupOperationUnknown(index, word) => write!(
                f,
                "unknown value {word} for operand kind GroupOperation at index {index}"
            ),
            Error::KernelEnqueueFlagsUnknown(index, word) => write!(
                f,
                "unknown value {word} for operand kind KernelEnqueueFlags at index {index}"
            ),
            Error::CapabilityUnknown(index, word) => write!(
                f,
                "unknown value {word} for operand kind Capability at index {index}"
            ),
            Error::RayQueryIntersectionUnknown(index, word) => write!(
                f,
                "unknown value {word} for operand kind RayQueryIntersection at index {index}"
            ),
            Error::RayQueryCommittedIntersectionTypeUnknown(index, word) => write!(
                f,
                "unknown value {word} for operand kind RayQueryCommittedIntersectionType at index {index}"
            ),
            Error::RayQueryCandidateIntersectionTypeUnknown(index, word) => write!(
                f,
                "unknown value {word} for operand kind RayQueryCandidateIntersectionType at index {index}"
            ),
            Error::PackedVectorFormatUnknown(index, word) => write!(
                f,
                "unknown value {word} for operand kind PackedVectorFormat at index {index}"
            ),
            Error::CooperativeMatrixOperandsUnknown(index, word) => write!(
                f,
                "unknown value {word} for operand kind CooperativeMatrixOperands at index {index}"
            ),
            Error::CooperativeMatrixLayoutUnknown(index, word) => write!(
                f,
                "unknown value {word} for operand kind CooperativeMatrixLayout at index {index}"
            ),
            Error::CooperativeMatrixUseUnknown(index, word) => write!(
                f,
                "unknown value {word} for operand kind CooperativeMatrixUse at index {index}"
            ),
            Error::CooperativeMatrixReduceUnknown(index, word) => write!(
                f,
                "unknown value {word} for operand kind CooperativeMatrixReduce at index {index}"
            ),
            Error::TensorClampModeUnknown(index, word) => write!(
                f,
                "unknown value {word} for operand kind TensorClampMode at index {index}"
            ),
            Error::TensorAddressingOperandsUnknown(index, word) => write!(
                f,
                "unknown value {word} for operand kind TensorAddressingOperands at index {index}"
            ),
            Error::InitializationModeQualifierUnknown(index, word) => write!(
                f,
                "unknown value {word} for operand kind InitializationModeQualifier at index {index}"
            ),
            Error::LoadCacheControlUnknown(index, word) => write!(
                f,
                "unknown value {word} for operand kind LoadCacheControl at index {index}"
            ),
            Error::StoreCacheControlUnknown(index, word) => write!(
                f,
                "unknown value {word} for operand kind StoreCacheControl at index {index}"
            ),
            Error::NamedMaximumNumberOfRegistersUnknown(index, word) => write!(
                f,
                "unknown value {word} for operand kind NamedMaximumNumberOfRegisters at index {index}"
            ),
            Error::MatrixMultiplyAccumulateOperandsUnknown(index, word) => write!(
                f,
                "unknown value {word} for operand kind MatrixMultiplyAccumulateOperands at index {index}"
            ),
            Error::FPEncodingUnknown(index, word) => write!(
                f,
                "unknown value {word} for operand kind FPEncoding at index {index}"
            ),
            Error::CooperativeVectorMatrixLayoutUnknown(index, word) => write!(
                f,
                "unknown value {word} for operand kind CooperativeVectorMatrixLayout at index {index}"
            ),
            Error::ComponentTypeUnknown(index, word) => write!(
                f,
                "unknown value {word} for operand kind ComponentType at index {index}"
            ),
            Error::DecodeStringFailed(index, ref e) => {
                write!(f, "cannot decode string at index {index}: {e}")
            }
        }
    }
}
impl error::Error for Error {}
