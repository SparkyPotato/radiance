; SPIR-V
; Version: 1.5
; Generator: Khronos; 40
; Bound: 567
; Schema: 0
               OpCapability VariablePointers
               OpCapability PhysicalStorageBufferAddresses
               OpCapability Int64ImageEXT
               OpCapability Int64
               OpCapability RuntimeDescriptorArray
               OpCapability ImageQuery
               OpCapability Int8
               OpCapability Shader
               OpExtension "SPV_KHR_variable_pointers"
               OpExtension "SPV_KHR_physical_storage_buffer"
               OpExtension "SPV_EXT_shader_image_int64"
        %107 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel PhysicalStorageBuffer64 GLSL450
               OpEntryPoint Fragment %meshlets "main" %Constants %entryPointParam_meshlets %input_uv %STextures
               OpExecutionMode %meshlets OriginUpperLeft
               OpSource Slang 1
               OpName %MeshletPointer "MeshletPointer"
               OpMemberName %MeshletPointer 0 "instance"
               OpMemberName %MeshletPointer 1 "meshlet_offset"
               OpName %MeshletQueue_std140 "MeshletQueue_std140"
               OpMemberName %MeshletQueue_std140 0 "data"
               OpName %STex_std140 "STex_std140"
               OpMemberName %STex_std140 0 "index"
               OpName %VisBufferTex_std140 "VisBufferTex_std140"
               OpMemberName %VisBufferTex_std140 0 "tex"
               OpName %PushConstants_std140 "PushConstants_std140"
               OpMemberName %PushConstants_std140 0 "instances"
               OpMemberName %PushConstants_std140 1 "camera"
               OpMemberName %PushConstants_std140 2 "early"
               OpMemberName %PushConstants_std140 3 "late"
               OpMemberName %PushConstants_std140 4 "read"
               OpMemberName %PushConstants_std140 5 "bottom"
               OpMemberName %PushConstants_std140 6 "top"
               OpName %Constants "Constants"
               OpName %STex "STex"
               OpMemberName %STex 0 "index"
               OpName %VisBufferTex "VisBufferTex"
               OpMemberName %VisBufferTex 0 "tex"
               OpName %input_uv "input.uv"
               OpName %VisBufferData "VisBufferData"
               OpMemberName %VisBufferData 0 "meshlet_id"
               OpMemberName %VisBufferData 1 "triangle_id"
               OpName %VisBuffer "VisBuffer"
               OpMemberName %VisBuffer 0 "depth"
               OpMemberName %VisBuffer 1 "data"
               OpName %STextures "STextures"
               OpName %MeshletQueue "MeshletQueue"
               OpMemberName %MeshletQueue 0 "data"
               OpName %MeshletPointer_natural "MeshletPointer_natural"
               OpMemberName %MeshletPointer_natural 0 "instance"
               OpMemberName %MeshletPointer_natural 1 "meshlet_offset"
               OpName %entryPointParam_meshlets "entryPointParam_meshlets"
               OpName %meshlets "meshlets"
               OpName %_Array_natural_float12 "_Array_natural_float12"
               OpMemberName %_Array_natural_float12 0 "data"
               OpName %Aabb_natural "Aabb_natural"
               OpMemberName %Aabb_natural 0 "center"
               OpMemberName %Aabb_natural 1 "half_extent"
               OpName %Instance_natural "Instance_natural"
               OpMemberName %Instance_natural 0 "transform"
               OpMemberName %Instance_natural 1 "mesh"
               OpMemberName %Instance_natural 2 "aabb"
               OpName %_MatrixStorage_float4x4_ColMajornatural "_MatrixStorage_float4x4_ColMajornatural"
               OpMemberName %_MatrixStorage_float4x4_ColMajornatural 0 "data"
               OpName %Camera_natural "Camera_natural"
               OpMemberName %Camera_natural 0 "view"
               OpMemberName %Camera_natural 1 "view_proj"
               OpMemberName %Camera_natural 2 "w"
               OpMemberName %Camera_natural 3 "h"
               OpMemberName %Camera_natural 4 "near"
               OpMemberName %Camera_natural 5 "_pad"
               OpMemberName %Camera_natural 6 "frustum"
               OpName %MeshletQueueData_natural "MeshletQueueData_natural"
               OpMemberName %MeshletQueueData_natural 0 "dispatch"
               OpMemberName %MeshletQueueData_natural 1 "pointers"
               OpMemberDecorate %MeshletPointer 0 Offset 0
               OpMemberDecorate %MeshletPointer 1 Offset 4
               OpDecorate %_ptr_PhysicalStorageBuffer_Instance_natural ArrayStride 80
               OpDecorate %_ptr_PhysicalStorageBuffer_Camera_natural ArrayStride 160
               OpDecorate %_ptr_PhysicalStorageBuffer_MeshletQueueData_natural ArrayStride 65535
               OpMemberDecorate %MeshletQueue_std140 0 Offset 0
               OpMemberDecorate %STex_std140 0 Offset 0
               OpMemberDecorate %VisBufferTex_std140 0 Offset 0
               OpDecorate %PushConstants_std140 Block
               OpMemberDecorate %PushConstants_std140 0 Offset 0
               OpMemberDecorate %PushConstants_std140 1 Offset 8
               OpMemberDecorate %PushConstants_std140 2 Offset 16
               OpMemberDecorate %PushConstants_std140 3 Offset 32
               OpMemberDecorate %PushConstants_std140 4 Offset 48
               OpMemberDecorate %PushConstants_std140 5 Offset 64
               OpMemberDecorate %PushConstants_std140 6 Offset 68
               OpMemberDecorate %STex 0 Offset 0
               OpMemberDecorate %VisBufferTex 0 Offset 0
               OpDecorate %input_uv Location 0
               OpMemberDecorate %VisBufferData 0 Offset 0
               OpMemberDecorate %VisBufferData 1 Offset 4
               OpMemberDecorate %VisBuffer 0 Offset 0
               OpMemberDecorate %VisBuffer 1 Offset 4
               OpDecorate %_runtimearr_87 ArrayStride 8
               OpDecorate %STextures Binding 1
               OpDecorate %STextures DescriptorSet 0
               OpMemberDecorate %MeshletQueue 0 Offset 0
               OpDecorate %_ptr_PhysicalStorageBuffer_v3uint ArrayStride 12
               OpDecorate %_runtimearr_MeshletPointer ArrayStride 8
               OpDecorate %_ptr_PhysicalStorageBuffer__runtimearr_MeshletPointer ArrayStride 65535
               OpDecorate %_ptr_PhysicalStorageBuffer_MeshletPointer_natural ArrayStride 8
               OpMemberDecorate %MeshletPointer_natural 0 Offset 0
               OpMemberDecorate %MeshletPointer_natural 1 Offset 4
               OpDecorate %entryPointParam_meshlets Location 0
               OpDecorate %_arr_float_int_12 ArrayStride 4
               OpMemberDecorate %_Array_natural_float12 0 Offset 0
               OpDecorate %_ptr_PhysicalStorageBuffer_uchar ArrayStride 1
               OpMemberDecorate %Aabb_natural 0 Offset 0
               OpMemberDecorate %Aabb_natural 1 Offset 12
               OpMemberDecorate %Instance_natural 0 Offset 0
               OpMemberDecorate %Instance_natural 1 Offset 48
               OpMemberDecorate %Instance_natural 2 Offset 56
               OpDecorate %_arr_v4float_int_4 ArrayStride 16
               OpMemberDecorate %_MatrixStorage_float4x4_ColMajornatural 0 Offset 0
               OpMemberDecorate %Camera_natural 0 Offset 0
               OpMemberDecorate %Camera_natural 1 Offset 64
               OpMemberDecorate %Camera_natural 2 Offset 128
               OpMemberDecorate %Camera_natural 3 Offset 132
               OpMemberDecorate %Camera_natural 4 Offset 136
               OpMemberDecorate %Camera_natural 5 Offset 140
               OpMemberDecorate %Camera_natural 6 Offset 144
               OpDecorate %MeshletQueueData_natural Block
               OpMemberDecorate %MeshletQueueData_natural 0 Offset 0
               OpMemberDecorate %MeshletQueueData_natural 1 Offset 12
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
       %uint = OpTypeInt 32 0
%MeshletPointer = OpTypeStruct %uint %uint
%_ptr_Function_MeshletPointer = OpTypePointer Function %MeshletPointer
               OpTypeForwardPointer %_ptr_PhysicalStorageBuffer_Instance_natural PhysicalStorageBuffer
               OpTypeForwardPointer %_ptr_PhysicalStorageBuffer_Camera_natural PhysicalStorageBuffer
               OpTypeForwardPointer %_ptr_PhysicalStorageBuffer_MeshletQueueData_natural PhysicalStorageBuffer
%MeshletQueue_std140 = OpTypeStruct %_ptr_PhysicalStorageBuffer_MeshletQueueData_natural
%STex_std140 = OpTypeStruct %uint
%VisBufferTex_std140 = OpTypeStruct %STex_std140
%PushConstants_std140 = OpTypeStruct %_ptr_PhysicalStorageBuffer_Instance_natural %_ptr_PhysicalStorageBuffer_Camera_natural %MeshletQueue_std140 %MeshletQueue_std140 %VisBufferTex_std140 %uint %uint
%_ptr_PushConstant_PushConstants_std140 = OpTypePointer PushConstant %PushConstants_std140
        %int = OpTypeInt 32 1
      %int_4 = OpConstant %int 4
%_ptr_PushConstant_VisBufferTex_std140 = OpTypePointer PushConstant %VisBufferTex_std140
       %STex = OpTypeStruct %uint
%VisBufferTex = OpTypeStruct %STex
         %35 = OpTypeFunction %VisBufferTex %VisBufferTex_std140
         %41 = OpTypeFunction %STex %STex_std140
      %float = OpTypeFloat 32
    %v2float = OpTypeVector %float 2
%_ptr_Input_v2float = OpTypePointer Input %v2float
%VisBufferData = OpTypeStruct %uint %uint
         %57 = OpTypeFunction %VisBufferData %VisBufferTex %v2float
  %VisBuffer = OpTypeStruct %float %uint
         %64 = OpTypeFunction %VisBuffer %VisBufferTex %v2float
     %v2uint = OpTypeVector %uint 2
         %72 = OpTypeFunction %v2uint %STex %v2float
         %78 = OpTypeFunction %v2uint %STex
%_ptr_Function_uint = OpTypePointer Function %uint
      %ulong = OpTypeInt 64 0
         %87 = OpTypeImage %ulong 2D 2 0 0 2 R64ui
%_runtimearr_87 = OpTypeRuntimeArray %87
%_ptr_UniformConstant__runtimearr_87 = OpTypePointer UniformConstant %_runtimearr_87
%_ptr_UniformConstant_87 = OpTypePointer UniformConstant %87
  %float_0_5 = OpConstant %float 0.5
        %112 = OpTypeFunction %ulong %STex %v2uint
      %v2int = OpTypeVector %int 2
    %v4ulong = OpTypeVector %ulong 4
        %127 = OpTypeFunction %VisBuffer %ulong
%_ptr_Function_VisBuffer = OpTypePointer Function %VisBuffer
      %int_0 = OpConstant %int 0
%_ptr_Function_float = OpTypePointer Function %float
     %int_32 = OpConstant %int 32
      %int_1 = OpConstant %int 1
%ulong_18446744073709551615 = OpConstant %ulong 18446744073709551615
        %152 = OpTypeFunction %VisBufferData %uint
%_ptr_Function_VisBufferData = OpTypePointer Function %VisBufferData
      %int_7 = OpConstant %int 7
   %uint_127 = OpConstant %uint 127
       %bool = OpTypeBool
     %int_n1 = OpConstant %int -1
      %int_2 = OpConstant %int 2
%_ptr_PushConstant_MeshletQueue_std140 = OpTypePointer PushConstant %MeshletQueue_std140
%MeshletQueue = OpTypeStruct %_ptr_PhysicalStorageBuffer_MeshletQueueData_natural
        %182 = OpTypeFunction %MeshletQueue %MeshletQueue_std140
        %190 = OpTypeFunction %uint %MeshletQueue
     %v3uint = OpTypeVector %uint 3
%_ptr_PhysicalStorageBuffer_v3uint = OpTypePointer PhysicalStorageBuffer %v3uint
      %int_3 = OpConstant %int 3
        %209 = OpTypeFunction %MeshletPointer %MeshletQueue %uint
%_runtimearr_MeshletPointer = OpTypeRuntimeArray %MeshletPointer
%_ptr_PhysicalStorageBuffer__runtimearr_MeshletPointer = OpTypePointer PhysicalStorageBuffer %_runtimearr_MeshletPointer
               OpTypeForwardPointer %_ptr_PhysicalStorageBuffer_MeshletPointer_natural PhysicalStorageBuffer
%MeshletPointer_natural = OpTypeStruct %uint %uint
        %223 = OpTypeFunction %MeshletPointer %MeshletPointer_natural
        %242 = OpTypeFunction %uint %uint
%uint_2127912214 = OpConstant %uint 2127912214
     %int_12 = OpConstant %int 12
%uint_3345072700 = OpConstant %uint 3345072700
     %int_19 = OpConstant %int 19
%uint_374761393 = OpConstant %uint 374761393
      %int_5 = OpConstant %int 5
%uint_3550635116 = OpConstant %uint 3550635116
      %int_9 = OpConstant %int 9
%uint_4251993797 = OpConstant %uint 4251993797
%uint_3042594569 = OpConstant %uint 3042594569
     %int_16 = OpConstant %int 16
   %uint_255 = OpConstant %uint 255
      %int_8 = OpConstant %int 8
    %v3float = OpTypeVector %float 3
  %float_255 = OpConstant %float 255
    %v4float = OpTypeVector %float 4
    %float_1 = OpConstant %float 1
%_ptr_Output_v4float = OpTypePointer Output %v4float
%_arr_float_int_12 = OpTypeArray %float %int_12
%_Array_natural_float12 = OpTypeStruct %_arr_float_int_12
      %uchar = OpTypeInt 8 0
%_ptr_PhysicalStorageBuffer_uchar = OpTypePointer PhysicalStorageBuffer %uchar
%Aabb_natural = OpTypeStruct %v3float %v3float
%Instance_natural = OpTypeStruct %_Array_natural_float12 %_ptr_PhysicalStorageBuffer_uchar %Aabb_natural
%_ptr_PhysicalStorageBuffer_Instance_natural = OpTypePointer PhysicalStorageBuffer %Instance_natural
%_arr_v4float_int_4 = OpTypeArray %v4float %int_4
%_MatrixStorage_float4x4_ColMajornatural = OpTypeStruct %_arr_v4float_int_4
%Camera_natural = OpTypeStruct %_MatrixStorage_float4x4_ColMajornatural %_MatrixStorage_float4x4_ColMajornatural %float %float %float %float %v4float
%_ptr_PhysicalStorageBuffer_Camera_natural = OpTypePointer PhysicalStorageBuffer %Camera_natural
%MeshletQueueData_natural = OpTypeStruct %v3uint %_runtimearr_MeshletPointer
%_ptr_PhysicalStorageBuffer_MeshletQueueData_natural = OpTypePointer PhysicalStorageBuffer %MeshletQueueData_natural
%_ptr_PhysicalStorageBuffer_MeshletPointer_natural = OpTypePointer PhysicalStorageBuffer %MeshletPointer_natural
  %Constants = OpVariable %_ptr_PushConstant_PushConstants_std140 PushConstant
   %input_uv = OpVariable %_ptr_Input_v2float Input
  %STextures = OpVariable %_ptr_UniformConstant__runtimearr_87 UniformConstant
%entryPointParam_meshlets = OpVariable %_ptr_Output_v4float Output
%_ptr_Function_VisBufferTex = OpTypePointer Function %VisBufferTex
%_ptr_Function_STex = OpTypePointer Function %STex
%_ptr_Function_v2uint = OpTypePointer Function %v2uint
%_ptr_Function_ulong = OpTypePointer Function %ulong
%_ptr_Function_MeshletQueue = OpTypePointer Function %MeshletQueue
%_ptr_Function__ptr_PhysicalStorageBuffer_MeshletQueueData_natural = OpTypePointer Function %_ptr_PhysicalStorageBuffer_MeshletQueueData_natural
        %529 = OpUndef %float
        %535 = OpUndef %uint
        %563 = OpConstantComposite %v2float %float_0_5 %float_0_5
        %564 = OpConstantComposite %v3float %float_255 %float_255 %float_255
%float_0_00392156886 = OpConstant %float 0.00392156886
        %566 = OpConstantComposite %v3float %float_0_00392156886 %float_0_00392156886 %float_0_00392156886
   %meshlets = OpFunction %void None %3
          %4 = OpLabel
        %559 = OpVariable %_ptr_Function_uint Function
        %548 = OpVariable %_ptr_Function_uint Function
        %547 = OpVariable %_ptr_Function_uint Function
        %539 = OpVariable %_ptr_Function_uint Function
        %534 = OpVariable %_ptr_Function_uint Function
        %530 = OpVariable %_ptr_Function_uint Function
        %523 = OpVariable %_ptr_Function_uint Function
        %522 = OpVariable %_ptr_Function_float Function
        %518 = OpVariable %_ptr_Function_uint Function
        %517 = OpVariable %_ptr_Function_float Function
        %511 = OpVariable %_ptr_Function_uint Function
        %510 = OpVariable %_ptr_Function_uint Function
        %506 = OpVariable %_ptr_Function_uint Function
        %505 = OpVariable %_ptr_Function_uint Function
        %501 = OpVariable %_ptr_Function__ptr_PhysicalStorageBuffer_MeshletQueueData_natural Function
        %497 = OpVariable %_ptr_Function__ptr_PhysicalStorageBuffer_MeshletQueueData_natural Function
        %491 = OpVariable %_ptr_Function_uint Function
        %490 = OpVariable %_ptr_Function_uint Function
        %484 = OpVariable %_ptr_Function_uint Function
        %483 = OpVariable %_ptr_Function_uint Function
        %479 = OpVariable %_ptr_Function__ptr_PhysicalStorageBuffer_MeshletQueueData_natural Function
        %472 = OpVariable %_ptr_Function_uint Function
        %471 = OpVariable %_ptr_Function_uint Function
        %465 = OpVariable %_ptr_Function_uint Function
        %464 = OpVariable %_ptr_Function_uint Function
        %444 = OpVariable %_ptr_Function_uint Function
        %424 = OpVariable %_ptr_Function_uint Function
        %386 = OpVariable %_ptr_Function_uint Function
        %354 = OpVariable %_ptr_Function_ulong Function
        %340 = OpVariable %_ptr_Function_uint Function
        %341 = OpVariable %_ptr_Function_uint Function
        %342 = OpVariable %_ptr_Function_v2uint Function
        %331 = OpVariable %_ptr_Function_v2uint Function
         %29 = OpAccessChain %_ptr_PushConstant_VisBufferTex_std140 %Constants %int_4
         %30 = OpLoad %VisBufferTex_std140 %29
        %311 = OpCompositeExtract %STex_std140 %30 0
        %317 = OpCompositeExtract %uint %311 0
        %318 = OpCompositeConstruct %STex %317
               OpStore %539 %317
        %542 = OpCompositeConstruct %STex %317
        %313 = OpCompositeConstruct %VisBufferTex %542
               OpStore %559 %317
        %562 = OpCompositeConstruct %STex %317
        %546 = OpCompositeConstruct %VisBufferTex %562
         %51 = OpLoad %v2float %input_uv
        %345 = OpAccessChain %_ptr_UniformConstant_87 %STextures %317
        %346 = OpLoad %87 %345
        %347 = OpImageQuerySize %v2uint %346
        %348 = OpCompositeExtract %uint %347 0
               OpStore %340 %348
        %349 = OpCompositeExtract %uint %347 1
               OpStore %341 %349
        %352 = OpCompositeConstruct %v2uint %348 %349
               OpStore %342 %352
        %334 = OpConvertUToF %v2float %352
        %335 = OpFMul %v2float %51 %334
        %337 = OpFSub %v2float %335 %563
        %338 = OpExtInst %v2float %107 Round %337
        %339 = OpConvertFToU %v2uint %338
               OpStore %331 %339
        %357 = OpAccessChain %_ptr_UniformConstant_87 %STextures %317
        %358 = OpBitcast %v2int %339
        %359 = OpLoad %87 %357
        %360 = OpImageRead %v4ulong %359 %358
        %361 = OpCompositeExtract %ulong %360 0
               OpStore %354 %361
        %366 = OpShiftRightLogical %ulong %361 %int_32
        %367 = OpUConvert %uint %366
        %368 = OpBitcast %float %367
               OpStore %517 %368
        %370 = OpBitwiseAnd %ulong %361 %ulong_18446744073709551615
        %371 = OpUConvert %uint %370
               OpStore %518 %371
        %521 = OpCompositeConstruct %VisBuffer %368 %371
               OpStore %522 %368
               OpStore %523 %371
        %528 = OpCompositeConstruct %VisBuffer %368 %371
               OpStore %530 %371
        %533 = OpCompositeConstruct %VisBuffer %529 %371
        %377 = OpShiftRightLogical %uint %371 %int_7
               OpStore %505 %377
        %379 = OpBitwiseAnd %uint %371 %uint_127
               OpStore %506 %379
        %509 = OpCompositeConstruct %VisBufferData %377 %379
               OpStore %510 %377
               OpStore %511 %379
        %516 = OpCompositeConstruct %VisBufferData %377 %379
               OpStore %534 %377
        %538 = OpCompositeConstruct %VisBufferData %377 %535
        %169 = OpBitcast %int %377
        %171 = OpIEqual %bool %169 %int_n1
               OpSelectionMerge %10 None
               OpBranchConditional %171 %9 %10
          %9 = OpLabel
               OpKill
         %10 = OpLabel
        %177 = OpAccessChain %_ptr_PushConstant_MeshletQueue_std140 %Constants %int_2
        %178 = OpLoad %MeshletQueue_std140 %177
        %384 = OpCompositeExtract %_ptr_PhysicalStorageBuffer_MeshletQueueData_natural %178 0
        %385 = OpCompositeConstruct %MeshletQueue %384
               OpStore %501 %384
        %504 = OpCompositeConstruct %MeshletQueue %384
        %389 = OpAccessChain %_ptr_PhysicalStorageBuffer_v3uint %384 %int_0
        %390 = OpLoad %v3uint %389 Aligned 4
        %391 = OpCompositeExtract %uint %390 0
               OpStore %386 %391
        %200 = OpULessThan %bool %377 %391
               OpSelectionMerge %13 None
               OpBranchConditional %200 %12 %11
         %11 = OpLabel
        %203 = OpAccessChain %_ptr_PushConstant_MeshletQueue_std140 %Constants %int_3
        %204 = OpLoad %MeshletQueue_std140 %203
        %394 = OpCompositeExtract %_ptr_PhysicalStorageBuffer_MeshletQueueData_natural %204 0
        %395 = OpCompositeConstruct %MeshletQueue %394
               OpStore %497 %394
        %500 = OpCompositeConstruct %MeshletQueue %394
        %206 = OpISub %uint %377 %391
        %399 = OpAccessChain %_ptr_PhysicalStorageBuffer__runtimearr_MeshletPointer %394 %int_1
        %400 = OpAccessChain %_ptr_PhysicalStorageBuffer_MeshletPointer_natural %399 %206
        %401 = OpLoad %MeshletPointer_natural %400 Aligned 4
        %405 = OpCompositeExtract %uint %401 0
        %406 = OpCompositeExtract %uint %401 1
        %407 = OpCompositeConstruct %MeshletPointer %405 %406
               OpStore %483 %405
               OpStore %484 %406
        %489 = OpCompositeConstruct %MeshletPointer %405 %406
               OpStore %490 %405
               OpStore %491 %406
        %496 = OpCompositeConstruct %MeshletPointer %405 %406
               OpStore %547 %405
               OpStore %548 %406
               OpBranch %13
         %12 = OpLabel
        %233 = OpLoad %MeshletQueue_std140 %177
        %410 = OpCompositeExtract %_ptr_PhysicalStorageBuffer_MeshletQueueData_natural %233 0
        %411 = OpCompositeConstruct %MeshletQueue %410
               OpStore %479 %410
        %482 = OpCompositeConstruct %MeshletQueue %410
        %415 = OpAccessChain %_ptr_PhysicalStorageBuffer__runtimearr_MeshletPointer %410 %int_1
        %416 = OpAccessChain %_ptr_PhysicalStorageBuffer_MeshletPointer_natural %415 %377
        %417 = OpLoad %MeshletPointer_natural %416 Aligned 4
        %421 = OpCompositeExtract %uint %417 0
        %422 = OpCompositeExtract %uint %417 1
        %423 = OpCompositeConstruct %MeshletPointer %421 %422
               OpStore %464 %421
               OpStore %465 %422
        %470 = OpCompositeConstruct %MeshletPointer %421 %422
               OpStore %471 %421
               OpStore %472 %422
        %477 = OpCompositeConstruct %MeshletPointer %421 %422
               OpStore %547 %421
               OpStore %548 %422
               OpBranch %13
         %13 = OpLabel
        %554 = OpLoad %uint %548
        %553 = OpLoad %uint %547
        %555 = OpCompositeConstruct %MeshletPointer %553 %554
        %426 = OpIAdd %uint %553 %uint_2127912214
        %427 = OpShiftLeftLogical %uint %553 %int_12
        %428 = OpIAdd %uint %426 %427
        %429 = OpBitwiseXor %uint %428 %uint_3345072700
        %430 = OpShiftRightLogical %uint %428 %int_19
        %431 = OpBitwiseXor %uint %429 %430
        %432 = OpIAdd %uint %431 %uint_374761393
        %433 = OpShiftLeftLogical %uint %431 %int_5
        %434 = OpIAdd %uint %432 %433
        %435 = OpIAdd %uint %434 %uint_3550635116
        %436 = OpShiftLeftLogical %uint %434 %int_9
        %437 = OpBitwiseXor %uint %435 %436
        %438 = OpIAdd %uint %437 %uint_4251993797
        %439 = OpShiftLeftLogical %uint %437 %int_3
        %440 = OpIAdd %uint %438 %439
        %441 = OpBitwiseXor %uint %440 %uint_3042594569
        %442 = OpShiftRightLogical %uint %440 %int_16
        %443 = OpBitwiseXor %uint %441 %442
               OpStore %424 %443
        %558 = OpCompositeConstruct %MeshletPointer %553 %554
        %446 = OpIAdd %uint %554 %uint_2127912214
        %447 = OpShiftLeftLogical %uint %554 %int_12
        %448 = OpIAdd %uint %446 %447
        %449 = OpBitwiseXor %uint %448 %uint_3345072700
        %450 = OpShiftRightLogical %uint %448 %int_19
        %451 = OpBitwiseXor %uint %449 %450
        %452 = OpIAdd %uint %451 %uint_374761393
        %453 = OpShiftLeftLogical %uint %451 %int_5
        %454 = OpIAdd %uint %452 %453
        %455 = OpIAdd %uint %454 %uint_3550635116
        %456 = OpShiftLeftLogical %uint %454 %int_9
        %457 = OpBitwiseXor %uint %455 %456
        %458 = OpIAdd %uint %457 %uint_4251993797
        %459 = OpShiftLeftLogical %uint %457 %int_3
        %460 = OpIAdd %uint %458 %459
        %461 = OpBitwiseXor %uint %460 %uint_3042594569
        %462 = OpShiftRightLogical %uint %460 %int_16
        %463 = OpBitwiseXor %uint %461 %462
               OpStore %444 %463
        %278 = OpBitwiseXor %uint %443 %463
        %279 = OpBitwiseAnd %uint %278 %uint_255
        %281 = OpConvertUToF %float %279
        %282 = OpShiftRightLogical %uint %278 %int_8
        %284 = OpBitwiseAnd %uint %282 %uint_255
        %285 = OpConvertUToF %float %284
        %286 = OpShiftRightLogical %uint %278 %int_16
        %287 = OpBitwiseAnd %uint %286 %uint_255
        %288 = OpConvertUToF %float %287
        %290 = OpCompositeConstruct %v3float %281 %285 %288
        %293 = OpFMul %v3float %290 %566
        %295 = OpCompositeConstruct %v4float %293 %float_1
               OpStore %entryPointParam_meshlets %295
               OpReturn
               OpFunctionEnd
