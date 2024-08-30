implementing graph;

public T wave_shuffle_xor<T : __BuiltinType>(T value, u32 mask) {
	return spirv_asm {
		OpCapability GroupNonUniformShuffle;
		OpGroupNonUniformShuffleXor $$T result Subgroup $value $mask;
	};
}

public u32 wave_id() {
	return spirv_asm {
		result:$$u32 = OpLoad builtin(SubgroupId:u32);
	};
}
