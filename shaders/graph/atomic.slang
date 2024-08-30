implementing graph;

public enum Scope : u32 {
	Device = 1,
	Workgroup = 2,
	Subgroup = 3,
	Invocation = 4,
	Queue = 5,
	ShaderCall = 6,
}

public enum Semantics : u32 {
	Relaxed = 0x0,
	Acquire = 0x2,
	Release = 0x4,
	AcqRel = 0x8,
	SeqCst = 0x10,
	Buffer = 0x40,
	Subgroup = 0x80,
	Workgroup = 0x100,
	Image = 0x800,
	MakeAvailable = 0x2000,
	MakeVisible = 0x4000,
}

// clang-format off
public T atomic_add<T : __BuiltinArithmeticType>(__ref T dest, T value, constexpr Scope sc = Scope.Device, constexpr Semantics se = Semantics.Relaxed) {
	// clang-format on
	if (__isFloat<T>()) {
		return spirv_asm {
			result:$$T = OpAtomicFAddEXT &dest $sc $se $value;
		};
	} else if (__isUnsignedInt<T>() || __isInt<T>()) {
		return spirv_asm {
			OpCapability Int64Atomics;
			result:$$T = OpAtomicIAdd &dest $sc $se $value;
		};
	} else {
		spirv_asm { "<invalid atomic_add>" };
	}
	return {};
}

// clang-format off
public T atomic_max<T : __BuiltinArithmeticType>(__ref T dest, T value, constexpr Scope sc = Scope.Device, constexpr Semantics se = Semantics.Relaxed) {
	// clang-format on
	if (__isFloat<T>()) {
		return spirv_asm {
			result:$$T = OpAtomicFMaxEXT &dest $sc $se $value;
		};
	} else if (__isUnsignedInt<T>()) {
		return spirv_asm {
			OpCapability Int64Atomics;
			result:$$T = OpAtomicUMax &dest $sc $se $value;
		};
	} else if (__isInt<T>()) {
		return spirv_asm {
			OpCapability Int64Atomics;
			result:$$T = OpAtomicSMax &dest $sc $se $value
		};
	} else {
		spirv_asm { "<invalid atomic_max>" };
	}
	return {};
}

public void barrier(constexpr Scope sc, constexpr Semantics se) {
	spirv_asm {
		OpMemoryBarrier $sc $se;
	};
}

public void sync_barrier(constexpr Scope esc, constexpr Scope sc, constexpr Semantics se) {
	spirv_asm {
		OpControlBarrier $esc $sc $se;
	};
}

public void workgroup_sync_barrier() {
	sync_barrier(Scope::Workgroup, Scope::Workgroup, Semantics::Workgroup | Semantics::AcqRel);
}

public void all_sync_barrier() {
	sync_barrier(Scope::Workgroup, Scope::Device,
				 Semantics::Buffer | Semantics::Image | Semantics::Workgroup | Semantics::AcqRel);
}
