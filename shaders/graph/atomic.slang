implementing graph;

public enum Scope : u32 {
	Device = 1,
	Workgroup = 2,
	Subgroup = 3,
	Invocation = 4,
	Queue = 5,
	ShaderCall = 6,
}

public enum Sem : u32 {
	Relaxed = 0x0,
	Acquire = 0x2,
	Release = 0x4,
	AcqRel = 0x8,
}

public enum Loc : u32 {
	None = 0x0,
	Buffer = 0x40,
	Subgroup = 0x80,
	Workgroup = 0x100,
	Image = 0x800,
}

public static const Scope DEFAULT_SCOPE = Scope.Queue;
public static const Sem DEFAULT_SEM = Sem.Relaxed;
public static const Loc DEFAULT_LOC = Loc.None;

public T atomic_add<T : __BuiltinArithmeticType>(__ref T dest, T value, constexpr Scope sc = DEFAULT_SCOPE,
												 constexpr Loc loc = DEFAULT_LOC, constexpr Sem se = DEFAULT_SEM) {
	u32 sem = loc | se;
	if (__isFloat<T>()) {
		return spirv_asm {
			result:$$T = OpAtomicFAddEXT &dest $sc $sem $value;
		};
	} else if (__isUnsignedInt<T>() || __isInt<T>()) {
		return spirv_asm {
			OpCapability Int64Atomics;
			result:$$T = OpAtomicIAdd &dest $sc $sem $value;
		};
	} else {
		spirv_asm { "<invalid atomic_add>" };
	}
	return {};
}

public T atomic_sub<T : __BuiltinArithmeticType>(__ref T dest, T value, constexpr Scope sc = DEFAULT_SCOPE,
												 constexpr Loc loc = DEFAULT_LOC, constexpr Sem se = DEFAULT_SEM) {
	u32 sem = loc | se;
	if (__isFloat<T>()) {
		let neg = -value;
		return spirv_asm {
			result:$$T = OpAtomicFAddEXT &dest $sc $sem $neg;
		};
	} else if (__isUnsignedInt<T>() || __isInt<T>()) {
		return spirv_asm {
			OpCapability Int64Atomics;
			result:$$T = OpAtomicISub &dest $sc $sem $value;
		};
	} else {
		spirv_asm { "<invalid atomic_add>" };
	}
	return {};
}

public T atomic_max<T : __BuiltinArithmeticType>(__ref T dest, T value, constexpr Scope sc = DEFAULT_SCOPE,
												 constexpr Loc loc = DEFAULT_LOC, constexpr Sem se = DEFAULT_SEM) {
	u32 sem = loc | se;
	if (__isFloat<T>()) {
		return spirv_asm {
			result:$$T = OpAtomicFMaxEXT &dest $sc $sem $value;
		};
	} else if (__isUnsignedInt<T>()) {
		return spirv_asm {
			OpCapability Int64Atomics;
			result:$$T = OpAtomicUMax &dest $sc $sem $value;
		};
	} else if (__isInt<T>()) {
		return spirv_asm {
			OpCapability Int64Atomics;
			result:$$T = OpAtomicSMax &dest $sc $sem $value
		};
	} else {
		spirv_asm { "<invalid atomic_max>" };
	}
	return {};
}

public void barrier(constexpr Scope sc, constexpr Loc loc, constexpr Sem se) {
	u32 sem = loc | se;
	spirv_asm {
		OpMemoryBarrier $sc $sem;
	};
}

public void sync_barrier(constexpr Scope esc, constexpr Scope sc, constexpr Loc loc, constexpr Sem se) {
	u32 sem = loc | se;
	spirv_asm {
		OpControlBarrier $esc $sc $sem;
	};
}

public void workgroup_sync_barrier() {
	sync_barrier(Scope.Workgroup, Scope.Workgroup, Loc.Workgroup, Sem.AcqRel);
}

public void image_sync_barrier() {
	sync_barrier(Scope.Workgroup, DEFAULT_SCOPE, Loc.Image | Loc.Workgroup, Sem.AcqRel);
}
