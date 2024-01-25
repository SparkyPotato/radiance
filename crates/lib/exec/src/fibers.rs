use std::arch::asm;

pub struct Fiber {
	context: ThreadContext,
	stack: *mut u8,
}

impl Fiber {
	pub fn new(stack_base: *mut u8) {}
}

#[repr(C)]
struct ThreadContext {
	rip: u64,
	rsp: u64,
	rbx: u64,
	rbp: u64,
	r12: u64,
	r13: u64,
	r14: u64,
	r15: u64,
}

#[naked]
unsafe extern "sysv64" fn swap_context(store: *mut ThreadContext, load: *mut ThreadContext) {
	asm!(
		"mov r8, [rsp + 8 * 0]",
		"mov [rdi + 8 * 0], r8",
		"mov r8, [rdi + 8 * 1]",
		"mov [rdi + 8 * 1], r8",
		"mov [rdi + 8 * 2], rbx",
		"mov [rdi + 8 * 3], rbp",
		"mov [rdi + 8 * 4], r12",
		"mov [rdi + 8 * 5], r13",
		"mov [rdi + 8 * 6], r14",
		"mov [rdi + 8 * 7], r15",
		"mov r8, [rsi + 8 * 0]",
		"mov rsp, [rsi + 8 * 1]",
		"mov rbx, [rsi + 8 * 2]",
		"mov rbp, [rsi + 8 * 3]",
		"mov r12, [rsi + 8 * 4]",
		"mov r13, [rsi + 8 * 5]",
		"mov r14, [rsi + 8 * 6]",
		"mov r15, [rsi + 8 * 7]",
		"push r8",
		"xor eax, eax",
		"ret",
		options(noreturn)
	);
}

