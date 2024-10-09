use std::{arch::naked_asm, marker::PhantomPinned, pin::Pin};

#[repr(C)]
#[derive(Default)]
pub struct Fiber {
	rip: u64,
	rsp: u64,
	rbx: u64,
	rbp: u64,
	r12: u64,
	r13: u64,
	r14: u64,
	r15: u64,
	_pin: PhantomPinned,
}

impl Fiber {
	pub fn null() -> Self {
		Self {
			rip: 0,
			rsp: 0,
			rbx: 0,
			rbp: 0,
			r12: 0,
			r13: 0,
			r14: 0,
			r15: 0,
			_pin: PhantomPinned,
		}
	}

	pub fn init(
		self: &mut Pin<&mut Self>, stack: *mut u8, size: u64, exec: extern "sysv64" fn(*mut Self, *mut u8),
		arg: *mut u8,
	) {
		let this = unsafe { &mut self.as_mut().get_unchecked_mut() };
		this.rsp = stack as u64 + size;
		this.rip = Self::internal_trampoline as _;
		this.rbx = exec as _;
		this.rbp = this.rsp;
		this.r12 = this as *mut _ as _;
		this.r13 = arg as _;
	}

	#[naked]
	unsafe extern "sysv64" fn internal_trampoline() {
		naked_asm!("mov rdi, r12", "mov rsi, r13", "jmp rbx",);
	}
}

#[naked]
pub unsafe extern "sysv64" fn swap_context(store: *mut Fiber, load: *mut Fiber) {
	naked_asm!(
		"mov r8, [rsp]", // mov r8, return-addr
		"mov [rdi + 8 * 0], r8",
		"lea r8, [rsp + 8]", // r8 = rsp + 8 (whatever it was before the call)
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
		"xor eax, eax",
		"jmp r8",
	);
}

#[test]
fn can_swap_context() {
	extern "sysv64" fn other_fn(us: *mut Fiber, arg: *mut u8) {
		println!("hello from other fiber");
		unsafe {
			swap_context(us, arg as _);
		}
	}

	let mut stack = vec![0; 4096];
	let mut us = Fiber::null();
	let mut other = Fiber::null();
	let mut other = std::pin::pin!(other);
	other.init(stack.as_mut_ptr(), stack.len() as u64, other_fn, &mut us as *mut _ as _);
	unsafe {
		swap_context(&mut us, other.get_unchecked_mut());
	}
	println!("we're back");
}
