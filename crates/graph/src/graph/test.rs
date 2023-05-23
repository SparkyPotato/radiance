use super::*;

mod data {
	use std::panic::AssertUnwindSafe;

	use super::*;
	use crate::arena::IteratorAlloc;

	#[test]
	fn basic() {
		let mut arena = Arena::new();
		let device = Device::new().unwrap();
		let mut graph = RenderGraph::new(&device).unwrap();

		for _ in 0..2 {
			arena.reset();
			let mut frame = graph.frame(&arena);

			struct Data<'graph>(Vec<usize, &'graph Arena>);

			let mut p1 = frame.pass("Pass 1");
			let (set, get) = p1.data_output::<Data>();
			p1.build(|mut ctx| {
				let v = [1, 2].into_iter().collect_in(ctx.arena);
				ctx.set_data(set, Data(v));
			});

			let mut p2 = frame.pass("Pass 2");
			p2.data_input(&get);
			p2.build(|mut ctx| {
				let data = ctx.get_data(get);
				assert_eq!(data.0, vec![1, 2]);
			});

			frame.run(&device).unwrap();
		}
	}

	#[test]
	#[should_panic]
	fn try_access_data_from_previous_frame() {
		let mut arena = Arena::new();
		let device = Device::new().unwrap();
		let mut graph = RenderGraph::new(&device).unwrap();

		struct Data<'graph>(Vec<usize, &'graph Arena>);

		let mut id: Option<RefId<Data>> = None;
		for _ in 0..2 {
			if std::panic::catch_unwind(AssertUnwindSafe(|| arena.reset())).is_err() {
				// Don't panic if we failed to reset.
				return;
			}

			let mut frame = graph.frame(&arena);

			let mut p1 = frame.pass("Pass 1");
			let (set, get) = p1.data_output::<Data>();
			p1.build(|mut ctx| {
				let mut v = Vec::new_in(ctx.arena);
				v.push(1);
				v.push(2);
				ctx.set_data(set, Data(v));
			});

			let ref_id = get.to_ref();

			let mut p2 = frame.pass("Pass 2");
			p2.data_input_ref(ref_id);
			p2.build(|mut ctx| {
				let data = ctx.get_data_ref(ref_id);
				assert_eq!(data.0, vec![1, 2]);
				if let Some(id) = id {
					let _ = ctx.get_data_ref(id);
				}
			});

			frame.run(&device).unwrap();

			id = Some(ref_id);
		}
	}
}
