use std::{
	cell::UnsafeCell,
	ffi::{CStr, c_void},
	io::Write,
	mem::ManuallyDrop,
	sync::{Arc, Mutex},
};

use ash::{
	ext,
	khr,
	vk::{self, TaggedStructure},
};
use gpu_allocator::{
	AllocationSizes,
	AllocatorDebugSettings,
	vulkan::{Allocator, AllocatorCreateDesc},
};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle, RawDisplayHandle, RawWindowHandle};
use tracing::{error, info, trace, warn};

use crate::{
	Error,
	Result,
	device::{
		Device,
		DeviceInner,
		QueueData,
		Queues,
		descriptor::Descriptors,
		sampler::Samplers,
		shader::ShaderRuntime,
	},
};

#[derive(Default)]
pub struct DeviceBuilder<'a> {
	pub layers: &'a [&'static CStr],
	pub instance_extensions: &'a [&'static CStr],
	pub device_extensions: &'a [&'static CStr],
	pub window: Option<(&'a dyn HasWindowHandle, &'a dyn HasDisplayHandle)>,
	pub features: vk::PhysicalDeviceFeatures2<'a>,
}

impl<'a> DeviceBuilder<'a> {
	pub fn layers(mut self, layers: &'a [&'static CStr]) -> Self {
		self.layers = layers;
		self
	}

	pub fn instance_extensions(mut self, extensions: &'a [&'static CStr]) -> Self {
		self.instance_extensions = extensions;
		self
	}

	pub fn device_extensions(mut self, extensions: &'a [&'static CStr]) -> Self {
		self.device_extensions = extensions;
		self
	}

	/// # Safety
	/// `window` and `display` must outlive the `SurfaceKHR` returned from [`Self::build`].
	pub unsafe fn window(mut self, window: &'a dyn HasWindowHandle, display: &'a dyn HasDisplayHandle) -> Self {
		self.window = Some((window, display));
		self
	}

	/// Any extra features required should be appended to the `p_next` chain.
	pub fn features(mut self, features: vk::PhysicalDeviceFeatures2<'a>) -> Self {
		self.features = features;
		self
	}

	pub fn build(self) -> Result<(Device, vk::SurfaceKHR)> {
		let entry = Self::load_entry()?;

		let window = self
			.window
			.map(|(window, display)| (window.window_handle().unwrap(), display.display_handle().unwrap()));

		let (layers, extensions) = Self::get_instance_layers_and_extensions(
			&entry,
			window.map(|x| x.0.as_raw()),
			self.layers,
			self.instance_extensions,
		)?;
		let (instance, debug_utils_messenger) = Self::create_instance(&entry, &layers, &extensions)?;

		let surface_ext = khr::surface::Instance::new(&entry, &instance);
		let surface = window
			.map(|(window, display)| unsafe {
				Self::create_surface_inner(&entry, &instance, window.as_raw(), display.as_raw())
			})
			.transpose()?;

		let s = surface.unwrap_or(vk::SurfaceKHR::null());

		let (device, physical_device, queues, debug_utils_ext) = Self::create_device(
			&instance,
			surface.map(|s| (&surface_ext, s)),
			self.device_extensions,
			self.features,
		)?;

		let allocator = Allocator::new(&AllocatorCreateDesc {
			instance: instance.clone(),
			device: device.clone(),
			physical_device,
			debug_settings: AllocatorDebugSettings::default(),
			buffer_device_address: true,
			allocation_sizes: AllocationSizes::default(),
		})
		.map_err(|e| Error::Message(e.to_string()))?;

		let as_ext = khr::acceleration_structure::Device::new(&instance, &device);
		let rt_ext = khr::ray_tracing_pipeline::Device::new(&instance, &device);

		let descriptors = Descriptors::new(&device)?;
		let dev = Device {
			inner: Arc::new(DeviceInner {
				entry,
				instance,
				as_ext,
				debug_utils_ext,
				debug_utils_messenger,
				surface_ext,
				physical_device,
				queues,
				allocator: ManuallyDrop::new(Mutex::new(allocator)),
				shaders: UnsafeCell::new(None),
				rt_ext,
				descriptors,
				samplers: Mutex::new(Samplers::new()),
				device,
			}),
		};
		let shaders = ShaderRuntime::new(dev.clone());
		unsafe { *dev.inner.shaders.get() = Some(shaders) };

		Ok((dev, s))
	}

	fn load_entry() -> Result<ash::Entry> {
		match unsafe { ash::Entry::load() } {
			Ok(entry) => Ok(entry),
			Err(err) => Err(format!("Failed to load Vulkan: {err}").into()),
		}
	}

	fn get_instance_layers_and_extensions(
		entry: &ash::Entry, window: Option<RawWindowHandle>, layers: &[&'static CStr], extensions: &[&'static CStr],
	) -> Result<(Vec<&'static CStr>, Vec<&'static CStr>)> {
		unsafe {
			let mut exts: Vec<&CStr> = Self::get_surface_extensions(window)?.to_vec();
			if entry
				.enumerate_instance_extension_properties(None)?
				.into_iter()
				.any(|props| CStr::from_ptr(props.extension_name.as_ptr()) == ext::debug_utils::NAME)
			{
				exts.push(ext::debug_utils::NAME);
			}
			exts.extend_from_slice(extensions);

			Ok((layers.to_vec(), exts))
		}
	}

	fn get_surface_extensions(handle: Option<RawWindowHandle>) -> Result<[&'static CStr; 4]> {
		Ok(match handle {
			Some(handle) => match handle {
				RawWindowHandle::Win32(_) => [
					khr::surface::NAME,
					ext::surface_maintenance1::NAME,
					khr::get_surface_capabilities2::NAME,
					khr::win32_surface::NAME,
				],
				RawWindowHandle::Wayland(_) => [
					khr::surface::NAME,
					ext::surface_maintenance1::NAME,
					khr::get_surface_capabilities2::NAME,
					khr::wayland_surface::NAME,
				],
				RawWindowHandle::Xlib(_) => [
					khr::surface::NAME,
					ext::surface_maintenance1::NAME,
					khr::get_surface_capabilities2::NAME,
					khr::xlib_surface::NAME,
				],
				RawWindowHandle::Xcb(_) => [
					khr::surface::NAME,
					ext::surface_maintenance1::NAME,
					khr::get_surface_capabilities2::NAME,
					khr::xcb_surface::NAME,
				],
				RawWindowHandle::AndroidNdk(_) => [
					khr::surface::NAME,
					ext::surface_maintenance1::NAME,
					khr::get_surface_capabilities2::NAME,
					khr::android_surface::NAME,
				],
				_ => return Err(vk::Result::ERROR_EXTENSION_NOT_PRESENT.into()),
			},
			None => [
				khr::surface::NAME,
				ext::surface_maintenance1::NAME,
				khr::get_surface_capabilities2::NAME,
				if cfg!(target_os = "android") {
					khr::android_surface::NAME
				} else if cfg!(target_os = "windows") {
					khr::win32_surface::NAME
				} else if cfg!(target_os = "linux") {
					khr::xlib_surface::NAME
				} else {
					return Err(vk::Result::ERROR_EXTENSION_NOT_PRESENT.into());
				},
			],
		})
	}

	fn create_instance(
		entry: &ash::Entry, layers: &[&'static CStr], extensions: &[&'static CStr],
	) -> Result<(ash::Instance, Option<vk::DebugUtilsMessengerEXT>)> {
		unsafe {
			let instance = entry.create_instance(
				&vk::InstanceCreateInfo::default()
					.application_info(
						&vk::ApplicationInfo::default()
							.application_name(c"radiance")
							.engine_name(c"radiance")
							.api_version(vk::make_api_version(0, 1, 3, 0)),
					)
					.enabled_layer_names(&layers.iter().map(|x| x.as_ptr()).collect::<Vec<_>>())
					.enabled_extension_names(&extensions.iter().map(|x| x.as_ptr()).collect::<Vec<_>>()),
				None,
			)?;

			unsafe extern "system" fn debug_messenger(
				severity: vk::DebugUtilsMessageSeverityFlagsEXT, _: vk::DebugUtilsMessageTypeFlagsEXT,
				cb: *const vk::DebugUtilsMessengerCallbackDataEXT, _: *mut c_void,
			) -> u32 {
				unsafe {
					let cb = &*cb;
					let bt = std::backtrace::Backtrace::force_capture();
					match severity {
						vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => warn!(
							"[{}][{}] {}\n{bt:#?}",
							cb.message_id_number,
							CStr::from_ptr(cb.p_message_id_name).to_str().unwrap(),
							CStr::from_ptr(cb.p_message).to_str().unwrap(),
						),
						_ => error!(
							"[{}][{}] {}\n{bt:#?}",
							cb.message_id_number,
							CStr::from_ptr(cb.p_message_id_name).to_str().unwrap(),
							CStr::from_ptr(cb.p_message).to_str().unwrap(),
						),
					}
				}
				let _ = std::io::stdout().flush();
				0
			}

			let messenger = ext::debug_utils::Instance::new(entry, &instance)
				.create_debug_utils_messenger(
					&vk::DebugUtilsMessengerCreateInfoEXT::default()
						.message_severity(
							vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
								| vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
						)
						.message_type(vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION)
						.pfn_user_callback(Some(debug_messenger)),
					None,
				)
				.ok();

			Ok((instance, messenger))
		}
	}

	unsafe fn create_surface_inner(
		entry: &ash::Entry, instance: &ash::Instance, window: RawWindowHandle, display: RawDisplayHandle,
	) -> Result<vk::SurfaceKHR> {
		unsafe {
			match (window, display) {
				(RawWindowHandle::Win32(handle), _) => {
					let surface_fn = khr::win32_surface::Instance::new(entry, instance);
					surface_fn.create_win32_surface(
						&vk::Win32SurfaceCreateInfoKHR::default()
							.hinstance(handle.hinstance.map_or(0, |x| x.get()))
							.hwnd(handle.hwnd.get()),
						None,
					)
				},
				(RawWindowHandle::Wayland(window), RawDisplayHandle::Wayland(display)) => {
					let surface_fn = khr::wayland_surface::Instance::new(entry, instance);
					surface_fn.create_wayland_surface(
						&vk::WaylandSurfaceCreateInfoKHR::default()
							.display(display.display.as_ptr())
							.surface(window.surface.as_ptr()),
						None,
					)
				},
				(RawWindowHandle::Xlib(window), RawDisplayHandle::Xlib(display)) => {
					let surface_fn = khr::xlib_surface::Instance::new(entry, instance);
					surface_fn.create_xlib_surface(
						&vk::XlibSurfaceCreateInfoKHR::default()
							.dpy(display.display.map_or(std::ptr::null_mut(), |x| x.as_ptr()))
							.window(window.window),
						None,
					)
				},
				(RawWindowHandle::Xcb(window), RawDisplayHandle::Xcb(display)) => {
					let surface_fn = khr::xcb_surface::Instance::new(entry, instance);
					surface_fn.create_xcb_surface(
						&vk::XcbSurfaceCreateInfoKHR::default()
							.connection(display.connection.map_or(std::ptr::null_mut(), |x| x.as_ptr()))
							.window(window.window.get()),
						None,
					)
				},
				(RawWindowHandle::AndroidNdk(handle), _) => {
					let surface_fn = khr::android_surface::Instance::new(entry, instance);
					surface_fn.create_android_surface(
						&vk::AndroidSurfaceCreateInfoKHR::default().window(handle.a_native_window.as_ptr()),
						None,
					)
				},
				_ => Err(vk::Result::ERROR_EXTENSION_NOT_PRESENT),
			}
			.map_err(Into::into)
		}
	}

	fn create_device(
		instance: &ash::Instance, surface: Option<(&khr::surface::Instance, vk::SurfaceKHR)>,
		extensions: &[&'static CStr], features: vk::PhysicalDeviceFeatures2<'a>,
	) -> Result<(
		ash::Device,
		vk::PhysicalDevice,
		Queues<QueueData>,
		Option<ext::debug_utils::Device>,
	)> {
		let extensions = Self::get_device_extensions(extensions);
		trace!("using device extensions: {:?}", extensions);
		let extensions: Vec<_> = extensions.into_iter().map(|extension| extension.as_ptr()).collect();

		for (physical_device, queues, name) in Self::get_physical_devices(instance, surface)? {
			let props = unsafe { instance.get_physical_device_properties(physical_device) };
			if props.api_version < vk::make_api_version(0, 1, 3, 0) {
				continue;
			}

			trace!("trying device: {}", name);

			#[repr(C)]
			struct VkStructHeader {
				ty: vk::StructureType,
				next: *mut VkStructHeader,
			}

			let mut features = features;

			// Push the features if they don't already exist.
			let mut features12 = vk::PhysicalDeviceVulkan12Features::default();
			let mut features13 = vk::PhysicalDeviceVulkan13Features::default();
			let mut as_features = vk::PhysicalDeviceAccelerationStructureFeaturesKHR::default();
			let mut rt_features = vk::PhysicalDeviceRayTracingPipelineFeaturesKHR::default();
			let mut rq_features = vk::PhysicalDeviceRayQueryFeaturesKHR::default();
			let mut maint5_features = vk::PhysicalDeviceMaintenance5FeaturesKHR::default();
			let mut swapchain_maintenance1_features =
				vk::PhysicalDeviceSwapchainMaintenance1FeaturesEXT::default().swapchain_maintenance1(true);
			{
				let mut next = features.p_next as *mut VkStructHeader;
				let mut found_12 = false;
				let mut found_13 = false;
				let mut found_as = false;
				let mut found_rt = false;
				let mut found_rq = false;
				let mut found_maint5 = false;
				let mut found_swapchain_maintenance1 = false;
				while !next.is_null() {
					unsafe {
						match (*next).ty {
							vk::PhysicalDeviceVulkan12Features::STRUCTURE_TYPE => found_12 = true,
							vk::PhysicalDeviceVulkan13Features::STRUCTURE_TYPE => found_13 = true,
							vk::PhysicalDeviceAccelerationStructureFeaturesKHR::STRUCTURE_TYPE => found_as = true,
							vk::PhysicalDeviceRayTracingPipelineFeaturesKHR::STRUCTURE_TYPE => found_rt = true,
							vk::PhysicalDeviceRayQueryFeaturesKHR::STRUCTURE_TYPE => found_rq = true,
							vk::PhysicalDeviceMaintenance5FeaturesKHR::STRUCTURE_TYPE => found_maint5 = true,
							vk::PhysicalDeviceSwapchainMaintenance1FeaturesEXT::STRUCTURE_TYPE => {
								found_swapchain_maintenance1 = true
							},
							_ => {},
						}
						next = (*next).next;
					}
				}

				features = if !found_12 {
					features.push_next(&mut features12)
				} else {
					features
				};
				features = if !found_13 {
					features.push_next(&mut features13)
				} else {
					features
				};
				features = if !found_as {
					features.push_next(&mut as_features)
				} else {
					features
				};
				features = if !found_rt {
					features.push_next(&mut rt_features)
				} else {
					features
				};
				features = if !found_rq {
					features.push_next(&mut rq_features)
				} else {
					features
				};
				features = if !found_maint5 {
					features.push_next(&mut maint5_features)
				} else {
					features
				};
				features = if !found_swapchain_maintenance1 {
					features.push_next(&mut swapchain_maintenance1_features)
				} else {
					features
				};
			}

			let mut next = features.p_next as *mut VkStructHeader;
			while !next.is_null() {
				unsafe {
					match (*next).ty {
						vk::PhysicalDeviceVulkan12Features::STRUCTURE_TYPE => {
							let features12 = &mut *(next as *mut vk::PhysicalDeviceVulkan12Features);
							features12.descriptor_indexing = true as _;
							features12.runtime_descriptor_array = true as _;
							features12.descriptor_binding_partially_bound = true as _;
							features12.descriptor_binding_update_unused_while_pending = true as _;
							features12.descriptor_binding_variable_descriptor_count = true as _;
							features12.descriptor_binding_storage_buffer_update_after_bind = true as _;
							features12.descriptor_binding_sampled_image_update_after_bind = true as _;
							features12.descriptor_binding_storage_image_update_after_bind = true as _;
							features12.shader_storage_buffer_array_non_uniform_indexing = true as _;
							features12.shader_sampled_image_array_non_uniform_indexing = true as _;
							features12.shader_storage_image_array_non_uniform_indexing = true as _;
							features12.timeline_semaphore = true as _;
							features12.buffer_device_address = true as _;
							features12.vulkan_memory_model = true as _;
						},
						vk::PhysicalDeviceVulkan13Features::STRUCTURE_TYPE => {
							let features13 = &mut *(next as *mut vk::PhysicalDeviceVulkan13Features);
							features13.synchronization2 = true as _;
							features13.maintenance4 = true as _;
							features13.subgroup_size_control = true as _;
							features13.compute_full_subgroups = true as _;
						},
						vk::PhysicalDeviceAccelerationStructureFeaturesKHR::STRUCTURE_TYPE => {
							let as_features = &mut *(next as *mut vk::PhysicalDeviceAccelerationStructureFeaturesKHR);
							as_features.acceleration_structure = true as _;
							as_features.descriptor_binding_acceleration_structure_update_after_bind = true as _;
						},
						vk::PhysicalDeviceRayTracingPipelineFeaturesKHR::STRUCTURE_TYPE => {
							let rt_features = &mut *(next as *mut vk::PhysicalDeviceRayTracingPipelineFeaturesKHR);
							rt_features.ray_tracing_pipeline = true as _;
						},
						vk::PhysicalDeviceRayQueryFeaturesKHR::STRUCTURE_TYPE => {
							let rq_features = &mut *(next as *mut vk::PhysicalDeviceRayQueryFeaturesKHR);
							rq_features.ray_query = true as _;
						},
						vk::PhysicalDeviceMaintenance5FeaturesKHR::STRUCTURE_TYPE => {
							let maint5_features = &mut *(next as *mut vk::PhysicalDeviceMaintenance5FeaturesKHR);
							maint5_features.maintenance5 = true as _;
						},
						_ => {},
					}
					next = (*next).next;
				}
			}

			let info = vk::DeviceCreateInfo::default()
				.enabled_extension_names(&extensions)
				.push_next(&mut features);

			match unsafe {
				match queues {
					Queues::Multiple {
						graphics,
						compute,
						transfer,
					} => instance.create_device(
						physical_device,
						&info.queue_create_infos(&[
							vk::DeviceQueueCreateInfo::default()
								.queue_family_index(graphics)
								.queue_priorities(&[1.0]),
							vk::DeviceQueueCreateInfo::default()
								.queue_family_index(compute)
								.queue_priorities(&[1.0]),
							vk::DeviceQueueCreateInfo::default()
								.queue_family_index(transfer)
								.queue_priorities(&[1.0]),
						]),
						None,
					),
					Queues::Single(graphics) => instance.create_device(
						physical_device,
						&info.queue_create_infos(&[vk::DeviceQueueCreateInfo::default()
							.queue_family_index(graphics)
							.queue_priorities(&[1.0])]),
						None,
					),
				}
			} {
				Ok(device) => {
					info!("created device: {}", name);

					let queues = queues.try_map(|family| QueueData::new(&device, family))?;
					let debug = ext::debug_utils::Device::new(instance, &device);
					return Ok((device, physical_device, queues, Some(debug)));
				},
				Err(err) => {
					warn!("failed to create device: {}", err);
					continue;
				},
			};
		}

		Err(
			"failed to find suitable device: radiance needs mesh shaders, raytracing, and ReBAR"
				.to_string()
				.into(),
		)
	}

	fn get_device_extensions(extensions: &[&'static CStr]) -> Vec<&'static CStr> {
		let mut extensions = extensions.to_vec();
		extensions.extend([
			khr::swapchain::NAME,
			ext::swapchain_maintenance1::NAME,
			khr::acceleration_structure::NAME,
			khr::ray_query::NAME,
			khr::ray_tracing_pipeline::NAME,
			khr::ray_tracing_maintenance1::NAME,
			khr::deferred_host_operations::NAME,
			khr::maintenance5::NAME,
		]);
		extensions
	}

	fn get_physical_devices<'i>(
		instance: &'i ash::Instance, surface: Option<(&'i khr::surface::Instance, vk::SurfaceKHR)>,
	) -> Result<impl IntoIterator<Item = (vk::PhysicalDevice, Queues<u32>, String)> + 'i> {
		let iter = unsafe { instance.enumerate_physical_devices()? }
			.into_iter()
			.flat_map(move |device| {
				Self::get_device_suitability(instance, device, surface).map(|(q, n)| (device, q, n))
			});
		Ok(iter)
	}

	fn get_device_suitability(
		instance: &ash::Instance, device: vk::PhysicalDevice,
		surface: Option<(&khr::surface::Instance, vk::SurfaceKHR)>,
	) -> Option<(Queues<u32>, String)> {
		let properties = unsafe { instance.get_physical_device_properties(device) };

		if properties.api_version < vk::make_api_version(0, 1, 3, 0) {
			return None;
		}

		// Check if the device supports the queues required.
		let queues = Self::get_queue_families(instance, device, surface)?;

		Some((queues, unsafe {
			CStr::from_ptr(properties.device_name.as_ptr())
				.to_str()
				.unwrap()
				.to_string()
		}))
	}

	fn get_queue_families(
		instance: &ash::Instance, device: vk::PhysicalDevice,
		surface: Option<(&khr::surface::Instance, vk::SurfaceKHR)>,
	) -> Option<Queues<u32>> {
		let mut graphics = None;
		let mut transfer = None;
		let mut compute = None;

		let queue_families = unsafe { instance.get_physical_device_queue_family_properties(device) };

		for (i, family) in queue_families.iter().enumerate() {
			if family.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
				if let Some((surface_ext, surface)) = surface
					&& !unsafe {
						surface_ext
							.get_physical_device_surface_support(device, i as u32, surface)
							.ok()?
					} {
					return None;
				}
				graphics = Some(i as u32);
			} else if family.queue_flags.contains(vk::QueueFlags::COMPUTE) {
				compute = Some(i as u32);
			} else if family.queue_flags.contains(vk::QueueFlags::TRANSFER) {
				transfer = Some(i as u32);
			}
		}

		match (graphics, compute, transfer) {
			(Some(g), Some(c), Some(t)) => Some(Queues::Multiple {
				graphics: g,
				compute: c,
				transfer: t,
			}),
			(Some(g), ..) => Some(Queues::Single(g)),
			_ => None,
		}
	}
}

impl Device {
	/// Get a device builder.
	pub fn builder<'a>() -> DeviceBuilder<'a> { DeviceBuilder::default() }

	/// # Safety
	/// `window` and `display` must outlive the returned `SurfaceKHR`.
	pub unsafe fn create_surface(
		&self, window: &dyn HasWindowHandle, display: &dyn HasDisplayHandle,
	) -> Result<vk::SurfaceKHR> {
		unsafe {
			DeviceBuilder::create_surface_inner(
				&self.inner.entry,
				&self.inner.instance,
				window.window_handle().unwrap().as_raw(),
				display.display_handle().unwrap().as_raw(),
			)
		}
	}
}
