use std::{
	ffi::{c_void, CStr},
	mem::ManuallyDrop,
	sync::Mutex,
};

use ash::{
	extensions::{ext, khr},
	vk,
	vk::TaggedStructure,
};
use gpu_allocator::{
	vulkan::{Allocator, AllocatorCreateDesc},
	AllocationSizes,
	AllocatorDebugSettings,
};
use raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle, RawDisplayHandle, RawWindowHandle};
use tracing::{error, info, trace, warn};

use crate::{
	device::{descriptor::Descriptors, Device, QueueData, Queues},
	Error,
	Result,
};

const VALIDATION_LAYER: &'static CStr =
	unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_LAYER_KHRONOS_validation\0") };

pub struct DeviceBuilder<'a> {
	pub validation: bool,
	pub layers: &'a [&'static CStr],
	pub instance_extensions: &'a [&'static CStr],
	pub device_extensions: &'a [&'static CStr],
	pub window: Option<(&'a dyn HasRawWindowHandle, &'a dyn HasRawDisplayHandle)>,
	pub features: vk::PhysicalDeviceFeatures2Builder<'a>,
}

impl Default for DeviceBuilder<'_> {
	fn default() -> Self {
		Self {
			validation: false,
			layers: &[],
			instance_extensions: &[],
			device_extensions: &[],
			window: None,
			features: vk::PhysicalDeviceFeatures2::builder(),
		}
	}
}

impl<'a> DeviceBuilder<'a> {
	pub fn validation(mut self, validation: bool) -> Self {
		self.validation = validation;
		self
	}

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
	pub unsafe fn window(mut self, window: &'a dyn HasRawWindowHandle, display: &'a dyn HasRawDisplayHandle) -> Self {
		self.window = Some((window, display));
		self
	}

	/// Aany extra features required should be appended to the `p_next` chain.
	pub fn features(mut self, features: vk::PhysicalDeviceFeatures2Builder<'a>) -> Self {
		self.features = features;
		self
	}

	pub fn build(self) -> Result<(Device, vk::SurfaceKHR)> {
		let entry = Self::load_entry()?;

		let window = self
			.window
			.map(|(window, display)| (window.raw_window_handle(), display.raw_display_handle()));

		let (layers, extensions) = Self::get_instance_layers_and_extensions(
			&entry,
			window.map(|x| x.0),
			self.validation,
			self.layers,
			self.instance_extensions,
		)?;
		let (instance, debug_utils_ext, debug_messenger) = Self::create_instance(&entry, &layers, &extensions)?;

		let surface = window.map(|(window, display)| {
			let surface_ext = khr::Surface::new(&entry, &instance);
			let surface = unsafe { Self::create_surface_inner(&entry, &instance, window, display) };
			(surface, surface_ext)
		});

		let s = surface
			.as_ref()
			.map(|(surface, _)| surface.clone())
			.transpose()?
			.unwrap_or(vk::SurfaceKHR::null());
		let surface_ext = surface.map(|(_, ext)| ext);

		let (device, physical_device, queues) = Self::create_device(
			&instance,
			surface_ext.as_ref().map(|ext| (ext, s)),
			self.device_extensions,
			self.features,
		)?;

		let allocator = Allocator::new(&AllocatorCreateDesc {
			instance: instance.clone(),
			device: device.clone(),
			physical_device,
			debug_settings: AllocatorDebugSettings::default(),
			buffer_device_address: false,
			allocation_sizes: AllocationSizes::default(),
		})
		.map_err(|e| Error::Message(e.to_string()))?;

		let descriptors = Descriptors::new(&device)?;

		Ok((
			Device {
				entry,
				instance,
				debug_messenger,
				debug_utils_ext,
				surface_ext,
				device,
				physical_device,
				queues,
				allocator: ManuallyDrop::new(Mutex::new(allocator)),
				descriptors,
			},
			s,
		))
	}

	fn load_entry() -> Result<ash::Entry> {
		match unsafe { ash::Entry::load() } {
			Ok(entry) => Ok(entry),
			Err(err) => Err(format!("Failed to load Vulkan: {}", err).into()),
		}
	}

	fn get_instance_layers_and_extensions(
		entry: &ash::Entry, window: Option<RawWindowHandle>, validation: bool, layers: &[&'static CStr],
		extensions: &[&'static CStr],
	) -> Result<(Vec<&'static CStr>, Vec<&'static CStr>)> {
		let validation = if validation {
			if entry
				.enumerate_instance_layer_properties()?
				.into_iter()
				.any(|props| unsafe { CStr::from_ptr(props.layer_name.as_ptr()) } == VALIDATION_LAYER)
			{
				Some(VALIDATION_LAYER)
			} else {
				warn!("validation layer not found, continuing without");
				None
			}
		} else {
			None
		};

		let mut exts: Vec<&CStr> = Self::get_surface_extensions(window)?.to_vec();
		if validation.is_some()
			&& entry
				.enumerate_instance_extension_properties(None)?
				.into_iter()
				.any(|props| unsafe { CStr::from_ptr(props.extension_name.as_ptr()) } == ext::DebugUtils::name())
		{
			exts.push(ext::DebugUtils::name());
		}
		exts.extend_from_slice(extensions);

		Ok((
			validation.into_iter().chain(layers.into_iter().copied()).collect(),
			exts,
		))
	}

	fn get_surface_extensions(handle: Option<RawWindowHandle>) -> Result<&'static [&'static CStr]> {
		Ok(match handle {
			Some(handle) => match handle {
				RawWindowHandle::Win32(_) => {
					const S: &[&CStr] = &[khr::Surface::name(), khr::Win32Surface::name()];
					S
				},
				RawWindowHandle::Wayland(_) => {
					const S: &[&CStr] = &[khr::Surface::name(), khr::WaylandSurface::name()];
					S
				},
				RawWindowHandle::Xlib(_) => {
					const S: &[&CStr] = &[khr::Surface::name(), khr::XlibSurface::name()];
					S
				},
				RawWindowHandle::Xcb(_) => {
					const S: &[&CStr] = &[khr::Surface::name(), khr::XcbSurface::name()];
					S
				},
				RawWindowHandle::AndroidNdk(_) => {
					const S: &[&CStr] = &[khr::Surface::name(), khr::AndroidSurface::name()];
					S
				},
				_ => return Err(vk::Result::ERROR_EXTENSION_NOT_PRESENT.into()),
			},
			None => &[],
		})
	}

	fn create_instance(
		entry: &ash::Entry, layers: &[&'static CStr], extensions: &[&'static CStr],
	) -> Result<(ash::Instance, Option<ext::DebugUtils>, vk::DebugUtilsMessengerEXT)> {
		let has_validation = layers.contains(&VALIDATION_LAYER);

		let instance = unsafe {
			entry.create_instance(
				&vk::InstanceCreateInfo::builder()
					.application_info(
						&vk::ApplicationInfo::builder()
							.application_name(CStr::from_bytes_with_nul(b"radiance\0").unwrap())
							.engine_name(CStr::from_bytes_with_nul(b"radiance\0").unwrap())
							.api_version(vk::make_api_version(0, 1, 3, 0)),
					)
					.enabled_layer_names(&layers.into_iter().map(|x| x.as_ptr()).collect::<Vec<_>>())
					.enabled_extension_names(&extensions.into_iter().map(|x| x.as_ptr()).collect::<Vec<_>>()),
				None,
			)?
		};

		let (utils, messenger) = if has_validation {
			let info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
				.message_severity(
					vk::DebugUtilsMessageSeverityFlagsEXT::INFO | {
						vk::DebugUtilsMessageSeverityFlagsEXT::WARNING | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
					},
				)
				.message_type(
					vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
						| vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE
						| vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION,
				)
				.pfn_user_callback(Some(debug_callback));

			unsafe {
				let utils = ext::DebugUtils::new(entry, &instance);
				let messenger = utils.create_debug_utils_messenger(&info, None)?;

				trace!("created debug utils messenger");
				(Some(utils), messenger)
			}
		} else {
			(None, vk::DebugUtilsMessengerEXT::null())
		};

		Ok((instance, utils, messenger))
	}

	unsafe fn create_surface_inner(
		entry: &ash::Entry, instance: &ash::Instance, window: RawWindowHandle, display: RawDisplayHandle,
	) -> Result<vk::SurfaceKHR> {
		match (window, display) {
			(RawWindowHandle::Win32(handle), _) => {
				let surface_fn = khr::Win32Surface::new(entry, instance);
				surface_fn.create_win32_surface(
					&vk::Win32SurfaceCreateInfoKHR::builder()
						.hinstance(handle.hinstance)
						.hwnd(handle.hwnd),
					None,
				)
			},
			(RawWindowHandle::Wayland(window), RawDisplayHandle::Wayland(display)) => {
				let surface_fn = khr::WaylandSurface::new(entry, instance);
				surface_fn.create_wayland_surface(
					&vk::WaylandSurfaceCreateInfoKHR::builder()
						.display(display.display)
						.surface(window.surface),
					None,
				)
			},
			(RawWindowHandle::Xlib(window), RawDisplayHandle::Xlib(display)) => {
				let surface_fn = khr::XlibSurface::new(entry, instance);
				surface_fn.create_xlib_surface(
					&vk::XlibSurfaceCreateInfoKHR::builder()
						.dpy(display.display as *mut _)
						.window(window.window),
					None,
				)
			},
			(RawWindowHandle::Xcb(window), RawDisplayHandle::Xcb(display)) => {
				let surface_fn = khr::XcbSurface::new(entry, instance);
				surface_fn.create_xcb_surface(
					&vk::XcbSurfaceCreateInfoKHR::builder()
						.connection(display.connection)
						.window(window.window),
					None,
				)
			},
			(RawWindowHandle::AndroidNdk(handle), _) => {
				let surface_fn = khr::AndroidSurface::new(entry, instance);
				surface_fn.create_android_surface(
					&vk::AndroidSurfaceCreateInfoKHR::builder().window(handle.a_native_window),
					None,
				)
			},
			_ => Err(vk::Result::ERROR_EXTENSION_NOT_PRESENT),
		}
		.map_err(Into::into)
	}

	fn create_device(
		instance: &ash::Instance, surface: Option<(&khr::Surface, vk::SurfaceKHR)>, extensions: &[&'static CStr],
		features: vk::PhysicalDeviceFeatures2Builder,
	) -> Result<(ash::Device, vk::PhysicalDevice, Queues<QueueData>)> {
		let extensions = Self::get_device_extensions(surface.is_some(), extensions);
		trace!("using device extensions: {:?}", extensions);
		let extensions: Vec<_> = extensions.into_iter().map(|extension| extension.as_ptr()).collect();

		for (physical_device, queues, name) in Self::get_physical_devices(instance, surface)? {
			let props = unsafe { instance.get_physical_device_properties(physical_device) };
			if props.api_version < vk::make_api_version(0, 1, 3, 0) {
				continue;
			}

			trace!("trying device: {}", name);

			let mut features: vk::PhysicalDeviceFeatures2Builder = unsafe { std::mem::transmute(features.clone()) };

			#[repr(C)]
			struct VkStructHeader {
				ty: vk::StructureType,
				next: *mut VkStructHeader,
			}

			// Push the features if they don't already exist.
			let mut features12 = vk::PhysicalDeviceVulkan12Features::default();
			let mut features13 = vk::PhysicalDeviceVulkan13Features::default();
			{
				let mut next = features.p_next as *mut VkStructHeader;
				let mut found_12 = false;
				let mut found_13 = false;
				while !next.is_null() {
					unsafe {
						match (*next).ty {
							vk::PhysicalDeviceVulkan12Features::STRUCTURE_TYPE => found_12 = true,
							vk::PhysicalDeviceVulkan13Features::STRUCTURE_TYPE => found_13 = true,
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
							features12.descriptor_binding_storage_buffer_update_after_bind = true as _;
							features12.descriptor_binding_sampled_image_update_after_bind = true as _;
							features12.descriptor_binding_storage_image_update_after_bind = true as _;
							features12.shader_storage_buffer_array_non_uniform_indexing = true as _;
							features12.shader_sampled_image_array_non_uniform_indexing = true as _;
							features12.shader_storage_image_array_non_uniform_indexing = true as _;
							features12.timeline_semaphore = true as _;
						},
						vk::PhysicalDeviceVulkan13Features::STRUCTURE_TYPE => {
							let features13 = &mut *(next as *mut vk::PhysicalDeviceVulkan13Features);
							features13.synchronization2 = true as _;
						},
						_ => {},
					}
					next = (*next).next;
				}
			}

			let info = vk::DeviceCreateInfo::builder()
				.enabled_extension_names(&extensions)
				.push_next(&mut features);

			match unsafe {
				match queues {
					Queues::Separate {
						graphics,
						compute,
						transfer,
					} => instance.create_device(
						physical_device,
						&info.queue_create_infos(&[
							vk::DeviceQueueCreateInfo::builder()
								.queue_family_index(graphics)
								.queue_priorities(&[1.0])
								.build(),
							vk::DeviceQueueCreateInfo::builder()
								.queue_family_index(compute)
								.queue_priorities(&[1.0])
								.build(),
							vk::DeviceQueueCreateInfo::builder()
								.queue_family_index(transfer)
								.queue_priorities(&[1.0])
								.build(),
						]),
						None,
					),
					Queues::Single(graphics) => instance.create_device(
						physical_device,
						&info.queue_create_infos(&[vk::DeviceQueueCreateInfo::builder()
							.queue_family_index(graphics)
							.queue_priorities(&[1.0])
							.build()]),
						None,
					),
				}
			} {
				Ok(device) => {
					info!("created device: {}", name);

					let queues = queues.map_ref(|index| QueueData {
						queue: Mutex::new(unsafe { device.get_device_queue(*index, 0) }),
						family: *index,
					});

					return Ok((device, physical_device, queues));
				},
				Err(err) => {
					warn!("failed to create device: {}", err);
					continue;
				},
			};
		}

		Err("failed to find suitable device".to_string().into())
	}

	fn get_device_extensions(swapchain: bool, extensions: &[&'static CStr]) -> Vec<&'static CStr> {
		let mut extensions = extensions.to_vec();
		if swapchain {
			extensions.push(khr::Swapchain::name());
		}
		extensions
	}

	fn get_physical_devices<'i>(
		instance: &'i ash::Instance, surface: Option<(&'i khr::Surface, vk::SurfaceKHR)>,
	) -> Result<impl IntoIterator<Item = (vk::PhysicalDevice, Queues<u32>, String)> + 'i> {
		let iter = unsafe { instance.enumerate_physical_devices()? }
			.into_iter()
			.flat_map(move |device| {
				Self::get_device_suitability(instance, device, surface).map(|(q, n)| (device, q, n))
			});
		Ok(iter)
	}

	fn get_device_suitability(
		instance: &ash::Instance, device: vk::PhysicalDevice, surface: Option<(&khr::Surface, vk::SurfaceKHR)>,
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
		instance: &ash::Instance, device: vk::PhysicalDevice, surface: Option<(&khr::Surface, vk::SurfaceKHR)>,
	) -> Option<Queues<u32>> {
		let mut graphics = None;
		let mut transfer = None;
		let mut compute = None;

		let queue_families = unsafe { instance.get_physical_device_queue_family_properties(device) };

		for (i, family) in queue_families.iter().enumerate() {
			if family.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
				if let Some((surface_ext, surface)) = surface {
					if !unsafe {
						surface_ext
							.get_physical_device_surface_support(device, i as u32, surface)
							.ok()?
					} {
						return None;
					}
				}
				graphics = Some(i as u32);
			} else if family.queue_flags.contains(vk::QueueFlags::COMPUTE) {
				compute = Some(i as u32);
			} else if family.queue_flags.contains(vk::QueueFlags::TRANSFER) {
				transfer = Some(i as u32);
			}
		}

		match (graphics, compute, transfer) {
			(Some(g), Some(c), Some(t)) => Some(Queues::Separate {
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
		&self, window: &dyn HasRawWindowHandle, display: &dyn HasRawDisplayHandle,
	) -> Result<vk::SurfaceKHR> {
		DeviceBuilder::create_surface_inner(
			&self.entry,
			&self.instance,
			window.raw_window_handle(),
			display.raw_display_handle(),
		)
	}
}

unsafe extern "system" fn debug_callback(
	message_severity: vk::DebugUtilsMessageSeverityFlagsEXT, _message_types: vk::DebugUtilsMessageTypeFlagsEXT,
	p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT, _p_user_data: *mut c_void,
) -> vk::Bool32 {
	match message_severity {
		vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => {
			warn!("{}", CStr::from_ptr((*p_callback_data).p_message).to_str().unwrap());
			// let b = Backtrace::force_capture();
			// trace!("debug callback occurred at\n{}", b);
		},
		vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => {
			error!("{}", CStr::from_ptr((*p_callback_data).p_message).to_str().unwrap());
			let b = std::backtrace::Backtrace::force_capture();
			error!("debug callback occurred at\n{}", b);
		},
		_ => {},
	}

	0
}
