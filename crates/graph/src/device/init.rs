use std::{
	ffi::{c_void, CStr},
	mem::ManuallyDrop,
	sync::Mutex,
};

use ash::{
	extensions::{ext, khr},
	vk,
};
use gpu_allocator::{
	vulkan::{Allocator, AllocatorCreateDesc},
	AllocatorDebugSettings,
};
use raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle, RawDisplayHandle, RawWindowHandle};
use tracing::{error, info, trace, warn};

use crate::{
	device::{descriptor::Descriptors, Device, QueueData, Queues},
	Error,
	Result,
};

impl Device {
	const VALIDATION_LAYER: &'static CStr =
		unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_LAYER_KHRONOS_validation\0") };

	/// Create a device with no layers, extensions, or surface.
	///
	/// Enables the validation if built in debug (and if the layers are available).
	pub fn new() -> Result<Self> { unsafe { Self::new_inner(None, &[], &[], &[]).map(|x| x.0) } }

	/// Create a device with the given layers and extensions, but no surface.
	///
	/// Enables the validation if built in debug (and if the layers are available).
	pub fn with_layers_and_extensions(
		layers: &[&'static CStr], instance_extensions: &[&'static CStr], device_extensions: &[&'static CStr],
	) -> Result<Self> {
		unsafe { Self::new_inner(None, layers, instance_extensions, device_extensions).map(|x| x.0) }
	}

	/// Create a device with a surface on the window, but no layers and extensions.
	///
	/// Enables the validation if built in debug (and if the layers are available).
	///
	/// # Safety
	/// `window` and `display` must outlive the returned `SurfaceKHR`.
	pub unsafe fn with_window(
		window: &dyn HasRawWindowHandle, display: &dyn HasRawDisplayHandle,
	) -> Result<(Self, vk::SurfaceKHR)> {
		Self::new_inner(
			Some((window.raw_window_handle(), display.raw_display_handle())),
			&[],
			&[],
			&[],
		)
		.map(|x| (x.0, x.1.unwrap()))
	}

	/// Create a device with a surface on the window, with the given layers and extensions.
	///
	/// Enables the validation if built in debug (and if the layers are available).
	///
	/// # Safety
	/// `window` and `display` must outlive the returned `SurfaceKHR`.
	pub unsafe fn with_window_and_layers_and_extensions(
		window: &dyn HasRawWindowHandle, display: &dyn HasRawDisplayHandle, layers: &[&'static CStr],
		instance_extensions: &[&'static CStr], device_extensions: &[&'static CStr],
	) -> Result<(Self, vk::SurfaceKHR)> {
		Self::new_inner(
			Some((window.raw_window_handle(), display.raw_display_handle())),
			layers,
			instance_extensions,
			device_extensions,
		)
		.map(|x| (x.0, x.1.unwrap()))
	}

	/// # Safety
	/// `window` and `display` must outlive the returned `SurfaceKHR`.
	pub unsafe fn create_surface(
		&self, window: &dyn HasRawWindowHandle, display: &dyn HasRawDisplayHandle,
	) -> Result<vk::SurfaceKHR> {
		Self::create_surface_inner(
			&self.entry,
			&self.instance,
			window.raw_window_handle(),
			display.raw_display_handle(),
		)
	}

	unsafe fn new_inner(
		window: Option<(RawWindowHandle, RawDisplayHandle)>, layers: &[&'static CStr],
		instance_extensions: &[&'static CStr], device_extensions: &[&'static CStr],
	) -> Result<(Self, Option<vk::SurfaceKHR>)> {
		let entry = Self::load_entry()?;
		let (instance, debug_utils_ext, debug_messenger) =
			Self::create_instance(&entry, window.map(|x| x.0), layers, instance_extensions)?;

		let surface = window.map(|(window, display)| {
			let surface_ext = khr::Surface::new(&entry, &instance);
			let surface = Self::create_surface_inner(&entry, &instance, window, display);
			(surface, surface_ext)
		});

		let s = surface.as_ref().map(|(surface, _)| surface.clone()).transpose()?;
		let surface_ext = surface.map(|(_, ext)| ext);

		let (device, physical_device, queues) = Self::create_device(
			&instance,
			match s {
				Some(s) => Some((surface_ext.as_ref().unwrap(), s)),
				None => None,
			},
			device_extensions,
		)?;

		let allocator = Allocator::new(&AllocatorCreateDesc {
			instance: instance.clone(),
			device: device.clone(),
			physical_device,
			debug_settings: AllocatorDebugSettings::default(),
			buffer_device_address: false,
		})
		.map_err(|e| Error::Message(e.to_string()))?;

		let descriptors = Descriptors::new(&device)?;

		Ok((
			Self {
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

	fn create_instance(
		entry: &ash::Entry, window: Option<RawWindowHandle>, user_layers: &[&'static CStr],
		instance_extensions: &[&'static CStr],
	) -> Result<(ash::Instance, Option<ext::DebugUtils>, vk::DebugUtilsMessengerEXT)> {
		let (mut layers, mut extensions) = Self::get_instance_layers_and_extensions(entry, window)?;
		let has_validation = layers.contains(&Self::VALIDATION_LAYER);
		layers.extend_from_slice(user_layers);
		extensions.extend_from_slice(instance_extensions);

		let instance = unsafe {
			entry.create_instance(
				&vk::InstanceCreateInfo::builder()
					.application_info(
						&vk::ApplicationInfo::builder()
							.application_name(CStr::from_bytes_with_nul(b"radiance\0").unwrap())
							.engine_name(CStr::from_bytes_with_nul(b"radiance\0").unwrap())
							.api_version(ash::vk::make_api_version(0, 1, 3, 0)),
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
				let utils = ext::DebugUtils::new(&entry, &instance);
				let messenger = utils.create_debug_utils_messenger(&info, None)?;

				trace!("created debug utils messenger");
				(Some(utils), messenger)
			}
		} else {
			(None, vk::DebugUtilsMessengerEXT::null())
		};

		Ok((instance, utils, messenger))
	}

	fn get_instance_layers_and_extensions(
		entry: &ash::Entry, window: Option<RawWindowHandle>,
	) -> Result<(Vec<&'static CStr>, Vec<&'static CStr>)> {
		let validation = if cfg!(debug_assertions) {
			if entry
				.enumerate_instance_layer_properties()?
				.into_iter()
				.find(|props| unsafe { CStr::from_ptr(props.layer_name.as_ptr()) } == Self::VALIDATION_LAYER)
				.is_some()
			{
				Some(Self::VALIDATION_LAYER)
			} else {
				warn!("validation layer not found, continuing without");
				None
			}
		} else {
			None
		};

		let mut extensions: Vec<&CStr> = Self::get_window_extensions(window)?.to_vec();
		if validation.is_some() {
			extensions.push(ext::DebugUtils::name());
		}

		Ok((validation.into_iter().collect(), extensions))
	}

	fn get_window_extensions(handle: Option<RawWindowHandle>) -> Result<&'static [&'static CStr]> {
		Ok(match handle {
			Some(handle) => match handle {
				RawWindowHandle::Win32(_) => {
					const S: &'static [&'static CStr] = &[khr::Surface::name(), khr::Win32Surface::name()];
					S
				},
				RawWindowHandle::Wayland(_) => {
					const S: &'static [&'static CStr] = &[khr::Surface::name(), khr::WaylandSurface::name()];
					S
				},
				RawWindowHandle::Xlib(_) => {
					const S: &'static [&'static CStr] = &[khr::Surface::name(), khr::XlibSurface::name()];
					S
				},
				RawWindowHandle::Xcb(_) => {
					const S: &'static [&'static CStr] = &[khr::Surface::name(), khr::XcbSurface::name()];
					S
				},
				RawWindowHandle::AndroidNdk(_) => {
					const S: &'static [&'static CStr] = &[khr::Surface::name(), khr::AndroidSurface::name()];
					S
				},
				_ => return Err(vk::Result::ERROR_EXTENSION_NOT_PRESENT.into()),
			},
			None => &[],
		})
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

	fn select_physical_device(
		instance: &ash::Instance, surface: Option<(&khr::Surface, vk::SurfaceKHR)>,
	) -> Result<(vk::PhysicalDevice, Queues<u32>)> {
		for device in unsafe { instance.enumerate_physical_devices()? } {
			if let Some((strategy, name)) = Self::get_device_suitability(instance, device, surface) {
				info!("selected GPU `{}`", name);
				return Ok((device, strategy));
			}
		}

		Err("no suitable GPU found".to_string().into())
	}

	fn get_device_suitability(
		instance: &ash::Instance, device: vk::PhysicalDevice, surface: Option<(&khr::Surface, vk::SurfaceKHR)>,
	) -> Option<(Queues<u32>, String)> {
		let properties = unsafe { instance.get_physical_device_properties(device) };

		if properties.api_version < ash::vk::make_api_version(0, 1, 3, 0) {
			return None;
		}

		let mut features13 = vk::PhysicalDeviceVulkan13Features::default();
		let mut features12 = vk::PhysicalDeviceVulkan12Features::default();
		let mut features = vk::PhysicalDeviceFeatures2::builder()
			.push_next(&mut features12)
			.push_next(&mut features13);
		unsafe { instance.get_physical_device_features2(device, &mut features) }

		// Check if the device supports the features required.
		if features.features.sampler_anisotropy == false as _ {
			return None;
		}
		if features12.descriptor_indexing == false as _
			|| features12.runtime_descriptor_array == false as _
			|| features12.descriptor_binding_partially_bound == false as _
			|| features12.descriptor_binding_update_unused_while_pending == false as _
			|| features12.descriptor_binding_storage_buffer_update_after_bind == false as _
			|| features12.descriptor_binding_sampled_image_update_after_bind == false as _
			|| features12.descriptor_binding_storage_image_update_after_bind == false as _
			|| features12.shader_storage_buffer_array_non_uniform_indexing == false as _
			|| features12.shader_sampled_image_array_non_uniform_indexing == false as _
			|| features12.shader_storage_image_array_non_uniform_indexing == false as _
			|| features12.timeline_semaphore == false as _
		{
			return None;
		}
		if features13.dynamic_rendering == false as _ || features13.synchronization2 == false as _ {
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

	fn create_device(
		instance: &ash::Instance, surface: Option<(&khr::Surface, vk::SurfaceKHR)>, user_extensions: &[&'static CStr],
	) -> Result<(ash::Device, vk::PhysicalDevice, Queues<QueueData>)> {
		let (physical_device, queues) = Self::select_physical_device(instance, surface)?;
		let mut extensions = if surface.is_some() {
			vec![khr::Swapchain::name()]
		} else {
			Vec::new()
		};
		extensions.extend_from_slice(user_extensions);
		trace!("using device extensions: {:?}", extensions);
		let extensions: Vec<_> = extensions.into_iter().map(|extension| extension.as_ptr()).collect();

		let features = vk::PhysicalDeviceFeatures::builder().sampler_anisotropy(true);
		let mut features12 = vk::PhysicalDeviceVulkan12Features::builder()
			.descriptor_indexing(true)
			.runtime_descriptor_array(true)
			.descriptor_binding_partially_bound(true)
			.descriptor_binding_update_unused_while_pending(true)
			.descriptor_binding_storage_buffer_update_after_bind(true)
			.descriptor_binding_sampled_image_update_after_bind(true)
			.descriptor_binding_storage_image_update_after_bind(true)
			.shader_storage_buffer_array_non_uniform_indexing(true)
			.shader_sampled_image_array_non_uniform_indexing(true)
			.shader_storage_image_array_non_uniform_indexing(true)
			.timeline_semaphore(true);
		let mut features13 = vk::PhysicalDeviceVulkan13Features::builder()
			.dynamic_rendering(true)
			.synchronization2(true);
		let info = vk::DeviceCreateInfo::builder()
			.enabled_extension_names(&extensions)
			.enabled_features(&features)
			.push_next(&mut features12)
			.push_next(&mut features13);

		let device = unsafe {
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
			}?
		};

		let queues = queues.map_ref(|index| QueueData {
			queue: Mutex::new(unsafe { device.get_device_queue(*index, 0) }),
			family: *index,
		});

		Ok((device, physical_device, queues))
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
			// let b = std::backtrace::Backtrace::force_capture();
			// error!("debug callback occurred at\n{}", b);
		},
		_ => {},
	}

	0
}
