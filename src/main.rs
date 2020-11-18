use ash::version::{EntryV1_0, InstanceV1_0, DeviceV1_0};
use const_cstr::const_cstr;
use bitflags::bitflags;

const ENABLE_VALIDATION: bool = true;

const FRAMES_IN_FLIGHT: u32 = 2;

pub struct Instance {
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    pub validation_enabled: bool,
    ext_debug_utils: ash::extensions::ext::DebugUtils,
    debug_messenger: ash::vk::DebugUtilsMessengerEXT,
    ext_surface: ash::extensions::khr::Surface
}

unsafe extern "system" fn debug_callback(
    _message_severity: ash::vk::DebugUtilsMessageSeverityFlagsEXT,
    _message_type: ash::vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const ash::vk::DebugUtilsMessengerCallbackDataEXT,
    _p_user_data: *mut std::os::raw::c_void) -> ash::vk::Bool32
{
    let msg = std::ffi::CStr::from_ptr((*p_callback_data).p_message);
    println!("[Vulkan] {:?}", msg);
    ash::vk::FALSE
}

const_cstr! {
    VALIDATION_LAYER_NAME = "VK_LAYER_KHRONOS_validation";
    SWAPCHAIN_EXT_NAME = "VK_KHR_swapchain";
    PHYS_DEV_PROP2_EXT_NAME = "VK_KHR_get_physical_device_properties2";
}

impl Instance {
    pub fn new(window: &winit::window::Window, enable_validation: bool) -> Self {
        let entry = ash::Entry::new().expect("Failed to initialize Vulkan loader");

        let layer_properties = entry.enumerate_instance_layer_properties().expect("Failed to enumerate instance layer properties");
        let available_layer_names = layer_properties.iter().map(|s| unsafe { std::ffi::CStr::from_ptr(s.layer_name.as_ptr()) }).collect::<Vec<_>>();
        println!("Available layers: {:?}", available_layer_names);

        let mut layers: smallvec::SmallVec<[*const std::os::raw::c_char; 4]> = smallvec::smallvec![];
        if enable_validation {
            layers.push(VALIDATION_LAYER_NAME.as_ptr());
        }

        let extension_properties = entry.enumerate_instance_extension_properties().expect("Failed to enumerate instanance extension properties");
        let available_extension_names = extension_properties.iter().map(|s| unsafe { std::ffi::CStr::from_ptr(s.extension_name.as_ptr()) }).collect::<Vec<_>>();
        println!("Available extensions: {:?}", available_extension_names);

        println!("Enabling layers: {:?}", layers.iter().map(|s| unsafe { std::ffi::CStr::from_ptr(*s) }).collect::<smallvec::SmallVec<[_; 4]>>());

        let surface_extensions = ash_window::enumerate_required_extensions(window).unwrap();
        let mut extensions = surface_extensions.iter().map(|s| s.as_ptr()).collect::<smallvec::SmallVec<[_; 8]>>();
        extensions.push(ash::extensions::ext::DebugUtils::name().as_ptr());
        extensions.push(PHYS_DEV_PROP2_EXT_NAME.as_ptr());

        println!("Enabling extensions: {:?}", extensions.iter().map(|s| unsafe { std::ffi::CStr::from_ptr(*s) }).collect::<smallvec::SmallVec<[_; 8]>>());

        let instance_create_info = ash::vk::InstanceCreateInfo {
            enabled_layer_count: layers.len() as u32,
            pp_enabled_layer_names: layers.as_ptr(),
            enabled_extension_count: extensions.len() as u32,
            pp_enabled_extension_names: extensions.as_ptr(),
            ..Default::default()
        };
        let instance = unsafe { entry.create_instance(&instance_create_info, None).expect("Failed to create Vulkan instance") };

        let ext_debug_utils = ash::extensions::ext::DebugUtils::new(&entry, &instance);
        let debug_create_info = ash::vk::DebugUtilsMessengerCreateInfoEXT {
            message_severity: ash::vk::DebugUtilsMessageSeverityFlagsEXT::WARNING | ash::vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
            message_type: ash::vk::DebugUtilsMessageTypeFlagsEXT::GENERAL | ash::vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE | ash::vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION,
            pfn_user_callback: Some(debug_callback),
            ..Default::default()
        };
        let debug_messenger = unsafe { ext_debug_utils.create_debug_utils_messenger(&debug_create_info, None).expect("Failed to create debug utils messenger") };

        let ext_surface = ash::extensions::khr::Surface::new(&entry, &instance);

        Instance {
            entry,
            instance,
            validation_enabled: enable_validation,
            ext_debug_utils,
            debug_messenger,
            ext_surface
        }
    }

    pub fn release_resources(&self) {
        unsafe {
            self.ext_debug_utils.destroy_debug_utils_messenger(self.debug_messenger, None);
            self.instance.destroy_instance(None);
        }
    }
}

pub struct WindowSurface {
    pub surface: ash::vk::SurfaceKHR
}

impl WindowSurface {
    pub fn new(instance: &Instance, window: &winit::window::Window) -> Self {
        let surface = unsafe { ash_window::create_surface(&instance.entry, &instance.instance, window, None).expect("Failed to create VkSurface") };
        WindowSurface {
            surface
        }
    }

    pub fn pixel_size(&self, instance: &Instance, physical_device: &PhysicalDevice, window: &winit::window::Window) -> ash::vk::Extent2D {
        let caps = unsafe { instance.ext_surface.get_physical_device_surface_capabilities(physical_device.physical_device, self.surface).expect("Failed to query surface capabilities") };
        if caps.current_extent.width == u32::max_value() {
            let window_physical_size = window.inner_size();
            ash::vk::Extent2D {
                width: window_physical_size.width,
                height: window_physical_size.height
            }
        } else {
            caps.current_extent
        }
    }

    pub fn release_resources(&self, instance: &Instance) {
        unsafe {
            instance.ext_surface.destroy_surface(self.surface, None);
        }
    }
}

pub struct PhysicalDevice {
    pub physical_device: ash::vk::PhysicalDevice,
    pub properties: ash::vk::PhysicalDeviceProperties,
    pub memory_properties: ash::vk::PhysicalDeviceMemoryProperties,
    pub features: ash::vk::PhysicalDeviceFeatures,
    pub queue_family_properties: Vec<ash::vk::QueueFamilyProperties>,
    pub gfx_compute_present_queue_family_index: u32 // for now just assumes that a combined graphics+compute+present will be available
}

impl PhysicalDevice {
    pub fn new(instance: &Instance, surface: &ash::vk::SurfaceKHR) -> Self {
        let mut result: Option<PhysicalDevice> = None;
        let physical_devices = unsafe { instance.instance.enumerate_physical_devices().expect("Failed to enumerate physical devices") };
        if physical_devices.len() == 0 {
            println!("No physical devices");
        }

        // For now just picks the first physical device, unless overriden via the env.var.
        let requested_index = match std::env::var("RVK_PHYSICAL_DEVICE_INDEX") {
            Ok(s) => match s.parse::<i32>() { Ok(i) => i, _ => -1 },
            _ => -1
        };

        for (physical_device_index, &physical_device) in physical_devices.iter().enumerate() {
            let mut pdev = PhysicalDevice {
                physical_device: physical_device,
                properties: unsafe { instance.instance.get_physical_device_properties(physical_device) },
                memory_properties: unsafe { instance.instance.get_physical_device_memory_properties(physical_device) },
                features: unsafe { instance.instance.get_physical_device_features(physical_device) },
                queue_family_properties: unsafe { instance.instance.get_physical_device_queue_family_properties(physical_device) },
                gfx_compute_present_queue_family_index: 0
            };

            println!("Physical device {}: {:?} {}.{}.{} api {}.{}.{} vendor id 0x{:X} device id 0x{:X} device type {}",
                     physical_device_index,
                     unsafe { std::ffi::CStr::from_ptr(pdev.properties.device_name.as_ptr()) },
                     ash::vk::version_major(pdev.properties.driver_version),
                     ash::vk::version_minor(pdev.properties.driver_version),
                     ash::vk::version_patch(pdev.properties.driver_version),
                     ash::vk::version_major(pdev.properties.api_version),
                     ash::vk::version_minor(pdev.properties.api_version),
                     ash::vk::version_patch(pdev.properties.api_version),
                     pdev.properties.vendor_id,
                     pdev.properties.device_id,
                     pdev.properties.device_type.as_raw());

            if result.is_none() && (requested_index < 0 || requested_index == physical_device_index as i32) {
                println!("  Using physical device {}", physical_device_index);
                let mut chosen_queue_family_index: Option<u32> = None;
                for (queue_family_index, &queue_family) in pdev.queue_family_properties.iter().enumerate() {
                    println!("Queue family {}: flags 0x{:X} count {}",
                             queue_family_index,
                             queue_family.queue_flags.as_raw(),
                             queue_family.queue_count);
                    if chosen_queue_family_index.is_none()
                        && queue_family.queue_flags.contains(ash::vk::QueueFlags::GRAPHICS | ash::vk::QueueFlags::COMPUTE)
                        && queue_family.queue_count > 0
                        && unsafe { instance.ext_surface.get_physical_device_surface_support(physical_device, queue_family_index as u32, *surface).unwrap() }
                    {
                        chosen_queue_family_index = Some(queue_family_index as u32);
                    }
                }
                pdev.gfx_compute_present_queue_family_index = chosen_queue_family_index.expect("Could not find graphics+compute queue");
                println!("  Using queue family {}", pdev.gfx_compute_present_queue_family_index);
                result = Some(pdev);
            }
        }

        result.expect("No physical device chosen")
    }
}

pub struct Device {
    pub device: ash::Device,
    pub queue: ash::vk::Queue,
    pub ext_swapchain: ash::extensions::khr::Swapchain
}

impl Device {
    pub fn new(instance: &Instance, physical_device: &PhysicalDevice) -> Self {
        let queue_priorities = [ 1.0f32 ];
        let queue_create_info = ash::vk::DeviceQueueCreateInfo {
            queue_family_index: physical_device.gfx_compute_present_queue_family_index,
            p_queue_priorities: queue_priorities.as_ptr(),
            queue_count: queue_priorities.len() as u32,
            ..Default::default()
        };

        let enabled_features = ash::vk::PhysicalDeviceFeatures {
            wide_lines: physical_device.features.wide_lines,
            large_points: physical_device.features.large_points,
            texture_compression_etc2: physical_device.features.texture_compression_etc2,
            texture_compression_astc_ldr: physical_device.features.texture_compression_astc_ldr,
            texture_compression_bc: physical_device.features.texture_compression_bc,
            ..Default::default()
        };

        let mut layers: smallvec::SmallVec<[*const std::os::raw::c_char; 4]> = smallvec::smallvec![];
        if instance.validation_enabled {
            layers.push(VALIDATION_LAYER_NAME.as_ptr());
        }

        let extension_properties = unsafe { instance.instance.enumerate_device_extension_properties(physical_device.physical_device).expect("Failed to enumerate device extension properties") };
        let available_extension_names = extension_properties.iter().map(|s| unsafe { std::ffi::CStr::from_ptr(s.extension_name.as_ptr()) }).collect::<Vec<_>>();
        println!("Available device extensions: {:?}", available_extension_names);

        let mut extensions: smallvec::SmallVec<[*const std::os::raw::c_char; 8]> = smallvec::smallvec![];
        extensions.push(SWAPCHAIN_EXT_NAME.as_ptr());

        println!("Enabling device extensions: {:?}", extensions.iter().map(|s| unsafe { std::ffi::CStr::from_ptr(*s) }).collect::<smallvec::SmallVec<[_; 8]>>());

        let device_create_info = ash::vk::DeviceCreateInfo {
            queue_create_info_count: 1,
            p_queue_create_infos: &queue_create_info,
            enabled_layer_count: layers.len() as u32,
            pp_enabled_layer_names: layers.as_ptr(),
            enabled_extension_count: extensions.len() as u32,
            pp_enabled_extension_names: extensions.as_ptr(),
            p_enabled_features: &enabled_features,
            ..Default::default()
        };

        let device = unsafe { instance.instance.create_device(physical_device.physical_device, &device_create_info, None).expect("Failed to create Vulkan device") };
        let queue = unsafe { device.get_device_queue(physical_device.gfx_compute_present_queue_family_index, 0) };
        let ext_swapchain = ash::extensions::khr::Swapchain::new(&instance.instance, &device);

        Device {
            device,
            queue,
            ext_swapchain
        }
    }

    pub fn release_resources(&self) {
        unsafe {
            self.device.destroy_device(None);
        }
    }

    pub fn wait_idle(&self) {
        unsafe { self.device.device_wait_idle().expect("DeviceWaitIdle failed") };
    }

    pub fn wait_fence(&self, fence: &ash::vk::Fence) {
        let fence_list = [*fence];
        unsafe { self.device.wait_for_fences(&fence_list, true, u64::max_value()).expect("Fence wait failed") };
    }

    pub fn wait_reset_fence(&self, fence: &ash::vk::Fence) {
        let fence_list = [*fence];
        unsafe {
            self.device.wait_for_fences(&fence_list, true, u64::max_value()).expect("Fence wait failed");
            self.device.reset_fences(&fence_list).expect("Fence reset failed");
        }
    }
}

bitflags! {
    pub struct SwapchainFlags: u32 {
        const ALLOW_READBACK = 0x01;
        const NO_VSYNC = 0x02;
        const PREMUL_ALPHA = 0x04;
        const NON_PREMUL_ALPHA = 0x08;
        const SRGB = 0x10;
    }
}

pub struct Swapchain {
    pub swapchain: Option<ash::vk::SwapchainKHR>,
    pub format: ash::vk::Format,
    pub flags: SwapchainFlags
}

fn is_srgb_format(format: ash::vk::Format) -> bool {
    match format {
        ash::vk::Format::R8_SRGB
            | ash::vk::Format::R8G8_SRGB
            | ash::vk::Format::R8G8B8_SRGB
            | ash::vk::Format::B8G8R8_SRGB
            | ash::vk::Format::R8G8B8A8_SRGB
            | ash::vk::Format::B8G8R8A8_SRGB
            | ash::vk::Format::A8B8G8R8_SRGB_PACK32
            => true,
        _ => false
    }
}

impl Swapchain {
    pub fn new() -> Self {
        Swapchain {
            swapchain: None,
            format: ash::vk::Format::UNDEFINED,
            flags: SwapchainFlags::empty()
        }
    }

    pub fn new_with_flags(flags: SwapchainFlags) -> Self {
        Swapchain {
            swapchain: None,
            format: ash::vk::Format::UNDEFINED,
            flags
        }
    }

    pub fn release_resources(&self, device: &Device) {
        if self.swapchain.is_some() {
            unsafe { device.ext_swapchain.destroy_swapchain(self.swapchain.unwrap(), None) };
        }
    }

    pub fn resize(&mut self, instance: &Instance, physical_device: &PhysicalDevice, device: &Device, surface: &WindowSurface, window: &winit::window::Window) {
        let image_extent = surface.pixel_size(instance, physical_device, window);
        if image_extent.width == 0 || image_extent.height == 0 {
            return;
        }

        device.wait_idle();

        let caps = unsafe { instance.ext_surface.get_physical_device_surface_capabilities(physical_device.physical_device, surface.surface).unwrap() };
        let buffer_count = std::cmp::max(std::cmp::min(3, caps.max_image_count), caps.min_image_count);
        let pre_transform = if caps.supported_transforms.contains(ash::vk::SurfaceTransformFlagsKHR::IDENTITY) {
            ash::vk::SurfaceTransformFlagsKHR::IDENTITY
        } else {
            caps.current_transform
        };
        let composite_alpha = if self.flags.contains(SwapchainFlags::PREMUL_ALPHA) && caps.supported_composite_alpha.contains(ash::vk::CompositeAlphaFlagsKHR::PRE_MULTIPLIED) {
            ash::vk::CompositeAlphaFlagsKHR::PRE_MULTIPLIED
        } else if self.flags.contains(SwapchainFlags::NON_PREMUL_ALPHA) && caps.supported_composite_alpha.contains(ash::vk::CompositeAlphaFlagsKHR::POST_MULTIPLIED) {
            ash::vk::CompositeAlphaFlagsKHR::POST_MULTIPLIED
        } else if caps.supported_composite_alpha.contains(ash::vk::CompositeAlphaFlagsKHR::INHERIT) {
            ash::vk::CompositeAlphaFlagsKHR::INHERIT
        } else {
            ash::vk::CompositeAlphaFlagsKHR::OPAQUE
        };
        let image_usage = if self.flags.contains(SwapchainFlags::ALLOW_READBACK) && caps.supported_usage_flags.contains(ash::vk::ImageUsageFlags::TRANSFER_SRC) {
            ash::vk::ImageUsageFlags::COLOR_ATTACHMENT | ash::vk::ImageUsageFlags::TRANSFER_SRC
        } else {
            ash::vk::ImageUsageFlags::COLOR_ATTACHMENT
        };
        let present_mode = if self.flags.contains(SwapchainFlags::NO_VSYNC) {
            let supported_present_modes = unsafe { instance.ext_surface.get_physical_device_surface_present_modes(physical_device.physical_device, surface.surface).unwrap() };
            if supported_present_modes.contains(&ash::vk::PresentModeKHR::MAILBOX) {
                ash::vk::PresentModeKHR::MAILBOX
            } else {
                ash::vk::PresentModeKHR::IMMEDIATE
            }
        } else {
            ash::vk::PresentModeKHR::FIFO
        };
        let supported_formats = unsafe { instance.ext_surface.get_physical_device_surface_formats(physical_device.physical_device, surface.surface).unwrap() };
        let mut chosen_format: Option<ash::vk::SurfaceFormatKHR> = None;
        let wants_srgb = self.flags.contains(SwapchainFlags::SRGB);
        for format in &supported_formats {
            if format.format != ash::vk::Format::UNDEFINED && wants_srgb == is_srgb_format(format.format) {
                chosen_format = Some(*format);
                break;
            }
        }

        let swapchain_create_info = ash::vk::SwapchainCreateInfoKHR {
            surface: surface.surface,
            min_image_count: buffer_count,
            image_color_space: chosen_format.unwrap().color_space,
            image_format: chosen_format.unwrap().format,
            image_extent,
            image_usage,
            image_sharing_mode: ash::vk::SharingMode::EXCLUSIVE,
            pre_transform,
            composite_alpha,
            present_mode,
            clipped: ash::vk::TRUE,
            old_swapchain: if self.swapchain.is_some() { self.swapchain.unwrap() } else { ash::vk::SwapchainKHR::null() },
            image_array_layers: 1,
            ..Default::default()
        };

        let swapchain = unsafe { device.ext_swapchain.create_swapchain(&swapchain_create_info, None).expect("Failed to create swapchain") };
        if self.swapchain.is_some() {
            unsafe { device.ext_swapchain.destroy_swapchain(self.swapchain.unwrap(), None) };
        }
        self.swapchain = Some(swapchain);
        self.format = chosen_format.unwrap().format;
    }
}

pub struct SwapchainImages {
    pub images: Vec<ash::vk::Image>,
    pub views: smallvec::SmallVec<[ash::vk::ImageView; 8]>,
}

impl SwapchainImages {
    pub fn new(device: &Device, swapchain: &Swapchain) -> Self {
        let images = unsafe { device.ext_swapchain.get_swapchain_images(swapchain.swapchain.unwrap()).expect("Failed to get swapchain images") };
        let mut views: smallvec::SmallVec<[ash::vk::ImageView; 8]> = smallvec::smallvec![];
        for &image in images.iter() {
            let view_create_info = ash::vk::ImageViewCreateInfo {
                image,
                view_type: ash::vk::ImageViewType::TYPE_2D,
                format: swapchain.format,
                components: ash::vk::ComponentMapping {
                    r: ash::vk::ComponentSwizzle::IDENTITY,
                    g: ash::vk::ComponentSwizzle::IDENTITY,
                    b: ash::vk::ComponentSwizzle::IDENTITY,
                    a: ash::vk::ComponentSwizzle::IDENTITY
                },
                subresource_range: ash::vk::ImageSubresourceRange {
                    aspect_mask: ash::vk::ImageAspectFlags::COLOR,
                    level_count: 1,
                    layer_count: 1,
                    ..Default::default()
                },
                ..Default::default()
            };
            let view = unsafe { device.device.create_image_view(&view_create_info, None).expect("Failed to create swapchain image view") };
            views.push(view);
        }
        SwapchainImages {
            images,
            views
        }
    }

    pub fn release_resources(&self, device: &Device) {
        for &view in self.views.iter() {
            unsafe { device.device.destroy_image_view(view, None) };
        }
    }
}

pub struct CommandPool {
    pub pools: smallvec::SmallVec<[ash::vk::CommandPool; FRAMES_IN_FLIGHT as usize]>,
}

impl CommandPool {
    pub fn new(physical_device: &PhysicalDevice, device: &Device) -> Self {
        let mut pools: smallvec::SmallVec<[ash::vk::CommandPool; FRAMES_IN_FLIGHT as usize]> = smallvec::smallvec![];
        let pool_create_info = ash::vk::CommandPoolCreateInfo {
            queue_family_index: physical_device.gfx_compute_present_queue_family_index,
            ..Default::default()
        };
        for _ in 0..FRAMES_IN_FLIGHT {
            let pool = unsafe { device.device.create_command_pool(&pool_create_info, None).expect("Failed to create command pool") };
            pools.push(pool);
        }
        CommandPool {
            pools
        }
    }

    pub fn release_resources(&self, device: &Device) {
        for &pool in self.pools.iter() {
            unsafe { device.device.destroy_command_pool(pool, None) };
        }
    }

    pub fn reset(&self, device: &Device, slot: u32) {
        unsafe { device.device.reset_command_pool(self.pools[slot as usize], ash::vk::CommandPoolResetFlags::empty()).expect("Failed to reset command pool") };
    }
}

pub struct CommandList {
    pub command_buffers: smallvec::SmallVec<[ash::vk::CommandBuffer; FRAMES_IN_FLIGHT as usize]>,
}

impl CommandList {
    pub fn new(device: &Device, cmd_pool: &CommandPool) -> Self {
        let mut command_buffers: smallvec::SmallVec<[ash::vk::CommandBuffer; FRAMES_IN_FLIGHT as usize]> = smallvec::smallvec![];
        for slot in 0..FRAMES_IN_FLIGHT {
            let allocate_info = ash::vk::CommandBufferAllocateInfo {
                command_pool: cmd_pool.pools[slot as usize],
                level: ash::vk::CommandBufferLevel::PRIMARY,
                command_buffer_count: 1,
                ..Default::default()
            };
            let buffers = unsafe { device.device.allocate_command_buffers(&allocate_info).expect("Failed to allocate command buffer") };
            command_buffers.push(buffers[0]);
        }
        CommandList {
            command_buffers
        }
    }

    pub fn begin(&self, device: &Device, slot: u32) {
        let begin_info = ash::vk::CommandBufferBeginInfo {
            ..Default::default()
        };
        unsafe { device.device.begin_command_buffer(self.command_buffers[slot as usize], &begin_info).expect("Failed to begin command buffer") };
    }

    pub fn end(&self, device: &Device, slot: u32) {
        unsafe { device.device.end_command_buffer(self.command_buffers[slot as usize]).expect("Failed to end command buffer") };
    }
}

pub struct SwapchainFrameSyncObjects {
    pub image_fence: ash::vk::Fence,
    pub cmd_fence: ash::vk::Fence,
    pub image_sem: ash::vk::Semaphore,
    pub present_sem: ash::vk::Semaphore,
    pub image_acquired: bool,
    pub image_fence_waitable: bool,
    pub cmd_fence_waitable: bool,
    pub image_sem_waitable: bool
}

fn make_swapchain_frame_sync_objects(device: &Device) -> smallvec::SmallVec<[SwapchainFrameSyncObjects; FRAMES_IN_FLIGHT as usize]> {
    let mut sync_objects: smallvec::SmallVec<[SwapchainFrameSyncObjects; FRAMES_IN_FLIGHT as usize]> = smallvec::smallvec![];
    for _ in 0..FRAMES_IN_FLIGHT {
        let fence_create_info = ash::vk::FenceCreateInfo {
            flags: ash::vk::FenceCreateFlags::SIGNALED,
            ..Default::default()
        };
        let sem_create_info = ash::vk::SemaphoreCreateInfo {
            ..Default::default()
        };
        sync_objects.push(SwapchainFrameSyncObjects {
            image_fence: unsafe { device.device.create_fence(&fence_create_info, None).expect("Failed to create swapchain image fence") },
            cmd_fence: unsafe { device.device.create_fence(&fence_create_info, None).expect("Failed to create swapchain command fence") },
            image_sem: unsafe { device.device.create_semaphore(&sem_create_info, None).expect("Failed to create swapchain image semaphore") },
            present_sem: unsafe { device.device.create_semaphore(&sem_create_info, None).expect("Failed to create present semaphore") },
            image_acquired: false,
            image_fence_waitable: true,
            cmd_fence_waitable: true,
            image_sem_waitable: false
        });
    }
    sync_objects
}

pub struct SwapchainFrameState {
    pub sync_objects: smallvec::SmallVec<[SwapchainFrameSyncObjects; FRAMES_IN_FLIGHT as usize]>,
    current_frame_slot: u32,
    current_image_index: u32
}

impl SwapchainFrameState {
    pub fn new(device: &Device) -> Self {
        SwapchainFrameState {
            sync_objects: make_swapchain_frame_sync_objects(device),
            current_frame_slot: 0,
            current_image_index: 0
        }
    }

    pub fn release_resources(&self, device: &Device) {
        for sync in self.sync_objects.iter() {
            if sync.image_fence_waitable {
                device.wait_fence(&sync.image_fence);
            }
            if sync.cmd_fence_waitable {
                device.wait_fence(&sync.cmd_fence);
            }
            unsafe {
                device.device.destroy_fence(sync.image_fence, None);
                device.device.destroy_fence(sync.cmd_fence, None);
                device.device.destroy_semaphore(sync.image_sem, None);
                device.device.destroy_semaphore(sync.present_sem, None);
            }
        }
    }

    pub fn begin_frame(&mut self, device: &Device, swapchain: &Swapchain, cmd_pool: &CommandPool, cmd_list: &CommandList) -> Result<u32, ash::vk::Result> {
        let s = &mut self.sync_objects[self.current_frame_slot as usize];
        if !s.image_acquired {
            if s.image_fence_waitable {
                device.wait_reset_fence(&s.image_fence);
                s.image_fence_waitable = false;
            }
            let index_and_suboptimal = unsafe { device.ext_swapchain.acquire_next_image(swapchain.swapchain.unwrap(), u64::max_value(), s.image_sem, s.image_fence) };
            match index_and_suboptimal {
                Err(r) => return Err(r),
                _ => ()
            };
            self.current_image_index = index_and_suboptimal.unwrap().0;
            s.image_acquired = true;
            s.image_fence_waitable = true;
            s.image_sem_waitable = true;
        }
        //println!("{} {}", self.current_frame_slot, self.current_image_index);

        if s.cmd_fence_waitable {
            device.wait_reset_fence(&s.cmd_fence);
            s.cmd_fence_waitable = false;
        }

        cmd_pool.reset(device, self.current_frame_slot);
        cmd_list.begin(device, self.current_frame_slot);

        return Ok(self.current_frame_slot);
    }

    pub fn end_frame(&mut self, device: &Device, swapchain: &Swapchain, swapchain_images: &SwapchainImages, cmd_list: &CommandList) {
        self.transition(device, swapchain_images, cmd_list,
                        ash::vk::AccessFlags::empty(),
                        ash::vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                        ash::vk::ImageLayout::UNDEFINED,
                        ash::vk::ImageLayout::PRESENT_SRC_KHR,
                        ash::vk::PipelineStageFlags::TOP_OF_PIPE,
                        ash::vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT);

        cmd_list.end(&device, self.current_frame_slot);

        let s = &mut self.sync_objects[self.current_frame_slot as usize];
        let wait_dst_stage_mask = ash::vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT;
        let submit_info = ash::vk::SubmitInfo {
            wait_semaphore_count: if s.image_sem_waitable { 1 } else  { 0 },
            p_wait_semaphores: &s.image_sem,
            p_wait_dst_stage_mask: &wait_dst_stage_mask,
            command_buffer_count: 1,
            p_command_buffers: &cmd_list.command_buffers[self.current_frame_slot as usize],
            signal_semaphore_count: 1,
            p_signal_semaphores: &s.present_sem,
            ..Default::default()
        };
        let submits = [submit_info];
        unsafe { device.device.queue_submit(device.queue, &submits, s.cmd_fence).expect("Failed to submit to queue") };
        s.cmd_fence_waitable = true;
        s.image_sem_waitable = false;

        let present_info = ash::vk::PresentInfoKHR {
            wait_semaphore_count: 1,
            p_wait_semaphores: &s.present_sem,
            swapchain_count: 1,
            p_swapchains: &swapchain.swapchain.unwrap(),
            p_image_indices: &self.current_image_index,
            p_results: std::ptr::null_mut(),
            ..Default::default()
        };
        unsafe { device.ext_swapchain.queue_present(device.queue, &present_info).expect("Failed to enqueue present") };
        s.image_acquired = false;

        self.current_frame_slot = (self.current_frame_slot + 1) % FRAMES_IN_FLIGHT;
    }

    fn transition(&self, device: &Device, swapchain_images: &SwapchainImages, cmd_list: &CommandList,
                  src_access_mask: ash::vk::AccessFlags,
                  dst_access_mask: ash::vk::AccessFlags,
                  old_layout: ash::vk::ImageLayout,
                  new_layout: ash::vk::ImageLayout,
                  src_stage_mask: ash::vk::PipelineStageFlags,
                  dst_stage_mask: ash::vk::PipelineStageFlags)
    {
        let image_barriers = [ash::vk::ImageMemoryBarrier {
            src_access_mask,
            dst_access_mask,
            old_layout,
            new_layout,
            image: swapchain_images.images[self.current_image_index as usize],
            subresource_range: ash::vk::ImageSubresourceRange {
                aspect_mask: ash::vk::ImageAspectFlags::COLOR,
                level_count: 1,
                layer_count: 1,
                ..Default::default()
            },
            ..Default::default()
        }];
        unsafe {
            device.device.cmd_pipeline_barrier(cmd_list.command_buffers[self.current_frame_slot as usize],
                                               src_stage_mask,
                                               dst_stage_mask,
                                               ash::vk::DependencyFlags::empty(),
                                               &[],
                                               &[],
                                               &image_barriers);
        }
    }
}

pub struct Scene {
}

impl Scene {
    pub fn new() -> Self {
        Scene {
        }
    }

    pub fn sync(&self) -> bool {
        true
    }

    pub fn render(&self) {
    }
}

fn main() {
    let event_loop = winit::event_loop::EventLoop::new();

    let window = winit::window::WindowBuilder::new()
        .with_inner_size(winit::dpi::LogicalSize::new(1280, 720))
        .build(&event_loop)
        .expect("Failed to create window");

    let instance = Instance::new(&window, ENABLE_VALIDATION);
    let surface = WindowSurface::new(&instance, &window);
    let physical_device = PhysicalDevice::new(&instance, &surface.surface);
    let device = Device::new(&instance, &physical_device);
    let cmd_pool = CommandPool::new(&physical_device, &device);
    let cmd_list = CommandList::new(&device, &cmd_pool);
    let mut swapchain = Swapchain::new();
    swapchain.resize(&instance, &physical_device, &device, &surface, &window);
    let mut swapchain_images = SwapchainImages::new(&device, &swapchain);
    let mut frame_state = SwapchainFrameState::new(&device);
    let mut running = true;
    let mut surface_size: ash::vk::Extent2D = Default::default();
    let mut frame_time = std::time::Instant::now();

    let scene = Scene::new();

    event_loop.run(move |event, _, control_flow| {
        *control_flow = winit::event_loop::ControlFlow::Wait;
        match event {
            winit::event::Event::WindowEvent { window_id, event } if window_id == window.id() => match event {
                winit::event::WindowEvent::CloseRequested => {
                    *control_flow = winit::event_loop::ControlFlow::Exit;
                    running = false;
                    frame_state.release_resources(&device);
                    swapchain_images.release_resources(&device);
                    swapchain.release_resources(&device);
                    cmd_pool.release_resources(&device);
                    device.release_resources();
                    surface.release_resources(&instance);
                    instance.release_resources();
                },
                _ => (),
            },
            winit::event::Event::MainEventsCleared => if scene.sync() { window.request_redraw(); },
            winit::event::Event::RedrawRequested(window_id) if window_id == window.id() && running => {
                let current_pixel_size = surface.pixel_size(&instance, &physical_device, &window);
                if current_pixel_size.width != 0 && current_pixel_size.height != 0 {
                    if surface_size != current_pixel_size {
                        surface_size = current_pixel_size;
                        swapchain_images.release_resources(&device);
                        frame_state.release_resources(&device);
                        swapchain.resize(&instance, &physical_device, &device, &surface, &window);
                        swapchain_images = SwapchainImages::new(&device, &swapchain);
                        frame_state = SwapchainFrameState::new(&device);
                        println!("Resized swapchain to {:?}", surface_size);
                    }
                    match frame_state.begin_frame(&device, &swapchain, &cmd_pool, &cmd_list) {
                        Ok(current_frame_slot) => {
                            println!("Render, elapsed since last {:?}", frame_time.elapsed());
                            frame_time = std::time::Instant::now();
                            scene.render();
                            frame_state.end_frame(&device, &swapchain, &swapchain_images, &cmd_list);
                        },
                        Err(r) => println!("{:?}", r)
                    }
                }
            },
            _ => ()
        };
    });
}
