use ash::version::{DeviceV1_0, EntryV1_0, InstanceV1_0};
use bitflags::bitflags;
use const_cstr::const_cstr;
use std::rc::Rc;

pub const ENABLE_VALIDATION: bool = true;

pub const FRAMES_IN_FLIGHT: u32 = 2;

pub struct Instance {
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    pub validation_enabled: bool,
    ext_debug_utils: ash::extensions::ext::DebugUtils,
    debug_messenger: ash::vk::DebugUtilsMessengerEXT,
    ext_surface: ash::extensions::khr::Surface,
    valid: bool,
}

unsafe extern "system" fn debug_callback(
    _message_severity: ash::vk::DebugUtilsMessageSeverityFlagsEXT,
    _message_type: ash::vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const ash::vk::DebugUtilsMessengerCallbackDataEXT,
    _p_user_data: *mut std::os::raw::c_void,
) -> ash::vk::Bool32 {
    let msg = std::ffi::CStr::from_ptr((*p_callback_data).p_message);
    println!("[Vulkan] {:?}", msg);
    ash::vk::FALSE
}

const_cstr! {
    VALIDATION_LAYER_NAME = "VK_LAYER_KHRONOS_validation";
    SWAPCHAIN_EXT_NAME = "VK_KHR_swapchain";
    PHYS_DEV_PROP2_EXT_NAME = "VK_KHR_get_physical_device_properties2";
    DEFAULT_SHADER_ENTRY_POINT = "main";
}

impl Instance {
    pub fn new(window: &winit::window::Window, enable_validation: bool) -> Self {
        let entry = ash::Entry::new().expect("Failed to initialize Vulkan loader");

        let layer_properties = entry
            .enumerate_instance_layer_properties()
            .expect("Failed to enumerate instance layer properties");
        let available_layer_names = layer_properties
            .iter()
            .map(|s| unsafe { std::ffi::CStr::from_ptr(s.layer_name.as_ptr()) })
            .collect::<Vec<_>>();
        println!("Available layers: {:?}", available_layer_names);

        let mut layers: smallvec::SmallVec<[*const std::os::raw::c_char; 4]> =
            smallvec::smallvec![];
        if enable_validation {
            layers.push(VALIDATION_LAYER_NAME.as_ptr());
        }

        let extension_properties = entry
            .enumerate_instance_extension_properties()
            .expect("Failed to enumerate instanance extension properties");
        let available_extension_names = extension_properties
            .iter()
            .map(|s| unsafe { std::ffi::CStr::from_ptr(s.extension_name.as_ptr()) })
            .collect::<Vec<_>>();
        println!("Available extensions: {:?}", available_extension_names);

        println!(
            "Enabling layers: {:?}",
            layers
                .iter()
                .map(|s| unsafe { std::ffi::CStr::from_ptr(*s) })
                .collect::<smallvec::SmallVec<[_; 4]>>()
        );

        let surface_extensions = ash_window::enumerate_required_extensions(window).unwrap();
        let mut extensions = surface_extensions
            .iter()
            .map(|s| s.as_ptr())
            .collect::<smallvec::SmallVec<[_; 8]>>();
        extensions.push(ash::extensions::ext::DebugUtils::name().as_ptr());
        extensions.push(PHYS_DEV_PROP2_EXT_NAME.as_ptr());

        println!(
            "Enabling extensions: {:?}",
            extensions
                .iter()
                .map(|s| unsafe { std::ffi::CStr::from_ptr(*s) })
                .collect::<smallvec::SmallVec<[_; 8]>>()
        );

        let instance_create_info = ash::vk::InstanceCreateInfo {
            enabled_layer_count: layers.len() as u32,
            pp_enabled_layer_names: layers.as_ptr(),
            enabled_extension_count: extensions.len() as u32,
            pp_enabled_extension_names: extensions.as_ptr(),
            ..Default::default()
        };
        let instance = unsafe {
            entry
                .create_instance(&instance_create_info, None)
                .expect("Failed to create Vulkan instance")
        };

        let ext_debug_utils = ash::extensions::ext::DebugUtils::new(&entry, &instance);
        let debug_create_info = ash::vk::DebugUtilsMessengerCreateInfoEXT {
            message_severity: ash::vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                | ash::vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
            message_type: ash::vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                | ash::vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE
                | ash::vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION,
            pfn_user_callback: Some(debug_callback),
            ..Default::default()
        };
        let debug_messenger = unsafe {
            ext_debug_utils
                .create_debug_utils_messenger(&debug_create_info, None)
                .expect("Failed to create debug utils messenger")
        };

        let ext_surface = ash::extensions::khr::Surface::new(&entry, &instance);

        Instance {
            entry,
            instance,
            validation_enabled: enable_validation,
            ext_debug_utils,
            debug_messenger,
            ext_surface,
            valid: true,
        }
    }

    pub fn release_resources(&mut self) {
        if self.valid {
            unsafe {
                self.ext_debug_utils
                    .destroy_debug_utils_messenger(self.debug_messenger, None);
                self.instance.destroy_instance(None);
            }
            self.valid = false;
        }
    }
}

impl Drop for Instance {
    fn drop(&mut self) {
        self.release_resources();
    }
}

pub struct WindowSurface {
    pub surface: ash::vk::SurfaceKHR,
    instance: Option<Rc<Instance>>,
}

impl WindowSurface {
    pub fn new(instance: &Rc<Instance>, window: &winit::window::Window) -> Self {
        let surface = unsafe {
            ash_window::create_surface(&instance.entry, &instance.instance, window, None)
                .expect("Failed to create VkSurface")
        };
        WindowSurface {
            surface,
            instance: Some(Rc::clone(instance)),
        }
    }

    pub fn pixel_size(
        &self,
        instance: &Instance,
        physical_device: &PhysicalDevice,
        window: &winit::window::Window,
    ) -> ash::vk::Extent2D {
        let caps = unsafe {
            instance
                .ext_surface
                .get_physical_device_surface_capabilities(
                    physical_device.physical_device,
                    self.surface,
                )
                .expect("Failed to query surface capabilities")
        };
        if caps.current_extent.width == u32::max_value() {
            let window_physical_size = window.inner_size();
            ash::vk::Extent2D {
                width: window_physical_size.width,
                height: window_physical_size.height,
            }
        } else {
            caps.current_extent
        }
    }

    pub fn release_resources(&mut self) {
        if self.instance.is_some() {
            unsafe {
                self.instance
                    .as_ref()
                    .unwrap()
                    .ext_surface
                    .destroy_surface(self.surface, None);
            }
            self.instance = None;
        }
        self.surface = ash::vk::SurfaceKHR::null();
    }
}

impl Drop for WindowSurface {
    fn drop(&mut self) {
        self.release_resources();
    }
}

fn aligned(v: u64, byte_alignment: u64) -> u64 {
    (v + byte_alignment - 1) & !(byte_alignment - 1)
}

pub struct PhysicalDevice {
    pub physical_device: ash::vk::PhysicalDevice,
    pub properties: ash::vk::PhysicalDeviceProperties,
    pub memory_properties: ash::vk::PhysicalDeviceMemoryProperties,
    pub features: ash::vk::PhysicalDeviceFeatures,
    pub queue_family_properties: Vec<ash::vk::QueueFamilyProperties>,
    pub gfx_compute_present_queue_family_index: u32, // for now just assumes that a combined graphics+compute+present will be available
    pub optimal_depth_stencil_format: ash::vk::Format,
    pub ubuf_offset_alignment: u64,
    pub staging_buf_offset_alignment: u64,
}

fn optimal_depth_stencil_format(
    instance: &Instance,
    physical_device: &PhysicalDevice,
) -> ash::vk::Format {
    let candidates = [
        ash::vk::Format::D24_UNORM_S8_UINT,
        ash::vk::Format::D32_SFLOAT_S8_UINT,
        ash::vk::Format::D16_UNORM_S8_UINT,
    ];
    for &format in candidates.iter() {
        let format_properties = unsafe {
            instance
                .instance
                .get_physical_device_format_properties(physical_device.physical_device, format)
        };
        if format_properties
            .optimal_tiling_features
            .contains(ash::vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT)
        {
            return format;
        }
    }
    println!("Failed to find an optimal depth-stencil format");
    return ash::vk::Format::D24_UNORM_S8_UINT;
}

impl PhysicalDevice {
    pub fn new(instance: &Instance, surface: &ash::vk::SurfaceKHR) -> Self {
        let mut result: Option<PhysicalDevice> = None;
        let physical_devices = unsafe {
            instance
                .instance
                .enumerate_physical_devices()
                .expect("Failed to enumerate physical devices")
        };
        if physical_devices.len() == 0 {
            println!("No physical devices");
        }

        // For now just picks the first physical device, unless overriden via the env.var.
        let requested_index = match std::env::var("RVK_PHYSICAL_DEVICE_INDEX") {
            Ok(s) => match s.parse::<i32>() {
                Ok(i) => i,
                _ => -1,
            },
            _ => -1,
        };

        for (physical_device_index, &physical_device) in physical_devices.iter().enumerate() {
            let mut pdev = PhysicalDevice {
                physical_device: physical_device,
                properties: unsafe {
                    instance
                        .instance
                        .get_physical_device_properties(physical_device)
                },
                memory_properties: unsafe {
                    instance
                        .instance
                        .get_physical_device_memory_properties(physical_device)
                },
                features: unsafe {
                    instance
                        .instance
                        .get_physical_device_features(physical_device)
                },
                queue_family_properties: unsafe {
                    instance
                        .instance
                        .get_physical_device_queue_family_properties(physical_device)
                },
                gfx_compute_present_queue_family_index: 0,
                optimal_depth_stencil_format: ash::vk::Format::UNDEFINED,
                ubuf_offset_alignment: 0,
                staging_buf_offset_alignment: 0,
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

            if result.is_none()
                && (requested_index < 0 || requested_index == physical_device_index as i32)
            {
                println!("  Using physical device {}", physical_device_index);
                let mut chosen_queue_family_index: Option<u32> = None;
                for (queue_family_index, &queue_family) in
                    pdev.queue_family_properties.iter().enumerate()
                {
                    println!(
                        "Queue family {}: flags 0x{:X} count {}",
                        queue_family_index,
                        queue_family.queue_flags.as_raw(),
                        queue_family.queue_count
                    );
                    if chosen_queue_family_index.is_none()
                        && queue_family
                            .queue_flags
                            .contains(ash::vk::QueueFlags::GRAPHICS | ash::vk::QueueFlags::COMPUTE)
                        && queue_family.queue_count > 0
                        && unsafe {
                            instance
                                .ext_surface
                                .get_physical_device_surface_support(
                                    physical_device,
                                    queue_family_index as u32,
                                    *surface,
                                )
                                .unwrap()
                        }
                    {
                        chosen_queue_family_index = Some(queue_family_index as u32);
                    }
                }
                pdev.gfx_compute_present_queue_family_index =
                    chosen_queue_family_index.expect("Could not find graphics+compute queue");
                println!(
                    "  Using queue family {}",
                    pdev.gfx_compute_present_queue_family_index
                );
                pdev.optimal_depth_stencil_format = optimal_depth_stencil_format(instance, &pdev);
                pdev.ubuf_offset_alignment =
                    pdev.properties.limits.min_uniform_buffer_offset_alignment;
                pdev.staging_buf_offset_alignment = std::cmp::max(
                    4,
                    pdev.properties.limits.optimal_buffer_copy_offset_alignment,
                );
                result = Some(pdev);
            }
        }

        result.expect("No physical device chosen")
    }
}

pub struct Device {
    pub device: ash::Device,
    pub queue: ash::vk::Queue,
    pub ext_swapchain: ash::extensions::khr::Swapchain,
    valid: bool,
}

impl Device {
    pub fn new(instance: &Instance, physical_device: &PhysicalDevice) -> Self {
        let queue_priorities = [1.0f32];
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

        let mut layers: smallvec::SmallVec<[*const std::os::raw::c_char; 4]> =
            smallvec::smallvec![];
        if instance.validation_enabled {
            layers.push(VALIDATION_LAYER_NAME.as_ptr());
        }

        let extension_properties = unsafe {
            instance
                .instance
                .enumerate_device_extension_properties(physical_device.physical_device)
                .expect("Failed to enumerate device extension properties")
        };
        let available_extension_names = extension_properties
            .iter()
            .map(|s| unsafe { std::ffi::CStr::from_ptr(s.extension_name.as_ptr()) })
            .collect::<Vec<_>>();
        println!(
            "Available device extensions: {:?}",
            available_extension_names
        );

        let mut extensions: smallvec::SmallVec<[*const std::os::raw::c_char; 8]> =
            smallvec::smallvec![];
        extensions.push(SWAPCHAIN_EXT_NAME.as_ptr());

        println!(
            "Enabling device extensions: {:?}",
            extensions
                .iter()
                .map(|s| unsafe { std::ffi::CStr::from_ptr(*s) })
                .collect::<smallvec::SmallVec<[_; 8]>>()
        );

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

        let device = unsafe {
            instance
                .instance
                .create_device(physical_device.physical_device, &device_create_info, None)
                .expect("Failed to create Vulkan device")
        };
        let queue = unsafe {
            device.get_device_queue(physical_device.gfx_compute_present_queue_family_index, 0)
        };
        let ext_swapchain = ash::extensions::khr::Swapchain::new(&instance.instance, &device);

        Device {
            device,
            queue,
            ext_swapchain,
            valid: true,
        }
    }

    pub fn release_resources(&mut self) {
        if self.valid {
            unsafe {
                self.device.destroy_device(None);
            }
            self.valid = false;
        }
    }

    pub fn wait_idle(&self) {
        unsafe {
            self.device
                .device_wait_idle()
                .expect("DeviceWaitIdle failed")
        };
    }

    pub fn wait_fence(&self, fence: &ash::vk::Fence) {
        unsafe {
            self.device
                .wait_for_fences(&[*fence], true, u64::max_value())
                .expect("Fence wait failed")
        };
    }

    pub fn wait_reset_fence(&self, fence: &ash::vk::Fence) {
        unsafe {
            self.device
                .wait_for_fences(&[*fence], true, u64::max_value())
                .expect("Fence wait failed");
            self.device
                .reset_fences(&[*fence])
                .expect("Fence reset failed");
        }
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        self.release_resources();
    }
}

fn memory_type_index_for_transient_image(
    physical_device: &PhysicalDevice,
    mem_req: &ash::vk::MemoryRequirements,
) -> Option<u32> {
    let mut first_device_local_idx: Option<u32> = None;
    if mem_req.memory_type_bits != 0 {
        for i in 0..physical_device.memory_properties.memory_type_count {
            if (mem_req.memory_type_bits & (1 << i)) != 0 {
                let memory_type = physical_device.memory_properties.memory_types[i as usize];
                if memory_type
                    .property_flags
                    .contains(ash::vk::MemoryPropertyFlags::DEVICE_LOCAL)
                {
                    if first_device_local_idx.is_none() {
                        first_device_local_idx = Some(i);
                    }
                    if memory_type
                        .property_flags
                        .contains(ash::vk::MemoryPropertyFlags::LAZILY_ALLOCATED)
                    {
                        return Some(i);
                    }
                }
            }
        }
    }
    return first_device_local_idx;
}

pub fn identity_component_mapping() -> ash::vk::ComponentMapping {
    ash::vk::ComponentMapping {
        r: ash::vk::ComponentSwizzle::IDENTITY,
        g: ash::vk::ComponentSwizzle::IDENTITY,
        b: ash::vk::ComponentSwizzle::IDENTITY,
        a: ash::vk::ComponentSwizzle::IDENTITY,
    }
}

pub fn base_level_subres_range(
    aspect_mask: ash::vk::ImageAspectFlags,
) -> ash::vk::ImageSubresourceRange {
    ash::vk::ImageSubresourceRange {
        aspect_mask,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
    }
}

pub struct DepthStencilBuffer {
    image: ash::vk::Image,
    memory: ash::vk::DeviceMemory,
    view: ash::vk::ImageView,
    device: Option<Rc<Device>>,
}

impl DepthStencilBuffer {
    pub fn new(
        physical_device: &PhysicalDevice,
        device: &Rc<Device>,
        pixel_size: ash::vk::Extent2D,
    ) -> Self {
        let format = physical_device.optimal_depth_stencil_format;
        let image_create_info = ash::vk::ImageCreateInfo {
            image_type: ash::vk::ImageType::TYPE_2D,
            format,
            extent: ash::vk::Extent3D {
                width: pixel_size.width,
                height: pixel_size.height,
                depth: 1,
            },
            mip_levels: 1,
            array_layers: 1,
            samples: ash::vk::SampleCountFlags::TYPE_1,
            tiling: ash::vk::ImageTiling::OPTIMAL,
            usage: ash::vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
                | ash::vk::ImageUsageFlags::TRANSIENT_ATTACHMENT,
            sharing_mode: ash::vk::SharingMode::EXCLUSIVE,
            initial_layout: ash::vk::ImageLayout::UNDEFINED,
            ..Default::default()
        };
        let image = unsafe {
            device
                .device
                .create_image(&image_create_info, None)
                .expect("Failed to create VkImage for depth-stencil buffer")
        };
        let mem_req = unsafe { device.device.get_image_memory_requirements(image) };
        let memory_type_index = memory_type_index_for_transient_image(physical_device, &mem_req)
            .expect("Failed to find suitable memory index for depth-stencil buffer");
        let memory_allocate_info = ash::vk::MemoryAllocateInfo {
            allocation_size: aligned(mem_req.size, mem_req.alignment),
            memory_type_index,
            ..Default::default()
        };
        let memory = unsafe {
            device
                .device
                .allocate_memory(&memory_allocate_info, None)
                .expect("Failed to allocate memory for depth-stencil buffer")
        };
        unsafe {
            device
                .device
                .bind_image_memory(image, memory, 0)
                .expect("Failed to bind memory for depth-stencil buffer")
        };
        let view_create_info = ash::vk::ImageViewCreateInfo {
            image,
            view_type: ash::vk::ImageViewType::TYPE_2D,
            format,
            components: identity_component_mapping(),
            subresource_range: base_level_subres_range(
                ash::vk::ImageAspectFlags::DEPTH | ash::vk::ImageAspectFlags::STENCIL,
            ),
            ..Default::default()
        };
        let view = unsafe {
            device
                .device
                .create_image_view(&view_create_info, None)
                .expect("Failed to create depth-stencil image view")
        };
        DepthStencilBuffer {
            image,
            memory,
            view,
            device: Some(Rc::clone(device)),
        }
    }

    pub fn release_resources(&mut self) {
        if self.device.is_some() {
            let device = self.device.as_ref().unwrap();
            unsafe {
                device.device.destroy_image_view(self.view, None);
                device.device.destroy_image(self.image, None);
                device.device.free_memory(self.memory, None);
            }
            self.device = None;
        }
        self.image = ash::vk::Image::null();
        self.memory = ash::vk::DeviceMemory::null();
        self.view = ash::vk::ImageView::null();
    }
}

impl Drop for DepthStencilBuffer {
    fn drop(&mut self) {
        self.release_resources();
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
    pub swapchain: ash::vk::SwapchainKHR,
    pub format: ash::vk::Format,
    pub flags: SwapchainFlags,
    pub pixel_size: ash::vk::Extent2D,
    device: Option<Rc<Device>>,
}

fn is_srgb_format(format: ash::vk::Format) -> bool {
    match format {
        ash::vk::Format::R8_SRGB
        | ash::vk::Format::R8G8_SRGB
        | ash::vk::Format::R8G8B8_SRGB
        | ash::vk::Format::B8G8R8_SRGB
        | ash::vk::Format::R8G8B8A8_SRGB
        | ash::vk::Format::B8G8R8A8_SRGB
        | ash::vk::Format::A8B8G8R8_SRGB_PACK32 => true,
        _ => false,
    }
}

impl Swapchain {
    pub fn release_resources(&mut self) {
        if self.device.is_some() {
            unsafe {
                self.device
                    .as_ref()
                    .unwrap()
                    .ext_swapchain
                    .destroy_swapchain(self.swapchain, None);
            }
            self.device = None;
        }
        self.swapchain = ash::vk::SwapchainKHR::null();
    }
}

impl Drop for Swapchain {
    fn drop(&mut self) {
        self.release_resources();
    }
}

pub struct SwapchainBuilder {
    swapchain: ash::vk::SwapchainKHR,
    format: ash::vk::Format,
    flags: SwapchainFlags,
}

impl SwapchainBuilder {
    pub fn new() -> Self {
        SwapchainBuilder {
            swapchain: ash::vk::SwapchainKHR::null(),
            format: ash::vk::Format::UNDEFINED,
            flags: SwapchainFlags::empty(),
        }
    }

    pub fn with_flags(&mut self, flags: SwapchainFlags) -> &mut Self {
        self.flags = flags;
        self
    }

    pub fn with_existing(&mut self, swapchain: &Swapchain) -> &mut Self {
        self.swapchain = swapchain.swapchain;
        self.format = swapchain.format;
        self.flags = swapchain.flags;
        self
    }

    pub fn build(
        &self,
        instance: &Instance,
        physical_device: &PhysicalDevice,
        device: &Rc<Device>,
        surface: &WindowSurface,
        pixel_size: ash::vk::Extent2D,
    ) -> Result<Swapchain, ()> {
        let image_extent = pixel_size;
        if image_extent.width == 0 || image_extent.height == 0 {
            return Err(());
        }

        let caps = unsafe {
            instance
                .ext_surface
                .get_physical_device_surface_capabilities(
                    physical_device.physical_device,
                    surface.surface,
                )
                .unwrap()
        };
        let buffer_count =
            std::cmp::max(std::cmp::min(3, caps.max_image_count), caps.min_image_count);
        let pre_transform = if caps
            .supported_transforms
            .contains(ash::vk::SurfaceTransformFlagsKHR::IDENTITY)
        {
            ash::vk::SurfaceTransformFlagsKHR::IDENTITY
        } else {
            caps.current_transform
        };
        let composite_alpha = if self.flags.contains(SwapchainFlags::PREMUL_ALPHA)
            && caps
                .supported_composite_alpha
                .contains(ash::vk::CompositeAlphaFlagsKHR::PRE_MULTIPLIED)
        {
            ash::vk::CompositeAlphaFlagsKHR::PRE_MULTIPLIED
        } else if self.flags.contains(SwapchainFlags::NON_PREMUL_ALPHA)
            && caps
                .supported_composite_alpha
                .contains(ash::vk::CompositeAlphaFlagsKHR::POST_MULTIPLIED)
        {
            ash::vk::CompositeAlphaFlagsKHR::POST_MULTIPLIED
        } else if caps
            .supported_composite_alpha
            .contains(ash::vk::CompositeAlphaFlagsKHR::INHERIT)
        {
            ash::vk::CompositeAlphaFlagsKHR::INHERIT
        } else {
            ash::vk::CompositeAlphaFlagsKHR::OPAQUE
        };
        let image_usage = if self.flags.contains(SwapchainFlags::ALLOW_READBACK)
            && caps
                .supported_usage_flags
                .contains(ash::vk::ImageUsageFlags::TRANSFER_SRC)
        {
            ash::vk::ImageUsageFlags::COLOR_ATTACHMENT | ash::vk::ImageUsageFlags::TRANSFER_SRC
        } else {
            ash::vk::ImageUsageFlags::COLOR_ATTACHMENT
        };
        let supported_present_modes = unsafe {
            instance
                .ext_surface
                .get_physical_device_surface_present_modes(
                    physical_device.physical_device,
                    surface.surface,
                )
                .unwrap()
        };
        let present_mode = if self.flags.contains(SwapchainFlags::NO_VSYNC) {
            if supported_present_modes.contains(&ash::vk::PresentModeKHR::MAILBOX) {
                ash::vk::PresentModeKHR::MAILBOX
            } else {
                ash::vk::PresentModeKHR::IMMEDIATE
            }
        } else {
            ash::vk::PresentModeKHR::FIFO
        };
        let supported_formats = unsafe {
            instance
                .ext_surface
                .get_physical_device_surface_formats(
                    physical_device.physical_device,
                    surface.surface,
                )
                .unwrap()
        };
        let mut chosen_format: Option<ash::vk::SurfaceFormatKHR> = None;
        let wants_srgb = self.flags.contains(SwapchainFlags::SRGB);
        for format in &supported_formats {
            if format.format != ash::vk::Format::UNDEFINED
                && wants_srgb == is_srgb_format(format.format)
            {
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
            old_swapchain: self.swapchain,
            image_array_layers: 1,
            ..Default::default()
        };

        let swapchain = unsafe {
            device
                .ext_swapchain
                .create_swapchain(&swapchain_create_info, None)
                .expect("Failed to create swapchain")
        };
        Ok(Swapchain {
            swapchain,
            format: chosen_format.unwrap().format,
            flags: self.flags,
            pixel_size: image_extent,
            device: Some(Rc::clone(device)),
        })
    }
}

pub struct SwapchainImages {
    pub images: Vec<ash::vk::Image>,
    pub views: smallvec::SmallVec<[ash::vk::ImageView; 8]>,
    device: Option<Rc<Device>>,
}

impl SwapchainImages {
    pub fn new(device: &Rc<Device>, swapchain: &Swapchain) -> Self {
        let images = unsafe {
            device
                .ext_swapchain
                .get_swapchain_images(swapchain.swapchain)
                .expect("Failed to get swapchain images")
        };
        let mut views: smallvec::SmallVec<[ash::vk::ImageView; 8]> = smallvec::smallvec![];
        for &image in images.iter() {
            let view_create_info = ash::vk::ImageViewCreateInfo {
                image,
                view_type: ash::vk::ImageViewType::TYPE_2D,
                format: swapchain.format,
                components: identity_component_mapping(),
                subresource_range: base_level_subres_range(ash::vk::ImageAspectFlags::COLOR),
                ..Default::default()
            };
            let view = unsafe {
                device
                    .device
                    .create_image_view(&view_create_info, None)
                    .expect("Failed to create swapchain image view")
            };
            views.push(view);
        }
        SwapchainImages {
            images,
            views,
            device: Some(Rc::clone(device)),
        }
    }

    pub fn release_resources(&mut self) {
        if self.device.is_some() {
            for &view in self.views.iter() {
                unsafe {
                    self.device
                        .as_ref()
                        .unwrap()
                        .device
                        .destroy_image_view(view, None)
                };
            }
            self.device = None;
        }
        self.views.clear();
    }
}

impl Drop for SwapchainImages {
    fn drop(&mut self) {
        self.release_resources();
    }
}

pub struct SwapchainRenderPass {
    pub render_pass: ash::vk::RenderPass,
    device: Option<Rc<Device>>,
}

impl SwapchainRenderPass {
    pub fn new(
        physical_device: &PhysicalDevice,
        device: &Rc<Device>,
        swapchain: &Swapchain,
    ) -> Self {
        let color_attachment = ash::vk::AttachmentDescription {
            format: swapchain.format,
            samples: ash::vk::SampleCountFlags::TYPE_1,
            load_op: ash::vk::AttachmentLoadOp::CLEAR,
            store_op: ash::vk::AttachmentStoreOp::STORE,
            stencil_load_op: ash::vk::AttachmentLoadOp::DONT_CARE,
            stencil_store_op: ash::vk::AttachmentStoreOp::DONT_CARE,
            initial_layout: ash::vk::ImageLayout::UNDEFINED,
            final_layout: ash::vk::ImageLayout::PRESENT_SRC_KHR,
            ..Default::default()
        };
        let color_attachment_ref = ash::vk::AttachmentReference {
            attachment: 0,
            layout: ash::vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        };
        let ds_attachment = ash::vk::AttachmentDescription {
            format: physical_device.optimal_depth_stencil_format,
            samples: ash::vk::SampleCountFlags::TYPE_1,
            load_op: ash::vk::AttachmentLoadOp::CLEAR,
            store_op: ash::vk::AttachmentStoreOp::DONT_CARE,
            stencil_load_op: ash::vk::AttachmentLoadOp::CLEAR,
            stencil_store_op: ash::vk::AttachmentStoreOp::DONT_CARE,
            initial_layout: ash::vk::ImageLayout::UNDEFINED,
            final_layout: ash::vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            ..Default::default()
        };
        let ds_attachment_ref = ash::vk::AttachmentReference {
            attachment: 1,
            layout: ash::vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        };
        let subpass_desc = ash::vk::SubpassDescription {
            pipeline_bind_point: ash::vk::PipelineBindPoint::GRAPHICS,
            color_attachment_count: 1,
            p_color_attachments: &color_attachment_ref,
            p_depth_stencil_attachment: &ds_attachment_ref,
            ..Default::default()
        };
        let subpass_deps = [
            ash::vk::SubpassDependency {
                src_subpass: ash::vk::SUBPASS_EXTERNAL,
                dst_subpass: 0,
                src_stage_mask: ash::vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                dst_stage_mask: ash::vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                src_access_mask: ash::vk::AccessFlags::empty(),
                dst_access_mask: ash::vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                ..Default::default()
            },
            ash::vk::SubpassDependency {
                src_subpass: ash::vk::SUBPASS_EXTERNAL,
                dst_subpass: 0,
                src_stage_mask: ash::vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                    | ash::vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
                dst_stage_mask: ash::vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                    | ash::vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
                src_access_mask: ash::vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                dst_access_mask: ash::vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                    | ash::vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                ..Default::default()
            },
        ];
        let attachments = [color_attachment, ds_attachment];
        let renderpass_create_info = ash::vk::RenderPassCreateInfo {
            attachment_count: attachments.len() as u32,
            p_attachments: attachments.as_ptr(),
            subpass_count: 1,
            p_subpasses: &subpass_desc,
            dependency_count: subpass_deps.len() as u32,
            p_dependencies: subpass_deps.as_ptr(),
            ..Default::default()
        };
        let render_pass = unsafe {
            device
                .device
                .create_render_pass(&renderpass_create_info, None)
                .expect("Failed to create renderpass")
        };
        SwapchainRenderPass {
            render_pass,
            device: Some(Rc::clone(device)),
        }
    }

    pub fn release_resources(&mut self) {
        if self.device.is_some() {
            unsafe {
                self.device
                    .as_ref()
                    .unwrap()
                    .device
                    .destroy_render_pass(self.render_pass, None)
            };
            self.device = None;
        }
        self.render_pass = ash::vk::RenderPass::null();
    }
}

impl Drop for SwapchainRenderPass {
    fn drop(&mut self) {
        self.release_resources();
    }
}

pub struct SwapchainFramebuffers {
    pub framebuffers: smallvec::SmallVec<[ash::vk::Framebuffer; 8]>,
    device: Option<Rc<Device>>,
}

impl SwapchainFramebuffers {
    pub fn new(
        device: &Rc<Device>,
        swapchain: &Swapchain,
        images: &SwapchainImages,
        render_pass: &SwapchainRenderPass,
        depth_stencil_buffer: &DepthStencilBuffer,
    ) -> Self {
        let mut framebuffers: smallvec::SmallVec<[ash::vk::Framebuffer; 8]> = smallvec::smallvec![];
        for &view in images.views.iter() {
            let attachments = [view, depth_stencil_buffer.view];
            let framebuffer_create_info = ash::vk::FramebufferCreateInfo {
                render_pass: render_pass.render_pass,
                attachment_count: attachments.len() as u32,
                p_attachments: attachments.as_ptr(),
                width: swapchain.pixel_size.width,
                height: swapchain.pixel_size.height,
                layers: 1,
                ..Default::default()
            };
            let framebuffer = unsafe {
                device
                    .device
                    .create_framebuffer(&framebuffer_create_info, None)
                    .expect("Failed to create framebuffer")
            };
            framebuffers.push(framebuffer);
        }
        SwapchainFramebuffers {
            framebuffers,
            device: Some(Rc::clone(device)),
        }
    }

    pub fn release_resources(&mut self) {
        if self.device.is_some() {
            for &framebuffer in self.framebuffers.iter() {
                unsafe {
                    self.device
                        .as_ref()
                        .unwrap()
                        .device
                        .destroy_framebuffer(framebuffer, None)
                };
            }
            self.device = None;
        }
        self.framebuffers.clear();
    }
}

impl Drop for SwapchainFramebuffers {
    fn drop(&mut self) {
        self.release_resources();
    }
}

pub struct CommandPool {
    pub pools: smallvec::SmallVec<[ash::vk::CommandPool; FRAMES_IN_FLIGHT as usize]>,
    device: Option<Rc<Device>>,
}

impl CommandPool {
    pub fn new(physical_device: &PhysicalDevice, device: &Rc<Device>) -> Self {
        let mut pools: smallvec::SmallVec<[ash::vk::CommandPool; FRAMES_IN_FLIGHT as usize]> =
            smallvec::smallvec![];
        let pool_create_info = ash::vk::CommandPoolCreateInfo {
            queue_family_index: physical_device.gfx_compute_present_queue_family_index,
            ..Default::default()
        };
        for _ in 0..FRAMES_IN_FLIGHT {
            let pool = unsafe {
                device
                    .device
                    .create_command_pool(&pool_create_info, None)
                    .expect("Failed to create command pool")
            };
            pools.push(pool);
        }
        CommandPool {
            pools,
            device: Some(Rc::clone(device)),
        }
    }

    pub fn release_resources(&mut self) {
        if self.device.is_some() {
            for &pool in self.pools.iter() {
                unsafe {
                    self.device
                        .as_ref()
                        .unwrap()
                        .device
                        .destroy_command_pool(pool, None)
                };
            }
            self.device = None;
        }
        self.pools.clear();
    }

    pub fn reset(&self, device: &Device, slot: u32) {
        unsafe {
            device
                .device
                .reset_command_pool(
                    self.pools[slot as usize],
                    ash::vk::CommandPoolResetFlags::empty(),
                )
                .expect("Failed to reset command pool")
        };
    }
}

impl Drop for CommandPool {
    fn drop(&mut self) {
        self.release_resources();
    }
}

pub struct CommandList {
    pub command_buffers: smallvec::SmallVec<[ash::vk::CommandBuffer; FRAMES_IN_FLIGHT as usize]>,
}

impl CommandList {
    pub fn new(device: &Device, command_pool: &CommandPool) -> Self {
        let mut command_buffers: smallvec::SmallVec<
            [ash::vk::CommandBuffer; FRAMES_IN_FLIGHT as usize],
        > = smallvec::smallvec![];
        for slot in 0..FRAMES_IN_FLIGHT {
            let allocate_info = ash::vk::CommandBufferAllocateInfo {
                command_pool: command_pool.pools[slot as usize],
                level: ash::vk::CommandBufferLevel::PRIMARY,
                command_buffer_count: 1,
                ..Default::default()
            };
            let buffers = unsafe {
                device
                    .device
                    .allocate_command_buffers(&allocate_info)
                    .expect("Failed to allocate command buffer")
            };
            command_buffers.push(buffers[0]);
        }
        CommandList { command_buffers }
    }

    pub fn begin(&self, device: &Device, slot: u32) {
        let begin_info = ash::vk::CommandBufferBeginInfo {
            ..Default::default()
        };
        unsafe {
            device
                .device
                .begin_command_buffer(self.command_buffers[slot as usize], &begin_info)
                .expect("Failed to begin command buffer")
        };
    }

    pub fn end(&self, device: &Device, slot: u32) {
        unsafe {
            device
                .device
                .end_command_buffer(self.command_buffers[slot as usize])
                .expect("Failed to end command buffer")
        };
    }
}

struct SwapchainFrameSyncObjects {
    image_fence: ash::vk::Fence,
    cmd_fence: ash::vk::Fence,
    image_sem: ash::vk::Semaphore,
    present_sem: ash::vk::Semaphore,
    image_acquired: bool,
    image_fence_waitable: bool,
    cmd_fence_waitable: bool,
    image_sem_waitable: bool,
}

fn make_swapchain_frame_sync_objects(
    device: &Device,
) -> smallvec::SmallVec<[SwapchainFrameSyncObjects; FRAMES_IN_FLIGHT as usize]> {
    let mut sync_objects: smallvec::SmallVec<
        [SwapchainFrameSyncObjects; FRAMES_IN_FLIGHT as usize],
    > = smallvec::smallvec![];
    for _ in 0..FRAMES_IN_FLIGHT {
        let fence_create_info = ash::vk::FenceCreateInfo {
            flags: ash::vk::FenceCreateFlags::SIGNALED,
            ..Default::default()
        };
        let sem_create_info = ash::vk::SemaphoreCreateInfo {
            ..Default::default()
        };
        sync_objects.push(SwapchainFrameSyncObjects {
            image_fence: unsafe {
                device
                    .device
                    .create_fence(&fence_create_info, None)
                    .expect("Failed to create swapchain image fence")
            },
            cmd_fence: unsafe {
                device
                    .device
                    .create_fence(&fence_create_info, None)
                    .expect("Failed to create swapchain command fence")
            },
            image_sem: unsafe {
                device
                    .device
                    .create_semaphore(&sem_create_info, None)
                    .expect("Failed to create swapchain image semaphore")
            },
            present_sem: unsafe {
                device
                    .device
                    .create_semaphore(&sem_create_info, None)
                    .expect("Failed to create present semaphore")
            },
            image_acquired: false,
            image_fence_waitable: true,
            cmd_fence_waitable: true,
            image_sem_waitable: false,
        });
    }
    sync_objects
}

bitflags! {
    pub struct EndFrameFlags: u32 {
        const NO_RENDER_PASS = 0x01;
        const SKIP_PRESENT = 0x02;
    }
}

pub enum DeferredReleaseEntry {
    Buffer(u32, (ash::vk::Buffer, vk_mem::Allocation)),
    Image(u32, (ash::vk::Image, vk_mem::Allocation)),
}

pub struct SwapchainFrameState {
    sync_objects: smallvec::SmallVec<[SwapchainFrameSyncObjects; FRAMES_IN_FLIGHT as usize]>,
    pub current_frame_slot: u32,
    pub current_image_index: u32,
    pub frame_count: u64,
    device: Option<Rc<Device>>,
    allocator: Option<Rc<MemAllocator>>,
    deferred_release_queue: smallvec::SmallVec<[DeferredReleaseEntry; 32]>,
}

impl SwapchainFrameState {
    pub fn new(device: &Rc<Device>, allocator: &Rc<MemAllocator>) -> Self {
        SwapchainFrameState {
            sync_objects: make_swapchain_frame_sync_objects(device),
            current_frame_slot: 0,
            current_image_index: 0,
            frame_count: 0,
            device: Some(Rc::clone(device)),
            allocator: Some(Rc::clone(allocator)),
            deferred_release_queue: smallvec::smallvec![],
        }
    }

    pub fn release_resources(&mut self) {
        if self.allocator.is_some() {
            self.run_deferred_releases(true);
            self.allocator = None;
        }
        if self.device.is_some() {
            let device = self.device.as_ref().unwrap();
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
            self.device = None;
        }
        self.sync_objects.clear();
    }

    pub fn begin_frame(
        &mut self,
        swapchain: &Swapchain,
        command_pool: &CommandPool,
        command_list: &CommandList,
    ) -> Result<u32, ash::vk::Result> {
        let device = self.device.as_ref().unwrap();
        let s = &mut self.sync_objects[self.current_frame_slot as usize];
        if !s.image_acquired {
            if s.image_fence_waitable {
                device.wait_reset_fence(&s.image_fence);
                s.image_fence_waitable = false;
            }
            let index_and_suboptimal = unsafe {
                device.ext_swapchain.acquire_next_image(
                    swapchain.swapchain,
                    u64::max_value(),
                    s.image_sem,
                    s.image_fence,
                )?
            };
            self.current_image_index = index_and_suboptimal.0;
            s.image_acquired = true;
            s.image_fence_waitable = true;
            s.image_sem_waitable = true;
        }
        //println!("{} {}", self.current_frame_slot, self.current_image_index);

        if s.cmd_fence_waitable {
            device.wait_reset_fence(&s.cmd_fence);
            s.cmd_fence_waitable = false;
        }

        command_pool.reset(device, self.current_frame_slot);
        command_list.begin(device, self.current_frame_slot);

        self.run_deferred_releases(false);

        Ok(self.current_frame_slot)
    }

    pub fn end_frame(
        &mut self,
        swapchain: &Swapchain,
        swapchain_images: &SwapchainImages,
        command_list: &CommandList,
        flags: EndFrameFlags,
    ) {
        if flags.contains(EndFrameFlags::NO_RENDER_PASS) {
            self.transition_current_swapchain_image(
                swapchain_images,
                command_list,
                ash::vk::AccessFlags::empty(),
                ash::vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                ash::vk::ImageLayout::UNDEFINED,
                ash::vk::ImageLayout::PRESENT_SRC_KHR,
                ash::vk::PipelineStageFlags::TOP_OF_PIPE,
                ash::vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            );
        }

        let device = self.device.as_ref().unwrap();
        command_list.end(device, self.current_frame_slot);

        let needs_present = !flags.contains(EndFrameFlags::SKIP_PRESENT);
        let cb_ref = self.current_frame_command_buffer(command_list);
        let s = &mut self.sync_objects[self.current_frame_slot as usize];
        let wait_dst_stage_mask = ash::vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT;
        let submit_info = ash::vk::SubmitInfo {
            wait_semaphore_count: if s.image_sem_waitable { 1 } else { 0 },
            p_wait_semaphores: &s.image_sem,
            p_wait_dst_stage_mask: &wait_dst_stage_mask,
            command_buffer_count: 1,
            p_command_buffers: cb_ref,
            signal_semaphore_count: if needs_present { 1 } else { 0 },
            p_signal_semaphores: if needs_present {
                &s.present_sem
            } else {
                std::ptr::null()
            },
            ..Default::default()
        };
        unsafe {
            device
                .device
                .queue_submit(device.queue, &[submit_info], s.cmd_fence)
                .expect("Failed to submit to queue")
        };
        s.cmd_fence_waitable = true;
        s.image_sem_waitable = false;

        if needs_present {
            let present_info = ash::vk::PresentInfoKHR {
                wait_semaphore_count: 1,
                p_wait_semaphores: &s.present_sem,
                swapchain_count: 1,
                p_swapchains: &swapchain.swapchain,
                p_image_indices: &self.current_image_index,
                p_results: std::ptr::null_mut(),
                ..Default::default()
            };
            let present_result = unsafe {
                device
                    .ext_swapchain
                    .queue_present(device.queue, &present_info)
            };
            match present_result {
                Err(r) => {
                    if r != ash::vk::Result::ERROR_OUT_OF_DATE_KHR {
                        panic!("{}", r)
                    }
                }
                _ => (),
            }
            s.image_acquired = false;
            self.current_frame_slot = (self.current_frame_slot + 1) % FRAMES_IN_FLIGHT;
        }

        self.frame_count = self.frame_count.wrapping_add(1);
    }

    pub fn current_frame_command_buffer<'a>(
        &self,
        command_list: &'a CommandList,
    ) -> &'a ash::vk::CommandBuffer {
        &command_list.command_buffers[self.current_frame_slot as usize]
    }

    fn transition_current_swapchain_image(
        &self,
        swapchain_images: &SwapchainImages,
        command_list: &CommandList,
        src_access_mask: ash::vk::AccessFlags,
        dst_access_mask: ash::vk::AccessFlags,
        old_layout: ash::vk::ImageLayout,
        new_layout: ash::vk::ImageLayout,
        src_stage_mask: ash::vk::PipelineStageFlags,
        dst_stage_mask: ash::vk::PipelineStageFlags,
    ) {
        let image_barriers = [ash::vk::ImageMemoryBarrier {
            src_access_mask,
            dst_access_mask,
            old_layout,
            new_layout,
            image: swapchain_images.images[self.current_image_index as usize],
            subresource_range: base_level_subres_range(ash::vk::ImageAspectFlags::COLOR),
            ..Default::default()
        }];
        unsafe {
            self.device.as_ref().unwrap().device.cmd_pipeline_barrier(
                *self.current_frame_command_buffer(command_list),
                src_stage_mask,
                dst_stage_mask,
                ash::vk::DependencyFlags::empty(),
                &[],
                &[],
                &image_barriers,
            );
        }
    }

    fn run_deferred_releases(&mut self, forced: bool) {
        let allocator = self.allocator.as_ref().unwrap();
        let mut i = 0;
        while i != self.deferred_release_queue.len() {
            let mut remove = false;
            match self.deferred_release_queue[i] {
                DeferredReleaseEntry::Buffer(slot, buf_and_alloc) => {
                    if forced || slot == self.current_frame_slot {
                        allocator.destroy_buffer(&buf_and_alloc);
                        remove = true;
                    }
                }
                DeferredReleaseEntry::Image(slot, image_and_alloc) => {
                    if forced || slot == self.current_frame_slot {
                        allocator.destroy_image(&image_and_alloc);
                        remove = true;
                    }
                }
            }
            if remove {
                self.deferred_release_queue.remove(i);
            } else {
                i += 1;
            }
        }
    }

    pub fn deferred_release_buffer(
        &mut self,
        buf_and_alloc: &(ash::vk::Buffer, vk_mem::Allocation),
    ) {
        self.deferred_release_queue
            .push(DeferredReleaseEntry::Buffer(
                self.current_frame_slot,
                *buf_and_alloc,
            ));
    }

    pub fn deferred_release_image(
        &mut self,
        image_and_alloc: &(ash::vk::Image, vk_mem::Allocation),
    ) {
        self.deferred_release_queue
            .push(DeferredReleaseEntry::Image(
                self.current_frame_slot,
                *image_and_alloc,
            ));
    }
}

pub struct SwapchainResizer {
    surface_size: ash::vk::Extent2D,
}

impl SwapchainResizer {
    pub fn new() -> Self {
        SwapchainResizer {
            surface_size: ash::vk::Extent2D {
                width: 0,
                height: 0,
            },
        }
    }

    pub fn ensure_up_to_date(
        &mut self,
        instance: &Instance,
        physical_device: &PhysicalDevice,
        surface: &WindowSurface,
        window: &winit::window::Window,
        device: &Rc<Device>,
        allocator: &Rc<MemAllocator>,
        swapchain: &mut Swapchain,
        swapchain_render_pass: &SwapchainRenderPass,
        swapchain_images: &mut SwapchainImages,
        swapchain_framebuffers: &mut SwapchainFramebuffers,
        swapchain_frame_state: &mut SwapchainFrameState,
        depth_stencil_buffer: &mut DepthStencilBuffer,
    ) -> bool {
        let current_pixel_size = surface.pixel_size(instance, physical_device, window);
        if current_pixel_size.width != 0 && current_pixel_size.height != 0 {
            if self.surface_size != current_pixel_size {
                self.surface_size = current_pixel_size;
                device.wait_idle();
                swapchain_framebuffers.release_resources();
                swapchain_images.release_resources();
                swapchain_frame_state.release_resources();
                depth_stencil_buffer.release_resources();
                *swapchain = SwapchainBuilder::new()
                    .with_existing(swapchain)
                    .build(
                        instance,
                        physical_device,
                        device,
                        surface,
                        current_pixel_size,
                    )
                    .unwrap();
                *swapchain_images = SwapchainImages::new(device, swapchain);
                *depth_stencil_buffer =
                    DepthStencilBuffer::new(physical_device, device, swapchain.pixel_size);
                *swapchain_framebuffers = SwapchainFramebuffers::new(
                    device,
                    swapchain,
                    swapchain_images,
                    swapchain_render_pass,
                    depth_stencil_buffer,
                );
                *swapchain_frame_state = SwapchainFrameState::new(device, allocator);
                println!("Resized swapchain to {:?}", self.surface_size);
            }
            true
        } else {
            false
        }
    }
}

impl Drop for SwapchainFrameState {
    fn drop(&mut self) {
        self.release_resources();
    }
}

pub struct PipelineCache {
    pub cache: ash::vk::PipelineCache,
    device: Option<Rc<Device>>,
}

impl PipelineCache {
    pub fn new(device: &Rc<Device>) -> Self {
        let pipeline_cache_create_info = ash::vk::PipelineCacheCreateInfo {
            ..Default::default()
        };
        let cache = unsafe {
            device
                .device
                .create_pipeline_cache(&pipeline_cache_create_info, None)
                .expect("Failed to create pipeline cache")
        };
        PipelineCache {
            cache,
            device: Some(Rc::clone(device)),
        }
    }

    pub fn release_resources(&mut self) {
        if self.device.is_some() {
            unsafe {
                self.device
                    .as_ref()
                    .unwrap()
                    .device
                    .destroy_pipeline_cache(self.cache, None)
            };
            self.device = None;
        }
        self.cache = ash::vk::PipelineCache::null();
    }
}

impl Drop for PipelineCache {
    fn drop(&mut self) {
        self.release_resources();
    }
}

pub struct MemAllocator {
    pub allocator: vk_mem::Allocator,
    valid: bool,
}

impl MemAllocator {
    pub fn new(instance: &Instance, physical_device: &PhysicalDevice, device: &Device) -> Self {
        let create_info = vk_mem::AllocatorCreateInfo {
            physical_device: physical_device.physical_device,
            device: device.device.clone(),
            instance: instance.instance.clone(),
            flags: vk_mem::AllocatorCreateFlags::EXTERNALLY_SYNCHRONIZED,
            preferred_large_heap_block_size: 0,
            frame_in_use_count: 0,
            heap_size_limits: None,
        };
        let allocator =
            vk_mem::Allocator::new(&create_info).expect("Failed to create memory allocator");
        MemAllocator {
            allocator,
            valid: true,
        }
    }

    pub fn release_resources(&mut self) {
        if self.valid {
            self.allocator.destroy();
            self.valid = false;
        }
    }

    pub fn create_host_visible_buffer(
        &self,
        size: usize,
        usage: ash::vk::BufferUsageFlags,
    ) -> Result<(ash::vk::Buffer, vk_mem::Allocation), vk_mem::Error> {
        let buffer_create_info = ash::vk::BufferCreateInfo {
            size: size as u64,
            usage,
            sharing_mode: ash::vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };
        let allocation_create_info = vk_mem::AllocationCreateInfo {
            usage: vk_mem::MemoryUsage::CpuToGpu,
            flags: vk_mem::AllocationCreateFlags::MAPPED,
            ..Default::default()
        };
        match self
            .allocator
            .create_buffer(&buffer_create_info, &allocation_create_info)
        {
            Ok((buffer, allocation, _)) => Ok((buffer, allocation)),
            Err(e) => {
                println!("Failed to create host visible buffer: {}", e);
                Err(e)
            }
        }
    }

    pub fn create_device_local_buffer(
        &self,
        size: usize,
        usage: ash::vk::BufferUsageFlags,
    ) -> Result<(ash::vk::Buffer, vk_mem::Allocation), vk_mem::Error> {
        let buffer_create_info = ash::vk::BufferCreateInfo {
            size: size as u64,
            usage: usage | ash::vk::BufferUsageFlags::TRANSFER_DST,
            sharing_mode: ash::vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };
        let allocation_create_info = vk_mem::AllocationCreateInfo {
            usage: vk_mem::MemoryUsage::GpuOnly,
            ..Default::default()
        };
        match self
            .allocator
            .create_buffer(&buffer_create_info, &allocation_create_info)
        {
            Ok((buffer, allocation, _)) => Ok((buffer, allocation)),
            Err(e) => {
                println!("Failed to create device local buffer: {}", e);
                Err(e)
            }
        }
    }

    pub fn create_staging_buffer(
        &self,
        size: usize,
    ) -> Result<(ash::vk::Buffer, vk_mem::Allocation), vk_mem::Error> {
        let buffer_create_info = ash::vk::BufferCreateInfo {
            size: size as u64,
            usage: ash::vk::BufferUsageFlags::TRANSFER_SRC,
            sharing_mode: ash::vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };
        let allocation_create_info = vk_mem::AllocationCreateInfo {
            usage: vk_mem::MemoryUsage::CpuOnly,
            ..Default::default()
        };
        match self
            .allocator
            .create_buffer(&buffer_create_info, &allocation_create_info)
        {
            Ok((buffer, allocation, _)) => Ok((buffer, allocation)),
            Err(e) => {
                println!("Failed to create staging buffer: {}", e);
                Err(e)
            }
        }
    }

    pub fn destroy_buffer(&self, buf_and_alloc: &(ash::vk::Buffer, vk_mem::Allocation)) {
        match self
            .allocator
            .destroy_buffer(buf_and_alloc.0, &buf_and_alloc.1)
        {
            Err(e) => println!("Failed to destroy buffer: {}", e),
            _ => (),
        }
    }

    pub fn update_host_visible_buffer(
        &self,
        allocation: &vk_mem::Allocation,
        flush_range_offset: usize,
        flush_range_len: usize,
        base_offset: usize,
        chunks: &[(*const u8, usize, usize)],
    ) {
        // map_memory should mostly be no-op due to creating as MAPPED
        match self.allocator.map_memory(allocation) {
            Ok(p) => {
                unsafe {
                    for (src, copy_offset, copy_len) in chunks {
                        p.add(base_offset + copy_offset)
                            .copy_from_nonoverlapping(*src, *copy_len);
                    }
                };
            }
            Err(r) => panic!("{}", r),
        }
        self.allocator.unmap_memory(allocation).unwrap();
        self.allocator
            .flush_allocation(allocation, flush_range_offset, flush_range_len)
            .unwrap();
    }

    pub fn create_image(
        &self,
        image_create_info: &ash::vk::ImageCreateInfo,
    ) -> Result<(ash::vk::Image, vk_mem::Allocation), vk_mem::Error> {
        let allocation_create_info = vk_mem::AllocationCreateInfo {
            usage: vk_mem::MemoryUsage::GpuOnly,
            ..Default::default()
        };
        match self
            .allocator
            .create_image(image_create_info, &allocation_create_info)
        {
            Ok((image, allocation, _)) => Ok((image, allocation)),
            Err(e) => {
                println!("Failed to create image: {}", e);
                Err(e)
            }
        }
    }

    pub fn destroy_image(&self, image_and_alloc: &(ash::vk::Image, vk_mem::Allocation)) {
        match self
            .allocator
            .destroy_image(image_and_alloc.0, &image_and_alloc.1)
        {
            Err(e) => println!("Failed to destroy image: {}", e),
            _ => (),
        }
    }
}

impl Drop for MemAllocator {
    fn drop(&mut self) {
        self.release_resources();
    }
}

pub struct DescriptorSetLayout {
    pub layout: ash::vk::DescriptorSetLayout,
    device: Option<Rc<Device>>,
}

impl DescriptorSetLayout {
    pub fn new(device: &Rc<Device>, bindings: &[ash::vk::DescriptorSetLayoutBinding]) -> Self {
        let layout_create_info = ash::vk::DescriptorSetLayoutCreateInfo {
            binding_count: bindings.len() as u32,
            p_bindings: bindings.as_ptr(),
            ..Default::default()
        };
        let layout = unsafe {
            device
                .device
                .create_descriptor_set_layout(&layout_create_info, None)
                .expect("Failed to create descriptor set layout")
        };
        DescriptorSetLayout {
            layout,
            device: Some(Rc::clone(device)),
        }
    }

    pub fn release_resources(&mut self) {
        if self.device.is_some() {
            unsafe {
                self.device
                    .as_ref()
                    .unwrap()
                    .device
                    .destroy_descriptor_set_layout(self.layout, None)
            };
            self.device = None;
        }
        self.layout = ash::vk::DescriptorSetLayout::null();
    }
}

impl Drop for DescriptorSetLayout {
    fn drop(&mut self) {
        self.release_resources();
    }
}

pub struct DescriptorPool {
    pub pool: ash::vk::DescriptorPool,
    device: Option<Rc<Device>>,
}

impl DescriptorPool {
    pub fn new(device: &Rc<Device>, max_sets: u32, sizes: &[ash::vk::DescriptorPoolSize]) -> Self {
        let pool_create_info = ash::vk::DescriptorPoolCreateInfo {
            max_sets,
            pool_size_count: sizes.len() as u32,
            p_pool_sizes: sizes.as_ptr(),
            ..Default::default()
        };
        let pool = unsafe {
            device
                .device
                .create_descriptor_pool(&pool_create_info, None)
                .expect("Failed to create descriptor pool")
        };
        DescriptorPool {
            pool,
            device: Some(Rc::clone(device)),
        }
    }

    pub fn release_resources(&mut self) {
        if self.device.is_some() {
            unsafe {
                self.device
                    .as_ref()
                    .unwrap()
                    .device
                    .destroy_descriptor_pool(self.pool, None)
            };
            self.device = None;
        }
        self.pool = ash::vk::DescriptorPool::null();
    }

    pub fn allocate(
        &self,
        layouts: &[&DescriptorSetLayout],
    ) -> Result<Vec<ash::vk::DescriptorSet>, ash::vk::Result> {
        let set_layouts = layouts
            .iter()
            .map(|layout| layout.layout)
            .collect::<smallvec::SmallVec<[_; 8]>>();
        let allocate_info = ash::vk::DescriptorSetAllocateInfo {
            descriptor_pool: self.pool,
            descriptor_set_count: set_layouts.len() as u32,
            p_set_layouts: set_layouts.as_ptr(),
            ..Default::default()
        };
        unsafe {
            self.device
                .as_ref()
                .unwrap()
                .device
                .allocate_descriptor_sets(&allocate_info)
        }
    }
}

impl Drop for DescriptorPool {
    fn drop(&mut self) {
        self.release_resources();
    }
}

pub struct Shader {
    pub module: ash::vk::ShaderModule,
    stage: ash::vk::ShaderStageFlags,
    device: Option<Rc<Device>>,
}

impl Shader {
    pub fn new(device: &Rc<Device>, spv: &[u8], stage: ash::vk::ShaderStageFlags) -> Self {
        let shader_module_create_info = ash::vk::ShaderModuleCreateInfo {
            code_size: spv.len(),
            p_code: spv.as_ptr() as *const u32,
            ..Default::default()
        };
        let module = unsafe {
            device
                .device
                .create_shader_module(&shader_module_create_info, None)
                .expect("Failed to create shader module")
        };
        Shader {
            module,
            stage,
            device: Some(Rc::clone(device)),
        }
    }

    pub fn release_resources(&mut self) {
        if self.device.is_some() {
            unsafe {
                self.device
                    .as_ref()
                    .unwrap()
                    .device
                    .destroy_shader_module(self.module, None)
            };
            self.device = None;
        }
        self.module = ash::vk::ShaderModule::null();
    }
}

impl Drop for Shader {
    fn drop(&mut self) {
        self.release_resources();
    }
}

pub struct PipelineLayout {
    pub layout: ash::vk::PipelineLayout,
    device: Option<Rc<Device>>,
}

impl PipelineLayout {
    pub fn new(
        device: &Rc<Device>,
        desc_set_layouts: &[&DescriptorSetLayout],
        push_constant_ranges: &[ash::vk::PushConstantRange],
    ) -> Self {
        let set_layouts = desc_set_layouts
            .iter()
            .map(|layout| layout.layout)
            .collect::<smallvec::SmallVec<[_; 8]>>();
        let layout_create_info = ash::vk::PipelineLayoutCreateInfo {
            set_layout_count: set_layouts.len() as u32,
            p_set_layouts: set_layouts.as_ptr(),
            push_constant_range_count: push_constant_ranges.len() as u32,
            p_push_constant_ranges: push_constant_ranges.as_ptr(),
            ..Default::default()
        };
        let layout = unsafe {
            device
                .device
                .create_pipeline_layout(&layout_create_info, None)
                .expect("Failed to create pipeline layout")
        };
        PipelineLayout {
            layout,
            device: Some(Rc::clone(device)),
        }
    }

    pub fn release_resources(&mut self) {
        if self.device.is_some() {
            unsafe {
                self.device
                    .as_ref()
                    .unwrap()
                    .device
                    .destroy_pipeline_layout(self.layout, None)
            };
            self.device = None;
        }
        self.layout = ash::vk::PipelineLayout::null();
    }
}

impl Drop for PipelineLayout {
    fn drop(&mut self) {
        self.release_resources();
    }
}

pub struct GraphicsPipeline {
    pub pipeline: ash::vk::Pipeline,
    device: Option<Rc<Device>>,
}

impl GraphicsPipeline {
    pub fn release_resources(&mut self) {
        if self.device.is_some() {
            unsafe {
                self.device
                    .as_ref()
                    .unwrap()
                    .device
                    .destroy_pipeline(self.pipeline, None)
            };
            self.device = None;
        }
        self.pipeline = ash::vk::Pipeline::null();
    }
}

impl Drop for GraphicsPipeline {
    fn drop(&mut self) {
        self.release_resources();
    }
}

pub struct GraphicsPipelineBuilder<'a> {
    shader_stages: smallvec::SmallVec<[&'a Shader; 4]>,
    layout: Option<&'a PipelineLayout>,
    render_pass: Option<&'a ash::vk::RenderPass>,
    vertex_input_bindings: smallvec::SmallVec<[ash::vk::VertexInputBindingDescription; 4]>,
    vertex_input_attributes: smallvec::SmallVec<[ash::vk::VertexInputAttributeDescription; 8]>,
    topology: ash::vk::PrimitiveTopology,
    cull_mode: ash::vk::CullModeFlags,
    front_face: ash::vk::FrontFace,
    depth_stencil_state: ash::vk::PipelineDepthStencilStateCreateInfo,
    blend_state: ash::vk::PipelineColorBlendAttachmentState,
}

pub fn rgba_color_write_mask() -> ash::vk::ColorComponentFlags {
    ash::vk::ColorComponentFlags::R
        | ash::vk::ColorComponentFlags::G
        | ash::vk::ColorComponentFlags::B
        | ash::vk::ColorComponentFlags::A
}

impl<'a> GraphicsPipelineBuilder<'a> {
    pub fn new() -> Self {
        GraphicsPipelineBuilder {
            shader_stages: smallvec::smallvec![],
            layout: None,
            render_pass: None,
            vertex_input_bindings: smallvec::smallvec![],
            vertex_input_attributes: smallvec::smallvec![],
            topology: ash::vk::PrimitiveTopology::TRIANGLE_LIST,
            cull_mode: ash::vk::CullModeFlags::BACK,
            front_face: ash::vk::FrontFace::COUNTER_CLOCKWISE,
            depth_stencil_state: Default::default(),
            blend_state: ash::vk::PipelineColorBlendAttachmentState {
                color_write_mask: rgba_color_write_mask(),
                ..Default::default()
            },
        }
    }

    pub fn with_shader_stages(&mut self, shaders: &[&'a Shader]) -> &mut Self {
        for shader in shaders {
            self.shader_stages.push(shader);
        }
        self
    }

    pub fn with_layout(&mut self, layout: &'a PipelineLayout) -> &mut Self {
        self.layout = Some(layout);
        self
    }

    pub fn with_render_pass(&mut self, render_pass: &'a ash::vk::RenderPass) -> &mut Self {
        self.render_pass = Some(render_pass);
        self
    }

    pub fn with_vertex_input_layout(
        &mut self,
        bindings: &[ash::vk::VertexInputBindingDescription],
        attributes: &[ash::vk::VertexInputAttributeDescription],
    ) -> &mut Self {
        self.vertex_input_bindings = bindings.iter().map(|b| *b).collect();
        self.vertex_input_attributes = attributes.iter().map(|a| *a).collect();
        self
    }

    pub fn with_topology(&mut self, topology: ash::vk::PrimitiveTopology) -> &mut Self {
        self.topology = topology;
        self
    }

    pub fn with_cull_mode(&mut self, cull_mode: ash::vk::CullModeFlags) -> &mut Self {
        self.cull_mode = cull_mode;
        self
    }

    pub fn with_front_face(&mut self, front_face: ash::vk::FrontFace) -> &mut Self {
        self.front_face = front_face;
        self
    }

    pub fn with_depth_stencil_state(
        &mut self,
        state: ash::vk::PipelineDepthStencilStateCreateInfo,
    ) -> &mut Self {
        self.depth_stencil_state = state;
        self
    }

    pub fn with_blend_state(
        &mut self,
        state: ash::vk::PipelineColorBlendAttachmentState,
    ) -> &mut Self {
        self.blend_state = state;
        self
    }

    pub fn build(
        &self,
        device: &Rc<Device>,
        pipeline_cache: &PipelineCache,
    ) -> Result<GraphicsPipeline, ash::vk::Result> {
        let mut stages: smallvec::SmallVec<[ash::vk::PipelineShaderStageCreateInfo; 4]> =
            smallvec::smallvec![];
        for shader in &self.shader_stages {
            stages.push(ash::vk::PipelineShaderStageCreateInfo {
                stage: shader.stage,
                module: shader.module,
                p_name: DEFAULT_SHADER_ENTRY_POINT.as_ptr(),
                ..Default::default()
            });
        }
        let vertex_input_state = ash::vk::PipelineVertexInputStateCreateInfo {
            vertex_binding_description_count: self.vertex_input_bindings.len() as u32,
            p_vertex_binding_descriptions: self.vertex_input_bindings.as_ptr(),
            vertex_attribute_description_count: self.vertex_input_attributes.len() as u32,
            p_vertex_attribute_descriptions: self.vertex_input_attributes.as_ptr(),
            ..Default::default()
        };
        let primitive_restart_enable = match self.topology {
            ash::vk::PrimitiveTopology::LINE_STRIP
            | ash::vk::PrimitiveTopology::TRIANGLE_STRIP
            | ash::vk::PrimitiveTopology::TRIANGLE_FAN => ash::vk::TRUE,
            _ => ash::vk::FALSE,
        };
        let input_assembly_state = ash::vk::PipelineInputAssemblyStateCreateInfo {
            topology: self.topology,
            primitive_restart_enable,
            ..Default::default()
        };
        let viewport_state = ash::vk::PipelineViewportStateCreateInfo {
            viewport_count: 1,
            scissor_count: 1,
            ..Default::default()
        };
        let rasterization_state = ash::vk::PipelineRasterizationStateCreateInfo {
            cull_mode: self.cull_mode,
            front_face: self.front_face,
            line_width: 1.0,
            ..Default::default()
        };
        let multisample_state = ash::vk::PipelineMultisampleStateCreateInfo {
            rasterization_samples: ash::vk::SampleCountFlags::TYPE_1,
            ..Default::default()
        };
        let color_blend_attachment_list = [self.blend_state];
        let color_blend_state = ash::vk::PipelineColorBlendStateCreateInfo {
            attachment_count: color_blend_attachment_list.len() as u32,
            p_attachments: color_blend_attachment_list.as_ptr(),
            ..Default::default()
        };
        let dynamic_state_list = [
            ash::vk::DynamicState::VIEWPORT,
            ash::vk::DynamicState::SCISSOR,
        ];
        let dynamic_state = ash::vk::PipelineDynamicStateCreateInfo {
            dynamic_state_count: dynamic_state_list.len() as u32,
            p_dynamic_states: dynamic_state_list.as_ptr(),
            ..Default::default()
        };
        let graphics_pipeline_create_info = ash::vk::GraphicsPipelineCreateInfo {
            stage_count: stages.len() as u32,
            p_stages: stages.as_ptr(),
            p_vertex_input_state: &vertex_input_state,
            p_input_assembly_state: &input_assembly_state,
            p_viewport_state: &viewport_state,
            p_rasterization_state: &rasterization_state,
            p_multisample_state: &multisample_state,
            p_depth_stencil_state: &self.depth_stencil_state,
            p_color_blend_state: &color_blend_state,
            p_dynamic_state: &dynamic_state,
            layout: self
                .layout
                .expect("No layout specified for GraphicsPipeline")
                .layout,
            render_pass: *self
                .render_pass
                .expect("No renderpass specified for GraphicsPipeline"),
            ..Default::default()
        };
        match unsafe {
            device.device.create_graphics_pipelines(
                pipeline_cache.cache,
                &[graphics_pipeline_create_info],
                None,
            )
        } {
            Ok(pipelines) => Ok(GraphicsPipeline {
                pipeline: pipelines[0],
                device: Some(Rc::clone(device)),
            }),
            Err((_, r)) => Err(r),
        }
    }
}

pub struct Sampler {
    pub sampler: ash::vk::Sampler,
    device: Option<Rc<Device>>,
}

impl Sampler {
    pub fn new(device: &Rc<Device>, create_info: &ash::vk::SamplerCreateInfo) -> Self {
        let sampler = unsafe {
            device
                .device
                .create_sampler(&create_info, None)
                .expect("Failed to create sampler")
        };
        Sampler {
            sampler,
            device: Some(Rc::clone(device)),
        }
    }

    pub fn release_resources(&mut self) {
        if self.device.is_some() {
            unsafe {
                self.device
                    .as_ref()
                    .unwrap()
                    .device
                    .destroy_sampler(self.sampler, None);
            };
            self.device = None;
        }
        self.sampler = ash::vk::Sampler::null();
    }
}

impl Drop for Sampler {
    fn drop(&mut self) {
        self.release_resources();
    }
}

pub struct ImageView {
    pub view: ash::vk::ImageView,
    device: Option<Rc<Device>>,
}

impl ImageView {
    pub fn new(device: &Rc<Device>, create_info: &ash::vk::ImageViewCreateInfo) -> Self {
        let view = unsafe {
            device
                .device
                .create_image_view(&create_info, None)
                .expect("Failed to create image view")
        };
        ImageView {
            view,
            device: Some(Rc::clone(device)),
        }
    }

    pub fn release_resources(&mut self) {
        if self.device.is_some() {
            unsafe {
                self.device
                    .as_ref()
                    .unwrap()
                    .device
                    .destroy_image_view(self.view, None);
            };
            self.device = None;
        }
        self.view = ash::vk::ImageView::null();
    }
}

impl Drop for ImageView {
    fn drop(&mut self) {
        self.release_resources();
    }
}

pub fn buffer_copy_from_staging(
    device: &Device,
    cb: &ash::vk::CommandBuffer,
    dst: &ash::vk::Buffer,
    staging: &ash::vk::Buffer,
    size: usize,
) {
    let copy_info = [ash::vk::BufferCopy {
        size: size as u64,
        ..Default::default()
    }];
    unsafe {
        device
            .device
            .cmd_copy_buffer(*cb, *staging, *dst, &copy_info);
    }
}

pub fn buffer_barrier(
    device: &Device,
    cb: &ash::vk::CommandBuffer,
    buffer: &ash::vk::Buffer,
    size: usize,
    src_stage_mask: ash::vk::PipelineStageFlags,
    src_access_mask: ash::vk::AccessFlags,
    dst_stage_mask: ash::vk::PipelineStageFlags,
    dst_access_mask: ash::vk::AccessFlags,
) {
    let barrier_info = [ash::vk::BufferMemoryBarrier {
        src_access_mask,
        dst_access_mask,
        src_queue_family_index: ash::vk::QUEUE_FAMILY_IGNORED,
        dst_queue_family_index: ash::vk::QUEUE_FAMILY_IGNORED,
        buffer: *buffer,
        size: size as u64,
        ..Default::default()
    }];
    unsafe {
        device.device.cmd_pipeline_barrier(
            *cb,
            src_stage_mask,
            dst_stage_mask,
            ash::vk::DependencyFlags::empty(),
            &[],
            &barrier_info,
            &[],
        );
    }
}

pub enum VertexIndexBufferType {
    Vertex,
    Index,
}

pub fn create_or_reuse_vertexindex_buffer_with_data(
    device: &Device,
    allocator: &MemAllocator,
    swapchain_frame_state: &mut SwapchainFrameState,
    cb: &ash::vk::CommandBuffer,
    typ: VertexIndexBufferType,
    total_size: usize,
    chunks: &[(*const u8, usize, usize)],
    old_buffer: Option<(ash::vk::Buffer, vk_mem::Allocation, usize)>,
) -> (ash::vk::Buffer, vk_mem::Allocation) {
    let mut existing_buf_and_alloc: Option<(ash::vk::Buffer, vk_mem::Allocation)> = None;
    match old_buffer {
        Some((existing_buf, existing_alloc, existing_size)) => {
            if existing_size < total_size {
                swapchain_frame_state.deferred_release_buffer(&(existing_buf, existing_alloc));
            } else if existing_buf != ash::vk::Buffer::null() {
                existing_buf_and_alloc = Some((existing_buf, existing_alloc));
            }
        }
        _ => (),
    }
    let usage = match typ {
        VertexIndexBufferType::Vertex => ash::vk::BufferUsageFlags::VERTEX_BUFFER,
        VertexIndexBufferType::Index => ash::vk::BufferUsageFlags::INDEX_BUFFER,
    };
    let buf = if existing_buf_and_alloc.is_some() {
        existing_buf_and_alloc.unwrap()
    } else {
        allocator
            .create_device_local_buffer(total_size, usage)
            .unwrap()
    };
    let staging_buf = allocator.create_staging_buffer(total_size).unwrap();
    swapchain_frame_state.deferred_release_buffer(&staging_buf);
    allocator.update_host_visible_buffer(&staging_buf.1, 0, total_size, 0, chunks);
    buffer_copy_from_staging(device, cb, &buf.0, &staging_buf.0, total_size);
    let dst_access_mask = match typ {
        VertexIndexBufferType::Vertex => ash::vk::AccessFlags::VERTEX_ATTRIBUTE_READ,
        VertexIndexBufferType::Index => ash::vk::AccessFlags::INDEX_READ,
    };
    buffer_barrier(
        device,
        cb,
        &buf.0,
        total_size,
        ash::vk::PipelineStageFlags::TRANSFER,
        ash::vk::AccessFlags::TRANSFER_WRITE,
        ash::vk::PipelineStageFlags::VERTEX_INPUT,
        dst_access_mask,
    );
    buf
}

pub fn create_or_reuse_host_visible_vertexindex_buffer_with_data(
    allocator: &MemAllocator,
    swapchain_frame_state: &mut SwapchainFrameState,
    typ: VertexIndexBufferType,
    total_size: usize,
    chunks: &[(*const u8, usize, usize)],
    old_buffer: Option<(ash::vk::Buffer, vk_mem::Allocation, usize)>,
) -> (ash::vk::Buffer, vk_mem::Allocation) {
    let mut existing_buf_and_alloc: Option<(ash::vk::Buffer, vk_mem::Allocation)> = None;
    match old_buffer {
        Some((existing_buf, existing_alloc, existing_size)) => {
            if existing_size < total_size {
                swapchain_frame_state.deferred_release_buffer(&(existing_buf, existing_alloc));
            } else if existing_buf != ash::vk::Buffer::null() {
                existing_buf_and_alloc = Some((existing_buf, existing_alloc));
            }
        }
        _ => (),
    }
    let usage = match typ {
        VertexIndexBufferType::Vertex => ash::vk::BufferUsageFlags::VERTEX_BUFFER,
        VertexIndexBufferType::Index => ash::vk::BufferUsageFlags::INDEX_BUFFER,
    };
    let buf = if existing_buf_and_alloc.is_some() {
        existing_buf_and_alloc.unwrap()
    } else {
        allocator
            .create_host_visible_buffer(total_size, usage)
            .unwrap()
    };
    allocator.update_host_visible_buffer(&buf.1, 0, total_size, 0, chunks);
    buf
}

pub fn create_base_rgba_2d_texture_for_sampling(
    device: &Device,
    allocator: &MemAllocator,
    swapchain_frame_state: &mut SwapchainFrameState,
    cb: &ash::vk::CommandBuffer,
    pixel_size: ash::vk::Extent2D,
    pixels: &[u8],
    sampled_in_stage_mask: ash::vk::PipelineStageFlags,
) -> (ash::vk::Image, vk_mem::Allocation) {
    let extent = ash::vk::Extent3D {
        width: pixel_size.width,
        height: pixel_size.height,
        depth: 1,
    };
    let texture = allocator
        .create_image(&ash::vk::ImageCreateInfo {
            image_type: ash::vk::ImageType::TYPE_2D,
            format: ash::vk::Format::R8G8B8A8_UNORM,
            extent,
            mip_levels: 1,
            array_layers: 1,
            samples: ash::vk::SampleCountFlags::TYPE_1,
            tiling: ash::vk::ImageTiling::OPTIMAL,
            usage: ash::vk::ImageUsageFlags::SAMPLED | ash::vk::ImageUsageFlags::TRANSFER_DST,
            sharing_mode: ash::vk::SharingMode::EXCLUSIVE,
            initial_layout: ash::vk::ImageLayout::PREINITIALIZED,
            ..Default::default()
        })
        .expect("Failed to create image object for 2D texture");
    let image_staging_buf = allocator.create_staging_buffer(pixels.len()).unwrap();
    swapchain_frame_state.deferred_release_buffer(&image_staging_buf);
    allocator.update_host_visible_buffer(
        &image_staging_buf.1,
        0,
        pixels.len(),
        0,
        &[(pixels.as_ptr() as *const u8, 0, pixels.len())],
    );
    let buffer_image_copy = [ash::vk::BufferImageCopy {
        buffer_offset: 0,
        buffer_row_length: pixel_size.width,
        buffer_image_height: pixel_size.height,
        image_subresource: ash::vk::ImageSubresourceLayers {
            aspect_mask: ash::vk::ImageAspectFlags::COLOR,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        },
        image_offset: ash::vk::Offset3D { x: 0, y: 0, z: 0 },
        image_extent: extent,
        ..Default::default()
    }];
    let buffer_image_copy_before_barrier = [ash::vk::ImageMemoryBarrier {
        src_access_mask: ash::vk::AccessFlags::empty(),
        dst_access_mask: ash::vk::AccessFlags::TRANSFER_WRITE,
        old_layout: ash::vk::ImageLayout::PREINITIALIZED,
        new_layout: ash::vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        image: texture.0,
        subresource_range: base_level_subres_range(ash::vk::ImageAspectFlags::COLOR),
        ..Default::default()
    }];
    let buffer_image_copy_after_barrier = [ash::vk::ImageMemoryBarrier {
        src_access_mask: ash::vk::AccessFlags::TRANSFER_WRITE,
        dst_access_mask: ash::vk::AccessFlags::SHADER_READ,
        old_layout: ash::vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        new_layout: ash::vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        image: texture.0,
        subresource_range: base_level_subres_range(ash::vk::ImageAspectFlags::COLOR),
        ..Default::default()
    }];
    unsafe {
        device.device.cmd_pipeline_barrier(
            *cb,
            ash::vk::PipelineStageFlags::TOP_OF_PIPE,
            ash::vk::PipelineStageFlags::TRANSFER,
            ash::vk::DependencyFlags::empty(),
            &[],
            &[],
            &buffer_image_copy_before_barrier,
        );
        device.device.cmd_copy_buffer_to_image(
            *cb,
            image_staging_buf.0,
            texture.0,
            ash::vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &buffer_image_copy,
        );
        device.device.cmd_pipeline_barrier(
            *cb,
            ash::vk::PipelineStageFlags::TRANSFER,
            sampled_in_stage_mask,
            ash::vk::DependencyFlags::empty(),
            &[],
            &[],
            &buffer_image_copy_after_barrier,
        );
    }
    texture
}
