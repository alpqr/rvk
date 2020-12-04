use ash::version::{DeviceV1_0, EntryV1_0, InstanceV1_0};
use bitflags::bitflags;
use const_cstr::const_cstr;
use std::rc::Rc;

const ENABLE_VALIDATION: bool = true;

const FRAMES_IN_FLIGHT: u32 = 2;

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
            instance: Some(instance.clone()),
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

pub struct PhysicalDevice {
    pub physical_device: ash::vk::PhysicalDevice,
    pub properties: ash::vk::PhysicalDeviceProperties,
    pub memory_properties: ash::vk::PhysicalDeviceMemoryProperties,
    pub features: ash::vk::PhysicalDeviceFeatures,
    pub queue_family_properties: Vec<ash::vk::QueueFamilyProperties>,
    pub gfx_compute_present_queue_family_index: u32, // for now just assumes that a combined graphics+compute+present will be available
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
        let fence_list = [*fence];
        unsafe {
            self.device
                .wait_for_fences(&fence_list, true, u64::max_value())
                .expect("Fence wait failed")
        };
    }

    pub fn wait_reset_fence(&self, fence: &ash::vk::Fence) {
        let fence_list = [*fence];
        unsafe {
            self.device
                .wait_for_fences(&fence_list, true, u64::max_value())
                .expect("Fence wait failed");
            self.device
                .reset_fences(&fence_list)
                .expect("Fence reset failed");
        }
    }
}

impl Drop for Device {
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
        window: &winit::window::Window,
    ) -> Result<Swapchain, ()> {
        let image_extent = surface.pixel_size(instance, physical_device, window);
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
        let present_mode = if self.flags.contains(SwapchainFlags::NO_VSYNC) {
            let supported_present_modes = unsafe {
                instance
                    .ext_surface
                    .get_physical_device_surface_present_modes(
                        physical_device.physical_device,
                        surface.surface,
                    )
                    .unwrap()
            };
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
            device: Some(device.clone()),
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
                components: ash::vk::ComponentMapping {
                    r: ash::vk::ComponentSwizzle::IDENTITY,
                    g: ash::vk::ComponentSwizzle::IDENTITY,
                    b: ash::vk::ComponentSwizzle::IDENTITY,
                    a: ash::vk::ComponentSwizzle::IDENTITY,
                },
                subresource_range: ash::vk::ImageSubresourceRange {
                    aspect_mask: ash::vk::ImageAspectFlags::COLOR,
                    level_count: 1,
                    layer_count: 1,
                    ..Default::default()
                },
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
            device: Some(device.clone()),
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
    render_pass: ash::vk::RenderPass,
    device: Option<Rc<Device>>,
}

impl SwapchainRenderPass {
    pub fn new(device: &Rc<Device>, swapchain: &Swapchain) -> Self {
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
        let subpass_desc = ash::vk::SubpassDescription {
            pipeline_bind_point: ash::vk::PipelineBindPoint::GRAPHICS,
            color_attachment_count: 1,
            p_color_attachments: &color_attachment_ref,
            ..Default::default()
        };
        let subpass_dep = ash::vk::SubpassDependency {
            src_subpass: ash::vk::SUBPASS_EXTERNAL,
            dst_subpass: 0,
            src_stage_mask: ash::vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            dst_stage_mask: ash::vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            src_access_mask: ash::vk::AccessFlags::empty(),
            dst_access_mask: ash::vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            ..Default::default()
        };
        let attachments = [color_attachment];
        let dependencies = [subpass_dep];
        let renderpass_create_info = ash::vk::RenderPassCreateInfo {
            attachment_count: attachments.len() as u32,
            p_attachments: attachments.as_ptr(),
            subpass_count: 1,
            p_subpasses: &subpass_desc,
            dependency_count: 1,
            p_dependencies: dependencies.as_ptr(),
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
            device: Some(device.clone()),
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
    ) -> Self {
        let mut framebuffers: smallvec::SmallVec<[ash::vk::Framebuffer; 8]> = smallvec::smallvec![];
        for &view in images.views.iter() {
            let attachments = [view];
            let framebuffer_create_info = ash::vk::FramebufferCreateInfo {
                render_pass: render_pass.render_pass,
                attachment_count: 1,
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
            device: Some(device.clone()),
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
            device: Some(device.clone()),
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

pub struct SwapchainFrameSyncObjects {
    pub image_fence: ash::vk::Fence,
    pub cmd_fence: ash::vk::Fence,
    pub image_sem: ash::vk::Semaphore,
    pub present_sem: ash::vk::Semaphore,
    pub image_acquired: bool,
    pub image_fence_waitable: bool,
    pub cmd_fence_waitable: bool,
    pub image_sem_waitable: bool,
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

pub struct SwapchainFrameState {
    pub sync_objects: smallvec::SmallVec<[SwapchainFrameSyncObjects; FRAMES_IN_FLIGHT as usize]>,
    current_frame_slot: u32,
    current_image_index: u32,
    frame_count: u64,
    render_pass_count: u32,
    device: Option<Rc<Device>>,
}

impl SwapchainFrameState {
    pub fn new(device: &Rc<Device>) -> Self {
        SwapchainFrameState {
            sync_objects: make_swapchain_frame_sync_objects(device),
            current_frame_slot: 0,
            current_image_index: 0,
            frame_count: 0,
            render_pass_count: 0,
            device: Some(device.clone()),
        }
    }

    pub fn release_resources(&mut self) {
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
        device: &Device,
        swapchain: &Swapchain,
        command_pool: &CommandPool,
        command_list: &CommandList,
    ) -> Result<u32, ash::vk::Result> {
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

        self.render_pass_count = 0;

        Ok(self.current_frame_slot)
    }

    pub fn end_frame(
        &mut self,
        device: &Device,
        swapchain: &Swapchain,
        swapchain_images: &SwapchainImages,
        command_list: &CommandList,
    ) {
        if self.render_pass_count == 0 {
            self.transition(
                device,
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

        command_list.end(device, self.current_frame_slot);

        let s = &mut self.sync_objects[self.current_frame_slot as usize];
        let wait_dst_stage_mask = ash::vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT;
        let submit_info = ash::vk::SubmitInfo {
            wait_semaphore_count: if s.image_sem_waitable { 1 } else { 0 },
            p_wait_semaphores: &s.image_sem,
            p_wait_dst_stage_mask: &wait_dst_stage_mask,
            command_buffer_count: 1,
            p_command_buffers: &command_list.command_buffers[self.current_frame_slot as usize],
            signal_semaphore_count: 1,
            p_signal_semaphores: &s.present_sem,
            ..Default::default()
        };
        let submits = [submit_info];
        unsafe {
            device
                .device
                .queue_submit(device.queue, &submits, s.cmd_fence)
                .expect("Failed to submit to queue")
        };
        s.cmd_fence_waitable = true;
        s.image_sem_waitable = false;

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
                    panic!(r)
                }
            }
            _ => (),
        }
        s.image_acquired = false;

        self.current_frame_slot = (self.current_frame_slot + 1) % FRAMES_IN_FLIGHT;
        self.frame_count += 1;
    }

    fn transition(
        &self,
        device: &Device,
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
            subresource_range: ash::vk::ImageSubresourceRange {
                aspect_mask: ash::vk::ImageAspectFlags::COLOR,
                level_count: 1,
                layer_count: 1,
                ..Default::default()
            },
            ..Default::default()
        }];
        unsafe {
            device.device.cmd_pipeline_barrier(
                command_list.command_buffers[self.current_frame_slot as usize],
                src_stage_mask,
                dst_stage_mask,
                ash::vk::DependencyFlags::empty(),
                &[],
                &[],
                &image_barriers,
            );
        }
    }

    pub fn begin_render_pass(
        &self,
        device: &Device,
        swapchain: &Swapchain,
        framebuffers: &SwapchainFramebuffers,
        render_pass: &SwapchainRenderPass,
        command_list: &CommandList,
        clear_color: [f32; 4],
    ) {
        let cb = command_list.command_buffers[self.current_frame_slot as usize];
        let clear_values = [ash::vk::ClearValue {
            color: ash::vk::ClearColorValue {
                float32: clear_color,
            },
        }];
        let begin_info = ash::vk::RenderPassBeginInfo {
            render_pass: render_pass.render_pass,
            framebuffer: framebuffers.framebuffers[self.current_image_index as usize],
            render_area: ash::vk::Rect2D {
                offset: ash::vk::Offset2D { x: 0, y: 0 },
                extent: swapchain.pixel_size,
            },
            clear_value_count: clear_values.len() as u32,
            p_clear_values: clear_values.as_ptr(),
            ..Default::default()
        };
        unsafe {
            device
                .device
                .cmd_begin_render_pass(cb, &begin_info, ash::vk::SubpassContents::INLINE)
        };
    }

    pub fn end_render_pass(&mut self, device: &Device, command_list: &CommandList) {
        let cb = command_list.command_buffers[self.current_frame_slot as usize];
        unsafe { device.device.cmd_end_render_pass(cb) };
        self.render_pass_count += 1;
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
        swapchain: &mut Swapchain,
        swapchain_render_pass: &SwapchainRenderPass,
        swapchain_images: &mut SwapchainImages,
        swapchain_framebuffers: &mut SwapchainFramebuffers,
        swapchain_frame_state: &mut SwapchainFrameState,
    ) -> bool {
        let current_pixel_size = surface.pixel_size(instance, physical_device, window);
        if current_pixel_size.width != 0 && current_pixel_size.height != 0 {
            if self.surface_size != current_pixel_size {
                self.surface_size = current_pixel_size;
                device.wait_idle();
                swapchain_framebuffers.release_resources();
                swapchain_images.release_resources();
                swapchain_frame_state.release_resources();
                *swapchain = SwapchainBuilder::new()
                    .with_existing(swapchain)
                    .build(instance, physical_device, device, surface, window)
                    .unwrap();
                *swapchain_images = SwapchainImages::new(device, swapchain);
                *swapchain_framebuffers = SwapchainFramebuffers::new(
                    device,
                    swapchain,
                    swapchain_images,
                    swapchain_render_pass,
                );
                *swapchain_frame_state = SwapchainFrameState::new(device);
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
    cache: ash::vk::PipelineCache,
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
            device: Some(device.clone()),
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
        size: u64,
        usage: ash::vk::BufferUsageFlags,
    ) -> Result<(ash::vk::Buffer, vk_mem::Allocation), vk_mem::Error> {
        let buffer_create_info = ash::vk::BufferCreateInfo {
            size,
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

    pub fn destroy_buffer(&self, buffer: ash::vk::Buffer, allocation: &vk_mem::Allocation) {
        match self.allocator.destroy_buffer(buffer, allocation) {
            Err(e) => println!("Failed to destroy buffer: {}", e),
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
    layout: ash::vk::DescriptorSetLayout,
    device: Option<Rc<Device>>
}

impl DescriptorSetLayout {
    pub fn new(device: &Rc<Device>, bindings: &[ash::vk::DescriptorSetLayoutBinding]) -> Self {
        let layout_create_info = ash::vk::DescriptorSetLayoutCreateInfo {
            binding_count: bindings.len() as u32,
            p_bindings: bindings.as_ptr(),
            ..Default::default()
        };
        let layout = unsafe { device.device.create_descriptor_set_layout(&layout_create_info, None).expect("Failed to create descriptor set layout") };
        DescriptorSetLayout {
            layout,
            device: Some(device.clone())
        }
    }

    pub fn release_resources(&mut self) {
        if self.device.is_some() {
            unsafe { self.device.as_ref().unwrap().device.destroy_descriptor_set_layout(self.layout, None) };
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
    pool: ash::vk::DescriptorPool,
    device: Option<Rc<Device>>
}

impl DescriptorPool {
    pub fn new(device: &Rc<Device>, max_sets: u32, sizes: &[ash::vk::DescriptorPoolSize]) -> Self {
        let pool_create_info = ash::vk::DescriptorPoolCreateInfo {
            max_sets,
            pool_size_count: sizes.len() as u32,
            p_pool_sizes: sizes.as_ptr(),
            ..Default::default()
        };
        let pool = unsafe { device.device.create_descriptor_pool(&pool_create_info, None).expect("Failed to create descriptor pool") };
        DescriptorPool {
            pool,
            device: Some(device.clone())
        }
    }

    pub fn release_resources(&mut self) {
        if self.device.is_some() {
            unsafe { self.device.as_ref().unwrap().device.destroy_descriptor_pool(self.pool, None) };
            self.device = None;
        }
        self.pool = ash::vk::DescriptorPool::null();
    }

    pub fn allocate(&self, layouts: &[&DescriptorSetLayout]) -> Result<Vec<ash::vk::DescriptorSet>, ash::vk::Result> {
        let set_layouts: smallvec::SmallVec::<[ash::vk::DescriptorSetLayout; 8]> = layouts.iter().map(|layout| layout.layout).collect();
        let allocate_info = ash::vk::DescriptorSetAllocateInfo {
            descriptor_pool: self.pool,
            descriptor_set_count: set_layouts.len() as u32,
            p_set_layouts: set_layouts.as_ptr(),
            ..Default::default()
        };
        unsafe { self.device.as_ref().unwrap().device.allocate_descriptor_sets(&allocate_info) }
    }
}

impl Drop for DescriptorPool {
    fn drop(&mut self) {
        self.release_resources();
    }
}

pub struct Shader {
    module: ash::vk::ShaderModule,
    stage: ash::vk::ShaderStageFlags,
    device: Option<Rc<Device>>,
}

impl Shader {
    pub fn new(spv: &[u8], stage: ash::vk::ShaderStageFlags, device: &Rc<Device>) -> Self {
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
            device: Some(device.clone()),
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
    layout: ash::vk::PipelineLayout,
    device: Option<Rc<Device>>
}

impl PipelineLayout {
    pub fn new(device: &Rc<Device>, desc_set_layouts: &[&DescriptorSetLayout], push_constant_ranges: &[ash::vk::PushConstantRange]) -> Self {
        let set_layouts: smallvec::SmallVec::<[ash::vk::DescriptorSetLayout; 8]> = desc_set_layouts.iter().map(|layout| layout.layout).collect();
        let layout_create_info = ash::vk::PipelineLayoutCreateInfo {
            set_layout_count: set_layouts.len() as u32,
            p_set_layouts: set_layouts.as_ptr(),
            push_constant_range_count: push_constant_ranges.len() as u32,
            p_push_constant_ranges: push_constant_ranges.as_ptr(),
            ..Default::default()
        };
        let layout = unsafe { device.device.create_pipeline_layout(&layout_create_info, None).expect("Failed to create pipeline layout") };
        PipelineLayout {
            layout,
            device: Some(device.clone())
        }
    }

    pub fn release_resources(&mut self) {
        if self.device.is_some() {
            unsafe { self.device.as_ref().unwrap().device.destroy_pipeline_layout(self.layout, None) };
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
    pipeline: ash::vk::Pipeline,
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
    render_pass: Option<&'a ash::vk::RenderPass>,
}

impl<'a> GraphicsPipelineBuilder<'a> {
    pub fn new() -> Self {
        GraphicsPipelineBuilder {
            shader_stages: smallvec::smallvec![],
            render_pass: None,
        }
    }

    pub fn with_shader_stages(&mut self, shaders: &[&'a Shader]) -> &mut Self {
        for shader in shaders {
            self.shader_stages.push(shader);
        }
        self
    }

    pub fn with_render_pass(&mut self, render_pass: &'a ash::vk::RenderPass) -> &mut Self {
        self.render_pass = Some(render_pass);
        self
    }

    pub fn build(
        &self,
        device: &Rc<Device>,
        pipeline_cache: &PipelineCache,
    ) -> Result<GraphicsPipeline, ash::vk::Result> {
        let mut stage_create_infos: smallvec::SmallVec<
            [ash::vk::PipelineShaderStageCreateInfo; 4],
        > = smallvec::smallvec![];
        for shader in &self.shader_stages {
            stage_create_infos.push(ash::vk::PipelineShaderStageCreateInfo {
                stage: shader.stage,
                module: shader.module,
                p_name: DEFAULT_SHADER_ENTRY_POINT.as_ptr(),
                ..Default::default()
            });
        }
        let graphics_pipeline_create_info = ash::vk::GraphicsPipelineCreateInfo {
            stage_count: stage_create_infos.len() as u32,
            p_stages: stage_create_infos.as_ptr(),
            render_pass: *self
                .render_pass
                .expect("No RenderPass specified for GraphicsPipeline, this is wrong"),
            ..Default::default()
        };
        let create_infos = [graphics_pipeline_create_info];
        // match unsafe {
        //     device
        //         .device
        //         .create_graphics_pipelines(pipeline_cache.cache, &create_infos, None)
        // } {
        //     Ok(pipelines) => Ok(GraphicsPipeline {
        //         pipeline: pipelines[0],
        //     }),
        //     Err((_, r)) => Err(r),
        // }
        Ok(GraphicsPipeline {
            pipeline: ash::vk::Pipeline::null(),
            device: Some(device.clone()),
        })
    }
}

const VS_COLOR: &[u8] = std::include_bytes!("shaders/color.vert.spv");
const FS_COLOR: &[u8] = std::include_bytes!("shaders/color.frag.spv");

#[repr(C)]
struct TriangleVertex {
    pos: [f32; 3],
}

const TRIANGLE_VERTICES: [TriangleVertex; 3] = [
    TriangleVertex {
        pos: [-0.5, -0.5, 0.0],
    },
    TriangleVertex {
        pos: [0.0, 0.5, 0.0],
    },
    TriangleVertex {
        pos: [0.5, -0.5, 0.0],
    },
];

pub struct Scene {
    green: f32,
    triangle_ready: bool,
    triangle_vbuf: ash::vk::Buffer,
    triangle_vbuf_alloc: vk_mem::Allocation,
    color_pipeline: Option<GraphicsPipeline>,
    device: Option<Rc<Device>>,
    allocator: Option<Rc<MemAllocator>>,
}

impl Scene {
    pub fn new(device: &Rc<Device>, allocator: &Rc<MemAllocator>) -> Self {
        Scene {
            green: 0.0,
            triangle_ready: false,
            triangle_vbuf: ash::vk::Buffer::null(),
            triangle_vbuf_alloc: vk_mem::Allocation::null(),
            color_pipeline: None,
            device: Some(device.clone()),
            allocator: Some(allocator.clone()),
        }
    }

    pub fn release_resources(&mut self) {
        self.color_pipeline = None;
        if self.allocator.is_some() {
            let allocator = self.allocator.as_ref().unwrap();
            allocator.destroy_buffer(self.triangle_vbuf, &self.triangle_vbuf_alloc);
            self.allocator = None;
        }
        self.device = None;
    }

    pub fn sync(&self) -> bool {
        true
    }

    pub fn prepare(
        &mut self,
        swapchain: &Swapchain,
        swapchain_render_pass: &SwapchainRenderPass,
        command_list: &CommandList,
        pipeline_cache: &PipelineCache,
    ) {
        let device = self.device.as_ref().unwrap();
        let allocator = self.allocator.as_ref().unwrap();
        if !self.triangle_ready {
            let (buf, alloc) = allocator
                .create_host_visible_buffer(256, ash::vk::BufferUsageFlags::VERTEX_BUFFER)
                .unwrap();
            let copy_len = TRIANGLE_VERTICES.len() * std::mem::size_of::<TriangleVertex>();
            match allocator.allocator.map_memory(&alloc) {
                Ok(p) => {
                    unsafe {
                        p.copy_from_nonoverlapping(
                            TRIANGLE_VERTICES.as_ptr() as *const u8,
                            copy_len,
                        )
                    };
                }
                Err(r) => panic!(r),
            }
            allocator.allocator.unmap_memory(&alloc).unwrap();
            allocator
                .allocator
                .flush_allocation(&alloc, 0, copy_len)
                .unwrap();
            self.triangle_vbuf = buf;
            self.triangle_vbuf_alloc = alloc;

            let color_vs = Shader::new(VS_COLOR, ash::vk::ShaderStageFlags::VERTEX, device);
            let color_fs = Shader::new(VS_COLOR, ash::vk::ShaderStageFlags::FRAGMENT, device);
            let color_shaders = [&color_vs, &color_fs];
            let mut pipeline_builder = GraphicsPipelineBuilder::new();
            self.color_pipeline = Some(
                pipeline_builder
                    .with_shader_stages(&color_shaders)
                    .with_render_pass(&swapchain_render_pass.render_pass)
                    .build(device, pipeline_cache)
                    .expect("Failed to build simple graphics pipeline"),
            );

            self.triangle_ready = true;
        }
    }

    pub fn render(
        &mut self,
        swapchain: &Swapchain,
        swapchain_framebuffers: &SwapchainFramebuffers,
        swapchain_render_pass: &SwapchainRenderPass,
        swapchain_frame_state: &mut SwapchainFrameState,
        command_list: &CommandList,
    ) {
        let device = self.device.as_ref().unwrap();
        swapchain_frame_state.begin_render_pass(
            device,
            swapchain,
            swapchain_framebuffers,
            swapchain_render_pass,
            command_list,
            [0.0, self.green, 0.0, 1.0],
        );
        swapchain_frame_state.end_render_pass(device, command_list);
        self.green += 0.01;
        if self.green > 1.0 {
            self.green = 0.0;
        }
    }
}

impl Drop for Scene {
    fn drop(&mut self) {
        self.release_resources();
    }
}

pub struct App {
    window: winit::window::Window,
    instance: Rc<Instance>,
    surface: WindowSurface,
    physical_device: PhysicalDevice,
    device: Rc<Device>,
    command_pool: CommandPool,
    command_list: CommandList,
    swapchain: Swapchain,
    swapchain_images: SwapchainImages,
    swapchain_render_pass: SwapchainRenderPass,
    swapchain_framebuffers: SwapchainFramebuffers,
    swapchain_frame_state: SwapchainFrameState,
    swapchain_resizer: SwapchainResizer,
    allocator: Rc<MemAllocator>,
    pipeline_cache: PipelineCache,
    scene: Scene,
}

impl App {
    pub fn new(event_loop: &winit::event_loop::EventLoop<()>) -> Self {
        let window = winit::window::WindowBuilder::new()
            .with_inner_size(winit::dpi::LogicalSize::new(1280, 720))
            .build(event_loop)
            .expect("Failed to create window");
        let instance = Rc::new(Instance::new(&window, ENABLE_VALIDATION));
        let surface = WindowSurface::new(&instance, &window);
        let physical_device = PhysicalDevice::new(&instance, &surface.surface);
        let device = Rc::new(Device::new(&instance, &physical_device));
        let command_pool = CommandPool::new(&physical_device, &device);
        let command_list = CommandList::new(&device, &command_pool);
        let swapchain = SwapchainBuilder::new()
            .build(&instance, &physical_device, &device, &surface, &window)
            .unwrap();
        let swapchain_images = SwapchainImages::new(&device, &swapchain);
        let swapchain_render_pass = SwapchainRenderPass::new(&device, &swapchain);
        let swapchain_framebuffers = SwapchainFramebuffers::new(
            &device,
            &swapchain,
            &swapchain_images,
            &swapchain_render_pass,
        );
        let swapchain_frame_state = SwapchainFrameState::new(&device);
        let swapchain_resizer = SwapchainResizer::new();
        let allocator = Rc::new(MemAllocator::new(&instance, &physical_device, &device));
        let pipeline_cache = PipelineCache::new(&device);
        let scene = Scene::new(&device, &allocator);
        App {
            window,
            instance,
            surface,
            physical_device,
            device,
            command_pool,
            command_list,
            swapchain,
            swapchain_images,
            swapchain_render_pass,
            swapchain_framebuffers,
            swapchain_frame_state,
            swapchain_resizer,
            allocator,
            pipeline_cache,
            scene,
        }
    }

    fn release_resources(&mut self) {
        // to be called on window close, the order matters, don't rely on Drop here
        self.device.wait_idle();
        self.scene.release_resources();
        self.swapchain_frame_state.release_resources();
        self.swapchain_framebuffers.release_resources();
        self.swapchain_render_pass.release_resources();
        self.swapchain_images.release_resources();
        self.swapchain.release_resources();
        self.command_pool.release_resources();
        self.pipeline_cache.release_resources();
        Rc::get_mut(&mut self.allocator)
            .unwrap()
            .release_resources();
        Rc::get_mut(&mut self.device).unwrap().release_resources();
        self.surface.release_resources();
        Rc::get_mut(&mut self.instance).unwrap().release_resources();
    }

    fn ensure_swapchain(&mut self) -> bool {
        return self.swapchain_resizer.ensure_up_to_date(
            &self.instance,
            &self.physical_device,
            &self.surface,
            &self.window,
            &self.device,
            &mut self.swapchain,
            &self.swapchain_render_pass,
            &mut self.swapchain_images,
            &mut self.swapchain_framebuffers,
            &mut self.swapchain_frame_state,
        );
    }

    pub fn run(mut self, event_loop: winit::event_loop::EventLoop<()>) {
        let mut running = true;
        let mut frame_time = std::time::Instant::now();
        event_loop.run(move |event, _, control_flow| {
            *control_flow = winit::event_loop::ControlFlow::Poll;
            match event {
                winit::event::Event::WindowEvent { window_id, event }
                    if window_id == self.window.id() =>
                {
                    match event {
                        winit::event::WindowEvent::CloseRequested => {
                            *control_flow = winit::event_loop::ControlFlow::Exit;
                            running = false;
                            self.release_resources();
                        }
                        _ => (),
                    }
                }
                winit::event::Event::MainEventsCleared => {
                    if self.scene.sync() {
                        self.window.request_redraw();
                    }
                }
                winit::event::Event::RedrawRequested(window_id)
                    if window_id == self.window.id() && running =>
                {
                    if self.ensure_swapchain() {
                        match self.swapchain_frame_state.begin_frame(
                            &self.device,
                            &self.swapchain,
                            &self.command_pool,
                            &self.command_list,
                        ) {
                            Ok(current_frame_slot) => {
                                println!(
                                    "Render, elapsed since last {:?} (slot {})",
                                    frame_time.elapsed(),
                                    current_frame_slot
                                );
                                frame_time = std::time::Instant::now();
                                self.scene.prepare(
                                    &self.swapchain,
                                    &self.swapchain_render_pass,
                                    &self.command_list,
                                    &self.pipeline_cache,
                                );
                                self.scene.render(
                                    &self.swapchain,
                                    &self.swapchain_framebuffers,
                                    &self.swapchain_render_pass,
                                    &mut self.swapchain_frame_state,
                                    &self.command_list,
                                );
                                self.swapchain_frame_state.end_frame(
                                    &self.device,
                                    &self.swapchain,
                                    &self.swapchain_images,
                                    &self.command_list,
                                );
                            }
                            Err(r) => {
                                if r != ash::vk::Result::ERROR_OUT_OF_DATE_KHR {
                                    panic!(r)
                                }
                            }
                        }
                    }
                }
                _ => (),
            };
        });
    }
}

fn main() {
    let event_loop = winit::event_loop::EventLoop::new();
    let app = App::new(&event_loop);
    app.run(event_loop);
}
