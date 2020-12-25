use crate::fw::*;
use ash::version::DeviceV1_0;
use nalgebra_glm as glm;
use std::rc::Rc;

pub struct MaterialPipeline {
    pub desc_set_layout: DescriptorSetLayout,
    pub pipeline_layout: PipelineLayout,
    pub pipeline: GraphicsPipeline,
}

const COLOR_MATERIAL_VS: &[u8] = std::include_bytes!("shaders/color.vert.spv");
const COLOR_MATERIAL_FS: &[u8] = std::include_bytes!("shaders/color.frag.spv");
const COLOR_MATERIAL_UBUF_SIZE: usize = 68;

fn new_color_material_pipeline(
    device: &Rc<Device>,
    pipeline_cache: &PipelineCache,
    render_pass: &ash::vk::RenderPass,
) -> MaterialPipeline {
    let desc_set_layout = DescriptorSetLayout::new(
        device,
        &[ash::vk::DescriptorSetLayoutBinding {
            binding: 0,
            descriptor_type: ash::vk::DescriptorType::UNIFORM_BUFFER,
            descriptor_count: 1,
            stage_flags: ash::vk::ShaderStageFlags::VERTEX | ash::vk::ShaderStageFlags::FRAGMENT,
            ..Default::default()
        }],
    );
    let pipeline_layout = PipelineLayout::new(device, &[&desc_set_layout], &[]);

    let vs = Shader::new(device, COLOR_MATERIAL_VS, ash::vk::ShaderStageFlags::VERTEX);
    let fs = Shader::new(
        device,
        COLOR_MATERIAL_FS,
        ash::vk::ShaderStageFlags::FRAGMENT,
    );

    let vertex_bindings = [ash::vk::VertexInputBindingDescription {
        binding: 0,
        stride: 6 * std::mem::size_of::<f32>() as u32,
        ..Default::default()
    }];
    let vertex_attributes = [
        ash::vk::VertexInputAttributeDescription {
            location: 0,
            binding: 0,
            format: ash::vk::Format::R32G32B32_SFLOAT,
            offset: 0,
        },
        ash::vk::VertexInputAttributeDescription {
            location: 1,
            binding: 0,
            format: ash::vk::Format::R32G32B32_SFLOAT,
            offset: 3 * std::mem::size_of::<f32>() as u32,
        },
    ];

    let pipeline = GraphicsPipelineBuilder::new()
        .with_shader_stages(&[&vs, &fs])
        .with_layout(&pipeline_layout)
        .with_render_pass(render_pass)
        .with_vertex_input_layout(&vertex_bindings, &vertex_attributes)
        .with_cull_mode(ash::vk::CullModeFlags::NONE)
        .with_depth_stencil_state(ash::vk::PipelineDepthStencilStateCreateInfo {
            depth_test_enable: ash::vk::TRUE,
            depth_write_enable: ash::vk::TRUE,
            depth_compare_op: ash::vk::CompareOp::LESS,
            ..Default::default()
        })
        .with_blend_state(ash::vk::PipelineColorBlendAttachmentState {
            blend_enable: ash::vk::TRUE,
            src_color_blend_factor: ash::vk::BlendFactor::SRC_ALPHA,
            dst_color_blend_factor: ash::vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
            color_blend_op: ash::vk::BlendOp::ADD,
            src_alpha_blend_factor: ash::vk::BlendFactor::SRC_ALPHA,
            dst_alpha_blend_factor: ash::vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
            alpha_blend_op: ash::vk::BlendOp::ADD,
            color_write_mask: rgba_color_write_mask(),
            ..Default::default()
        })
        .build(device, pipeline_cache)
        .expect("Failed to build simple graphics pipeline");

    MaterialPipeline {
        desc_set_layout,
        pipeline_layout,
        pipeline,
    }
}

const TEXTURE_MATERIAL_VS: &[u8] = std::include_bytes!("shaders/texture.vert.spv");
const TEXTURE_MATERIAL_FS: &[u8] = std::include_bytes!("shaders/texture.frag.spv");
const TEXTURE_MATERIAL_UBUF_SIZE: usize = 64;

fn new_texture_material_pipeline(
    device: &Rc<Device>,
    pipeline_cache: &PipelineCache,
    render_pass: &ash::vk::RenderPass,
) -> MaterialPipeline {
    let desc_set_layout = DescriptorSetLayout::new(
        device,
        &[
            ash::vk::DescriptorSetLayoutBinding {
                binding: 0,
                descriptor_type: ash::vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: 1,
                stage_flags: ash::vk::ShaderStageFlags::VERTEX
                    | ash::vk::ShaderStageFlags::FRAGMENT,
                ..Default::default()
            },
            ash::vk::DescriptorSetLayoutBinding {
                binding: 1,
                descriptor_type: ash::vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: 1,
                stage_flags: ash::vk::ShaderStageFlags::FRAGMENT,
                ..Default::default()
            },
        ],
    );
    let pipeline_layout = PipelineLayout::new(device, &[&desc_set_layout], &[]);

    let vs = Shader::new(
        device,
        TEXTURE_MATERIAL_VS,
        ash::vk::ShaderStageFlags::VERTEX,
    );
    let fs = Shader::new(
        device,
        TEXTURE_MATERIAL_FS,
        ash::vk::ShaderStageFlags::FRAGMENT,
    );

    let vertex_bindings = [ash::vk::VertexInputBindingDescription {
        binding: 0,
        stride: 5 * std::mem::size_of::<f32>() as u32,
        ..Default::default()
    }];
    let vertex_attributes = [
        ash::vk::VertexInputAttributeDescription {
            location: 0,
            binding: 0,
            format: ash::vk::Format::R32G32B32_SFLOAT,
            offset: 0,
        },
        ash::vk::VertexInputAttributeDescription {
            location: 1,
            binding: 0,
            format: ash::vk::Format::R32G32_SFLOAT,
            offset: 3 * std::mem::size_of::<f32>() as u32,
        },
    ];

    let pipeline = GraphicsPipelineBuilder::new()
        .with_shader_stages(&[&vs, &fs])
        .with_layout(&pipeline_layout)
        .with_render_pass(render_pass)
        .with_vertex_input_layout(&vertex_bindings, &vertex_attributes)
        .with_cull_mode(ash::vk::CullModeFlags::NONE)
        .with_depth_stencil_state(ash::vk::PipelineDepthStencilStateCreateInfo {
            depth_test_enable: ash::vk::TRUE,
            depth_write_enable: ash::vk::TRUE,
            depth_compare_op: ash::vk::CompareOp::LESS,
            ..Default::default()
        })
        .build(device, pipeline_cache)
        .expect("Failed to build simple graphics pipeline");

    MaterialPipeline {
        desc_set_layout,
        pipeline_layout,
        pipeline,
    }
}

#[repr(C)]
struct TriangleVertex {
    pos: [f32; 3],
    color: [f32; 3],
}

const TRIANGLE_VERTICES: [TriangleVertex; 3] = [
    // Y up, front=CCW
    TriangleVertex {
        pos: [0.0, 0.5, 0.0],
        color: [1.0, 0.0, 0.0],
    },
    TriangleVertex {
        pos: [-0.5, -0.5, 0.0],
        color: [0.0, 1.0, 0.0],
    },
    TriangleVertex {
        pos: [0.5, -0.5, 0.0],
        color: [0.0, 0.0, 1.0],
    },
];

fn create_triangle_vertex_buffer(
    device: &Device,
    allocator: &MemAllocator,
    swapchain_frame_state: &mut SwapchainFrameState,
    cb: &ash::vk::CommandBuffer,
) -> (ash::vk::Buffer, vk_mem::Allocation) {
    let size = TRIANGLE_VERTICES.len() * std::mem::size_of::<TriangleVertex>();
    let data = TRIANGLE_VERTICES.as_ptr() as *const u8;
    let data_offset_size = [(data, 0, size)];
    create_or_reuse_vertexindex_buffer_with_data(
        device,
        allocator,
        swapchain_frame_state,
        cb,
        VertexIndexBufferType::Vertex,
        size,
        &data_offset_size,
        None,
    )
}

#[repr(C)]
struct TexturedQuadVertex {
    pos: [f32; 3],
    uv: [f32; 2],
}

const TEXTURED_QUAD_VERTICES: [TexturedQuadVertex; 4] = [
    // Y up, V up, front=CCW
    TexturedQuadVertex {
        pos: [-1.0, -1.0, 0.0],
        uv: [0.0, 0.0],
    },
    TexturedQuadVertex {
        pos: [-1.0, 1.0, 0.0],
        uv: [0.0, 1.0],
    },
    TexturedQuadVertex {
        pos: [1.0, 1.0, 0.0],
        uv: [1.0, 1.0],
    },
    TexturedQuadVertex {
        pos: [1.0, -1.0, 0.0],
        uv: [1.0, 0.0],
    },
];

const TEXTURED_QUAD_INDICES: [u16; 6] = [0, 1, 2, 0, 2, 3];

fn create_textured_quad_vertex_buffer(
    device: &Device,
    allocator: &MemAllocator,
    swapchain_frame_state: &mut SwapchainFrameState,
    cb: &ash::vk::CommandBuffer,
) -> (ash::vk::Buffer, vk_mem::Allocation) {
    let size = TEXTURED_QUAD_VERTICES.len() * std::mem::size_of::<TexturedQuadVertex>();
    let data = TEXTURED_QUAD_VERTICES.as_ptr() as *const u8;
    let data_offset_size = [(data, 0, size)];
    create_or_reuse_vertexindex_buffer_with_data(
        device,
        allocator,
        swapchain_frame_state,
        cb,
        VertexIndexBufferType::Vertex,
        size,
        &data_offset_size,
        None,
    )
}

fn create_textured_quad_index_buffer(
    device: &Device,
    allocator: &MemAllocator,
    swapchain_frame_state: &mut SwapchainFrameState,
    cb: &ash::vk::CommandBuffer,
) -> (ash::vk::Buffer, vk_mem::Allocation) {
    let size = TEXTURED_QUAD_INDICES.len() * std::mem::size_of::<u16>();
    let data = TEXTURED_QUAD_INDICES.as_ptr() as *const u8;
    let data_offset_size = [(data, 0, size)];
    create_or_reuse_vertexindex_buffer_with_data(
        device,
        allocator,
        swapchain_frame_state,
        cb,
        VertexIndexBufferType::Index,
        size,
        &data_offset_size,
        None,
    )
}

const IMAGE: &[u8] = std::include_bytes!("../data/something.png");

fn create_something_texture(
    device: &Device,
    allocator: &MemAllocator,
    swapchain_frame_state: &mut SwapchainFrameState,
    cb: &ash::vk::CommandBuffer,
) -> (ash::vk::Image, vk_mem::Allocation) {
    let image_data = image::load_from_memory_with_format(IMAGE, image::ImageFormat::Png)
        .expect("Failed to load image")
        .flipv();
    let rgba8_image_data = match image_data {
        image::DynamicImage::ImageRgb8(_)
        | image::DynamicImage::ImageRgba8(_)
        | image::DynamicImage::ImageBgr8(_)
        | image::DynamicImage::ImageBgra8(_) => image_data.into_rgba8(),
        _ => panic!("Unsupported image format"),
    };
    let pixels: &Vec<u8> = rgba8_image_data.as_raw();
    create_base_rgba_2d_texture_for_sampling(
        device,
        allocator,
        swapchain_frame_state,
        cb,
        ash::vk::Extent2D {
            width: rgba8_image_data.width(),
            height: rgba8_image_data.height(),
        },
        pixels,
        ash::vk::PipelineStageFlags::FRAGMENT_SHADER,
    )
}

pub struct Scene {
    ready: bool,
    device: Option<Rc<Device>>,
    allocator: Option<Rc<MemAllocator>>,
    descriptor_pool: Option<DescriptorPool>,
    output_pixel_size: ash::vk::Extent2D,
    projection: glm::Mat4,

    color_material_pipeline: Option<MaterialPipeline>,
    texture_material_pipeline: Option<MaterialPipeline>,

    triangle_vbuf: (ash::vk::Buffer, vk_mem::Allocation),
    quad_vbuf: (ash::vk::Buffer, vk_mem::Allocation),
    quad_ibuf: (ash::vk::Buffer, vk_mem::Allocation),
    some_texture: (ash::vk::Image, vk_mem::Allocation),
    some_texture_view: Option<ImageView>,
    linear_sampler: Option<Sampler>,

    triangle_ubufs: [(ash::vk::Buffer, vk_mem::Allocation); FRAMES_IN_FLIGHT as usize],
    triangle_desc_sets: Vec<ash::vk::DescriptorSet>,
    triangle_view_matrix: glm::Mat4,
    triangle_rotation: f32,

    textured_quad_ubufs: [(ash::vk::Buffer, vk_mem::Allocation); FRAMES_IN_FLIGHT as usize],
    textured_quad_desc_sets: Vec<ash::vk::DescriptorSet>,
    textured_quad_view_matrix: glm::Mat4,
}

impl Scene {
    pub fn new(device: &Rc<Device>, allocator: &Rc<MemAllocator>) -> Self {
        let null_buf_and_alloc = (ash::vk::Buffer::null(), vk_mem::Allocation::null());
        let null_image_and_alloc = (ash::vk::Image::null(), vk_mem::Allocation::null());
        Scene {
            ready: false,
            device: Some(Rc::clone(device)),
            allocator: Some(Rc::clone(allocator)),
            descriptor_pool: None,
            output_pixel_size: ash::vk::Extent2D {
                width: 0,
                height: 0,
            },
            projection: glm::identity(),

            color_material_pipeline: None,
            texture_material_pipeline: None,

            triangle_vbuf: null_buf_and_alloc,
            quad_vbuf: null_buf_and_alloc,
            quad_ibuf: null_buf_and_alloc,
            some_texture: null_image_and_alloc,
            some_texture_view: None,
            linear_sampler: None,

            triangle_ubufs: [null_buf_and_alloc; FRAMES_IN_FLIGHT as usize],
            triangle_desc_sets: vec![],
            triangle_view_matrix: glm::identity(),
            triangle_rotation: 0.0,

            textured_quad_ubufs: [null_buf_and_alloc; FRAMES_IN_FLIGHT as usize],
            textured_quad_desc_sets: vec![],
            textured_quad_view_matrix: glm::identity(),
        }
    }

    pub fn release_resources(&mut self) {
        // Objects holding a ref to the Device or Allocator must be dropped
        // here, before this function returns.
        self.some_texture_view = None;
        self.linear_sampler = None;
        self.color_material_pipeline = None;
        self.texture_material_pipeline = None;
        self.descriptor_pool = None;

        if self.allocator.is_some() {
            let allocator = self.allocator.as_ref().unwrap();
            for buf_and_alloc in &self.triangle_ubufs {
                allocator.destroy_buffer(buf_and_alloc);
            }
            for buf_and_alloc in &self.textured_quad_ubufs {
                allocator.destroy_buffer(buf_and_alloc);
            }
            allocator.destroy_buffer(&self.triangle_vbuf);
            allocator.destroy_buffer(&self.quad_vbuf);
            allocator.destroy_buffer(&self.quad_ibuf);
            allocator.destroy_image(&self.some_texture);
            self.allocator = None;
        }
        self.device = None;
    }

    pub fn sync(&mut self) -> bool {
        self.triangle_rotation += 1.0;
        true
    }

    pub fn prepare(
        &mut self,
        swapchain: &Swapchain,
        swapchain_render_pass: &SwapchainRenderPass,
        swapchain_frame_state: &mut SwapchainFrameState,
        command_list: &CommandList,
        pipeline_cache: &PipelineCache,
    ) {
        let device = self.device.as_ref().unwrap();
        let allocator = self.allocator.as_ref().unwrap();
        let current_frame_slot = swapchain_frame_state.current_frame_slot;
        let cb = swapchain_frame_state.current_frame_command_buffer(command_list);

        if self.output_pixel_size != swapchain.pixel_size {
            self.output_pixel_size = swapchain.pixel_size;
            self.projection = glm::perspective_fov_zo(
                45.0f32.to_radians(),
                self.output_pixel_size.width as f32,
                self.output_pixel_size.height as f32,
                0.01,
                1000.0,
            );
            self.projection[5] *= -1.0; // vertex data is Y up
        }

        if !self.ready {
            self.color_material_pipeline = Some(new_color_material_pipeline(
                device,
                pipeline_cache,
                &swapchain_render_pass.render_pass,
            ));
            self.texture_material_pipeline = Some(new_texture_material_pipeline(
                device,
                pipeline_cache,
                &swapchain_render_pass.render_pass,
            ));

            const MAX_DESC_SETS: u32 = 128;
            self.descriptor_pool = Some(DescriptorPool::new(
                device,
                MAX_DESC_SETS,
                &[
                    ash::vk::DescriptorPoolSize {
                        ty: ash::vk::DescriptorType::UNIFORM_BUFFER,
                        descriptor_count: MAX_DESC_SETS,
                    },
                    ash::vk::DescriptorPoolSize {
                        ty: ash::vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                        descriptor_count: MAX_DESC_SETS,
                    },
                ],
            ));

            self.triangle_vbuf =
                create_triangle_vertex_buffer(device, &allocator, swapchain_frame_state, cb);

            self.quad_vbuf =
                create_textured_quad_vertex_buffer(device, &allocator, swapchain_frame_state, cb);
            self.quad_ibuf =
                create_textured_quad_index_buffer(device, &allocator, swapchain_frame_state, cb);

            self.some_texture =
                create_something_texture(device, &allocator, swapchain_frame_state, cb);

            self.some_texture_view = Some(ImageView::new(
                device,
                &ash::vk::ImageViewCreateInfo {
                    image: self.some_texture.0,
                    view_type: ash::vk::ImageViewType::TYPE_2D,
                    format: ash::vk::Format::R8G8B8A8_UNORM,
                    components: identity_component_mapping(),
                    subresource_range: base_level_subres_range(ash::vk::ImageAspectFlags::COLOR),
                    ..Default::default()
                },
            ));

            self.linear_sampler = Some(Sampler::new(
                device,
                &ash::vk::SamplerCreateInfo {
                    mag_filter: ash::vk::Filter::LINEAR,
                    min_filter: ash::vk::Filter::LINEAR,
                    address_mode_u: ash::vk::SamplerAddressMode::CLAMP_TO_EDGE,
                    address_mode_v: ash::vk::SamplerAddressMode::CLAMP_TO_EDGE,
                    max_lod: 0.25,
                    ..Default::default()
                },
            ));

            self.triangle_view_matrix =
                glm::translate(&glm::identity(), &glm::vec3(0.0, 0.0, -4.0));

            for frame_slot in 0..FRAMES_IN_FLIGHT {
                self.triangle_ubufs[frame_slot as usize] = allocator
                    .create_host_visible_buffer(
                        COLOR_MATERIAL_UBUF_SIZE,
                        ash::vk::BufferUsageFlags::UNIFORM_BUFFER,
                    )
                    .unwrap();
                let opacity = [0.5f32];
                allocator.update_host_visible_buffer(
                    &self.triangle_ubufs[frame_slot as usize].1,
                    64,
                    4,
                    0,
                    &[(opacity.as_ptr() as *const u8, 64, 4)],
                );
            }

            self.triangle_desc_sets = self
                .descriptor_pool
                .as_ref()
                .unwrap()
                .allocate(
                    &[&self
                        .color_material_pipeline
                        .as_ref()
                        .unwrap()
                        .desc_set_layout; FRAMES_IN_FLIGHT as usize],
                )
                .expect("Failed to allocate descriptor sets for triangle");
            let mut desc_buffer_infos: smallvec::SmallVec<[ash::vk::DescriptorBufferInfo; 4]> =
                smallvec::smallvec![];
            for frame_slot in 0..FRAMES_IN_FLIGHT {
                desc_buffer_infos.push(ash::vk::DescriptorBufferInfo {
                    buffer: self.triangle_ubufs[frame_slot as usize].0,
                    offset: 0,
                    range: COLOR_MATERIAL_UBUF_SIZE as u64,
                });
            }
            let mut desc_writes: smallvec::SmallVec<[ash::vk::WriteDescriptorSet; 4]> =
                smallvec::smallvec![];
            for frame_slot in 0..FRAMES_IN_FLIGHT {
                desc_writes.push(ash::vk::WriteDescriptorSet {
                    dst_set: self.triangle_desc_sets[frame_slot as usize],
                    dst_binding: 0,
                    descriptor_count: 1,
                    descriptor_type: ash::vk::DescriptorType::UNIFORM_BUFFER,
                    p_buffer_info: &desc_buffer_infos[frame_slot as usize],
                    ..Default::default()
                });
            }
            unsafe {
                device.device.update_descriptor_sets(&desc_writes, &[]);
            }

            self.textured_quad_view_matrix =
                glm::translate(&glm::identity(), &glm::vec3(-4.0, 0.0, -8.0));

            for i in 0..FRAMES_IN_FLIGHT {
                self.textured_quad_ubufs[i as usize] = allocator
                    .create_host_visible_buffer(
                        TEXTURE_MATERIAL_UBUF_SIZE,
                        ash::vk::BufferUsageFlags::UNIFORM_BUFFER,
                    )
                    .unwrap();
            }

            self.textured_quad_desc_sets = self
                .descriptor_pool
                .as_ref()
                .unwrap()
                .allocate(
                    &[&self
                        .texture_material_pipeline
                        .as_ref()
                        .unwrap()
                        .desc_set_layout; FRAMES_IN_FLIGHT as usize],
                )
                .expect("Failed to allocate descriptor sets for textured quad");
            let mut desc_buffer_infos: smallvec::SmallVec<[ash::vk::DescriptorBufferInfo; 4]> =
                smallvec::smallvec![];
            for i in 0..FRAMES_IN_FLIGHT {
                desc_buffer_infos.push(ash::vk::DescriptorBufferInfo {
                    buffer: self.textured_quad_ubufs[i as usize].0,
                    offset: 0,
                    range: TEXTURE_MATERIAL_UBUF_SIZE as u64,
                });
            }
            let desc_image_info = ash::vk::DescriptorImageInfo {
                sampler: self.linear_sampler.as_ref().unwrap().sampler,
                image_view: self.some_texture_view.as_ref().unwrap().view,
                image_layout: ash::vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                ..Default::default()
            };
            let mut desc_writes: smallvec::SmallVec<[ash::vk::WriteDescriptorSet; 4]> =
                smallvec::smallvec![];
            for i in 0..FRAMES_IN_FLIGHT {
                desc_writes.push(ash::vk::WriteDescriptorSet {
                    dst_set: self.textured_quad_desc_sets[i as usize],
                    dst_binding: 0,
                    descriptor_count: 1,
                    descriptor_type: ash::vk::DescriptorType::UNIFORM_BUFFER,
                    p_buffer_info: &desc_buffer_infos[i as usize],
                    ..Default::default()
                });
                desc_writes.push(ash::vk::WriteDescriptorSet {
                    dst_set: self.textured_quad_desc_sets[i as usize],
                    dst_binding: 1,
                    descriptor_count: 1,
                    descriptor_type: ash::vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                    p_image_info: &desc_image_info,
                    ..Default::default()
                });
            }
            unsafe {
                device.device.update_descriptor_sets(&desc_writes, &[]);
            }

            self.ready = true;
        }

        let triangle_model_matrix = glm::rotate(
            &glm::identity(),
            self.triangle_rotation.to_radians(),
            &glm::vec3(0.0, 1.0, 0.0),
        );
        let mvp = self.projection * self.triangle_view_matrix * triangle_model_matrix;
        allocator.update_host_visible_buffer(
            &self.triangle_ubufs[current_frame_slot as usize].1,
            0,
            64,
            0,
            &[(mvp.as_ptr() as *const u8, 0, 64)],
        );

        let mvp = self.projection * self.textured_quad_view_matrix;
        allocator.update_host_visible_buffer(
            &self.textured_quad_ubufs[current_frame_slot as usize].1,
            0,
            64,
            0,
            &[(mvp.as_ptr() as *const u8, 0, 64)],
        );
    }

    pub fn begin_main_render_pass(
        &self,
        swapchain: &Swapchain,
        swapchain_framebuffers: &SwapchainFramebuffers,
        swapchain_render_pass: &SwapchainRenderPass,
        swapchain_frame_state: &SwapchainFrameState,
        command_list: &CommandList,
    ) {
        let clear_values = [
            ash::vk::ClearValue {
                color: ash::vk::ClearColorValue {
                    float32: [0.0, 0.5, 0.0, 1.0],
                },
            },
            ash::vk::ClearValue {
                depth_stencil: ash::vk::ClearDepthStencilValue {
                    depth: 1.0,
                    stencil: 0,
                },
            },
        ];
        let pass_begin_info = ash::vk::RenderPassBeginInfo {
            render_pass: swapchain_render_pass.render_pass,
            framebuffer: swapchain_framebuffers.framebuffers
                [swapchain_frame_state.current_image_index as usize],
            render_area: ash::vk::Rect2D {
                offset: ash::vk::Offset2D { x: 0, y: 0 },
                extent: swapchain.pixel_size,
            },
            clear_value_count: clear_values.len() as u32,
            p_clear_values: clear_values.as_ptr(),
            ..Default::default()
        };
        let device = self.device.as_ref().unwrap();
        let cb = swapchain_frame_state.current_frame_command_buffer(command_list);
        unsafe {
            device.device.cmd_begin_render_pass(
                *cb,
                &pass_begin_info,
                ash::vk::SubpassContents::INLINE,
            );
        }
    }

    pub fn end_main_render_pass(
        &self,
        swapchain_frame_state: &SwapchainFrameState,
        command_list: &CommandList,
    ) {
        let device = self.device.as_ref().unwrap();
        let cb = swapchain_frame_state.current_frame_command_buffer(command_list);
        unsafe {
            device.device.cmd_end_render_pass(*cb);
        }
    }

    pub fn render_main_pass(
        &self,
        swapchain_frame_state: &SwapchainFrameState,
        command_list: &CommandList,
    ) {
        let device = self.device.as_ref().unwrap();
        let cb = swapchain_frame_state.current_frame_command_buffer(command_list);
        let current_frame_slot = swapchain_frame_state.current_frame_slot;

        unsafe {
            device.device.cmd_set_viewport(
                *cb,
                0,
                &[ash::vk::Viewport {
                    x: 0.0,
                    y: 0.0,
                    width: self.output_pixel_size.width as f32,
                    height: self.output_pixel_size.height as f32,
                    min_depth: 0.0,
                    max_depth: 1.0,
                }],
            );
            device.device.cmd_set_scissor(
                *cb,
                0,
                &[ash::vk::Rect2D {
                    offset: ash::vk::Offset2D { x: 0, y: 0 },
                    extent: self.output_pixel_size,
                }],
            );

            // triangle
            device.device.cmd_bind_pipeline(
                *cb,
                ash::vk::PipelineBindPoint::GRAPHICS,
                self.color_material_pipeline
                    .as_ref()
                    .unwrap()
                    .pipeline
                    .pipeline,
            );
            device.device.cmd_bind_descriptor_sets(
                *cb,
                ash::vk::PipelineBindPoint::GRAPHICS,
                self.color_material_pipeline
                    .as_ref()
                    .unwrap()
                    .pipeline_layout
                    .layout,
                0,
                &[self.triangle_desc_sets[current_frame_slot as usize]],
                &[],
            );
            device
                .device
                .cmd_bind_vertex_buffers(*cb, 0, &[self.triangle_vbuf.0], &[0]);
            device
                .device
                .cmd_draw(*cb, TRIANGLE_VERTICES.len() as u32, 1, 0, 0);

            // textured quad
            device.device.cmd_bind_pipeline(
                *cb,
                ash::vk::PipelineBindPoint::GRAPHICS,
                self.texture_material_pipeline
                    .as_ref()
                    .unwrap()
                    .pipeline
                    .pipeline,
            );
            device.device.cmd_bind_descriptor_sets(
                *cb,
                ash::vk::PipelineBindPoint::GRAPHICS,
                self.texture_material_pipeline
                    .as_ref()
                    .unwrap()
                    .pipeline_layout
                    .layout,
                0,
                &[self.textured_quad_desc_sets[current_frame_slot as usize]],
                &[],
            );
            device
                .device
                .cmd_bind_vertex_buffers(*cb, 0, &[self.quad_vbuf.0], &[0]);
            device.device.cmd_bind_index_buffer(
                *cb,
                self.quad_ibuf.0,
                0,
                ash::vk::IndexType::UINT16,
            );
            device
                .device
                .cmd_draw_indexed(*cb, TEXTURED_QUAD_INDICES.len() as u32, 1, 0, 0, 0);
        }
    }
}

impl Drop for Scene {
    fn drop(&mut self) {
        self.release_resources();
    }
}
