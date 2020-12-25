use crate::fw::*;
use crate::scene::MaterialPipeline;
use crate::uicontent::UiContent;
use ash::version::DeviceV1_0;
use nalgebra_glm as glm;
use std::rc::Rc;

const IMGUI_MATERIAL_VS: &[u8] = std::include_bytes!("shaders/imgui.vert.spv");
const IMGUI_MATERIAL_FS: &[u8] = std::include_bytes!("shaders/imgui.frag.spv");
const IMGUI_MATERIAL_UBUF_SIZE: usize = 64;

fn new_imgui_material_pipeline(
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

    let vs = Shader::new(device, IMGUI_MATERIAL_VS, ash::vk::ShaderStageFlags::VERTEX);
    let fs = Shader::new(
        device,
        IMGUI_MATERIAL_FS,
        ash::vk::ShaderStageFlags::FRAGMENT,
    );

    let vertex_bindings = [ash::vk::VertexInputBindingDescription {
        binding: 0,
        stride: (4 * std::mem::size_of::<f32>() + 4 * std::mem::size_of::<u8>()) as u32,
        ..Default::default()
    }];
    let vertex_attributes = [
        ash::vk::VertexInputAttributeDescription {
            location: 0,
            binding: 0,
            format: ash::vk::Format::R32G32_SFLOAT,
            offset: 0,
        },
        ash::vk::VertexInputAttributeDescription {
            location: 1,
            binding: 0,
            format: ash::vk::Format::R32G32_SFLOAT,
            offset: 2 * std::mem::size_of::<f32>() as u32,
        },
        ash::vk::VertexInputAttributeDescription {
            location: 2,
            binding: 0,
            format: ash::vk::Format::R8G8B8A8_UNORM,
            offset: 4 * std::mem::size_of::<f32>() as u32,
        },
    ];

    let pipeline = GraphicsPipelineBuilder::new()
        .with_shader_stages(&[&vs, &fs])
        .with_layout(&pipeline_layout)
        .with_render_pass(render_pass)
        .with_vertex_input_layout(&vertex_bindings, &vertex_attributes)
        .with_cull_mode(ash::vk::CullModeFlags::NONE)
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

fn prepare_imgui_font_texture(
    device: &Device,
    allocator: &MemAllocator,
    swapchain_frame_state: &mut SwapchainFrameState,
    cb: &ash::vk::CommandBuffer,
    ctx: &mut imgui::Context,
    texture_list: &mut imgui::Textures<(ash::vk::Image, vk_mem::Allocation)>,
) -> (ash::vk::Image, vk_mem::Allocation) {
    let mut fonts = ctx.fonts();
    let tex_data = fonts.build_rgba32_texture();
    let texture = create_base_rgba_2d_texture_for_sampling(
        device,
        allocator,
        swapchain_frame_state,
        cb,
        ash::vk::Extent2D {
            width: tex_data.width,
            height: tex_data.height,
        },
        tex_data.data,
        ash::vk::PipelineStageFlags::FRAGMENT_SHADER,
    );
    fonts.tex_id = texture_list.insert(texture);
    fonts.clear_tex_data();
    texture
}

#[derive(Debug)]
struct ImGuiDrawCommand {
    texture_id: imgui::TextureId,
    scissor_offset: ash::vk::Offset2D,
    scissor_extent: ash::vk::Extent2D,
    base_vertex: i32,
    first_index: u32,
    index_count: u32,
}

pub struct ImGui {
    pub active: bool,
    pub winit_support: imgui_winit_support::WinitPlatform,
    pub ctx: imgui::Context,
    device: Option<Rc<Device>>,
    allocator: Option<Rc<MemAllocator>>,
    descriptor_pool: Option<DescriptorPool>,
    material_pipeline: Option<MaterialPipeline>,
    font_texture: (ash::vk::Image, vk_mem::Allocation),
    font_texture_view: Option<ImageView>,
    sampler: Option<Sampler>,
    texture_list: imgui::Textures<(ash::vk::Image, vk_mem::Allocation)>,
    vbuf: (ash::vk::Buffer, vk_mem::Allocation, usize),
    ibuf: (ash::vk::Buffer, vk_mem::Allocation, usize),
    ubufs: [(ash::vk::Buffer, vk_mem::Allocation); FRAMES_IN_FLIGHT as usize],
    desc_sets: Vec<ash::vk::DescriptorSet>,
    draw_commands: smallvec::SmallVec<[ImGuiDrawCommand; 16]>,
    last_display_size: [f32; 2],
    projection: glm::Mat4,
    ui_content: UiContent,
}

impl ImGui {
    pub fn new(
        device: &Rc<Device>,
        allocator: &Rc<MemAllocator>,
        window: &winit::window::Window,
    ) -> Self {
        let mut ctx = imgui::Context::create();
        ctx.set_ini_filename(None);
        assert!(std::mem::size_of::<imgui::DrawIdx>() == 2);
        let mut winit_support = imgui_winit_support::WinitPlatform::init(&mut ctx);
        winit_support.attach_window(
            ctx.io_mut(),
            window,
            imgui_winit_support::HiDpiMode::Default,
        );
        ImGui {
            active: false,
            winit_support,
            ctx,
            device: Some(Rc::clone(device)),
            allocator: Some(Rc::clone(allocator)),
            descriptor_pool: None,
            material_pipeline: None,
            font_texture: (ash::vk::Image::null(), vk_mem::Allocation::null()),
            font_texture_view: None,
            sampler: None,
            texture_list: imgui::Textures::new(),
            vbuf: (ash::vk::Buffer::null(), vk_mem::Allocation::null(), 0),
            ibuf: (ash::vk::Buffer::null(), vk_mem::Allocation::null(), 0),
            ubufs: [(ash::vk::Buffer::null(), vk_mem::Allocation::null());
                FRAMES_IN_FLIGHT as usize],
            desc_sets: vec![],
            draw_commands: smallvec::smallvec![],
            last_display_size: [0.0, 0.0],
            projection: glm::identity(),
            ui_content: UiContent::new(),
        }
    }

    pub fn release_resources(&mut self) {
        if self.device.is_none() {
            return;
        }
        let allocator = self.allocator.as_ref().unwrap();
        self.sampler = None;
        self.font_texture_view = None;
        self.material_pipeline = None;
        self.descriptor_pool = None;
        allocator.destroy_image(&self.font_texture);
        allocator.destroy_buffer(&(self.vbuf.0, self.vbuf.1));
        allocator.destroy_buffer(&(self.ibuf.0, self.ibuf.1));
        for buf_and_alloc in &self.ubufs {
            allocator.destroy_buffer(buf_and_alloc);
        }
        self.allocator = None;
        self.device = None;
    }

    pub fn prepare(
        &mut self,
        swapchain_frame_state: &mut SwapchainFrameState,
        command_list: &CommandList,
        pipeline_cache: &PipelineCache,
        render_pass: &ash::vk::RenderPass,
        window: &winit::window::Window,
    ) {
        if !self.active {
            return;
        }
        let device = self.device.as_ref().unwrap();
        let allocator = self.allocator.as_ref().unwrap();
        let cb = swapchain_frame_state.current_frame_command_buffer(command_list);
        let current_frame_slot = swapchain_frame_state.current_frame_slot;
        let scale_factor: [f32; 2];
        self.draw_commands.clear();
        if self.material_pipeline.is_none() {
            self.material_pipeline = Some(new_imgui_material_pipeline(
                device,
                pipeline_cache,
                render_pass,
            ));
        }
        if self.font_texture.0 == ash::vk::Image::null() {
            self.font_texture = prepare_imgui_font_texture(
                device,
                allocator,
                swapchain_frame_state,
                cb,
                &mut self.ctx,
                &mut self.texture_list,
            );
        }
        if self.font_texture_view.is_none() {
            self.font_texture_view = Some(ImageView::new(
                device,
                &ash::vk::ImageViewCreateInfo {
                    image: self.font_texture.0,
                    view_type: ash::vk::ImageViewType::TYPE_2D,
                    format: ash::vk::Format::R8G8B8A8_UNORM,
                    components: identity_component_mapping(),
                    subresource_range: base_level_subres_range(ash::vk::ImageAspectFlags::COLOR),
                    ..Default::default()
                },
            ));
        }
        if self.sampler.is_none() {
            self.sampler = Some(Sampler::new(
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
        }

        let ui = self.ctx.frame();
        self.ui_content.update(&ui);
        self.winit_support.prepare_render(&ui, window);
        let draw_data = ui.render();
        scale_factor = draw_data.framebuffer_scale;
        self.draw_commands.clear();

        let mut vbuf_chunks: smallvec::SmallVec<[(*const u8, usize, usize); 16]> =
            smallvec::smallvec![];
        let mut ibuf_chunks: smallvec::SmallVec<[(*const u8, usize, usize); 16]> =
            smallvec::smallvec![];
        let mut total_vbuf_size: usize = 0;
        let mut total_ibuf_size: usize = 0;
        let mut global_base_vertex: usize = 0;
        let mut global_first_index: usize = 0;

        for draw_list in draw_data.draw_lists() {
            let vertex_count = draw_list.vtx_buffer().len();
            let vbuf_size = vertex_count * std::mem::size_of::<imgui::DrawVert>();
            let index_count = draw_list.idx_buffer().len();
            let ibuf_size = index_count * std::mem::size_of::<imgui::DrawIdx>();
            vbuf_chunks.push((
                draw_list.vtx_buffer().as_ptr() as *const u8,
                total_vbuf_size,
                vbuf_size,
            ));
            ibuf_chunks.push((
                draw_list.idx_buffer().as_ptr() as *const u8,
                total_ibuf_size,
                ibuf_size,
            ));

            for cmd in draw_list.commands() {
                match cmd {
                    imgui::DrawCmd::Elements { count, cmd_params } => {
                        let scissor_offset = ash::vk::Offset2D {
                            x: std::cmp::max(0, (cmd_params.clip_rect[0] * scale_factor[0]) as i32),
                            y: std::cmp::max(0, (cmd_params.clip_rect[1] * scale_factor[1]) as i32),
                        };
                        let scissor_extent = ash::vk::Extent2D {
                            width: ((cmd_params.clip_rect[2] - cmd_params.clip_rect[0])
                                * scale_factor[0]) as u32,
                            height: ((cmd_params.clip_rect[3] - cmd_params.clip_rect[1])
                                * scale_factor[1]) as u32,
                        };
                        self.draw_commands.push(ImGuiDrawCommand {
                            texture_id: cmd_params.texture_id,
                            scissor_offset,
                            scissor_extent,
                            base_vertex: (cmd_params.vtx_offset + global_base_vertex) as i32,
                            first_index: (cmd_params.idx_offset + global_first_index) as u32,
                            index_count: count as u32,
                        });
                    }
                    _ => (),
                }
            }

            total_vbuf_size += vbuf_size;
            total_ibuf_size += ibuf_size;
            global_base_vertex += vertex_count;
            global_first_index += index_count;
        }

        if total_vbuf_size > 0 {
            let vbuf = create_or_reuse_host_visible_vertexindex_buffer_with_data(
                allocator,
                swapchain_frame_state,
                VertexIndexBufferType::Vertex,
                total_vbuf_size,
                &vbuf_chunks,
                Some(self.vbuf),
            );
            self.vbuf = (vbuf.0, vbuf.1, total_vbuf_size);
            let ibuf = create_or_reuse_host_visible_vertexindex_buffer_with_data(
                allocator,
                swapchain_frame_state,
                VertexIndexBufferType::Index,
                total_ibuf_size,
                &ibuf_chunks,
                Some(self.ibuf),
            );
            self.ibuf = (ibuf.0, ibuf.1, total_ibuf_size);
        }

        if self.ubufs[0].0 == ash::vk::Buffer::null() {
            for frame_slot in 0..FRAMES_IN_FLIGHT {
                self.ubufs[frame_slot as usize] = allocator
                    .create_host_visible_buffer(
                        IMGUI_MATERIAL_UBUF_SIZE,
                        ash::vk::BufferUsageFlags::UNIFORM_BUFFER,
                    )
                    .unwrap();
            }
        }

        if self.descriptor_pool.is_none() {
            self.descriptor_pool = Some(DescriptorPool::new(
                device,
                FRAMES_IN_FLIGHT,
                &[
                    ash::vk::DescriptorPoolSize {
                        ty: ash::vk::DescriptorType::UNIFORM_BUFFER,
                        descriptor_count: FRAMES_IN_FLIGHT,
                    },
                    ash::vk::DescriptorPoolSize {
                        ty: ash::vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                        descriptor_count: FRAMES_IN_FLIGHT,
                    },
                ],
            ));
        }

        if self.desc_sets.len() == 0 {
            self.desc_sets = self
                .descriptor_pool
                .as_ref()
                .unwrap()
                .allocate(
                    &[&self.material_pipeline.as_ref().unwrap().desc_set_layout;
                        FRAMES_IN_FLIGHT as usize],
                )
                .expect("Failed to allocate descriptor sets for ImGui content");
            let mut desc_buffer_infos: smallvec::SmallVec<[ash::vk::DescriptorBufferInfo; 4]> =
                smallvec::smallvec![];
            for i in 0..FRAMES_IN_FLIGHT {
                desc_buffer_infos.push(ash::vk::DescriptorBufferInfo {
                    buffer: self.ubufs[i as usize].0,
                    offset: 0,
                    range: IMGUI_MATERIAL_UBUF_SIZE as u64,
                });
            }
            let desc_image_info = ash::vk::DescriptorImageInfo {
                sampler: self.sampler.as_ref().unwrap().sampler,
                image_view: self.font_texture_view.as_ref().unwrap().view,
                image_layout: ash::vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                ..Default::default()
            };
            let mut desc_writes: smallvec::SmallVec<[ash::vk::WriteDescriptorSet; 4]> =
                smallvec::smallvec![];
            for i in 0..FRAMES_IN_FLIGHT {
                desc_writes.push(ash::vk::WriteDescriptorSet {
                    dst_set: self.desc_sets[i as usize],
                    dst_binding: 0,
                    descriptor_count: 1,
                    descriptor_type: ash::vk::DescriptorType::UNIFORM_BUFFER,
                    p_buffer_info: &desc_buffer_infos[i as usize],
                    ..Default::default()
                });
                desc_writes.push(ash::vk::WriteDescriptorSet {
                    dst_set: self.desc_sets[i as usize],
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
        }

        if self.last_display_size != draw_data.display_size {
            self.projection = glm::ortho_zo(
                0.0f32,
                draw_data.display_size[0],
                0.0f32,
                draw_data.display_size[1],
                1.0f32,
                -1.0f32,
            );
            self.last_display_size = draw_data.display_size;
        }
        allocator.update_host_visible_buffer(
            &self.ubufs[current_frame_slot as usize].1,
            0,
            64,
            0,
            &[(self.projection.as_ptr() as *const u8, 0, 64)],
        );
    }

    pub fn render(&self, swapchain_frame_state: &SwapchainFrameState, command_list: &CommandList) {
        if !self.active {
            return;
        }
        let device = self.device.as_ref().unwrap();
        let cb = swapchain_frame_state.current_frame_command_buffer(command_list);
        let current_frame_slot = swapchain_frame_state.current_frame_slot;
        unsafe {
            device.device.cmd_bind_pipeline(
                *cb,
                ash::vk::PipelineBindPoint::GRAPHICS,
                self.material_pipeline.as_ref().unwrap().pipeline.pipeline,
            );
            device.device.cmd_bind_descriptor_sets(
                *cb,
                ash::vk::PipelineBindPoint::GRAPHICS,
                self.material_pipeline
                    .as_ref()
                    .unwrap()
                    .pipeline_layout
                    .layout,
                0,
                &[self.desc_sets[current_frame_slot as usize]],
                &[],
            );
            device
                .device
                .cmd_bind_vertex_buffers(*cb, 0, &[self.vbuf.0], &[0]);
            device
                .device
                .cmd_bind_index_buffer(*cb, self.ibuf.0, 0, ash::vk::IndexType::UINT16);
            let mut last_scissor: Option<ash::vk::Rect2D> = None;
            for cmd in self.draw_commands.iter() {
                let scissor = ash::vk::Rect2D {
                    offset: cmd.scissor_offset,
                    extent: cmd.scissor_extent,
                };
                if last_scissor.is_none() || last_scissor.unwrap() != scissor {
                    device.device.cmd_set_scissor(*cb, 0, &[scissor]);
                    last_scissor = Some(scissor);
                }
                device.device.cmd_draw_indexed(
                    *cb,
                    cmd.index_count,
                    1,
                    cmd.first_index,
                    cmd.base_vertex,
                    0,
                );
            }
        }
    }
}

impl Drop for ImGui {
    fn drop(&mut self) {
        self.release_resources();
    }
}
