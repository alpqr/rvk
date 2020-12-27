use crate::fw::*;
use crate::scene::Scene;
use crate::ui::ImGui;
use std::rc::Rc;

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
    depth_stencil_buffer: DepthStencilBuffer,
    allocator: Rc<MemAllocator>,
    pipeline_cache: PipelineCache,
    imgui: ImGui,
    scene: Scene,
}

impl App {
    pub fn new(event_loop: &winit::event_loop::EventLoop<()>) -> Self {
        let window = winit::window::WindowBuilder::new()
            .with_inner_size(winit::dpi::LogicalSize::new(1280, 720))
            .with_title("Rust Vulkan Test")
            .build(event_loop)
            .expect("Failed to create window");
        let instance = Rc::new(Instance::new(&window, ENABLE_VALIDATION));
        let surface = WindowSurface::new(&instance, &window);
        let physical_device = PhysicalDevice::new(&instance, &surface.surface);
        let device = Rc::new(Device::new(&instance, &physical_device));
        let allocator = Rc::new(MemAllocator::new(&instance, &physical_device, &device));
        let command_pool = CommandPool::new(&physical_device, &device);
        let command_list = CommandList::new(&device, &command_pool);
        let swapchain = SwapchainBuilder::new()
            .build(
                &instance,
                &physical_device,
                &device,
                &surface,
                surface.pixel_size(&instance, &physical_device, &window),
            )
            .unwrap();
        let swapchain_images = SwapchainImages::new(&device, &swapchain);
        let depth_stencil_buffer =
            DepthStencilBuffer::new(&physical_device, &device, swapchain.pixel_size);
        let swapchain_render_pass = SwapchainRenderPass::new(&physical_device, &device, &swapchain);
        let swapchain_framebuffers = SwapchainFramebuffers::new(
            &device,
            &swapchain,
            &swapchain_images,
            &swapchain_render_pass,
            &depth_stencil_buffer,
        );
        let swapchain_frame_state = SwapchainFrameState::new(&device, &allocator);
        let swapchain_resizer = SwapchainResizer::new();
        let pipeline_cache = PipelineCache::new(&device);
        let imgui = ImGui::new(&device, &allocator, &window);
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
            depth_stencil_buffer,
            allocator,
            pipeline_cache,
            imgui,
            scene,
        }
    }

    fn release_resources(&mut self) {
        // to be called on window close, the order matters, don't rely on Drop here
        self.device.wait_idle();
        self.imgui.release_resources();
        self.scene.release_resources();
        self.swapchain_frame_state.release_resources();
        self.swapchain_framebuffers.release_resources();
        self.swapchain_render_pass.release_resources();
        self.swapchain_images.release_resources();
        self.swapchain.release_resources();
        self.depth_stencil_buffer.release_resources();
        self.command_pool.release_resources();
        self.pipeline_cache.release_resources();
        Rc::get_mut(&mut self.allocator)
            .expect("Some resources referencing the allocator have not been released")
            .release_resources();
        Rc::get_mut(&mut self.device)
            .expect("Some resources referencing the device have not been released")
            .release_resources();
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
            &self.allocator,
            &mut self.swapchain,
            &self.swapchain_render_pass,
            &mut self.swapchain_images,
            &mut self.swapchain_framebuffers,
            &mut self.swapchain_frame_state,
            &mut self.depth_stencil_buffer,
        );
    }

    fn record_frame(&mut self) {
        self.scene.prepare(
            &self.swapchain,
            &self.swapchain_render_pass,
            &mut self.swapchain_frame_state,
            &self.command_list,
            &self.pipeline_cache,
        );

        self.imgui.prepare(
            &mut self.swapchain_frame_state,
            &self.command_list,
            &self.pipeline_cache,
            &self.swapchain_render_pass.render_pass,
            &self.window,
        );

        self.scene.begin_main_render_pass(
            &self.swapchain,
            &self.swapchain_framebuffers,
            &self.swapchain_render_pass,
            &self.swapchain_frame_state,
            &self.command_list,
        );

        self.scene
            .render_main_pass(&self.swapchain_frame_state, &self.command_list);

        self.imgui
            .render(&self.swapchain_frame_state, &self.command_list);

        self.scene
            .end_main_render_pass(&self.swapchain_frame_state, &self.command_list);
    }

    pub fn run(mut self, event_loop: winit::event_loop::EventLoop<()>) {
        let mut running = true;
        let mut frame_time = std::time::Instant::now();
        event_loop.run(move |event, _, control_flow| {
            *control_flow = winit::event_loop::ControlFlow::Poll;
            match event {
                winit::event::Event::WindowEvent {
                    window_id,
                    event: winit::event::WindowEvent::CloseRequested,
                } if window_id == self.window.id() => {
                    *control_flow = winit::event_loop::ControlFlow::Exit;
                    running = false;
                    self.release_resources();
                }
                winit::event::Event::MainEventsCleared => {
                    if self.imgui.active {
                        self.imgui
                            .winit_support
                            .prepare_frame(self.imgui.ctx.io_mut(), &self.window)
                            .unwrap();
                    }
                    if self.scene.sync() || self.imgui.active {
                        self.window.request_redraw();
                    }
                }
                winit::event::Event::RedrawRequested(window_id)
                    if window_id == self.window.id() && running =>
                {
                    if self.ensure_swapchain() {
                        match self.swapchain_frame_state.begin_frame(
                            &self.swapchain,
                            &self.command_pool,
                            &self.command_list,
                        ) {
                            Ok(current_frame_slot) => {
                                if cfg!(feature = "log-redraw") {
                                    println!(
                                        "Render, elapsed since last {:?} (slot {})",
                                        frame_time.elapsed(),
                                        current_frame_slot
                                    );
                                }
                                frame_time = std::time::Instant::now();
                                self.record_frame();
                                self.swapchain_frame_state.end_frame(
                                    &self.swapchain,
                                    &self.swapchain_images,
                                    &self.command_list,
                                    EndFrameFlags::empty(),
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
                event => {
                    let mut active_just_changed = false;
                    let mut forward_always = false;
                    match event {
                        winit::event::Event::WindowEvent {
                            window_id,
                            event: winit::event::WindowEvent::KeyboardInput { input, .. },
                        } if window_id == self.window.id()
                            && input.state == winit::event::ElementState::Pressed =>
                        {
                            match input.virtual_keycode {
                                Some(winit::event::VirtualKeyCode::Grave) => {
                                    self.imgui.active = !self.imgui.active;
                                    active_just_changed = true;
                                }
                                _ => (),
                            }
                        }
                        winit::event::Event::WindowEvent {
                            window_id,
                            event: winit::event::WindowEvent::Resized(_),
                        } if window_id == self.window.id() => {
                            forward_always = true;
                        }
                        _ => (),
                    }
                    if (self.imgui.active && !active_just_changed) || forward_always {
                        self.imgui.winit_support.handle_event(
                            self.imgui.ctx.io_mut(),
                            &self.window,
                            &event,
                        );
                    }
                }
            };
        });
    }
}
