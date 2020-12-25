pub struct UiContent {
    demo_open: bool,
}

impl UiContent {
    pub fn new() -> Self {
        UiContent { demo_open: true }
    }

    pub fn update(&mut self, ui: &imgui::Ui) {
        let window = imgui::Window::new(imgui::im_str!("Hello world"));
        window
            .size([320.0, 200.0], imgui::Condition::FirstUseEver)
            .build(ui, || {
                ui.text(imgui::im_str!("blah blah"));
                ui.separator();
            });
        ui.show_demo_window(&mut self.demo_open);
    }
}
