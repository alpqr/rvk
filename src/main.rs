mod app;
mod fw;
mod scene;
mod ui;
mod uicontent;

use app::*;

fn main() {
    let event_loop = winit::event_loop::EventLoop::new();
    let app = App::new(&event_loop);
    app.run(event_loop);
}
