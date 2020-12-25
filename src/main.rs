mod fw;
mod ui;
mod scene;
mod app;

use app::*;

fn main() {
    let event_loop = winit::event_loop::EventLoop::new();
    let app = App::new(&event_loop);
    app.run(event_loop);
}
