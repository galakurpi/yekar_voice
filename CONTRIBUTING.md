# Contributing

## Scope

Current target environment:
- Linux desktop
- X11 first
- GNOME / Pop!_OS is the main tested path

If you are proposing Wayland, macOS, or Windows support, keep it separate from fixes that improve the current X11 path.

## Before Opening a PR

- Run `cargo fmt`
- Run `cargo check`
- Run `cargo test`
- If you touched dictation flow behavior, include a short note about what you tested manually

## Local Setup

```bash
. "$HOME/.cargo/env"
cargo run --release -- doctor
cargo run --release -- daemon
```

Optional:

```bash
cargo run --release -- download-model
cargo run --release -- benchmark
```

If you want OpenAI-backed STT/rewrite, create a local `.env` with:

```bash
OPENAI_API_KEY=your_key_here
```

Do not commit `.env`.

## Good Contributions

High-value areas:
- X11 paste reliability
- terminal detection and paste behavior
- overlay UX that does not steal focus
- instrumentation and latency analysis
- local/offline fallback improvements
- docs that reduce setup friction

## PR Notes

Keep changes focused.
If you change behavior, update the README or config example in the same PR.
