# Voice Dictation

Rust background dictation tool for Pop!_OS / GNOME / X11.

Current runtime:
- Hold `Ctrl+Alt+Q` to record
- Shows a tiny floating status bubble while listening and finalizing
- Release to stream/finalize OpenAI STT by default
- Local native Rust Whisper inference through Candle remains available as an opt-in fallback
- Captures at 24kHz for the OpenAI streaming path and resamples automatically for local Whisper
- Always runs the developer-focused rewrite pass when enabled
- Copies the final transcript into the clipboard on release
- Stores per-session timings in SQLite
- `Esc` cancels the current recording
- `Ctrl+Alt+V` pastes the last transcript again if you want an explicit paste step

## Quick Start

```bash
git clone https://github.com/galakurpi/yekar_voice.git
cd yekar_voice
. "$HOME/.cargo/env"
cargo run --release -- doctor
# optional: preload the local Whisper fallback
cargo run --release -- download-model
cargo run --release -- daemon
```

Useful commands:

```bash
cargo run --release -- stats
cargo run --release -- recent --limit 20
cargo run --release -- benchmark
```

## Config

Config file:
- `~/.config/yekar/voice_dictation.toml`

Metrics DB:
- `~/.local/share/yekar/voice_dictation/metrics.sqlite3`

Optional repo-local secrets file:
- `.env`
- expected key: `OPENAI_API_KEY=...`

## Systemd User Service

The included [voice-dictation.service](voice-dictation.service) assumes the repo is cloned to `%h/src/yekar_voice`.
If you clone somewhere else, edit `WorkingDirectory` and `ExecStart` first.
