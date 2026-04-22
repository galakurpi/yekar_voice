# Voice Dictation

Rust background dictation tool for Pop!_OS / GNOME / X11.

Current runtime:
- Hold `Ctrl+Alt+Q` to record
- Shows a tiny floating status bubble while listening and finalizing
- Release to stream/finalize OpenAI STT by default
- Local native Rust Whisper inference through Candle remains available as an opt-in fallback
- Captures at 24kHz for the OpenAI streaming path and resamples automatically for local Whisper
- Always runs the developer-focused rewrite pass when enabled
- Auto-pastes into the focused field and also leaves the final transcript in the clipboard
- Stores per-session timings in SQLite
- `Esc` cancels the current recording
- `Ctrl+Alt+V` pastes the last transcript again

Current scope:
- Linux desktop
- X11 only for now
- GNOME/Pop!_OS is the main tested environment
- Terminal paste is best-effort and currently uses `Ctrl+Shift+V`

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
cargo test
```

## Config

Config file:
- `~/.config/yekar/voice_dictation.toml`

Metrics DB:
- `~/.local/share/yekar/voice_dictation/metrics.sqlite3`

Optional repo-local secrets file:
- `.env`
- expected key: `OPENAI_API_KEY=...`

## Privacy

Default behavior:
- If `asr.backend = "openai"`, microphone audio is sent to OpenAI for transcription
- If rewrite is enabled, transcript text is sent to OpenAI for cleanup
- Metrics are stored locally in SQLite
- Full transcript storage is disabled by default unless you enable `metrics.store_transcript_text`

Local-first options:
- You can switch STT to the local Whisper backend in config
- Even with local STT, the rewrite step is still networked unless you disable rewrite

Operational expectations:
- Bring your own OpenAI API key
- Do not commit `.env`
- Review your config before sharing logs or metrics dumps

## Open Source Notes

- Public repo does not mean API usage is free; users pay their own provider costs
- This project is still early and desktop-specific
- Contributions that improve X11 paste reliability, terminal handling, and instrumentation are useful

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).

## Systemd User Service

The included [voice-dictation.service](voice-dictation.service) assumes the repo is cloned to `%h/src/yekar_voice`.
If you clone somewhere else, edit `WorkingDirectory` and `ExecStart` first.

Important behavior:
- `Ctrl+Alt+Q` is not a GNOME custom shortcut or desktop launcher
- the daemon itself grabs that hotkey on X11
- if the daemon is not already running, pressing the shortcut does nothing

For this checkout, install the user service with:

```bash
cd integrations/yekar_voice
./install-user-service.sh
```

That script writes `~/.config/systemd/user/voice-dictation.service` with the correct local path, enables it for `graphical-session.target`, and restarts it immediately.
