use anyhow::{anyhow, Context, Result};
use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use base64::Engine;
use byteorder::{ByteOrder, LittleEndian};
use candle_core::{Device, IndexOp, Tensor};
use candle_nn::{ops::softmax, VarBuilder};
use candle_transformers::models::whisper::{self as m, audio, Config as WhisperConfig};
use clap::{Parser, Subcommand};
use global_hotkey::hotkey::{Code, HotKey, Modifiers};
use global_hotkey::{GlobalHotKeyEvent, GlobalHotKeyManager, HotKeyState};
use hf_hub::{api::sync::Api, Repo, RepoType};
use reqwest::blocking::{multipart, Client};
use rusqlite::{params, Connection};
use serde::Deserialize;
use serde_json::json;
use std::env;
use std::fs;
use std::io::{Cursor, Read, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdout, Command, Stdio};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender, TryRecvError};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokenizers::Tokenizer;
use tungstenite::client::IntoClientRequest;
use tungstenite::stream::MaybeTlsStream;
use tungstenite::{connect, Message, WebSocket};

mod multilingual;

#[derive(Debug, Parser)]
#[command(name = "voice-dictation")]
#[command(about = "Background voice dictation with metrics")]
struct Cli {
    #[arg(long)]
    config: Option<PathBuf>,
    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Subcommand)]
enum Commands {
    Daemon,
    Doctor,
    Stats,
    Recent {
        #[arg(long, default_value_t = 10)]
        limit: usize,
    },
    Benchmark {
        #[arg(long)]
        manifest: Option<PathBuf>,
    },
    DownloadModel,
}

#[derive(Debug, Clone, Deserialize)]
struct Config {
    #[serde(default)]
    hotkey: HotkeyConfig,
    #[serde(default)]
    audio: AudioConfig,
    #[serde(default)]
    asr: AsrConfig,
    #[serde(default)]
    rewrite: RewriteConfig,
    #[serde(default)]
    paste: PasteConfig,
    #[serde(default)]
    metrics: MetricsConfig,
    #[serde(default)]
    developer_profile: DeveloperProfile,
}

#[derive(Debug, Clone, Deserialize)]
struct HotkeyConfig {
    #[serde(default = "default_hotkey_modifiers")]
    modifiers: Vec<String>,
    #[serde(default = "default_trigger")]
    trigger: String,
    #[serde(default = "default_paste_last")]
    paste_last: Vec<String>,
    #[serde(default = "default_cancel")]
    cancel: String,
}

#[derive(Debug, Clone, Deserialize)]
struct AudioConfig {
    #[serde(default = "default_target_sample_rate")]
    target_sample_rate_hz: u32,
    #[serde(default = "default_channels")]
    channels: u16,
    #[serde(default = "default_tail_ms")]
    tail_ms: u64,
    #[serde(default = "default_min_audio_ms")]
    min_audio_ms: u64,
    #[serde(default)]
    device: String,
}

#[derive(Debug, Clone, Deserialize)]
struct AsrConfig {
    #[serde(default = "default_asr_backend")]
    backend: String,
    #[serde(default = "default_true")]
    streaming_enabled: bool,
    #[serde(default = "default_openai_asr_model")]
    openai_model: String,
    #[serde(default = "default_local_asr_model")]
    local_model: String,
    #[serde(default)]
    language: String,
    #[serde(default = "default_asr_timeout_ms")]
    timeout_ms: u64,
}

#[derive(Debug, Clone, Deserialize)]
struct RewriteConfig {
    #[serde(default = "default_true")]
    enabled: bool,
    #[serde(default = "default_rewrite_model")]
    model: String,
    #[serde(default = "default_timeout_ms")]
    timeout_ms: u64,
}

#[derive(Debug, Clone, Deserialize)]
struct PasteConfig {
    #[serde(default = "default_true")]
    restore_clipboard: bool,
    #[serde(default = "default_paste_binding")]
    keybinding: String,
}

#[derive(Debug, Clone, Deserialize, Default)]
struct MetricsConfig {
    #[serde(default)]
    store_transcript_text: bool,
}

#[derive(Debug, Clone, Deserialize, Default)]
struct DeveloperProfile {
    #[serde(default)]
    glossary: Vec<String>,
    #[serde(default)]
    replacements: Vec<(String, String)>,
}

#[derive(Debug, Clone, Copy)]
enum AsrBackend {
    Local,
    OpenAi,
}

impl AsrConfig {
    fn backend_kind(&self) -> Result<AsrBackend> {
        match self.backend.trim().to_ascii_lowercase().as_str() {
            "local" => Ok(AsrBackend::Local),
            "openai" => Ok(AsrBackend::OpenAi),
            other => Err(anyhow!("unsupported ASR backend: {other}")),
        }
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            hotkey: HotkeyConfig::default(),
            audio: AudioConfig::default(),
            asr: AsrConfig::default(),
            rewrite: RewriteConfig::default(),
            paste: PasteConfig::default(),
            metrics: MetricsConfig::default(),
            developer_profile: DeveloperProfile::default(),
        }
    }
}

impl Default for HotkeyConfig {
    fn default() -> Self {
        Self {
            modifiers: default_hotkey_modifiers(),
            trigger: default_trigger(),
            paste_last: default_paste_last(),
            cancel: default_cancel(),
        }
    }
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            target_sample_rate_hz: default_target_sample_rate(),
            channels: default_channels(),
            tail_ms: default_tail_ms(),
            min_audio_ms: default_min_audio_ms(),
            device: String::new(),
        }
    }
}

impl Default for AsrConfig {
    fn default() -> Self {
        Self {
            backend: default_asr_backend(),
            streaming_enabled: default_true(),
            openai_model: default_openai_asr_model(),
            local_model: default_local_asr_model(),
            language: String::new(),
            timeout_ms: default_asr_timeout_ms(),
        }
    }
}

impl Default for RewriteConfig {
    fn default() -> Self {
        Self {
            enabled: default_true(),
            model: default_rewrite_model(),
            timeout_ms: default_timeout_ms(),
        }
    }
}

impl Default for PasteConfig {
    fn default() -> Self {
        Self {
            restore_clipboard: false,
            keybinding: default_paste_binding(),
        }
    }
}

fn default_hotkey_modifiers() -> Vec<String> {
    vec!["ctrl".to_string(), "alt".to_string()]
}

fn default_trigger() -> String {
    "q".to_string()
}

fn default_paste_last() -> Vec<String> {
    vec!["ctrl".to_string(), "alt".to_string(), "v".to_string()]
}

fn default_cancel() -> String {
    "escape".to_string()
}

fn default_target_sample_rate() -> u32 {
    24_000
}

fn default_channels() -> u16 {
    1
}

fn default_tail_ms() -> u64 {
    120
}

fn default_min_audio_ms() -> u64 {
    250
}

fn default_asr_backend() -> String {
    "openai".to_string()
}

fn default_openai_asr_model() -> String {
    "gpt-4o-mini-transcribe".to_string()
}

fn default_local_asr_model() -> String {
    "base.en".to_string()
}

fn default_asr_timeout_ms() -> u64 {
    5_000
}

fn default_true() -> bool {
    true
}

fn default_rewrite_model() -> String {
    "gpt-5.3-chat-latest".to_string()
}

fn default_timeout_ms() -> u64 {
    3000
}

fn default_paste_binding() -> String {
    "ctrl+v".to_string()
}

fn workspace_root() -> Result<PathBuf> {
    let mut current = env::current_dir()?.canonicalize()?;
    loop {
        if current.join(".git").exists() {
            return Ok(current);
        }
        let Some(parent) = current.parent() else {
            return Err(anyhow!("workspace root not found"));
        };
        current = parent.to_path_buf();
    }
}

fn default_config_path() -> PathBuf {
    dirs::config_dir()
        .unwrap_or_else(|| PathBuf::from("/tmp"))
        .join("yekar/voice_dictation.toml")
}

fn default_metrics_db() -> PathBuf {
    dirs::data_dir()
        .unwrap_or_else(|| PathBuf::from("/tmp"))
        .join("yekar/voice_dictation/metrics.sqlite3")
}

fn voice_manifest_path() -> Result<PathBuf> {
    Ok(workspace_root()?.join("docs/voice-agent-test-fixtures/manifest.json"))
}

fn load_repo_env() -> Result<()> {
    let env_path = workspace_root()?.join(".env");
    if !env_path.exists() {
        return Ok(());
    }
    let contents = fs::read_to_string(env_path)?;
    for line in contents.lines() {
        if let Some((key, value)) = line.split_once('=') {
            if env::var_os(key).is_none() {
                env::set_var(key, value.trim().trim_matches('"').trim_matches('\''));
            }
        }
    }
    Ok(())
}

fn load_config(path: Option<PathBuf>) -> Result<(Config, PathBuf, PathBuf)> {
    load_repo_env()?;
    let config_path = path.unwrap_or_else(default_config_path);
    if let Some(parent) = config_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let config = if config_path.exists() {
        toml::from_str(&fs::read_to_string(&config_path)?)?
    } else {
        Config::default()
    };
    let metrics_path = default_metrics_db();
    if let Some(parent) = metrics_path.parent() {
        fs::create_dir_all(parent)?;
    }
    Ok((config, config_path, metrics_path))
}

#[derive(Debug, Clone)]
struct SessionTrace {
    session_id: String,
    wall_started_at: f64,
    started_at: Instant,
    events: Vec<(&'static str, f64)>,
}

impl SessionTrace {
    fn new() -> Self {
        Self {
            session_id: format!(
                "{:x}",
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_nanos()
            ),
            wall_started_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs_f64(),
            started_at: Instant::now(),
            events: Vec::new(),
        }
    }

    fn mark(&mut self, name: &'static str) {
        self.events
            .push((name, self.started_at.elapsed().as_secs_f64() * 1000.0));
    }

    fn mark_at(&mut self, name: &'static str, offset_ms: f64) {
        self.events.push((name, offset_ms));
    }

    fn offset_ms(&self, name: &str) -> Option<f64> {
        self.events
            .iter()
            .find(|(event, _)| *event == name)
            .map(|(_, offset)| *offset)
    }

    fn duration_ms(&self, start: &str, end: &str) -> Option<f64> {
        let a = self.offset_ms(start)?;
        let b = self.offset_ms(end)?;
        Some(b - a)
    }
}

struct MetricsStore {
    path: PathBuf,
}

impl MetricsStore {
    fn new(path: PathBuf) -> Result<Self> {
        let store = Self { path };
        store.init()?;
        Ok(store)
    }

    fn conn(&self) -> Result<Connection> {
        Ok(Connection::open(&self.path)?)
    }

    fn init(&self) -> Result<()> {
        let conn = self.conn()?;
        conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                started_at REAL NOT NULL,
                status TEXT NOT NULL,
                active_app TEXT,
                audio_duration_ms REAL,
                raw_text_chars INTEGER,
                final_text_chars INTEGER,
                rewrite_applied INTEGER NOT NULL DEFAULT 0,
                rewrite_fallback INTEGER NOT NULL DEFAULT 0,
                asr_backend TEXT,
                asr_mode TEXT,
                asr_fallback INTEGER NOT NULL DEFAULT 0,
                asr_model TEXT,
                rewrite_model TEXT,
                error_code TEXT,
                transcript_raw TEXT,
                transcript_final TEXT,
                t_hotkey_to_recording_ms REAL,
                t_hotkey_to_stream_ready_ms REAL,
                t_hotkey_to_first_partial_ms REAL,
                t_release_to_audio_finalized_ms REAL,
                t_release_to_asr_done_ms REAL,
                t_release_to_rewrite_done_ms REAL,
                t_release_to_paste_done_ms REAL,
                t_rewrite_ms REAL,
                t_copy_ms REAL,
                t_end_to_end_ms REAL
            );
            CREATE TABLE IF NOT EXISTS session_events (
                session_id TEXT NOT NULL,
                event_name TEXT NOT NULL,
                offset_ms REAL NOT NULL
            );
            "#,
        )?;
        ensure_column(&conn, "sessions", "asr_backend", "TEXT")?;
        ensure_column(&conn, "sessions", "asr_mode", "TEXT")?;
        ensure_column(
            &conn,
            "sessions",
            "asr_fallback",
            "INTEGER NOT NULL DEFAULT 0",
        )?;
        ensure_column(&conn, "sessions", "t_hotkey_to_stream_ready_ms", "REAL")?;
        ensure_column(&conn, "sessions", "t_hotkey_to_first_partial_ms", "REAL")?;
        ensure_column(&conn, "sessions", "t_release_to_audio_finalized_ms", "REAL")?;
        ensure_column(&conn, "sessions", "t_rewrite_ms", "REAL")?;
        ensure_column(&conn, "sessions", "t_copy_ms", "REAL")?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn record_session(
        &self,
        trace: &SessionTrace,
        status: &str,
        active_app: &str,
        audio_duration_ms: f64,
        raw_text: &str,
        final_text: &str,
        rewrite_applied: bool,
        rewrite_fallback: bool,
        asr_backend: &str,
        asr_mode: &str,
        asr_fallback: bool,
        asr_model: &str,
        rewrite_model: &str,
        error_code: Option<&str>,
        store_transcript_text: bool,
    ) -> Result<()> {
        let conn = self.conn()?;
        conn.execute(
            r#"
            INSERT INTO sessions (
                session_id, started_at, status, active_app, audio_duration_ms,
                raw_text_chars, final_text_chars, rewrite_applied, rewrite_fallback,
                asr_backend, asr_mode, asr_fallback, asr_model, rewrite_model, error_code,
                transcript_raw, transcript_final, t_hotkey_to_recording_ms,
                t_hotkey_to_stream_ready_ms, t_hotkey_to_first_partial_ms, t_release_to_audio_finalized_ms,
                t_release_to_asr_done_ms, t_release_to_rewrite_done_ms, t_release_to_paste_done_ms,
                t_rewrite_ms, t_copy_ms, t_end_to_end_ms
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19, ?20, ?21, ?22, ?23, ?24, ?25, ?26, ?27)
            "#,
            params![
                trace.session_id,
                trace.wall_started_at,
                status,
                active_app,
                audio_duration_ms,
                raw_text.len() as i64,
                final_text.len() as i64,
                rewrite_applied as i64,
                rewrite_fallback as i64,
                asr_backend,
                asr_mode,
                asr_fallback as i64,
                asr_model,
                rewrite_model,
                error_code,
                if store_transcript_text { Some(raw_text) } else { None },
                if store_transcript_text { Some(final_text) } else { None },
                trace.duration_ms("hotkey_down", "recording_started"),
                trace.duration_ms("hotkey_down", "stream_session_ready"),
                trace.duration_ms("hotkey_down", "first_partial_received"),
                trace.duration_ms("hotkey_up", "audio_finalized"),
                trace.duration_ms("hotkey_up", "asr_finished"),
                trace.duration_ms("hotkey_up", "rewrite_finished"),
                trace.duration_ms("hotkey_up", "paste_finished"),
                trace.duration_ms("rewrite_started", "rewrite_finished"),
                trace.duration_ms("paste_started", "paste_finished"),
                trace.duration_ms("hotkey_down", "session_done"),
            ],
        )?;
        for (event, offset) in &trace.events {
            conn.execute(
                "INSERT INTO session_events (session_id, event_name, offset_ms) VALUES (?1, ?2, ?3)",
                params![trace.session_id, event, offset],
            )?;
        }
        Ok(())
    }

    fn print_stats(&self) -> Result<()> {
        let conn = self.conn()?;
        let payload = json!({
            "session_count": scalar_i64(&conn, "SELECT COUNT(*) FROM sessions")?,
            "rewrite_fallback_count": scalar_i64(&conn, "SELECT COUNT(*) FROM sessions WHERE rewrite_fallback = 1")?,
            "asr_stream_count": scalar_i64(&conn, "SELECT COUNT(*) FROM sessions WHERE asr_mode = 'openai_stream'")?,
            "asr_fallback_count": scalar_i64(&conn, "SELECT COUNT(*) FROM sessions WHERE asr_fallback = 1")?,
            "t_hotkey_to_stream_ready_ms_p50": percentile_metric(&conn, "t_hotkey_to_stream_ready_ms", 0.50)?,
            "t_hotkey_to_stream_ready_ms_p95": percentile_metric(&conn, "t_hotkey_to_stream_ready_ms", 0.95)?,
            "t_hotkey_to_first_partial_ms_p50": percentile_metric(&conn, "t_hotkey_to_first_partial_ms", 0.50)?,
            "t_hotkey_to_first_partial_ms_p95": percentile_metric(&conn, "t_hotkey_to_first_partial_ms", 0.95)?,
            "t_release_to_audio_finalized_ms_p50": percentile_metric(&conn, "t_release_to_audio_finalized_ms", 0.50)?,
            "t_release_to_audio_finalized_ms_p95": percentile_metric(&conn, "t_release_to_audio_finalized_ms", 0.95)?,
            "t_release_to_asr_done_ms_avg": average_metric(&conn, "t_release_to_asr_done_ms")?,
            "t_release_to_asr_done_ms_p50": percentile_metric(&conn, "t_release_to_asr_done_ms", 0.50)?,
            "t_release_to_asr_done_ms_p95": percentile_metric(&conn, "t_release_to_asr_done_ms", 0.95)?,
            "t_rewrite_ms_p50": percentile_metric(&conn, "t_rewrite_ms", 0.50)?,
            "t_rewrite_ms_p95": percentile_metric(&conn, "t_rewrite_ms", 0.95)?,
            "t_copy_ms_p50": percentile_metric(&conn, "t_copy_ms", 0.50)?,
            "t_copy_ms_p95": percentile_metric(&conn, "t_copy_ms", 0.95)?,
            "t_release_to_paste_done_ms_avg": average_metric(&conn, "t_release_to_paste_done_ms")?,
            "t_release_to_paste_done_ms_p50": percentile_metric(&conn, "t_release_to_paste_done_ms", 0.50)?,
            "t_release_to_paste_done_ms_p95": percentile_metric(&conn, "t_release_to_paste_done_ms", 0.95)?,
            "t_end_to_end_ms_avg": average_metric(&conn, "t_end_to_end_ms")?,
            "t_end_to_end_ms_p50": percentile_metric(&conn, "t_end_to_end_ms", 0.50)?,
            "t_end_to_end_ms_p95": percentile_metric(&conn, "t_end_to_end_ms", 0.95)?,
        });
        println!("{}", serde_json::to_string_pretty(&payload)?);
        Ok(())
    }

    fn print_recent(&self, limit: usize) -> Result<()> {
        let conn = self.conn()?;
        let mut stmt = conn.prepare(
            "SELECT started_at, status, active_app, asr_mode, audio_duration_ms, t_hotkey_to_first_partial_ms, t_release_to_paste_done_ms, error_code FROM sessions ORDER BY started_at DESC LIMIT ?1",
        )?;
        let rows = stmt.query_map(params![limit as i64], |row| {
            Ok((
                row.get::<_, f64>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)
                    .unwrap_or_else(|_| "unknown".to_string()),
                row.get::<_, Option<String>>(3)?,
                row.get::<_, Option<f64>>(4)?,
                row.get::<_, Option<f64>>(5)?,
                row.get::<_, Option<f64>>(6)?,
                row.get::<_, Option<String>>(7)?,
            ))
        })?;
        for row in rows {
            let (
                started_at,
                status,
                active_app,
                asr_mode,
                audio_ms,
                first_partial_ms,
                release_to_paste,
                error_code,
            ) = row?;
            println!(
                "{}  status={}  app={}  asr_mode={:?}  audio_ms={:?}  first_partial_ms={:?}  release_to_copied_ms={:?}  error={:?}",
                started_at, status, active_app, asr_mode, audio_ms, first_partial_ms, release_to_paste, error_code
            );
        }
        Ok(())
    }
}

fn ensure_column(conn: &Connection, table: &str, column: &str, definition: &str) -> Result<()> {
    let mut stmt = conn.prepare(&format!("PRAGMA table_info({table})"))?;
    let rows = stmt.query_map([], |row| row.get::<_, String>(1))?;
    for row in rows {
        if row? == column {
            return Ok(());
        }
    }
    conn.execute(
        &format!("ALTER TABLE {table} ADD COLUMN {column} {definition}"),
        [],
    )?;
    Ok(())
}

fn scalar_i64(conn: &Connection, sql: &str) -> Result<i64> {
    Ok(conn.query_row(sql, [], |row| row.get(0))?)
}

fn metric_values(conn: &Connection, column: &str) -> Result<Vec<f64>> {
    let mut stmt = conn.prepare(&format!(
        "SELECT {column} FROM sessions WHERE {column} IS NOT NULL ORDER BY {column}"
    ))?;
    let rows = stmt.query_map([], |row| row.get::<_, f64>(0))?;
    let mut values = Vec::new();
    for row in rows {
        values.push(row?);
    }
    Ok(values)
}

fn average_metric(conn: &Connection, column: &str) -> Result<Option<f64>> {
    let values = metric_values(conn, column)?;
    if values.is_empty() {
        return Ok(None);
    }
    Ok(Some(values.iter().sum::<f64>() / values.len() as f64))
}

fn percentile_metric(conn: &Connection, column: &str, q: f64) -> Result<Option<f64>> {
    let values = metric_values(conn, column)?;
    if values.is_empty() {
        return Ok(None);
    }
    let pos = ((values.len() - 1) as f64 * q).clamp(0.0, (values.len() - 1) as f64);
    let low = pos.floor() as usize;
    let high = pos.ceil() as usize;
    if low == high {
        return Ok(Some(values[low]));
    }
    let weight = pos - low as f64;
    Ok(Some(values[low] * (1.0 - weight) + values[high] * weight))
}

struct Recorder {
    config: AudioConfig,
    child: Option<Child>,
    reader: Option<JoinHandle<Result<Vec<u8>>>>,
    chunk_rx: Option<Receiver<Vec<u8>>>,
}

impl Recorder {
    fn new(config: AudioConfig) -> Self {
        Self {
            config,
            child: None,
            reader: None,
            chunk_rx: None,
        }
    }

    fn start_recording(&mut self, stream_chunks: bool) -> Result<()> {
        if self.child.is_some() {
            return Ok(());
        }
        let source = if self.config.device.trim().is_empty() {
            "default".to_string()
        } else {
            self.config.device.clone()
        };
        let mut child = Command::new("ffmpeg")
            .args([
                "-loglevel",
                "error",
                "-nostdin",
                "-f",
                "pulse",
                "-i",
                &source,
                "-ac",
                &self.config.channels.to_string(),
                "-ar",
                &self.config.target_sample_rate_hz.to_string(),
                "-f",
                "f32le",
                "pipe:1",
            ])
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .context("failed to start ffmpeg recorder")?;
        let stdout = child.stdout.take().context("missing ffmpeg stdout")?;
        let (chunk_tx, chunk_rx) = if stream_chunks {
            let (tx, rx) = mpsc::channel();
            (Some(tx), Some(rx))
        } else {
            (None, None)
        };
        self.reader = Some(spawn_reader(stdout, chunk_tx));
        self.chunk_rx = chunk_rx;
        self.child = Some(child);
        Ok(())
    }

    fn take_chunk_rx(&mut self) -> Option<Receiver<Vec<u8>>> {
        self.chunk_rx.take()
    }

    fn stop_recording(&mut self) -> Result<Vec<f32>> {
        thread::sleep(Duration::from_millis(self.config.tail_ms));
        self.stop_process()?;
        let bytes = self.take_bytes()?;
        Ok(bytes_to_samples(&bytes))
    }

    fn cancel_recording(&mut self) -> Result<()> {
        self.stop_process()?;
        let _ = self.take_bytes()?;
        Ok(())
    }

    fn stop_process(&mut self) -> Result<()> {
        if let Some(child) = self.child.as_mut() {
            let _ = child.kill();
            let _ = child.wait();
        }
        self.child = None;
        self.chunk_rx = None;
        Ok(())
    }

    fn take_bytes(&mut self) -> Result<Vec<u8>> {
        if let Some(reader) = self.reader.take() {
            reader
                .join()
                .map_err(|_| anyhow!("recorder reader thread panicked"))?
        } else {
            Ok(Vec::new())
        }
    }
}

fn spawn_reader(
    mut stdout: ChildStdout,
    chunk_tx: Option<Sender<Vec<u8>>>,
) -> JoinHandle<Result<Vec<u8>>> {
    thread::spawn(move || {
        let mut bytes = Vec::new();
        let mut buf = [0u8; 8192];
        loop {
            match stdout.read(&mut buf) {
                Ok(0) => break,
                Ok(count) => {
                    let chunk = &buf[..count];
                    bytes.extend_from_slice(chunk);
                    if let Some(tx) = &chunk_tx {
                        let _ = tx.send(chunk.to_vec());
                    }
                }
                Err(err) => return Err(anyhow!("recorder read: {err}")),
            }
        }
        Ok(bytes)
    })
}

fn bytes_to_samples(bytes: &[u8]) -> Vec<f32> {
    let mut samples = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        samples.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    samples
}

fn f32_to_pcm16(sample: f32) -> i16 {
    (sample.clamp(-1.0, 1.0) * i16::MAX as f32).round() as i16
}

fn resample_linear_mono(samples: &[f32], input_rate: u32, output_rate: u32) -> Vec<f32> {
    if samples.is_empty() || input_rate == output_rate {
        return samples.to_vec();
    }
    let output_len =
        ((samples.len() as u128 * output_rate as u128) / input_rate as u128).max(1) as usize;
    let last = samples.len().saturating_sub(1);
    let step = input_rate as f64 / output_rate as f64;
    let mut out = Vec::with_capacity(output_len);
    for index in 0..output_len {
        let position = index as f64 * step;
        let left = position.floor() as usize;
        let right = usize::min(left + 1, last);
        let fraction = (position - left as f64) as f32;
        let left_sample = samples[usize::min(left, last)];
        let right_sample = samples[right];
        out.push(left_sample + (right_sample - left_sample) * fraction);
    }
    out
}

#[derive(Debug, Clone)]
struct RewriteResult {
    text: String,
    used_model: bool,
    fallback_used: bool,
    error_code: Option<String>,
}

struct Rewriter {
    config: RewriteConfig,
    profile: DeveloperProfile,
    client: Client,
    api_key: Option<String>,
}

impl Rewriter {
    fn new(config: RewriteConfig, profile: DeveloperProfile) -> Result<Self> {
        Ok(Self {
            client: Client::builder()
                .timeout(Duration::from_millis(config.timeout_ms))
                .build()?,
            api_key: env::var("OPENAI_API_KEY").ok(),
            config,
            profile,
        })
    }

    fn rewrite(&self, input: &str) -> RewriteResult {
        let normalized = normalize_text(input, &self.profile);
        if normalized.is_empty() {
            return RewriteResult {
                text: String::new(),
                used_model: false,
                fallback_used: false,
                error_code: None,
            };
        }
        if !self.config.enabled {
            return RewriteResult {
                text: normalized,
                used_model: false,
                fallback_used: false,
                error_code: None,
            };
        }
        let Some(api_key) = &self.api_key else {
            return RewriteResult {
                text: normalized,
                used_model: false,
                fallback_used: true,
                error_code: Some("missing_api_key".to_string()),
            };
        };
        let system_prompt = r#"You rewrite speech-to-text into clean written text for software development work.

Preserve meaning exactly. Fix punctuation, capitalization, spacing, and obvious grammar. Keep code identifiers, filenames, library names, commands, acronyms, and technical words intact. Do not invent code, filenames, or facts. If the text is already clean, make minimal edits.

When the speaker uses spoken software-style markers, convert them into the written form that a developer would expect. This includes things like slash commands, underscores, and explicit spoken list delimiters when the intent is clear.

Examples:
- "Execute slash go command" -> "Execute /go command."
- "Check whatsexec underscore jack folder" -> "Check whatsexec_jack folder"
- "So we have to do, open list, A, B and C, close list, then verify all" -> "So we have to do:
- A
- B
- C
Then verify all"

Output only the final rewritten text."#;
        let payload = json!({
            "model": self.config.model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": json!({
                        "text": normalized,
                        "technical_glossary": self.profile.glossary.join(", "),
                    }).to_string()
                }
            ],
            "max_completion_tokens": 400
        });
        match self
            .client
            .post("https://api.openai.com/v1/chat/completions")
            .bearer_auth(api_key)
            .json(&payload)
            .send()
        {
            Ok(response) => {
                if !response.status().is_success() {
                    return RewriteResult {
                        text: normalized,
                        used_model: true,
                        fallback_used: true,
                        error_code: Some(format!("http_{}", response.status().as_u16())),
                    };
                }
                match response.json::<serde_json::Value>() {
                    Ok(value) => {
                        let content = value["choices"][0]["message"]["content"]
                            .as_str()
                            .unwrap_or("")
                            .trim()
                            .trim_matches('`')
                            .trim()
                            .to_string();
                        if content.is_empty() {
                            RewriteResult {
                                text: normalized,
                                used_model: true,
                                fallback_used: true,
                                error_code: Some("empty_rewrite".to_string()),
                            }
                        } else {
                            RewriteResult {
                                text: apply_replacements(&content, &self.profile),
                                used_model: true,
                                fallback_used: false,
                                error_code: None,
                            }
                        }
                    }
                    Err(err) => RewriteResult {
                        text: normalized,
                        used_model: true,
                        fallback_used: true,
                        error_code: Some(format!("json:{err}")),
                    },
                }
            }
            Err(err) => RewriteResult {
                text: normalized,
                used_model: true,
                fallback_used: true,
                error_code: Some(format!("request:{err}")),
            },
        }
    }
}

fn normalize_text(text: &str, profile: &DeveloperProfile) -> String {
    let mut out = text.trim().replace("\r\n", "\n");
    out = out.replace(" new paragraph ", "\n\n");
    out = out.replace(" new line ", "\n");
    out = out.replace(" newline ", "\n");
    out = out.replace(" slash ", " /");
    out = out.replace(" Slash ", " /");
    out = out.replace(" underscore ", "_");
    out = out.replace(" Underscore ", "_");
    out = apply_replacements(&out, profile);
    out.lines()
        .map(|line| line.split_whitespace().collect::<Vec<_>>().join(" "))
        .collect::<Vec<_>>()
        .join("\n")
        .trim()
        .to_string()
}

fn apply_replacements(text: &str, profile: &DeveloperProfile) -> String {
    let mut out = text.to_string();
    for (source, target) in &profile.replacements {
        out = out.replace(source, target);
        out = out.replace(&source.to_lowercase(), target);
    }
    out
}

enum OverlayBackend {
    Zenity,
    NotifySend,
}

struct Overlay {
    backend: Option<OverlayBackend>,
    active_child: Option<Child>,
}

fn overlay_text(title: &str, body: &str) -> String {
    if body.trim().is_empty() {
        title.to_string()
    } else {
        format!("{}\n{}", title, body)
    }
}

impl Overlay {
    fn new() -> Self {
        let backend = if command_exists("zenity") {
            Some(OverlayBackend::Zenity)
        } else if command_exists("notify-send") {
            Some(OverlayBackend::NotifySend)
        } else {
            None
        };
        Self {
            backend,
            active_child: None,
        }
    }

    fn close_active(&mut self) {
        if let Some(mut child) = self.active_child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }

    fn refocus(&self, window_id: Option<&str>) {
        let Some(window_id) = window_id else { return };
        if !command_exists("xdotool") {
            return;
        }
        thread::sleep(Duration::from_millis(60));
        let _ = Command::new("xdotool")
            .args(["windowactivate", "--sync", window_id])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
    }

    fn show_state(&mut self, title: &str, body: &str, restore_focus: Option<&str>) {
        self.close_active();
        match self.backend {
            Some(OverlayBackend::Zenity) => {
                match Command::new("zenity")
                    .args([
                        "--progress",
                        "--pulsate",
                        "--no-cancel",
                        "--title",
                        "Voice Dictation",
                        "--text",
                        &overlay_text(title, body),
                        "--width",
                        "320",
                    ])
                    .stdout(Stdio::null())
                    .stderr(Stdio::null())
                    .spawn()
                {
                    Ok(child) => {
                        self.active_child = Some(child);
                        self.refocus(restore_focus);
                    }
                    Err(_) => self.backend = None,
                }
            }
            Some(OverlayBackend::NotifySend) => {
                let _ = Command::new("notify-send")
                    .args([
                        "-u",
                        "normal",
                        "-t",
                        "20000",
                        "-a",
                        "voice-dictation",
                        "-e",
                        title,
                        body,
                    ])
                    .stdout(Stdio::null())
                    .stderr(Stdio::null())
                    .status();
            }
            None => {}
        }
    }

    fn show_notice(
        &mut self,
        title: &str,
        body: &str,
        timeout_secs: u64,
        restore_focus: Option<&str>,
    ) {
        self.close_active();
        match self.backend {
            Some(OverlayBackend::Zenity) => {
                let text = overlay_text(title, body);
                let script = r#"sleep "$VOICE_DICTATION_TIMEOUT"; printf '100\n' | zenity --progress --no-cancel --auto-close --percentage=0 --title="Voice Dictation" --text="$VOICE_DICTATION_TEXT" --width=340"#;
                match Command::new("sh")
                    .args(["-lc", script])
                    .env("VOICE_DICTATION_TIMEOUT", timeout_secs.to_string())
                    .env("VOICE_DICTATION_TEXT", text)
                    .stdout(Stdio::null())
                    .stderr(Stdio::null())
                    .spawn()
                {
                    Ok(child) => {
                        self.active_child = Some(child);
                        self.refocus(restore_focus);
                    }
                    Err(_) => self.backend = None,
                }
            }
            Some(OverlayBackend::NotifySend) => {
                let _ = Command::new("notify-send")
                    .args([
                        "-u",
                        "critical",
                        "-t",
                        &(timeout_secs.saturating_mul(1000)).to_string(),
                        "-a",
                        "voice-dictation",
                        "-e",
                        title,
                        body,
                    ])
                    .stdout(Stdio::null())
                    .stderr(Stdio::null())
                    .status();
            }
            None => {}
        }
    }
}

#[derive(Clone, Copy)]
struct WhisperModelSpec {
    alias: &'static str,
    repo_id: &'static str,
    revision: &'static str,
    multilingual: bool,
}

fn whisper_model_spec(alias: &str) -> Result<WhisperModelSpec> {
    match alias {
        "tiny" => Ok(WhisperModelSpec {
            alias: "tiny",
            repo_id: "openai/whisper-tiny",
            revision: "main",
            multilingual: true,
        }),
        "tiny.en" => Ok(WhisperModelSpec {
            alias: "tiny.en",
            repo_id: "openai/whisper-tiny.en",
            revision: "refs/pr/15",
            multilingual: false,
        }),
        "base" => Ok(WhisperModelSpec {
            alias: "base",
            repo_id: "openai/whisper-base",
            revision: "refs/pr/22",
            multilingual: true,
        }),
        "base.en" => Ok(WhisperModelSpec {
            alias: "base.en",
            repo_id: "openai/whisper-base.en",
            revision: "refs/pr/13",
            multilingual: false,
        }),
        "small" => Ok(WhisperModelSpec {
            alias: "small",
            repo_id: "openai/whisper-small",
            revision: "main",
            multilingual: true,
        }),
        "small.en" => Ok(WhisperModelSpec {
            alias: "small.en",
            repo_id: "openai/whisper-small.en",
            revision: "refs/pr/10",
            multilingual: false,
        }),
        "medium" => Ok(WhisperModelSpec {
            alias: "medium",
            repo_id: "openai/whisper-medium",
            revision: "main",
            multilingual: true,
        }),
        "medium.en" => Ok(WhisperModelSpec {
            alias: "medium.en",
            repo_id: "openai/whisper-medium.en",
            revision: "main",
            multilingual: false,
        }),
        "large-v3" => Ok(WhisperModelSpec {
            alias: "large-v3",
            repo_id: "openai/whisper-large-v3",
            revision: "main",
            multilingual: true,
        }),
        "large-v3-turbo" => Ok(WhisperModelSpec {
            alias: "large-v3-turbo",
            repo_id: "openai/whisper-large-v3-turbo",
            revision: "main",
            multilingual: true,
        }),
        other => Err(anyhow!("unsupported whisper model alias: {other}")),
    }
}

pub enum Model {
    Normal(m::model::Whisper),
}

impl Model {
    pub fn config(&self) -> &WhisperConfig {
        match self {
            Self::Normal(model) => &model.config,
        }
    }

    pub fn encoder_forward(&mut self, x: &Tensor, flush: bool) -> candle_core::Result<Tensor> {
        match self {
            Self::Normal(model) => model.encoder.forward(x, flush),
        }
    }

    pub fn decoder_forward(
        &mut self,
        x: &Tensor,
        xa: &Tensor,
        flush: bool,
    ) -> candle_core::Result<Tensor> {
        match self {
            Self::Normal(model) => model.decoder.forward(x, xa, flush),
        }
    }

    pub fn decoder_final_linear(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        match self {
            Self::Normal(model) => model.decoder.final_linear(x),
        }
    }
}

#[derive(Debug)]
struct DecodingResult {
    text: String,
    avg_logprob: f64,
    no_speech_prob: f64,
}

struct Decoder<'a> {
    model: &'a mut Model,
    tokenizer: &'a Tokenizer,
    suppress_tokens: Tensor,
    sot_token: u32,
    transcribe_token: u32,
    eot_token: u32,
    no_speech_token: u32,
    no_timestamps_token: u32,
    language_token: Option<u32>,
}

impl<'a> Decoder<'a> {
    fn new(
        model: &'a mut Model,
        tokenizer: &'a Tokenizer,
        device: &Device,
        language_token: Option<u32>,
    ) -> Result<Self> {
        let suppress_tokens: Vec<f32> = (0..model.config().vocab_size as u32)
            .map(|i| {
                if model.config().suppress_tokens.contains(&i) {
                    f32::NEG_INFINITY
                } else {
                    0.0
                }
            })
            .collect();
        let suppress_tokens = Tensor::new(suppress_tokens.as_slice(), device)?;
        let sot_token = token_id(tokenizer, m::SOT_TOKEN)?;
        let transcribe_token = token_id(tokenizer, m::TRANSCRIBE_TOKEN)?;
        let eot_token = token_id(tokenizer, m::EOT_TOKEN)?;
        let no_timestamps_token = token_id(tokenizer, m::NO_TIMESTAMPS_TOKEN)?;
        let no_speech_token = m::NO_SPEECH_TOKENS
            .iter()
            .find_map(|token| token_id(tokenizer, token).ok())
            .ok_or_else(|| anyhow!("unable to find any non-speech token"))?;
        Ok(Self {
            model,
            tokenizer,
            suppress_tokens,
            sot_token,
            transcribe_token,
            eot_token,
            no_speech_token,
            no_timestamps_token,
            language_token,
        })
    }

    fn decode_segment(&mut self, mel: &Tensor) -> Result<DecodingResult> {
        let audio_features = self.model.encoder_forward(mel, true)?;
        let sample_len = self.model.config().max_target_positions / 2;
        let mut sum_logprob = 0.0;
        let mut measured_tokens = 0usize;
        let mut no_speech_prob = f64::NAN;
        let mut tokens = vec![self.sot_token];
        if let Some(language_token) = self.language_token {
            tokens.push(language_token);
        }
        tokens.push(self.transcribe_token);
        tokens.push(self.no_timestamps_token);

        for i in 0..sample_len {
            let tokens_t = Tensor::new(tokens.as_slice(), mel.device())?.unsqueeze(0)?;
            let ys = self
                .model
                .decoder_forward(&tokens_t, &audio_features, i == 0)?;
            if i == 0 {
                let logits = self.model.decoder_final_linear(&ys.i(..1)?)?.i(0)?.i(0)?;
                no_speech_prob = softmax(&logits, 0)?
                    .i(self.no_speech_token as usize)?
                    .to_scalar::<f32>()? as f64;
            }
            let (_, seq_len, _) = ys.dims3()?;
            let logits = self
                .model
                .decoder_final_linear(&ys.i((..1, seq_len - 1..))?)?
                .i(0)?
                .i(0)?
                .broadcast_add(&self.suppress_tokens)?;
            let probs = softmax(&logits, candle_core::D::Minus1)?;
            let logits_v: Vec<f32> = logits.to_vec1()?;
            let next_token = logits_v
                .iter()
                .enumerate()
                .max_by(|(_, left), (_, right)| left.total_cmp(right))
                .map(|(index, _)| index as u32)
                .ok_or_else(|| anyhow!("failed to decode next token"))?;
            tokens.push(next_token);
            if next_token == self.eot_token
                || tokens.len() > self.model.config().max_target_positions
            {
                break;
            }
            let prob = probs.i(next_token as usize)?.to_scalar::<f32>()? as f64;
            sum_logprob += prob.max(f64::MIN_POSITIVE).ln();
            measured_tokens += 1;
        }

        let text = self
            .tokenizer
            .decode(&tokens, true)
            .map_err(|err| anyhow!(err.to_string()))?;
        let avg_logprob = if measured_tokens == 0 {
            f64::NAN
        } else {
            sum_logprob / measured_tokens as f64
        };
        Ok(DecodingResult {
            text,
            avg_logprob,
            no_speech_prob,
        })
    }
}

struct LocalTranscriber {
    language: String,
    spec: WhisperModelSpec,
    device: Device,
    whisper_config: WhisperConfig,
    tokenizer: Tokenizer,
    mel_filters: Vec<f32>,
    model: Model,
}

impl LocalTranscriber {
    fn new(model_alias: &str, language: &str) -> Result<Self> {
        let spec = whisper_model_spec(model_alias)?;
        let api = Api::new()?;
        let repo = api.repo(Repo::with_revision(
            spec.repo_id.to_string(),
            RepoType::Model,
            spec.revision.to_string(),
        ));
        let config_path = repo.get("config.json")?;
        let tokenizer_path = repo.get("tokenizer.json")?;
        let weights_path = repo.get("model.safetensors")?;

        let whisper_config: WhisperConfig =
            serde_json::from_str(&fs::read_to_string(&config_path)?)?;
        let mel_filters = load_mel_filters(whisper_config.num_mel_bins)?;
        let tokenizer =
            Tokenizer::from_file(tokenizer_path).map_err(|err| anyhow!(err.to_string()))?;
        let device = Device::Cpu;
        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], m::DTYPE, &device)? };
        let model = Model::Normal(m::model::Whisper::load(&vb, whisper_config.clone())?);

        Ok(Self {
            language: language.to_string(),
            spec,
            device,
            whisper_config,
            tokenizer,
            mel_filters,
            model,
        })
    }

    fn warm_up(&mut self) -> Result<()> {
        let sample_rate = m::SAMPLE_RATE as u32;
        let silence = vec![0.0f32; sample_rate as usize];
        let _ = self.transcribe_audio(&silence, sample_rate)?;
        Ok(())
    }

    fn transcribe_audio(&mut self, audio_samples: &[f32], sample_rate: u32) -> Result<String> {
        let resampled;
        let audio_samples = if sample_rate != m::SAMPLE_RATE as u32 {
            resampled = resample_linear_mono(audio_samples, sample_rate, m::SAMPLE_RATE as u32);
            resampled.as_slice()
        } else {
            audio_samples
        };
        let mel = audio::pcm_to_mel(&self.whisper_config, audio_samples, &self.mel_filters);
        let mel_len = mel.len();
        let mel = Tensor::from_vec(
            mel,
            (
                1,
                self.whisper_config.num_mel_bins,
                mel_len / self.whisper_config.num_mel_bins,
            ),
            &self.device,
        )?;
        self.transcribe_mel(&mel)
    }

    fn transcribe_wav(&mut self, wav_path: &Path) -> Result<String> {
        let (audio_samples, sample_rate) = read_wav_mono_f32(wav_path)?;
        self.transcribe_audio(&audio_samples, sample_rate)
    }

    fn model_name(&self) -> &str {
        self.spec.alias
    }

    fn transcribe_mel(&mut self, mel: &Tensor) -> Result<String> {
        let language_token = match (self.spec.multilingual, self.language.trim()) {
            (true, "") => Some(multilingual::detect_language(
                &mut self.model,
                &self.tokenizer,
                mel,
            )?),
            (true, language) => Some(token_id(&self.tokenizer, &format!("<|{language}|>"))?),
            (false, "") => None,
            (false, _) => {
                return Err(anyhow!(
                    "language cannot be set for English-only Whisper models"
                ))
            }
        };
        let mut decoder = Decoder::new(
            &mut self.model,
            &self.tokenizer,
            &self.device,
            language_token,
        )?;
        let (_, _, content_frames) = mel.dims3()?;
        let mut seek = 0;
        let mut pieces = Vec::new();
        while seek < content_frames {
            let segment_size = usize::min(content_frames - seek, m::N_FRAMES);
            let mel_segment = mel.narrow(2, seek, segment_size)?;
            seek += segment_size;
            let decoded = decoder.decode_segment(&mel_segment)?;
            if decoded.no_speech_prob > m::NO_SPEECH_THRESHOLD
                && decoded.avg_logprob < m::LOGPROB_THRESHOLD
            {
                continue;
            }
            let trimmed = decoded.text.trim();
            if !trimmed.is_empty() {
                pieces.push(trimmed.to_string());
            }
        }
        Ok(pieces.join(" ").trim().to_string())
    }
}

struct AsrResult {
    text: String,
    mode: &'static str,
    fallback_used: bool,
    error_code: Option<String>,
    stream_ready_ms: Option<f64>,
    first_partial_ms: Option<f64>,
}

struct OpenAiStreamHandle {
    control_tx: Sender<OpenAiStreamCommand>,
    result_rx: Receiver<OpenAiStreamResult>,
    join: Option<JoinHandle<()>>,
}

enum OpenAiStreamCommand {
    Commit,
    Cancel,
}

struct OpenAiStreamResult {
    transcript: Option<String>,
    error_code: Option<String>,
    stream_ready_ms: Option<f64>,
    first_partial_ms: Option<f64>,
}

impl OpenAiStreamHandle {
    fn finish(mut self, timeout_ms: u64) -> OpenAiStreamResult {
        let _ = self.control_tx.send(OpenAiStreamCommand::Commit);
        match self
            .result_rx
            .recv_timeout(Duration::from_millis(timeout_ms))
        {
            Ok(result) => {
                if let Some(join) = self.join.take() {
                    let _ = join.join();
                }
                result
            }
            Err(RecvTimeoutError::Timeout) => OpenAiStreamResult {
                transcript: None,
                error_code: Some("stream_timeout".to_string()),
                stream_ready_ms: None,
                first_partial_ms: None,
            },
            Err(RecvTimeoutError::Disconnected) => OpenAiStreamResult {
                transcript: None,
                error_code: Some("stream_disconnected".to_string()),
                stream_ready_ms: None,
                first_partial_ms: None,
            },
        }
    }

    fn cancel(self) {
        let _ = self.control_tx.send(OpenAiStreamCommand::Cancel);
    }
}

struct OpenAiTranscriber {
    client: Client,
    api_key: String,
    model: String,
    language: String,
    timeout_ms: u64,
    streaming_enabled: bool,
}

impl OpenAiTranscriber {
    fn new(config: &AsrConfig) -> Result<Self> {
        let api_key = env::var("OPENAI_API_KEY")
            .context("OPENAI_API_KEY is required for the openai ASR backend")?;
        let client = Client::builder()
            .timeout(Duration::from_millis(config.timeout_ms))
            .build()?;
        Ok(Self {
            client,
            api_key,
            model: config.openai_model.clone(),
            language: config.language.clone(),
            timeout_ms: config.timeout_ms,
            streaming_enabled: config.streaming_enabled,
        })
    }

    fn streaming_enabled(&self) -> bool {
        self.streaming_enabled
    }

    fn start_streaming(
        &self,
        session_started_at: Instant,
        chunk_rx: Receiver<Vec<u8>>,
    ) -> OpenAiStreamHandle {
        let (control_tx, control_rx) = mpsc::channel();
        let (result_tx, result_rx) = mpsc::channel();
        let client = self.client.clone();
        let api_key = self.api_key.clone();
        let model = self.model.clone();
        let language = self.language.clone();
        let join = thread::spawn(move || {
            let result = run_openai_realtime_stream(
                client,
                api_key,
                model,
                language,
                session_started_at,
                chunk_rx,
                control_rx,
            );
            let _ = result_tx.send(result);
        });
        OpenAiStreamHandle {
            control_tx,
            result_rx,
            join: Some(join),
        }
    }

    fn transcribe_audio(&mut self, audio_samples: &[f32], sample_rate: u32) -> Result<String> {
        let wav_bytes = encode_wav_mono_i16(audio_samples, sample_rate)?;
        self.transcribe_wav_bytes(wav_bytes, "dictation.wav")
    }

    fn transcribe_wav(&mut self, wav_path: &Path) -> Result<String> {
        let wav_bytes = fs::read(wav_path)?;
        let file_name = wav_path
            .file_name()
            .and_then(|value| value.to_str())
            .unwrap_or("audio.wav");
        self.transcribe_wav_bytes(wav_bytes, file_name)
    }

    fn model_name(&self) -> &str {
        &self.model
    }

    fn transcribe_wav_bytes(&self, wav_bytes: Vec<u8>, file_name: &str) -> Result<String> {
        let file_part = multipart::Part::bytes(wav_bytes)
            .file_name(file_name.to_string())
            .mime_str("audio/wav")?;
        let mut form = multipart::Form::new()
            .text("model", self.model.clone())
            .part("file", file_part);
        if !self.language.trim().is_empty() {
            form = form.text("language", self.language.clone());
        }
        let response = self
            .client
            .post("https://api.openai.com/v1/audio/transcriptions")
            .bearer_auth(&self.api_key)
            .multipart(form)
            .send()?;
        let status = response.status();
        if !status.is_success() {
            let body = response.text().unwrap_or_default();
            let snippet = body.chars().take(200).collect::<String>();
            return Err(anyhow!(
                "openai transcription http_{}: {}",
                status.as_u16(),
                snippet
            ));
        }
        let payload = response.json::<serde_json::Value>()?;
        Ok(payload["text"].as_str().unwrap_or("").trim().to_string())
    }

    fn create_realtime_client_secret(&self) -> Result<String> {
        let mut transcription = json!({
            "model": self.model,
        });
        if !self.language.trim().is_empty() {
            transcription["language"] = json!(self.language);
        }
        let payload = json!({
            "session": {
                "type": "transcription",
                "audio": {
                    "input": {
                        "format": {
                            "type": "audio/pcm",
                            "rate": 24000,
                        },
                        "transcription": transcription,
                        "turn_detection": serde_json::Value::Null,
                    }
                }
            }
        });
        let response = self
            .client
            .post("https://api.openai.com/v1/realtime/client_secrets")
            .bearer_auth(&self.api_key)
            .json(&payload)
            .send()?;
        let status = response.status();
        if !status.is_success() {
            let body = response.text().unwrap_or_default();
            let snippet = body.chars().take(200).collect::<String>();
            return Err(anyhow!(
                "openai client_secret http_{}: {}",
                status.as_u16(),
                snippet
            ));
        }
        let payload = response.json::<serde_json::Value>()?;
        payload["value"]
            .as_str()
            .map(str::to_string)
            .context("missing realtime client secret value")
    }
}

fn run_openai_realtime_stream(
    client: Client,
    api_key: String,
    model: String,
    language: String,
    session_started_at: Instant,
    chunk_rx: Receiver<Vec<u8>>,
    control_rx: Receiver<OpenAiStreamCommand>,
) -> OpenAiStreamResult {
    let transcriber = OpenAiTranscriber {
        client,
        api_key,
        model,
        language,
        timeout_ms: 5_000,
        streaming_enabled: true,
    };
    let secret = match transcriber.create_realtime_client_secret() {
        Ok(secret) => secret,
        Err(err) => {
            return OpenAiStreamResult {
                transcript: None,
                error_code: Some(format!("stream_secret:{err}")),
                stream_ready_ms: None,
                first_partial_ms: None,
            };
        }
    };
    let mut request = match "wss://api.openai.com/v1/realtime".into_client_request() {
        Ok(request) => request,
        Err(err) => {
            return OpenAiStreamResult {
                transcript: None,
                error_code: Some(format!("stream_request:{err}")),
                stream_ready_ms: None,
                first_partial_ms: None,
            };
        }
    };
    let auth_header = match format!("Bearer {secret}").parse() {
        Ok(value) => value,
        Err(err) => {
            return OpenAiStreamResult {
                transcript: None,
                error_code: Some(format!("stream_auth_header:{err}")),
                stream_ready_ms: None,
                first_partial_ms: None,
            };
        }
    };
    request.headers_mut().insert("Authorization", auth_header);
    let (mut socket, _) = match connect(request) {
        Ok(value) => value,
        Err(err) => {
            return OpenAiStreamResult {
                transcript: None,
                error_code: Some(format!("stream_connect:{err}")),
                stream_ready_ms: None,
                first_partial_ms: None,
            };
        }
    };
    if let Err(err) = set_websocket_nonblocking(&mut socket) {
        return OpenAiStreamResult {
            transcript: None,
            error_code: Some(format!("stream_nonblocking:{err}")),
            stream_ready_ms: None,
            first_partial_ms: None,
        };
    }

    let mut pending_f32_bytes = Vec::new();
    let mut audio_closed = false;
    let mut commit_requested = false;
    let mut commit_sent = false;
    let mut transcript = None;
    let mut error_code = None;
    let mut stream_ready_ms = None;
    let mut first_partial_ms = None;

    loop {
        let mut did_work = false;

        loop {
            match chunk_rx.try_recv() {
                Ok(chunk) => {
                    pending_f32_bytes.extend_from_slice(&chunk);
                    did_work = true;
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    audio_closed = true;
                    break;
                }
            }
        }

        let aligned_len = pending_f32_bytes.len() - (pending_f32_bytes.len() % 4);
        if aligned_len > 0 {
            let mut pcm16_bytes = Vec::with_capacity(aligned_len / 2);
            for chunk in pending_f32_bytes[..aligned_len].chunks_exact(4) {
                let sample = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                pcm16_bytes.extend_from_slice(&f32_to_pcm16(sample).to_le_bytes());
            }
            if !pcm16_bytes.is_empty() {
                let event = json!({
                    "type": "input_audio_buffer.append",
                    "audio": BASE64_STANDARD.encode(pcm16_bytes),
                });
                if let Err(err) = socket.send(Message::Text(event.to_string().into())) {
                    error_code = Some(format!("stream_append:{err}"));
                    break;
                }
                did_work = true;
            }
            pending_f32_bytes.drain(..aligned_len);
        }

        match control_rx.try_recv() {
            Ok(OpenAiStreamCommand::Commit) => commit_requested = true,
            Ok(OpenAiStreamCommand::Cancel) => {
                return OpenAiStreamResult {
                    transcript: None,
                    error_code: Some("stream_cancelled".to_string()),
                    stream_ready_ms,
                    first_partial_ms,
                };
            }
            Err(TryRecvError::Empty) => {}
            Err(TryRecvError::Disconnected) => commit_requested = true,
        }

        if audio_closed && !pending_f32_bytes.is_empty() {
            pending_f32_bytes.clear();
        }

        if commit_requested && audio_closed && pending_f32_bytes.is_empty() && !commit_sent {
            let event = json!({ "type": "input_audio_buffer.commit" });
            if let Err(err) = socket.send(Message::Text(event.to_string().into())) {
                error_code = Some(format!("stream_commit:{err}"));
                break;
            }
            commit_sent = true;
            did_work = true;
        }

        loop {
            match socket.read() {
                Ok(Message::Text(message)) => {
                    did_work = true;
                    match serde_json::from_str::<serde_json::Value>(&message) {
                        Ok(event) => match event["type"].as_str().unwrap_or("") {
                            "session.created" => {
                                if stream_ready_ms.is_none() {
                                    stream_ready_ms =
                                        Some(session_started_at.elapsed().as_secs_f64() * 1000.0);
                                }
                            }
                            "conversation.item.input_audio_transcription.delta" => {
                                if first_partial_ms.is_none() && event["delta"].as_str().is_some() {
                                    first_partial_ms =
                                        Some(session_started_at.elapsed().as_secs_f64() * 1000.0);
                                }
                            }
                            "conversation.item.input_audio_transcription.completed" => {
                                transcript = Some(
                                    event["transcript"]
                                        .as_str()
                                        .unwrap_or("")
                                        .trim()
                                        .to_string(),
                                );
                                return OpenAiStreamResult {
                                    transcript,
                                    error_code,
                                    stream_ready_ms,
                                    first_partial_ms,
                                };
                            }
                            "error" => {
                                error_code = Some(
                                    event["error"]["message"]
                                        .as_str()
                                        .map(str::to_string)
                                        .unwrap_or_else(|| "stream_error".to_string()),
                                );
                                return OpenAiStreamResult {
                                    transcript: None,
                                    error_code,
                                    stream_ready_ms,
                                    first_partial_ms,
                                };
                            }
                            _ => {}
                        },
                        Err(err) => {
                            return OpenAiStreamResult {
                                transcript: None,
                                error_code: Some(format!("stream_json:{err}")),
                                stream_ready_ms,
                                first_partial_ms,
                            };
                        }
                    }
                }
                Ok(Message::Binary(_))
                | Ok(Message::Ping(_))
                | Ok(Message::Pong(_))
                | Ok(Message::Frame(_)) => {
                    did_work = true;
                }
                Ok(Message::Close(_)) => {
                    break;
                }
                Err(tungstenite::Error::Io(err))
                    if err.kind() == std::io::ErrorKind::WouldBlock =>
                {
                    break
                }
                Err(tungstenite::Error::ConnectionClosed) => break,
                Err(err) => {
                    error_code = Some(format!("stream_read:{err}"));
                    break;
                }
            }
        }

        if error_code.is_some() {
            break;
        }

        if commit_sent && audio_closed && transcript.is_some() {
            break;
        }

        if !did_work {
            thread::sleep(Duration::from_millis(8));
        }
    }

    OpenAiStreamResult {
        transcript,
        error_code: error_code.or_else(|| Some("stream_incomplete".to_string())),
        stream_ready_ms,
        first_partial_ms,
    }
}

fn set_websocket_nonblocking(socket: &mut WebSocket<MaybeTlsStream<TcpStream>>) -> Result<()> {
    match socket.get_mut() {
        MaybeTlsStream::Plain(stream) => stream.set_nonblocking(true)?,
        MaybeTlsStream::Rustls(stream) => stream.sock.set_nonblocking(true)?,
        _ => return Err(anyhow!("unsupported websocket TLS stream")),
    }
    Ok(())
}

enum Transcriber {
    Local(LocalTranscriber),
    OpenAi(OpenAiTranscriber),
}

impl Transcriber {
    fn new(config: AsrConfig) -> Result<Self> {
        match config.backend_kind()? {
            AsrBackend::Local => Ok(Self::Local(LocalTranscriber::new(
                &config.local_model,
                &config.language,
            )?)),
            AsrBackend::OpenAi => Ok(Self::OpenAi(OpenAiTranscriber::new(&config)?)),
        }
    }

    fn start_streaming(
        &self,
        session_started_at: Instant,
        chunk_rx: Option<Receiver<Vec<u8>>>,
    ) -> Option<OpenAiStreamHandle> {
        match (self, chunk_rx) {
            (Self::OpenAi(transcriber), Some(chunk_rx)) if transcriber.streaming_enabled() => {
                Some(transcriber.start_streaming(session_started_at, chunk_rx))
            }
            _ => None,
        }
    }

    fn transcribe_after_release(
        &mut self,
        audio_samples: &[f32],
        sample_rate: u32,
        stream: Option<OpenAiStreamHandle>,
    ) -> AsrResult {
        match self {
            Self::Local(transcriber) => {
                match transcriber.transcribe_audio(audio_samples, sample_rate) {
                    Ok(text) => AsrResult {
                        text,
                        mode: "local",
                        fallback_used: false,
                        error_code: None,
                        stream_ready_ms: None,
                        first_partial_ms: None,
                    },
                    Err(err) => AsrResult {
                        text: String::new(),
                        mode: "local",
                        fallback_used: false,
                        error_code: Some(format!("local:{err}")),
                        stream_ready_ms: None,
                        first_partial_ms: None,
                    },
                }
            }
            Self::OpenAi(transcriber) => {
                if let Some(stream) = stream {
                    let stream_result = stream.finish(transcriber.timeout_ms);
                    if stream_result.error_code.is_none() {
                        return AsrResult {
                            text: stream_result.transcript.unwrap_or_default(),
                            mode: "openai_stream",
                            fallback_used: false,
                            error_code: None,
                            stream_ready_ms: stream_result.stream_ready_ms,
                            first_partial_ms: stream_result.first_partial_ms,
                        };
                    }
                    let stream_error = stream_result.error_code.clone();
                    match transcriber.transcribe_audio(audio_samples, sample_rate) {
                        Ok(text) => AsrResult {
                            text,
                            mode: "openai_file",
                            fallback_used: true,
                            error_code: stream_error,
                            stream_ready_ms: stream_result.stream_ready_ms,
                            first_partial_ms: stream_result.first_partial_ms,
                        },
                        Err(err) => AsrResult {
                            text: String::new(),
                            mode: "openai_file",
                            fallback_used: true,
                            error_code: Some(match stream_error {
                                Some(stream_error) => format!("{stream_error}; file:{err}"),
                                None => format!("file:{err}"),
                            }),
                            stream_ready_ms: stream_result.stream_ready_ms,
                            first_partial_ms: stream_result.first_partial_ms,
                        },
                    }
                } else {
                    match transcriber.transcribe_audio(audio_samples, sample_rate) {
                        Ok(text) => AsrResult {
                            text,
                            mode: "openai_file",
                            fallback_used: false,
                            error_code: None,
                            stream_ready_ms: None,
                            first_partial_ms: None,
                        },
                        Err(err) => AsrResult {
                            text: String::new(),
                            mode: "openai_file",
                            fallback_used: false,
                            error_code: Some(format!("file:{err}")),
                            stream_ready_ms: None,
                            first_partial_ms: None,
                        },
                    }
                }
            }
        }
    }

    fn transcribe_wav(&mut self, wav_path: &Path) -> Result<String> {
        match self {
            Self::Local(transcriber) => transcriber.transcribe_wav(wav_path),
            Self::OpenAi(transcriber) => transcriber.transcribe_wav(wav_path),
        }
    }

    fn model_name(&self) -> &str {
        match self {
            Self::Local(transcriber) => transcriber.model_name(),
            Self::OpenAi(transcriber) => transcriber.model_name(),
        }
    }

    fn backend_name(&self) -> &'static str {
        match self {
            Self::Local(_) => "local",
            Self::OpenAi(_) => "openai",
        }
    }
}

fn encode_wav_mono_i16(audio_samples: &[f32], sample_rate: u32) -> Result<Vec<u8>> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut cursor = Cursor::new(Vec::new());
    {
        let mut writer = hound::WavWriter::new(&mut cursor, spec)?;
        for sample in audio_samples {
            let pcm = (sample.clamp(-1.0, 1.0) * i16::MAX as f32).round() as i16;
            writer.write_sample(pcm)?;
        }
        writer.finalize()?;
    }
    Ok(cursor.into_inner())
}

fn load_mel_filters(num_mel_bins: usize) -> Result<Vec<f32>> {
    let mel_bytes = match num_mel_bins {
        80 => include_bytes!("melfilters.bytes").as_slice(),
        128 => include_bytes!("melfilters128.bytes").as_slice(),
        other => return Err(anyhow!("unexpected num_mel_bins {other}")),
    };
    let mut mel_filters = vec![0f32; mel_bytes.len() / 4];
    LittleEndian::read_f32_into(mel_bytes, &mut mel_filters);
    Ok(mel_filters)
}

pub fn token_id(tokenizer: &Tokenizer, token: &str) -> candle_core::Result<u32> {
    match tokenizer.token_to_id(token) {
        None => candle_core::bail!("no token-id for {token}"),
        Some(id) => Ok(id),
    }
}

fn read_wav_mono_f32(path: &Path) -> Result<(Vec<f32>, u32)> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    let channels = spec.channels.max(1) as usize;
    let mut mono = Vec::new();
    match (spec.sample_format, spec.bits_per_sample) {
        (hound::SampleFormat::Float, bits) if bits == 32 => {
            let mut frame = Vec::with_capacity(channels);
            for sample in reader.samples::<f32>() {
                frame.push(sample?);
                if frame.len() == channels {
                    mono.push(frame.iter().copied().sum::<f32>() / channels as f32);
                    frame.clear();
                }
            }
        }
        (hound::SampleFormat::Float, bits) => {
            return Err(anyhow!("unsupported float wav bit depth: {bits}"));
        }
        (hound::SampleFormat::Int, bits) if bits <= 16 => {
            let scale = i16::MAX as f32;
            let mut frame = Vec::with_capacity(channels);
            for sample in reader.samples::<i16>() {
                frame.push(sample? as f32 / scale);
                if frame.len() == channels {
                    mono.push(frame.iter().copied().sum::<f32>() / channels as f32);
                    frame.clear();
                }
            }
        }
        (hound::SampleFormat::Int, _) => {
            let scale = i32::MAX as f32;
            let mut frame = Vec::with_capacity(channels);
            for sample in reader.samples::<i32>() {
                frame.push(sample? as f32 / scale);
                if frame.len() == channels {
                    mono.push(frame.iter().copied().sum::<f32>() / channels as f32);
                    frame.clear();
                }
            }
        }
    }
    Ok((mono, spec.sample_rate))
}

fn active_app() -> String {
    command_output(&["xdotool", "getwindowfocus", "getwindowclassname"])
        .unwrap_or_else(|_| "unknown".to_string())
}

fn active_window_id() -> Option<String> {
    command_output(&["xdotool", "getwindowfocus"]).ok()
}

fn write_clipboard(text: &str) -> Result<()> {
    let mut child = Command::new("xclip")
        .args(["-selection", "clipboard"])
        .stdin(Stdio::piped())
        .spawn()?;
    if let Some(stdin) = child.stdin.as_mut() {
        stdin.write_all(text.as_bytes())?;
    }
    let status = child.wait()?;
    if !status.success() {
        return Err(anyhow!("xclip failed"));
    }
    Ok(())
}

fn paste_text(text: &str, config: &PasteConfig) -> Result<()> {
    write_clipboard(text)?;
    if config.restore_clipboard {
        return Err(anyhow!(
            "restore_clipboard is no longer supported; keep it false"
        ));
    }
    run_command(&["xdotool", "key", "--clearmodifiers", &config.keybinding])?;
    Ok(())
}

fn run_command(args: &[&str]) -> Result<()> {
    let status = Command::new(args[0]).args(&args[1..]).status()?;
    if !status.success() {
        return Err(anyhow!("command failed: {}", args.join(" ")));
    }
    Ok(())
}

fn command_output(args: &[&str]) -> Result<String> {
    let output = Command::new(args[0]).args(&args[1..]).output()?;
    if !output.status.success() {
        return Err(anyhow!("command failed: {}", args.join(" ")));
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

fn key_from_name(name: &str) -> Result<Code> {
    match name.to_ascii_lowercase().as_str() {
        "q" => Ok(Code::KeyQ),
        "v" => Ok(Code::KeyV),
        "escape" | "esc" => Ok(Code::Escape),
        other => Err(anyhow!("unsupported key: {other}")),
    }
}

fn modifiers_from_names(names: &[String]) -> Modifiers {
    let mut modifiers = Modifiers::empty();
    for name in names {
        match name.to_ascii_lowercase().as_str() {
            "ctrl" | "control" => modifiers |= Modifiers::CONTROL,
            "alt" => modifiers |= Modifiers::ALT,
            "shift" => modifiers |= Modifiers::SHIFT,
            "super" | "meta" => modifiers |= Modifiers::SUPER,
            _ => {}
        }
    }
    modifiers
}

fn combine_error_codes(primary: Option<&str>, secondary: Option<&str>) -> Option<String> {
    match (primary, secondary) {
        (Some(left), Some(right)) if left != right => Some(format!("{left}; {right}")),
        (Some(left), _) => Some(left.to_string()),
        (_, Some(right)) => Some(right.to_string()),
        _ => None,
    }
}

fn run_daemon(config: Config, metrics: &MetricsStore) -> Result<()> {
    let mut recorder = Recorder::new(config.audio.clone());
    let mut transcriber = Transcriber::new(config.asr.clone())?;
    let asr_backend_name = transcriber.backend_name().to_string();
    let asr_model_name = transcriber.model_name().to_string();
    let rewriter = Rewriter::new(config.rewrite.clone(), config.developer_profile.clone())?;
    let mut overlay = Overlay::new();
    let manager = GlobalHotKeyManager::new()?;
    let dictate = HotKey::new(
        Some(modifiers_from_names(&config.hotkey.modifiers)),
        key_from_name(&config.hotkey.trigger)?,
    );
    let paste_last_key = HotKey::new(
        Some(modifiers_from_names(
            &config.hotkey.paste_last[..config.hotkey.paste_last.len().saturating_sub(1)],
        )),
        key_from_name(
            config
                .hotkey
                .paste_last
                .last()
                .map(String::as_str)
                .unwrap_or("v"),
        )?,
    );
    let cancel_key = HotKey::new(None, key_from_name(&config.hotkey.cancel)?);
    manager.register(dictate)?;
    manager.register(paste_last_key)?;
    manager.register(cancel_key)?;
    println!("listening for hold-to-talk on Ctrl+Alt+Q");
    let mut last_text = String::new();
    let mut active_trace: Option<SessionTrace> = None;
    let mut active_stream: Option<OpenAiStreamHandle> = None;
    let mut active_app_name = String::from("unknown");
    let mut active_window = None::<String>;

    loop {
        let event = GlobalHotKeyEvent::receiver().recv()?;
        if event.id == dictate.id() {
            match event.state {
                HotKeyState::Pressed => {
                    if active_trace.is_none() {
                        let mut trace = SessionTrace::new();
                        active_app_name = active_app();
                        active_window = active_window_id();
                        trace.mark("hotkey_down");
                        let stream_chunks = config.audio.target_sample_rate_hz == 24_000
                            && matches!(&transcriber, Transcriber::OpenAi(inner) if inner.streaming_enabled());
                        recorder.start_recording(stream_chunks)?;
                        trace.mark("recording_started");
                        active_stream =
                            transcriber.start_streaming(trace.started_at, recorder.take_chunk_rx());
                        println!("Listening...");
                        overlay.show_state("Listening", "", active_window.as_deref());
                        active_trace = Some(trace);
                    }
                }
                HotKeyState::Released => {
                    let Some(mut trace) = active_trace.take() else {
                        continue;
                    };
                    trace.mark("hotkey_up");
                    overlay.show_state("Finalizing", "", active_window.as_deref());
                    let audio = recorder.stop_recording()?;
                    let duration_ms =
                        audio.len() as f64 / config.audio.target_sample_rate_hz as f64 * 1000.0;
                    trace.mark("audio_finalized");
                    if duration_ms < config.audio.min_audio_ms as f64 {
                        if let Some(stream) = active_stream.take() {
                            stream.cancel();
                        }
                        trace.mark("session_done");
                        metrics.record_session(
                            &trace,
                            "ignored",
                            &active_app_name,
                            duration_ms,
                            "",
                            "",
                            false,
                            false,
                            &asr_backend_name,
                            "ignored",
                            false,
                            &asr_model_name,
                            &config.rewrite.model,
                            Some("too_short"),
                            config.metrics.store_transcript_text,
                        )?;
                        println!("Ignored short utterance");
                        overlay.show_notice("Ignored", "", 2, active_window.as_deref());
                        continue;
                    }
                    trace.mark("asr_started");
                    let asr = transcriber.transcribe_after_release(
                        &audio,
                        config.audio.target_sample_rate_hz,
                        active_stream.take(),
                    );
                    if let Some(offset_ms) = asr.stream_ready_ms {
                        trace.mark_at("stream_session_ready", offset_ms);
                    }
                    if let Some(offset_ms) = asr.first_partial_ms {
                        trace.mark_at("first_partial_received", offset_ms);
                    }
                    trace.mark("asr_finished");
                    let raw_text = asr.text;
                    trace.mark("rewrite_started");
                    let rewrite = rewriter.rewrite(&raw_text);
                    trace.mark("rewrite_finished");
                    if rewrite.text.is_empty() {
                        trace.mark("session_done");
                        metrics.record_session(
                            &trace,
                            "ignored",
                            &active_app_name,
                            duration_ms,
                            &raw_text,
                            "",
                            rewrite.used_model,
                            rewrite.fallback_used,
                            &asr_backend_name,
                            asr.mode,
                            asr.fallback_used,
                            &asr_model_name,
                            &config.rewrite.model,
                            combine_error_codes(
                                asr.error_code.as_deref(),
                                rewrite.error_code.as_deref(),
                            )
                            .as_deref()
                            .or(Some("empty_transcript")),
                            config.metrics.store_transcript_text,
                        )?;
                        println!("No speech detected");
                        overlay.show_notice("No speech", "", 2, active_window.as_deref());
                        continue;
                    }
                    trace.mark("paste_started");
                    let copy_result = write_clipboard(&rewrite.text);
                    trace.mark("paste_finished");
                    trace.mark("session_done");
                    last_text = rewrite.text.clone();
                    let combined_error = combine_error_codes(
                        asr.error_code.as_deref(),
                        rewrite.error_code.as_deref(),
                    );
                    let (status, error_code) = if let Err(err) = copy_result {
                        println!("Clipboard copy failed: {err}");
                        overlay.show_notice("Failed", "", 2, active_window.as_deref());
                        ("copy_failed", Some("copy_failed".to_string()))
                    } else {
                        println!("Copied: {}", rewrite.text);
                        overlay.show_notice("Copied", "", 2, active_window.as_deref());
                        ("copied", combined_error)
                    };
                    metrics.record_session(
                        &trace,
                        status,
                        &active_app_name,
                        duration_ms,
                        &raw_text,
                        &rewrite.text,
                        rewrite.used_model,
                        rewrite.fallback_used,
                        &asr_backend_name,
                        asr.mode,
                        asr.fallback_used,
                        &asr_model_name,
                        &config.rewrite.model,
                        error_code.as_deref(),
                        config.metrics.store_transcript_text,
                    )?;
                }
            }
        } else if event.id == paste_last_key.id() && event.state == HotKeyState::Pressed {
            if !last_text.is_empty() {
                let _ = paste_text(&last_text, &config.paste);
                println!("Re-pasted last transcript");
                overlay.show_notice("Retry", "", 2, active_window_id().as_deref());
            }
        } else if event.id == cancel_key.id() && event.state == HotKeyState::Pressed {
            if let Some(mut trace) = active_trace.take() {
                recorder.cancel_recording()?;
                if let Some(stream) = active_stream.take() {
                    stream.cancel();
                }
                trace.mark("cancelled");
                trace.mark("session_done");
                metrics.record_session(
                    &trace,
                    "cancelled",
                    &active_app_name,
                    0.0,
                    "",
                    "",
                    false,
                    false,
                    &asr_backend_name,
                    "cancelled",
                    false,
                    &asr_model_name,
                    &config.rewrite.model,
                    Some("cancelled"),
                    config.metrics.store_transcript_text,
                )?;
                println!("Cancelled current dictation");
                overlay.show_notice("Cancelled", "", 2, active_window.as_deref());
            }
        }
    }
}

#[derive(Debug, Deserialize)]
struct BenchmarkFixture {
    file: String,
    case: Option<String>,
}

fn run_benchmark(config: Config, metrics: &MetricsStore, manifest_path: PathBuf) -> Result<()> {
    let mut transcriber = Transcriber::new(config.asr.clone())?;
    let asr_backend_name = transcriber.backend_name().to_string();
    let asr_model_name = transcriber.model_name().to_string();
    let rewriter = Rewriter::new(config.rewrite.clone(), config.developer_profile.clone())?;
    let manifest: Vec<BenchmarkFixture> =
        serde_json::from_str(&fs::read_to_string(&manifest_path)?)?;
    for fixture in manifest {
        let wav_path = manifest_path.parent().unwrap().join(&fixture.file);
        let (_, sample_rate) = read_wav_mono_f32(&wav_path)?;
        let duration_ms =
            hound::WavReader::open(&wav_path)?.duration() as f64 / sample_rate as f64 * 1000.0;
        let mut trace = SessionTrace::new();
        trace.mark("hotkey_down");
        trace.mark("recording_started");
        trace.mark("hotkey_up");
        trace.mark("audio_finalized");
        trace.mark("asr_started");
        let raw_text = transcriber.transcribe_wav(&wav_path)?;
        trace.mark("asr_finished");
        trace.mark("rewrite_started");
        let rewrite = rewriter.rewrite(&raw_text);
        trace.mark("rewrite_finished");
        trace.mark("paste_started");
        trace.mark("paste_finished");
        trace.mark("session_done");
        let asr_mode = if asr_backend_name == "openai" {
            "openai_file"
        } else {
            "local"
        };
        metrics.record_session(
            &trace,
            "benchmark",
            fixture.case.as_deref().unwrap_or("benchmark"),
            duration_ms,
            &raw_text,
            &rewrite.text,
            rewrite.used_model,
            rewrite.fallback_used,
            &asr_backend_name,
            asr_mode,
            false,
            &asr_model_name,
            &config.rewrite.model,
            rewrite.error_code.as_deref(),
            config.metrics.store_transcript_text,
        )?;
        println!("{}: {}", fixture.file, raw_text);
    }
    metrics.print_stats()?;
    Ok(())
}

fn command_exists(command: &str) -> bool {
    Command::new("sh")
        .args(["-lc", &format!("command -v {command} >/dev/null 2>&1")])
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

fn run_doctor(config: &Config, config_path: &Path, metrics_path: &Path) -> Result<()> {
    println!("Config: {}", config_path.display());
    println!("Metrics DB: {}", metrics_path.display());
    println!("Active app: {}", active_app());
    println!(
        "OpenAI configured: {}",
        env::var_os("OPENAI_API_KEY").is_some()
    );
    println!("ASR backend: {}", config.asr.backend);
    match config.asr.backend_kind()? {
        AsrBackend::Local => {
            let spec = whisper_model_spec(&config.asr.local_model)?;
            println!("Local Whisper model alias: {}", spec.alias);
            println!("Local Whisper repo: {} @ {}", spec.repo_id, spec.revision);
        }
        AsrBackend::OpenAi => {
            println!("OpenAI transcription model: {}", config.asr.openai_model);
            println!("OpenAI streaming enabled: {}", config.asr.streaming_enabled);
            println!("OpenAI transcription timeout ms: {}", config.asr.timeout_ms);
            println!(
                "Audio capture sample rate hz: {}",
                config.audio.target_sample_rate_hz
            );
            println!("Local fallback model alias: {}", config.asr.local_model);
        }
    }
    println!("ffmpeg: {}", command_exists("ffmpeg"));
    println!("xclip: {}", command_exists("xclip"));
    println!("xdotool: {}", command_exists("xdotool"));
    println!("notify-send: {}", command_exists("notify-send"));
    println!("pactl: {}", command_exists("pactl"));
    println!("Audio sources:");
    match Command::new("pactl")
        .args(["list", "short", "sources"])
        .output()
    {
        Ok(out) if out.status.success() => print!("{}", String::from_utf8_lossy(&out.stdout)),
        _ => println!("  pactl unavailable"),
    }
    Ok(())
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let (config, config_path, metrics_path) = load_config(cli.config)?;
    let metrics = MetricsStore::new(metrics_path.clone())?;
    match cli.command {
        Commands::Daemon => run_daemon(config, &metrics),
        Commands::Doctor => run_doctor(&config, &config_path, &metrics_path),
        Commands::Stats => metrics.print_stats(),
        Commands::Recent { limit } => metrics.print_recent(limit),
        Commands::Benchmark { manifest } => {
            run_benchmark(config, &metrics, manifest.unwrap_or(voice_manifest_path()?))
        }
        Commands::DownloadModel => {
            let mut transcriber =
                LocalTranscriber::new(&config.asr.local_model, &config.asr.language)?;
            transcriber.warm_up()?;
            println!("loaded {}", transcriber.spec.repo_id);
            Ok(())
        }
    }
}
