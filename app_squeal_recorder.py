import base64
import configparser
import io
import json
import math
import threading
import time
import zipfile
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pydeck as pdk
import serial
from scipy.io import wavfile
from scipy.signal import spectrogram
from serial.tools import list_ports
import streamlit as st


# ============================================================
# Configuration
# ============================================================

@dataclass
class AppConfig:
    app_title: str
    sample_rate_hz: int
    live_refresh_seconds: float
    live_buffer_seconds: float
    live_wave_seconds: float
    spectrogram_seconds: float
    live_plot_points: int
    spectrogram_nperseg: int
    spectrogram_noverlap: int
    squeal_band_low_hz: float
    squeal_band_high_hz: float
    event_rms_threshold: float
    event_band_ratio_threshold: float
    event_min_duration_s: float
    record_preroll_s: float
    record_postroll_s: float
    telemetry_history_rows: int
    map_height_px: int
    theme_accent: str


def load_app_config(path: str = "config_refactored.ini") -> AppConfig:
    parser = configparser.ConfigParser()
    parser.read(path, encoding="utf-8")
    return AppConfig(
        app_title=parser.get("app", "title", fallback="NewWave Squeal Recorder"),
        sample_rate_hz=parser.getint("audio", "sample_rate_hz", fallback=16000),
        live_refresh_seconds=parser.getfloat("live", "refresh_seconds", fallback=0.5),
        live_buffer_seconds=parser.getfloat("live", "buffer_seconds", fallback=30.0),
        live_wave_seconds=parser.getfloat("live", "wave_seconds", fallback=3.0),
        spectrogram_seconds=parser.getfloat("live", "spectrogram_seconds", fallback=3.0),
        live_plot_points=parser.getint("live", "plot_points", fallback=1800),
        spectrogram_nperseg=parser.getint("audio", "spectrogram_nperseg", fallback=256),
        spectrogram_noverlap=parser.getint("audio", "spectrogram_noverlap", fallback=192),
        squeal_band_low_hz=parser.getfloat("detector", "squeal_band_low_hz", fallback=900.0),
        squeal_band_high_hz=parser.getfloat("detector", "squeal_band_high_hz", fallback=3500.0),
        event_rms_threshold=parser.getfloat("detector", "event_rms_threshold", fallback=0.06),
        event_band_ratio_threshold=parser.getfloat("detector", "event_band_ratio_threshold", fallback=0.28),
        event_min_duration_s=parser.getfloat("detector", "event_min_duration_s", fallback=0.20),
        record_preroll_s=parser.getfloat("record", "preroll_s", fallback=0.50),
        record_postroll_s=parser.getfloat("record", "postroll_s", fallback=1.20),
        telemetry_history_rows=parser.getint("live", "telemetry_history_rows", fallback=20),
        map_height_px=parser.getint("map", "height_px", fallback=560),
        theme_accent=parser.get("theme", "accent", fallback="#22d3ee"),
    )


# ============================================================
# Tagging constants
# ============================================================

TAG_OPTIONS = ["Untagged", "TOR", "Flange", "NotWheel"]
TAG_COLORS = {
    "TOR": "#ef4444",
    "Flange": "#f97316",
    "NotWheel": "#6b7280",
    "Untagged": "#22d3ee",
}


# ============================================================
# Helpers
# ============================================================

def _extract_speed_kmh(telemetry: Dict) -> Optional[float]:
    try:
        if telemetry.get("speed_kmh") is not None:
            return float(telemetry["speed_kmh"])
        if telemetry.get("speed_mps") is not None:
            return float(telemetry["speed_mps"]) * 3.6
        if telemetry.get("speed_knots") is not None:
            return float(telemetry["speed_knots"]) * 1.852
    except Exception:
        pass
    return None


# ============================================================
# Audio ring buffer
# ============================================================

class AudioRingBuffer:
    def __init__(self, capacity_samples: int) -> None:
        self.capacity = max(1, int(capacity_samples))
        self.data = np.zeros(self.capacity, dtype=np.int16)
        self.write_pos = 0
        self.size = 0
        self.total_samples_written = 0

    def clear(self) -> None:
        self.data.fill(0)
        self.write_pos = 0
        self.size = 0
        self.total_samples_written = 0

    def append(self, samples: np.ndarray) -> None:
        if samples.size == 0:
            return
        samples = np.asarray(samples, dtype=np.int16).reshape(-1)
        n = samples.size
        self.total_samples_written += n
        if n >= self.capacity:
            self.data[:] = samples[-self.capacity:]
            self.write_pos = 0
            self.size = self.capacity
            return
        end_space = self.capacity - self.write_pos
        if n <= end_space:
            self.data[self.write_pos:self.write_pos + n] = samples
        else:
            self.data[self.write_pos:] = samples[:end_space]
            self.data[:n - end_space] = samples[end_space:]
        self.write_pos = (self.write_pos + n) % self.capacity
        self.size = min(self.capacity, self.size + n)

    def get_last(self, n_samples: int) -> np.ndarray:
        n = min(max(0, int(n_samples)), self.size)
        if n == 0:
            return np.array([], dtype=np.int16)
        start = (self.write_pos - n) % self.capacity
        if start < self.write_pos:
            return self.data[start:self.write_pos].copy()
        return np.concatenate((self.data[start:], self.data[:self.write_pos]))


# ============================================================
# Auto-triggered clip recorder
# ============================================================

class TriggeredClipRecorder:
    def __init__(self, sample_rate_hz: int, preroll_s: float, postroll_s: float) -> None:
        self.sample_rate_hz = int(sample_rate_hz)
        self.preroll_samples = max(0, int(preroll_s * self.sample_rate_hz))
        self.postroll_samples_default = max(0, int(postroll_s * self.sample_rate_hz))
        self.pretrigger = AudioRingBuffer(self.preroll_samples + self.sample_rate_hz)
        self.active_pcm: List[np.ndarray] = []
        self.active_meta: Dict = {}
        self.clip_index = 0
        self.samples_since_trigger = 0
        self.postroll_remaining = 0
        self.is_active = False
        self.captured: Deque[Dict] = deque(maxlen=100)

    def clear(self) -> None:
        self.pretrigger.clear()
        self.active_pcm.clear()
        self.active_meta = {}
        self.samples_since_trigger = 0
        self.postroll_remaining = 0
        self.is_active = False
        self.captured.clear()

    def _start_clip(self, trigger_reason: str, telemetry: Dict, button_mark: bool) -> None:
        self.is_active = True
        self.active_pcm = []
        preroll = self.pretrigger.get_last(self.preroll_samples)
        if preroll.size:
            self.active_pcm.append(preroll)
        self.samples_since_trigger = 0
        self.postroll_remaining = self.postroll_samples_default
        self.clip_index += 1
        self.active_meta = {
            "clip_id": self.clip_index,
            "source": "auto",
            "trigger_reason": trigger_reason,
            "button_mark": int(button_mark),
            "start_wall_time": time.time(),
            "start_uptime_ms": telemetry.get("uptime_ms"),
            "lat": telemetry.get("lat"),
            "lon": telemetry.get("lon"),
            "utc": telemetry.get("utc"),
            "gps_valid": telemetry.get("gps_valid"),
            "speed_kmh": _extract_speed_kmh(telemetry),
            "sample_rate_hz": self.sample_rate_hz,
        }

    def _finalize_clip(self) -> Optional[Dict]:
        if not self.active_pcm:
            self.is_active = False
            self.active_meta = {}
            return None
        pcm = np.concatenate(self.active_pcm).astype(np.int16)
        clip = dict(self.active_meta)
        clip["pcm16"] = pcm
        clip["duration_s"] = len(pcm) / float(self.sample_rate_hz)
        clip["rms"] = float(np.sqrt(np.mean((pcm.astype(np.float32) / 32768.0) ** 2))) if pcm.size else 0.0
        self.captured.appendleft(clip)
        self.is_active = False
        self.active_pcm = []
        self.active_meta = {}
        self.samples_since_trigger = 0
        self.postroll_remaining = 0
        return clip

    def process_frame(
        self,
        pcm: np.ndarray,
        telemetry: Dict,
        trigger_active: bool,
        trigger_reason: str,
        button_mark: bool,
    ) -> Optional[Dict]:
        pcm = np.asarray(pcm, dtype=np.int16).reshape(-1)
        self.pretrigger.append(pcm)

        if trigger_active and not self.is_active:
            self._start_clip(trigger_reason, telemetry, button_mark)

        if self.is_active:
            self.active_pcm.append(pcm.copy())
            self.samples_since_trigger += pcm.size
            if trigger_active:
                self.postroll_remaining = self.postroll_samples_default
            else:
                self.postroll_remaining = max(0, self.postroll_remaining - pcm.size)
                if self.postroll_remaining == 0:
                    return self._finalize_clip()
        return None


# ============================================================
# Live serial manager
# ============================================================

class LiveSerialManager:
    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self._ser: Optional[serial.Serial] = None
        self._reader_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._telemetry_buffer: Deque[Dict] = deque(maxlen=4000)
        self._button_buffer: Deque[Dict] = deque(maxlen=1000)
        self._event_rows: Deque[Dict] = deque(maxlen=2000)
        self._error_buffer: Deque[str] = deque(maxlen=100)
        self.sample_rate_hz = cfg.sample_rate_hz
        self.audio_ring = AudioRingBuffer(int(cfg.sample_rate_hz * cfg.live_buffer_seconds))
        self.clip_recorder = TriggeredClipRecorder(
            sample_rate_hz=cfg.sample_rate_hz,
            preroll_s=cfg.record_preroll_s,
            postroll_s=cfg.record_postroll_s,
        )
        # Manual recording state
        self.recording_active = False
        self.record_chunks: List[np.ndarray] = []
        self.record_start_telemetry: Dict = {}
        self.manual_recordings: Deque[Dict] = deque(maxlen=100)
        self.manual_record_index = 0

        self.connected_port: Optional[str] = None
        self.connected_baud: Optional[int] = None
        self.last_packet_time: Optional[float] = None
        self.last_frame_metrics: Dict = {}
        self.latest_telemetry: Dict = {}
        self.last_spectrogram_update_t = 0.0
        self.last_spectrogram: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None

    def connect(self, port: str, baudrate: int, timeout_s: float = 0.05) -> Tuple[bool, str]:
        with self._lock:
            self.disconnect()
            try:
                self._ser = serial.Serial(port=port, baudrate=baudrate, timeout=timeout_s)
            except Exception as exc:
                return False, f"Open failed: {exc}"
            self.connected_port = port
            self.connected_baud = baudrate
            self._stop_event.clear()
            self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
            self._reader_thread.start()
            return True, f"Connected to {port} at {baudrate} baud"

    def disconnect(self) -> None:
        self._stop_event.set()
        if self._reader_thread is not None and self._reader_thread.is_alive():
            self._reader_thread.join(timeout=0.5)
        self._reader_thread = None
        if self._ser is not None:
            try:
                self._ser.close()
            except Exception:
                pass
        self._ser = None
        self.connected_port = None
        self.connected_baud = None

    def clear_buffers(self) -> None:
        with self._lock:
            self.audio_ring.clear()
            self.clip_recorder.clear()
            self._telemetry_buffer.clear()
            self._button_buffer.clear()
            self._event_rows.clear()
            self._error_buffer.clear()
            self.last_frame_metrics = {}
            self.latest_telemetry = {}
            self.last_packet_time = None
            self.last_spectrogram = None
            self.last_spectrogram_update_t = 0.0
            self.recording_active = False
            self.record_chunks = []
            self.record_start_telemetry = {}

    def is_connected(self) -> bool:
        return self._ser is not None and self._ser.is_open

    def start_recording(self) -> None:
        with self._lock:
            self.recording_active = True
            self.record_chunks = []
            self.record_start_telemetry = dict(self.latest_telemetry)

    def stop_recording(self) -> Optional[Dict]:
        with self._lock:
            self.recording_active = False
            if not self.record_chunks:
                return None
            pcm = np.concatenate(self.record_chunks).astype(np.int16)
            self.record_chunks = []
            telem = self.record_start_telemetry
            self.manual_record_index += 1
            rec = {
                "clip_id": self.manual_record_index,
                "source": "manual",
                "trigger_reason": "manual",
                "button_mark": 0,
                "start_wall_time": time.time(),
                "lat": telem.get("lat"),
                "lon": telem.get("lon"),
                "utc": telem.get("utc"),
                "gps_valid": telem.get("gps_valid"),
                "speed_kmh": _extract_speed_kmh(telem),
                "sample_rate_hz": self.sample_rate_hz,
                "pcm16": pcm,
                "duration_s": len(pcm) / float(self.sample_rate_hz) if pcm.size else 0.0,
                "rms": float(np.sqrt(np.mean((pcm.astype(np.float32) / 32768.0) ** 2))) if pcm.size else 0.0,
            }
            self.manual_recordings.appendleft(rec)
            return rec

    def _reader_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                if self._ser is None:
                    time.sleep(0.02)
                    continue
                line = self._ser.readline()
                if not line:
                    continue
                self.last_packet_time = time.time()
                try:
                    packet = json.loads(line.decode("utf-8", errors="replace"))
                except json.JSONDecodeError:
                    self._error_buffer.append("Non-JSON serial packet received")
                    continue
                self._handle_packet(packet)
            except Exception as exc:
                self._error_buffer.append(f"Reader error: {exc}")
                time.sleep(0.1)

    def _handle_packet(self, packet: Dict) -> None:
        packet_type = str(packet.get("type", ""))
        with self._lock:
            if packet_type == "audio":
                payload_b64 = packet.get("payload_b64")
                if not payload_b64:
                    return
                pcm = np.frombuffer(base64.b64decode(payload_b64), dtype=np.int16).copy()
                sr = int(packet.get("sample_rate_hz", self.sample_rate_hz))
                if sr != self.sample_rate_hz:
                    self.sample_rate_hz = sr
                self.audio_ring.append(pcm)
                if self.recording_active:
                    self.record_chunks.append(pcm.copy())
                metrics = compute_frame_metrics(
                    pcm.astype(np.float32) / 32768.0,
                    self.sample_rate_hz,
                    self.cfg.squeal_band_low_hz,
                    self.cfg.squeal_band_high_hz,
                )
                metrics["frame_index"] = int(packet.get("frame_index", -1))
                metrics["uptime_ms"] = packet.get("uptime_ms")
                self.last_frame_metrics = metrics
                trigger_active = (
                    metrics["rms"] >= self.cfg.event_rms_threshold
                    and metrics["band_ratio"] >= self.cfg.event_band_ratio_threshold
                )
                finalized = self.clip_recorder.process_frame(
                    pcm=pcm,
                    telemetry=self.latest_telemetry,
                    trigger_active=trigger_active,
                    trigger_reason="auto_squeal",
                    button_mark=False,
                )
                if finalized is not None:
                    self._event_rows.append({
                        "event_type": "auto_clip",
                        "clip_id": finalized["clip_id"],
                        "source": "auto",
                        "duration_s": finalized["duration_s"],
                        "rms": finalized["rms"],
                        "lat": finalized.get("lat"),
                        "lon": finalized.get("lon"),
                        "utc": finalized.get("utc"),
                        "trigger_reason": finalized.get("trigger_reason"),
                    })
            elif packet_type == "telemetry":
                self.latest_telemetry = packet.copy()
                self._telemetry_buffer.append(packet.copy())
            elif packet_type == "button_mark":
                self._button_buffer.append(packet.copy())
                self.latest_telemetry = {**self.latest_telemetry, **packet}
                self._event_rows.append({
                    "event_type": "button_mark",
                    "clip_id": None,
                    "source": "button",
                    "duration_s": 0.0,
                    "rms": self.last_frame_metrics.get("rms"),
                    "lat": packet.get("lat"),
                    "lon": packet.get("lon"),
                    "utc": packet.get("utc"),
                    "trigger_reason": "button_mark",
                })
                self.clip_recorder.process_frame(
                    pcm=np.array([], dtype=np.int16),
                    telemetry=packet.copy(),
                    trigger_active=True,
                    trigger_reason="button_mark",
                    button_mark=True,
                )
            else:
                self._telemetry_buffer.append(packet.copy())

    def latest_audio_seconds(self, seconds: float) -> Tuple[int, np.ndarray]:
        with self._lock:
            return self.sample_rate_hz, self.audio_ring.get_last(int(seconds * self.sample_rate_hz))

    def telemetry_df(self) -> pd.DataFrame:
        with self._lock:
            return pd.DataFrame(list(self._telemetry_buffer))

    def button_df(self) -> pd.DataFrame:
        with self._lock:
            return pd.DataFrame(list(self._button_buffer))

    def event_df(self) -> pd.DataFrame:
        with self._lock:
            rows = list(self._event_rows)
            return pd.DataFrame(rows) if rows else pd.DataFrame()

    def all_clips(self) -> List[Dict]:
        """Return all clips (auto-triggered + manual) newest first."""
        with self._lock:
            auto = list(self.clip_recorder.captured)
            manual = list(self.manual_recordings)
        combined = auto + manual
        combined.sort(key=lambda c: c.get("start_wall_time", 0.0), reverse=True)
        return combined

    def get_clip(self, source: str, clip_id: int) -> Optional[Dict]:
        with self._lock:
            pool = self.clip_recorder.captured if source == "auto" else self.manual_recordings
            for clip in pool:
                if int(clip["clip_id"]) == int(clip_id):
                    return clip
        return None

    def latest_status(self) -> Dict:
        with self._lock:
            return {
                "connected": self.is_connected(),
                "port": self.connected_port,
                "baud": self.connected_baud,
                "sample_rate_hz": self.sample_rate_hz,
                "last_packet_age_s": None if self.last_packet_time is None else max(0.0, time.time() - self.last_packet_time),
                "last_frame_metrics": dict(self.last_frame_metrics),
                "latest_telemetry": dict(self.latest_telemetry),
                "telemetry_count": len(self._telemetry_buffer),
                "button_count": len(self._button_buffer),
                "auto_clip_count": len(self.clip_recorder.captured),
                "manual_clip_count": len(self.manual_recordings),
                "clip_active": self.clip_recorder.is_active,
                "recording_active": self.recording_active,
                "errors": list(self._error_buffer),
            }

    def get_cached_spectrogram(self) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        now = time.time()
        with self._lock:
            if self.last_spectrogram is not None and (now - self.last_spectrogram_update_t) < 0.8:
                return self.last_spectrogram
            pcm = self.audio_ring.get_last(int(self.cfg.spectrogram_seconds * self.sample_rate_hz))
            if pcm.size < 128:
                return None
            signal = pcm.astype(np.float32) / 32768.0
            spec = compute_spectrogram(signal, self.sample_rate_hz, self.cfg.spectrogram_nperseg, self.cfg.spectrogram_noverlap)
            self.last_spectrogram = spec
            self.last_spectrogram_update_t = now
            return spec


@st.cache_resource
def get_live_manager(cfg: AppConfig) -> LiveSerialManager:
    return LiveSerialManager(cfg)


# ============================================================
# Signal processing
# ============================================================

def compute_frame_metrics(signal: np.ndarray, sr: int, band_low: float, band_high: float) -> Dict:
    if signal.size == 0:
        return {"rms": 0.0, "peak": 0.0, "band_ratio": 0.0, "dominant_freq_hz": 0.0}
    rms = float(np.sqrt(np.mean(signal ** 2)))
    peak = float(np.max(np.abs(signal)))
    spectrum = np.fft.rfft(signal * np.hanning(signal.size))
    power = np.abs(spectrum) ** 2
    freqs = np.fft.rfftfreq(signal.size, d=1.0 / sr)
    total_power = float(np.sum(power) + 1e-12)
    band_mask = (freqs >= band_low) & (freqs <= band_high)
    band_ratio = float(np.sum(power[band_mask])) / total_power
    dominant_freq_hz = float(freqs[np.argmax(power)]) if freqs.size else 0.0
    return {"rms": rms, "peak": peak, "band_ratio": band_ratio, "dominant_freq_hz": dominant_freq_hz}


def compute_spectrogram(signal: np.ndarray, sr: int, nperseg: int, noverlap: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    nperseg = max(64, min(int(nperseg), len(signal)))
    noverlap = max(0, min(int(noverlap), nperseg - 1))
    freqs, times_s, spec = spectrogram(signal, fs=sr, nperseg=nperseg, noverlap=noverlap, scaling="density", mode="magnitude")
    return freqs, times_s, 20.0 * np.log10(spec + 1e-12)


def minmax_envelope(signal: np.ndarray, sr: int, seconds: float, max_points: int) -> Tuple[np.ndarray, np.ndarray]:
    if signal.size == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)
    max_samples = int(seconds * sr)
    visible = signal[-max_samples:] if max_samples > 0 else signal
    n = visible.size
    if n <= max_points:
        x = np.arange(n, dtype=np.float32) / sr
        return x, visible.astype(np.float32)
    bins = max(10, max_points // 2)
    chunk = int(math.ceil(n / bins))
    xs: List[float] = []
    ys: List[float] = []
    for idx in range(0, n, chunk):
        seg = visible[idx:idx + chunk]
        if seg.size == 0:
            continue
        x = idx / sr
        xs.extend([x, x])
        ys.extend([float(seg.min()), float(seg.max())])
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


# ============================================================
# Export helpers
# ============================================================

def clip_to_wav_bytes(clip: Dict) -> bytes:
    pcm = np.asarray(clip["pcm16"], dtype=np.int16)
    buf = io.BytesIO()
    wavfile.write(buf, int(clip["sample_rate_hz"]), pcm)
    return buf.getvalue()


def clip_to_json_bytes(clip: Dict, tag: str) -> bytes:
    wav_filename = _clip_wav_filename(clip, tag)
    meta = {
        "clip_id": clip["clip_id"],
        "source": clip.get("source", "auto"),
        "trigger_reason": clip.get("trigger_reason"),
        "tag": tag,
        "duration_s": round(float(clip.get("duration_s", 0.0)), 4),
        "sample_rate_hz": int(clip.get("sample_rate_hz", 0)),
        "rms": round(float(clip.get("rms", 0.0)), 5),
        "lat": clip.get("lat"),
        "lon": clip.get("lon"),
        "utc": clip.get("utc"),
        "gps_valid": clip.get("gps_valid"),
        "speed_kmh": round(float(clip["speed_kmh"]), 2) if clip.get("speed_kmh") is not None else None,
        "wav_filename": wav_filename,
    }
    return json.dumps(meta, indent=2).encode("utf-8")


def _clip_wav_filename(clip: Dict, tag: str) -> str:
    src = clip.get("source", "auto")
    cid = int(clip["clip_id"])
    return f"clip_{src}_{cid:04d}_{tag}.wav"


def build_export_zip(clips: List[Dict], tags: Dict[str, str]) -> bytes:
    """Build a ZIP containing WAV + JSON per clip and a catalog.json."""
    buf = io.BytesIO()
    catalog = []
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for clip in clips:
            key = _clip_key(clip)
            tag = tags.get(key, "Untagged")
            wav_name = _clip_wav_filename(clip, tag)
            json_name = wav_name.replace(".wav", ".json")
            zf.writestr(wav_name, clip_to_wav_bytes(clip))
            meta_bytes = clip_to_json_bytes(clip, tag)
            zf.writestr(json_name, meta_bytes)
            catalog.append(json.loads(meta_bytes.decode("utf-8")))
        zf.writestr("catalog.json", json.dumps(catalog, indent=2).encode("utf-8"))
    return buf.getvalue()


def _clip_key(clip: Dict) -> str:
    return f"{clip.get('source', 'auto')}_{clip['clip_id']}"


# ============================================================
# Plots
# ============================================================

def inject_custom_css(accent: str) -> None:
    st.markdown(
        f"""
        <style>
        :root {{
            --accent: {accent};
            --panel-bg: rgba(8, 18, 28, 0.92);
            --panel-border: rgba(34, 211, 238, 0.18);
            --text-main: #d9f7ff;
            --text-soft: #9ec8d5;
        }}
        .stApp {{
            background:
                radial-gradient(circle at top right, rgba(34,211,238,0.08), transparent 28%),
                linear-gradient(180deg, #061018 0%, #091723 42%, #08131d 100%);
            color: var(--text-main);
        }}
        .block-container {{
            max-width: 1550px;
            padding-top: 1rem;
            padding-bottom: 1rem;
        }}
        div[data-testid="stMetric"] {{
            background: var(--panel-bg);
            border: 1px solid var(--panel-border);
            padding: 0.65rem 0.8rem;
            border-radius: 12px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def build_waveform_figure(signal: np.ndarray, sr: int, seconds: float, max_points: int, title: str = "Waveform") -> go.Figure:
    x, y = minmax_envelope(signal, sr, seconds, max_points)
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=x, y=y, mode="lines", name="Amplitude"))
    fig.update_layout(
        title=title,
        template="plotly_dark",
        xaxis_title="Time [s]",
        yaxis_title="Amplitude",
        height=270,
        margin=dict(l=10, r=10, t=40, b=10),
        uirevision=title,
    )
    return fig


def build_spectrogram_figure(freqs: np.ndarray, times_s: np.ndarray, spec_db: np.ndarray, band_low: float, band_high: float) -> go.Figure:
    fig = go.Figure(data=go.Heatmap(x=times_s, y=freqs, z=spec_db, colorscale="Turbo", colorbar=dict(title="dB")))
    fig.add_hrect(y0=band_low, y1=band_high, line_width=1, line_dash="dash", line_color="cyan", opacity=0.15)
    fig.update_layout(
        title="Spectrogram",
        template="plotly_dark",
        xaxis_title="Time [s]",
        yaxis_title="Frequency [Hz]",
        height=330,
        margin=dict(l=10, r=10, t=40, b=10),
        uirevision="spec",
    )
    return fig


def build_map(df: pd.DataFrame, height_px: int) -> pdk.Deck:
    if df.empty or "lat" not in df.columns or "lon" not in df.columns:
        df = pd.DataFrame({"lat": [59.9127], "lon": [10.7461], "rms": [0.1], "trigger_reason": ["n/a"]})
    center_lat = float(df["lat"].astype(float).mean())
    center_lon = float(df["lon"].astype(float).mean())
    plot_df = df.copy()
    plot_df["rms"] = pd.to_numeric(plot_df.get("rms", 0.1), errors="coerce").fillna(0.1)
    plot_df["radius"] = np.clip(80 + 1500 * plot_df["rms"], 80, 900)
    is_button = plot_df.get("trigger_reason", pd.Series([""] * len(plot_df))).astype(str).str.contains("button")
    plot_df["color_r"] = np.where(is_button, 255, 64)
    plot_df["color_g"] = np.where(is_button, 160, 225)
    plot_df["color_b"] = np.where(is_button, 90, 255)
    layer = pdk.Layer(
        "ScatterplotLayer",
        plot_df,
        get_position="[lon, lat]",
        get_radius="radius",
        get_fill_color="[color_r, color_g, color_b, 170]",
        pickable=True,
    )
    return pdk.Deck(
        map_style="dark",
        initial_view_state=pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=13, pitch=30),
        layers=[layer],
        tooltip={"text": "Reason: {trigger_reason}\nRMS: {rms}\nLat: {lat}\nLon: {lon}\nUTC: {utc}"},
        height=height_px,
    )


# ============================================================
# Live fragment (auto-refreshing)
# ============================================================

@st.fragment(run_every=0.5)
def render_live_fragment(cfg: AppConfig) -> None:
    manager = get_live_manager(cfg)
    status = manager.latest_status()
    sr, pcm = manager.latest_audio_seconds(cfg.live_wave_seconds)
    signal = pcm.astype(np.float32) / 32768.0 if pcm.size else np.array([], dtype=np.float32)

    # Row 1 — connection + clip counts
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Serial", "Connected" if status["connected"] else "Disconnected")
    c2.metric("Rate", f"{status['sample_rate_hz']} Hz")
    c3.metric("Auto clips", int(status["auto_clip_count"]))
    c4.metric("Manual clips", int(status["manual_clip_count"]))
    c5.metric("Buttons", int(status["button_count"]))
    c6.metric("Link age", "n/a" if status["last_packet_age_s"] is None else f"{status['last_packet_age_s']:.2f} s")

    # Row 2 — GPS + audio frame metrics
    latest_telem = status.get("latest_telemetry", {}) or {}
    gps_valid = bool(latest_telem.get("gps_valid", False))
    lat_val = latest_telem.get("lat")
    lon_val = latest_telem.get("lon")
    utc_val = latest_telem.get("utc")
    speed_kmh = _extract_speed_kmh(latest_telem)
    speed_display = f"{speed_kmh:.1f} km/h" if speed_kmh is not None else "n/a"

    fm = status.get("last_frame_metrics", {})
    g1, g2, g3, g4, g5, g6, g7, g8 = st.columns(8)
    g1.metric("LAT", "n/a" if lat_val is None else f"{float(lat_val):.5f}")
    g2.metric("LON", "n/a" if lon_val is None else f"{float(lon_val):.5f}")
    g3.metric("Speed", speed_display)
    g4.metric("UTC", str(utc_val)[-8:] if utc_val else ("No fix" if not gps_valid else "n/a"))
    g5.metric("RMS", f"{float(fm.get('rms', 0.0)):.4f}")
    g6.metric("Band ratio", f"{float(fm.get('band_ratio', 0.0)):.3f}")
    g7.metric("Peak", f"{float(fm.get('peak', 0.0)):.3f}")
    g8.metric("Dom. Hz", f"{float(fm.get('dominant_freq_hz', 0.0)):.0f}")

    if status["recording_active"]:
        st.warning("RECORDING — click **Stop recording** to save.")

    left, right = st.columns(2)
    with left:
        if signal.size:
            st.plotly_chart(build_waveform_figure(signal, sr, cfg.live_wave_seconds, cfg.live_plot_points, title="Live waveform"), use_container_width=True)
        else:
            st.info("Waiting for audio frames.")
    with right:
        spec = manager.get_cached_spectrogram()
        if spec is not None:
            freqs, times_s, spec_db = spec
            st.plotly_chart(build_spectrogram_figure(freqs, times_s, spec_db, cfg.squeal_band_low_hz, cfg.squeal_band_high_hz), use_container_width=True)
        else:
            st.info("Waiting for enough samples for spectrogram.")


# ============================================================
# Live tab
# ============================================================

def render_live_tab(cfg: AppConfig) -> None:
    st.subheader("Live stream and recording")
    manager = get_live_manager(cfg)

    # Connection controls
    ports = [p.device for p in list_ports.comports()]
    conn_cols = st.columns([2.4, 1.2, 1.0, 1.0, 1.0, 1.0])
    with conn_cols[0]:
        selected_port = st.selectbox("Serial port", options=ports if ports else [""], index=0)
    with conn_cols[1]:
        baud = st.selectbox("Baud", options=[921600, 460800, 230400, 115200], index=0)
    with conn_cols[2]:
        if st.button("Connect", use_container_width=True):
            ok, msg = manager.connect(selected_port, int(baud))
            if ok:
                st.success(msg)
            else:
                st.error(msg)
    with conn_cols[3]:
        if st.button("Disconnect", use_container_width=True):
            manager.disconnect()
            st.info("Disconnected.")
    with conn_cols[4]:
        if st.button("Clear", use_container_width=True):
            manager.clear_buffers()
            st.info("Buffers cleared.")
    with conn_cols[5]:
        if st.button("Refresh ports", use_container_width=True):
            st.rerun()

    # Manual recording controls
    st.divider()
    rec_cols = st.columns([1.0, 1.0, 2.0])
    with rec_cols[0]:
        if st.button("Start recording", type="primary", use_container_width=True):
            manager.start_recording()
            st.success("Recording started.")
    with rec_cols[1]:
        if st.button("Stop recording", use_container_width=True):
            rec = manager.stop_recording()
            if rec is None:
                st.warning("No audio was captured.")
            else:
                st.success(f"Saved manual clip {rec['clip_id']} — {rec['duration_s']:.2f} s, RMS {rec['rms']:.4f}")
    with rec_cols[2]:
        st.caption("Auto clips are captured automatically when RMS + band-ratio thresholds are exceeded. Manual recording captures continuously between Start and Stop.")

    render_live_fragment(cfg)

    # ---- Clip browser --------------------------------------------------
    st.divider()
    st.subheader("Clips")

    clips = manager.all_clips()
    if not clips:
        st.info("No clips yet. Auto clips are created when squeal is detected; use Start/Stop to record manually.")
        return

    # Ensure tag state exists
    if "clip_tags" not in st.session_state:
        st.session_state.clip_tags = {}

    # Build summary table
    rows = []
    for clip in clips:
        key = _clip_key(clip)
        tag = st.session_state.clip_tags.get(key, "Untagged")
        rows.append({
            "key": key,
            "source": clip.get("source", "auto"),
            "clip_id": clip["clip_id"],
            "trigger": clip.get("trigger_reason", ""),
            "duration_s": round(float(clip.get("duration_s", 0.0)), 2),
            "rms": round(float(clip.get("rms", 0.0)), 4),
            "speed_kmh": round(float(clip["speed_kmh"]), 1) if clip.get("speed_kmh") is not None else None,
            "lat": clip.get("lat"),
            "lon": clip.get("lon"),
            "utc": clip.get("utc"),
            "tag": tag,
        })

    summary_df = pd.DataFrame(rows)
    st.dataframe(
        summary_df.drop(columns=["key"]),
        use_container_width=True,
        hide_index=True,
        column_config={
            "tag": st.column_config.TextColumn("Tag"),
            "speed_kmh": st.column_config.NumberColumn("Speed (km/h)", format="%.1f"),
        },
    )

    # Batch export
    if st.button("Export all clips (ZIP)", use_container_width=False):
        zip_bytes = build_export_zip(clips, st.session_state.clip_tags)
        st.download_button(
            "Download ZIP",
            data=zip_bytes,
            file_name="squeal_clips_export.zip",
            mime="application/zip",
        )

    st.divider()
    st.subheader("Inspect / tag clip")

    # Clip selector
    clip_keys = [_clip_key(c) for c in clips]
    clip_labels = [
        f"{c.get('source','auto')} #{c['clip_id']} | {c.get('duration_s',0):.2f}s | {c.get('trigger_reason','')}"
        for c in clips
    ]
    selected_idx = st.selectbox("Select clip", options=range(len(clips)), format_func=lambda i: clip_labels[i])
    clip = clips[selected_idx]
    clip_key = clip_keys[selected_idx]

    # Tag selector
    current_tag = st.session_state.clip_tags.get(clip_key, "Untagged")
    tag_idx = TAG_OPTIONS.index(current_tag) if current_tag in TAG_OPTIONS else 0
    chosen_tag = st.radio(
        "Squeal type",
        options=TAG_OPTIONS,
        index=tag_idx,
        horizontal=True,
        help="TOR = top-of-rail squeal · Flange = flange squeal · NotWheel = noise not related to wheels",
    )
    if chosen_tag != current_tag:
        st.session_state.clip_tags[clip_key] = chosen_tag

    # Clip details
    speed_str = f"{clip['speed_kmh']:.1f} km/h" if clip.get("speed_kmh") is not None else "n/a"
    d1, d2, d3, d4, d5 = st.columns(5)
    d1.metric("Duration", f"{float(clip.get('duration_s', 0)):.2f} s")
    d2.metric("RMS", f"{float(clip.get('rms', 0)):.4f}")
    d3.metric("GPS speed", speed_str)
    d4.metric("LAT", f"{float(clip['lat']):.5f}" if clip.get("lat") else "n/a")
    d5.metric("LON", f"{float(clip['lon']):.5f}" if clip.get("lon") else "n/a")

    clip_signal = clip["pcm16"].astype(np.float32) / 32768.0
    st.plotly_chart(
        build_waveform_figure(clip_signal, int(clip["sample_rate_hz"]), float(clip["duration_s"]), cfg.live_plot_points, title=f"Clip waveform — {chosen_tag}"),
        use_container_width=True,
    )
    freqs, times_s, spec_db = compute_spectrogram(clip_signal, int(clip["sample_rate_hz"]), cfg.spectrogram_nperseg, cfg.spectrogram_noverlap)
    st.plotly_chart(
        build_spectrogram_figure(freqs, times_s, spec_db, cfg.squeal_band_low_hz, cfg.squeal_band_high_hz),
        use_container_width=True,
    )

    # Per-clip downloads
    dl1, dl2 = st.columns(2)
    wav_name = _clip_wav_filename(clip, chosen_tag)
    with dl1:
        st.download_button(
            "Download WAV",
            data=clip_to_wav_bytes(clip),
            file_name=wav_name,
            mime="audio/wav",
            use_container_width=True,
        )
    with dl2:
        st.download_button(
            "Download JSON metadata",
            data=clip_to_json_bytes(clip, chosen_tag),
            file_name=wav_name.replace(".wav", ".json"),
            mime="application/json",
            use_container_width=True,
        )

    # Diagnostics expander
    with st.expander("Telemetry and diagnostics"):
        tdf = manager.telemetry_df()
        bdf = manager.button_df()
        if not tdf.empty:
            st.dataframe(tdf.tail(cfg.telemetry_history_rows), use_container_width=True)
        if not bdf.empty:
            st.dataframe(bdf.tail(cfg.telemetry_history_rows), use_container_width=True)
        errors = manager.latest_status().get("errors", [])
        if errors:
            st.write(errors[-20:])


# ============================================================
# Map tab
# ============================================================

def render_map_tab(cfg: AppConfig) -> None:
    st.subheader("Event map")
    manager = get_live_manager(cfg)
    df = manager.event_df()
    if df.empty:
        st.info("No GPS-tagged events yet.")
        return
    for col in ("lat", "lon"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["lat", "lon"])
    if df.empty:
        st.info("Events exist, but none have valid GPS coordinates yet.")
        return
    st.pydeck_chart(build_map(df, cfg.map_height_px), use_container_width=True)
    st.dataframe(df, use_container_width=True, hide_index=True)


# ============================================================
# Offline import tab
# ============================================================

def render_offline_import_tab(cfg: AppConfig) -> None:
    st.subheader("Offline review")
    uploaded = st.file_uploader("Upload captured WAV", type=["wav"])
    if uploaded is None:
        return
    sr, signal = wavfile.read(io.BytesIO(uploaded.read()))
    if signal.ndim > 1:
        signal = signal[:, 0]
    if np.issubdtype(signal.dtype, np.integer):
        signal = signal.astype(np.float32) / float(np.iinfo(signal.dtype).max)
    else:
        signal = signal.astype(np.float32)
    metrics = compute_frame_metrics(signal, int(sr), cfg.squeal_band_low_hz, cfg.squeal_band_high_hz)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("RMS", f"{metrics['rms']:.4f}")
    c2.metric("Band ratio", f"{metrics['band_ratio']:.3f}")
    c3.metric("Peak", f"{metrics['peak']:.3f}")
    c4.metric("Dominant Hz", f"{metrics['dominant_freq_hz']:.0f}")
    st.plotly_chart(build_waveform_figure(signal, int(sr), len(signal) / float(sr), cfg.live_plot_points), use_container_width=True)
    freqs, times_s, spec_db = compute_spectrogram(signal, int(sr), cfg.spectrogram_nperseg, cfg.spectrogram_noverlap)
    st.plotly_chart(build_spectrogram_figure(freqs, times_s, spec_db, cfg.squeal_band_low_hz, cfg.squeal_band_high_hz), use_container_width=True)


# ============================================================
# Main
# ============================================================

def main() -> None:
    cfg = load_app_config("config_refactored.ini")
    st.set_page_config(page_title=cfg.app_title, page_icon="🎙️", layout="wide")
    inject_custom_css(cfg.theme_accent)
    st.title(cfg.app_title)
    st.caption("Live squeal recorder — manual/auto capture, GPS speed, TOR/Flange/NotWheel tagging, WAV + JSON export for ML.")

    if "clip_tags" not in st.session_state:
        st.session_state.clip_tags = {}

    tabs = st.tabs(["Live / Record", "Map", "Offline review"])
    with tabs[0]:
        render_live_tab(cfg)
    with tabs[1]:
        render_map_tab(cfg)
    with tabs[2]:
        render_offline_import_tab(cfg)


if __name__ == "__main__":
    main()
