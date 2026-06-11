# NewWave Squeal Recorder

A Streamlit app for recording, tagging, and exporting rail wheel squeal events captured from an ESP32 over serial. Designed as the data-collection front-end for a machine-learning pipeline that automatically classifies squeal type (TOR, Flange, or unrelated noise).

---

## Features

- **Live serial stream** — connects to the ESP32, displays waveform and spectrogram in real time
- **GPS speed** — reads `speed_kmh`, `speed_mps`, or `speed_knots` from telemetry and displays in km/h
- **Auto-triggered clips** — clips are automatically captured when RMS + band-ratio thresholds are exceeded (pre-roll + post-roll)
- **Manual recording** — Start / Stop buttons for free-form recording sessions
- **Tagging** — assign a squeal type to each clip:
  - **TOR** — top-of-rail squeal
  - **Flange** — flange squeal
  - **NotWheel** — noise not related to wheels
- **Per-clip export** — download WAV + JSON metadata sidecar for any clip
- **Batch export** — ZIP archive containing every WAV, its JSON sidecar, and a `catalog.json` array ready for ML import
- **Event map** — GPS-tagged events plotted on a map
- **Offline review** — upload and inspect a previously recorded WAV

---

## Hardware

The app expects an ESP32 sending JSON-lines over serial at configurable baud (default 921600). Each line is one of:

```json
{"type": "audio", "payload_b64": "<base64 PCM int16>", "sample_rate_hz": 16000, "frame_index": 42, "uptime_ms": 12345}
{"type": "telemetry", "lat": 59.912345, "lon": 10.746123, "speed_kmh": 25.3, "utc": "10:05:32.00", "gps_valid": true}
{"type": "button_mark", "lat": 59.912345, "lon": 10.746123, "uptime_ms": 12345}
```

Speed can be supplied as `speed_kmh`, `speed_mps`, or `speed_knots` — the app normalises all to km/h.

---

## Export format

Each clip produces:

**WAV** — 16-bit PCM, sample rate as received from the device
`clip_auto_0001_TOR.wav`

**JSON sidecar**
```json
{
  "clip_id": 1,
  "source": "auto",
  "trigger_reason": "auto_squeal",
  "tag": "TOR",
  "duration_s": 2.48,
  "sample_rate_hz": 16000,
  "rms": 0.09312,
  "lat": 59.912345,
  "lon": 10.746123,
  "speed_kmh": 25.3,
  "utc": "10:05:32.00",
  "gps_valid": true,
  "wav_filename": "clip_auto_0001_TOR.wav"
}
```

The batch ZIP also contains `catalog.json` — an array of all clip metadata — for direct ingestion into a training pipeline.

---

## Setup

```bash
git clone https://github.com/Erneland/NewWave-Recorder.git
cd NewWave-Recorder
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app_squeal_recorder.py
```

The app reads `config_refactored.ini` for all tunable parameters (sample rate, squeal band, RMS threshold, pre/post-roll, map height, etc.).

---

## Configuration

Key settings in `config_refactored.ini`:

| Section | Key | Default | Description |
|---|---|---|---|
| `[audio]` | `sample_rate_hz` | 16000 | Expected sample rate |
| `[detector]` | `squeal_band_low_hz` | 1260 | Lower bound of squeal band |
| `[detector]` | `squeal_band_high_hz` | 2860 | Upper bound of squeal band |
| `[detector]` | `event_rms_threshold` | 0.06 | RMS level to trigger auto capture |
| `[detector]` | `event_band_ratio_threshold` | 0.28 | Band energy ratio to trigger auto capture |
| `[live]` | `buffer_seconds` | 45 | Size of the rolling audio ring buffer |
| `[record]` | `preroll_s` | 0.5 | Pre-trigger audio prepended to auto clips |
| `[record]` | `postroll_s` | 1.2 | Post-trigger tail appended to auto clips |
