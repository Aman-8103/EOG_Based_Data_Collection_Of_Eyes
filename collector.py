#!/usr/bin/env python3
"""
iSync — EOG Based Multi-Device Assistive Control
Data Collection & Labeling Engine (collector.py)

Features:
- Dual hardware support: Real Arduino Serial (115200 baud, 250 Hz) or Mock Hardware Stream
- Strict Subject (S01, S02) and per-subject Attempt (A01, A02...) tracking
- 20-second synchronized recording with 3-2-1-START countdown
- Independent timestamp (restarts from 0.000s) and Sample_Index (restarts from 1)
- Raw CSV storage in 'New Data/{Subject_ID}_{Attempt_ID}.csv'
- Full-recording PNG generation in 'New Data/{Subject_ID}_{Attempt_ID}.png'
- Data quality verification (duration, sample count, monotonicity, numeric values)
- Manual Quality Check: SELECT, REJECT, RECORD AGAIN
- Final dataset curation in 'LabelData/Eog Data.csv' with duplicate protection & replacement option
- Automatic ground-truth label mapping based on stimulus timeline (REST, RIGHT, LEFT, UP, DOWN)
- Built-in HTTP server serving index.html and REST API for synchronized browser stimulus
"""

import os
import time
import math
import glob
import json
import random
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from datetime import datetime
import pandas as pd  # pyrefly: ignore [missing-import]
import numpy as np  # pyrefly: ignore [missing-import]
import matplotlib  # pyrefly: ignore [missing-import]
matplotlib.use('Agg')  # Non-interactive backend for server generation
import matplotlib.pyplot as plt  # pyrefly: ignore [missing-import]

try:
    import serial
    import serial.tools.list_ports
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False

# ==============================================================================
# CONFIGURATION & PROTOCOLS
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NEW_DATA_DIR = os.path.join(BASE_DIR, "New Data Plots") if os.path.isdir(os.path.join(BASE_DIR, "New Data Plots")) else os.path.join(BASE_DIR, "New Data")
LABEL_DATA_DIR = os.path.join(BASE_DIR, "LabelData")
FINAL_CSV_PATH = os.path.join(LABEL_DATA_DIR, "Eog Data.csv")
METADATA_CSV_PATH = os.path.join(LABEL_DATA_DIR, "trials_metadata.csv")

TARGET_DURATION = 24.0  # seconds
EXPECTED_FS = 250.0     # Hz
SERIAL_BAUD = 115200

# Deterministic Stimulus Sequences (24s / 12 phases / 2.0s each)
SEQUENCES = {
    "Sequence_A": {
        "id": "Sequence_A",
        "description": "Standard: REST -> RIGHT -> LEFT -> REST -> UP -> DOWN -> REST -> RIGHT -> LEFT -> REST -> UP -> DOWN",
        "timeline": [
            {"start": 0.0,  "end": 2.0,  "label": "REST",  "target": "center"},
            {"start": 2.0,  "end": 4.0,  "label": "RIGHT", "target": "right"},
            {"start": 4.0,  "end": 6.0,  "label": "LEFT",  "target": "left"},
            {"start": 6.0,  "end": 8.0,  "label": "REST",  "target": "center"},
            {"start": 8.0,  "end": 10.0, "label": "UP",    "target": "up"},
            {"start": 10.0, "end": 12.0, "label": "DOWN",  "target": "down"},
            {"start": 12.0, "end": 14.0, "label": "REST",  "target": "center"},
            {"start": 14.0, "end": 16.0, "label": "RIGHT", "target": "right"},
            {"start": 16.0, "end": 18.0, "label": "LEFT",  "target": "left"},
            {"start": 18.0, "end": 20.0, "label": "REST",  "target": "center"},
            {"start": 20.0, "end": 22.0, "label": "UP",    "target": "up"},
            {"start": 22.0, "end": 24.0, "label": "DOWN",  "target": "down"},
        ]
    },
    "Sequence_B": {
        "id": "Sequence_B",
        "description": "Alternative: REST -> LEFT -> RIGHT -> REST -> UP -> DOWN -> REST -> LEFT -> RIGHT -> REST -> UP -> DOWN",
        "timeline": [
            {"start": 0.0,  "end": 2.0,  "label": "REST",  "target": "center"},
            {"start": 2.0,  "end": 4.0,  "label": "LEFT",  "target": "left"},
            {"start": 4.0,  "end": 6.0,  "label": "RIGHT", "target": "right"},
            {"start": 6.0,  "end": 8.0,  "label": "REST",  "target": "center"},
            {"start": 8.0,  "end": 10.0, "label": "UP",    "target": "up"},
            {"start": 10.0, "end": 12.0, "label": "DOWN",  "target": "down"},
            {"start": 12.0, "end": 14.0, "label": "REST",  "target": "center"},
            {"start": 14.0, "end": 16.0, "label": "LEFT",  "target": "left"},
            {"start": 16.0, "end": 18.0, "label": "RIGHT", "target": "right"},
            {"start": 18.0, "end": 20.0, "label": "REST",  "target": "center"},
            {"start": 20.0, "end": 22.0, "label": "UP",    "target": "up"},
            {"start": 22.0, "end": 24.0, "label": "DOWN",  "target": "down"},
        ]
    },
    "Sequence_C": {
        "id": "Sequence_C",
        "description": "Alternative: REST -> RIGHT -> LEFT -> REST -> DOWN -> UP -> REST -> RIGHT -> LEFT -> REST -> DOWN -> UP",
        "timeline": [
            {"start": 0.0,  "end": 2.0,  "label": "REST",  "target": "center"},
            {"start": 2.0,  "end": 4.0,  "label": "RIGHT", "target": "right"},
            {"start": 4.0,  "end": 6.0,  "label": "LEFT",  "target": "left"},
            {"start": 6.0,  "end": 8.0,  "label": "REST",  "target": "center"},
            {"start": 8.0,  "end": 10.0, "label": "DOWN",  "target": "down"},
            {"start": 10.0, "end": 12.0, "label": "UP",    "target": "up"},
            {"start": 12.0, "end": 14.0, "label": "REST",  "target": "center"},
            {"start": 14.0, "end": 16.0, "label": "RIGHT", "target": "right"},
            {"start": 16.0, "end": 18.0, "label": "LEFT",  "target": "left"},
            {"start": 18.0, "end": 20.0, "label": "REST",  "target": "center"},
            {"start": 20.0, "end": 22.0, "label": "DOWN",  "target": "down"},
            {"start": 22.0, "end": 24.0, "label": "UP",    "target": "up"},
        ]
    },
    "Sequence_D": {
        "id": "Sequence_D",
        "description": "Alternative: REST -> LEFT -> RIGHT -> REST -> DOWN -> UP -> REST -> LEFT -> RIGHT -> REST -> DOWN -> UP",
        "timeline": [
            {"start": 0.0,  "end": 2.0,  "label": "REST",  "target": "center"},
            {"start": 2.0,  "end": 4.0,  "label": "LEFT",  "target": "left"},
            {"start": 4.0,  "end": 6.0,  "label": "RIGHT", "target": "right"},
            {"start": 6.0,  "end": 8.0,  "label": "REST",  "target": "center"},
            {"start": 8.0,  "end": 10.0, "label": "DOWN",  "target": "down"},
            {"start": 10.0, "end": 12.0, "label": "UP",    "target": "up"},
            {"start": 12.0, "end": 14.0, "label": "REST",  "target": "center"},
            {"start": 14.0, "end": 16.0, "label": "LEFT",  "target": "left"},
            {"start": 16.0, "end": 18.0, "label": "RIGHT", "target": "right"},
            {"start": 18.0, "end": 20.0, "label": "REST",  "target": "center"},
            {"start": 20.0, "end": 22.0, "label": "DOWN",  "target": "down"},
            {"start": 22.0, "end": 24.0, "label": "UP",    "target": "up"},
        ]
    }
}

os.makedirs(NEW_DATA_DIR, exist_ok=True)
os.makedirs(LABEL_DATA_DIR, exist_ok=True)


# ==============================================================================
# HARDWARE / STREAM ACQUISITION
# ==============================================================================
class EOGAcquisitionEngine:
    def __init__(self, com_port="COM3", force_mock=False):
        self.com_port = com_port
        self.force_mock = force_mock
        self.serial_conn = None
        self.is_mock = force_mock
        self.recording = False
        self.data_buffer = []  # list of (timestamp, eog_value)
        self.start_time = 0.0
        self.lock = threading.Lock()
        self.thread = None
        self._init_hardware()

    def _init_hardware(self):
        if self.force_mock:
            self.is_mock = True
            print("[EOG Engine] Initialized in FORCED MOCK mode.")
            return

        if not HAS_SERIAL:
            self.is_mock = True
            print("[EOG Engine] pyserial not available. Defaulting to MOCK mode.")
            return

        try:
            self.serial_conn = serial.Serial(self.com_port, SERIAL_BAUD, timeout=0.1)
            time.sleep(1.5)  # Arduino boot reset delay
            self.is_mock = False
            print(f"[EOG Engine] Successfully connected to Arduino on {self.com_port}.")
        except Exception as e:
            self.is_mock = True
            print(f"[EOG Engine] Could not open {self.com_port} ({e}). Running in MOCK mode.")

    def start_recording(self, duration=TARGET_DURATION, sequence_id="Sequence_A"):
        with self.lock:
            self.data_buffer = []
            self.recording = True
            self.duration = duration
            self.sequence_id = sequence_id

        if not self.is_mock and self.serial_conn and self.serial_conn.is_open:
            try:
                self.serial_conn.reset_input_buffer()
            except Exception:
                pass

        self.thread = threading.Thread(target=self._acquisition_loop, daemon=True)
        self.thread.start()

    def _acquisition_loop(self):
        self.start_time = time.perf_counter()

        # Mock signal simulation state
        mock_phase = 0.0
        timeline = SEQUENCES.get(self.sequence_id, SEQUENCES["Sequence_A"])["timeline"]

        while True:
            with self.lock:
                if not self.recording:
                    break

            now = time.perf_counter()
            elapsed = now - self.start_time
            if elapsed >= self.duration:
                break

            if self.is_mock:
                # Generate realistic 250 Hz synthetic EOG signal
                current_label = "REST"
                for phase in timeline:
                    if phase["start"] <= elapsed < phase["end"]:
                        current_label = phase["label"]
                        break

                offset = 0.0
                if current_label == "RIGHT":
                    offset = 120.0
                elif current_label == "LEFT":
                    offset = -120.0
                elif current_label == "UP":
                    offset = 80.0
                elif current_label == "DOWN":
                    offset = -80.0

                mock_phase += 2 * math.pi * 50.0 / EXPECTED_FS
                noise = random.gauss(0, 3.5)
                baseline = 512.0 + 10.0 * math.sin(elapsed * 0.5)
                val = int(baseline + offset + 6.0 * math.sin(mock_phase) + noise)
                val = max(0, min(1023, val))

                with self.lock:
                    self.data_buffer.append((elapsed, val))

                time.sleep(1.0 / EXPECTED_FS)

            else:
                try:
                    line = self.serial_conn.readline().decode('utf-8', errors='ignore').strip()
                    if line:
                        try:
                            val = int(line)
                            with self.lock:
                                self.data_buffer.append((elapsed, val))
                        except ValueError:
                            pass
                except Exception as e:
                    print(f"[EOG Engine Serial Read Error]: {e}")
                    time.sleep(0.005)

        with self.lock:
            self.recording = False

    def stop_recording(self):
        with self.lock:
            self.recording = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)

    def get_status(self):
        with self.lock:
            count = len(self.data_buffer)
            elapsed = (time.perf_counter() - self.start_time) if self.recording else 0.0
            is_rec = self.recording
        return {
            "recording": is_rec,
            "samples": count,
            "elapsed": round(elapsed, 2),
            "is_mock": self.is_mock,
            "port": self.com_port
        }

    def get_data(self):
        with self.lock:
            return list(self.data_buffer)


# ==============================================================================
# ATTEMPT & SUBJECT NUMBERING LOGIC
# ==============================================================================
def normalize_subject_id(raw_id: str) -> str:
    cleaned = raw_id.strip().upper()
    if cleaned.startswith("S") and cleaned[1:].isdigit():
        return f"S{int(cleaned[1:]):02d}"
    elif cleaned.isdigit():
        return f"S{int(cleaned):02d}"
    return cleaned if cleaned else "S01"

def get_next_attempt_id(subject_id: str) -> str:
    norm_subj = normalize_subject_id(subject_id)
    pattern = os.path.join(NEW_DATA_DIR, f"{norm_subj}_A*.csv")
    files = glob.glob(pattern)
    existing_nums = []
    for f in files:
        base = os.path.basename(f)
        parts = base.replace(".csv", "").split("_")
        if len(parts) >= 2 and parts[1].startswith("A") and parts[1][1:].isdigit():
            existing_nums.append(int(parts[1][1:]))

    next_num = max(existing_nums, default=0) + 1
    return f"{norm_subj}_A{next_num:02d}"

def check_subject_in_final_dataset(subject_id: str):
    norm_subj = normalize_subject_id(subject_id)
    if not os.path.exists(FINAL_CSV_PATH):
        return False, None

    try:
        df = pd.read_csv(FINAL_CSV_PATH)
        if "Subject_ID" in df.columns:
            matches = df[df["Subject_ID"] == norm_subj]
            if not matches.empty:
                chosen_att = str(matches["Attempt_ID"].iloc[0])
                return True, chosen_att
    except Exception as e:
        print(f"[Dataset Check Error]: {e}")
    return False, None


# ==============================================================================
# DATA QUALITY CHECK & VALIDATION
# ==============================================================================
def validate_recording(data_buffer, min_duration=18.0, min_samples=3500):
    if not data_buffer or len(data_buffer) < 100:
        return False, "Critical: Insufficient data collected (under 100 samples)."

    timestamps = [t for t, v in data_buffer]
    duration = timestamps[-1] - timestamps[0]
    sample_count = len(timestamps)

    warnings = []
    if duration < min_duration:
        warnings.append(f"Duration {duration:.2f}s is shorter than target ({TARGET_DURATION}s).")
    if sample_count < min_samples:
        warnings.append(f"Sample count ({sample_count}) is lower than expected (~5000).")

    for i in range(1, len(timestamps)):
        if timestamps[i] < timestamps[i - 1]:
            return False, f"Data Error: Non-monotonic timestamp detected at index {i}."

    msg = "Data valid." if not warnings else "Notice: " + "; ".join(warnings)
    return True, msg


# ==============================================================================
# RAW STORAGE & PLOTTING
# ==============================================================================
def save_raw_attempt(subject_id: str, attempt_id: str, data_buffer):
    csv_filename = f"{attempt_id}.csv"
    csv_path = os.path.join(NEW_DATA_DIR, csv_filename)

    t0 = data_buffer[0][0] if data_buffer else 0.0

    rows = []
    for idx, (t, val) in enumerate(data_buffer, start=1):
        rel_time = round(t - t0, 4)
        rows.append({
            "Subject_ID": subject_id,
            "Attempt_ID": attempt_id,
            "Sample_Index": idx,
            "Timestamp": rel_time,
            "EOG": int(val)
        })

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"[Raw Storage] Saved: {csv_path} ({len(df)} samples)")
    return csv_path, df

def generate_attempt_plot(subject_id: str, attempt_id: str, df: pd.DataFrame, sequence_id: str = "Sequence_A"):
    png_filename = f"{attempt_id}.png"
    png_path = os.path.join(NEW_DATA_DIR, png_filename)

    timestamps = df["Timestamp"].values
    eog = df["EOG"].values
    total_samples = len(df)

    fig, ax = plt.subplots(figsize=(15, 6), dpi=150)
    ax.plot(timestamps, eog, label="Raw EOG Signal", color="#2563eb", linewidth=1.2, zorder=3)

    timeline = SEQUENCES.get(sequence_id, SEQUENCES["Sequence_A"])["timeline"]
    phase_styles = {
        "REST":  {"bg": "#f1f5f9", "border": "#94a3b8", "text": "#475569", "icon": "REST"},
        "RIGHT": {"bg": "#dbeafe", "border": "#60a5fa", "text": "#1d4ed8", "icon": "RIGHT \u2192"},
        "LEFT":  {"bg": "#d1fae5", "border": "#34d399", "text": "#047857", "icon": "\u2190 LEFT"},
        "UP":    {"bg": "#fef3c7", "border": "#fbbf24", "text": "#b45309", "icon": "UP \u2191"},
        "DOWN":  {"bg": "#ffe4e6", "border": "#f87171", "text": "#b91c1c", "icon": "DOWN \u2193"}
    }

    y_min, y_max = float(np.min(eog)), float(np.max(eog))
    y_range = y_max - y_min if y_max != y_min else 1.0
    ax.set_ylim(y_min - 0.08 * y_range, y_max + 0.24 * y_range)

    for i, phase in enumerate(timeline):
        start_t = float(phase["start"])
        end_t = float(phase["end"])
        lbl = str(phase["label"]).upper()
        style = phase_styles.get(lbl, phase_styles["REST"])

        ax.axvspan(start_t, end_t, color=style["bg"], alpha=0.6, zorder=1)
        ax.axvline(start_t, color=style["border"], linestyle="--", linewidth=1.0, alpha=0.8, zorder=2)

        mid_t = (start_t + end_t) / 2.0
        y_pos = y_max + (0.16 if (i % 2 == 0) else 0.05) * y_range
        badge_text = f"{style['icon']}\n({start_t:.0f}s \u2013 {end_t:.0f}s)"
        ax.text(mid_t, y_pos, badge_text, ha='center', va='center', fontsize=8.5, fontweight='bold',
                color=style["text"], zorder=5,
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor=style["border"], linewidth=1.2, alpha=0.95))

    duration = timestamps[-1] if len(timestamps) > 0 else 0
    fs = len(timestamps) / duration if duration > 0 else 0
    ax.set_title(f"EOG Signal \u2014 {attempt_id} (Subject: {subject_id})\nDuration: {duration:.2f}s | Samples: {len(timestamps)} | Sampling Rate: {fs:.1f} Hz",
                 fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Time (seconds) [Stimulus-Synchronized Timebase]", fontsize=11, fontweight='bold', labelpad=8)
    ax.set_ylabel("EOG Amplitude (ADC Units)", fontsize=11, fontweight='bold', labelpad=8)
    ax.set_xlim(0, max(20.0, duration))
    ax.grid(True, linestyle=":", alpha=0.5, color="#cbd5e1", zorder=2)

    # Top sample count axis
    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())
    time_ticks = np.arange(0, duration + 0.1, 2.0)
    sample_indices = [int(np.searchsorted(timestamps, t)) for t in time_ticks]
    sample_indices[-1] = min(sample_indices[-1], total_samples - 1)
    ax_top.set_xticks(time_ticks)
    ax_top.set_xticklabels([f"S#{idx}" for idx in sample_indices], fontsize=8, color="#64748b")
    ax_top.set_xlabel("Corresponding Sample Index (at each phase boundary)", fontsize=9, fontweight="bold", color="#64748b", labelpad=6)

    plt.tight_layout()
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot Generation] Saved: {png_path}")
    return png_path


# ==============================================================================
# FINAL LABELED DATASET CURATION (ONE PER SUBJECT)
# ==============================================================================
def assign_timeline_labels(df: pd.DataFrame, sequence_id: str = "Sequence_A") -> pd.DataFrame:
    timeline = SEQUENCES.get(sequence_id, SEQUENCES["Sequence_A"])["timeline"]

    def map_label(t):
        for phase in timeline:
            if phase["start"] <= t < phase["end"]:
                return phase["label"]
        return "REST"

    df_labeled = df.copy()
    df_labeled["Label"] = df_labeled["Timestamp"].apply(map_label)
    return df_labeled

def commit_selected_attempt(subject_id: str, attempt_id: str, df_raw: pd.DataFrame, sequence_id: str = "Sequence_A", replace: bool = False):
    norm_subj = normalize_subject_id(subject_id)
    df_labeled = assign_timeline_labels(df_raw, sequence_id)

    desired_cols = ["Subject_ID", "Attempt_ID", "Sample_Index", "Timestamp", "EOG", "Label"]
    df_new = df_labeled[desired_cols].copy()

    file_exists = os.path.exists(FINAL_CSV_PATH)

    if file_exists:
        df_master = pd.read_csv(FINAL_CSV_PATH)
        already_present = norm_subj in df_master["Subject_ID"].values

        if already_present:
            if not replace:
                return False, f"Subject {norm_subj} already has a selected recording. Explicit replacement required."
            print(f"[Curator] Replacing previous selected recording for {norm_subj}...")
            df_master = df_master[df_master["Subject_ID"] != norm_subj]

        df_combined = pd.concat([df_master, df_new], ignore_index=True)
        df_combined.to_csv(FINAL_CSV_PATH, index=False)
    else:
        df_new.to_csv(FINAL_CSV_PATH, index=False)

    record_trial_metadata(subject_id, attempt_id, sequence_id, "SELECTED", len(df_raw), df_raw["Timestamp"].iloc[-1], True)
    print(f"[Curator] Successfully recorded selected trial {attempt_id} into {FINAL_CSV_PATH}.")
    return True, f"Attempt {attempt_id} successfully saved as final recording for {norm_subj}."

def record_trial_metadata(subject_id: str, attempt_id: str, sequence_id: str, status: str, sample_count: int, duration: float, selected: bool):
    fs = sample_count / duration if duration > 0 else 0
    record = {
        "Subject_ID": subject_id,
        "Attempt_ID": attempt_id,
        "Sequence_ID": sequence_id,
        "Status": status,
        "Sample_Count": sample_count,
        "Duration": round(duration, 3),
        "Sampling_Rate": round(fs, 1),
        "Selected": selected,
        "Recording_DateTime": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    df_rec = pd.DataFrame([record])
    if os.path.exists(METADATA_CSV_PATH):
        df_rec.to_csv(METADATA_CSV_PATH, mode='a', header=False, index=False)
    else:
        df_rec.to_csv(METADATA_CSV_PATH, mode='w', header=True, index=False)


# ==============================================================================
# HTTP & REST API SERVER (FOR SYNCHRONIZED BROWSER STIMULUS)
# ==============================================================================
class CollectorServerState:
    def __init__(self, engine: EOGAcquisitionEngine):
        self.engine = engine
        self.current_subject = "S01"
        self.current_sequence = "Sequence_A"
        self.last_attempt_id = None
        self.last_raw_df = None
        self.last_png_path = None
        self.last_metrics = {}

state: CollectorServerState = None

class iSyncHTTPRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=BASE_DIR, **kwargs)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/status":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()

            engine_status = state.engine.get_status()
            curr_subj = normalize_subject_id(state.current_subject)
            next_attempt = get_next_attempt_id(curr_subj)
            has_selected, selected_att = check_subject_in_final_dataset(curr_subj)

            resp = {
                **engine_status,
                "subject_id": curr_subj,
                "next_attempt_id": next_attempt,
                "has_selected": has_selected,
                "selected_attempt_id": selected_att,
                "sequence_id": state.current_sequence,
                "sequences": list(SEQUENCES.keys())
            }
            self.wfile.write(json.dumps(resp).encode("utf-8"))

        elif path == "/api/get_plot":
            query = parse_qs(parsed.query)
            filename = query.get("file", [""])[0]
            file_path = os.path.join(NEW_DATA_DIR, os.path.basename(filename))
            if os.path.exists(file_path):
                self.send_response(200)
                self.send_header("Content-type", "image/png")
                self.end_headers()
                with open(file_path, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.send_error(404, "Plot not found")

        else:
            super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path
        length = int(self.headers.get('content-length', 0))
        body = self.rfile.read(length).decode('utf-8') if length > 0 else "{}"
        try:
            payload = json.loads(body)
        except Exception:
            payload = {}

        if path == "/api/set_subject":
            new_subj = payload.get("subject_id", "S01")
            state.current_subject = normalize_subject_id(new_subj)
            seq = payload.get("sequence_id", state.current_sequence)
            if seq in SEQUENCES:
                state.current_sequence = seq

            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            has_sel, sel_att = check_subject_in_final_dataset(state.current_subject)
            self.wfile.write(json.dumps({
                "status": "ok",
                "subject_id": state.current_subject,
                "next_attempt_id": get_next_attempt_id(state.current_subject),
                "has_selected": has_sel,
                "selected_attempt_id": sel_att
            }).encode("utf-8"))

        elif path == "/api/start_trial":
            subj = payload.get("subject_id", state.current_subject)
            seq = payload.get("sequence_id", state.current_sequence)
            state.current_subject = normalize_subject_id(subj)
            state.current_sequence = seq

            attempt_id = get_next_attempt_id(state.current_subject)
            state.last_attempt_id = attempt_id

            state.engine.start_recording(duration=TARGET_DURATION, sequence_id=seq)

            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "status": "started",
                "subject_id": state.current_subject,
                "attempt_id": attempt_id,
                "duration": TARGET_DURATION,
                "sequence": SEQUENCES.get(seq, SEQUENCES["Sequence_A"])
            }).encode("utf-8"))

        elif path == "/api/finish_trial":
            state.engine.stop_recording()
            data = state.engine.get_data()

            attempt_id = state.last_attempt_id or get_next_attempt_id(state.current_subject)
            valid, msg = validate_recording(data)

            csv_path, df = save_raw_attempt(state.current_subject, attempt_id, data)
            png_path = generate_attempt_plot(state.current_subject, attempt_id, df, state.current_sequence)

            state.last_raw_df = df
            state.last_png_path = png_path

            dur = df["Timestamp"].iloc[-1] if len(df) > 0 else 0
            fs = len(df) / dur if dur > 0 else 0
            state.last_metrics = {
                "sample_count": len(df),
                "duration": round(dur, 2),
                "sampling_rate": round(fs, 1),
                "validation_msg": msg
            }

            has_sel, sel_att = check_subject_in_final_dataset(state.current_subject)

            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "status": "complete",
                "subject_id": state.current_subject,
                "attempt_id": attempt_id,
                "metrics": state.last_metrics,
                "plot_url": f"/api/get_plot?file={os.path.basename(png_path)}",
                "has_selected": has_sel,
                "selected_attempt_id": sel_att
            }).encode("utf-8"))

        elif path == "/api/decision":
            action = payload.get("action")  # 'SELECT', 'REJECT', 'RECORD_AGAIN'
            replace = payload.get("replace", False)
            attempt_id = payload.get("attempt_id", state.last_attempt_id)
            subj = state.current_subject

            if action == "SELECT":
                if state.last_raw_df is None or state.last_attempt_id != attempt_id:
                    csv_p = os.path.join(NEW_DATA_DIR, f"{attempt_id}.csv")
                    if os.path.exists(csv_p):
                        state.last_raw_df = pd.read_csv(csv_p)

                success, message = commit_selected_attempt(subj, attempt_id, state.last_raw_df, state.current_sequence, replace=replace)
                resp = {"status": "ok" if success else "error", "message": message, "action": "SELECT"}

            elif action in ("REJECT", "RECORD_AGAIN"):
                record_trial_metadata(subj, attempt_id, state.current_sequence, "REJECTED",
                                      state.last_metrics.get("sample_count", 0),
                                      state.last_metrics.get("duration", 0), False)
                resp = {
                    "status": "ok",
                    "action": action,
                    "message": f"Attempt {attempt_id} rejected and archived in New Data.",
                    "next_attempt_id": get_next_attempt_id(subj)
                }
            else:
                resp = {"status": "error", "message": f"Unknown action: {action}"}

            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(resp).encode("utf-8"))

        else:
            self.send_error(404, "Endpoint not found")


# ==============================================================================
# CLI EXECUTION ENTRY POINT
# ==============================================================================
def run_cli_flow(engine: EOGAcquisitionEngine):
    print("\n=======================================================")
    print(" iSync EOG Interactive Data Collection & Curation CLI  ")
    print("=======================================================")

    while True:
        raw_subj = input("\nEnter Subject ID (e.g. S01, S02) [or 'exit' to quit]: ").strip()
        if raw_subj.lower() in ("exit", "quit"):
            break

        subject_id = normalize_subject_id(raw_subj)
        has_sel, sel_att = check_subject_in_final_dataset(subject_id)
        if has_sel:
            print(f"\n[WARNING] Subject {subject_id} ALREADY HAS a selected recording: {sel_att}")
            ans = input("Do you want to continue making attempts for this subject? (y/n): ").strip().lower()
            if ans != 'y':
                continue

        while True:
            attempt_id = get_next_attempt_id(subject_id)
            print("\n-------------------------------------------------------")
            print(f" Subject: {subject_id} | Attempt: {attempt_id}")
            print("-------------------------------------------------------")
            input("Press [ENTER] to begin 3-second countdown and 20s recording...")

            for countdown in [3, 2, 1]:
                print(f"  {countdown}...")
                time.sleep(1.0)
            print("  >>> START! Recording 20 seconds of EOG...")

            engine.start_recording(duration=TARGET_DURATION, sequence_id="Sequence_A")

            while True:
                st = engine.get_status()
                if not st["recording"]:
                    break
                print(f"\r  Recording: {st['elapsed']:.1f} / {TARGET_DURATION}s | Samples: {st['samples']}", end="", flush=True)
                time.sleep(0.2)

            print("\n  [+] 20-second recording finished!")
            data = engine.get_data()
            valid, msg = validate_recording(data)
            print(f"  Status: {msg}")

            csv_path, df = save_raw_attempt(subject_id, attempt_id, data)
            png_path = generate_attempt_plot(subject_id, attempt_id, df, "Sequence_A")

            dur = df["Timestamp"].iloc[-1] if len(df) > 0 else 0
            fs = len(df) / dur if dur > 0 else 0
            print("\n  Inspection Graph Generated:")
            print(f"  File: {png_path}")
            print(f"  Duration: {dur:.2f}s | Samples: {len(df)} | Estimated Fs: {fs:.1f} Hz")

            print("\n  [1] SELECT (Add to final labeled dataset)")
            print("  [2] REJECT (Keep in New Data, but do not label)")
            print("  [3] RECORD AGAIN (Reject current and re-record attempt)")
            choice = input("  Select action [1/2/3]: ").strip()

            if choice == "1":
                has_sel, sel_att = check_subject_in_final_dataset(subject_id)
                replace = False
                if has_sel:
                    print(f"\n  [ALERT] Subject {subject_id} already has selected recording: {sel_att}")
                    rep_input = input(f"  Do you want to REPLACE {sel_att} with {attempt_id}? (yes/no): ").strip().lower()
                    if rep_input in ("yes", "y"):
                        replace = True
                    else:
                        print("  Selection cancelled. Attempt kept in New Data.")
                        break

                commit_selected_attempt(subject_id, attempt_id, df, "Sequence_A", replace=replace)
                print(f"  [SUCCESS] Subject {subject_id} complete!\n")
                break

            elif choice == "2":
                record_trial_metadata(subject_id, attempt_id, "Sequence_A", "REJECTED", len(df), dur, False)
                print(f"  Attempt {attempt_id} rejected. Saved in New Data.\n")
                cont = input("  Make another attempt for this subject? (y/n): ").strip().lower()
                if cont != 'y':
                    break
            else:
                record_trial_metadata(subject_id, attempt_id, "Sequence_A", "REJECTED", len(df), dur, False)
                print(f"  Attempt {attempt_id} saved. Preparing next attempt...\n")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="iSync EOG Data Collection & Curation Server")
    parser.add_argument("--port", type=str, default="COM3", help="Serial COM port for Arduino")
    parser.add_argument("--mock", action="store_true", help="Force synthetic mock EOG data stream")
    parser.add_argument("--cli", action="store_true", help="Run interactive terminal CLI mode instead of web server")
    parser.add_argument("--server-port", type=int, default=5000, help="Web server HTTP port (default: 5000)")
    args = parser.parse_args()

    global state
    engine = EOGAcquisitionEngine(com_port=args.port, force_mock=args.mock)
    state = CollectorServerState(engine)

    if args.cli:
        run_cli_flow(engine)
    else:
        server_address = ('', args.server_port)
        httpd = HTTPServer(server_address, iSyncHTTPRequestHandler)
        mode_str = "MOCK MODE" if engine.is_mock else f"ARDUINO ({args.port})"
        print("\n" + "=" * 65)
        print(" iSync EOG Visual Stimulus & Acquisition Server Running ")
        print("=" * 65)
        print(f" Hardware Mode:   {mode_str}")
        print(f" Local Web UI:    http://127.0.0.1:{args.server_port}/index.html")
        print(f" Raw Storage:     {NEW_DATA_DIR}")
        print(f" Final Dataset:   {FINAL_CSV_PATH}")
        print("=" * 65 + "\n")
        print("Press Ctrl+C to terminate server.\n")

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server...")
            httpd.server_close()

if __name__ == "__main__":
    main()
