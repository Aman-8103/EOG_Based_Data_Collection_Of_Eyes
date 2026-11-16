"""
eog_plotter.py - Advanced EOG Signal Processing & Visualization Engine
=======================================================================
Implements a multi-stage signal processing pipeline specifically designed
for single-channel EOG data collected via Arduino (250 Hz, 10-bit ADC).

Signal Processing Pipeline:
  1. Adaptive Resampling   - Uniform 250 Hz grid via cubic interpolation
  2. DC Removal            - Baseline wander removal via median filter
  3. Powerline Rejection   - Cascaded 50 Hz + 100 Hz notch filters
  4. Bandpass Filtering     - 0.05-35 Hz (wide) to preserve saccade transients
  5. Wavelet Denoising      - Stationary Wavelet Transform (sym4) soft threshold
  6. Derivative Enhancement - 1st derivative highlights saccadic spikes
  7. Per-Phase Adaptive Gain - Normalizes weak channels (UP/DOWN) to match strong

Design rationale:
  - Standard 0.1-10 Hz bandpass is TOO AGGRESSIVE for saccade detection.
    Saccadic eye movements contain energy up to 30-40 Hz. Cutting at 10 Hz
    smooths out the sharp spikes that distinguish movements from rest.
  - Single-channel horizontal electrode pairs inherently have weak vertical
    sensitivity. Adaptive gain per-phase compensates for this.
  - Wavelet denoising preserves edge structure (saccade onsets) far better
    than linear filters, which blur transient events.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import (
    butter, filtfilt, iirnotch, medfilt,
    find_peaks, savgol_filter
)
from scipy.interpolate import interp1d

# Try to import PyWavelets for wavelet denoising; fall back gracefully
try:
    import pywt
    HAS_PYWT = True
except ImportError:
    HAS_PYWT = False

# ==========================================================================
# VISUAL STYLE CONSTANTS
# ==========================================================================
PHASE_STYLES = {
    "REST":   {"bg": "#f1f5f9", "border": "#94a3b8", "text": "#475569", "icon": "REST"},
    "RIGHT":  {"bg": "#dbeafe", "border": "#60a5fa", "text": "#1d4ed8", "icon": "RIGHT \u2192"},
    "LEFT":   {"bg": "#d1fae5", "border": "#34d399", "text": "#047857", "icon": "\u2190 LEFT"},
    "UP":     {"bg": "#fef3c7", "border": "#fbbf24", "text": "#b45309", "icon": "UP \u2191"},
    "DOWN":   {"bg": "#ffe4e6", "border": "#f87171", "text": "#b91c1c", "icon": "DOWN \u2193"},
    "CENTER": {"bg": "#e0e7ff", "border": "#818cf8", "text": "#4338ca", "icon": "CENTER \u25ce"}
}

DEFAULT_SEQUENCES = {
    "Sequence_A": [
        {"start": 0.0, "end": 4.0, "label": "REST"},
        {"start": 4.0, "end": 6.0, "label": "RIGHT"},
        {"start": 6.0, "end": 8.0, "label": "LEFT"},
        {"start": 8.0, "end": 10.0, "label": "CENTER"},
        {"start": 10.0, "end": 12.0, "label": "UP"},
        {"start": 12.0, "end": 14.0, "label": "DOWN"},
        {"start": 14.0, "end": 18.0, "label": "REST"},
        {"start": 18.0, "end": 20.0, "label": "RIGHT"},
        {"start": 20.0, "end": 22.0, "label": "LEFT"},
        {"start": 22.0, "end": 24.0, "label": "CENTER"},
        {"start": 24.0, "end": 26.0, "label": "UP"},
        {"start": 26.0, "end": 28.0, "label": "DOWN"},
        {"start": 28.0, "end": 30.0, "label": "REST"},
    ],
    "Sequence_B": [
        {"start": 0.0, "end": 4.0, "label": "REST"},
        {"start": 4.0, "end": 6.0, "label": "LEFT"},
        {"start": 6.0, "end": 8.0, "label": "RIGHT"},
        {"start": 8.0, "end": 10.0, "label": "CENTER"},
        {"start": 10.0, "end": 12.0, "label": "UP"},
        {"start": 12.0, "end": 14.0, "label": "DOWN"},
        {"start": 14.0, "end": 18.0, "label": "REST"},
        {"start": 18.0, "end": 20.0, "label": "LEFT"},
        {"start": 20.0, "end": 22.0, "label": "RIGHT"},
        {"start": 22.0, "end": 24.0, "label": "CENTER"},
        {"start": 24.0, "end": 26.0, "label": "UP"},
        {"start": 26.0, "end": 28.0, "label": "DOWN"},
        {"start": 28.0, "end": 30.0, "label": "REST"},
    ],
    "Sequence_C": [
        {"start": 0.0, "end": 4.0, "label": "REST"},
        {"start": 4.0, "end": 6.0, "label": "RIGHT"},
        {"start": 6.0, "end": 8.0, "label": "LEFT"},
        {"start": 8.0, "end": 10.0, "label": "CENTER"},
        {"start": 10.0, "end": 12.0, "label": "DOWN"},
        {"start": 12.0, "end": 14.0, "label": "UP"},
        {"start": 14.0, "end": 18.0, "label": "REST"},
        {"start": 18.0, "end": 20.0, "label": "RIGHT"},
        {"start": 20.0, "end": 22.0, "label": "LEFT"},
        {"start": 22.0, "end": 24.0, "label": "CENTER"},
        {"start": 24.0, "end": 26.0, "label": "DOWN"},
        {"start": 26.0, "end": 28.0, "label": "UP"},
        {"start": 28.0, "end": 30.0, "label": "REST"},
    ],
    "Sequence_D": [
        {"start": 0.0, "end": 4.0, "label": "REST"},
        {"start": 4.0, "end": 6.0, "label": "LEFT"},
        {"start": 6.0, "end": 8.0, "label": "RIGHT"},
        {"start": 8.0, "end": 10.0, "label": "CENTER"},
        {"start": 10.0, "end": 12.0, "label": "DOWN"},
        {"start": 12.0, "end": 14.0, "label": "UP"},
        {"start": 14.0, "end": 18.0, "label": "REST"},
        {"start": 18.0, "end": 20.0, "label": "LEFT"},
        {"start": 20.0, "end": 22.0, "label": "RIGHT"},
        {"start": 22.0, "end": 24.0, "label": "CENTER"},
        {"start": 24.0, "end": 26.0, "label": "DOWN"},
        {"start": 26.0, "end": 28.0, "label": "UP"},
        {"start": 28.0, "end": 30.0, "label": "REST"},
    ]
}


# ==========================================================================
# CORE SIGNAL PROCESSING FUNCTIONS
# ==========================================================================

def resample_uniform(time_arr, eog_raw, target_fs=250.0):
    """
    Resample non-uniform Arduino timestamps to a uniform grid.

    Arduino serial communication introduces jitter (your data shows
    intervals ranging from 0ms to 162ms). Non-uniform sampling causes
    filter artifacts. Cubic interpolation onto a uniform grid fixes this.

    Parameters
    ----------
    time_arr : np.ndarray
        Original timestamps (seconds), starting from 0.
    eog_raw : np.ndarray
        Raw ADC values (0-1023).
    target_fs : float
        Desired uniform sampling rate in Hz.

    Returns
    -------
    t_uniform : np.ndarray
        Uniform time grid.
    eog_uniform : np.ndarray
        Resampled EOG values.
    """
    # Ensure time_arr is strictly monotonically increasing
    diffs = np.diff(time_arr)
    if np.any(diffs <= 0):
        df_temp = pd.DataFrame({'t': time_arr, 'v': eog_raw})
        df_temp = df_temp.groupby('t', as_index=False).mean()
        time_arr = df_temp['t'].values
        eog_raw = df_temp['v'].values

    duration = time_arr[-1]
    n_samples = int(duration * target_fs) + 1
    t_uniform = np.linspace(0, duration, n_samples)

    # Cubic interpolation preserves waveform shape better than linear
    interpolator = interp1d(time_arr, eog_raw, kind='cubic',
                            fill_value='extrapolate')
    eog_uniform = interpolator(t_uniform)

    return t_uniform, eog_uniform


def remove_baseline_wander(eog, fs, window_sec=2.0):
    """
    Remove slow baseline drift using a median filter.

    EOG signals suffer from electrode drift, skin potential changes, and
    slow DC wander. A median filter with a wide window (2 seconds = 500
    samples at 250 Hz) estimates the baseline trend without distorting
    saccadic spikes (which are ~50-200ms wide).

    This is superior to simple mean subtraction because it tracks
    time-varying drift rather than assuming a constant offset.

    Parameters
    ----------
    eog : np.ndarray
        Input signal.
    fs : float
        Sampling rate in Hz.
    window_sec : float
        Median filter window in seconds.

    Returns
    -------
    eog_detrended : np.ndarray
        Signal with baseline wander removed.
    baseline : np.ndarray
        The estimated baseline for diagnostic plotting.
    """
    kernel_size = int(window_sec * fs)
    # medfilt requires odd kernel size
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel_size = max(kernel_size, 3)

    baseline = medfilt(eog, kernel_size=kernel_size)
    eog_detrended = eog - baseline

    return eog_detrended, baseline


def apply_notch_filters(eog, fs, freqs=[50.0, 100.0], Q=30.0):
    """
    Apply cascaded notch filters for powerline hum and harmonics.

    Parameters
    ----------
    eog : np.ndarray
        Input signal.
    fs : float
        Sampling rate in Hz.
    freqs : list of float
        Frequencies to notch out (50 Hz fundamental + 100 Hz harmonic).
    Q : float
        Quality factor (higher = narrower notch).

    Returns
    -------
    eog_notched : np.ndarray
        Signal with powerline interference removed.
    """
    nyquist = fs / 2.0
    eog_notched = eog.copy()

    for freq in freqs:
        if freq < nyquist:
            b_notch, a_notch = iirnotch(freq, Q, fs)
            eog_notched = filtfilt(b_notch, a_notch, eog_notched)

    return eog_notched


def apply_bandpass(eog, fs, lowcut=0.05, highcut=35.0, order=4):
    """
    Wide bandpass filter preserving saccadic spike energy.

    CRITICAL DESIGN DECISION:
    The previous pipeline used 0.1-10 Hz, which is standard for smooth
    pursuit EOG analysis. However, saccadic eye movements generate
    transient spikes with energy content up to 30-40 Hz. Cutting at
    10 Hz removes this energy, making spikes appear "flat" or "rounded."

    The new range 0.05-35 Hz:
    - 0.05 Hz low-cut: removes DC + ultra-slow drift while preserving
      the sustained deflection during gaze-hold
    - 35 Hz high-cut: preserves saccadic onset/offset transients
      while still removing high-frequency EMG/noise

    A 4th-order Butterworth provides a smooth roll-off without ringing.

    Parameters
    ----------
    eog : np.ndarray
        Input signal.
    fs : float
        Sampling rate in Hz.
    lowcut : float
        Lower cutoff frequency in Hz.
    highcut : float
        Upper cutoff frequency in Hz.
    order : int
        Filter order.

    Returns
    -------
    eog_filtered : np.ndarray
        Bandpass-filtered signal.
    """
    nyquist = fs / 2.0

    # Ensure highcut is below Nyquist
    if highcut >= nyquist:
        highcut = nyquist - 1.0

    Wn = [lowcut / nyquist, highcut / nyquist]
    b, a = butter(order, Wn, btype='band')
    eog_filtered = filtfilt(b, a, eog)

    return eog_filtered


def wavelet_denoise(eog, wavelet='sym4', level=4, threshold_mode='soft'):
    """
    Wavelet denoising using Stationary Wavelet Transform (SWT).

    Why wavelets beat linear filters for EOG saccade detection:
    - Linear filters (Butterworth, etc.) blur ALL transients equally
    - Wavelets decompose the signal into time-frequency components
    - We can selectively remove noise coefficients while preserving
      the sharp edge structure of saccadic onsets

    SWT (rather than DWT) is used because it is shift-invariant,
    meaning the denoising result doesn't change with signal alignment.

    The sym4 wavelet is chosen for its good time-frequency localization
    and symmetry properties.

    Parameters
    ----------
    eog : np.ndarray
        Input signal.
    wavelet : str
        Wavelet family to use.
    level : int
        Decomposition level.
    threshold_mode : str
        'soft' or 'hard' thresholding.

    Returns
    -------
    eog_denoised : np.ndarray
        Denoised signal.
    """
    if not HAS_PYWT:
        # Fallback: Savitzky-Golay filter (polynomial smoothing)
        # window=31 (~124ms at 250Hz), order=3 preserves peaks well
        window = min(31, len(eog) - 1)
        if window % 2 == 0:
            window -= 1
        if window >= 5:
            return savgol_filter(eog, window, polyorder=3)
        return eog

    # Pad signal to required length for SWT (must be divisible by 2^level)
    original_len = len(eog)
    pad_len = int(2 ** level * np.ceil(original_len / 2 ** level)) - original_len
    eog_padded = np.pad(eog, (0, pad_len), mode='reflect')

    # Stationary Wavelet Transform
    coeffs = pywt.swt(eog_padded, wavelet, level=level)

    # Universal threshold (VisuShrink)
    # We apply threshold only to detail coefficients, not the approximation
    sigma = np.median(np.abs(coeffs[-1][1])) / 0.6745  # MAD estimator
    threshold = sigma * np.sqrt(2 * np.log(len(eog_padded)))

    denoised_coeffs = []
    for i, (cA, cD) in enumerate(coeffs):
        if i == 0:
            # Keep approximation coefficients unchanged
            denoised_coeffs.append((cA, cD))
        else:
            # Apply soft thresholding to detail coefficients
            cD_thresh = pywt.threshold(cD, threshold * (0.8 ** i),
                                        mode=threshold_mode)
            denoised_coeffs.append((cA, cD_thresh))

    # Inverse SWT
    eog_denoised = pywt.iswt(denoised_coeffs, wavelet)

    return eog_denoised[:original_len]


def compute_saccade_derivative(eog, fs):
    """
    Compute the first derivative of the EOG signal to highlight saccades.

    Saccadic eye movements are characterized by RAPID changes in the
    corneoretinal potential. The derivative (rate of change) amplifies
    these transients while suppressing the sustained gaze-hold plateau.

    This is particularly useful for detecting UP/DOWN movements on a
    horizontal electrode pair, where the sustained deflection is weak
    but the onset transient may still be detectable.

    Parameters
    ----------
    eog : np.ndarray
        Input (filtered) signal.
    fs : float
        Sampling rate in Hz.

    Returns
    -------
    eog_derivative : np.ndarray
        First derivative of the signal (same length, zero-padded).
    """
    # Central difference for interior points
    dt = 1.0 / fs
    eog_derivative = np.gradient(eog, dt)

    # Smooth the derivative slightly to reduce noise amplification
    window = min(11, len(eog_derivative) - 1)
    if window % 2 == 0:
        window -= 1
    if window >= 5:
        eog_derivative = savgol_filter(eog_derivative, window, polyorder=2)

    return eog_derivative


def adaptive_phase_gain(eog, time_arr, timeline, rest_labels={"REST"}):
    """
    Normalize signal amplitude per-phase so weak movements (UP/DOWN)
    become visually comparable to strong movements (LEFT/RIGHT).

    Method:
    1. Compute the RMS amplitude of each non-REST phase
    2. Find the maximum RMS across all movement phases
    3. Scale each phase so its RMS matches the target (max RMS)
    4. Apply smooth cross-fade at phase boundaries to avoid discontinuities

    This is an OPTIONAL visualization enhancement. It does NOT change the
    raw data - it produces a separate "gain-normalized" view.

    Parameters
    ----------
    eog : np.ndarray
        Input filtered signal.
    time_arr : np.ndarray
        Time array (uniform grid).
    timeline : list of dict
        Phase definitions with start, end, label.
    rest_labels : set
        Labels to treat as rest (no gain applied).

    Returns
    -------
    eog_normalized : np.ndarray
        Gain-normalized signal.
    gain_profile : np.ndarray
        Per-sample gain factor applied (for diagnostic display).
    """
    eog_normalized = eog.copy()
    gain_profile = np.ones_like(eog, dtype=float)

    # Compute per-phase RMS
    phase_rms = {}
    for phase in timeline:
        label = phase["label"].upper()
        if label in rest_labels:
            continue
        mask = (time_arr >= phase["start"]) & (time_arr < phase["end"])
        if np.sum(mask) > 0:
            segment = eog[mask]
            rms = np.sqrt(np.mean(segment ** 2))
            if rms > 0:
                if label not in phase_rms:
                    phase_rms[label] = []
                phase_rms[label].append(rms)

    if not phase_rms:
        return eog_normalized, gain_profile

    # Average RMS per movement type
    avg_rms = {label: np.mean(rms_vals) for label, rms_vals in phase_rms.items()}
    target_rms = max(avg_rms.values())

    # Apply gain to each phase
    fs = len(time_arr) / (time_arr[-1] - time_arr[0]) if time_arr[-1] > time_arr[0] else 250.0
    crossfade_samples = int(0.1 * fs)  # 100ms crossfade

    for phase in timeline:
        label = phase["label"].upper()
        if label in rest_labels:
            continue

        mask = (time_arr >= phase["start"]) & (time_arr < phase["end"])
        indices = np.where(mask)[0]
        if len(indices) == 0 or avg_rms.get(label, 0) == 0:
            continue

        gain = target_rms / avg_rms[label]
        # Limit gain to prevent excessive amplification of noise
        gain = min(gain, 4.0)

        # Apply gain with crossfade at boundaries
        for idx in indices:
            # Distance from phase boundaries for crossfade
            dist_start = idx - indices[0]
            dist_end = indices[-1] - idx
            fade = 1.0
            if dist_start < crossfade_samples:
                fade = dist_start / max(crossfade_samples, 1)
            if dist_end < crossfade_samples:
                fade = min(fade, dist_end / max(crossfade_samples, 1))

            effective_gain = 1.0 + (gain - 1.0) * fade
            eog_normalized[idx] = eog[idx] * effective_gain
            gain_profile[idx] = effective_gain

    return eog_normalized, gain_profile


# ==========================================================================
# COMPLETE PROCESSING PIPELINE
# ==========================================================================

def process_eog_signal(eog_raw, time_arr, fs_target=250.0, timeline=None,
                       enable_wavelet=True, enable_adaptive_gain=True):
    """
    Full multi-stage EOG signal processing pipeline.

    Parameters
    ----------
    eog_raw : np.ndarray
        Raw ADC values from Arduino.
    time_arr : np.ndarray
        Timestamps (seconds, starting from 0).
    fs_target : float
        Target uniform sampling rate.
    timeline : list of dict, optional
        Phase definitions for adaptive gain.
    enable_wavelet : bool
        Whether to apply wavelet denoising.
    enable_adaptive_gain : bool
        Whether to apply per-phase adaptive gain normalization.

    Returns
    -------
    results : dict
        Dictionary containing all intermediate and final signals:
        - 't_uniform': uniform time grid
        - 'eog_resampled': after resampling
        - 'eog_detrended': after baseline removal
        - 'eog_notched': after notch filtering
        - 'eog_bandpass': after bandpass filtering
        - 'eog_denoised': after wavelet denoising
        - 'eog_derivative': saccade-enhanced derivative signal
        - 'eog_normalized': after adaptive gain (if enabled)
        - 'gain_profile': per-sample gain factors
        - 'baseline': estimated baseline wander
        - 'fs': actual sampling rate
    """
    results = {}

    # Stage 1: Resample to uniform grid
    t_uniform, eog_resampled = resample_uniform(time_arr, eog_raw, fs_target)
    fs = fs_target
    results['t_uniform'] = t_uniform
    results['eog_resampled'] = eog_resampled
    results['fs'] = fs

    # Stage 2: Remove baseline wander (median filter, 2s window)
    eog_detrended, baseline = remove_baseline_wander(eog_resampled, fs, window_sec=2.0)
    results['eog_detrended'] = eog_detrended
    results['baseline'] = baseline

    # Stage 3: Notch filters (50 Hz + 100 Hz harmonics)
    eog_notched = apply_notch_filters(eog_detrended, fs, freqs=[50.0, 100.0])
    results['eog_notched'] = eog_notched

    # Stage 4: Wide bandpass (0.05-35 Hz) preserving saccade energy
    eog_bandpass = apply_bandpass(eog_notched, fs, lowcut=0.05, highcut=35.0, order=4)
    results['eog_bandpass'] = eog_bandpass

    # Stage 5: Wavelet denoising (preserves edge structure)
    if enable_wavelet:
        eog_denoised = wavelet_denoise(eog_bandpass, wavelet='sym4', level=4)
    else:
        eog_denoised = eog_bandpass
    results['eog_denoised'] = eog_denoised

    # Stage 6: Saccade derivative
    eog_derivative = compute_saccade_derivative(eog_denoised, fs)
    results['eog_derivative'] = eog_derivative

    # Stage 7: Adaptive per-phase gain normalization
    if enable_adaptive_gain and timeline is not None:
        eog_normalized, gain_profile = adaptive_phase_gain(
            eog_denoised, t_uniform, timeline
        )
    else:
        eog_normalized = eog_denoised
        gain_profile = np.ones_like(eog_denoised)
    results['eog_normalized'] = eog_normalized
    results['gain_profile'] = gain_profile

    return results


# ==========================================================================
# PLOTTING FUNCTIONS
# ==========================================================================

def _add_phase_overlays(ax, timeline, y_min, y_max, y_range):
    """Add color-differentiated phase overlays to an axis."""
    for i, phase in enumerate(timeline):
        s_t = float(phase["start"])
        e_t = float(phase["end"])
        lbl = str(phase["label"]).upper()
        style = PHASE_STYLES.get(lbl, PHASE_STYLES["REST"])

        ax.axvspan(s_t, e_t, color=style["bg"], alpha=0.6, zorder=1)
        ax.axvline(s_t, color=style["border"], linestyle="--",
                   linewidth=1.0, alpha=0.8, zorder=2)

        mid_t = (s_t + e_t) / 2.0
        y_badge = y_max + (0.16 if (i % 2 == 0) else 0.05) * y_range
        badge_text = f"{style['icon']}\n({s_t:.0f}s \u2013 {e_t:.0f}s)"
        ax.text(mid_t, y_badge, badge_text, ha="center", va="center",
                fontsize=8.5, fontweight="bold", color=style["text"], zorder=5,
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                          edgecolor=style["border"], linewidth=1.2, alpha=0.95))

    target_dur = timeline[-1]["end"]
    ax.axvline(target_dur, color="#94a3b8", linestyle="--",
               linewidth=1.0, alpha=0.8, zorder=2)


def _add_sample_axis(ax, time_arr, eog, target_dur, phase_dur=2.0):
    """Add top axis showing sample indices at phase boundaries."""
    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())
    time_ticks = np.arange(0, target_dur + 0.1, phase_dur)
    sample_indices = [int(np.searchsorted(time_arr, t)) for t in time_ticks]
    sample_indices[-1] = min(sample_indices[-1], len(eog) - 1)
    ax_top.set_xticks(time_ticks)
    ax_top.set_xticklabels([f"S#{idx}" for idx in sample_indices],
                            fontsize=8, color="#64748b")
    ax_top.set_xlabel("Corresponding Sample Index (at each phase boundary)",
                       fontsize=9, fontweight="bold", color="#64748b", labelpad=6)


def plot_raw_eog(csv_path, sequence_id="Sequence_A", output_path=None, show=True):
    """Plot raw EOG with color-differentiated movement bands."""
    df = pd.read_csv(csv_path) if isinstance(csv_path, str) else csv_path
    timestamps = df["Timestamp"].values
    eog_raw = df["EOG"].values
    total_samples = len(df)
    subj = df["Subject_ID"].iloc[0] if "Subject_ID" in df.columns else "S01"
    att = df["Attempt_ID"].iloc[0] if "Attempt_ID" in df.columns else "A01"

    timeline = DEFAULT_SEQUENCES.get(sequence_id, DEFAULT_SEQUENCES["Sequence_A"])
    target_dur = timeline[-1]["end"]

    fig, ax = plt.subplots(figsize=(15, 6), dpi=150)
    ax.plot(timestamps, eog_raw, color="#2563eb", linewidth=1.2, label="Raw EOG Signal", zorder=3)

    y_min, y_max = float(np.min(eog_raw)), float(np.max(eog_raw))
    y_range = y_max - y_min if y_max != y_min else 1.0
    ax.set_ylim(y_min - 0.08 * y_range, y_max + 0.24 * y_range)

    _add_phase_overlays(ax, timeline, y_min, y_max, y_range)

    ax.set_xlim(0, max(target_dur, timestamps[-1]))
    ax.set_xlabel("Time (seconds) [Stimulus-Synchronized Timebase]", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_ylabel("Raw EOG / ADC Value (0\u20131023)", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_title(f"Raw EOG Signal with Ground-Truth Stimulus Alignment \u2014 {subj} | {att} | {sequence_id}",
                 fontsize=13, fontweight="bold", pad=12)
    ax.grid(True, linestyle=":", alpha=0.5, color="#cbd5e1", zorder=2)

    _add_sample_axis(ax, timestamps, eog_raw, target_dur)

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_cleaned_eog(csv_path, sequence_id="Sequence_A", output_path=None, show=True):
    """
    Plot advanced-processed EOG signal with full pipeline.

    This replaces the old simple bandpass+notch approach with the
    complete multi-stage pipeline for maximum spike visibility.
    """
    df = pd.read_csv(csv_path) if isinstance(csv_path, str) else csv_path
    timestamps = df["Timestamp"].values
    eog_raw = df["EOG"].values.astype(float)
    subj = df["Subject_ID"].iloc[0] if "Subject_ID" in df.columns else "S01"
    att = df["Attempt_ID"].iloc[0] if "Attempt_ID" in df.columns else "A01"

    time_arr = timestamps - timestamps[0]
    timeline = DEFAULT_SEQUENCES.get(sequence_id, DEFAULT_SEQUENCES["Sequence_A"])
    target_dur = timeline[-1]["end"]

    # Run full processing pipeline
    results = process_eog_signal(
        eog_raw, time_arr,
        fs_target=250.0,
        timeline=timeline,
        enable_wavelet=True,
        enable_adaptive_gain=True
    )

    t_plot = results['t_uniform']
    eog_plot = results['eog_normalized']

    fig, ax = plt.subplots(figsize=(15, 6), dpi=150)
    ax.plot(t_plot, eog_plot, color="#0284c7", linewidth=1.2,
            label="Cleaned EOG Signal", zorder=3)

    y_min, y_max = float(np.min(eog_plot)), float(np.max(eog_plot))
    y_range = y_max - y_min if y_max != y_min else 1.0
    ax.set_ylim(y_min - 0.08 * y_range, y_max + 0.24 * y_range)

    _add_phase_overlays(ax, timeline, y_min, y_max, y_range)

    ax.set_xlim(0, max(target_dur, t_plot[-1]))
    ax.set_xlabel("Time (seconds) [Stimulus-Synchronized Timebase]",
                  fontsize=11, fontweight="bold", labelpad=8)
    ax.set_ylabel("Processed EOG Amplitude (Adaptive Gain Normalized)",
                  fontsize=11, fontweight="bold", labelpad=8)
    pipeline_desc = "Resample+Detrend+Notch+BP(0.05-35Hz)"
    if HAS_PYWT:
        pipeline_desc += "+Wavelet"
    pipeline_desc += "+AdaptiveGain"
    ax.set_title(
        f"Advanced Processed EOG ({pipeline_desc})\n{subj} | {att} | {sequence_id}",
        fontsize=12, fontweight="bold", pad=12
    )
    ax.grid(True, linestyle=":", alpha=0.5, color="#cbd5e1", zorder=2)

    _add_sample_axis(ax, t_plot, eog_plot, target_dur)

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_diagnostic_panel(csv_path, sequence_id="Sequence_A",
                          output_path=None, show=True):
    """
    Generate a 4-panel diagnostic plot showing each processing stage.

    Panel layout:
    1. Raw signal + baseline estimate
    2. After detrending + bandpass (no wavelet, no gain)
    3. After wavelet denoising
    4. Saccade derivative (spike-enhanced)

    This is invaluable for understanding what each stage contributes
    and for tuning filter parameters.
    """
    df = pd.read_csv(csv_path) if isinstance(csv_path, str) else csv_path
    timestamps = df["Timestamp"].values
    eog_raw = df["EOG"].values.astype(float)
    subj = df["Subject_ID"].iloc[0] if "Subject_ID" in df.columns else "S01"
    att = df["Attempt_ID"].iloc[0] if "Attempt_ID" in df.columns else "A01"

    time_arr = timestamps - timestamps[0]
    timeline = DEFAULT_SEQUENCES.get(sequence_id, DEFAULT_SEQUENCES["Sequence_A"])
    target_dur = timeline[-1]["end"]

    results = process_eog_signal(
        eog_raw, time_arr,
        fs_target=250.0,
        timeline=timeline,
        enable_wavelet=True,
        enable_adaptive_gain=True
    )

    t = results['t_uniform']

    fig, axes = plt.subplots(4, 1, figsize=(16, 16), dpi=150, sharex=True)

    panels = [
        (axes[0], results['eog_resampled'],
         "Stage 1: Raw Resampled (250 Hz uniform grid)", "#2563eb"),
        (axes[1], results['eog_bandpass'],
         "Stage 2: Detrended + Notch + Bandpass (0.05-35 Hz)", "#059669"),
        (axes[2], results['eog_denoised'],
         f"Stage 3: Wavelet Denoised ({'SWT sym4' if HAS_PYWT else 'Savgol fallback'})",
         "#7c3aed"),
        (axes[3], results['eog_derivative'],
         "Stage 4: Saccade Derivative (rate of change)", "#dc2626"),
    ]

    for ax, signal, title, color in panels:
        ax.plot(t, signal, color=color, linewidth=1.0, zorder=3)
        y_min, y_max = float(np.min(signal)), float(np.max(signal))
        y_range = y_max - y_min if y_max != y_min else 1.0
        ax.set_ylim(y_min - 0.08 * y_range, y_max + 0.15 * y_range)

        # Simplified phase overlays (just colors, no text badges)
        for phase in timeline:
            s_t, e_t = float(phase["start"]), float(phase["end"])
            lbl = str(phase["label"]).upper()
            style = PHASE_STYLES.get(lbl, PHASE_STYLES["REST"])
            ax.axvspan(s_t, e_t, color=style["bg"], alpha=0.4, zorder=1)
            ax.axvline(s_t, color=style["border"], linestyle="--",
                       linewidth=0.5, alpha=0.6, zorder=2)

        ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
        ax.grid(True, linestyle=":", alpha=0.4, color="#cbd5e1", zorder=2)

    # Add phase labels only on top panel
    ax0 = axes[0]
    y_min0, y_max0 = ax0.get_ylim()
    y_range0 = y_max0 - y_min0
    for i, phase in enumerate(timeline):
        s_t, e_t = float(phase["start"]), float(phase["end"])
        lbl = str(phase["label"]).upper()
        style = PHASE_STYLES.get(lbl, PHASE_STYLES["REST"])
        mid_t = (s_t + e_t) / 2.0
        ax0.text(mid_t, y_max0 - 0.02 * y_range0, style["icon"],
                 ha="center", va="top", fontsize=7.5, fontweight="bold",
                 color=style["text"], zorder=5)

    axes[-1].set_xlabel("Time (seconds)", fontsize=11, fontweight="bold", labelpad=8)
    axes[0].set_xlim(0, max(target_dur, t[-1]))

    fig.suptitle(
        f"EOG Signal Processing Pipeline Diagnostic \u2014 {subj} | {att} | {sequence_id}",
        fontsize=14, fontweight="bold", y=0.98
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_gain_comparison(csv_path, sequence_id="Sequence_A",
                         output_path=None, show=True):
    """
    Side-by-side comparison: without vs with adaptive gain normalization.

    This clearly shows how weak UP/DOWN phases are amplified to match
    the stronger LEFT/RIGHT movements.
    """
    df = pd.read_csv(csv_path) if isinstance(csv_path, str) else csv_path
    timestamps = df["Timestamp"].values
    eog_raw = df["EOG"].values.astype(float)
    subj = df["Subject_ID"].iloc[0] if "Subject_ID" in df.columns else "S01"
    att = df["Attempt_ID"].iloc[0] if "Attempt_ID" in df.columns else "A01"

    time_arr = timestamps - timestamps[0]
    timeline = DEFAULT_SEQUENCES.get(sequence_id, DEFAULT_SEQUENCES["Sequence_A"])
    target_dur = timeline[-1]["end"]

    # Process without adaptive gain
    results_no_gain = process_eog_signal(
        eog_raw, time_arr, fs_target=250.0, timeline=timeline,
        enable_wavelet=True, enable_adaptive_gain=False
    )

    # Process with adaptive gain
    results_with_gain = process_eog_signal(
        eog_raw, time_arr, fs_target=250.0, timeline=timeline,
        enable_wavelet=True, enable_adaptive_gain=True
    )

    t = results_no_gain['t_uniform']

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), dpi=150, sharex=True)

    # Panel 1: Without gain
    sig1 = results_no_gain['eog_denoised']
    ax1.plot(t, sig1, color="#0284c7", linewidth=1.2, zorder=3)
    y1_min, y1_max = float(np.min(sig1)), float(np.max(sig1))
    y1_range = y1_max - y1_min if y1_max != y1_min else 1.0
    ax1.set_ylim(y1_min - 0.08 * y1_range, y1_max + 0.24 * y1_range)
    _add_phase_overlays(ax1, timeline, y1_min, y1_max, y1_range)
    ax1.set_title("Without Adaptive Gain (note flat UP/DOWN phases)",
                  fontsize=12, fontweight="bold", pad=8)
    ax1.set_ylabel("EOG Amplitude", fontsize=10, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.5, color="#cbd5e1", zorder=2)

    # Panel 2: With gain
    sig2 = results_with_gain['eog_normalized']
    ax2.plot(t, sig2, color="#7c3aed", linewidth=1.2, zorder=3)
    y2_min, y2_max = float(np.min(sig2)), float(np.max(sig2))
    y2_range = y2_max - y2_min if y2_max != y2_min else 1.0
    ax2.set_ylim(y2_min - 0.08 * y2_range, y2_max + 0.24 * y2_range)
    _add_phase_overlays(ax2, timeline, y2_min, y2_max, y2_range)
    ax2.set_title("With Adaptive Gain Normalization (UP/DOWN enhanced)",
                  fontsize=12, fontweight="bold", pad=8)
    ax2.set_ylabel("Normalized EOG Amplitude", fontsize=10, fontweight="bold")
    ax2.set_xlabel("Time (seconds)", fontsize=11, fontweight="bold", labelpad=8)
    ax2.grid(True, linestyle=":", alpha=0.5, color="#cbd5e1", zorder=2)

    ax1.set_xlim(0, max(target_dur, t[-1]))

    fig.suptitle(
        f"Adaptive Gain Comparison \u2014 {subj} | {att} | {sequence_id}",
        fontsize=14, fontweight="bold", y=0.98
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


# ==========================================================================
# CLI ENTRY POINT
# ==========================================================================

if __name__ == "__main__":
    import glob
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = current_dir if os.path.basename(current_dir).lower() != "demo" else os.path.dirname(current_dir)
    data_dir = os.path.join(project_dir, "New Data Plots")
    csv_files = sorted(glob.glob(os.path.join(data_dir, "S*.csv")), key=os.path.getmtime)

    if csv_files:
        latest_csv = csv_files[-1]
        print(f"Processing latest attempt: {latest_csv}")
        print(f"PyWavelets available: {HAS_PYWT}")
        print()

        # Generate all plot types
        plot_raw_eog(latest_csv, "Sequence_A", show=False)
        plot_cleaned_eog(latest_csv, "Sequence_A", show=False)
        plot_diagnostic_panel(latest_csv, "Sequence_A", show=False)
        plot_gain_comparison(latest_csv, "Sequence_A", show=False)
        print("All plots generated successfully!")
    else:
        print(f"No attempt CSVs found in {data_dir}")
