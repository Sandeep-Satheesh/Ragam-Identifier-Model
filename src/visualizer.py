import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sounddevice as sd
import time
import threading
from . import constants as const
from . import util

# ---- Mixer that plays a background buffer + a real-time sine ----
class MixerWithSine:
    def __init__(self, sr=16000, channels=1, blocksize=const.AUDIO_BLOCK_SIZE, dtype='float32'):
        self.sr = sr
        self.channels = channels
        self.blocksize = blocksize
        self.dtype = dtype

        # single background source (mono/stereo) stored as {'data','pos','gain','active'}
        self.source = None
        self.lock = threading.Lock()

        # sine generator state (protected by lock)
        self.sine = {
            'freq': 440.0,
            'gain': 0.0,
            'active': False,
            'phase': 0.0
        }

        self.stream = sd.OutputStream(
            samplerate=self.sr,
            channels=self.channels,
            dtype=self.dtype,
            blocksize=self.blocksize,
            callback=self._callback
        )

    def add_background(self, data, gain=1.0, start_active=True):
        """
        Add a background buffer to be played from start. Data -> (N,) or (N,channels).
        """
        data = np.asarray(data, dtype=np.float32)
        if data.ndim == 1:
            if self.channels == 1:
                data = data.reshape(-1, 1)
            else:
                data = np.stack([data] * self.channels, axis=1)
        elif data.ndim == 2 and data.shape[1] != self.channels:
            # try to adapt channels
            if data.shape[1] == 1 and self.channels == 2:
                data = np.repeat(data, 2, axis=1)
            elif data.shape[1] == 2 and self.channels == 1:
                data = data.mean(axis=1, keepdims=True)
            else:
                raise ValueError("Background channel mismatch")
        with self.lock:
            self.source = {'data': data, 'pos': 0, 'gain': float(gain), 'active': bool(start_active)}

    def start(self):
        self.stream.start()

    def stop(self):
        with self.lock:
            if self.source is not None:
                self.source['active'] = False
        self.stream.stop()

    def close(self):
        self.stream.close()

    def set_sine(self, freq=None, gain=None, active=None):
        with self.lock:
            if freq is not None:
                self.sine['freq'] = float(freq)
            if gain is not None:
                self.sine['gain'] = float(gain)
            if active is not None:
                self.sine['active'] = bool(active)

    def _callback(self, outdata, frames, time_info, status):
        out = np.zeros((frames, self.channels), dtype=np.float32)

        # mix background
        with self.lock:
            src = self.source.copy() if self.source is not None else None

        if src is not None and src['active']:
            pos = src['pos']
            data = src['data']
            available = data.shape[0] - pos
            if available > 0:
                take = min(frames, available)
                chunk = data[pos:pos + take] * src['gain']
                out[:take, :] += chunk
                with self.lock:
                    self.source['pos'] += take
            else:
                # background finished
                with self.lock:
                    self.source['active'] = False

        # generate sine
        with self.lock:
            sine_state = dict(self.sine)

        if sine_state['active'] and sine_state['gain'] > 0.0:
            freq = sine_state['freq']
            gain = sine_state['gain']
            phase = sine_state['phase']

            # sample times for this block
            t = (np.arange(frames) / self.sr).astype(np.float32)
            angles = 2.0 * np.pi * freq * t + phase
            chunk = (gain * np.sin(angles)).astype(np.float32)

            # tile into channels
            if self.channels == 1:
                out[:, 0] += chunk
            else:
                out[:] += np.expand_dims(chunk, 1)

            # compute new phase for continuity
            last_angle = angles[-1]
            # advance phase by one sample step to keep continuity next callback
            new_phase = (last_angle + 2.0 * np.pi * freq / self.sr) % (2.0 * np.pi)
            with self.lock:
                self.sine['phase'] = float(new_phase)

        # simple limiter to avoid clipping
        peak = np.max(np.abs(out))
        if peak > 1.0:
            out = out / peak

        outdata[:] = out

# ---- helper to map label -> frequency from swara_map (robust to suffixes) ----
def label_to_freq(label, swara_map):
    if label is None:
        return None
    if label in swara_map:
        return float(swara_map[label])
    # try prefix match: prefer the longest match
    best = None
    best_len = -1
    for k in swara_map.keys():
        if label.startswith(k) and len(k) > best_len:
            best = k
            best_len = len(k)
    if best is not None:
        return float(swara_map[best])
    return None

# ---- Modified visualize_notes ----
def visualize_notes(
    waveform,
    f0_pitch,
    swara_labels,
    swara_map,
    play_sine_audio: bool = False,
    hop_len=const.HOP_LENGTH,
    sr=const.SAMPLE_RATE
):
    """
    waveform: 1D numpy audio array (mono) or (N,channels) matching sr
    f0_pitch: array-like length N_frames (one value per frame; can be unused)
    swara_labels: list of labels per frame (length N_frames)
    swara_map: dict {label: freq_hz} (can be base labels; label_to_freq handles suffixes)
    play_sine_audio: if True, will synthesize sine for each frame's swara freq
    """

    # --- times + y-values ---
    f0 = util.convert_to_numpy_cpu(f0_pitch)
    times = np.arange(len(f0)) * hop_len / sr

    # Use keys of swara_map in insertion order (dict preserves insertion order)
    unique_swaras = [l for l in dict.fromkeys(swara_map)]
    # keep the same special labels
    unique_swaras.append("irregular_pitch")
    unique_swaras.append("lowconf")
    swara_to_y = {sw: i for i, sw in enumerate(unique_swaras)}

    y_vals = [swara_to_y[l] if l in swara_to_y else np.nan for l in swara_labels]

    # --- setup mixer if required ---
    mixer = None
    if play_sine_audio:
        # ensure waveform is numpy float32 and mono
        waveform = np.asarray(waveform, dtype=np.float32)
        if waveform.ndim == 2:
            # take mean to mono
            waveform = waveform.mean(axis=1)
        mixer = MixerWithSine(sr=sr, channels=1, blocksize=const.AUDIO_BLOCK_SIZE)
        mixer.add_background(waveform, gain=0.5, start_active=True)
        # start stream before animation to have stream.time valid
        mixer.start()

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(16, 9))
    scatter = ax.scatter(times, y_vals, s=10, c=y_vals, cmap="tab20", alpha=0.7)

    ax.set_yticks(list(swara_to_y.values()))
    ax.set_yticklabels(list(swara_to_y.keys()))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Swara")
    ax.set_title("Detected swaras over time")

    # Seek bar line
    seek_line = ax.axvline(0, color="red", linewidth=2)
    ax.set_xlim(0, times[-1] if len(times) > 0 else 1)
    ax.set_ylim(-1, len(swara_to_y))

    # dynamic label (large enough to be visible)
    text_obj = ax.text(0.75, 0.9, "Current Swara: -",
                       transform=ax.transAxes, ha="center", va="top",
                       fontsize=10, fontweight="bold", color="darkred",
                       bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"))

    # sync reference: use stream time if we have mixer else wall-clock with sd.play()
    started = {'flag': False}
    start_time_ref = {'time': None}

    # audio latency compensation (only used when not using mixer)

    # sine volume & safety limits
    sine_gain = 0.50
    min_freq = 30.0
    max_freq = sr / 2.0 - 100.0
    t0 = None

    def update(frame):
        nonlocal started, start_time_ref

        # determine elapsed using mixer.stream.time (preferred) or wall-clock with sd
        if play_sine_audio and mixer is not None:
            if not started['flag']:
                # anchor the start time to current stream time
                start_time_ref['time'] = mixer.stream.time
                started['flag'] = True
            block_latency = mixer.blocksize / mixer.sr
            elapsed = (mixer.stream.time - start_time_ref['time']) + block_latency * 8
        else:
            # fallback: use wall time + sd.play latency compensation
            nonlocal t0
            try:
                if not started['flag']:
                    start_time_ref['time'] = time.time()
                    # start background with sd.play once
                    sd.play(waveform, samplerate=sr)
                    started['flag'] = True
                elapsed = time.time() - start_time_ref['time'] - const.AUDIO_LATENCY_SEC
            except Exception:
                elapsed = time.time() - start_time_ref.get('time', time.time())

        elapsed = max(0.0, min(elapsed, times[-1] if len(times) > 0 else 0.0))
        seek_line.set_xdata([elapsed, elapsed])

        # frame index for this time
        idx = int(np.searchsorted(times, elapsed))
        if idx >= len(swara_labels):
            idx = len(swara_labels) - 1

        label = swara_labels[idx] if idx >= 0 and idx < len(swara_labels) else "-"
        # decide display label
        display_label = label if label not in ("irregular_pitch", "lowconf") else "-"

        text_obj.set_text(f"Current Swara: {display_label} ({elapsed:.2f}s)")

        # handle sine: set frequency/gain on mixer
        if play_sine_audio and mixer is not None:
            if label not in ("irregular_pitch", "lowconf", "-", None):
                freq = label_to_freq(label, swara_map)
                if freq is not None and not np.isnan(freq):
                    freq_clamped = float(np.clip(freq, min_freq, max_freq))
                    mixer.set_sine(freq=freq_clamped, gain=sine_gain, active=True)
                else:
                    mixer.set_sine(gain=0.0, active=False)
            else:
                mixer.set_sine(gain=0.0, active=False)

        return seek_line, text_obj

    ani = animation.FuncAnimation(fig, update, interval=30, blit=True)

    # ensure mixer stops when figure is closed
    def on_close(event):
        try:
            if mixer is not None:
                mixer.stop()
                mixer.close()
        except Exception:
            pass

    fig.canvas.mpl_connect('close_event', on_close)

    # position window top-left (best-effort)
    mngr = plt.get_current_fig_manager()
    try:
        mngr.window.wm_geometry("+0+0")  # Tk
    except Exception:
        try:
            mngr.window.move(0, 0)       # Qt
        except Exception:
            pass

    plt.show()

    # cleanup after plt.show() returns
    try:
        if mixer is not None:
            mixer.stop()
            mixer.close()
    except Exception:
        pass