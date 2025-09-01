import numpy as np
import torchcrepe
import torch
import torch.nn.functional as F
import logging
from collections import OrderedDict, Counter
import sounddevice as sd
from . import constants as const
from . import util
from scipy.ndimage import gaussian_filter1d
from scipy.stats import circmean

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

def torch_medfilt(x, kernel_size=5):
    # x: (1, time) 2D tensor
    pad = kernel_size // 2
    x_padded = F.pad(x.unsqueeze(0), (pad, pad), mode="reflect")
    # Unfold into sliding windows
    x_unfold = x_padded.unfold(dimension=2, size=kernel_size, step=1)  # (1, 1, time, k)
    # Take median along the window dimension
    x_med, _ = x_unfold.median(dim=-1)
    return x_med.view(-1)

def extract_pitch(y, sr, hop_length : int = const.HOP_LENGTH):
    logger.info("\t--> Extracting pitch frequencies in the audio sample...")
    y = torch.tensor(y).unsqueeze(0).to("cuda")

    freqs, harmonicity = torchcrepe.predict(
        y,sr,hop_length,const.FMIN_HZ,const.FMAX_HZ,
        model='full',   # 'full' = higher accuracy, 'tiny' = faster
        decoder=torchcrepe.decode.weighted_argmax,
        batch_size=const.AUDIO_BATCH_SIZE,device="cuda",return_harmonicity=True
    )
    
    smoothed_pitch = torch_medfilt(freqs, kernel_size=const.TORCH_MEDFILT_KERNEL_SIZE)
    
    return smoothed_pitch, harmonicity

def predict_sa_frequency(f0, conf, fmin=const.FMIN_HZ, fmax=const.FMAX_HZ, bins=const.HISTOGRAM_BINS_SA_ESTIMATION, smooth_sigma=1.0):
    logger.info("\t--> Predicting Sa frequency...")
    # f0 = your smoothed pitch array (Hz)
    f0 = util.convert_to_numpy_cpu(f0)
    conf = util.convert_to_numpy_cpu(conf)

    f0 = f0[(f0 > 0)]
    if f0.size == 0:
        return None

    # range filter
    mask = (f0 >= fmin) & (f0 <= fmax)
    f0 = f0[mask]
    if conf is not None:
        conf = np.asarray(conf).reshape(-1)[mask]
    else:
        conf = np.ones_like(f0)

    if f0.size == 0:
        return None

    # log2 and fold to 0..1
    logf = np.log2(f0)
    folded = np.mod(logf, 1.0)  # values in [0,1)

    # weighted histogram
    hist, edges = np.histogram(folded, bins=bins, range=(0.0, 1.0), weights=conf)
    hist_smooth = gaussian_filter1d(hist.astype(float), sigma=smooth_sigma)

    peak = np.argmax(hist_smooth)
    folded_center = 0.5 * (edges[peak] + edges[peak+1])

    # Map back to a real octave near the median logf
    median_oct = int(np.round(np.median(logf)))
    sa_log = folded_center + median_oct
    sa_freq = 2.0 ** sa_log

    # As fallback use circular mean if histogram is flat
    if hist_smooth.max() < 1e-6:
        circ = circmean(2*np.pi*folded, high=2*np.pi) / (2*np.pi)
        folded_center = circ % 1.0
        sa_log = folded_center + median_oct
        sa_freq = 2.0 ** sa_log
    
    # Clamp to expected range
    while sa_freq < const.EXPECTED_SA_RANGE_HZ[0]:
        sa_freq *= 2.0
    
    while sa_freq > const.EXPECTED_SA_RANGE_HZ[1]:
        sa_freq /= 2.0

    return float(sa_freq)


def build_swara_map(
    smooth_pitch: np.ndarray,
    sa_freq: float,
    search_octaves=const.SWARA_MAPPING_SEARCH_OCTAVES
):
    """
    Build a swara map around a given Sa (tonic) frequency using 12-TET.
    Each octave is labeled explicitly: Sa_o0, Ri1_o0, etc.
    
    Octaves are only generated if at least one swara in that octave 
    falls within the observed pitch range of the audio.

    Args:
        smooth_pitch: numpy array of f0 in Hz (already smoothed, >0 only)
        sa_freq: tonic frequency in Hz
        search_octaves: candidate octave indices to consider (relative to base Sa)
        margin_cents: how much tolerance beyond the observed pitch range to allow

    Returns:
        OrderedDict mapping swara label with octave to frequency in Hz
    """
    # Swara offsets in semitones from Sa
    swaras = [
        ("Sa", 0),
        ("Ri1", 1), ("Ri2", 2),
        ("Ga1", 3), ("Ga2", 4),
        ("Ma1", 5), ("Ma2", 6),
        ("Pa", 7),
        ("Da1", 8), ("Da2", 9),
        ("Ni1", 10), ("Ni2", 11),
    ]

    # observed pitch range
    valid = smooth_pitch[(smooth_pitch > 0) & np.isfinite(smooth_pitch)]
    if len(valid) == 0:
        return OrderedDict()

    min_pitch = valid.min()
    max_pitch = valid.max()

    swara_map = OrderedDict()

    for o in search_octaves:
        octave_has_overlap = False
        octave_entries = []
        for label, semitone in swaras:
            freq = sa_freq * (2 ** ((semitone + 12 * o) / 12))
            if min_pitch <= freq <= max_pitch:
                octave_has_overlap = True
            octave_entries.append((f"{label}_o{o}", freq))

        if octave_has_overlap:
            for lbl, freq in octave_entries:
                swara_map[lbl] = freq

    return swara_map

def map_pitch_to_swaras_direct(
    f0: np.ndarray,
    sa_hz: float,
    swara_map: dict,
    confidence: np.ndarray = None,
    conf_thresh: float = const.SWARA_MAPPING_CONFIDENCE_THRESHOLD,
    swara_tolerances: dict = const.SWARA_MAPPING_TOLERANCES,   # NEW: optional per-swara tolerance dict
):
    """
    Map each f0 frame to nearest swara (with octave) if within tolerance.
    Uses confidence from torchcrepe to mask low-confidence frames.

    Args:
        f0: (N,) or (1,N) array of frequencies in Hz
        sa_hz: tonic frequency
        swara_map: dict {swara_name: frequency in Hz}
        confidence: (N,) array of torchcrepe confidence scores [0,1]
        tolerance_cents: default max deviation from swara center (used if no per-swara tolerance given)
        conf_thresh: minimum confidence to accept mapping
        swara_tolerances: dict {swara_name_base: cents} to override tolerance for each swara
                         Example: {"Sa": 50, "Pa": 50, "Ri1": 30, "Ri2": 30, "Ga1": 30, ...}

    Returns:
        list of swara labels (length N),
        'lowconf' for unvoiced/low-conf frames,
        'irregular_pitch' for out-of-tolerance frames
    """
    # flatten
    f0 = np.asarray(f0).reshape(-1)

    # optional confidence mask
    if confidence is None:
        confidence = np.ones_like(f0)
    else:
        confidence = np.asarray(confidence).reshape(-1)

    # swara centers in semitones relative to Sa
    swara_names = list(swara_map.keys())
    swara_vals = np.array(list(swara_map.values()))
    swara_vals_st = 12 * np.log2(swara_vals / sa_hz)

    # f0 -> semitones wrt Sa
    valid = (f0 > 0) & np.isfinite(f0) & (confidence >= conf_thresh)
    st = np.full_like(f0, np.nan, dtype=float)
    st[valid] = 12 * np.log2(f0[valid] / sa_hz)

    labels = []
    irregular_pitch_count = 0

    for val, is_ok in zip(st, valid):
        if not is_ok:
            labels.append("lowconf")
            continue

        # distance to each swara
        cents_dist = 100 * np.abs(swara_vals_st - val)
        idx = np.argmin(cents_dist)
        swara_name = swara_names[idx]

        # extract base swara (strip octave suffix like "_o0")
        base_swara = swara_name.split("_")[0]

        # pick tolerance: per-swara if given, else global
        tol = (
            swara_tolerances.get(base_swara, const.SWARA_MAPPING_DEFAULT_TOLERANCE_CENTS)
            if swara_tolerances is not None
            else tolerance_cents
        )

        if cents_dist[idx] <= tol:
            labels.append(swara_name)
        else:
            irregular_pitch_count += 1
            labels.append("irregular_pitch")

    logger.info(
        f"\t--> Detected {irregular_pitch_count}/{len(valid)} ({round(irregular_pitch_count/len(valid), 2)}) frames with irregular pitch (not matching any swara within tolerance)"
    )

    return labels


def play_sine_tone(frequency, duration, samplerate=44100, amplitude=1):
    """
    Plays a sine tone of a given frequency and duration.

    Args:
        frequency (float): The frequency of the sine wave in Hz.
        duration (float): The duration of the tone in seconds.
        samplerate (int): The sampling rate in samples per second (Hz).
        amplitude (float): The amplitude of the sine wave (0.0 to 1.0).
    """
    t = np.linspace(0, duration, int(duration * samplerate), endpoint=False)
    # Generate the sine wave
    sine_wave = amplitude * np.sin(2 * np.pi * frequency * t)

    # Play the sine wave
    sd.play(sine_wave, samplerate)
    sd.wait()

def get_swaras_for_frames(smooth_pitch: torch.Tensor, confidence: torch.Tensor, sa_freq: float) -> dict:
    """
    Map each frame's pitch to swaras using the detected Sa frequency.
    """

    # --- numpy conversion ---
    smooth_pitch = util.convert_to_numpy_cpu(smooth_pitch)
    confidence = util.convert_to_numpy_cpu(confidence)

    # --- map to swaras ---
    swara_map = build_swara_map(smooth_pitch, sa_freq)
    swara_seq = map_pitch_to_swaras_direct(smooth_pitch, sa_freq, swara_map, confidence)
    
    # for s in swara_map:
    #     if s in swara_seq.keys():
    #         print('Playing swara:', s)
    #         play_sine_tone(swara_map[s], 2)

    return swara_seq, swara_map