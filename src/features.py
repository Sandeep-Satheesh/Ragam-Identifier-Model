import numpy as np
import torchcrepe
import torch
import torch.nn.functional as F
import logging
from collections import Counter
import sounddevice as sd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from collections import Counter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

fmin = 100  # Hz, adapt for your singer
fmax = 400  # Hz, adapt for your singer

def torch_moving_average(x, window=5):
    # x: (time,) 1D tensor
    kernel = torch.ones(1, 1, window, device=x.device) / window
    x_smooth = F.conv1d(x, kernel, padding=window//2)
    return x_smooth  # back to (time,)

def extract_pitch(y, sr):
    logger.info("\t--> Extracting pitch frequencies in the audio sample...")
    y = torch.tensor(y).unsqueeze(0).to("cuda")

    hop_length = 80  # 10 ms hop length
    freqs, harmonicity = torchcrepe.predict(
        y,sr,hop_length,fmin,fmax,
        model='full',   # 'full' = higher accuracy, 'tiny' = faster
        batch_size=128,device="cuda",return_harmonicity=True
    )

    smoothed_pitch = torch_moving_average(freqs, window=3)
    
    return smoothed_pitch, harmonicity

def predict_sa_frequency(f0):
    logger.info("\t--> Predicting Sa frequency...")
    # f0 = your smoothed pitch array (Hz)
    f0 = f0.cpu().numpy() if isinstance(f0, torch.Tensor) else np.array(f0)
    f0 = f0[f0 > 0]    # remove unvoiced
    
    pitch_mask = (f0 >= fmin) & (f0 <= fmax)
    f0[~pitch_mask] = np.nan
    f0 = f0[~np.isnan(f0)]  # remove frequencies outside range
    log_f0 = np.log2(f0)  # convert to log2 scale (octaves)

    logger.info("\t--> Building frequency histogram...")
    # fold into 1 octave around middle C (~261 Hz) or any ref
    folded = (log_f0 % 1.0)  # keep only fractional part (within one octave)

    hist, bin_edges = np.histogram(folded, bins=120)
    peak_bin = np.argmax(hist)
    sa_folded = 0.5 * (bin_edges[peak_bin] + bin_edges[peak_bin+1])

    # map back to actual tonic near median pitch
    median_octave = int(np.median(log_f0))
    sa_log = sa_folded + median_octave
    sa_freq = 2 ** sa_log
    
    return sa_freq

def build_swara_map(sa_freq: float, octaves=[0,1]):
    """
    Build a swara map around a given Sa (tonic) frequency using 12-TET.
    Each octave is labeled explicitly: Sa2, Ri1, Ri2, etc.
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
    
    swara_map = {}
    
    for label, semitone in swaras:
        freq = sa_freq * (2 ** (semitone / 12))
        swara_map[f"{label}"] = freq
    
    swara_map['Sa\''] = sa_freq * 2
    return swara_map

def map_pitch_to_swaras_clustered(
    pitch_contour: np.ndarray,
    swara_map: dict,
    min_frames_ratio: float = 0.02,
    merge_cents: float = 30.0,
    k_range=(5, 9),
):
    """
    Cluster pitch contour using a music-aware heuristic for number of swaras.
    Returns swara_counter and confidence.
    """

    def freq_to_cents(f1, f2):
        return 1200 * np.log2(f1 / f2)

    # Remove zeros
    pitch_contour = pitch_contour[pitch_contour > 0]
    if len(pitch_contour) == 0:
        return Counter(), {}

    log_pitch = np.log2(pitch_contour).reshape(-1, 1)
    best_k = None
    best_score = -np.inf
    best_labels = None
    best_centroids = None
    total_frames = len(pitch_contour)

    # --- Heuristic loop ---
    for k in range(k_range[0], k_range[1]+1):
        kmeans = KMeans(n_clusters=k, n_init=50, max_iter=500, random_state=42).fit(log_pitch)
        labels = kmeans.labels_
        centroids = 2 ** kmeans.cluster_centers_.flatten()
        cluster_counts = np.array([np.sum(labels == i) for i in range(k)])
        # Filter tiny clusters
        valid_idx = cluster_counts / total_frames >= min_frames_ratio
        cluster_counts = cluster_counts[valid_idx]
        centroids = centroids[valid_idx]
        # Merge close clusters
        merged = []
        merged_counts = []
        used = np.zeros(len(centroids), dtype=bool)
        for i, c in enumerate(centroids):
            if used[i]:
                continue
            group = [i]
            for j, c2 in enumerate(centroids):
                if i != j and not used[j]:
                    if abs(freq_to_cents(c, c2)) < merge_cents:
                        group.append(j)
            merged_freq = np.average(centroids[group], weights=cluster_counts[group])
            merged_count = np.sum(cluster_counts[group])
            merged.append(merged_freq)
            merged_counts.append(merged_count)
            for g in group:
                used[g] = True
        merged = np.array(merged)
        merged_counts = np.array(merged_counts)

        # Compute heuristic score: total frames / (number of clusters)^0.5
        coverage = np.sum(merged_counts) / total_frames
        score = coverage / np.sqrt(len(merged))
        if score > best_score:
            best_score = score
            best_k = k
            best_centroids = merged
            best_labels = merged_counts

    # --- Map centroids to swaras ---
    swara_names = list(swara_map.keys())
    swara_freqs = np.array(list(swara_map.values()))

    swara_counter = Counter()
    swara_confidence = {}
    for c, count in zip(best_centroids, best_labels):
        cents_diff = np.abs([freq_to_cents(c, f) for f in swara_freqs])
        min_idx = np.argmin(cents_diff)
        swara_name = swara_names[min_idx]
        swara_counter[swara_name] += count
        swara_confidence[swara_name] = swara_confidence.get(swara_name, 0) + count / total_frames

    return swara_counter, swara_confidence

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

def swara_counter(smooth_pitch: torch.Tensor, confidence: torch.Tensor, sa_freq: float) -> dict:
    """
    Filter low-confidence frames, map to swaras, and count.
    """

    # --- numpy conversion ---
    smooth_pitch = smooth_pitch.cpu().numpy() if isinstance(smooth_pitch, torch.Tensor) else np.array(smooth_pitch)
    confidence = confidence.cpu().numpy() if isinstance(confidence, torch.Tensor) else np.array(confidence)

    # --- map to swaras ---
    swara_map = build_swara_map(sa_freq)
    swara_seq, swara_confidence = map_pitch_to_swaras_clustered(smooth_pitch, swara_map)

    for s in swara_map:
        if s in swara_seq.keys():
            print('Playing swara:', s)
            play_sine_tone(swara_map[s], 1)
    # --- aggregate counts ---
    print(swara_confidence)
    return swara_seq