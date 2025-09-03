import os
import torch
from torch.utils.data import Dataset
import numpy as np
import json
import logging
from tqdm import tqdm
from src import constants

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

class CarnaticPitchDataset(Dataset):
    def __init__(self, base_dir, snippet_seconds, overlap, downsample_factor,
                 force_rebuild=False, convert_to_npy=True, augment=False):
        self.base_dir = base_dir
        self.feat_dir = os.path.join(base_dir, "features")
        info_dir = os.path.join(base_dir, "_info_")
        self.augment = augment
        self.index_cache = os.path.join(info_dir, "snippet_index.json")

        logger.info("\t--> Loading metadata files...")

        with open(os.path.join(info_dir, "path_mbid_ragaid.json")) as f:
            mbid_to_meta = json.load(f)
        with open(os.path.join(info_dir, "ragaId_to_ragaName_mapping.json")) as f:
            ragaid_to_name = json.load(f)

        logger.info("\t--> Building metadata for lookups...")

        self.metadata = []
        for mbid, meta in mbid_to_meta.items():
            rel_path = meta["path"]
            rel_path = rel_path.replace("RagaDataset/Carnatic/", "").replace("audio", "features")
            raga_name = ragaid_to_name[str(meta["ragaid"])]
            self.metadata.append({"file": rel_path, "raga": raga_name})

        raga_names = sorted({m["raga"] for m in self.metadata})
        self.raga2idx = {r: i for i, r in enumerate(raga_names)}

        example_file = os.path.join(self.base_dir, self.metadata[0]["file"] + ".pitchSilIntrpPP")
        f0_data = np.loadtxt(example_file)
        hop = np.mean(np.diff(f0_data[:, 0]))
        hop *= downsample_factor
        self.downsample_factor = downsample_factor

        self.snippet_frames = int(snippet_seconds / hop)
        self.hop_frames = int(self.snippet_frames * (1 - overlap))
        logger.info(f"\t--> hop={hop:.4f}s (downsample={downsample_factor}) → "
                    f"snippet_frames={self.snippet_frames}, hop_frames={self.hop_frames}")

        if os.path.exists(self.index_cache) and not force_rebuild:
            logger.info(f"\t--> Loading cached snippet index from {self.index_cache}")
            with open(self.index_cache, "r") as f:
                self.samples = json.load(f)
            logger.info(f"\t--> Loaded {len(self.samples)} snippets from cache.")
        else:
            logger.info("\t--> Building snippet samples from scratch...")
            self.samples = []
            for m in self.metadata:
                base_path = os.path.join(self.base_dir, m["file"])
                pitch_file = base_path + ".pitchSilIntrpPP"
                with open(pitch_file) as f:
                    num_frames = sum(1 for _ in f) // downsample_factor
                    start = 0
                    while start + self.snippet_frames <= num_frames:
                        self.samples.append({
                            "file": m["file"],
                            "raga": m["raga"],
                            "start": start
                        })
                        start += self.hop_frames

            with open(self.index_cache, "w") as f:
                json.dump(self.samples, f)
            logger.info(f"\t--> Built and cached {len(self.samples)} snippets.")
        
        if convert_to_npy:
            self._maybe_convert_to_npy()

    def _maybe_convert_to_npy(self):
        logger.info("\t--> Checking for .npy cache files and converting if not available...")
        for s in tqdm(self.samples):
            base_path = os.path.join(self.base_dir, s["file"])
            pitch_txt = base_path + ".pitchSilIntrpPP"
            pitch_npy = base_path + ".freqs.npy"

            if not os.path.exists(pitch_npy):
                try:
                    f0_data = np.loadtxt(pitch_txt)
                    freqs = f0_data[:, 1].copy()
                    np.save(pitch_npy, freqs)
                except Exception as e:
                    logger.warning(f"\t--> Failed to convert {pitch_txt}: {e}")
        logger.info("\t--> .npy conversion done.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        entry = self.samples[idx]
        base_path = os.path.join(self.base_dir, entry["file"])

        pitch_npy = base_path + ".freqs.npy"
        pitch_txt = base_path + ".pitchSilIntrpPP"

        if os.path.exists(pitch_npy):
            freqs = np.load(pitch_npy)
        else:
            f0_data = np.loadtxt(pitch_txt)
            freqs = f0_data[:, 1]

        freqs = freqs.copy()

        # --- Augmentation ---
        if self.augment:
            freqs = self.apply_augmentation(freqs)

        # --- Downsampling ---
        freqs = freqs[::self.downsample_factor]

        # Load tonic
        tonic_file = base_path + ".tonicFine"
        tonic = float(open(tonic_file).read().strip())

        # Normalize to cents relative to tonic
        cents = np.full_like(freqs, -100.0, dtype=np.float32)
        valid = freqs > 0
        cents[valid] = 1200 * np.log2(freqs[valid] / tonic)

        # --- Slice snippet ---
        start = entry["start"]
        seq = cents[start:start + self.snippet_frames]

        # --- Guarantee fixed length ---
        if len(seq) < self.snippet_frames:
            seq = np.pad(seq, (0, self.snippet_frames - len(seq)), constant_values=-100.0)
        elif len(seq) > self.snippet_frames:
            seq = seq[:self.snippet_frames]

        # Final safety: enforce float32 + contiguous memory
        seq = np.ascontiguousarray(seq, dtype=np.float32)

        # Shape = (1, T)
        x = torch.from_numpy(seq).unsqueeze(0).clone()
        y = self.raga2idx[entry["raga"]]
        return x, y

    def apply_augmentation(self, freqs):
        """Apply random combination of tonic shift, tempo stretch, and noise.
        Ensures output length = self.snippet_frames.
        """

        # --- Tonic shift (Hz scaling) ---
        if np.random.rand() < constants.AUGMENT_PROBS["tonic_shift"]:
            shift_cents = np.random.uniform(-constants.AUGMENT_RANGES["tonic_shift"],
                                            constants.AUGMENT_RANGES["tonic_shift"])
            factor = 2 ** (shift_cents / 1200)  # cents → multiplicative factor
            freqs = freqs * factor

        # --- Tempo stretch (time-warp with fixed output length) ---
        if np.random.rand() < constants.AUGMENT_PROBS["tempo"]:
            stretch = np.random.uniform(1 - constants.AUGMENT_RANGES["tempo"],
                                        1 + constants.AUGMENT_RANGES["tempo"])
            # target timeline (fixed length)
            idx = np.linspace(0, len(freqs) - 1, self.snippet_frames)
            # warp timeline by dividing by stretch
            warped_idx = idx / stretch
            # clip to valid range
            warped_idx = np.clip(warped_idx, 0, len(freqs) - 1)
            freqs = np.interp(warped_idx, np.arange(len(freqs)), freqs)

        # --- Noise injection (still in Hz) ---
        if np.random.rand() < constants.AUGMENT_PROBS["noise"]:
            noise_cents = np.random.normal(0, constants.AUGMENT_RANGES["noise"], size=freqs.shape)
            freqs = freqs * (2 ** (noise_cents / 1200))

        # --- Ensure clean contiguous float32 array ---
        return np.ascontiguousarray(freqs, dtype=np.float32).copy()
