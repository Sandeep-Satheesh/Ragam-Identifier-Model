import os
import torch
from torch.utils.data import Dataset
import numpy as np
import json
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

class CarnaticPitchDataset(Dataset):
    def __init__(self, base_dir, snippet_seconds=45.0, overlap=0.5, downsample_factor=2):
        """
        base_dir: path to 'Carnatic/' (with features/ and _info_/)
        snippet_seconds: window size in seconds (default=45)
        overlap: fraction overlap between windows (0=no overlap, 0.5=50% overlap)
        downsample_factor: keep 1 frame every n frames (default=2)
        """
        self.base_dir = base_dir
        self.feat_dir = os.path.join(base_dir, "features")
        info_dir = os.path.join(base_dir, "_info_")

        logger.info("\t--> Loading metadata files...")

        # --- Load metadata files ---
        with open(os.path.join(info_dir, "path_mbid_ragaid.json")) as f:
            mbid_to_meta = json.load(f)
        with open(os.path.join(info_dir, "ragaId_to_ragaName_mapping.json")) as f:
            ragaid_to_name = json.load(f)

        # --- Build metadata list ---
        logger.info("\t--> Building metadata for lookups...")

        self.metadata = []
        for mbid, meta in mbid_to_meta.items():
            rel_path = meta["path"]
            rel_path = rel_path.replace("RagaDataset/Carnatic/", "").replace("audio", "features")
            raga_name = ragaid_to_name[str(meta["ragaid"])]
            self.metadata.append({"file": rel_path, "raga": raga_name})

        # --- Build raga2idx ---
        raga_names = sorted({m["raga"] for m in self.metadata})
        self.raga2idx = {r: i for i, r in enumerate(raga_names)}

        # --- Detect hop size from first file ---
        example_file = os.path.join(self.base_dir, self.metadata[0]["file"] + ".pitchSilIntrpPP")
        f0_data = np.loadtxt(example_file)
        hop = np.mean(np.diff(f0_data[:,0]))  # seconds per frame

        # Apply downsampling to hop size
        hop *= downsample_factor
        self.downsample_factor = downsample_factor

        self.snippet_frames = int(snippet_seconds / hop)
        self.hop_frames = int(self.snippet_frames * (1 - overlap))
        logger.info(f"\t--> hop={hop:.4f}s (downsample={downsample_factor}) → "
              f"snippet_frames={self.snippet_frames}, hop_frames={self.hop_frames}")

        # --- Build list of snippet samples ---
        logger.info(f"\t--> Building snippet samples of {snippet_seconds}s each...")
        self.samples = []
        for m in self.metadata:
            base_path = os.path.join(self.base_dir, m["file"])
            pitch_file = base_path + ".pitchSilIntrpPP"
            f0_data = np.loadtxt(pitch_file)
            num_frames = len(f0_data) // downsample_factor  # adjusted for downsampling

            start = 0
            while start + self.snippet_frames <= num_frames:
                self.samples.append({
                    "file": m["file"],
                    "raga": m["raga"],
                    "start": start
                })
                start += self.hop_frames

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        entry = self.samples[idx]
        base_path = os.path.join(self.base_dir, entry["file"])

        # Load pitch + tonic
        pitch_file = base_path + ".pitchSilIntrpPP"
        tonic_file = base_path + ".tonicFine"

        f0_data = np.loadtxt(pitch_file)
        times, freqs = f0_data[:,0], f0_data[:,1]

        # --- Downsampling ---
        freqs = freqs[::self.downsample_factor]

        # Load tonic
        tonic = float(open(tonic_file).read().strip())

        # Normalize to cents relative to tonic
        cents = np.full_like(freqs, -100.0, dtype=np.float32)
        valid = freqs > 0
        cents[valid] = 1200 * np.log2(freqs[valid] / tonic)

        # Slice snippet
        start = entry["start"]
        seq = cents[start:start+self.snippet_frames]

        # Shape = (1, 1, T) for CRNN
        x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
        y = self.raga2idx[entry["raga"]]
        return x, y