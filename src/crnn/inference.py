import src.util as util
import torch
import numpy as np
from src.crnn.dataset import CarnaticPitchDataset
from src.crnn.model import RagamCRNN
from src import constants

def load_model(checkpoint_path, num_classes, device="cpu", pooling="attention"):
    model = RagamCRNN(num_classes=num_classes, pooling=pooling).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_state" in state_dict:
        model.load_state_dict(state_dict["model_state"])  # from full checkpoint
    else:
        model.load_state_dict(state_dict)  # direct weights
    model.eval()
    return model

import torch
import numpy as np

def preprocess_sample(f0_pitch: torch.Tensor,
                         tonic: float,
                         snippet_frames: int,
                         downsample_factor: int = 2,
                         pad_value: float = -100.0):
    """
    Convert torchcrepe f0 tensor into CRNN-ready input.

    Args:
        f0_pitch: torch.Tensor of shape (1, T) or (T,) in Hz
        tonic: float, estimated tonic frequency
        snippet_frames: int, expected length for model (e.g. 5062)
        downsample_factor: int, downsampling factor used in training
        pad_value: float, filler value for padding (default=-100.0)
    """

    # Ensure shape (T,)
    if f0_pitch.ndim == 2:
        freqs = f0_pitch.squeeze(0)  # (T,)
    else:
        freqs = f0_pitch

    # To numpy for consistency
    freqs = freqs.cpu().numpy()

    # --- Downsample (same as training) ---
    freqs = freqs[::downsample_factor]

    # --- Normalize to cents relative to tonic ---
    cents = np.full_like(freqs, pad_value, dtype=np.float32)
    valid = freqs > 0
    cents[valid] = 1200 * np.log2(freqs[valid] / tonic)

    # --- Trim or pad ---
    if len(cents) >= snippet_frames:
        seq = cents[:snippet_frames]
    else:
        seq = np.pad(cents, (0, snippet_frames - len(cents)), constant_values=pad_value)

    # --- Convert to tensor with channel + batch dims ---
    x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1,1,T)
    return x.to('cuda')

def predict(model, x, idx2raga, device="cpu", return_attention=False):
    #x = x.unsqueeze(0).to(device)  # (1, 1, T)
    with torch.no_grad():
        if return_attention:
            logits, attn = model(x, return_attention=True)
        else:
            logits = model(x)
            attn = None

        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))
        pred_raga = idx2raga[pred_idx]

    return pred_raga, probs, attn
