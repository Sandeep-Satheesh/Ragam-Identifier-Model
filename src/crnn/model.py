import torch
import torch.nn as nn
import torch.nn.functional as F

class RagamCRNN(nn.Module):
    def __init__(self, num_classes, rnn_hidden=256, rnn_layers=2,
                 dropout=0.3, pooling="attention"):
        super(RagamCRNN, self).__init__()

        # --- 1D CNN feature extractor ---
        self.conv_block = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        # --- BiLSTM ---
        self.rnn = nn.LSTM(
            input_size=128,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=True
        )

        # --- Pooling type ---
        self.pooling = pooling
        if pooling == "attention":
            self.attn = nn.Linear(rnn_hidden * 2, 1)  # learns weights per timestep

        # --- Classifier ---
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(rnn_hidden * 2, num_classes)

    def forward(self, x, return_attention=False):
        # --- CNN feature extraction ---
        out = self.conv_block(x)       # (B, C, T)

        # --- BiLSTM expects (B, T, C) ---
        out = out.permute(0, 2, 1)     # (B, T, C)
        out, _ = self.rnn(out)         # (B, T, H*2)

        # --- Pooling ---
        if self.pooling == "attention":
            attn_scores = self.attn(out)                 # (B, T, 1)
            attn_weights = torch.softmax(attn_scores, dim=1)  # (B, T, 1)
            out = torch.sum(out * attn_weights, dim=1)        # (B, H*2)
            if return_attention:
                return self.fc(self.dropout(out)), attn_weights.squeeze(-1)  # (B, num_classes), (B, T)
        elif self.pooling == "avg":
            out = out.mean(dim=1)
        elif self.pooling == "max":
            out, _ = out.max(dim=1)

        # --- Classifier ---
        out = self.dropout(out)
        return self.fc(out)
