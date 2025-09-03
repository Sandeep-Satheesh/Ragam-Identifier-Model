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

    def forward(self, x):
        """
        x: (B, 1, T) → pitch contour snippet
        """
        # CNN: (B,1,T) → (B,128,T')
        out = self.conv_block(x)

        # Prepare for RNN: (B,128,T') → (B,T',128)
        out = out.permute(0, 2, 1)

        # BiLSTM: (B,T',2*hidden)
        out, _ = self.rnn(out)

        # --- Pooling ---
        if self.pooling == "mean":
            out = out.mean(dim=1)  # average pooling
        elif self.pooling == "max":
            out = out.max(dim=1).values  # max pooling
        elif self.pooling == "attention":
            attn_weights = torch.softmax(self.attn(out).squeeze(-1), dim=1)  # (B,T')
            out = torch.sum(out * attn_weights.unsqueeze(-1), dim=1)  # weighted sum
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling}")

        # --- Classifier ---
        out = self.dropout(out)
        out = self.fc(out)
        return out
