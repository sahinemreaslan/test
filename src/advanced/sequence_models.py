"""
LSTM/GRU and Transformer Models for Sequence Learning

Learns temporal patterns in candle sequences.

References:
- Hochreiter & Schmidhuber (1997) - LSTM
- Fischer & Krauss (2018) - Deep learning with LSTM in stock trading
- Zhou et al. (2021) - Informer
- Wu et al. (2023) - TimesNet
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class LSTMSequenceModel(nn.Module):
    """LSTM model for learning candle sequences"""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = True
    ):
        """
        Initialize LSTM model

        Args:
            input_size: Number of features per timestep
            hidden_size: Hidden state dimension
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            bidirectional: Use bidirectional LSTM
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )

        # Output dimension
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size

        # Attention over sequence
        self.attention = nn.Sequential(
            nn.Linear(lstm_output_size, lstm_output_size // 2),
            nn.Tanh(),
            nn.Linear(lstm_output_size // 2, 1)
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input sequence [batch, seq_len, features]

        Returns:
            Predictions and attention weights
        """
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)  # [batch, seq, hidden*2]

        # Attention weights
        attn_weights = self.attention(lstm_out)  # [batch, seq, 1]
        attn_weights = torch.softmax(attn_weights, dim=1)

        # Weighted sum
        context = torch.sum(lstm_out * attn_weights, dim=1)  # [batch, hidden*2]

        # Classify
        output = self.classifier(context)

        return output.squeeze(-1), attn_weights.squeeze(-1)


class TransformerSequenceModel(nn.Module):
    """
    Transformer model for multi-timeframe sequence fusion

    Uses transformer encoder to process sequences from different timeframes
    """

    def __init__(
        self,
        input_size: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1
    ):
        """
        Initialize Transformer model

        Args:
            input_size: Number of features per timestep
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: FFN dimension
            dropout: Dropout rate
        """
        super().__init__()

        self.d_model = d_model

        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input sequence [batch, seq_len, features]

        Returns:
            Predictions
        """
        # Project to d_model
        x = self.input_projection(x)  # [batch, seq, d_model]

        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer encoding
        x = self.transformer_encoder(x)  # [batch, seq, d_model]

        # Global pooling
        x = x.transpose(1, 2)  # [batch, d_model, seq]
        x = self.global_pool(x).squeeze(-1)  # [batch, d_model]

        # Classify
        output = self.classifier(x)

        return output.squeeze(-1)


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]

        self.register_buffer('pe', pe)

    def forward(self, x):
        """Add positional encoding to input"""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MultiTimeframeTransformer(nn.Module):
    """
    Transformer that processes multiple timeframes simultaneously

    Each timeframe is a separate sequence, cross-attention fuses them
    """

    def __init__(
        self,
        num_timeframes: int,
        features_per_timeframe: int,
        sequence_length: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        """
        Initialize multi-timeframe transformer

        Args:
            num_timeframes: Number of timeframes (e.g., 11)
            features_per_timeframe: Features per timeframe per candle
            sequence_length: Lookback window (number of candles)
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout rate
        """
        super().__init__()

        self.num_timeframes = num_timeframes
        self.sequence_length = sequence_length

        # Separate transformer for each timeframe
        self.timeframe_encoders = nn.ModuleList([
            TransformerSequenceModel(
                input_size=features_per_timeframe,
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                dim_feedforward=d_model * 4,
                dropout=dropout
            )
            for _ in range(num_timeframes)
        ])

        # Cross-attention to fuse timeframes
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )

        self.norm = nn.LayerNorm(d_model)

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model * num_timeframes, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x_list):
        """
        Forward pass

        Args:
            x_list: List of sequences, one per timeframe
                   Each: [batch, seq_len, features]

        Returns:
            Predictions
        """
        # Encode each timeframe
        encoded_timeframes = []

        for i, x in enumerate(x_list):
            # Get representation from transformer
            # Use last hidden state
            encoded = self.timeframe_encoders[i](x)
            encoded_timeframes.append(encoded.unsqueeze(1))

        # Stack: [batch, num_timeframes, d_model]
        encoded_stack = torch.cat(encoded_timeframes, dim=1)

        # Cross-attention (each timeframe attends to all others)
        attn_output, _ = self.cross_attention(
            encoded_stack,
            encoded_stack,
            encoded_stack
        )

        # Residual + norm
        fused = self.norm(encoded_stack + attn_output)

        # Flatten and classify
        fused_flat = fused.view(fused.size(0), -1)
        output = self.classifier(fused_flat)

        return output.squeeze(-1)
