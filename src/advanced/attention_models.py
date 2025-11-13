"""
Attention Mechanisms for Feature Importance

Multi-head attention to dynamically weight:
- Different timeframes
- Different indicators
- Different fractal patterns

References:
- Vaswani et al. (2017) - Attention is All You Need
- Lim et al. (2021) - Temporal Fusion Transformers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism"""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        """
        Initialize multi-head attention

        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        """
        Forward pass

        Args:
            query: Query tensor [batch, seq_len, d_model]
            key: Key tensor
            value: Value tensor
            mask: Optional mask

        Returns:
            Output tensor and attention weights
        """
        batch_size = query.size(0)

        # Linear projections
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        context = torch.matmul(attn_weights, V)

        # Concat heads and project
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(context)

        return output, attn_weights


class TimeframeAttentionModel(nn.Module):
    """
    Attention model for multi-timeframe features

    Learns which timeframes are most important for prediction
    """

    def __init__(
        self,
        num_timeframes: int,
        features_per_timeframe: int,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        """
        Initialize timeframe attention model

        Args:
            num_timeframes: Number of timeframes (e.g., 11)
            features_per_timeframe: Features per timeframe
            d_model: Model dimension
            num_heads: Number of attention heads
            num_layers: Number of attention layers
            dropout: Dropout rate
        """
        super().__init__()

        self.num_timeframes = num_timeframes
        self.features_per_timeframe = features_per_timeframe
        self.d_model = d_model

        # Project each timeframe features to d_model
        self.input_projection = nn.Linear(features_per_timeframe, d_model)

        # Multi-head attention layers
        self.attention_layers = nn.ModuleList([
            MultiHeadAttention(d_model, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model)
            for _ in range(num_layers)
        ])

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )

        self.ffn_norm = nn.LayerNorm(d_model)

        # Final prediction layer
        self.classifier = nn.Sequential(
            nn.Linear(d_model * num_timeframes, 256),
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
            x: Input tensor [batch, num_timeframes, features_per_timeframe]

        Returns:
            Predictions and attention weights
        """
        batch_size = x.size(0)

        # Project to d_model
        x = self.input_projection(x)  # [batch, num_timeframes, d_model]

        attention_weights_all = []

        # Apply attention layers
        for attn_layer, norm in zip(self.attention_layers, self.layer_norms):
            # Self-attention across timeframes
            attn_output, attn_weights = attn_layer(x, x, x)
            attention_weights_all.append(attn_weights)

            # Residual connection and layer norm
            x = norm(x + attn_output)

        # Feed-forward network
        ffn_output = self.ffn(x)
        x = self.ffn_norm(x + ffn_output)

        # Flatten and classify
        x = x.view(batch_size, -1)
        predictions = self.classifier(x)

        return predictions.squeeze(-1), attention_weights_all

    def get_timeframe_importance(self, x):
        """
        Get importance scores for each timeframe

        Args:
            x: Input tensor

        Returns:
            Importance scores [num_timeframes]
        """
        with torch.no_grad():
            _, attn_weights = self.forward(x)

            # Average across all heads and layers
            importance = []
            for layer_weights in attn_weights:
                # Average across heads: [batch, num_heads, seq, seq] -> [batch, seq, seq]
                layer_avg = layer_weights.mean(dim=1)
                # Average across batch and queries: [seq]
                importance.append(layer_avg.mean(dim=(0, 1)))

            # Average across layers
            final_importance = torch.stack(importance).mean(dim=0)

        return final_importance.cpu().numpy()


class FeatureAttentionModel(nn.Module):
    """
    Attention model to learn important features

    Dynamically weights different technical indicators
    """

    def __init__(
        self,
        num_features: int,
        d_model: int = 128,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        """
        Initialize feature attention model

        Args:
            num_features: Total number of features
            d_model: Model dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()

        self.num_features = num_features
        self.d_model = d_model

        # Feature embedding
        self.feature_embedding = nn.Linear(1, d_model)

        # Self-attention
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # Feed-forward
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model * num_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input features [batch, num_features]

        Returns:
            Predictions and feature importance
        """
        batch_size = x.size(0)

        # Reshape to [batch, num_features, 1]
        x = x.unsqueeze(-1)

        # Embed each feature
        x = self.feature_embedding(x)  # [batch, num_features, d_model]

        # Self-attention across features
        attn_output, attn_weights = self.attention(x, x, x)
        x = self.norm1(x + attn_output)

        # Feed-forward
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)

        # Flatten and classify
        x = x.view(batch_size, -1)
        predictions = self.classifier(x)

        return predictions.squeeze(-1), attn_weights
