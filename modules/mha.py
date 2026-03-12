import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # [batch_size, seq_len (or 1), d_model]
        self.W_q, self.W_k, self.W_v, self.W_o = self._generate_matrices(
            layer=nn.Linear(d_model, d_model)
        )

    def _generate_matrices(self, layer, amount=4):
        """
        Helper function for initializing ModuleList for W_k, W_q, W_v

        layer: layer to generate (should be nn.Linear as input)
        amount: amount of layers
        """
        return nn.ModuleList([layer for _ in range(amount)])

    def _make_matrices(self, linears, values, batch_size):
        """
        Helper function for applying mapping for each component (Q, K, V)
        & dividing them for num_heads

        linears: list of mappings
        values: list of components
        batch_size: obvious
        """
        # after method(value): [batch_size, seq_len, d_model]
        # after view(...): [batch_size, seq_len, num_heads, d_k]
        # after transpose(1, 2): [batch_size, num_heads, seq_len, d_k]
        return [
            method(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
            for method, value in zip(linears, values)
        ]

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        Q, K, V = self._make_matrices(
            [self.W_q, self.W_k, self.W_v], [q, k, v], batch_size
        )  # each: [batch_size, num_heads, seq_len, d_k]

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(
            self.d_k
        )  # [batch_size, num_heads, seq_len, seq_len]

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_score = F.softmax(
            scores, dim=-1
        )  # still [batch_size, num_heads, seq_len, seq_len]
        context = torch.matmul(
            attention_score, V
        )  # [batch_size, num_heads, seq_len, d_k]

        context = (
            context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        )  # [batch_size, seq_len, d_model]
        output = self.W_o(context)

        return output
