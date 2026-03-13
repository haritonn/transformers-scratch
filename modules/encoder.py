import torch.nn as nn
from ffn import FeedForwardNetwork
from mha import MultiHeadAttention


class EncoderClass(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderClass, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForwardNetwork(d_model, d_ff)
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Immediately with skip connections
        """
        attn_scores = self.mha(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_scores))

        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))

        return x
