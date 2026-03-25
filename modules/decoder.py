import torch.nn as nn

from modules.ffn import FeedForwardNetwork
from modules.mha import MultiHeadAttention


class DecoderClass(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderClass, self).__init__()
        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForwardNetwork(d_model, d_ff)
        self.norm1, self.norm2, self.norm3 = (
            nn.LayerNorm(d_model),
            nn.LayerNorm(d_model),
            nn.LayerNorm(d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, self_mask=None, cross_mask=None):
        mha1_out = self.mha1(x, x, x, self_mask)
        x = self.norm1(x + self.dropout(mha1_out))
        mha2_out = self.mha2(x, encoder_output, encoder_output, cross_mask)
        x = self.norm2(x + self.dropout(mha2_out))

        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_out))

        return x
