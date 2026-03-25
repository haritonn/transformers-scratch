import copy

import torch
import torch.nn as nn

# from modules.ffn import FeedForwardNetwork
# from modules.mha import MultiHeadAttention
from modules.decoder import DecoderClass
from modules.encoder import EncoderClass
from modules.positional_enc import PositionalEncoding
from modules.transformer import BaseTransformer


def subsequent_mask(size):
    attn_shape = (1, size, size)
    mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
    return mask == 0


def make_std_mask(src, tgt, pad):
    src_mask = (src != pad).unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len]
    tgt_mask = (tgt != pad).unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len]
    tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
    return src_mask, tgt_mask


def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    c = copy.deepcopy

    # attn = MultiHeadAttention(d_model, h)
    # ff = FeedForwardNetwork(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model)

    encoder = nn.ModuleList([EncoderClass(d_model, h, d_ff, dropout) for _ in range(N)])
    decoder = nn.ModuleList([DecoderClass(d_model, h, d_ff, dropout) for _ in range(N)])

    src_emb = nn.Sequential(nn.Embedding(src_vocab, d_model), c(position))
    tgt_emb = nn.Sequential(nn.Embedding(tgt_vocab, d_model), c(position))

    head = nn.Sequential(nn.Linear(d_model, tgt_vocab), nn.LogSoftmax(dim=-1))

    model = BaseTransformer(encoder, decoder, src_emb, tgt_emb, head)

    # xavier weight init
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model
