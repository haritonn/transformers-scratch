import torch.nn as nn


class BaseTransformer(nn.Module):
    def __init__(self, encoder, decoder, src_emb, tgt_emb, head):
        super(BaseTransformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_emb = src_emb
        self.tgt_emb = tgt_emb
        self.head = head

    def encode(self, src, src_mask):
        x = self.src_emb(src)
        for layer in self.encoder:
            x = layer(x, src_mask)
        return x

    def decode(self, memory, src_mask, tgt, tgt_mask):
        x = self.tgt_emb(tgt)
        for layer in self.decoder:
            x = layer(x, memory, tgt_mask, src_mask)
        return x

    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.head(
            self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)
        )
