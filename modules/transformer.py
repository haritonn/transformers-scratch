import torch.nn as nn


class BaseTransformer(nn.Module):
    def __init__(self, encoder, decoder, src_emb, tgt_emb, head):
        super(BaseTransformer, self).__init__()
        self.encoder = encoder  # N * EncoderLayer
        self.decoder = decoder  # N * DecoderLayer
        self.src_emb, self.tgt_emb = (
            src_emb,
            tgt_emb,
        )  # nn sequential(embedding, positionalenc)
        self.head = head  # nn sequential(linear, softmax)

    def encode(self, src, src_mask):
        return self.encoder(self.src_emb(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_emb(tgt), memory, src_mask, tgt_mask)

    def forward(self, src, tgt, src_mask, tgt_mask):
        memory = self.encode(src, src_mask)
        decoded = self.decode(memory, src_mask, tgt, tgt_mask)

        return self.head(decoded)
