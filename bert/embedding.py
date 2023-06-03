

import torch
import torch.nn as nn


class LearnedPositionalEncoding(nn.Module):
    """
    Learned Positional Embedding
    """

    def __init__(self, d_model, max_len):
        super(LearnedPositionalEncoding, self).__init__()
        self.encoding = nn.Parameter(torch.randn(max_len, d_model),
                                     requires_grad=True)

    def forward(self, x):
        batch_size, seq_len, d_model = x.size()
        return self.encoding[:seq_len, :]


class TokenEmbedding(nn.Module):
    """
    Token Embedding
    """

    def __init__(self, vocab_size, d_model):
        super(TokenEmbedding, self).__init__()
        self.emb = nn.Embedding(vocab_size, d_model)

    def forward(self, ids):
        """
        :param  [batch_size, length]
        :return [batch_size, length, dim]
        """
        token_emb = self.emb(ids)
        return token_emb


class TransformerEmbedding(nn.Module):
    """
    Transformer Embedding
    """

    def __init__(self, vocab_size, d_model, max_len):
        super(TransformerEmbedding, self).__init__()
        self.tok_emb = TokenEmbedding(vocab_size, d_model)
        self.pos_emb = LearnedPositionalEncoding(d_model, max_len)

    def forward(self, x):
        """
        :param  [batch_size, length]
        :return [batch_size, length, dim]
        """
        token_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(token_emb)
        return token_emb + pos_emb

class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        scale_base = 512,
        theta = 10000
    ):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent = False)

        self.scale_base = scale_base
        scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)
        self.register_buffer('scale', scale, persistent = False)

        self.register_buffer('cached_freqs', None, persistent = False)
        self.register_buffer('cached_scales', None, persistent = False)

    @property
    def device(self):
        return next(self.buffers()).device

    def forward(self, seq_len):
        device = self.device

        if self.cached_freqs is not None:
            cached_seq_len = self.cached_freqs.shape[-2]
            if cached_seq_len >= seq_len:
                return self.cached_freqs[:seq_len], self.cached_scales[:seq_len]

        t = torch.arange(seq_len, device = device).type_as(self.inv_freq)
        freqs = torch.einsum('i , j -> i j', t, self.inv_freq)
        freqs = torch.cat((freqs, freqs), dim = -1)

        power = (t - (seq_len // 2)) / self.scale_base
        scale = self.scale ** power.unsqueeze(1)
        scale = torch.cat((scale, scale), dim = -1)

        self.register_buffer('cached_freqs', freqs, persistent = False)
        self.register_buffer('cached_scales', scale, persistent = False)
        return freqs, scale