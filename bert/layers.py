

import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from .embedding import LearnedPositionalEncoding


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot Product Attention for Transformers

    Query : [batch_size, head, length, d_tensor]
    Key : T [batch_size, head, d_tensor, length]
    Value : [batch_size, head, length, d_tensor]

    score : [batch_size, head, length, length]
    v_out : [batch_size, head, length, d_tensor]
    """

    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None, e=1e-12):
        batch_size, head, length, d_tensor = q.shape
        k_t = k.transpose(2, 3)

        score = torch.matmul(q, k_t) / math.sqrt(d_tensor)

        if mask is not None:
            score = score.masked_fill(mask == 0, -e)

        score = self.softmax(score)
        v = torch.matmul(score, v)

        return v, score


class Attention(nn.Module):
    """
    Attention module for Transformer layers
    """

    def __init__(self, d_model, n_head):
        super(Attention, self).__init__()
        self.n_head = n_head
        self.attention = ScaledDotProductAttention()

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_kv = nn.Linear(d_model, 2 * (d_model // n_head), bias=False)
        self.w_concat = nn.Linear(d_model, d_model, bias=False)

    def forward(self, q, kv, mask=None):
        """
        * k and v are the same thing

        :param q, k, v: [batch_size, length, d_model]
        :return: out:   [batch_size, length, d_model]
        """
        q, k, v = self.w_q(q), *self.w_kv(kv).chunk(2, dim=-1)
        q, k, v = self.split(q), k.unsqueeze(1), v.unsqueeze(1)

        out, attention = self.attention(q, k, v, mask=mask)

        out = self.concat(out)
        out = self.w_concat(out)

        return out

    def split(self, tensor):
        """
        Split tensor into number of head

        :param tensor: [batch_size, length, d_model]
        :return: [batch_size, head, length, d_tensor]
        """
        batch_size, length, d_model = tensor.shape

        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)

        return tensor

    def concat(self, tensor):
        """
        Inverse function of self.split(tensor : torch.Tensor)

        :param tensor: [batch_size, head, length, d_tensor]
        :return: [batch_size, length, d_model]
        """
        batch_size, head, length, d_tensor = tensor.shape
        d_model = head * d_tensor

        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor


class RecurrentAttention(nn.Module):
    """
    Recurrent Attention module for Block Recurrent Layer
    """

    def __init__(self, d_model, n_head):
        super(RecurrentAttention, self).__init__()
        self.n_head = n_head
        self.attention = ScaledDotProductAttention()

        # get q, k, v for x and state
        self.w_qx1 = nn.Linear(d_model, d_model, bias=False)
        self.w_qs1 = nn.Linear(d_model, d_model, bias=False)
        self.w_qx2 = nn.Linear(d_model, d_model, bias=False)
        self.w_qs2 = nn.Linear(d_model, d_model, bias=False)

        self.w_kvx = nn.Linear(d_model, 2 * (d_model // n_head), bias=False)
        self.w_kvs = nn.Linear(d_model, 2 * (d_model // n_head), bias=False)

        # linear projection
        self.x_proj = nn.Linear(2 * d_model, d_model)
        self.s_proj = nn.Linear(2 * d_model, d_model)

    def forward(self, qx, kvx, qs, kvs, mask=None):
        # compute 4 distinct queries
        qx1, qs1, qx2, qs2 = self.w_qx1(qx), self.w_qs1(qs), self.w_qx2(qx), self.w_qs2(qs)
        qx1, qs1, qx2, qs2 = self.split(qx1), self.split(qs1), self.split(qx2), self.split(qs2)

        # compute shared keys and values
        kx, vx = self.w_kvx(kvx).chunk(2, dim=-1)
        kx, vx = kx.unsqueeze(1), vx.unsqueeze(1)

        ks, vs = self.w_kvs(kvs).chunk(2, dim=-1)
        ks, vs = ks.unsqueeze(1), vs.unsqueeze(1)

        # perform self attention and cross attention
        x, _ = self.attention(qx1, kx, vx, mask=mask)
        s, _ = self.attention(qs1, ks, vs, mask=mask)

        xs, _ = self.attention(qx2, ks, vs, mask=mask)
        sx, _ = self.attention(qs2, kx, vx, mask=mask)

        # concatenate and linear projection
        x_proj = self.concat(torch.concat((xs, x), dim=-1))
        s_proj = self.concat(torch.concat((sx, s), dim=-1))

        x_proj = self.x_proj(x_proj)
        s_proj = self.s_proj(s_proj)

        return x_proj, s_proj

    def split(self, tensor):
        """
        Split tensor into number of head

        :param tensor: [batch_size, length, d_model]
        :return: [batch_size, head, length, d_tensor]
        """
        batch_size, length, d_model = tensor.shape

        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)

        return tensor

    def concat(self, tensor):
        """
        Inverse function of self.split(tensor : torch.Tensor)

        :param tensor: [batch_size, head, length, d_tensor]
        :return: [batch_size, length, d_model]
        """
        batch_size, head, length, d_tensor = tensor.shape
        d_model = head * d_tensor

        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor


class AttentionLayer(nn.Module):
    """
    Standard transformer layer
    """

    def __init__(self, d_model, ffn_hidden, n_head, p):
        super(AttentionLayer, self).__init__()
        self.attention = Attention(d_model=d_model, n_head=n_head)
        self.norm1 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=p)

        self.ffn = FeedForward(dim=d_model, inner_dim=ffn_hidden)
        self.norm2 = LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p=p)

    def forward(self, x, src_mask=None):
        _x = x
        x = self.attention(q=x, kv=x, mask=src_mask)

        x = self.norm1(x + _x)
        x = self.dropout1(x)

        _x = x
        x = self.ffn(x)

        x = self.norm2(x + _x)
        x = self.dropout2(x)

        return x


class CrossAttentionLayer(nn.Module):
    """
    Cross attention layer to take in state
    """

    def __init__(self, d_model, ffn_hidden, n_head, p):
        super(CrossAttentionLayer, self).__init__()
        self.self_attention = Attention(d_model=d_model, n_head=n_head)
        self.norm1 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=p)

        self.cross_attention = Attention(d_model=d_model, n_head=n_head)
        self.norm2 = LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p=p)

        self.ffn = FeedForward(dim=d_model, inner_dim=ffn_hidden)
        self.norm3 = LayerNorm(d_model)
        self.dropout3 = nn.Dropout(p=p)

    def forward(self, x, s, x_mask=None, s_mask=None):
        _x = x
        x = self.self_attention(q=x, kv=x, mask=x_mask)

        x = self.norm1(x + _x)
        x = self.dropout1(x)

        _x = x
        x = self.cross_attention(q=x, kv=s, mask=s_mask)

        x = self.norm2(x + _x)
        x = self.dropout2(x)

        _x = x
        x = self.ffn(x)

        x = self.norm3(x + _x)
        x = self.dropout3(x)

        return x


class RecurrentAttentionLayer(nn.Module):
    """
    A recurrent transformer layer from block-recurrent transformer
    """
    def __init__(self, d_model, ffn_hidden, n_head, p, max_len=512, qk_rmsnorm_scale=8):
        super(RecurrentAttentionLayer, self).__init__()

        # learned ids
        self.state_norm = LayerNorm(d_model)
        self.state_ids = LearnedPositionalEncoding(d_model=d_model, max_len=max_len)

        # attention
        self.attention = RecurrentAttention(d_model=d_model, n_head=n_head)

        # forget gates
        self.proj_gate = FixedGate(d_model)
        self.ff_gate = FixedGate(d_model)

        # linear projection
        self.x_proj = nn.Linear(2 * d_model, d_model)
        self.s_proj = nn.Linear(2 * d_model, d_model)

        # feed forward model
        self.ffn = FeedForward(d_model, inner_dim=ffn_hidden)

    def forward(self, x, s, x_mask=None, s_mask=None):
        _x = x
        _s = s

        s = self.state_norm(s)
        s = s + self.state_ids(s)

        x_proj, s_proj = self.attention(qx=x, kvx=x, qs=s, kvs=s)

        # finish computing out
        x_residual = x_proj + _x
        out = self.ffn(x_residual) + x_residual

        # fixed simple gate
        next_s = self.proj_gate(s_proj, _s)

        return out, next_s


class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim = -1)
        return F.gelu(gate) * x


class FeedForward(nn.Module):
    """
    Sequential(
        Linear(dim, inner_dim)
        GELU()
        Linear(inner_dim, dim)
    )
    """

    def __init__(self, dim, inner_dim):
        super(FeedForward, self).__init__()

        self.ff = nn.Sequential(
            LayerNorm(dim),
            nn.Linear(dim, inner_dim * 2, bias=False),
            GEGLU(),
            nn.Linear(inner_dim, dim, bias=False)
        )

        # self.ff = nn.Sequential(
        #     nn.Linear(dim, inner_dim),
        #     nn.GELU(),
        #     nn.Linear(inner_dim, dim)
        # )

    def forward(self, x):
        return self.ff(x)


class LSTMGate(nn.Module):
    """
    LSTM Gate for block-recurrent transformer
    """
    def __init__(self, dim: int):
        super().__init__()
        self.main_proj = nn.Linear(dim, dim, bias=True)
        self.input_proj = nn.Linear(dim, dim, bias=True)
        self.forget_proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x, state):
        z = torch.tanh(self.main_proj(x))
        i = torch.sigmoid(self.input_proj(x) - 1)
        f = torch.sigmoid(self.forget_proj(x) + 1)
        return torch.mul(state, f) + torch.mul(z, i)


class FixedGate(nn.Module):
    """
    Fixed Gate for block-recurrent transformer
    """
    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=True)
        self.bias = nn.Parameter(torch.randn(dim), requires_grad=True)

    def forward(self, x, state):
        z = self.proj(x)
        g = torch.sigmoid(self.bias)
        return torch.mul(state, g) + torch.mul(z, 1-g)

# bias-less layernorm
class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)