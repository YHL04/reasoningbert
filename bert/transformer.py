

import torch
import torch.nn as nn
import numpy as np


from .embedding import TransformerEmbedding, RotaryEmbedding
from .layers import AttentionLayer, CrossAttentionLayer, RecurrentAttentionLayer


class Transformer(nn.Module):
    """
    Standard Transformer
    """

    def __init__(self,
                 vocab_size,
                 max_len=512,
                 n_layers=12,
                 d_model=768,
                 n_head=8,
                 p=0.1
                 ):

        super(Transformer, self).__init__()

        self.emb = TransformerEmbedding(
            vocab_size=vocab_size,
            d_model=d_model,
            max_len=max_len
        )
        self.layers = nn.ModuleList([
            AttentionLayer(
                d_model=d_model,
                ffn_hidden=4 * d_model,
                n_head=n_head,
                p=p
            )
        ])

    def forward(self, ids):
        """
        :param   [batch_size, length]
        :return: [batch_size, d_model]
        """
        x = self.emb(ids)

        for layer in self.layers:
            x = layer(x)

        return x


class ReasoningTransformer(nn.Module):
    """
    Reasoning Transformer composed of vertical and horizontal components
    """

    def __init__(self,
                 vocab_size,
                 max_len=512,
                 h_layers=2,
                 v_layers=2,
                 d_model=512,
                 n_head=8,
                 p=0.1
                 ):

        super(ReasoningTransformer, self).__init__()

        self.d_model = d_model

        self.h_emb = TransformerEmbedding(
            vocab_size=vocab_size,
            d_model=d_model,
            max_len=max_len
        )
        self.v_emb = TransformerEmbedding(
            vocab_size=vocab_size,
            d_model=d_model,
            max_len=max_len
        )

        self.h_cross = CrossAttentionLayer(
            d_model=d_model,
            ffn_hidden=4 * d_model,
            n_head=n_head,
            p=p
        )
        self.v_cross = CrossAttentionLayer(
            d_model=d_model,
            ffn_hidden=4 * d_model,
            n_head=n_head,
            p=p
        )

        self.h_layers = nn.ModuleList([
            AttentionLayer(
                d_model=d_model,
                ffn_hidden=4 * d_model,
                n_head=n_head,
                p=p
            )
            for _ in range(h_layers)
        ])
        self.v_layers = nn.ModuleList([
            AttentionLayer(
                d_model=d_model,
                ffn_hidden=4 * d_model,
                n_head=n_head,
                p=p
            )
            for _ in range(v_layers)
        ])

        self.tanh = nn.Tanh()

    def init_state(self, batch_size, state_len, device="cuda"):
        return torch.zeros(
            batch_size,
            self.d_model,
            device=device
        )

    def h_forward(self, ids, state):
        """
        Horizontal pass of Reasoning Transformer
        """
        state = state.unsqueeze(dim=1)

        x = self.h_emb(ids)

        state = self.h_cross(x, state)
        for layer in self.h_layers:
            state = layer(state)

        state = self.tanh(state.mean(dim=1))
        return state

    def v_forward(self, ids, state):
        """
        Vertical pass of Reasoning Transformer
        """
        state = state.unsqueeze(dim=1)

        x = self.v_emb(ids)

        x = self.v_cross(x, state)
        for layer in self.v_layers:
            x = layer(x)

        return x

    def forward(self, ids, state):
        x = self.v_forward(ids, state)
        state = self.h_forward(ids, state)

        return x, state


class BlockRecurrentTransformer(nn.Module):

    def __init__(self,
                 vocab_size,
                 max_len=512,
                 n_layers=4,
                 d_model=512,
                 n_head=8,
                 p=0.1,
                 state_in=False):

        super(BlockRecurrentTransformer, self).__init__()
        # transform token ids to embedding and add positional encoding
        # self.rotary = RotaryEmbedding(dim=d_model // n_head)

        self.state_in = state_in

        self.embedding = TransformerEmbedding(vocab_size=vocab_size,
                                              d_model=d_model,
                                              max_len=max_len)

        if self.state_in:
            self.first_layer = RecurrentAttentionLayer(d_model=d_model,
                                                       ffn_hidden=4 * d_model,
                                                       n_head=n_head,
                                                       p=p)
            self.layers1 = nn.ModuleList([AttentionLayer(d_model=d_model,
                                                         ffn_hidden=4 * d_model,
                                                         n_head=n_head,
                                                         p=p)
                                          for _ in range(n_layers//2 - 1)])

        else:
            self.layers1 = nn.ModuleList([AttentionLayer(d_model=d_model,
                                                         ffn_hidden=4 * d_model,
                                                         n_head=n_head,
                                                         p=p)
                                          for _ in range(n_layers//2)])

        self.recurrent = RecurrentAttentionLayer(d_model=d_model,
                                                 ffn_hidden=4 * d_model,
                                                 n_head=n_head,
                                                 p=p)
        self.layers2 = nn.ModuleList([AttentionLayer(d_model=d_model,
                                                     ffn_hidden=4 * d_model,
                                                     n_head=n_head,
                                                     p=p)
                                      for _ in range(n_layers//2)])

    def init_state(self, batch_size, state_len, device="cuda"):
        return torch.zeros(
            batch_size,
            self.d_model,
            device=device
        )

    def forward(self, ids, state):
        """
        :param ids: torch.Tensor [batch_size, length]
        :return: torch.Tensor [batch_size, 1]
        """
        # rotary_emb, xpos_scale = self.rotary(ids.size(1))

        x = self.embedding(ids)

        if self.state_in:
            x, _ = self.first_layer(x, state.detach())

        # transformer layers (recurrent layer sandwiched between standard layers)
        for layer in self.layers1:
            x = layer(x)
        x, state = self.recurrent(x, state)
        for layer in self.layers2:
            x = layer(x)

        return x, state


class TrainerTransformer(nn.Module):
    """
    Standard Transformer with two cross attention layers
    """

    def __init__(self,
                 vocab_size,
                 max_len=512,
                 n_layers=3,
                 d_model=512,
                 n_head=8,
                 p=0.1
                 ):

        super(TrainerTransformer, self).__init__()

        self.emb = TransformerEmbedding(
            vocab_size=vocab_size,
            d_model=d_model,
            max_len=max_len
        )
        self.layer = AttentionLayer(
            d_model=d_model,
            ffn_hidden=4 * d_model,
            n_head=n_head,
            p=p
        )
        self.state1 = CrossAttentionLayer(
            d_model=d_model,
            ffn_hidden=4 * d_model,
            n_head=n_head,
            p=p
        )
        self.state2 = CrossAttentionLayer(
            d_model=d_model,
            ffn_hidden=4 * d_model,
            n_head=n_head,
            p=p
        )
        self.layers = nn.ModuleList([
            AttentionLayer(
                d_model=d_model,
                ffn_hidden=4 * d_model,
                n_head=n_head,
                p=p
            )
            for _ in range(n_layers)
        ])

    def forward(self, ids, state1, state2):
        """
        :param   [batch_size, length]
        :return: [batch_size, d_model]
        """

        x = self.layer(self.emb(ids))

        x = self.state1(state1.unsqueeze(dim=1), x)
        x = self.state2(state2.unsqueeze(dim=1), x)

        for layer in self.layers:
            x = layer(x)

        return x
