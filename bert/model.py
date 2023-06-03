

import torch
import torch.nn as nn

from .transformer import Transformer, BlockRecurrentTransformer, ReasoningTransformer, TrainerTransformer
from block_recurrent_transformer_pytorch import BlockRecurrentTransformer \
    as BlockRecurrentTransformerLucidrains

import numpy as np


class BERT(nn.Module):
    """
    BERT
    """

    def __init__(self,
                 vocab_size,
                 n_layers=4,
                 d_model=128,
                 n_head=8,
                 p=0.1
                 ):

        super(BERT, self).__init__()

        self.transformer = Transformer(
            vocab_size=vocab_size,
            n_layers=n_layers,
            d_model=d_model,
            n_head=n_head,
            p=p
        )

        self.linear = nn.Linear(d_model, 1)
        self.softmax = nn.LogSoftmax(dim=-1)

    def init_state(self, batch_size, state_len):
        return self.transformer.init_state(batch_size, state_len)

    def state_forward(self, ids, state):
        return state

    def forward(self, ids, state):
        x = self.transformer.forward(ids)

        # x = self.linear(x.mean(dim=1))
        x = self.softmax(self.linear(x))

        return x, state


class BlockBERT(nn.Module):

    def __init__(self,
                 vocab_size,
                 n_layers=4,
                 d_model=512,
                 n_head=8,
                 p=0.1,
                 state_in=False,
                 bert=False
                 ):
        super(BlockBERT, self).__init__()

        self.bert = bert

        self.transformer = BlockRecurrentTransformer(
            vocab_size=vocab_size,
            n_layers=n_layers,
            d_model=d_model,
            n_head=n_head,
            p=p,
            state_in=state_in
        )

        if bert:
            self.linear = nn.Linear(d_model, vocab_size)
            self.softmax = nn.LogSoftmax(dim=-1)
        else:
            self.linear = nn.Linear(d_model, 1)

    def init_state(self, batch_size, state_len):
        return self.transformer.init_state(batch_size, state_len)

    def state_forward(self, ids, state):
        state = state.unsqueeze(1)
        _, state = self.transformer.forward(ids, state)
        state = state.squeeze()

        return state

    def forward(self, ids, state):
        state = state.unsqueeze(1)
        x, state = self.transformer.forward(ids, state)
        state = state.squeeze()

        if self.bert:
            x = self.softmax(self.linear(x))
        else:
            x = self.linear(x.mean(dim=1))

        return x, state


class BlockBERTlucidrains(nn.Module):
    """
    lucidrains' block recurrent transformer
    """

    def __init__(self,
                 vocab_size,
                 n_layers=4,
                 d_model=128,
                 n_head=8,
                 p=0.1
                 ):
        super(BlockBERTlucidrains, self).__init__()

        self.transformer = BlockRecurrentTransformerLucidrains(
            num_tokens=vocab_size,
            dim=d_model,
            depth=n_layers,
            heads=n_head,
            num_state_vectors=1,
            max_seq_len=512
        )

        self.linear = nn.Linear(d_model, 1)
        self.softmax = nn.LogSoftmax(dim=-1)

    def state_forward(self, ids, state):
        state = state.unsqueeze(1)
        _, state = self.transformer.forward(ids, states=[state])
        state = state[0].squeeze()

        return state

    def forward(self, ids, state):
        state = state.unsqueeze(1)
        x, state = self.transformer(ids, states=[state])
        state = state[0].squeeze()

        # x = self.linear(x.mean(dim=1))
        x = self.softmax(self.linear(x))

        return x, state


class ReasoningBERT(nn.Module):
    """
    Reasoning BERT (Attention layer outputs fed back into model)
    """

    def __init__(self,
                 vocab_size,
                 h_layers=2,
                 v_layers=4,
                 d_model=128,
                 n_head=8,
                 p=0.1
                 ):

        super(ReasoningBERT, self).__init__()

        self.transformer = ReasoningTransformer(
            vocab_size=vocab_size,
            h_layers=h_layers,
            v_layers=v_layers,
            d_model=d_model,
            n_head=n_head,
            p=p
        )

        # self.linear = nn.Linear(d_model, vocab_size)

        self.linear = nn.Linear(d_model, 1)
        self.softmax = nn.LogSoftmax(dim=-1)

    def init_state(self, batch_size, state_len):
        return self.transformer.init_state(batch_size, state_len)

    def state_forward(self, ids, state):
        return self.transformer.h_forward(ids, state)

    def forward(self, ids, state):
        x, state = self.transformer.forward(ids, state)

        # x = self.linear(x.mean(dim=1))
        x = self.softmax(self.linear(x))

        return x, state


class RecurrentTrainer(nn.Module):
    """
    Recurrent Trainer (Approximates aggregate loss of Reasoning BERT to update recurrent state)
    """

    def __init__(self,
                 vocab_size,
                 n_layers=4,
                 d_model=128,
                 n_head=8,
                 n_cos=64,
                 p=0.1,
                 device="cuda"):

        super(RecurrentTrainer, self).__init__()

        self.transformer = TrainerTransformer(
            vocab_size=vocab_size,
            n_layers=n_layers,
            d_model=d_model,
            n_head=n_head,
            p=p
        )

        self.linear = nn.Linear(d_model, d_model)
        self.gelu = nn.GELU()

        # Implicit Quantile Network (IQN)
        self.n_cos = n_cos
        self.d_model = d_model
        self.device = device
        self.cos_embedding = nn.Linear(n_cos, d_model)
        self.pis = torch.FloatTensor([np.pi*i for i in range(1, n_cos+1)]).view(1, 1, n_cos).to(device)

        # Critic out
        self.cos = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, 1)

    def calc_cos(self, batch_size, n_tau=8):
        """
        Calculate the cosin values depending on the number of tau samples
        """
        taus = torch.rand(batch_size, n_tau).unsqueeze(-1).to(self.device)  # (batch_size, n_tau, 1)
        cos = torch.cos(taus*self.pis)

        assert cos.shape == (batch_size, n_tau, self.n_cos)

        cos = cos.view(batch_size*n_tau, self.n_cos)
        cos = self.gelu(self.cos_embedding(cos))
        cos = cos.view(batch_size, n_tau, self.d_model)

        return cos, taus

    def forward(self, x, state1, state2, n_tau=8):
        batch_size = x.shape[0]

        # Pass everything through transformer
        x = self.transformer(x, state1, state2)
        x = x.mean(dim=1)
        x = self.gelu(self.linear(x))
        x = x.view(batch_size, 1, self.d_model)

        # Calculate cos and taus for IQN
        cos, taus = self.calc_cos(batch_size, n_tau)
        cos = cos.view(batch_size, n_tau, self.d_model)
        taus = taus.view(batch_size, n_tau)

        """
        x    [batch_size, 1, d_model]
        cos  [batch_size, n_tau, d_model]
        taus [batch_size, n_tau]
        """

        # Critic out
        x = (x*cos).view(batch_size*n_tau, self.d_model)
        x = self.gelu(self.cos(x))
        x = self.out(x)
        x = x.view(batch_size, n_tau)

        return x, taus

