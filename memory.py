import torch
import random

random.seed(0)


class Memory:

    def __init__(self,
                 data,
                 target=None,
                 dim=512,
                 statelen=1):
        """
        :param data:   List[Tensor(length, d_model)]
        :param target: List[Tensor(length, d_model)]
        """
        self.data = data
        self.target = target
        self.size = len(data)

        self.dim = dim
        self.statelen = statelen

        self.state = [torch.zeros(seq.size(0), dim) for seq in data]

    def update_state(self, idxs, t, states):
        """
        :param idxs:    [buffer_idx, time_idx] for each sample in the batch
        :param t:       timestep according to the batch of states
        :param states:  a batch of updated stored states
        """

        for idx, state in zip(idxs, states):
            self.state[idx[0]][idx[1]+t] = state

    def regenerate(self, model):
        """
        regenerate self.data and self.target and generate states according to new model
        (to create unmemorizable data)
        """
        X = torch.zeros(2000, 501, dtype=torch.int32)
        Y = torch.zeros(2000, 1, dtype=torch.float32)

        for i in range(X.size(0) - 60):
            if random.random() < 0.05:
                X[i][:250] = torch.ones(250,)
                for j in range(5, 60, 5):
                    Y[i + j] = torch.tensor([1])

            # if random.random() < 0.05:
            #     X[i][251:] = torch.ones(250,)
            #     for j in range(40, 60):
            #         Y[i + j] = torch.tensor([1])

        with torch.no_grad():

            states = torch.zeros(X.size(0), self.dim)

            state = torch.zeros(1, self.dim)
            for t in range(X.size(0)):
                states[t] = state
                _, state = model.module.forward(X[t].unsqueeze(0).to("cuda"), state=state.to("cuda"))
                state = state.unsqueeze(0).to("cpu")

        self.data = [X]
        self.target = [Y]
        self.state = [states]

    def mask_tokens(self, tokens, p):
        """
        :param tokens: Tensor[total_len, max_len]
        :param p:      mask probability
        """
        target = torch.zeros(*tokens.size(), dtype=torch.int64)

        for i in range(len(tokens)):
            for j in range(len(tokens[i])):
                prob = random.random()

                if prob < p:
                    # index for [MASK] is 103
                    target[i][j] = tokens[i][j]
                    tokens[i][j] = 103

        return tokens, target

    def get_batch(self, batch_size=32, length=30):
        X = []
        Y = []
        states = []
        idxs = []

        for i in range(batch_size):
            bufferidx = random.randrange(0, self.size)
            timeidx = random.randrange(0, self.data[bufferidx].size(0)-length+1)
            idxs.append([bufferidx, timeidx])

            if self.target is None:
                # if self.target is None, then use bert mlm
                tokens = self.data[bufferidx][timeidx:timeidx+length]
                x, y = self.mask_tokens(tokens, p=0.25)
                X.append(x)
                Y.append(y)
            else:
                X.append(self.data[bufferidx][timeidx:timeidx+length])
                Y.append(self.target[bufferidx][timeidx:timeidx+length])

            states.append(self.state[bufferidx][timeidx])

        X = torch.stack(X).to("cuda")
        Y = torch.stack(Y).to("cuda")
        states = torch.stack(states).to("cuda")

        X = X.transpose(0, 1)
        Y = Y.transpose(0, 1)

        assert X.shape == (length, batch_size, X.size(2))
        assert Y.shape == (length, batch_size, Y.size(2))
        assert states.shape == (batch_size, self.dim)

        return X, Y, states, idxs
