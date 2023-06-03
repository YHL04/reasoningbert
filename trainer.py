import torch
import torch.nn as nn
from torch.optim import Adam

from alrc import ALRC


class BERTTrainer:

    def __init__(self,
                 bert,
                 trainer,
                 memory,
                 lr=1e-4,
                 batch_size=32,
                 n_accumulate=1,
                 tau=0.01,
                 statelen=1,
                 burnin=20,
                 rollout=40,
                 use_trainer=False,
                 trainer_gamma=0.95
                 ):

        self.bert = nn.DataParallel(bert).to("cuda")
        self.trainer = nn.DataParallel(trainer).to("cuda")
        self.target_trainer = nn.DataParallel(trainer).to("cuda")

        self.hard_update(self.target_trainer, self.trainer)

        self.optimizer = Adam(self.bert.parameters(), lr=lr)
        self.trainer_optimizer = Adam(self.trainer.parameters(), lr=lr)

        if self.bert.module.bert:
            self.criterion = nn.NLLLoss(ignore_index=0)
        else:
            self.criterion = nn.MSELoss()

        self.memory = memory

        self.batch_size = batch_size
        self.n_accumulate = n_accumulate
        self.tau = tau
        self.statelen = statelen
        self.burnin = burnin
        self.rollout = rollout
        self.length = burnin + rollout
        self.use_trainer = use_trainer
        self.trainer_gamma = trainer_gamma

        print("self.use_trainer ", self.use_trainer)

    def train_step(self):
        # self.memory.regenerate(self.bert)
        # itr = iter(self.trainer.parameters())
        # print('before ', next(itr))
        # print('before ', next(itr))
        # print('before ', next(itr))

        bert_grad = [torch.zeros(x.shape, device="cuda") for x in self.bert.parameters()]
        trainer_grad = [torch.zeros(x.shape, device="cuda") for x in self.trainer.parameters()]

        total_bert_loss = 0
        total_trainer_loss = 0

        for i in range(self.n_accumulate):
            X, Y, states, idxs = self.memory.get_batch(batch_size=self.batch_size,
                                                       length=self.burnin+self.rollout)

            bert_loss, trainer_input = self.get_bert_grad(
                X,
                Y,
                states,
                idxs
            )
            trainer_loss = self.get_trainer_grad(
                X1       =trainer_input["X1"],
                X2       =trainer_input["X2"],
                bert_loss=trainer_input["bert_loss"],
                Sp       =trainer_input["Sp"],
                S        =trainer_input["S"],
                Sn       =trainer_input["Sn"]
            )

            total_bert_loss += bert_loss
            total_trainer_loss += trainer_loss

            for x, grad in zip(self.bert.parameters(), bert_grad):
                if x.grad is not None:
                    grad += x.grad
            for x, grad in zip(self.trainer.parameters(), trainer_grad):
                if x.grad is not None:
                    grad += x.grad

        for x, grad in zip(self.bert.parameters(), bert_grad):
            x.grad = (grad / self.n_accumulate)
        for x, grad in zip(self.trainer.parameters(), trainer_grad):
            x.grad = (grad / self.n_accumulate)

        self.optimizer.step()
        self.trainer_optimizer.step()

        self.soft_update(self.target_trainer, self.trainer, self.tau)

        # itr = iter(self.trainer.parameters())
        # print('after ', next(itr))
        # print('after ', next(itr))
        # print('after ', next(itr))
        return total_bert_loss, total_trainer_loss

    def use_trainer_independently(self):
        X, Y, states, idxs = self.memory.get_batch(batch_size=self.batch_size, length=2)
        X = X[0]
        target = Y[0]

        # get loss gradients
        expected, new_states = self.bert.module.forward(X, state=states)
        loss = self.bert_loss(target, expected)

        # get state gradients
        self.trainer.zero_grad()

        state1 = states.detach()
        state2 = new_states.detach()
        state2.requires_grad = True

        agg_loss, _ = self.trainer(X, state1=state1, state2=state2)
        agg_loss = agg_loss.mean()
        agg_loss.backward()

        save_grad = state2.grad

        variables = [loss, new_states]
        grads = [None, save_grad]
        torch.autograd.backward(variables, grads)

        self.optimizer.step()

        return loss

    def get_trainer_acc(self):
        """
        :return: agg_loss: trainer predicted aggregate loss
                 ground_truth: under-approximation of ground truth within length
        """
        with torch.no_grad():
            X, Y, states, idxs = self.memory.get_batch(batch_size=self.batch_size, length=self.burnin+self.rollout)

            _, new_states = self.bert.module.forward(X[0], state=states)

            agg_loss, _ = self.trainer(X[0], state1=states, state2=new_states)
            agg_loss = agg_loss.mean().item()

            ground_truth = 0
            for t in range(1, self.burnin+self.rollout):
                expected, new_states = self.bert.module.forward(X[t], state=new_states)
                ground_truth += (self.bert_loss(Y[t], expected) * (self.trainer_gamma ** (t-1))).item()

        return agg_loss, ground_truth

    def get_bert_grad(self, X, Y, state, idxs):
        self.bert.zero_grad()

        with torch.no_grad():
            state = state.detach()

            for t in range(self.burnin):
                self.memory.update_state(idxs, t, state.detach())

                state = self.bert.module.state_forward(X[t], state=state)

        trainer_inputs = {}

        with torch.no_grad():
            states = {}

            for t in range(self.burnin, self.length):
                states[t] = state.detach()
                self.memory.update_state(idxs, t, state.detach())

                state = self.bert.module.state_forward(X[t], state=state)

            trainer_inputs["Sn"] = states[self.length-1].detach()
            trainer_inputs["S"] = states[self.length-2].detach()
            trainer_inputs["Sp"] = states[self.length-3].detach()

        self.trainer.zero_grad()

        state = state.detach()
        state.requires_grad = True

        loss, _ = self.trainer(
            X[t],
            state1=states[self.length-1],
            state2=state
        )
        loss = loss.mean()
        loss.backward()

        save_grad = state.grad
        save_std = save_grad.squeeze().std()

        # get projection of save_grad onto loss gradient
        temp = states[self.length-1].detach()
        temp.requires_grad = True
        expected, _ = self.bert.forward(X[t], state=temp)
        loss = self.bert_loss(Y[t], expected)
        loss.backward()

        onto_grad = temp.grad
        self.bert.zero_grad()

        # print(save_grad.squeeze().std())
        # print(onto_grad.squeeze().std())

        # calculate projection (if projection is opposite direction then subtract from original)
        proj_direction = torch.sum(save_grad * onto_grad) / torch.sum(onto_grad * onto_grad)
        # save_grad = proj_direction * onto_grad (BUG!!)
        save_grad = save_grad - torch.min(proj_direction, torch.tensor(0.)) * onto_grad

        if not self.use_trainer:
            # if trainer is not used then shuffle grad to check if grad contains info
            # save_grad = save_grad[:, torch.randperm(save_grad.shape[1])]
            # save_grad = onto_grad
            save_grad *= 0

        # print('first ', save_grad.squeeze().std())

        loss = 0
        ckpt_std = []
        intervals = list(range(self.burnin, self.length, 1))

        for ckpt in reversed(intervals):
            assert states[ckpt].grad == None
            states[ckpt].requires_grad = True
            state = states[ckpt]

            expected, state = self.bert.forward(X[ckpt], state=state)
            target = Y[ckpt]
            ckpt_loss = self.bert_loss(target, expected)

            if ckpt == self.length - 2:
                trainer_inputs["bert_loss"] = ckpt_loss.detach()

            variables = [ckpt_loss, state]
            grads = [None, save_grad]
            torch.autograd.backward(variables, grads)

            if ckpt != self.burnin:
                save_grad = states[ckpt].grad
                ckpt_std.append(save_grad.squeeze().std())
                # print(save_grad.squeeze().std())

            loss += ckpt_loss

        loss /= self.rollout

        trainer_inputs["X1"] = X[self.length-3].detach()
        trainer_inputs["X2"] = X[self.length-2].detach()

        return loss, trainer_inputs

    def get_trainer_grad(self, X1, X2, bert_loss,
                                Sp, S, Sn):
        self.trainer.zero_grad()

        with torch.no_grad():
            next_q_values, _ = self.trainer.module.forward(X2, state1=S, state2=Sn)
            target = bert_loss + self.trainer_gamma * next_q_values
            target = target.unsqueeze(1)

        expected, taus = self.trainer(X1, state1=Sp, state2=S)
        expected = expected.unsqueeze(2)
        taus = taus.unsqueeze(2)

        """
        target    [batch_size, 1, n]
        expected  [batch_size, n, 1]
        taus      [batch_size, n, 1]
        """

        loss = self.quantile_loss(target, expected, taus)
        loss.backward()

        return loss

    def bert_loss(self, target, expected):
        """
        :param target:   [batch_size, max_len]
        :param expected: [batch_size, max_len, vocab_size]
        """
        if self.bert.module.bert:
            expected = expected.transpose(1, 2)

        return self.criterion(expected, target)

    def quantile_loss(self, target, expected, taus):
        td_error = target - expected

        huber_loss = torch.where(td_error.abs() <= 1, 0.5 * td_error.pow(2), td_error.abs() - 0.5)
        quantile_loss = abs(taus - (td_error.detach() < 0).float()) * huber_loss

        critic_loss = quantile_loss.sum(dim=1).mean(dim=1)
        critic_loss = critic_loss.mean()
        return critic_loss

    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau
            )

    def hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

