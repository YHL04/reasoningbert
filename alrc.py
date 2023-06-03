import math


class ALRC:
    """
    Define ALRC without stop gradient
    (Step returns true if L > Lmax)
    """

    def __init__(self,
                 n=3,
                 decay=0.99,
                 mu1_start=1,
                 mu2_start=1
                 ):
        """
        :param n:         number of standard deviation
        :param decay:     decay rate for mu1 and mu2
        :param mu1_start: initial mu1 estimate
        :param mu2_start: initial mu2 estimate
        """

        self.n = n
        self.decay = decay
        self.mu1 = mu1_start
        self.mu2 = mu2_start

    def step(self, loss_tensor):
        """
        :param loss: a scalar value in a Tensor
        :return:     boolean indicating whether loss has exceeded max loss
        """
        assert loss_tensor.dim() == 0
        loss = loss_tensor.item()
        print(loss)

        sigma = math.sqrt(self.mu2 - self.mu1**2 + 1.e-8)

        boolean = (loss > (self.mu1 + self.n * sigma))

        print('term inside sigma ', (self.mu2 - self.mu1**2 + 1.e-8))
        print('ceil ', (self.mu1 + self.n * sigma))

        # Update mu1 and mu2 (only if loss is not divergent)
        if not boolean:
            self.mu1 = self.decay*self.mu1 + (1-self.decay)*loss
            self.mu2 = self.decay*self.mu2 + (1-self.decay)*(loss**2)

        print('mu1 ', self.mu1)
        print('mu2 ', self.mu2)

        return boolean


if __name__ == "__main__":
    import torch

    test = ALRC()

    test.step(torch.tensor(1.1))
    print(test.step(torch.tensor(1.1)))

