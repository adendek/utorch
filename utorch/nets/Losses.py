import nets
from simplegrad import Variable
import numpy as np


class CrossEntropyWithLogitsLoss(nets.Model):
    """
    This class implements CE loss with logits as an input.
    This implementation is more computationally stable than, x->exp->log->sum,
    see https://github.com/pytorch/pytorch/issues/751
    """

    def __init__(self, n_classes=2):
        self.epsilon = Variable(1e-5)
        self.n_classes = n_classes

    def _to_one_hot(self, target):
        return Variable(np.eye(self.n_classes)[target.value])

    def forward(self, x, target):
        target = self._to_one_hot(target)
        max_val = Variable.clip_min(x * (-1), 0)
        loss = (1 - target) * x + max_val + Variable.log(
            Variable.exp(max_val * (-1)) + Variable.exp((x + max_val) * (-1)))
        return Variable.sum(loss)


class L2Loss(nets.Model):
    def __init__(self, model):
        self.parameters = model.get_parameters()

    def forward(self, output, target ):
        loss = Variable(0)
        for param in self.parameters:
            loss += Variable.sum(param*param)
        return loss
