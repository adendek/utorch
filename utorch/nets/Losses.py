import utorch
from utorch.nets.Model import Model
import utorch.simplegrad as sg
import numpy as np


class CrossEntropyWithLogitsLoss(Model):
    """
    This class implements CE loss with logits as an input.
    This implementation is more computationally stable than, x->exp->log->sum,
    see https://github.com/pytorch/pytorch/issues/751
    """

    def __init__(self, n_classes=2):
        self.epsilon = sg.Variable(1e-5)
        self.n_classes = n_classes

    def _to_one_hot(self, target):
        return sg.Variable(np.eye(self.n_classes)[target.value])

    def forward(self, x, target):
        target = self._to_one_hot(target)
        max_val = sg.Variable.clip_min(x * (-1), 0)
        loss = (1 - target) * x + max_val + sg.Variable.log(
            sg.Variable.exp(max_val * (-1)) + sg.Variable.exp((x + max_val) * (-1)))
        return sg.Variable.sum(loss)


class L2Loss(Model):
    def __init__(self, model):
        self.parameters = model.get_parameters()

    def forward(self, output, target ):
        loss = sg.Variable(0)
        for param in self.parameters:
            loss += sg.Variable.sum(param*param)
        return loss
