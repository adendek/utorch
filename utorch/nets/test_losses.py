from unittest import TestCase
import numpy as np
from numpy.testing import assert_almost_equal

import torch
import utorch.simplegrad as sg
from utorch.nets import Losses as Losses

def build_torch_tensors(value_list):
    """
    Build a list of torch tensors initialized according to the input list
    """
    return [torch.tensor(value, requires_grad=True,dtype=float) for value in value_list]


def build_simplegrad_variables(value_list):
    """
    Build a list of simplegrad variables initialized according to the input list
    """
    return [sg.Variable(value) for value in value_list]


class TestCrossEntropyWithLogitsLoss(TestCase):

    def setUp(self):
        """A workaround to solve numpy installation issue,
        see https://stackoverflow.com/questions/20554074/sklearn-omp-error-15-when-fitting-models
        """
        import os
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    def validate_vs_pytorch(self, sg_variables, torch_tensors):
        """
        loop over collection of simplegrad variables and check if they are almost the same as
        pytorch tensors.
        """
        if not isinstance(sg_variables, list):
            assert_almost_equal(sg_variables.value, torch_tensors.detach().numpy())
        for sg_variable, torch_tensor in zip(sg_variables, torch_tensors):
            assert_almost_equal(sg_variable.value, torch_tensor.detach().numpy())

    def test_cross_entropy_with_logits(self):
        np.random.seed(10)
        n_classes = 3
        target_values = np.random.randint(0, n_classes, size=10)
        input_values = [np.random.normal(1, 1, size=(10, n_classes)), # data
                        target_values, # target logit
                        np.eye(n_classes)[target_values], # target one_hot
        ]

        pred_sg, target_sg, _ = build_simplegrad_variables(input_values)

        loss_sg = Losses.CrossEntropyWithLogitsLoss(n_classes)(pred_sg, target_sg)
        loss_sg.backward()

        pred_pt, _, target_pt = build_torch_tensors(input_values)

        loss_pt = torch.nn.BCEWithLogitsLoss(reduction="sum")(
            pred_pt, target_pt)

        loss_pt.backward()

        self.validate_vs_pytorch(
            [loss_sg, pred_sg.grad],
            [loss_pt, pred_pt.grad]
        )


