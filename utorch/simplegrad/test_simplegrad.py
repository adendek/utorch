from unittest import TestCase
import numpy as np
from numpy.testing import assert_almost_equal

import simplegrad
import torch


def build_torch_tensors(value_list):
    """
    Build a list of torch tensors initialized according to the input list
    """
    return [torch.tensor(value, requires_grad=True,dtype=float) for value in value_list]


def build_simplegrad_variables(value_list):
    """
    Build a list of simplegrad variables initialized according to the input list
    """
    return [simplegrad.Variable(value) for value in value_list]

class TestSimplegrad(TestCase):
    """
    The idea to check whether simplegrad returns a proper gradient value is to check it versus the one returned by PyTorch
    """

    def setUp(self):
        """A workaround to solve numpy installation issue,
        see https://stackoverflow.com/questions/20554074/sklearn-omp-error-15-when-fitting-models
        """
        import os
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    def vallidate_vs_pytorch(self, sg_variables, torch_tensors ):
        """
        loop over collection of simplegrad variables and check if they are almost the same as
        pytorch tensors.
        """
        if not isinstance(sg_variables, list):
            assert_almost_equal(sg_variables.value, torch_tensors.numpy())
        for sg_variable, torch_tensor in zip(sg_variables, torch_tensors):
            assert_almost_equal(sg_variable.value, torch_tensor.numpy())


    def test_sum_and_multilication(self):
        input_values = [2.,3.,4.]
        a, b, d = build_simplegrad_variables(input_values)
        c = a * a * b + d + d * a * b + a * a * 2
        c.backward()

        A, B, D = build_torch_tensors(input_values)
        C = A * A * B + D + D * A * B + A * A * 2

        C.backward()
        self.vallidate_vs_pytorch(
            [a.grad, b.grad, d.grad],
            [A.grad, B.grad, D.grad]
        )

    def test_division_and_multiplication(self):
        input_values = [2.,3.,4.]
        a, b, d = build_simplegrad_variables(input_values)
        c = a/(b*d)
        c.backward()

        A, B, D = build_torch_tensors(input_values)
        C = A/(B*D)

        C.backward()
        self.vallidate_vs_pytorch(
            [a.grad, b.grad, d.grad],
            [A.grad, B.grad, D.grad]
        )


    def test_sin_division(self):
        input_values = [2.,3.,4.]
        a, b, d = build_simplegrad_variables(input_values)
        c = a/(b*simplegrad.Variable.sin(d))
        c.backward()

        A, B, D = build_torch_tensors(input_values)
        C = A/(B*torch.sin(D))

        C.backward()
        self.vallidate_vs_pytorch(
            [a.grad, b.grad, d.grad],
            [A.grad, B.grad, D.grad]
        )

    def test_sigmoid_relu(self):
        input_values = [2., 3., 4.]
        a, b, d = build_simplegrad_variables(input_values)
        c = simplegrad.Variable.relu(a / (b * simplegrad.Variable.sigmoid(d)))
        c.backward()

        A, B, D = build_torch_tensors(input_values)
        C = torch.relu(A / (B * torch.sigmoid(D)))

        C.backward()
        self.vallidate_vs_pytorch(
            [a.grad, b.grad, d.grad],
            [A.grad, B.grad, D.grad]
        )

    def test_vector_multiplication_transpose(self):
        np.random.seed(10)
        input_values = [np.random.normal(1,10,size=(2, 3)),
                        np.random.normal(1,10,size=(10, 3)),
                        np.random.normal(1, 10, size=10)
                        ]
        d, w, b = build_simplegrad_variables(input_values)
        c = simplegrad.Variable.sum(d@simplegrad.Variable.transpose(w)*b)
        c.backward()

        D, W, B = build_torch_tensors(input_values)
        C = torch.sum(D@W.T*B)


        C.backward()
        self.vallidate_vs_pytorch(
            [d.grad, w.grad, b.grad],
            [D.grad, W.grad, B.grad]
        )

    def test_sum_over_a_single_axis(self):
        np.random.seed(10)
        input_values = [np.random.randint(1,10, size=(5,3)),
                        np.random.randint(1,10, size=(5,3))]
        a,b = build_simplegrad_variables(input_values)
        c = simplegrad.Variable.sum(
                        simplegrad.Variable.sin(
                            simplegrad.Variable.sum(
                                simplegrad.Variable.sin(a+b),1
                            )
                        )
        )
        c.backward()

        A, B = build_torch_tensors(input_values)
        C = torch.sum(
                torch.sin(
                    torch.sum(
                        torch.sin(A+B),1)
                )
            )
        C.backward()

        self.vallidate_vs_pytorch(
            [a.grad, b.grad],
            [A.grad, B.grad]
        )

    def test_binary_cross_entropy_with_logits(self):
        def bce_with_logits(x, target):
            """
              Binary Cross Entropy Loss
              Should be numerically stable, built based on: https://github.com/pytorch/pytorch/issues/751
              :param x: Input tensor
              :return: Scalar value (sum)
            """

            max_val = simplegrad.Variable.clip_min(x * (-1), 0)
            loss = (1 - target) * x + max_val + simplegrad.Variable.log(
                simplegrad.Variable.exp(max_val * (-1)) + simplegrad.Variable.exp((x + max_val) * (-1)))
            return simplegrad.Variable.sum(loss)

        np.random.seed(10)
        target_values = np.random.randint(0, 3, size=10)
        input_values = [np.random.normal(1, 1, size=(10, 10)), # data
                        np.eye(3)[target_values], # target
                        np.random.normal(1, 1, size=(5, 10)), #w1
                        np.random.normal(1, 1, size=(5)),# b1
                        np.random.normal(1, 1, size=(3, 5)),# w2
                        np.random.normal(1, 1, size=(3)) # b2
                        ]
        d, t, w1, b1, w2, b2 = build_simplegrad_variables(input_values)

        c = bce_with_logits(
            simplegrad.Variable.relu(d @ simplegrad.Variable.transpose(w1) + b1) @ simplegrad.Variable.transpose(w2) + b2,
            t)

        c.backward()

        D, T, W1, B1, W2, B2 = build_torch_tensors(input_values)

        C = torch.nn.BCEWithLogitsLoss(reduction="sum")(
            torch.relu(D @ W1.T + B1) @ W2.T + B2,
            T)
        C.backward()
        self.vallidate_vs_pytorch(
            [w1.grad, b1.grad, w2.grad, b2.grad],
            [W1.grad, B1.grad, W2.grad, B2.grad]
        )
