import numpy as np
from utorch.simplegrad import Variable


def sum_backward(grad, x, axis):
    """
    This function implement backward of a sum.
    It needs to take care of two cases. The first one is when we sum over all dims
    eg.  sum([[1,2],[3,4]]) = 10.
    In this case we need to return a gradient from the previous step ( a single number) multiplied by the tensor which shape match the shape of the input tensor.
    The second scenario is when the user specify axe(s):
    sum([[1,2],[3,4]], axis=0) = [4,6]
    In this case we need to broadcast the result to the shape of the input tensor.
    TODO: add figures, that visualize this concepts.
    """
    if axis is None:
        return grad*np.ones_like(x.value)
    return Variable.Variable(np.broadcast_to(grad.value, x.shape()[::-1]).T)


primitives = {"__add__": [lambda grad, left, right, args: grad] * 2,
              "__radd__": [lambda grad, left, right, args: grad] * 2,

              "__mul__": [lambda grad, left, right, args: grad * right.value,
                          lambda grad, left, right, args: grad * left.value],

              "__sub__": [lambda grad, left, right, args: grad,
                          lambda grad, left, right, args: (-1) * grad],

              "__rsub__": [lambda grad, left, right, args: (-1) * grad,
                           lambda grad, left, right, args: grad],

              "__truediv__": [lambda grad, left, right, args: grad / right,
                              lambda grad, left, right, args: grad * (-1) * left / right ** 2],
              # based on https://math.stackexchange.com/a/3850121
              "__matmul__": [lambda grad, left, right, args: grad @ Variable.Variable.transpose(right),
                             lambda grad, left, right, args: Variable.Variable.transpose(left) @ grad
                             ],
              "sin": lambda grad, x, args: grad * Variable.Variable.cos(x),
              "cos": lambda grad, x, args: (-1) * grad * Variable.Variable.sin(x),
              "log": lambda grad, x, args: grad / x,
              "relu": lambda grad, x, args: grad * np.where(x > Variable.Variable(0), 1, 0),
              "sigmoid": lambda grad, x, args: grad *Variable.Variable.sigmoid(x) * (1 - Variable.Variable.sigmoid(x)),
              "sum": lambda grad, x, args: sum_backward(grad, x, args),
              "exp": lambda grad, x, args: grad * Variable.Variable.exp(x),
              "transpose": lambda grad, x, args: Variable.Variable.transpose(grad),
              "clip_min": lambda grad, x, args: grad * (x >= args),
              "abs": lambda grad, x, args: grad * Variable.Variable.sign(x),
              "sign": lambda grad, x, args: np.zeros_like(x)
              }