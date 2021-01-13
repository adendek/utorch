import numpy as np
from simplegrad import Variable

primitives = {"__add__": [lambda grad, left, right: grad] * 2,
              "__radd__": [lambda grad, left, right: grad] * 2,

              "__mul__": [lambda grad, left, right: grad * right.value,
                          lambda grad, left, right: grad * left.value],

              "__sub__": [lambda grad, left, right: grad,
                          lambda grad, left, right: (-1) * grad],

              "__truediv__": [lambda grad, left, right: grad / right,
                              lambda grad, left, right: grad * (-1) * left / right ** 2],
              # based on https://math.stackexchange.com/a/3850121
              "__matmul__": [lambda grad, left, right: Variable.dot(grad, Variable.transpose(right)),
                             lambda grad, left, right: Variable.dot(Variable.transpose(left), grad)
                             ],
              "sin": lambda grad, x: grad * Variable.cos(x),
              "cos": lambda grad, x: (-1) * grad * Variable.sin(x),
              "log": lambda grad, x: grad / x,
              "relu": lambda grad, x: grad * np.where(x > Variable(0), 1, 0),
              "sigmoid": lambda grad, x: grad * Variable.sigmoid(x) * (1 - Variable.sigmoid(x)),
              "sum": lambda grad, x: grad * np.ones_like(x.value),
              "transpose": lambda grad, x: Variable.transpose(grad)

              }