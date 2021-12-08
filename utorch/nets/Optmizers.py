from utorch.simplegrad import Variable


class Optimizer(object):
    def __init__(self):
        self.model_params = None

    def update_model(self):
        pass

    def zero_grad(self):
        """
        This functions set gradient of a given model to zero.
        It is sufficient since graph is generated dynamically, which means the intermediate nodes
        (results of some calculations) will be recreated each time when forward method is called.
        """
        if self.model_params is None:
            raise RuntimeError(
                "No model has been given. You should initialize the Optimize by"
                " providing model  as a constructor's parameter"
            )

        for param in self.model_params:
            self._zero_grad(param)
            if hasattr(param, "velocity"):
                del param.velocity
            if hasattr(param, "mean_square"):
                del param.mean_square
            param.grad = Variable(0)
            param.velocity=Variable(0)
            param.mean_square=Variable(0)

    def _zero_grad(self, param_grad):
        grad_parents = param_grad.parents
        while grad_parents:
            parent =  grad_parents.pop()
            self._zero_grad(parent)
            del parent
        del param_grad

class SGD(Optimizer):
    def __init__(self, model, learning_rate):
        self.model_params = model.get_parameters()
        self.learning_rate = learning_rate

    def update_model(self):
        for parameter in self.model_params:
            parameter.value -= self.learning_rate * parameter.grad.value


class SGDm(Optimizer): #TODO: WIP, make tests
    """
    SGD with momentum
    http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf
    The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and is following the PyTorch implementation
    https://github.com/pytorch/pytorch/blob/master/torch/optim/sgd.py
    """
    def __init__(self, model, learning_rate=1e-3, momentum=0.9):
        self.model_params = model.get_parameters()
        self.learning_rate = learning_rate
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        self.momentum=momentum

    def update_model(self):
        for parameter in self.model_params:
            parameter.velocity *= self.momentum
            parameter.velocity += self.learning_rate * parameter.grad.value
            parameter.value -= self.learning_rate * parameter.velocity.value


class RMSProp(Optimizer): #TODO: WIP, make tests
    """Implements RMSprop algorithm.
        Proposed by G. Hinton in his course <https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf>
    """
    def __init__(self, model, learning_rate=1e-3, decay=0.9, eps=1e-8):
        self.model_params = model.get_parameters()
        self.lr, self.decay, self.eps = learning_rate, decay, eps

    def update_model(self):
        for i, parameter in enumerate(self.model_params):
            parameter.mean_square.value *= self.decay
            parameter.mean_square += (1.0 - self.decay) * parameter.grad.value * parameter.grad.value
            parameter.mean_square += self.eps
            parameter.mean_square = parameter.mean_square.__pow__(1/2)

            parameter.value -= (self.lr * parameter.grad.value).__truediv__(parameter.mean_square.value)


class Adam(Optimizer):  # TODO: WIP, make tests
    def __init__(self, model, learning_rate=1e-3, momentum=0.9, decay=0.9, eps=1e-8):
        self.model_params = model.get_parameters()
        self.lr, self.b1, self.b2, self.eps, self.t = learning_rate, momentum, decay, eps, 0

    def update_model(self):
        self.t = self.t + 1
        a = self.lr * ((1.0 - self.b2 ** self.t) ** 0.5) / (1.0 - self.b1 ** self.t)
        for parameter in self.model_params:
            parameter.velocity.value *= self.b1
            parameter.velocity.value +=  (1.0 - self.b1) * parameter.grad.value

            parameter.mean_square.value *= self.b2
            parameter.mean_square += (1.0 - self.b2) * parameter.grad.value * parameter.grad.value
            parameter.mean_square += self.eps
            parameter.mean_square = parameter.mean_square.__pow__(1/2)

            parameter.value -= a * parameter.velocity.value.__truediv__(parameter.mean_square.value)

