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
            param.grad = Variable(0)
            param.velocity=Variable(0)


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
