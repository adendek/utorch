from simplegrad import  Variable


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


class SGD(Optimizer):
    def __init__(self, model, learning_rate):
        self.model_params = model.get_parameters()
        self.learning_rate = learning_rate

    def update_model(self):
        for parameter in self.model_params:
            parameter.value -= self.learning_rate * parameter.grad.value

