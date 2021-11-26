import numpy as np

import utorch.simplegrad as sg
from utorch.nets.Model import Model, NetworkParameter


class LinearLayer(Model):
    def __init__(self, n_input, n_hidden, has_bias=False,name=None):
        layer_name = "_".join([name, "weights"])
        # I can move the layer initialization into the different method, right now it is what it is.
        self.weights = NetworkParameter(np.random.normal(0,1, size=(n_hidden, n_input)),name=layer_name)
        self.has_bias = has_bias
        if has_bias:
            layer_name = "_".join([name, "bias"])
            self.bias = NetworkParameter(np.zeros(n_hidden),layer_name)

    def forward(self, x, *args, **kwargs):
        if self.has_bias:
            output = x@sg.Variable.transpose(self.weights) + self.bias
        else:
            output = x@sg.Variable.transpose(self.weights)
        return output


class ReLULayer(Model):
    def __init__(self):
        self.relu = sg.Variable.relu

    def forward(self, x, *args, **kwargs):
        return self.relu(x)


class StackedLayers(Model):
    def __init__(self, layers):
        StackedLayers._validate_layers(layers)
        self.layers = layers

    def forward(self, x, *args, **kwargs):
        x = self.layers[0](x)
        for layer in self.layers[1:]:
            x = layer(x)
        return x

    def __getitem__(self, index):
        return self.layers[index]

    @classmethod
    def _validate_layers(cls, layers):
        if len(layers) < 2:
            raise RuntimeError(
                "You should use *StackedLayers* if you want to stack a couple of layers, no need to use it with a single layer!")
        if not all([map(lambda layer: isinstance(layer, Model), layers)]):
            raise RuntimeError(
                "The input to the *StackedLayers* is a list of layers or Models, please provide a correct one!")