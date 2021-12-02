import numpy as np
import abc

import utorch.simplegrad as sg
from utorch.nets.Model import Model, NetworkParameter


class LinearLayer(Model):
    def __init__(self, n_input, n_hidden, has_bias=False,name=None):
        layer_name = "_".join([name, "weights"])
        # TODO: move layer init into the separated method.
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


class RNNBase(Model):
    """
    This class defines an interface that every recurrent model has to implement.
    Currently, this is interface required the following methods to be overridden.
    :meth: 'initial_state" used to construct a initial core state.
    "forward: that propagates the input sequence through the network.
    """
    @abc.abstractmethod
    def init_state(self, batch_size):
        """
        This method construct an initial state for a given implementation of RNN.
        Args:
            batch_size: integer that defines the batch_size.
        """


class RNNCell(RNNBase):
    """
    This class implements Elman Net RNN, also called VanillaRNN.
    For each input element :math: `x_{i}` the output state :math: `h_{i}` is calculated using the following formula:
    :math:
    h_{i+1} = a( Wh_{i} + Ux_{i} + b)
    where:
    a is a activation function (reLu)
    W is a weight matrix that map between previous hidden state :math: `h_{i-1}`  the current one :math: `h_{i}` ,
    U is a weight matrix that map between input :math: `x_{i}`
    """
    def __init__(self, input_size: int, hidden_size: int, activation: Model):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_input_hidden = NetworkParameter(np.random.uniform(0, np.sqrt(2/(self.input_size+self.hidden_size)),
                                                    size=(self.hidden_size, self.input_size)),
                                                    name="weight input hidden")
        self.weight_hidden_hidden = NetworkParameter(np.random.normal(0, np.sqrt(1/(self.hidden_size)),
                                                    size=(self.hidden_size, self.hidden_size)),
                                                    name="weight hidden hidden")
        self.bias = NetworkParameter(np.zeros(self.hidden_size), "bias")
        self.activation = activation

    def init_state(self, batch_size):
        """
        Initialize the hidden state of shape (batch_size, hidden_size)
        """
        return sg.Variable(np.zeros(shape=(batch_size, self.hidden_size)))

    def forward(self, x: sg.Variable, hidden_previous: NetworkParameter, ):
        """
        This function implements the forward propagation of the VanillaRNN. See class docstring for formula.
        It process a single input :math: 'x_{i}' only. In order to build RNN network
        the unfold method has to be implemented, see example.

        :param x: a single timestamp of tensor of shape (n_batch, n_features)
        :param hidden_previous: previously calculated hidden state of shape (n_hidden, )
        :return: current hidden state of shape(batch_size, hidden_size).
        """
        return self.activation(
            x @ sg.Variable.transpose(self.weight_input_hidden) +
            hidden_previous @ sg.Variable.transpose(self.weight_hidden_hidden) +
            self.bias
        )





