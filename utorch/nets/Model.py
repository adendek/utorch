import utorch.simplegrad as sg

from abc import abstractmethod
from collections.abc import Iterable
from functools import reduce


class NetworkParameter(sg.Variable):
  """
  This class represent a network parameter and it is a Variable type. All the networks parameters should be initialized using this wrapper.
  It is used to provide the list of model's parameters to the optimizer.
  """
  pass


class Model(object):
  """
  This is a base class for all the Neural Network models.
  Right now it is just a simple abstract class that define the interface.
  In the future it will provide more functionality, similarly to the torch.nn.Module.
  """

  @abstractmethod
  def forward(self, x, *args, **kwargs):
    """
    This method is called in order to propagate input tensor through the model or layer.
    :arg x: An input tensor
    """
    pass

  def __call__(self, x, *args, **kwargs):
    """ This method calls self.forward. it is implemented to mimic the interface of pytorch """
    return self.forward(x, *args, **kwargs)

  def get_parameters(self):
    """
    This function returns a list of all trainable model parameters.
    Useful when init optimizer, which has to keep track of model's parameters and updates.
    """

    def get_typed_parameter(model, parameter_type):
      typed_parameters = [getattr(model, atribute) for atribute in
                          filter(lambda x: not x.startswith("__"), dir(model))
                          if isinstance(getattr(model, atribute), parameter_type)]
      # the line above is unfortunetely not enough. The parameters can be storad in the list or some other iterable container. We need to extract them as well.
      # perhaps in the future this function will have to be recurrent, since it should support nested container. Right now, it supports one level only
      containers = [getattr(model, atribute) for atribute in
                    filter(lambda x: not x.startswith("__"), dir(model))
                    if isinstance(getattr(model, atribute), Iterable)]
      for container in containers:
        typed_parameters.extend([parameter for parameter in container if isinstance(parameter, parameter_type)])
      return typed_parameters

    def get_parameter_impl(model):
      internal_models = get_typed_parameter(model, Model)
      model_parameters = get_typed_parameter(model, NetworkParameter)
      parameter_list.extend(model_parameters)
      for submodel in internal_models: get_parameter_impl(submodel)

    parameter_list = []
    get_parameter_impl(self)
    return parameter_list

  def get_n_params(self):
    """
    This function allows to calculate the number of parameters that constitute a given model.
    """
    return sum([reduce(lambda x, y: x * y, layer.shape())
                for layer in self.get_parameters()])

