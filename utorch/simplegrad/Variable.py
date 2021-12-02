import numpy as np
import utorch
from utorch.simplegrad.Primitives import primitives


class Variable(object):
    def __init__(self, value=None, parents=None, fun=None, name=None, args=None):
        if value is not None:
            self._init_if_leaf(value)
        else:
            self._init_as_inner_node(parents, fun, args)
        self.adj_list = []
        self.name = name

    def is_leaf(self):
        return self.is_leaf

    def _init_if_leaf(self, value):
        self.value = value
        self.is_leaf = True
        self.fun = None
        self.args = None
        self.parents = []
        self.grad = 0

    def _init_as_inner_node(self, parents, fun, args=None):
        if args is None:
            self.value = fun(*list(map(lambda x: x.value, parents)))
        else:
            fun_params = list(map(lambda x: x.value, parents))
            fun_params.append(args)
            self.value = fun(*fun_params)

        self.fun = fun
        self.args = args
        self.is_leaf = False
        self.parents = parents
        self.grad = 0

        # No idea if we need this. We will see
        list(map(lambda parent: parent.adj_list.append(self), parents))

    def shape(self):
        if isinstance(self.value, int) or isinstance(self.value, float):
            return (1,)
        else:
            return self.value.shape

    def single_operator(self, fun, args=None):
        return Variable(parents=[self], fun=fun, args=args)

    def _two_variable_operator(self, other, fun, args=None):
        if not isinstance(other, Variable):
            other = Variable(other)
        return Variable(parents=[self, other], fun=fun, args=args)

    def backward(self):
        sorted_nodes = topological_sort(self)
        backward(sorted_nodes)

    def __ge__(self, other):
        if isinstance(other, Variable):
            return np.greater_equal(self.value, other.value)
        else:
            return np.greater_equal(self.value, other)

    def __le__(self, other):
        if isinstance(other, Variable):
            return np.less_equal(self.value, other.value)
        else:
            return np.less_equal(self.value, other)

    def __eq__(self, other):
        if isinstance(other, Variable):
            return np.equal(self.value, other.value)
        else:
            return np.equal(self.value, other)

    def __lt__(self, other):
        if isinstance(other, Variable):
            return np.less(self.value, other.value)
        else:
            return np.less(self.value, other)

    def __ne__(self, other):
        if isinstance(other, Variable):
            return not self.__eq__(other)

    def __hash__(self):
        return hash(id(self))

    def __add__(self, other):
        return self._two_variable_operator(other, lambda x, y: x + y)

    def __mul__(self, other):
        return self._two_variable_operator(other, lambda x, y: x * y)

    def __radd__(self, other):
        return self._two_variable_operator(other, lambda x, y: x + y)

    def __rmul__(self, other):
        return self._two_variable_operator(other, lambda x, y: x * y)

    def __matmul__(self, other):
        return self._two_variable_operator(other, lambda x, y: np.matmul(x, y))

    def __sub__(self, other):
        return self._two_variable_operator(other, lambda x, y: np.subtract(x, y))

    def __rsub__(self, other):
        return self._two_variable_operator(other, lambda x, y: np.subtract(y, x))

    def __truediv__(self, other):
        return self._two_variable_operator(other, lambda x, y: np.divide(x, y))

    def __pow__(self, other):
        return self.single_operator(lambda x: np.power(x, other))

    @classmethod
    def exp(cls, variable):
        return variable.single_operator(lambda x: np.exp(x))

    @classmethod
    def sin(cls, variable):
        return variable.single_operator(lambda x: np.sin(x))

    def __repr__(self):
        return "Node {}, name {},\n value {},\n fun {}, \n grad {} \n *****************".format(hex(id(self)),
                                                                                                self.name, self.value,
                                                                                                self.fun, self.grad)

    @classmethod
    def log(cls, variable):
        return variable.single_operator(lambda x: np.log(x))

    @classmethod
    def cos(cls, variable):
        return variable.single_operator(lambda x: np.cos(x))

    @classmethod
    def relu(cls, variable):
        return variable.single_operator(lambda x: np.where(x > Variable(0), x, 0))

    @classmethod
    def sigmoid(cls, variable):
        return variable.single_operator(lambda x: 1 / (1 + np.exp((-1) * x)))

    @classmethod
    def tanh(cls, variable):
        return variable.single_operator(lambda x: np.tanh(x))

    @classmethod
    def cosh(cls, variable):
        return variable.single_operator(lambda x: np.cosh(x))

    @classmethod
    def dot(clas, l_var, r_val):
        return l_var.__matmul__(r_val)

    @classmethod
    def abs(cls, variable):
        return variable.single_operator(lambda x: np.abs(x))

    @classmethod
    def sum(cls, variable, axis=None):
        return variable.single_operator(lambda x, args=None: np.sum(x, args), args=axis)

    @classmethod
    def transpose(cls, variable, args=None):
        return variable.single_operator(lambda x: np.transpose(x, args), args=args)

    @classmethod
    def clip_min(cls, variable, min):
        return variable.single_operator(lambda x, args: np.clip(x, a_min=args, a_max=None), args=min)

    @classmethod
    def sign(cls, variable):
        return variable.single_operator(lambda x: np.sign(x))


def backward(sorted_nodes):
    def get_function_name(fun):
        # This is the easiest way of extracting function name, and finding associated primitive in the dictionary.
        return str(fun).split(".")[1]

    def reverse_broadcasting(node, gradient):
        if len(node.shape()) < len(gradient.shape()):
            # print("reverse broadcast", node.shape(), gradient.shape())
            gradient = Variable.sum(gradient, axis=0)
            # print("reversed: ",gradient.shape() )
        return gradient

    sorted_nodes[0].grad = Variable(1)
    for node in sorted_nodes:
        if node.fun is None:
            continue
        # print("processing node", node)

        primitive_function = primitives[get_function_name(node.fun)]
        # right now, there are 2 options. Each node has two or one parent(s).
        # no option to have more than 2, since e.g. a+b+c will be converted into two nodes (a+b) + c
        if len(node.parents) == 2:
            # gradient with respect to one parent or with respect to another makes a fucking difference.
            # thus we need to treat them separately.
            parent_left, parent_right = node.parents
            left_update = primitive_function[0](node.grad, parent_left, parent_right, node.args)
            # print("left update", left_update)
            parent_left.grad += left_update
            parent_left.grad = reverse_broadcasting(parent_left, parent_left.grad)

            right_update = primitive_function[1](node.grad, parent_left, parent_right, node.args)
            # print("right_update update", right_update)
            parent_right.grad += right_update
            parent_right.grad = reverse_broadcasting(parent_right, parent_right.grad)

        else:
            parent = node.parents[0]
            update = primitive_function(node.grad, parent, node.args)

            # print("parent grad update", update)
            parent.grad += update
            parent.grad = reverse_broadcasting(parent, parent.grad)


def topological_sort(node):
  """
  A simple implementation of the topological sort of a given graph.
  For those of you, who want to know how to implement it we highly recommend this video
  https://www.youtube.com/watch?v=AfSk24UTFS8&t=603s&ab_channel=MITOpenCourseWare
  and the entire course.
  :param node: the node that you can use as a starting point
  :return: a list of topologically sorted nodes from a given input graph
  """
  def depth_first_search(node):
    visited.add(node)
    for parent in node.parents:
      if parent not in visited:
        depth_first_search(parent)
    sorted_nodes.insert(0, node)

  sorted_nodes = []
  visited = set()
  for parent in node.parents:
    if parent not in visited:
      depth_first_search(parent)
  sorted_nodes.insert(0, node) # add the very first element
  return sorted_nodes


if __name__=="__main__":
    def fib(n):
        if n == 0 or n == 1:
            return n
        return fib(n - 1) + fib(n - 2)


    n = Variable(10)
    b = fib(n)
    print(b)
    b.backward()
    print(n.grad)


