from graphviz import Digraph
import numpy as np
import matplotlib.pyplot as plt
import simplegrad

def draw_graph(node):
  """
  Function to visualize the structure of a computational graph.
  :return: graphviz.dot object
  """
  node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')
  dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
  seen = set()

  def add_node(node):
    if node in seen:
      return

    if node.is_leaf:
      dot.node(str(id(node)), str(node.value), fillcolor='lightblue')
    else:
      fun_name = str(node.fun).split(".")[1]
      dot.node(str(id(node)), fun_name)

    seen.add(node)
    for parent in node.parents:
      dot.edge(str(id(parent)), str(id(node)))
      add_node(parent)
  add_node(node)
  return dot


def plot_function_and_der(fun, range=[0,1], n_points=100):

  def evaluate_derivative(x, fun):
    x_var = simplegrad.Variable(x)
    y = fun(x_var)
    y.backward()
    return y.value, x_var.grad.value

  x_points = np.linspace(start = range[0], stop=range[1], num=n_points)
  y = np.transpose(np.array(list(map(lambda x: evaluate_derivative(x, fun), x_points))))
  plt.plot(x_points, y[0], color='blue')
  plt.plot(x_points, y[1], color='red', linestyle='dashed')
