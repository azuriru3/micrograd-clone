"""
Draw the computation graph of a single tiny 2D neuron, showing data and
grad at every node. Run with: python examples/trace_graph.py
(needs graphviz's `dot` binary installed on your system, not just the
pip package)
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from micrograd.engine import Value
from micrograd.nn import Neuron
from micrograd.draw import draw_dot


def main():
    n = Neuron(2)
    x = [Value(1.0), Value(-2.0)]
    y = n(x)
    y.backward()

    dot = draw_dot(y)
    out_path = os.path.join(os.path.dirname(__file__), 'trace_graph_result')
    dot.render(out_path, cleanup=True)
    print(f"saved graph to {out_path}.svg")


if __name__ == '__main__':
    main()
