"""
Render a Value's computation graph with graphviz. Needs the graphviz
python package plus the actual graphviz binaries (`dot`) installed on
the system, not just the pip package.
"""
from graphviz import Digraph


def trace(root):
    # walk the graph backward from root and collect every node + edge once
    nodes, edges = set(), set()

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges


def draw_dot(root, rankdir='LR'):
    """Build a graphviz Digraph for the graph rooted at `root`.

    Each Value becomes a record node showing its data and grad. If a
    Value was produced by an op (e.g. '+', '*', 'tanh'), a small extra
    node for that op sits between it and the values that fed into it,
    so the picture reads left-to-right as data flowing through ops.
    """
    assert rankdir in ('LR', 'TB')
    nodes, edges = trace(root)
    dot = Digraph(format='svg', graph_attr={'rankdir': rankdir})

    for v in nodes:
        uid = str(id(v))
        dot.node(uid, label=f"{{ data {v.data:.4f} | grad {v.grad:.4f} }}", shape='record')
        if v._op:
            op_uid = uid + v._op
            dot.node(op_uid, label=v._op)
            dot.edge(op_uid, uid)

    for child, parent in edges:
        # edge goes into the op node that produced the parent, not the
        # parent's value node directly, so ops visually sit in between
        dot.edge(str(id(child)), str(id(parent)) + parent._op)

    return dot
