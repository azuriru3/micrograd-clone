"""
Scalar-valued autograd engine.

Everything here is a graph of Value nodes. Every op that combines Values
records how to route gradients backward through itself (_backward), and
Value.backward() walks that graph in reverse topological order, calling
each node's _backward in turn. That's the whole trick behind backprop -
it's just the chain rule applied node by node, in the right order.
"""

class Value:
    """Stores a single scalar value and its gradient."""

    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0.0
        # internal bookkeeping used to build the autograd graph
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op  # the op that produced this node, just for debugging/repr

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            # d(out)/d(self) = 1, d(out)/d(other) = 1, so just pass the
            # upstream gradient straight through to both parents. We use
            # += because a Value can feed into more than one op (e.g. x + x),
            # and gradients from each path need to accumulate, not overwrite.
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            # product rule: d(out)/d(self) = other.data, d(out)/d(other) = self.data
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data ** other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data ** (other - 1)) * out.grad
        out._backward = _backward

        return out

    def tanh(self):
        x = self.data
        t = (pow(2.718281828459045, 2 * x) - 1) / (pow(2.718281828459045, 2 * x) + 1)
        out = Value(t, (self,), 'tanh')

        def _backward():
            # d/dx tanh(x) = 1 - tanh(x)^2
            self.grad += (1 - t ** 2) * out.grad
        out._backward = _backward

        return out

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'relu')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out

    def backward(self):
        # build a topological ordering of the graph so that every node
        # is processed only after everything that depends on it has
        # already had its gradient finalized. classic DFS post-order.
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # seed the gradient at the output (dL/dL = 1), then walk the
        # topo order backward, applying each node's local backward rule.
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * other**-1

    def __rtruediv__(self, other):
        return other * self**-1

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
