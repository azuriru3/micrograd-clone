import pytest
from micrograd.engine import Value
from micrograd.nn import Neuron, Layer, MLP


def test_neuron_parameter_count():
    n = Neuron(3)
    assert len(n.parameters()) == 4  # 3 weights + 1 bias


def test_neuron_output_is_tanh_bounded():
    n = Neuron(2)
    out = n([Value(1.0), Value(-1.0)])
    assert -1.0 < out.data < 1.0


def test_linear_neuron_skips_nonlinearity():
    # nonlin=False should give back the raw weighted sum, so it can
    # blow past tanh's (-1, 1) range if the weights push it there.
    n = Neuron(1, nonlin=False)
    n.w[0] = Value(5.0)
    n.b = Value(0.0)
    out = n([Value(10.0)])
    assert out.data == pytest.approx(50.0)


def test_layer_parameter_count_and_output_size():
    layer = Layer(3, 4)
    assert len(layer.parameters()) == 4 * (3 + 1)
    out = layer([Value(1.0), Value(2.0), Value(3.0)])
    assert len(out) == 4


def test_layer_with_single_output_unwraps_to_scalar():
    layer = Layer(2, 1)
    out = layer([Value(1.0), Value(1.0)])
    assert isinstance(out, Value)


def test_mlp_parameter_count():
    mlp = MLP(2, [4, 4, 1])
    # layer sizes: (2->4), (4->4), (4->1)
    expected = 4 * (2 + 1) + 4 * (4 + 1) + 1 * (4 + 1)
    assert len(mlp.parameters()) == expected


def test_mlp_last_layer_is_linear():
    mlp = MLP(2, [4, 1])
    assert mlp.layers[-1].neurons[0].nonlin is False
    assert mlp.layers[0].neurons[0].nonlin is True


def test_zero_grad_resets_all_parameters():
    mlp = MLP(2, [4, 1])
    out = mlp([Value(1.0), Value(-1.0)])
    out.backward()
    assert any(p.grad != 0.0 for p in mlp.parameters())

    mlp.zero_grad()
    assert all(p.grad == 0.0 for p in mlp.parameters())
