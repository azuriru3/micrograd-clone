import pytest
from micrograd.engine import Value

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def test_add_mul_pow_chain():
    # z = x*y + y**2, hand-derived:
    # dz/dx = y
    # dz/dy = x + 2y
    x = Value(3.0)
    y = Value(-4.0)
    z = x * y + y**2
    z.backward()

    assert z.data == pytest.approx(4.0)
    assert x.grad == pytest.approx(-4.0)
    assert y.grad == pytest.approx(3.0 + 2 * -4.0)


def test_sub_div_chain():
    # w = (x - y) / (x + y), hand-derived via quotient rule:
    # dw/dx = 2y / (x+y)^2
    # dw/dy = -2x / (x+y)^2
    x = Value(3.0)
    y = Value(-4.0)
    w = (x - y) / (x + y)
    w.backward()

    s = 3.0 + -4.0
    assert w.data == pytest.approx((3.0 - -4.0) / s)
    assert x.grad == pytest.approx(2 * -4.0 / s**2)
    assert y.grad == pytest.approx(-2 * 3.0 / s**2)


def test_gradient_accumulates_when_value_reused():
    # x used twice in the same expression - grad should sum across both
    # paths, not get overwritten by the second one.
    x = Value(5.0)
    y = x + x
    y.backward()
    assert x.grad == pytest.approx(2.0)


def test_tanh_matches_manual_derivative():
    x = Value(0.5)
    y = x.tanh()
    y.backward()

    import math
    t = math.tanh(0.5)
    assert y.data == pytest.approx(t)
    assert x.grad == pytest.approx(1 - t**2)


def test_relu_zeroes_gradient_for_negative_input():
    x = Value(-2.0)
    y = x.relu()
    y.backward()
    assert y.data == 0.0
    assert x.grad == 0.0


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_matches_torch_autograd():
    # same computation, once in our engine and once in torch, then compare.
    a, b, c = -2.0, 3.0, -1.0

    xa, xb, xc = Value(a), Value(b), Value(c)
    out = (xa * xb + xc).tanh() * xa
    out.backward()

    ta = torch.tensor(a, requires_grad=True, dtype=torch.double)
    tb = torch.tensor(b, requires_grad=True, dtype=torch.double)
    tc = torch.tensor(c, requires_grad=True, dtype=torch.double)
    tout = (ta * tb + tc).tanh() * ta
    tout.backward()

    assert out.data == pytest.approx(tout.item())
    assert xa.grad == pytest.approx(ta.grad.item())
    assert xb.grad == pytest.approx(tb.grad.item())
    assert xc.grad == pytest.approx(tc.grad.item())
