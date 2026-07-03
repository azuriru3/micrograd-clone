# micrograd-clone

This is my own from-scratch rebuild of Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd), done as a learning exercise. The idea is to actually understand backpropagation by implementing it myself instead of just calling `.backward()` on a PyTorch tensor and trusting it works.

I followed the same shape as the original (a tiny `Value` class that tracks a computation graph, plus a small MLP built on top of it) but wrote the code myself rather than copying it, so some of the naming and structure differs.

## what this actually is

An autograd engine is the thing under the hood of every deep learning framework. When you do `z = x * y`, it doesn't just compute the number — it also remembers *how* `z` was computed (from `x` and `y`, via multiplication), so that later, when you have some final loss value, you can ask "how much would the loss change if I nudged `x` a tiny bit?" for every value in the whole graph, automatically. That's what `backward()` does — it's reverse-mode automatic differentiation, applied one scalar at a time.

Once you have that, a neural net is nothing special. It's just a chain of `Value` multiplications, additions, and a nonlinearity, and training is: run the graph forward, call `backward()`, nudge every weight a little bit in the direction that reduces the loss, repeat.

## structure

```
micrograd/
  engine.py   - the Value class, this is the whole autograd engine
  nn.py       - Neuron, Layer, MLP, built entirely out of Value ops
tests/
  test_engine.py
examples/
  train_toy.py - trains a small MLP on a toy 2D classification problem
```

## running it

```
pip install -r requirements.txt
pytest tests/
python examples/train_toy.py
```

`train_toy.py` generates a small two-moons-ish dataset, trains a `[2, 16, 16, 1]` MLP on it with plain SGD (no Adam, no momentum, nothing fancy), and pops up two plots: the loss curve and the learned decision boundary. It also saves the plot to `examples/train_toy_result.png`.

If you have `torch` installed, `tests/test_engine.py` will also run a test that checks the gradients from this engine against PyTorch's autograd on the same computation, as a sanity check. It's skipped automatically if torch isn't there.

## why tanh

`engine.py` implements both `tanh` and `relu` as nonlinearities, but `Neuron` in `nn.py` defaults to tanh. Mostly because that's what I originally saw this done with and the smooth gradient made debugging easier when I was checking my backward pass by hand — no dead-neuron problem to worry about while I was still figuring out if my chain rule implementation was even correct. Once the engine was solid, tanh vs relu barely matters for a toy problem this small, so I never bothered switching the default.

## what I learned

*(filling this in later)*
