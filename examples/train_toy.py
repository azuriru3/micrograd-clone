"""
Train the MLP on a toy 2D binary classification problem and plot the
loss curve + decision boundary. Run with: python examples/train_toy.py
"""
import random
import numpy as np
import matplotlib.pyplot as plt

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from micrograd.engine import Value
from micrograd.nn import MLP


def make_moons(n_samples=100, noise=0.1):
    # avoid a hard sklearn dependency - two interleaving half-moons, hand rolled
    n_per_class = n_samples // 2

    theta = np.linspace(0, np.pi, n_per_class)
    x0 = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    x1 = np.stack([1 - np.cos(theta), 1 - np.sin(theta) - 0.5], axis=1)

    X = np.concatenate([x0, x1], axis=0)
    X += np.random.normal(scale=noise, size=X.shape)
    y = np.array([0] * n_per_class + [1] * n_per_class)
    return X, y


def loss_fn(model, X, y):
    # max-margin (hinge) loss, same idea SVMs use - encourages the correct
    # class score to beat the wrong one by a margin of at least 1
    inputs = [[Value(xi) for xi in row] for row in X]
    scores = [model(xrow) for xrow in inputs]

    labels = [1 if yi == 1 else -1 for yi in y]
    losses = [(1 + -yi * scorei).relu() for yi, scorei in zip(labels, scores)]
    data_loss = sum(losses) * (1.0 / len(losses))

    # small L2 term so the weights don't blow up
    alpha = 1e-4
    reg_loss = alpha * sum((p * p for p in model.parameters()))
    total_loss = data_loss + reg_loss

    accuracy = [(yi > 0) == (scorei.data > 0) for yi, scorei in zip(labels, scores)]
    return total_loss, sum(accuracy) / len(accuracy)


def train():
    random.seed(42)
    np.random.seed(42)

    X, y = make_moons(n_samples=100, noise=0.1)
    model = MLP(2, [16, 16, 1])
    print(model)
    print("number of parameters:", len(model.parameters()))

    steps = 100
    loss_history = []

    for k in range(steps):
        total_loss, acc = loss_fn(model, X, y)

        model.zero_grad()
        total_loss.backward()

        # step size decay, otherwise it starts oscillating once it gets close
        learning_rate = 1.0 - 0.9 * k / steps
        for p in model.parameters():
            p.data -= learning_rate * p.grad

        loss_history.append(total_loss.data)
        if k % 10 == 0 or k == steps - 1:
            print(f"step {k}: loss {total_loss.data:.4f}, accuracy {acc * 100:.1f}%")

    plot_results(model, X, y, loss_history)


def plot_results(model, X, y, loss_history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    ax1.plot(loss_history)
    ax1.set_xlabel("step")
    ax1.set_ylabel("loss")
    ax1.set_title("training loss")

    h = 0.05
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    grid_inputs = [[Value(xi) for xi in row] for row in np.c_[xx.ravel(), yy.ravel()]]
    scores = [model(xrow).data for xrow in grid_inputs]
    Z = np.array(scores).reshape(xx.shape)

    ax2.contourf(xx, yy, Z > 0, alpha=0.3, cmap='coolwarm')
    ax2.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k')
    ax2.set_title("decision boundary")

    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'train_toy_result.png'))
    print("saved plot to examples/train_toy_result.png")
    plt.show()


if __name__ == '__main__':
    train()
