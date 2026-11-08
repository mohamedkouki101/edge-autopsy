"""P0.5: set_all_seeds makes every RNG we use reproducible."""

import os
import random

import numpy as np
import torch

from edge_autopsy.seeds import set_all_seeds


def _draw():
    # legacy global numpy RNG on purpose: third-party code may use it
    return (random.random(), np.random.rand(), torch.rand(1).item())  # noqa: NPY002


def test_same_seed_same_draws():
    set_all_seeds(123)
    a = _draw()
    set_all_seeds(123)
    b = _draw()
    assert a == b


def test_different_seed_different_draws():
    set_all_seeds(1)
    a = _draw()
    set_all_seeds(2)
    assert a != _draw()


def test_returns_generator_and_sets_env():
    rng = set_all_seeds(7)
    assert isinstance(rng, np.random.Generator)
    assert rng.integers(0, 10**9) == np.random.default_rng(7).integers(0, 10**9)
    assert os.environ["PYTHONHASHSEED"] == "7"
    assert torch.are_deterministic_algorithms_enabled()
