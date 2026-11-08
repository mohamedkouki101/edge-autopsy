"""Seed every RNG the project uses (PLAN §4 "Seeds")."""

from __future__ import annotations

import os
import random

import numpy as np


def set_all_seeds(seed: int) -> np.random.Generator:
    """Seed random, numpy (legacy + returned Generator), torch; enable deterministic torch.

    PYTHONHASHSEED only affects interpreters started *after* this call (e.g. subprocesses);
    the current process's str hashing is fixed at startup. Code must therefore never rely on
    set/dict iteration order of strings for anything that reaches results.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - legacy global RNG still seeded for third-party code
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a pinned dependency
        pass
    else:
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
    return np.random.default_rng(seed)
