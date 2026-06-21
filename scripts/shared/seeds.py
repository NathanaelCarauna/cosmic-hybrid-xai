"""Global seed helper — call set_global_seeds() at the top of every runner main()."""
from __future__ import annotations

import os
import random


def set_global_seeds(seed: int = 42) -> None:
    """Fix all randomness sources for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
