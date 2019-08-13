from contextlib import contextmanager
import json
import os
import random
import time

from attrdict import AttrDict
import numpy as np
import torch


@contextmanager
def timer(msg):
    t0 = time.time()
    print(f'[{msg}] start.')
    yield
    elapsed_time = time.time() - t0
    print(f'[{msg}] done in {elapsed_time / 60:.2f} min.')


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def load_config(config_path: str) -> AttrDict:
    with open(config_path) as f:
        config = json.load(f, object_hook=AttrDict)
    return config
