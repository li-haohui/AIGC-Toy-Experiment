import os
import torch
import random
import numpy as np
import logging

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class AvgMeter:
    def __init__(self) -> None:
        self.value = [0]

    def update(self, val):
        self.value.append(val)

        while len(self.value) > 100:
            self.value.pop(0)

    def statistic(self):
        return sum(self.value) / len(self.value)

def training_setup():
    os.makedirs("./runs", exist_ok=True)
    num_run_dirs = len(os.listdir("./runs"))
    curr_run_dir = f"./runs/run_{num_run_dirs}"
    os.makedirs(curr_run_dir, exist_ok=False)

    logger = logging.getLogger(__name__)

    logging.basicConfig(
        filename=os.path.join(curr_run_dir,'info.log'),
        level=logging.INFO
    )

    return curr_run_dir, logger