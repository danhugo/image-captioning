import torch
import numpy as np
import os
import json
import random

def init_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def read_json_file(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data

def write_json_file(path, data):
    with open(path, "w") as f:
        data = json.dump(data, f)
    return data
