"""
This is a boilerplate pipeline 'pipelines'
generated using Kedro 0.18.7
"""

import gc

import numpy as np
from matplotlib import pyplot as plt
import cv2

def _clean_gpu_memory():
    gc.collect()
    torch.cuda.empty_cache()
