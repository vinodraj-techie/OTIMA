import numpy as np
import random

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

NUM_SKUS = 1000
NUM_DAYS = 365
NUM_ZONES = 3
AISLES_PER_ZONE = 5
RACKS_PER_AISLE = 10
BINS_PER_RACK = 10

NUM_ORDERS = 30000
MAX_ORDER_LINES = 8
