from params import *

import numpy as np


class Sensor:
    """Sensor class
    Model sensors and provide functions used to locate them.
    """

    def __init__(self, length_range, priority_range, seed=None):
        self.x = np.random.randint(0, length_range)
        self.y = np.random.randint(0, length_range)
        self.p = np.random.randint(1, priority_range + 1)
        self.records = []

    def gen(self):
        self.x = 400
        self.y = 400
        self.p = 5
