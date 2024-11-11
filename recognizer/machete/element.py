import numpy as np

class Element:
    def __init__(self):
        self.start_frame = -1
        self.end_frame = -1
        self.cumulative_cost = 0
        self.cumulative_length = 0
        self.score = np.inf
