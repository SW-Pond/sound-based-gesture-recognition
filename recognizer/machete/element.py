import numpy as np

class Element:
    def __init__(self):
        self.gesture_frame_len = 1
        self.cumulative_cost = 0
        self.cumulative_length = 0
        self.score = np.inf
