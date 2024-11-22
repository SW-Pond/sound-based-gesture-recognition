import numpy as np
from collections import deque
from .element import Element


class Template:
    def __init__(self, name, sample_pts, theta, epsilon):
        self.name = name
        # Circular buffer over two CDP rows
        self.row = deque([[],[]], maxlen=2)
        self.T = []
        self.gesture_frame_len = 1
        self.curr_row_idx = 0
        self.s1 = 0
        self.s2 = 0
        self.s3 = 0
        self.total = 0
        self.n = 0
        self.prev = []
        self.first_col_val = (1 - np.cos(theta)) ** 2

    
        # Resample using angular DP and build template and initial CDP matrix
        pts = self.angular_dp(np.array(sample_pts), epsilon)
        N = len(pts)
        for i in range(N):
            elem1 = Element()
            elem2 = Element()
            if i == 0: 
                elem1.score = self.first_col_val
                elem2.score = self.first_col_val

            self.row[0].append(elem1)
            self.row[1].append(elem2)

            if i > 0:
                v = pts[i] - pts[i - 1]
                v /= np.linalg.norm(v)
                self.T.append(v)
        
        # Calculate correction factor values and weights
        f2l = pts[N - 1] - pts[0]
        diag_length = self.diagonal_length(pts)
        length = self.path_length(pts)
        f2l_length = np.linalg.norm(f2l)
        self.f2l = f2l / np.linalg.norm(f2l)
        self.openness = f2l_length / length
        self.w_closedness = 1 - f2l_length / diag_length
        self.w_f2l = min(1, 2 * f2l_length / diag_length)

    def reset(self):
        self.row = deque([[],[]], maxlen=2)
        self.gesture_frame_len = 1
        self.curr_row_idx = 0
        self.s1 = 0
        self.s2 = 0
        self.s3 = 0
        self.total = 0
        self.n = 0
        self.prev = []

        # Reset CDP matrix
        for i in range(len(self.T) + 1):
            elem1 = Element()
            elem2 = Element()
            if i == 0: 
                elem1.score = self.first_col_val
                elem2.score = self.first_col_val

            self.row[0].append(elem1)
            self.row[1].append(elem2)
        
    def angular_dp(self, trajectory, epsilon):
        # Determine threshold that stops recursion
        diag_length = self.diagonal_length(trajectory)
        epsilon *= diag_length

        # Recursively find most descriptive points
        new_pts = []
        N = len(trajectory)
        new_pts.append(trajectory[0])
        self.angular_dp_recursive(trajectory, 0, N - 1, new_pts, epsilon)
        new_pts.append(trajectory[N - 1])
        return new_pts
    
    def angular_dp_recursive(self, trajectory, start, end, new_pts, epsilon):
        # Base case
        if start + 1 >= end: return

        # Calculate distance from every point in [start+1, end-1] to the line
        # defined by trajectory[start] to trajectory[end]
        AB = trajectory[end] - trajectory[start]
        denom = np.inner(AB, AB)
        if denom == 0: return

        largest = epsilon
        selected = -1

        for idx in range(start + 1, end):
            # Project point onto line segment AB
            AC = trajectory[idx] - trajectory[start]
            numer = np.inner(AB, AC)
            d2 = np.inner(AC, AC) - (numer ** 2 / denom)

            # Get vectors made by end points and this point
            v1 = trajectory[idx] - trajectory[start]
            v2 = trajectory[end] - trajectory[idx]
            l1 = np.linalg.norm(v1)
            l2 = np.linalg.norm(v2)
            if l1 * l2 == 0: continue

            # Calculate weighted distance and save if it's the best so far
            d = np.inner(v1, v2) / (l1 * l2)
            distance = d2 * np.arccos(d) / np.pi
            if distance >= largest:
                largest = distance
                selected = idx
            
        if selected == -1: return

        # If the subsequence is split, then recurse into each half. Also save
        # the split point.
        self.angular_dp_recursive(trajectory, start, selected, new_pts, epsilon)
        new_pts.append(trajectory[selected])
        self.angular_dp_recursive(trajectory, selected, end, new_pts, epsilon)

    def diagonal_length(self, trajectory):
        components_per_vec = len(trajectory[0])
        bb_max = np.copy(trajectory[0])
        bb_min = np.copy(trajectory[0])

        for i in range(1, len(trajectory)):
            for j in range(components_per_vec):
                bb_max[j] = max(bb_max[j], trajectory[i][j])
                bb_min[j] = min(bb_min[j], trajectory[i][j])

        bb_vec = bb_max - bb_min

        return np.linalg.norm(bb_vec)

    def path_length(self, pts):
        length = 0

        for i in range(1, len(pts)):
            curr_point = np.array(pts[i])
            last_point = np.array(pts[i - 1])
            length += np.linalg.norm(curr_point - last_point)

        return length