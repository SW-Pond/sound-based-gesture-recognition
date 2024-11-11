import random as rand
import numpy as np


class Gesture:
    def __init__(self):
        self.points = []
        self.gpdvs = []
        self.features = []

    def add_point(self, point):
        self.points.append(point)

    """
    n and variance are specified for stochastic resampling, otherwise, resample
    to default n with a uniform distance between each point.
    """
    def resample_points(self, n=16, variance=0):
        resampled_points = [] 
        resampled_points.append(self.points[0])
        intervals = np.empty(n - 1)

        # Uniform resampling
        if variance == 0:
            intervals = np.full((n - 1), 1 / (n - 1))
        # Stochastic resampling; randomly determine the fractions of the total
        #   path length at which each successive point will be interpolated
        else:
            for i in range(n - 1):
                r = rand.random()
                intervals[i] = 1 + r * np.sqrt(12 * variance)

            intervals /= np.sum(intervals)

        """
        I: interval (distance along path to interpolate from last point)
        d: distance between current point and last
        D: accumulates d until D + d is enough to interpolate at I along path
           from last interpolated point
        t: factor for calculating the interpolated point
        cnt: count of interpolated points
        """
        path_dist = self.path_len()
        cnt = 0
        I = path_dist * intervals[0]
        D = 0
        i = 1
        while i < len(self.points):
            curr_point = np.array(self.points[i])
            prev_point = np.array(self.points[i - 1])
            
            d = np.linalg.norm(curr_point - prev_point)

            if D + d >= I:
                t = min(max((I - D) / d, 0), 1)

                interp_point = (1 - t) * prev_point + t * curr_point
                resampled_points.append(interp_point)
                self.points.insert(i, interp_point)
                D = 0
                cnt += 1

                if cnt < len(intervals):
                    I = path_dist * intervals[cnt]
            else:   
                D += d

            i += 1
        
        while len(resampled_points) < n:
            resampled_points.append(self.points[-1])
        
        self.points = resampled_points

    def path_len(self):
        length = 0

        for i in range(1, len(self.points)):
            curr_point = np.array(self.points[i])
            last_point = np.array(self.points[i - 1])
            length += np.linalg.norm(curr_point - last_point)

        return length

    # Convert n points to n-1 gesture path direction (unit) vectors
    def populate_gpdvs(self):
        # Convert to numpy array for ease of vector operations
        np_points = np.array(self.points)

        for i in range(len(np_points) - 1):
            between_pnt_vec = np_points[i + 1] - np_points[i]
            vec_norm = np.linalg.norm(between_pnt_vec)

            # Handles division by 0
            if vec_norm != 0:
                # Normalize
                gpdv = between_pnt_vec / vec_norm
            else:
                gpdv = between_pnt_vec

            self.gpdvs.append(gpdv)

    def extract_features(self):
        num_vecs = len(self.gpdvs)
        components_per_vec = len(self.gpdvs[0])
        # Per-component absolute distance traveled in gesture path
        abs_dist_vec = np.zeros(components_per_vec)
        bb_max = np.copy(self.points[0])
        bb_min = np.copy(self.points[0])

        for i in range(num_vecs):
            for j in range(components_per_vec):
                abs_dist_vec[j] += np.abs(self.gpdvs[i][j])

                # Using i + 1 for points since points will always have one
                # more than gpdvs
                bb_max[j] = max(bb_max[j], self.points[i + 1][j])
                bb_min[j] = min(bb_min[j], self.points[i + 1][j])

        abs_dist_vec_norm = np.linalg.norm(abs_dist_vec)
        # Handles division by 0
        if abs_dist_vec_norm != 0:
            abs_dist_vec /= abs_dist_vec_norm

        bb_vec = bb_max - bb_min
        bb_vec_norm = np.linalg.norm(bb_vec)
        # Handles division by 0
        if bb_vec_norm != 0:
            bb_vec /= bb_vec_norm
        
        self.features.append(abs_dist_vec)
        self.features.append(bb_vec)


class Template(Gesture):
    def __init__(self, name, frames):
        super().__init__()
        self.name = name
        self.points = frames
        self.rejection_threshold = np.inf
        
        self.resample_points()
        self.populate_gpdvs()
        self.extract_features()


class Query(Gesture):
    def __init__(self):
        super().__init__()