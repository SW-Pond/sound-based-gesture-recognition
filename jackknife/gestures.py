import numpy as np
import csv
from . import data_utils as d_u


class Gesture:
    def __init__(self):
        self.points = []
        self.gpdvs = []
        self.features = []

    def add_point(self, point):
        self.points.append(point)

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
            # Normalizing to give a relative measure of per component abs dist
            abs_dist_vec /= abs_dist_vec_norm

        bb = bb_max - bb_min

        self.features.append(abs_dist_vec)
        self.features.append(bb)


class Query(Gesture):
    def __init__(self):
        super().__init__()


class Template(Gesture):
    def __init__(self, name=""):
        super().__init__()
        self.name = name

        # Upper and lower bands for determining lowest possible DTW score
        self.upper = []
        self.lower = []

    def envelop(self):
        num_vecs = len(self.gpdvs)
        components_per_vec = len(self.gpdvs[0])

        for i in range(num_vecs):
            maximum = np.full(components_per_vec, -np.inf)
            minimum = np.full(components_per_vec, np.inf)

            for j in range(max(0, i - d_u.R), min(i + d_u.R + 1, num_vecs)):
                for k in range(components_per_vec):
                    maximum[k] = max(maximum[k], self.gpdvs[j][k])
                    minimum[k] = min(minimum[k], self.gpdvs[j][k])

            self.upper.append(maximum)
            self.lower.append(minimum)

    # For template logging only
    def record_point(self, point):
        if len(self.points) == 0 or point is not self.points[-1]:
            print(f"Recording point {len(self.points)}")
            self.add_point(point)

    def log(self, g_key):
        gesture_type = d_u.GESTURE_TYPES[g_key]

        dir = d_u.PARENT_DIR + gesture_type + "\\"

        for log_file_num in range(d_u.TEMPLATES_PER_GESTURE):
            log_file_path = f"{dir}t{log_file_num}.csv"

            with open(log_file_path, "r+", newline='') as log_file:
                if log_file.read(1) == '':
                    print(f"Logging template {log_file_num + 1} for gesture: "
                          f"{gesture_type} ...")
                    
                    writer = csv.writer(log_file,)

                    for point in self.points:
                        writer.writerow(point)

                    log_file.close()

                    print("Successfully logged template")

                    break

                else:
                    if log_file_num == 2:
                        print(f"Three templates have already been logged for "
                              f"gesture: {gesture_type}")
                        
                    log_file.close()