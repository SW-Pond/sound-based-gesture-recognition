import numpy as np
import csv


TEMPLATES_DIR = "jackknife\\templates\\"
# Key-gesture associations
GESTURE_TYPES = {'1':"zigzag", '2':"triangle", '3':"rectangle", '4':"x", 
                 '5':"c", '6':"arrow", '7':"check", '8':"caret", '9':"star",
                 'a':"double arch", 's':"s", 'w':"w", 'y':"y", 'z':"z"}
TEMPLATES_PER_GESTURE = 1
UNIFORM_RESAMPLE_PTS = 16 # For standard (non-stochastic) resampling
# Set Sakoe-Chiba band radius to 10% of resampled time series length
Radius = int(np.ceil(0.1 * UNIFORM_RESAMPLE_PTS))


class Gesture:
    def __init__(self):
        self.points = []
        self.gpdvs = []
        self.features = []

    def add_point(self, point):
        self.points.append(point)

    """
    n and variance are specified for stochastic resampling, otherwise, resample
    to UNIFORM_RESAMPLE_PTS with a uniform distance between each point.
    """
    def resample_points(self, n=UNIFORM_RESAMPLE_PTS, variance=0):
        resampled_points = [] 
        resampled_points.append(self.points[0])

        point_spacing = self.path_len() / (n - 1)

        # Used in case dist between curr point and last point < point spacing
        accumulated_dist = 0

        i = 1
        while i < len(self.points) and point_spacing > 0:
            curr_point = np.array(self.points[i])
            last_point = np.array(self.points[i - 1])
            curr_dist = np.linalg.norm(curr_point - last_point)

            if accumulated_dist + curr_dist >= point_spacing:
                curr_diff_vec = curr_point - last_point
                if curr_dist != 0:
                    next_pnt_factor = (point_spacing - accumulated_dist) / curr_dist
                else:
                    next_pnt_factor = 0.5

                resampled_point = self.points[i - 1] + next_pnt_factor * curr_diff_vec
                resampled_points.append(resampled_point)
                self.points.insert(i, resampled_point)
                accumulated_dist = 0
            else:   
                accumulated_dist += curr_dist

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

            for j in range(max(0, i - Radius), min(i + Radius + 1, num_vecs)):
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
        gesture_type = GESTURE_TYPES[g_key]

        dir = f"{TEMPLATES_DIR}{gesture_type}\\"

        for log_file_num in range(TEMPLATES_PER_GESTURE):
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
                        print(f"{TEMPLATES_PER_GESTURE} templates have "
                              f"already been logged for gesture: {gesture_type}")
                        
                    log_file.close()