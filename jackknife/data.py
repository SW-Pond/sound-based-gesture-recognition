import numpy as np
import csv
from . import data_utils as d_u
from collections import deque


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
                #  more than gpdvs
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


class Manager:
    def __init__(self, pipe_conn):
        self.pipe_conn = pipe_conn
        # Flag for communicating with classifier through pipe;
        #   FLAG_DEFAULT ==> do nothing
        self.FLAG_DEFAULT = 0
        self.flag = self.FLAG_DEFAULT

        self.MAX_HISTORY_POINTS = 25
        self.point_history = deque(maxlen=self.MAX_HISTORY_POINTS)

        self.WINDOW_INCREMENT = 5
        self.INIT_WINDOW_SIZE = 5
        self.window_size = self.INIT_WINDOW_SIZE

        # For template logging only
        self.curr_template = Template()
        self.curr_point = []

        # For these, first element is the score, second is the gesture name
        self.scnd_last_best_match = None
        self.last_best_match = None
        self.best_match = None

    def process_point(self, point):
        self.curr_point = np.copy(point)
        self.point_history.append(point)

        if self.pipe_conn.poll():
            self.flag = self.pipe_conn.recv()

        # If classifier is ready and there are enough points in history
        if self.flag == 1 and len(self.point_history) >= self.window_size:
            window_points = []
            window_end = len(self.point_history)

            for i in range(window_end - self.window_size, window_end):
                window_points.append(self.point_history[i])
                
            self.pipe_conn.send(window_points)

            self.window_size += self.WINDOW_INCREMENT

            self.flag = self.FLAG_DEFAULT

        # If classifier is done with latest window
        if self.flag == 2:
            match = self.pipe_conn.recv()
            
            if self.best_match == None:
                self.best_match = match

            elif match[0] < self.best_match[0]:
                self.best_match = match
            
            if self.window_size > self.MAX_HISTORY_POINTS:
                self.window_size = self.INIT_WINDOW_SIZE

                if self.last_best_match != None and \
                   self.scnd_last_best_match != None:
                    
                    if self.best_match[1] == self.last_best_match[1] and \
                    self.last_best_match[1] == self.scnd_last_best_match[1]:
                        
                        print(self.best_match[1])
                        self.scnd_last_best_match = None
                        self.last_best_match = None
                        self.best_match = None
                        #self.point_history.clear()
                        
                self.scnd_last_best_match = self.last_best_match
                self.last_best_match = self.best_match
                self.best_match = None

    def check_pressed_key(self, key_event):
        key = key_event.name

        if key == 'r':
            self.reset_curr_template()

        if key == 't':
            self.curr_template.record_point(self.curr_point)
        
        if key in d_u.GESTURE_TYPES.keys():
            self.curr_template.log(g_key=key)
            self.reset_curr_template()

    def reset_curr_template(self):
        self.curr_template = Template()
        print("Current template reset\n")