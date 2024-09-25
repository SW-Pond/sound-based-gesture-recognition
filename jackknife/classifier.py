import numpy as np
from . import data


class Classifier:
    def __init__(self, pipe_conn):
        self.NUM_RESAMPLE_POINTS = 16
        self.pipe_conn = pipe_conn
        self.gesture_templates = data.get_all_templates()
        self.num_gestures = len(self.gesture_templates)
        self.templates_per_gest = len(self.gesture_templates[0])

        # Resample templates and convert to gesture path direction vectors
        for gest_type in self.gesture_templates:
            for template in gest_type:
                template.points = self.resample(template.points, 
                                                self.NUM_RESAMPLE_POINTS)
                template.points = self.to_gpdvs(template.points)

    def classify(self):
        while True:
            # Ready for new points
            self.pipe_conn.send(1)

            num_recv_points = self.pipe_conn.recv()
            # Data manager ready to send to points
            if num_recv_points != 0:
                points = []
                scores = {}

                for i in range(num_recv_points):
                    points.append(self.pipe_conn.recv())
                
                for i in range(5, len(points), 5):
                    query = data.Query()
                    for j in range(i):
                        query.add_point(points[j])
                    
                    # Resample query and convert to gesture path direction vectors
                    query.points = self.resample(query.points, 
                                                 self.NUM_RESAMPLE_POINTS)
                    query.points = self.to_gpdvs(query.points)

                    for i in range(1):    
                    #for i in range(self.templates_per_gest):
                        for j in range(self.num_gestures):
                            template = self.gesture_templates[j][i]
                            gesture_name = template.name
                            score = self.dtw(template.points, query.points)
                            scores.update({score: gesture_name})
                    
                # Decision using argmin
                #print(scores.keys())
                smallest_score = min(scores.keys())
                print(smallest_score)
                gesture_guess = scores[smallest_score]
                print(gesture_guess)

    # Resample to n equidistant points along gesture path
    def resample(self, points, n):
        points_copy = points.copy()
        resampled_points = [] 
        resampled_points.append(points_copy[0])

        path_spacing = self.path_len(points_copy) / (n - 1)

        # Used in case dist between curr point and last < path spacing
        accumulated_dist = 0

        i = 1
        while i < len(points_copy) and path_spacing > 0:
            curr_point = np.array(points_copy[i])
            last_point = np.array(points_copy[i - 1])
            curr_dist = np.linalg.norm(curr_point - last_point)

            if accumulated_dist + curr_dist >= path_spacing:
                curr_diff_vec = curr_point - last_point
                if curr_dist != 0:
                    next_pnt_factor = (path_spacing - accumulated_dist) / curr_dist
                else:
                    next_pnt_factor = 0.5

                resampled_point = points_copy[i - 1] + next_pnt_factor * curr_diff_vec
                resampled_points.append(resampled_point)
                points_copy.insert(i, resampled_point)
                accumulated_dist = 0
            else:   
                accumulated_dist += curr_dist

            i += 1

        while len(resampled_points) < n:
            resampled_points.append(points_copy[-1])

        return resampled_points

    def path_len(self, points):
        length = 0

        for i in range(1, len(points)):
            curr_point = np.array(points[i])
            last_point = np.array(points[i - 1])
            length += np.linalg.norm(curr_point - last_point)

        return length
    
    def dtw(self, template_gpdvs, query_gpdvs):
        n = len(template_gpdvs) + 1
        m = len(query_gpdvs) + 1
        r = int(np.ceil(0.1 * n)) # Set Sakoe-Chiba band radius to 10% of length

        cost_matrix = np.empty((n, m))

        cost_matrix[:, 0] = np.inf
        cost_matrix[0, :] = np.inf
        cost_matrix[0, 0] = 0

        for i in range(1, n):
            for j in range(max(1, i - r), min(m, i + r), 1):
                cost = self.local_cost(template_gpdvs[i - 1], query_gpdvs[j - 1])
                cost += np.min([ cost_matrix[i - 1][j - 1],
                                 cost_matrix[i - 1][j],
                                 cost_matrix[i][j - 1] ])
                cost_matrix[i][j] = cost

        return cost_matrix[n - 1][m - 1]

    def local_cost(self, template_gpdv, query_gpdv):
        return 1 - np.inner(template_gpdv, query_gpdv)
    
    # Convert n points to n-1 gesture path direction (unit) vectors
    def to_gpdvs(self, points):
        # Convert to numpy array for ease of vector operations
        np_points = np.array(points)
        gpdvs = []

        for i in range(len(np_points) - 1):
            diff_vec = np_points[i + 1] - np_points[i]
            diff_vec_norm = np.linalg.norm(diff_vec)

            # Handles division by 0
            if diff_vec_norm != 0:
                gpdv = diff_vec / diff_vec_norm
            else:
                gpdv = diff_vec

            gpdvs.append(gpdv)

        return gpdvs