import numpy as np
import csv
from . import gestures as g


# gpdvs := gesture path direction vectors
class Classifier:
    def __init__(self, pipe_conn):
        self.pipe_conn = pipe_conn
        self.gesture_templates = self.get_all_templates()

        for gesture_type in self.gesture_templates:
            for template in gesture_type:
                template.resample_points()
                template.populate_gpdvs()
                template.envelop()
                template.extract_features()

    def get_all_templates(self):
        templates = []

        for gesture_type in g.GESTURE_TYPES.values():
            dir = f"{g.TEMPLATES_DIR}{gesture_type}\\"
            curr_gest_templates = []

            for template_num in range(g.TEMPLATES_PER_GESTURE):
                curr_template = g.Template(name=gesture_type)
                template_path = f"{dir}t{template_num}.csv"
                template_file_empty = True # Assume the file is empty

                with open(template_path, "r") as template_file:
                    temp_file_reader = csv.reader(template_file)

                    for line in temp_file_reader:
                        if line: # If there is a non-empty line
                            template_file_empty = False
                            curr_template.add_point([float(val) for val in line])

                    template_file.close()

                if not template_file_empty:
                    curr_gest_templates.append(curr_template)

            # Only use gestures that have at least one recorded template
            if len(curr_gest_templates) > 0:
                templates.append(curr_gest_templates)
        
        return templates
    
    def classify(self):
        INIT_WINDOW_SIZE = 5
        WINDOW_INCREMENT = 5
        
        while True:
            best_score = np.inf
            best_match_name = None

            """
            When ready to process new points, notify data manager and block 
            until points are received.
            """
            self.pipe_conn.send(1)
            recvd_points = self.pipe_conn.recv()

            window_end = len(recvd_points)

            for window_size in range(INIT_WINDOW_SIZE, window_end + 1, 
                                     WINDOW_INCREMENT):
                window_points = []
                for i in range(window_end - window_size, window_end):
                    window_points.append(recvd_points[i])

                query = g.Query()
                query.points = window_points
                    
                query.resample_points()
                query.populate_gpdvs()
                query.extract_features()
   
                for gesture_type in self.gesture_templates:
                    for template in gesture_type:
                        gesture_name = template.name

                        score = 1
                        correction_factors = self.correction_factors(template,
                                                                    query)
                        for factor in correction_factors:
                            score *= factor
                        
                        # Skip DTW check if last best score is lower than
                        # best possible score for query and current template
                        lower_bound = self.lower_bound(template, query) * score
                        if best_score < lower_bound:
                            continue
                        
                        score *= self.dtw(template, query)

                        # Using argmin of DTW for best gesture match
                        if score < best_score:
                            best_score = score
                            best_match_name = gesture_name

            best_match = [best_score, best_match_name]

            # Notify data manager that classification is done and send match
            self.pipe_conn.send(2)
            self.pipe_conn.send(best_match)
    
    def dtw(self, template, query):
        n = len(template.gpdvs) + 1
        m = len(query.gpdvs) + 1

        cost_matrix = np.full((n, m), np.inf)

        cost_matrix[0, 0] = 0

        for i in range(1, n):
            for j in range(max(1, i - g.Radius), min(m, i + g.Radius + 1)):
                cost = self.local_cost(template.gpdvs[i - 1], query.gpdvs[j - 1])

                cost += np.min([ cost_matrix[i - 1][j - 1],
                                 cost_matrix[i - 1][j],
                                 cost_matrix[i][j - 1] ])

                cost_matrix[i][j] = cost
        
        #self.print_cost_matrix(cost_matrix)

        return cost_matrix[n - 1][m - 1]

    def local_cost(self, template_vec, query_vec):
        return 1 - np.inner(template_vec, query_vec)
    
    def lower_bound(self, template, query):
        num_vecs = len(template.gpdvs)
        components_per_vec = len(template.gpdvs[0])
        lb = 0

        for i in range(num_vecs):
            lb_inner_product = 0
            for j in range(components_per_vec):
                if query.gpdvs[i][j] >= 0:
                    lb_inner_product += template.upper[i][j] * query.gpdvs[i][j]
                else:
                    lb_inner_product += template.lower[i][j] * query.gpdvs[i][j]

            lb += 1 - min(1, max(-1, lb_inner_product))

        return lb
    
    def correction_factors(self, template, query):
        num_feature_vecs = len(template.features)
        correction_factors = []

        for i in range(num_feature_vecs):
            factor = 1
            feature_inner_product = np.inner(template.features[i], 
                                             query.features[i])
            
            # Handles division by 0
            if feature_inner_product != 0:
                factor = 1 / feature_inner_product
        
            correction_factors.append(factor)

        return correction_factors
    
    # Costs are truncated to make matrix easily interpretable
    def print_cost_matrix(self, cost_matrix):
        for row in cost_matrix:
            for val in row:
                if val >= 10 and val != np.inf:
                    print(f"{int(val)}., " , end="")
                else:
                    print(f"{val:3.1f}, ", end="")
            print()
        print()
        print()