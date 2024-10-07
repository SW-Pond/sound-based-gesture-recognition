import numpy as np
import time
import csv
from . import data_utils as d_u
from . import data


# gpdvs := gesture path direction vectors
class Classifier:
    def __init__(self, pipe_conn):
        # Number of points to add to the query for each dtw check against
        #  all templates
        self.pipe_conn = pipe_conn
        self.gesture_templates = self.get_all_templates()

        for gest_type in self.gesture_templates:
            for template in gest_type:
                template.points = d_u.resample(template.points)
                template.gpdvs = d_u.to_gpdvs(template.points)
                template.envelop()
                template.extract_features()

    def get_all_templates(self):
        templates = []

        for gesture_type in d_u.GESTURE_TYPES.values():
            dir = f"{d_u.PARENT_DIR}{gesture_type}\\"
            curr_gest_templates = []

            for template_num in range(d_u.TEMPLATES_PER_GESTURE):
                curr_template = data.Template(name=gesture_type)

                template_path = f"{dir}t{template_num}.csv"

                with open(template_path, "r") as template_file:
                    temp_file_reader = csv.reader(template_file)
                    for line in temp_file_reader:
                        curr_template.add_point([float(val) for val in line])

                    template_file.close()

                curr_gest_templates.append(curr_template)  
            templates.append(curr_gest_templates)
        
        return templates   
    
    def classify(self):
        best_score = np.inf
        best_match = None
        prev_best_match = None
        prev2_best_match = None

        while True:
            """
            When ready to process a new query, notify data manager and block 
            until a list of points is received.
            """
            self.pipe_conn.send(1)
            recvd_points = self.pipe_conn.recv()

            query = data.Query()
            query.points = recvd_points
                
            query.points = d_u.resample(query.points)
            query.gpdvs = d_u.to_gpdvs(query.points)
            query.extract_features()

            classified = False
            for i in range(1):    
            #for i in range(d_u.TEMPLATES_PER_GESTURE):
                for j in range(d_u.NUM_GESTURES):
                    template = self.gesture_templates[j][i]
                    gesture_name = template.name

                    score = 1
                    correction_factors = self.correction_factors(template,
                                                                 query)
                    for factor in correction_factors:
                        score *= factor
                    """
                    # Skip DTW check if last best score is lower than
                    # best possible score for query and current template
                    lower_bound = self.lower_bound(template, query) * score
                    if best_score < lower_bound:
                        continue
                    """
                    score *= self.dtw(template.gpdvs, query.gpdvs)
                    
                    # Using argmin of DTW for best gesture match
                    if score < best_score:
                        best_score = score
                        best_match = gesture_name

                        if best_match == prev_best_match and \
                           prev_best_match == prev2_best_match:
                            classified = True
                            break

                if classified:
                    print(best_match)
                    best_match = None
                    prev_best_match = None
                    prev2_best_match = None
                    time.sleep(2)
                    # Notify data manager that a gesture has been detected
                    self.pipe_conn.send(2)
                    break
        
            prev2_best_match = prev_best_match
            prev_best_match = best_match

            # Reset for next query
            best_score = np.inf

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
    
    def dtw(self, template_gpdvs, query_gpdvs):
        n = len(template_gpdvs) + 1
        m = len(query_gpdvs) + 1

        cost_matrix = np.empty((n, m))

        cost_matrix[:, 0] = np.inf
        cost_matrix[0, :] = np.inf
        cost_matrix[0, 0] = 0

        for i in range(1, n):
            for j in range(max(1, i - d_u.R), min(m, i + d_u.R), 1):
                cost = self.local_cost(template_gpdvs[i - 1], query_gpdvs[j - 1])
                cost += np.min([ cost_matrix[i - 1][j - 1],
                                 cost_matrix[i - 1][j],
                                 cost_matrix[i][j - 1] ])
                cost_matrix[i][j] = cost

        return cost_matrix[n - 1][m - 1]

    def local_cost(self, template_gpdv, query_gpdv):
        return 1 - np.inner(template_gpdv, query_gpdv)
    
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