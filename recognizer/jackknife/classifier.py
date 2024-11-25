import random as rand
import numpy as np
from . import gestures as g
from ..logger import Logger


# gpdvs := gesture path direction vectors
class Classifier:
    def __init__(self):
        raw_templates = Logger.get_all_templates()

        # Create Jackknife Template objects from raw frames
        self.templates = {} # For direct access in is_match() using names
        for raw_t in raw_templates:
            template = g.Template(name=raw_t[0], points=raw_t[1])
            self.templates[template.name] = template # Add to dict

        self.train()
    
    def train(self):
        """
        Generate per-template rejection thresholds using synthetic samples
        """

        NUM_SYNTHETIC_SAMPLES = 1000
        BIN_COUNT = 1000
        BETA = 1
        SAMPLES_TO_SPLICE = 2 # For creating negative samples
        template_list = list(self.templates.values()) # Convert to list for indexing
        template_cnt = len(template_list)
        distributions = []
        worst_score = 0

        # Generate negative samples
        for i in range(NUM_SYNTHETIC_SAMPLES):
            neg_sample = g.Gesture()

            # Randomly grab templates and splice them together
            for j in range(SAMPLES_TO_SPLICE):
                t = rand.randint(0, template_cnt - 1)
                rand_template = template_list[t]

                template_len = len(rand_template.points)
                point_start_idx = rand.randint(0, template_len // SAMPLES_TO_SPLICE)

                for k in range(template_len // SAMPLES_TO_SPLICE):
                    neg_sample.add_point(rand_template.points[point_start_idx + k])

            neg_sample.resample_points()
            neg_sample.populate_gpdvs()

            for t in range(template_cnt):
                template = template_list[t]

                score = self.dtw(template, neg_sample)

                if worst_score < score:
                    worst_score = score

                if i > 50:
                    distributions[t].add_negative_score(score)
            
            if i != 50:
                continue

            for t in range(template_cnt):
                distributions.append(Distributions(worst_score, BIN_COUNT))

        # Generate positive samples
        for t in range(template_cnt):
            for i in range(NUM_SYNTHETIC_SAMPLES):
                pos_sample = g.Gesture()
                template = template_list[t]

                for point in template.points:
                    pos_sample.add_point(point)

                self.gpsr(pos_sample)
                pos_sample.resample_points()
                pos_sample.populate_gpdvs()

                score = self.dtw(template, pos_sample)    

                distributions[t].add_positive_score(score)   

        # Get rejection thresholds
        for t in range(template_cnt):
            template = template_list[t]
            threshold = distributions[t].rejection_threshold(BETA)
            template.rejection_threshold = threshold
            print(f"gesture: {template.name}; threshold: {template.rejection_threshold}")
    
    def gpsr(self, gesture):
        N = 6
        POINTS_TO_REMOVE = 2
        VARIANCE = 0.25

        gesture.resample_points(N + POINTS_TO_REMOVE, VARIANCE)

        for i in range(POINTS_TO_REMOVE):
            remove_idx = rand.randint(0, N + POINTS_TO_REMOVE - i - 1)
            gesture.points.pop(remove_idx)

    def is_match(self, trajectory, gesture_name):
        match = False

        template = self.templates[gesture_name]

        query = g.Query()
        query.points = trajectory
        
        query.resample_points()
        query.populate_gpdvs()
        query.extract_features()

        score = 1
        correction_factors = self.correction_factors(template, query)
        for factor in correction_factors:
            score *= factor
        
        score *= self.dtw(template, query)
        
        if score < template.rejection_threshold:
            match = True
        ###############################
        print()
        print(f"gesture: {template.name}; score: {score}; threshold: {template.rejection_threshold}")
        ###############################
        return match, score
    
    # r := Sakoe-Chiba band radius
    def dtw(self, template, query, r=2):
        n = len(template.gpdvs) + 1
        m = len(query.gpdvs) + 1

        cost_matrix = np.full((n, m), np.inf)

        cost_matrix[0, 0] = 0

        for i in range(1, n):
            for j in range(max(1, i - r), min(m, i + r + 1)):
                cost = self.local_cost(template.gpdvs[i - 1], query.gpdvs[j - 1])

                cost += np.min([ cost_matrix[i - 1][j - 1],
                                 cost_matrix[i - 1][j],
                                 cost_matrix[i][j - 1] ])

                cost_matrix[i][j] = cost
        
        #self.print_cost_matrix(cost_matrix)

        return cost_matrix[n - 1][m - 1]

    def local_cost(self, template_vec, query_vec):
        return 1 - np.inner(template_vec, query_vec)
    
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


class Distributions:
    def __init__(self, max_score, bin_cnt):
        self.neg = np.zeros(bin_cnt)
        self.pos = np.zeros(bin_cnt)
        self.max_score = max_score

    def bin(self, score):
        val1 = int(score * len(self.neg) / self.max_score)
        val2 = len(self.neg) - 1
        return min(val1, val2)

    def add_negative_score(self, score):
        self.neg[self.bin(score)] += 1

    def add_positive_score(self, score):
        self.pos[self.bin(score)] += 1

    def rejection_threshold(self, beta):
        self.neg /= np.sum(self.neg)
        self.neg = np.cumsum(self.neg)
        assert (abs(self.neg[len(self.neg) - 1] - 1.0) < .00001)

        self.pos /= np.sum(self.pos)
        self.pos = np.cumsum(self.pos)
        assert (abs(self.pos[len(self.pos) - 1] - 1.0) < .00001)

        alpha = 1 / (1 + beta ** 2)
        precision = self.pos / (self.pos + self.neg)
        recall = self.pos

        best_score = 0
        best_idx = -1

        for i in range(len(self.neg)):
            # Handles division by 0
            if precision[i] == 0 or recall[i] == 0:
                continue
            
            E = (alpha / precision[i]) + ((1 - alpha) / recall[i])
            f_score = 1 / E

            if f_score > best_score:
                best_score = f_score
                best_idx = i

        ret = best_idx + 0.5
        ret *= self.max_score / len(self.neg)

        return ret