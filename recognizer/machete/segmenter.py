import numpy as np
from .template import Template
from ..logger import Logger


class Segmenter:
    def __init__(self):
        THETA = 40
        EPSILON = 0.01
        raw_templates = Logger.get_all_templates()

        # Create Machete Template objects from raw frames
        self.templates = []
        for raw_t in raw_templates:
            template = Template(name=raw_t[0], sample_pts=raw_t[1], 
                                theta=THETA, epsilon=EPSILON)
            self.templates.append(template)

    # Could be static, but leaving it for sake of clarity
    def consume_input(self, template, x, frame_number):
        # Add input to buffer
        template.buffer.append(x)

        length = np.linalg.norm(x - template.prev)

        # Convert to direction vector
        x_vec = (x - template.prev) / length
        template.prev = x

        # Store two rows of matrix data, accessed as a circular buffer
        prev_row = template.row[template.curr_row_idx]
        template.curr_row_idx = (template.curr_row_idx + 1) % 2
        curr_row = template.row[template.curr_row_idx]
        curr_row[0].start_frame = frame_number

        # Update current row with new input
        T = template.T
        T_N = len(T)

        for col in range(1, T_N + 1):
            # Determine which one of the three paths to extend
            best = curr_row[col - 1]
            path2 = prev_row[col - 1]
            path3 = prev_row[col]

            if path2.score <= best.score: best = path2
            if path3.score <= best.score: best = path3

            # Extend selected path through current column
            local_cost = length * (1 - np.inner(x_vec, T[col - 1])) ** 2
            curr_row[col].start_frame = best.start_frame
            curr_row[col].end_frame = frame_number
            curr_row[col].cumulative_cost = best.cumulative_cost + local_cost
            curr_row[col].cumulative_length = best.cumulative_length + length
            curr_row[col].score = curr_row[col].cumulative_cost / curr_row[col].cumulative_length

        cf = self.calculate_correction_factors(template, curr_row[T_N])
        corrected_score = cf * curr_row[T_N].score

        # Determine if the underlying recognizer should be called
        template.do_check = False
        template.total = template.total + curr_row[T_N].score
        template.n += 1
        template.s1 = template.s2
        template.s2 = template.s3
        template.s3 = corrected_score

        # If new low, save segmentation information
        if template.s3 < template.s2:
            template.start_frame = curr_row[T_N].start_frame
            template.end_frame = curr_row[T_N].end_frame
            return
        
        # If previous frame is a minimum below the threshold, trigger check
        mean = template.total / (2 * template.n)
        template.do_check = template.s2 < mean and \
                            template.s2 < template.s1 and \
                            template.s2 < template.s3
    
    def calculate_correction_factors(self, template, cdp_elem):
        # Calculate the first-to-last vector, then the closeness and first-to-last correction factors
        f2l = template.buffer[cdp_elem.end_frame] - template.buffer[cdp_elem.start_frame]
        f2l_length = np.linalg.norm(f2l)
        openness = f2l_length / cdp_elem.cumulative_length

        max_min_ratio = max(openness, template.openness) / min(openness, template.openness)
        cf_openness = 1 + template.w_closedness * (max_min_ratio - 1)
        cf_openness = min(2, cf_openness)

        cf_f2l = 1 + 0.5 * template.w_f2l * (1 - np.inner(f2l / f2l_length, template.f2l))
        cf_f2l = min(2, cf_f2l) 
        return cf_openness * cf_f2l