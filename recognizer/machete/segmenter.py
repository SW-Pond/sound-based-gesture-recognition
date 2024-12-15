import numpy as np
from .template import Template
from collections import deque
from ..logger import Logger


class Segmenter:
    def __init__(self):
        THETA_D = 75
        THETA_R = THETA_D * np.pi / 180
        EPSILON = 0.01
        self.MAX_HISTORY_FRAMES = 100
        self.frame_hist = deque(maxlen=self.MAX_HISTORY_FRAMES)
        raw_templates = Logger.get_all_templates()

        # Create Machete Template objects from raw frames
        self.templates = []
        for raw_t in raw_templates:
            template = Template(name=raw_t[0], sample_pts=raw_t[1],
                                theta=THETA_R, epsilon=EPSILON)
            self.templates.append(template)

    def consume_input(self, template, x):
        """
        Returns the segmented frames if gesture end points have been detected,
        otherwise, None.
        """

        # Only add a new frame to history once
        if len(self.frame_hist) == 0:
            self.frame_hist.append(x)
        elif x is not self.frame_hist[len(self.frame_hist) - 1]:
            self.frame_hist.append(x)

        if len(template.prev) != 0:
            # Convert to direction vector and normalize
            x_vec = x - template.prev
            length = np.linalg.norm(x_vec)
            # Handles division by 0
            if length == 0:
                # Set an element to a non-zero value so no vector has only zeros
                x_vec[0] = 0.00001
                length = 0.00001
            x_vec /= length
            template.prev = x
        # Can't make a direction vector on first frame since two are needed
        else:
            template.prev = x
            template.row[0][0].score = 0
            
            return

        # Store two rows of matrix data, accessed as a circular buffer
        prev_row = template.row[template.curr_row_idx]
        template.curr_row_idx = (template.curr_row_idx + 1) % 2
        curr_row = template.row[template.curr_row_idx]

        # Update current row with new input
        T = template.T
        T_N = len(T)

        for col in range(1, T_N + 1):
            curr_elem = curr_row[col]

            # Determine which one of the three paths to extend
            best = curr_row[col - 1]
            path2 = prev_row[col - 1]
            path3 = prev_row[col]

            if path2.score <= best.score: best = path2
            if path3.score <= best.score: best = path3

            curr_elem.gesture_frame_len = best.gesture_frame_len
            # If gesture is extended from last frame, accumulate frame length
            curr_elem.gesture_frame_len = best.gesture_frame_len
            if best == path2 or best == path3:
                curr_elem.gesture_frame_len += 1

            # Limit size of gesture if gesture has been extended for a while
            # without a match
            if curr_elem.gesture_frame_len > self.MAX_HISTORY_FRAMES:
                curr_elem.gesture_frame_len = self.MAX_HISTORY_FRAMES
            
            # Extend selected path through current column
            local_cost = (1 - np.inner(x_vec, T[col - 1])) ** 2
            curr_elem.cumulative_length = best.cumulative_length + length
            curr_elem.score = local_cost + best.score
        
        # Change element score at [0][0] to default first column value
        # after first full iteration since row is reused
        if prev_row[0].score == 0:
            prev_row[0].score = template.first_col_val
        
        # Determine if the underlying recognizer should be called
        template.total += curr_row[T_N].score
        template.n += 1
        template.s1 = template.s2
        template.s2 = template.s3
        template.s3 = curr_row[T_N].score

        # If new low, save segmentation information
        if template.s3 < template.s2:
            template.gesture_frame_len = curr_row[T_N].gesture_frame_len
            # Not a gesture end point
            return None
        
        # If previous frame is a minimum below the threshold, trigger check
        mean = template.total / (2 * template.n)
        
        if template.s2 < mean and template.s2 < template.s1 and template.s2 < template.s3:
            # Does not include the current frame
            start_frame_idx = len(self.frame_hist) - template.gesture_frame_len - 1
            return list(self.frame_hist)[start_frame_idx : -1]
        else:
            return None