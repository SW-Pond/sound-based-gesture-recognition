import pickle
import os
from .jackknife.classifier import Classifier
from .machete.segmenter import Segmenter
from collections import deque


class DataHandler:
    def __init__(self, data_q, res_q):
        self.data_q = data_q
        self.res_q = res_q

        MAX_HISTORY_FRAMES = 50
        self.frame_history = deque(maxlen=MAX_HISTORY_FRAMES)

        """
        Get Classifier and Segmenter if already pickled; if not, initialize.
        """
        self.classifier = None
        self.segmenter = None
        objects_dir = os.path.join("recognizer", "objects")
        classifier_path = os.path.join(objects_dir, "classifier.pkl")
        segmenter_path = os.path.join(objects_dir, "segmenter.pkl")
        if os.path.exists(classifier_path):
            with open(classifier_path, 'rb') as f:
                self.classifier = pickle.load(f)
        else:
            self.classifier = Classifier()
            with open(classifier_path, 'wb') as f:
                pickle.dump(self.classifier, f)
        if os.path.exists(segmenter_path):
            with open(segmenter_path, 'rb') as f:
                self.segmenter = pickle.load(f)
        else:
            self.segmenter = Segmenter()
            with open(segmenter_path, 'wb') as f:
                pickle.dump(self.segmenter, f)

    def process_data(self):
        frame_num = -1

        while True:
            """
            Grab the latest input frame in case frames are created faster than
            they can be processed. If no frames are available (likely only on
            startup), keep looping until there are.
            """
            frame = None
            if self.data_q.empty():
                continue
            else:
                while not self.data_q.empty():
                    frame = self.data_q.get()

            frame_num += 1

            for template in self.segmenter.templates:
                self.segmenter.consume_input(template, frame, frame_num)
                if template.do_check:
                    start_idx = len(self.frame_history) - \
                                   (template.end_frame - template.start_frame)
                    end_idx = len(self.frame_history)
                    trajectory = list(self.frame_history)[start_idx:end_idx]

                    match, score = self.classifier(trajectory, template.name)
                    if match:
                        self.res_q.put((template.name, score))
                        template.reset()

            self.frame_history.append(frame)