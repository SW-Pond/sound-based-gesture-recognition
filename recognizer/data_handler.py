import pickle
import os
from .jackknife.classifier import Classifier
from .machete.segmenter import Segmenter


class DataHandler:
    def __init__(self, data_q, res_q):
        self.data_q = data_q
        self.res_q = res_q

        # Get Classifier and Segmenter if already pickled; if not, initialize.
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
        if False:#os.path.exists(segmenter_path):
            with open(segmenter_path, 'rb') as f:
                self.segmenter = pickle.load(f)
        else:
            self.segmenter = Segmenter()
            with open(segmenter_path, 'wb') as f:
                pickle.dump(self.segmenter, f)

    def process_data(self):
        while True:
            # Grab the latest frame
            frame = None
            if self.data_q.empty():
                continue
            else:
                #################################
                if self.data_q.qsize() > 1:
                    for i in range(10):
                        print(f"Queue length: {self.data_q.qsize()}")
                #################################
                while not self.data_q.empty():
                    frame = self.data_q.get()

            for template in self.segmenter.templates:
                trajectory = self.segmenter.consume_input(template, frame)

                if trajectory == None or len(trajectory) == 0:
                    continue
                else:
                    match, score = self.classifier.is_match(trajectory, template.name)
                    if match:
                        self.res_q.put((template.name, score))
                        template.reset()