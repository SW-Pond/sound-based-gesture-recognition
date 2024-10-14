import numpy as np
import time
from collections import deque
from . import gestures as g


class Manager:
    def __init__(self, pipe_conn, res_q):
        self.pipe_conn = pipe_conn

        self.res_q = res_q

        # For communicating with classifier through pipe;
        # FLAG_DEFAULT ==> do nothing
        self.FLAG_DEFAULT = 0
        self.flag = self.FLAG_DEFAULT

        self.MAX_HISTORY_POINTS = 25
        self.point_history = deque(maxlen=self.MAX_HISTORY_POINTS)

        # For template logging only
        self.curr_template = g.Template()
        self.curr_point = []

        # For these, first element is the score, second is the gesture name
        self.scnd_last_best_match = None
        self.last_best_match = None
        self.best_match = None

        self.BETWEEN_GESTURE_DELAY = 2 # Seconds to wait between gestures
        self.timerStart = 0

    def process_point(self, point):
        self.curr_point = np.copy(point)
        self.point_history.append(point)

        # Check status of classifier
        if self.pipe_conn.poll():
            self.flag = self.pipe_conn.recv()

        # If classifier is done
        if self.flag == 2:
            match = self.pipe_conn.recv()
            
            if self.best_match == None:
                self.best_match = match

            elif match[0] < self.best_match[0]:
                self.best_match = match

            if self.last_best_match != None and \
               self.scnd_last_best_match != None:

                # If a match has persisted over three frames
                if self.best_match[1] == self.last_best_match[1] and \
                   self.last_best_match[1] == self.scnd_last_best_match[1]:
                    
                    self.res_q.put(self.best_match)

                    self.scnd_last_best_match = None
                    self.last_best_match = None
                    self.best_match = None

                    self.point_history.clear()

                    self.timerStart = time.time()

                    # Do not send current points to classifier
                    self.flag = self.FLAG_DEFAULT
                
                # If a match has not persisted over three frames, send
                # current points to classifier
                else:
                    # Since last flag was 2 (classifier is done with last
                    # points), recv() will not block and this flag will always
                    # be 1
                    self.flag = self.pipe_conn.recv()
                    
            self.scnd_last_best_match = self.last_best_match
            self.last_best_match = self.best_match
            self.best_match = None

        # If classifier is ready, the minimum delay between gestures has 
        # passed, and there are enough points in history
        if self.flag == 1 and \
           time.time() - self.timerStart >= self.BETWEEN_GESTURE_DELAY and \
           len(self.point_history) == self.MAX_HISTORY_POINTS:
            self.pipe_conn.send(self.point_history)

            self.flag = self.FLAG_DEFAULT

    def check_pressed_key(self, key_event):
        key = key_event.name

        if key == 'r':
            self.reset_curr_template()

        if key == 't':
            self.curr_template.record_point(self.curr_point)
        
        if key in g.GESTURE_TYPES.keys():
            self.curr_template.log(g_key=key)
            self.reset_curr_template()

    def reset_curr_template(self):
        self.curr_template = g.Template()
        print("Current template reset\n")