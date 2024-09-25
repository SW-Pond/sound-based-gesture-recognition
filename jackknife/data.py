import numpy as np
import csv
from collections import deque


PARENT_DIR = "jackknife\\templates\\"
# Key-gesture associations
GESTURE_TYPES = {'1':"zigzag", '2':"triangle", '3':"rectangle", 
                        '4':"x", '5':"c", '6':"arrow"}
TEMPLATES_PER_GESTURE = 3

def get_all_templates():
    templates = []

    for gesture_type in GESTURE_TYPES.values():
        dir = PARENT_DIR + gesture_type + "\\"
        curr_gest_templates = []

        for template_num in range(TEMPLATES_PER_GESTURE):
            curr_template = Template(name=gesture_type)

            template_path = dir + "t" + str(template_num) + ".csv"

            with open(template_path, "r") as template_file:
                temp_file_reader = csv.reader(template_file)
                for line in temp_file_reader:
                    curr_template.add_point([float(val) for val in line])

                template_file.close()
            
            curr_gest_templates.append(curr_template)  
        templates.append(curr_gest_templates)
    
    return templates


class Query:
    def __init__(self):
        self.points = []

    def add_point(self, point):
        self.points.append(point)


class Template(Query):
    def __init__(self, name=""):
        super().__init__()
        self.name = name
    
    def record_point(self, point):
        if len(self.points) == 0 or point is not self.points[-1]:
            self.add_point(point)

    def log(self, g_key):
        gesture_type = GESTURE_TYPES[g_key]

        dir = PARENT_DIR + gesture_type + "\\"

        for log_file_num in range(TEMPLATES_PER_GESTURE):
            log_file_path = dir + "t" + str(log_file_num) + ".csv"

            with open(log_file_path, "r+", newline='') as log_file:
                # If file is empty
                if log_file.read(1) == '':
                    print(f"Logging template {log_file_num + 1} for gesture: {gesture_type} ...")
                    
                    writer = csv.writer(log_file,)

                    for point in self.points:
                        writer.writerow(point)

                    log_file.close()

                    print("Successfully logged template")

                    break

                else:
                    if log_file_num == 2:
                        print(f"Three templates have already been logged for gesture: {gesture_type}")
                        
                    log_file.close()


class Manager:
    def __init__(self, pipe_conn):
        self.pipe_conn = pipe_conn
        self.curr_template = Template()
        self.curr_point = []
        self.MAX_HISTORY_POINTS = 25
        self.point_history = deque(maxlen=self.MAX_HISTORY_POINTS)
    
    def pass_point(self, point):
        self.curr_point = np.copy(point)
        self.point_history.append(point)

        if self.pipe_conn.poll():
            if (self.pipe_conn.recv() == 1 and 
                len(self.point_history) == self.MAX_HISTORY_POINTS):
                # Send MAX_HISTORY_POINTS so classifier knows how many points to expect
                self.pipe_conn.send(self.MAX_HISTORY_POINTS)
                for i in range(self.MAX_HISTORY_POINTS):
                    latest_point = self.point_history.pop()
                    self.pipe_conn.send(latest_point)
            else:
                # Not ready to send points
                self.pipe_conn.send(0)


    def check_pressed_key(self, key_event):
        key = key_event.name

        if key == 'r':
            self.reset_curr_template()

        if key == 't':
            self.curr_template.record_point(self.curr_point)
        
        if key in GESTURE_TYPES.keys():
            self.curr_template.log(g_key=key)
            self.reset_curr_template()

    def reset_curr_template(self):
        self.curr_template = Template()
        print("Current template reset\n")