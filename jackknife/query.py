import numpy as np
import csv


class Query:
    def __init__(self):
        self.points = []

    def add_point(self, point):
        self.points.append(point)

class QueryMgr:
    def __init__(self):
        self.DEFAULT_DIR = "jackknife\\templates\\"
        # Associate numeric key presses with gesture types for template logging
        self.GESTURE_TYPES = {'1':"zigzag", '2':"triangle", '3':"rectangle", '4':"x", '5':"c", '6':"arrow"}
        self.curr_query = Query()
        self.curr_point = []
        self.template_points = []
    
    def pass_point(self, point, last_in_curr_query):
        self.curr_query.add_point(np.copy(point))
        self.curr_point = point

        # This is for processing queries later.
        # Need to implement logic for determining start and end of a query;
        #   for now, creates new query once curr_query has 30 points
        if last_in_curr_query or len(self.curr_query.points) >= 30:
            self.curr_query = Query()

    def add_to_template(self):
        self.template_points.append(self.curr_point)
        print(f"Adding current point to template; # of template points: {len(self.template_points)}\n")

    def check_pressed_key(self, key_event):
        key = key_event.name

        if key == 'c':
            self.clear_template()
        
        if key in self.GESTURE_TYPES.keys():
            self.log_template(key)

    def log_template(self, key):
        gesture_type = self.GESTURE_TYPES[key]

        dir = self.DEFAULT_DIR + self.GESTURE_TYPES[key] + "\\"

        for log_file_num in range(3):
            log_file_path = dir + "t" + str(log_file_num) + ".csv"

            with open(log_file_path, "r+", newline='') as log_file:
                # If file is empty
                if log_file.read(1) == '':
                    print(f"Logging template {log_file_num + 1} for gesture: {gesture_type} ...")
                    
                    writer = csv.writer(log_file,)

                    for point in self.template_points:
                        writer.writerow(point)

                    log_file.close()

                    print("Successfully logged template")

                    break

                else:
                    if log_file_num == 2:
                        print(f"Three templates have already been logged for gesture: {gesture_type}")
                        
                    log_file.close()
                    
        self.clear_template()

    def clear_template(self):
            self.template_points = []
            print("Template points cleared\n")