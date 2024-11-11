import csv
import os

# Key-gesture associations
GESTURE_TYPES = {'1':"zigzag", '2':"triangle", '3':"rectangle", '4':"x", 
                 '5':"c", '6':"arrow", '7':"check", '8':"caret", '9':"star", 
                 'a':"double arch", 's':"s", 'w':"w", 'y':"y", 'z':"z"}
TEMPLATES_DIR = os.path.join("recognizer", "templates")


"""
For creating/logging new templates and getting raw template frames for 
Jackknife Classifier and Machete Segmenter.
"""
class Logger:
    def __init__(self):
        self.template_frames = []

    def record_frame(self, frame):
        num_template_frames = len(self.template.points)
        if num_template_frames == 0 or frame is not self.template.points[-1]:
            print(f"Recording frame {num_template_frames + 1}")
            self.template.add_point(frame)

    def reset_template(self):
        self.template_frames = []
        print("Current template reset\n")
    
    def log(self, g_key):
        template_name = GESTURE_TYPES[g_key]
        template_file = f"{template_name}.csv"
        template_path = os.path.join(TEMPLATES_DIR, template_file)

        with open(template_path, "r+", newline='') as log_file:
            if log_file.read(1) == '':
                print(f"Logging template for gesture: {template_name} ...")
                
                writer = csv.writer(log_file,)

                for frame in self.template_frames:
                    writer.writerow(frame)

                log_file.close()

                print("Successfully logged template")

            else:
                print(f"Template has already been logged for gesture: "
                      f"{template_name}")
                    
                log_file.close()

    @staticmethod
    def get_all_templates():
        templates = []

        for gesture_type in GESTURE_TYPES.values():
            template_frames = []
            template_name = gesture_type
            template_file = f"{template_name}.csv"
            template_path = os.path.join(TEMPLATES_DIR, template_file)
            template_file_empty = True # Assume the file is empty

            with open(template_path, "r") as template_file:
                temp_file_reader = csv.reader(template_file)

                for line in temp_file_reader:
                    if line: # If there is a non-empty line
                        template_file_empty = False
                        template_frames.append([float(val) for val in line])

                template_file.close()

            if template_file_empty:
                print(f"Empty template file for gesture {template_name}")
            else:
                templates.append((template_name, template_frames))
        
        return templates