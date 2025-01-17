import csv
import os


# Key-gesture associations for logging
GESTURE_TYPES = {'1':"star_y(1)", '2':"x_w_l(2)", '3':"z_triangle(3)"}

TEMPLATES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")


class Logger:
    """
    For creating/logging new templates and getting raw template frames for 
    Jackknife Classifier and Machete Segmenter.
    """
    
    def __init__(self):
        self.template_frames = []

    def record_frame(self, frame):
        num_template_frames = len(self.template_frames)
        if num_template_frames == 0 or frame is not self.template_frames[-1]:
            print(f"Recording frame {num_template_frames + 1}")
            self.template_frames.append(frame)

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

        for template_file in os.listdir(TEMPLATES_DIR):
            template_path = os.path.join(TEMPLATES_DIR, template_file)
            template_frames = []
            template_name = template_file[:-4] # Don't include .csv
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