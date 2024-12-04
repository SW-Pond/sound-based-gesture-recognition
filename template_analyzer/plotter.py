import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
from matplotlib.animation import FuncAnimation
from pathlib import Path
import os
import csv
import keyboard
import sys



class SpectrumPlotter:
    def __init__(self):
        self.fig = plt.figure(figsize=(7,6))

        LOW_FREQ = 18000
        HIGH_FREQ = 18500
        IN_BUFFER_SIZE = 2048
        SAMPLE_RATE = 44100
        FREQ_BIN_RES = SAMPLE_RATE / (IN_BUFFER_SIZE * 2)
        PEAK_MARGIN_F = 250

        self.MIN_PLOT_F = LOW_FREQ - PEAK_MARGIN_F
        self.MAX_PLOT_F = HIGH_FREQ + PEAK_MARGIN_F
        MIN_F_IDX = int(np.floor(self.MIN_PLOT_F / FREQ_BIN_RES))
        MAX_F_IDX = int(np.ceil(self.MAX_PLOT_F / FREQ_BIN_RES))
        self.TEMPLATES_DIR = Path(__file__).parent.absolute().joinpath("templates")

        freq_spec = np.linspace(0, SAMPLE_RATE / 2, IN_BUFFER_SIZE + 1)
        self.freqs = freq_spec[MIN_F_IDX : MAX_F_IDX + 1]
        self.templates = self.get_all_templates()

    def get_all_templates(self):
        templates = []

        for template_file in os.listdir(self.TEMPLATES_DIR):
            template_path = os.path.join(self.TEMPLATES_DIR, template_file)
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

    def plot_templates(self):
        f_spec_line = self.init_f_spec_plot()
        frame_num = 0
        template_num = -1
        template = None
        name = None
        frames = None
        zero_frames = np.zeros((4,95))
        zero_frame_num = 0

        def next_frame(i):
            nonlocal frame_num, template_num, template, name, frames, zero_frames, zero_frame_num
            keyboard.wait('space')
            if zero_frame_num < len(zero_frames):
                f_spec_line.set_data(self.freqs, zero_frames[zero_frame_num])
                zero_frame_num += 1
                return [f_spec_line]
            if frame_num == len(self.templates[template_num][1]) or \
               (template_num == -1 and frame_num == 0):
                template_num += 1
                if template_num == len(self.templates):
                    sys.exit(0)
                frame_num = 0
                template = self.templates[template_num]
                name = template[0]
                frames = template[1]
                print(f"template name: {name}")

            print(f"frame num: {frame_num + 1}")
            frame = frames[frame_num]
            first_half = frame[0:46]
            second_half = frame[46:90]
            for j in range(2):
                first_half.insert(0,0)
                second_half.append(0)
            for j in range(1):
                first_half.append(0)
            frame = np.concatenate((first_half, second_half))
            f_spec_line.set_data(self.freqs, frame)
            frame_num += 1

            if frame_num == len(frames):
                print(f"end of template {name}")

            return [f_spec_line]

        animated_plot = FuncAnimation(fig=self.fig, func=next_frame, 
                                interval=1, blit=True, 
                                cache_frame_data=False)
        plt.show()

    def init_f_spec_plot(self):
        ax = self.fig.add_subplot(xlim=(self.MIN_PLOT_F, self.MAX_PLOT_F), 
                                  ylim=(0, 1.2))
        ax.set_title("Frequency Spectrum")
        ax.xaxis.set_minor_locator(MultipleLocator(25))
        ax.xaxis.set_major_locator(MultipleLocator(250))
        ax.xaxis.set_tick_params(which='minor', length=3, width=0.75)
        ax.xaxis.set_tick_params(which='major', length=6, width=1)
        ax.yaxis.set_minor_locator(MultipleLocator(1))
        ax.yaxis.set_major_locator(MultipleLocator(10))
        ax.yaxis.set_tick_params(which='minor', length=3, width=0.3)
        ax.yaxis.set_tick_params(which='major', length=6, width=0.5)
        ax.set_xlabel("Frequency(Hz)", size=10)
        ax.set_ylabel("Amplitude", size=10)
        ax.grid(linestyle='-', linewidth=0.5)
        ax.axhline(0, color='red', linestyle='-', lw=1)
        line, = ax.plot([], [])
        line.set_linewidth(1.25)

        return line
    
if __name__ == "__main__":
    s_p = SpectrumPlotter()
    s_p.plot_templates()