import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import MultipleLocator

class Plotter:
    def __init__(self, in_plot_q, v_plot_q, low_freq, high_freq, f_bin_res):
        self.in_plot_q = in_plot_q
        self.v_plot_q = v_plot_q
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.f_bin_res = f_bin_res
        self.containing_fig = plt.figure(figsize=(12, 5))
        self.PEAK_MARGIN_F = 250
        self.MIN_PLOT_F = self.low_freq - self.PEAK_MARGIN_F
        self.MAX_PLOT_F = self.high_freq + self.PEAK_MARGIN_F
        self.MIN_F_IDX = int(np.floor(self.MIN_PLOT_F / self.f_bin_res))
        self.MAX_F_IDX = int(np.ceil(self.MAX_PLOT_F / self.f_bin_res))

    def plot_all(self):
        f_spec_line = self.init_f_spec_plot()

        v_line = self.init_v_plot()
        # Start velocity line at origin
        v_x_vals = [0]
        v_y_vals = [0]

        def update_plot(i):
            if not self.in_plot_q.empty():
                data = self.in_plot_q.get()
                freqs = data[0][self.MIN_F_IDX : self.MAX_F_IDX + 1]
                amps = data[1][self.MIN_F_IDX : self.MAX_F_IDX + 1]

                f_spec_line.set_data(freqs, amps)

            if not self.v_plot_q.empty():
                velocity = self.v_plot_q.get()
                v_x = velocity[0]
                v_y = velocity[1]

                # Restart at origin if there is no movement
                if v_x == 0 and v_y == 0:
                    v_x_vals.clear()
                    v_y_vals.clear()
                    v_x_vals.append(0)
                    v_y_vals.append(0)
                
                else:
                    v_x_vals.append(v_x + v_x_vals[-1])
                    v_y_vals.append(v_y + v_y_vals[-1])

                v_line.set_data(v_x_vals, v_y_vals)

            return f_spec_line, v_line

        animated_plot = FuncAnimation(fig=self.containing_fig, func=update_plot, 
                                      interval=60, blit=True, cache_frame_data=False)
        
        plt.show()

    # Returns line associated with freq spectrum subplot
    def init_f_spec_plot(self):
        ax = self.containing_fig.add_subplot(1, 2, 1, 
                                             xlim=(self.MIN_PLOT_F, self.MAX_PLOT_F), 
                                             ylim=(0, 100))
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
        ax.set_ylabel("Amplitude(dB)", size=10)
        ax.grid(linestyle='-', linewidth=0.5)
        ax.axhline(0, color='red', linestyle='-', lw=1)
        line, = ax.plot([], [])
        line.set_linewidth(1.25)

        return line

    # Returns line associated with velocity subplot
    def init_v_plot(self):
        ax = self.containing_fig.add_subplot(1, 2, 2, 
                                             xlim=(-10, 10), ylim=(-10, 10), 
                                             xticks=[], yticks=[])
        ax.set_title("Velocity")
        line, = ax.plot([], [])
        line.set_linewidth(1.25)

        return line