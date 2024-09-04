import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from multiprocessing import Queue


class VelocityAnalyzer:
    def __init__(self, f_bin_res, peak_freqs, f_domain_q, v_q):
        self.C = 343 # Speed of sound in air (m/s)
        # Constants for defining what constitutes detected movement
        self.MIN_FREQ_SHIFT = 30
        self.AMP_CUTOFF_FACTOR = 0.3
        # Angles (degrees) from mic
        self.L_SPKR_ANGLE = -45
        self.R_SPKR_ANGLE = 45

        self.f_bin_res = f_bin_res
        self.l_f = np.min(peak_freqs)
        self.r_f = np.max(peak_freqs)
        # Round func used in case SAMPLE_RATE or IN_BUFFER_SIZE is changed
        #   such that SAMPLE_RATE is not an integer multiple of IN_BUFFER_SIZE.
        self.l_f_idx = int(np.round(self.l_f / self.f_bin_res))
        self.r_f_idx = int(np.round(self.r_f / self.f_bin_res))
        self.f_domain_q = f_domain_q
        self.v_q = v_q
        self.plot_q = Queue()

    def get_v(self):
        while True:
            if not self.f_domain_q.empty():
                data = self.f_domain_q.get()
                freqs = data[0]
                amps = data[1]

                v_vec = self.scan(amps)

                self.plot_q.put(np.copy(v_vec))
                self.v_q.put(v_vec)

    def scan(self, amps):
        l_v_vec = self.peak_scan(amps, 'L')
        r_v_vec = self.peak_scan(amps, 'R')

        v_x = l_v_vec[0] + r_v_vec[0]
        v_y = l_v_vec[1] + r_v_vec[1]
        v_vec = [v_x, v_y]

        return v_vec

    def peak_scan(self, amps, peak):
        """
        ToDo:
            Use MAG of freq shift to determine v (subtract left shift from right shift); may need to 
            define each shift as the difference in freq between current time series sequence and last.
            
            May need to define each shift as starting from the edge frequency bins of the peak tones.
            
            May need to scale left/right vectors to have similar vector mag as forward/back movement.
        """

        peak_idx = 0
        speaker_angle = 0
        speaker_freq = 0

        if peak == 'L':
            speaker_angle = self.L_SPKR_ANGLE
            speaker_freq = self.l_f
            peak_idx = self.l_f_idx
        else:
            speaker_angle = self.R_SPKR_ANGLE   
            speaker_freq = self.r_f         
            peak_idx = self.r_f_idx

        amp_cutoff = self.AMP_CUTOFF_FACTOR * amps[peak_idx]
        freq_shift = 0
        low_shift = 0
        high_shift = 0
        low_idx = peak_idx - 1
        high_idx = peak_idx + 1

        # Scan bins left of peak
        while amps[low_idx] > amp_cutoff and low_idx >= peak_idx - 16:
            low_shift -= self.f_bin_res
            low_idx -= 1

        # Scan bins right of peak
        while amps[high_idx] > amp_cutoff and high_idx <= peak_idx + 16:
            high_shift += self.f_bin_res
            high_idx += 1

        if np.abs(low_shift) > high_shift:
            freq_shift = low_shift
        if high_shift > np.abs(low_shift):
            freq_shift = high_shift

        if np.abs(freq_shift) < self.MIN_FREQ_SHIFT:
            freq_shift = 0

        mic_freq = speaker_freq + freq_shift
        # Handles division by zero error
        v = self.C * (mic_freq - speaker_freq) / (mic_freq + speaker_freq) if (mic_freq + speaker_freq) != 0 else 0
        v_x = v * np.sin(speaker_angle * (np.pi / 180))
        v_y = v * np.cos(speaker_angle * (np.pi / 180))
        v_vec = [v_x, v_y]

        return v_vec

    def plot(self):
        fig = plt.figure()
        ax = fig.add_subplot(xlim=(-10, 10), ylim=(-10, 10), xticks=[], yticks=[])
        line, = ax.plot([], [])
        line.set_linewidth(1.25)
        v_x_vals = [0]
        v_y_vals = [0]

        def update_plot(i):
            nonlocal v_x_vals, v_y_vals
            if not self.plot_q.empty():
                velocity = self.plot_q.get()
                v_x = velocity[0]
                v_y = velocity[1]

                if v_x == 0 and v_y == 0:
                    v_x_vals = [0]
                    v_y_vals = [0]
                
                else:
                    v_x_vals.append(v_x + v_x_vals[-1])
                    v_y_vals.append(v_y + v_y_vals[-1])

            line.set_data(v_x_vals, v_y_vals)
            return [line]

        animated_plot = FuncAnimation(fig=fig, func=update_plot, interval=0, 
                                      blit=True, cache_frame_data=False)
        plt.show()