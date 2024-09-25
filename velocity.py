import numpy as np


class VelocityAnalyzer:
    def __init__(self, f_domain_q, v_plot_q, f_bin_res, peak_freqs):
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
        self.v_plot_q = v_plot_q

    def get_v(self):
        while True:
            if not self.f_domain_q.empty():
                data = self.f_domain_q.get()
                freqs = data[0]
                amps = data[1]

                v_vec = self.scan(amps)

                self.v_plot_q.put(v_vec)

    def scan(self, amps):
        l_v_vec = self.peak_scan(amps, 'L')
        r_v_vec = self.peak_scan(amps, 'R')

        v_x = l_v_vec[0] + r_v_vec[0]
        v_y = l_v_vec[1] + r_v_vec[1]
        v_vec = [v_x, v_y]

        return v_vec

    def peak_scan(self, amps, peak):
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