import matplotlib.pyplot as plt
import numpy as np
import pyaudio
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import MultipleLocator
from multiprocessing import Queue
from scipy.signal import windows


class Output:
    def __init__(self, low_freq, high_freq, sample_rate):
        self.left_freq = low_freq
        self.right_freq = high_freq
        self.sample_rate = sample_rate
        self.samples = self.generate_samples()
        
    def start(self):
        out_stream = pyaudio.PyAudio().open(format=pyaudio.paFloat32, channels=2, rate=self.sample_rate, output=True)
        
        while True:
            out_stream.write(self.samples)

    def generate_samples(self):
        samples = np.empty(self.sample_rate * 2, dtype=np.float32)
        left_samples = np.array( [0.25 * np.sin(2 * np.pi * self.left_freq * (i / self.sample_rate))
                                  for i in range(self.sample_rate)] ).astype(np.float32)
        right_samples = np.array( [0.25 * np.sin(2 * np.pi * self.right_freq * (i / self.sample_rate))
                                   for i in range(self.sample_rate)] ).astype(np.float32)

        # Interleaving samples
        for i in range(0, len(samples), 2):
            samples[i] = left_samples[i // 2]
            samples[i + 1] = right_samples[i // 2]

        # Makes samples longer, creating smoother tone
        while(len(samples) / self.sample_rate < 10000):
            samples = np.concatenate((samples, samples), axis=None)

        return samples.tobytes()


class Input:
    def __init__(self, input_queue, buffer_size, sample_rate, peak_freqs, f_bin_res):
        self.FILTER_FACTOR = 1.3

        self.plot_q = Queue()
        self.in_q = input_queue
        self.buffer_size = buffer_size
        self.sample_rate = sample_rate
        self.l_f = np.min(peak_freqs)
        self.r_f = np.max(peak_freqs)
        self.f_bin_res = f_bin_res
    
    # Gets input in freq domain
    def get(self):
        in_stream = pyaudio.PyAudio().open(format=pyaudio.paInt16, channels=1, rate=self.sample_rate, 
                                           input=True, frames_per_buffer=self.buffer_size)

        window = windows.blackmanharris(self.buffer_size)
        freqs = np.linspace(0, self.sample_rate / 2, (self.buffer_size // 2) + 1)

        while True:
            t_domain_amps = np.frombuffer(in_stream.read(self.buffer_size), dtype=np.int16) * window
            f_domain_amps = np.abs((np.fft.rfft(t_domain_amps)))
            f_domain_dB_amps = self.to_dB_and_filter(f_domain_amps)

            data = np.vstack((freqs, f_domain_dB_amps))

            self.plot_q.put(np.copy(data))
            self.in_q.put(data)

    def to_dB_and_filter(self, amps):
        pre_shift_mean = 0
        post_shift_mean = 0

        for i in range(len(amps)):
            if amps[i] != 0:
                amps[i] = 20 * np.log10(amps[i])

            pre_shift_mean += amps[i]
        pre_shift_mean /= (len(amps))

        for i in range(len(amps)):
            amps[i] -= pre_shift_mean

            if(amps[i] < 0):
                amps[i] = 0

            post_shift_mean += amps[i]
        post_shift_mean /= len(amps)
            
        for i in range(len(amps)):
            if amps[i] < post_shift_mean * self.FILTER_FACTOR:
                amps[i] = 0

        return amps

    def plot(self):
        PEAK_MARGIN = 250
        MIN_F = self.l_f - PEAK_MARGIN
        MAX_F = self.r_f + PEAK_MARGIN
        MIN_F_IDX = int(np.floor(MIN_F / self.f_bin_res))
        MAX_F_IDX = int(np.ceil(MAX_F / self.f_bin_res))

        fig = plt.figure()
        ax = fig.add_subplot(xlim=(MIN_F, MAX_F), ylim=(0, 100))
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

        def update_plot(i):
            if not self.plot_q.empty():
                data = self.plot_q.get()
                freqs = data[0][MIN_F_IDX : MAX_F_IDX + 1]
                amps = data[1][MIN_F_IDX : MAX_F_IDX + 1]

                line.set_data(freqs, amps)
            return [line]

        animated_plot = FuncAnimation(fig=fig, func=update_plot, interval=50, 
                                      blit=True, cache_frame_data=False)
        plt.show()