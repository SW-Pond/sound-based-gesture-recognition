import numpy as np
import keyboard
import pyaudio
from scipy.signal import windows
from recognizer import logger


class Output:
    def __init__(self, low_freq, high_freq, sample_rate):
        self.left_freq = low_freq
        self.right_freq = high_freq
        self.sample_rate = sample_rate
        self.samples = self.generate_samples()
        
    def start(self):
        out_stream = pyaudio.PyAudio().open(format=pyaudio.paFloat32, 
                                            channels=2, rate=self.sample_rate, 
                                            output=True)
        
        while True:
            out_stream.write(self.samples)

    def generate_samples(self):
        samples = np.empty(self.sample_rate * 2, dtype=np.float32)
        left_samples = np.array( [0.25 * np.sin(2 * np.pi * self.left_freq * (i / self.sample_rate))
                                  for i in range(self.sample_rate)] ).astype(np.float32)
        right_samples = np.array( [0.25 * np.sin(2 * np.pi * self.right_freq * (i / self.sample_rate))
                                   for i in range(self.sample_rate)] ).astype(np.float32)

        # Interleaving samples for output to correct speakers
        for i in range(0, len(samples), 2):
            samples[i] = left_samples[i // 2]
            samples[i + 1] = right_samples[i // 2]

        return samples.tobytes()


class Input:
    def __init__(self, v_q, in_plot_q, recognition_q, buffer_size, sample_rate, 
                 peak_freqs, f_bin_res):
        self.FILTER_FACTOR = 1.3
        self.v_q = v_q
        self.plot_q = in_plot_q
        self.recognition_q = recognition_q
        self.buffer_size = buffer_size
        self.sample_rate = sample_rate
        l_f = np.min(peak_freqs)
        r_f = np.max(peak_freqs)
        f_bin_res = f_bin_res
        self.l_f_bin = int(np.round(l_f / f_bin_res))
        self.r_f_bin = int(np.round(r_f / f_bin_res))
        self.template_logger = logger.Logger()
        self.curr_frame = None # For template logging only
        self.last_amps = []

    def get(self):
        in_stream = pyaudio.PyAudio().open(format=pyaudio.paInt16, channels=1, 
                                           rate=self.sample_rate, input=True, 
                                           frames_per_buffer=self.buffer_size)

        window = windows.blackmanharris(self.buffer_size * 2)
        freqs = np.linspace(0, self.sample_rate / 2, self.buffer_size + 1)

        while True:
            curr_amps = np.frombuffer(in_stream.read(self.buffer_size), 
                                          dtype=np.int16)
            
            # If not first buffer on startup
            if len(self.last_amps) != 0:
                t_domain_amps = np.concatenate((self.last_amps, curr_amps))
                t_domain_amps = t_domain_amps * window
                self.last_amps = curr_amps
            else:
                self.last_amps = curr_amps
                continue

            f_domain_amps = np.abs(np.fft.rfft(t_domain_amps))
            f_domain_dB_amps = self.to_dB_and_filter(f_domain_amps)

            # Create current frame and pass to data manager
            f_domain_vec = np.copy(f_domain_dB_amps)

            l_f_domain_vec = f_domain_vec[self.l_f_bin - 22 : self.l_f_bin + 23]
            r_f_domain_vec = f_domain_vec[self.r_f_bin - 22 : self.r_f_bin + 23]

            self.normalize(l_f_domain_vec)
            self.normalize(r_f_domain_vec)

            frame = np.concatenate((l_f_domain_vec, r_f_domain_vec))

            self.recognition_q.put(np.copy(frame))
            self.curr_frame = frame # For template logging only
            
            # For template logging only
            #keyboard.on_press(self.check_pressed_key)
            #keyboard.on_release(lambda _:_) # Stop indefinite key press event

            visualization_data = np.vstack((freqs, f_domain_dB_amps))
            self.plot_q.put(np.copy(visualization_data))
            self.v_q.put(visualization_data)

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

    def normalize(self, vector):
        """
        Rescale vector components to range [0,1]
        """

        if not len(vector) == 0:
            max = np.max(vector)

            # Avoid division by 0
            if max != 0:
                for i in range(len(vector)):
                    vector[i] /= max

    def check_pressed_key(self, key_event):
        key = key_event.name

        if key == 'r':
            self.template_logger.reset_template()

        if key == 't':
            self.template_logger.record_frame(self.curr_frame)
        
        if key in logger.GESTURE_TYPES.keys():
            self.template_logger.log(g_key=key)
            self.template_logger.reset_template()