import multiprocessing as mp

import matplotlib.pyplot as plt
import numpy as np
import pyaudio
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import MultipleLocator

# CHANGE AS NEEDED
SAMPLE_RATE = 44100
IN_BUFFER_SIZE = 4096
PEAK_MARGIN = 250 # in Hz; cannot be < 250
LEFT_OUT_FREQ = 18000 # must be less than right
RIGHT_OUT_FREQ = 18500
FILTER_FACTOR = 1.3

# DON'T CHANGE
FREQ_SPACING = SAMPLE_RATE / IN_BUFFER_SIZE
# Setting lower bound of interest to the frequency bin corresponding to: LEFT_OUT_FREQ - PEAK_MARGIN
RANGE_START = int(np.floor((LEFT_OUT_FREQ - PEAK_MARGIN) / FREQ_SPACING))
# Setting higher bound of interest to the frequency bin corresponding to: RIGHT_OUT_FREQ + PEAK_MARGIN
RANGE_END = int(np.ceil((RIGHT_OUT_FREQ + PEAK_MARGIN) / FREQ_SPACING))
LEFT_PEAK_IDX = int(np.ceil(LEFT_OUT_FREQ / FREQ_SPACING))
RIGHT_PEAK_IDX = int(np.ceil(RIGHT_OUT_FREQ / FREQ_SPACING))

def output_tone():
    out_stream = pyaudio.PyAudio().open(format=pyaudio.paFloat32, channels=2, rate=SAMPLE_RATE, output=True)

    samples = generate_samples()
    
    while True:
        out_stream.write(samples)
    
def generate_samples():
    samples = np.empty(SAMPLE_RATE * 2, dtype=np.float32)
    left_samples = np.array( [np.sin(2 * np.pi * LEFT_OUT_FREQ * (i / SAMPLE_RATE)) for i in range(SAMPLE_RATE)] ).astype(np.float32)
    right_samples = np.array( [np.sin(2 * np.pi * RIGHT_OUT_FREQ * (i / SAMPLE_RATE)) for i in range(SAMPLE_RATE)] ).astype(np.float32)

    # Interleaving samples
    for i in range(0, len(samples), 2):
        samples[i] = right_samples[i // 2]
        samples[i + 1] = left_samples[i // 2]

    # Makes samples longer, creating smoother tone
    while(len(samples) / SAMPLE_RATE < 10000):
        samples = np.concatenate((samples, samples), axis=None)

    return samples.tobytes()

def get_audio_input(input_q, scan_q):
    in_stream = pyaudio.PyAudio().open(format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE, input=True, frames_per_buffer=IN_BUFFER_SIZE)

    window = np.hamming(IN_BUFFER_SIZE)
    freqs = np.linspace(0, SAMPLE_RATE / 2, (IN_BUFFER_SIZE // 2) + 1)
    freqs = freqs[RANGE_START:RANGE_END]

    while True:
        time_amps = np.frombuffer(in_stream.read(IN_BUFFER_SIZE), dtype=np.int16) * window
        freq_amps = np.abs((np.fft.rfft(time_amps)))
        freq_dB_amps = to_dB_and_filter(freq_amps)

        data = np.vstack((freqs, freq_dB_amps))
        scan_q.put(np.copy(data))
        input_q.put(data)

def to_dB_and_filter(amps): 
    spectrum_mean = 0
    non_peak_mean = 0

    for i in range(len(amps)):
        if amps[i] != 0:
            amps[i] = 20 * np.log10(amps[i])

        if i != LEFT_PEAK_IDX and i != RIGHT_PEAK_IDX:
            non_peak_mean += amps[i]
    non_peak_mean /= (len(amps) - 2)

    for i in range(len(amps)):
        amps[i] -= non_peak_mean
        if(amps[i] < 0):
            amps[i] = 0
        spectrum_mean += amps[i]
    spectrum_mean /= len(amps)

    amps = amps[RANGE_START:RANGE_END]
    
    for i in range(len(amps)):
        if amps[i] < spectrum_mean * FILTER_FACTOR:
            amps[i] = 0
    
    return amps

def plot(input_q):
    fig = plt.figure()
    ax = fig.add_subplot(xlim=(LEFT_OUT_FREQ - PEAK_MARGIN, RIGHT_OUT_FREQ + PEAK_MARGIN), ylim=(0, 100))
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
        if not input_q.empty():
            data = input_q.get()
            freqs = data[0]
            amps = data[1]

            line.set_data(freqs, amps)
        return [line]

    animated_plot = FuncAnimation(fig=fig, func=update_plot, interval=1, blit=True, cache_frame_data=False)
    plt.show()

def scan(scan_q):
    while True:
        if not scan_q.empty():
            data = scan_q.get()
            freqs = data[0]
            amps = data[1]
            primary_scan(freqs, amps)

######################### 16 bins on either side of peak, not 33
def primary_scan(freqs, amps):
    pilot_idx = len(freqs) // 2 - 1
    pilot_amp = amps[pilot_idx]
    AMP_SHIFT_THRESHOLD = 0.5 * pilot_amp
    BIN_SHIFT_THRESHOLD = 4

    if amps[pilot_idx] > 60:
        left_shift_bins = 0
        right_shift_bins = 0
        left_idx = pilot_idx - 1
        right_idx = pilot_idx + 1

        #Left scan
        while amps[left_idx] > AMP_SHIFT_THRESHOLD and left_idx >= pilot_idx - 33:
            left_shift_bins += 1
            left_idx -= 1
        
        #Right scan
        while amps[right_idx] > AMP_SHIFT_THRESHOLD and right_idx <= pilot_idx + 33:
            right_shift_bins += 1
            right_idx += 1

        if left_shift_bins > BIN_SHIFT_THRESHOLD or right_shift_bins > BIN_SHIFT_THRESHOLD:
            if left_shift_bins > right_shift_bins:
                print("Backward movement")

            if right_shift_bins > left_shift_bins:
                print("Forward movement")

def secondary_scan():
    return 0

if __name__ == "__main__":
    input_queue = mp.Queue()
    scan_queue = mp.Queue()

    out_p = mp.Process(target=output_tone)
    in_p = mp.Process(target=get_audio_input, args=(input_queue, scan_queue, ))
    plot_p = mp.Process(target=plot, args=(input_queue, ))
    scan_p = mp.Process(target=scan, args=(scan_queue, ))

    out_p.start()
    in_p.start()
    plot_p.start()
    scan_p.start()

    out_p.join()
    in_p.join()
    plot_p.join()
    scan_p.join()