import multiprocessing as mp

import matplotlib.pyplot as plt
import numpy as np
import pyaudio
from matplotlib.animation import FuncAnimation

# CHANGE AS NEEDED
SAMPLE_RATE = 44100
INPUT_FRAMES_PER_BUFFER = 4096
PEAK_MARGIN = 250 # in Hz
LEFT_OUT_FREQ = 18000 # must be lower than right
RIGHT_OUT_FREQ = 18500

# DON'T CHANGE
FREQ_SPACING = SAMPLE_RATE / INPUT_FRAMES_PER_BUFFER

def output_tone():
    out_stream = pyaudio.PyAudio().open(format=pyaudio.paFloat32, channels=2, rate=SAMPLE_RATE, output=True)

    samples = generate_samples()
    
    while True:
        out_stream.write(samples)
    
def generate_samples():
    samples = np.empty(SAMPLE_RATE * 2, dtype=np.float32)
    left_samples = np.array( [np.sin(2 * np.pi * LEFT_OUT_FREQ * (i / SAMPLE_RATE)) for i in range(SAMPLE_RATE)] ).astype(np.float32)
    right_samples = np.array( [np.sin(2 * np.pi * RIGHT_OUT_FREQ * (i / SAMPLE_RATE)) for i in range(SAMPLE_RATE)] ).astype(np.float32)

    for i in range(0, len(samples), 2):
        samples[i] = right_samples[i // 2]
        samples[i + 1] = left_samples[i // 2]

    while(len(samples) / SAMPLE_RATE < 10000):
        samples = np.concatenate((samples, samples), axis=None)

    return samples.tobytes()

def get_audio_input(input_q, scan_q):
    #Setting lower bound of interest to the frequency: LEFT_OUT_FREQ - PEAK_MARGIN
    RANGE_START = int(np.floor((LEFT_OUT_FREQ - PEAK_MARGIN) / FREQ_SPACING))
    #Setting higher bound of interest to the frequency: RIGHT_OUT_FREQ + PEAK_MARGIN
    RANGE_END = int(np.ceil((RIGHT_OUT_FREQ + PEAK_MARGIN) / FREQ_SPACING))

    in_stream = pyaudio.PyAudio().open(format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE, input=True, frames_per_buffer=INPUT_FRAMES_PER_BUFFER)

    window = np.hamming(INPUT_FRAMES_PER_BUFFER)
    freqs = np.linspace(0, SAMPLE_RATE / 2, (INPUT_FRAMES_PER_BUFFER // 2) + 1)
    freqs = freqs[RANGE_START:RANGE_END]

    while True:
        time_amps = np.frombuffer(in_stream.read(INPUT_FRAMES_PER_BUFFER), dtype=np.int16) * window
        freq_amps = np.abs(np.fft.rfft(time_amps))
        freq_dB_amps = 20 * np.log10(freq_amps, where=(freq_amps != 0)) - 50

        freq_dB_amps = freq_dB_amps[RANGE_START:RANGE_END]

        ###########Filter input by scaling up by 1.5 x mean amp and zeroing out values below 30% of that

        data = np.vstack((freqs, freq_dB_amps))
        scan_q.put(np.copy(data))
        input_q.put(data)

def plot(input_q):
    fig = plt.figure()
    ax = fig.add_subplot(xlim=(LEFT_OUT_FREQ - PEAK_MARGIN, RIGHT_OUT_FREQ + PEAK_MARGIN), ylim=(-30, 100))
    ax.set_xlabel("Frequency(Hz)", size=10)
    ax.set_ylabel("Amplitude(dB?)", size=10)
    ax.axhline(0, color='red', linestyle='-', lw=1)
    line, = ax.plot([], [])

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