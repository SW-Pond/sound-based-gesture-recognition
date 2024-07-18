import multiprocessing as mp

import matplotlib.pyplot as plt
import numpy as np
import pyaudio
from matplotlib.animation import FuncAnimation

SAMPLE_RATE = 44100
INPUT_FRAMES_PER_BUFFER = 2048
OUTPUT_FREQ = 18000
NUM_OUT_CHANNELS = 2
NUM_IN_CHANNELS = 1


##############Make code more adaptive to changing constants

def output_tone():
    out_stream = pyaudio.PyAudio().open(format=pyaudio.paFloat32, channels=NUM_OUT_CHANNELS, rate=SAMPLE_RATE, output=True)

    duration = 1
    num_samples = SAMPLE_RATE * duration

    ############Figure out a way to create a less choppy sine wave (potentially smaller intervals for samples)
    samples = np.array( [np.sin(2 * np.pi * (OUTPUT_FREQ / NUM_OUT_CHANNELS) * (i / SAMPLE_RATE)) for i in range(num_samples)] ).astype(np.float32)
    byte_samples = samples.tobytes()

    while True:
        out_stream.write(byte_samples)

def get_audio_input(in_q, scan_q):
    #Indices for frequencies in 17kHz-19kHz
    RANGE_START = 790
    RANGE_END = 884

    in_stream = pyaudio.PyAudio().open(format=pyaudio.paInt16, channels=NUM_IN_CHANNELS, rate=SAMPLE_RATE, input=True, frames_per_buffer=INPUT_FRAMES_PER_BUFFER)

    window = np.hamming(INPUT_FRAMES_PER_BUFFER * NUM_IN_CHANNELS)
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
        in_q.put(data)

def plot(in_q):
    fig = plt.figure()
    ax = fig.add_subplot(xlim=(17000, 19000), ylim=(-30, 100))
    ax.set_xlabel("Frequency(Hz)", size=10)
    ax.set_ylabel("Amplitude(dB?)", size=10)
    ax.axhline(0, color='red', linestyle='-', lw=1)
    line, = ax.plot([], [])

    def update_plot(i):
        if not in_q.empty():
            data = in_q.get()
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

def primary_scan(freqs, amps):
    #check_vals(freqs, amps)

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

def check_vals(freqs, amps):
        print("Amplitudes : Frequencies : Index")
        for i in range(min(len(freqs), len(amps))):
            print(str(freqs[i]) + " : " + str(amps[i]) + " : " + str(i))
        
        print()
        max_idx = np.argmax(amps)
        max_amp = amps[max_idx]
        corresponding_freq = freqs[max_idx]
        print("Greatest Amp / Corresponding Freq | " + str(max_amp) + " / " + str(corresponding_freq))
        print("Amp at next 3 freqs to the left: " + str(amps[max_idx-3]) + ", " + str(amps[max_idx-2]) + ", "+ str(amps[max_idx-1]))
        print("Amp at next 3 freqs to the right: " + str(amps[max_idx+1]) + ", " + str(amps[max_idx+2]) + ", "+ str(amps[max_idx+3]))
        print("-----------------")

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