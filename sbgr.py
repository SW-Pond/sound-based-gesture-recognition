import multiprocessing as mp

import matplotlib.pyplot as plt
import numpy as np
import pyaudio
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import MultipleLocator
from scipy.signal import windows

# CHANGE AS NEEDED
LEFT_OUT_FREQ = 20000 # must be less than RIGHT_OUT_FREQ
RIGHT_OUT_FREQ = 20500
PEAK_MARGIN = 250 # in Hz; cannot be < 250 or < LEFT_OUT_FREQ

# DON'T CHANGE
C = 343 # speed of sound in air (m/s)
SAMPLE_RATE = 44100
IN_BUFFER_SIZE = 4410
FILTER_FACTOR = 7
FREQ_SHIFT_THRESHOLD = 10
SHIFT_THRESHOLD_FACTOR = 0.3
# Angles (degrees) from mic
LEFT_SPEAKER_ANGLE = -45
RIGHT_SPEAKER_ANGLE = 45
FREQ_BIN_RES = SAMPLE_RATE / IN_BUFFER_SIZE
# Setting lower bound of interest to the frequency bin corresponding to: LEFT_OUT_FREQ - PEAK_MARGIN
RANGE_START = int(np.floor((LEFT_OUT_FREQ - PEAK_MARGIN) / FREQ_BIN_RES))
# Setting higher bound of interest to the frequency bin corresponding to: RIGHT_OUT_FREQ + PEAK_MARGIN
RANGE_END = int(np.ceil((RIGHT_OUT_FREQ + PEAK_MARGIN) / FREQ_BIN_RES))
LEFT_PEAK_IDX = int(np.ceil(LEFT_OUT_FREQ / FREQ_BIN_RES))
RIGHT_PEAK_IDX = int(np.ceil(RIGHT_OUT_FREQ / FREQ_BIN_RES))

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
        samples[i] = left_samples[i // 2]
        samples[i + 1] = right_samples[i // 2]

    # Makes samples longer, creating smoother tone
    while(len(samples) / SAMPLE_RATE < 10000):
        samples = np.concatenate((samples, samples), axis=None)

    return samples.tobytes()

def get_audio_input(input_q, scan_q):
    in_stream = pyaudio.PyAudio().open(format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE, input=True, frames_per_buffer=IN_BUFFER_SIZE)

    window = windows.blackmanharris(IN_BUFFER_SIZE)
    freqs = np.linspace(0, SAMPLE_RATE / 2, (IN_BUFFER_SIZE // 2) + 1)

    while True:
        time_amps = np.frombuffer(in_stream.read(IN_BUFFER_SIZE), dtype=np.int16) * window
        freq_amps = np.abs((np.fft.rfft(time_amps)))
        freq_dB_amps = to_dB_and_filter(freq_amps)

        data = np.vstack((freqs, freq_dB_amps))
        scan_q.put(np.copy(data))
        input_q.put(data)

def to_dB_and_filter(amps): 
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
        if amps[i] < post_shift_mean * FILTER_FACTOR:
            amps[i] = 0

    return amps

def spectrum_plot(input_queue):
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
        if not input_queue.empty():
            data = input_queue.get()
            freqs = data[0][RANGE_START:RANGE_END]
            amps = data[1][RANGE_START:RANGE_END]

            line.set_data(freqs, amps)
        return [line]

    animated_plot = FuncAnimation(fig=fig, func=update_plot, interval=70, blit=True, cache_frame_data=False)
    plt.show()

def scan(scan_queue, velocity_queue):
    while True:
        if not scan_queue.empty():
            data = scan_queue.get()
            freqs = data[0]
            amps = data[1]
            v_vec = primary_scan(amps)
            velocity_queue.put(v_vec)

def primary_scan(amps):
    left_v_vec = peak_scan(amps, 'L')
    right_v_vec = peak_scan(amps, 'R')

    v_x = left_v_vec[0] + right_v_vec[0]
    v_y = left_v_vec[1] + right_v_vec[1]
    v_vec = [v_x, v_y]

    return v_vec

def secondary_scan():
    return 0

def peak_scan(amps, peak):
    peak_idx = 0
    speaker_angle = 0
    speaker_freq = 0
    if peak == 'L':
        peak_idx = LEFT_PEAK_IDX
        speaker_angle = LEFT_SPEAKER_ANGLE
        speaker_freq = LEFT_OUT_FREQ
    else:
        peak_idx = RIGHT_PEAK_IDX
        speaker_angle = RIGHT_SPEAKER_ANGLE
        speaker_freq = RIGHT_OUT_FREQ
    amp_shift_threshold = SHIFT_THRESHOLD_FACTOR * amps[peak_idx]
    freq_shift = 0
    low_shift = 0
    high_shift = 0
    low_idx = peak_idx - 1
    high_idx = peak_idx + 1

    # Scan bins left of peak
    while amps[low_idx] > amp_shift_threshold and low_idx >= peak_idx - 16:
        low_shift -= FREQ_BIN_RES
        low_idx -= 1

    # Scan bins right of peak
    while amps[high_idx] > amp_shift_threshold and high_idx <= peak_idx + 16:
        high_shift += FREQ_BIN_RES
        high_idx += 1
    #############################################################################################
    # Use MAG of freq shift to determine v (subtract left shift from right shift); may need to 
    #  define each shift as the difference in freq between current time series data and last.
    #
    # May need to define each shift as starting from the edge frequency bins of the peak tones, 
    #  then counting out from there.
    #
    # May need to scale left/right vectors to have same vector mag as forward/back movement
    #  (get mag shift correct first to see if it fixes vector scaling).
    #############################################################################################

    if np.abs(low_shift) > high_shift:
        freq_shift = low_shift
    if high_shift > np.abs(low_shift):
        freq_shift = high_shift

    if np.abs(freq_shift) < FREQ_SHIFT_THRESHOLD:
        freq_shift = 0

    mic_freq = speaker_freq + freq_shift
    v = C * (mic_freq - speaker_freq) / (mic_freq + speaker_freq) if (mic_freq + speaker_freq) != 0 else 0
    v_x = v * np.sin(speaker_angle * (np.pi / 180))
    v_y = v * np.cos(speaker_angle * (np.pi / 180))
    v_vec = [v_x, v_y]

    return v_vec

def velocity_plot(velocity_queue):
    fig = plt.figure()
    ax = fig.add_subplot(xlim=(-6, 6), ylim=(-4, 4), xticks=[], yticks=[])
    line, = ax.plot([], [])
    line.set_linewidth(1.25)
    v_x_vals = [0]
    v_y_vals = [0]

    def update_plot(i):
        nonlocal v_x_vals, v_y_vals
        if not velocity_queue.empty():
            velocity = velocity_queue.get()
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

    animated_plot = FuncAnimation(fig=fig, func=update_plot, interval=0, blit=True, cache_frame_data=False)
    plt.show()

if __name__ == "__main__":
    input_queue = mp.Queue()
    scan_queue = mp.Queue()
    velocity_queue = mp.Queue()

    out_p = mp.Process(target=output_tone)
    in_p = mp.Process(target=get_audio_input, args=(input_queue, scan_queue, ))
    spectrum_plot_p = mp.Process(target=spectrum_plot, args=(input_queue, ))
    scan_p = mp.Process(target=scan, args=(scan_queue, velocity_queue, ))
    velocity_plot_p = mp.Process(target=velocity_plot, args=(velocity_queue, ))

    out_p.start()
    in_p.start()
    spectrum_plot_p.start()
    scan_p.start()
    velocity_plot_p.start()

    out_p.join()
    in_p.join()
    spectrum_plot_p.join()
    scan_p.join()
    velocity_plot_p.join()