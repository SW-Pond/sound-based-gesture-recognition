import multiprocessing as mp
import velocity
import audio

LOW_FREQ = 18000
HIGH_FREQ = 18500
IN_BUFFER_SIZE = 4410
SAMPLE_RATE = 44100

FREQ_BIN_RES = SAMPLE_RATE / IN_BUFFER_SIZE

if __name__ == "__main__":
    input_queue = mp.Queue()
    velocity_queue = mp.Queue()

    output_tones = audio.Output(low_freq=LOW_FREQ, high_freq=HIGH_FREQ, sample_rate=SAMPLE_RATE)
    audio_in = audio.Input(input_queue=input_queue, buffer_size=IN_BUFFER_SIZE, sample_rate=SAMPLE_RATE, 
                           peak_freqs=(LOW_FREQ, HIGH_FREQ), f_bin_res=FREQ_BIN_RES)
    velocity_analyzer = velocity.VelocityAnalyzer(f_domain_q=input_queue, v_q=velocity_queue, 
                                                  peak_freqs=(LOW_FREQ, HIGH_FREQ), f_bin_res=FREQ_BIN_RES)

    output_p = mp.Process(target=output_tones.start)
    input_p = mp.Process(target=audio_in.get)
    f_spectrum_plot_p = mp.Process(target=audio_in.plot)
    get_velocity_p = mp.Process(target=velocity_analyzer.get_v)
    velocity_plot_p = mp.Process(target=velocity_analyzer.plot)

    output_p.start()
    input_p.start()
    f_spectrum_plot_p.start()
    get_velocity_p.start()
    velocity_plot_p.start()

    output_p.join()
    input_p.join()
    f_spectrum_plot_p.join()
    get_velocity_p.join()
    velocity_plot_p.join()