import multiprocessing as mp
import keyboard
import velocity
import audio
from recognizer.data_handler import DataHandler
from plotter import Plotter

LOW_FREQ = 18000
HIGH_FREQ = 18500
IN_BUFFER_SIZE = 2048
SAMPLE_RATE = 44100
FREQ_BIN_RES = SAMPLE_RATE / (IN_BUFFER_SIZE * 2)


def end_program(processes):
    print("\nEnding processes...")
    for p in processes:
        p.terminate()
    print("Program has ended.")


if __name__ == "__main__":
    velocity_queue = mp.Queue()
    velocity_plot_queue = mp.Queue()
    input_plot_queue = mp.Queue()
    result_queue = mp.Queue()
    recognition_queue = mp.Queue()

    output_tones = audio.Output(low_freq=LOW_FREQ, high_freq=HIGH_FREQ, 
                                sample_rate=SAMPLE_RATE)
    audio_in = audio.Input(v_q=velocity_queue, in_plot_q=input_plot_queue, 
                           recognition_q=recognition_queue, 
                           buffer_size=IN_BUFFER_SIZE, sample_rate=SAMPLE_RATE, 
                           peak_freqs=(LOW_FREQ, HIGH_FREQ), 
                           f_bin_res=FREQ_BIN_RES)
    velocity_analyzer = velocity.VelocityAnalyzer(f_domain_q=velocity_queue, 
                                                  v_plot_q=velocity_plot_queue, 
                                                  peak_freqs=(LOW_FREQ, HIGH_FREQ), 
                                                  f_bin_res=FREQ_BIN_RES)
    plotter = Plotter(in_plot_q=input_plot_queue, v_plot_q=velocity_plot_queue,
                      res_q=result_queue, low_freq=LOW_FREQ, high_freq=HIGH_FREQ, 
                      f_bin_res=FREQ_BIN_RES)
    recognizer = DataHandler(data_q=recognition_queue, res_q=result_queue)

    output_p = mp.Process(target=output_tones.start)
    input_p = mp.Process(target=audio_in.get)
    get_velocity_p = mp.Process(target=velocity_analyzer.get_v)
    plot_p = mp.Process(target=plotter.plot_all)
    recognition_p = mp.Process(target=recognizer.process_data)

    processes = [output_p, input_p, get_velocity_p, plot_p, recognition_p]

    for p in processes:
        p.start()

    keyboard.wait('esc')
    end_program(processes)