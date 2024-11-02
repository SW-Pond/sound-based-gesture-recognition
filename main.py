import multiprocessing as mp
import keyboard
import velocity
import audio
from plotter import Plotter
from jackknife.data_mgr import Manager
from jackknife.recognizer import Classifier


LOW_FREQ = 18000
HIGH_FREQ = 18500
IN_BUFFER_SIZE = 4096
SAMPLE_RATE = 44100

FREQ_BIN_RES = SAMPLE_RATE / IN_BUFFER_SIZE


def end_program(processes):
    print("Ending processes...")
    for p in processes:
        p.terminate()
    print("Program has ended.")


if __name__ == "__main__":
    input_queue = mp.Queue()
    input_plot_queue = mp.Queue()
    velocity_plot_queue = mp.Queue()
    result_queue = mp.Queue()
    classifier_conn, data_mgr_conn = mp.Pipe()
    data_mgr = Manager(pipe_conn=data_mgr_conn, res_q=result_queue)

    output_tones = audio.Output(low_freq=LOW_FREQ, high_freq=HIGH_FREQ, 
                                sample_rate=SAMPLE_RATE)
    audio_in = audio.Input(in_q=input_queue, in_plot_q=input_plot_queue, 
                           buffer_size=IN_BUFFER_SIZE, sample_rate=SAMPLE_RATE, 
                           peak_freqs=(LOW_FREQ, HIGH_FREQ), 
                           f_bin_res=FREQ_BIN_RES, data_mgr=data_mgr)
    velocity_analyzer = velocity.VelocityAnalyzer(f_domain_q=input_queue, 
                                                  v_plot_q=velocity_plot_queue, 
                                                  peak_freqs=(LOW_FREQ, HIGH_FREQ), 
                                                  f_bin_res=FREQ_BIN_RES)
    plotter = Plotter(in_plot_q=input_plot_queue, v_plot_q=velocity_plot_queue,
                      res_q=result_queue, low_freq=LOW_FREQ, high_freq=HIGH_FREQ, 
                      f_bin_res=FREQ_BIN_RES)
    classifier = Classifier(pipe_conn=classifier_conn)

    output_p = mp.Process(target=output_tones.start)
    input_p = mp.Process(target=audio_in.get)
    get_velocity_p = mp.Process(target=velocity_analyzer.get_v)
    plot_p = mp.Process(target=plotter.plot_all)
    classification_p = mp.Process(target=classifier.classify)

    processes = [output_p, input_p, get_velocity_p, plot_p, classification_p]

    for p in processes:
        p.start()

    keyboard.wait('esc')
    end_program(processes)
