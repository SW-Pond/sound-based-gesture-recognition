import tkinter as tk
import keyboard
import sys
import time

class Gui:
    def __init__(self, res_q, end_program_q):
        self.res_q = res_q
        self.end_program_q = end_program_q
        self.accepted_gestures = {"star_y(1)", "x_w_l(2)", "z_triangle(3)"}

    def check_for_gesture(self):
        window = tk.Tk()
        window.attributes("-topmost", True)
        window_width = 1200
        window_height = 200

        screen_width = window.winfo_screenwidth()
        screen_height = window.winfo_screenheight()

        x_position = (screen_width // 2) - (window_width // 2)
        y_position = (screen_height // 2) - (window_height // 2)

        window.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")

        window.overrideredirect(True)

        window.configure(bg="white")

        label = tk.Label(
            window,
            text="Perform gesture to unlock",
            font=("Helvetica", 45, "bold"),
            bg="white",
            fg="black"
        )
        label.pack(expand=True)

        window.update()

        while True:
            if not self.res_q.empty():
                result = self.res_q.get()
                match_name = result[0]

                if match_name in self.accepted_gestures:
                    window.config(bg="green")
                    label.config(text=f"GESTURE {match_name[-2]} ACCEPTED", background="green", foreground="white")
                    window.update()
                    keyboard.send("alt+tab")
                    time.sleep(0.03)
                    keyboard.write("Password@123!")
                    keyboard.send("enter")
                    window.after(2000)
                    self.end_program_q.put(1)
                    sys.exit(0)