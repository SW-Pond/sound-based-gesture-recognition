import multiprocessing as mp
import pyautogui
import keyboard

pyautogui.PAUSE = 0.0005
pyautogui.FAILSAFE = False

class Mapper:
    """Gesture-action mapping for demo"""

    def __init__(self, action_q):
        self.action_q = action_q

        # Mode states
        self.MOVING = 0
        self.LOOKING = 1
        self.INTERACTING = 2

        # Directions
        self.STOP = 0
        self.UP_BACK = 1
        self.DOWN_FORWARD = 2
        self.LEFT = 3
        self.RIGHT = 4

        # Action flags
        self.UPDATE_DIRECTION = 1
        self.NEW_GESTURE = 2
        self.CHANGE_MODE = 3

    def state_machine(self):
        looking_flag = mp.Value('i', self.STOP)
        looking_p = mp.Process(target=Mapper.update_look_direction,
                                    args=(looking_flag,))
        
        moving_flag = mp.Value('i', self.STOP)
        moving_p = mp.Process(target=Mapper.update_move_direction,
                                    args=(moving_flag,))

        looking_p.start()
        moving_p.start()

        prev_action_flag = 0
        prev2_action_flag = 0
        prev3_action_flag = 0

        prev4_direction = self.STOP
        prev3_direction = self.STOP
        prev2_direction = self.STOP
        prev_direction = self.STOP

        mode = self.INTERACTING
        last_mode = mode

        while True:
            if self.action_q.empty():
                continue
            action = self.action_q.get()
            action_flag = action[0]

            
            if action_flag == self.CHANGE_MODE and \
               prev_action_flag != self.CHANGE_MODE and \
               prev2_action_flag != self.CHANGE_MODE and \
               prev3_action_flag != self.CHANGE_MODE:
                
                mode = (mode + 1) % 3
                print(f"mode change; mode: {mode}")
                prev3_action_flag = prev2_action_flag
                prev2_action_flag = prev_action_flag
                prev_action_flag = action_flag
                continue
            else:
                prev3_action_flag = prev2_action_flag
                prev2_action_flag = prev_action_flag
                prev_action_flag = action_flag

            if last_mode != mode:
                looking_flag.value = self.STOP
                moving_flag.value = self.STOP
                last_mode = mode
                
            match mode:
                case self.MOVING:
                    if action_flag == self.UPDATE_DIRECTION:
                        l_tone_shift = action[1]
                        r_tone_shift = action[2]
                        
                        direction = self.get_direction(l_tone_shift, r_tone_shift)
                        
                        match direction:
                            case self.UP_BACK:
                                if not (prev_direction == self.UP_BACK \
                                   and prev2_direction == self.UP_BACK \
                                   and prev3_direction == self.UP_BACK):
                                    moving_flag.value = self.STOP
                                else:
                                    moving_flag.value = self.UP_BACK
                                prev3_direction = prev2_direction
                                prev2_direction = prev_direction
                                prev_direction = self.UP_BACK
                            case self.DOWN_FORWARD:
                                if not (prev_direction == self.DOWN_FORWARD \
                                   and prev2_direction == self.DOWN_FORWARD \
                                   and prev3_direction == self.DOWN_FORWARD):
                                    moving_flag.value = self.STOP
                                else:
                                    moving_flag.value = self.DOWN_FORWARD
                                prev3_direction = prev2_direction
                                prev2_direction = prev_direction
                                prev_direction = self.DOWN_FORWARD
                            case self.LEFT:
                                if not (prev_direction == self.LEFT \
                                   and prev2_direction == self.LEFT \
                                   and prev3_direction == self.LEFT):
                                    moving_flag.value = self.STOP
                                else:
                                    moving_flag.value = self.LEFT
                                prev3_direction = prev2_direction
                                prev2_direction = prev_direction
                                prev_direction = self.LEFT
                            case self.RIGHT:
                                if not (prev_direction == self.RIGHT \
                                   and prev2_direction == self.RIGHT \
                                   and prev3_direction == self.RIGHT):
                                    moving_flag.value = self.STOP
                                else:
                                    moving_flag.value = self.RIGHT
                                prev3_direction = prev2_direction
                                prev2_direction = prev_direction
                                prev_direction = self.RIGHT
                            case default:
                                continue
                    else:
                        continue
                
                case self.LOOKING:
                    if action_flag == self.UPDATE_DIRECTION:
                        l_tone_shift = action[1]
                        r_tone_shift = action[2]
                        
                        direction = self.get_direction(l_tone_shift, r_tone_shift)
                        
                        match direction:
                            case self.UP_BACK:
                                if not (prev_direction == self.UP_BACK \
                                   and prev2_direction == self.UP_BACK \
                                   and prev3_direction == self.UP_BACK):
                                    looking_flag.value = self.STOP
                                else:
                                    looking_flag.value = self.UP_BACK
                                prev3_direction = prev2_direction
                                prev2_direction = prev_direction
                                prev_direction = self.UP_BACK
                            case self.DOWN_FORWARD:
                                if not (prev_direction == self.DOWN_FORWARD \
                                   and prev2_direction == self.DOWN_FORWARD \
                                   and prev3_direction == self.DOWN_FORWARD):
                                    looking_flag.value = self.STOP
                                else:
                                    looking_flag.value = self.DOWN_FORWARD
                                prev3_direction = prev2_direction
                                prev2_direction = prev_direction
                                prev_direction = self.DOWN_FORWARD
                            case self.LEFT:
                                if not (prev_direction == self.LEFT \
                                   and prev2_direction == self.LEFT \
                                   and prev3_direction == self.LEFT):
                                    looking_flag.value = self.STOP
                                else:
                                    looking_flag.value = self.LEFT
                                prev3_direction = prev2_direction
                                prev2_direction = prev_direction
                                prev_direction = self.LEFT
                            case self.RIGHT:
                                if not (prev_direction == self.RIGHT \
                                   and prev2_direction == self.RIGHT \
                                   and prev3_direction == self.RIGHT):
                                    looking_flag.value = self.STOP
                                else:
                                    looking_flag.value = self.RIGHT
                                prev3_direction = prev2_direction
                                prev2_direction = prev_direction
                                prev_direction = self.RIGHT
                            case default:
                                continue
                    else:
                        continue

                case self.INTERACTING:
                    if action_flag == self.NEW_GESTURE:
                        name = action[1]

                        match name:
                            case "x":
                                pyautogui.leftClick()
                            case "triangle":
                                pyautogui.rightClick()
                            case "w":
                                keyboard.press_and_release("e")
                            case "z":
                                pyautogui.scroll(1)
                            case "star":
                                pyautogui.scroll(-1)
                            case default:
                                continue
                    else:
                        continue
            
    def get_direction(self, l_tone_shift, r_tone_shift):
        direction = self.STOP

        if l_tone_shift < -100 and r_tone_shift < -100:
            direction = self.UP_BACK
        if l_tone_shift > 90 and r_tone_shift > 90:
            direction = self.DOWN_FORWARD
        if l_tone_shift < -60 and r_tone_shift > 50:
            direction = self.RIGHT
        if l_tone_shift > 50 and r_tone_shift < -40:
            direction = self.LEFT

        return direction

    @staticmethod
    def update_look_direction(looking_flag):
        while True:
            match looking_flag.value:
                case 0:
                    continue
                case 1:
                    pyautogui.moveRel(0,-1,0.0000001)
                case 2:
                    pyautogui.moveRel(0,1,0.0000001)
                case 3:
                    pyautogui.moveRel(-1,0,0.0000001)
                case 4:
                    pyautogui.moveRel(1,0,0.0000001)
    
    @staticmethod
    def update_move_direction(moving_flag):
        while True:
            match moving_flag.value:
                case 0:
                    continue
                case 1:
                    keyboard.press_and_release("s")
                case 2:
                    keyboard.press_and_release("w")
                case 3:
                    keyboard.press_and_release("a")
                case 4:
                    keyboard.press_and_release("d")