from djitellopy import Tello
from pynput import keyboard
import cv2
import time

tello = Tello()
tello.connect(wait_for_state=False)

speed = 60

def on_press(key):
    if hasattr(key, "char"):
        if key.char == 't':
            tello.takeoff()

        elif key.char == 'w':
            tello.send_rc_control(0, 0, speed, 0)

        elif key.char == 's':
            tello.send_rc_control(0, 0, -speed, 0)

        elif key.char == 'a':
            tello.send_rc_control(0, 0, 0, -speed)

        elif key.char == 'd':
            tello.send_rc_control(0, 0, 0, speed)

        elif key.char == 'q':
            tello.send_rc_control(0, 0, 0, 0)
            tello.land()
            return False

    else:
        if key == keyboard.Key.space:
            tello.send_rc_control(0, 0, 0, 0)
            tello.land()

        elif key == keyboard.Key.up:
            tello.send_rc_control(0, speed, 0, 0)

        elif key == keyboard.Key.down:
            tello.send_rc_control(0, -speed, 0, 0)

        elif key == keyboard.Key.left:
            tello.send_rc_control(-speed, 0, 0, 0)

        elif key == keyboard.Key.right:
            tello.send_rc_control(speed, 0, 0, 0)


def on_release(key):
    if hasattr(key, "char"):
        if key.char in ['w', 's', 'a', 'd']:
            tello.send_rc_control(0, 0, 0, 0)

    else:
        if key in [
            keyboard.Key.up,
            keyboard.Key.down,
            keyboard.Key.left,
            keyboard.Key.right
        ]:
            tello.send_rc_control(0, 0, 0, 0)

#listener = keyboard.Listener(on_press=on_press, on_release=on_release)
#listener.start()

#running = True

#while running:
#    frame_read = tello.get_frame_read()
#    img = frame_read.frame   
    
#    if img is not None:
#        cv2.imshow("Tello Video", img)

#    if cv2.waitKey(1) & 0xFF == 27:
#        running = False
#        tello.send_rc_control(0, 0, 0, 0)
#        tello.land()
#        break

with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    listener.join()

#listener.stop()
#tello.streamoff()
tello.end()
#cv2.destroyAllWindows()