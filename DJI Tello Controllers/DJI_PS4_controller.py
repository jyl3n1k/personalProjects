import sys
import pygame
from djitellopy import Tello
from ultralytics import YOLO
import cv2
import time

pygame.init()
pygame.joystick.init()

if pygame.joystick.get_count() == 0:
    print("No controller found! Connect your PS4 controller and try again.")
    sys.exit()

controller = pygame.joystick.Joystick(0)
controller.init()
print(f" Connected to: {controller.get_name()}")

model = YOLO(r"yolov8n.pt")
DETECT_EVERY = 3
FRAME_COUNT = 0
last_annotated_frame = None
display_frame = None


speed = 40
tello = Tello()
tello.connect(wait_for_state=False)

# Axes
AXIS_L3_X = 0  # Left / Right
AXIS_L3_Y = 1  # Up / Down
AXIS_R3_X = 2  # Left / Right
AXIS_R3_Y = 3  # Up / Down
AXIS_L2   = 4  # Left Analog Trigger
AXIS_R2   = 5  # Right Analog Trigger

# Buttons (Pygame indexes standard PS4 layouts)
BUTTON_CIRCLE   = 1
BUTTON_TRIANGLE = 3

# 4. CONFIGURATION
DEADZONE = 0.4  # Ignore small stick twitches (40% threshold)
clock = pygame.time.Clock()
running = True


while running:
    # Handle event queue for presses/releases
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        # Detect discrete Button Presses
        elif event.type == pygame.JOYBUTTONDOWN:
            if event.button == BUTTON_TRIANGLE:
                tello.takeoff()

            elif event.button == BUTTON_CIRCLE:
                tello.send_rc_control(0, 0, 0, 0)
                tello.land()
                running = False

        elif event.type == pygame.JOYBUTTONUP:
            tello.send_rc_control(0, 0, 0, 0)

    # Read Analog Joystick States
    l3_x = controller.get_axis(AXIS_L3_X)
    l3_y = controller.get_axis(AXIS_L3_Y)
    r3_x = controller.get_axis(AXIS_R3_X)
    r3_y = controller.get_axis(AXIS_R3_Y)

    lr = 0    # left/right
    fb = 0    # forward/backward
    ud = 0    # up/down
    yaw = 0   # rotate left/right

# --- L3 controls yaw and up/down ---
    if l3_x < -DEADZONE:
        yaw = -speed   # rotate left
    elif l3_x > DEADZONE:
        yaw = speed    # rotate right

    if l3_y < -DEADZONE:
        ud = speed     # up
    elif l3_y > DEADZONE:
        ud = -speed    # down

    if r3_x < -DEADZONE:
        lr = -speed    # left
    elif r3_x > DEADZONE:
        lr = speed     # right

    if r3_y < -DEADZONE:
        fb = speed     # forward
    elif r3_y > DEADZONE:
        fb = -speed    # backward

    tello.send_rc_control(lr, fb, ud, yaw)

    l2_val = controller.get_axis(AXIS_L2)
    r2_val = controller.get_axis(AXIS_R2)

    if l2_val > -0.9:
        speed = speed - 2
        if speed < 10:
            speed = 10

    if r2_val > -0.9:
        speed = speed + 2
        if speed > 100:
            speed = 100    

    clock.tick(30)

pygame.quit()
tello.end()
