import cv2
import time

from main import *

cap = cv2.VideoCapture(0)

frame_rate = 30
previous_time = 0

while True:
    time_elapsed = time.time() - previous_time
    _, initial_image = cap.read()

    if time_elapsed > 1. / frame_rate:
        previous_time = time.time()

        solved_solution =  ar_sudoku_solver(initial_image)

        if solved_solution is not None:
            cv2.imshow('frame', solved_solution)
        else:
            cv2.imshow('frame', initial_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()