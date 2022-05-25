import cv2

from main import *

cv2.namedWindow('Sudoku Solver')

initial_image = cv2.imread("./test_images/image_1.jpg")

solution = ar_sudoku_solver(initial_image)

if solution is not None:

    cv2.imshow('Sudoku Solver', solution)

    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()

else:
    print("This image does not work properly, please retake the image with better image quality and lighting")