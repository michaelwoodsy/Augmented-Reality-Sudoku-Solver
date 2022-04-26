import cv2
import numpy as np

from preprocessing import *
from utils import *

cv2.namedWindow('Sudoku Solver')


while True:
    # Open sudoku image file
    image = cv2.imread('sudoku_background_noise.jpeg')

    # Initialize image width and height (will make perspective transformation easier later on)
    image_width = 500
    image_height = 500

    # Preprocess the image
    preprocessed_image = initial_preprocess(image)

    # Make a copy of the image to draw the contours on
    contour_image = image.copy()

    # Find all the contours on the preprocessed image
    contours, _ = cv2.findContours(preprocessed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw all the contours onto the contour_image
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)

    # Find the largest contour on the contour_image, and use it as the Sudoku board
    sudoku_contour = largest_contour(contours)

    # Make a copy of the image to draw the largest_contour on
    sudoku_corner_image = image.copy()

    # Draw the largest contour on largest_contour_image
    cv2.drawContours(sudoku_corner_image, sudoku_contour, -1, (0, 0, 255), 5)

    # Organise the corners (TODO)
    organised_corners = organise_corners(sudoku_contour)

    # Make a copy of the image to perform perspective transformation
    warped_image = image.copy()

    # Warp the image to the sudoku grid
    warped_image = warp_image(warped_image, organised_corners, image_width, image_height)

    # Apply preprocessing to warped grid (TODO)

    cv2.imshow('Sudoku Solver', warped_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()