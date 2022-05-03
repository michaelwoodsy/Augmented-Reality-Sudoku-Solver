import cv2
import numpy as np
import tensorflow as tf

from preprocessing import *
from utils import *

cv2.namedWindow('Sudoku Solver')

# Open sudoku image file
image = cv2.imread('sudoku_noise.jpeg')

# Initialize image width and height (will make perspective transformation easier later on)
# Dimensions a multiple of 9 so that the image can be split into 81 evenly sized boxes later on
image_width = 450
image_height = 450

# Initialise dimension of images in CNN (in current one they are 28x28 pixels)
model_dimension = 28

# Initialise the CNN model for digit classification
#model = tf.keras.models.load_model('digit_classification.h5')

# Preprocess the image
preprocessed_image = preprocess(image)

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

# Organise the corners
organised_corners = organise_corners(sudoku_contour)

# Make a copy of the image to perform perspective transformation
warped_image = image.copy()

# Warp the image to the sudoku grid
warped_image = warp_image(warped_image, organised_corners, image_width, image_height)

# Make a copy of the image to perform preprocessing on warped image
warp_preprocessed_image = warped_image.copy()

# Apply preprocessing to warped grid
warp_preprocessed_image = preprocess(warp_preprocessed_image)

# Make two copies of the image to get the horizontal and vertical lines
horizontal_lines = warp_preprocessed_image.copy()
vertical_lines = warp_preprocessed_image.copy()

# Get the horizontal and vertical lines
horizontal_lines = get_lines(horizontal_lines, axis=1)
vertical_lines = get_lines(vertical_lines, axis=0)

# Add vertical and horizontal lines together
grid_image = cv2.add(horizontal_lines, vertical_lines)

# Dilate grid so that the grid lines are larger
preprocess_grid_lines = dilate_grid(grid_image)

# Make a copy of the preprocessed grid so that we can apply Hough Line transformations on it
complete_grid = preprocess_grid_lines.copy()

# Apply Hough Line transformations on preprocessed grid
complete_grid = hough_line_transform(complete_grid)

# Create a copy of of preprocessed warped image so that we can add it to the complete_grid
numbers_only_image = warp_preprocessed_image.copy()

# Apply bitwise and on numbers_only_image and complete_grid to only get the numbers
numbers_only_image = cv2.bitwise_and(numbers_only_image, complete_grid)

# Create copy of numbers_only_image, which will be split into 81 evenly sized images (number of boxes in a sudoku)
number_image_boxes = numbers_only_image.copy()

# Divide image with only numbers into 81 evenly sized boxes
number_image_boxes = split_image_boxes(number_image_boxes)

# Resize the boxes for their predictions
resized_number_images = resize_number_images(number_image_boxes, model_dimension)

# Predict a number as a test
#sudoku = get_sudoku(resized_number_images, model)

cv2.imshow('Sudoku Solver', numbers_only_image)

cv2.waitKey(0)