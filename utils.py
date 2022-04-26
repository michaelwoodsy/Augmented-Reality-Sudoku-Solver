import cv2
import numpy as np

# We are assuming that the largest contour of the image is the sudoku board's border
def largest_contour(contours):
    largest_contour = None
    max_area = 0

    for contour in contours:
        contour_area = cv2.contourArea(contour)
        contour_perimeter = cv2.arcLength(contour, closed=True)
        approximate_polygon = cv2.approxPolyDP(contour, 0.02 * contour_perimeter, closed=True)
        if len(approximate_polygon) == 4 and contour_area > max_area:
            largest_contour = approximate_polygon
            max_area = contour_area

    return largest_contour

# Organise the corners anti-clockwise for perspective transformation
def organise_corners(sudoku_contour):
    sudoku_contour = sudoku_contour.reshape((4, 2))

    # Create an empty array which we will set the corner points to in the right order
    organised_corners = np.zeros((4, 1, 2), dtype=np.int32)

    add = sudoku_contour.sum(1)
    # Bottom Left
    organised_corners[0] = sudoku_contour[np.argmin(add)] # Has smallest (x + y) value
    # Top Right
    organised_corners[3] =sudoku_contour[np.argmax(add)] # Has largest (x + y) value

    diff = np.diff(sudoku_contour, axis=1)
    # Top Left
    organised_corners[1] =sudoku_contour[np.argmin(diff)] # Has smallest (x - y) value
    # Bottom Right
    organised_corners[2] = sudoku_contour[np.argmax(diff)] # Has smallest (x - y) value

    return organised_corners

# Apply perspective transformation so only the sudoku board is displayed on the image
def warp_image(image, organised_corners, image_width, image_height):
    # Input and output points are now in right order (in an anti-clockwise direction)
    input_points = np.float32(organised_corners)
    output_points = np.float32([[0, 0],[image_width, 0], [0, image_height],[image_width, image_height]])

    transformation_matrix = cv2.getPerspectiveTransform(input_points, output_points)
    output_image = cv2.warpPerspective(image, transformation_matrix, (image_width, image_height))
    return output_image