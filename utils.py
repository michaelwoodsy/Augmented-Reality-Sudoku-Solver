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
    output_points = np.float32([[0, 0],[image_width - 1, 0], [0, image_height - 1],[image_width - 1, image_height - 1]])

    transformation_matrix = cv2.getPerspectiveTransform(input_points, output_points)
    output_image = cv2.warpPerspective(image, transformation_matrix, (image_width, image_height))
    return output_image

# axis=0 for vertical axis, axis=1 for horizontal axis
def get_lines(image, axis):
    # Specify size on vertical/horizontal axis
    rows = image.shape[axis]
    linesize = rows // 10 # 10 horizontal and vertical lines on a sudoku grid (includes border)
    
    # Create structure element for extracting vertical/horizontal lines through morphology operations
    if axis == 0:
        lineStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, linesize))
    else:
        lineStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (linesize, 1))
    
    # Apply morphology operations
    image = cv2.erode(image, lineStructure)
    image = cv2.dilate(image, lineStructure)
    return image

# Retrieved from https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html
def hough_line_transform(grid):
    # If rho is any larger some lines come out angled, which is not what we want.
    hough_lines = cv2.HoughLines(grid, 0.3, np.pi / 90, 200)
    
    # Removes axes of length 1
    hough_lines = np.squeeze(hough_lines)

    for rho, theta in hough_lines:
        # find out where the line stretches to and draw them
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * a)
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * a)

        # Draws a line of thickness four and colour white on the image
        cv2.line(grid, (x1, y1), (x2, y2), (255, 255, 255), 4) 

    # We need to invert grid so we can add the grid and preprocessed warp image together
    # Doing so removes the grid lines (so we only have the numbers)
    inverted_grid = cv2.bitwise_not(grid)
    return inverted_grid

# Splits sudoku grid into 81 evenly sized boxes
def split_image_boxes(number_image):
    boxes = list()
    # Splits image into 9 evenly sized rows
    rows = np.hsplit(number_image, 9)
    for row in rows:
        # Splits row into 9 evenly sized columns (9 boxes)
        columns = np.vsplit(row, 9)
        for col in columns:
            # Add each box to the list of boxes
            boxes.append(col)
    return boxes