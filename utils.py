import cv2
import numpy as np
import tensorflow as tf


def initialise_model():
    json_file = open('./model/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = tf.keras.models.model_from_json(loaded_model_json)
    model.load_weights("./model/weights.h5")
    return model


# We are assuming that the largest contour of the image is the sudoku board's border
def largest_contour(contours):
    large_contour = None
    max_area = 1000

    for contour in contours:
        contour_area = cv2.contourArea(contour)
        contour_perimeter = cv2.arcLength(contour, closed=True)
        approximate_polygon = cv2.approxPolyDP(contour, 0.02 * contour_perimeter, closed=True)
        if len(approximate_polygon) == 4 and contour_area > max_area:
            large_contour = approximate_polygon
            max_area = contour_area

    if large_contour is not None:
        large_contour = organise_corners(large_contour)

    return large_contour


# Organise the corners anti-clockwise for perspective transformation
def organise_corners(sudoku_contour):
    sudoku_contour = sudoku_contour.reshape((4, 2))

    # Create an empty array which we will set the corner points to in the right order
    organised_corners = np.zeros((4, 1, 2), dtype=np.int32)

    add = sudoku_contour.sum(1)
    # Bottom Left
    organised_corners[0] = sudoku_contour[np.argmin(add)]  # Has smallest (x + y) value
    # Top Right
    organised_corners[3] = sudoku_contour[np.argmax(add)]  # Has largest (x + y) value

    diff = np.diff(sudoku_contour, axis=1)
    # Top Left
    organised_corners[1] = sudoku_contour[np.argmin(diff)]  # Has smallest (x - y) value
    # Bottom Right
    organised_corners[2] = sudoku_contour[np.argmax(diff)]  # Has smallest (x - y) value

    return organised_corners


# Apply perspective transformation so only the sudoku board is displayed on the image
def warp_image(image, organised_corners, image_width, image_height):
    # Input and output points are now in right order (in an anti-clockwise direction)
    input_points = np.float32(organised_corners)
    output_points = np.float32(
        [[0, 0], [image_width - 1, 0], [0, image_height - 1], [image_width - 1, image_height - 1]])

    transformation_matrix = cv2.getPerspectiveTransform(input_points, output_points)
    output_image = cv2.warpPerspective(image, transformation_matrix, (image_width, image_height))
    return output_image


# axis=0 for vertical axis, axis=1 for horizontal axis
def get_lines(image, axis):
    # Specify size on vertical/horizontal axis
    rows = image.shape[axis]
    line_size = rows // 10  # 10 horizontal and vertical lines on a sudoku grid (includes border)

    # Create structure element for extracting vertical/horizontal lines through morphology operations
    if axis == 0:
        line_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, line_size))
    else:
        line_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (line_size, 1))

    # Apply morphology operations
    image = cv2.erode(image, line_structure)
    image = cv2.dilate(image, line_structure)
    return image


# https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html
def hough_line_transform(grid):
    # If rho is any larger some lines come out angled, which is not what we want.
    hough_lines = cv2.HoughLines(grid, 0.3, np.pi / 90, 150)

    # Remove axes of length 1
    hough_lines = np.squeeze(hough_lines)

    if len(hough_lines.shape) < 2:

        return None

    else:

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
            cv2.line(grid, (x1, y1), (x2, y2), (255, 255, 255), 3)

            # We need to invert grid, so we can add the grid and preprocessed warp image together
        # Doing so removes the grid lines (so we only have the numbers)
        inverted_grid = cv2.bitwise_not(grid)
        return inverted_grid


# Splits sudoku grid into 81 evenly sized boxes
def split_image_boxes(number_image):
    boxes = list()
    # Splits image into 9 evenly sized columns
    cols = np.vsplit(number_image, 9)
    for col in cols:
        # Splits column into 9 evenly sized rows (9 boxes)
        rows = np.hsplit(col, 9)
        for row in rows:
            # Add each box to the list of boxes
            boxes.append(row)
    return boxes


# Cleans the images (ie makes boxes with no numbers completely black and centers boxes with numbers)
def clean_number_images(images):
    clean_boxes = list()
    for image in images:
        height, width = image.shape
        mid = width // 2
        if number_image_check(image, height, width, mid):
            clean_boxes.append(False)  # Sets it to False if not a number
        else:
            # Center the number in the box
            contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            x, y, w, h = cv2.boundingRect(contours[0])

            start_x = (width - w) // 2
            start_y = (height - h) // 2

            # Removes all noise from the box (leaving just the number)
            new_image = np.zeros_like(image)
            new_image[start_y:start_y + h, start_x:start_x + w] = image[y:y + h, x:x + w]

            clean_boxes.append(new_image)

    return clean_boxes


def get_num_clues(images):
    counter = 0
    for image in images:
        if type(image) is not bool:
            counter += 1
    return counter


# Checks where the box contains a number or not
def number_image_check(image, height, width, mid):
    # Checks that the majority of the image is black
    if np.isclose(image, 0).sum() / (image.shape[0] * image.shape[1]) >= 0.95:
        return True
    # Checks that the majority of the center of the image is black
    elif np.isclose(image[:, int(mid - width * 0.4):int(mid + width * 0.4)], 0).sum() / (
            2 * width * 0.4 * height) >= 0.925:
        return True
    # If it reaches here then we know that the box must contain a number
    else:
        return False


# Resizes the image so that they are the correct shape for the CNN
def resize_number_images(images, dimension):
    new_images = list()
    for image in images:
        if type(image) is not bool:
            # Image is resized
            image = cv2.resize(image, (dimension, dimension))

            # Image is reshaped so it can be passed into CNN
            image = np.reshape(image, (1, dimension, dimension, 1))
            new_images.append(image)
        else:
            new_images.append(False)
    return new_images


# Gets all the numbers for the sudoku puzzle
def get_sudoku(images, model):
    sudoku = str()
    checker = list()
    # Appends nine lists of nine numbers to the sudoku list
    for j in range(0, len(images), 9):
        sudoku_row = list()
        start = j
        stop = j + 9
        for i in range(start, stop):
            if type(images[i]) is not bool:
                prediction = model.predict(images[i])  # Predicts what the digit is using the CNN
                prediction_value = np.argmax(prediction)
                sudoku += str(prediction_value + 1)
                sudoku_row.append(str(prediction_value + 1))
            else:
                sudoku += str(0)
                sudoku_row.append(str(0))
        checker.append(sudoku_row)
    return sudoku, checker


# Overlay the solution found on the warped image
def overlay_solution(image, solved_puzzle, initial_puzzle, dimension, text_colour):
    for y in range(len(solved_puzzle)):
        for x in range(len(solved_puzzle[y])):
            # Check to make sure the number wasn't already on the sudoku board
            if initial_puzzle[y][x] == "0":
                number = solved_puzzle[y][x]

                # Retrieves the corners and center of the box
                top_left = (x * dimension, y * dimension)
                bottom_right = ((x + 1) * dimension, (y + 1) * dimension)
                center = ((top_left[0] + bottom_right[0]) // 2, (top_left[1] + bottom_right[1]) // 2)

                # Calculates the size and position of there the number will go in the box
                text_size, _ = cv2.getTextSize(number, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 4)
                text_position = (center[0] - text_size[0] // 2, center[1] + text_size[1] // 2)

                # Adds number to the right box on the board
                cv2.putText(image, number, text_position, cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, text_colour, 2)
    return image


def unwarp_image(sudoku_solution_image, original_image, corners, overlay_width, overlay_height, original_width,
                 original_height):
    # Input and output points are now in right order (in an anti-clockwise direction)
    input_points = np.float32(corners)
    output_points = np.float32(
        [[0, 0], [overlay_width - 1, 0], [0, overlay_height - 1], [overlay_width - 1, overlay_height - 1]])

    # Applies the inverse transformation
    transformation_matrix = cv2.getPerspectiveTransform(input_points, output_points)
    unwarped = cv2.warpPerspective(sudoku_solution_image, transformation_matrix, (original_height, original_width),
                                   flags=cv2.WARP_INVERSE_MAP)

    # Combines the original and unwarped image to get the final result
    final_result = np.where(unwarped.sum(axis=-1, keepdims=True) != 0, unwarped, original_image)

    return final_result
