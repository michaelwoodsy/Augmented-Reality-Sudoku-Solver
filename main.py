import cv2
import copy
# import time

from preprocessing import *
from utils import *
from sudoku import *

def ar_sudoku_solver(image):
    # Initialise timer
    # start_time = time.time()

    # Retrieve the original dimensions of the image
    original_image_width, original_image_height = image.shape[0], image.shape[1]

    # Initialize image width and height (will make perspective transformation easier later on)
    # Dimensions a multiple of 9 so that the image can be split into 81 evenly sized boxes later on
    image_width, image_height = 576, 576

    # Initialise dimension of images in CNN (in current one they are 28x28 pixels)
    model_dimension = 64

    # Initialise colour of text of solution
    number_colour = (0, 0, 255) # Red

    # Initialise the CNN model for digit classification
    model = initialise_model()

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

    if sudoku_contour is not None:

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

        # Record time taken to warp the image
        # warp_time = time.time()
        # print("Time taken to detect the grid is {} seconds".format(warp_time - start_time))

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

        if complete_grid is not None:
            # Create a copy of of preprocessed warped image so that we can add it to the complete_grid
            numbers_only_image = warp_preprocessed_image.copy()

            # Apply bitwise and on numbers_only_image and complete_grid to only get the numbers
            numbers_only_image = cv2.bitwise_and(numbers_only_image, complete_grid)

            # Create copy of numbers_only_image, which will be split into 81 evenly sized images (number of boxes in a sudoku)
            number_image_boxes = numbers_only_image.copy()

            # Divide image with only numbers into 81 evenly sized boxes
            number_image_boxes = split_image_boxes(number_image_boxes)

            # Cleans all the image boxes (ie removes boxes which don't have numbers)
            cleaned_number_images = clean_number_images(number_image_boxes)

            # Resize the boxes for their predictions
            resized_number_images = resize_number_images(cleaned_number_images, model_dimension)

            # Predict a number as a test
            sudoku = get_sudoku(resized_number_images, model)

            # Record time taken to get sudoku numbers
            # sudoku_time = time.time()
            # print("Time taken to detect the sudoku numbers is {} seconds".format(sudoku_time - warp_time))

            # Make a copy of the initial sudoku for when overlaying the solution
            initial_sudoku = copy.deepcopy(sudoku)

            # Solve the sudoku
            solved_sudoku = solve_sudoku(sudoku)

            if type(solved_sudoku) is not bool:

                # Record total time taken
                # solve_time = time.time()
                # print("Time taken to solve sudoku is {} seconds".format(solve_time - sudoku_time))

                # Overlay the solution to the sudoku on the warped image
                overlayed_warped_image = overlay_solution(warped_image, solved_sudoku, initial_sudoku, model_dimension, number_colour)

                # Unwarp the solution onto the original image
                final_solution = unwarp_image(overlayed_warped_image, image, organised_corners, image_width, image_height, original_image_width, original_image_height)

                # Record total time taken
                # total_time = time.time()
                # print("Time taken to detect, solve and overlay sudoku is {} seconds".format(total_time - start_time))

                return final_solution
            
            else:
                return None
        else:
            return None

    else:
        return None