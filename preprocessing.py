import cv2
import numpy as np

def preprocess(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur
    gaussian_blur = cv2.GaussianBlur(gray, (9, 9), 0)

    # Apply adaptive thresholding
    thresholding = cv2.adaptiveThreshold(gaussian_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Invert the image
    inverted_image = cv2.bitwise_not(thresholding)

    # Get a rectangular kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

    # Morph to remove noise (like the random dots)
    morphology = cv2.morphologyEx(inverted_image, cv2.MORPH_OPEN, kernel)

    # Dilate the image
    dilated_image = cv2.dilate(morphology, kernel, iterations=1)

    return dilated_image

# Dilate the grid to make it larger
def dilate_grid(grid):
    # Get a rectangular kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

    # Dilate the image
    dilated_image = cv2.dilate(grid, kernel, iterations=1)

    return dilated_image