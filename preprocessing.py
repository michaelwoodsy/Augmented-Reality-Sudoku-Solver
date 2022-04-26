import cv2

def initial_preprocess(image):
    # Convert the image to greyscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur
    gaussian_blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply adaptive thresholding
    thresholding = cv2.adaptiveThreshold(gaussian_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Invert the image
    inverted_image = cv2.bitwise_not(thresholding)

    return inverted_image

def board_preprocess(image):
    # Convert the image to greyscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur
    gaussian_blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply adaptive thresholding
    thresholding = cv2.adaptiveThreshold(gaussian_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Invert the image
    inverted_image = cv2.bitwise_not(thresholding)

    return inverted_image