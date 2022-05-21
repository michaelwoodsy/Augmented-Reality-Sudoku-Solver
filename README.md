# COSC428 Project
This is a sudoku solver in which users can input an image of an empty sudoku board 
and the program will generate a solution overlaying the original image.

## How to run the program
To choose the image you would like the program to solve, change the name of the image file on the following line in main.py
```python
image = cv2.imread("./test_images/sudoku_test.jpeg")
```

Run the following command in the root directory:
```bash
python3 main.py
```

## How to run the OCR model
Run the following command in the model directory:
```bash
python3 model.py
```

## How to add your own images to use
Add the image you would like the program to solve in the test_images directory.
Then, as mentioned above, change the name of the image file on the following line in main.py
```python
image = cv2.imread("./test_images/sudoku_test.jpeg")
```