# Checks that a number in a given position is valid on the sudoku board
def valid_number(board, row, col, num):
    # Check the row for duplicate numbers
    for i in range(9):
        if board[row][i] == num:
            return False

    # Check the column for duplicate numbers
    for i in range(9):
        if board[i][col] == num:
            return False

    # Get the top left corner of the number's 3x3 grid
    corner_row = row - row % 3
    corner_col = col - col % 3

    # Check the 3x3 grid for duplicate numbers
    for i in range(corner_row, corner_row + 3):
        for j in range(corner_col, corner_col + 3):
            if board[i][j] == num:
                return False

    # Returns True if there are no duplicate numbers
    return True


# Backtracking algorithm for solving the sudoku
# Returns True if there is a valid solution, False otherwise
def check_solution(board):
    for i in range(9):
        for j in range(9):
            if board[i][j] == 0:
                for num in range(1, 10):
                    if valid_number(board, i, j, num):
                        board[i][j] = num
                        result = check_solution(board)  # Recursive process
                        if result:
                            return True
                        else:
                            board[i][j] = 0
                return False
    return True


# Calls the backtracking algorithm to find the solution
# Returns the solution if there is one, False otherwise
def solve_sudoku(board):
    if check_solution(board):
        return board
    else:
        return False
