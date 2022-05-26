# Validates that the board is solvable
def validate_sudoku(board):
    # Iterates through rows
    for i in range(9):
        row = {}
        column = {}
        block = {}
        row_cube = 3 * (i // 3)
        column_cube = 3 * (i % 3)

        # Iterates through columns
        for j in range(9):
            # Checks for duplicates in columns
            if board[i][j] != "0" and board[i][j] in row:
                return False

            row[board[i][j]] = 1
            # Checks for duplicates in columns
            if board[j][i] != "0" and board[j][i] in column:
                return False

            column[board[j][i]] = 1
            rc = row_cube + j // 3
            cc = column_cube + j % 3
            # Checks for duplicates in 3x3 sudoku
            if board[rc][cc] in block and board[rc][cc] != "0":
                return False

            block[board[rc][cc]] = 1
    return True


# Solves board using solve(sudoku) and reformats to nested lists
def get_solution(sudoku):
    solved_solution = solve(sudoku)
    if solved_solution is not False:
        values = list(solved_solution.values())
        return [values[i : i + 9] for i in range(0, len(values), 9)]
    else:
        return solved_solution


# Solve Every Sudoku Puzzle

# See http://norvig.com/sudoku.html

# Throughout this program we have:
# r is a row,    e.g. 'A'
# c is a column, e.g. '3'
# s is a square, e.g. 'A3'
# d is a digit,  e.g. '9'
# u is a unit,   e.g. ['A1','B1','C1','D1','E1','F1','G1','H1','I1']
# grid is a grid,e.g. 81 non-blank chars, e.g. starting with '.18...7...
# values is a dict of possible values, e.g. {'A1':'12349', 'A2':'8', ...}


def cross(A, B):
    "Cross product of elements in A and elements in B."
    return [a + b for a in A for b in B]


digits = "123456789"
rows = "ABCDEFGHI"
cols = digits
squares = cross(rows, cols)
unitlist = (
    [cross(rows, c) for c in cols]
    + [cross(r, cols) for r in rows]
    + [cross(rs, cs) for rs in ("ABC", "DEF", "GHI") for cs in ("123", "456", "789")]
)
units = dict((s, [u for u in unitlist if s in u]) for s in squares)
peers = dict((s, set(sum(units[s], [])) - set([s])) for s in squares)


################ Parse a Grid ################


def parse_grid(grid):
    """Convert grid to a dict of possible values, {square: digits}, or
    return False if a contradiction is detected."""
    # To start, every square can be any digit; then assign values from the grid.
    values = dict((s, digits) for s in squares)
    for s, d in grid_values(grid).items():
        if d in digits and not assign(values, s, d):
            return False  # (Fail if we can't assign d to square s.)
    return values


def grid_values(grid):
    "Convert grid into a dict of {square: char} with '0' or '.' for empties."
    chars = [c for c in grid if c in digits or c in "0."]
    assert len(chars) == 81
    return dict(zip(squares, chars))


################ Constraint Propagation ################


def assign(values, s, d):
    """Eliminate all the other values (except d) from values[s] and propagate.
    Return values, except return False if a contradiction is detected."""
    other_values = values[s].replace(d, "")
    if all(eliminate(values, s, d2) for d2 in other_values):
        return values
    else:
        return False


def eliminate(values, s, d):
    """Eliminate d from values[s]; propagate when values or places <= 2.
    Return values, except return False if a contradiction is detected."""
    if d not in values[s]:
        return values  # Already eliminated
    values[s] = values[s].replace(d, "")
    # (1) If a square s is reduced to one value d2, then eliminate d2 from the peers.
    if len(values[s]) == 0:
        return False  # Contradiction: removed last value
    elif len(values[s]) == 1:
        d2 = values[s]
        if not all(eliminate(values, s2, d2) for s2 in peers[s]):
            return False
    # (2) If a unit u is reduced to only one place for a value d, then put it there.
    for u in units[s]:
        dplaces = [s for s in u if d in values[s]]
        if len(dplaces) == 0:
            return False  # Contradiction: no place for this value
        elif len(dplaces) == 1:
            # d can only be in one place in unit; assign it there
            if not assign(values, dplaces[0], d):
                return False
    return values


################ Search ################


def solve(grid):
    return search(parse_grid(grid))


def search(values):
    "Using depth-first search and propagation, try all possible values."
    if values is False:
        return False  # Failed earlier
    if all(len(values[s]) == 1 for s in squares):
        return values  # Solved!
    # Chose the unfilled square s with the fewest possibilities
    n, s = min((len(values[s]), s) for s in squares if len(values[s]) > 1)
    return some(search(assign(values.copy(), s, d)) for d in values[s])


################ Utilities ################


def some(seq):
    "Return some element of seq that is true."
    for e in seq:
        if e:
            return e
    return False


def solved(values):
    "A puzzle is solved if each unit is a permutation of the digits 1 to 9."

    def unitsolved(unit):
        return set(values[s] for s in unit) == set(digits)

    return values is not False and all(unitsolved(unit) for unit in unitlist)


# References used:
# http://www.scanraid.com/BasicStrategies.htm
# http://www.sudokudragon.com/sudokustrategy.htm
# http://www.krazydad.com/blog/2005/09/29/an-index-of-sudoku-strategies/
# http://www2.warwick.ac.uk/fac/sci/moac/currentstudents/peter_cock/python/sudoku/
