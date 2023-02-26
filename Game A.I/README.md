An A.I. for a modified version of the game Teeko.

The rules for this version of the game are largely the same except we exchanged the 2x2 box winning condition for a 3x3 box winning condition.

The winning conditions are summarized below:

> Four markers of the same color in a row horizontally, vertically, or diagonally.
> Four markers of the same color in the four corners of a 3x3 box.

1) make_move(self, state) - takes in the current state of the game and returns a list of move tuples in the format of:
   [(row, col), (src_row, src_col)]
   
2) randomMove(self, state) - similar to above, except it returns some random move.

3) succ(self, state) - returns a list of legal successors.

4) eval_succ(self, successors) - evaluates each of the given successors, and returns a list of their corresponding immediate values.

5) heuristic_game_value(self, state) - analyzes a given state of the game and determines if it's non-terminal, that is a losing state or a non-winning one.

6) max_value(self, state, depth, player) - minimax function that determines a given states utility. Each recursive call increases the value of depth, terminating the recursion if either it reaches the tested depth limit or finds a terminal state.

7)

8)

9)

10)