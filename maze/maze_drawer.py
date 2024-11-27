import matplotlib.pyplot as plt
import matplotlib.patches as patches
from IPython.display import display, clear_output

# Constants for visualization
WALL_COLOR = 'black'
EXPLORED_COLOR = 'gray'
PATH_COLOR = 'blue'
START_COLOR = 'green'
END_COLOR = 'red'
EMPTY_COLOR = 'white'

class MazeDrawer:
    def __init__(self, grid, start, end):
        self.grid = grid
        self.start = start
        self.end = end
        self.rows = len(grid)
        self.cols = len(grid[0])
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.ax.set_aspect('equal')
        self.draw_grid()

    def draw_grid(self):
        self.ax.clear()
        self.ax.set_xlim(0, self.cols)
        self.ax.set_ylim(0, self.rows)
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        for row in range(self.rows):
            for col in range(self.cols):
                color = WALL_COLOR if self.grid[row][col] == 1 else EMPTY_COLOR
                self.ax.add_patch(
                    patches.Rectangle((col, self.rows - row - 1), 1, 1, color=color)
                )

        # Draw start and end points
        self.ax.add_patch(
            patches.Rectangle(
                (self.start[1], self.rows - self.start[0] - 1), 1, 1, color=START_COLOR
            )
        )
        self.ax.add_patch(
            patches.Rectangle(
                (self.end[1], self.rows - self.end[0] - 1), 1, 1, color=END_COLOR
            )
        )

    def mark_explored(self, pos):
        if pos != self.start and pos != self.end:
            row, col = pos
            self.ax.add_patch(
                patches.Rectangle((col, self.rows - row - 1), 1, 1, color=EXPLORED_COLOR)
            )
            clear_output(wait=True)  # Clear previous output
            display(self.fig)
            plt.pause(0.1)

    def mark_path(self, path):
        for row, col in path:
            if (row, col) != self.start and (row, col) != self.end:
                self.ax.add_patch(
                    patches.Rectangle((col, self.rows - row - 1), 1, 1, color=PATH_COLOR)
                )
        clear_output(wait=True)  # Clear previous output
        display(self.fig)
        plt.pause(0.1)

    def finalize(self):
        plt.show()
