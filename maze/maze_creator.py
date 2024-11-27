import os
from ipywidgets import Button, GridBox, Layout, VBox, Output, HBox
import numpy as np
from IPython.display import display, clear_output

class MazeCreator:
    def __init__(self, rows, cols, out_path="./"):
        """
        Initialize the MazeCreator with grid dimensions and output path.
        
        :param rows: Number of rows in the maze grid.
        :param cols: Number of columns in the maze grid.
        :param out_path: Directory to save the maze files.
        """
        self.rows = rows
        self.cols = cols
        self.grid = np.zeros((rows, cols), dtype=int)  # Maze grid (0: empty, 1: wall)
        self.start_position = None
        self.end_position = None
        self.out_path = out_path
        self.buttons = []
        self.output = Output()

        # Ensure output directory exists
        os.makedirs(self.out_path, exist_ok=True)

        self.init_buttons()
        self.init_control_buttons()
    
    def init_buttons(self):
        """Initialize the grid as a set of toggleable buttons with grid lines."""
        cell_size = "20px"  # Adjust cell size for smaller grid
        self.buttons = [
            Button(
                description="",  # No text on the button
                layout=Layout(width=cell_size, height=cell_size, border="1px solid black"),  # Add grid lines
                style={"button_color": "white"},
            )
            for _ in range(self.rows * self.cols)
        ]

        # Assign toggle functionality
        for i, button in enumerate(self.buttons):
            button.on_click(self.make_toggle_handler(i))
        
        # Create a grid layout
        self.gridbox = GridBox(
            children=self.buttons,
            layout=Layout(
                width=f"{self.cols * 25}px",  # Adjust total grid width
                height=f"{self.rows * 25}px",
                grid_template_columns=f"repeat({self.cols}, {cell_size})",
                grid_template_rows=f"repeat({self.rows}, {cell_size})",  # Ensures uniform row spacing
                grid_gap="0px",  # No space between rows or columns
            ),
        )
    
    def make_toggle_handler(self, index):
        """Create a handler to toggle cell state."""
        def handler(button):
            row, col = divmod(index, self.cols)
            
            # If setting start or end position, mark accordingly
            if self.setting_start:
                if self.start_position:
                    self.buttons[self.start_position[0] * self.cols + self.start_position[1]].style.button_color = "white"
                self.start_position = (row, col)
                button.style.button_color = "green"
                self.setting_start = False
            elif self.setting_end:
                if self.end_position:
                    self.buttons[self.end_position[0] * self.cols + self.end_position[1]].style.button_color = "white"
                self.end_position = (row, col)
                button.style.button_color = "red"
                self.setting_end = False
            else:
                # Toggle cell state for maze walls
                self.grid[row, col] = 1 - self.grid[row, col]
                button.style.button_color = "black" if self.grid[row, col] == 1 else "white"
        return handler
    
    def init_control_buttons(self):
        """Initialize control buttons for setting start/end positions and saving."""
        self.setting_start = False
        self.setting_end = False

        self.start_button = Button(
            description="Set Start",
            layout=Layout(width="100px"),
            style={"button_color": "lightblue"},
        )
        self.start_button.on_click(self.set_start)

        self.end_button = Button(
            description="Set End",
            layout=Layout(width="100px"),
            style={"button_color": "lightcoral"},
        )
        self.end_button.on_click(self.set_end)

        self.save_button = Button(
            description="Save Maze",
            layout=Layout(width="100px"),
            style={"button_color": "lightgreen"},
        )
        self.save_button.on_click(self.save_maze)

        self.control_buttons = HBox([self.start_button, self.end_button, self.save_button])
    
    def set_start(self, button):
        """Enable start position setting mode."""
        self.setting_start = True
        self.setting_end = False
        with self.output:
            print("Click a grid cell to set the start position.")

    def set_end(self, button):
        """Enable end position setting mode."""
        self.setting_start = False
        self.setting_end = True
        with self.output:
            print("Click a grid cell to set the end position.")

    def save_maze(self, button):
        """Save the maze to a .txt file with an incremental index."""
        # Find the next available index for the filename
        existing_files = [f for f in os.listdir(self.out_path) if f.startswith("maze_") and f.endswith(".txt")]
        indices = [
            int(f.split("_")[1].split(".")[0]) for f in existing_files if f.split("_")[1].split(".")[0].isdigit()
        ]
        next_index = max(indices) + 1 if indices else 1
        filename = os.path.join(self.out_path, f"maze_{next_index}.txt")
        
        # Save the grid and positions
        with open(filename, "w") as f:
            np.savetxt(f, self.grid, fmt="%d", delimiter="")
            if self.start_position:
                f.write(f"Start: {self.start_position}\n")
            if self.end_position:
                f.write(f"End: {self.end_position}\n")
        
        with self.output:
            clear_output(wait=True)
            print(f"Maze saved to '{filename}' with start and end positions.")
    
    def display(self):
        """Display the interactive maze grid, control buttons, and output."""
        with self.output:
            display(VBox([self.gridbox, self.control_buttons]))
        display(self.output)
    
    def get_maze(self):
        """Get the current maze grid."""
        return self.grid
