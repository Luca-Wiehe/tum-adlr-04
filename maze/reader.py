# Function to read maze file
def read_maze(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    grid = []
    for line in lines[:-2]:  # Last two lines are start and end
        grid.append([int(ch) for ch in line.strip()])

    # Extract start and end points
    start_line = lines[-2].strip()
    end_line = lines[-1].strip()
    start = tuple(map(int, start_line.split(":")[1].strip(" ()").split(", ")))
    end = tuple(map(int, end_line.split(":")[1].strip(" ()").split(", ")))

    return grid, start, end
