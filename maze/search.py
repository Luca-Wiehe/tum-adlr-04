import heapq

# A-Star Search with exploration tracking
def a_star_search(grid, start, end, drawer):
    rows, cols = len(grid), len(grid[0])
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # right, down, left, up

    def is_valid(x, y):
        return 0 <= x < rows and 0 <= y < cols and grid[x][y] == 0

    def manhattan_distance(x1, y1, x2, y2):
        return abs(x1 - x2) + abs(y1 - y2)

    open_list = []
    heapq.heappush(open_list, (0, start))  # (priority, (x, y))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: manhattan_distance(*start, *end)}
    explored = set()

    while open_list:
        _, current = heapq.heappop(open_list)
        if current in explored:
            continue
        explored.add(current)

        # Visualize explored tiles
        drawer.mark_explored(current)

        if current == end:
            # Reconstruct path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path = path[::-1]

            # Visualize the path
            drawer.mark_path(path)
            return path

        for dx, dy in directions:
            neighbor = (current[0] + dx, current[1] + dy)
            if not is_valid(*neighbor) or neighbor in explored:
                continue

            tentative_g_score = g_score[current] + 1
            if tentative_g_score < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + manhattan_distance(*neighbor, *end)
                if neighbor not in [i[1] for i in open_list]:
                    heapq.heappush(open_list, (f_score[neighbor], neighbor))

    return None  # No path found