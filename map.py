import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import medial_axis
from skimage.util import invert
from behavioral_model import Action
import time
from typing import List
from queue import PriorityQueue
from tqdm import tqdm
import csv
import sys


def valid_destination(grid: np.array, start: tuple, goal: tuple) -> bool:
    if grid[start[0], start[1]] != 0:
        print(f'INVALID start position. {grid[start[0], start[1]]}', file=sys.stderr)
        return False
    elif grid[goal[0], goal[1]] != 0:
        print(f'INVALID goal position. {grid[goal[0], goal[1]]}', file=sys.stderr)
        return False
    return True


def valid_actions(grid, current_node):
    """
    Returns a list of valid_actions actions given a grid and current node.
    """
    valid = [Action.UP, Action.LEFT, Action.RIGHT, Action.DOWN]
    n, m = grid.shape[0] - 1, grid.shape[1] - 1 # max is 921, 921
    x, y = current_node
    # check if the node is off the grid or
    # it's an obstacle
    if x - 1 < 0 or grid[x-1, y] == 1:
        valid.remove(Action.UP)
    if x + 1 > n or grid[x+1, y] == 1:
        valid.remove(Action.DOWN)
    if y - 1 < 0 or grid[x, y-1] == 1:
        valid.remove(Action.LEFT)
    if y + 1 > m or grid[x, y+1] == 1:
        valid.remove(Action.RIGHT)
    return valid


def normalize_grid(grid: np.array) -> None:
    now_time = time.time()
    # contrary, if the grid consists of the height of obstacles, we may consider height above threshold as an obstacle
    threshold: int = 0
    progress_bar = tqdm(total=len(grid)*len(grid[0]))
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            progress_bar.update()
            if grid[i, j] > threshold:
                grid[i, j] = 1 # 1 indicates the existence of the unavoidable static obstacle
            else:
                grid[i, j] = 0
    time_diff = time.time()-now_time
    progress_bar.close()
    print('='*5+f'Normalization time: {time_diff}'+'='*5)


def save_grid(grid_filename, grid: np.array) -> None:
    with open(grid_filename,'w+') as file:
        wr = csv.writer(file) #, quoting=csv.QUOTE_ALL)
        wr.writerows(grid)
        file.close()
    print(f'GRID HAS BEEN SAVED TO {grid_filename}')


def create_grid(data, safe_distance):
    """
    Create a 2.5D grid from given obstacle data.
    """
    # minimum and maximum north coordinates
    north_min = np.floor(np.min(data[:, 0] - data[:, 3]))
    north_max = np.ceil(np.max(data[:, 0] + data[:, 3]))

    # minimum and maximum east coordinates
    east_min = np.floor(np.min(data[:, 1] - data[:, 4]))
    east_max = np.ceil(np.max(data[:, 1] + data[:, 4]))

    # given the minimum and maximum coordinates we can
    # calculate the size of the grid.
    north_size = int(np.ceil((north_max - north_min + 1)))
    east_size = int(np.ceil((east_max - east_min + 1)))

    # Initialize an empty grid
    grid = np.zeros((north_size, east_size))

    # Populate the grid with obstacles
    for i in range(data.shape[0]):
        north, east, alt, d_north, d_east, d_alt = data[i, :]
        obstacle = [
            int(np.clip(north - d_north - safe_distance - north_min, 0, north_size - 1)),
            int(np.clip(north + d_north + safe_distance - north_min, 0, north_size - 1)),
            int(np.clip(east - d_east - safe_distance - east_min, 0, east_size - 1)),
            int(np.clip(east + d_east + safe_distance - east_min, 0, east_size - 1)),
        ]
        obs = grid[obstacle[0]:obstacle[1] + 1, obstacle[2]:obstacle[3] + 1]
        np.maximum(obs, np.ceil(alt + d_alt + safe_distance), obs)

    return grid, int(north_min), int(east_min)


def heuristic_func(position, goal_position):
    return np.sqrt((position[0] - goal_position[0])**2 + (position[1] - goal_position[1])**2)


def a_star(grid, h, start, goal):
    """
    Here we have implemented A* search with the help of a priority queue.
    """
    queue = PriorityQueue()
    queue.put((0, start))
    visited = set(start)

    branch = {}
    found = False
    # TODO: add animation for path calculation

    while not queue.empty():
        item = queue.get()
        current_cost = item[0]
        current_node = item[1]

        if current_node == goal:
            print('Found a path.')
            found = True
            break
        else:
            for action in valid_actions(grid, current_node):
                # get the tuple representation
                da = action.delta
                cost = action.cost
                next_node = (current_node[0] + da[0], current_node[1] + da[1])
                new_cost = current_cost + cost + h(next_node, goal)
                if next_node not in visited:
                    visited.add(next_node)
                    queue.put((new_cost, next_node))

                    branch[next_node] = (new_cost, current_node, action)
    path = []
    path_cost = 0
    if found:

        # retrace steps
        path = []
        n = goal
        path_cost = branch[n][0]
        while branch[n][1] != start:
            path.append(branch[n][2])
            n = branch[n][1]
        path.append(branch[n][2])

    return path[::-1], path_cost


def get_path(grid, start, goal):
    start_time = time.time()
    path = a_star(grid=grid, h=heuristic_func, start=start, goal=goal)
    # path = path_prune(path, collinear_points)
    # path = path_simplify(grid=grid, path=path)
    time_taken = time.time() - start_time
    time_taken = round(time_taken, 2) # round to the 2 digits after comma
    print("--- %s seconds ---" % time_taken)
    return path


def create_custom_map(filename: str, safety_distance=0):
    """=== Function consider the Start and Goal position of North and East coordinates ===
    Start = Goal = (North, East)
    """
    plt.rcParams['figure.figsize'] = 12, 12
    # getting obstacle data
    data = np.loadtxt(filename, delimiter=',', dtype='Float64', skiprows=2)
    grid, north_offset, east_offset = create_grid(data, safety_distance)
    print("North offset = {0}, east offset = {1}".format(north_offset, east_offset))

    skeleton, distances = medial_axis(invert(grid), return_distance=True)
    return grid, skeleton, distances, data


def show_map(grid, skeleton, start, goal, path=None):
    # plot the edges on top of the grid along with start and goal locations
    plt.imshow(grid, origin='lower')
    plt.imshow(skeleton, cmap='Greys', origin='lower', alpha=0.7)

    if path is not None:
        pp = np.array(path)
        plt.plot(pp[:, 1], pp[:, 0], 'g')

    plt.plot(start[1], start[0], 'rx')
    plt.plot(goal[1], goal[0], 'rx')

    plt.xlabel('EAST')
    plt.ylabel('NORTH')
    plt.show()


def reformat_path(path: list, n_set, e_set):
    """ The path represented in the form of Action objects are converted to
    the list of waypoint with x,y coordinates.
    This approach significantly simplify path visualization
    """
    time_started = time.time()
    waypoints = []
    new_n = n_set
    new_e = e_set
    progress_bar = tqdm(total=len(path))
    for i in range(len(path)):
        p = path[i].value
        new_n += p[0]
        new_e += p[1]
        new_coordinate = [new_n, new_e]
        waypoints.append(new_coordinate)
        progress_bar.update(n=1)
    progress_bar.close()
    print('='*5+f'Path reformation: {time.time()-time_started}'+'='*5)
    return waypoints


def read_grid(file_path, dtype) -> np.array:
    """ The function is reading the csv file to download 2D matrix
    This can be used for testing different normalizations and format of the maps
    """
    data = np.loadtxt(file_path, delimiter=',', dtype=dtype)
    return data
