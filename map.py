import numpy as np
import matplotlib.pyplot as plt
import pandas.errors
from skimage.morphology import medial_axis
from skimage.util import invert
from behavioral_model import Action
import time
from queue import PriorityQueue
from tqdm import tqdm
import csv
import sys
from math import sin, cos, radians
from typing import List
import pandas as pd
import config


class Path:
    start: tuple
    end: tuple
    waypoints: np.array

    def __init__(self, current_location: tuple, destination: tuple):
        self.start = current_location
        self.end = destination

    def a_star(self, va):  # PATH
        """
        Here we have implemented A* search with the help of a priority queue.
        :parameter va is a function returning valid actions at certain position
        """
        queue = PriorityQueue()
        queue.put((0, self.start))
        visited = set(self.start)
        branch = {}
        found = False
        while not queue.empty():
            item = queue.get()
            current_cost = item[0]
            current_node = item[1]
            if current_node == self.end:
                print('Found a path.')
                found = True
                break
            else:
                for action in va(current_node):
                    # get the tuple representation
                    da = action.delta
                    cost = action.cost
                    next_node = (current_node[0] + da[0], current_node[1] + da[1])
                    new_cost = current_cost + cost + self.heuristic_func(next_node, self.end)
                    if next_node not in visited:
                        visited.add(next_node)
                        queue.put((new_cost, next_node))
                        branch[next_node] = (new_cost, current_node, action)
        path = []
        path_cost = 0
        if found:
            # retrace steps
            path = []
            n = self.end
            path_cost = branch[n][0]
            while branch[n][1] != self.start:
                path.append(branch[n][2])
                n = branch[n][1]
            path.append(branch[n][2])
        return path[::-1], path_cost

    def animate_path(self,
                     grid: np.array,
                     skeleton: np.array,
                     start=None, goal=None,
                     animation_speed=10) -> None:
        """ The function is showing the figure with animated path
        The path is represented as a list of waypoints and coordinates on the map
        """
        # Set up the figure and axis
        fig, ax = plt.subplots()
        """ Center according to the grid """
        # define figure dimensions
        ax.set_xlim(0, grid.shape[0])
        ax.set_ylim(0, grid.shape[1])
        line, = ax.plot([], [], marker='o')

        if start is not None:
            ax.plot(start[1], start[0], 'rx')
        if goal is not None:
            ax.plot(goal[1], goal[0], 'rx')

        # Plot the grid
        ax.imshow(grid, origin='lower')
        ax.imshow(skeleton, cmap='Greys', origin='lower', alpha=0.7)
        # Plot the path
        ax.plot(self.waypoints[:, 1], self.waypoints[:, 0], 'g')

        # Define the update function
        def update(frame):
            y = self.waypoints[:frame + 1, 0]
            x = self.waypoints[:frame + 1, 1]
            line.set_data(x, y)
            return line,

        # Display the animation
        plt.show()

    def reformat_path(self, path: list) -> np.array:
        """ The path represented in the form of Action objects are converted to
        the list of waypoint with x,y coordinates.
        This approach significantly simplify path visualization
        """
        waypoints = []
        new_n = self.start[0]
        new_e = self.start[1]
        progress_bar = tqdm(total=len(path))
        for i in range(len(path)):
            p = path[i].value
            new_n += p[0]
            new_e += p[1]
            new_coordinate = [new_n, new_e]
            waypoints.append(new_coordinate)
            progress_bar.update(n=1)
        progress_bar.close()
        return np.array(waypoints)

    @staticmethod
    def heuristic_func(position, goal_position):
        return np.sqrt((position[0] - goal_position[0]) ** 2 + (position[1] - goal_position[1]) ** 2)


class MapIsland:
    id: int
    name: str
    coordinates: np.array  # [[x, y], [x, y]]
    destination: List[int]  # [x, y]
    text_location: List[int]  # [x, y]

    def __init__(self, id: int, text_location: List[int]):
        self.id = id
        self.name = ""
        self.coordinates = np.zeros([2, 2])
        self.text_location = text_location[::-1]

    def set_name(self, new_name: str):
        self.name = new_name

    def set_destination(self, new_coordinate: List[int]):
        self.destination = new_coordinate

    def set_text_location(self, new_location: List[int]):
        self.text_location = new_location

    def update_coordinates(self, coordinates: np.array):
        self.coordinates = coordinates

    def show_island(self):
        assert self.coordinates is not None, "ERROR: coordinates are not updated"
        x = [coord[1] for coord in self.coordinates]
        y = [coord[0] for coord in self.coordinates]
        plt.plot(x, y, 'bo')
        plt.text(self.text_location[0], self.text_location[1], self.name, fontsize=config.get('font_size'))
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Plot of Island Coordinates')
        plt.show()


class Map:
    grid: np.array
    skeleton: np.array
    size: int  # size of the image
    safe_d: int  # safety distance
    north_offset: int
    east_offset: int
    path: Path
    start: tuple
    end: tuple
    islands: List[MapIsland]

    def __init__(self, filename: str, size: int, safety_distance=0):
        """=== Function consider the Start and Goal position of North and East coordinates ===
        Start = Goal = (North, East)
        """
        self.size = size
        self.safe_d = safety_distance
        # getting obstacle data
        obstacles_data = np.loadtxt(filename, delimiter=',', dtype='Float64', skiprows=2)
        self.grid, self.north_offset, self.east_offset = self.create_grid(obstacles_data)
        self.create_skeleton()
        self.normalize_grid()
        self.islands = []
        try:
            self.get_islands()
        except pandas.errors.EmptyDataError:
            print(config.get("islands_data_path"), " is empty and has been skipped.")

    def select_start(self) -> tuple:
        sp = self.select_point(title="Select current position")
        if not self.valid_destination(sp, "Start"): raise Exception
        self.start = sp
        return self.start

    def select_end(self) -> tuple:
        threshold = 1
        selected_coord = self.select_point(start=self.start, title="Select Destination")
        if self.valid_destination(coordinate=selected_coord):
            self.end = selected_coord
            return self.end
        return self.get_island_destination(selected_destination=list(selected_coord),
                                           threshold=threshold)

    def create_path(self, start, end) -> np.array:
        start_time = time.time()
        self.path = Path(current_location=start, destination=end)
        path, waypoints = self.path.a_star(self.valid_actions)
        # path = path_prune(path, collinear_points)
        # path = path_simplify(grid=grid, path=path)
        if len(path) == 0:
            print('Path was not found.', file=sys.stderr)
            raise Exception
        print("--- %s seconds ---" % round(time.time() - start_time), 2)
        self.path.waypoints = self.path.reformat_path(path)

    def select_point(self, start=None, goal=None, title='') -> tuple:
        """ Show the 2D environment map for selection of coordinates by mouse click
        Passed parameters start and goal will be displayed on the map if passed
        :returns (y, x) coordinates of the map
        """
        # Display the figure
        fig, ax = plt.subplots()
        ax.set_xlim(0, self.grid.shape[1])
        ax.set_ylim(0, self.grid.shape[0])
        # Plot the grid
        ax.imshow(self.grid, origin='lower')
        ax.imshow(self.skeleton, cmap='Greys', origin='lower', alpha=0.7)
        # Plot start or goal locations
        if start is not None:
            ax.plot(start[1], start[0], 'rx')
        if goal is not None:
            ax.plot(goal[1], goal[0], 'rx')
        for island in self.islands:
            plt.text(island.text_location[0],
                     island.text_location[1],
                     island.name,
                     fontsize=config.get('font_size'))
        # Show
        plt.show(block=False)
        plt.title(title)
        # Wait for the user to click on the figure
        points = plt.ginput(n=1, timeout=-1)
        # Close the figure
        plt.close(fig)
        # convert to the proper format
        location = (round(points[0][1]), round(points[0][0]))
        return location

    def create_skeleton(self) -> None:
        self.skeleton = medial_axis(invert(self.grid))

    def create_grid(self, data):
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
                int(np.clip(north - d_north - self.safe_d - north_min, 0, north_size - 1)),
                int(np.clip(north + d_north + self.safe_d - north_min, 0, north_size - 1)),
                int(np.clip(east - d_east - self.safe_d - east_min, 0, east_size - 1)),
                int(np.clip(east + d_east + self.safe_d - east_min, 0, east_size - 1)),
            ]
            obs = grid[obstacle[0]:obstacle[1] + 1, obstacle[2]:obstacle[3] + 1]
            np.maximum(obs, np.ceil(alt + d_alt + self.safe_d), obs)
        print("North offset = {0}, east offset = {1}".format(north_min, east_min))
        return grid, int(north_min), int(east_min)

    def add_obstacles(self, n: int) -> None:
        """ The function add the obstacle locations to the main map
        and save the new map to the csv file.
        It allows to test the obstacle avoidance system and A* algorithm,
        in case no maneuvers are available
        """
        # Get the dimensions of the grid
        indices = np.argwhere(self.grid == 0)

        # If there are fewer available indices than the desired number of obstacles, return the grid as is
        if len(indices) < n:
            return None

        # Randomly select n indices from the available indices
        selected_indices = np.random.choice(len(indices), n, replace=False)
        selected_coordinates: np.array = indices[selected_indices]

        x_offset, y_offset = self.grid.shape[1], self.grid.shape[0]

        # Set the selected indices to 1
        for coord in selected_coordinates:
            x, y = coord[0], coord[1]
            if x + 1 >= x_offset or y + 1 >= y_offset:
                continue
            self.grid[(x + 1, y + 1)] = 1
            self.grid[(x + 1, y)] = 1
            self.grid[(x, y + 1)] = 1
            self.grid[(x, y)] = 1

    def get_island_coordinates(self, seed_point, threshold) -> np.array:
        # Create a mask to store the regions where the algorithm has expanded
        mask = np.zeros_like(self.grid, dtype=np.uint8)

        # Set up a queue to store the points to be expanded
        queue = []
        queue.append(seed_point)

        # Start the growing algorithm
        while len(queue) > 0:
            # Get the next point from the queue
            current_point = queue.pop(0)
            x = current_point[0]
            y = current_point[1]
            # Check if the current point is within the grid boundaries
            if (0 <= y < self.grid.shape[1]) and (0 <= x < self.grid.shape[0]):
                # Check if the current point is already expanded or not
                if mask[x, y] == 0:
                    # Check if the current point has a value of 1
                    if self.grid[x, y] == threshold:
                        # Expand the region by setting the pixel value and updating the mask
                        mask[x, y] = threshold

                        # Add the neighboring points to the queue for further expansion
                        queue.append((x - 1, y))  # Left neighbor
                        queue.append((x + 1, y))  # Right neighbor
                        queue.append((x, y - 1))  # Top neighbor
                        queue.append((x, y + 1))  # Bottom neighbor

        # Find the coordinates of the regions with only 1s
        segment_coordinates = np.argwhere(mask == 1)
        return segment_coordinates

    def add_island(self) -> None:
        # select the object we want to remember
        init_coordinates = self.select_point(title="Select the object")
        threshold = 1
        # get all coordinates
        island_coordinates = self.get_island_coordinates(seed_point=init_coordinates, threshold=threshold)
        print("Island has been detected")
        assert len(island_coordinates) > 0, "ERROR, island has not been found.\n"
        # store the island details
        island_name = input("Provide the name of the building / room: ")
        island_id = len(self.islands)
        island = MapIsland(id=island_id, text_location=list(init_coordinates))
        island.set_name(island_name)
        self.islands.append(island)
        # save islands
        compress_islands_data = [[island.id, island.name, island.text_location[0], island.text_location[1]]
                                 for island in self.islands]
        islands_df = pd.DataFrame(data=compress_islands_data, columns=config.get("islands_file_headers"))
        islands_df.to_csv(config.get("islands_data_path"), header=True)  # mode="a" prevent overwriting file

    def get_islands(self):
        df = pd.read_csv(config.get("islands_data_path"))
        # store the islands in the object
        for i in range(df.shape[0]):
            il_detail = df.iloc[i]
            text_loc = [il_detail["Coordinate_y"], il_detail["Coordinate_x"]]
            island = MapIsland(id=il_detail["ID"], text_location=text_loc)
            island_coordinates = self.get_island_coordinates(text_loc, threshold=1)
            island.set_name(new_name=il_detail["Name"])
            island.update_coordinates(coordinates=island_coordinates)
            self.islands.append(island)

    def get_island_destination(self, selected_destination: List[int], threshold: int) -> tuple:
        # get the destination coordinates
        coordinates = self.get_island_coordinates(seed_point=selected_destination, threshold=threshold)
        # convert coordinates to the list of the strings
        coordinates = list(map(lambda x: str(x[0])+','+str(x[1]), coordinates))
        print("Island has been detected")
        assert len(coordinates) > 0, "ERROR, island has not been found.\n"
        # check to which island the coord belongs
        df = pd.read_csv(config.get("islands_data_path"), usecols=config.get("islands_file_headers"))
        df['Coordinate'] = df.apply(lambda x: str(x["Coordinate_y"])+','+str(x["Coordinate_x"]), axis=1)
        df["Detected"] = np.where((df["Coordinate"].isin(coordinates)), 1, 0)
        detected_islands = df[df["Detected"] == 1].iloc[0]
        print(detected_islands)
        island = self.islands[detected_islands["ID"]]
        island.update_coordinates(coordinates)
        self.end = island.coordinates[0]
        return self.end

    def normalize_grid(self) -> None:
        """ Convert the grid of 1s and 0s """
        now_time = time.time()
        # contrary, if the grid consists of the height of obstacles, we may consider height above threshold as an obstacle
        threshold: int = 0
        progress_bar = tqdm(total=len(self.grid) * len(self.grid[0]))
        for i in range(len(self.grid)):
            for j in range(len(self.grid[i])):
                progress_bar.update()
                if self.grid[i, j] > threshold:
                    self.grid[i, j] = 1  # 1 indicates the existence of the unavoidable static obstacle
                else:
                    self.grid[i, j] = 0
        time_diff = time.time() - now_time
        progress_bar.close()
        print('=' * 5 + f'Normalization time: {time_diff}' + '=' * 5)

    def valid_destination(self, coordinate: tuple, coordinate_name="") -> bool:
        """ Determine if the selected start and goal positions are valid for creating a path """
        if self.grid[coordinate[0], coordinate[1]] != 0:
            return False
        return True

    def valid_actions(self, current_node):
        """
        Returns a list of valid_actions actions given a grid and current node.
        """
        valid = [Action.UP, Action.LEFT, Action.RIGHT, Action.DOWN]
        n, m = self.grid.shape[0] - 1, self.grid.shape[1] - 1  # max is 921, 921
        x, y = current_node
        # check if the node is off the grid or
        # it's an obstacle
        if x - 1 < 0 or self.grid[x - 1, y] == 1:
            valid.remove(Action.UP)
        if x + 1 > n or self.grid[x + 1, y] == 1:
            valid.remove(Action.DOWN)
        if y - 1 < 0 or self.grid[x, y - 1] == 1:
            valid.remove(Action.LEFT)
        if y + 1 > m or self.grid[x, y + 1] == 1:
            valid.remove(Action.RIGHT)
        return valid

    def save_map(self, grid_filename: str, skeleton_filename=None):
        """ Store the map into csv file """
        with open(grid_filename, 'w+') as file:
            wr = csv.writer(file)  # quoting=csv.QUOTE_ALL)
            wr.writerows(self.grid)
            file.close()
        if skeleton_filename is not None:
            with open(skeleton_filename, 'w+') as file:
                wr = csv.writer(file)
                wr.writerows(self.skeleton)
                file.close()
        print(f'GRID HAS BEEN SAVED TO {grid_filename}')

    def read_grid(self, file_path, dtype) -> np.array:
        """ The function is reading the csv file to download 2D matrix
        This can be used for testing different normalizations and format of the maps
        """
        self.grid = np.loadtxt(file_path, delimiter=',', dtype=dtype)
        return self.grid

    def read_skeleton(self, file_path, dtype) -> np.array:
        """ The function is reading the csv file to download 2D matrix
        This can be used for testing different normalizations and format of the maps
        """
        self.skeleton = np.loadtxt(file_path, delimiter=',', dtype=dtype)
        return self.skeleton

    def show_map(self, start=None, initial_vector=None, goal=None, path=None, save_path=None, title="") -> None:
        """ Plot the graph using matplotlib to show objects based on the parameters
        If parameter is not provided, it will not be displayed on the map
        Parameter initial vector is a tuple (angle, length) of the vector
        :returns None
        """
        # plot the edges on top of the grid along with start and goal locations
        plt.rcParams['figure.figsize'] = self.size, self.size
        plt.imshow(self.grid, origin='lower')
        plt.imshow(self.skeleton, cmap='Greys', origin='lower', alpha=0.7)

        if path is not None:
            self.show_path()

        if start is not None:
            plt.plot(start[1], start[0], 'rx')
            angle, length = 0, 10  # in degrees and meters
            if initial_vector is not None:
                angle, length = initial_vector
            vector_x = start[0] + cos(radians(angle)) * length
            vector_y = start[1] + sin(radians(angle)) * length
            plt.quiver(start[1], start[0], vector_y, vector_x, color='b')
            # show angle difference on the map
            plt.quiver(start[1], start[0], sin(0) * length, cos(0) * length, color='r')
        if goal is not None:
            plt.plot(goal[1], goal[0], 'rx')
        plt.title(title)
        plt.xlabel('EAST')
        plt.ylabel('NORTH')
        if save_path is not None:
            plt.savefig(save_path)
            print('=' * 5 + f'MAP HAS BEEN SAVED TO {save_path}' + '=' * 5)
        plt.show()

    def show_path(self):
        """ The function shows two different paths to estimate performance """
        plt.imshow(self.grid, origin='lower')
        plt.imshow(self.skeleton, cmap='Greys', origin='lower', alpha=0.7)
        # show path 1
        plt.plot(self.path.waypoints[:, 1], self.path.waypoints[:, 0], 'r')
        plt.show()

    def show_islands(self):
        # show obstacles
        plt.imshow(self.grid, origin='lower')
        plt.imshow(self.skeleton, cmap='Greys', origin='lower', alpha=0.7)
        # show islands
        for island in self.islands:
            plt.text(island.text_location[0],
                     island.text_location[1],
                     island.name,
                     fontsize=config.get('font_size'))
        plt.title("Objects selected")
        plt.show()


if __name__ == '__main__':
    start_default: tuple    = config.get('initial_position')
    safety_distance: int    = config.get('safety_distance')
    filename: str           = config.get('colliders')

    testing_map = Map(filename, 10, safety_distance)
    testing_map.start = start_default
    testing_map.add_island()
    testing_map.select_end()
    # testing_map.show_islands()

    # start = map.select_start()
    # end = map.select_end()
    # map.create_path(start, end)
    # map.show_path()
