""" This file is responsible for simulating the measured values from the LIDAR
It is used to prepare data processing upon application of real LIDAR and determine the performance of code
"""
from queue import Queue
import time
import numpy as np
import math
from pprint import pprint
import matplotlib.pyplot as plt


def get_obstacle_vector(next_node: tuple, current_node: tuple) -> float:
    """
    Returns arctan between two points as radians
    """
    y_diff = next_node[0] - current_node[1]
    x_diff = next_node[1] - current_node[0]
    angle: float = math.atan2(y_diff, x_diff)
    return angle


class LIDAR:

    """ The object will contain measurement results as an angle and distance toward obstacle
    The real LIDAR system is scanning the environment and send the results one-by-one, similarly to the queue:
    FIFO (First-In-First-Out)
    """

    measuring_radius: int # radius is given in meters
    measurement_results: Queue # queue of tuples (angle: float, distance: float)

    def __init__(self, radius: int):
        self.measuring_radius = radius
        self.measurement_results = Queue()

    def __str__(self):
        text = f'LIDAR:\n' \
               f'Scanning radius: {self.measuring_radius}\n' \
               f'Measurements Queue: ({self.get_values()})\n'
        return text

    def get_values(self) -> list:
        """ Assumption: LIDAR provides measurements one-by-one in the format:
        angle, distance for each point
        The values are appended to the list for storage and easier data validation.
        :returns [(angle1, distance1), (angle2, distance2)]
        """
        start_time = time.time()
        data = []
        while not self.measurement_results.empty():
            data.append(self.measurement_results.get())
        print('='*5+f'LIDAR DATA RECEIVED {time.time()-start_time}'+'='*5)
        data.sort() # sort by angle in ascending order
        return data

    def get_scanning_area(self, grid: np.array, current_node: tuple) -> np.array:
        """
        The function returns the coordinates of the matrix points
        which are inside the circle of the measuring radius
        :return 2D matrix with the coordinates of [[x1, y1], [x2, y2], ...]
        """
        # Create an array of indices for the grid matrix
        indices = np.indices(grid.shape)
        # Calculate the distances from the initial coordinates (x, y) to all cells in the grid
        distances = np.sqrt((indices[0] - current_node[0])**2 + (indices[1] - current_node[1])**2)
        # Find the indices of cells within the specified radius
        indices_within_radius = np.where(distances <= self.measuring_radius)
        # Get the coordinates of the cells within the radius
        coordinates_within_radius = np.transpose(indices_within_radius)
        print('SCANNING AREA COORDINATES')
        pprint(coordinates_within_radius)
        return coordinates_within_radius

    def scan(self, grid: np.array, current_location: tuple) -> None:
        """ The purchased LIDAR has a limit of scanning distance
        We may assume that any measurements received from the LIDAR are the location of obstacles withing scanning area
        Otherwise, freeway is assumed if no measurement detected for specific angle
        This function is only simulating which data will be passed to the system
        Initial angle is given in degrees
        """
        coordinates = self.get_scanning_area(grid, current_location)
        for coor in coordinates:
            x, y = coor[0], coor[1] # get the coordinates x and y
            if grid[x, y] == 1: # if obstacle is detected
                distance: float = np.sqrt((current_location[0] - x)**2 + (current_location[1] - y)**2)
                angle:    float = get_obstacle_vector(current_node=current_location, next_node=(y, x))
                record = (angle, distance) # compress the data into tuple
                self.measurement_results.put(record)
        print('ENVIRONMENT HAS BEEN SCANNED')

    def show_scanning_area(self, grid: np.array, skeleton: np.array, current_node: tuple) -> None:
        area: np.array = self.get_scanning_area(grid, current_node)
        # expand area to fit grid size
        normalized_area = np.zeros((grid.shape[0], grid.shape[1]))
        for coor in area:
            x, y = coor[0], coor[1]
            normalized_area[x][y] = 1
        plt.imshow(grid, origin='lower')
        plt.imshow(skeleton, cmap='Greys', origin='lower', alpha=0.7)
        plt.imshow(normalized_area, origin='lower', alpha=0.5)
        # show the current position
        plt.plot(current_node[1], current_node[0], 'rx')
        plt.xlabel('EAST')
        plt.ylabel('NORTH')
        plt.title('Scanning range of LIDAR')
        plt.show()
