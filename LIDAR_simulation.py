""" This file is responsible for simulating the measured values from the LIDAR
It is used to prepare data processing upon application of real LIDAR and determine the performance of code
"""
from queue import Queue
import time
import numpy as np
import math
from pprint import pprint
import matplotlib.pyplot as plt


class LIDAR:

    """ The object will contain measurement results as an angle and distance toward obstacle
    The real LIDAR system is scanning the environment and send the results one-by-one, similarly to the queue:
    FIFO (First-In-First-Out)
    """

    measuring_radius: int # radius is given in meters
    measurement_results: Queue # queue of tuples (angle, distance)

    def __init__(self, radius: int):
        self.measuring_radius = radius
        self.measurement_results = Queue()

    def __str__(self):
        text = f'LIDAR:\n' \
               f'Scanning radius: {self.measuring_radius}\n' \
               f'Measurements Queue: ({self.get_values()})\n'
        return text

    def get_values(self) -> list:
        start_time = time.time()
        data = []
        while not self.measurement_results.empty():
            data.append(self.measurement_results.get())
        print('='*5+f'LIDAR DATA RECEIVED {time.time()-start_time}'+'='*5)
        return data

    def get_scanning_area(self, grid: np.array, current_node: tuple) -> np.array:
        """
        The function returns the coordinates of the matrix points
        which are inside the circle of the measuring radius
        :return 2D matrix with the coordinates of [[x, y],...]
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
        coordinates = self.get_scanning_area(grid, current_location)
        for coor in coordinates:
            i, j = coor[0], coor[1]
            if grid[i, j] == 1:
                distance = np.sqrt((current_location[0] - i)**2 + (current_location[1] - j)**2)
                angle = self.get_obstacle_vector(current_node=current_location, next_node=(i, j))
                print(i, j)
                record = (angle, distance)
                self.measurement_results.put(record)
        print('ENVIRONMENT HAS BEEN SCANNED')

    def show_scanning_area(self, grid: np.array, skeleton: np.array, current_node: tuple) -> None:
        area: np.array = self.get_scanning_area(grid, current_node)
        plt.imshow(grid, origin='lower')
        plt.imshow(skeleton, cmap='Greys', origin='lower', alpha=0.7)
        for coor in area:
            plt.plot(coor[1], coor[0], 'gx')
        # show the current position
        plt.plot(current_node[1], current_node[0], 'rx')
        plt.xlabel('EAST')
        plt.ylabel('NORTH')
        plt.show()
        """ Show the scanning area on the figure"""

    @staticmethod
    def get_obstacle_vector(next_node: tuple, current_node: tuple) -> float:
        """
        Returns arctan between two points in radians
        """
        y_diff = next_node[1] - current_node[1]
        x_diff = next_node[0] - current_node[0]
        angle = math.atan2(y_diff, x_diff)
        print(f'Rotation angle is {angle}')
        return angle
