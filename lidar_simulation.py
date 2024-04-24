""" This file is responsible for simulating the measured values from the LIDAR
It is used to prepare data processing upon application of real LIDAR and determine the performance of code
"""
from queue import Queue
import config
import numpy as np
import matplotlib.pyplot as plt
from common_functions import get_distance, get_vector_angle


class LidarSimulation:

    """ The object will contain measurement results as an angle and distance toward obstacle
    The real LIDAR system is scanning the environment and send the results one-by-one, similarly to the queue:
    FIFO (First-In-First-Out)
    """

    measuring_radius: int  # radius is given in meters
    measurement_results: Queue  # queue of tuples (angle: float, distance: float)
    start_blind_spot: int
    end_blind_spot: int
    current_angle: float  # current direction of the lidar
    blind_spot_overflow: bool

    def __init__(self, radius: int, direction: float):
        self.measuring_radius = radius
        self.measurement_results = Queue()
        self.current_angle = direction
        self.start_blind_spot = config.get('blind_spot_range')[0]
        self.end_blind_spot   = config.get('blind_spot_range')[1]
        self.blind_spot_overflow = False

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
        data = []
        while not self.measurement_results.empty():
            data.append(self.measurement_results.get())
        data.sort() # sort by angle in ascending order
        if len(data) == 0:
            return []
        return data

    def update_blind_spot(self) -> None:
        if not self.blind_spot_overflow:
            delta = self.end_blind_spot - self.start_blind_spot
        else:
            delta = (360 - self.end_blind_spot) + self.start_blind_spot
        self.end_blind_spot += self.current_angle
        if self.end_blind_spot < 0:
            self.end_blind_spot += 360
        if self.end_blind_spot > 360:
            self.end_blind_spot %= 360
        self.start_blind_spot += self.current_angle
        if self.start_blind_spot < 0:
            self.start_blind_spot += 360
        if self.start_blind_spot > 360:
            self.start_blind_spot %= 360

        if self.end_blind_spot < self.start_blind_spot:
            self.swap_blind_spot_range()

        if self.end_blind_spot - self.start_blind_spot != delta:
            self.blind_spot_overflow = True
        else:
            self.blind_spot_overflow = False

    def swap_blind_spot_range(self) -> None:
        var = self.end_blind_spot
        self.end_blind_spot = self.start_blind_spot
        self.start_blind_spot = var

    def get_scanning_area(self, grid: np.array, current_node: tuple) -> np.array:
        """
        The function returns the coordinates of the matrix points
        which are inside the circle of the measuring radius
        :return 2D matrix with the coordinates of [[x1, y1], [x2, y2], ...]
        """
        # Create an array of indices for the grid matrix
        indices = np.indices(grid.shape)
        # Calculate the distances from the initial coordinates (x, y) to all cells in the grid
        distances = np.sqrt((indices[0] - current_node[0])**2 + (indices[1] - current_node[1])**2)  # FIXME: overflow encountered
        # Find the indices of cells within the specified radius
        indices_within_radius = np.where(distances <= self.measuring_radius)
        # Get the coordinates of the cells within the radius
        coordinates_within_radius = np.transpose(indices_within_radius)
        return coordinates_within_radius

    def scan(self, grid: np.array, current_location: tuple) -> None:
        """ The purchased LIDAR has a limit of scanning distance
        We may assume that any measurements received from the LIDAR are the location of obstacles withing scanning area
        Otherwise, freeway is assumed if no measurement detected for specific angle
        This function is only simulating which data will be passed to the system
        Initial angle is given in degrees
        """
        coordinates = self.get_scanning_area(grid, current_location)
        self.update_blind_spot()
        for coord in coordinates:
            x, y = coord[0], coord[1]  # get the coordinates x and y
            if grid[x, y] == 1:  # if obstacle is detected
                distance: float = get_distance(current_node=current_location, next_node=(x, y))
                angle:    float = get_vector_angle(current_node=current_location, next_node=(x, y))
                angle += np.radians(self.current_angle)  # consider the current direction of the lidar
                record = (angle, distance)  # compress the data into tuple
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
