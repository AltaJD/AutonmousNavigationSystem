import threading

from map import Map
import numpy as np
from lidar_simulation import get_vector_angle
from typing import List
from math import cos, sin
import matplotlib.pyplot as plt
import config


class VFH:
    desired_direction: float    # represents the angle from current node to target node
    steering_direction: float
    measurements: List[tuple]
    histogram: np.array         # histogram represented as np.array with a shape (n, 0)
    threshold: float            # in % specifies the minimum obstacle probability acceptable as obstacle-free path
    steering_direction: int
    a: int
    b: int
    alpha: int
    l: int

    def __init__(self, threshold: float,
                 a, b, alpha, l_param):
        """
        :param threshold:
        :param lidar_measurements:
        :param a:
        :param b:
        :param alpha:
        :param l_param:
        """
        self.threshold = threshold
        self.alpha = alpha
        self.a = a
        self.b = b
        self.l = l_param
        # set up figure instance
        fig, (ax1, ax2) = plt.subplots(2, 1)
        self.fig = fig
        self.ax1 = ax1
        self.ax2 = ax2

    def update_measurements(self, values: List[tuple]):
        self.measurements = values

    def get_rotation_angle(self, current_node: tuple, next_node: tuple) -> int:
        """ Get the angle with the lowest probability of the obstacles ahead
        :param next_node provides the data for choosing the most appropriate steering direction
        :param current_node is the (x, y) of current location of the LIDAR/Wheelchair
        :return angle in degrees
        """
        self.generate_vfh()
        # determining angle that is the closest to the target point
        obstacle_free_sectors: List[int] = np.where(self.histogram <= self.threshold)[0]
        assert len(obstacle_free_sectors) > 0
        # merge neighbor sectors
        merged_sectors: List[list] = []
        wide_sector = []
        for i in range(len(obstacle_free_sectors) - 1):
            current_value = obstacle_free_sectors[i]
            next_value = obstacle_free_sectors[i+1]
            difference = next_value - current_value
            if difference == 1:
                wide_sector.append(current_value)
                if (i + 1) != len(obstacle_free_sectors) - 1:
                    continue
                else:
                    wide_sector.append(next_value)
            if len(wide_sector) != 0:
                # append the last value
                wide_sector.append(current_value)
                merged_sectors.append(wide_sector)
                wide_sector = []
        desired_angle = get_vector_angle(next_node=next_node, current_node=current_node) * 180 / np.pi
        # select the angle closest to the desired angle
        if desired_angle < 0:
            desired_angle += 360
        desired_angle = np.floor(desired_angle/config.get('sector_angle'))
        self.desired_direction = desired_angle
        minimums_diff = list(map(lambda x: abs(x-desired_angle), obstacle_free_sectors))
        best_angle = obstacle_free_sectors[minimums_diff.index(min(minimums_diff))]
        # convert angle to degrees
        best_angle *= config.get('sector_angle')
        self.steering_direction = best_angle
        return best_angle

    def generate_vfh(self) -> None:
        """ The function will generate the Vector Field Histogram
        The magnitude of the obstacle is represented as a formula:
        m[i][j] = (c[i][j])**2 * (a - b * d[i][j])
        where:
        1. c[i][j] = certainty value
        2. d[i][j] = distances to each obstacle
        3. a and b are positive constants, where a = b*max(distance)
        4. m[i][j] will be a one dimensional array consisting of values greater than 0

        alpha is an angle division that should be in degrees
        :returns np.array with shape: (n, 0)
        """
        c: np.array = self.get_certainty_values()
        magnitudes = self.get_magnitudes(c, self.b)
        self.histogram = self.get_sectors(magnitudes, self.alpha)
        # histogram is updated inplace
        self.smooth_histogram(l=self.l)
        self.normalize_histogram()

    def get_certainty_values(self) -> np.array:
        """
        The function determine the location of the obstacles and create virtual AxA map of their locations,
        where A is a measuring distance as an int
        The measurements include the coordinates (angle, distance) of the obstacles detected by the virtual LIDAR
        :returns the list with 1+distance: int and 0, where 1+distance is a certainty value of obstacle
        otherwise, the [0] is returned
        """
        obstacle_magnitudes = []
        for value in self.measurements:
            # angle in radians
            # distance in meters
            angle, distance = value
            # assign 1+distance for obstacle and 0 as freeway
            # the certainty value based on the distance is assigned to the obstacle map
            obstacle_magnitudes.append(1 + round(distance))
        if len(obstacle_magnitudes) > 0:
            return np.array(obstacle_magnitudes)
        return np.zeros(1, dtype=np.float)

    def get_sectors(self, magnitudes: np.array, alpha: float) -> np.array:
        """ The function creates the grid as np.array of the angles to the obstacle
        The measurements include the coordinates (angle, distance) of the obstacles detected by the virtual LIDAR
        :returns: polar obstacle density as np.array, dtype=int
        """
        # determine polar obstacle density h_k, where k is each sector divided by alpha
        h_k = []  # List[int]
        m_sum = 0  # sum of magnitudes per sector
        # set initial/starting angle
        starting_angle: float = self.measurements[0][0] * 180 / np.pi  # in degrees
        for i in range(len(self.measurements)):
            beta: float = self.measurements[i][0]  # get angle in radiance
            beta = beta * 180 / np.pi  # convert radiance to degrees
            # determine sum of the magnitudes for particular sector
            if beta - starting_angle <= alpha:
                m_sum += magnitudes[i]  # if the magnitude is in the sector, we add it to the sum
            else:
                h_k.append(round(m_sum, 0))  # store the sector cumulative magnitude
                m_sum = 0  # reset sum
                starting_angle = beta  # set the new starting angle for the sector in degrees
        # expand the angle ranges from 0 to 360 degrees
        initial_angle: float = self.measurements[0][0] * 180 / np.pi  # degrees
        if initial_angle < 0:
            initial_angle += 360  # convert to the positive sign angle
        sectors_num = int(360 / alpha)  # number of sectors
        initial_sector = int(initial_angle / alpha)  # starting sector
        normalized_histogram = self.expand_histogram(angles=h_k, max_size=sectors_num, starting_index=initial_sector)
        return np.array(normalized_histogram)

    def smooth_histogram(self, l: int) -> None:
        """ Initial mapping may appear ragged and cause errors in the selection of the steering direction
        The function suggest to consider smooth the graph for better angle selection
        :parameter
        h is a polar obstacle density
        l is a constant integer, chosen by experiment or simulation.
        For this system, l is a tolerance for steering direction
        :returns: smoothed polar obstacle density as List[float]
        """
        hist = np.array(self.histogram, dtype=int)
        smoothed_data = []
        # cumulative = 0 # cumulative sum
        for i in range(len(hist)):
            start_index = max(0, i - l + 1)
            end_index = min(len(hist), i + l)
            window = hist[start_index:end_index]
            smoothed_value = sum(window) / len(window)
            smoothed_data.append(smoothed_value)
        self.histogram = smoothed_data

    def normalize_histogram(self) -> None:
        highest_magnitude = max(self.histogram)
        self.histogram = np.array(
            [magnitude / highest_magnitude if highest_magnitude != 0 else 0 for magnitude in self.histogram])

    @staticmethod
    def get_magnitudes(certainty_values: np.array, b: int, a=None) -> np.array:
        """ The magnitudes are calculated using: m[i][j] = (c[i][j])**2 * (a - b * d[i][j])
        The distances are included in the certainty values as (d[i][j] = c[i][j]-1)
        """
        # since the distance is included in the certainty values, it can be extracted as follows:
        d: np.array = certainty_values.copy()  # distances
        for i in range(d.shape[0]):
            cell_value = d[i]
            if cell_value > 0:
                d[i] = cell_value - 1
        if a is None:
            a = b * max(d)
        return (certainty_values ** 2) * (a - b * d)  # m[i][j]

    @staticmethod
    def expand_histogram(angles: List[int], max_size: int, starting_index: int) -> List[int]:
        """ Considering that the histogram may have a size of 5 or 5*alpha degrees range
        It should be expanded to 360 degrees to be considered for VFH algorithm
        The function inserts the angles list to the resulting list from starting index
        :returns the list of size max_size (36) and including the angles list from the starting index
        """

        def shift_to_right(values: List[int], size: int) -> List[int]:
            """ If indexes > size => shift to the beginning of the list
            Example: [0, 0, value3]
            size = 2 => [value3, 0]
            """
            for i in range(size, len(values)):
                values[i - size] = values[i]
            return values

        def fill_to_right(values: List[int], size: int) -> List[int]:
            """ If indexes < size => expand the list up to max size
            Example: [value1, value2]; size = 3 => [value1, value2, 0]
            """
            values = values + [0] * (size - len(values))
            return values

        if len(angles) == max_size:
            # if no insertion is required:
            return angles
        # insertion process
        """ Shift the indexes by starting index 
        Example:
        starting_index = 1 => [0, value1, value2, ...]
        starting_index = 2 => [0, 0, value1, value2, ...]
        """
        expanded_list = list(np.zeros(max_size))[:starting_index] + angles
        if len(expanded_list) < max_size:
            expanded_list = fill_to_right(expanded_list, max_size)
        else:
            expanded_list = shift_to_right(expanded_list, max_size)
        expanded_list = expanded_list[:max_size]
        assert len(expanded_list) == max_size
        return expanded_list

    def show_histogram(self, current_node=None, target_node=None, env=None) -> None:
        """
        The function is using plt to plot the histogram and
        the map of the environment as subplots
        The current location is indicated as a cross
        The steering direction is indicated as a blue vector passed in degrees
         """
        # get x axis for histogram
        x = np.arange(self.histogram.shape[0]) * config.get('sector_angle')
        # plot histogram
        self.ax1.bar(x, self.histogram)
        self.ax1.set_xlabel('Angle (in degrees)')
        self.ax1.set_ylabel('Probability')
        self.ax1.set_title('Vector Field Histogram')
        if env is not None and current_node is not None: self.show_map(self.ax2, env, current_node)
        if target_node is not None:
            plt.plot(target_node[1], target_node[0], 'gx')
            # show 0 angle vector
            plt.quiver(current_node[1], current_node[0], sin(0) * 1, cos(0) * 1, color='r')
            # show the steering direction
            plt.quiver(current_node[1], current_node[0],
                       sin(np.radians(self.steering_direction)),
                       cos(np.radians(self.steering_direction)),
                       color='b')
        # adjusting spacing between subplots
        plt.tight_layout()
        # Update the plot
        plt.pause(1)  # Add a small delay (e.g., 0.1 seconds)

    def get_obstacle_map(self, measuring_distance: int):
        """
        The function determine the location of the obstacles and create virtual AxA map of their locations,
        where A is a measuring distance as an int
        The measurements include the coordinates (angle, distance) of the obstacles detected by the virtual LIDAR
        The current location in (y, x) is coordinates of the wheelchair according to the map
        The grid with 1+distance: int and 0, where 1+distance is a certainty value of obstacle
        """
        # TODO: increase accuracy.
        # create obstacle_grid simulation with zeros by default
        obstacle_grid = np.zeros((measuring_distance + 1, measuring_distance + 1))
        for value in self.measurements:
            # angle in radians
            # distance in meters
            angle, distance = value
            # get the location of the obstacle on the map
            location_y = round(distance * cos(angle))
            location_x = round(distance * sin(angle))
            # assign 1+distance for obstacle and 0 as freeway
            # the certainty value based on the distance is assigned to the obstacle map
            obstacle_grid[location_y][location_x] = 1 + round(distance)
        return obstacle_grid

    def show_obstacle_map(self, measuring_distance: int) -> None:
        plt.imshow(self.get_obstacle_map(measuring_distance), cmap='Greys', origin='lower', alpha=0.7)
        plt.show()

    @staticmethod
    def show_map(ax2, env: Map, current_node: tuple) -> None:
        # plot grid
        ax2.imshow(env.grid, origin='lower')
        ax2.imshow(env.skeleton, cmap='Greys', origin='lower', alpha=0.7)
        ax2.set_xlabel('North')
        ax2.set_ylabel('East')
        ax2.set_title('Environment map')
        # show the current location
        ax2.plot(current_node[1], current_node[0], 'rx')
        ax2.figure.set_size_inches(10, 10)


def test_simulation_lidar(start: tuple, filename: str, safety_distance: int, goal: tuple) -> None:
    # test with simulation Lidar
    from lidar_simulation import LidarSimulation
    map_2d = Map(filename=filename, size=12, safety_distance=safety_distance)
    lidar_simulation = LidarSimulation(radius=config.get('lidar_radius'))
    lidar_simulation.scan(grid=map_2d.grid, current_location=start)
    vfh = VFH(a=config.get('a'),
              b=config.get('b'),
              alpha=config.get('sector_angle'),
              l_param=config.get('l'),
              threshold=config.get('vfh_threshold'))
    vfh.update_measurements(lidar_simulation.get_values())
    vfh.get_rotation_angle(current_node=start, next_node=goal)
    vfh.show_histogram(current_node=start, env=map_2d)


if __name__ == '__main__':
    """ Retrieve testing data """
    start_default:      tuple   = config.get('initial_position')
    file_name:          str     = config.get('colliders')
    safety_distance:    int     = config.get('safety_distance')
    goal_default:       tuple   = config.get('final_position')

    test_simulation_lidar(start_default, file_name, safety_distance, goal_default)
