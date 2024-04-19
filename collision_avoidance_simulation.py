from map import Map
import numpy as np
from common_functions import get_vector_angle, convert_to_degrees
from typing import List, Tuple
from math import cos, sin
import matplotlib.pyplot as plt
import config


class VFH:
    desired_direction: float  # represents the angle from current node to target node
    steering_direction: float
    measurements: List[tuple]  # first value is a distance and second is an angle in tuple
    free_sectors_num: List[Tuple[int, float]]  # number of obstacle-free sectors to evaluate performance of LIDAR
    histogram: np.array  # histogram represented as np.array with a shape (n, 0)
    threshold: float  # in % specifies the minimum obstacle probability acceptable as obstacle-free path
    safety_distance: float
    steering_direction: int
    a: float
    b: float
    c: int  # certainty value
    alpha: int
    l: int

    def __init__(self, safety_distance: float, b, alpha, l_param, a=None, keep_images=False):
        """
        Sets the parameters for the VFH generation
        :param safety_distance: min distance to the obstacle in m (0.1 = 10 cm)
        :param a: constant for determining obstacle probability
        :param b: constant for determining obstacle probability
        :param alpha: width of the sector in degrees
        :param l_param: constant for smoothing histogram
        """
        self.safety_distance = safety_distance
        self.threshold = config.get("vfh_threshold")
        self.alpha = alpha
        self.a = a
        self.b = b
        self.c = 1
        self.l = l_param
        self.free_sectors_num = [(0, 0.0)]  # [(num_of_sectors, time_in_ms)]
        self.measurements = []
        self.histogram = np.zeros(int(360 / self.alpha))  # by default it contains only zeros
        # set up figure instance
        if keep_images:
            fig, (ax1, ax2) = plt.subplots(2, 1)
            self.fig = fig
            self.ax1 = ax1
            self.ax2 = ax2

    def update_measurements(self, values: List[tuple]):
        self.measurements = values

    def update_free_sectors_num(self, num: int, time: float):
        last_time = self.free_sectors_num[-1][1]  # get time
        time += last_time
        self.free_sectors_num.append((num, time))

    def get_free_sectors_num(self) -> int:
        return np.count_nonzero(self.histogram == 0)

    def get_rotation_angle(self, current_node: tuple, next_node: tuple) -> int:
        """ Get the angle with the lowest probability of the obstacles ahead
        :param next_node provides the data for choosing the most appropriate steering direction
        :param current_node is the (x, y) of current location of the LIDAR/Wheelchair
        :return angle in degrees
        """
        # determining angle that is the closest to the target point
        obstacle_free_sectors: List[int] = np.where(self.histogram <= self.threshold)[0]
        if len(obstacle_free_sectors) <= 0:
            return -1
        # merge neighbor sectors
        merged_sectors: List[list] = []
        wide_sector = []
        for i in range(len(obstacle_free_sectors) - 1):
            current_value = obstacle_free_sectors[i]
            next_value = obstacle_free_sectors[i + 1]
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
        desired_angle = np.floor(desired_angle / config.get('sector_angle'))
        self.desired_direction = desired_angle
        minimums_diff = list(map(lambda x: abs(x - desired_angle), obstacle_free_sectors))
        best_angle = obstacle_free_sectors[minimums_diff.index(min(minimums_diff))]
        # convert angle to degrees
        best_angle *= config.get('sector_angle')
        self.steering_direction = best_angle
        return best_angle

    def get_magnitude(self, d: float) -> float:
        # d represents distance to the obstacle
        m: float = (self.c ** 2) * (self.a - self.b * d)  # magnitude
        if m < 0:
            m = 0
        assert m >= 0, "Magnitude is a negative number"
        return m

    def fill_histogram(self) -> None:
        """ Array is filled according to the angle sector """
        if len(self.measurements) == 0:
            # if no obstacles are detected, all probabilities are zero
            return None
        highest_distance = max(self.measurements, key=lambda x: x[1])[1]
        assert highest_distance > 0, "HIGHEST DISTANCE IS 0"
        if self.a is None:
            self.a = self.b * highest_distance
        for i in range(len(self.measurements)):
            angle: float = self.measurements[i][0]  # radians
            angle = convert_to_degrees(angle)  # degrees
            distance: float = self.measurements[i][1]  # meters
            sector: int = round(np.floor(angle / self.alpha))
            if self.histogram[sector] != 0:
                self.c += 1
            else:
                self.c = 1  # reset to 1
            m = self.get_magnitude(distance)
            self.histogram[sector] += m

    def neglect_angles(self, angles: tuple, overflow: bool) -> None:
        """ The function set probability of 1 (max) to the sectors, which should be automatically neglected
        :param angles is a tuple representing (from, to) range of the angles to be neglected.
        :param overflow is a bool responsible for detecting whether range (30, 270) is:
        from 30 to 270 with blind spot width of 240 degrees
        or
        from 270 to 30 with blind spot width of 120 degrees
        Both angles are included
        """
        start_sector = int(np.floor(angles[0] / self.alpha))
        end_sector = int(np.floor(angles[1] / self.alpha))
        if not overflow:
            for angle in range(start_sector, end_sector, self.alpha):
                self.histogram[angle] = 1  # set max probability
        else:
            for angle in range(end_sector, int(360 / self.alpha), self.alpha):
                self.histogram[angle] = 1  # set max probability
            for angle in range(0, start_sector, self.alpha):
                self.histogram[angle] = 1  # set max probability

    def generate_vfh(self, blind_spot_range=None, blind_spot_overflow=None) -> None:
        """ The function will generate the Vector Field Histogram
        The magnitude of the obstacle is represented as a formula:
        m[i][j] = (c[i][j])**2 * (a - b * d[i][j])
        where:
        1. c[i][j] = certainty value
        2. d[i][j] = distances to each obstacle
        3. a and b are positive constants, where a = b*max(distance) if a is not provided
        4. m[i][j] will be a one dimensional array consisting of values greater than 0

        alpha is an angle division that should be in degrees
        :param blind_spot_range is a tuple (from, to) to set the max probability

        Parameters used for simulation testing only:
        :param blind_spot_overflow is a bool responsible for detecting whether range (30, 270) is:
        from 30 to 270 with blind spot width of 240 degrees
        or
        from 270 to 30 with blind spot width of 120 degrees
        :returns np.array with shape: (n, 0)
        """
        self.empty_histogram()
        self.fill_histogram()
        self.smooth_histogram(l=self.l)
        self.normalize_histogram()
        if blind_spot_range is not None and type(blind_spot_range) == tuple:
            self.neglect_angles(angles=blind_spot_range,
                                overflow=blind_spot_overflow)

        self.detect_danger(observation_range=config.get("observation_range"))

    def detect_danger(self, observation_range: list) -> None:
        """ The function to determine whether the danger ahead
        If obstacle closer than safety distance, the machine should stop
        Sets the whole histogram as 1 => best angle becomes -1 and machine stops
        """
        def angle_within_range(angles: list, target_angle: int) -> bool:
            normalized_angle = target_angle % 360
            if angles[0] <= normalized_angle <= angles[1]:
                print('WITHIN RANGE !!!!!!!!!!!!!!!!!!!!!!!!!!')
                return True
            return False

        def get_lowest_dist_tuple() -> tuple:
            min_distance = float('inf')
            result = (-1, -1)
            for angle, dist in self.measurements:
                if dist < min_distance:
                    min_distance = dist
                    result = (dist, convert_to_degrees(angle))
            return result

        closest_distance, targ_angle = get_lowest_dist_tuple()
        print("CLOSEST DIST AND ANGLE: ", closest_distance, targ_angle)
        if closest_distance <= self.safety_distance and angle_within_range(observation_range, target_angle=targ_angle):
            self.histogram = np.ones(int(360 / self.alpha))  # set as 1

    def get_histogram(self) -> np.array:
        return self.histogram

    def empty_histogram(self):
        self.histogram * 0  # reset histogram

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

    def get_threshold_magnitude(self, min_distance: float) -> float:
        if self.a is None:
            self.a = self.b * min_distance
        return ((min_distance + 1) ** 2) * (self.a - self.b * min_distance)

    def get_obstacle_map(self, measuring_distance: int) -> np.array:
        """
        The function determine the location of the obstacles and create virtual AxA map of their locations,
        where A is a measuring distance as an int
        The measurements include the coordinates (angle, distance) of the obstacles detected by the virtual LIDAR
        The current location in (y, x) is coordinates of the wheelchair according to the map
        The grid with 1+distance: int and 0, where 1+distance is a certainty value of obstacle
        """
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

    def show_histogram(self, current_node=None, target_node=None, env=None, show_rate=None) -> None:
        """
        The function is using plt to plot the histogram and
        the map of the environment as subplots
        The current location is indicated as a cross
        The steering direction is indicated as a blue vector passed in degrees
         """
        # get x axis for histogram

        self.ax1.clear()
        # plot histogram
        self.ax1.plot(self.histogram, color='b')
        # plot threshold
        x = np.zeros(self.histogram.shape[0]) + self.threshold
        self.ax1.plot(x, color='r')
        self.ax1.set_xlabel('Angle (in degrees)')
        self.ax1.set_ylabel('Probability')
        self.ax1.set_title('Vector Field Histogram')
        if env is not None and current_node is not None:
            self.show_map(env=env, current_node=current_node)
        if show_rate is not None:
            self.show_free_sectors_num()
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
        plt.pause(0.1)  # Add a small delay (e.g., 0.1 seconds)

    def show_obstacle_map(self, measuring_distance: int) -> None:
        plt.imshow(self.get_obstacle_map(measuring_distance), cmap='Greys', origin='lower', alpha=0.7)
        plt.show()

    def show_map(self, env: Map, current_node: tuple) -> None:
        # plot grid
        self.ax2.imshow(env.grid, origin='lower')
        self.ax2.imshow(env.skeleton, cmap='Greys', origin='lower', alpha=0.7)
        self.ax2.set_xlabel('North')
        self.ax2.set_ylabel('East')
        self.ax2.set_title('Environment map')
        # show the current location
        self.ax2.plot(current_node[1], current_node[0], 'rx')
        self.ax2.figure.set_size_inches(12, 12)

    def show_free_sectors_num(self):
        # Extract the x and y values from the input data
        time_in_ms = [entry[1] for entry in self.free_sectors_num]
        sectors_num = [entry[0] for entry in self.free_sectors_num]
        self.ax2.plot(time_in_ms, sectors_num)
        self.ax2.set_title('VFH update rate')
        self.ax2.set_xlabel('Time (ms)')
        self.ax2.set_ylabel('Number of Free Sectors')


if __name__ == '__main__':
    """ Retrieve testing data """
    start: tuple = config.get('initial_position')
    file_name: str = config.get('colliders')
    safety_distance: float = config.get('safety_distance')
    goal: tuple = config.get('final_position')

    # test with simulation Lidar
    from lidar_simulation import LidarSimulation

    map_2d = Map(map_image_size=12, safety_distance=safety_distance)
    map_2d.load_grid(config.get('grid_save'), dtype=np.int)
    map_2d.load_skeleton(config.get('skeleton_save'), dtype=np.int)
    lidar_simulation = LidarSimulation(radius=config.get('lidar_radius'), direction=0)
    lidar_simulation.scan(grid=map_2d.grid, current_location=start)
    vfh = VFH(b=config.get('b'),
              alpha=config.get('sector_angle'),
              l_param=config.get('l'),
              safety_distance=safety_distance)
    vfh.update_measurements(lidar_simulation.get_values())
    vfh.generate_vfh()
    vfh.get_rotation_angle(current_node=start, next_node=goal)
    vfh.show_histogram(current_node=start)
