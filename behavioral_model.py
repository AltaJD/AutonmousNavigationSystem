import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Tuple
from enum import Enum
from collision_avoidance_simulation import get_vfh, get_rotation_angle, show_histogram
from math import sin, cos, radians
from LIDAR_simulation import LIDAR, get_distance, get_vector_angle
import config as config
import time


class WheelchairStatus(Enum):
    """
    An object contain different status of wheelchair
    MOVING = performing maneuvering between two different points
    IDLE = no tasks has been given to the wheelchair
    WAITING = the task has not been finished and requires processing path plan or
    unavoidable object has been encountered
    INTERRUPTED = considering that user has manually stopped the wheelchair
    ERROR = in case the path was not found or any program bug has occurred
    """
    MOVING = 'moving'
    IDLE = 'idle'
    WAITING = 'waiting'
    INTERRUPTED = 'interrupted'
    ERROR = 'error'


class IntelligentWheelchair:
    """
    The object consider the list of the functions available
    for different services
    """
    name: Optional[str]
    current_position: Tuple[float, float]  # (y, x)
    current_angle: float  # in degrees
    current_speed: float
    goal_position: tuple
    length: float
    width: float
    height: Optional[float]
    battery_level: float
    status: str
    lidar: LIDAR

    def __init__(self, current_position: tuple, current_angle: float, lidar: LIDAR):
        """
        Set the values by default, such as status of the wheelchair
        The position vector is essential to determine the next maneuver to be correctly calculated
        """
        self.status = WheelchairStatus.IDLE.value
        self.current_position = current_position  # (y, x)
        self.current_angle = current_angle
        self.lidar = lidar

    def __str__(self) -> str:
        return f"Class:\tIntelligentWheelchair\n" \
               f"Name:\t{self.name}\n" \
               f"Current position:\t{self.current_position}\n" \
               f"Current angle:\t{self.current_angle}\n" \
               f"Goal position:\t{self.goal_position}\n" \
               f"Size (l,w, h):\t{(self.length, self.width, self.height)}\n"

    def show_current_position(self, map: np.array) -> None:
        """
        The function is showing the recent location of wheelchair
        and presents on the map using plt
        """
        plt.imshow(map)
        plt.plot(self.current_position[0], self.current_position[1], 'rx')
        plt.show()

    def move_to(self, target_node: tuple, grid: np.array, show_map=False) -> None:
        """
        The function is calculating the current position and vector
        to determine the rotation angle required and
        """
        # get parameters required for vfh calculation FIXME
        sector_angle: int = config.get('sector_angle')  # degrees
        a, b = config.get('a'), config.get('b')
        distance_tolerance = config.get('distance_tolerance')  # in meters
        vfh_threshold: float = config.get('vfh_threshold')

        start_time = time.time()
        self.status = WheelchairStatus.MOVING.value
        reached_target: bool = (get_distance(target_node, self.current_position) < distance_tolerance)
        # target_reachable: bool = histogram[int(get_vector_angle(target_node, self.current_position)/config.get('sector_angle'))] > vfh_threshold

        while not reached_target:
            """==== Path selection with lowest probability of obstacles ===="""
            histogram = get_vfh(measurements=self.lidar.get_values(), alpha=sector_angle, b=b)
            angle     = get_rotation_angle(h=histogram, threshold=vfh_threshold,
                                           current_node=self.current_position,
                                           next_node=target_node)
            if round(abs(angle-self.current_angle)) == 180:
                break
            distance = get_distance(target_node, self.current_position)  # on average, the wheelchair is moving 1 meter
            next_node = (distance * cos(radians(angle)) + self.current_position[0],
                         distance * sin(radians(angle)) + self.current_position[1])
            # update self steering direction and current node
            self.current_position = next_node
            self.current_angle = float(angle)
            self.lidar.scan(grid, current_location=self.current_position)
            reached_target: bool = (distance < distance_tolerance)
            print('DISTANCE', round(distance, 2))
            time_diff = time.time()-start_time
            if time_diff > 10:
                show_histogram(h=histogram,
                               grid=grid,
                               current_location=self.current_position,
                               steering_direction=self.current_angle)
                start_time = time.time()
            # show the result
            print(f"Current location:\t {self.current_position}\n"
                  f"Target location:\t {target_node}\n"
                  f"Steering direction:\t {self.current_angle}")

        if show_map: show_histogram(h=histogram,
                                    grid=grid,
                                    current_location=self.current_position,
                                    steering_direction=self.current_angle)
        self.stop()

    def stop(self) -> None:
        self.status = WheelchairStatus.WAITING.value
        pass


class Action(Enum):
    """
    An action is represented by a 3 element tuple.

    The first 2 values are the delta of the action relative
    to the current grid position. The third and final value
    is the cost of performing the action.

    The rotation cost is higher, because the system also should consider whether
    more straight and simpli path can be chosen.
    """
    LEFT = (0, -1, 1)
    RIGHT = (0, 1, 1)
    UP = (-1, 0, 1)
    DOWN = (1, 0, 1)

    def __str__(self):
        if self == self.LEFT:
            return '<'
        elif self == self.RIGHT:
            return '>'
        elif self == self.UP:
            return '^'
        elif self == self.DOWN:
            return 'v'

    @property
    def cost(self):
        return self.value[2]

    @property
    def delta(self):
        return (self.value[0], self.value[1])
