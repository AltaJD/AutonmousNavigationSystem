import sys

import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Tuple
from enum import Enum
from collision_avoidance_simulation import VFH
from math import sin, cos, radians
from lidar_simulation import LidarSimulation
from lidar import LIDAR
from common_functions import get_distance
from map import Map
import config


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
    MOVING = 'MOVING'
    IDLE = 'IDLE'
    WAITING = 'WAITING'
    INTERRUPTED = 'INTERRUPTED'
    ERROR = 'ERROR'


class IntelligentWheelchair:
    """
    The object consider the list of the functions available
    for different services
    """
    map: Map
    lidar: LIDAR
    lidar_sim: LidarSimulation
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

    def __init__(self, current_position: tuple, current_angle: float, lidar: LidarSimulation, env: Map):
        """
        Set the values by default, such as status of the wheelchair
        The position vector is essential to determine the next maneuver to be correctly calculated
        """
        self.status = WheelchairStatus.IDLE.value
        self.current_position = current_position  # (y, x)
        self.current_angle = current_angle
        self.lidar_sim = lidar
        self.map = env

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

    def move_to(self, target_node: tuple, vfh: VFH, show_map=False) -> None:
        """
        The function is calculating the current position and vector
        to determine the rotation angle required and
        """
        distance_tolerance = config.get('distance_tolerance')  # in meters
        self.status = WheelchairStatus.MOVING.value
        reached_target: bool = (get_distance(target_node, self.current_position) < distance_tolerance)
        # target_reachable: bool = histogram[int(get_vector_angle(target_node, self.current_position)/config.get('sector_angle'))] > vfh_threshold
        while not reached_target:
            self.lidar.scan(self.map.grid, current_location=self.current_position)
            vfh.update_measurements(self.lidar.get_values())
            angle = vfh.get_rotation_angle(current_node=self.current_position, next_node=target_node)
            if round(abs(angle - self.current_angle)) == 180:
                print("Wheelchair tried to go backward", file=sys.stderr)
                break
            distance = get_distance(target_node, self.current_position)  # on average, the wheelchair is moving 1 meter
            next_node = (distance * cos(radians(angle)) + self.current_position[0],
                         distance * sin(radians(angle)) + self.current_position[1])
            # update self steering direction and current node
            self.current_position = next_node
            self.current_angle = float(angle)
            reached_target: bool = (distance < distance_tolerance)
            # show the result
            print(f"Current location:\t {self.current_position}\n"
                  f"Target location:\t {target_node}\n"
                  f"Steering direction:\t {self.current_angle}")

        if show_map: vfh.show_histogram(current_node=self.current_position)
        self.stop()

    def stop(self) -> None:
        self.status = WheelchairStatus.WAITING.value
        # TODO: make emergency stop
        pass
