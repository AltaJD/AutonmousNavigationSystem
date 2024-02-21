import sys
from typing import List
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Tuple
from enum import Enum
from collision_avoidance_simulation import VFH
from math import sin, cos, radians
from lidar_simulation import LidarSimulation
from lidar import LIDAR
from common_functions import get_distance, get_vector_angle
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

    def __init__(self, current_position: tuple,
                 current_angle: float,
                 lidar: LIDAR,
                 env: Map):
        """ The object obtains necessary services for manipulating with the LIDAR and Map data extraction """
        self.current_position = current_position
        self.current_angle = current_angle
        self.lidar = lidar
        self.map = env

    def __str__(self) -> str:
        return f"Class:\tIntelligentWheelchair\n" \
               f"Name:\t{self.name}\n" \
               f"Current position:\t{self.current_position}\n" \
               f"Current angle:\t{self.current_angle}\n" \
               f"Goal position:\t{self.goal_position}\n" \
               f"Size (l,w, h):\t{(self.length, self.width, self.height)}\n"

    def stop(self) -> None:
        self.status = WheelchairStatus.WAITING.value
        # TODO: make proper emergency stop
        pass

    def show_current_position(self) -> None:
        """
        The function is showing the recent location of wheelchair
        and presents on the map using plt
        """
        plt.imshow(self.map.grid, origin='lower')
        plt.imshow(self.map.skeleton, cmap='Greys', origin='lower', alpha=0.7)
        plt.plot(self.current_position[0], self.current_position[1], 'rx')
        plt.show()

    def start(self) -> None:
        path = self.map.path.waypoints
        self.lidar.get_values()
        for waypoint in path:
            self.move_to(next_node=waypoint)
            # TODO

    def move_to(self, next_node: List[int]) -> None:
        """ The function to communicate with the hardware control """
        # TODO: make hardware communication for the motors
        pass


class IntelligentWheelchairSim:
    """
    The object consider the list of the functions available
    for different services
    """
    map: Map
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

    def __init__(self, current_position: tuple,
                 current_angle: float,
                 lidar_simulation: LidarSimulation,
                 env: Map):
        """
        Set the values by default, such as status of the wheelchair
        The position vector is essential to determine the next maneuver to be correctly calculated
        """
        self.status = WheelchairStatus.IDLE.value
        self.current_position = current_position  # (y, x)
        self.current_angle = current_angle
        self.lidar_sim = lidar_simulation
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
        reached_target: bool = get_distance(target_node, self.current_position) < distance_tolerance
        while not reached_target:
            """ Generate VFH """
            self.lidar_sim.scan(self.map.grid,
                                current_location=self.current_position)
            vfh.update_measurements(self.lidar_sim.get_values())
            vfh.generate_vfh(blind_spot_range=(self.lidar_sim.start_blind_spot, self.lidar_sim.end_blind_spot))  # TODO: change blind spot range
            """ Get parameters to the next node """
            angle       = vfh.get_rotation_angle(current_node=self.current_position,
                                                 next_node=target_node)
            distance    = get_distance(target_node, self.current_position)
            """ Get the next coordinates from VFH """
            if self.lidar_sim.start_blind_spot < angle < self.lidar_sim.end_blind_spot:
                print("Waypoint has been passed", file=sys.stderr)
                break
            next_node:  tuple = self.next_coordinate(angle, distance)
            """ Update wheelchair parameters """
            self.current_position   = next_node
            self.current_angle      = float(angle)
            self.lidar_sim.current_angle = self.current_angle
            reached_target: bool = (distance < distance_tolerance)
            """ Show the result """
            print(f"Current location:\t {self.current_position}\n"
                  f"Target location:\t {target_node}\n"
                  f"Steering direction:\t {self.current_angle}")

        if show_map:
            vfh.show_histogram(current_node=self.current_position,
                               env=self.map,
                               target_node=target_node)
        self.stop()

    def stop(self) -> None:
        self.status = WheelchairStatus.WAITING.value
        pass

    def next_coordinate(self, angle: float, distance: float) -> tuple:
        """ Return the next coordinate
        :param angle in degrees
        :param distance in m
        :return (x, y) as tuple
        """
        return (distance * cos(radians(angle)) + self.current_position[0],
                distance * sin(radians(angle)) + self.current_position[1])
