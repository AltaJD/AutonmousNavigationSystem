import sys
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Tuple
from enum import Enum
from collision_avoidance_simulation import VFH
from math import sin, cos, radians
from lidar_simulation import LidarSimulation
from lidar import LIDAR
from common_functions import get_distance, get_angle_difference
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
    ROTATING = 'ROTATING'
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
    current_position: Tuple[float, float]  # (y, x)
    current_angle: float  # in degrees
    goal_angle: float
    goal_distance: float
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
        self.goal_angle = -1  # set invalid angle by default

    def __str__(self) -> str:
        return f"Class:\tIntelligentWheelchair\n" \
               f"Current position:\t{self.current_position}\n" \
               f"Current angle:\t{self.current_angle}\n" \
               f"Size (l,w, h):\t{(self.length, self.width, self.height)}\n" \
               f"Status:\t{self.status}"

    def stop(self) -> None:
        self.status = WheelchairStatus.WAITING.value
        # TODO: make proper emergency stop hardware call

    def emergency_stop(self) -> None:
        self.status = WheelchairStatus.INTERRUPTED.value
        # TODO: make proper emergency emergency hardware call

    def show_current_position(self) -> None:
        """
        The function is showing the recent location of wheelchair
        and presents on the map using plt
        """
        plt.imshow(self.map.grid, origin='lower')
        plt.imshow(self.map.skeleton, cmap='Greys', origin='lower', alpha=0.7)
        plt.plot(self.current_position[0], self.current_position[1], 'rx')
        plt.show()

    def move_forward(self, d: float) -> None:
        self.status = WheelchairStatus.MOVING.value
        # TODO: add hardware communication
        print("STATUS: ", self.status)
        self.stop()
        pass

    def rotate(self, angle: float) -> None:
        print(f"LIDAR: {self.lidar.current_angle}\t"
              f"WHEELCHAIR: {self.current_angle}\t"
              f"GOAL: {self.goal_angle}\t"
              f"DIFF: {angle}\t"
              f"DIRECTION: {self.status}")
        # TODO: add hardware communication
        # self.stop()
        pass

    def move_to(self, direction_angle: float, distance: float) -> None:
        """ The function to communicate with the hardware control
        :param direction_angle is an angle in degrees
        :param distance is distance in m toward goal
        """
        # TODO: refactor
        # angle by which the wheelchair has rotated
        rotated_angle, rotating_direction = get_angle_difference(self.lidar.current_angle,
                                                                 self.current_angle)
        rotation_angle_left, self.status = get_angle_difference(direction_angle,
                                                                rotated_angle)
        self.goal_angle = direction_angle
        if rotation_angle_left != 0:
            self.rotate(angle=rotation_angle_left)
        else:
            self.move_forward(distance)


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
    speed: float

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
        self.speed = 0.5

    def __str__(self) -> str:
        return f"Class:\tIntelligentWheelchair\n" \
               f"Name:\t{self.name}\n" \
               f"Current position:\t{self.current_position}\n" \
               f"Current angle:\t{self.current_angle}\n" \
               f"Goal position:\t{self.goal_position}\n" \
               f"Size (l,w, h):\t{(self.length, self.width, self.height)}\n"

    def show_current_position(self, env: np.array) -> None:
        """
        The function is showing the recent location of wheelchair
        and presents on the map using plt
        """
        plt.imshow(env)
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
            vfh.generate_vfh(blind_spot_range=(self.lidar_sim.start_blind_spot, self.lidar_sim.end_blind_spot),
                             blind_spot_overflow=self.lidar_sim.blind_spot_overflow)
            """ Get parameters to the next node """
            angle = vfh.get_rotation_angle(current_node=self.current_position,
                                           next_node=target_node)
            if angle == -1:
                self.emergency_stop()
                print("NO FREE SPACE HAS BEEN FOUND", file=sys.stderr)
                break
            distance = get_distance(target_node, self.current_position)
            """ Get the next coordinates from VFH """
            next_node: tuple = self.next_coordinate(angle, distance)
            """ Update wheelchair parameters """
            self.current_position = next_node
            self.current_angle = float(angle)
            self.lidar_sim.current_angle = float(angle)
            if distance < distance_tolerance:
                break
            """ Show the result """
            print(f"Current location:\t {self.current_position}\n"
                  f"Target location:\t {target_node}\n"
                  f"Steering direction:\t {self.current_angle}")
        if show_map:
            vfh.show_histogram(current_node=self.current_position,
                               env=self.map,
                               target_node=target_node)
        print("Reached the Waypoint")
        self.stop()

    def stop(self) -> None:
        self.status = WheelchairStatus.WAITING.value

    def emergency_stop(self) -> None:
        self.status = WheelchairStatus.INTERRUPTED.value

    def next_coordinate(self, angle: float, distance: float) -> tuple:
        """ Return the next coordinate
        :param angle in degrees
        :param distance in m
        :return (x, y) as tuple
        """
        return (self.speed * distance * cos(radians(angle)) + self.current_position[0],
                self.speed * distance * sin(radians(angle)) + self.current_position[1])
