from enum import Enum
import math
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional
from enum import Enum


class WheelchairStatus(Enum):
    """ An object contain different status of wheelchair
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
    current_position: tuple
    current_angle: float
    goal_position: tuple
    length: float
    width: float
    height: Optional[float]
    battery_level: float
    status: WheelchairStatus

    def __init__(self, current_position: tuple, current_angle: float):
        """ Set the values by default, such as status of the wheelchair
        The position vector is essential to determine the next maneuver to be correctly calculated
        """
        self.status = WheelchairStatus.IDLE
        self.current_position = current_position
        self.current_angle = current_angle

    def __str__(self) -> str:
        return f"Class:\tIntelligentWheelchair\n" \
               f"Name:\t{self.name}\n" \
               f"Current position:\t{self.current_position}\n" \
               f"Goal position:\t{self.goal_position}\n" \
               f"Size (l,w, h):\t{(self.length, self.width, self.height)}\n"

    def show_current_position(self, map: np.array) -> None:
        """The function is showing the recent location of wheelchair
        and presents on the map using plt
        """
        plt.imshow(map)
        plt.plot(self.current_position[0], self.current_position[1], 'rx')
        plt.show()

    def move_to(self, next_node: tuple) -> None:
        """ The function is calculating the current position and vector
        to determine the rotation angle required and
        """
        rotation = self.rotation_angle(next_node=next_node)
        self.current_position = next_node
        print(f'Wheelchair moved to: {next_node}')
        pass

    def stop(self) -> None:
        pass

    """The functions accessible only for calculation"""
    def rotation_angle(self, next_node: tuple) -> float:
        """ The function calculate the angle for rotation the wheelchair
        It is based on the arctan between two points
        """
        y_diff = next_node[1]-self.current_position[1]
        x_diff = next_node[0]-self.current_position[0]
        angle = math.atan2(y_diff, x_diff)
        print(f'Rotation angle is {angle}')
        return angle


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
