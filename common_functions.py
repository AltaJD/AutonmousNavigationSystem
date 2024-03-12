import math
import numpy as np


def get_vector_angle(next_node: tuple, current_node: tuple) -> float:
    """
    Returns arctan between two points as radians
    """
    y_diff = next_node[1] - current_node[1]
    x_diff = next_node[0] - current_node[0]
    return math.atan2(y_diff, x_diff)


def get_distance(next_node: tuple, current_node: tuple) -> float:
    """
    Return Euclidean distance between nodes in meters
    """
    x_diff = next_node[0]-current_node[0]
    y_diff = next_node[1]-current_node[1]
    return np.sqrt(x_diff**2 + y_diff**2)


def convert_to_degrees(radians: float):
    """
    :param radians: angle in radiance in R range
    :return: angle in range [0, 360) degrees
    """
    angle = math.degrees(radians)
    if angle < 0:
        angle += 360  # Adjust negative degrees to positive range
    if angle >= 360:
        angle %= 360
    return angle


def get_angle_difference(angle1, angle2):
    """
    Calculates the difference between two angles in degrees.
    The result will always be positive and in the range of 0 to 180 degrees.
    """

    # Normalize the angles to be between 0 and 360 degrees
    angle1 = angle1 % 360
    angle2 = angle2 % 360

    # Calculate the absolute difference between the angles
    diff = abs(angle1 - angle2)

    # Determine the direction of rotation
    if angle1 < angle2:
        direction = "right"
    else:
        direction = "left"

    # Take the smaller difference between the angles
    diff = min(diff, 360 - diff)

    return diff, direction
