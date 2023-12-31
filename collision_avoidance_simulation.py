import config as config
import map as mp
import numpy as np
from LIDAR_simulation import LIDAR, get_vector_angle
from typing import List
from math import cos, sin
import matplotlib.pyplot as plt

""" === Functions to get Vector Field Histogram === """


def show_histogram(h: np.array, current_location: tuple, steering_direction=None, skeleton=None, grid=None, target_node=None) -> None:
    """
    The function is using plt to plot the histogram and
    the map of the environment as subplots
    The current location is indicated as a cross
    The steering direction is indicated as a blue vector passed in degrees
     """
    # get subplot objects ax1 and ax2
    fig, (ax1, ax2) = plt.subplots(2, 1)
    # get x axis for histogram
    x = np.arange(h.shape[0])*config.get('sector_angle')
    # plot histogram
    ax1.bar(x, h)
    ax1.set_xlabel('Angle (*10 degrees)')
    ax1.set_ylabel('Probability')
    ax1.set_title('Vector Field Histogram')
    # plot grid
    if grid is not None:
        ax2.imshow(grid, origin='lower')
        if skeleton: ax2.imshow(skeleton, cmap='Greys', origin='lower', alpha=0.7)
        ax2.set_xlabel('North')
        ax2.set_ylabel('East')
        ax2.set_title('Environment map')
        # show the current location
        ax2.plot(current_location[1], current_location[0], 'rx')
        ax2.figure.set_size_inches(10, 10)
    # show 0 angle vector
    plt.quiver(current_location[1], current_location[0], sin(0) * 1, cos(0) * 1, color='r')
    # show the steering direction
    if steering_direction: plt.quiver(current_location[1], current_location[0],
                                      sin(np.radians(steering_direction)), cos(np.radians(steering_direction)),
                                      color='b')
    if target_node: plt.plot(target_node[1], target_node[0], 'gx')
    # adjusting spacing between subplots
    plt.tight_layout()
    plt.show()


def show_obstacle_map(measurements: List[tuple], measuring_distance: int) -> None:
    """
    The function determine the location of the obstacles and create virtual AxA map of their locations,
    where A is a measuring distance as an int
    The measurements include the coordinates (angle, distance) of the obstacles detected by the virtual LIDAR
    The current location in (y, x) is coordinates of the wheelchair according to the map
    The grid with 1+distance: int and 0, where 1+distance is a certainty value of obstacle
    """
    # FIXME: increase accuracy.
    # create obstacle_grid simulation with zeros by default
    obstacle_grid = np.zeros((measuring_distance+1, measuring_distance+1))
    for value in measurements:
        # angle in radians
        # distance in meters
        angle, distance = value
        # get the location of the obstacle on the map
        location_y = round(distance*cos(angle))
        location_x = round(distance*sin(angle))
        # assign 1+distance for obstacle and 0 as freeway
        # the certainty value based on the distance is assigned to the obstacle map
        obstacle_grid[location_y][location_x] = 1+round(distance)
    print("OBSTACLE GRID")
    print(obstacle_grid)
    plt.imshow(obstacle_grid, cmap='Greys', origin='lower', alpha=0.7)
    plt.show()


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
        values = values + [0]*(size-len(values))
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


def get_certainty_values(measurements: List[tuple]) -> np.array:
    """
    The function determine the location of the obstacles and create virtual AxA map of their locations,
    where A is a measuring distance as an int
    The measurements include the coordinates (angle, distance) of the obstacles detected by the virtual LIDAR
    :returns the list with 1+distance: int and 0, where 1+distance is a certainty value of obstacle
    otherwise, the [0] is returned
    """
    obstacle_magnitudes = []
    for value in measurements:
        # angle in radians
        # distance in meters
        angle, distance = value
        # assign 1+distance for obstacle and 0 as freeway
        # the certainty value based on the distance is assigned to the obstacle map
        obstacle_magnitudes.append(1+round(distance))
    if len(obstacle_magnitudes) > 0:
        return np.array(obstacle_magnitudes)
    return np.zeros(1, dtype=np.float)


def smooth_histogram(h: List[int], l: int) -> np.array:
    """ Initial mapping may appear ragged and cause errors in the selection of the steering direction
    The function suggest to consider smooth the graph for better angle selection
    :parameter
    h is a polar obstacle density
    l is a constant integer, chosen by experiment or simulation.
    For this system, l is a tolerance for steering direction
    :returns: smoothed polar obstacle density as List[float]
    """
    hist = np.array(h, dtype=int)
    smoothed_data = []
    # cumulative = 0 # cumulative sum
    for i in range(len(hist)):
        start_index = max(0, i - l + 1)
        end_index = min(len(hist), i + l)
        window = hist[start_index:end_index]
        smoothed_value = sum(window) / len(window)
        smoothed_data.append(smoothed_value)
    return smoothed_data


def get_sectors(measurements: List[tuple], magnitudes: np.array, alpha: float) -> List[int]:
    """ The function creates the grid as np.array of the angles to the obstacle
    The measurements include the coordinates (angle, distance) of the obstacles detected by the virtual LIDAR
    :returns: polar obstacle density as np.array, dtype=int
    """
    # determine polar obstacle density h_k, where k is each sector divided by alpha
    h_k = []  # List[int]
    m_sum = 0  # sum of magnitudes per sector
    # set initial/starting angle
    starting_angle: float = measurements[0][0]*180/np.pi # in degrees
    for i in range(len(measurements)):
        beta: float = measurements[i][0]  # get angle in radiance
        beta = beta*180/np.pi  # convert radiance to degrees
        # determine sum of the magnitudes for particular sector
        if beta-starting_angle <= alpha:
            m_sum += magnitudes[i]  # if the magnitude is in the sector, we add it to the sum
        else:
            h_k.append(round(m_sum, 0))  # store the sector cumulative magnitude
            m_sum = 0  # reset sum
            starting_angle = beta  # set the new starting angle for the sector in degrees
    # expand the angle ranges from 0 to 360 degrees
    initial_angle: float = measurements[0][0]*180/np.pi  # degrees
    if initial_angle < 0:
        initial_angle += 360  # convert to the positive sign angle
    sectors_num = int(360/alpha)  # number of sectors
    initial_sector = int(initial_angle/alpha) # starting sector
    normalized_histogram = expand_histogram(angles=h_k, max_size=sectors_num, starting_index=initial_sector)
    return normalized_histogram


def get_magnitudes(certainty_values: np.array, a: int, b: int) -> np.array:
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
        a = b*max(d)
    return (certainty_values**2)*(a-b*d)  # m[i][j]


def get_vfh(measurements: List[tuple], alpha: int, b: int, a=None) -> np.array:
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

    c: np.array = get_certainty_values(measurements)
    magnitudes  = get_magnitudes(c, a, b)
    histogram   = get_sectors(measurements, magnitudes, alpha)
    histogram_smoothed = smooth_histogram(h=histogram, l=config.get('l'))
    # show magnitudes as percentages (from 0 to 1)
    highest_magnitude    = max(histogram_smoothed)
    normalized_histogram = np.array([magnitude/highest_magnitude if highest_magnitude != 0 else 0 for magnitude in histogram_smoothed])
    return normalized_histogram


def get_rotation_angle(h: np.array, current_node: tuple, next_node=None, threshold=0.0) -> int:
    """ Get the angle with the lowest probability of the obstacles ahead 
    :param h is histogram represented as np.array with a shape (n, 0)
    :param threshold in % specifies the minimum obstacle probability acceptable as obstacle-free path
    :param next_node provides the data for choosing the most appropriate steering direction
    :return angle in degrees
    """
    # determining angle that is the closest to the target point
    obstacle_free_sectors: List[int] = np.where(h <= threshold)[0]
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
    minimums_diff = list(map(lambda x: abs(x-desired_angle), obstacle_free_sectors))
    best_angle = obstacle_free_sectors[minimums_diff.index(min(minimums_diff))]
    # convert angle to degrees
    best_angle *= config.get('sector_angle')
    return best_angle


if __name__ == '__main__':
    # TESTING CODE
    from behavioral_model import IntelligentWheelchair
    """ === Get default values ==="""
    grid_file            = config.get('grid_save')
    skeleton_file        = config.get('skeleton_save')
    grid: np.array       = mp.read_grid(file_path=grid_file, dtype=np.float)
    skeleton: np.array   = mp.read_grid(file_path=skeleton_file, dtype=np.int)
    goal_default:  tuple = config.get('final_position')
    """ Select the current position of the wheelchair """
    start_default = mp.select_point(grid, skeleton) # update starting position
    """ Create instances of the objects """
    # set scanning radius of the lidar as 30 meters or 30x30 matrix within grid
    lidar = LIDAR(radius=config.get('lidar_radius'))
    wheelchair = IntelligentWheelchair(current_position=start_default, current_angle=0.0, lidar=lidar)
    """ Find the location of the new obstacles """
    lidar.scan(grid=grid, current_location=wheelchair.current_position)
    """ Show obstacles detected by LIDAR """
    # show_obstacle_mp(lidar.get_values(), measuring_distance=lidar.measuring_radius)
    """ Get Vector Field Histogram (VFH) """
    sector_angle: int = config.get('sector_angle')  # degrees
    a, b = config.get('a'), config.get('b')
    histogram = get_vfh(measurements=lidar.get_values(),
                        alpha=sector_angle, b=b)
    """ Path selection with lowest probability of obstacles """
    angle = get_rotation_angle(h=histogram, threshold=config.get('vfh_threshold'),
                               current_node=wheelchair.current_position)
    print('NEXT ROTATION ANGLE and VALUE')
    print(angle)
    wheelchair.current_angle = float(angle) # update wheelchair steering direction
    show_histogram(h=histogram, grid=grid, skeleton=skeleton, current_location=wheelchair.current_position)
    lidar.show_scanning_area(grid, skeleton, current_node=start_default)
    mp.show_map(grid, skeleton, start=wheelchair.current_position, initial_vector=(wheelchair.current_angle, 1))
