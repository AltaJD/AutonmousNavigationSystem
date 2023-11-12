import config_extractor as config
import map
import numpy as np
from LIDAR_simulation import LIDAR
from behavioral_model import IntelligentWheelchair
from typing import List
from math import cos, sin
import matplotlib.pyplot as plt

""" === Functions to get Vector Field Histogram === """


def show_histogram(h: np.array, grid: np.array, skeleton: np.array, current_location: tuple) -> None:
    """
    The function is using plt to plot the histogram and
    the map of the environment as subplots
    The current location is indicated as a cross
     """
    # get subplot objects ax1 and ax2
    fig, (ax1, ax2) = plt.subplots(2, 1)
    # get x axis for histogram
    x = np.arange(h.shape[0])
    # plot histogram
    ax1.bar(x, h)
    ax1.set_xlabel('Angle (*10 degrees)')
    ax1.set_ylabel('Probability')
    ax1.set_title('Vector Field Histogram')
    # plot grid
    ax2.imshow(grid, origin='lower')
    ax2.imshow(skeleton, cmap='Greys', origin='lower', alpha=0.7)
    ax2.set_xlabel('North')
    ax2.set_ylabel('East')
    ax2.set_title('Environment map')
    # show the current location
    ax2.plot(current_location[1], current_location[0], 'rx')
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
    # FIXME: increase accuracy. The map is incorrect
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
    if len(angles) == max_size:
        # if no insertion is required:
        return angles
    # insert process
    expanded_list = np.zeros(max_size)
    expanded_list = list(expanded_list)[:starting_index] + angles
    print(expanded_list)
    # shift values from max_size to len(expanded_list) indexes to the beginning of the list
    for i in range(max_size, len(expanded_list)):
        expanded_list[i-max_size] = expanded_list[i]
    expanded_list = expanded_list[:max_size]
    return expanded_list


def get_certainty_values(measurements: List[tuple]) -> np.array:
    """
    The function determine the location of the obstacles and create virtual AxA map of their locations,
    where A is a measuring distance as an int
    The measurements include the coordinates (angle, distance) of the obstacles detected by the virtual LIDAR
    :returns the list with 1+distance: int and 0, where 1+distance is a certainty value of obstacle
    """
    obstacle_magnitudes = []
    for value in measurements:
        # angle in radians
        # distance in meters
        angle, distance = value
        # assign 1+distance for obstacle and 0 as freeway
        # the certainty value based on the distance is assigned to the obstacle map
        obstacle_magnitudes.append(1+round(distance))
    return np.array(obstacle_magnitudes)


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
    cumulative = 0 # cumulative sum
    for k in range(hist.shape[0]-l):
        for j in range(1, l):
            multiplier = l-j+1
            cumulative += (multiplier*h[k-j] + multiplier*h[k+j])
        cumulative += l*h[k] # for j==0 which is always True
        hist[k] = cumulative # update array
        cumulative = 0 # reset
    return hist


def get_sectors(measurements: List[tuple], magnitudes: np.array, alpha: float) -> List[int]:
    """ The function creates the grid as np.array of the angles to the obstacle
    The measurements include the coordinates (angle, distance) of the obstacles detected by the virtual LIDAR
    :returns: polar obstacle density as np.array, dtype=int
    """
    # determine polar obstacle density h_k, where k is each sector divided by alpha
    h_k = [] # List[int]
    m_sum = 0 # sum of magnitudes per sector
    # set initial/starting angle
    starting_angle: float = measurements[0][0]*180/np.pi # in degrees
    for i in range(len(measurements)):
        beta: float = measurements[i][0] # get angle in radiance
        beta = beta*180/np.pi # convert radiance to degrees
        # determine sum of the magnitudes for particular sector
        if beta-starting_angle <= alpha:
            m_sum += magnitudes[i] # if the magnitude is in the sector, we add it to the sum
        else:
            h_k.append(round(m_sum, 0)) # store the sector cumulative magnitude
            m_sum = 0 # reset sum
            starting_angle = beta # set the new starting angle for the sector in degrees
    print("RAW OBSTACLE HISTOGRAM")
    print(h_k)
    # expand the angle ranges from 0 to 360 degrees
    initial_angle: float = measurements[0][0]*180/np.pi # degrees
    if initial_angle < 0:
        initial_angle += 360 # convert to the positive sign angle
    sectors_num = int(360/alpha) # number of sectors
    initial_sector = int(initial_angle/alpha) # starting sector
    normalized_histogram = expand_histogram(angles=h_k, max_size=sectors_num, starting_index=initial_sector)
    print("EXPANDED OBSTACLE HISTOGRAM")
    print(normalized_histogram)
    return normalized_histogram


def get_vfh(measurements: List[tuple], alpha: int, a: int, b: int) -> np.array:
    """ The function will generate the Vector Field Histogram
    The magnitude of the obstacle is represented as a formula:
    m[i][j] = (c[i][j])**2 * (a - b * d[i][j])
    where:
    1. c[i][j] = certainty value
    2. d[i][j] = distances to each obstacle
    3. a and b are positive constants
    4. m[i][j] will be a one dimensional array consisting of values greater than 0

    alpha is an angle division that should be in degrees
    """
    c: np.array = get_certainty_values(measurements)
    # since the distance is included in the certainty values, it can be extracted as follows:
    d: np.array = c.copy() # distances
    for i in range(d.shape[0]):
        cell_value = d[i]
        if cell_value > 0:
            d[i] = cell_value - 1
    magnitudes: np.array = (c**2)*(a-b*d) # m[i][j]
    # since a constant is very small, we can convert negative values back to positive values
    magnitudes = 0-magnitudes
    print('MAGNITUDES')
    print(magnitudes)
    histogram = get_sectors(measurements, magnitudes, alpha)
    # smooth histogram
    print("""==== RECEIVED HISTOGRAM ====""")
    print(np.array(histogram))
    # show_histogram(np.array(histogram))
    histogram_smoothed = smooth_histogram(h=histogram, l=3)
    print("""==== SMOOTHED HISTOGRAM ====""")
    print(histogram_smoothed)
    # show_histogram(histogram)
    return histogram_smoothed


if __name__ == '__main__':
    # TESTING CODE
    """ === Get default values ==="""
    grid_file            = config.get('grid_save')
    skeleton_file        = config.get('skeleton_save')
    grid: np.array       = map.read_grid(file_path=grid_file, dtype=np.float)
    skeleton: np.array   = map.read_grid(file_path=skeleton_file, dtype=np.int)
    start_default: tuple = config.get('initial_position')
    goal_default:  tuple = config.get('final_position')
    """ Select the current position of the wheelchair """
    start_default = map.select_point(grid, skeleton) # update starting position
    print(start_default)
    """ Create instances of the objects """
    lidar = LIDAR(radius=config.get('lidar_radius')) # set scanning radius of the lidar as 30 meters or 30x30 matrix within grid
    wheelchair = IntelligentWheelchair(current_position=start_default, current_angle=0.0)
    """ Find the location of the new obstacles """
    lidar.scan(grid=grid, current_location=wheelchair.current_position, initial_angle=wheelchair.current_angle)
    """ Show obstacles detected by LIDAR """
    # show_obstacle_map(lidar.get_values(), measuring_distance=lidar.measuring_radius)
    """ Get Vector Field Histogram (VFH) """
    sector_angle: int = 10 # degrees
    # TODO: determine a and b such that a-b*dmax = 0
    a, b = 1, 1
    histogram = get_vfh(measurements=lidar.get_values(),
                        alpha=sector_angle,
                        a=a, b=b)
    show_histogram(h=histogram, grid=grid, skeleton=skeleton, current_location=wheelchair.current_position)
    lidar.show_scanning_area(grid, skeleton, current_node=start_default)
