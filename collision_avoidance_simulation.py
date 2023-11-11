import config_extractor as config
import map
import numpy as np
from LIDAR_simulation import LIDAR
from behavioral_model import IntelligentWheelchair
from typing import List
from math import cos, sin
import matplotlib.pyplot as plt

""" === Functions to get Vector Field Histogram === """


def show_histogram(h: np.array) -> None:
    """ The function is using plt to plot the histogram """
    # get x axis
    x = np.arange(h.shape[0])
    # x = x*10 # convert to degrees (from 0 to 360)
    # plot histogram
    plt.bar(x, h)
    plt.ylabel('Probability')
    plt.xlabel('Angle')
    plt.title('Vector Field Histogram')
    plt.show()


def show_obstacle_map(measurements: List[tuple], measuring_distance: int) -> None:
    """
    The function determine the location of the obstacles and create virtual AxA map of their locations,
    where A is a measuring distance as an int
    The measurements include the coordinates (angle, distance) of the obstacles detected by the virtual LIDAR
    The current location in (y, x) is coordinates of the wheelchair according to the map
    The grid with 1+distance: int and 0, where 1+distance is a certainty value of obstacle
    """
    # create obstacle_grid simulation with zeros
    obstacle_grid = np.zeros((measuring_distance, measuring_distance))
    for value in measurements:
        # angle in radians
        # distance in meters
        angle, distance = value
        # get the location of the obstacle on the map
        location_x = round(distance*cos(angle))
        location_y = round(distance*sin(angle))
        # assign 1+distance for obstacle and 0 as freeway
        # the certainty value based on the distance is assigned to the obstacle map
        obstacle_grid[location_x][location_y] = 1+round(distance)
    plt.imshow(obstacle_grid, cmap='Greys', origin='lower', alpha=0.7)
    plt.show()


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
    """ The function creates the grid as np.array of the angle to the obstacle
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
            starting_angle = beta # set the new starting angle for the sector
    # expand the angle ranges from 0 to 360 degrees
    initial_angle: float = measurements[0][0]*180/np.pi # degrees
    if initial_angle < 0:
        initial_angle += 360 # convert to the positive sign angle
    sectors_num = int(360/alpha) # number of sectors
    initial_sector = int(initial_angle/alpha) # starting sector
    histogram: List[int] = list(np.zeros(sectors_num))
    histogram = histogram[:initial_sector-1]+h_k
    if len(histogram) != sectors_num: histogram += list(np.zeros(sectors_num-len(histogram)))
    return histogram


def get_vfh(measurements: List[tuple], alpha: int, a: int, b: int) -> np.array:
    """ The function will generate the Vector Field Histogram
    The magnitude of the obstacle is represented as a formula:
    m[i][j] = (c[i][j])**2 * (a - b * d[i][j])
    where:
    1. c[i][j] = certainty value
    2. d[i][j] = distances to each obstacle
    3. a and b are positive constants

     alpha is an angle division that should be in degrees
    """
    c: np.array = get_certainty_values(measurements)
    # since the distance is included in the certainty values, it can be extracted as follows:
    d: np.array = c.copy() # distances
    for i in range(d.shape[0]):
        cell_value = d[i]
        if cell_value > 0:
            d[i] = cell_value - 1
    magnitudes: np.array = (c**2)*(a-b*d)
    # since a constant is very small, we can convert negative values back to positive values
    magnitudes = 0-magnitudes
    histogram = get_sectors(measurements, magnitudes, alpha)
    # smooth histogram
    print("""==== RECEIVED HISTOGRAM ====""")
    print(np.array(histogram))
    # show_histogram(np.array(histogram))
    histogram_smoothed = smooth_histogram(h=histogram, l=3) # TODO: choose the l const
    print("""==== SMOOTHED HISTOGRAM ====""")
    print(histogram_smoothed)
    # show_histogram(histogram)
    return histogram_smoothed


if __name__ == '__main__':
    # TEST
    """ === Get default values ==="""
    grid_file            = config.get('grid_save')
    skeleton_file        = config.get('skeleton_save')
    grid: np.array       = map.read_grid(file_path=grid_file, dtype=np.float)
    skeleton: np.array   = map.read_grid(file_path=skeleton_file, dtype=np.int)
    start_default: tuple = config.get('initial_position')
    goal_default:  tuple = config.get('final_position')
    """ Create instances of the objects """
    lidar = LIDAR(radius=30) # set scanning radius of the lidar as 30 meters or 30x30 matrix within grid
    wheelchair = IntelligentWheelchair(current_position=start_default, current_angle=0.0)
    """ Find the location of the new obstacles """
    lidar.scan(grid=grid, current_location=wheelchair.current_position)
    """ Show obstacles detected by LIDAR """
    # lidar.show_scanning_area(grid, skeleton, wheelchair.current_position)
    # show_obstacle_map(lidar.get_values(), measuring_distance=lidar.measuring_radius)
    """ Get Vector Field Histogram (VFH) """
    angle_d: int = 10 # degrees
    # TODO: determine a and b such that a-b*dmax = 0
    a, b = 1, 1
    histogram = get_vfh(measurements=lidar.get_values(),
                        alpha=angle_d,
                        a=a, b=b)
    show_histogram(h=histogram)
