import numpy as np
import map
from pprint import pprint
import sys
import config_extractor as config
from behavioral_model import IntelligentWheelchair
import LIDAR_simulation
import sys
from collision_avoidance_simulation import get_vfh, show_histogram, get_rotation_angle
from math import sin, cos, radians


if __name__ == '__main__':
    """ === Get configuration === 
    It includes default data for testing the system, such as:
    1. Start coordinate
    2. Goal coordinate
    3. Safety distance for wheelchair
    4. Path to the colliders coordinates
    5. Path to save the map as csv file for better view of the array and debugging
    """
    if sys.version_info[0:2] != (3, 6):
        raise Exception('Requires python 3.6')
    start_default:      tuple = config.get('initial_position')
    goal_default:       tuple = config.get('final_position')
    safety_distance:    float = config.get('safety_distance')
    filename:           str   = config.get('colliders')
    grid_filename:      str   = config.get('grid_save')
    skeleton_file_name: str   = config.get('skeleton_save')
    path_file_name:     str   = config.get('path_save')

    """ === Map generation === """
    # convert coordinates into tuple
    start_default, goal_default = (start_default[0], start_default[1]), (goal_default[0], goal_default[1])
    # grid, skeleton, distances, original_data = map.create_custom_map(filename=filename, safety_distance=safety_distance)
    """=== Read Grids for testing multiple versions in the future ==="""
    grid     = map.read_grid(file_path=grid_filename, dtype=np.float)
    skeleton = map.read_grid(file_path=skeleton_file_name, dtype=np.int)
    # select start and goal locations
    # goal = map.select_point(grid, skeleton, start_default)
    goal = goal_default
    print(goal)
    if not map.valid_destination(grid, start_default, goal):
        map.show_map(grid, skeleton, start_default, goal)
        raise Exception
    """=== Generate wheelchair object ==="""
    wheelchair: IntelligentWheelchair = IntelligentWheelchair(current_position=start_default, current_angle=0)
    """=== Normalize Grid ==="""
    map.normalize_grid(grid) # inplace action
    skeleton = skeleton.astype(int)
    """=== Save Grids for providing multiple and different formats ==="""
    # map.save_grid(grid_filename, grid)
    # map.save_grid(skeleton_file_name, skeleton)
    """=== Print map data === """
    print(10*'='+f"Grid"+"="*10)
    pprint(grid)
    print(10 * '=' + f"Sceleton" + "=" * 10)
    pprint(skeleton)
    print(10 * '=' + f"Distances" + "=" * 10)
    # pprint(distances)
    # map.show_map(grid, skeleton, start_default, goal_default)
    """ === Path planning === """
    # absolute_path: np.array = map.get_path(grid=grid, start=start_default, goal=goal) # get list of waypoints
    # map.save_grid(path_file_name, absolute_path)
    absolute_path: np.array = map.read_grid(file_path=path_file_name, dtype=np.int)
    print(absolute_path)
    map.show_map(grid=grid, skeleton=skeleton, path=absolute_path, start=start_default, goal=goal, save_path='./data_storage/images/images.png')
    # map.animate_path(absolute_path, grid, absolute_path, skeleton,
    #                  start=start_default, goal=goal,
    #                  animation_speed=5)
    """ ==== Append VFH ==== """
    lidar = LIDAR_simulation.LIDAR(radius=config.get('lidar_radius'))
    """ ==== Find the location of the new obstacles ==== """
    lidar.scan(grid=grid, current_location=wheelchair.current_position)
    """ ==== Show obstacles detected by LIDAR ==== """
    # show_obstacle_map(lidar.get_values(), measuring_distance=lidar.measuring_radius)
    """ ==== Get Vector Field Histogram (VFH) ==== """
    sector_angle: int = 10  # degrees
    a, b = 1, 1
    wheelchair.move_to(next_node=start_default)
    lidar.scan(grid, current_location=start_default)
    iteration = 0
    for node in absolute_path:
        histogram = get_vfh(measurements=lidar.get_values(), alpha=sector_angle, b=b)
        """ Path selection with lowest probability of obstacles """
        angle = get_rotation_angle(h=histogram, threshold=0.0, current_node=wheelchair.current_position, next_node=node)
        angle *= 10
        # update wheelchair steering direction
        previous_node = wheelchair.current_position
        wheelchair.current_angle = float(angle)
        distance = 1 # let's say every meter, the lidar starts to scan
        next_node = (distance*cos(radians(angle))+previous_node[0], distance*sin(radians(angle))+previous_node[1])
        wheelchair.move_to(next_node)
        lidar.scan(grid, current_location=next_node)
        print(f"Current location\t {wheelchair.current_position}\n"
              f"Previous location:\t {previous_node}\n"
              f"Steering direction:\t {angle}")
        iteration += 1
        if iteration % 20 == 0:
            show_histogram(h=histogram,
                           grid=grid,
                           skeleton=skeleton,
                           current_location=wheelchair.current_position,
                           steering_direction=angle)
            # map.show_map(grid, skeleton, start=wheelchair.current_position, initial_vector=(wheelchair.current_angle, 1))
