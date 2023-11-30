import numpy as np
import map
from pprint import pprint
import config_extractor as config
from behavioral_model import IntelligentWheelchair
import LIDAR_simulation
import sys
import time


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

    """ === Convert coordinates into tuples === """
    start_default, goal_default = (start_default[0], start_default[1]), (goal_default[0], goal_default[1])
    """=== Read Grids for testing multiple versions in the future ==="""
    grid     = map.read_grid(file_path=grid_filename, dtype=np.float)
    skeleton = map.read_grid(file_path=skeleton_file_name, dtype=np.int)
    # select start or goal locations
    goal          = map.select_point(grid, skeleton, title='Select the destination')
    start_default = map.select_point(grid, skeleton, title='Select the initial point')
    # goal = goal_default
    print(goal)
    if not map.valid_destination(grid, start_default, goal):
        map.show_map(grid, skeleton, start_default, goal)
        raise Exception
    """ ==== Generate Objects ==== """
    lidar = LIDAR_simulation.LIDAR(radius=config.get('lidar_radius'))
    wheelchair: IntelligentWheelchair = IntelligentWheelchair(current_position=start_default, current_angle=0, lidar=lidar)
    """ ==== Find the location of the new obstacles ==== """
    lidar.scan(grid=grid, current_location=wheelchair.current_position)
    """ ==== Show obstacles detected by LIDAR ==== """
    # show_obstacle_map(lidar.get_values(), measuring_distance=lidar.measuring_radius)
    """=== Normalize Grid ==="""
    map.normalize_grid(grid) # inplace action
    skeleton = skeleton.astype(int)
    """ === Path planning === """
    absolute_path: np.array = map.get_path(grid=grid, start=start_default, goal=goal) # get list of waypoints
    """=== Save Grids for providing multiple and different formats ==="""
    # map.save_grid(grid_filename, grid)
    # map.save_grid(skeleton_file_name, skeleton)
    # map.save_grid(path_file_name, absolute_path)
    # absolute_path: np.array = map.read_grid(file_path=path_file_name, dtype=np.int)  # read predefined path
    """=== Show map === """
    map.add_obstacles(grid, n=200)

    map.show_map(grid=grid, skeleton=skeleton, path=absolute_path, start=start_default, goal=goal, save_path='./data_storage/images/images.png')
    # map.animate_path(absolute_path, grid, absolute_path, skeleton,
    #                  start=start_default, goal=goal,
    #                  animation_speed=5)
    iteration = 0
    steps_taken = []
    starting_time = time.time()
    for node in absolute_path:
        wheelchair.move_to(target_node=node, grid=grid, show_map=False)
        steps_taken.append(wheelchair.current_position)
    # show the path followed by the wheelchair
    print(" TIME SPENT FOR MOVEMENT: ", time.time()-starting_time)
    map.animate_path(path=np.array(steps_taken), grid=grid, skeleton=skeleton)
    print('FINAL DESTINATION: ', absolute_path[-1])
