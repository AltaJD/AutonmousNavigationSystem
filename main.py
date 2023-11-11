import numpy as np
import map
from pprint import pprint
import sys
import config_extractor as config
from behavioral_model import IntelligentWheelchair
import sys



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
    grid, skeleton, distances, original_data = map.create_custom_map(filename=filename, safety_distance=safety_distance)
    # select start and goal locations
    goal = map.select_point(grid, skeleton, start_default)
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
    """=== Read Grids for testing multiple versions in the future ==="""
    # grid     = map.read_grid(file_path=grid_filename, dtype=np.float)
    # skeleton = map.read_grid(file_path=skeleton_file_name, dtype=np.int)
    """=== Print map data === """
    print(10*'='+f"Grid"+"="*10)
    pprint(grid)
    print(10 * '=' + f"Sceleton" + "=" * 10)
    pprint(skeleton)
    print(10 * '=' + f"Distances" + "=" * 10)
    # pprint(distances)
    # map.show_map(grid, skeleton, start_default, goal_default)
    """ === Path planning === """
    absolute_path: np.array = map.get_path(grid=grid, start=start_default, goal=goal) # get list of waypoints
    # map.save_grid(path_file_name, absolute_path)
    # absolute_path: np.array = map.read_grid(file_path=path_file_name, dtype=np.int)
    map.show_map(grid=grid, skeleton=skeleton, path=absolute_path, start=start_default, goal=goal, save_path='./data_storage/images/images.png')
    map.animate_path(absolute_path, grid, absolute_path, skeleton,
                     start=start_default, goal=goal,
                     animation_speed=5)
