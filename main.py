import numpy as np
import map
from pprint import pprint
import sys
import config_extractor as config
from behavioral_model import IntelligentWheelchair

if __name__ == '__main__':
    """ === Get configuration === 
    It includes default data for testing the system, such as:
    1. Start coordinate
    2. Goal coordinate
    3. Safety distance for wheelchair
    4. Path to the colliders coordinates
    5. Path to save the map as csv file for better view of the array and debugging
    """
    start:              tuple = config.get('initial_position')
    goal:               tuple = config.get('final_position')
    safety_distance:    float = config.get('safety_distance')
    filename:           str   = config.get('colliders')
    grid_filename:      str   = config.get('grid_save')
    skeleton_file_name: str   = config.get('skeleton_save')

    """ === Map generation === """
    # convert coordinates into tuple
    start, goal = (start[0], start[1]), (goal[0], goal[1])
    # grid, skeleton, distances, original_data = map.create_custom_map(filename=filename, safety_distance=safety_distance)
    # if not map.valid_destination(grid, start, goal):
    #     map.show_map(grid, skeleton, start, goal)
    #     raise Exception
    """=== Generate wheelchair object ==="""
    wheelchair: IntelligentWheelchair = IntelligentWheelchair(current_position=start, current_angle=0)
    """=== Normalize Grid ==="""
    # map.normalize_grid(grid) # inplace action
    # skeleton = skeleton.astype(int)
    """=== Save Grids for providing multiple and different formats ==="""
    # map.save_grid(grid_filename, grid)
    # map.save_grid(skeleton_file_name, skeleton)
    """=== Read Grids for testing multiple versions ==="""
    grid     = map.read_grid(file_path=grid_filename, dtype=np.float)
    skeleton = map.read_grid(file_path=skeleton_file_name, dtype=np.int)
    """=== Print map data === """
    print(10*'='+f"Grid"+"="*10)
    pprint(grid)
    print(10 * '=' + f"Sceleton" + "=" * 10)
    pprint(skeleton)
    print(10 * '=' + f"Distances" + "=" * 10)
    # pprint(distances)
    # map.show_map(grid, skeleton, start, goal)
    """ === Path planning === """
    path, waypoint_values = map.get_path(grid=grid, start=start, goal=goal)
    if len(path) == 0:
        print('Path was not found.', file=sys.stderr)
        raise Exception
    path = map.reformat_path(path, start[0], start[1])
    wheelchair.move_to(next_node=path[-1])
    pprint(path[::-1]) # from the start
    map.show_map(grid, skeleton, start, goal, path)
