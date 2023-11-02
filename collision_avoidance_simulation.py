import config_extractor as config
import map
import numpy as np
from LIDAR_simulation import LIDAR
from behavioral_model import IntelligentWheelchair

""" === Functions to get Vector Field Histogram === """


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
    print(lidar)
    lidar.scan(grid=grid, current_location=wheelchair.current_position)
    lidar.show_scanning_area(grid, skeleton, wheelchair.current_position)
