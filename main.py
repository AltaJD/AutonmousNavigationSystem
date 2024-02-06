from map import Map
from LIDAR_simulation import LIDAR
from wheelchair import IntelligentWheelchair
from collision_avoidance_simulation import VFH
import config as config
import sys
import numpy as np


def main():
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
    # start_default:      tuple = config.get('initial_position')
    # goal_default:       tuple = config.get('final_position')
    safety_distance:    float   = config.get('safety_distance')
    filename:           str     = config.get('colliders')
    lidar_radius:       int     = config.get('lidar_radius')
    map_figure_size:    int     = config.get('map_figure_size')

    # prepare Map
    env_map = Map(filename, map_figure_size, safety_distance)
    start = env_map.select_start()
    end = env_map.select_end()
    env_map.create_path(start, end)
    env_map.show_path()

    # prepare LIDAR
    lidar_simulation = LIDAR(lidar_radius)

    # prepare VFH
    vfh = VFH(env=env_map,
              a=config.get('a'),
              b=config.get('b'),
              alpha=config.get('sector_angle'),
              l_param=config.get('l'),
              threshold=config.get('vfh_threshold'),
              lidar_measurements=lidar_simulation.get_values())

    # create Wheelchair
    intel_wheelchair = IntelligentWheelchair(current_position=env_map.path.start,
                                             current_angle=0,
                                             lidar=lidar_simulation,
                                             env=env_map)

    path = env_map.path
    path_taken = []
    for coord in path.waypoints:
        intel_wheelchair.move_to(target_node=coord, vfh=vfh, show_map=False)
        print(intel_wheelchair.current_position)
        path_taken.append([intel_wheelchair.current_position[0], intel_wheelchair.current_position[1]])

    env_map.path.waypoints = np.array(path_taken)
    env_map.show_path()


if __name__ == '__main__':
    main()
