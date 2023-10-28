import map
from pprint import pprint
import sys
import config_extractor as config

if __name__ == '__main__':
    start: tuple = config.get('initial_position')
    goal:  tuple = config.get('final_position')
    safety_distance: float = config.get('safety_distance')
    print(start, goal, safety_distance)
    filename = 'data_storage/colliders.csv'
    grid_filename = 'data_storage/grid.csv'
    grid, skeleton, distances, original_data = map.get_custom_map(filename=filename, safety_distance=safety_distance)
    # map.save_grid(grid_filename, grid)
    map.normalized_grid(grid) # inplace = True
    if not map.valid_destination(grid, start, goal):
        map.show_map(grid, skeleton, start, goal)
        raise Exception
    print(10*'='+f"Grid"+"="*10)
    pprint(grid)
    print(10 * '=' + f"Sceleton" + "=" * 10)
    pprint(skeleton)
    print(10 * '=' + f"Distances" + "=" * 10)
    pprint(distances)
    map.show_map(grid, skeleton, start, goal)
    # path, waypoint_values = map.get_path(grid=grid, start=start, goal=goal)
    # if len(path) == 0:
    #     print('Path was not found.', file=sys.stderr)
    #     raise Exception
    # pprint(path[::-1]) # considering from the start point
    # path = map.reformat_path(path, start[0], start[1])
    # pprint(path)
    # map.show_map(grid, skeleton, start, goal, path)
