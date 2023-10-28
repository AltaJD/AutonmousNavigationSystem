import map
from pprint import pprint
import sys

if __name__ == '__main__':
    start = (25,  100)
    goal = (650, 530)
    filename = 'data_storage/colliders.csv'
    grid_filename = 'data_storage/grid.csv'
    grid, skeleton, distances, original_data = map.get_custom_map(filename=filename)
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
    path, waypoint_values = map.get_path(grid=grid, start=start, goal=goal)
    if len(path) == 0:
        print('Path was not found.', file=sys.stderr)
        raise Exception
    pprint(path[::-1]) # considering from the start point
    path = map.reformat_path(path, start[0], start[1])
    pprint(path)
    map.show_map(grid, skeleton, start, goal, path)
