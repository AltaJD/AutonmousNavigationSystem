import map as mp
import numpy as np

if __name__ == '__main__':
    path_map = "data_storage/testing_grid.csv"
    environment = mp.Map()
    environment.load_grid(path_map, dtype=np.int8)
    environment.create_skeleton()
    obstacle_coordinates = environment.select_point(title="Add obstacle")
    environment.add_obstacle_at(obstacle_coordinates, w=1)
    environment.show_map(title="FJ Laboratory Map")
    # environment.save_map(path_map)
