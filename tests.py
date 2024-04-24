import map as mp
import numpy as np

if __name__ == '__main__':
    path_map = "data_storage/testing_grid.csv"
    environment = mp.Map()
    environment.load_grid(path_map, dtype=np.int8)
    environment.create_skeleton()
    # environment.show_map(title="FJ Laboratory Map")
    start_pos = environment.select_start()
    end_pos   = environment.select_end()
    environment.create_path(start=start_pos, end=end_pos)
    environment.show_path()
