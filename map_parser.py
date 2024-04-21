import rospy
from nav_msgs.msg import OccupancyGrid, MapMetaData
import numpy as np
""" Connect to the topic """
rospy.init_node('map_parser', anonymous=True)
map_coordinates = rospy.wait_for_message('/map', topic_type=OccupancyGrid, timeout=10)
map_meta_data = rospy.wait_for_message('/map_metadata', topic_type=MapMetaData, timeout=10)


def parse_map_coordinates(data: OccupancyGrid, meta_data: MapMetaData):
    
    def normalize_matrix(arr: list):
        return [1 if x == -1 or x == 100 else x for x in arr]

    map_array = list(data.data)
    # print(map_array)
    # print(len(map_array))
    map_array = normalize_matrix(map_array)

    # Get the height and width of the picture
    h, n = meta_data.height, meta_data.width

    # Calculate the number of rows
    num_rows = len(map_array) // n
    
    # Create the 2D matrix
    matrix = []
    for i in range(num_rows):
        start = i * n
        end = start + n
        row = map_array[start:end]
        matrix.append(row)
    result = np.array(matrix)
    assert result.shape == (h, n), "INCORRECTLY PARSED MAP DATA!"
    return result


# Save data
map_array = parse_map_coordinates(map_coordinates, map_meta_data)
print(map_array.shape)
np.savetxt("testing_grid.csv", map_array, delimiter=',')

if __name__ == '__main__':
    import map as mp
    map = mp.Map()
    map.load_grid('testing_grid.csv', dtype=np.int8)
    map.create_skeleton()
    map.show_map(title="Testing Grid")
