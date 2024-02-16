import numpy as np
from map import Map
import matplotlib.pyplot as plt


def growing_algorithm(grid, seed, threshold):
    # Create a mask to store the regions where the algorithm has expanded
    mask = np.zeros_like(grid, dtype=np.uint8)

    # Set the seed point as the starting point
    seed_point = tuple(seed)

    # Set up a queue to store the points to be expanded
    queue = []
    queue.append(seed_point)

    # Start the growing algorithm
    while len(queue) > 0:
        # Get the next point from the queue
        current_point = queue.pop(0)
        x = current_point[0]
        y = current_point[1]
        # Check if the current point is within the grid boundaries
        if (0 <= y < grid.shape[1]) and (0 <= x < grid.shape[0]):
            # Check if the current point is already expanded or not
            if mask[x, y] == 0:
                # Check if the current point has a value of 1
                if grid[x, y] == threshold:
                    # Expand the region by setting the pixel value and updating the mask
                    mask[x, y] = threshold

                    # Add the neighboring points to the queue for further expansion
                    queue.append((current_point[0] - 1, current_point[1]))  # Left neighbor
                    queue.append((current_point[0] + 1, current_point[1]))  # Right neighbor
                    queue.append((current_point[0], current_point[1] - 1))  # Top neighbor
                    queue.append((current_point[0], current_point[1] + 1))  # Bottom neighbor

    # Find the coordinates of the regions with only 1s
    segment_coordinates = np.argwhere(mask == 1)
    return segment_coordinates


import config
file_path = "data_storage/images/test_results/map.png"

safety_distance: int = config.get('safety_distance')
filename: str = config.get('colliders')
testing_map = Map(filename, 10, safety_distance)

# Set the seed point (e.g., (x, y) coordinates)
seed_point = testing_map.select_point(title="Select Seed")
# seed_point = seed_point[::-1] # swap x and y
print("Initial points", seed_point)

# Set the threshold for expanding the region
threshold = 1  # 1 for outdoor environment and 0 for indoor environment

# Apply the growing algorithm
coordinates = growing_algorithm(testing_map.grid, seed_point, threshold)
print(coordinates)

x_coords = [coord[1] for coord in coordinates]
y_coords = [coord[0] for coord in coordinates]

# Plot the coordinates
plt.plot(x_coords, y_coords, 'bo')

# Set the axis labels and title
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Plot of Coordinates')

# Display the plot
plt.show()
