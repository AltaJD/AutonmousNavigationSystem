import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication, QMainWindow, QGraphicsView, QWidget, QVBoxLayout
import numpy as np
from map import Map
import config


def get_map_image(coordinates: np.array) -> np.array:
    height, width = coordinates.shape
    image = np.zeros((height, width, 3), dtype=np.uint8)  # RGB image

    path_way_color = (255, 255, 255)
    obstacle_color = (0, 0, 0)
    for i in range(height):
        for j in range(width):
            if coordinates[i, j] == 0:
                image[i, j] = path_way_color
            else:
                image[i, j] = obstacle_color

    return image


if __name__ == '__main__':
    grid_file = config.get('grid_save')
    skeleton_file = config.get('skeleton_save')

    env = Map(map_image_size=12, safety_distance=0)
    env.load_grid(file_path=grid_file, dtype=np.int)
    env.load_skeleton(file_path=skeleton_file, dtype=np.int)

    app = QApplication([])
    map_image = get_map_image(env.grid)

    # Create a QMainWindow and set it up
    main_window = QMainWindow()
    main_window.setWindowTitle("Color Map")
    main_window.showMaximized()

    # Create a QWidget as the central widget
    central_widget = QWidget(main_window)
    main_window.setCentralWidget(central_widget)

    # Create a QVBoxLayout to hold the PlotWidget
    layout = QVBoxLayout(central_widget)

    # Create a PlotWidget from pyqtgraph and display the color map
    plot_widget = pg.PlotWidget()
    layout.addWidget(plot_widget)

    # Create an ImageItem from the map image and add it to the PlotWidget
    img_item = pg.ImageItem()
    plot_widget.addItem(img_item)
    img_item.setImage(map_image)

    # Set the aspect ratio and scale of the image
    plot_widget.setAspectLocked(True)
    plot_widget.setRange(xRange=[0, env.grid.shape[1]], yRange=[0, env.grid.shape[0]])

    main_window.show()
    app.exec_()
