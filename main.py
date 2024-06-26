import matplotlib.pyplot as plt

from lidar import LIDAR
from collision_avoidance_simulation import VFH
from wheelchair import IntelligentWheelchairSim, IntelligentWheelchair, WheelchairStatus
from map import Map
import config
import numpy as np
import struct
import time
import threading


class AutonomousNavigationProgram:
    destination: tuple  # tuple of floats
    interrupt_program: bool
    show_histogram: bool

    def __init__(self):
        self.destination = (0.1033686161041259766, 0.0)  # (x, y)
        self.interrupt_program = False
        self.show_histogram = False

    def testing_run(self):
        print("Start listening to LIDAR")
        while not self.interrupt_program:
            print("HI")
        else:
            print("Testing Run has been successfully stopped")

    def run(self):
        print("Start listening to LIDAR")
        lidar = LIDAR()
        vfh = VFH(b=config.get('b'),
                  alpha=config.get('sector_angle'),
                  l_param=config.get('l'),
                  safety_distance=config.get('safety_distance'),
                  keep_images=self.show_histogram)
        env_map = Map()
        env_map.load_grid(config.get('grid_save'), dtype=np.int8)
        env_map.load_skeleton(config.get('skeleton_save'), dtype=np.int8)
        wheelchair = IntelligentWheelchair(current_position=(0, 0),
                                           current_angle=lidar.current_angle,
                                           lidar=lidar,
                                           env=env_map)
        wheelchair.status = WheelchairStatus.IDLE.value
        sock = lidar.get_socket()
        iteration = 0
        start_time = time.time()

        emergency_stop: bool = wheelchair.status == WheelchairStatus.INTERRUPTED.value

        while not emergency_stop:
            if self.interrupt_program:
                break
            # Recv data
            data, addr = sock.recvfrom(10000)
            # print(f"Received data from {addr[0]}:{addr[1]}")

            msg_type = struct.unpack("=I", data[:4])[0]

            if msg_type == 101:  # IMU Message
                """=== Process new position of the LIDAR ==="""
                lidar.update_imu_data(data)

            elif msg_type == 102:  # Scan Message
                """=== Preprocess received data ==="""
                lidar.process_scan_data(data)
                time_diff = round(time.time() - start_time, 4) * 1000  # in milliseconds
                vfh.update_measurements(lidar.get_values())
                if time_diff < config.get('vfh_time_delay'):
                    continue  # set a delay for collecting enough points

                """=== Generate VFH ==="""
                vfh.generate_vfh(blind_spot_overflow=False, blind_spot_range=(lidar.start_blind_spot,
                                                                              lidar.end_blind_spot))
                steering_direction = vfh.get_rotation_angle(current_node=(lidar.x, lidar.y),
                                                            next_node=self.destination)
                if steering_direction == -1:
                    wheelchair.stop()
                processing_time = round(time.time() - start_time, 4) * 1000  # processing time in ms
                vfh.update_free_sectors_num(num=vfh.get_free_sectors_num(), time=processing_time)
                wheelchair.move_to(steering_direction, 1.0)
                """=== Show the results ==="""
                # print("ITERATION: ", iteration)
                # print("Cloud points num: ", lidar.get_valid_points_num())
                # print("HISTOGRAM: ", vfh.histogram)
                # print("ANGLES AND DISTANCES: ", lidar.values)
                # print(f"BLIND SPOT: from {lidar.start_blind_spot} to {lidar.end_blind_spot} degrees")
                # print("=" * 10 + f"Received {len(lidar.values)} LIDAR points  in {processing_time} ms" + 10 * "=")
                if self.show_histogram is True:
                    vfh.show_histogram()

                """=== Reset values ==="""
                lidar.empty_values()
                vfh.empty_histogram()
                start_time = time.time()
                iteration += 1

            # print("STATUS: ", wheelchair.status)
        wheelchair.stop()
        print("ANP has been successfully stopped")
        sock.close()

    @staticmethod
    def run_simulation():
        from lidar_simulation import LidarSimulation
        """ === Get configuration === """
        lidar_radius: int = config.get('lidar_radius')
        wheelchair_direction = 0

        # prepare Map
        env_map = Map(map_image_size=10)
        env_map.load_grid(config.get('lab_grid_save'), dtype=np.int)
        env_map.create_skeleton()
        start = env_map.select_start()
        end = env_map.select_end()
        env_map.create_path(start, end)
        obstacle_coordinates = env_map.select_point(title="Add obstacle")
        env_map.add_obstacle_at(obstacle_coordinates, w=1)
        env_map.show_path()

        # prepare LIDAR
        lidar_simulation = LidarSimulation(lidar_radius, direction=wheelchair_direction)

        # prepare VFH
        vfh = VFH(b=config.get('b'),
                  alpha=config.get('sector_angle'),
                  l_param=config.get('l'),
                  safety_distance=config.get('safety_distance'),
                  keep_images=True)
        vfh.update_measurements(lidar_simulation.get_values())

        # create Wheelchair
        intel_wheelchair = IntelligentWheelchairSim(current_position=env_map.path.start,
                                                    current_angle=wheelchair_direction,
                                                    lidar_simulation=lidar_simulation,
                                                    env=env_map)

        path = env_map.path
        print('Path Length: ', len(path.waypoints)*config.get('map_density'))
        path_taken = []
        for coord in path.waypoints:
            intel_wheelchair.move_to(target_node=coord, vfh=vfh, show_map=True)
            path_taken.append([intel_wheelchair.current_position[0], intel_wheelchair.current_position[1]])
            if intel_wheelchair.status == WheelchairStatus.INTERRUPTED.value:
                break

        """ Visualizing path taken by the simulation """
        print("=== REACHED DESTINATION ===")
        # vfh.ax2.remove()
        # vfh.ax1.remove()
        # env_map.path.waypoints = np.array(path_taken)  # change the generated path to path taken
        # env_map.show_path()  # show path taken
        plt.show()


autonomous_navigation_program = AutonomousNavigationProgram()


def start_testing_run():
    autonomous_navigation_program.run()  # select the method to run


# Create a thread and start it
thread = threading.Thread(target=start_testing_run, daemon=True)


def start_program():
    thread.start()
    # thread.join()  # freezing the main thread until thread stops


def interrupt_program():
    autonomous_navigation_program.interrupt_program = True


if __name__ == '__main__':
    # start_time = time.time()
    # start_program()
    # time.sleep(3)  # 2 seconds
    # interrupt_program()
    # print(f"Program stopped after: {time.time()-start_time} seconds")
    autonomous_navigation_program.run_simulation()

