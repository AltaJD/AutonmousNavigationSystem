from lidar import LIDAR, PointUnitree, ScanUnitree, IMUUnitree
from collision_avoidance_simulation import VFH
from wheelchair import IntelligentWheelchairSim, WheelchairStatus
from common_functions import get_vector_angle, get_distance
from map import Map
import config
import sys
import numpy as np
import struct
import time
import math

DESTINATION = (0.1033686161041259766, 0)  # (x, y)


def main_simulation():
    from lidar_simulation import LidarSimulation
    """ === Get configuration === """
    filename: str = config.get('colliders')
    lidar_radius: int = config.get('lidar_radius')
    map_figure_size: int = config.get('map_figure_size')
    wheelchair_direction = 0

    # prepare Map
    env_map = Map(map_image_size=10)
    env_map.load_grid(config.get('grid_save'), dtype=np.int)
    env_map.load_skeleton(config.get('skeleton_save'), dtype=np.int)
    start = env_map.select_start()
    end = env_map.select_end()
    env_map.create_path(start, end)
    env_map.show_path()

    # prepare LIDAR
    lidar_simulation = LidarSimulation(lidar_radius, direction=wheelchair_direction)

    # prepare VFH
    vfh = VFH(b=config.get('b'),
              alpha=config.get('sector_angle'),
              l_param=config.get('l'),
              safety_distance=config.get('safety_distance'))
    vfh.update_measurements(lidar_simulation.get_values())

    # create Wheelchair
    intel_wheelchair = IntelligentWheelchairSim(current_position=env_map.path.start,
                                                current_angle=wheelchair_direction,
                                                lidar_simulation=lidar_simulation,
                                                env=env_map)

    path = env_map.path
    path_taken = []
    for coord in path.waypoints:
        intel_wheelchair.move_to(target_node=coord, vfh=vfh, show_map=True)
        path_taken.append([intel_wheelchair.current_position[0], intel_wheelchair.current_position[1]])
        if intel_wheelchair.status == WheelchairStatus.INTERRUPTED.value:
            break

    """ Visualizing path taken by the simulation """
    print("=== REACHED DESTINATION ===")
    vfh.ax2.remove()
    vfh.ax1.remove()
    env_map.path.waypoints = np.array(path_taken)  # change the generated path to path taken
    env_map.show_path()  # show path taken


def main(show_histogram=None):
    def show_scan_message(message: ScanUnitree, points_num: int) -> None:
        """ Print raw data received from the LIDAR """
        print("A Scan msg is parsed!")
        print("\tstamp =", message.stamp, "id =", message.id)
        print("\tScan size =", message.validPointsNum)
        print("\tfirst 10 points (x, y, z, intensity, time, ring) =")
        for i in range(min(points_num, message.validPointsNum)):
            cloud_point = message.points[i]
            print("\t", cloud_point.x, cloud_point.y, cloud_point.z, cloud_point.intensity, cloud_point.time, cloud_point.ring)
        print("\n")

    def get_rotation_angle(imu: IMUUnitree):
        # Access the quaternion values from the IMU message
        x = imu.quaternion[0]
        y = imu.quaternion[1]
        z = imu.quaternion[2]
        w = imu.quaternion[3]

        # Convert quaternion to Euler angles
        roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        pitch = math.asin(2 * (w * y - z * x))
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

        # Convert the yaw angle to degrees
        yaw_degrees = math.degrees(yaw)

        # Adjust the yaw angle to be within the range of [0, 360)
        if yaw_degrees < 0:
            yaw_degrees += 360

        return yaw_degrees

    lidar = LIDAR()
    vfh = VFH(b=config.get('b'),
              alpha=config.get('sector_angle'),
              l_param=config.get('l'),
              safety_distance=config.get('safety_distance'))
    sock = lidar.get_socket()
    print("Testing real lidar")
    # print("pointSize = " + str(lidar.pointSize) +
    #       ", scanDataSize = " + str(lidar.scanDataSize) +
    #       ", imuDataSize = " + str(lidar.imuDataSize))
    iteration = 0
    start_time = time.time()

    while True:
        # Recv data
        data, addr = sock.recvfrom(10000)
        # print(f"Received data from {addr[0]}:{addr[1]}")

        msg_type = struct.unpack("=I", data[:4])[0]
        # print("msgType =", msg_type)

        if msg_type == 101:  # IMU Message
            length = struct.unpack("=I", data[4:8])[0]
            imu_data = struct.unpack(lidar.imuDataStr, data[8:8 + lidar.imuDataSize])
            imu_msg = IMUUnitree(imu_data[0], imu_data[1], imu_data[2:6], imu_data[6:9], imu_data[9:12])
            # print("An IMU msg")
            # print("\tstamp =", imu_msg.stamp, "id =", imu_msg.id)
            # print("\tquaternion (x, y, z, w) =", imu_msg.quaternion)
            # print("\tangular velocity =", imu_msg.angular_velocity)
            # print("\tlinear acceleration =", imu_msg.linear_acceleration)
            # print("\n")
            """=== Process new position of the LIDAR ==="""
            angle = get_rotation_angle(imu_msg)
            print("ROTATION ANGLE: ", angle)

        elif msg_type == 102:  # Scan Message
            """ Preprocess received data """
            length = struct.unpack("=I", data[4:8])[0]
            stamp = struct.unpack("=d", data[8:16])[0]
            id_msg = struct.unpack("=I", data[16:20])[0]
            valid_points_num = struct.unpack("=I", data[20:24])[0]
            scan_points = []
            point_start_addr = 24
            for i in range(valid_points_num):
                pointData = struct.unpack(lidar.pointDataStr, data[point_start_addr: point_start_addr + lidar.pointSize])
                point_start_addr = point_start_addr + lidar.pointSize
                point = PointUnitree(*pointData)
                scan_points.append(point)
            scan_msg = ScanUnitree(stamp, id_msg, valid_points_num, scan_points)
            # show_scan_message(scan_msg, 10)
            """=== Collect data ==="""
            time_diff = round(time.time() - start_time, 4) * 1000  # in milliseconds
            lidar.append_values(scan_msg)
            vfh.update_measurements(lidar.get_values())
            if time_diff < config.get('vfh_time_delay'):
                continue  # set a delay for collecting enough points
            """=== Generate VFH ==="""
            vfh.generate_vfh(blind_spot_overflow=False, blind_spot_range=(lidar.start_blind_spot,
                                                                          lidar.end_blind_spot))
            steering_direction = vfh.get_rotation_angle(current_node=(lidar.x, lidar.y),
                                                        next_node=DESTINATION)

            """=== Show the results ==="""
            # print("ITERATION: ", iteration)
            # print("Cloud points num: ", valid_points_num)
            # print("BEST ANGLE: ", steering_direction)
            # print("HISTOGRAM: ", vfh.histogram)
            # print("ANGLES AND DISTANCES: ", lidar.values)
            # print(f"BLIND SPOT: from {lidar.start_blind_spot} to {lidar.end_blind_spot} degrees")
            if show_histogram is True: vfh.show_histogram()

            """=== Reset values ==="""
            processing_time = round(time.time() - start_time, 4) * 1000  # processing time in ms
            vfh.update_free_sectors(num=vfh.get_free_sectors_num(), time=processing_time)
            lidar.empty_values()
            vfh.empty_histogram()
            start_time = time.time()
            iteration += 1
            # print("=" * 10 + f"Received {len(lidar.values)} LIDAR points  in {processing_time} ms" + 10 * "=")
    sock.close()


if __name__ == '__main__':
    if sys.version_info[0:2] != (3, 6):
        raise Exception('Requires python 3.6')
    main(show_histogram=False)
