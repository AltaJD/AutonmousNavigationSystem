from map import Map
from wheelchair import IntelligentWheelchair
from collision_avoidance_simulation import VFH
import config as config
import sys
import numpy as np
import struct


def main_simulation():
    from lidar_simulation import LidarSimulation
    """ === Get configuration === """
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
    lidar_simulation = LidarSimulation(lidar_radius)

    # prepare VFH
    vfh = VFH(a=config.get('a'),
              b=config.get('b'),
              alpha=config.get('sector_angle'),
              l_param=config.get('l'),
              threshold=config.get('vfh_threshold'))
    vfh.update_measurements(lidar_simulation.get_values())

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

    """ Visualizing path taken by the simulation """
    env_map.path.waypoints = np.array(path_taken) # change the generated path to path taken
    env_map.show_path() # show path taken


def main():
    from lidar import LIDAR, PointUnitree, ScanUnitree, IMUUnitree
    print("Testing real lidar")
    lidar = LIDAR()
    vfh = VFH(a=config.get('a'),
              b=config.get('b'),
              alpha=config.get('sector_angle'),
              l_param=config.get('l'),
              threshold=config.get('vfh_threshold'))
    sock = lidar.get_socket()
    print("pointSize = " + str(lidar.pointSize) +
          ", scanDataSize = " + str(lidar.scanDataSize) +
          ", imuDataSize = " + str(lidar.imuDataSize))
    iteration = 0
    while True:
        # Recv data
        data, addr = sock.recvfrom(10000)
        print(f"Received data from {addr[0]}:{addr[1]}")

        msgType = struct.unpack("=I", data[:4])[0]
        print("msgType =", msgType)

        if msgType == 101:  # IMU Message
            length = struct.unpack("=I", data[4:8])[0]
            imuData = struct.unpack(lidar.imuDataStr, data[8:8 + lidar.imuDataSize])
            imuMsg = IMUUnitree(imuData[0], imuData[1], imuData[2:6], imuData[6:9], imuData[9:12])
            print("An IMU msg is parsed!")
            print("\tstamp =", imuMsg.stamp, "id =", imuMsg.id)
            print("\tquaternion (x, y, z, w) =", imuMsg.quaternion)
            print("\tangular velocity =", imuMsg.angular_velocity)
            print("\tlinear acceleration =", imuMsg.linear_acceleration)
            print("\n")

            # Update current position
            lidar.update_position(x=imuMsg.quaternion[0], y=imuMsg.quaternion[1], z=imuMsg.quaternion[2])

        elif msgType == 102:  # Scan Message
            length = struct.unpack("=I", data[4:8])[0]
            stamp = struct.unpack("=d", data[8:16])[0]
            id = struct.unpack("=I", data[16:20])[0]
            validPointsNum = struct.unpack("=I", data[20:24])[0]
            scanPoints = []
            pointStartAddr = 24
            for i in range(validPointsNum):
                pointData = struct.unpack(lidar.pointDataStr, data[pointStartAddr: pointStartAddr + lidar.pointSize])
                pointStartAddr = pointStartAddr + lidar.pointSize
                point = PointUnitree(*pointData)
                scanPoints.append(point)
            scanMsg = ScanUnitree(stamp, id, validPointsNum, scanPoints)
            print("A Scan msg is parsed!")
            print("\tstamp =", scanMsg.stamp, "id =", scanMsg.id)
            print("\tScan size =", scanMsg.validPointsNum)
            print("\tfirst 10 points (x, y, z, intensity, time, ring) =")
            for i in range(min(10, scanMsg.validPointsNum)):
                point = scanMsg.points[i]
                print("\t", point.x, point.y, point.z, point.intensity, point.time, point.ring)
            print("\n")
            # Save data
            lidar.update_values(scanMsg)
            vfh.update_measurements(lidar.get_values())
            vfh.generate_vfh()
            iteration += 1
            # vfh.show_histogram()
            print("ITERATION: ", iteration)
            print("HISTOGRAM: ", vfh.histogram)
            print("BEST ANGLE: ", vfh.get_rotation_angle(current_node=(lidar.x, lidar.y), next_node=(lidar.x+1, lidar.y)))
    sock.close()


if __name__ == '__main__':
    if sys.version_info[0:2] != (3, 6):
        raise Exception('Requires python 3.6')
    main()
