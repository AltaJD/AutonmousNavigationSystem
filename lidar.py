import socket
import struct
from typing import List
from common_functions import get_distance, get_vector_angle, convert_to_degrees
import config
import math


class PointUnitree:
    def __init__(self, x, y, z, intensity, time, ring):
        self.x = x
        self.y = y
        self.z = z
        self.intensity = intensity
        self.time = time
        self.ring = ring


# Scan Type
class ScanUnitree:
    def __init__(self, stamp, id, validPointsNum, points: List[PointUnitree]):
        self.stamp = stamp
        self.id = id
        self.validPointsNum = validPointsNum
        self.points = points


# IMU Type
class IMUUnitree:
    def __init__(self, stamp, id, quaternion, angular_velocity, linear_acceleration):
        self.stamp = stamp
        self.id = id
        self.quaternion = quaternion
        self.angular_velocity = angular_velocity
        self.linear_acceleration = linear_acceleration


class LIDAR:
    """ The object will contain measurement results as an angle and distance toward obstacle
    The real LIDAR system is scanning the environment and send the results one-by-one, similarly to the queue:
    FIFO (First-In-First-Out)
    """
    values: List[tuple]
    current_angle: float
    z_threshold: float
    x: float
    y: float
    z: float
    start_blind_spot: int
    end_blind_spot: int

    """ Values to extract and communicate with the LIDAR """
    UDP_IP = "127.0.0.1"
    UDP_PORT = 12345
    imuDataStr = "=dI4f3f3f"
    pointDataStr = "=fffffI"
    scanDataStr = "=dII" + 120 * "fffffI"

    def __init__(self, max_height_threshold=config.get('lidar_height_threshold'), lidar_x=0, lidar_y=0, lidar_z=0):
        self.z_threshold = max_height_threshold
        self.start_blind_spot = config.get('blind_spot_range')[0]
        self.end_blind_spot   = config.get('blind_spot_range')[1]
        self.x = lidar_x
        self.y = lidar_y
        self.z = lidar_z
        self.values = []
        """ Configure LIDAR for data parsing """
        self.imuDataSize = struct.calcsize(self.imuDataStr)
        self.pointSize = struct.calcsize(self.pointDataStr)
        self.scanDataSize = struct.calcsize(self.scanDataStr)

    def append_values(self, measurements: ScanUnitree) -> None:
        """
        Convert the (x,y,z,w) values to the appropriate values
        for VFH generation
        """
        current_position = (self.x, self.y)
        for cloud_point in measurements.points:
            # neglect third dimension
            if cloud_point.z > self.z_threshold:
                continue
            # calculate data
            x = cloud_point.x
            y = cloud_point.y
            distance:   float = get_distance(current_node=current_position, next_node=(x, y))
            angle:      float = get_vector_angle(current_node=current_position, next_node=(x, y))
            # neglect blind spots
            if self.start_blind_spot < convert_to_degrees(angle) < self.end_blind_spot:
                continue
            # zip data
            record = (angle, distance)
            self.values.append(record)

    def empty_values(self):
        self.values.clear()

    def update_position(self, x: float, y: float, z: float):
        """ Updates the current coordinates of the LIDAR sensor """
        self.x = x
        self.y = y
        self.z = z

    def get_values(self) -> list:
        """ Assumption: LIDAR provides measurements one-by-one in the format:
        angle, distance for each point
        The values are appended to the list for storage and easier data validation.
        :returns [(angle1, distance1), (angle2, distance2)]
        """
        assert self.values is not None, "LIDAR has not updated measurements values"
        self.values.sort()  # sort by angle in ascending order
        if len(self.values) == 0:
            return [(0, 0)]
        # self.values.task_done()
        return self.values

    def process_scan_data(self, data):
        length = struct.unpack("=I", data[4:8])[0]
        stamp = struct.unpack("=d", data[8:16])[0]
        id_msg = struct.unpack("=I", data[16:20])[0]
        valid_points_num = struct.unpack("=I", data[20:24])[0]
        scan_points = []
        point_start_addr = 24
        for i in range(valid_points_num):
            pointData = struct.unpack(self.pointDataStr, data[point_start_addr: point_start_addr + self.pointSize])
            point_start_addr = point_start_addr + self.pointSize
            point = PointUnitree(*pointData)
            scan_points.append(point)
        scan_msg = ScanUnitree(stamp, id_msg, valid_points_num, scan_points)
        # show_scan_message(scan_msg, 10)
        """=== Collect data ==="""
        self.append_values(scan_msg)

    def get_socket(self):
        """ Create UDP socket """
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((self.UDP_IP, self.UDP_PORT))
        return sock

    def get_valid_points_num(self) -> int:
        return len(self.values)

    def update_imu_data(self, data) -> None:
        length = struct.unpack("=I", data[4:8])[0]
        imu_data = struct.unpack(self.imuDataStr, data[8:8 + self.imuDataSize])
        imu = IMUUnitree(imu_data[0], imu_data[1], imu_data[2:6], imu_data[6:9], imu_data[9:12])
        # Access the quaternion values from the IMU message
        x = imu.quaternion[0]
        y = imu.quaternion[1]
        z = imu.quaternion[2]
        w = imu.quaternion[3]

        # Convert quaternion to Euler angles
        # roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        # pitch = math.asin(2 * (w * y - z * x))
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

        # Convert the yaw angle to degrees
        yaw_degrees = math.degrees(yaw)

        # Adjust the yaw angle to be within the range of [0, 360)
        if yaw_degrees < 0:
            yaw_degrees += 360

        self.current_angle = yaw_degrees

    @staticmethod
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

    @staticmethod
    def show_imu_message(message: IMUUnitree) -> None:
        print("An IMU msg")
        print("\tstamp =", message.stamp, "id =", message.id)
        print("\tquaternion (x, y, z, w) =", message.quaternion)
        print("\tangular velocity =", message.angular_velocity)
        print("\tlinear acceleration =", message.linear_acceleration)
        print("\n")


def run_lidar_test(lidar: LIDAR):
    """ Testing Lidar communication """
    sock = lidar.get_socket()
    print("pointSize = " + str(lidar.pointSize) +
          ", scanDataSize = " + str(lidar.scanDataSize) +
          ", imuDataSize = " + str(lidar.imuDataSize))

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
            # lidar.update_position(x=imuMsg.quaternion[0], y=imuMsg.quaternion[1], z=imuMsg.quaternion[2])

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
            lidar.append_values(scanMsg)
            print("LIDAR values:")
            print(lidar.values)
    sock.close()


if __name__ == '__main__':
    lidar_sensor = LIDAR()
    run_lidar_test(lidar_sensor)
    values = lidar_sensor.get_values()
    print("LIDAR values:")
    print(values)
