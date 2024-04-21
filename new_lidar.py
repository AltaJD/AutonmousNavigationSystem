from typing import List
from common_functions import get_distance, get_vector_angle, convert_to_degrees
import config
import math
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import PointCloud2, PointField, Imu
import sensor_msgs.point_cloud2 as pc2
import struct
import numpy as np
import time
import threading


class NewLIDAR:
    values: List[tuple]
    current_angle: float
    z_threshold: float
    x: float
    y: float
    z: float
    start_blind_spot: int
    end_blind_spot: int
    topic_point_cloud: str
    topic_imu: str

    def __init__(self) -> None:
        self.z_threshold = config.get('lidar_height_threshold')
        self.current_angle = 0.0
        self.start_blind_spot = config.get('blind_spot_range')[0]
        self.end_blind_spot = config.get('blind_spot_range')[1]
        self.x = 0
        self.y = 0
        self.z = 0
        self.values = []
        """ Setup topics for comm with the new lidar """
        self.topic_point_cloud = "/livox/lidar"
        self.topic_imu = "/livox/imu"


    def append_values(self, measurements: List[List[float]]) -> None:
        """
        Convert the (x,y,z,w) values to the appropriate values
        for VFH generation
            Measurements structure:
            [[x,y,z]]
        """
        current_position = (self.x, self.y)
        for cloud_point in measurements:
            # neglect third dimension
            if cloud_point[2] > self.z_threshold:
                continue
            # calculate data
            x = cloud_point[0]
            y = cloud_point[1]
            distance:   float = get_distance(current_node=current_position, next_node=(x, y))
            angle:      float = get_vector_angle(current_node=current_position, next_node=(x, y))
            # neglect blind spots
            # if self.start_blind_spot < convert_to_degrees(angle) < self.end_blind_spot:
            #     continue
            # zip data
            record = (angle, distance)
            self.values.append(record)
    
    def empty_values(self):
        self.values.clear()

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

    def process_scan_data(self, data: PointCloud2):
        """ Function to extract the obstacle coordinates """
        # Extract the field names for x, y, and z coordinates
        field_names = [field.name for field in data.fields]
        if 'x' not in field_names or 'y' not in field_names or 'z' not in field_names:
            rospy.logwarn("PointCloud2 does not contain x, y, or z fields.")
            return
        
        """ Convert the coordinates to measurements type """
        processed_points = []
        for point in pc2.read_points(data, field_names=('x', 'y', 'z'), skip_nans=True):
            x, y, z = point
            processed_points.append([x,y,z])
        
        self.append_values(measurements=processed_points)

    def update_position(self, x: float, y: float, z: float):
        """ Updates the current coordinates of the LIDAR sensor """
        self.x = x
        self.y = y
        self.z = z

    def get_valid_points_num(self) -> int:
        return len(self.values)
    
    def update_imu_data(self, data: Imu) -> None:
        """ Update the current direction and position of the lidar """
        def get_imu_chars(orient):
            x = orient.x
            y = orient.y
            z = orient.z
            w = orient.w
            return x, y, z, w
        
        print("Parsing IMU Data")
        print(data.orientation)
        x, y, z, w = get_imu_chars(data.orientation)
        print(x, y, z, w)

        # Convert quaternion to Euler angles
        roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        pitch = math.asin(2 * (w * y - z * x))
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

        # Convert the yaw angle to degrees
        yaw_degrees = math.degrees(yaw)
        yaw_degrees = round(yaw_degrees)

        # Adjust the yaw angle to be within the range of [0, 360)
        if yaw_degrees < 0:
            yaw_degrees += 360

        self.current_angle = yaw_degrees


def point_cloud_callback(data):
    # Convertimos el PointCloud2 ajustando los campos 'line' a 'ring' y 'timestamp' a 'time'
    new_fields = []
    # Inicializar el ajuste del offset
    offset_adjustment = 0
    for field in data.fields:
        # Ajustar el offset para cada campo despus de remover 'tag'
        if field.name == 'line':
            # Cambiamos el nombre del campo de 'line' a 'ring' y ajustamos su tipo y offset
            # Suponemos que el campo 'ring' debe ser UINT16 (2 bytes) (de uint8 a uint16)
            new_field = PointField(name='ring', offset=16,
                                   datatype=4, count=field.count)
            offset_adjustment += 1  # 'ring' ahora es de 2 bytes, 'line' era de 1 byte
        elif field.name == 'timestamp':
            # Cambiamos el nombre del campo de 'timestamp' a 'time', ajustamos su tipo y offset
            # Suponemos que el campo 'time' debe ser FLOAT32 (4 bytes) (float64 a float32)
            new_field = PointField(name='time', offset=18,
                                   datatype=7, count=field.count)
            # Ajustamos el offset de acuerdo al nuevo tamao de 'time' (FLOAT32)
            offset_adjustment += 4  # timestamp era FLOAT64 (8 bytes), ahora es FLOAT32 (4 bytes)
        elif field.name == 'tag':
            # Si el campo es 'tag', lo omitimos y ajustamos el offset en 1 byte
            offset_adjustment += 1
            continue
        else:
            new_field = field
            #new_field.offset -= offset_adjustment
        new_fields.append(new_field)

    # Crear un nuevo PointCloud2 con los campos actualizados
    transformed_cloud = PointCloud2()
    transformed_cloud.header = data.header
    transformed_cloud.height = data.height
    transformed_cloud.width = data.width
    transformed_cloud.fields = new_fields
    transformed_cloud.is_bigendian = data.is_bigendian
    transformed_cloud.point_step = data.point_step -4#- offset_adjustment
    transformed_cloud.row_step = transformed_cloud.point_step * data.width
    transformed_cloud.is_dense = data.is_dense
    
    # Reestructurar 'data' para eliminar el campo 'tag' y ajustar el campo 'timestamp'
    new_data = bytearray()
    for i in range(0, len(data.data), data.point_step):
        point_data = data.data[i:i+data.point_step]
        
        # x, y, z, intensity (cada uno FLOAT32, 4 bytes)
        new_data.extend(point_data[0:16])
        # ring (UINT16, 2 bytes), leemos los datos originales como UINT8 y los empacamos como UINT16
        line_value = struct.unpack_from('B', point_data, 17)[0]  # Leer 'line' como UINT8
        new_data.extend(struct.pack('H', line_value))  # Convertir y escribir 'ring' como UINT16
       
        # time (FLOAT32, 4 bytes), convertido de FLOAT64
        timestamp = struct.unpack_from('d', point_data, 18)[0] # Leer como FLOAT64
        # time= np.float32(np.float64(timestamp)-np.float64(timestamp) ) # no se xq pero funciona con este
        time = np.float32(np.float64(timestamp)-17e+18 ) # Con este tambien funciona
        new_data.extend(struct.pack('f', time))  # Escribir como FLOAT32

    transformed_cloud.data = bytes(new_data)

    new_lidar.process_scan_data(data=transformed_cloud)

    # rospy.loginfo("Published transformed PointCloud2")    


def imu_callback(data):
    rospy.loginfo("Received message: %s", data)
    new_lidar.update_imu_data(data)


new_lidar = NewLIDAR()
""" Connect to the lidar topics """
rospy.init_node("an_program", anonymous=True)
point_cloud_listener = rospy.Subscriber(new_lidar.topic_point_cloud,
                                        PointCloud2,
                                        callback=point_cloud_callback)

imu_listener = rospy.Subscriber(new_lidar.topic_imu,
                                Imu,
                                callback=imu_callback)


def start_event_loop():
    rospy.spin()


def stop_comm():
    global imu_listener
    global point_cloud_listener
    imu_listener.unregister()
    point_cloud_listener.unregister()

    def hook():
        print("shutdown time!")

    rospy.on_shutdown(hook)
    rospy.signal_shutdown(reason=None)


def start_comm():
    thread1 = threading.Thread(target=start_event_loop, daemon=True)
    thread1.start()
    return thread1


if __name__ == "__main__":
    
    thd = start_comm()
    time.sleep(0.5)
    stop_comm()
    print("Stop all communications")
    thd.join()

    # start_comm()
    # time.sleep(1)
    # stop_comm()
    # print("Stop all communications")
    # thd.join()

    # print("Processed values: ", new_lidar.get_values())
    print("The current direction of the LIDAR: ", new_lidar.current_angle)

