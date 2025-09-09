#!/usr/bin/env python3
"""
ROS2 driver for the DJI Tello drone.

This node interfaces with the Tello drone using the djitellopy library,
exposing control and sensor data through ROS2 topics. It allows for takeoff,
landing, RC control, and streams video, IMU, battery, and other status data.
"""

import pprint
import math
import rclpy
import threading
import numpy
import time
import av
import tf2_ros
import cv2
import yaml

from djitellopy import Tello

from rclpy.node import Node
from tello_msg.msg import TelloStatus, TelloID, TelloWifiConfig
from std_msgs.msg import Empty, UInt8, Bool, String
from sensor_msgs.msg import Image, Imu, BatteryState, Temperature, CameraInfo
from geometry_msgs.msg import Twist, TransformStamped
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
from ament_index_python.packages import get_package_share_directory

# Tello ROS node class, inherits from the Tello controller object.
#
# Can be configured to be used by multiple drones, publishes, all data collected from the drone and provides control using ROS messages.
class TelloNode(Node):
    """
    Tello ROS2 node class for controlling and receiving data from a Tello drone.

    This class initializes a connection to the drone, sets up ROS2 publishers
    and subscribers for various data streams and commands, and runs timers
    to periodically publish sensor data.
    """
    def __init__(self):
        """
        Initialize the TelloNode.

        Declares and retrieves ROS parameters, connects to the Tello drone,
        sets up camera and ROS interfaces, and starts timers for data publishing.
        """
        super().__init__('tello')

        # Declare parameters
        self.declare_parameter('connect_timeout', 10.0)
        self.declare_parameter('tello_ip', '192.168.10.1')
        self.declare_parameter('tf_base', 'map')
        self.declare_parameter('tf_drone', 'drone')
        self.declare_parameter('tf_pub', False)
        self.declare_parameter('camera_info_file', '')

        # Get parameters
        self.connect_timeout = float(self.get_parameter('connect_timeout').value)
        self.tello_ip = str(self.get_parameter('tello_ip').value)
        self.tf_base = str(self.get_parameter('tf_base').value)
        self.tf_drone = str(self.get_parameter('tf_drone').value)
        self.tf_pub = bool(self.get_parameter('tf_pub').value)
        self.camera_info_file = str(self.get_parameter('camera_info_file').value)

        # Camera related setup for video streaming
        self.setup_camera()

        # Configure drone connection
        Tello.TELLO_IP = self.tello_ip
        Tello.RESPONSE_TIMEOUT = int(self.connect_timeout)

        # Connect to drone
        self.get_logger().info('Tello: Connecting to drone...')
        self.tello = Tello()
        self.tello.connect()
        self.get_logger().info('Tello: Connected to drone!')

        # Publishers and subscribers
        self.setup_publishers()
        self.setup_subscribers()

        # Start timers for periodic data publishing
        self.video_timer = self.create_timer(1.0/10.0, self.cb_video_timer)     # 10 Hz
        self.odom_timer = self.create_timer(1.0/10.0, self.cb_odom_timer)       # 10 Hz
        self.status_timer = self.create_timer(1.0/2.0, self.cb_status_timer)    # 2 Hz

        self.get_logger().info('Tello: Driver node ready')

    def setup_camera(self):
        """
        Initialize camera-related objects and load calibration data.
        
        This method sets up the CvBridge for converting image formats and
        loads the camera calibration data from a specified YAML file.
        """        
        self.bridge = CvBridge()
        self.frame_reader = None
        self.camera_info = None

        # Check if camera info file was received as argument
        if len(self.camera_info_file) == 0:
            share_directory = get_package_share_directory('tello')
            self.camera_info_file = share_directory + '/ost.yaml'

        # Read camera info from YAML file
        with open(self.camera_info_file, 'r') as file:
            self.camera_info = yaml.load(file, Loader=yaml.FullLoader)
            # self.get_logger().info('Tello: Camera information YAML' + self.camera_info.__str__())

    def setup_publishers(self):
        """
        Create all necessary ROS publishers for the Tello drone data.
        """
        self.pub_image_raw = self.create_publisher(Image, 'image_raw', 1)
        self.pub_camera_info = self.create_publisher(CameraInfo, 'camera_info', 1)
        self.pub_status = self.create_publisher(TelloStatus, 'status', 1)
        self.pub_id = self.create_publisher(TelloID, 'id', 1)
        self.pub_imu = self.create_publisher(Imu, 'imu', 1)
        self.pub_battery = self.create_publisher(BatteryState, 'battery', 1)
        self.pub_temperature = self.create_publisher(Temperature, 'temperature', 1)
        self.pub_odom = self.create_publisher(Odometry, 'odom', 1)

        # TF broadcaster
        if self.tf_pub:
            self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
    
    def setup_subscribers(self):
        """
        Create all necessary ROS subscribers for Tello drone commands.
        """
        self.sub_emergency = self.create_subscription(Empty, 'emergency', self.cb_emergency, 1)
        self.sub_takeoff = self.create_subscription(Empty, 'takeoff', self.cb_takeoff, 1)
        self.sub_land = self.create_subscription(Empty, 'land', self.cb_land, 1)
        self.sub_control = self.create_subscription(Twist, 'cmd_vel', self.cb_control, 1)
        self.sub_flip = self.create_subscription(String, 'flip', self.cb_flip, 1)
        self.sub_wifi_config = self.create_subscription(TelloWifiConfig, 'wifi_config', self.cb_wifi_config, 1)

    def get_orientation_quaternion(self):
        """
        Get the drone's orientation as a quaternion.

        Retrieves yaw, pitch, and roll from the drone, converts them to radians,
        and then computes the corresponding quaternion.

        Returns:
            list: A list [qx, qy, qz, qw] representing the orientation quaternion.
        """
        deg_to_rad = math.pi / 180.0
        return euler_to_quaternion([
            self.tello.get_yaw() * deg_to_rad,
            self.tello.get_pitch() * deg_to_rad,
            self.tello.get_roll() * deg_to_rad
        ])

    def cb_status_timer(self):
        """
        Timer callback to periodically publish the drone's status.

        This method is called by a timer and publishes various status messages
        including battery state, temperature, general Tello status, ID info,
        and camera information.
        """
        # Battery
        if self.pub_battery.get_subscription_count() > 0:
            msg = BatteryState()
            msg.header.frame_id = self.tf_drone
            msg.percentage = float(self.tello.get_battery()) / 100.0
            msg.voltage = 3.8
            msg.design_capacity = 1.1
            msg.present = True
            msg.power_supply_technology = 2 # POWER_SUPPLY_TECHNOLOGY_LION
            msg.power_supply_status = 2 # POWER_SUPPLY_STATUS_DISCHARGING
            self.pub_battery.publish(msg)

        # Temperature
        if self.pub_temperature.get_subscription_count() > 0:
            msg = Temperature()
            msg.header.frame_id = self.tf_drone
            msg.temperature = self.tello.get_temperature()
            msg.variance = 0.0
            self.pub_temperature.publish(msg)

        # Tello Status
        if self.pub_status.get_subscription_count() > 0:
            msg = TelloStatus()
            msg.acceleration.x = self.tello.get_acceleration_x()
            msg.acceleration.y = self.tello.get_acceleration_y()
            msg.acceleration.z = self.tello.get_acceleration_z()

            msg.speed.x = float(self.tello.get_speed_x())
            msg.speed.y = float(self.tello.get_speed_y())
            msg.speed.z = float(self.tello.get_speed_z())

            msg.pitch = self.tello.get_pitch()
            msg.roll = self.tello.get_roll()
            msg.yaw = self.tello.get_yaw()

            msg.barometer = int(self.tello.get_barometer())
            msg.distance_tof = self.tello.get_distance_tof()

            msg.flight_time = self.tello.get_flight_time()

            msg.battery = self.tello.get_battery()

            msg.highest_temperature = self.tello.get_highest_temperature()
            msg.lowest_temperature = self.tello.get_lowest_temperature()
            msg.temperature = self.tello.get_temperature()

            msg.wifi_snr = self.tello.query_wifi_signal_noise_ratio()

            self.pub_status.publish(msg)

        # Tello ID
        if self.pub_id.get_subscription_count() > 0:
            msg = TelloID()
            msg.sdk_version = self.tello.query_sdk_version()
            msg.serial_number = self.tello.query_serial_number()
            self.pub_id.publish(msg)

        # Camera info
        if self.pub_camera_info.get_subscription_count() > 0:
            msg = CameraInfo()
            msg.height = self.camera_info['image_height']
            msg.width = self.camera_info['image_width']
            msg.distortion_model = self.camera_info['distortion_model']
            msg.D = self.camera_info['distortion_coefficients']['data']
            msg.K = self.camera_info['camera_matrix']['data']
            msg.R = self.camera_info['rectification_matrix']['data']
            msg.P = self.camera_info['projection_matrix']['data']
            self.pub_camera_info.publish(msg)

    def cb_odom_timer(self):
        """
        Timer callback to periodically publish odometry and IMU data.

        This function also broadcasts the drone's transform if enabled.
        """
        # TF
        if self.tf_pub:
            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = self.tf_base
            t.child_frame_id = self.tf_drone
            t.transform.translation.x = 0.0
            t.transform.translation.y = 0.0
            t.transform.translation.z = (self.tello.get_barometer()) / 100.0
            self.tf_broadcaster.sendTransform(t)
        
        # IMU
        if self.pub_imu.get_subscription_count() > 0:
            q = self.get_orientation_quaternion()

            msg = Imu()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = self.tf_drone
            msg.linear_acceleration.x = self.tello.get_acceleration_x() / 100.0
            msg.linear_acceleration.y = self.tello.get_acceleration_y() / 100.0
            msg.linear_acceleration.z = self.tello.get_acceleration_z() / 100.0
            msg.orientation.x = q[0]
            msg.orientation.y = q[1]
            msg.orientation.z = q[2]
            msg.orientation.w = q[3]
            self.pub_imu.publish(msg)

        # Odometry
        if self.pub_odom.get_subscription_count() > 0:
            q = self.get_orientation_quaternion()

            odom_msg = Odometry()
            odom_msg.header.stamp = self.get_clock().now().to_msg()
            odom_msg.header.frame_id = self.tf_base
            odom_msg.pose.pose.orientation.x = q[0]
            odom_msg.pose.pose.orientation.y = q[1]
            odom_msg.pose.pose.orientation.z = q[2]
            odom_msg.pose.pose.orientation.w = q[3]
            odom_msg.twist.twist.linear.x = float(self.tello.get_speed_x()) / 100.0
            odom_msg.twist.twist.linear.y = float(self.tello.get_speed_y()) / 100.0
            odom_msg.twist.twist.linear.z = float(self.tello.get_speed_z()) / 100.0
            self.pub_odom.publish(odom_msg)

    def cb_video_timer(self):
        """
        Timer callback to capture and publish video frames.

        This function ensures the video stream is active, reads the latest
        frame, converts it to a ROS Image message, and publishes it.
        """
        try:
            # Attempt to start stream if not already started
            if self.frame_reader is None:
                try:
                    self.tello.streamon()
                except Exception as e:
                    self.get_logger().debug(f"streamon() raised: {e}")
                self.frame_reader = self.tello.get_frame_read()

            frame = None
            if self.frame_reader is not None and hasattr(self.frame_reader, 'frame'):
                frame = self.frame_reader.frame

            if frame is None:
                return

            # Publish opencv frame using CV bridge
            img_msg = self.bridge.cv2_to_imgmsg(numpy.array(frame), 'bgr8') # TO TEST: might be "rgb8"
            img_msg.header.stamp = self.get_clock().now().to_msg()
            img_msg.header.frame_id = self.tf_drone
            self.pub_image_raw.publish(img_msg)

        except Exception as e:
            self.get_logger().warning(f"Video callback error: {str(e)}")
                
    def shutdown(self):
        """
        Cleanly shut down the Tello connection.
        """
        self.get_logger().info('Tello: shutting down')
        try:
            self.tello.end()
        except Exception as e:
            self.get_logger().warning(f"Tello shutdown error: {e}")

    def cb_emergency(self, msg):
        """
        Callback for the emergency stop command. Halts all motors.

        Args:
            msg (std_msgs.msg.Empty): The received message (content is ignored).
        """
        try:
            self.tello.emergency()
            self.get_logger().warning('Tello: emergency stop executed!')
        except Exception as e:
            self.get_logger().error(f"Emergency stop failed: {str(e)}")

    def cb_takeoff(self, msg):
        """
        Callback to command the drone to take off.

        Args:
            msg (std_msgs.msg.Empty): The received message (content is ignored).
        """
        if getattr(self.tello, 'is_flying', False):
            self.get_logger().warning('Tello is already flying!')
            return
        
        try:
            self.tello.takeoff()
            self.get_logger().info('Tello: takeoff successful!')
        except Exception as e:
            self.get_logger().error(f"Takeoff failed: {str(e)}")

    def cb_land(self, msg):
        """
        Callback to command the drone to land.

        Args:
            msg (std_msgs.msg.Empty): The received message (content is ignored).
        """
        if not getattr(self.tello, 'is_flying', False):
            self.get_logger().warning('Tello has already landed!')
            return
        
        try:
            self.tello.land()
            self.get_logger().info('Tello: landing successful!')
        except Exception as e:
            self.get_logger().error(f"Landing failed: {str(e)}")
        
    def cb_control(self, msg):
        """
        Callback for remote control commands.

        Receives a Twist message and sends the corresponding RC control values
        to the drone for manual flight control.

        Args:
            msg (geometry_msgs.msg.Twist): Velocity commands for the drone.
                Linear x/y/z and angular z are used, expected from -100 to 100.
        """
        if not getattr(self.tello, 'is_flying', False):
            self.get_logger().warning('Tello is not flying!')
            return
        
        try:
            self.tello.send_rc_control(
                int(msg.linear.x),   # TO TEST: might need to swap linear.x and linear.y
                int(msg.linear.y), 
                int(msg.linear.z), 
                int(msg.angular.z)
            )
        except Exception as e:
            self.get_logger().warning(f"send_rc_control failed: {e}")        

    def cb_wifi_config(self, msg):
        """
        Callback to configure the drone's Wi-Fi credentials.

        The drone will restart after the credentials are changed.

        Args:
            msg (tello_msg.msg.TelloWifiConfig): Contains the new SSID and password.
        """
        try:
            self.tello.set_wifi_credentials(msg.ssid, msg.password)
            self.get_logger().info("Tello: wifi credentials set (drone may restart).")
        except Exception as e:
            self.get_logger().error(f"Failed to set wifi credentials: {e}")
    
    def cb_flip(self, msg):
        """
        Callback to perform a flip in a specified direction.

        Args:
            msg (std_msgs.msg.String): The direction to flip.
                Can be "r" (right), "l" (left), "f" (forward), or "b" (backward).
        """
        self.tello.flip(msg.data)


def euler_to_quaternion(r):
    """
    Convert a rotation from Euler angles to a quaternion.

    Args:
        r (list): A list [yaw, pitch, roll] in radians.

    Returns:
        list: The corresponding [qx, qy, qz, qw] quaternion.
    """
    (yaw, pitch, roll) = (r[0], r[1], r[2])
    qx = math.sin(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) - math.cos(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
    qy = math.cos(roll/2) * math.sin(pitch/2) * math.cos(yaw/2) + math.sin(roll/2) * math.cos(pitch/2) * math.sin(yaw/2)
    qz = math.cos(roll/2) * math.cos(pitch/2) * math.sin(yaw/2) - math.sin(roll/2) * math.sin(pitch/2) * math.cos(yaw/2)
    qw = math.cos(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) + math.sin(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
    return [qx, qy, qz, qw]


def quaternion_to_euler(q):
    """
    Convert a rotation from a quaternion to Euler angles.

    Args:
        q (list): A list [qx, qy, qz, qw] representing the quaternion.

    Returns:
        list: The corresponding [yaw, pitch, roll] in radians.
    """
    (x, y, z, w) = (q[0], q[1], q[2], q[3])
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = math.asin(t2)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)
    return [yaw, pitch, roll]


def main(args=None):
    """
    Main entry point for the Tello ROS2 driver node.
    Ensures safe shutdown on crash or Ctrl+C.
    """
    rclpy.init(args=args)

    node = TelloNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("KeyboardInterrupt received. Shutting down node.")
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
