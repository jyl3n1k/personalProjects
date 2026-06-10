import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from rclpy.qos import ReliabilityPolicy, QoSProfile
import math

LINEAR_VEL = 0.22
STOP_DISTANCE = 0.2
LIDAR_ERROR = 0.05
LIDAR_AVOID_DISTANCE = 0.7
SAFE_STOP_DISTANCE = STOP_DISTANCE + LIDAR_ERROR
RIGHT_SIDE_INDEX = 270
RIGHT_FRONT_INDEX = 210
LEFT_FRONT_INDEX = 150
LEFT_SIDE_INDEX = 90
ZONE_THRESHOLD = 0.2  # Threshold to consider robot "inside" a zone

class RandomWalk(Node):

    def __init__(self):
        super().__init__('random_walk_node')
        self.scan_cleaned = []
        self.stall = False
        self.turtlebot_moving = False
        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10)
        self.subscriber1 = self.create_subscription(
            LaserScan, '/scan', self.listener_callback1,
            QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.subscriber2 = self.create_subscription(
            Odometry, '/odom', self.listener_callback2,
            QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.pose_saved = None
        self.cmd = Twist()
        timer_period = 0.5
        self.timer = self.create_timer(timer_period, self.timer_callback)

        # Define zones with position, rotation, name, and visited flag
        self.zones = [
            {'name': 'Zone 1', 'x': 7.6, 'y': 3.5, 'z': 0.05, 'visited': False},
            {'name': 'Zone 2', 'x': 5.2, 'y': 6.0, 'z': 0.05, 'visited': False},
            {'name': 'Zone 3', 'x': 1.8, 'y': 6.3, 'z': 0.05, 'visited': False},
            {'name': 'Zone 4', 'x': 0.96, 'y': 0.492, 'z': 0.05, 'visited': False}
        ]

    def listener_callback1(self, msg1):
        scan = msg1.ranges
        self.scan_cleaned = []
        for reading in scan:
            if reading == float('Inf'):
                self.scan_cleaned.append(3.5)
            elif math.isnan(reading):
                self.scan_cleaned.append(0.0)
            else:
                self.scan_cleaned.append(reading)

    def listener_callback2(self, msg2):
        position = msg2.pose.pose.position
        self.pose_saved = position

        # Check if starting position is inside any zone
        for zone in self.zones:
            if not zone['visited']:
                if (abs(position.x - zone['x']) < ZONE_THRESHOLD) and \
                   (abs(position.y - zone['y']) < ZONE_THRESHOLD):
                    zone['visited'] = True
                    self.get_logger().info(f"Starting in {zone['name']}, marking it visited.")

    def timer_callback(self):
        if len(self.scan_cleaned) == 0:
            self.turtlebot_moving = False
            return

        left_lidar_min = min(self.scan_cleaned[LEFT_SIDE_INDEX:LEFT_FRONT_INDEX])
        right_lidar_min = min(self.scan_cleaned[RIGHT_FRONT_INDEX:RIGHT_SIDE_INDEX])
        front_lidar_min = min(self.scan_cleaned[LEFT_FRONT_INDEX:RIGHT_FRONT_INDEX])

        # Obstacle handling
        if front_lidar_min < SAFE_STOP_DISTANCE:
            if self.turtlebot_moving:
                self.cmd.linear.x = 0.0
                self.cmd.angular.z = 0.0
                self.publisher_.publish(self.cmd)
                self.turtlebot_moving = False
                self.get_logger().info('Stopping due to obstacle')
                return
        elif front_lidar_min < LIDAR_AVOID_DISTANCE:
            self.cmd.linear.x = 0.07
            if right_lidar_min > left_lidar_min:
                self.cmd.angular.z = -0.3
            else:
                self.cmd.angular.z = 0.3
            self.publisher_.publish(self.cmd)
            self.turtlebot_moving = True
            self.get_logger().info('Avoiding obstacle')
        else:
            self.cmd.linear.x = 0.3
            self.cmd.angular.z = 0.0
            self.publisher_.publish(self.cmd)
            self.turtlebot_moving = True

        # Check if the robot reached any zone
        if self.pose_saved:
            for zone in self.zones:
                if not zone['visited']:
                    if (abs(self.pose_saved.x - zone['x']) < ZONE_THRESHOLD) and \
                       (abs(self.pose_saved.y - zone['y']) < ZONE_THRESHOLD):
                        zone['visited'] = True
                        self.get_logger().info(f"Reached {zone['name']}! Marking as visited.")

        # Check if all zones are visited
        if all(zone['visited'] for zone in self.zones):
            self.get_logger().info("All zones visited! Exploration complete.")

def main(args=None):
    rclpy.init(args=args)
    random_walk_node = RandomWalk()
    rclpy.spin(random_walk_node)
    random_walk_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
