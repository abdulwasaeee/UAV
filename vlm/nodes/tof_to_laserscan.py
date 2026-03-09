#!/usr/bin/env python3
"""
tof_to_laserscan.py
Converts 6x ToF depth images into a 360° LaserScan.
Each ToF reads the center pixel depth from its depth image.

ToF layout (60° spacing):
  tof_0:   0°   → front
  tof_1:  60°   → front-left
  tof_2: 120°   → rear-left
  tof_3: 180°   → rear
  tof_4: -120°  → rear-right
  tof_5:  -60°  → front-right
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Image
from cv_bridge import CvBridge
import numpy as np
import math


class TofToLaserScan(Node):
    def __init__(self):
        super().__init__("tof_to_laserscan")

        self.bridge = CvBridge()

        # ToF readings — default to max range
        self.ranges = {
            "tof_0": 2.5,  # front        0°
            "tof_1": 2.5,  # front-left  60°
            "tof_2": 2.5,  # rear-left  120°
            "tof_3": 2.5,  # rear       180°
            "tof_4": 2.5,  # rear-right -120°
            "tof_5": 2.5,  # front-right -60°
        }

        ns = "/simple_drone"

        # Subscribe to depth image for each ToF
        for i in range(6):
            name = f"tof_{i}"
            topic = f"{ns}/{name}/depth/image_raw"
            # Use lambda with default arg to capture i correctly
            self.create_subscription(
                Image, topic,
                lambda msg, n=name: self.depth_cb(msg, n),
                10)

        # Publish LaserScan at 20Hz
        self.pub = self.create_publisher(
            LaserScan, "/simple_drone/scan", 10)
        self.create_timer(0.05, self.publish_scan)

        self.get_logger().info("✅ ToF → LaserScan (6 sensors) ready")

    def depth_cb(self, msg, name):
        try:
            depth = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding="passthrough")

            h, w = depth.shape[:2]
            cx, cy = w // 2, h // 2

            # Sample 5x5 region at center
            r = 2
            region = depth[
                max(0, cy-r):min(h, cy+r),
                max(0, cx-r):min(w, cx+r)
            ]
            valid = region[
                (region > 0.2) &
                (region < 2.5) &
                np.isfinite(region)
            ]

            if len(valid) > 0:
                self.ranges[name] = float(np.median(valid))
            else:
                self.ranges[name] = 2.5  # max range if no valid reading

        except Exception as e:
            self.get_logger().warn(f"{name} depth error: {e}")

    def publish_scan(self):
        scan = LaserScan()
        scan.header.stamp    = self.get_clock().now().to_msg()
        scan.header.frame_id = "simple_drone/base_footprint"

        # 6 rays at 60° intervals starting at 0°
        scan.angle_min       = 0.0
        scan.angle_max       = 2.0 * math.pi
        scan.angle_increment = math.pi / 3.0  # 60°
        scan.time_increment  = 0.0
        scan.scan_time       = 0.05
        scan.range_min       = 0.2
        scan.range_max       = 2.5

        # Order matches angle: 0°, 60°, 120°, 180°, -120°, -60°
        scan.ranges = [
            self.ranges["tof_0"],  # 0°   front
            self.ranges["tof_1"],  # 60°  front-left
            self.ranges["tof_2"],  # 120° rear-left
            self.ranges["tof_3"],  # 180° rear
            self.ranges["tof_4"],  # 240° rear-right
            self.ranges["tof_5"],  # 300° front-right
        ]
        scan.intensities = [1.0] * 6

        self.pub.publish(scan)


def main(args=None):
    rclpy.init(args=args)
    node = TofToLaserScan()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
