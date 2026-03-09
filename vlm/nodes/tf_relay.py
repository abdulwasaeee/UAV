#!/usr/bin/env python3
"""
tf_relay.py - Complete TF fix for sjtu_drone
1. Strips leading slashes from /tf and /tf_static
2. Publishes odom->base_footprint from odometry as TFMessage
3. Fixes camera/depth topic frame_ids
"""

import rclpy
from rclpy.node import Node
from tf2_msgs.msg import TFMessage
from sensor_msgs.msg import CameraInfo, Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy


class TFRelay(Node):
    def __init__(self):
        super().__init__("tf_relay")

        static_qos = QoSProfile(
            depth=100,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE
        )

        # ── Publishers ────────────────────────────────────────
        self.pub_tf        = self.create_publisher(TFMessage, "/tf", 10)
        self.pub_tf_static = self.create_publisher(
            TFMessage, "/tf_static", static_qos)
        self.pub_rgb       = self.create_publisher(
            Image,      "/simple_drone/rgbd/image_raw_fixed",         10)
        self.pub_depth     = self.create_publisher(
            Image,      "/simple_drone/rgbd/depth/image_raw_fixed",   10)
        self.pub_caminfo   = self.create_publisher(
            CameraInfo, "/simple_drone/rgbd/camera_info_fixed",       10)
        self.pub_caminfo_d = self.create_publisher(
            CameraInfo, "/simple_drone/rgbd/depth/camera_info_fixed", 10)

        # ── Subscribers ───────────────────────────────────────
        self.create_subscription(TFMessage, "/tf",        self.tf_cb,        10)
        self.create_subscription(TFMessage, "/tf_static", self.tf_static_cb, static_qos)
        self.create_subscription(Odometry,  "/simple_drone/odom", self.odom_cb, 10)
        self.create_subscription(Image,      "/simple_drone/rgbd/image_raw",           self.rgb_cb,       10)
        self.create_subscription(Image,      "/simple_drone/rgbd/depth/image_raw",     self.depth_cb,     10)
        self.create_subscription(CameraInfo, "/simple_drone/rgbd/camera_info",         self.caminfo_cb,   10)
        self.create_subscription(CameraInfo, "/simple_drone/rgbd/depth/camera_info",   self.caminfo_d_cb, 10)

        self.published_static = set()
        self.get_logger().info("✅ TF Relay ready")

    def strip(self, s):
        return s.lstrip("/")

    # ── Odom → publishes as TFMessage directly ────────────────
    def odom_cb(self, msg):
        t = TransformStamped()
        t.header.stamp    = msg.header.stamp
        t.header.frame_id = "simple_drone/odom"
        t.child_frame_id  = "simple_drone/base_footprint"
        t.transform.translation.x = msg.pose.pose.position.x
        t.transform.translation.y = msg.pose.pose.position.y
        t.transform.translation.z = msg.pose.pose.position.z
        t.transform.rotation      = msg.pose.pose.orientation

        out = TFMessage()
        out.transforms.append(t)
        self.pub_tf.publish(out)

    # ── TF relay ──────────────────────────────────────────────
    def fix_tf(self, msg):
        for t in msg.transforms:
            t.header.frame_id = self.strip(t.header.frame_id)
            t.child_frame_id  = self.strip(t.child_frame_id)
        return msg

    def tf_cb(self, msg):
        if any(t.header.frame_id.startswith("/") or
               t.child_frame_id.startswith("/")
               for t in msg.transforms):
            self.pub_tf.publish(self.fix_tf(msg))

    def tf_static_cb(self, msg):
        if any(t.header.frame_id.startswith("/") or
               t.child_frame_id.startswith("/")
               for t in msg.transforms):
            fixed = self.fix_tf(msg)
            self.pub_tf_static.publish(fixed)
            for t in fixed.transforms:
                key = (t.header.frame_id, t.child_frame_id)
                if key not in self.published_static:
                    self.published_static.add(key)
                    self.get_logger().info(
                        f"  static TF: {t.header.frame_id} → {t.child_frame_id}")

    # ── Camera frame fix ──────────────────────────────────────
    def fix_header(self, msg):
        msg.header.frame_id = self.strip(msg.header.frame_id)
        return msg

    def rgb_cb(self, msg):      self.pub_rgb.publish(self.fix_header(msg))
    def depth_cb(self, msg):    self.pub_depth.publish(self.fix_header(msg))
    def caminfo_cb(self, msg):  self.pub_caminfo.publish(self.fix_header(msg))
    def caminfo_d_cb(self, msg): self.pub_caminfo_d.publish(self.fix_header(msg))


def main(args=None):
    rclpy.init(args=args)
    node = TFRelay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
