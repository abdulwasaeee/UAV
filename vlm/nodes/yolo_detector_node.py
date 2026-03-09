#!/usr/bin/env python3
"""
yolo_detector_node.py
Detects objects in RGB-D camera feed using YOLOv8.
Uses odom for world position instead of TF lookup.

Publishes:
  /detections          (JSON string)
  /detections/image    (annotated image)
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import json
import math
import numpy as np
import threading

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# ── Config ──────────────────────────────────────────────────────
YOLO_MODEL   = "yolov8n.pt"
DETECT_RATE  = 2.0       # Hz
CONFIDENCE   = 0.5
MAX_DEPTH    = 8.0       # meters
# ────────────────────────────────────────────────────────────────


class YoloDetectorNode(Node):
    def __init__(self):
        super().__init__("yolo_detector_node")

        if not YOLO_AVAILABLE:
            self.get_logger().error("ultralytics not installed!")
            return

        self.get_logger().info("Loading YOLOv8 model...")
        self.model = YOLO(YOLO_MODEL)
        self.get_logger().info("✅ YOLOv8 loaded!")

        self.bridge      = CvBridge()
        self.fx = self.fy = self.cx = self.cy = None
        self.rgb_frame   = None
        self.depth_frame = None
        self.drone_pose  = None
        self._running    = False

        # ── Subscribers ────────────────────────────────────────
        self.create_subscription(
            Image, "/simple_drone/rgbd/image_raw_fixed",
            self.rgb_cb, 10)
        self.create_subscription(
            Image, "/simple_drone/rgbd/depth/image_raw_fixed",
            self.depth_cb, 10)
        self.create_subscription(
            CameraInfo, "/simple_drone/rgbd/camera_info_fixed",
            self.caminfo_cb, 10)
        self.create_subscription(
            Odometry, "/simple_drone/odom",
            self.odom_cb, 10)

        # ── Publishers ─────────────────────────────────────────
        self.det_pub = self.create_publisher(String, "/detections", 10)
        self.img_pub = self.create_publisher(Image, "/detections/image", 10)

        # ── Timer ──────────────────────────────────────────────
        self.create_timer(1.0 / DETECT_RATE, self.detect)

        self.get_logger().info("✅ YOLO Detector ready!")

    # ── Callbacks ──────────────────────────────────────────────

    def caminfo_cb(self, msg):
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]

    def rgb_cb(self, msg):
        try:
            self.rgb_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().warn(f"RGB error: {e}")

    def depth_cb(self, msg):
        try:
            self.depth_frame = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding="passthrough")
        except Exception as e:
            self.get_logger().warn(f"Depth error: {e}")

    def odom_cb(self, msg):
        self.drone_pose = msg.pose.pose

    # ── World position from odom ───────────────────────────────

    def _to_world(self, X_cam, Y_cam, Z):
        """
        Convert camera frame point to world frame using drone odom.
        No TF needed — uses odometry directly.
        """
        if self.drone_pose is None:
            # Return camera frame coords as fallback
            return X_cam, Y_cam, Z

        # Drone world position
        dx = self.drone_pose.position.x
        dy = self.drone_pose.position.y
        dz = self.drone_pose.position.z

        # Drone yaw from quaternion
        q = self.drone_pose.orientation
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )

        # Rotate camera frame point to world frame
        # Camera: Z=forward, X=right, Y=down
        world_x = dx + Z * math.cos(yaw) - X_cam * math.sin(yaw)
        world_y = dy + Z * math.sin(yaw) + X_cam * math.cos(yaw)
        world_z = dz - Y_cam

        return round(world_x, 2), round(world_y, 2), round(world_z, 2)

    # ── Detection ──────────────────────────────────────────────

    def detect(self):
        if self._running:
            return
        if self.rgb_frame is None or self.depth_frame is None:
            return
        if self.fx is None:
            return
        threading.Thread(target=self._run_detection, daemon=True).start()

    def _run_detection(self):
        self._running = True
        try:
            frame = self.rgb_frame.copy()
            depth = self.depth_frame.copy()

            results   = self.model(frame, conf=CONFIDENCE, verbose=False)
            detections = []
            annotated  = frame.copy()

            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    u     = (x1 + x2) // 2
                    v     = (y1 + y2) // 2
                    conf  = float(box.conf[0])
                    cls   = int(box.cls[0])
                    label = self.model.names[cls]

                    # Metric depth at center pixel
                    h, w = depth.shape[:2]
                    u_c  = max(0, min(u, w - 1))
                    v_c  = max(0, min(v, h - 1))
                    r    = 5
                    region = depth[
                        max(0, v_c-r):min(h, v_c+r),
                        max(0, u_c-r):min(w, u_c+r)
                    ]
                    valid = region[
                        (region > 0.1) & (region < MAX_DEPTH) & np.isfinite(region)]

                    if len(valid) == 0:
                        continue

                    Z     = float(np.median(valid))
                    X_cam = (u - self.cx) * Z / self.fx
                    Y_cam = (v - self.cy) * Z / self.fy

                    # Convert to world frame using odom
                    wx, wy, wz = self._to_world(X_cam, Y_cam, Z)

                    det = {
                        "label":      label,
                        "confidence": round(conf, 2),
                        "pixel":      [u, v],
                        "depth_m":    round(Z, 2),
                        "world":      {"x": wx, "y": wy, "z": wz}
                    }
                    detections.append(det)

                    # Annotate image
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(annotated, (u, v), 4, (0, 0, 255), -1)
                    cv2.putText(
                        annotated,
                        f"{label} {conf:.0%} {Z:.1f}m",
                        (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    self.get_logger().info(
                        f"🎯 {label} ({conf:.0%}) "
                        f"depth={Z:.1f}m "
                        f"world=({wx}, {wy}, {wz})")

            if detections:
                self.det_pub.publish(
                    String(data=json.dumps(detections)))

            self.img_pub.publish(
                self.bridge.cv2_to_imgmsg(annotated, "bgr8"))

        except Exception as e:
            self.get_logger().warn(f"Detection error: {e}")
        finally:
            self._running = False


def main(args=None):
    rclpy.init(args=args)
    node = YoloDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
