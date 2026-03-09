#!/usr/bin/env python3
"""
controller_node.py
20Hz P-controller with 6 ToF obstacle avoidance.
Reads center pixel depth from each ToF depth image.

ToF layout:
  tof_0:   0°   front
  tof_1:  60°   front-left
  tof_2: 120°   rear-left
  tof_3: 180°   rear
  tof_4: -120°  rear-right
  tof_5:  -60°  front-right
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from std_msgs.msg import Empty, String
from cv_bridge import CvBridge
import math
import threading
import json
import time
import numpy as np

# ── Config ──────────────────────────────────────────────────────
KP_XY         = 0.4
KP_Z          = 0.3
KP_YAW        = 0.5
MAX_VEL_XY    = 0.5
MAX_VEL_Z     = 0.3
GOAL_THRESH   = 0.4    # meters
OBSTACLE_DIST = 1.0    # meters — avoidance trigger
MIN_HEIGHT    = 0.8    # meters
TOF_MAX       = 2.5    # meters — max ToF range
# ────────────────────────────────────────────────────────────────


class ControllerNode(Node):
    def __init__(self):
        super().__init__("controller_node")

        self.bridge    = CvBridge()
        self.goal      = None
        self.curr_pose = None
        self.flying    = False
        self.state     = "idle"

        # 6 ToF readings
        self.tof = {
            "tof_0": TOF_MAX,  # front
            "tof_1": TOF_MAX,  # front-left
            "tof_2": TOF_MAX,  # rear-left
            "tof_3": TOF_MAX,  # rear
            "tof_4": TOF_MAX,  # rear-right
            "tof_5": TOF_MAX,  # front-right
        }

        ns = "/simple_drone"

        # ── ToF depth image subscribers ───────────────────────
        for i in range(6):
            name  = f"tof_{i}"
            topic = f"{ns}/{name}/depth/image_raw"
            self.create_subscription(
                Image, topic,
                lambda msg, n=name: self.tof_cb(msg, n),
                10)

        # ── Other subscribers ─────────────────────────────────
        self.create_subscription(
            PoseStamped, "/drone/goal_pose", self.goal_cb, 10)
        self.create_subscription(
            Odometry, "/simple_drone/odom", self.odom_cb, 10)

        # ── Publishers ─────────────────────────────────────────
        self.cmd_pub     = self.create_publisher(
            Twist,  "/cmd_vel/input", 10)
        self.takeoff_pub = self.create_publisher(
            Empty,  "/simple_drone/takeoff", 10)
        self.land_pub    = self.create_publisher(
            Empty,  "/simple_drone/land",    10)
        self.status_pub  = self.create_publisher(
            String, "/drone/status",         10)

        # ── Timers ─────────────────────────────────────────────
        self.create_timer(0.05, self.control_loop)   # 20Hz
        self.create_timer(1.0,  self.publish_status) # 1Hz

        self.get_logger().info("✅ Controller ready — 6 ToF sensors")
        threading.Thread(target=self.cli, daemon=True).start()

    # ── ToF depth callback ────────────────────────────────────

    def tof_cb(self, msg, name):
        try:
            depth = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding="passthrough")
            h, w = depth.shape[:2]
            cx, cy = w // 2, h // 2
            r = 2
            region = depth[
                max(0, cy-r):min(h, cy+r),
                max(0, cx-r):min(w, cx+r)
            ]
            valid = region[
                (region > 0.2) &
                (region < TOF_MAX) &
                np.isfinite(region)
            ]
            self.tof[name] = float(np.median(valid)) if len(valid) > 0 else TOF_MAX
        except:
            pass

    # ── Other callbacks ───────────────────────────────────────

    def goal_cb(self, msg):
        self.goal  = msg
        self.state = "navigating"
        p = msg.pose.position
        self.get_logger().info(
            f"🎯 Goal: ({p.x:.1f}, {p.y:.1f}, {p.z:.1f})")

    def odom_cb(self, msg):
        self.curr_pose = msg.pose.pose

    # ── Control loop (20Hz) ───────────────────────────────────

    def control_loop(self):
        if not self.flying:
            return

        cmd = Twist()

        # ── Obstacle avoidance (priority) ─────────────────────
        # Group sensors by direction
        front      = min(self.tof["tof_0"], self.tof["tof_5"], self.tof["tof_1"])
        rear       = min(self.tof["tof_3"], self.tof["tof_4"], self.tof["tof_2"])
        front_left = self.tof["tof_1"]
        front_right= self.tof["tof_5"]

        avoided = False

        if front < OBSTACLE_DIST:
            # Back up and rise
            cmd.linear.x = -0.3
            cmd.linear.z =  0.2
            self.state   = "avoiding"
            avoided      = True
        elif front_left < OBSTACLE_DIST and front_right >= OBSTACLE_DIST:
            # Dodge right
            cmd.linear.y = -0.3
            self.state   = "avoiding"
            avoided      = True
        elif front_right < OBSTACLE_DIST and front_left >= OBSTACLE_DIST:
            # Dodge left
            cmd.linear.y =  0.3
            self.state   = "avoiding"
            avoided      = True
        elif rear < OBSTACLE_DIST:
            # Move forward
            cmd.linear.x =  0.3
            self.state   = "avoiding"
            avoided      = True

        if avoided:
            self.cmd_pub.publish(cmd)
            return

        # ── Goal tracking ──────────────────────────────────────
        if self.goal is None or self.curr_pose is None:
            return
        if self.state != "navigating":
            return

        gx = self.goal.pose.position.x
        gy = self.goal.pose.position.y
        gz = max(self.goal.pose.position.z, MIN_HEIGHT)

        cx = self.curr_pose.position.x
        cy = self.curr_pose.position.y
        cz = self.curr_pose.position.z

        ex = gx - cx
        ey = gy - cy
        ez = gz - cz

        dist = math.sqrt(ex**2 + ey**2 + ez**2)

        if dist < GOAL_THRESH:
            self.get_logger().info("✅ Goal reached!")
            self.goal  = None
            self.state = "hovering"
            self.cmd_pub.publish(Twist())
            return

        # P-controller
        vx = max(-MAX_VEL_XY, min(MAX_VEL_XY, KP_XY * ex))
        vy = max(-MAX_VEL_XY, min(MAX_VEL_XY, KP_XY * ey))
        vz = max(-MAX_VEL_Z,  min(MAX_VEL_Z,  KP_Z  * ez))

        yaw_err = math.atan2(ey, ex)
        cmd.linear.x  = vx
        cmd.linear.y  = vy
        cmd.linear.z  = vz
        cmd.angular.z = max(-0.5, min(0.5, KP_YAW * yaw_err))

        self.cmd_pub.publish(cmd)

    def publish_status(self):
        tof_str = " | ".join(
            f"{k[-1]}:{v:.1f}" for k, v in self.tof.items())
        status = {
            "state":  self.state,
            "flying": self.flying,
            "tof":    tof_str,
        }
        self.status_pub.publish(String(data=json.dumps(status)))

    # ── CLI ───────────────────────────────────────────────────

    def cli(self):
        print("\n" + "="*45)
        print("🚁 DRONE CONTROLLER")
        print("="*45)
        print("  takeoff  — take off")
        print("  land     — land")
        print("  stop     — hover")
        print("  status   — show state + ToF readings")
        print("  quit     — exit")
        print("="*45 + "\n")

        while rclpy.ok():
            try:
                raw = input("ctrl> ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                break

            if raw == "takeoff":
                self.takeoff_pub.publish(Empty())
                self.flying = True
                self.state  = "hovering"
                print("✅ Takeoff!")

            elif raw == "land":
                self.goal   = None
                self.state  = "landing"
                self.cmd_pub.publish(Twist())
                time.sleep(0.3)
                self.land_pub.publish(Empty())
                self.flying = False
                self.state  = "idle"
                print("✅ Landed!")

            elif raw == "stop":
                self.goal  = None
                self.state = "hovering"
                self.cmd_pub.publish(Twist())
                print("⏹ Hovering")

            elif raw == "status":
                print(f"\n  state:  {self.state}")
                print(f"  flying: {self.flying}")
                print(f"  ToF readings (meters):")
                dirs = ["front", "front-left", "rear-left",
                        "rear", "rear-right", "front-right"]
                for i, (k, v) in enumerate(self.tof.items()):
                    bar = "🟢" if v > OBSTACLE_DIST else "🔴"
                    print(f"    {bar} tof_{i} ({dirs[i]:12s}): {v:.2f}m")
                print()

            elif raw == "quit":
                break
            else:
                print("Unknown command. Try: takeoff, land, stop, status")


def main(args=None):
    rclpy.init(args=args)
    node = ControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()