#!/usr/bin/env python3
"""
obstacle_mux_node.py - Fast, simple obstacle avoidance mux
Reads center pixel from each ToF depth image.
Simple threshold check — no complex grouping logic.
Runs at 50Hz.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import numpy as np
import time

# ── Config ──────────────────────────────────────────────────────
WARN_DIST     = 1.2   # meters — start slowing down
AVOID_DIST    = 0.8   # meters — active avoidance
CRITICAL_DIST = 0.35  # meters — emergency stop
AVOID_VEL     = 0.25  # m/s
RISE_VEL      = 0.15  # m/s
TOF_MAX       = 2.5   # meters
INPUT_TIMEOUT = 0.5   # seconds — stale input threshold
MUX_HZ        = 50.0
# ────────────────────────────────────────────────────────────────


class ObstacleMuxNode(Node):
    def __init__(self):
        super().__init__("obstacle_mux_node")

        self.bridge    = CvBridge()
        self.input_cmd = Twist()
        self.input_time = 0.0
        self.mode      = "INIT"

        # 6 ToF readings indexed 0-5
        self.tof = [TOF_MAX] * 6

        # tof_0=front, tof_1=front-left, tof_2=rear-left
        # tof_3=rear,  tof_4=rear-right, tof_5=front-right
        self.dirs = ["front", "front-left", "rear-left",
                     "rear",  "rear-right", "front-right"]

        ns = "/simple_drone"

        # ── ToF subscribers ───────────────────────────────────
        for i in range(6):
            topic = f"{ns}/tof_{i}/depth/image_raw"
            self.create_subscription(
                Image, topic,
                lambda msg, idx=i: self.tof_cb(msg, idx),
                10)

        # ── Input from VLM/controller ─────────────────────────
        self.create_subscription(
            Twist, "/cmd_vel/input", self.input_cb, 10)

        # ── Output to drone ───────────────────────────────────
        self.cmd_pub    = self.create_publisher(
            Twist,  "/simple_drone/cmd_vel", 10)
        self.status_pub = self.create_publisher(
            String, "/obstacle_mux/status",  10)

        # ── Loops ─────────────────────────────────────────────
        self.create_timer(1.0 / MUX_HZ, self.mux_loop)
        self.create_timer(1.0,           self.print_status)

        self.get_logger().info(
            f"✅ Obstacle Mux ready — "
            f"warn@{WARN_DIST}m avoid@{AVOID_DIST}m "
            f"emergency@{CRITICAL_DIST}m")

    # ── ToF callback ──────────────────────────────────────────

    def tof_cb(self, msg, idx):
        try:
            depth = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding="passthrough")
            h, w  = depth.shape[:2]
            cx, cy = w // 2, h // 2
            r = 3
            patch = depth[
                max(0, cy-r):min(h, cy+r),
                max(0, cx-r):min(w, cx+r)]
            valid = patch[
                (patch > 0.15) &
                (patch < TOF_MAX) &
                np.isfinite(patch)]
            self.tof[idx] = float(np.median(valid)) \
                if len(valid) > 0 else TOF_MAX
        except:
            pass

    def input_cb(self, msg):
        self.input_cmd  = msg
        self.input_time = time.time()

    # ── Main mux loop 50Hz ────────────────────────────────────

    def mux_loop(self):
        cmd, mode = self._compute()
        self.mode = mode
        self.cmd_pub.publish(cmd)

    def _compute(self):
        t = self.tof

        # Sensor groups
        front  = min(t[0], t[1], t[5])   # front cone
        rear   = min(t[3], t[2], t[4])   # rear cone
        left   = min(t[1], t[2])          # left side
        right  = min(t[4], t[5])          # right side
        all_min = min(t)

        cmd = Twist()

        # ── Emergency — any sensor critical ───────────────────
        if all_min < CRITICAL_DIST:
            cmd.linear.z = RISE_VEL
            return cmd, "EMERGENCY"

        # ── Active avoidance ──────────────────────────────────
        if front < AVOID_DIST:
            cmd.linear.x = -AVOID_VEL
            cmd.linear.z =  RISE_VEL * 0.5
            # pick clearer side
            cmd.linear.y = AVOID_VEL * 0.5 if right > left else -AVOID_VEL * 0.5
            return cmd, "AVOID_FRONT"

        if rear < AVOID_DIST:
            cmd.linear.x = AVOID_VEL
            return cmd, "AVOID_REAR"

        if left < AVOID_DIST and right >= AVOID_DIST:
            cmd.linear.y = -AVOID_VEL
            return cmd, "AVOID_LEFT"

        if right < AVOID_DIST and left >= AVOID_DIST:
            cmd.linear.y = AVOID_VEL
            return cmd, "AVOID_RIGHT"

        if left < AVOID_DIST and right < AVOID_DIST:
            cmd.linear.z = RISE_VEL
            return cmd, "AVOID_RISE"

        # ── Warning zone — scale down input velocity ───────────
        if front < WARN_DIST:
            inp = self.input_cmd
            scale = max(0.1, (front - AVOID_DIST) / (WARN_DIST - AVOID_DIST))
            cmd.linear.x  = inp.linear.x  * scale
            cmd.linear.y  = inp.linear.y
            cmd.linear.z  = inp.linear.z
            cmd.angular.z = inp.angular.z
            return cmd, "WARNING"

        # ── Passthrough ───────────────────────────────────────
        if time.time() - self.input_time < INPUT_TIMEOUT:
            return self.input_cmd, "PASSTHROUGH"

        return Twist(), "HOVER"

    # ── Status print 1Hz ──────────────────────────────────────

    def print_status(self):
        icons = {
            "PASSTHROUGH": "✅",
            "HOVER":       "⏸ ",
            "WARNING":     "⚠️ ",
            "AVOID_FRONT": "⬅️ ",
            "AVOID_REAR":  "➡️ ",
            "AVOID_LEFT":  "➡️ ",
            "AVOID_RIGHT": "⬅️ ",
            "AVOID_RISE":  "⬆️ ",
            "EMERGENCY":   "🚨",
            "INIT":        "⏳",
        }
        icon = icons.get(self.mode, "❓")

        readings = "  ".join(
            f"{'🔴' if self.tof[i] < AVOID_DIST else '🟡' if self.tof[i] < WARN_DIST else '🟢'}"
            f"{self.dirs[i][:2]}:{self.tof[i]:.1f}"
            for i in range(6)
        )
        self.get_logger().info(f"{icon}[{self.mode:12s}] {readings}")
        self.status_pub.publish(String(data=self.mode))


def main(args=None):
    rclpy.init(args=args)
    node = ObstacleMuxNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()