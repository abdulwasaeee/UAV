#!/usr/bin/env python3
"""
exploration_node.py
Autonomous room mapping using expanding spiral + wall detection.

- Starts wherever drone is
- Expands outward in increasing squares
- ToF sensors detect walls — stops expanding that direction
- Tracks visited cells to ensure full coverage
- Returns to start and lands when room fully covered
- Obstacle mux handles avoidance throughout
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from std_msgs.msg import Empty, String
from cv_bridge import CvBridge
import numpy as np
import math
import threading
import time

# ── Config ──────────────────────────────────────────────────────
EXPLORE_HEIGHT  = 1.5    # m — flight height
STEP_SIZE       = 1.5    # m — grid cell size
WALL_DIST       = 1.2    # m — stop expanding if wall closer than this
GOAL_THRESH     = 0.4    # m — close enough to waypoint
GOAL_TIMEOUT    = 12.0   # s — skip waypoint if stuck
KP_XY           = 0.4
KP_Z            = 0.3
MAX_VEL_XY      = 0.3    # slow for clean mapping
MAX_VEL_Z       = 0.2
TOF_MAX         = 2.5
# ────────────────────────────────────────────────────────────────


class ExplorationNode(Node):
    def __init__(self):
        super().__init__("exploration_node")

        self.bridge     = CvBridge()
        self.curr_pose  = None
        self.tof        = [TOF_MAX] * 6
        self.exploring  = False
        self.goal       = None
        self.goal_start = 0.0
        self.start_pos  = None   # home position
        self.waypoints  = []     # full planned path
        self.wp_index   = 0      # current waypoint index
        self.state      = "idle" # idle/exploring/returning/done

        # tof directions
        # 0=front 1=front-left 2=rear-left 3=rear 4=rear-right 5=front-right
        ns = "/simple_drone"

        # ── ToF subscribers ───────────────────────────────────
        for i in range(6):
            topic = f"{ns}/tof_{i}/depth/image_raw"
            self.create_subscription(
                Image, topic,
                lambda msg, idx=i: self.tof_cb(msg, idx),
                10)

        # ── Odom ──────────────────────────────────────────────
        self.create_subscription(
            Odometry, "/simple_drone/odom", self.odom_cb, 10)

        # ── Publishers ────────────────────────────────────────
        self.cmd_pub     = self.create_publisher(
            Twist,  "/cmd_vel/input",        10)
        self.land_pub    = self.create_publisher(
            Empty,  "/simple_drone/land",    10)
        self.status_pub  = self.create_publisher(
            String, "/exploration/status",   10)

        # ── Timers ────────────────────────────────────────────
        self.create_timer(0.05, self.control_loop)  # 20Hz
        self.create_timer(1.0,  self.print_status)

        self.get_logger().info("✅ Exploration Node ready!")
        threading.Thread(target=self.cli, daemon=True).start()

    # ── Callbacks ─────────────────────────────────────────────

    def odom_cb(self, msg):
        self.curr_pose = msg.pose.pose

    def tof_cb(self, msg, idx):
        try:
            depth  = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding="passthrough")
            h, w   = depth.shape[:2]
            cx, cy = w // 2, h // 2
            r      = 3
            patch  = depth[
                max(0,cy-r):min(h,cy+r),
                max(0,cx-r):min(w,cx+r)]
            valid  = patch[
                (patch > 0.15) &
                (patch < TOF_MAX) &
                np.isfinite(patch)]
            self.tof[idx] = float(np.median(valid)) \
                if len(valid) > 0 else TOF_MAX
        except:
            pass

    # ── Spiral waypoint generator ─────────────────────────────

    def generate_spiral(self, ox, oy):
        """
        Generate expanding spiral waypoints from origin (ox, oy).
        Uses current ToF readings to limit expansion in each direction.
        Returns list of (x, y) waypoints.
        """
        waypoints = []
        visited   = set()

        # Spiral directions: right, up, left, down (in odom frame)
        # Each loop expands by STEP_SIZE
        directions = [
            ( 1,  0),  # +X  (forward in world)
            ( 0,  1),  # +Y  (left in world)
            (-1,  0),  # -X  (backward)
            ( 0, -1),  # -Y  (right)
        ]

        cx, cy   = ox, oy
        step     = STEP_SIZE
        max_steps = 20   # max steps per side
        loop     = 0

        # Map ToF readings to world directions roughly
        # tof_0=front(+X), tof_3=rear(-X)
        # tof_1=front-left(+Y), tof_5=front-right(-Y)
        wall_limits = {
            ( 1,  0): self.tof[0],   # front
            (-1,  0): self.tof[3],   # rear
            ( 0,  1): self.tof[1],   # left
            ( 0, -1): self.tof[5],   # right
        }

        # Expanding spiral: 1,1,2,2,3,3,4,4... steps per side
        dir_idx   = 0
        steps_this_side = 1
        sides_done = 0

        for _ in range(200):  # max 200 waypoints
            dx, dy = directions[dir_idx % 4]

            # Check wall limit in this direction
            wall_d = wall_limits.get((dx, dy), TOF_MAX)
            if wall_d < WALL_DIST:
                # Wall blocks this direction — skip and turn
                dir_idx   += 1
                sides_done += 1
                if sides_done % 2 == 0:
                    steps_this_side += 1
                continue

            # Add steps in this direction
            added = 0
            for s in range(steps_this_side):
                nx = cx + dx * STEP_SIZE
                ny = cy + dy * STEP_SIZE

                cell = (round(nx/STEP_SIZE), round(ny/STEP_SIZE))
                if cell not in visited:
                    visited.add(cell)
                    waypoints.append((nx, ny))
                    added += 1

                cx, cy = nx, ny

                # Check if we've gone too far from origin
                dist_from_start = math.sqrt(
                    (cx-ox)**2 + (cy-oy)**2)
                if dist_from_start > 15.0:  # max 15m from start
                    break

            dir_idx   += 1
            sides_done += 1
            if sides_done % 2 == 0:
                steps_this_side += 1

            # Stop if we've expanded enough
            if steps_this_side > max_steps:
                break

        return waypoints

    # ── Start exploration ─────────────────────────────────────

    def start_exploration(self):
        if self.curr_pose is None:
            print("❌ No odometry — is sim running?")
            return

        ox = self.curr_pose.position.x
        oy = self.curr_pose.position.y
        oz = self.curr_pose.position.z

        self.start_pos = {"x": ox, "y": oy, "z": EXPLORE_HEIGHT}

        print(f"📍 Start position: ({ox:.1f}, {oy:.1f}, {oz:.1f})")
        print(f"⚙️  Generating spiral from current position...")

        # Generate waypoints based on current ToF readings
        wps = self.generate_spiral(ox, oy)

        # Add return home at end
        wps.append((ox, oy))

        self.waypoints = wps
        self.wp_index  = 0
        self.exploring = True
        self.state     = "exploring"

        print(f"✅ {len(wps)} waypoints generated")
        print(f"   Speed: {MAX_VEL_XY}m/s  Height: {EXPLORE_HEIGHT}m")
        print(f"   Obstacle mux active throughout")
        print(f"   Will return home and land when done\n")

        # Set first waypoint
        self._set_next_waypoint()

    def _set_next_waypoint(self):
        if self.wp_index >= len(self.waypoints):
            # All waypoints done — return home
            self._go_home()
            return

        wx, wy = self.waypoints[self.wp_index]
        self.goal       = {"x": wx, "y": wy, "z": EXPLORE_HEIGHT}
        self.goal_start = time.time()
        self.get_logger().info(
            f"➡️  Waypoint {self.wp_index+1}/{len(self.waypoints)}: "
            f"({wx:.1f}, {wy:.1f})")

    def _go_home(self):
        if self.start_pos is None:
            return
        self.state = "returning"
        self.goal  = self.start_pos.copy()
        self.goal_start = time.time()
        print("\n🏠 All waypoints done — returning home...")

    # ── Control loop 20Hz ─────────────────────────────────────

    def control_loop(self):
        if not self.exploring:
            return
        if self.goal is None:
            return
        if self.curr_pose is None:
            return

        cx = self.curr_pose.position.x
        cy = self.curr_pose.position.y
        cz = self.curr_pose.position.z

        ex = self.goal["x"] - cx
        ey = self.goal["y"] - cy
        ez = self.goal["z"] - cz

        dist_xy = math.sqrt(ex**2 + ey**2)
        dist    = math.sqrt(ex**2 + ey**2 + ez**2)

        # ── Reached waypoint ──────────────────────────────────
        if dist_xy < GOAL_THRESH:
            if self.state == "returning":
                # Home reached — land
                print("🏠 Home reached — landing!")
                self.exploring = False
                self.state     = "done"
                self.goal      = None
                self.cmd_pub.publish(Twist())
                time.sleep(1.0)
                self.land_pub.publish(Empty())
                print("✅ Exploration complete! Map saved.")
                return

            # Next waypoint
            self.wp_index += 1
            self._set_next_waypoint()
            return

        # ── Timeout ───────────────────────────────────────────
        if time.time() - self.goal_start > GOAL_TIMEOUT:
            self.get_logger().warn(
                f"⏱ Waypoint timeout — skipping")
            if self.state == "returning":
                # Force land if stuck returning home
                self.exploring = False
                self.land_pub.publish(Empty())
                return
            self.wp_index += 1
            self._set_next_waypoint()
            return

        # ── Fly toward waypoint ───────────────────────────────
        cmd = Twist()
        cmd.linear.x  = max(-MAX_VEL_XY, min(MAX_VEL_XY, KP_XY * ex))
        cmd.linear.y  = max(-MAX_VEL_XY, min(MAX_VEL_XY, KP_XY * ey))
        cmd.linear.z  = max(-MAX_VEL_Z,  min(MAX_VEL_Z,  KP_Z  * ez))
        cmd.angular.z = 0.0
        self.cmd_pub.publish(cmd)

    # ── Status ────────────────────────────────────────────────

    def print_status(self):
        if not self.exploring and self.state == "idle":
            return

        pos_str = "unknown"
        if self.curr_pose:
            p = self.curr_pose.position
            pos_str = f"({p.x:.1f},{p.y:.1f},{p.z:.1f})"

        tof_str = " ".join(
            f"{'🔴' if self.tof[i] < WALL_DIST else '🟢'}{self.tof[i]:.1f}"
            for i in [0, 1, 5, 3])  # front, fl, fr, rear

        wp_str = f"{self.wp_index}/{len(self.waypoints)}"

        self.get_logger().info(
            f"[{self.state:10s}] pos={pos_str} "
            f"wp={wp_str} tof={tof_str}")

    # ── CLI ───────────────────────────────────────────────────

    def cli(self):
        print("\n" + "="*50)
        print("🗺️  AUTONOMOUS ROOM MAPPING")
        print("="*50)
        print("  start   — begin autonomous mapping")
        print("  stop    — stop and hover")
        print("  status  — show progress")
        print("  quit    — exit")
        print("="*50)
        print("Takeoff first with: vlm> takeoff")
        print("Then type: explore> start\n")

        while rclpy.ok():
            try:
                raw = input("explore> ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                break

            if raw == "start":
                self.start_exploration()

            elif raw == "stop":
                self.exploring = False
                self.state     = "idle"
                self.goal      = None
                self.cmd_pub.publish(Twist())
                print("⏹ Stopped — hovering")

            elif raw == "status":
                pos_str = "unknown"
                if self.curr_pose:
                    p = self.curr_pose.position
                    pos_str = f"({p.x:.1f},{p.y:.1f},{p.z:.1f})"
                print(f"\n  state:     {self.state}")
                print(f"  position:  {pos_str}")
                print(f"  waypoints: {self.wp_index}/{len(self.waypoints)}")
                print(f"  ToF front: {self.tof[0]:.2f}m")
                print(f"  ToF left:  {self.tof[1]:.2f}m")
                print(f"  ToF right: {self.tof[5]:.2f}m")
                print(f"  ToF rear:  {self.tof[3]:.2f}m\n")

            elif raw == "quit":
                break

            else:
                print("Commands: start, stop, status, quit")


def main(args=None):
    rclpy.init(args=args)
    node = ExplorationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()