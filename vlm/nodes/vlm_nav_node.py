#!/usr/bin/env python3
"""
vlm_nav_node.py — VLM Navigation + Obstacle Avoidance
ToF sensors read via subprocess to avoid rclpy Range bug.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_msgs.msg import Empty

from cv_bridge import CvBridge
import cv2, base64, json, threading, requests, time, subprocess, re

# ── Config ─────────────────────────────────────────────────────
OLLAMA_URL    = "http://localhost:11434/api/generate"
VLM_MODEL     = "llava"
IMAGE_W       = 320
IMAGE_H       = 240
VLM_INTERVAL  = 3.0
OBSTACLE_DIST = 1.0
FORWARD_SPEED = 0.3
# ───────────────────────────────────────────────────────────────


class VLMNavNode(Node):
    def __init__(self):
        super().__init__("vlm_nav_node")
        self.bridge = CvBridge()

        # State
        self.frame        = None
        self.task         = None
        self.flying       = False
        self.vlm_active   = False
        self._vlm_running = False
        self.tof          = {"front": 9.9, "back": 9.9, "left": 9.9, "right": 9.9}

        # ── Image subscriber (works fine) ─────────────────────
        self.create_subscription(
            Image, "/simple_drone/rgbd/image_raw", self.img_cb, 10)

        # ── Publishers ────────────────────────────────────────
        self.cmd_pub     = self.create_publisher(Twist, "/simple_drone/cmd_vel", 10)
        self.takeoff_pub = self.create_publisher(Empty, "/simple_drone/takeoff", 10)
        self.land_pub    = self.create_publisher(Empty, "/simple_drone/land",    10)

        # ── Timers ────────────────────────────────────────────
        self.create_timer(VLM_INTERVAL, self.vlm_loop)
        self.create_timer(0.1, self.avoid_loop)   # 10Hz avoidance

        # ── Poll ToF via subprocess (avoids rclpy Range bug) ──
        threading.Thread(target=self._poll_tof, daemon=True).start()

        self.get_logger().info("✅ VLM Nav Node ready!")
        threading.Thread(target=self.cli, daemon=True).start()

    # ── ToF polling via ros2 topic echo ───────────────────────

    def _poll_tof(self):
        """Read all 4 ToF sensors in background threads."""
        for direction, topic in [
            ("front", "/simple_drone/tof_front_plugin/out"),
            ("back",  "/simple_drone/tof_back_plugin/out"),
            ("left",  "/simple_drone/tof_left_plugin/out"),
            ("right", "/simple_drone/tof_right_plugin/out"),
        ]:
            threading.Thread(
                target=self._echo_tof,
                args=(direction, topic),
                daemon=True
            ).start()

    def _echo_tof(self, direction, topic):
        """Subscribe to a single ToF topic via subprocess."""
        cmd = [
            "bash", "-c",
            f"source /opt/ros/humble/setup.bash && "
            f"ros2 topic echo --no-arr {topic} sensor_msgs/msg/Range"
        ]
        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, text=True)
            for line in proc.stdout:
                line = line.strip()
                if line.startswith("range:"):
                    try:
                        val = float(line.split(":")[1].strip())
                        self.tof[direction] = val
                    except:
                        pass
        except Exception as e:
            self.get_logger().warn(f"ToF {direction} poll error: {e}")

    # ── Image callback ────────────────────────────────────────

    def img_cb(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.frame = cv2.resize(frame, (IMAGE_W, IMAGE_H))
        except Exception as e:
            self.get_logger().warn(f"Image error: {e}")

    # ── Obstacle Avoidance ────────────────────────────────────

    def avoid_loop(self):
        if not self.flying:
            return

        cmd   = Twist()
        block = False

        if self.tof["front"] < OBSTACLE_DIST:
            cmd.linear.x = -0.3
            cmd.linear.z =  0.2
            block = True
            self.get_logger().warn(f"🚨 FRONT {self.tof['front']:.2f}m — backing up!")
        elif self.tof["left"] < OBSTACLE_DIST:
            cmd.linear.y = -0.3
            block = True
            self.get_logger().warn(f"🚨 LEFT {self.tof['left']:.2f}m — moving right!")
        elif self.tof["right"] < OBSTACLE_DIST:
            cmd.linear.y =  0.3
            block = True
            self.get_logger().warn(f"🚨 RIGHT {self.tof['right']:.2f}m — moving left!")
        elif self.tof["back"] < OBSTACLE_DIST:
            cmd.linear.x =  0.3
            block = True
            self.get_logger().warn(f"🚨 BACK {self.tof['back']:.2f}m — moving fwd!")

        if block:
            self.vlm_active = False
            self.cmd_pub.publish(cmd)
        elif self.task and not self.vlm_active:
            self.vlm_active = True  # re-enable after obstacle cleared

    # ── VLM Loop ──────────────────────────────────────────────

    def vlm_loop(self):
        if not self.vlm_active or not self.flying:
            return
        if self.frame is None:
            self.get_logger().warn("⏳ Waiting for camera...")
            return
        if self._vlm_running:
            return
        threading.Thread(target=self._run_vlm, daemon=True).start()

    def _run_vlm(self):
        self._vlm_running = True
        try:
            frame = self.frame.copy()
            task  = self.task

            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            b64 = base64.b64encode(buf).decode()

            prompt = (
                f'You are a drone navigation AI. Task: "{task}"\n'
                f"Find the target in this image.\n"
                f"Reply ONLY with JSON:\n"
                f'{{"found": true/false, "u": <0-{IMAGE_W}>, '
                f'"v": <0-{IMAGE_H}>, "depth": <1-5>, "note": "<what you see>"}}\n'
                f"depth: 1=very close, 2=close, 3=medium, 4=far, 5=very far\n"
                f"JSON only, no other text."
            )

            resp = requests.post(OLLAMA_URL, json={
                "model":  VLM_MODEL,
                "prompt": prompt,
                "images": [b64],
                "stream": False,
                "format": "json"
            }, timeout=20)

            if resp.status_code != 200:
                self.get_logger().warn(f"Ollama {resp.status_code}")
                return

            raw    = resp.json().get("response", "{}")
            parsed = json.loads(raw)

            found = bool(parsed.get("found", False))
            u     = max(0, min(int(parsed.get("u", IMAGE_W // 2)), IMAGE_W))
            v     = max(0, min(int(parsed.get("v", IMAGE_H // 2)), IMAGE_H))
            depth = max(1, min(int(parsed.get("depth", 3)), 5))
            note  = parsed.get("note", "")

            icon = "✅ FOUND" if found else "🔍 SEARCHING"
            self.get_logger().info(f"{icon} | ({u},{v}) d={depth} | {note}")

            # Debug overlay
            debug = frame.copy()
            cv2.circle(debug, (u, v), 10, (0, 255, 0), -1)
            cv2.putText(debug, note[:50], (5, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            cv2.putText(debug, f"depth={depth}", (5, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            cv2.imshow("VLM View", debug)
            cv2.waitKey(1)

            if self.vlm_active and self.flying:
                cmd = self._pixel_to_vel(u, v, depth, found)
                self.cmd_pub.publish(cmd)

        except requests.exceptions.ConnectionError:
            self.get_logger().error("❌ Ollama not reachable! Run: ollama serve")
        except json.JSONDecodeError as e:
            self.get_logger().warn(f"JSON error: {e}")
        except Exception as e:
            self.get_logger().warn(f"VLM error: {e}")
        finally:
            self._vlm_running = False

    def _pixel_to_vel(self, u, v, depth, found):
        cmd = Twist()
        if not found:
            cmd.angular.z = 0.3
            return cmd
        if depth == 1:
            self.get_logger().info("🎯 Target reached!")
            return cmd
        ex  = (u - IMAGE_W / 2.0) / IMAGE_W
        ey  = (IMAGE_H / 2.0 - v) / IMAGE_H
        spd = FORWARD_SPEED * (depth / 3.0)
        cmd.linear.x  =  spd
        cmd.linear.y  = -ex * spd * 1.5
        cmd.linear.z  =  ey * 0.4
        cmd.angular.z = -ex * 0.5
        return cmd

    # ── CLI ───────────────────────────────────────────────────

    def cli(self):
        print("\n" + "="*50)
        print("🚁 VLM DRONE — type commands below")
        print("="*50)
        print("  takeoff")
        print("  land")
        print("  stop")
        print("  status")
        print("  quit")
        print("  <anything else> = natural language task")
        print("="*50 + "\n")

        while rclpy.ok():
            try:
                raw = input("drone> ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            cmd = raw.lower()

            if cmd == "takeoff":
                self.takeoff_pub.publish(Empty())
                self.flying = True
                print("✅ Takeoff!")

            elif cmd == "land":
                self.vlm_active = False
                self.task = None
                self.cmd_pub.publish(Twist())
                time.sleep(0.3)
                self.land_pub.publish(Empty())
                self.flying = False
                print("✅ Landing!")

            elif cmd == "stop":
                self.vlm_active = False
                self.cmd_pub.publish(Twist())
                print("⏹ Hovering.")

            elif cmd == "status":
                print(f"""
  flying={self.flying}  task='{self.task}'  vlm={self.vlm_active}
  tof: front={self.tof['front']:.2f}  back={self.tof['back']:.2f}
       left={self.tof['left']:.2f}   right={self.tof['right']:.2f}
  frame={'ready' if self.frame is not None else 'waiting'}
""")
            elif cmd == "quit":
                break

            elif raw:
                # Everything else = natural language task
                self.task = raw
                self.vlm_active = True
                print(f"✅ Task: '{self.task}'")


def main(args=None):
    rclpy.init(args=args)
    node = VLMNavNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()