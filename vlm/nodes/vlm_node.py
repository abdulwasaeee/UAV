#!/usr/bin/env python3
"""
vlm_node.py - Fast keyword-based drone control
No Mistral — instant parsing, no lag.
Goal stored in odom frame to fix coordinate mismatch.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Empty, String
import json
import math
import threading
import time
import os

try:
    import chromadb
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

# ── Config ──────────────────────────────────────────────────────
DB_PATH      = "/root/.ros/semantic_map_db"
GOAL_THRESH  = 0.5    # meters — close enough
MIN_HEIGHT   = 1.0    # meters
KP_XY        = 0.4
KP_Z         = 0.3
MAX_VEL_XY   = 0.4
MAX_VEL_Z    = 0.25
# ────────────────────────────────────────────────────────────────


class VLMNode(Node):
    def __init__(self):
        super().__init__("vlm_node")

        self.curr_pose  = None
        self.goal       = None   # {x, y, z} in ODOM frame
        self.flying     = False
        self.state      = "idle"
        self.col        = None

        # ── ChromaDB ─────────────────────────────────────────
        self._connect_db()

        # ── Subscribers ───────────────────────────────────────
        self.create_subscription(
            Odometry, "/simple_drone/odom", self.odom_cb, 10)
        self.create_subscription(
            String, "/semantic_map/objects", self.objects_cb, 10)

        # ── Publishers ────────────────────────────────────────
        self.cmd_pub     = self.create_publisher(
            Twist,  "/cmd_vel/input",        10)
        self.takeoff_pub = self.create_publisher(
            Empty,  "/simple_drone/takeoff", 10)
        self.land_pub    = self.create_publisher(
            Empty,  "/simple_drone/land",    10)

        # ── Control loop 20Hz ─────────────────────────────────
        self.create_timer(0.05, self.control_loop)

        self.get_logger().info("✅ VLM Node ready — keyword parser active")
        threading.Thread(target=self.cli, daemon=True).start()

    def _connect_db(self):
        if not CHROMA_AVAILABLE:
            return
        if not os.path.exists(DB_PATH):
            return
        try:
            client   = chromadb.PersistentClient(path=DB_PATH)
            self.col = client.get_collection("objects")
            self.get_logger().info(
                f"✅ Memory: {self.col.count()} objects")
        except:
            pass

    # ── Callbacks ─────────────────────────────────────────────

    def odom_cb(self, msg):
        self.curr_pose = msg.pose.pose

    def objects_cb(self, msg):
        if self.col is None:
            self._connect_db()

    # ── Control loop 20Hz ─────────────────────────────────────

    def control_loop(self):
        if not self.flying:
            return
        if self.goal is None:
            return
        if self.state != "navigating":
            return
        if self.curr_pose is None:
            return

        cx = self.curr_pose.position.x
        cy = self.curr_pose.position.y
        cz = self.curr_pose.position.z

        ex = self.goal["x"] - cx
        ey = self.goal["y"] - cy
        ez = self.goal["z"] - cz

        dist = math.sqrt(ex**2 + ey**2 + ez**2)

        if dist < GOAL_THRESH:
            print(f"\n✅ Reached! Hovering.")
            self.goal  = None
            self.state = "hovering"
            self.cmd_pub.publish(Twist())
            return

        cmd = Twist()
        cmd.linear.x  = max(-MAX_VEL_XY, min(MAX_VEL_XY, KP_XY * ex))
        cmd.linear.y  = max(-MAX_VEL_XY, min(MAX_VEL_XY, KP_XY * ey))
        cmd.linear.z  = max(-MAX_VEL_Z,  min(MAX_VEL_Z,  KP_Z  * ez))
        cmd.angular.z = 0.0  # no yaw rotation — stops spinning
        self.cmd_pub.publish(cmd)

    # ── Keyword parser — instant, no AI needed ────────────────

    def parse(self, text):
        t = text.lower().strip()

        if any(w in t for w in ["take off", "takeoff", "launch", "fly up"]):
            return "takeoff", None
        if any(w in t for w in ["land", "descend", "come down"]):
            return "land", None
        if any(w in t for w in ["stop", "hover", "halt", "hold"]):
            return "stop", None
        if t in ("list", "what do you know", "what have you seen"):
            return "list", None
        if t == "status":
            return "status", None

        # goto — strip all goto keywords and use remainder as target
        for kw in ["go to", "goto", "fly to", "navigate to",
                   "move to", "head to", "return to"]:
            if kw in t:
                target = t.replace(kw, "").strip()
                return "goto", target

        # find
        for kw in ["find", "search for", "look for", "locate", "where is"]:
            if kw in t:
                target = t.replace(kw, "").strip()
                return "goto", target

        # remember
        for kw in ["remember", "recall", "did you see"]:
            if kw in t:
                target = t.replace(kw, "").strip()
                return "remember", target

        # default — treat whole string as goto target
        return "goto", text.strip()

    # ── Memory search ─────────────────────────────────────────

    def search_memory(self, query):
        if self.col is None:
            self._connect_db()
        if self.col is None or self.col.count() == 0:
            print("❌ Memory empty — fly around first")
            return None

        try:
            results = self.col.get(include=["metadatas", "documents"])
        except Exception as e:
            print(f"❌ Memory error: {e}")
            return None

        q = query.lower()
        scored = []

        for meta, doc in zip(results["metadatas"], results["documents"]):
            score = 0
            label = meta["label"].lower()
            desc  = doc.lower()

            if label in q:           score += 10
            if q in label:           score += 8
            for word in q.split():
                if len(word) < 3:    continue
                if word in label:    score += 5
                if word in desc:     score += 2
            for word in label.split():
                if len(word) < 3:    continue
                if word in q:        score += 3

            if score > 0:
                scored.append((score, meta, doc))

        if not scored:
            print(f"❌ Nothing matching '{query}' in memory")
            print("   Type 'list' to see all known objects")
            return None

        scored.sort(key=lambda x: x[0], reverse=True)
        _, meta, doc = scored[0]

        print(f"\n✅ Found: {meta['label']}")
        print(f"   Description: {doc[:60]}")
        print(f"   Position: x={meta['x']:.1f}  "
              f"y={meta['y']:.1f}  z={meta['z']:.1f}")

        if len(scored) > 1:
            print("   Other matches:")
            for _, m, d in scored[1:3]:
                print(f"   - {m['label']:12s} "
                      f"({m['x']:.1f},{m['y']:.1f},{m['z']:.1f})"
                      f"  \"{d[:35]}\"")
        return meta

    def list_memory(self):
        if self.col is None or self.col.count() == 0:
            print("Memory empty — fly around first")
            return
        try:
            results = self.col.get(include=["metadatas", "documents"])
            print(f"\n📚 Known objects ({len(results['metadatas'])}):")
            print("-" * 60)
            for meta, doc in zip(results["metadatas"], results["documents"]):
                t = time.strftime(
                    "%H:%M:%S", time.localtime(meta.get("timestamp", 0)))
                print(f"  [{t}] {meta['label']:15s} "
                      f"({meta['x']:.1f},{meta['y']:.1f},{meta['z']:.1f})")
                print(f"         \"{doc[:55]}\"")
            print("-" * 60)
        except Exception as e:
            print(f"Error: {e}")

    # ── Execute ───────────────────────────────────────────────

    def execute(self, action, target):
        if action == "takeoff":
            self.takeoff_pub.publish(Empty())
            self.flying = True
            self.state  = "hovering"
            print("✅ Takeoff!")

        elif action == "land":
            self.goal   = None
            self.state  = "landing"
            self.cmd_pub.publish(Twist())
            time.sleep(0.3)
            self.land_pub.publish(Empty())
            self.flying = False
            self.state  = "idle"
            print("✅ Landed!")

        elif action == "stop":
            self.goal  = None
            self.state = "hovering"
            self.cmd_pub.publish(Twist())
            print("⏹ Hovering")

        elif action == "list":
            self.list_memory()

        elif action == "status":
            mem = self.col.count() if self.col else 0
            print(f"\n  state:   {self.state}")
            print(f"  flying:  {self.flying}")
            print(f"  memory:  {mem} objects known")
            if self.curr_pose:
                p = self.curr_pose.position
                print(f"  odom pos: ({p.x:.2f}, {p.y:.2f}, {p.z:.2f})")
            if self.goal:
                g = self.goal
                print(f"  goal:    ({g['x']:.2f}, {g['y']:.2f}, {g['z']:.2f})")
            print()

        elif action == "remember":
            self.search_memory(target or "")

        elif action == "goto":
            if not target:
                print("❌ No target")
                return
            if not self.flying:
                print("❌ Take off first!")
                return

            meta = self.search_memory(target)
            if meta is None:
                return

            # ── Key fix: store goal in odom frame ─────────────
            # The semantic map stores positions using drone odom
            # directly so no frame conversion needed
            self.goal = {
                "x": float(meta["x"]),
                "y": float(meta["y"]),
                "z": max(float(meta["z"]), MIN_HEIGHT),
            }
            self.state = "navigating"
            print(f"\n🚁 Navigating to {meta['label']} "
                  f"({self.goal['x']:.1f}, "
                  f"{self.goal['y']:.1f}, "
                  f"{self.goal['z']:.1f})")
            print("   Type 'stop' to cancel")

    # ── CLI ───────────────────────────────────────────────────

    def cli(self):
        print("\n" + "="*50)
        print("🤖 VLM DRONE — KEYWORD CONTROL")
        print("="*50)
        print("  takeoff              — take off")
        print("  land                 — land")
        print("  stop                 — hover")
        print("  go to <object>       — fly to object")
        print("  find <object>        — locate object")
        print("  remember <object>    — search memory")
        print("  list                 — all known objects")
        print("  status               — current state")
        print("  quit                 — exit")
        print("="*50 + "\n")

        while rclpy.ok():
            try:
                raw = input("vlm> ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if not raw:
                continue
            if raw.lower() == "quit":
                break

            action, target = self.parse(raw)
            self.execute(action, target)


def main(args=None):
    rclpy.init(args=args)
    node = VLMNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()