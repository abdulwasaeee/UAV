#!/usr/bin/env python3
"""
memory_query_node.py
Natural language search over semantic map.
Uses simple label/description string matching — no vector search needed.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
import json
import threading
import time
import os

try:
    import chromadb
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

DB_PATH = "/root/.ros/semantic_map_db"


class MemoryQueryNode(Node):
    def __init__(self):
        super().__init__("memory_query_node")

        if not CHROMA_AVAILABLE:
            self.get_logger().error("chromadb not installed!")
            return

        self.client = chromadb.PersistentClient(path=DB_PATH)
        self.col    = None

        self.goal_pub = self.create_publisher(
            PoseStamped, "/drone/goal_pose", 10)

        # Try to connect to collection
        self._refresh_collection()

        # Refresh every 5s in case semantic_map_node starts later
        self.create_timer(5.0, self._refresh_collection)

        threading.Thread(target=self.cli, daemon=True).start()
        self.get_logger().info("✅ Memory Query Node ready!")

    def _refresh_collection(self):
        if self.col is not None:
            return
        try:
            self.col = self.client.get_collection("objects")
            self.get_logger().info(
                f"✅ Connected to memory — {self.col.count()} objects")
        except:
            pass

    # ── Query by string matching ──────────────────────────────

    def query(self, text):
        if self.col is None:
            print("❌ No memory yet — start semantic_map_node and fly around")
            return None

        count = self.col.count()
        if count == 0:
            print("❌ Memory is empty — fly around to discover objects")
            return None

        print(f"\n🔍 Searching {count} known objects for: '{text}'")

        try:
            results = self.col.get(include=["metadatas", "documents"])
        except Exception as e:
            print(f"❌ Could not read memory: {e}")
            return None

        text_lower = text.lower()
        scored = []

        for meta, doc in zip(results["metadatas"], results["documents"]):
            score = 0
            label = meta["label"].lower()
            desc  = doc.lower()

            # Exact label match — highest score
            if label in text_lower:
                score += 10

            # Word-level match against label
            for word in text_lower.split():
                if len(word) < 3:
                    continue
                if word in label:
                    score += 5
                if word in desc:
                    score += 2

            # Partial match — any label word in query
            for word in label.split():
                if len(word) < 3:
                    continue
                if word in text_lower:
                    score += 3

            if score > 0:
                scored.append((score, meta, doc))

        if not scored:
            print(f"❌ Nothing matching '{text}' found in memory")
            print("   Type 'list' to see all known objects")
            return None

        # Sort by score descending
        scored.sort(key=lambda x: x[0], reverse=True)
        score, meta, doc = scored[0]

        print(f"\n✅ Best match:")
        print(f"   Label:       {meta['label']}")
        print(f"   Description: {doc}")
        print(f"   Position:    "
              f"x={meta['x']:.1f}  y={meta['y']:.1f}  z={meta['z']:.1f}")

        if len(scored) > 1:
            print(f"\n   Other candidates:")
            for s, m, d in scored[1:4]:
                print(f"   - {m['label']:15s} "
                      f"({m['x']:.1f}, {m['y']:.1f}, {m['z']:.1f})"
                      f"  \"{d[:40]}\"")

        return meta

    def goto(self, text):
        meta = self.query(text)
        if meta is None:
            return

        goal = PoseStamped()
        goal.header.frame_id    = "map"
        goal.header.stamp       = self.get_clock().now().to_msg()
        goal.pose.position.x    = float(meta["x"])
        goal.pose.position.y    = float(meta["y"])
        goal.pose.position.z    = max(float(meta["z"]), 1.0)
        goal.pose.orientation.w = 1.0

        self.goal_pub.publish(goal)
        print(f"\n🚁 Flying to {meta['label']} "
              f"at ({meta['x']:.1f}, {meta['y']:.1f}, {meta['z']:.1f})")

    def list_objects(self):
        if self.col is None:
            print("❌ Not connected to memory")
            return

        try:
            count = self.col.count()
        except:
            print("❌ Could not read memory")
            return

        if count == 0:
            print("Memory is empty — fly around to discover objects")
            return

        try:
            results = self.col.get(include=["metadatas", "documents"])
        except Exception as e:
            print(f"❌ Error reading objects: {e}")
            return

        print(f"\n📚 Known objects ({count}):")
        print("-" * 65)
        for meta, doc in zip(results["metadatas"], results["documents"]):
            t = time.strftime(
                "%H:%M:%S", time.localtime(meta.get("timestamp", 0)))
            print(f"  [{t}] {meta['label']:15s} "
                  f"({meta['x']:.1f}, {meta['y']:.1f}, {meta['z']:.1f})")
            print(f"         \"{doc[:60]}\"")
        print("-" * 65)

    def clear_memory(self):
        confirm = input("Clear ALL memory? (yes/no): ").strip().lower()
        if confirm != "yes":
            print("Cancelled")
            return
        try:
            self.client.delete_collection("objects")
            self.col = self.client.create_collection(
                "objects",
                metadata={"hnsw:space": "cosine"})
            print("✅ Memory cleared")
        except Exception as e:
            print(f"Error: {e}")

    # ── CLI ───────────────────────────────────────────────────

    def cli(self):
        print("\n" + "="*55)
        print("🧠 DRONE MEMORY")
        print("="*55)
        print("  list                — show all known objects")
        print("  goto <description>  — fly to object")
        print("  clear               — wipe memory")
        print("  quit                — exit")
        print("  <anything>          — search memory")
        print("="*55)
        print("Examples:")
        print("  vase")
        print("  goto umbrella")
        print("  person near dumpster")
        print("  goto bottle")
        print("="*55 + "\n")

        while rclpy.ok():
            try:
                raw = input("memory> ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if not raw:
                continue

            cmd = raw.lower()

            if cmd == "list":
                self.list_objects()

            elif cmd == "clear":
                self.clear_memory()

            elif cmd == "quit":
                break

            elif cmd.startswith("goto "):
                self.goto(raw[5:].strip())

            else:
                # Anything else = search
                self.query(raw)


def main(args=None):
    rclpy.init(args=args)
    node = MemoryQueryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
