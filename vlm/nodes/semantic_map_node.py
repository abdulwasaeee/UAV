#!/usr/bin/env python3
"""
semantic_map_node.py
Stores YOLO detections in ChromaDB with LLaVA descriptions.
Proper deduplication — one entry per object location.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker, MarkerArray
from cv_bridge import CvBridge
import json
import time
import base64
import requests
import cv2
import threading
import os
import math

try:
    import chromadb
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

# ── Config ──────────────────────────────────────────────────────
OLLAMA_URL    = "http://localhost:11434"
EMBED_MODEL   = "nomic-embed-text"
LLAVA_MODEL   = "llava"
DB_PATH       = "/root/.ros/semantic_map_db"
MIN_DIST      = 1.5    # meters — don't re-add if same object nearby
DESCRIBE_EVERY = 10.0  # seconds between LLaVA calls per object
# ────────────────────────────────────────────────────────────────


class SemanticMapNode(Node):
    def __init__(self):
        super().__init__("semantic_map_node")

        self.bridge      = CvBridge()
        self.rgb_frame   = None
        self._llava_lock = threading.Lock()
        self._llava_busy = False

        # In-memory object store — label -> list of {pos, time, described}
        # Used for fast dedup before hitting ChromaDB
        self._seen = {}

        if not CHROMA_AVAILABLE:
            self.get_logger().error("chromadb not installed!")
            return

        os.makedirs(DB_PATH, exist_ok=True)

        # Fresh client each time
        self.client = chromadb.PersistentClient(path=DB_PATH)

        # Delete and recreate collection to avoid stale index issues
        try:
            self.client.delete_collection("objects")
        except:
            pass

        self.col = self.client.create_collection(
            name="objects",
            metadata={"hnsw:space": "cosine"}
        )

        self.get_logger().info(f"✅ SemanticMap ready — fresh DB at {DB_PATH}")

        # ── Subscribers ──────────────────────────────────────
        self.create_subscription(
            String, "/detections", self.det_cb, 10)
        self.create_subscription(
            Image, "/simple_drone/rgbd/image_raw_fixed",
            self.rgb_cb, 10)

        # ── Publishers ────────────────────────────────────────
        self.marker_pub  = self.create_publisher(
            MarkerArray, "/semantic_map/markers", 10)
        self.objects_pub = self.create_publisher(
            String, "/semantic_map/objects", 10)

        self.create_timer(2.0, self.publish_markers)

    # ── Callbacks ────────────────────────────────────────────

    def rgb_cb(self, msg):
        try:
            self.rgb_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except:
            pass

    def det_cb(self, msg):
        try:
            dets = json.loads(msg.data)
            for d in dets:
                threading.Thread(
                    target=self._handle,
                    args=(d,), daemon=True).start()
        except Exception as e:
            self.get_logger().warn(f"Det parse: {e}")

    # ── Handle Detection ─────────────────────────────────────

    def _handle(self, det):
        label = det["label"]
        w     = det["world"]
        pos   = (w["x"], w["y"], w["z"])

        # Fast in-memory dedup
        if self._near_known(label, pos):
            return

        # Register in memory immediately to block duplicates
        if label not in self._seen:
            self._seen[label] = []
        self._seen[label].append({
            "pos": pos,
            "time": time.time(),
            "described": False
        })

        self.get_logger().info(f"📍 New: {label} at {pos}")

        # Get description from LLaVA (non-blocking, best effort)
        description = label
        if not self._llava_busy and self.rgb_frame is not None:
            description = self._describe(label, det.get("depth_m", 2.0))

        # Get embedding
        emb = self._embed(f"{label}: {description}")
        if emb is None:
            self.get_logger().warn(f"No embedding for {label}")
            return

        # Store in ChromaDB
        obj_id = f"{label}_{int(time.time()*1000)}"
        try:
            self.col.add(
                ids=[obj_id],
                embeddings=[emb],
                documents=[description],
                metadatas=[{
                    "label":       label,
                    "description": description,
                    "x": pos[0], "y": pos[1], "z": pos[2],
                    "confidence":  det.get("confidence", 0.0),
                    "timestamp":   time.time(),
                }]
            )
            self.get_logger().info(
                f"✅ Saved: '{label}' — \"{description[:60]}\"")
            self._publish_objects()
        except Exception as e:
            self.get_logger().warn(f"ChromaDB add error: {e}")

    def _near_known(self, label, pos, dist=MIN_DIST):
        """Check in-memory store if same label exists nearby."""
        for entry in self._seen.get(label, []):
            p = entry["pos"]
            d = math.sqrt(
                (p[0]-pos[0])**2 +
                (p[1]-pos[1])**2 +
                (p[2]-pos[2])**2
            )
            if d < dist:
                return True
        return False

    # ── LLaVA ────────────────────────────────────────────────

    def _describe(self, label, depth):
        with self._llava_lock:
            if self._llava_busy:
                return label
            self._llava_busy = True

        try:
            frame = self.rgb_frame.copy()
            _, buf = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            b64 = base64.b64encode(buf).decode()

            prompt = (
                f"There is a {label} about {depth:.1f}m away. "
                f"Describe it in one short sentence: color, size, location in frame."
            )
            resp = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model":  LLAVA_MODEL,
                    "prompt": prompt,
                    "images": [b64],
                    "stream": False,
                },
                timeout=12
            )
            if resp.status_code == 200:
                raw = resp.json().get("response", label).strip()
                return raw.split(".")[0].strip()
        except Exception as e:
            self.get_logger().warn(f"LLaVA: {e}")
        finally:
            self._llava_busy = False

        return label

    # ── Embedding ─────────────────────────────────────────────

    def _embed(self, text):
        try:
            resp = requests.post(
                f"{OLLAMA_URL}/api/embeddings",
                json={"model": EMBED_MODEL, "prompt": text},
                timeout=10
            )
            if resp.status_code == 200:
                return resp.json()["embedding"]
        except Exception as e:
            self.get_logger().warn(f"Embed error: {e}")
        return None

    # ── RViz Markers ──────────────────────────────────────────

    def publish_markers(self):
        try:
            if self.col.count() == 0:
                return

            results  = self.col.get(include=["metadatas", "documents"])
            markers  = MarkerArray()
            stamp    = self.get_clock().now().to_msg()

            for i, (meta, doc) in enumerate(
                    zip(results["metadatas"], results["documents"])):

                # Text label
                m            = Marker()
                m.header.frame_id = "map"
                m.header.stamp    = stamp
                m.ns         = "labels"
                m.id         = i
                m.type       = Marker.TEXT_VIEW_FACING
                m.action     = Marker.ADD
                m.pose.position.x = meta["x"]
                m.pose.position.y = meta["y"]
                m.pose.position.z = meta["z"] + 0.4
                m.pose.orientation.w = 1.0
                m.scale.z    = 0.25
                m.color.a    = 1.0
                m.color.r    = 0.2
                m.color.g    = 1.0
                m.color.b    = 0.8
                m.text       = f"{meta['label']}\n{doc[:40]}"
                m.lifetime.sec = 3
                markers.markers.append(m)

                # Sphere
                s            = Marker()
                s.header.frame_id = "map"
                s.header.stamp    = stamp
                s.ns         = "spheres"
                s.id         = i + 1000
                s.type       = Marker.SPHERE
                s.action     = Marker.ADD
                s.pose.position.x = meta["x"]
                s.pose.position.y = meta["y"]
                s.pose.position.z = meta["z"]
                s.pose.orientation.w = 1.0
                s.scale.x = s.scale.y = s.scale.z = 0.25
                s.color.a    = 0.9
                s.color.r    = 1.0
                s.color.g    = 0.4
                s.color.b    = 0.0
                s.lifetime.sec = 3
                markers.markers.append(s)

            self.marker_pub.publish(markers)

        except Exception as e:
            self.get_logger().warn(f"Marker error: {e}")

    def _publish_objects(self):
        try:
            if self.col.count() == 0:
                return
            results = self.col.get(include=["metadatas", "documents"])
            objects = [
                {
                    "label":       m["label"],
                    "description": d,
                    "position":    {
                        "x": m["x"], "y": m["y"], "z": m["z"]},
                    "confidence":  m.get("confidence", 0),
                    "timestamp":   m.get("timestamp", 0),
                }
                for m, d in zip(results["metadatas"], results["documents"])
            ]
            self.objects_pub.publish(
                String(data=json.dumps(objects)))
        except Exception as e:
            self.get_logger().warn(f"Publish objects: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = SemanticMapNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
