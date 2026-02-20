#!/bin/bash
# ─────────────────────────────────────────────
# build_base.sh  —  Run this ONCE (takes ~15min)
# Builds the slow base image with all apt packages
# and Gazebo models baked in.
# After this, run_docker.sh only takes ~1 min.
# ─────────────────────────────────────────────

ROS_DISTRO=${ROS_DISTRO:-"humble"}

echo "🔨 Building base image (one time only, ~15 min)..."
docker build \
    --build-arg ROS_DISTRO=${ROS_DISTRO} \
    -t sjtu_drone_base:${ROS_DISTRO} \
    -f Dockerfile \
    .

if [ $? -ne 0 ]; then
    echo "❌ Base build failed!"
    exit 1
fi

echo ""
echo "✅ Base image built: sjtu_drone_base:${ROS_DISTRO}"
echo "From now on just run: bash run_docker.sh"