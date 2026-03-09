#!/bin/bash
# ─────────────────────────────────────────────
# run_docker.sh  —  Fast rebuild + run (~1 min)
# Requires base image built first: bash build_base.sh
# ─────────────────────────────────────────────

usage(){
    echo "Usage: $0 [-r <humble|iron|rolling>]"
    exit 1
}

ROS_DISTRO=${ROS_DISTRO:-"humble"}

while getopts "r:" opt; do
    case $opt in
        r)
            if [ $OPTARG != "humble" ] && [ $OPTARG != "iron" ] && [ $OPTARG != "rolling" ]; then
                echo "Invalid ROS distro: $OPTARG" >&2
                usage
            fi
            ROS_DISTRO=$OPTARG
            ;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            usage
            ;;
    esac
done

XSOCK=/tmp/.X11-unix
XAUTH=$HOME/.Xauthority

# Check base image exists, build it if not
if ! docker image inspect sjtu_drone_base:${ROS_DISTRO} &>/dev/null; then
    echo "⚠️  Base image not found! Building it now (one-time, ~15 min)..."
    bash build_base.sh
fi

# Fast rebuild from local files only (~1 min)
echo "🔨 Rebuilding from local files (fast)..."
docker build \
    --build-arg ROS_DISTRO=${ROS_DISTRO} \
    -f Dockerfile.dev \
    -t sjtu_drone_local:${ROS_DISTRO} \
    .

if [ $? -ne 0 ]; then
    echo "❌ Build failed!"
    exit 1
fi
echo "✅ Build done!"

# Stop existing container
docker stop sjtu_drone 2>/dev/null
docker rm sjtu_drone 2>/dev/null

# Run
xhost +local:docker

docker run \
    -it --rm \
    -v ${XSOCK}:${XSOCK} \
    -v ${XAUTH}:${XAUTH} \
    -v ~/vlm/nodes:/nodes \
    -v ${HOME}/vlm/maps:/root/.ros \
    -e DISPLAY=${DISPLAY} \
    -e XAUTHORITY=${XAUTH} \
    --env=QT_X11_NO_MITSHM=1 \
    --privileged \
    --net=host \
    --name="sjtu_drone" \
    sjtu_drone_local:${ROS_DISTRO}

xhost -local:docker