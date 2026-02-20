ARG ROS_DISTRO=humble
FROM ros:${ROS_DISTRO}-ros-core-jammy

ARG ROS_DISTRO=humble
ENV ROS_DISTRO=${ROS_DISTRO}

RUN apt-get update \
    && apt-get install -y \
        wget curl unzip \
        lsb-release \
        mesa-utils \
        build-essential \
    && apt-get clean

# Get Gazebo binaries
RUN echo "deb http://packages.osrfoundation.org/gazebo/ubuntu `lsb_release -cs` main" > /etc/apt/sources.list.d/gazebo-stable.list \
    && wget http://packages.osrfoundation.org/gazebo.key -O - | apt-key add - \
    && apt-get update \
    && apt-get install -y \
        gazebo \
        ros-${ROS_DISTRO}-gazebo-ros-pkgs \
        ros-${ROS_DISTRO}-gazebo-plugins \
        ros-${ROS_DISTRO}-rviz2 \
        ros-${ROS_DISTRO}-xacro \
        ros-${ROS_DISTRO}-robot-state-publisher \
        ros-${ROS_DISTRO}-joint-state-publisher \
        ros-${ROS_DISTRO}-joy \
        python3-colcon-common-extensions \
        python3-rosdep \
        xterm \
        --no-install-recommends \
    && apt-get clean

RUN rosdep init && rosdep update

# Gazebo world models (slow download — baked into base so never repeated)
RUN curl -L https://github.com/osrf/gazebo_models/archive/refs/heads/master.zip \
        -o /tmp/gazebo_models.zip \
    && unzip /tmp/gazebo_models.zip -d /tmp \
    && mkdir -p ~/.gazebo/models/ \
    && mv /tmp/gazebo_models-master/* ~/.gazebo/models/ \
    && rm -rf /tmp/gazebo_models.zip /tmp/gazebo_models-master

# Create workspace dir (files copied in Dockerfile.dev)
RUN mkdir -p /ros2_ws/src

WORKDIR /ros2_ws