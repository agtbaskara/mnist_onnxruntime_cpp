#!/bin/bash

# Set the DEBIAN_FRONTEND to noninteractive for auto yes
export DEBIAN_FRONTEND=noninteractive

# Check if 'sudo' exists
if ! command -v sudo &> /dev/null
then
    echo "sudo not found, installing sudo"
    apt update
    apt install -y sudo
else
    echo "sudo is available."
fi

# create lib directory
mkdir -p lib

# move to lib directory
cd lib

# apt update
sudo apt update
sudo apt upgrade -y

# install dependency
sudo apt install -y \
    g++ \
    gcc \
    build-essential \
    cmake \
    git \
    git-lfs \
    zip \
    unzip \
    ffmpeg

sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    libgtk2.0-dev \
    pkg-config \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libtbb2 \
    libtbb-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libdc1394-dev \
    libeigen3-dev

sudo apt-get install -y opencl-headers ocl-icd-opencl-dev
sudo apt-get install -y build-essential cmake git unzip pkg-config
sudo apt-get install -y libjpeg-dev libpng-dev
sudo apt-get install -y libavcodec-dev libavformat-dev libswscale-dev
sudo apt-get install -y libgtk2.0-dev libcanberra-gtk* libgtk-3-dev
sudo apt-get install -y libgstreamer1.0-dev gstreamer1.0-gtk3
sudo apt-get install -y libgstreamer-plugins-base1.0-dev gstreamer1.0-gl
sudo apt-get install -y libxvidcore-dev libx264-dev
sudo apt-get install -y python3-dev python3-numpy python3-pip
sudo apt-get install -y libtbb2 libtbb-dev libdc1394-22-dev
sudo apt-get install -y libv4l-dev v4l-utils
sudo apt-get install -y libopenblas-dev libatlas-base-dev libblas-dev
sudo apt-get install -y liblapack-dev gfortran libhdf5-dev
sudo apt-get install -y libprotobuf-dev libgoogle-glog-dev libgflags-dev
sudo apt-get install -y protobuf-compiler

sudo apt install -y gstreamer1.0*
sudo apt install -y ubuntu-restricted-extras
sudo apt install -y libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev

# download opencv
wget -O opencv.zip https://github.com/opencv/opencv/archive/4.8.0.zip
wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.8.0.zip

# unpack
unzip opencv.zip
unzip opencv_contrib.zip

# clean up the zip files
rm opencv.zip
rm opencv_contrib.zip

# create build directory
cd opencv-4.8.0
mkdir build
cd build

# cmake opencv
cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-4.8.0/modules \
    -D WITH_OPENMP=ON \
    -D BUILD_TIFF=ON \
    -D WITH_FFMPEG=ON \
    -D WITH_TBB=ON \
    -D BUILD_TBB=ON \
    -D WITH_GSTREAMER=ON \
    -D BUILD_TESTS=OFF \
    -D WITH_V4L=ON \
    -D WITH_LIBV4L=ON \
    -D WITH_PROTOBUF=ON \
    -D OPENCV_ENABLE_NONFREE=ON \
    -D INSTALL_C_EXAMPLES=OFF \
    -D INSTALL_PYTHON_EXAMPLES=OFF \
    -D OPENCV_GENERATE_PKGCONFIG=ON \
    -D BUILD_EXAMPLES=OFF \
    -D PYTHON_EXECUTABLE=$(which python2) \
    -D BUILD_opencv_python2=OFF \
    -D PYTHON3_EXECUTABLE=$(which python3) \
    ..

# make opencv
make -j$(nproc)
