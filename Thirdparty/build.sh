#!/bin/bash 
# nyk 22/05/2023
# Run this script once as it will clean up after itself. Everytime you run it will recompile packages (except opencv)
BuildType="RelWithDebInfo"

SCRIPTPATH=$(dirname $0)
if [ $SCRIPTPATH = '.' ]
then
SCRIPTPATH=$(pwd)
fi

mkdir -p $SCRIPTPATH/CompiledLibs
InstallDir=$SCRIPTPATH/CompiledLibs


#install system wide dependencies
#================================
export DEBIAN_FRONTEND=noninteractive
#sudo apt -y install libgl1-mesa-dev libglew-dev libsuitesparse-dev libeigen3-dev libboost-all-dev cmake build-essential git libzip-dev ccache freeglut3-dev libgoogle-glog-dev libatlas-base-dev ninja-build

#install libceres for compatibility with ubuntu 22:
#cd $SCRIPTPATH/Thirdparty/
#wget ceres-solver.org/ceres-solver-1.14.0.tar.gz
#tar -zxf ceres-solver-1.14.0.tar.gz && 
echo -e "Compiling Ceres\n"
cd ceres-solver-1.14.0 && mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$InstallDir -DBUILD_EXAMPLES=OFF -DBUILD_TESTING=OFF -DCXX11=ON 
make -j $(nproc) && make install
cd .. && rm -r build
cd .. && rm ceres-solver-1.14.0.tar.gz && rm -r ceres-solver-1.14.0
echo -e "Ceres Installed\n"

#optional libs to record pangolin gui
#sudo apt -y install ffmpeg libavcodec-dev libavutil-dev libavformat-dev libswscale-dev libavdevice-dev

#if you have OpenCV3.4 comment out the following and specify the directory later
#sudo apt -y install libjpeg8-dev libpng-dev libtiff5-dev libtiff-dev libavcodec-dev libavformat-dev libv4l-dev libgtk2.0-dev qtbase5-dev qtchooser qt5-qmake qtbase5-dev-tools v4l-utils
# OpenCV installation
echo -e "Compiling OpenCV3.4.6\n"
cvVersion=3.4.6
# Define directories
OPENCV_DIR=$SCRIPTPATH/opencv-${cvVersion}
OPENCV_CONTRIB_DIR=$OPENCV_DIR/opencv_contrib-${cvVersion}
#if [ ! -d "$SCRIPTPATH/Thirdparty/opencv-${cvVersion}" ]; then
#  DL_opencv="https://github.com/opencv/opencv/archive/${cvVersion}.zip"
#  DL_contrib="https://github.com/opencv/opencv_contrib/archive/${cvVersion}.zip"
#  cd $SCRIPTPATH/Thirdparty/
#  wget -O opencv.zip -nc "${DL_opencv}" && unzip opencv.zip && rm opencv.zip && cd opencv-${cvVersion}
#  wget -O opencv_contrib.zip -nc "${DL_contrib}" && unzip opencv_contrib.zip && rm opencv_contrib.zip
#fi
cd $OPENCV_DIR && mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=$BuildType -DCMAKE_INSTALL_PREFIX=$InstallDir -DWITH_V4L=ON -DWITH_CUDA=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_TESTS=OFF -DWITH_QT=ON -DCMAKE_CXX_FLAGS=-std=c++11 -DWITH_OPENGL=ON -DOPENCV_EXTRA_MODULES_PATH=$OPENCV_CONTRIB_DIR/modules -DOPENCV_ENABLE_NONFREE=ON -DCeres_DIR=$InstallDir/lib/cmake/Ceres

make -j $(nproc) && make install
cd .. && rm -rf build
#cd $SCRIPTPATH/opencv-${cvVersion} && mkdir -p build && cd build &&cmake .. -DCMAKE_BUILD_TYPE=$BuildType -DCMAKE_INSTALL_PREFIX=$InstallDir -DWITH_V4L=ON -DWITH_CUDA=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_TESTS=OFF -DWITH_QT=ON -DCMAKE_CXX_FLAGS=-std=c++11 -DWITH_OPENGL=ON -DOPENCV_EXTRA_MODULES_PATH=$SCRIPTPATH/opencv-${cvVersion}/opencv_contrib-${cvVersion}/modules -DOPENCV_ENABLE_NONFREE=ON -DCeres_DIR=$InstallDir/lib/cmake/Ceres && make -j $(nproc) && make install && cd .. && rm -r build
#end comment out OpenCV
# Ensure CMake can find OpenCV
export OpenCV_DIR=$InstallDir
export PKG_CONFIG_PATH=$InstallDir/lib/pkgconfig:$PKG_CONFIG_PATH
export LD_LIBRARY_PATH=$InstallDir/lib:$LD_LIBRARY_PATH
#Build Thirdparty libs	
#=====================
echo -e "Compiling Pangolin\n"
cd $SCRIPTPATH/Pangolin
mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=$BuildType -DCMAKE_INSTALL_PREFIX=$InstallDir -DBUILD_PANGOLIN_PYTHON=OFF -DDISPLAY_WAYLAND=OFF -DBUILD_TESTS=OFF -DBUILD_EXAMPLES=OFF && make -j $(nproc) && make install && cd .. && rm -r build

#echo -e "Compiling G2O\n"
cd $SCRIPTPATH/g2o
mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=$BuildType -DCMAKE_INSTALL_PREFIX=$InstallDir -DCMAKE_RELWITHDEBINFO_POSTFIX="" -DCMAKE_MINSIZEREL_POSTFIX="" -DG2O_BUILD_APPS=OFF -DG2O_BUILD_EXAMPLES=OFF -DBUILD_WITH_MARCH_NATIVE=ON -DG2O_USE_OPENMP=OFF
make -j $(nproc) && make install && cd .. && rm -r build && rm -r bin && rm -r lib

#echo -e "Compiling DBoW3\n"
cd $SCRIPTPATH/DBow3
mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=$BuildType -DCMAKE_INSTALL_PREFIX=$InstallDir -DBUILD_UTILS=OFF -DCMAKE_CXX_FLAGS=-std=c++11 -DUSE_CONTRIB=true -DOpenCV_DIR=$InstallDir/share/OpenCV && make -j $(nproc) && make install && cd .. && rm -r build

#set environment settings
#==========================
if grep -Fxq 'PATH=${PATH}'":${InstallDir}/bin" ~/.bashrc 
then :
else
  echo 'PATH=${PATH}'":${InstallDir}/bin:$InstallDir/share/OpenCV" >> ~/.bashrc 
  echo 'LD_LIBRARY_PATH=${LD_LIBRARY_PATH}'":${InstallDir}/lib" >> ~/.bashrc
  source ~/.bashrc 
fi


#build SLAM
#==========
#cmake_prefix=$InstallDir/lib/cmake
#cd $SCRIPTPATH && mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=$BuildType && make -j $(nproc)



