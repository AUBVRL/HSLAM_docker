# H-SLAM (Hybrid Simultaneous Localization and Mapping).

A containerized "ready-to-use" SLAM application that leverages both direct and indirect methods.

### Related Publications:
[A Unified Hybrid Formulation for Visual SLAM](https://scholarworks.aub.edu.lb/bitstream/handle/10938/22253/YounesGeorges_2021.pdf?sequence=5) (Doctoral dissertation), Younes, G. (2021).

Please cite the paper if used in an academic context.


## Table of Contents

- [Project Description](#project-description)
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Contributing](#contributing)
- [License](#license)

## Project Description

The H-SLAM project is an implementation of a visual simultaneous localization and mapping algorithm. 
It utilizes the ROS (Robot Operating System) Noetic and is designed for Ubuntu 20.04.



## Installation

To use this Dockerfile and build the H-SLAM project, follow these steps:

1. Install Docker on your machine.
2. Create a new directory and navigate to it in the terminal.
3. Clone this repository into the previously created directory.
4. Open a terminal and navigate to the directory containing the Dockerfile and project files.
```bash
    cd ~/<YOUR_PATH>/fslam_ros_docker
```
6. Run the following command to build the Docker image:

```bash
    docker build -t fslam .
```
The build will start using the Dockerfile structure, which is divided into multiple stages. 
In the first stage, it sets up the base image with Ubuntu 20.04 and ROS Noetic using OSRF official noetic image. It also installs the necessary dependencies for catkin tools.
In the second stage, it installs additional dependencies and the Realsense SDK. These dependencies include various libraries and tools required for camera support, graphics, system functionality, and GUI elements. Additionally, it downloads and extracts third-party libraries, including Ceres Solver and OpenCV with OpenCV Contrib.
The project files are then copied into the container, and the Thirdparty libraries are built using the provided `build.sh` script. After that, the H-SLAM project is built using CMake, and the `fslam_ros` wrapper is copied and built using catkin.
Finally, the container's entry point is specified, and calibration files are copied into the project directory.

## Usage

To run the H-SLAM project you will need to run two containers of the same image.
One to to publish images from the camera and the other to run the H-SLAM main application.

0. Allow access to containers:
``` bash
    xhost +
```

1. Open two terminals and execute the following command in each:
``` bash
    docker run -it --net=host --privileged -e DISPLAY=unix$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:rw --device /dev/video0:/dev/video0  fslam /bin/bash
```
This command starts the container and provides an interactive terminal within it.

2. In the first terminal run the following command to open a camera stream through ROS and publish the camera's images unto a ROS topic:
``` bash
    roslaunch usb_cam usb_cam-test.launch
```
A display window will pop with the camera's stream.

3. In the second terminal, execute this command to run the H-SLAM algorithm on the image stream.
``` bash
    rosrun fslam_ros fslam_live image:=/usb_cam/image_raw calib=/catkin_ws/src/res/camera.txt gamma=/catkin_ws/src/res/pcalib.txt vignette=/catkin_ws/src/res/vignette.png
```
Start moving the camera/computer around and perform SLAM.

## Features

- Utilizes the H-SLAM algorithm for simultaneous localization and mapping.
- Integrates with ROS Noetic and utilizes various ROS functionalities.
- Supports camera integration, including Realsense cameras.
- Provides a wrapper for ROS integration and additional functionality.

## Contributing

Contributions to the H-SLAM project are welcome. If you would like to contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make the necessary changes and commit them.
4. Push your changes to your forked repository.
5. Submit a pull request detailing the changes you have made.

## License
This project is licensed under the [MIT License](LICENSE).
