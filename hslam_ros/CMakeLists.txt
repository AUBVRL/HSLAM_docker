cmake_minimum_required(VERSION 2.4.6)
project(hslam_ros)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  roscpp
  sensor_msgs
  cv_bridge
  pcl_conversions

)

#SET PATH TO HSLAM

set(HSLAM_PATH /catkin_ws/src/HSLAM)
#set(HSLAM_LIBRARY_PATH ~/catkin_ws/src/HSLAM/lib)
set(HSLAM_LIBRARY_PATH /catkin_ws/src/HSLAM/build/lib)

# set values from environment variables
if (NOT HSLAM_PATH)
    set(HSLAM_PATH $ENV{HSLAM_PATH})
endif()
if (NOT HSLAM_LIBRARY_PATH)
    set(HSLAM_LIBRARY_PATH $ENV{HSLAM_LIBRARY_PATH})
    if (NOT HSLAM_LIBRARY_PATH AND HSLAM_PATH)
        set(HSLAM_LIBRARY_PATH "${HSLAM_PATH}/build/lib")
    endif()
endif()

if (HSLAM_PATH)
    # Manually set path to HSLAM source folder
    message("HSLAM_PATH set, trying to find library.")
    message("---- EXPECTING HSLAM sources at\n\"${HSLAM_PATH}\"")
    set(HSLAM_INCLUDE_DIRS "${HSLAM_PATH}/src" "${HSLAM_PATH}/Thirdparty/Sophus" "${HSLAM_PATH}/Thirdparty/g2o" "${HSLAM_PATH}/Thirdparty/DBow3" "${HSLAM_PATH}/Thirdparty/opencv-3.4.6")
    message("---- LOOKING FOR HSLAM library at\n\"${HSLAM_LIBRARY_PATH}\"")
    find_library(HSLAM_LIBRARY HSLAM_lib ${HSLAM_LIBRARY_PATH})
else()
    # Detect HSLAM install (also catkin devel folder)
    message("HSLAM_PATH not set yet, trying to find installed hslam headers and library.")
    find_path(HSLAM_INCLUDE_DIRS HSLAM)
    if (HSLAM_INCLUDE_DIRS)
        set(HSLAM_INCLUDE_DIRS "${HSLAM_INCLUDE_DIRS}/HSLAM")
        message("---- FOUND HSLAM headers at \"${HSLAM_INCLUDE_DIRS}\"")
    endif()
    find_library(HSLAM_LIBRARY hslam)
endif()


if (NOT HSLAM_INCLUDE_DIRS)
    message(FATAL_ERROR "HSLAM headers not found and/or HSLAM_PATH not set.")
endif()
if (NOT HSLAM_LIBRARY)
    message(FATAL_ERROR "HSLAM library not found and/or HSLAM_LIBRARY_PATH not set.")
endif()

message("---- Found HSLAM library at \"${HSLAM_LIBRARY}\"")


set(CMAKE_MODULE_PATH /catkin_ws/src/hslam_ros/cmake)
set(OpenCV_DIR /catkin_ws/src/HSLAM/Thirdparty/CompiledLibs/share/OpenCV)
set(Pangolin_DIR /catkin_ws/src/HSLAM/Thirdparty/CompiledLibs/lib/cmake)
set(DBoW3_DIR /catkin_ws/src/HSLAM/Thirdparty/CompiledLibs/lib/cmake/DBoW3)
set(G2O_DIR /catkin_ws/src/HSLAM/Thirdparty/CompiledLibs/lib/cmake/g2o)


find_package(Pangolin 0.2 REQUIRED PATHS /catkin_ws/src/HSLAM/Thirdparty/CompiledLibs/lib/cmake)
message("\n\n---- FOUND Pangolin Headers at\n\"${Pangolin_INCLUDE_DIRS}\"")
message("---- FOUND Pangolin Libs at\n\"${Pangolin_LIBRARIES}\"")

find_package(OpenCV 3.4.6 REQUIRED PATHS /catkin_ws/src/HSLAM/Thirdparty/CompiledLibs/lib/cmake)
message("\n\n---- FOUND OpenCV Headers at\n\"${OpenCV_INCLUDE_DIRS}\"")
message("---- FOUND OpenCV Libs at\n\"${OpenCV_LIBS}\"")

find_package(DBoW3 REQUIRED)
message("\n\n---- FOUND DBoW3 Headers at\n\"${DBoW3_INCLUDE_DIRS}\"")
message("---- FOUND DBoW3 Libs at\n\"${DBoW3_LIBRARIES}\"")

find_package(SuiteParse REQUIRED)
message("\n\n---- FOUND CSPARSE at\n\"${CSPARSE_INCLUDE_DIR}\"")
message("\n\n---- FOUND CSPARSE LIBS at\n\"${CSPARSE_LIBRARY}\"")
message("\n\n---- FOUND CHOLMOD at\n\"${CHOLMOD_INCLUDE_DIR}\"")
message("\n\n---- FOUND CHOLMOD LIBS at\n\"${CHOLMOD_LIBRARIES}\"")

find_package(Eigen3 REQUIRED)
find_package(Boost COMPONENTS system thread)

#G2O not found troubleshooting
find_package(G2O COMPONENTS)
message("-----G2O troubleshooting-----")
message("The G2O directory is : ${G2O_DIR}" )
message("The G2O include  directory is : ${G2O_INCLUDE_DIR}" )
message("The G2O libraries is : ${G2O_LIBS}" )
#find_package(G2O REQUIRED COMPONENTS PATHS /catkin_ws/src/HSLAM/Thirdparty/CompiledLibs/lib/cmake/g2o)
message ("The following were considered for g2o: ${G2O_CONSIDERED_CONFIGS}")
message("\n\n---- FOUND G2O Headers at\n\"${G2O_INCLUDE_DIR}\"")
message("---- FOUND G2O Libs at\n\"${G2O_LIBS}\"")

catkin_package(
  CATKIN_DEPENDS
    geometry_msgs
    roscpp
    pcl_conversions
    sensor_msgs
    cv_bridge
)


set(LIBS
  ${OpenCV_LIBS}
  ${EIGEN3_LIBS}
  ${G2O_LIBS}
  ${Pangolin_LIBRARIES}
  ${DBoW3_LIB_DIR}/libDBoW3.so
  ${BOOST_THREAD_LIBRARY}
  boost_system
  cxsparse
  cholmod
)
###########
## Build ##
###########

set(CMAKE_CXX_FLAGS "-std=c++14 -march=native"
#   "${SSE_FLAGS} -O2 -g -std=c++0x -march=native -fno-omit-frame-pointer"
)

include_directories(
  ${PROJECT_SOURCE_DIR}/src
  ${HSLAM_INCLUDE_DIRS}
  ${Pangolin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${catkin_INCLUDE_DIRS}
  ${G2O_INCLUDE_DIR}
  ${DBoW3_INCLUDE_DIRS}
  ${CSPARSE_INCLUDE_DIR}
  ${CHOLMOD_INCLUDE_DIR}
  
)

link_directories(
  ${G2O_INCLUDE_DIR}
  ${DBoW3_INCLUDE_DIR}
  ${CSPARSE_INCLUDE_DIR}
  ${CHOLMOD_INCLUDE_DIR}
)
## Declare a C++ executable
add_executable(hslam_live src/main.cpp)


target_link_libraries(hslam_live
    ${HSLAM_LIBRARY}
    ${Pangolin_LIBRARIES}
    ${OpenCV_LIBS}
    ${G2O_LIBS}
    ${DBoW3_LIB_DIR}/libDBoW3.so
    ${catkin_LIBRARIES}
    ${CSPARSE_LIBRARY}
    ${CHOLMOD_LIBRARIES}
    
   )

   ## Declare a C++ executable
add_executable(hslam_live_2 src/main-2.cpp)


target_link_libraries(hslam_live_2
    ${HSLAM_LIBRARY}
    ${Pangolin_LIBRARIES}
    ${OpenCV_LIBS}
    ${G2O_LIBS}
    ${DBoW3_LIB_DIR}/libDBoW3.so
    ${catkin_LIBRARIES}
    ${CSPARSE_LIBRARY}
    ${CHOLMOD_LIBRARIES}
    
   )
