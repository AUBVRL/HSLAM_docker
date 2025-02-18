SET(PROJECT_NAME HSLAM)
CMAKE_MINIMUM_REQUIRED(VERSION 2.9)

set(OpenCV_DIR ${CMAKE_CURRENT_SOURCE_DIR}/Thirdparty/CompiledLibs/share/OpenCV)
set(CMAKE_PREFIX_PATH ${CMAKE_CURRENT_SOURCE_DIR}/Thirdparty/CompiledLibs/lib/cmake)
# Ensure these paths are set correctly before calling project()
message ("OPENCV Directory: ${OpenCV_DIR}")
message("The CMAKE prefix path is: ${CMAKE_PREFIX_PATH}")

PROJECT(${PROJECT_NAME})
#set(CMAKE_VERBOSE_MAKEFILE ON)


#set(BUILD_TYPE Release)
set(BUILD_TYPE RelWithDebInfo)

set(EXECUTABLE_OUTPUT_PATH bin)
set(LIBRARY_OUTPUT_PATH lib)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# required libraries
find_package(SuiteParse REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Boost COMPONENTS system thread) 
find_package(OpenCV 3.4.6 REQUIRED)
#find_package(OpenCV 3.4.6 REQUIRED PATHS /catkin_ws/src/HSLAM/Thirdparty/CompiledLibs NO_DEFAULT_PATH)
find_package(G2O REQUIRED)
find_package(DBoW3 REQUIRED) #PATHS  ~/catkin_ws/src/HSLAM/Thirdparty/CompiledLibs)

# optional libraries
find_package(LibZip QUIET)
find_package(Pangolin 0.2 QUIET)# PATHS  ~/catkin_ws/src/HSLAM/Thirdparty/CompiledLibs)

#pcl libraries
find_package(PCL 1.8 REQUIRED)



# flags
add_definitions("-DENABLE_SSE")
#detect and number of processors
include(ProcessorCount)
ProcessorCount(NUMTHREADS)
add_definitions("-DThreadCount=${NUMTHREADS}")

set(CMAKE_GENERATOR "Unix Makefiles")
set(CMAKE_CXX_FLAGS
   "${SSE_FLAGS} -O3 -g -std=c++0x -march=native -Wno-deprecated -Wno-deprecated-declarations" #-fopenmp is added if g2o is compiled with openmp capabilities
#   "${SSE_FLAGS} -O3 -g -std=c++0x -fno-omit-frame-pointer"
)
#SET CXX FLAGS to use march antive and gcc
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -std=gnu++0x")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -march=native -std=gnu++0x")
set(CMAKE_CXX_FLAGS_MINSIZEREL "${CMAKE_CXX_FLAGS_MINSIZEREL} -march=native -std=gnu++0x")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native -std=gnu++0x")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -march=native -std=gnu++0x")

if (MSVC)
     set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")
endif (MSVC)

# Sources files
set(SOURCE_FILES
  ${PROJECT_SOURCE_DIR}/src/FullSystem/FullSystem.cpp
  ${PROJECT_SOURCE_DIR}/src/FullSystem/FullSystemOptimize.cpp
  ${PROJECT_SOURCE_DIR}/src/FullSystem/FullSystemOptPoint.cpp
  ${PROJECT_SOURCE_DIR}/src/FullSystem/FullSystemDebugStuff.cpp
  ${PROJECT_SOURCE_DIR}/src/FullSystem/FullSystemMarginalize.cpp
  ${PROJECT_SOURCE_DIR}/src/FullSystem/Residuals.cpp
  ${PROJECT_SOURCE_DIR}/src/FullSystem/CoarseTracker.cpp
  ${PROJECT_SOURCE_DIR}/src/FullSystem/CoarseInitializer.cpp
  ${PROJECT_SOURCE_DIR}/src/FullSystem/ImmaturePoint.cpp
  ${PROJECT_SOURCE_DIR}/src/FullSystem/HessianBlocks.cpp
  ${PROJECT_SOURCE_DIR}/src/FullSystem/PixelSelector2.cpp
  ${PROJECT_SOURCE_DIR}/src/OptimizationBackend/EnergyFunctional.cpp
  ${PROJECT_SOURCE_DIR}/src/OptimizationBackend/AccumulatedTopHessian.cpp
  ${PROJECT_SOURCE_DIR}/src/OptimizationBackend/AccumulatedSCHessian.cpp
  ${PROJECT_SOURCE_DIR}/src/OptimizationBackend/EnergyFunctionalStructs.cpp
  ${PROJECT_SOURCE_DIR}/src/util/settings.cpp
  ${PROJECT_SOURCE_DIR}/src/util/Undistort.cpp
  ${PROJECT_SOURCE_DIR}/src/util/globalCalib.cpp
  ${PROJECT_SOURCE_DIR}/src/Indirect/Frame.cpp
  ${PROJECT_SOURCE_DIR}/src/Indirect/Detector.cpp
  ${PROJECT_SOURCE_DIR}/src/Indirect/MapPoint.cpp
  ${PROJECT_SOURCE_DIR}/src/Indirect/Matcher.cpp
  ${PROJECT_SOURCE_DIR}/src/Indirect/Map.cpp
  ${PROJECT_SOURCE_DIR}/src/Indirect/IndirectTracker.cpp
  ${PROJECT_SOURCE_DIR}/src/Indirect/Optimizer.cpp
  ${PROJECT_SOURCE_DIR}/src/Indirect/Sim3Solver.cpp
  ${PROJECT_SOURCE_DIR}/src/Indirect/LoopCloser.cpp
)



include_directories(
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/Thirdparty/Sophus
  ${PROJECT_SOURCE_DIR}/Thirdparty/sse2neon
  ${EIGEN3_INCLUDE_DIR}
  ${G2O_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
) 


# decide if we have pangolin
if (Pangolin_FOUND)
	message("--- found PANGOLIN, compiling pangolin library.")
	include_directories( ${Pangolin_INCLUDE_DIRS} ) 
	set(pangolin_SOURCE_FILES 
	  ${PROJECT_SOURCE_DIR}/src/IOWrapper/Pangolin/KeyFrameDisplay.cpp
	  ${PROJECT_SOURCE_DIR}/src/IOWrapper/Pangolin/PangolinDSOViewer.cpp)
	set(HAS_PANGOLIN 1)
else ()
	message("--- could not find PANGOLIN, not compiling pangolin library.")
	message("    this means there will be no 3D display / GUI available.")
	set(pangolin_SOURCE_FILES )
	set(HAS_PANGOLIN 0)
endif ()

# decide if we have openCV
if (OpenCV_FOUND)
	message("--- found OpenCV, compiling with opencv library.")
	include_directories( ${OpenCV_INCLUDE_DIRS} )
	set(opencv_SOURCE_FILES 
	  ${PROJECT_SOURCE_DIR}/src/IOWrapper/OpenCV/ImageDisplay_OpenCV.cpp
	  ${PROJECT_SOURCE_DIR}/src/IOWrapper/OpenCV/ImageRW_OpenCV.cpp)
	set(HAS_OPENCV 1)
else ()
	message("--- could not find OpenCV, not compiling opencv library.")
	message("    this means there will be no image display, and image read / load functionality.")
	set(opencv_SOURCE_FILES 
	  ${PROJECT_SOURCE_DIR}/src/IOWrapper/ImageDisplay_dummy.cpp
	  ${PROJECT_SOURCE_DIR}/src/IOWrapper/ImageRW_dummy.cpp)
	set(HAS_OPENCV 0)
endif ()

# decide if we have ziplib.
if (LIBZIP_LIBRARY)
	message("--- found ziplib (${LIBZIP_VERSION}), compiling with zip capability.")
	add_definitions(-DHAS_ZIPLIB=1)
	include_directories( ${LIBZIP_INCLUDE_DIR_ZIP} ${LIBZIP_INCLUDE_DIR_ZIPCONF} ) 
else()
	message("--- not found ziplib (${LIBZIP_LIBRARY}), compiling without zip capability.")
	set(LIBZIP_LIBRARY "")
endif()


# compile main library.
include_directories( ${CSPARSE_INCLUDE_DIR} ${CHOLMOD_INCLUDE_DIR}) 
add_library(HSLAM_lib ${SOURCE_FILES} ${opencv_SOURCE_FILES} ${pangolin_SOURCE_FILES})



if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin") # OSX
    set(BOOST_THREAD_LIBRARY boost_thread-mt)
else()
    set(BOOST_THREAD_LIBRARY boost_thread)
endif()


set(LIBS
  ${OpenCV_LIBS}
  ${EIGEN3_LIBS}
  ${G2O_LIBS}
  ${LIBZIP_LIBRARY} 
  ${Pangolin_LIBRARIES}
  ${DBoW3_LIB_DIR}/libDBoW3.so
  ${BOOST_THREAD_LIBRARY}
  ${PCL_LIBRARIES}
  boost_system
  cxsparse
  cholmod
)

# build main executable (only if we have both OpenCV and Pangolin)
if (OpenCV_FOUND AND Pangolin_FOUND)
	message("--- compiling executable.")
	add_executable(HSLAM ${PROJECT_SOURCE_DIR}/src/main.cpp )
    target_link_libraries(HSLAM HSLAM_lib ${LIBS} )
else()
	message("--- not building executable, since either don't have openCV or Pangolin.")
endif()

