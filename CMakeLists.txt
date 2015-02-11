cmake_minimum_required(VERSION 2.8.3)
project(stereo_assignment)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  cmake_modules
  roscpp
)

## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)
find_package(Eigen REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED COMPONENTS system)
find_package(PCL REQUIRED COMPONENTS common io)
find_package(PCL 1.2 REQUIRED)

catkin_package(
   DEPENDS OpenCV Boost Eigen PCL
#  INCLUDE_DIRS include
#  LIBRARIES stereo_assignment
   CATKIN_DEPENDS roscpp
#  DEPENDS system_lib
)

include_directories(${Boost_INCLUDE_DIRS}
                    ${catkin_INCLUDE_DIRS}
                    ${OpenCV_INCLUDE_DIRS}
                    ${Eigen_INCLUDE_DIRS}
                    ${PCL_INCLUDE_DIRS})

## Declare a cpp executable
add_executable(stereo_assignment_main src/main.cpp)
add_executable(stereo_assignment_extra src/extra.cpp)
add_executable(stereo_assignment_alt src/alt.cpp)
target_link_libraries( stereo_assignment_main
    ${catkin_LIBRARIES}
    ${PCL_LIBRARIES}
    ${OpenCV_LIBRARIES}
    ${Boost_LIBRARIES}
    ${OpenCV_LIBS})

target_link_libraries( stereo_assignment_extra
    ${catkin_LIBRARIES}
    ${PCL_LIBRARIES}
    ${OpenCV_LIBRARIES}
    ${Boost_LIBRARIES}
    ${OpenCV_LIBS})
target_link_libraries( stereo_assignment_alt
    ${catkin_LIBRARIES}
    ${PCL_LIBRARIES}
    ${OpenCV_LIBRARIES}
    ${Boost_LIBRARIES}
    ${OpenCV_LIBS})

