cmake_minimum_required(VERSION 2.8)
project( SLAM )
# set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/)
find_package( OpenCV REQUIRED )
add_executable( SLAM SLAM.cpp )
target_link_libraries( SLAM ${OpenCV_LIBS} )