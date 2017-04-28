cmake_minimum_required(VERSION 3.7)
project(Stereo_Matcher)
find_package( OpenCV REQUIRED )
set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES main.cpp StereoMatcher.cpp StereoMatcher.h)
add_executable(StereoMatcher ${SOURCE_FILES})
target_link_libraries( StereoMatcher ${OpenCV_LIBS} )