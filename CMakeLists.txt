cmake_minimum_required(VERSION 2.8)

project( eye_pos_tracker )

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR})
set(CMAKE_INSTALL_PREFIX "./install") 

message(STATUS "")
message(STATUS "--- Please edit the following paths according to your environment ---" )
set(CMAKE_INCLUDE_PATH ${CMAKE_INCLUDE_PATH} "C:/SDK/boost_1_54_0")
set(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} "C:/SDK/boost_1_54_0/lib")
message(STATUS " CMAKE_INCLUDE_PATH=${CMAKE_INCLUDE_PATH}" )
message(STATUS " CMAKE_LIBRARY_PATH=${CMAKE_LIBRARY_PATH}" )
message(STATUS "---------------------------" )
message(STATUS "")

if (NOT MSVC)
    if (${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
        execute_process(COMMAND ${CMAKE_CXX_COMPILER} -dumpversion OUTPUT_VARIABLE GCC_VERSION)
        if (GCC_VERSION VERSION_LESS 4.7)
            set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
        else ()
            set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
        endif ()
    else ()
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
    endif ()
endif ()

set(Boost_USE_STATIC_LIBS ON)
set(Boost_USE_MULTITHREADED ON)
#set(Boost_USE_STATIC_RUNTIME OFF)

find_package ( Boost REQUIRED )
find_package ( OpenCV REQUIRED )
find_package ( TBB REQUIRED )
find_package ( Eigen3 REQUIRED )

message(STATUS "")
message(STATUS "--- Include directories ---" )
message(STATUS " OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}" )
message(STATUS " TBB_INCLUDE_DIRS: ${TBB_INCLUDE_DIRS}" )
message(STATUS " Eigen3_INCLUDE_DIRS: ${EIGEN3_INCLUDE_DIR}" )
message(STATUS " Boost_INCLUDE_DIRS: ${Boost_INCLUDE_DIRS}" )
message(STATUS "---------------------------" )
message(STATUS "")

include_directories (
  "${PROJECT_SOURCE_DIR}"
  "external/LSD"
  "external/Peeters"
  "external/Swirski"
  ${OpenCV_INCLUDE_DIRS}
  ${TBB_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${Boost_INCLUDE_DIRS}
)

file (GLOB SRCS 
  "*.cpp" 
  "external/LSD/lsd_opencv3.cpp"
  "external/Swirski/PupilTracker.cpp"
  )

file (GLOB HEADERS 
  *.h *.hpp 
  "external/LSD/lsd_opencv3.h"
  "external/Swirski/pupiltracker/PupilTracker.h"
  )

add_executable( main ${SRCS} ${HEADERS} )
target_link_libraries( main 
		       ${OpenCV_LIBS}
		       ${TBB_LIBRARIES}
		       )
