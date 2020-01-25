cmake_minimum_required (VERSION 3.9)

if (APPLE)
    set(CMAKE_MACOSX_RPATH ON)
endif ()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

## These two things are needed for YCM
SET( CMAKE_EXPORT_COMPILE_COMMANDS ON )
IF( EXISTS "${CMAKE_CURRENT_BINARY_DIR}/compile_commands.json" )
  EXECUTE_PROCESS( COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_BINARY_DIR}/compile_commands.json
    ${CMAKE_CURRENT_SOURCE_DIR}/compile_commands.json
  )
ENDIF()

# add_subdirectory(abseil-cpp)
#include(cmake/tensorflow.cmake)

if (APPLE)
	find_package (Protobuf REQUIRED)
else()
	find_package (Protobuf REQUIRED)
	find_package(PkgConfig REQUIRED)
endif()

find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package(absl REQUIRED)
find_library(TENSORFLOW tensorflow_cc REQUIRED)
message("Tensorflow: " ${TENSORFLOW})

add_library(deepcpp include/deepcpp.hpp lib/model.cpp lib/batch.cpp)
target_include_directories(deepcpp PUBLIC include)
target_include_directories(deepcpp PUBLIC lib)

target_include_directories(deepcpp PRIVATE ~/standalone/include)
SET_TARGET_PROPERTIES(deepcpp PROPERTIES LINKER_LANGUAGE CXX)

target_link_libraries(deepcpp absl::strings ${TENSORFLOW} Eigen3::Eigen)
