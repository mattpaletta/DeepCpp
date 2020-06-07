cmake_minimum_required (VERSION 3.9)

if (APPLE)
    set(CMAKE_MACOSX_RPATH ON)
endif ()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

## These two things are needed for YCM
SET( CMAKE_EXPORT_COMPILE_COMMANDS ON )
IF( EXISTS "${CMAKE_CURRENT_BINARY_DIR}/compile_commands.json" )
  EXECUTE_PROCESS( COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_BINARY_DIR}/compile_commands.json
    ${CMAKE_CURRENT_SOURCE_DIR}/compile_commands.json
  )
ENDIF()

add_subdirectory(tensorflow-cmake)

file(GLOB SRC_FILES lib/*.cpp)
file(GLOB INC_FILES include/*.hpp)
add_library(deepcpp ${SRC_FILES} ${INC_FILES})
target_include_directories(deepcpp PUBLIC include)
target_link_libraries(deepcpp tensorflow)
