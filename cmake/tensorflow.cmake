cmake_minimum_required(VERSION 3.7)

project(tensorflow)

find_package(Git)
if (NOT GIT_FOUND)
	message(FATAL_ERROR "Requires Git")
endif()
message(STATUS "Cloning Tensorflow Repo")

set(TENSORFLOW_DOWNLOAD_ROOT ${CMAKE_CURRENT_BINARY_DIR})
execute_process(
	COMMAND "${GIT_EXECUTABLE} clone --recursive --depth 1 https://github.com/tensorflow/tensorflow.git ${TENSORFLOW_DOWNLOAD_ROOT}"
	RESULT_VARIABLE error_code
	OUTPUT_QUIET ERROR_QUIET)
if(error_code)
	message(FATAL_ERROR "CPM failed to get the hash for HEAD")
endif()

execute_process(COMMAND ./configure DIRECTORY ${TENSORFLOW_DOWNLOAD_ROOT})
execute_process(COMMAND bazel build --config=monolithic --config=v2 --config=noaws --config=nogcp --config=nohdfs --config=nonccl //tensorflow:tensorflow_cc DIRECTORY ${TENSORFLOW_DOWNLOAD_ROOT})

include_directories(${TENSORFLOW_DOWNLOAD_ROOT}/tensorflow-src/tensorflow)
