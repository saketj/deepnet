# Set the required version of CMake
cmake_minimum_required(VERSION 2.8.11)
# Set your project title
project(DeepNet)

#The following will but all binary files in a "bin" directory
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

#Include macros from the ParallelUtils library
include(ParallelUtils.cmake)

enable_cuda_support()
add_executable(deepnet main.cpp data_loader.cpp utils.cpp neural_network.cpp)
cuda_add_executable(deepnet-cuda main.cpp data_loader.cpp utils.cu neural_network.cu backpropagation.cu)
# set_target_properties(deepnet PROPERTIES COMPILE_FLAGS "-O4")
