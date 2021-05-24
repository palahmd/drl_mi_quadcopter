## CMake File for compiling raisim scripts

cmake_minimum_required(VERSION 3.10)
project(Quadcopter LANGUAGES CXX)

# Add dependencies

find_package(raisim CONFIG REQUIRED)
find_package(Eigen3 REQUIRED)
include_directories (${EIGEN3_INCLUDE_DIRS})

# make sure activation.raisim file and all models are in here
file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/../rsc" DESTINATION ${CMAKE_BINARY_DIR})

add_executable(pid_control_quadcopter src/quadcopter_pid_control.cpp)
target_link_libraries(pid_control_quadcopter PUBLIC raisim::raisim pthread)
target_include_directories(pid_control_quadcopter PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(gyroscope_effect src/gyroscopic_effect.cpp)
target_link_libraries(gyroscope_effect PUBLIC raisim::raisim pthread)
target_include_directories(gyroscope_effect PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
