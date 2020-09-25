cmake_minimum_required(VERSION 3.10)
project(Quadcopter LANGUAGES CXX)

# Add dependencies

find_package(raisim CONFIG REQUIRED)
find_package(Eigen3 REQUIRED)
include_directories (${EIGEN3_INCLUDE_DIRS})

# make sure activation.raisim file and all models are in here
file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/rsc" DESTINATION ${CMAKE_BINARY_DIR})

add_executable(spawn_quadcopter src/spawnQuadcopter.cpp)
target_link_libraries(spawn_quadcopter PUBLIC raisim::raisim pthread)
target_include_directories(spawn_quadcopter PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
