## CMake File for compiling raisim scripts

cmake_minimum_required(VERSION 3.10)
project(Quadcopter LANGUAGES C CXX)

# for macOS
set(CMAKE_MACOSX_RPATH 1)

# set compiler standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99 -DEXT_DEP -fdiagnostics-show-option")

# Add dependencies
# raisim setup
list(APPEND CMAKE_PREFIX_PATH "/home/arnold/Develop/raisimLib/build")
find_package(raisim CONFIG REQUIRED)
find_package(Eigen3 REQUIRED)
include_directories (${EIGEN3_INCLUDE_DIRS})
# make sure activation.raisim file and all models are in here
file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/../rsc" DESTINATION ${CMAKE_BINARY_DIR})

# acados
# include some more paths
include_directories($ENV{ACADOS_SOURCE_DIR}/include)
include_directories($ENV{ACADOS_SOURCE_DIR}/include/blasfeo/include)
include_directories($ENV{ACADOS_SOURCE_DIR}/include/hpipm/include)
include_directories($ENV{ACADOS_SOURCE_DIR}/include/acados)
include_directories(./acados_code)
include_directories(./include)

# load acados libs
link_directories($ENV{ACADOS_SOURCE_DIR}/lib)

file(GLOB ocp_solver
        ./acados_code/acados_solver_quadrotor_q.c
        )

file(GLOB casadi_fun
        ./acados_code/quadrotor_q_model/quadrotor_q_expl_ode_fun.c
        ./acados_code/quadrotor_q_model/quadrotor_q_expl_vde_forw.c)

file(GLOB sim_solver
        ./acados_code/acados_sim_solver_quadrotor_q.c
        )

add_library(ocp_shared_lib SHARED ${ocp_solver} ${casadi_fun} )
target_link_libraries(ocp_shared_lib acados hpipm blasfeo)

add_library(sim_shared_lib SHARED ${sim_solver} ${casadi_fun})
target_link_libraries(sim_shared_lib acados hpipm blasfeo)


add_executable(pid_control_quadcopter src/quadcopter_pid_control.cpp)
target_link_libraries(pid_control_quadcopter PUBLIC raisim::raisim pthread )
# target_include_directories(pid_control_quadcopter PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# add_executable(gyroscope_effect src/gyroscopic_effect.cpp)
# target_link_libraries(gyroscope_effect PUBLIC raisim::raisim pthread)
# target_include_directories(gyroscope_effect PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(mpc_control_quadrotor src/quadcopter_mpc_q_control.cpp src/mpc_controller.cpp)
target_link_libraries(mpc_control_quadrotor PUBLIC raisim::raisim pthread ocp_shared_lib sim_shared_lib)
