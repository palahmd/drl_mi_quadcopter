/*
 * @Author: Wei Luo
 * @Date: 2021-05-21 12:20:01
 * @LastEditors: Wei Luo
 * @LastEditTime: 2021-05-21 12:31:04
 * @Note: Note
 */

#ifndef _MPC_CONTROLLER_HPP_
#define _MPC_CONTROLLER_HPP_

#include <chrono>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
// Eigen
#include <Eigen/Dense>

// acados
#include <acados/utils/math.h>
#include <acados_c/ocp_nlp_interface.h>
#include <acados_sim_solver_quadrotor_q.h>
#include <acados_solver_quadrotor_q.h>


class MPCAcadosController
{
public:
    MPCAcadosController();
    ~MPCAcadosController();
    void solvingACADOS(Eigen::VectorXd current_state, Eigen::MatrixXd ref);
    void setTargetPoint(double x, double y, double z);

    Eigen::VectorXd targetPoint;
    double robot_command[4];

private:
    Eigen::Vector3d ToEulerAngles(Eigen::VectorXd q);

    Eigen::Vector3d eulerAngles, angVel_Body, desAcc;
    /* ACADOS */
    nlp_solver_capsule *acados_ocp_capsule;
    int acados_status;
    ocp_nlp_config *nlp_config;
    ocp_nlp_dims *nlp_dims;
    ocp_nlp_in *nlp_in;
    ocp_nlp_out *nlp_out;
    int time_horizon;
    int num_states;
    int num_controls;
    double robot_current_state[10];
};
#endif /* _MPC_CONTROLLER_HPP_ */
