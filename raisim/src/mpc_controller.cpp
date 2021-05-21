/*
 * @Author: Wei Luo
 * @Date: 2021-05-21 12:19:49
 * @LastEditors: Wei Luo
 * @LastEditTime: 2021-05-21 14:41:40
 * @Note: Note
 */
#include "mpc_controller.hpp"

MPCAcadosController::MPCAcadosController()
{
    /* init ACADOS */
    acados_ocp_capsule = quadrotor_q_acados_create_capsule();
    acados_status = quadrotor_q_acados_create(acados_ocp_capsule);
    if (acados_status)
    {
        std::cout << "Cannot create the ACADOS solver!!!" << std::endl;
    }
    nlp_config = quadrotor_q_acados_get_nlp_config(acados_ocp_capsule);
    nlp_dims = quadrotor_q_acados_get_nlp_dims(acados_ocp_capsule);
    nlp_in = quadrotor_q_acados_get_nlp_in(acados_ocp_capsule);
    nlp_out = quadrotor_q_acados_get_nlp_out(acados_ocp_capsule);

    // get model information
    time_horizon = nlp_dims->N;
    num_states = *nlp_dims->nx;
    num_controls = *nlp_dims->nu;
}

MPCAcadosController::~MPCAcadosController()
{
}

void MPCAcadosController::solvingACADOS(Eigen::VectorXd current_state, Eigen::MatrixXd ref)
{
    if (sizeof(robot_current_state) != current_state.size())
    {
        std::cout << "current state size error" << std::endl;
    }
    for (int i=0; i<current_state.size(); i++)
    {
        robot_current_state[i] = current_state(i);
    }

    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, 0, "lbx", robot_current_state);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, 0, "ubx", robot_current_state);

    std::vector<double> end_term_ref;
    for (int i = 0; i < 3; i++)
    {
        end_term_ref.push_back(ref(time_horizon, i));
    }
    for (int i = 7; i < num_states; i++)
    {
        end_term_ref.push_back(ref(time_horizon, i));
    }
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, time_horizon, "yref", &end_term_ref[0]);

    for (int i = 0; i < time_horizon; i++)
    {
        std::vector<double> y_ref;
        for (int j = 0; j < num_controls + num_states; j++)
        {
            y_ref.push_back(ref(i, j));
        }
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "yref", &y_ref[0]);
    }

    // solve MPC
    acados_status = quadrotor_q_acados_solve(acados_ocp_capsule);
    if (acados_status)
    {
        robot_command[0] = 0.0;
        robot_command[1] = 0.0;
        robot_command[2] = 0.0;
        robot_command[3] = 9.8066;
    }
    else
        ocp_nlp_out_get(nlp_config, nlp_dims, nlp_out, 0, "u", &robot_command);
}

void MPCAcadosController::setTargetPoint(double x, double y, double z)
{
    targetPoint.setZero(12);
    targetPoint.head(3) << x, y, z;
}

Eigen::Vector3d MPCAcadosController::ToEulerAngles(Eigen::VectorXd q) {
    // quaternions inside observation vector start from ob_q[0]
    // ob_q[0] = w, ob_q[1] = x, ob_q[2] = y, ob_q[3] = z
    Eigen::Vector3d angles;

    // roll (x-axis rotation)
    double sinr_cosp = 2 * (q[0] * q[1] + q[2] * q[3]);
    double cosr_cosp = 1 - 2 * (q[1] * q[1] + q[2] * q[2]);
    angles[0] = std::atan2(sinr_cosp, cosr_cosp);

    // pitch (y-axis rotation)
    double sinp = 2 * (q[0] * q[2] - q[3] * q[1]);
    if (std::abs(sinp) >= 1)
        angles[1] = std::copysign(M_PI / 2, sinp); // use 90 degrees if out of range
    else
        angles[1] = std::asin(sinp);


    // yaw (z-axis rotation)
    double siny_cosp = 2 * (q[0] * q[3] + q[1] * q[2]);
    double cosy_cosp = 1 - 2 * (q[2] * q[2] + q[3] * q[3]);
    angles[2] = std::atan2(siny_cosp, cosy_cosp);

    return angles;
}