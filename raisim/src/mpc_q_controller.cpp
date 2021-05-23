/*
 * @Author: Wei Luo
 * @Date: 2021-05-21 12:19:49
 * @LastEditors: Wei Luo
 * @LastEditTime: 2021-05-21 14:41:40
 * @Note: Note
 */
#include "mpc_q_controller.hpp"

MPCAcadosController::MPCAcadosController(double m, double dt): timeStep(dt), mass(m)
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

    std::cout << time_horizon << num_states << num_controls << std::endl;

    trajectory_reference = Eigen::MatrixXd::Zero(time_horizon + 1, num_states + num_controls);
    currentState.setZero(10);

    hoverThrust = mass * 9.81 / 4;
}

MPCAcadosController::~MPCAcadosController()
{
}


void MPCAcadosController::solvingACADOS(Eigen::Matrix4d rot, Eigen::Vector4d& thrusts) //Eigen::VectorXd current_state, Eigen::MatrixXd ref
{
    if (sizeof(robot_current_state_q) / sizeof(double) != currentState.size())
    {
        std::cout << "current state size error" << std::endl;
    }
    for (int i = 0; i < currentState.size(); i++)
    {
        robot_current_state_q[i] = currentState(i);
    }

    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, 0, "lbx", robot_current_state_q);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, 0, "ubx", robot_current_state_q);

    std::vector<double> end_term_ref;
    for (int i = 0; i < 3; i++)
    {
        end_term_ref.push_back(trajectory_reference(time_horizon, i));

    }
    for (int i = 7; i < num_states; i++)
    {
        end_term_ref.push_back(trajectory_reference(time_horizon, i));

    }
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, time_horizon, "yref", &end_term_ref[0]);

    for (int i = 0; i < time_horizon; i++)
    {
        std::vector<double> y_ref;
        for (int j = 0; j < num_controls + num_states; j++)
        {
            y_ref.push_back(trajectory_reference(i, j));
        }
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "yref", &y_ref[0]);
    }

    // solve MPC
    acados_status = quadrotor_q_acados_solve(acados_ocp_capsule);

    if (acados_status)
    {
        robot_rate_command[0] = 0.0;
        robot_rate_command[1] = 0.0;
        robot_rate_command[2] = 0.0;
        robot_rate_command[3] = 9.81;
    }
    else
        ocp_nlp_out_get(nlp_config, nlp_dims, nlp_out, 0, "u", &robot_rate_command);

    Eigen::Vector3d rpy;
    rpy = ToEulerAngles(currentState.segment(3, 4));

    std::cout << "rpy = " << rpy << std::endl;
    u[0] = robot_rate_command[3] * 9.81 / 6.;
    u[1] = -robot_rate_command[1] * timeStep + rpy[0];
    u[2] = robot_rate_command[0] * timeStep + rpy[1];
    u[3] = robot_rate_command[2] * timeStep + rpy[2];
    std::cout << u << std::endl;

    /** Motor Model for motor i:
     **         time_constant_up = 0.0125 sec
     **         time_const_down = 0.025 sec
     **         thrust(t) = thrust(t-1) + (controlThrust(t-1) - thrust(t-1)) * timeStep/time_constant **/
    for (int i = 0; i < 4; i++) {
        if (thrusts[i] < controlThrusts[i]) { // time constant for increasing rotor speed
            thrusts[i] = thrusts[i] + (controlThrusts[i] - thrusts[i]) * timeStep / 0.0125;
        } else if (thrusts[i] > controlThrusts[i]) { // time constant for decreasing rotor speed
            thrusts[i] = thrusts[i] + (controlThrusts[i] - thrusts[i]) * timeStep / 0.025;
        }
    }

    controlThrusts = rot.inverse() * u;
    std::cout << "control trust: " << controlThrusts << std::endl;
    /// scale controlThrusts and avoid thrusts out of range: 0.5 - 1.5 * hoverThrust
    double max_scale = controlThrusts.maxCoeff();
    if (max_scale > 1.5 * hoverThrust) {
        controlThrusts = 1.5 / max_scale * hoverThrust * controlThrusts;
    }
    for (int i = 0; i < 4; i++) {
        if (controlThrusts[i] < 0.5 * hoverThrust) {
            controlThrusts[i] = 0.5 * hoverThrust;
        }
    }

}

void MPCAcadosController::setTargetPoint(double x, double y, double z)
{
    targetPoint.setZero(12);
    targetPoint.head(3) << x, y, z;

    for (int i = 0; i < time_horizon + 1; i++)
    {
        trajectory_reference(i, 0) = x;
        trajectory_reference(i, 1) = y;
        trajectory_reference(i, 2) = z;

        trajectory_reference(i, 3) = 1.0;
        trajectory_reference(i, 4) = 0.0;
        trajectory_reference(i, 5) = 0.0;
        trajectory_reference(i, 6) = 0.0;
        trajectory_reference(i, 7) = 0.0;
        trajectory_reference(i, 8) = 0.0;
        trajectory_reference(i, 9) = 0.0;
        // control
        trajectory_reference(i, 10) = 0.0;
        trajectory_reference(i, 11) = 0.0;
        trajectory_reference(i, 12) = 0.0;
        trajectory_reference(i, 13) = 9.8066;
    }
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