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