//
// Created by pala on 02.11.20.
//
// This header initializes all dynamics and model properties and provides basic functions

#ifndef QUADCOPTER_INIT_HPP
#define QUADCOPTER_INIT_HPP
#include "Eigen/Dense"
#include "raisim/World.hpp"

    raisim::ArticulatedSystem* robot;

    // generalized coordinates, generalized velocities and state
    Eigen::VectorXd gc_init, gv_init, gc, gv;
    int gvDim, gcDim, nRotors;
    raisim::Mat<3,3> rot;
    raisim::Vec<3> pos, linVel_W, angVel_W;
    raisim::Vec<4> rot_q;
    Eigen::VectorXd ob, ob_q;

    // thrusts and forces
    Eigen::Vector4d thrusts;
    Eigen::Matrix4d thrusts2TorquesAndForces;
    Eigen::Vector4d torquesAndForces;
    Eigen::Vector3d torques_baseFrame, forces_baseFrame;
    raisim::Vec<3> torques_worldFrame, forces_worldFrame;
    Eigen::VectorXd genForces;


    // quadcopter model parameters
    const double rotorPos = 0.17104913036744201, momentConst = 0.016;
    const double rps = 2 * M_PI, rpm = rps/60;
    const double g = 9.81, m = 1.727;
    const double max_thrust_i = 8;
    const Eigen::Vector4d hoverThrusts = {m * g / 4, m * g / 4, m * g / 4, m * g / 4};


/// Basic Functions
void updateState(){
    robot->getBasePosition(pos);
    robot->getBaseOrientation(rot);
    robot->getBaseOrientation(rot_q);
    robot->getVelocity(0, linVel_W);
    robot->getAngularVelocity(0, angVel_W);

    ///
    for(size_t i=0; i<9; i++)
        ob[i] = rot[i];

    for(size_t i=0; i<4; i++)
        ob_q[i] = rot_q[i];

    /// target position. the target position is always 0,0,0 in the world frame but not in the hummingbird frame
    for(size_t i=0; i<3; i++) {
        ob[i + 9] = pos[i];
        ob_q[i + 4] = pos[i];
    }

    /// body velocities
    for(size_t i=0; i<3; i++) {
        ob[i + 12] = linVel_W[i];
        ob_q[i+7] = linVel_W[i];
    }

    for(size_t i=0; i<3; i++) {
        ob[i + 15] = angVel_W[i];
        ob_q[i + 10] = angVel_W[i];
    }
}

void applyThrusts(){
    torquesAndForces = thrusts2TorquesAndForces * thrusts;
    torques_baseFrame = torquesAndForces.segment(0, 3);
    forces_baseFrame << 0.0, 0.0, torquesAndForces(3);

    torques_worldFrame.e() = rot.e() * torques_baseFrame;
    forces_worldFrame.e() = rot.e() * forces_baseFrame;
    genForces.head(6) << forces_worldFrame.e(), torques_worldFrame.e();

    // robot->setGeneralizedForce(genForces);

/// also possible to put an external force, to vis. forces and torque
    robot->setExternalForce(0, forces_worldFrame);
    robot->setExternalTorque(0, torques_worldFrame);
}

#endif //QUADCOPTER_INIT_HPP