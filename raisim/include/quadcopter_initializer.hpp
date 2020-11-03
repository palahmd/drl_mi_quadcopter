#ifndef QUADCOPTER_INITIALIZER_HPP
#define QUADCOPTER_INITIALIZER_HPP
#include "Eigen/Dense"
#include "raisim/World.hpp"

    raisim::ArticulatedSystem* robot;

    // generalized coordinates, generalized velocities and state
    Eigen::VectorXd gc_init, gv_init, gc, gv;
    int gvDim, gcDim, nRotors;
    raisim::Mat<3,3> rot;
    raisim::Vec<3> pos, linVel, angVel;

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
    const Eigen::Vector4d levitationThrusts = {m*g/4, m*g/4, m*g/4, m*g/4};


/// Basic Functions
void updateState(){
    robot->getState(gc, gv);
    robot->getBasePosition(pos);
    robot->getBaseOrientation(rot);
    robot->getVelocity(0, linVel);
    robot->getAngularVelocity(0, angVel);
}

void applyThrusts(){
    torquesAndForces = thrusts2TorquesAndForces * thrusts;
    torques_baseFrame = torquesAndForces.segment(0, 3);
    forces_baseFrame << 0.0, 0.0, torquesAndForces(3);

    torques_worldFrame.e() = rot.e() * torques_baseFrame;
    forces_worldFrame.e() = rot.e() * forces_baseFrame;
    genForces.head(6) << forces_worldFrame.e(), torques_worldFrame.e();

    robot->setGeneralizedForce(genForces);

/// also possible to put an external force, to vis. forces and torque
//    robot->setExternalForce(0, force_WorldFrame);
//    robot->setExternalTorque(0, torque_WorldFrame);
}

#endif //QUADCOPTER_INITIALIZER_HPP