#ifndef QUADCOPTER_PID_CONTROLLER_HPP
#define QUADCOPTER_PID_CONTROLLER_HPP
#include "Eigen/Dense"
#include "raisim/World.hpp"
#include "quadcopter_initializer.hpp"


    raisim::Vec<3> desiredPos;
    Eigen::Vector4d controlThrusts;
    double errorPos;

void calculateThrusts(){
    Eigen::Vector3d hilf1 = desiredPos.e() - pos.e();
    Eigen::Vector3d hilf2 = {1, 1, 1};
    errorPos = hilf1.transpose()*hilf2;

    controlThrusts = 0.1*errorPos*levitationThrusts;
    thrusts = levitationThrusts + controlThrusts;
}


#endif //QUADCOPTER_PID_CONTROLLER_HPP
