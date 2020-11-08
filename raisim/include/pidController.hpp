//
// Created by pala on 08.11.20.
//
#ifndef QUADCOPTER_PIDCONTROLLER_HPP
#define QUADCOPTER_PIDCONTROLLER_HPP
#include "Eigen/Dense"

class pidController{
public:

    static void calculateThrusts(const Eigen::VectorXd& desiredPos);

private:
//    static Eigen::Vector3d eulerAngles;
//    static Eigen::VectorXd currState;
//    static Eigen::VectorXd errState;
//    static Eigen::VectorXd errPos;
//    static Eigen::Vector3d desiredAcc, desiredVel, desiredAng;
//    static Eigen::Vector4d controlThrusts;
//
//    static int pGain, dGain, iGain;
};

#endif //QUADCOPTER_PIDCONTROLLER_HPP
