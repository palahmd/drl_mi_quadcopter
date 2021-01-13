//
// Created by pala on 08.11.20.
//
#ifndef QUADCOPTER_PIDCONTROLLER_HPP
#define QUADCOPTER_PIDCONTROLLER_HPP

#include "Eigen/Dense"

class pidController {
public:
    pidController(double P, double I, double D);

<<<<<<< HEAD
    pidController() {};
    ~pidController() {};

=======
>>>>>>> master
    void smallAnglesControl();

    void setTargetPoint(double x, double y, double z);

    Eigen::VectorXd targetPoint;
private:
    Eigen::Vector3d eulerAngles, angVel_Body, desAcc;
    Eigen::VectorXd currState; //pos, eulerAngles, linVel, angVel
    Eigen::VectorXd errState; //pos, eulerAngles, linVel, angVel
    Eigen::VectorXd desState; //pos, eulerAngles, linVel, angVel
    Eigen::Vector4d u, controlThrusts;
    double pGain, dGain, iGain;
};

#endif //QUADCOPTER_PIDCONTROLLER_HPP
