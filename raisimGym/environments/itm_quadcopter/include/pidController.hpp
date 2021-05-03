//
// Created by pala on 08.11.20.
//
#ifndef QUADCOPTER_PIDCONTROLLER_HPP
#define QUADCOPTER_PIDCONTROLLER_HPP

#include "Eigen/Dense"

class pidController {
public:
    pidController(){};
    ~pidController(){};

    Eigen::Vector4d smallAnglesControl(Eigen::VectorXd gc_, Eigen::VectorXd gv_, Eigen::Matrix3d bodyRot_);

    void setTargetPoint(Eigen::VectorXd target);

    void setParameters(double P, double I, double D, double contr_dt, Eigen::Matrix4d thrustMatrix,
                                      double hoverT, bool normAct, bool maxScal);

    Eigen::VectorXd targetPoint;
private:
    Eigen::Vector3d eulerAngles, angVel_Body, desAcc;
    Eigen::VectorXd currState; //pos, eulerAngles, linVel, angVel
    Eigen::VectorXd errState; //pos, eulerAngles, linVel, angVel
    Eigen::VectorXd desState; //pos, eulerAngles, linVel, angVel
    Eigen::Vector4d u, controlThrusts;
    double pGain, dGain, iGain;
    Eigen::Matrix4d thrusts2TorquesAndForces;
    bool normAction, maxScale;
    double control_dt;
    double hoverThrust;
    Eigen::Vector4d thrustOffset;
};

#endif //QUADCOPTER_PIDCONTROLLER_HPP
