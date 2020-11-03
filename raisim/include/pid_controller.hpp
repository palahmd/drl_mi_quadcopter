//
// Created by pala on 03.11.20.
//

#ifndef QUADCOPTER_PID_CONTROLLER_HPP
#define QUADCOPTER_PID_CONTROLLER_HPP
#include "Eigen/Dense"
#include "raisim/World.hpp"
#include "quadcopter_init.hpp"
#include "quanternion_to_euler.h"
#include "iostream"

class pidController {

public:

    static void calculateThrusts(Eigen::VectorXd desiredPos){
        Eigen::Vector3d currentAngles = ToEulerAngles(ob_q);
        Eigen::VectorXd curState; curState.setZero(12);
        Eigen::VectorXd errState; errState.setZero(12);
        Eigen::VectorXd errPos; errPos.setZero(12);
        Eigen::VectorXd desiredState; desiredState.setZero(12);
        Eigen::Vector4d controlThrusts;

        int pGain = 1;
        int dGain = 0.5;

        // current state, currentAngles is a conversion from quaternions to Euler Angles
        curState.head(3) = currentAngles;
        curState.tail(9) = ob_q.tail(9);

        // desired Position is the target position. This should be the outer PD loop for the position controller
        // and to calculate the desired roll and pitch angles for the attitude control
        errPos = desiredPos - curState;
        double v_x_err = errPos[3]*sin(currentAngles[2])-errPos[4]*cos(currentAngles[2]);
        double v_y_err = errPos[3]*cos(currentAngles[2])+errPos[4]*sin(currentAngles[2]);
        double v_dot_x_err = errPos[6]*sin(currentAngles[2])-errPos[6]*cos(currentAngles[2]);
        double v_dot_y_err = errPos[6]*cos(currentAngles[2])+errPos[6]*sin(currentAngles[2]);

        // this is the inner PD loop for attitude control (or somehow)
        desiredState[0] = 1/g*(pGain*v_x_err - dGain*v_dot_x_err);
        desiredState[1] = 1/g*(pGain*v_y_err - dGain*v_dot_y_err);
        desiredState.tail(10) = desiredPos.tail(10);
        errState = desiredState - curState;

        controlThrusts[0] = pGain*errState[0] + dGain*errState[9];
        controlThrusts[1] = pGain*errState[1] + dGain*errState[10];
        controlThrusts[2] = pGain*errState[2] + dGain*errState[11];
        controlThrusts[3] = (pGain*errState[5] + dGain*errState[5]);

        std::cout << controlThrusts << std::endl;

        thrusts = hoverThrusts + thrusts2TorquesAndForces.inverse()*controlThrusts;

    }


};

#endif //QUADCOPTER_PID_CONTROLLER_HPP
