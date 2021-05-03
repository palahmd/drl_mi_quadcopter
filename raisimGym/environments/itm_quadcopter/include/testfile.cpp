//
// Created by pala on 03.11.20.
//
#include "pidController.hpp"
#include "quaternionToEuler.hpp"
#include "iostream"

using namespace raisim;

pidController::pidController(double P, double I, double D) {
    this->pGain = P;
    this->iGain = I;
    this->dGain = D;

    currState.setZero(12); //pos, linVel, angVel, eulerAngles
    desState.setZero(12); // != desired Position which is the target point
    errState.setZero(12);
    controlThrusts.setZero();
}

Eigen::VectorXd pidController::smallAnglesControl(Eigen::VectorXd& gc_, Eigen::VectorXd gv_, Eigen::Matrix3d& bodyRot_,
                                                  Eigen::VectorXd& thrusts_) {
    /*** PD Controller for small angles ***/
    // current state and error. eulerAngles is a conversion from quaternions to Euler Angles
    eulerAngles = ToEulerAngles(gc_);
    angVel_Body = bodyRot_ * gv_.segment(3,3);
    currState << gc_.head(3), eulerAngles, gv_.head(3), angVel_Body;
    desState = targetPoint;
    errState = targetPoint - currState;

    /** outer PID controller for Position Control, runs with 20 Hz
     ** input: error between target point and current state
     ** output: desired acceleration, desired pitch and roll angles and u[0] for altitude control **/
    if (loopCount_ > 4) {
        desAcc = pGain * errState.segment(0, 3) + dGain * errState.segment(6, 3);
        desState[3] = 1 / 9.81 * (desAcc[0] * sin(currState[5]) - desAcc[1] * cos(currState[5]));
        desState[4] = 1 / 9.81 * (desAcc[0] * cos(currState[5]) + desAcc[1] * sin(currState[5]));
        u[0] = 1.727 * 9.81 + 1.727 * desAcc[2] + 1.727 * iGain * errState[2] * control_dt_;
    } else {
        u[0] = u[0];
    }

    /** inner PD controller for Attitude Control, runs with 100 Hz
     ** input: desired acceleration and error between desired state and current state
     ** output: u[1] - u[3] for attitude control **/
    errState = desState - currState;
    u[1] = 0.006687 * 81 * errState[3] + 2 * 0.006687 * 9 * errState[9];
    u[2] = 0.0101 * 81 * errState[4] + 2 * 0.0101 * 9 * errState[10];
    u[3] = 0.00996 * 81 * errState[5] + 2 * 0.00996 * 9 * errState[11];

    /** the input u for reaching the the desired state has to be transformed into each rotor thrust.
     ** The control thrust from this time step will be partly applied in the next time step
     ** due to the rotor velocity change delay in the motor model **/
    controlThrusts = thrusts2TorquesAndForces_.inverse() * u;

    /// scale controlThrusts and avoid thrusts_ out of range: 0.5 - 1.5 * hoverThrust_
    double max_scale = controlThrusts.maxCoeff();
    if (max_scale > (1.5 * 1.727 * 9.81/4)){
        controlThrusts = 1.5 / max_scale * (1.727 * 9.81/4) * controlThrusts;
    }
    for (int i = 0; i<4; i++){
        if (controlThrusts[i]< (0.5 * 1.727 * 9.81/4)){
            controlThrusts[i] = 0.5 * 1.727 * 9.81/4;
        }
    }

    /** Motor Model for motor i:
 **         time_constant_up = 0.0125 sec
 **         time_const_down = 0.025 sec
 **         thrust(t) = thrust(t-1) + (controlThrust(t-1) - thrust(t-1)) * control_dt_/time_constant **/
    for (int i = 0; i<4; i++){
        if (thrusts_[i]<controlThrusts[i]) {  // time constant for increasing rotor speed
            thrusts_[i] = thrusts_[i] + (controlThrusts[i] - thrusts_[i]) * control_dt_ / 0.0125;
        } else if (thrusts_[i]>controlThrusts[i]){   // time constant for decreasing rotor speed
            thrusts_[i] = thrusts_[i] + (controlThrusts[i] - thrusts_[i]) * control_dt_ / 0.025;
        }
    }

    return thrusts_;
}

void pidController::setTargetPoint(double x, double y, double z){
    targetPoint.setZero(12);
    targetPoint.head(3) << x, y, z;
}