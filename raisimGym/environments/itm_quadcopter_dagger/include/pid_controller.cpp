//
// Created by pala on 03.11.20.
//
#include "pidController.hpp"
#include "quaternionToEuler.hpp"


Eigen::Vector4d pidController::smallAnglesControl(Eigen::VectorXd gc_, Eigen::VectorXd gv_, Eigen::Matrix3d bodyRot_) {
    /*** PD Controller for small angles ***/
    // current state and error. eulerAngles is a conversion from quaternions to Euler Angles
    eulerAngles = ToEulerAngles(gc_.segment(3,4));
    angVel_Body = bodyRot_ * gv_.segment(3,3);
    currState << gc_.head(3), eulerAngles, gv_.head(3), angVel_Body;
    desState = targetPoint;
    errState = targetPoint - currState;

    /** outer PID controller for Position Control, runs with same frequency
     ** input: error between target point and current state
     ** output: desired acceleration, desired pitch and roll angles and u[0] for altitude control **/
    desAcc = pGain * errState.segment(0, 3) + dGain * errState.segment(6, 3);
    desState[3] = (1 / 9.81 * (desAcc[0] * sin(currState[5]) - desAcc[1] * cos(currState[5])))/4;
    desState[4] = (1 / 9.81 * (desAcc[0] * cos(currState[5]) + desAcc[1] * sin(currState[5])))/4;
    u[0] = 1.727 * 9.81 + 1.727 * desAcc[2] + 1.727 * iGain * errState[2] * control_dt;


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
    controlThrusts = thrusts2TorquesAndForces.inverse() * u;

    /// maximize action by scaling to [0.5, 1.5] * hoverThrusts
    if (maxScale){
        double max_scale = controlThrusts.maxCoeff();
        if (max_scale > (1.5 * hoverThrust)){
            controlThrusts = 1.5 / max_scale * hoverThrust * controlThrusts;
        }
        for (int i = 0; i<4; i++){
            if (controlThrusts[i]< (0.5 * hoverThrust)){
                controlThrusts[i] = 0.5 * hoverThrust;
            }
        }
    }
    /// bound action to [-1, 1] for the Neural Network to evaluate
    if (normAction){
        controlThrusts = controlThrusts / (0.5 * hoverThrust) - thrustOffset;
    }
    return controlThrusts;
}

void pidController::setTargetPoint(Eigen::VectorXd target){
   targetPoint.setZero(12);
   targetPoint.head(3) = target.head(3);
}

void pidController::setParameters(double P, double I, double D, double contr_dt, Eigen::Matrix4d thrustMatrix,
                                  double hoverT, bool normAct, bool maxScal) {
    pGain = P;
    iGain = I;
    dGain = D;
    thrusts2TorquesAndForces = thrustMatrix;
    normAction = normAct;
    control_dt = contr_dt;
    hoverThrust = hoverT;
    maxScale = maxScal;

    currState.setZero(12); //pos, linVel, angVel, eulerAngles
    desState.setZero(12); // != desired Position which is the target point
    errState.setZero(12);
    controlThrusts.setZero();
    thrustOffset = 2*Eigen::Vector4d::Ones();
}
