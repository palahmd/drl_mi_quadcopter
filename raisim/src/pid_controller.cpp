//
// Created by pala on 03.11.20.
//
#include "pidController.hpp"
#include "quaternionToEuler.hpp"
#include "quadcopterInit.hpp"
#include "iostream"

pidController::pidController(double P, double I, double D) {
    this->pGain = P;
    this->iGain = I;
    this->dGain = D;

    currState.setZero(12); //pos, linVel, angVel, eulerAngles
    desState.setZero(12); // != desPos which is the target point
    errState.setZero(12);
}

void pidController::smallAnglesController() {
    /*** PD Controller for small angles ***/
    // current state and error. eulerAngles is a conversion from quaternions to Euler Angles
    eulerAngles = ToEulerAngles(ob_q);
    angVel_Body = rot.e().inverse() * angVel_W.e();
    currState << pos.e(), eulerAngles, linVel_W.e(), angVel_Body;
    desState = targetPoint;
    errState = targetPoint - currState;

    /** outer PID controller for Position Control
     ** input: error between target point and current state
     ** output: desired acceleration, desired pitch and roll angles and u[0] for altitude control **/
    desAcc = pGain * errState.segment(0, 3) + dGain * errState.segment(6, 3);
    desState[3] = 1 / g * (desAcc[0] * sin(currState[5]) - desAcc[1] * cos(currState[5]));
    desState[4] = 1 / g * (desAcc[0] * cos(currState[5]) + desAcc[1] * sin(currState[5]));
    u[0] = m * g + m * desAcc[2] + m * iGain * errState[2] * timeStep;

    /** inner PID controller for Attitude Control
     ** input: desired acceleration, desired State and current state
     ** output: u[1] - u[3] for attitude control **/
    errState = desState - currState;
    u[1] = pGain / 2 * errState[3] + dGain / 20 * errState[9] + iGain  * errState[3] * timeStep;
    u[2] = pGain / 2 * errState[4] + dGain / 20 * errState[10] + iGain  * errState[4] * timeStep;
    u[3] = pGain / 2 * errState[5] + dGain / 20 * errState[11] + iGain  * errState[5] * timeStep;

    /** the input u is the desired input needed to reach the desired state,
     ** thus it has to be transformed into each rotor controlThrusts **/
    controlThrusts = thrusts2TorquesAndForces.inverse() * u;

    // scale controlThrusts and avoid thrusts out of range: 0.5 - 1.5 * hoverThrust_i
    double max_scale = controlThrusts.maxCoeff();
    if (max_scale > 1.5*hoverThrust_i){
        controlThrusts = 1.5/max_scale * hoverThrust_i * controlThrusts;
    }
    for (int i = 0; i<4; i++){
        if (controlThrusts[i]<0.5*hoverThrust_i){
            controlThrusts[i] = 0.5*hoverThrust_i;
        }
    }

    /** Motor Model:
     **             delta_omega = 20 * (omega(t) - omega(t-1))
     **             thrust_i = k_f * omega_i²
     **             delta_thrust_i = 20² *(thrust_i-thrust(t-1)) * timeStep
     ** OR: time_constant_up = 0.0125 sec, time_const_down=0.025 sec **/
     for (int i = 0; i<4; i++){
         if (thrusts[i]<controlThrusts[i]) {
             thrusts[i] = thrusts[i] + timeStep / 0.0125 * (controlThrusts[i] - thrusts[i]);
         } else if (thrusts[i]>controlThrusts[i]){
             thrusts[i] = thrusts[i] + timeStep / 0.025 * (controlThrusts[i] - thrusts[i]);
         }
     }

    std::cout << timeStep/0.025*(controlThrusts[0]-thrusts[0]) << "\n ___ " << std::endl;
}

void pidController::setTargetPoint(double x, double y, double z){
   targetPoint.setZero(12);
   targetPoint.head(3) << x, y, z;
}
