//
// Created by pala on 03.11.20.
//
#include "pidController.hpp"
#include "quaternionToEuler.hpp"
#include "quadcopterInit.hpp"

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
    u[1] = pGain / 2 * errState[3] + dGain / 10 * errState[9] + iGain * errState[3] * timeStep;
    u[2] = pGain / 2 * errState[4] + dGain / 10 * errState[10] + iGain * errState[4] * timeStep;
    u[3] = pGain / 2 * errState[5] + dGain / 10 * errState[11] + iGain * errState[5] * timeStep;

    // range for maximum and minimum forces i: 0.5 =< thrusts_i =< 1.5
    if (u[0] > 1.5 * m * g) {
        u[0] = 1.5 * m * g;
    } else if (u[0] < 0.5 * m * g) {
        u[0] = 0.5 * m * g;
    } else {
        u[0] = u[0];
    }

    /** the input u is the desired input needed to reach the desired state,
     ** thus it has to be transformed into each rotor thrusts **/
    thrusts = thrusts2TorquesAndForces.inverse() * u;
}

void pidController::setTargetPoint(double x, double y, double z){
   targetPoint.setZero(12);
   targetPoint.head(3) << x, y, z;
}
