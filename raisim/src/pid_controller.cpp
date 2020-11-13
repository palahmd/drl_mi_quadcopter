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

void pidController::smallAnglesControl() {
    /*** PD Controller for small angles ***/
    // current state and error. eulerAngles is a conversion from quaternions to Euler Angles
    eulerAngles = ToEulerAngles(ob_q);
    angVel_Body = rot.e().inverse() * angVel_W.e();
    currState << pos.e(), eulerAngles, linVel_W.e(), angVel_Body;
    desState = targetPoint;
    errState = targetPoint - currState;

    /** outer PID controller for Position Control, runs with 20 Hz
     ** input: error between target point and current state
     ** output: desired acceleration, desired pitch and roll angles and u[0] for altitude control **/
    if (loopCount > 4) {
        desAcc = pGain * errState.segment(0, 3) + dGain * errState.segment(6, 3);
        desState[3] = 1 / g * (desAcc[0] * sin(currState[5]) - desAcc[1] * cos(currState[5]));
        desState[4] = 1 / g * (desAcc[0] * cos(currState[5]) + desAcc[1] * sin(currState[5]));
        u[0] = m * g + m * desAcc[2] + m * iGain * errState[2] * timeStep;
    } else {
        u[0] = u[0];
    }

    /** inner PD controller for Attitude Control, runs with 100 Hz
     ** input: desired acceleration, desired State and current state
     ** output: u[1] - u[3] for attitude control **/
    errState = desState - currState;
    u[1] = inertiaDiagVec[0] * 81 * errState[3] + 2 * inertiaDiagVec[0] * 9 * errState[9];
    u[2] = inertiaDiagVec[1] * 81 * errState[4] + 2 * inertiaDiagVec[1] * 9 * errState[10];
    u[3] = inertiaDiagVec[2] * 81 * errState[5] + 2 * inertiaDiagVec[2] * 9 * errState[11];

    /** the input u is the desired input needed to reach the desired state,
     ** thus it has to be transformed into each rotor controlThrusts **/
    controlThrusts = thrusts2TorquesAndForces.inverse() * u;

    // scale controlThrusts and avoid thrusts out of range: 0.5 - 1.5 * hoverThrust
    double max_scale = controlThrusts.maxCoeff();
    if (max_scale > 1.5 * hoverThrust){
        controlThrusts = 1.5 / max_scale * hoverThrust * controlThrusts;
    }
    for (int i = 0; i<4; i++){
        if (controlThrusts[i]< 0.5 * hoverThrust){
            controlThrusts[i] = 0.5 * hoverThrust;
        }
    }

    /** Motor Model:
     **             delta_omega = 20 * (omega(t) - omega(t-1))
     **             thrust_i = k_f * omega_i²
     **             delta_thrust_i = 20² *(thrust_i-thrust(t-1)) * timeStep
     ** OR: time_constant_up = 0.0125 sec, time_const_down=0.025 sec **/
     for (int i = 0; i<4; i++){
         if (thrusts[i]<controlThrusts[i]) {  // time constant for increasing rotor speed
             thrusts[i] = thrusts[i] + (controlThrusts[i] - thrusts[i]) * timeStep / 0.0125;
         } else if (thrusts[i]>controlThrusts[i]){   // time constant for decreasing rotor speed
             thrusts[i] = thrusts[i] + (controlThrusts[i] - thrusts[i]) * timeStep / 0.025;
         }
     }
}

void pidController::setTargetPoint(double x, double y, double z){
   targetPoint.setZero(12);
   targetPoint.head(3) << x, y, z;
}
