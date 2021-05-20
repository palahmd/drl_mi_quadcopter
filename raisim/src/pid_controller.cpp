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
    desState.setZero(12); // != desired Position which is the target point
    errState.setZero(13);
    controlThrusts.setZero();
}

void pidController::smallAnglesControl() {
    /*** pd controller for small angles ***/
    // current state and error. eulerangles is a conversion from quaternions to euler angles
    eulerangles = toeulerangles(ob_q);
    angvel_body = rot.e().inverse() * angvel_w.e();
    currstate << pos.e(), eulerangles, linvel_w.e(), angvel_body;
    desstate = targetpoint;
    errstate = targetpoint - currstate;

    /** outer pid controller for position control, runs with 20 hz
     ** input: error between target point and current state
     ** output: desired acceleration, desired pitch and roll angles and u[0] for altitude control **/
    if (loopcount > 4) {
        desacc = pgain * errstate.segment(0, 3) + dgain * errstate.segment(6, 3);
        desstate[3] = 1 / g * (desacc[0] * sin(currstate[5]) - desacc[1] * cos(currstate[5]));
        desstate[4] = 1 / g * (desacc[0] * cos(currstate[5]) + desacc[1] * sin(currstate[5]));
        u[0] = m * g + m * desacc[2] + m * igain * errstate[2] * timestep;
    } else {
        u[0] = u[0];
    }

    /** inner pd controller for attitude control, runs with 100 hz
     ** input: desired acceleration and error between desired state and current state
     ** output: u[1] - u[3] for attitude control **/
    errstate = desstate - currstate;
    u[1] = inertiadiagvec[0] * 81 * errstate[3] + 2 * inertiadiagvec[0] * 9 * errstate[9];
    u[2] = inertiadiagvec[1] * 81 * errstate[4] + 2 * inertiadiagvec[1] * 9 * errstate[10];
    u[3] = inertiadiagvec[2] * 81 * errstate[5] + 2 * inertiadiagvec[2] * 9 * errstate[11];

    /** motor model for motor i:
     **         time_constant_up = 0.0125 sec
     **         time_const_down = 0.025 sec
     **         thrust(t) = thrust(t-1) + (controlthrust(t-1) - thrust(t-1)) * timestep/time_constant **/
    for (int i = 0; i<4; i++){
        if (thrusts[i]<controlthrusts[i]) {  // time constant for increasing rotor speed
            thrusts[i] = thrusts[i] + (controlthrusts[i] - thrusts[i]) * timestep / 0.0125;
        } else if (thrusts[i]>controlthrusts[i]){   // time constant for decreasing rotor speed
            thrusts[i] = thrusts[i] + (controlthrusts[i] - thrusts[i]) * timestep / 0.025;
        }
    }

    /** the input u for reaching the the desired state has to be transformed into each rotor thrust.
     ** the control thrust from this time step will be partly applied in the next time step
     ** due to the rotor velocity change delay in the motor model **/
    controlthrusts = thrusts2torquesandforces.inverse() * u;

    /// scale controlthrusts and avoid thrusts out of range: 0.5 - 1.5 * hoverthrust
    double max_scale = controlthrusts.maxcoeff();
    if (max_scale > 1.5 * hoverthrust){
        controlthrusts = 1.5 / max_scale * hoverthrust * controlthrusts;
    }
    for (int i = 0; i<4; i++) {
        if (controlthrusts[i] < 0.5 * hoverthrust) {
            controlthrusts[i] = 0.5 * hoverthrust;
        }
    }

}

void pidController::setTargetPoint(double x, double y, double z){
   targetPoint.setZero(12);
   targetPoint.head(3) << x, y, z;
}
