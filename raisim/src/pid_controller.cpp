//
// Created by pala on 03.11.20.
//
#include "pidController.hpp"
#include "quaternionToEuler.hpp"
#include "quadcopterInit.hpp"
#include "iostream"

    void pidController::calculateThrusts(const Eigen::VectorXd& desiredPos){
         Eigen::Vector3d eulerAngles;
         Eigen::VectorXd currState;
         Eigen::VectorXd errState;
         Eigen::VectorXd errPos;
         Eigen::Vector3d desiredAcc, desiredVel, desiredAng;
         Eigen::Vector4d controlThrusts;


         int pGain, dGain, iGain;
        currState.setZero(12);
        errState.setZero(12); errPos.setZero(12);
        eulerAngles = ToEulerAngles(ob_q);

        /// P, D and I gain
        pGain = 20;
        dGain = 30;
        iGain = 0.1;


        /// current state, eulerAngles is a conversion from quaternions to Euler Angles
        currState << ob_q.head(9), eulerAngles;
        errPos = desiredPos - currState;


        /// outer PID Controller for Position control
        desiredAcc = pGain*errPos.segment(0,3) + dGain * errPos.segment(3,3) + linAcc_W.e();
        desiredAng[1] = 1/g*(desiredAcc[0]*sin(currState[3])-desiredAcc[1]*cos(currState[3]));
        desiredAng[2] = 1/g*(desiredAcc[0]*cos(currState[3])+desiredAcc[1]*sin(currState[3]));

        /// desired Position is the target position. This should be the outer PD loop for the position controller
        /// and to calculate the desired roll and pitch angles for the attitude control
        errPos[9] = desiredAng[0] - currState[9];
        errPos[10] = desiredAng[1] - currState[10];

        controlThrusts[0] = m*desiredAcc[2]/5;
        controlThrusts[1] = (pGain*errPos[9] + dGain*errState[6])*10;
        controlThrusts[2] = (pGain*errState[10] + dGain*errState[7])*10;
        controlThrusts[3] = (pGain*errState[11] + dGain*errState[8])*10;
        
        /// range for maximum and minimum applied forces 
        for (size_t i=0; i<4; i++){
            if (controlThrusts[i]>0.5*hoverThrust_i) {
                controlThrusts[i] = 0.5*hoverThrust_i;
            }
            else if (controlThrusts[i]<-0.5*hoverThrust_i){
                controlThrusts[i] = -0.5*hoverThrust_i;
            }
            else{
                controlThrusts[i] = controlThrusts[i];
            }
        }
        thrusts = hoverThrusts + thrusts2TorquesAndForces.inverse()*controlThrusts;
        std::cout << controlThrusts << "\n" << std::endl;
    }


