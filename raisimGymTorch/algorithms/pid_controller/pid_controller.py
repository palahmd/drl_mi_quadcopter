import numpy as np
import torch

## PID Controller, only tuned for recent quadcopter model

class PID:
    def __init__(self, pGain, iGain, dGain, obs_dim, act_dim, timeStep, mass):
        self.pGain = pGain
        self.iGain = iGain
        self.dGain = dGain
        self.m = mass
        self.timeStep = timeStep
        self.obs_dim = obs_dim

        self.u = np.zeros(act_dim)
        self.controlThrusts = np.zeros(act_dim)
        self.hoverThrust = self.m * 9.81 / 4
        self.thrusts2TorquesAndForces = np.array([[1.0, 1.0, 1.0, 1.0], [0.1710491, -0.1710491, -0.1710491, 0.1710491], [-0.1710491, -0.1710491, 0.1710491, 0.1710491],[0.016, -0.016, 0.016, -0.016]])


    def smallAnglesControl(self, obs, target, loopCount):

        eulerAngles = self.toEuler(obs[3:12])
        currState = np.concatenate([obs[0:3], eulerAngles, obs[12:18]])
        errState = target - currState

        desState = np.zeros(currState.shape[0])

        # outer PID controller for Position Control, runs with 20 Hz
        # input: error between target point and current state
        # output: desired acceleration, desired pitch and roll angles and u[0] for altitude control
        if loopCount > 4:
            desAcc = self.pGain * errState[0:3] + self.dGain * errState[6:9]
            desState[3] = 1 / 9.81 * (desAcc[0] * np.sin(currState[5]) - desAcc[1] * np.cos(currState[5]))
            desState[4] = 1 / 9.81 * (desAcc[0] * np.cos(currState[5]) + desAcc[1] * np.sin(currState[5]))
            self.u[0] = self.m * 9.81 + self.m * desAcc[2] + self.m * self.iGain * errState[2] * self.timeStep
        else:
            self.u[0] = self.u[0]

        # inner PD controller for Attitude Control, runs with 100 Hz
        # input: desired acceleration and error between desired state and current state
        # output: u[1] - u[3] for attitude control
        errState = desState - currState
        self.u[1] = 0.006687 * 81 * errState[3] + 2 * 0.006687 * 9 * errState[9]
        self.u[2] = 0.0101 * 81 * errState[4] + 2 * 0.0101 * 9 * errState[10]
        self.u[3] = 0.00996 * 81 * errState[5] + 2 * 0.00996 * 9 * errState[11]

        self.controlThrusts = self.thrusts2TorquesAndForces.dot(self.u)

        max_scale = np.max(self.controlThrusts)
        min_scale = np.min(self.controlThrusts)
        if min_scale < 0.5 * self.hoverThrust:
            self.controlThrusts = 0.5 / min_scale * self.hoverThrust * self.controlThrusts
        if max_scale > 1.5 * self.hoverThrust:
            self.controlThrusts = 1.5 / max_scale * self.hoverThrust * self.controlThrusts
        print(eulerAngles)
        print(self.controlThrusts)
        return self.controlThrusts



    def toEuler(self, obs):

        sy = np.sqrt(obs[0] * obs[0] +  obs[3] * obs[3])
        singular = sy < 1e-6

        if  not singular :
            x = np.arctan2(obs[7] , obs[8])
            y = np.arctan2(-obs[6], sy)
            z = np.arctan2(obs[3], obs[0])
        else :
            x = np.arctan2(-obs[5], obs[4])
            y = np.arctan2(-obs[6], sy)
            z = 0

        return np.array([x, y, z])
