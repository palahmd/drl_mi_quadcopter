import numpy as np
from scipy.spatial.transform import Rotation as R


## PID Controller, only tuned for recent quadcopter model

class PID:
    def __init__(self, pGain, iGain, dGain, obs_dim, act_dim, timeStep, mass):
        self.pGain = pGain
        self.iGain = iGain
        self.dGain = dGain
        self.m = mass
        self.timeStep = timeStep
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.u = np.zeros(shape=(act_dim, 1))
        self.controlThrusts = np.zeros(shape=(act_dim, 1))
        self.hoverThrust = self.m * 9.81 / 4
        self.thrusts2TorquesAndForcesInv = np.linalg.inv(np.array([[1.0, 1.0, 1.0, 1.0], [0.1710491, -0.1710491, -0.1710491, 0.1710491],
                                                  [-0.1710491, -0.1710491, 0.1710491, 0.1710491],
                                                  [0.016, -0.016, 0.016, -0.016]]))

    def smallAnglesControl(self, obs, target, loopCount):

        eulerAngles = np.array(R.from_matrix(obs[3:12].reshape(3,3)).as_rotvec()).reshape(3,1)
        currState = np.concatenate([obs[0:3], eulerAngles, obs[12:18]])
        errState = target - currState
        desState = np.zeros(shape=(currState.shape[0], 1))

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

        self.controlThrusts = self.thrusts2TorquesAndForcesInv.dot(self.u)

        # action scaling within the boundaries of 0.5 * hoverThrust < thrust < 1.5 * hoverThrust
        max_scale = np.max(self.controlThrusts)
        min_scale = np.min(self.controlThrusts) + 1e-3
        if min_scale < 0.5 * self.hoverThrust:
            self.controlThrusts = 0.5 / min_scale * self.hoverThrust * self.controlThrusts
        if max_scale > 1.5 * self.hoverThrust:
            self.controlThrusts = 1.5 / max_scale * self.hoverThrust * self.controlThrusts


        return self.controlThrusts.reshape(1, self.act_dim).astype(dtype="float32")

