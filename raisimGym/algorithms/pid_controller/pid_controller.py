import numpy as np
from scipy.spatial.transform import Rotation as R


## PID Controller, only tuned for recent quadcopter model

class PID:
    def __init__(self, pGain, iGain, dGain, obs_dim, act_dim, timeStep, mass, normalize_action=True):
        self.pGain = pGain
        self.iGain = iGain
        self.dGain = dGain
        self.m = mass
        self.timeStep = timeStep
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.normalize_action = normalize_action

        self.u = np.zeros(shape=(act_dim, 1))
        self.controlThrusts = np.zeros(shape=(act_dim, 1))
        self.hoverThrust = self.m * 9.81 / 4
        self.thrusts2TorquesAndForcesInv = np.linalg.inv(
            np.array([[1.0, 1.0, 1.0, 1.0], [0.1710491, -0.1710491, -0.1710491, 0.1710491],
                      [-0.1710491, -0.1710491, 0.1710491, 0.1710491],
                      [0.016, -0.016, 0.016, -0.016]]))

    def control(self, obs, target, loopCount):

        #eulerAngles = np.array(R.from_matrix(obs[3:12].reshape(3, 3)).as_rotvec()).reshape(3, 1) # does not work
        eulerAngles = self.quatToEuler(obs[18:22])
        #eulerAngles = obs[18:21]
        currState = np.concatenate([obs[0:3], eulerAngles, obs[12:18]])
        desState = np.zeros(shape=(12, 1))
        errState = target - currState

        # outer PID controller for Position Control, runs with 20 Hz
        # input: error between target point and current state
        # output: desired acceleration, desired pitch and roll angles and u[0] for altitude control
        if loopCount > 7:
            desAcc = self.pGain * errState[0:3] + self.dGain * errState[6:9]
            desState[3] = (1 / 9.81 * (desAcc[0] * np.sin(currState[5]) - desAcc[1] * np.cos(currState[5])))/4
            desState[4] = (1 / 9.81 * (desAcc[0] * np.cos(currState[5]) + desAcc[1] * np.sin(currState[5])))/4
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
        if max_scale > 1.5 * self.hoverThrust:
            self.controlThrusts = 1.5 / max_scale * self.hoverThrust * self.controlThrusts
        for i in range(len(self.controlThrusts)):
            if self.controlThrusts[i] < 0.5 * self.hoverThrust:
                self.controlThrusts[i] = 0.5 * self.hoverThrust

        if self.normalize_action:
            self.controlThrusts = self.controlThrusts / (0.5*self.hoverThrust) - 2

        return self.controlThrusts.reshape(1, self.act_dim).astype(dtype="float32")

    def quatToEuler(self, q):

        angles = np.zeros(shape=(3, 1))

        sinr_cosp = 2 * (q[0] * q[1] + q[2] * q[3])
        cosr_cosp = 1 - 2 * (q[1] * q[1] + q[2] * q[2])
        angles[0] = np.arctan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (q[0] * q[2] - q[3] * q[1])
        if np.abs(sinp) >= 1:
            angles[1] = np.copysign(np.pi / 2, sinp)
        else:
            angles[1] = np.arcsin(sinp)

        siny_cosp = 2 * (q[0] * q[3] + q[1] * q[2])
        cosy_cosp = 1 - 2 * (q[2] * q[2] + q[3] * q[3])
        angles[2] = np.arctan2(siny_cosp, cosy_cosp)

        return angles

