//
// Created by pala on 03.11.20.
//
#include <cmath>
#include "Eigen/Dense"

Eigen::Vector3d ToEulerAngles(Eigen::VectorXd& q) {
    // quaternions inside observation vector start from ob_q[9]
    // ob_q[9] = w, ob_q[10] = x, ob_q[11] = y, ob_q[12] = z
    Eigen::Vector3d angles;

    // roll (x-axis rotation)
    double sinr_cosp = 2 * (q[3] * q[4] + q[5] * q[6]);
    double cosr_cosp = 1 - 2 * (q[4] * q[4] + q[5] * q[5]);
    angles[0] = std::atan2(sinr_cosp, cosr_cosp);

    // pitch (y-axis rotation)
    double sinp = 2 * (q[3] * q[5] - q[6] * q[4]);
    if (std::abs(sinp) >= 1)
        angles[1] = std::copysign(M_PI / 2, sinp); // use 90 degrees if out of range
    else
        angles[1] = std::asin(sinp);


    // yaw (z-axis rotation)
    double siny_cosp = 2 * (q[3] * q[6] + q[4] * q[5]);
    double cosy_cosp = 1 - 2 * (q[5] * q[5] + q[6] * q[6]);
    angles[2] = std::atan2(siny_cosp, cosy_cosp);

    return angles;
}
