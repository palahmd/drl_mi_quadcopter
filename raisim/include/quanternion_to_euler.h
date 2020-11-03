//
// Created by pala on 03.11.20.
//

#define _USE_MATH_DEFINES
#include <cmath>
#include "Eigen/Dense"

struct Quaternion {
    double w, x, y, z;
};

struct EulerAngles {
    double roll, pitch, yaw;
};

Eigen::Vector3d ToEulerAngles(Eigen::VectorXd quaternion) {
    EulerAngles angles;
    Eigen::Vector3d anglesVec;

    Quaternion q;
    q.w = quaternion[0];
    q.x = quaternion[1];
    q.y = quaternion[2];
    q.z = quaternion[3];

    // roll (x-axis rotation)
    double sinr_cosp = 2 * (q.w * q.x + q.y * q.z);
    double cosr_cosp = 1 - 2 * (q.x * q.x + q.y * q.y);
    angles.roll = std::atan2(sinr_cosp, cosr_cosp);

    // pitch (y-axis rotation)
    double sinp = 2 * (q.w * q.y - q.z * q.x);
    if (std::abs(sinp) >= 1)
        angles.pitch = std::copysign(M_PI / 2, sinp); // use 90 degrees if out of range
    else
        angles.pitch = std::asin(sinp);

    // yaw (z-axis rotation)
    double siny_cosp = 2 * (q.w * q.z + q.x * q.y);
    double cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z);
    angles.yaw = std::atan2(siny_cosp, cosy_cosp);

    anglesVec << angles.roll, angles.pitch, angles.yaw;

    return anglesVec;
}
