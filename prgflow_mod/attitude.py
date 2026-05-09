import math

import numpy as np


def quat_normalize(q):
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    q = q / n
    if q[0] < 0.0:
        q = -q
    return q


def quat_to_matrix(q):
    q = quat_normalize(q)
    w, x, y, z = q
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


class Madgwick:
    def __init__(self, beta=0.08, q0=None):
        self.beta = float(beta)
        if q0 is None:
            q0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.q_wb = quat_normalize(q0)

    def reset(self, q0=None):
        if q0 is None:
            q0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.q_wb = quat_normalize(q0)

    def step(self, gyro, accel, dt):
        gx, gy, gz = np.asarray(gyro, dtype=np.float64)
        ax, ay, az = np.asarray(accel, dtype=np.float64)
        q1, q2, q3, q4 = self.q_wb

        norm_a = math.sqrt(ax * ax + ay * ay + az * az)
        if norm_a > 1e-12:
            ax /= norm_a
            ay /= norm_a
            az /= norm_a

            _2q1 = 2.0 * q1
            _2q2 = 2.0 * q2
            _2q3 = 2.0 * q3
            _2q4 = 2.0 * q4
            _4q1 = 4.0 * q1
            _4q2 = 4.0 * q2
            _4q3 = 4.0 * q3
            _8q2 = 8.0 * q2
            _8q3 = 8.0 * q3
            q1q1 = q1 * q1
            q2q2 = q2 * q2
            q3q3 = q3 * q3
            q4q4 = q4 * q4

            s1 = _4q1 * q3q3 + _2q3 * ax + _4q1 * q2q2 - _2q2 * ay
            s2 = _4q2 * q4q4 - _2q4 * ax + 4.0 * q1q1 * q2 - _2q1 * ay - _4q2 + _8q2 * q2q2 + _8q2 * q3q3 + _4q2 * az
            s3 = 4.0 * q1q1 * q3 + _2q1 * ax + _4q3 * q4q4 - _2q4 * ay - _4q3 + _8q3 * q2q2 + _8q3 * q3q3 + _4q3 * az
            s4 = 4.0 * q2q2 * q4 - _2q2 * ax + 4.0 * q3q3 * q4 - _2q3 * ay
            sn = math.sqrt(s1 * s1 + s2 * s2 + s3 * s3 + s4 * s4)
            if sn > 1e-12:
                s1 /= sn
                s2 /= sn
                s3 /= sn
                s4 /= sn
            else:
                s1 = 0.0
                s2 = 0.0
                s3 = 0.0
                s4 = 0.0
        else:
            s1 = 0.0
            s2 = 0.0
            s3 = 0.0
            s4 = 0.0

        qdot1 = 0.5 * (-q2 * gx - q3 * gy - q4 * gz) - self.beta * s1
        qdot2 = 0.5 * (q1 * gx + q3 * gz - q4 * gy) - self.beta * s2
        qdot3 = 0.5 * (q1 * gy - q2 * gz + q4 * gx) - self.beta * s3
        qdot4 = 0.5 * (q1 * gz + q2 * gy - q3 * gx) - self.beta * s4

        self.q_wb = quat_normalize(
            np.array(
                [
                    q1 + qdot1 * dt,
                    q2 + qdot2 * dt,
                    q3 + qdot3 * dt,
                    q4 + qdot4 * dt,
                ],
                dtype=np.float64,
            )
        )
        return self.q_wb

    def quaternion(self):
        return self.q_wb.copy()

    def R(self):
        return quat_to_matrix(self.q_wb)


def synthetic_check(seconds=30.0, fs=200.0, beta=0.05):
    dt = 1.0 / fs
    n = int(seconds * fs)
    filt = Madgwick(beta=beta)
    yaw_rate = math.radians(6.0)
    q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    def qmul(a, b):
        aw, ax, ay, az = a
        bw, bx, by, bz = b
        return np.array(
            [
                aw * bw - ax * bx - ay * by - az * bz,
                aw * bx + ax * bw + ay * bz - az * by,
                aw * by - ax * bz + ay * bw + az * bx,
                aw * bz + ax * by - ay * bx + az * bw,
            ],
            dtype=np.float64,
        )

    def rotate_world_to_body(q_wb, v):
        R = quat_to_matrix(q_wb)
        return R.T @ v

    gravity = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    for _ in range(n):
        wq = np.array([0.0, 0.0, 0.0, yaw_rate], dtype=np.float64)
        q = quat_normalize(q + 0.5 * qmul(q, wq) * dt)
        accel = rotate_world_to_body(q, gravity)
        filt.step(np.array([0.0, 0.0, yaw_rate]), accel, dt)

    R_err = filt.R().T @ quat_to_matrix(q)
    trace = np.clip((np.trace(R_err) - 1.0) * 0.5, -1.0, 1.0)
    return math.degrees(math.acos(trace))


if __name__ == "__main__":
    err = synthetic_check()
    print("synthetic rotation error deg:", err)
