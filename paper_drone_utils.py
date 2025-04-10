import numpy as np
from numpy import cos, sin, tan
from numpy.linalg import inv
from scipy.linalg import solve_continuous_are

# Model Parameters
m = 0.468  # mass
L = 0.225  # distance from center of mass to propeller
K = 2.980 * 10**-6  # thrust coefficient
B = 0.14 * 10 ** (-6)  # drag coefficient
g = 9.81  # gravity
I = np.diag(
    [
        4.856 * 10**-3,  # moment of inertia x
        4.856 * 10**-3,  # moment of inertia y
        8.801 * 10**-3,  # moment of inertia z
    ]
)
Ir = 3.357 * 10**-5
aero_drag = np.array([0.3, 0.3, 0.25])
Ar = 0  # TODO: is this right?
max_rpm = 10_000
max_rads = (max_rpm * 2 * np.pi) / 60

G = np.array([0, 0, -g])  # gravity vector


def rotation_matrix(phi, theta, psi):
    return np.array(
        [
            [
                cos(psi) * cos(theta),
                cos(psi) * sin(theta) * sin(phi) - sin(psi) * cos(phi),
                cos(psi) * sin(theta) * cos(phi) + sin(psi) * sin(phi),
            ],
            [
                sin(psi) * cos(theta),
                sin(psi) * sin(theta) * sin(phi) + cos(psi) * cos(phi),
                sin(psi) * sin(theta) * cos(phi) - cos(psi) * sin(phi),
            ],
            [
                -sin(theta),
                cos(theta) * sin(phi),
                cos(theta) * cos(phi),
            ],
        ]
    )


def body_to_inertial(phi, theta, psi):
    return np.array(
        [
            [1, sin(phi) * tan(theta), cos(phi) * tan(theta)],
            [0, cos(phi), -sin(phi)],
            [0, sin(phi) / cos(theta), cos(phi) / cos(theta)],
        ]
    )


def state_dot(t, y, rotor_ctrl_fn):
    rotor_ctrl = rotor_ctrl_fn(t, y)
    X, Y, Z, phi, theta, psi, U, V, W, P, Q, R = y
    linear_velocity = np.array([U, V, W])
    angular_velocity = np.array([P, Q, R])

    # Angular velocity
    phi_dot, theta_dot, psi_dot = body_to_inertial(phi, theta, psi) @ angular_velocity

    # Thrust components
    T = K * rotor_ctrl**2
    moment_body = np.array(
        [
            # Phi/Roll moment, caused by opposing x-axis rotors
            L * (T[3] - T[1]),
            # Theta/Pitch moment, caused by opposing y-axis rotors
            L * (T[2] - T[0]),
            # Psi/Yaw moment, caused by drag force on all rotors
            B
            * (
                -(rotor_ctrl[0] ** 2)
                + rotor_ctrl[1] ** 2
                - rotor_ctrl[2] ** 2
                + rotor_ctrl[3] ** 2
            ),
        ]
    )
    moment_drag = Ar * angular_velocity
    # Net thrust acting on body
    net_thrust = np.array([0, 0, np.sum(T)])
    # Drag force on the body (Fd)
    drag_body = aero_drag * linear_velocity
    U_dot, V_dot, W_dot = (
        G + (rotation_matrix(phi, theta, psi) @ net_thrust - drag_body) / m
    )

    I_inv = inv(I)
    P_dot, Q_dot, R_dot = I_inv @ (
        moment_body
        - moment_drag
        - np.cross(angular_velocity, I @ angular_velocity)
        # Gyroscopic effect
        - Ir
        * np.cross(
            angular_velocity,
            np.array(
                [0, 0, -rotor_ctrl[0] + rotor_ctrl[1] - rotor_ctrl[2] + rotor_ctrl[3]]
            ),
        )
    )

    return np.array(
        [
            # Linear velocity
            U,
            V,
            W,
            # Angular velocity, inertial
            phi_dot,
            theta_dot,
            psi_dot,
            # Linear Acceleration
            U_dot,
            V_dot,
            W_dot,
            # Angular Acceleration
            P_dot,
            Q_dot,
            R_dot,
        ]
    )


class LQR:
    def __init__(self):
        # LQR
        # NOTE: LQR is all based on the system as a function of u^2
        A_sub = np.block(
            [
                [
                    np.zeros((4, 4)),
                    np.array(
                        [
                            [1, 0, 0, 0],
                            # TODO: more specific angles here?
                            [0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1],
                        ]
                    ),
                ],
                [
                    np.zeros((4, 4)),
                    np.diag(
                        [-aero_drag[2] / m, -Ar / I[0, 0], -Ar / I[1, 1], -Ar / I[2, 2]]
                    ),
                ],
            ]
        )
        B_sub = np.vstack(
            [
                np.zeros((4, 4)),
                [
                    np.ones(4) / m,
                    [0, -K * L, 0, K * L] / I[0, 0],
                    [0, -K * L, 0, K * L] / I[1, 1],
                    [-B, B, -B, B] / I[2, 2],
                ],
            ]
        )

        Q_sub = np.diag([100, 10, 10, 10, 9.427, 0.1703, 0.1703, 0.0267])
        R = np.eye(4)
        P = solve_continuous_are(A_sub, B_sub, Q_sub, R)
        self.K = inv(R) @ B_sub.T @ P

    def get_control(self, x):
        # Extract Z, euler angles, and angular velocities
        # Goal is to go to specific Z, and stable angle
        x_relevant = x[[2, 3, 4, 5, 8, 9, 10, 11]]
        u = self.K @ x_relevant
        u = np.sqrt(np.maximum(u, 0))

        return np.clip(u, 0, max_rads)
