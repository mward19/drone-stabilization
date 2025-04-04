import numpy as np
from numpy import cos, sin
from numpy.linalg import inv

def rotation(phi, theta, psi):
    return np.array([
        [
            cos(theta) * cos(psi), 
            cos(theta) * sin(psi), 
            -sin(theta)
        ], 
        [
            sin(phi) * sin(theta) * cos(psi) - cos(phi) * sin(psi), 
            sin(phi) * sin(theta) * sin(psi) + cos(phi) * cos(psi), 
            sin(phi) * cos(theta)
        ],
        [
            cos(phi) * sin(theta) * cos(psi) + sin(phi) * sin(psi),
            cos(phi) * sin(theta) * sin(psi) - sin(phi) * cos(psi),
            cos(phi) * cos(theta)
        ]])

def rotation_phi(phi, theta, psi):
    # Deriviating of the rotation matrix with respect to phi
    return np.array([
        [
            0,0,0
        ],
        [
            cos(phi) * cos(psi) * sin(theta) * sin(phi) * sin(psi),
            -cos(psi) * sin(phi) + cos(phi) * sin(theta) * sin(psi),
            cos(theta) * cos(phi)
        ],
        [
            -cos(psi) * sin(theta) * sin(phi) + cos(phi) * sin(psi),
            -cos(phi) * cos(psi) - sin(theta) * sin(phi) * sin(psi),
            -cos(theta) * sin(phi)
        ]
    ])

def rotation_theta(phi, theta, psi):
    # Derivative of the rotation matrix with respect to theta
    return np.array([
        [
            -cos(psi) * sin(theta),
            -sin(theta) * sin(psi),
            -cos(theta)
        ],
        [
            cos(theta) * cos(psi) * sin(phi),
            cos(theta) * sin(phi) * sin(psi),
            -sin(theta) * sin(phi)
        ],
        [
            cos(theta) * cos(phi) * cos(psi),
            cos(theta) * cos(phi) * sin(psi),
            -cos(phi) * sin(theta)
        ]
    ])

def rotation_psi(phi, theta, psi):
    # Derivative of the rotation matrix with respect to psi
    return np.array([
        [
           -cos(theta) * sin(psi),
            cos(theta) * cos(psi),
            0
        ],
        [
           -cos(phi) * cos(psi) - sin(theta) * sin(phi) * sin(psi),
            cos(psi) * sin(theta) * sin(phi) - cos(phi) * sin(psi),
            0
        ],
        [
            cos(psi) * sin(phi) - cos(phi) * sin(theta) * sin(psi),
            cos(phi) * cos(psi) * sin(theta) + sin(phi) * sin(psi),
            0
        ]
    ])

def mass_inertia_matrix(mass=1):
    return np.diag([mass, mass, mass, 1, 1, 1]) # TODO: Make a more intelligent intertia matrix

def control(state, costate, lambda_):
    """
    `state`: The full 12 dimensional state, called `sigma` in derivation.
    `costate`: Associated costate, called `p` in derivation.

    'state' = (x, y, z, phi, theta, psi)
    """
    phi   = state[3]
    theta = state[4]
    psi   = state[5]

    R = rotation(phi, theta, phi)
    MI_inv = inv(mass_inertia_matrix())

    gamma = np.array([
        [0, 1, lambda_],
        [1, 0, -lambda_],
        [0, -1, lambda_],
        [-1, 0, -lambda_]
    ])

    # This is the vector with the sub i-6 after it in the derivation of the control
    def sum_vector(control_index):
        return MI_inv @ np.hstack([R[:, 2], R @ gamma[control_index]])
    
    control = 1/2 * np.array([
        np.sum(costate[6:12] * sum_vector(control_index))
        for control_index in range(4)
    ])
    
    return control
    