import numpy as np
from numpy import cos, sin
from numpy.linalg import inv

# Elements of sigma vector:
# 0:  x
# 1:  y
# 2:  z
# 3:  phi
# 4:  theta
# 5:  psi
# 6:  x'
# 7:  y'
# 8:  z'
# 9:  phi'
# 10: theta'
# 11: psi'




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

def mass_inertia_matrix(mass=10):
    return np.diag([mass, mass, mass, 1, 1, 1]) # TODO: Make a more intelligent intertia matrix

def gamma(i, lambda_):
    return np.array([
        [0, 1, lambda_],
        [1, 0, -lambda_],
        [0, -1, lambda_],
        [-1, 0, -lambda_]
    ])[i]

def control_from_state(state, costate, lambda_):
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


    # This is the vector with the sub i-6 after it in the derivation of the control, the thing in square brackets
    def sum_vector(control_index):
        return MI_inv @ np.hstack([R[:, 2], R @ gamma(control_index, lambda_)])
    
    return 1/2 * np.array([
        np.sum(costate[6:] * sum_vector(control_index)) # This is elementwise
        for control_index in range(4)
    ])


def D_norm_u_sigma(k, state, costate, lambda_):
    """
    This function computes the first term in the partial derivative of the costate vector.
    In other words the derivative of the norm of u squared with respect to the kth element of the state vector.
    """
    if k >= 6 or k < 3:
        return 0
        
    MI_inv = inv(mass_inertia_matrix())

    # Unpack angles
    phi, theta, psi = state[3:6]

    if k == 3: # Derivative with respect to phi
        R_der = rotation_phi(phi, theta, psi)
    elif k == 4:
        R_der = rotation_theta(phi, theta, psi)
    elif k == 5:
        R_der = rotation_psi(phi, theta, psi)
    
    control = control_from_state(state, costate, lambda_) # First sum term
    
    def sum_vector(control_index):
        return MI_inv @ np.hstack([R_der[:, 2], R_der @ gamma(control_index, lambda_)])
        
    second_sum_term = np.array([
        np.sum(costate[6:] * sum_vector(control_index)) # This is elementwise
        for control_index in range(4)
    ])

    return np.sum(control * second_sum_term) # This is the sum of the first term and the second term

def D_norm_s_sigma(k, state, alpha, lambda_):
    if k >= 6:
        return 0
    else:
        return 2 * alpha * state[k]

def costate_prime(state, costate, alpha, lambda_):
    return np.array([
        D_norm_u_sigma(k, state, costate, lambda_) + D_norm_s_sigma(k, state, alpha, lambda_)
        for k in range(12)
    ])

def position_angle_double_prime(state, costate, lambda_):
    """
    This function computes the second derivative of the state vector.
    """
    phi   = state[3]
    theta = state[4]
    psi   = state[5]

    R = rotation(phi, theta, phi)
    MI = mass_inertia_matrix()

    control = control_from_state(state, costate, lambda_)
    
    return MI @ np.hstack([
        R[:, 2] * np.sum(control), 
        R @ [
            control[1] - control[3], 
            control[0] - control[2], 
            lambda_ * (control[0] + control[2] - control[1] - control[3])
        ]
    ])

def state_prime(state, costate, lambda_):
    """sigma dot"""
    return np.hstack([
        state[6:],
        position_angle_double_prime(state[:6], costate, lambda_)
    ])