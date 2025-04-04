import numpy as np
from numpy import cos
from numpy import sin

def rotation_matrix(phi, theta, psi):
    rotation_matrix = np.array([
        [
            cos(theta) * cos(psi), 
            cos(theta) * sin(psi), 
            -sin(theta)
        ], 
        [
            sin(phi) * sin(theta) * cos(psi) - cos(phi) * sin(psi), 
            sin(phi) * sin(theta) * sin(psi) + cos(theta) * cos(psi), 
            sin(phi) * cos(theta)
        ],
        [
            cos(phi) * sin(theta) * cos(psi) + sin,
        ]])
    pass

def control(state, costate):
    """
    `state`: The full 12 dimensional state, called `sigma` in derivation.
    `costate`: Associated costate, called `p` in derivation.
    """
    phi   = state[3]
    theta = statte[4]
    psi   = state[5]

    R = rotation_matrix()