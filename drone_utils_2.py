import numpy as np

import numpy as np

def state_dot(state, c, inerx, inery, inerz, L, lam, g, m):
    # Unpack the state vector
    p, q, r, psi, theta, phi, u, v, w, x, y, z = state
    
    # Precompute trigonometric functions
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    sin_psi = np.sin(psi)
    cos_psi = np.cos(psi)
    
    # Avoid division by zero for cos(theta)
    #if np.isclose(cos_theta, 0):
    #    raise ValueError("cos(theta) is too close to zero, possible singularity.")
    
    # Angular rates
    pdot = ((inery - inerz) / inerx) * q * r + L / inerx * (c[2] + c[3] - c[0] - c[1])
    qdot = ((inerz - inerx) / inery) * p * r + L / inery * (c[0] + c[2] - c[1] - c[3])
    rdot = ((inerx - inery) / inerz) * p * q + lam / inerz * (c[1] + c[2] - c[0] - c[3])

    
    # Euler angle rates
    psidot = (q * sin_phi + r * cos_phi) / cos_theta
    thetadot = q * cos_phi - r * sin_phi
    phidot = (q * sin_theta * sin_phi + r * sin_theta * cos_phi) / cos_theta + p
    
    # Linear acceleration
    udot = r * v - q * w - g * np.sin(theta)
    vdot = p * w - r * u + g * cos_theta * sin_phi
    wdot = g * u - p * v + g * cos_theta * cos_phi - np.sum(c) / m
    
    # Position rates (inertial frame)
    xdot = (u * cos_phi * cos_psi +
            v * (sin_phi * sin_theta * cos_psi - cos_phi * sin_psi) +
            w * (cos_phi * sin_theta * cos_psi + sin_phi * sin_psi))
    
    ydot = (u * cos_theta * sin_psi +
            v * (sin_phi * sin_theta * sin_psi + cos_phi * cos_psi) +
            w * (cos_phi * sin_theta * sin_psi - sin_phi * cos_psi))
    
    zdot = (u * sin_theta -
            v * sin_phi * cos_theta -
            w * cos_phi * cos_theta)
    
    return np.array([pdot, qdot, rdot,
                     psidot, thetadot, phidot,
                     udot, vdot, wdot,
                     xdot, ydot, zdot])

def costate_dot(state, costate, inerx, inery, inerz, g, alpha=0, beta=0, gamma=0):
    # Unpack state and costate
    p, q, r, psi, theta, phi, u, v, w, x, y, z = state
    rho = costate

    # Trig shortcuts
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    tan_theta = np.tan(theta)
    sec_theta = 1 / cos_theta
    sin_psi = np.sin(psi)
    cos_psi = np.cos(psi)

    # Derivatives of costate
    drho1 = (inerx * r * rho[1]) / inery - (inerz * r * rho[1]) / inery - \
            (inerx * q * rho[2]) / inerz + (inery * q * rho[2]) / inerz - \
            rho[5] - w * rho[7] + v * rho[8]
    
    drho2 = -((inery * r * rho[0]) / inerx) + (inerz * r * rho[0]) / inerx - \
            (inerx * p * rho[2]) / inerz + (inery * p * rho[2]) / inerz - \
            sec_theta * sin_phi * rho[3] - cos_phi * rho[4] - \
            sin_phi * tan_theta * rho[5] + w * rho[6]
    
    drho3 = -((inery * q * rho[0]) / inerx) + (inerz * q * rho[0]) / inerx + \
            (inerx * p * rho[1]) / inery - (inerz * p * rho[1]) / inery + \
            sin_phi * rho[4] - cos_phi * sec_theta * (rho[3] + sin_theta * rho[5]) - \
            v * rho[6] + u * rho[7]
    
    drho4 = 2 * alpha * psi + \
            v * sin_theta * sin_phi * sin_psi * rho[9] - \
            w * sin_phi * sin_psi * rho[10] - \
            cos_psi * (w * sin_phi * rho[9] + u * cos_theta * rho[10] + v * sin_theta * sin_phi * rho[10]) + \
            cos_phi * (sin_psi * (u * rho[9] + w * sin_theta * rho[9] + v * rho[10]) + \
                       cos_psi * (v * rho[9] - w * sin_theta * rho[10]))
    
    drho5 = 2 * beta * theta - \
            q * sec_theta * sin_phi * tan_theta * rho[3] - \
            q * sec_theta**2 * sin_phi * rho[5] + \
            g * cos_theta * rho[6] + \
            u * sin_theta * sin_psi * rho[10] - \
            u * cos_theta * rho[11] - \
            cos_theta * sin_phi * (-g * tan_theta * rho[7] + \
            v * cos_psi * rho[9] + v * sin_psi * rho[10] + v * tan_theta * rho[11]) - \
            0.25 * cos_phi * (
                4 * r * sec_theta**2 * rho[5] + \
                sin_theta * (-3 * g * rho[8] - 4 * w * tan_theta * (cos_psi * rho[9] + sin_psi * rho[10]) + \
                             3 * w * rho[11] + tan_theta**2 * (g * rho[8] - w * rho[11])) + \
                sec_theta * (4 * w * (cos_psi * rho[9] + sin_psi * rho[10]) + \
                             tan_theta * (4 * r * rho[3] - g * rho[8] + w * rho[11]))
            )
    
    drho6 = 2 * beta * phi - q * cos_phi * sec_theta * rho[3] + \
            r * cos_phi * rho[4] - q * cos_phi * tan_theta * rho[5] - \
            g * cos_theta * cos_phi * rho[7] - \
            v * cos_phi * cos_psi * sin_theta * rho[9] - \
            w * cos_phi * sin_psi * rho[9] + \
            w * cos_phi * cos_psi * rho[10] - \
            v * cos_phi * sin_theta * sin_psi * rho[10] + \
            v * cos_theta * cos_phi * rho[11] + \
            0.25 * cos_theta * sin_phi * (
                3 * g * rho[8] + \
                4 * sec_theta * (q * rho[4] + r * tan_theta * rho[5] + \
                                 u * cos_psi * rho[9] - v * sin_psi * rho[9] + \
                                 v * cos_psi * rho[10]) + \
                4 * w * tan_theta * (cos_psi * rho[9] + sin_psi * rho[10]) - \
                3 * w * rho[11] + \
                sec_theta**2 * (4 * r * rho[3] + g * rho[8] - w * rho[11]) + \
                tan_theta**2 * (-g * rho[8] + w * rho[11])
            )
    
    drho7 = r * rho[7] - g * rho[8] - \
            cos_phi * cos_psi * rho[9] - \
            cos_theta * sin_psi * rho[10] - \
            sin_theta * rho[11]
    
    drho8 = -r * rho[6] + p * rho[8] + \
            cos_phi * sin_psi * rho[9] - \
            sin_theta * sin_phi * sin_psi * rho[10] - \
            cos_psi * (sin_theta * sin_phi * rho[9] + cos_phi * rho[10]) + \
            cos_theta * sin_phi * rho[11]
    
    drho9 = q * rho[6] - p * rho[7] + \
            sin_phi * (-sin_psi * rho[9] + cos_psi * rho[10]) - \
            cos_phi * (cos_psi * sin_theta * rho[9] + \
                       sin_theta * sin_psi * rho[10] - \
                       cos_theta * rho[11])
    
    drho10 = 2 * x * gamma
    drho11 = 2 * y * gamma
    drho12 = 2 * z * gamma

    return np.array([
        drho1, drho2, drho3,
        drho4, drho5, drho6,
        drho7, drho8, drho9,
        drho10, drho11, drho12
    ])

import numpy as np

def optimal_control(rho, inerx, inery, inerz, L, lam, m):
    # Precompute denominator
    denom = 2 * inerx * inery * inerz * m

    # Extract costates
    rho1, rho2, rho3, _, _, _, _, _, rho9 = rho[:9]

    # Compute each control input
    c1 = (-inery * inerz * L * m * rho1 +
           inerx * inerz * L * m * rho2 -
           inerx * inery * m * lam * rho3 -
           inerx * inery * inerz * rho9) / denom

    c2 = (-inery * inerz * L * m * rho1 -
           inerx * inerz * L * m * rho2 +
           inerx * inery * m * lam * rho3 -
           inerx * inery * inerz * rho9) / denom

    c3 = ( inery * inerz * L * m * rho1 +
           inerx * inerz * L * m * rho2 +
           inerx * inery * m * lam * rho3 -
           inerx * inery * inerz * rho9) / denom

    c4 = ( inery * inerz * L * m * rho1 -
           inerx * inerz * L * m * rho2 -
           inerx * inery * m * lam * rho3 -
           inerx * inery * inerz * rho9) / denom

    return np.array([c1, c2, c3, c4])



