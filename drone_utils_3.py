import numpy as np

def state_dot(state, control, iner_x, iner_y, iner_z, L, lam, g, m):
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
    pdot = ((iner_y - iner_z) / iner_x) * q * r + L / iner_x * (control[2] + control[3] - control[0] - control[1])
    qdot = ((iner_z - iner_x) / iner_y) * p * r + L / iner_y * (control[0] + control[2] - control[1] - control[3])
    rdot = ((iner_x - iner_y) / iner_z) * p * q + lam / iner_z * (control[1] + control[2] - control[0] - control[3])

    
    # Euler angle rates
    psidot = (q * sin_phi + r * cos_phi) / cos_theta
    thetadot = q * cos_phi - r * sin_phi
    phidot = (q * sin_theta * sin_phi + r * sin_theta * cos_phi) / cos_theta + p
    
    # Linear acceleration
    udot = r * v - q * w - g * np.sin(theta)
    vdot = p * w - r * u + g * cos_theta * sin_phi
    wdot = g * u - p * v + g * cos_theta * cos_phi - np.sum(control) / m
    
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

def costate_dot(state, costate, iner_x, iner_y, iner_z, g, alpha, beta, gamma, delta):
    # Unpack state variables
    p, q, r = state[0], state[1], state[2]
    theta, phi, psi = state[3], state[4], state[5]
    u, v, w = state[6], state[7], state[8]
    x, y, z = state[9], state[10], state[11]
    
    # Unpack costate components (ρ₁,...,ρ₁₂)
    rh1, rh2, rh3, rh4, rh5, rh6, rh7, rh8, rh9, rh10, rh11, rh12 = costate
    
    # Precompute trigonometric functions and derived functions
    sin_theta, cos_theta = np.sin(theta), np.cos(theta)
    tan_theta, sec_theta = np.tan(theta), 1/np.cos(theta)
    sin_phi, cos_phi = np.sin(phi), np.cos(phi)
    sin_psi, cos_psi = np.sin(psi), np.cos(psi)
    
    dot = np.zeros(12)
    
    dot[0] = (iner_x * r * rh2)/iner_y - (iner_z * r * rh2)/iner_y \
             - (iner_x * q * rh3)/iner_z + (iner_y * q * rh3)/iner_z \
             - rh6 - w * rh8 + v * rh9

    dot[1] = - (iner_y * r * rh1)/iner_x + (iner_z * r * rh1)/iner_x \
             - (iner_x * p * rh3)/iner_z + (iner_y * p * rh3)/iner_z \
             - sec_theta * sin_phi * rh4 - cos_phi * rh5 \
             - sin_phi * tan_theta * rh6 + w * rh7 - u * rh9

    dot[2] = - (iner_y * q * rh1)/iner_x + (iner_z * q * rh1)/iner_x \
             + (iner_x * p * rh2)/iner_y - (iner_z * p * rh2)/iner_y \
             + sin_phi * rh5 - cos_phi * sec_theta * (rh4 + sin_theta * rh6) \
             - v * rh7 + u * rh8

    dot[3] = 2 * alpha * psi + u * cos_theta * sin_psi * rh10 \
             + v * sin_theta * sin_phi * sin_psi * rh10 \
             - w * sin_phi * sin_psi * rh11 \
             - cos_psi * (w * sin_phi * rh10 + u * cos_theta * rh11 + v * sin_theta * sin_phi * rh11) \
             + cos_phi * ( sin_psi * (w * sin_theta * rh10 + v * rh11) \
             + cos_psi * (v * rh10 - w * sin_theta * rh11) )

    dot[4] = 2 * beta * theta - q * sec_theta * sin_phi * tan_theta * rh4 \
             - q * sec_theta**2 * sin_phi * rh6 + g * cos_theta * rh7 \
             + u * cos_psi * sin_theta * rh10 + u * sin_theta * sin_psi * rh11 \
             - u * cos_theta * rh12 \
             - cos_theta * sin_phi * (-g * tan_theta * rh8 + v * cos_psi * rh10 \
             + v * sin_psi * rh11 + v * tan_theta * rh12) \
             - 0.25 * cos_phi * ( 4 * r * sec_theta**2 * rh6 \
             + sin_theta * (-3 * g * rh9 - 4 * w * tan_theta * (cos_psi * rh10 + sin_psi * rh11) \
             + 3 * w * rh12 + tan_theta**2 * (g * rh9 - w * rh12)) \
             + sec_theta * (4 * w * (cos_psi * rh10 + sin_psi * rh11) \
             + tan_theta * (4 * r * rh4 - g * rh9 + w * rh12)) )

    dot[5] = 2 * beta * phi - q * cos_phi * sec_theta * rh4 \
             + r * cos_phi * rh5 - q * cos_phi * tan_theta * rh6 \
             - g * cos_theta * cos_phi * rh8 \
             - v * cos_phi * cos_psi * sin_theta * rh10 \
             - w * cos_phi * sin_psi * rh10 + w * cos_phi * cos_psi * rh11 \
             - v * cos_phi * sin_theta * sin_psi * rh11 \
             + v * cos_theta * cos_phi * rh12 \
             + 0.25 * cos_theta * sin_phi * ( 3 * g * rh9 \
             + 4 * sec_theta * (q * rh5 + r * tan_theta * rh6 - v * sin_psi * rh10 + v * cos_psi * rh11) \
             + 4 * w * tan_theta * (cos_psi * rh10 + sin_psi * rh11) \
             - 3 * w * rh12 + sec_theta**2 * (4 * r * rh4 + g * rh9 - w * rh12) \
             + tan_theta**2 * (-g * rh9 + w * rh12) )

    dot[6] = r * rh8 - q * rh9 - cos_theta * (cos_psi * rh10 + sin_psi * rh11) - sin_theta * rh12

    dot[7] = - r * rh7 + p * rh9 + cos_phi * sin_psi * rh10 \
             - sin_theta * sin_phi * sin_psi * rh11 \
             - cos_psi * (sin_theta * sin_phi * rh10 + cos_phi * rh11) \
             + cos_theta * sin_phi * rh12

    dot[8] = q * rh7 - p * rh8 + sin_phi * (-sin_psi * rh10 + cos_psi * rh11) \
             - cos_phi * (cos_psi * sin_theta * rh10 + sin_theta * sin_psi * rh11 - cos_theta * rh12)

    dot[9]  = 2 * x * gamma
    dot[10] = 2 * y * gamma
    dot[11] = 2 * z * delta

    return dot


def optimal_control(costate, inerx, inery, inerz, L, lam, m):
    denom = 2 * inerx * inery * inerz * m
    c1 = (-inery * inerz * L * m * costate[1] +
          inerx * inerz * L * m * costate[2] -
          inerx * inery * m * lam * costate[3] -
          inerx * inery * inerz * costate[9]) / denom

    c2 = (-inery * inerz * L * m * costate[1] -
          inerx * inerz * L * m * costate[2] +
          inerx * inery * m * lam * costate[3] -
          inerx * inery * inerz * costate[9]) / denom

    c3 = (inery * inerz * L * m * costate[1] +
          inerx * inerz * L * m * costate[2] +
          inerx * inery * m * lam * costate[3] -
          inerx * inery * inerz * costate[9]) / denom

    c4 = (inery * inerz * L * m * costate[1] -
          inerx * inerz * L * m * costate[2] -
          inerx * inery * m * lam * costate[3] -
          inerx * inery * inerz * costate[9]) / denom

    return (c1, c2, c3, c4)




