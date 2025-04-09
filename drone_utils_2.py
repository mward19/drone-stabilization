import numpy as np

def state_dot(state, c, inerx, inery, inerz, L, lam, g, m):
    """
    Compute the time derivative of the state vector state.
    
    Parameters:
        state: array-like, shape (12,)
            [p, q, r, psi, theta, phi, u, v, w, x, y, z]
        c: array-like, shape (4,)
            Control inputs c1 through c4
        inerx, inery, inerz: float
            Moments of inertia
        L: float
            Arm length
        lam: float
            A constant related to torque (lambda)
        g: float
            Gravitational acceleration
        m: float
            Mass of the object

    Returns:
        np.ndarray, shape (12,)
            Time derivative of state
    """
    p, q, r, psi, theta, phi, u, v, w, x, y, z = state
    c1, c2, c3, c4 = c

    # Precompute trigonometric values
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    sin_psi = np.sin(psi)
    cos_psi = np.cos(psi)

    # Angular velocities derivatives
    dp = ((inery - inerz) / inerx) * q * r + L / inerx * (c3 + c4 - c1 - c2)
    dq = ((inerz - inerx) / inery) * p * r + L / inery * (c1 + c3 - c2 - c4)
    dr = ((inerx - inery) / inerz) * p * q + (lam / inerz) * (c2 + c3 - c1 - c4)

    # Euler angles derivatives
    dpsi = (1 / cos_theta) * (q * sin_phi + r * cos_phi)
    dtheta = q * cos_phi - r * sin_phi
    dphi = (1 / cos_theta) * (q * sin_theta * sin_phi + r * sin_theta * cos_phi) + p

    # Linear velocities derivatives
    du = r * v - q * w - g * sin_theta
    dv = p * w - r * u + g * cos_theta * sin_phi
    dw = g * u - p * v + g * cos_theta * cos_phi - (np.sum(c)) / m

    # Position derivatives
    dx = u * cos_phi * cos_psi \
         + v * (sin_phi * sin_theta * cos_psi - cos_phi * sin_psi) \
         + w * (cos_phi * sin_theta * cos_psi + sin_phi * sin_psi)

    dy = u * cos_theta * sin_psi \
         + v * (sin_phi * sin_theta * sin_psi + cos_phi * cos_psi) \
         + w * (cos_phi * sin_theta * sin_psi - sin_phi * cos_psi)

    dz = u * sin_theta - v * sin_phi * cos_theta - w * cos_phi * cos_theta

    return np.array([dp, dq, dr, dpsi, dtheta, dphi, du, dv, dw, dx, dy, dz])

def costate_dot(state, costate, inerx, inery, inerz, g):
    """
    Compute the time derivative of the costate vector costate.

    Parameters:
        state: array-like, shape (12,)
            [p, q, r, psi, theta, phi, u, v, w, x, y, z]
        costate: array-like, shape (12,)
            Costate vector [rho1, ..., rho12]
        inerx, inery, inerz: float
            Moments of inertia
        g: float
            Gravitational acceleration

    Returns:
        np.ndarray, shape (12,)
            Time derivative of the costate vector
    """
    p, q, r, psi, theta, phi, u, v, w, x, y, z = state
    rho1, rho2, rho3, rho4, rho5, rho6, rho7, rho8, rho9, rho10, rho11, rho12 = costate

    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    tan_theta = np.tan(theta)
    sec_theta = 1 / cos_theta
    sec_theta2 = sec_theta ** 2

    sin_psi = np.sin(psi)
    cos_psi = np.cos(psi)

    # Each component
    dot1 = (inerx * r * rho2) / inery - (inerz * r * rho2) / inery - \
           (inerx * q * rho3) / inerz + (inery * q * rho3) / inerz - \
           rho6 - w * rho8 + v * rho9

    dot2 = - (inery * r * rho1) / inerx + (inerz * r * rho1) / inerx - \
           (inerx * p * rho3) / inerz + (inery * p * rho3) / inerz - \
           sec_theta * sin_phi * rho4 - cos_phi * rho5 - \
           sin_phi * tan_theta * rho6 + w * rho7

    dot3 = - (inery * q * rho1) / inerx + (inerz * q * rho1) / inerx + \
           (inerx * p * rho2) / inery - (inerz * p * rho2) / inery + \
           sin_phi * rho5 - cos_phi * sec_theta * (rho4 + sin_theta * rho6) - \
           v * rho7 + u * rho8

    dot4 = 2 * psi + \
           v * sin_theta * sin_phi * sin_psi * rho10 - \
           w * sin_phi * sin_psi * rho11 - \
           cos_psi * (w * sin_phi * rho10 + u * cos_theta * rho11 + v * sin_theta * sin_phi * rho11) + \
           cos_phi * (
               sin_psi * (u * rho10 + w * sin_theta * rho10 + v * rho11) +
               cos_psi * (v * rho10 - w * sin_theta * rho11)
           )

    dot5 = 2 * theta - \
           q * sec_theta * sin_phi * tan_theta * rho4 - \
           q * sec_theta2 * sin_phi * rho6 + \
           g * cos_theta * rho7 + \
           u * sin_theta * sin_psi * rho11 - \
           u * cos_theta * rho12 - \
           cos_theta * sin_phi * (-g * tan_theta * rho8 + v * cos_psi * rho10 + v * sin_psi * rho11 + v * tan_theta * rho12) - \
           0.25 * cos_phi * (
               4 * r * sec_theta2 * rho6 + 
               sin_theta * (
                   -3 * g * rho9 - 4 * w * tan_theta * (cos_psi * rho10 + sin_psi * rho11) +
                   3 * w * rho12 + tan_theta ** 2 * (g * rho9 - w * rho12)
               ) +
               sec_theta * (4 * w * (cos_psi * rho10 + sin_psi * rho11) +
                            tan_theta * (4 * r * rho4 - g * rho9 + w * rho12))
           )

    dot6 = 2 * phi - \
           q * cos_phi * sec_theta * rho4 + \
           r * cos_phi * rho5 - \
           q * cos_phi * tan_theta * rho6 - \
           g * cos_theta * cos_phi * rho8 - \
           v * cos_phi * cos_psi * sin_theta * rho10 - \
           w * cos_phi * sin_psi * rho10 + \
           w * cos_phi * cos_psi * rho11 - \
           v * cos_phi * sin_theta * sin_psi * rho11 + \
           v * cos_theta * cos_phi * rho12 + \
           0.25 * cos_theta * sin_phi * (
               3 * g * rho9 +
               4 * sec_theta * (
                   q * rho5 + r * tan_theta * rho6 + u * cos_psi * rho10 -
                   v * sin_psi * rho10 + v * cos_psi * rho11
               ) +
               4 * w * tan_theta * (cos_psi * rho10 + sin_psi * rho11) -
               3 * w * rho12 +
               sec_theta2 * (4 * r * rho4 + g * rho9 - w * rho12) +
               tan_theta ** 2 * (-g * rho9 + w * rho12)
           )

    dot7 = r * rho8 - g * rho9 - cos_phi * cos_psi * rho10 - \
           cos_theta * sin_psi * rho11 - sin_theta * rho12

    dot8 = -r * rho7 + p * rho9 + \
           cos_phi * sin_psi * rho10 - \
           sin_theta * sin_phi * sin_psi * rho11 - \
           cos_psi * (sin_theta * sin_phi * rho10 + cos_phi * rho11) + \
           cos_theta * sin_phi * rho12

    dot9 = q * rho7 - p * rho8 + \
           sin_phi * (-sin_psi * rho10 + cos_psi * rho11) - \
           cos_phi * (cos_psi * sin_theta * rho10 +
                      sin_theta * sin_psi * rho11 -
                      cos_theta * rho12)

    dot10 = 2 * x
    dot11 = 2 * y
    dot12 = 2 * z

    return np.array([
        dot1, dot2, dot3, dot4, dot5, dot6,
        dot7, dot8, dot9, dot10, dot11, dot12
    ])

def optimal_control(costate, inerx, inery, inerz, L, lam, m):
    """
    Compute the optimal control inputs c1, c2, c3, c4.

    Parameters:
        costate: array-like, shape (12,)
            Costate vector [rho1, ..., rho12]
        inerx, inery, inerz: float
            Moments of inertia
        L: float
            Arm length
        lam: float
            A constant related to torque (lambda)
        m: float
            Mass of the object

    Returns:
        np.ndarray, shape (4,)
            Optimal control inputs [c1, c2, c3, c4]
    """
    rho1, rho2, rho3, _, _, _, _, _, rho9, _, _, _ = costate

    denom = 2 * inerx * inery * inerz * m
    c1 = (-inery * inerz * L * m * rho1 +
          inerx * inerz * L * m * rho2 -
          inerx * inery * m * lam * rho3 -
          inerx * inery * inerz * rho9) / denom

    c2 = (-inery * inerz * L * m * rho1 -
          inerx * inerz * L * m * rho2 +
          inerx * inery * m * lam * rho3 -
          inerx * inery * inerz * rho9) / denom

    c3 = (inery * inerz * L * m * rho1 +
          inerx * inerz * L * m * rho2 +
          inerx * inery * m * lam * rho3 -
          inerx * inery * inerz * rho9) / denom

    c4 = (inery * inerz * L * m * rho1 -
          inerx * inerz * L * m * rho2 -
          inerx * inery * m * lam * rho3 -
          inerx * inery * inerz * rho9) / denom

    return np.array([c1, c2, c3, c4])


