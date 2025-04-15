import numpy as np

def state_dot(state, control, iner_x, iner_y, iner_z, L, lam, g, m):
    # Unpack the state vector
    p, q, r, psi, theta, phi, u, v, w, x, y, z = state
    c1, c2, c3, c4 = control  # Assume 4 control inputs
    
    # Precompute trigonometric functions
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    sin_psi = np.sin(psi)
    cos_psi = np.cos(psi)
    
    # Angular rates
    p_dot = L / iner_x * (c3 + c4 - c1 - c2)
    q_dot = L / iner_y * (c1 + c3 - c2 - c4)
    r_dot = lam / iner_z * (c2 + c3 - c1 - c4)
    
    # Euler angle rates
    psi_dot = (q * sin_phi + r * cos_phi) / cos_theta
    theta_dot = q * cos_phi - r * sin_phi
    phi_dot = (q * sin_theta * sin_phi + r * sin_theta * cos_phi) / cos_theta + p

    # Translational accelerations
    u_dot = g * sin_theta
    v_dot = g * cos_theta * sin_phi
    w_dot = g * cos_theta * cos_phi - sum(control) / m

    # Position derivatives (inertial frame)
    x_dot = (
        u * cos_theta * cos_psi +
        v * (sin_phi * sin_theta * cos_psi - cos_phi * sin_psi) +
        w * (cos_phi * sin_theta * cos_psi + sin_phi * sin_psi)
    )
    y_dot = (
        u * cos_theta * sin_psi +
        v * (sin_phi * sin_theta * sin_psi + cos_phi * cos_psi) +
        w * (cos_phi * sin_theta * sin_psi - sin_phi * cos_psi)
    )
    z_dot = (
        u * sin_theta -
        v * sin_phi * cos_theta -
        w * cos_phi * cos_theta
    )

    return np.array([
        p_dot, q_dot, r_dot,
        psi_dot, theta_dot, phi_dot,
        u_dot, v_dot, w_dot,
        x_dot, y_dot, z_dot
    ])

def costate_dot(state, costate, iner_x, iner_y, iner_z, g, alpha, beta, gamma, delta):
    # Unpack state variables
    p, q, r = state[0], state[1], state[2]
    psi, theta, phi = state[3], state[4], state[5]
    u, v, w = state[6], state[7], state[8]
    x, y, z = state[9], state[10], state[11]
    
    # Unpack costate variables
    rho = costate  # assumed to be a 13-element array

    # Trigonometric terms
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    tan_theta = np.tan(theta)
    sec_theta = 1 / cos_theta
    sin_psi = np.sin(psi)
    cos_psi = np.cos(psi)

    # Costate derivatives
    rho_dot = np.zeros_like(rho)

    rho_dot[0] = -rho[5]
    
    rho_dot[1] = -sec_theta * sin_phi * rho[3] - cos_phi * rho[4] - sin_phi * tan_theta * rho[5]
    
    rho_dot[2] = sin_phi * rho[4] - cos_phi * sec_theta * (rho[3] + sin_theta * rho[5])
    
    rho_dot[3] = (
        2 * alpha * psi +
        u * cos_theta * sin_psi * rho[9] +
        v * sin_theta * sin_phi * sin_psi * rho[9] -
        w * sin_phi * sin_psi * rho[10] -
        cos_psi * (w * sin_phi * rho[9] + u * cos_theta * rho[10] + v * sin_theta * sin_phi * rho[10]) +
        cos_phi * (
            sin_psi * (w * sin_theta * rho[9] + v * rho[10]) +
            cos_psi * (v * rho[9] - w * sin_theta * rho[10])
        )
    )
    
    rho_dot[4] = (
        2 * beta * theta -
        q * sec_theta * sin_phi * tan_theta * rho[3] -
        q * sec_theta**2 * sin_phi * rho[5] +
        g * sin_theta * sin_phi * rho[7] +
        u * cos_psi * sin_theta * rho[9] +
        u * sin_theta * sin_psi * rho[10] -
        v * sin_theta * sin_phi * rho[11] -
        cos_theta * (
            g * rho[6] + v * cos_psi * sin_phi * rho[9] + v * sin_phi * sin_psi * rho[10] + u * rho[11]
        ) -
        0.25 * cos_phi * (
            4 * r * sec_theta**2 * rho[5] +
            sin_theta * (
                -3 * g * rho[8] -
                4 * w * tan_theta * (cos_psi * rho[9] + sin_psi * rho[10]) +
                3 * w * rho[11] +
                tan_theta**2 * (g * rho[8] - w * rho[11])
            ) +
            sec_theta * (
                4 * w * (cos_psi * rho[9] + sin_psi * rho[10]) +
                tan_theta * (4 * r * rho[3] - g * rho[8] + w * rho[11])
            )
        )
    )
    
    rho_dot[5] = (
        2 * beta * phi -
        q * cos_phi * sec_theta * rho[3] +
        r * cos_phi * rho[4] -
        q * cos_phi * tan_theta * rho[5] -
        g * cos_theta * cos_phi * rho[7] -
        v * cos_phi * cos_psi * sin_theta * rho[9] -
        w * cos_phi * sin_psi * rho[9] +
        w * cos_phi * cos_psi * rho[10] -
        v * cos_phi * sin_theta * sin_psi * rho[10] +
        v * cos_theta * cos_phi * rho[11] +
        0.25 * cos_theta * sin_phi * (
            3 * g * rho[8] +
            4 * sec_theta * (q * rho[4] + r * tan_theta * rho[5] - v * sin_psi * rho[9] + v * cos_psi * rho[10]) +
            4 * w * tan_theta * (cos_psi * rho[9] + sin_psi * rho[10]) -
            3 * w * rho[11] +
            sec_theta**2 * (4 * r * rho[3] + g * rho[8] - w * rho[11]) +
            tan_theta**2 * (-g * rho[8] + w * rho[11])
        )
    )
    
    rho_dot[6] = -cos_theta * (cos_psi * rho[9] + sin_psi * rho[10]) - sin_theta * rho[11]
    
    rho_dot[7] = (
        cos_phi * sin_psi * rho[9] -
        cos_psi * (sin_theta * sin_phi * rho[9] + cos_phi * rho[10]) +
        sin_phi * (-sin_theta * sin_psi * rho[10] + cos_theta * rho[11])
    )
    
    rho_dot[8] = (
        sin_phi * (-sin_psi * rho[9] + cos_psi * rho[10]) -
        cos_phi * (
            cos_psi * sin_theta * rho[9] +
            sin_theta * sin_psi * rho[10] -
            cos_theta * rho[11]
        )
    )
    
    rho_dot[9] = 2 * gamma * x
    rho_dot[10] = 2 * gamma * y
    rho_dot[11] = 2 * delta * z

    return rho_dot


def optimal_control(costate, inerx, inery, inerz, L, lam, m):
    rho1, rho2, rho3, *_, rho9 = costate  # unpack needed costate components

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




