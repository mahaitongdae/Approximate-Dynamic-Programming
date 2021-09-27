from __future__ import print_function
import numpy as np
class GeneralConfig(object):
    BATCH_SIZE = 256
    DYNAMICS_DIM = 5
    STATE_DIM = 4
    ACTION_DIM = 1
    BUFFER_SIZE = 5000
    FORWARD_STEP = 20
    GAMMA_D = 1
    RESET_ITERATION = 10000

    SIMULATION_STEPS = 500
    NP_TOTAL = SIMULATION_STEPS+1


class DynamicsConfig(GeneralConfig):

    nonlinearity = True
    tire_model = 'Pacejka'    # Pacejka, Linear
    reference_traj = 'SIN'

    a = 1.14       # distance c.g.to front axle(m)
    L = 2.54       # wheel base(m)
    b = L - a      # distance c.g.to rear axle(m)
    m = 1500.      # mass(kg)
    I_zz = 2420.0  # yaw moment of inertia(kg * m ^ 2)
    C = 1.43       # parameter in Pacejka tire model
    B = 14.        # parameter in Pacejka tire model
    u = 15         # longitudinal velocity(m / s)
    g = 9.81
    D = 0.75
    k1 = 88000    # front axle cornering stiffness for linear model (N / rad)
    k2 = 94000    # rear axle cornering stiffness for linear model (N / rad)
    Is = 1.        # steering ratio
    Ts = 0.05      # control signal period
    N = 314        # total simulation steps

    F_z1 = m * g * b / L    # Vertical force on front axle
    F_z2 = m * g * a / L    # Vertical force on rear axle

    k_curve = 1/30        # curve shape of a * sin(kx)
    a_curve = 1           # curve shape of a * sin(kx)
    psi_init = a_curve * k_curve # initial position of psi

    # ADP reset state range
    y_range = 5
    psi_range = 1.3
    beta_range = 1.0

    x_init_s = [0, 0.0, psi_init + 0 / 180 * np.pi, 0.0, 0.0]
    x_init_pred = [-0.1, 0.0, psi_init + 3 / 180 * np.pi, 0.0, 0.0]

class PlotConfig(object):
    fig_size = (8.5, 6.5)
    dpi = 300
    pad = 0.2
    tick_size = 8
    legend_font = {'family': 'Times New Roman', 'size': '8', 'weight': 'normal'}
    label_font = {'family': 'Times New Roman', 'size': '9', 'weight': 'normal'}
    tick_label_font = 'Times New Roman'
