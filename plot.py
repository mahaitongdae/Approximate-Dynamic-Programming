import numpy as np
import os
from Config import DynamicsConfig
from utils import idplot, Numpy2Torch

import Dynamics
S_DIM = 4
A_DIM = 1


def plot_comparison(simu_dir, methods):
    '''
    Plot comparison figure among ADP, MPC & open-loop solution.
    Trajectory, tracking error and control signal plot

    Parameters
    ----------
    picture_dir: string
        location of figure saved.

    '''
    num_methods = len(methods)
    legends = ['MPC-3','MPC-5','MPC-10','ADP','Open-loop']
    picture_dir = simu_dir + "/Figures"
    if not os.path.exists(picture_dir): os.mkdir(picture_dir)
    config = DynamicsConfig()
    trajectory_data = []
    heading_angle = []
    error_data = []
    psi_error_data = []
    control_plot_data = []
    utilities_data = []
    dy = Dynamics.VehicleDynamics()

    def load_data(method):
        if method.startswith('MPC'):
            pred_steps = method.split('-')[1]
            state_fname, control_fname = 'MPC_' + pred_steps + '_state.txt', \
                                         'MPC_' + pred_steps + '_control.txt'
            state = np.loadtxt(os.path.join(simu_dir, state_fname))
            control = np.loadtxt(os.path.join(simu_dir, control_fname))
        elif method.startswith('ADP'):
            state = np.loadtxt(os.path.join(simu_dir, 'ADP_state.txt'))
            control = np.loadtxt(os.path.join(simu_dir, 'ADP_control.txt'))
        elif method.startswith('OP'):
            state = np.loadtxt(os.path.join(simu_dir, 'Open_loop_state.txt'))
            control = np.loadtxt(os.path.join(simu_dir, 'Open_loop_control.txt'))
        else:
            raise KeyError('invalid methods')
        trajectory = (state[:, 4], state[:, 0])
        heading = (state[:, 4], 180 / np.pi * state[:, 2])
        ref = dy.reference_trajectory(Numpy2Torch(state[:, 4], state[:, 4].shape)).numpy()
        error = (state[:, 4], state[:, 0] - ref[:, 0])
        psi_error = (state[:, 4], 180 / np.pi * (state[:, 2] - ref[:, 2]))
        control_tuple = (state[1:, 4], 180 / np.pi * control)
        utilities = 6 * (state[1:, 0]) ** 2 + 80 * control ** 2
        utilities_tuple = (state[1:, 4], utilities)

        trajectory_data.append(trajectory)
        heading_angle.append(heading)
        error_data.append(error)
        psi_error_data.append(psi_error)
        control_plot_data.append(control_tuple)
        utilities_data.append(utilities_tuple)

    for method in methods:
        load_data(method)
        idplot(trajectory_data, num_methods, "xy",
               fname=os.path.join(picture_dir, 'trajectory.png'),
               xlabel="Longitudinal position [m]",
               ylabel="Lateral position [m]",
               legend=legends,
               legend_loc="lower left"
               )
        idplot(utilities_data, num_methods, "xy",
               fname=os.path.join(picture_dir, 'utilities.png'),
               xlabel="Longitudinal position [m]",
               ylabel="Utilities",
               legend=legends,
               legend_loc="lower left"
               )
        idplot(heading_angle, num_methods, "xy",
               fname=os.path.join(picture_dir, 'trajectory_heading_angle.png'),
               xlabel="Longitudinal position [m]",
               ylabel=r"Heading angle [$\degree$]",
               legend=legends,
               legend_loc="lower left"
               )
        idplot(error_data, num_methods, "xy",
               fname=os.path.join(picture_dir, 'trajectory_error.png'),
               xlabel="Longitudinal position [m]",
               ylabel="Lateral position error [m]",
               legend=legends,
               legend_loc="upper left"
               )
        idplot(psi_error_data, num_methods, "xy",
               fname=os.path.join(picture_dir, 'head_angle_error.png'),
               xlabel="Longitudinal position [m]",
               ylabel=r"Head angle error [$\degree$]",
               legend=legends,
               legend_loc="lower left"
               )
        idplot(control_plot_data, num_methods, "xy",
               fname=os.path.join(picture_dir, 'control.png'),
               xlabel="Longitudinal position [m]",
               ylabel=r"Steering angle [$\degree$]",
               legend=legends,
               legend_loc="upper left"
               )


def adp_simulation_plot(simu_dir):
    '''
    Simulate and plot trajectory and control after ADP training algorithm.

    Parameters
    ----------
    simu_dir: string
        location of data and figures saved.

    '''
    state_history = np.loadtxt(os.path.join(simu_dir, 'ADP_state.txt'))
    control_history = np.loadtxt(os.path.join(simu_dir, 'ADP_control.txt'))
    trajectory = (state_history[:, -1], state_history[:, 0])
    figures_dir = simu_dir + "/Figures"
    os.makedirs(figures_dir, exist_ok=True)
    idplot(trajectory, 1, "xy",
           fname=os.path.join(figures_dir, 'adp_trajectory.png'),
           xlabel="longitudinal position [m]",
           ylabel="Lateral position [m]",
           legend=["trajectory"],
           legend_loc="upper left"
           )
    u_lat = (state_history[:, -1], state_history[:, 1])
    psi =(state_history[:, -1], state_history[:, 2])
    omega = (state_history[:, -1], state_history[:, 3])
    data = [u_lat, psi, omega]
    legend=["$u_{lat}$", "$\psi$", "$\omega$"]
    idplot(data, 3, "xy",
           fname=os.path.join(figures_dir, 'adp_other_state.png'),
           xlabel="longitudinal position [m]",
           legend=legend
           )
    control_history_plot = (state_history[1:, -1], 180 / np.pi * control_history)
    idplot(control_history_plot, 1, "xy",
           fname=os.path.join(figures_dir, 'adp_control.png'),
           xlabel="longitudinal position [m]",
           ylabel="steering angle [degree]"
           )



