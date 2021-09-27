"""
Copyright (c). All Rights Reserved.
<Reinforcement Learning and Control> (Year 2020)
Intelligent Driving Lab (iDLab), Tsinghua University

by Shengbo Eben Li & Haitong Ma

Description: Chapter 8: RL example for lane keeping problem in a curve road;
             Approximate dynamic programming with structured policy

Update Date: 2021-09-06, Haitong Ma: Rewrite code formats
"""
from config import GeneralConfig, DynamicsConfig, PlotConfig
import numpy as np
import torch
import time
import os
from network import Actor, Critic
from solver import Solver
from utils import idplot, numpy2torch, step_relative, recover_absolute_state, cm2inch
import matplotlib.pyplot as plt
from datetime import datetime

import dynamics
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
    legends = methods # ['MPC-3','MPC-5','MPC-10','ADP','Open-loop']
    picture_dir = simu_dir + "/Figures"
    if not os.path.exists(picture_dir): os.mkdir(picture_dir)
    config = DynamicsConfig()
    trajectory_data = []
    heading_angle = []
    error_data = []
    psi_error_data = []
    control_plot_data = []
    utilities_data = []
    control_error_data = []
    dy = dynamics.VehicleDynamics()
    op_control = np.loadtxt(os.path.join(simu_dir, 'Open_loop_control.txt'))[:-1]

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
            state = np.loadtxt(os.path.join(simu_dir, 'Open_loop_state.txt'))[:-1, :]
            control = np.loadtxt(os.path.join(simu_dir, 'Open_loop_control.txt'))[:-1]
        else:
            raise KeyError('invalid methods')


        trajectory = (state[:, 4], state[:, 0])
        heading = (state[:, 4], 180 / np.pi * state[:, 2])
        ref = dy.reference_trajectory(numpy2torch(state[:, 4], state[:, 4].shape)).numpy()
        error = (state[:, 4], state[:, 0] - ref[:, 0])
        psi_error = (state[:, 4], 180 / np.pi * (state[:, 2] - ref[:, 2]))
        control_tuple = (state[1:, 4], 180 / np.pi * control)
        control_error_tuple = (state[1:, 4], 180 / np.pi * (control - op_control))
        utilities = 0.2 * (state[1:, 0]) ** 2 + 1 * (state[1:, 2]) ** 2 + 5 * control ** 2
        utilities_tuple = (state[1:, 4], utilities)

        error[1][:] = 100 * error[1][:]
        trajectory_data.append(trajectory)
        heading_angle.append(heading)
        error_data.append(error)
        psi_error_data.append(psi_error)
        control_plot_data.append(control_tuple)
        control_error_data.append(control_error_tuple)
        utilities_data.append(utilities_tuple)

    for method in methods:
        load_data(method)
        idplot(trajectory_data, num_methods, "xy",
               fname=os.path.join(picture_dir, '1-Lat Position.png'),
               xlabel="Travel dist [m]",
               ylabel="Lateral position [m]",
               legend=legends,
               legend_loc="lower left"
               )
        idplot(utilities_data, num_methods, "xy",
               fname=os.path.join(picture_dir, '1-Utility func.png'),
               xlabel="Travel dist [m]",
               ylabel="Utility",
               legend=legends,
               legend_loc="lower left"
               )
        idplot(heading_angle, num_methods, "xy",
               fname=os.path.join(picture_dir, '1-Heading angle.png'),
               xlabel="Travel dist [m]",
               ylabel=r"Heading angle [$\degree$]",
               legend=legends,
               legend_loc="lower left",
               tight_layout=False
               )
        idplot(error_data, num_methods, "xy",
               fname=os.path.join(picture_dir, '1-Lat position error.png'),
               xlabel="Travel dist [m]",
               ylabel="Lateral position error [cm]",
               legend=legends,
               legend_loc="upper left",
               )
        idplot(psi_error_data, num_methods, "xy",
               fname=os.path.join(picture_dir, '1-Heading angle error.png'),
               xlabel="Travel dist [m]",
               ylabel=r"Heading angle error [$\degree$]",
               legend=legends,
               legend_loc="lower left"
               )
        idplot(control_plot_data, num_methods, "xy",
               fname=os.path.join(picture_dir, '1-Control input.png'),
               xlabel="Travel dist [m]",
               ylabel=r"Steering angle [$\degree$]",
               legend=legends,
               legend_loc="upper left"
               )
        idplot(control_error_data, num_methods, "xy",
               fname=os.path.join(picture_dir, '1-Control error.png'),
               xlabel="Travel dist [m]",
               ylabel=r"Steering angle error [$\degree$]",
               legend=legends,
               legend_loc="upper right"
               )


def plot_phase_plot(methods, log_dir, simu_dir,
                    data = '',
                    figsize_scalar=1,
                    x_init=None):
    '''

    Args:
        log_dir: str, model directory.
        simu_dir: str, simulation directory.
        ref: 'pos' or 'angle', which state to plot.

    Returns:

    '''
    config = DynamicsConfig()
    if x_init is None:
        x_init = config.x_init_pred
    S_DIM = config.STATE_DIM
    A_DIM = config.ACTION_DIM
    policy = Actor(S_DIM, A_DIM)
    value = Critic(S_DIM, A_DIM)
    config = DynamicsConfig()
    solver = Solver()
    load_dir = log_dir
    policy.load_parameters(load_dir)
    value.load_parameters(load_dir)
    statemodel_plt = dynamics.VehicleDynamics()
    if data == 'Phase plot':
        index_x = 0
        index_y = 2
    elif data == 'Lat position':
        index_x = -1
        index_y = 0
    elif data == 'Head angle':
        index_x = -1
        index_y = 2
    elif data == 'Control':
        index_x = -1
        index_y = -2

    axis_label = {"-1":"Predictive horizon [s]",
                  "2":r"Heading angle error [$\degree$]",
                  "0": "Lateral position error [cm]",
                  "-2": r"Steering angle [$\degree$]",
                  }

    # Open-loop reference


    fig_size = (PlotConfig.fig_size * figsize_scalar, PlotConfig.fig_size * figsize_scalar)
    ax_list = []
    fig_list = []
    state_list = []
    action_list = []
    for method in methods:
        state = torch.tensor([x_init])
        for step in range(1):
            state.requires_grad_(False)
            x_ref = statemodel_plt.reference_trajectory(state[:, -1])
            state_r = state.detach().clone()
            state_list.append(state_r.numpy())
            state_r[:, 0:4] = state_r[:, 0:4] - x_ref # refresh relative state

            if method.startswith('ADP'):
                fig, ax = plt.subplots(figsize=cm2inch(*fig_size), dpi=PlotConfig.dpi)
                ax_list.append(ax)
                fig_list.append(fig)
                state_r_predict = []
                virtual_state_r = state_r.clone().detach()
                state_r_predict.append(virtual_state_r.numpy().squeeze())
                for virtual_step in range(30):
                    virtual_u = policy.forward(virtual_state_r[:, 0:4])
                    if virtual_step == 0: u = virtual_u.clone().detach()
                    virtual_state_r, _, _, _, _, _, _ = statemodel_plt.step(virtual_state_r, virtual_u)
                    # virtual_state_r = virtual_state.detach().clone()[:, 0:4] - x_ref
                    state_r_predict.append(virtual_state_r.detach().numpy().squeeze())
                    action_list.append(virtual_u.detach().numpy().squeeze())
                state_r_predict = np.array(state_r_predict)
                label = 'ADP'
                marker = 'D'
                ms_size = 2.0
                state_list.append(state.detach().clone().numpy())

            elif method.startswith('MPC'):
                pred_steps = int(method.split('-')[1])
                x = state_r.tolist()[0]
                state_r_predict, action_list = solver.mpcSolver(x, pred_steps)
                # u = np.array(control[0], dtype='float32').reshape(-1, config.ACTION_DIM)
                # u = torch.from_numpy(u)
                label = 'MPC ' + str(pred_steps)
                action_list = action_list.squeeze()
                marker_dict = {'5': 'o', '10': '+', '30': 'x'}
                marker = marker_dict.get(str(pred_steps))
                ms_size = 4.0
                # state_predict, ref_predict = recover_absolute_state(state_r_predict, x_ref.numpy().squeeze())

            else:
                continue
            time = 0.1 * np.arange(len(state_r_predict[:, -1] - state_r_predict[0, -1]))
            state_r_predict[:, -1] = state_r_predict[:, -1] - state_r_predict[0, -1]
            data_x = 100 * state_r_predict[:, index_x] if index_x == 0 else time
            if index_y == 2:
                data_y = 180 / np.pi * state_r_predict[:, index_y]
            elif index_y == -2:
                data_y = np.array(action_list) / np.pi * 180.0
                data_x = data_x[:-1]
            elif index_y == 0:
                data_y = 100 * state_r_predict[:, index_y]
            else:
                data_y = state_r_predict[:, index_y]
            ax_list[step].plot(data_x, data_y, linestyle='--', label=label,
                     marker=marker, ms=ms_size, zorder=0)
            state, state_r, x_ref = step_relative(statemodel_plt, state, u)

    for ax in ax_list:
        if data == 'phase':
            ax.scatter([0.0], [0.0], color='red',
                        label='Ref point', marker='o', s=50, zorder=10)
            # ax.set_xlim([-1, 1])
            # ax.set_ylim([-0.25, 0.25])
        else:
            ax.plot(list(ax.get_xlim()), [0.0, 0.0], color='grey',
                    linestyle='--', label='Ref', zorder=0)
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontname(PlotConfig.tick_label_font) for label in labels]
        ax.legend(loc='best', prop=PlotConfig.legend_font)
        ax.tick_params(labelsize=PlotConfig.tick_size)
        ax.set_xlabel(axis_label.get(str(index_x)), PlotConfig.label_font)
        ax.set_ylabel(axis_label.get(str(index_y)), PlotConfig.label_font)
    figures_dir = simu_dir + "/Figures"
    os.makedirs(figures_dir, exist_ok=True)
    for i, fig in enumerate(fig_list):
        if i % 5 != 0 : continue
        fig.tight_layout(pad=PlotConfig.pad)
        fig_name = "2-" + data + '-' + str(i) + ' Step.png'
        fig_path = os.path.join(figures_dir, fig_name)
        fig.savefig(fig_path)



