"""
Copyright (c). All Rights Reserved.
<Reinforcement Learning and Control> (Year 2020)
Intelligent Driving Lab (iDLab), Tsinghua University

by Shengbo Eben Li & Haitong Ma

Description: Chapter 6: RL example for lane keeping problem in a curve road;
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
    dy = dynamics.VehicleDynamics()

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
        ref = dy.reference_trajectory(numpy2torch(state[:, 4], state[:, 4].shape)).numpy()
        error = (state[:, 4], state[:, 0] - ref[:, 0])
        # if method.startswith('ADP'):
        #     error[1][5:] = error[1][5:] + 0.0013
        #     error[1][5:] = 0.98 * error[1][5:]
        psi_error = (state[:, 4], 180 / np.pi * (state[:, 2] - ref[:, 2]))
        control_tuple = (state[1:, 4], 180 / np.pi * control)
        utilities = 6 * (state[1:, 0]) ** 2 + 80 * control ** 2
        utilities_tuple = (state[1:, 4], utilities)

        error[1][:] = 100 * error[1][:]
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
               ylabel="Lateral position error [cm]",
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

def plot_ref_and_state(log_dir, simu_dir, ref='angle', figsize_scalar=1, ms_size=2.0):
    '''

    Args:
        log_dir: str, model directory.
        simu_dir: str, simulation directory.
        ref: 'pos' or 'angle', which state to plot.

    Returns:

    '''
    config = GeneralConfig()
    S_DIM = config.STATE_DIM
    A_DIM = config.ACTION_DIM
    policy = Actor(S_DIM, A_DIM)
    value = Critic(S_DIM, A_DIM)
    config = DynamicsConfig()
    solver=Solver()
    load_dir = log_dir
    policy.load_parameters(load_dir)
    value.load_parameters(load_dir)
    statemodel_plt = dynamics.VehicleDynamics()

    # Open-loop reference
    x_init = [1.01, 0.0, 0.2/np.pi*180, 0.0, 15 * np.pi]
    index = 0 if ref == 'pos' else 2
    for step in [3,4,5]:
        cal_time = 0
        state = torch.tensor([x_init])
        state.requires_grad_(False)
        x_ref = statemodel_plt.reference_trajectory(state[:, -1])
        state_r = state.detach().clone()
        state_r[:, 0:4] = state_r[:, 0:4] - x_ref

        state_r_history = state.detach().numpy()
        state_history = []
        control_history = []
        ref_history = []
        fig_size = (PlotConfig.fig_size * figsize_scalar, PlotConfig.fig_size * figsize_scalar)
        _, ax = plt.subplots(figsize=cm2inch(*fig_size), dpi=PlotConfig.dpi)

        for i in range(step): # plot_length
            x = state_r.tolist()[0]
            time_start = time.time()
            state_r_predict, control = solver.mpcSolver(x, 10)
            cal_time += time.time() - time_start
            u = np.array(control[0], dtype='float32').reshape(-1, config.ACTION_DIM)
            u = torch.from_numpy(u)

            state, state_r, x_ref =step_relative(statemodel_plt, state, u)
            state_predict, ref_predict = recover_absolute_state(state_r_predict, x_ref.numpy().squeeze())
            ref_history.append(ref_predict[0])
            state_r_history = np.append(state_r_history, np.expand_dims(state_r_predict[0], axis=0), axis=0)
            state_history.append(state_predict[0])
            if i < step - 1:
                plt.plot(state_r_predict[:, -1], state_predict[:, index], linestyle='--', marker='D', color='deepskyblue', ms=ms_size)
                plt.plot(state_r_predict[:, -1], ref_predict[:, index], linestyle='--', color='grey', marker='D', ms=ms_size)
            else:
                plt.plot(state_r_predict[:, -1], state_predict[:, index], linestyle='--', label='Predictive trajectory', color='deepskyblue', marker='D', ms=ms_size)
                plt.plot(state_r_predict[:, -1], ref_predict[:, index], linestyle='--', color='grey',label='Predictive reference', marker='D', ms=ms_size)

        ref_history = np.array(ref_history)
        state_history = np.array(state_history)
        plt.plot(state_r_history[1:, -1], state_history[:, index], color='blue', label='Real trajectory', marker='1', ms=ms_size)
        plt.plot(state_r_history[1:, -1], ref_history[:, index], linestyle='-.', color='black', label='Real reference',
                 marker='1', ms=ms_size)

        plt.tick_params(labelsize=PlotConfig.tick_size)
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontname(PlotConfig.tick_label_font) for label in labels]
        plt.legend(loc='best', prop=PlotConfig.legend_font)
        plt.xlim([47, 57])
        if ref == 'pos':
            plt.ylim([0.990, 1.002])
        elif ref == 'angle':
            plt.ylim([-0.006, 0.0005])
        figures_dir = simu_dir + "/Figures"
        os.makedirs(figures_dir, exist_ok=True)
        fig_name = 'reference_' + ref + '_' + str(step) + '.png'
        fig_path = os.path.join(figures_dir, fig_name)
        plt.savefig(fig_path)


def plot_phase_plot(methods, log_dir, simu_dir,
                    data = 'phase',
                    figsize_scalar=1):
    '''

    Args:
        log_dir: str, model directory.
        simu_dir: str, simulation directory.
        ref: 'pos' or 'angle', which state to plot.

    Returns:

    '''
    config = GeneralConfig()
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
    if data == 'phase':
        index_x = 0
        index_y = 2
    elif data == 'position':
        index_x = -1
        index_y = 0
    elif data == 'angle':
        index_x = -1
        index_y = 2

    axis_label = {"-1":"Logitudinal position [m]",
                  "2":"Heading angle error [deg]",
                  "0": "Lateral position error [cm]",
                  }

    # Open-loop reference
    x_init = [1.1, 0.0, 5.0 / 180 * np.pi, 0.0, 15 * np.pi]

    fig_size = (PlotConfig.fig_size * figsize_scalar, PlotConfig.fig_size * figsize_scalar)
    ax_list = []
    fig_list = []
    state_list = []
    for method in methods:
        state = torch.tensor([x_init])
        for step in range(1):
            state.requires_grad_(False)
            x_ref = statemodel_plt.reference_trajectory(state[:, -1])
            state_r = state.detach().clone()
            state_r[:, 0:4] = state_r[:, 0:4] - x_ref # refresh relative state

            if method.startswith('ADP'):
                fig, ax = plt.subplots(figsize=cm2inch(*fig_size), dpi=PlotConfig.dpi)
                ax_list.append(ax)
                fig_list.append(fig)
                state_r_predict = []
                virtual_state_r = state_r.clone().detach()
                for virtual_step in range(30):
                    virtual_u = policy.forward(virtual_state_r[:, 0:4])
                    if virtual_step == 0: u = virtual_u.clone().detach()
                    virtual_state_r, _, _, _, _, _, _ = statemodel_plt.step(virtual_state_r, virtual_u)
                    # virtual_state_r = virtual_state.detach().clone()[:, 0:4] - x_ref
                    state_r_predict.append(virtual_state_r.detach().numpy().squeeze())
                state_r_predict = np.array(state_r_predict)
                label = 'ADP'
                marker = 'D'
                ms_size = 2.0
                state_list.append(state.detach().clone().numpy())

            elif method.startswith('MPC'):
                pred_steps = int(method.split('-')[1])
                x = state_r.tolist()[0]
                state_r_predict, control = solver.mpcSolver(x, pred_steps)
                u = np.array(control[0], dtype='float32').reshape(-1, config.ACTION_DIM)
                u = torch.from_numpy(u)
                label = 'MPC ' + str(pred_steps)
                marker_dict = {'5': 'o', '10': '+', '30': 'x'}
                marker = marker_dict.get(str(pred_steps))
                ms_size = 4.0
                # state_predict, ref_predict = recover_absolute_state(state_r_predict, x_ref.numpy().squeeze())

            else:
                continue
            data_y = 180 / np.pi * state_r_predict[:, index_y] if index_y == 2 else state_r_predict[:, index_y]
            data_x = 100 * state_r_predict[:, index_x] if index_x == 0 else state_r_predict[:, index_x]
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
        fig_name = data + '_plot_' + str(i) + '.png'
        fig_path = os.path.join(figures_dir, fig_name)
        fig.savefig(fig_path)

if __name__ == '__main__':
    LOG_DIR = "./trained_results/2020-10-09-14-42-10000"
    METHODS = ['MPC-10', 'ADP', 'OP']  #
    simu_dir = "./Simulation_dir/" + datetime.now().strftime("%Y-%m-%d-%H-%M")
    os.makedirs(simu_dir, exist_ok=True)
    plot_phase_plot(['ADP','MPC-5','MPC-10','MPC-50'], LOG_DIR, simu_dir)



