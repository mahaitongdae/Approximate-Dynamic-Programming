import dynamics
import numpy as np
import torch
import time
import os
from network import Actor, Critic
from config import DynamicsConfig
from datetime import datetime
from solver import Solver
from utils import step_relative
from plot import plot_comparison, plot_ref_and_state
from config import GeneralConfig

def simulation(methods, log_dir, simu_dir):
    '''

    Args:
        methods: list, methods to simulate
        log_dir: str, name of log dir
        simu_dir: str, name of log dir

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
    plot_length = config.SIMULATION_STEPS

    # Open-loop reference
    x_init = [0.0, 0.0, config.psi_init, 0.0, 0.0]
    op_state, op_control = solver.openLoopMpcSolver(x_init, config.NP_TOTAL)
    np.savetxt(os.path.join(simu_dir, 'Open_loop_control.txt'), op_control)

    for method in methods:
        cal_time = 0
        state = torch.tensor([[0.0, 0.0, config.psi_init, 0.0, 0.0]])
        state.requires_grad_(False)
        x_ref = statemodel_plt.reference_trajectory(state[:, -1])
        state_r = state.detach().clone()
        state_r[:, 0:4] = state_r[:, 0:4] - x_ref

        state_history = state.detach().numpy()
        control_history = []

        print('\nCALCULATION TIME:')
        for i in range(plot_length):
            if method == 'ADP':
                time_start = time.time()
                u = policy.forward(state_r[:, 0:4])
                cal_time += time.time() - time_start
            elif method.startswith('MPC'):
                pred_steps = int(method.split('-')[1])
                x = state_r.tolist()[0]
                time_start = time.time()
                _, control = solver.mpcSolver(x, pred_steps)
                cal_time += time.time() - time_start
                u = np.array(control[0], dtype='float32').reshape(-1, config.ACTION_DIM)
                u = torch.from_numpy(u)
            else:
                u = np.array(op_control[i], dtype='float32').reshape(-1, config.ACTION_DIM)
                u = torch.from_numpy(u)

            state, state_r, _ =step_relative(statemodel_plt, state, u)
            state_history = np.append(state_history, state.detach().numpy(), axis=0)
            control_history = np.append(control_history, u.detach().numpy())



        if method == 'ADP':
            print(" ADP: {:.3f}".format(cal_time) + "s")
            np.savetxt(os.path.join(simu_dir, 'ADP_state.txt'), state_history)
            np.savetxt(os.path.join(simu_dir, 'ADP_control.txt'), control_history)
        elif method.startswith('MPC'):
            pred_steps = method.split('-')[1]
            state_fname, control_fname = 'MPC_' + pred_steps + '_state.txt', \
                                       'MPC_' + pred_steps + '_control.txt'
            print(" MPC {}steps: {:.3f}".format(pred_steps, cal_time) + "s")
            np.savetxt(os.path.join(simu_dir, state_fname), state_history)
            np.savetxt(os.path.join(simu_dir, control_fname), control_history)

        else:
            np.savetxt(os.path.join(simu_dir, 'Open_loop_state.txt'), state_history)

    plot_comparison(simu_dir, methods)
    plot_ref_and_state(log_dir, simu_dir, ref='pos')
    plot_ref_and_state(log_dir, simu_dir, ref='angle')


# def plot_ref_and_state(log_dir, simu_dir, ref='angle', figsize_scalar=1, ms_size=2.0):
#     '''
#
#     Args:
#         log_dir: str, model directory.
#         simu_dir: str, simulation directory.
#         ref: 'pos' or 'angle', which state to plot.
#
#     Returns:
#
#     '''
#     config = GeneralConfig()
#     S_DIM = config.STATE_DIM
#     A_DIM = config.ACTION_DIM
#     policy = Actor(S_DIM, A_DIM)
#     value = Critic(S_DIM, A_DIM)
#     config = DynamicsConfig()
#     solver=Solver()
#     load_dir = log_dir
#     policy.load_parameters(load_dir)
#     value.load_parameters(load_dir)
#     statemodel_plt = dynamics.VehicleDynamics()
#
#     # Open-loop reference
#     x_init = [1.0, 0.0, 0.0, 0.0, 15 * np.pi]
#     index = 0 if ref == 'pos' else 2
#     for step in [3,4,5]:
#         cal_time = 0
#         state = torch.tensor([x_init])
#         state.requires_grad_(False)
#         x_ref = statemodel_plt.reference_trajectory(state[:, -1])
#         state_r = state.detach().clone()
#         state_r[:, 0:4] = state_r[:, 0:4] - x_ref
#
#         state_r_history = state.detach().numpy()
#         state_history = []
#         control_history = []
#         ref_history = []
#         fig_size = (PlotConfig.fig_size * figsize_scalar, PlotConfig.fig_size * figsize_scalar)
#         _, ax = plt.subplots(figsize=cm2inch(*fig_size), dpi=PlotConfig.dpi)
#
#         for i in range(step): # plot_length
#             x = state_r.tolist()[0]
#             time_start = time.time()
#             state_r_predict, control = solver.mpcSolver(x, 10)
#             cal_time += time.time() - time_start
#             u = np.array(control[0], dtype='float32').reshape(-1, config.ACTION_DIM)
#             u = torch.from_numpy(u)
#
#             state, state_r, x_ref =step_relative(statemodel_plt, state, u)
#
#
#             state_predict, ref_predict = recover_absolute_state(state_r_predict, x_ref.numpy().squeeze())
#             ref_history.append(ref_predict[0])
#             state_r_history = np.append(state_r_history, np.expand_dims(state_r_predict[0], axis=0), axis=0)
#             state_history.append(state_predict[0])
#             if i < step - 1:
#                 plt.plot(state_r_predict[:, -1], state_predict[:, index], marker='D', color='deepskyblue', ms=ms_size)
#                 plt.plot(state_r_predict[:, -1], ref_predict[:, index], linestyle='--', color='grey', marker='D', ms=ms_size)
#             else:
#                 plt.plot(state_r_predict[:, -1], state_predict[:, index], label='Predictive trajectory', color='deepskyblue', marker='D', ms=ms_size)
#                 plt.plot(state_r_predict[:, -1], ref_predict[:, index], linestyle='--', color='grey',label='Predictive reference', marker='D', ms=ms_size)
#
#         ref_history = np.array(ref_history)
#         state_history = np.array(state_history)
#         plt.plot(state_r_history[1:, -1], state_history[:, index], linestyle='-.', color='blue', label='Real trajectory', marker='1', ms=ms_size)
#         plt.plot(state_r_history[1:, -1], ref_history[:, index], linestyle='-.', color='black', label='Real reference',
#                  marker='1', ms=ms_size)
#
#         plt.tick_params(labelsize=PlotConfig.tick_size)
#         labels = ax.get_xticklabels() + ax.get_yticklabels()
#         [label.set_fontname(PlotConfig.tick_label_font) for label in labels]
#         plt.legend(loc='best', prop=PlotConfig.legend_font)
#         plt.xlim([47, 57])
#         if ref == 'pos':
#             plt.ylim([0.970, 1.002])
#         elif ref == 'angle':
#             plt.ylim([-0.006, 0.0005])
#         figures_dir = simu_dir + "/Figures"
#         os.makedirs(figures_dir, exist_ok=True)
#         fig_name = 'reference_' + ref + '_' + str(step) + '.png'
#         fig_path = os.path.join(figures_dir, fig_name)
#         plt.savefig(fig_path)


if __name__ == '__main__':
    LOG_DIR = "./Results_dir/2020-10-09-14-42-10000"
    METHODS = ['MPC-10', 'ADP', 'OP'] #
    simu_dir = "./Simulation_dir/" + datetime.now().strftime("%Y-%m-%d-%H-%M")
    os.makedirs(simu_dir, exist_ok=True)
    simulation(METHODS, LOG_DIR, simu_dir)
