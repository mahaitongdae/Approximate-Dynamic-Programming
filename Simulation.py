import Dynamics
import numpy as np
import torch
import time
import os
from Network import Actor, Critic
from Config import DynamicsConfig
from datetime import datetime
from Solver import Solver
from utils import step_relative
from plot import plot_comparison
from Config import GeneralConfig

def simulation(methods, log_dir, simu_dir):

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
    statemodel_plt = Dynamics.VehicleDynamics()
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

            state, state_r =step_relative(statemodel_plt, state, u)
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



if __name__ == '__main__':
    LOG_DIR = "./Results_dir/2020-10-09-14-42-10000"
    METHODS = ['MPC-3','MPC-5','MPC-10','ADP','OP']
    simu_dir = "./Simulation_dir/" + datetime.now().strftime("%Y-%m-%d-%H-%M")
    os.makedirs(simu_dir, exist_ok=True)
    simulation(METHODS, LOG_DIR, simu_dir)
