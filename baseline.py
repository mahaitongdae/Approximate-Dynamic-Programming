"""
    <Reinforcement Learning and Control>(Year 2020)
    by Shengbo Eben Li
        @ Intelligent Driving Lab, Tsinghua University

    ADP example for lane keeping problem in a circle road

    [Method]
    Model predictive control(MPC) as comparison

"""
from Solver import Solver
from Config import DynamicsConfig, GeneralConfig
from matplotlib import pyplot as plt
from utils import Numpy2Torch, step_relative
import numpy as np
import time
import os
import Dynamics
import torch

class Baseline(DynamicsConfig):
    def __init__(self, initial_state, baseline_dir):
        self.config = DynamicsConfig()
        self.solver = Solver()
        self.initial_state = initial_state
        self.baseline_dir = baseline_dir

    def mpcSolution(self):

        # initialize state model
        statemodel_plt = Dynamics.VehicleDynamics()
        state = self.initial_state
        # state = torch.tensor([[0.0, 0.0, self.psi_init, 0.0, 0.0]])
        state.requires_grad_(False)
        x_ref = statemodel_plt.reference_trajectory(state[:, -1])
        state_r = state.detach().clone()                # relative state
        state_r[:, 0:4] = state_r[:, 0:4] - x_ref

        self.state_history = state.detach().numpy()
        plot_length = self.config.SIMULATION_STEPS
        self.control_history = []
        self.state_r_history = state_r
        cal_time = 0
        plt.figure(0)

        for i in range(plot_length):
            x = state_r.tolist()[0]
            time_start = time.time()
            temp, control = self.solver.mpcSolver(x, self.config.NP)
            plt.plot(temp[:,-1],temp[:,0])
            cal_time += time.time() - time_start
            u = Numpy2Torch(control[0], (-1,self.config.ACTION_DIM))

            state, state_r = step_relative(statemodel_plt, state, u)

            self.state_history = np.append(self.state_history, state.detach().numpy(), axis=0)
            self.control_history = np.append(self.control_history, u.detach().numpy())
            self.state_r_history = np.append(self.state_history, state_r.detach().numpy())
        print("MPC calculating time: {:.3f}".format(cal_time) + "s")
        self.mpcSaveTraj()

    def mpcPlot(self):
        dy = Dynamics.VehicleDynamics()
        ref = dy.reference_trajectory(Numpy2Torch(self.state_history[:,-1],self.state_history[:,-1].shape))
        self.state_r_history = self.state_r_history.reshape([-1,5])
        plt.figure(1)
        plt.plot(self.state_history[:,-1],self.state_history[:,0], label="trajectory")
        # plt.plot(state_r_history[:,-1],state_r_history[:,0], label="$trajectory_r$")
        # plt.plot(self.state_history[:,-1],self.config.a_curve * np.sin(self.config.k_curve*self.state_history[:,-1]), label="reference")
        plt.plot(self.state_history[:,-1], ref[:,0], label="reference")
        plt.legend(loc="upper right")
        plt.figure(2)
        plt.plot(self.state_history[:, -1], self.state_history[:, 2], label="trajectory")
        # plt.plot(state_r_history[:,-1],state_r_history[:,0], label="$trajectory_r$")
        # plt.plot(self.state_history[:,-1],self.config.a_curve * np.sin(self.config.k_curve*self.state_history[:,-1]), label="reference")
        plt.plot(self.state_history[:, -1], ref[:, 2], label="reference")
        plt.legend(loc="upper right")
        plt.figure(3)
        plt.plot(self.state_history[0:-1,-1], self.control_history)
        plt.show()

    def mpcSaveTraj(self):
        np.savetxt(os.path.join(self.baseline_dir, 'structured_MPC_state.txt'), self.state_history)
        np.savetxt(os.path.join(self.baseline_dir, 'structured_MPC_control.txt'), self.control_history)

    def openLoopSolution(self):
        init_state = self.initial_state.detach().numpy().tolist()[0]
        state, control = self.solver.openLoopMpcSolver(init_state, self.config.NP_TOTAL)
        np.savetxt(os.path.join(self.baseline_dir, 'Open_loop_state.txt'), state)
        np.savetxt(os.path.join(self.baseline_dir, 'Open_loop_control.txt'), control)

if __name__ == '__main__':
    state = torch.tensor([[0.5, 0.0, 0.0, 0.0, 0.0]])
    baseline_dir = "./baseline"
    baseline = Baseline(state,baseline_dir)
    baseline.mpcSolution()
    baseline.mpcPlot()
    # baseline.openLoopSolution()