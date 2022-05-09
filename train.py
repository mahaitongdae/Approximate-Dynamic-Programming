"""
Copyright (c). All Rights Reserved.
<Reinforcement Learning and Control> (Year 2020)
Intelligent Driving Lab (iDLab), Tsinghua University

by Shengbo Eben Li & Haitong Ma

Description: Chapter 6: RL example for lane keeping problem in a curve road;
             Approximate dynamic programming with structured policy

Update Date: 2021-09-06, Haitong Ma: Rewrite code formats
"""
import numpy as np
import torch
import os
from matplotlib import pyplot as plt
from config import GeneralConfig, DynamicsConfig
from dynamics import VehicleDynamics
from utils import step_relative


class Train(DynamicsConfig):
    def __init__(self):
        super(Train, self).__init__()

        self.preprocessor = Preprocessor()

        self.agent_batch = torch.empty([self.BATCH_SIZE, self.DYNAMICS_DIM])
        self.state_batch = torch.empty([self.BATCH_SIZE, self.STATE_DIM])
        self.init_index = np.ones([self.BATCH_SIZE, 1])

        self.x_forward = []
        self.u_forward = []
        self.L_forward = []
        self.ref_forward = []
        self.iteration_index = 0

        self.value_loss = np.empty([0, 1])
        self.policy_loss = np.empty([0, 1])
        self.dynamics = VehicleDynamics()
        self.equilibrium_state = torch.tensor([[0.0, 0.0, 0.0, 0.0]])

        for i in range(self.FORWARD_STEP):
            self.u_forward.append([])
            self.L_forward.append([])
        for i in range(self.FORWARD_STEP+1):
            self.x_forward.append([])
            self.ref_forward.append([])
        self.initialize_state()

    def initialize_state(self):
        self.agent_batch[:, 0] = torch.normal(0.0, 0.1, [self.BATCH_SIZE, ])
        self.agent_batch[:, 1] = torch.normal(0.0, 0.05, [self.BATCH_SIZE, ])
        self.agent_batch[:, 2] = torch.normal(0.0, 0.05, [self.BATCH_SIZE, ])
        self.agent_batch[:, 3] = torch.normal(0.0, 0.02, [self.BATCH_SIZE, ])
        self.agent_batch[:, 4] = torch.linspace(0.0, np.pi, self.BATCH_SIZE)
        # init_ref = self.dynamics.reference_trajectory(self.agent_batch[:, 4])
        # self.agent_batch[:, 0:4] = self.state_batch + init_ref
        self.init_state = self.agent_batch

    def check_done(self, state):
        """
        Check if the states reach unreasonable zone and reset them
        Parameters
        ----------
        state: tensor   shape: [BATCH_SIZE, STATE_DIMENSION]
            state used for checking.
        Returns
        -------

        """
        threshold = np.kron(np.ones([self.BATCH_SIZE, 1]), np.array([self.y_range, self.psi_range]))
        threshold = np.array(threshold, dtype='float32')
        threshold = torch.from_numpy(threshold)
        ref_state = self.dynamics.reference_trajectory(state[:, -1])
        state = state - ref_state
        check_state = state[:, [0, 2]].clone()
        check_state.detach_()
        sign_error = torch.sign(torch.abs(check_state) - threshold) # if abs state is over threshold, sign_error = 1
        self._reset_index, _ = torch.max(sign_error, 1) # if one state is over threshold, _reset_index = 1
        if self.iteration_index == self.RESET_ITERATION:
            self._reset_index = torch.from_numpy(np.ones([self.BATCH_SIZE,],dtype='float32'))
            self.iteration_index = 0
            print('AGENT RESET')
        reset_state = self._reset_state(self.agent_batch)
        return reset_state

    def _reset_state(self, state):
        """
        reset state to initial state.
        Parameters
        ----------
        state: tensor   shape: [BATCH_SIZE, STATE_DIMENSION]
            state used for checking.

        Returns
        -------
        state: state after reset.

        """
        for i in range(self.BATCH_SIZE):
            if self._reset_index[i] == 1:
                state[i, :] = self.init_state[i, :]
        return state

    def update_state(self, policy, dynamics):
        """
        Update state using policy net and dynamics model.
        Parameters
        ----------
        policy: nn.Module
            policy net.
        dynamics: object dynamics.

        """
        self.agent_batch = self.check_done(self.agent_batch)
        self.agent_batch.detach_()
        # ref_trajectory = dynamics.reference_trajectory(self.agent_batch[:, -1])
        # self.state_batch = self.agent_batch[:, 0:4] - ref_trajectory
        control = policy.forward(self.preprocessor.torch_preprocess(self.agent_batch, self.agent_batch[:, -1]))
        # self.agent_batch, self.state_batch = dynamics.step_relative(self.agent_batch, control)
        self.agent_batch, _, _, _, _, _, _ = dynamics.step(self.agent_batch, control)
        self.iteration_index += 1

    def policy_evaluation(self, policy, value, dynamics):
        """
        Do n-step look-ahead policy evaluation.
        Parameters
        ----------
        policy: policy net
        value: value net
        dynamics: object dynamics

        """
        for i in range(self.FORWARD_STEP):
            if i == 0:
                self.x_forward[i] = self.agent_batch.detach() # 要存agent batch是因为step relative要用agent
                self.ref_forward[i] = dynamics.reference_trajectory(self.agent_batch[:,-1])
                self.virtual_ref_start = self.ref_forward[i].clone()
                # self.state_batch = dynamics.relative_state(self.x_forward[i])
                self.u_forward[i] = policy.forward(self.preprocessor.torch_preprocess(self.x_forward[i], self.virtual_ref_start[:, -1]))
                self.x_forward[i + 1], _, _, _, _, _, _ = dynamics.step(self.x_forward[i], self.u_forward[i])
                # self.ref_state_start = dynamics.reference_trajectory

            else:
                self.ref_forward[i] = dynamics.reference_trajectory_rollout(self.virtual_ref_start, self.x_forward[i])
                # ref_state = self.x_forward[i][:, 0:4] - reference
                self.u_forward[i] = policy.forward(self.preprocessor.torch_preprocess(self.x_forward[i], self.virtual_ref_start[:, -1]))
                self.x_forward[i + 1], _, _, _, _, _, _ = dynamics.step(self.x_forward[i],
                                                                                        self.u_forward[i])
                # ref_state_next = self.x_forward[i + 1][:, 0:4] - reference
                # self.L_forward[i] = dynamics.utility(ref_state_next, self.u_forward[i])
            self.L_forward[i] = self.GAMMA_D ** i * dynamics.utility(self.x_forward[i], self.ref_forward[i], self.u_forward[i])
        self.agent_batch_next = self.x_forward[-1]
        # self.state_batch_next = self.agent_batch_next[:, 0:4] - reference
        self.value_next = value.forward(self.preprocessor.torch_preprocess(self.agent_batch_next, self.virtual_ref_start[:, -1]))
        self.utility = torch.zeros([self.FORWARD_STEP, self.BATCH_SIZE], dtype=torch.float32)
        for i in range(self.FORWARD_STEP):
            self.utility[i] = self.L_forward[i].clone()
        self.sum_utility = torch.sum(self.utility,0)
        target_value = self.sum_utility.detach() + self.GAMMA_D ** self.FORWARD_STEP * self.value_next.detach()
        value_now = value.forward(self.preprocessor.torch_preprocess(self.agent_batch, self.virtual_ref_start[:, -1]))
        value_equilibrium = value.forward(self.preprocessor.torch_preprocess(self.virtual_ref_start.detach().clone(),
                                                                             self.virtual_ref_start[:, -1].detach().clone()))
        value_loss = 1 / 2 * torch.mean(torch.pow((target_value - value_now), 2)) # + 10 * torch.pow(value_equilibrium, 2)
        self.state_batch.requires_grad_(False)
        value.zero_grad()
        value_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(value.parameters(), 10.0)
        value.opt.step()
        value.scheduler.step()
        self.value_loss = np.append(self.value_loss, value_loss.detach().numpy())
        return value_loss.detach().numpy()

    def policy_improvement(self, policy, value):
        """
        Do n-step look-ahead policy improvement.
        Parameters
        ----------
        policy: policy net
        value: value net

        """
        self.value_next = value.forward(self.preprocessor.torch_preprocess(self.agent_batch_next, self.virtual_ref_start[:, -1]))
        policy_loss = torch.mean(self.sum_utility + self.GAMMA_D ** self.FORWARD_STEP * self.value_next)  # Hamilton
        #for i in range(1):
        policy.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
        policy.opt.step()
        policy.scheduler.step()
        self.policy_loss = np.append(self.policy_loss, policy_loss.detach().numpy())
        return policy_loss.detach().numpy()

    def save_data(self, log_dir):
        """
        Save loss data.
        Parameters
        ----------
        log_dir: str
            directory in ./Results_dir.

        Returns
        -------

        """
        np.savetxt(os.path.join(log_dir, "value_loss.txt"), self.value_loss)
        np.savetxt(os.path.join(log_dir, "policy_loss.txt"), self.policy_loss)

    def print_loss_figure(self, iteration, log_dir):
        """
        print figure of loss decent.
        Parameters
        ----------
        iteration: int
            number of iterations.
        log_dir: str
            directory in ./Results_dir.

        Returns
        -------

        """
        plt.figure()
        plt.scatter(range(iteration), np.log10(self.value_loss), c='r', marker=".", s=5., label="policy evaluation")
        plt.scatter(range(iteration), np.log10(self.policy_loss), c='b', marker=".", s=5., label="policy improvement")
        plt.legend(loc='upper right')
        plt.xlabel('iteration')
        plt.ylabel('loss')
        plt.savefig(os.path.join(log_dir, "loss.png"))

class Preprocessor(DynamicsConfig):
    def __init__(self):
        self._norm_matrix = 0.1 * torch.tensor([2, 5, 10, 10, 1/(2*np.pi*self.k_curve), 1/(2*np.pi*self.k_curve)], dtype=torch.float32)

    def torch_preprocess(self, state, ref_start_x):
        real_x = state[:, -1] % (2 * np.pi * self.k_curve)
        real_ref_start_x = ref_start_x % (2 * np.pi * self.k_curve)
        res_state = torch.cat((state[:,:-1], real_x[:, np.newaxis], real_ref_start_x[:, np.newaxis]), 1)
        return torch.mul(res_state, self._norm_matrix)