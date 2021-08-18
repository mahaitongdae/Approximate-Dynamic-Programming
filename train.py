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

        self.agent_batch = torch.empty([self.BATCH_SIZE, self.DYNAMICS_DIM])
        self.state_batch = torch.empty([self.BATCH_SIZE, self.STATE_DIM])
        self.init_index = np.ones([self.BATCH_SIZE, 1])

        self.x_forward = []
        self.u_forward = []
        self.L_forward = []
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
        self.initialize_state()

    def initialize_state(self):
        self.state_batch[:, 0] = torch.normal(0.0, 0.3, [self.BATCH_SIZE, ])
        self.state_batch[:, 1] = torch.normal(0.0, 0.2, [self.BATCH_SIZE, ])
        self.state_batch[:, 2] = torch.normal(0.0, 0.1, [self.BATCH_SIZE, ])
        self.state_batch[:, 3] = torch.normal(0.0, 0.06, [self.BATCH_SIZE, ])
        self.agent_batch[:, 4] = torch.linspace(0.0, np.pi, self.BATCH_SIZE)
        init_ref = self.dynamics.reference_trajectory(self.agent_batch[:, 4])
        self.agent_batch[:, 0:4] = self.state_batch + init_ref
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
        state = state[:, 0:4] - ref_state
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
        ref_trajectory = dynamics.reference_trajectory(self.agent_batch[:, -1])
        self.state_batch = self.agent_batch[:, 0:4] - ref_trajectory
        control = policy.forward(self.state_batch)
        self.agent_batch, self.state_batch = dynamics.step_relative(self.agent_batch, control)
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
                reference = dynamics.reference_trajectory(self.agent_batch[:,-1])
                self.state_batch = dynamics.relative_state(self.x_forward[i])
                self.u_forward[i] = policy.forward(self.state_batch)
                self.x_forward[i + 1], _, _, _, _, _, _ = dynamics.step(self.x_forward[i], self.u_forward[i])
                ref_state_next = self.x_forward[i + 1][:, 0:4] - reference
                self.L_forward[i] = dynamics.utility(ref_state_next, self.u_forward[i])
            else:
                ref_state = self.x_forward[i][:, 0:4] - reference
                self.u_forward[i] = policy.forward(ref_state)
                self.x_forward[i + 1], _, _, _, _, _, _ = dynamics.step(self.x_forward[i],
                                                                                        self.u_forward[i])
                ref_state_next = self.x_forward[i + 1][:, 0:4] - reference
                self.L_forward[i] = dynamics.utility(ref_state_next, self.u_forward[i])
        self.agent_batch_next = self.x_forward[-1]
        self.state_batch_next = self.agent_batch_next[:, 0:4] - reference
        self.value_next = value.forward(self.state_batch_next)
        self.utility = torch.zeros([self.FORWARD_STEP, self.BATCH_SIZE], dtype=torch.float32)
        for i in range(self.FORWARD_STEP):
            self.utility[i] = self.L_forward[i].clone()
        self.sum_utility = torch.sum(self.utility,0)
        target_value = self.sum_utility.detach() + self.value_next.detach()
        value_now = value.forward(self.state_batch)
        value_equilibrium = value.forward(self.equilibrium_state)
        value_loss = 1 / 2 * torch.mean(torch.pow((target_value - value_now), 2)) \
                     + 10 * torch.pow(value_equilibrium, 2)
        self.state_batch.requires_grad_(False)
        value.zero_grad()
        value_loss.backward()
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
        self.value_next = value.forward(self.state_batch_next)
        policy_loss = torch.mean(self.sum_utility + self.value_next)  # Hamilton
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
        torch.save(self.agent_batch, os.path.join(log_dir, "agent_buffer.pth"))

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

