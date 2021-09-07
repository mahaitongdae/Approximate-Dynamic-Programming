from __future__ import print_function
import torch
import numpy as np
from config import DynamicsConfig
import matplotlib.pyplot as plt
import math

PI = 3.1415926


class VehicleDynamics(DynamicsConfig):

    def __init__(self):
        self._state = torch.zeros([self.BATCH_SIZE, self.DYNAMICS_DIM])
        self.init_state = torch.zeros([self.BATCH_SIZE, self.DYNAMICS_DIM])
        self._reset_index = np.zeros([self.BATCH_SIZE, 1])
        self.initialize_state()
        super(VehicleDynamics, self).__init__()

    def initialize_state(self):
        """
        random initialization of state.

        Returns
        -------

        """
        self.init_state[:, 0] = torch.normal(0.0, 0.6, [self.BATCH_SIZE,])
        self.init_state[:, 1] = torch.normal(0.0, 0.4, [self.BATCH_SIZE,])
        self.init_state[:, 2] = torch.normal(0.0, 0.15, [self.BATCH_SIZE,])
        self.init_state[:, 3] = torch.normal(0.0, 0.1, [self.BATCH_SIZE,])
        self.init_state[:, 4] = torch.linspace(0.0, np.pi, self.BATCH_SIZE)
        init_ref = self.reference_trajectory(self.init_state[:, 4])
        init_ref_all = torch.cat((init_ref, torch.zeros([self.BATCH_SIZE,1])),1)
        self._state = self.init_state
        init_state = self.init_state + init_ref_all
        return init_state

    def relative_state(self, state):
        x_ref = self.reference_trajectory(state[:, -1])
        state_r = state.detach().clone()[:, 0:4] - x_ref  # relative state # todo:修改所有相对坐标更新
        return state_r

    def _state_function(self, state, control):
        """
        State function of vehicle with Pacejka tire model, i.e. \dot(x)=f(x,u)
        Parameters
        ----------
        state: tensor   shape: [BATCH_SIZE, STATE_DIMENSION]
            current state
        control: tensor   shape: [BATCH_SIZE, ACTION_DIMENSION]
            input
        Returns
        -------
        deri_state.T:   tensor shape: [BATCH_SIZE, ]
            f(x,u)
        F_y1:           tensor shape: [BATCH_SIZE, ]
            front axle lateral force
        F_y2:           tensor shape: [BATCH_SIZE, ]
            rear axle lateral force
        alpha_1:        tensor shape: [BATCH_SIZE, ]
            front wheel slip angle
        alpha_2:        tensor shape: [BATCH_SIZE, ]
            rear wheel slip angle

        """

        # state variable
        y = state[:, 0]                 # lateral position
        u_lateral = state[:, 1]         # lateral speed
        beta = u_lateral / self.u       # yaw angle
        psi = state[:, 2]               # heading angle
        omega_r = state[:, 3]           # yaw rate
        x = state[:, 4]                 # longitudinal position

        # inputs
        delta = control[:, 0]           # front wheel steering angle
        delta.requires_grad_(True)

        # slip angle of front and rear wheels
        alpha_1 = -delta + beta + self.a * omega_r / self.u
        alpha_2 = beta - self.b * omega_r / self.u

        # cornering force of front and rear angle, Pacejka tire model
        F_y1 = -self.D * torch.sin(self.C * torch.atan(self.B * alpha_1)) * self.F_z1
        F_y2 = -self.D * torch.sin(self.C * torch.atan(self.B * alpha_2)) * self.F_z2

        # derivative of state
        deri_y = self.u * torch.sin(psi) + u_lateral * torch.cos(psi)
        deri_u_lat = (torch.mul(F_y1, torch.cos(delta)) + F_y2) / (self.m) - self.u * omega_r
        deri_psi = omega_r
        deri_omega_r = (torch.mul(self.a * F_y1, torch.cos(delta)) - self.b * F_y2) / self.I_zz
        deri_x = self.u * torch.cos(psi) - u_lateral * torch.sin(psi)

        deri_state = torch.cat((deri_y[np.newaxis, :],
                                deri_u_lat[np.newaxis, :],
                                deri_psi[np.newaxis, :],
                                deri_omega_r[np.newaxis, :],
                                deri_x[np.newaxis, :]), 0)

        return deri_state.T, F_y1, F_y2, alpha_1, alpha_2

    def _state_function_linear(self, state, control):
        """
        State function of vehicle with linear tire model and linear approximation, i.e. \dot(x) = Ax + Bu
        Parameters
        ----------
        state: tensor   shape: [BATCH_SIZE, STATE_DIMENSION]
            current state
        control: tensor   shape: [BATCH_SIZE, ACTION_DIMENSION]
            input
        Returns
        -------
        deri_state.T:   tensor shape: [BATCH_SIZE, ]
            f(x,u)
        F_y1:           tensor shape: [BATCH_SIZE, ]
            front axle lateral force
        F_y2:           tensor shape: [BATCH_SIZE, ]
            rear axle lateral force
        alpha_1:        tensor shape: [BATCH_SIZE, ]
            front wheel slip angle
        alpha_2:        tensor shape: [BATCH_SIZE, ]
            rear wheel slip angle

        """

        # state variable
        y = state[:, 0]                 # lateral position
        u_lateral = state[:, 1]         # lateral speed
        beta = u_lateral / self.u       # yaw angle
        psi = state[:, 2]               # heading angle
        omega_r = state[:, 3]           # yaw rate
        x = state[:, 4]                 # longitudinal position

        # inputs
        delta = control[:, 0]           # front wheel steering angle
        delta.requires_grad_(True)

        # slip angle of front and rear wheels, with small angle approximation
        alpha_1 = -delta + beta + self.a * omega_r / self.u
        alpha_2 = beta - self.b * omega_r / self.u

        # cornering force of front and rear angle, linear tire model
        F_y1 = - self.k1 * alpha_1
        F_y2 = - self.k2 * alpha_2

        # derivative of state
        # deri_y = self.u * psi + u_lateral
        deri_y = self.u * torch.sin(psi) + u_lateral * torch.cos(psi)
        deri_u_lat = (torch.mul(F_y1, torch.cos(delta)) + F_y2) / (self.m) - self.u * omega_r
        deri_psi = omega_r
        deri_omega_r = (torch.mul(self.a * F_y1, torch.cos(delta)) - self.b * F_y2) / self.I_zz
        deri_x = self.u * torch.cos(psi) - u_lateral * torch.sin(psi)

        deri_state = torch.cat((deri_y[np.newaxis, :],
                                deri_u_lat[np.newaxis, :],
                                deri_psi[np.newaxis, :],
                                deri_omega_r[np.newaxis, :],
                                deri_x[np.newaxis, :]), 0)

        return deri_state.T, F_y1, F_y2, alpha_1, alpha_2

    def reference_trajectory(self, state):
        """

        Parameters
        ----------
        state               shape: [BATCH_SIZE,]       longitudinal location x

        Returns
        -------
        state_ref.T:        shape: [BATCH_SIZE, 4]      reference trajectory

        """

        if self.reference_traj == 'SIN':
            k = self.k_curve
            a = self.a_curve
            y_ref = a * torch.sin(k * state)
            psi_ref = torch.atan(a * k * torch.cos(k * state))
        elif self.reference_traj == 'DLC':
            width = 3.5
            line1 = 50
            straight = 50
            cycle = 3 * straight + 2 * line1
            x = state % cycle
            lane_position = torch.zeros([len(state), ])
            lane_angle = torch.zeros([len(state), ])
            for i in range(len(state)):
                if x[i] <= 50:
                    lane_position[i] = 0
                    lane_angle[i] = 0
                elif 50 < x[i] and x[i] <= 90:
                    lane_position[i] = 3.5 / 40 * x[i] - 4.375
                    lane_angle[i] = np.arctan(3.5 / 40)
                elif 90 < x[i] and x[i] <= 140:
                    lane_position[i] = 3.5
                    lane_angle[i] = 0
                elif x[i] > 180:
                    lane_position[i] = 0
                    lane_angle[i] = 0
                elif 140 < x[i] and x[i] <= 180:
                    lane_position[i] = -3.5 / 40 * x[i] + 15.75
                    lane_angle[i] = -np.arctan(3.5 / 40)
                else:
                    lane_position[i] = 0.
                    lane_angle[i] = 0.

            y_ref = lane_position
            psi_ref = lane_angle

        zeros = torch.zeros([len(state), ])
        state_ref = torch.cat((y_ref[np.newaxis, :],
                                zeros[np.newaxis, :],
                                psi_ref[np.newaxis, :],
                                zeros[np.newaxis, :]), 0)
        return state_ref.T


    def step(self, state, control):
        """
        step ahead with discrete state function, i.e. x'=f(x,u)
        Parameters
        ----------
        state: tensor   shape: [BATCH_SIZE, STATE_DIMENSION]
            current state
        control: tensor   shape: [BATCH_SIZE, ACTION_DIMENSION]
            current control signal

        Returns
        -------
        state_next:     tensor shape: [BATCH_SIZE, ]
            x'
        f_xu:           tensor shape: [BATCH_SIZE, ]
            f(x,u)
        utility:        tensor shape: [BATCH_SIZE, ]
        utility, i.e. l(x,u)
        F_y1:           tensor shape: [BATCH_SIZE, ]
            front axle lateral force
        F_y2:           tensor shape: [BATCH_SIZE, ]
            rear axle lateral force
        alpha_1:        tensor shape: [BATCH_SIZE, ]
            front wheel slip angle
        alpha_2:        tensor shape: [BATCH_SIZE, ]
            rear wheel slip angle

        """
        if self.nonlinearity:
            deri_state, F_y1, F_y2, alpha_1, alpha_2 = self._state_function(state, control)
        else:
            deri_state, F_y1, F_y2, alpha_1, alpha_2 = self._state_function_linear(state, control)
        state_next = state + self.Ts * deri_state
        utility = self.utility(state, control)
        f_xu = deri_state[:, 0:4]
        return state_next, f_xu, utility, F_y1, F_y2, alpha_1, alpha_2

    def step_relative(self, state, u):
        """

        Parameters
        ----------
        state_r
        u_r

        Returns
        -------

        """
        x_ref = self.reference_trajectory(state[:, -1])
        state_r = state.detach().clone()  # relative state
        state_r[:, 0:4] = state_r[:, 0:4] - x_ref
        state_next, deri_state, utility, F_y1, F_y2, alpha_1, alpha_2 = self.step(state, u)
        state_r_next_bias, _, _, _, _, _, _ = self.step(state_r, u)  # update by relative value
        state_r_next = state_r_next_bias.detach().clone()
        state_r_next_bias[:, [0, 2]] = state_next[:, [0, 2]]  # y psi with reference update by absolute value
        x_ref_next = self.reference_trajectory(state_next[:, -1])
        state_r_next[:, 0:4] = state_r_next_bias[:, 0:4] - x_ref_next
        utility = self.utility(state_r_next, u)
        return state_next.clone().detach(), state_r_next.clone().detach()

    @staticmethod
    def utility(state, control):
        """

        Parameters
        ----------
        state: tensor       shape: [BATCH_SIZE, STATE_DIMENSION]
            current state
        control: tensor     shape: [BATCH_SIZE, ACTION_DIMENSION]
            current control signal

        Returns
        -------
        utility: tensor   shape: [BATCH_SIZE, ]
            utility, i.e. l(x,u)
        """
        utility = 0.2 * torch.pow(state[:, 0], 2) + 1 * torch.pow(state[:, 2], 2) + 5 * torch.pow(control[:, 0], 2)
        return utility
