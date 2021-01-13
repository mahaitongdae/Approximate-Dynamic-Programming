"""
    <Reinforcement Learning and Control>(Year 2020)
    by Shengbo Eben Li
        @ Intelligent Driving Lab, Tsinghua University

    ADP example for lane keeping problem in a circle road

    [Method]
    Model predictive control(MPC) as comparison

"""
from Solver import Solver
from Config import DynamicsConfig
from matplotlib import pyplot as plt
from utils import Numpy2Torch
import numpy as np
import time
import os
import Dynamics
import torch





log_dir = "Results_dir/Comparison_Data"
config = DynamicsConfig()
solver=Solver()
# x = [0.0, 0.0, 0.033, 0.0, 0.0]
# state_history = np.array(x)
# control_history = np.empty([0, 1])
# time_init = time.time()
# for i in range(config.NP_TOTAL):
#     state, control = solver.mpc_solver(x, 60)
#     x = state[1]
#     u = control[0]
#     state_history = np.append(state_history, x)
#     control_history = np.append(control_history, u)
#     x = x.tolist()
#     print("steps:{:3d}".format(i) + " | state: " + str(x))
# print("MPC calculating time: {:.3f}".format(time.time() - time_init) + "s")
# state_history = state_history.reshape(-1,config.DYNAMICS_DIM)
# # np.savetxt(os.path.join(log_dir, 'structured_MPC_state.txt'), state_history)
# # np.savetxt(os.path.join(log_dir, 'structured_MPC_control.txt'), control_history)

statemodel_plt = Dynamics.VehicleDynamics()
if config.tire_model == 'Fiala':
    state = torch.tensor([[1.0, 0.0, config.psi_init, 0.0, config.u, 0.0]])
else:
    state = torch.tensor([[0.0, 0.0, config.psi_init, 0.0, 0.0]])
state.requires_grad_(False)
x_ref = statemodel_plt.reference_trajectory(state[:, -1])
state_r = state.detach().clone()
state_r[:, 0:4] = state_r[:, 0:4] - x_ref
state_history = state.detach().numpy()
x = np.array([0.])
plot_length = 300
control_history = []
state_r_history = state_r
cal_time = 0
plt.figure()
for i in range(plot_length):
    x = state_r.tolist()[0]
    time_start = time.time()
    temp, control = solver.mpcSolver(x, config.NP)
    plt.plot(temp[:,-1],temp[:,0])
    cal_time += time.time() - time_start
    u = Numpy2Torch(control[0], (-1,config.ACTION_DIM))

    state_next, deri_state, utility, F_y1, F_y2, alpha_1, alpha_2 = statemodel_plt.step(state, u)
    state_r_old, _, _, _, _, _, _ = statemodel_plt.step(state_r, u)
    state_r = state_r_old.detach().clone()
    state_r[:, [0, 2]] = state_next[:, [0, 2]]
    x_ref = statemodel_plt.reference_trajectory(state_next[:, -1])
    state_r[:, 0:4] = state_r[:, 0:4] - x_ref
    state = state_next.clone().detach()
    s = state_next.detach().numpy()
    state_history = np.append(state_history, s, axis=0)
    control_history = np.append(control_history, u.detach().numpy())
    state_r_history = np.append(state_history, state_r.detach().numpy())
print("MPC calculating time: {:.3f}".format(cal_time) + "s")
plt.show()
# np.savetxt(os.path.join(log_dir, 'structured_MPC_state.txt'), state_history)
# np.savetxt(os.path.join(log_dir, 'structured_MPC_control.txt'), control_history)
state_r_history = state_r_history.reshape([-1,5])
plt.figure(1)
plt.plot(state_history[:,-1],state_history[:,0], label="trajectory")
plt.plot(state_r_history[:,-1],state_r_history[:,0], label="$trajectory_r$")
plt.plot(state_history[:,-1],config.a_curve * np.sin(config.k_curve*state_history[:,-1]), label="reference")
plt.legend(loc="upper right")
plt.show()
plt.figure(2)
plt.plot(state_history[0:-1,-1], control_history)
plt.title('Control')
plt.show()