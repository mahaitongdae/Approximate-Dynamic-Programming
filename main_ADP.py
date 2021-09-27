"""
Copyright (c). All Rights Reserved.
<Reinforcement Learning and Control> (Year 2020)
Intelligent Driving Lab (iDLab), Tsinghua University

by Shengbo Eben Li & Haitong Ma

Description: Chapter 8: RL example for lane keeping problem in a curve road;
             Approximate dynamic programming with structured policy

"""

# =================== load package ====================
import dynamics
import numpy as np
import torch
import os
from network import Actor, Critic
from train import Train
from datetime import datetime
from main_simuMPC import simulation
from config import GeneralConfig


# ============= Setting hyper-parameters ===============
intro = 'DEMO OF CHAPTER 8: \n'+ \
                'ADP FOR Veh Track Ctrl \n'
print(intro)
METHODS = ['MPC-5',
           'MPC-10',
           'MPC-30', # MPC-"prediction steps of MPC",
           'ADP',    # Approximate dynamic programming,
           'OPEN']     # Open-loop
MAX_ITERATION = 15000        # max iterations todo
LR_P = 5.6e-4                  # learning rate of policy net todo
LR_V = 6e-3                  # learning rate of value net todo

# Environment tasks
TRAIN_FLAG = 1
LOAD_PARA_FLAG = 0
SIMULATION_FLAG = 1

# ============= Setting random seed ===============
np.random.seed(0)
torch.manual_seed(0)

#  ============= initialization  ==================
config = GeneralConfig()
policy = Actor(config.STATE_DIM, config.ACTION_DIM, lr=LR_P)
value = Critic(config.STATE_DIM, 1, lr=LR_V)
vehicleDynamics = dynamics.VehicleDynamics()
state_batch = vehicleDynamics.initialize_state()

# ================== Training =====================
iteration_index = 0

if TRAIN_FLAG == 1:
    print_iters = 50
    print("********************************** START TRAINING **********************************")
    print("************************** PRINT LOSS EVERY "+ str(print_iters) + "iterations ***************************")
    # train network by policy iteration
    train = Train()

    while True:
        train.update_state(policy, vehicleDynamics)
        value_loss = train.policy_evaluation(policy, value, vehicleDynamics)
        policy_loss = train.policy_improvement(policy, value)
        iteration_index += 1

        # print train information
        if iteration_index % print_iters == 0:
            log_trace = "Cycle: {:3d} | "\
                        "policy_loss: {:3.3f} | " \
                        "value_loss: {:3.3f}".format(iteration_index, float(policy_loss), float(value_loss))
            print(log_trace)

        # save parameters, run simulation and plot figures
        if iteration_index == MAX_ITERATION:
            log_dir = "./Results/" + datetime.now().strftime("%Y-%m%d-%H%M")
            os.makedirs(log_dir, exist_ok=True)
            value.save_parameters(log_dir)
            policy.save_parameters(log_dir)
            train.save_data(log_dir)
            break

# ================== Simulation =====================
if SIMULATION_FLAG == 1:
    print("********************************* START SIMULATION *********************************")
    simu_dir = log_dir + "-Simu"
    os.makedirs(simu_dir, exist_ok=True)
    simulation(METHODS, log_dir, simu_dir)
