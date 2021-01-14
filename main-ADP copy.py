"""
    <Reinforcement Learning and Control>(Year 2020)
    by Shengbo Eben Li
        @ Intelligent Driving Lab, Tsinghua University

    ADP example for lane keeping problem in a curve road

    [Method]
    Approximate dynamic programming with structured policy

    """
import Dynamics
import numpy as np
import torch
import os
from Network import Actor, Critic
from Train import Train
from datetime import datetime
from Simulation import simulation
from plot_figure import adp_simulation_plot, plot_comparison
from Config import GeneralConfig
from utils import init_print
from tensorboardX import SummaryWriter


# Parameters
init_print()
MAX_ITERATION = 1000       # max iterations
LR_P = 8e-4                 # learning rate of policy net
LR_V = 3e-3                # learning rate of value net

# tasks
TRAIN_FLAG = 0
LOAD_PARA_FLAG = 1
SIMULATION_FLAG = 1

# Set random seed
np.random.seed(0)
torch.manual_seed(0)

# initialize policy and value net, model of vehicle dynamics
config = GeneralConfig()
policy = Actor(config.STATE_DIM, config.ACTION_DIM, lr=LR_P)
value = Critic(config.STATE_DIM, 1, lr=LR_V)
vehicleDynamics = Dynamics.VehicleDynamics()
state_batch = vehicleDynamics.initialize_state()
writer = SummaryWriter()

# Training
iteration_index = 0
if LOAD_PARA_FLAG == 1:
    print("********************************* LOAD PARAMETERS *********************************")
    # load pre-trained parameters
    load_dir = "./Results_dir/2021-01-14-10-40-1000"
    policy.load_parameters(load_dir)
    value.load_parameters(load_dir)

if TRAIN_FLAG == 1:
    print_iters = 100
    print("********************************** START TRAINING **********************************")
    print("************************** PRINT LOSS EVERY "+ str(print_iters) + "iterations ***************************")
    # train the network by policy iteration
    train = Train()
    # train.agent_batch = vehicleDynamics.initialize_state()
    if LOAD_PARA_FLAG == 1:
        train.agent_batch = torch.load(os.path.join(load_dir,'agent_buffer.pth'))
        train.setInitState()
    else:
        # train.agent_batch = vehicleDynamics.initialize_state()
        train.initialize_state()

    while True:
        train.update_state(policy, vehicleDynamics)
        value_loss = train.policy_evaluation(policy, value, vehicleDynamics)
        policy_loss = train.policy_improvement(policy, value)
        writer.add_scalar("policy_loss", policy_loss, global_step=iteration_index)
        writer.add_scalar("value_loss", value_loss, global_step=iteration_index)
        iteration_index += 1

        # print train information
        if iteration_index % print_iters == 0:
            log_trace = "iteration:{:3d} | "\
                        "policy_loss:{:3.3f} | " \
                        "value_loss:{:3.3f}".format(iteration_index, float(policy_loss), float(value_loss))
            print(log_trace)

        # save parameters, run simulation and plot figures
        if iteration_index == MAX_ITERATION:
            # ==================== Set log path ====================
            print("********************************* FINISH TRAINING **********************************")
            log_dir = "./Results_dir/" + datetime.now().strftime("%Y-%m-%d-%H-%M-" + str(iteration_index))
            os.makedirs(log_dir, exist_ok=True)
            value.save_parameters(log_dir)
            policy.save_parameters(log_dir)
            train.print_loss_figure(MAX_ITERATION, log_dir)
            train.save_data(log_dir)
            break

if SIMULATION_FLAG == 1:
    print("********************************* START SIMULATION *********************************")
    methods = ['MPC','ADP']
    simu_dir = "./Simulation_dir/" + datetime.now().strftime("%Y-%m-%d-%H-%M")
    os.makedirs(simu_dir, exist_ok=True)
    if TRAIN_FLAG == 0:
        simulation(methods, load_dir, simu_dir)
    else:
        simulation(methods, log_dir, simu_dir)