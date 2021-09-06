"""
    <Reinforcement Learning and Control>(Year 2020)
    by Shengbo Eben Li
        @ Intelligent Driving Lab, Tsinghua University

    ADP example for lane keeping problem in a curve road

    [Method]
    Approximate dynamic programming with structured policy

"""
import dynamics
import numpy as np
import torch
import os
from network import Actor, Critic
from train import Train
from datetime import datetime
from simulation import simulation
from config import GeneralConfig


# Parameters
intro = 'DEMO OF CHPATER 8,  REINFORCEMENT LEARNING AND CONTROL\n'+ \
                'APPROXIMATE DYNAMIC PROGRMMING FOR LANE KEEPING TASK \n'
print(intro)
METHODS = ['MPC-5',
           'MPC-10',
           'MPC-30', # MPC-"prediction steps of MPC",
           'ADP',    # Approximate dynamic programming,
           'OP']     # Open-loop
MAX_ITERATION = 11000        # max iterations
LR_P = 6e-4                  # learning rate of policy net
LR_V = 6e-3                  # learning rate of value net

# tasks
TRAIN_FLAG = 1
LOAD_PARA_FLAG = 0
SIMULATION_FLAG = 0

# Set random seed
np.random.seed(0)
torch.manual_seed(0)

# initialize policy and value net, model of vehicle dynamics
config = GeneralConfig()
policy = Actor(config.STATE_DIM, config.ACTION_DIM, lr=LR_P)
value = Critic(config.STATE_DIM, 1, lr=LR_V)
vehicleDynamics = dynamics.VehicleDynamics()
state_batch = vehicleDynamics.initialize_state()

# Training
iteration_index = 0
if LOAD_PARA_FLAG == 1:
    print("********************************* LOAD PARAMETERS *********************************")
    # load pre-trained parameters
    load_dir = "./trained_results/2020-10-09-14-42-10000"
    policy.load_parameters(load_dir)
    value.load_parameters(load_dir)

if TRAIN_FLAG == 1:
    print_iters = 10
    print("********************************** START TRAINING **********************************")
    print("************************** PRINT LOSS EVERY "+ str(print_iters) + "iterations ***************************")
    # train the network by policy iteration
    train = Train()

    while True:
        train.update_state(policy, vehicleDynamics)
        value_loss = train.policy_evaluation(policy, value, vehicleDynamics)
        policy_loss = train.policy_improvement(policy, value)
        iteration_index += 1

        # print train information
        if iteration_index % print_iters == 0:
            log_trace = "iteration:{:3d} | "\
                        "policy_loss:{:3.3f} | " \
                        "value_loss:{:3.3f}".format(iteration_index, float(policy_loss), float(value_loss))
            print(log_trace)

        # save parameters, run simulation and plot figures
        if (iteration_index >= 10000 and iteration_index % 100 == 0) or iteration_index == MAX_ITERATION:
            # ==================== Set log path ====================
            #
            log_dir = "./Results_dir/" + datetime.now().strftime("%Y-%m-%d-%H-%M-" + str(iteration_index))
            os.makedirs(log_dir, exist_ok=True)
            value.save_parameters(log_dir)
            policy.save_parameters(log_dir)
            # train.print_loss_figure(MAX_ITERATION, log_dir)
            train.save_data(log_dir)
            # if iteration_index % 1000 == 0:
            simu_dir = "./Simulation_dir/" + datetime.now().strftime("%Y-%m-%d-%H-%M-" + str(iteration_index))
            os.makedirs(simu_dir, exist_ok=True)
            simulation(METHODS, log_dir, simu_dir)
            if iteration_index == MAX_ITERATION:
                break

if SIMULATION_FLAG == 1:
    print("********************************* START SIMULATION *********************************")
    simu_dir = "./Simulation_dir/" + datetime.now().strftime("%Y-%m-%d-%H-%M")
    os.makedirs(simu_dir, exist_ok=True)
    if TRAIN_FLAG == 0:
        simulation(METHODS, load_dir, simu_dir)
    else:
        simulation(METHODS, log_dir, simu_dir)
