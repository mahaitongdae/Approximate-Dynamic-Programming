import numpy as np
import torch
from matplotlib import pyplot as plt
from Dynamics import VehicleDynamics
from utils import Numpy2Torch, myplot

# path_dir: directory to save simulation results
# load_dir: directory to save network models

path_dir = "./Simulation_dir/2021-01-14-14-54"
load_dir = "./Results_dir/2020-06-11-13-24-10"


class DrawADP:
    def __init__(self, value, policy):
        self.value = value
        self.policy = policy

    def draw(self, network):
        '''
        Given fixed states:[v,psi,omega], show different value function of different y value.
        '''
        if network == "value":
            net = self.value
        elif network == "policy":
            net = self.policy

        y = np.arange(-1, 1, 0.002) 
        ly = len(y)
        v = np.zeros(ly) # value/policy function

        # states=[y,v,psi,omega],size:(ly,4),the last three colomuns are the same for each row.
        states = np.c_[y[:, np.newaxis],
                       np.dot(np.ones((ly, 1)), np.array([[1, 0.2, 1.0]]))]
        states = torch.from_numpy(states).float()

        with torch.no_grad():
            results = net.forward(states)  # value or policy
            results = results.numpy()

        plt.plot(y, results)
        plt.show()

    def show_info(self, path_dir):
        state = np.loadtxt(path_dir)
        y = state[:, 0]
        v = state[:, 1]
        psi = state[:, 2]
        omega = state[:, 3]
        plt.figure(figsize=(15, 8))
        for i in range(4):
            plt.subplot(2, 2, i + 1)
            plt.hist(state[:, i], bins=50)
        plt.show()

    def show_value_history(self,path_dir):
        value=np.loadtxt(os.path.join(path_dir,"ADP_value.txt"))
        state = np.loadtxt(os.path.join(path_dir,"ADP_state.txt"))
        y=state[:,0]
        # longitudinal_distance=np.linspace(0,500,len(value))
        longitudinal_distance=state[:,-1]

        dy=VehicleDynamics()
        adp_ref = dy.reference_trajectory(Numpy2Torch(state[:, 4], state[:, 4].shape)).numpy()
        y_ref=adp_ref[:,0]#y reference

        assert len(y_ref)==len(longitudinal_distance)
        assert len(value)==len(longitudinal_distance)

        fig,ax1=plt.subplots(figsize = (10, 5), facecolor='white')
        ax1.plot(longitudinal_distance,value,"b-")
        ax1.set_xlabel("longitudinal distance (m)")
        ax1.set_ylabel("value function")

        ax2 = ax1.twinx()
        ax2.plot(longitudinal_distance,y,"r:")
        ax2.plot(longitudinal_distance,y_ref,"c:")
        # ax2.plot(longitudinal_distance,y-y_ref,"m:")
        ax2.set_ylabel("lateral distance (m)")

        fig.legend(["value","y","y_ref"],loc=1, bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
        # fig.legend(["value","y-y_ref"],loc=1, bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)

        if not os.path.exists(os.path.join(path_dir,"visulization")):
            os.mkdir(os.path.join(path_dir,"visulization"))
        plt.savefig(os.path.join(path_dir,"visulization","value_history.png"))
        plt.show()


if __name__ == "__main__":
    import os

    from Config import GeneralConfig
    from Network import Actor, Critic
    LR_P = 8e-4
    LR_V = 3e-3
    config = GeneralConfig()
    policy = Actor(config.STATE_DIM, config.ACTION_DIM, lr=LR_P)
    value = Critic(config.STATE_DIM, 1, lr=LR_V)

    policy.load_parameters(load_dir)
    value.load_parameters(load_dir)

    drawing = DrawADP(value, policy)
    # drawing.show_info(os.path.join(path_dir,"ADP_state.txt"))
    # drawing.draw("value")
    # drawing.draw("policy")
    drawing.show_value_history(path_dir)
