import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn import init

PI = np.pi

class Actor(nn.Module):
    def __init__(self, input_size, output_size, order=1, lr=0.001):
        super(Actor, self).__init__()

        # parameters
        self._out_gain = PI / 9
        # self._norm_matrix = 0.1 * torch.tensor([2, 1, 10, 10], dtype=torch.float32)
        self._norm_matrix = 0.1 * torch.tensor([1, 1, 1, 1], dtype=torch.float32)

        # initial NNs
        self.layers = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, output_size),
            nn.Tanh()
        )
        # initial optimizer
        self.opt = torch.optim.Adam(self.parameters(), lr=lr)
        self._initialize_weights()

        # zeros state value
        self._zero_state = torch.tensor([0.0, 0.0, 0.0, 0.0])

    def forward(self, x):
        """
        Parameters
        ----------
        x: polynomial features, shape:[batch, feature dimension]

        Returns
        -------
        value of current state
        """
        temp = torch.mul(x, self._norm_matrix)
        x = torch.mul(self._out_gain, self.layers(temp))
        return x

    def _initialize_weights(self):
        """
        initial parameter using xavier
        """

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                init.constant_(m.bias, 0.0)

    def loss_function(self, utility, p_V_x, f_xu):

        hamilton = utility + torch.diag(torch.mm(p_V_x, f_xu.T))
        loss = torch.mean(hamilton)
        return loss

    def predict(self, x):

        return self.forward(x).detach().numpy()

    def save_parameters(self, logdir):
        """
        save model
        Parameters
        ----------
        logdir, the model will be saved in this path

        """
        torch.save(self.state_dict(), os.path.join(logdir, "actor.pth"))

    def load_parameters(self, load_dir):
        self.load_state_dict(torch.load(os.path.join(load_dir,'actor.pth')))


class Critic(nn.Module):
    """
    NN of value approximation
    """

    def __init__(self, input_size, output_size, order=1, lr=0.001):
        super(Critic, self).__init__()

        # initial parameters of actor
        self.layers = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, output_size),
            nn.ReLU()
        )
        self._norm_matrix = 0.1 * torch.tensor([2, 5, 10, 10], dtype=torch.float32)

        # initial optimizer
        self.opt = torch.optim.Adam(self.parameters(), lr=lr)
        self._initialize_weights()


        # zeros state value
        self._zero_state = torch.tensor([0.0, 0.0, 0.0, 0.0])

    def predict(self, state):
        """
        Parameters
        ----------
        state: current state [batch, feature dimension]

        Returns
        -------
        out: value np.array [batch, 1]
        """

        return self.forward(state).detach().numpy()

    def forward(self, x):
        """
        Parameters
        ----------
        x: polynomial features, shape:[batch, feature dimension]

        Returns
        -------
        value of current state
        """
        x = torch.mul(x, self._norm_matrix)
        x = self.layers(x)
        return x


    def _initialize_weights(self):
        """
        initial paramete using xavier
        """

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                init.constant_(m.bias, 0.0)


    def save_parameters(self, logdir):
        """
        save model
        Parameters
        ----------
        logdir, the model will be saved in this path

        """
        torch.save(self.state_dict(), os.path.join(logdir, "critic.pth"))

    def load_parameters(self, load_dir):
        self.load_state_dict(torch.load(os.path.join(load_dir,'critic.pth')))