"""
Copyright (c). All Rights Reserved.
<Reinforcement Learning and Control> (Year 2020)
Intelligent Driving Lab (iDLab), Tsinghua University

by Shengbo Eben Li & Haitong Ma

Description: Chapter 6: RL example for lane keeping problem in a curve road;
             Approximate dynamic programming with structured policy

Update Date: 2021-09-06, Haitong Ma: Rewrite code formats
"""
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from itertools import cycle
from config import PlotConfig, DynamicsConfig
import numpy as np
import torch

def cm2inch(*tupl):

    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)

def smooth(data, a=0.5):
    data = np.array(data).reshape(-1, 1)
    for ind in range(data.shape[0] - 1):
        data[ind + 1, 0] = data[ind, 0] * (1-a) + data[ind + 1, 0] * a
    return data

def numpy2torch(input, size):
    """

    Parameters
    ----------
    input

    Returns
    -------

    """
    u = np.array(input, dtype='float32').reshape(size)
    return torch.from_numpy(u)

def step_relative(statemodel, state, u):
    """

    Parameters
    ----------
    state_r
    u_r

    Returns
    -------

    """
    x_ref = statemodel.reference_trajectory(state[:, -1])
    state_r = state.detach().clone()  # relative state
    state_r[:, 0:4] = state_r[:, 0:4] - x_ref
    state_next, deri_state, utility, F_y1, F_y2, alpha_1, alpha_2 = statemodel.step(state, u)
    state_r_next_bias, _, _, _, _, _, _ = statemodel.step(state_r, u) # update by relative value
    state_r_next = state_r_next_bias.detach().clone()
    state_r_next_bias[:, [0, 2]] = state_next[:, [0, 2]]            # y psi with reference update by absolute value
    x_ref_next = statemodel.reference_trajectory(state_next[:, -1])
    state_r_next[:, 0:4] = state_r_next_bias[:, 0:4] - x_ref_next
    return state_next.clone().detach(), state_r_next.clone().detach(), x_ref.detach().clone()

def recover_absolute_state(state_r_predict, x_ref, length=None):
    if length is None:
        length = state_r_predict.shape[0]
    c = DynamicsConfig()
    ref_predict = [x_ref]
    for i in range(length-1):
        ref_t = np.copy(ref_predict[-1])
        # ref_t[0] += c.u * c.Ts * np.tan(x_ref[2])
        ref_predict.append(ref_t)
    state = state_r_predict[:, 0:4] + ref_predict
    return state, np.array(ref_predict)

def idplot(data,
           figure_num=1,
           mode="xy",
           fname=None,
           xlabel=None,
           ylabel=None,
           legend=None,
           legend_loc="best",
           color_list=None,
           xlim=None,
           ylim=None,
           ncol=1,
           figsize_scalar=1,
           tight_layout=True):
    """
    plot figures
    """
    if (color_list is None) or len(color_list) < figure_num:
        tableau_colors = cycle(mcolors.TABLEAU_COLORS)
        color_list = [next(tableau_colors) for _ in range(figure_num)]

    l = 5
    fig_size = (PlotConfig.fig_size * figsize_scalar, PlotConfig.fig_size * figsize_scalar)
    _, ax = plt.subplots(figsize=cm2inch(*fig_size), dpi=PlotConfig.dpi)
    if figure_num == 1:
        data = [data]

    if color_list is not None:
        for (i, d) in enumerate(data):
            if mode == "xy":
                if i == 0:
                    plt.plot(d[0], d[1], linestyle='-.', color=color_list[i])
                elif i == l - 1:
                    plt.plot(d[0], d[1], linestyle='dotted', color=color_list[i])
                else:
                    plt.plot(d[0], d[1], color=color_list[i])
            if mode == "y":
                plt.plot(d, color=color_list[i])
            if mode == "scatter":
                plt.scatter(d[0], d[1], color=color_list[i], marker=".", s =5.,)
    else:
        for (i, d) in enumerate(data):
            if mode == "xy":
                if i == 0:
                    plt.plot(d[0], d[1],  linestyle='-.')
                elif i == l - 1:
                    plt.plot(d[0], d[1],  linestyle='dotted')
                else:
                    plt.plot(d[0], d[1])
            if mode == "y":
                plt.plot(d)
            if mode == "scatter":
                plt.scatter(d[0], d[1], marker=".", s =5.,)

    plt.tick_params(labelsize=PlotConfig.tick_size)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname(PlotConfig.tick_label_font) for label in labels]
    if legend is not None:
        plt.legend(legend, loc=legend_loc, ncol=ncol, prop=PlotConfig.legend_font)
    plt.xlabel(xlabel, PlotConfig.label_font)
    plt.ylabel(ylabel, PlotConfig.label_font)
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    if tight_layout:
        plt.tight_layout(pad=PlotConfig.pad)
    else:
        plt.tight_layout(pad=3*PlotConfig.pad)

    if fname is None:
        plt.show()
    else:
        plt.savefig(fname)


