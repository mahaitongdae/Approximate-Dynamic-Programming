from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from itertools import cycle
from Config import GeneralConfig
from idplot.config import default_cfg
from idplot.utils import cm2inch
import numpy as np
import torch

def init_print():
    intro = 'DEMO OF CHPATER 8,  REINFORCEMENT LEARNING AND CONTROL\n'+ \
                'APPROXIMATE DYNAMIC PROGRMMING FOR LANE KEEPING TASK \n'
    print(intro)

def smooth(data, a=0.5):
    data = np.array(data).reshape(-1, 1)
    for ind in range(data.shape[0] - 1):
        data[ind + 1, 0] = data[ind, 0] * (1-a) + data[ind + 1, 0] * a
    return data

def Numpy2Torch(input,size):
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
    return state_next.clone().detach(), state_r_next.clone().detach()


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
           figsize_scalar=1):
    """
    plot figures
    """
    if (color_list is None) or len(color_list) < figure_num:
        tableau_colors = cycle(mcolors.TABLEAU_COLORS)
        color_list = [next(tableau_colors) for _ in range(figure_num)]

    l = 5
    fig_size = (default_cfg.fig_size * figsize_scalar, default_cfg.fig_size * figsize_scalar)
    _, ax = plt.subplots(figsize=cm2inch(*fig_size), dpi=default_cfg.dpi)
    if figure_num == 1:
        data = [data]

    if color_list is not None:
        for (i, d) in enumerate(data):
            if mode == "xy":
                if i == l - 2:
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
                if i == l - 2:
                    plt.plot(d[0], d[1],  linestyle='-.')
                elif i == l - 1:
                    plt.plot(d[0], d[1],  linestyle='dotted')
                else:
                    plt.plot(d[0], d[1])
            if mode == "y":
                plt.plot(d)
            if mode == "scatter":
                plt.scatter(d[0], d[1], marker=".", s =5.,)

    # plt.tick_params(labelsize=18)
    plt.tick_params(labelsize=default_cfg.tick_size)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname(default_cfg.tick_label_font) for label in labels]
    # labels = ax.get_xticklabels() + ax.get_yticklabels()
    # [label.set_fontname('Times New Roman') for label in labels]
    # font = {'family': 'Times New Roman'} # , 'size': '18'
    if legend is not None:
        plt.legend(legend, loc=legend_loc, ncol=ncol, prop=default_cfg.legend_font)
        # 'lower center'
    plt.xlabel(xlabel, default_cfg.label_font) # , font
    plt.ylabel(ylabel, default_cfg.label_font) # , font
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    plt.tight_layout(pad=default_cfg.pad)

    if fname is None:
        plt.show()
    else:
        plt.savefig(fname)