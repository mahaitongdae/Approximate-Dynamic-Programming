import numpy as np
import matplotlib.pyplot as plt
from Solver import Solver

def txt_read(filename):
    data = []
    with open(filename, 'r') as f:#with语句自动调用close()方法
        line = f.readline()
        while line:
            eachline = line.split()###按行读取文本文件，每行数据以列表形式返回
            read_data = [ float(x) for x in eachline[0:7] ] #TopN概率字符转换为float型
            # lable = [ (x) for x in eachline[-1] ]#lable转换为int型
            # read_data.append(lable[0])
            #read_data = list(map(float, eachline))
            data.append(read_data)
            line = f.readline()
        return np.array(data) #返回数据为双列表形式

def AnalyticalSolution(predict_nums, R, P2_flag):  # Todo 第一项应为T-t  input_data[:, :, 0:x_dim], output_data
    Ts = 1. / 200
    v_long = 16
    k1 = -88000
    k2 = -94000
    a = 1.14
    b = 1.4
    l = a + b
    m = 1500
    Izz = 2420
    Is = 1
    P1 = 5
    if int(P2_flag /10) == 1:
        P2 =0
    elif int(P2_flag /10) == 2:
        P2 = 200  # Todo

    A = [
        [0, v_long, 1, 0],
        [0, 0, 0, 1],
        [0, 0, (k1 + k2) / (m * v_long), -((b * k2 - a * k1) / (m * v_long) + v_long)],
        [0, 0, (a * k1 - b * k2) / (Izz * v_long), (a ** 2 * k1 + b ** 2 * k2) / (Izz * v_long)]
    ]
    B = [[0], [0], [-k1 / (m * Is)], [-a * k1 / (Izz * Is)]]

    G = np.eye(4) + np.dot(A, Ts)
    H = np.dot(B, Ts)

    N = predict_nums
    Q = np.zeros([4, 4])
    Q[0, 0] = P1
    Q[3, 3] = P2
    Q_bar = np.zeros([4 * N,4 * N])
    for i in range(N):
        Q_bar[4 * i, 4 * i] = P1
        Q_bar[4 * i + 3, 4 * i + 3] = P2
    # Q_bar[0 : 4, 4 * N - 4 : 4 * N ] = Q
    # R_bar = np.eye(N)
    # QQ2 = P2 * np.eye(N)
    R_bar = R * np.eye(N)
    S_bar = np.zeros((4 * N, N))
    # S_bar = S_bar.reshape([N, N, 4, 4])
    # WW2 = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if j > i:
                S_bar[4 * i : 4 * i + 4, j] = np.zeros(4)
                # WW2[i, j] = 0
            else:
                S_bar[4 * i : 4 * i + 4, j] = np.dot(np.linalg.matrix_power(G, i - j), H).squeeze()
                # WW2[i, j] = np.dot(np.dot(E2, np.linalg.matrix_power(G, i - j)), H)

    T_bar = np.zeros((4 * N, 4))
    # T_bar = T_bar.reshape([N, 1, 4, 4])
    # VV2 = np.zeros((N, 4))
    for i in range(N):
        T_bar[4 * i : 4 * i + 4, :] = np.linalg.matrix_power(G, i + 1)
        # VV2[i, :] = np.dot(E2, np.linalg.matrix_power(G, i + 1))

    H = 2.0 * (np.dot(np.dot(S_bar.T, Q_bar), S_bar) + R_bar)
    # F2 = np.dot(WW1.T, QQ1)
    F = 2.0 * np.dot(np.dot(T_bar.T, Q_bar), S_bar)
    F = F.T
    L = np.zeros((N, 1))
    Y = 2.0 * (Q + np.dot(T_bar.T ,np.dot(Q_bar, T_bar)))
    return Q, H, F, Y

def mpc_solution(x, tag):
    x_state = x[0:4]
    t = x[4]
    predict_num = int((0.5 - t) / 0.005)
    if predict_num == 0:
        predict_num = 1
    Q, H, F, Y= AnalyticalSolution(predict_num,5, tag) # TODO RRRR   # predict_horizon & weight matrix
    H_inv = np.linalg.inv(H)  # np 矩阵求逆
    # x_ref = np.zeros(predict_num)
    u_all = - np.dot(H_inv, (np.dot(F, x_state.T)))  # TODO T-t是说要改这个地方
    u_Analytical = u_all[0]
    value_Analytical = 0.5 * np.dot(u_all.T, np.dot(H, u_all)) + np.dot(x_state, np.dot(F.T, u_all))  + 0.5 * np.dot(x_state, np.dot(Y, x_state.T))
    value_Analytical = 1/200*value_Analytical
    # value_Analytical = np.dot(x_state, np.dot(Q, x_state.T)) + u_Analytical * u_Analytical
    return u_Analytical, value_Analytical

def caluculate_error(u_adp_array, u_mpc_array):
    u_adp_array_reshape = u_adp_array.reshape(-1, 500).T

    u_mpc_max = u_mpc_array.max()
    u_mpc_min = u_mpc_array.min()
    value_region = u_mpc_max - u_mpc_min

    value_region = np.reciprocal(value_region)
    u_error = np.zeros(u_adp_array_reshape.shape)
    for i in range(u_adp_array_reshape.shape[1]):
        u_error[:, i] = u_adp_array_reshape[:, i] - u_mpc_array

    u_abs_error = np.abs(u_error)
    u_mean_abs_error = np.sum(u_abs_error, axis=0) / 500
    u_mean_relative_error = u_mean_abs_error * value_region
    # print(u_abs_error)
    print(u_mean_relative_error)
    return u_mean_relative_error, u_mean_abs_error

def main():
    solver = Solver()
    file_x = './train_data/train.x_history_plot_mht.txt'
    file_u = './train_data/train.u_history_plot_mht.txt'
    file_v = './train_data/train.v_history_plot_mht.txt'


    x_history_plot = np.loadtxt(file_x)
    x_single_interation = x_history_plot[0:500, :]              # v_x, v_y, omega, psi, y, t    MPC: y, v_y, psi, omega
    u_mpc_array = np.zeros(0)
    v_mpc_array = np.zeros(0)
    for i in range(500):
        x = x_single_interation[i, :].tolist()
        x_fixed = [x[4], x[1], x[3], x[2], 10.0, x[0]]
        predict_num = int((0.5 - x[5]) / 0.005)
        if predict_num == 0:
            predict_num = 1
        state, control = solver.mpcSolver(x_fixed, predict_num)
        u_mpc = control[0][0]
        # u_mpc, v_mpc = mpc_solution(x_single_interation[i, :])
        u_mpc_array = np.append(u_mpc_array, u_mpc)
        # v_mpc_array = np.append(v_mpc_array, v_mpc)

    u_adp_array = np.loadtxt(file_u)
    # v_adp_array = txt_read(file_v)
    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 12,
             }

    # v_mean_relative_error, v_mean_abs_error = caluculate_error(v_adp_array, v_mpc_array)
    # plt.figure(1)
    # plt.plot(range(len(v_mean_relative_error)), np.log10(v_mean_relative_error))
    # plt.xlabel('100 * iteration', font2)
    # plt.ylabel('Log mean relative error', font2)
    # plt.title('Value error')


    u_mean_relative_error, u_mean_abs_error = caluculate_error(u_adp_array, u_mpc_array)
    plt.figure(2)
    plt.plot(range(len(u_mean_relative_error)), np.log10(u_mean_relative_error))
    plt.xlabel('100 * iteration', font2)
    plt.ylabel('Log mean relative error', font2)
    plt.title('Input error')
    plt.show()

    # np.savetxt(save_v, v_mpc_array)
    # np.savetxt(save_u, u_mpc_array)
    # np.savetxt('err_abs_relative_mean_00.txt', np.log10(u_mean_relative_error))
    # np.savetxt('err_abs_relative_mean_v_00.txt', np.log10(v_mean_relative_error))

if __name__ == '__main__':
    # data_old_flag = 0
    # main(data_old_flag, 10)
    main()
    # main(data_old_flag, 20)
    # main(data_old_flag, 21)





