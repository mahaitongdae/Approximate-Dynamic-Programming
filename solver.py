"""
    <Reinforcement Learning and Control>(Year 2020)
    by Shengbo Eben Li
        @ Intelligent Driving Lab, Tsinghua University

    OCP example for lane keeping problem in a circle road

    [Method]
    Model predictive control

"""
from  casadi import *
from config import DynamicsConfig
import math
from dynamics import VehicleDynamics
import matplotlib.pyplot as plt

class Solver(DynamicsConfig):
    """
    NLP solver for nonlinear model predictive control with Casadi.
    """
    def __init__(self):

        self._sol_dic = {'ipopt.print_level': 0, 'ipopt.sb': 'yes', 'print_time': 0}
        if self.tire_model == 'Fiala':
            self.X_init = [0.0, 0.0, self.psi_init, 0.0, self.u, 0.0]
            self.zero = [0., 0., 0., 0., 0., 0.]
            self.U_LOWER = [- math.pi / 9, -10]
            self.U_UPPER = [math.pi / 9, 10]
        else:
            self.X_init = [0.0, 0.0, 0.1, 0.0, 0.0]
            self.zero = [0., 0., 0., 0., 0.]
            self.U_LOWER = [- math.pi / 9]
            self.U_UPPER = [math.pi / 9]
        self.x_last = 0
        self.dynamics = VehicleDynamics()
        super(Solver, self).__init__()

    def dynamics(self,x, u):
        x1 = [0.0, 0.0, 0.0, 0.0, 0.0]
        x1[0] = x[0] + self.Ts * (self.u * sin(x[2]) + x[1] * cos(x[2])),
        x1[1] = x[1] + self.Ts * (-self.D * self.F_z1 * sin(
            self.C * arctan(self.B * (-u[0] + (x[1] + self.a * x[3]) / self.u))) * cos(u[0])
                          - self.D * self.F_z2 * sin(
                    self.C * arctan(self.B * ((x[1] - self.b * x[3]) / self.u))) / self.m - self.u * x[3]),
        x1[2] = x[2] + self.Ts * (x[3]),
        x1[3] = x[3] + self.Ts * (self.a * (-self.D * self.F_z1 * sin(
            self.C * arctan(self.B * (-u[0] + (x[1] + self.a * x[3]) / self.u)))) * cos(u[0])
                          - self.b * (-self.D * self.F_z2 * sin(
                    self.C * arctan(self.B * ((x[1] - self.b * x[3]) / self.u)))) / self.I_zz),
        x1[4] = x[4] + self.Ts * (self.u * cos(x[2]) - x[1] * sin(x[2]))

        return x1

    def openLoopMpcSolver(self, x_init, predict_steps):
        """
        Solver of nonlinear MPC

        Parameters
        ----------
        x_init: list
            input state for MPC.
        predict_steps: int
            steps of predict horizon.

        Returns
        ----------
        state: np.array     shape: [predict_steps+1, state_dimension]
            state trajectory of MPC in the whole predict horizon.
        control: np.array   shape: [predict_steps, control_dimension]
            control signal of MPC in the whole predict horizon.
        """
        x = SX.sym('x', self.DYNAMICS_DIM)
        u = SX.sym('u', self.ACTION_DIM)

        # discrete dynamic model
        self.f = vertcat(
            x[0] + self.Ts * (self.u * sin(x[2]) + x[1] * cos(x[2])),
            x[1] + self.Ts * ((-self.D * self.F_z1 * sin(
                self.C * arctan(self.B * (-u[0] + (x[1] + self.a * x[3]) / self.u))) * cos(u[0])
                               - self.D * self.F_z2 * sin(
                        self.C * arctan(self.B * ((x[1] - self.b * x[3]) / self.u)))) / self.m - self.u * x[3]),
            x[2] + self.Ts * (x[3]),
            x[3] + self.Ts * ((self.a * (-self.D * self.F_z1 * sin(
                self.C * arctan(self.B * (-u[0] + (x[1] + self.a * x[3]) / self.u)))) * cos(u[0])
                               - self.b * (-self.D * self.F_z2 * sin(
                        self.C * arctan(self.B * ((x[1] - self.b * x[3]) / self.u))))) / self.I_zz),
            x[4] + self.Ts * (self.u * cos(x[2]) - x[1] * sin(x[2]))
        )

        # Create solver instance
        self.F = Function("F", [x, u], [self.f])

        # Create empty NLP
        w = []
        lbw = []
        ubw = []
        lbg = []
        ubg = []
        G = []
        J = 0

        # Initial conditions
        Xk = MX.sym('X0', self.DYNAMICS_DIM)
        w += [Xk]
        lbw += x_init
        ubw += x_init

        for k in range(1, predict_steps + 1):
            # Local control
            Uname = 'U' + str(k - 1)
            Uk = MX.sym(Uname, self.ACTION_DIM)
            w += [Uk]
            lbw += self.U_LOWER
            ubw += self.U_UPPER

            Fk = self.F(Xk, Uk)
            Xname = 'X' + str(k)
            Xk = MX.sym(Xname, self.DYNAMICS_DIM)

            # Dynamic Constriants
            G += [Fk - Xk]
            lbg += self.zero
            ubg += self.zero
            w += [Xk]
            if self.tire_model == 'Fiala':
                lbw += [-inf, -20, -pi, -20, 50, -inf]
                ubw += [inf, 20, pi, 20, 0, inf]
            else:
                lbw += [-inf, -20, -pi, -20, -inf]
                ubw += [inf, 20, pi, 20, inf]
            F_cost = Function('F_cost', [x, u], [0.1 * (x[0] - self.a_curve * sin(self.k_curve * x[4])) ** 2
                                                 + 0.1 * (x[2] - arctan(
                self.a_curve * self.k_curve * cos(self.k_curve * x[4]))) ** 2
                                                 + 0.001 * u[0] ** 2])
            J += F_cost(w[k * 2], w[k * 2 - 1])

        # Create NLP solver
        nlp = dict(f=J, g=vertcat(*G), x=vertcat(*w))
        S = nlpsol('S', 'ipopt', nlp, self._sol_dic)

        # Solve NLP
        r = S(lbx=lbw, ubx=ubw, x0=0, lbg=lbg, ubg=ubg)
        # print(r['x'])
        state_all = np.array(r['x'])
        state = np.zeros([predict_steps, self.DYNAMICS_DIM])
        control = np.zeros([predict_steps, self.ACTION_DIM])
        nt = self.DYNAMICS_DIM + self.ACTION_DIM  # total variable per step

        # save trajectories
        for i in range(predict_steps):
            state[i] = state_all[nt * i: nt * i + nt - 1].reshape(-1)
            control[i] = state_all[nt * i + nt - 1]
        return state, control

    def mpcSolver(self, x_init, predict_steps):
        """
        Solver of nonlinear MPC

        Parameters
        ----------
        x_init: list
            input state for MPC.
        predict_steps: int
            steps of predict horizon.

        Returns
        ----------
        state: np.array     shape: [predict_steps+1, state_dimension]
            state trajectory of MPC in the whole predict horizon.
        control: np.array   shape: [predict_steps, control_dimension]
            control signal of MPC in the whole predict horizon.
        """
        tire_model = self.tire_model
        if tire_model == 'Fiala':
            DYNAMICS_DIM = 6
            ACTION_DIM = 2
        else:
            DYNAMICS_DIM = 5
            ACTION_DIM = 1
        x = SX.sym('x', DYNAMICS_DIM)
        u = SX.sym('u', ACTION_DIM)

        # Create solver instance
        if self.tire_model == 'Fiala':
            self.f_d = vertcat(
                x[0] + self.Ts * (x[4] * sin(x[2]) + x[1] * cos(x[2])),
                # y : lateral position
                x[1] + self.Ts * (((- self.k1 * tan(- u[0] + arctan((x[1] + self.a * x[3]) / x[4])) * (
                        pow(self.k1 * tan(- u[0] + arctan((x[1] + self.a * x[3]) / x[4])), 2) / (
                        27 * pow(self.D * self.F_z1, 2)) - self.k1 * fabs(
                    tan(- u[0] + arctan((x[1] + self.a * x[3]) / x[4]))) / (
                                3 * self.D * self.F_z1) + 1)) * cos(u[0]) - self.k2 * tan(
                    arctan((x[1] - self.b * x[3]) / x[4])) * (
                                           pow(self.k2 * tan(arctan((x[1] - self.b * x[3]) / x[4])), 2) / (
                                           27 * pow(self.D * self.F_z2, 2)) - self.k2 * fabs(tan(
                                       arctan((x[1] - self.b * x[3]) / x[4]))) / (
                                                   3 * self.D * self.F_z2) + 1)) / self.m - x[4] * x[3]),
                # v_y : lateral speed
                x[2] + self.Ts * (x[3]),
                # psi : heading angle
                x[3] + self.Ts * ((self.a * (- self.k1 * tan(- u[0] + arctan((x[1] + self.a * x[3]) / x[4])) * (
                        pow(self.k1 * tan(- u[0] + arctan((x[1] + self.a * x[3]) / x[4])), 2) / (
                        27 * pow(self.D * self.F_z1, 2)) - self.k1 * fabs(
                    tan(- u[0] + arctan((x[1] + self.a * x[3]) / x[4]))) / (
                                3 * self.D * self.F_z1) + 1)) * cos(u[0]) - self.b * (
                                           - self.k2 * tan(arctan((x[1] - self.b * x[3]) / x[4])) * (
                                           pow(self.k2 * tan(arctan((x[1] - self.b * x[3]) / x[4])), 2) / (
                                           27 * pow(self.D * self.F_z2, 2)) - self.k2 * fabs(tan(
                                       arctan((x[1] - self.b * x[3]) / x[4]))) / (
                                                   3 * self.D * self.F_z2) + 1))) / self.I_zz),
                # r : yaw rate
                x[4] + self.Ts * (
                            u[1] + x[2] * x[4] - (- self.k1 * tan(- u[0] + arctan((x[1] + self.a * x[3]) / x[4])) * (
                            pow(self.k1 * tan(- u[0] + arctan((x[1] + self.a * x[3]) / x[4])), 2) / (
                            27 * pow(self.D * self.F_z1, 2)) - self.k1 * fabs(
                        tan(- u[0] + arctan((x[1] + self.a * x[3]) / x[4]))) / (
                                    3 * self.D * self.F_z1) + 1)) * sin(u[0]) / self.m),
                # v_x : longitudinal speed
                x[5] + self.Ts * (x[4])
                # x : longitudinal position
            )
            self.F = Function("F", [x, u], [self.f_d])
        elif self.tire_model == 'Pacejka':
            # discrete dynamic model
            self.f = vertcat(
                x[0] + self.Ts * (self.u * sin(x[2]) + x[1] * cos(x[2])),
                x[1] + self.Ts * ((-self.D * self.F_z1 * sin(
                    self.C * arctan(self.B * (-u[0] + (x[1] + self.a * x[3]) / self.u))) * cos(u[0])
                                   - self.D * self.F_z2 * sin(
                            self.C * arctan(self.B * ((x[1] - self.b * x[3]) / self.u)))) / self.m - self.u * x[3]),
                x[2] + self.Ts * (x[3]),
                x[3] + self.Ts * ((self.a * (-self.D * self.F_z1 * sin(
                    self.C * arctan(self.B * (-u[0] + (x[1] + self.a * x[3]) / self.u)))) * cos(u[0])
                                   - self.b * (-self.D * self.F_z2 * sin(
                            self.C * arctan(self.B * ((x[1] - self.b * x[3]) / self.u))))) / self.I_zz),
                x[4] + self.Ts * (self.u * cos(x[2]) - x[1] * sin(x[2]))
            )
            # todo:retreve
            self.F = Function("F", [x, u], [self.f])
        elif self.tire_model == 'Linear':
            # linear model
            self.f = vertcat(
                x[0] + self.Ts * (self.u * sin(x[2]) + x[1] * cos(x[2])),
                x[1] + self.Ts * ((-self.k1*(-u[0] + (x[1] + self.a * x[3]) / self.u) * cos(u[0]) +
                                  -self.k2*((x[1] - self.b * x[3]) / self.u)) / self.m - self.u * x[3]),
                x[2] + self.Ts * (x[3]),
                x[3] + self.Ts * (self.a * (-self.D * self.F_z1 * sin(
                    self.C * arctan(self.B * (-u[0] + (x[1] + self.a * x[3]) / self.u)))) * cos(u[0])
                                  - self.b * (-self.D * self.F_z2 * sin(
                            self.C * arctan(self.B * ((x[1] - self.b * x[3]) / self.u)))) / self.I_zz),
                x[4] + self.Ts * (self.u * cos(x[2]) - x[1] * sin(x[2]))
                )
            self.F = Function("F", [x, u], [self.f])

        # Create empty NLP
        w = []
        lbw = []
        ubw = []
        lbg = []
        ubg = []
        G = []
        J = 0

        # Initial conditions
        Xk = MX.sym('X0', DYNAMICS_DIM)
        w += [Xk]
        lbw += x_init
        ubw += x_init

        for k in range(1, predict_steps + 1):
            # Local control
            Uname = 'U' + str(k - 1)
            Uk = MX.sym(Uname, ACTION_DIM)
            w += [Uk]
            lbw += self.U_LOWER
            ubw += self.U_UPPER
            # Gk = self.G_f(Xk,Uk)

            Fk = self.F(Xk, Uk)
            Xname = 'X' + str(k)
            Xk = MX.sym(Xname, DYNAMICS_DIM)


            # Dynamic Constriants
            G += [Fk - Xk]
            lbg += self.zero
            ubg += self.zero
            w += [Xk]
            if self.tire_model == 'Fiala':
                lbw += [-inf, -20, -pi, -20, 0, -inf]
                ubw += [inf, 20, pi, 20, 50, inf]
            else:
                lbw += [-inf, -20, -pi, -20, -inf]
                ubw += [inf, 20, pi, 20, inf]


            # Cost function
            if tire_model == 'Fiala':
                F_cost = Function('F_cost', [x, u], [6 * (x[0]) ** 2
                                                     + 0.2 * (x[4] - self.u) ** 2
                                                     + 80 * u[0] ** 2
                                                     + 0.3 * u[1] ** 2])
            else:
                F_cost = Function('F_cost', [x, u], [1 * (x[0]) ** 2
                                                     + 1 * (x[2]) ** 2
                                                     + 1 * u[0] ** 2])
            J += F_cost(w[k * 2], w[k * 2 - 1])
            # J += F_cost(w[k * 3 - 1], w[k * 3 - 2])

        # Create NLP solver
        nlp = dict(f=J, g=vertcat(*G), x=vertcat(*w))
        S = nlpsol('S', 'ipopt', nlp, self._sol_dic)

        # Solve NLP
        r = S(lbx=lbw, ubx=ubw, x0=0, lbg=lbg, ubg=ubg)
        # print(r['x'])
        state_all = np.array(r['x'])
        state = np.zeros([predict_steps, DYNAMICS_DIM])
        control = np.zeros([predict_steps, ACTION_DIM])
        nt = DYNAMICS_DIM  + ACTION_DIM  # total variable per step

        # save trajectories
        for i in range(predict_steps):
            state[i] = state_all[nt * i: nt * i + DYNAMICS_DIM].reshape(-1)
            control[i] = state_all[nt * i + DYNAMICS_DIM: nt * (i + 1)].reshape(-1)
        return state, control
