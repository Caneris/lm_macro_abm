import numpy as np
import numpy.random as rd

class Agent(object):

    def __init__(self, _id, A_init, T):

        self.id = _id
        self.T = T
        self.A = A_init
        self.cash = A_init

        # taxes that agent has to pay
        self.income_tax = 0
        self.wealth_tax = 0

        # tolerance level for setting values to zero
        self.tol = 10**(-10)

    def change_cash(self, c):
        self.cash += c
        if np.abs(self.cash) < self.tol:
            self.cash = 0

class Firm(Agent):

    def __init__(self, _id, A_init, T, y, nu, Wr_f, Wnr_f, delta, p, m, pi, div):
        super(Firm, self).__init__(_id, A_init, T)

        # employees, number of employees n_f, total wage bill
        self.r_employees, self.nr_employees = np.array([]), np.array([])

        # sold goods, expected sales
        self.s, self.s_e = y, y

        # number of employees and desired number of umployees
        self.Nr, self.d_Nr = 0, 0
        self.Nnr, self.d_Nnr = 0, 0
        self.n_r_fired, self.n_nr_fired = 0, 0

        # average wage to pay
        self.Wr, self.Wnr = Wr_f, Wnr_f
        self.Wr_e, self.Wnr_e = Wr_f, Wnr_f

        # total wage bill
        self.Wr_tot, self.Wnr_tot = 0, 0

        self.apps_r, self.apps_nr = np.array([]), np.array([])

        # desired production, production and price
        self.d_y, self.y, self.p = y, y, p
        self.d_y_diff = 0
        self.uc_arr = np.zeros(T)
        self.uc_arr[-1] = 0.4

        # number of vancancies
        self.v_r, self.v_nr = 0, 0

        # inventories share, inventories and mark up
        self.nu, self.m = nu, m
        self.inv = nu*y

        # profits, profits after dividends and dividend rate
        self.pi, self.div, self.delta = pi, div, delta
        self.pi2, self.neg_pi, self.pos_pi = 0, 0, 0
        self.pi_bar = 0

        # boolean for default and
        # parameter for share of wagebills paid if default
        self.default, self.par = False, 1
        # self.inactive = False


class Household(Agent):

    def __init__(self, _id, A_init, T, routine, W_r, W_nr, p):
        super(Household, self).__init__(_id, A_init, T)

        # dummy if TRUE -> routine type else -> non-routine
        self.routine = routine
        self.public_worker = False

        # Dummy: if True -> current job is non routine,
        #        else -> current job is routine
        self.nr_job = False

        # Dummy -> got job offer  at time t -> 1
        #       -> got no job offer at time t -> 0
        self.job_offer = np.zeros(T)

        self.fired = None
        self.fired_time = 0
        self.fired_time_max = 0

        # unemployed dummy,
        self.u = np.zeros(T)
        self.u[0] = 1
        self.exp, self.employer_id = 0, None
        # number of periods unemployed in a row
        self.xi = 0
        # desired wage and actual wage
        if routine:
            self.d_w, self.w = W_r, 0
            self.w_e = W_r
            self.last_w = W_r
        else:
            self.d_w, self.w = W_nr, 0
            self.w_e = W_nr
            self.last_w = W_nr
        # expected wealth
        self.A_e = A_init
        # price expectations, average price paid
        self.p = p
        # desired consumption, consumption
        if routine:
            self.d_c, self.c = 2, 2
        else:
            self.d_c, self.c = 3, 3
        # consumption expenditure
        self.expenditure = 1
        # dividend income
        self.div = 0.12
        self.div_e = 0.12
