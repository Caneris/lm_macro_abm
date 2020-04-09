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

        # percentage of wages paid respectively received
        self.par = 1

    def change_cash(self, c):
        self.cash += c
        if np.abs(self.cash) < self.tol:
            self.cash = 0

class Firm(Agent):

    def __init__(self, _id, A_init, T, y, nu, Wr_f, Wnr_f, delta, p, m, pi, div, uc):
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
        self.uc_arr[-1] = uc

        # number of vacancies
        self.v_r, self.v_nr = 0, 0

        # inventories share, inventories and mark up
        self.nu, self.m = nu, m
        self.inv = nu*y

        # profits, profits after dividends and dividend rate
        self.pi, self.div, self.delta = pi, div, delta

        # boolean for default and
        # parameter for share of wage bills paid if default
        self.default, self.par = False, 1


class Household(Agent):

    def __init__(self, _id, A_init, T, routine, w, p, c, div):
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

        # desired wage and actual wage
        self.d_w, self.w = w, w
        self.w_e = w
        self.last_w = w

        # average price paid
        self.p, self.p_e = p, p

        # desired consumption, consumption
        self.d_c, self.c = c, c

        # consumption expenditure
        self.expenditure = c*p

        # dividend income
        self.div = div
        self.div_e = div
