from agent_class import *
from initialize_tools import *
from calibration import calibrate_model
from hire_fire_routine import *
from hire_fire_non_routine import *
import numpy as np
import matplotlib.pyplot as plt


# rewrite the class such that you only have to type in following parameters:

# input
# u_r, mu_r, gamma_nr, H, F, W_r, m, sigma, delta, alpha_2, G, nu_Af

# output
# mu_nr, W_nr, Ah, Af, p, y, pi_f, DIV_h, DIV_f, C_h, alpha_1, Nr, Nnr


class Model:

    def __init__(self, H = 200, F = 20, u_r = 0.08, mu_r = 1, W_r = 1, gamma_nr = 0.33,
                 m = 0.1, sigma = 0.5, delta = 1, alpha_2 = 0.25, T = 500, tol = 1e-10):


        # exogenous parameters
        self.sigma_FN, self.chi_L, self.chi_C = sigma_FN, chi_L, chi_C
        self.lambda_LM = lambda_LM
        self.share_nr = share_nr
        self.lambda_exp = lambda_exp
        self.beta = beta
        self.delta = delta
        self.nu = nu

        self.min_realw_t = min_realw_t
        self.shock_t = shock_t

        self.T, self.t = T, 0
        self.tol = tol

        # exogenously chosen steady state parameters
        self.H, self.F = H, F
        self.mu_r, self.u_r, self.W_r, self.gamma_nr  = mu_r, u_r, W_r, gamma_nr
        self.m, self.sigma, self.delta, self.alpha_2 = m, sigma, delta, alpha_2

        # steady state calibration
        calibration = calibrate_model(H=H, F=F, Ah=Ah, u_r=u_r, mu_r=mu_r, W_r=W_r, gamma_nr=gamma_nr, m=m, sigma=sigma, delta=delta,
                        alpha_2=alpha_2)

        self.s_init = s_init

        # Number of routine resp. non-routine households
        self.H_r = int(np.round(self.H*(1-self.gamma_nr)))
        self.H_nr = int(np.round(self.H*self.gamma_nr))

        routine = True
        non_routine = False

        # create firms
        self.f_arr = np.array([Firm(j, Af_init, T,
                                    self.s_init, self.s_init, nu, self.w_init, self.div_rate) for j in range(F)])

        # create households
        self.h_arr = np.array([Household(j, self.Ah_init, T,
                                         routine, self.w_init) for j in range(self.H_r)])

        self.h_arr = np.append(self.h_arr, np.array([Household(j, self.Ah_init, T,
                                                               non_routine, self.w_init) for j in range(self.H_r, self.H_r + self.H_nr)]))

        # select routine resp. non routine workers
        self.routine_arr = np.array([h.routine for h in self.h_arr])
        self.non_routine_arr = np.array([not h.routine for h in self.h_arr])

        # Data

        # mean wages
        self.mean_w, mean_w_e, self.u_n = w_init, w_init, 0
        self.mean_r_w, self.mean_nr_w, self.mean_w_arr = np.zeros(T), np.zeros(T), np.zeros(T)
        self.mean_r_w[-1], self.mean_nr_w[-1] = w_init, w_init

        # unemployment
        self.u_r, self.mean_p_arr = 0, np.zeros(T)
        self.mean_p_arr[-1] = np.mean([f.p for f in self.f_arr])
        self.u_r_arr, self.mean_r_w_arr, self.mean_nr_w_arr = np.zeros(T), np.zeros(T), np.zeros(T)
        self.mean_nominal_w_arr = np.zeros(T)
        self.mean_r_w_arr[-1], self.mean_nr_w_arr[-1] = w_init, w_init
        self.ur_r_arr, self.unr_r_arr = np.zeros(T), np.zeros(T)
        self.u_n = 0

        self.SAVINGS = np.array(T)

        # GDP and open vacancies
        self.GDP, self.open_vs = np.zeros(T), np.zeros(T)

        # produced
        self.Y_arr = np.zeros(T)
        self.DC_arr, self.C_arr, self.DY_arr = np.zeros(T), np.zeros(T), np.zeros(T)
        self.INV_arr = np.zeros(T)


        # surviving and default firm masks
        self.default_fs, self.active_fs = np.full(F, False, dtype=bool), np.full(F, True, dtype=bool)

        # share of nr in r jobs, share of inactive firms
        self.share_nr_in_r = np.zeros(T)
        self.share_inactive = np.zeros(T)

        self.n_refinanced = np.zeros(T)

    def data_collector(self):

        ur_n = np.sum(np.array([h.u[self.t] for h in self.h_arr[self.routine_arr]]))
        unr_n = np.sum(np.array([h.u[self.t] for h in self.h_arr[self.non_routine_arr]]))
        u_n = np.sum(np.array([h.u[self.t] for h in self.h_arr]))

        self.u_n = u_n

        self.u_r_arr[self.t] = u_n / self.H
        self.ur_r_arr[self.t] = ur_n/self.H_r
        self.unr_r_arr[self.t] = unr_n / self.H_nr

        self.Y_arr[self.t] = np.sum(np.array([f.y + f.inv for f in self.f_arr]))
        self.mean_p_arr[self.t] = np.sum(np.array([f.p * (f.y + f.inv) for f in self.f_arr])) / self.Y_arr[self.t]

        self.DC_arr[self.t] = np.sum([h.d_c for h in self.h_arr])
        self.C_arr[self.t] = np.sum([h.c for h in self.h_arr])
        self.DY_arr[self.t] = np.sum([f.d_y for f in self.f_arr])
        self.INV_arr[self.t] = np.sum([f.inv for f in self.f_arr])

        # get mean wages
        self.mean_w_arr[self.t] = (np.sum(np.array([h.w for h in self.h_arr]))/(self.H-u_n))/self.mean_p_arr[self.t]
        self.mean_nominal_w_arr[self.t] = (np.sum(np.array([h.w for h in self.h_arr])) / (self.H - u_n))

        self.mean_r_w = np.sum(np.array([h.w for h in self.h_arr[self.routine_arr]]))/(self.H_r-ur_n)
        self.mean_r_w_arr[self.t] = self.mean_r_w/self.mean_p_arr[self.t]

        self.mean_nr_w = np.sum(np.array([h.w for h in self.h_arr[self.non_routine_arr]]))/(self.H_nr-unr_n)
        self.mean_nr_w_arr[self.t] = self.mean_nr_w/self.mean_p_arr[self.t]

        # get GDP
        self.GDP[self.t] = np.sum(np.array([h.w/self.mean_p_arr[self.t] for h in self.h_arr]))

        # open vacancies
        self.open_vs[self.t] = np.sum(np.array([(f.v_r>0)*f.v_r + (f.v_nr>0)*f.v_nr for f in self.f_arr ]))

        # share of default firms
        n_def = np.sum(self.default_fs)
        self.share_inactive[self.t] = n_def/self.F



