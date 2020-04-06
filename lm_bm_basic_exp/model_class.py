from agent_class import *
# from initialize_tools import *
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

    def __init__(self,
                 # exogenously chosen steady state parameters
                 H = 200, F = 20, Ar = 1, u_r = 0.08, mu_r = 1, W_r = 1, gamma_nr = 0.33,
                 m = 0.1, sigma = 0.5, delta = 1, alpha_2 = 0.25,
                 # exogenous model parameters
                 lambda_LM = 0.5, lambda_exp = 0.25, beta = 1, nu = 0.1, min_realw_t = 0,
                 shock_t = 0, sigma_FN = 0.01, chi_L = 0.1, chi_C = 0.2, T = 500, tol = 1e-10):


        # exogenous parameters
        self.sigma_FN, self.chi_L, self.chi_C = sigma_FN, chi_L, chi_C
        self.lambda_LM = lambda_LM
        self.lambda_exp = lambda_exp
        self.beta = beta
        self.nu = nu

        self.min_realw_t = min_realw_t
        self.shock_t = shock_t

        self.T, self.t = T, 0
        self.tol = tol

        # exogenously chosen steady state parameters
        self.H, self.F, self.Ar = H, F, Ar
        self.mu_r, self.u_r, self.W_r, self.gamma_nr  = mu_r, u_r, W_r, gamma_nr
        self.m, self.sigma, self.delta, self.alpha_2 = m, sigma, delta, alpha_2

        # steady state calibration
        calibration = calibrate_model(H=H, F=F, Ar=Ar, u_r=u_r, mu_r=mu_r, W_r=W_r, gamma_nr=gamma_nr,
                                      m=m, sigma=sigma, delta=delta, alpha_2=alpha_2)

        # parameters derived from steady state model

        mu_nr, W_nr, Anr, Af, p, y, pi_f, DIV_r, DIV_nr, DIV_f, c_r, c_nr, alpha_1, Nr, Nnr = calibration

        self.mu_nr, self.W_nr, self.Anr, self.Af, self.p, self.y = mu_nr, W_nr, Anr, Af, p, y
        self.pi_f, self.DIV_r, self.DIV_nr, self.DIV_f, self.c_r, self.c_nr = pi_f, DIV_r, DIV_nr, DIV_f, c_r, c_nr
        self.alpha_1, self.Nr, self.Nnr = alpha_1, Nr, Nnr

        # Number of routine resp. non-routine households
        self.H_r = int(np.round(self.H*(1-self.gamma_nr)))
        self.H_nr = int(np.round(self.H*self.gamma_nr))

        routine = True
        non_routine = False

        # create firms
        self.f_arr = np.array([Firm(j, self.Af, T, self.y, self.nu, self.W_r, self.W_nr,
                                    self.delta, self.p, self.m, self.pi_f, self.DIV_f) for j in range(F)])

        # create households
        self.h_arr = np.array([Household(j, self.Ar, T,
                                         routine, self.W_r, self.p, self.c_r, self.DIV_r) for j in range(self.H_r)])

        self.h_arr = np.append(self.h_arr, np.array([Household(j, self.Anr, T,
                                                               non_routine, self.W_nr, self.p, self.c_nr, self.DIV_nr)
                                                     for j in range(self.H_r, self.H_r + self.H_nr)]))

        # select routine resp. non routine workers
        self.routine_arr = np.array([h.routine for h in self.h_arr])
        self.non_routine_arr = np.array([not h.routine for h in self.h_arr])

        # Data

        # mean wages
        mean_w = (1 - gamma_nr)*W_r + gamma_nr*W_nr
        self.mean_w, self.u_n = mean_w, 0
        self.mean_r_w, self.mean_nr_w, self.mean_w_arr = W_r, W_nr, np.zeros(T)

        # unemployment
        self.u_r, self.mean_p_arr = 0, np.zeros(T)
        self.mean_p_arr[-1] = np.mean([f.p for f in self.f_arr])
        self.u_r_arr, self.mean_r_w_arr, self.mean_nr_w_arr = np.zeros(T), np.zeros(T), np.zeros(T)
        self.mean_nominal_w_arr = np.zeros(T)
        self.mean_r_w_arr[-1], self.mean_nr_w_arr[-1] = W_r, W_nr
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



