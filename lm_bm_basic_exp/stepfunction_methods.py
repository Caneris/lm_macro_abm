import numpy as np
from hire_fire_routine import *
from hire_fire_non_routine import *
from default_tools import *
from goods_market import gm_matching


def wage_decisions(m):

    # wage decisions
    update_N(m.f_arr)
    set_W_fs(m.f_arr[m.active_fs], m.h_arr)  # firms measure average wages paid to employees
    update_Wr_e(m.f_arr[m.active_fs], m.min_w, m.lambda_exp)  # firms build wage expectations
    update_Wnr_e(m.f_arr[m.active_fs], m.min_w, m.lambda_exp)
    update_d_w(m.h_arr, m.sigma_chi, m.mean_p_arr[m.t - 1], m.t)  # households decide for desired wages


def household_decisions(m):

    # households update work experience
    update_exp(m.h_arr, m.t, 4)

    # households make consumption decision
    update_d_c(m.h_arr, m.alpha_1, m.alpha_2)


def firm_decisions(m):

    # firms update sales expectations
    update_s_e(m.f_arr[m.active_fs], m.lambda_exp)

    # Firms decide for desired production
    update_d_y(m.f_arr[m.active_fs], m.mu_r, m.mu_nr, m.sigma)

    update_delta(m.f_arr[m.active_fs], m.sigma_delta)

    # Firms decide for labor demand
    update_d_N(m.f_arr[m.active_fs], m.mu_r, m.mu_nr, m.sigma)

    # Firms choose whether to hire or fire
    update_v(m.f_arr)

    # price decisions
    update_m(m.f_arr[m.active_fs], m.sigma_chi)  # choose markup
    update_uc_arr(m.f_arr, m.t)
    update_p(m.f_arr[m.active_fs], m.t)