import numpy as np
from hire_fire_routine import *
from hire_fire_non_routine import *
from default_tools import *
from goods_market import gm_matching


def wage_decisions(m):

    # wage decisions
    update_N(m.f_arr, m.emp_matrix, m.nr_job_arr)
    set_W_fs(m.f_arr[m.active_fs], m.emp_matrix, m.nr_job_arr, m.h_arr)  # firms measure average wages paid to employees
    update_Wr_e(m.f_arr[m.active_fs], m.min_w, m.lambda_exp)  # firms build wage expectations
    update_Wnr_e(m.f_arr[m.active_fs], m.min_w, m.lambda_exp)
    update_d_w(m.h_arr, m.sigma_w, m.t)  # households decide for desired wages


def household_decisions(m):

    # households update work experience
    update_exp(m.h_arr, m.t, 4)

    # households make consumption decision
    update_p_e(m.h_arr, m.lambda_exp)
    update_d_c(m.h_arr, m.alpha_1, m.alpha_2)


def firm_decisions(m):

    # Firms decide for desired production
    update_d_y(m.f_arr[m.active_fs], m.mu_r, m.mu_nr, m.sigma)

    # Firms decide for labor demand
    update_d_N(m.f_arr[m.active_fs], m.mu_r, m.mu_nr, m.sigma)

    # Firms choose whether to hire or fire
    update_v(m.f_arr)

    # price decisions
    update_m(m.f_arr[m.active_fs], m.sigma_m)  # choose markup
    update_uc_arr(m.f_arr, m.t)
    update_p(m.f_arr[m.active_fs], m.t)


def run_labor_market(m):

    ## Labor market matching
    # get vacancies Fx2 matrix
    v_mat = np.array([(f.id, f.v_r, f.v_nr) for f in m.f_arr[m.active_fs]])
    firms_fire_r_workers(v_mat, m.h_arr, m.f_arr, m.emp_matrix, m.nr_job_arr)
    firms_fire_nr_workers(v_mat, m.h_arr, m.f_arr, m.emp_matrix, m.nr_job_arr)
    update_N(m.f_arr, m.emp_matrix, m.nr_job_arr)

    # households apply
    # hs_send_nr_apps(m.f_arr, m.h_arr[m.non_routine_arr], m.chi_L, m.H_nr, m.H_r, m.beta)
    # hs_send_r_apps(m.f_arr, m.h_arr[m.routine_arr], m.chi_L, m.H_r, m.beta)
    h_send_apps(m.app_matrix, m.H, m.F, m.N_app)

    # firms hire
    update_v(m.f_arr)
    firms_employ_nr_applicants(m)
    update_N(m.f_arr, m.emp_matrix, m.nr_job_arr)
    set_W_fs(m.f_arr, m.emp_matrix, m.nr_job_arr, m.h_arr)

    firms_employ_r_applicants(m)
    update_N(m.f_arr, m.emp_matrix, m.nr_job_arr)
    set_W_fs(m.f_arr, m.emp_matrix, m.nr_job_arr, m.h_arr)
    # m.app_matrix = np.zeros((m.F, m.H))


def run_goods_market(m):

    # firms produce goods
    firms_produce(m.f_arr, m.mu_r, m.mu_nr, m.sigma)

    # firms sell goods
    clear_s(m.f_arr)
    clear_expenditure(m.h_arr)
    gm_matching(m.f_arr, m.h_arr, m.chi_C, m.tol)

    # firms update sales expectations
    update_s_e(m.f_arr[m.active_fs], m.lambda_exp)
    update_inv(m.f_arr[m.active_fs])
    update_delta(m.f_arr[m.active_fs], m.sigma_delta)

    # households update mean prices
    update_h_p(m.h_arr)


def firm_profits_and_dividends(m):

    # firms calculate profits
    update_pi(m.f_arr)
    update_div_f(m.f_arr[m.active_fs])

    # surviving firms pay dividends
    distribute_dividends(m.h_arr, m.f_arr)


def hh_refin_firms(m):

    # households refinance firms
    m.active_fs = surviving_firms(m.f_arr)
    m.default_fs = default_firms(m.f_arr)

    # change this later
    refin_firms(m.f_arr[m.default_fs], m.f_arr[m.active_fs], m.h_arr, m.n_refinanced,
                m.tol, m.t)

    m.active_fs = surviving_firms(m.f_arr)
    m.default_fs = default_firms(m.f_arr)