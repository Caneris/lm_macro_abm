from model_class import *
from default_tools import *
from plot_tool import plot_lm
from goods_market import gm_matching


T = 1000
rd.seed(135)
m = Model(lambda_LM=0.5, chi_L=0.2, chi_C=0.2, T=T,
          lambda_exp=0.25, share_nr=0.33, H=200,
          sigma_FN=0.01, w_init=1, F = 20, beta=1,
          mu_r=8.8, mu_nr=17.8, nu=0.1, Af_init=12, alpha_1=0.698762626262,
          alpha_2=0.25, min_real_w=0, shock_t=0, tau=0,
          delta=0, div_rate=1, sigma=0.5, psi=0, period=6,
          AG_init= 0, phi_w=0, s_init = 26, Ah_init=1.581199999)

# initialize employment
initialize_emp(m.h_arr, m.f_arr, m.F, 120, 59)

set_W_fs(m.f_arr, m.h_arr)
val = 0

for t in range(m.T):
    print("Period: {}".format(t))

    if t == m.shock_t:
        m.min_real_w = m.delta

    # count unemployed households
    if t>0:
        count_unemployed_hs(m.h_arr, m.t)

    # fired workers loose job
    count_fired_time(m.h_arr)
    fired_workers_loose_job(m.h_arr, m.f_arr, m.t)

    # wage decisions
    update_N(m.f_arr)
    set_W_fs(m.f_arr[m.active_fs], m.h_arr) # firms measure average wages paid to employees
    update_Wr_e(m.f_arr[m.active_fs], m.min_w, m.lambda_exp) # firms build wage expectations
    update_Wnr_e(m.f_arr[m.active_fs], m.min_w, m.lambda_exp)
    update_d_w(m.h_arr, m.sigma_FN, m.mean_p_arr[m.t-1], m.t) # households decide for desired wages

    # households update work experience
    update_exp(m.h_arr, m.t, 4)

    # households update expected wages and dividends
    # update_div_h_e(m.h_arr, lambda_exp)
    update_w_e(m.h_arr, m.lambda_exp)

    # households update expected prices
    update_p_e(m.h_arr, lambda_exp)

    # households make consumption decision
    update_d_c(m.h_arr, m.alpha_1, m.alpha_2)

    # firms update sales expectations
    update_s_e(m.f_arr[m.active_fs], lambda_exp)

    # Firms decide for desired production
    update_d_y(m.f_arr[m.active_fs], m.mu_r, m.mu_nr, m.sigma)

    # update_div_rate(m.f_arr[m.active_fs], m.sigma_FN)

    # Firms decide for labor demand
    update_d_N(m.f_arr[m.active_fs], m.mu_r, m.mu_nr, m.sigma)

    # Firms choose whether to hire or fire
    update_v(m.f_arr)

    # price decisions
    update_m(m.f_arr[m.active_fs], m.sigma_FN) # choose markup
    update_uc_arr(m.f_arr, m.t)
    update_p(m.f_arr[m.active_fs], m.t)

    ## Labor market matching
    # get vacancies Fx2 matrix
    v_mat = np.array([(f.id, f.v_r, f.v_nr) for f in m.f_arr[m.active_fs]])
    firms_fire_r_workers(v_mat, m.h_arr, m.f_arr, m.t)
    firms_fire_nr_workers(v_mat, m.h_arr, m.f_arr, m.t)
    update_N(m.f_arr[m.active_fs])

    # shuffle supply side (workers)
    h_indices = np.array([h.id for h in m.h_arr])
    h_shuffled = rd.choice(h_indices, m.H_r + m.H_nr, replace=False)
    h_arr_shuffled = m.h_arr[h_shuffled]

    # households apply
    hs_send_nr_apps(m.f_arr, m.h_arr[m.non_routine_arr], m.chi_L, m.H_nr, m.H_r, m.beta)
    hs_send_r_apps(m.f_arr, m.h_arr[m.routine_arr], m.chi_L, m.H_r, m.beta)

    # firms hire
    firms_employ_nr_applicants(m.f_arr, m.h_arr, m.lambda_LM, m.min_w, m.t)
    update_N(m.f_arr)
    set_W_fs(m.f_arr, m.h_arr)

    firms_employ_r_applicants(m.f_arr, m.h_arr, m.lambda_LM, m.min_w, m.t)
    update_N(m.f_arr)
    set_W_fs(m.f_arr, m.h_arr)

    clear_applications(m.f_arr)

    # firms produce goods
    firms_produce(m.f_arr, m.mu_r, m.mu_nr, m.sigma)

    # firms sell goods
    clear_s(m.f_arr)
    clear_expenditure(m.h_arr)
    gm_matching(m.f_arr, m.h_arr, m.chi_C, m.tol)
    update_inv(m.f_arr[m.active_fs])

    # households update mean prices
    update_h_p(m.h_arr)

    # agents pay income taxes
    update_pi(m.f_arr)
    update_pi_2(m.f_arr)
    update_neg_pi(m.f_arr)
    update_pos_pi(m.f_arr)

    # firms calculate profits
    update_pi_bar(m.f_arr[m.active_fs])
    update_div_f(m.f_arr[m.active_fs])

    # surviving firms pay dividends
    distribute_dividends(m.h_arr, m.f_arr)

    # households refinance firms
    m.active_fs = surviving_firms(m.f_arr)
    m.default_fs = default_firms(m.f_arr)
    m.default_fs = default_firms(m.f_arr)
    m.default_fs = default_firms(m.f_arr)
    mean_Af = np.mean(np.array([f.A for f in m.f_arr[m.active_fs]]))
    refin_firms(m.Af_init, m.f_arr[m.default_fs], m.f_arr[m.active_fs], m.h_arr, m.n_refinanced, m.tol, m.t)

    m.active_fs = surviving_firms(m.f_arr)
    m.default_fs = default_firms(m.f_arr)

    # defaulted firms, pay remaining wage bills
    unemp_arr = default_firms_pay_employees(m.f_arr[m.default_fs], m.h_arr)
    set_W_fs(m.f_arr, m.h_arr)

    update_Af(m.f_arr, m.tol)
    update_Ah(m.h_arr)

    for h in m.h_arr[unemp_arr.astype(int)]:
        get_unemployed(h, m.t)
        h.fired = None
        h.fired_time = 0
        h.fired_time_max = 0

    m.data_collector()

    if t%4 == 0:
        m.min_w = get_min_w(m.mean_p_arr[m.t], m.min_real_w)

    rhs = np.sum([f.Wnr_tot + f.Wr_tot for f in m.f_arr])
    lhs = np.sum([h.w for h in m.h_arr])

    print(np.array([(f.uc_arr[m.t-1], f.m, f.p) for f in m.f_arr]))

    # firms loose employees
    for f in m.f_arr[m.default_fs]:
        f.n_r_fired, f.n_nr_fired = 0, 0
        employees = np.concatenate((f.r_employees, f.nr_employees), axis=None)
        if len(employees) > 0:
            f.r_employees, f.nr_employees = np.array([]), np.array([])

    m.t += 1

f1, f2 = plot_lm(m, 1000, 1000)
f1.show()
f2.show()