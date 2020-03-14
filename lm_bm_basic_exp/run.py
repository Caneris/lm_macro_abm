from model_class import *
from default_tools import *
from plot_tool import plot_lm
from goods_market import gm_matching
from unemp_benefits import *
from public_worker_tools import *

T = 1000
rd.seed(135)
m = Model(lambda_LM=0.5, chi_L=0.2, chi_C=0.2, T=T,
          lambda_exp=0.25, share_nr=0.33, H=200, g=0,
          sigma_FN=0.01, w_init=1, F = 20, beta=1,
          mu_r=8.8, mu_nr=17.8, nu=0.001, Af_init=12, alpha_1=0.698762626262,
          alpha_2=0.25, min_real_w=0, shock_t=0, tau=0,
          delta=0, div_rate=1, sigma=0.5, psi=0, period=6,
          AG_init= 0, phi_w=0, s_init = 26, Ah_init=1.581199999)

# initialize employment
initialize_emp(m.h_arr, m.f_arr, m.F, 120, 59)
# update_pw_w(m.pw_arr, m.h_arr, m.gov, m.H_pw, 0)



set_W_fs(m.f_arr, m.h_arr)
val = 0

for t in range(m.T):
    print("Period: {}".format(t))

    Ah_tot = np.sum(np.array([h.A for h in m.htot_arr]))
    Af_tot = np.sum(np.array([f.A for f in m.f_arr]))

    if t == m.shock_t:
        m.min_real_w = m.min_real_w + m.delta

    # count unemployed households
    if t>0:
        count_unemployed_hs(m.htot_arr, m.t)

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
    # update_div_h_e(m.htot_arr, lambda_exp)
    update_w_e(m.htot_arr, m.lambda_exp)

    # households update expected prices
    update_p_e(m.htot_arr, 3*lambda_exp)

    # households make consumption decision
    update_d_c(m.htot_arr, m.alpha_1, m.alpha_2, m.mean_p_arr[m.t-1], m.tau)
    print(np.sum([h.d_c for h in m.h_arr]))
    Ah_tot = np.sum([h.A for h in m.h_arr])
    I = np.sum([h.w+h.div for h in m.h_arr])

    # firms update sales expectations
    update_s_e(m.f_arr[m.active_fs], lambda_exp)

    # Firms decide for desired production
    update_d_y(m.f_arr[m.active_fs], m.mu_r, m.mu_nr, m.sigma)
    # update_d_y2(m.f_arr, m.sigma_FN)
    # update_d_y3(m.f_arr)

    # update_div_rate(m.f_arr[m.active_fs], m.sigma_FN)

    # Firms decide for labor demand
    update_d_N(m.f_arr[m.active_fs], m.wr_bar, m.wnr_bar, m.mu_r, m.mu_nr, m.sigma)

    # Firms choose whether to hire or fire
    update_v(m.f_arr)

    # N_arr = np.array([(f.Nr, f.Nnr) for f in m.f_arr])
    # print(N_arr)

    # price decisions
    update_m(m.f_arr[m.active_fs], m.sigma_FN) # choose markup
    update_uc_arr2(m.f_arr, m.t)
    # update_uc_arr2(m.f_arr, m.t)
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
    hs_send_nr_apps(m.f_arr, m.htot_arr[m.non_routine_arr], m.chi_L, m.H_nr, m.H_r, m.beta)
    hs_send_r_apps(m.f_arr, m.htot_arr[m.routine_arr], m.chi_L, m.H_r, m.beta)

    # firms hire
    firms_employ_nr_applicants(m.f_arr, m.h_arr, m.lambda_LM, m.min_w, m.t)
    update_N(m.f_arr)
    set_W_fs(m.f_arr, m.h_arr)

    firms_employ_r_applicants(m.f_arr, m.h_arr, m.lambda_LM, m.min_w, m.t)
    update_N(m.f_arr)
    set_W_fs(m.f_arr, m.h_arr)

    # update_pw_w(m.pw_arr, m.h_arr, m.gov, m.H_pw, m.t)

    update_xi(m.h_arr, t) # parameter for being unemployed in a row
    clear_applications(m.f_arr)

    # firms produce goods
    firms_produce(m.f_arr, m.mu_r, m.mu_nr, m.sigma)

    # firms sell goods
    clear_s(m.f_arr)
    clear_expenditure(m.htot_arr)
    gm_matching(m.f_arr, m.htot_arr, m.chi_C, m.tol)
    update_inv(m.f_arr[m.active_fs])

    # households update mean prices
    update_h_p(m.htot_arr)

    # agents pay income taxes
    update_pi(m.f_arr)
    update_pi_2(m.f_arr)
    update_neg_pi(m.f_arr)
    update_pos_pi(m.f_arr)


    f_taxes = firms_pay_income_tax(m.f_arr, m.tau)

    # gov_decides_for_benefits(m.gov, m.h_arr, m.period, m.psi, t)

    # firms calculate profits
    update_pi_bar(m.f_arr[m.active_fs])
    update_div_f(m.f_arr[m.active_fs])

    # surviving firms pay dividends
    distribute_dividends(m.htot_arr, m.f_arr)

    # households refinance firms
    m.active_fs = surviving_firms(m.f_arr)
    m.default_fs = default_firms(m.f_arr)
    m.default_fs = default_firms(m.f_arr)
    m.default_fs = default_firms(m.f_arr)
    mean_Af = np.mean(np.array([f.A for f in m.f_arr[m.active_fs]]))
    refin_firms(m.Af_init, m.f_arr[m.default_fs], m.f_arr[m.active_fs], m.htot_arr, m.gov, m.n_refinanced, m.tol, m.t)

    m.active_fs = surviving_firms(m.f_arr)
    m.default_fs = default_firms(m.f_arr)

    # defaulted firms, pay remaining wage bills
    unemp_arr = default_firms_pay_employees(m.f_arr[m.default_fs], m.h_arr)
    set_W_fs(m.f_arr, m.h_arr)

    hh_taxes = hhs_pay_income_tax(m.htot_arr, m.tau, m.t)
    gov_collects_income_taxes(m.gov, hh_taxes, f_taxes)

    update_Af(m.f_arr, m.tol)
    update_Ah(m.htot_arr)
    update_AG(m.gov)

    for h in m.h_arr[unemp_arr.astype(int)]:
        get_unemployed(h, m.t)
        h.fired = None
        h.fired_time = 0
        h.fired_time_max = 0

    m.data_collector()

    if t%4 == 0:
        m.min_w = get_min_w(m.mean_p_arr[m.t], m.min_real_w)

    rhs = np.sum([f.Wnr_tot + f.Wr_tot for f in m.f_arr]) + m.gov.paid_benefits
    lhs = np.sum([h.w for h in m.h_arr])

    print(np.array([(f.uc_arr[m.t-1], f.m, f.p) for f in m.f_arr]))

    # firms loose employees
    for f in m.f_arr[m.default_fs]:
        f.n_r_fired, f.n_nr_fired = 0, 0
        employees = np.concatenate((f.r_employees, f.nr_employees), axis=None)
        if len(employees) > 0:
            f.r_employees, f.nr_employees = np.array([]), np.array([])

    tot_A = Ah_tot + Af_tot + m.gov.A
    m.t += 1


f1, f2 = plot_lm(m, 1000, 400)
f1.show()
f2.show()

a = np.array([h.A for h in m.h_arr])
ids = np.where(a<0)[0]
np.array([h.A for h in m.h_arr[ids]])

a = np.array([f.A for f in m.f_arr])
ids = np.where(a<0)[0]
np.array([f.A for f in m.f_arr[ids]])

sum_diff = 0
for f in m.f_arr:
    r_emps = f.r_employees.astype(int)
    wages_r = np.array([h.w for h in m.h_arr[r_emps]])
    Wr_tot = np.sum(wages_r)
    val = f.Wr_tot == Wr_tot
    sum_diff += f.Wr_tot - Wr_tot
    print(val, sum_diff, f.id)

for f in m.f_arr:
    nr_emps = f.nr_employees.astype(int)
    wages_nr = np.array([h.w for h in m.h_arr[nr_emps]])
    Wnr_tot = np.sum(wages_nr)
    val = f.Wnr_tot == Wnr_tot
    sum_diff += f.Wnr_tot - Wnr_tot
    print(val, sum_diff, f.id)

a = np.array([h.div for h in m.f_arr])
ids = np.where(a<0)[0]
np.array([h.A for h in m.f_arr[ids]])


emps = np.array([])
for f in m.f_arr:
    r_emps = f.r_employees.astype(int)
    nr_emps = f.nr_employees.astype(int)
    f_emps = np.concatenate((r_emps, nr_emps), axis=None)
    emps = np.append(emps, f_emps)

print(emps)
unique_elements, counts_elements = np.unique(emps, return_counts=True)
print(np.asarray((unique_elements, counts_elements)))
print(np.sum(counts_elements) == len(unique_elements))



print(m.H - m.u_n)

u_arr = np.array([h.u[m.t] == 0 for h in m.h_arr])
emp_h_arr = np.array([h.id for h in m.h_arr[u_arr]])
len(emp_h_arr)

m_ = np.array([False if h in emps else True for h in emp_h_arr])
np.sum(m_)
print(emp_h_arr[m_])