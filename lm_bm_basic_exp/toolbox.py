import numpy as np
import numpy.random as rd

def delete_from_old_r_job2(h, f_arr):
    f_id = h.employer_id # get id of old employer
    emp_arr = f_arr[f_id].r_employees
    h_i = np.where(emp_arr == h.id)[0][0] # get index in emp_arr
    f_arr[f_id].r_employees = np.delete(emp_arr, h_i)
    f_arr[f_id].n_r_fired -= 1

def delete_from_old_nr_job2(h, f_arr):
    f_id = h.employer_id  # get id of old employer
    emp_arr = f_arr[f_id].nr_employees
    h_i = np.where(emp_arr == h.id)[0][0]  # get index in emp_arr
    f_arr[f_id].nr_employees = np.delete(emp_arr, h_i)
    f_arr[f_id].n_nr_fired -= 1

# expectation function for a generic variable z

def update_wr_wnr_bar(w_bar, mean_w, phi_w):
    w_bar = phi_w*mean_w + (1-phi_w)*w_bar
    return w_bar

def expectation(z, z_e, lambda_exp):
    """
    this function returns the expected value for the
    variable z for the current period.

    :param z: previous period observation
    :param z_e: previous period expectation
    :param lambda_exp: adjustment parameter
    :return: current period observation
    """
    error = z - z_e
    return z_e + lambda_exp*error

def get_min_w(mean_p, min_r_w):
    return mean_p*min_r_w

# draw 1 with porbability P and 0 with prob. (1-P)

def draw_one(P):
    num = rd.uniform()
    return 0*(num >= P) + 1*(num<P)

# demand function


def get_N_sub(chi, N):
    if chi * N >= 1:
        return int(chi*N)
    elif chi * N > 0:
        return 1
    else:
        return 0


# test expectation function
z = 4
z_e = 6
lambda_exp = 0.5
expectation(z, z_e, lambda_exp)

#########################################################################################################
###################### TOOLS FOR HOUSEHOLDS #############################################################
#########################################################################################################


def get_unemployed(h, t):
    h.u[t], h.employer_id = 1, None
    h.nr_job = False
    h.w = 0
    h.last_w = h.w


def count_unemployed_hs(h_arr, t):
    for h in h_arr:
        unemp = (h.u[t-1] == 1)
        h.u[t] = 1*unemp


def Pr_LM(w_old, w_new, lambda_LM):
    diff = (w_old - w_new)/w_old
    if diff < 0:
        return 1 - np.exp(lambda_LM*diff)
    else:
        return 0


def update_exp(h_arr, t, diff):

    if t < diff:
        for h in h_arr:
            h.exp = np.sum(1 - h.u[0: t + 1])
    else:
        for h in h_arr:
            h.exp = np.sum(1 - h.u[t - 3: t + 1])


def update_d_w(h_arr, sigma_chi, mean_p, t):

    for h in h_arr:

        if h.job_offer[t-1]==0:
            h.d_w = h.d_w*(1-rd.chisquare(1)*sigma_chi)
        else:
            # note that employed worker's d_w is
            # not increasing fom period to period
            h.d_w = h.d_w*(1 + rd.chisquare(1)*sigma_chi)
        h.d_w = np.max([h.d_w, 0])


def update_w_e(h_arr, lambda_exp):
    for h in h_arr:
        h.w_e = expectation(h.w, h.w_e, lambda_exp)


def update_Ah(h_arr):
    for h in h_arr:
        h.A += (h.w + h.div)


def update_d_c(h_arr, alpha_1, alpha_2):
    for h in h_arr:
        h.d_c = alpha_1*((h.w + h.div)/h.p) + alpha_2*(h.A/h.p)


def clear_expenditure(h_arr):
    for h in h_arr:
        h.expenditure = 0
        h.c = 0


def update_h_p(h_arr):
    for h in h_arr:
        if (h.expenditure > 0) and (h.c>0):
            h.p = h.expenditure/h.c


def get_Ah_weights(h_arr):
    wealth_arr = np.array([h.A for h in h_arr])
    # print(wealth_arr)
    wealth_tot = np.sum(wealth_arr)
    if wealth_tot > 0:
        return wealth_arr/wealth_tot
    else:
        return np.ones(len(h_arr))*(1/len(h_arr))


def distribute_dividends(h_arr, f_arr):
    DIV = np.sum(np.array([f.div for f in f_arr]))
    weights = get_Ah_weights(h_arr)
    div = weights*DIV
    for h in h_arr:
        h.div = div[h.id]


def update_div_h_e(h_arr, lambda_exp):
    for h in h_arr:
        if h.div > 0:
            h.div_e = expectation(h.div, h.div_e, lambda_exp)


def count_fired_time(h_arr):
    for h in h_arr:
        if h.fired:
            h.fired_time += 1


def fired_workers_loose_job(h_arr, f_arr, t):
    for h in h_arr:
        if h.fired and (h.fired_time == h.fired_time_max):
            if h.nr_job:
                delete_from_old_nr_job2(h, f_arr)
            else:
                delete_from_old_r_job2(h, f_arr)
            get_unemployed(h, t)
            h.fired = False
            h.fired_time = 0
            h.fired_time_max = 0


#########################################################################################################
###################### TOOLS FOR FIRMS ##################################################################
#########################################################################################################




def update_d_N(f_arr, mu_r, mu_nr, sigma):
    for f in f_arr:
        # 1. Case
        Omega = get_Omega(f.Wr_e, f.Wnr_e, mu_r, mu_nr, sigma)
        f.d_Nnr = np.round(get_d_Nnr_non_binding(f.d_y, Omega, mu_r, mu_nr, sigma))
        f.d_Nr = np.round(get_d_Nr(f.d_Nnr, Omega))

        # check if feasible
        C = get_total_costs(f.Wr_e, f.Wnr_e, f.d_Nr, f.d_Nnr)
        if C > f.A + f.p*f.s_e: # check if feasible, IDEA: consider expected net earnings (after dividends)
            f.d_Nnr = np.round(get_d_Nnr_binding(f.A, f.Wr_e, f.Wnr_e, Omega))
            f.d_Nr = np.round(get_d_Nr(f.d_Nnr, Omega))

def update_N(f_arr):
    for f in f_arr:
        f.Nr = len(f.r_employees)
        f.Nnr = len(f.nr_employees)


def update_v(f_arr):
    for f in f_arr:
        f.v_r = f.d_Nr - (f.Nr - f.n_r_fired)
        f.v_nr = f.d_Nnr - (f.Nnr - f.n_nr_fired)


def initialize_emp(h_arr, f_arr, F, R, NR):
    ind = np.array([h.id for h in h_arr])
    routine_arr = np.array([h.routine for h in h_arr])
    non_routine_arr = np.array([not h.routine for h in h_arr])

    r_inds = rd.choice(ind[routine_arr], R, replace=False)
    nr_inds = rd.choice(ind[non_routine_arr], NR, replace=False)

    for i in range(len(r_inds)):
        ind_f = i%F
        r_id = int(r_inds[i])
        f_arr[ind_f].r_employees = np.append(f_arr[ind_f].r_employees,
                                             r_id)
        h_arr[r_id].employer_id = int(ind_f)
        h_arr[r_id].w = h_arr[r_id].d_w
        h_arr[r_id].u[0] = 0

    for i in range(len(nr_inds)):
        ind_f = i%F
        nr_id = int(nr_inds[i])
        f_arr[ind_f].nr_employees = np.append(f_arr[ind_f].nr_employees,
                                             nr_id)
        h_arr[nr_id].employer_id = int(ind_f)
        h_arr[nr_id].nr_job = True
        h_arr[nr_id].w = h_arr[nr_id].d_w
        h_arr[nr_id].u[0] = 0

    update_N(f_arr)
    set_W_fs(f_arr, h_arr)


def update_s_e(f_arr, lambda_exp):
    for f in f_arr:
        f.s_e = expectation(f.s, f.s_e, lambda_exp)


def set_W_fs(f_arr, h_arr):
    for f in f_arr:
        if f.Nr > 0:
            r_emps = f.r_employees.astype(int)
            wages_r = np.array([h.w for h in h_arr[r_emps]])
            f.Wr_tot = np.sum(wages_r)
            f.Wr = np.sum(wages_r) / f.Nr
        else:
            f.Wr_tot, f.Wr = 0, 0

        if f.Nnr > 0:
            nr_emps = f.nr_employees.astype(int)
            wages_nr = np.array([h.w for h in h_arr[nr_emps]])
            f.Wnr = np.sum(wages_nr) / f.Nnr
            f.Wnr_tot = np.sum(wages_nr)
        else:
            f.Wnr_tot, f.Wnr = 0, 0


def update_W_e(f_arr, lambda_exp):
    for f in f_arr:
        if not f.W == 0:
            f.W_e = expectation(f.W, f.W_e, lambda_exp)


def update_Wr_e(f_arr, min_w, lambda_exp):
    for f in f_arr:
        if f.Wr > 0:
            f.Wr_e = expectation(f.Wr, f.Wr_e, lambda_exp)
            f.Wr_e = np.maximum(f.Wr_e, min_w)


def update_Wnr_e(f_arr, min_w, lambda_exp):
    for f in f_arr:
        if f.Wr > 0:
            f.Wnr_e = expectation(f.Wnr, f.Wnr_e, lambda_exp)
            f.Wnr_e = np.maximum(f.Wnr_e, min_w)


def update_m(f_arr, sigma_chi):
    for f in f_arr:
        if f.inv < f.nu*f.s:
            f.m = f.m*(1+rd.chisquare(1)*sigma_chi)
        elif f.inv > f.nu*f.s:
            f.m = f.m*(1-rd.chisquare(1)*sigma_chi)
        f.m = np.maximum(1e-2, f.m)


def update_m2(f_arr, param):
    for f in f_arr:
        if f.inv < f.nu*f.s:
            f.m = f.m*(1+param*rd.uniform())
        elif f.inv > f.nu*f.s:
            f.m = f.m*(1-param*rd.uniform())
            print(f.m)


def update_uc_arr(f_arr, t):
    for f in f_arr:
        C = (f.Wr_e*f.d_Nr + f.Wnr_e*f.d_Nnr)
        f.uc_arr[t] = C/f.d_y
        if f.uc_arr[t] == 0:
            f.uc_arr[t] = f.uc_arr[t-1]


def update_p(f_arr, t):
    for f in f_arr:
        f.p = f.uc_arr[t]*(1+f.m)


def update_d_y(f_arr, mu_r, mu_nr, sigma):
    for f in f_arr:
        if not f.default:
            min_Nnr = 1
            Omega = np.round(get_Omega(f.Wr_e, f.Wnr_e, mu_r, mu_nr, sigma))
            min_Nr = Omega * min_Nnr
            min_d_y = CES_production(min_Nr, min_Nnr, mu_r, mu_nr, sigma)
            d_y = np.maximum(f.s_e * (1 + f.nu) - f.inv, min_d_y)
            f.d_y_diff = d_y - f.d_y
            f.d_y = d_y
        else:
            f.d_y = 0

###### Production Decision ##########


def firms_produce(f_arr, mu_r, mu_nr, sigma):
    for f in f_arr:
        f.y = CES_production(f.Nr, f.Nnr, mu_r, mu_nr, sigma)


def CES_production(Nr, Nnr, mu_r, mu_nr, sigma):
    rho = (sigma - 1)/sigma
    bool_1, bool_2 = Nr > 0, Nnr > 0
    if bool_1:
        X_1 = (mu_r*Nr)**rho
    else:
        X_1 = 0

    if bool_2:
        X_2 = (mu_nr*Nnr)**rho
    else:
        X_2 = 0

    if X_1 + X_2 > 0:
        CES = (X_1 + X_2)**(1/rho)
    else:
        CES = 0

    return CES

# Omega is the ratio of R to NR workers that's optimal
def get_Omega(Wr_e, Wnr_e, mu_r, mu_nr, sigma):
    rho = (sigma - 1)/sigma
    X_1, X_2, X_3 = Wr_e / mu_r, Wnr_e / mu_nr, mu_nr / mu_r
    Omega = ((X_1/X_2)**(1/rho-1))*X_3
    return Omega


def get_d_Nr(Nnr, Omega):
    return  Nnr * Omega

# 1. Case: Budget constraint is non-binding
def get_d_Nnr_non_binding(d_y, Omega, mu_r, mu_nr, sigma):
    rho = (sigma - 1)/sigma
    X_1, X_2 = (d_y/mu_nr), ((mu_r/mu_nr)*Omega)**rho
    Nnr = X_1*((1 + X_2)**(-1/rho))
    return Nnr

#  check whether the 1. Case is feasible
def get_total_costs(Wr_e, Wnr_e, Nr, Nnr):
    return Wr_e*Nr + Wnr_e*Nnr


# 2. Case: Budget constraint is binding, B is the firm f's udget
def get_d_Nnr_binding(B, Wr_e, Wnr_e, Omega):
    X_1 = Wr_e*Omega
    Nnr = B*((X_1 + Wnr_e)**(-1))
    return Nnr


def clear_applications(f_arr):
    for f in f_arr:
        f.apps_r, f.apps_nr = np.array([]), np.array([])


def update_pi(f_arr):
    for f in f_arr:
        f.pi = f.p*f.s - f.Wr_tot - f.Wnr_tot


def update_div_f(f_arr):
    for f in f_arr:
        if f.pi > 0:
            f.div = f.delta*f.pi
        else:
            f.div = 0


def update_delta(f_arr, sigma_chi):
    for f in f_arr:
        if f.d_y_diff > 0:
            f.delta = f.delta * (1 - rd.chisquare(1) * sigma_chi)
        elif f.d_y_diff < 0:
            f.delta = f.delta * (1 + rd.chisquare(1) * sigma_chi)

        f.delta = np.maximum(np.minimum(1, f.delta), 0)


def update_f_default(f_arr):
    for f in f_arr:
        default = (f.A + f.pi) <= 0
        f.default = default


def update_Af(f_arr, tol):
    for f in f_arr:
        f.A += f.s*f.p - (f.Wr_tot + f.Wnr_tot) - f.div
        if np.abs(f.A) <= tol:
            f.A = 0


def clear_s(f_arr):
    for f in f_arr:
        f.s = 0


def update_inv(f_arr):
    for f in f_arr:
        f.inv += f.y - f.s

