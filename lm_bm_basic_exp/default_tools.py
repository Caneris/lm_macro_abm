import numpy as np
import numpy.random as rd


def default_firms(f_arr):
    A_arr = np.array([f.A + f.pi for f in f_arr])
    y_arr = np.array([f.y for f in f_arr])
    return np.logical_or(A_arr <= 0, y_arr == 0)


def surviving_firms(f_arr):
    A_arr = np.array([f.A + f.pi for f in f_arr])
    y_arr = np.array([f.y for f in f_arr])
    return np.logical_or(A_arr > 0, y_arr > 0)


def firms_pay_employees(f_arr, h_arr, default_fs, emp_mat):
    unemp_arr = np.array([], dtype=np.int64)
    for f in f_arr:
        employees = np.nonzero(emp_mat[f.id, :])[0]
        if len(employees) > 0:
            par = np.maximum(f.A + f.s*f.p, 0) / (f.Wr_tot + f.Wnr_tot) # percentage of wage bills paid
            f.par = np.minimum(1, par)
            for h in h_arr[employees]:
                h.par = f.par
                if default_fs[f.id]:
                    unemp_arr = np.append(unemp_arr, h.id)
    return unemp_arr


def pay_refin_cost(h_arr, weights, tol, tot_refin_cost):
    for h in h_arr:
        cost = weights[h.id]*tot_refin_cost
        h.A -= cost
        h.refin = cost
        if np.abs(h.A) < tol:
            h.A = 0


def refin_firms(def_firms, surviving_firms, h_arr, n_refin, tol, t):

    wealth_arr = np.array([np.maximum(h.A, 0) for h in h_arr])
    wealth_tot = np.sum(wealth_arr)
    weights = wealth_arr / wealth_tot

    Af_arr = np.array([f.A for f in surviving_firms])
    Af_init = np.percentile(Af_arr, 50)
    sigma_A = np.std(Af_arr)

    netA_arr = np.array([f.A + f.pi for f in def_firms])
    ids = np.arange(len(netA_arr))
    rand_ids = rd.choice(ids, len(netA_arr), replace=False)

    s_e_arr = np.array([f.s_e for f in surviving_firms])
    s_e_init = np.percentile(s_e_arr, 50)
    sigma_s_e = np.std(s_e_arr)

    m_arr = np.array([f.m for f in surviving_firms])
    m_init = np.percentile(s_e_arr, 50)
    sigma_m = np.std(m_arr)

    for id in rand_ids:
        refin = np.maximum((-1)*netA_arr[id] + np.maximum(Af_init + rd.randn()*sigma_A, 0), 0)
        wealth_tot -= refin
        if wealth_tot >= 0:
            def_firms[id].A += refin
            def_firms[id].default = False
            def_firms[id].s_e = s_e_init + rd.randn()*sigma_s_e
            def_firms[id].m = m_init + rd.randn()*sigma_m
            n_refin[t] += 1

            pay_refin_cost(h_arr, weights, tol, refin)
