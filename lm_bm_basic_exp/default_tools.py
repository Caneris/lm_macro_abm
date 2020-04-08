import numpy as np
import numpy.random as rd
from toolbox import get_unemployed


def default_firms(f_arr):
    A_arr = np.array([f.A + f.pi for f in f_arr])
    y_arr = np.array([f.y for f in f_arr])
    return np.logical_or(A_arr <= 0, y_arr == 0)


def surviving_firms(f_arr):
    A_arr = np.array([f.A + f.pi for f in f_arr])
    y_arr = np.array([f.y for f in f_arr])
    return np.logical_or(A_arr > 0, y_arr > 0)


def default_firms_pay_employees(def_firms, h_arr):
    unemp_arr = np.array([])
    for f in def_firms:
        employees = np.concatenate((f.r_employees, f.nr_employees), axis=None)
        if len(employees) > 0:
            par = np.maximum(f.A + f.s*f.p, 0) / (f.Wr_tot + f.Wnr_tot) # percentage of wage bills paid
            if f.id == 19:
                print("par: {}".format(par))
            for h in h_arr[employees.astype(int)]:
                h.w = par * h.w
                unemp_arr = np.append(unemp_arr, h.id)
    return unemp_arr


def pay_refin_cost(h_arr, weights, tol, tot_refin_cost):
    for h in h_arr:
        cost = weights[h.id]*tot_refin_cost
        h.A -= cost
        if np.abs(h.A) < tol:
            h.A = 0


def refin_firms(Af_init, def_firms, surviving_firms, h_arr, n_refin, tol, t):

    wealth_arr = np.array([np.maximum(h.A, 0) for h in h_arr])
    wealth_tot = np.sum(wealth_arr)
    weights = wealth_arr / wealth_tot

    netA_arr = np.array([f.A + f.pi for f in def_firms])
    ids = np.arange(len(netA_arr))
    rand_ids = rd.choice(ids, len(netA_arr), replace=False)

    mean_s_e = np.mean(np.array([f.s_e for f in surviving_firms]))
    mean_m = np.mean(np.array([f.m for f in surviving_firms]))

    for id in rand_ids:
        refin = np.maximum((-1)*netA_arr[id] + Af_init, 0)
        wealth_tot -= refin
        if wealth_tot >= 0:
            def_firms[id].A += refin
            def_firms[id].default = False
            def_firms[id].s_e = mean_s_e + rd.randn()*0.01
            def_firms[id].m = mean_m + rd.randn()*0.01
            n_refin[t] += 1

            pay_refin_cost(h_arr, weights, tol, refin)
