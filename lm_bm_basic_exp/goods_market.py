import numpy as np
import numpy.random as rd
from toolbox import get_N_sub

def gm_matching(f_arr, h_arr, N_good, tol):
    h_indices = np.array([h.id for h in h_arr])
    f_indices = np.array([f.id for f in f_arr])

    demand = np.array([h.d_c for h in h_arr])

    supply = np.array([(f.y + f.inv)*(not f.default) for f in f_arr])

    while np.sum(demand > 0) and np.sum(supply > 0):
        # print("supply: {}".format(np.sum(supply)))
        # print("demand: {}".format(np.sum(demand)))
        # print("help")
        # print(np.sum(demand > 0), np.sum(supply > 0))

        rand_inds = rd.choice(h_indices, len(h_arr), replace=False).astype(int)
        h_arr_shuffled = h_arr[rand_inds]
        d_m = demand[rand_inds] > 0

        for h in h_arr_shuffled[d_m]:
            # observe firms
            subN = np.minimum(N_good, np.sum(supply>0))
            if subN > 0:
                f_arr_shuffled = f_arr[rd.choice(f_indices[supply>0], subN, replace=False)]
                # take the one with lowest price
                prices = np.array([f.p for f in f_arr_shuffled])
                ind = np.argsort(prices)[0]
                f = f_arr_shuffled[ind]
                # buy consumption good
                c = np.min([h.d_c/1, demand[int(h.id)], supply[int(f.id)]])
                # print(current_rw, demand[h.id], supply[f.id])
                expenditure = c*f.p
                if expenditure <= h.A:
                    h.expenditure += expenditure
                    h.A -= expenditure
                    h.c += c
                    f.s += c
                    demand[h.id] -= c
                    supply[f.id] -= c
                else:
                    demand[h.id] = 0

                if (np.abs(demand[h.id]) < tol) or (np.abs(h.A) < tol):
                    demand[h.id] = 0
                if np.abs(supply[f.id]) < tol:
                    supply[f.id] = 0

