import numpy as np
import numpy.random as rd

def gm_matching2(Ah_arr, d_c_arr, prices, demand, supply, F, H, N_good, tol):
    h_indices = np.arange(H)
    f_indices = np.arange(F)

    expenditure_arr = np.zeros(H)
    c_arr = np.zeros(H)
    sales_arr = np.zeros(F)

    while np.sum(demand > 0) and np.sum(supply > 0):

        rand_inds = rd.choice(h_indices, H, replace=False).astype(int)
        d_m = demand[rand_inds] > 0

        for h_id in rand_inds[d_m]:
            # observe firms
            subN = np.minimum(N_good, np.sum(supply > 0))
            if subN > 0:
                f_rand_inds = rd.choice(f_indices[supply>0], subN, replace=False)
                rand_prices = prices[f_rand_inds]
                f_id = np.argsort(rand_prices)[0]
                c = np.min([d_c_arr[h_id], demand[h_id], supply[f_id]])
                expenditure = c*prices[f_id]
                if expenditure <= Ah_arr[h_id]:
                    expenditure_arr[h_id] += expenditure
                    Ah_arr[h_id] -= expenditure
                    c_arr[h_id] += c
                    sales_arr[f_id] += c
                    demand[h_id] -= c
                    supply[f_id] -= c
                else:
                    demand[h_id] = 0

                if (np.abs(demand[h_id]) < tol) or (np.abs(Ah_arr[h_id]) < tol):
                    demand[h_id] = 0
                if np.abs(supply[f_id]) < tol:
                    supply[f_id] = 0

    return expenditure_arr, Ah_arr, c_arr, sales_arr