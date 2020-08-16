import numpy as np
import numpy.random as rd

cpdef gm_matching(double[:] Ah_arr, double[:] d_c_arr, prices, demand, supply, int F, int H, int N_good, float tol):
    h = np.arange(H, dtype=np.intc)
    f = np.arange(F, dtype=np.intc)

    cdef double[:] expenditure_arr = np.zeros(H);
    cdef double[:] c_arr = np.zeros(H);
    cdef double[:] sales_arr = np.zeros(F);
    cdef double[:] rand_prices = np.zeros(F);
    cdef int subN;
    cdef double c, expenditure;
    cdef int[:] f_rand_inds;
    cdef int[:] h_indices = h;
    # cdef int[:] f_indices = f;

    while np.sum(demand > 0) and np.sum(supply > 0):

        # print("demand: {}".format(demand))
        # print("supply: {}".format(supply))

        rand_inds = rd.choice(h_indices, H, replace=False).astype(int)
        d_m = demand[rand_inds] > 0
        for h_id in rand_inds[d_m]:
            # observe firms
            subN = np.minimum(N_good, np.sum(supply > 0))
            if subN > 0:
                f_rand_inds = rd.choice(f[supply>0], subN, replace=False).astype(np.intc)
                # print("observed f_ids: {}".format(f_rand_inds))


                rand_prices = prices[np.asarray(f_rand_inds, dtype=np.intc)]

                f_id = f_rand_inds[np.argsort(rand_prices)[0]]
                c = np.min([d_c_arr[h_id], demand[h_id], supply[f_id]])
                # print("c_h: {} d_c: {}".format(c, supply[f_id]))
                expenditure = c*prices[f_id]
                # print("expenditure: {}".format(expenditure))
                # print("Ah: {}".format(Ah_arr[h_id]))
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