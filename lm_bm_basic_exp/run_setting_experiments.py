import numpy as np
import numpy.random as rd
import time
from multiprocessing import Pool
from setting_generator import pick_element
from model_class import Model
import csv


def run_nc(args):
    ID, NC, T, periods, sigma_w_arr, sigma_m_arr, lambda_LM_arr = args
    print('start simulation {} with NC = {}'.format(ID, NC))
    run_perms(ID, NC, T, periods, sigma_w_arr, sigma_m_arr, lambda_LM_arr)


def run_perms(ID, NC, T, periods, sigma_w_arr, sigma_m_arr, lambda_LM_arr):
    tt = T-periods
    rng_states = np.array([])
    with open('run_ID{}.csv'.format(ID), 'w+', newline='') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        filewriter.writerow(["sim_num", "lambda_LM", "sigma_m", "sigma_w",
                             'unemp_rate', 'unemp_vol',
                             'mean_p', 'mean_p_vol',
                             'real_GDP', 'real_GDP_vol',
                             'mean_r_w', 'mean_r_w_vol',
                             'mean_nr_w', 'mean_nr_w_vol',
                             'share_inactive', 'share_inactive_vol'])
    perm_seq = rd.permutation(NC)
    for j in range(len(perm_seq)):
        print('start simulating case number {}'.format(j))
        num = perm_seq[j]
        i_1, num_1 = pick_element(sigma_w_arr, num)
        i_2, num_2 = pick_element(sigma_m_arr, num_1)
        i_3, num_3 = pick_element(lambda_LM_arr, num_2)

        sigma_w, sigma_m, lambda_LM = sigma_w_arr[i_1], sigma_m_arr[i_2], lambda_LM_arr[i_3]

        state = rd.get_state()
        rng_states = np.append(rng_states, state)
        m = Model(T=T, alpha_2=0.25, N_app=4, N_good=4, lambda_LM=lambda_LM, sigma_m=sigma_m,
                  sigma_w= sigma_w, nu = 0.1, u_r=0.08, beta=1, lambda_exp = 0.5, F = 80, H = 500)
        m.run()

        with open('run_ID{}.csv'.format(ID), 'a', newline='') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            filewriter.writerow([j, sigma_w, sigma_m, lambda_LM,
                                 100 * np.mean(m.u_r_arr[tt:T]), 100 * np.std(m.ur_r_arr[tt:T]),
                                 np.mean(m.mean_p_arr[tt:T]), np.std(m.mean_p_arr[tt:T]),
                                 np.mean(m.GDP[tt:T]), np.std(m.GDP[tt:T]),
                                 np.mean(m.mean_r_w_arr[tt:T]), np.std(m.mean_r_w_arr[tt:T]),
                                 np.mean(m.mean_nr_w_arr[tt:T]), np.std(m.mean_nr_w_arr[tt:T]),
                                 100 * np.mean(m.share_inactive[tt:T]), 100 * np.std(m.share_inactive[tt:T])])

    np.save('states_ID{}'.format(ID), rng_states)


def run_nc_with_mp(args_arr):

    start_time = time.time()

    p = Pool()
    p.map(run_nc, args_arr)

    p.close()
    p.join()

    end_time = time.time() - start_time
    print("Simulating {} mc simulations took {} time using mp".format(len(args_arr), end_time))


if __name__ == '__main__':

    # Parameter values
    N_app_arr = np.arange(2, 12, 2).astype(int)
    N_good_arr = np.arange(2, 12, 2).astype(int)
    lambda_LM_arr = np.arange(2, 12, 2).astype(int)
    sigma_w_arr = np.round(np.linspace(0.01, 0.4, 5), 2)
    sigma_m_arr = np.round(np.linspace(0.01, 0.4, 5), 2)
    min_w_par_arr = np.round(np.linspace(0.01, 0.6, 5), 2)

    print('sigma_m array: {}'.format(sigma_m_arr))
    print('sigma_w array: {}'.format(sigma_w_arr))
    print('lambda_LM array: {}'.format(lambda_LM_arr))

    # Number of periods per simulation
    T = 1000
    periods = 300
    tt = T - periods
    # Number of replications
    NR = 10
    # number of cases
    NC = 2
    args_arr = [(ID, NC, T, periods, sigma_m_arr, sigma_w_arr, lambda_LM_arr) for ID in range(NR)]
    run_nc_with_mp(args_arr)
