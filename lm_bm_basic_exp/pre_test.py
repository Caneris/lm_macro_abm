# import numpy as np
import numpy.random as rd
import time
from multiprocessing import Pool
from model_class import Model
import csv


def run_nc(args):
    ID, NC, T = args
    print('start simulation {} with NC = {}'.format(ID, NC))
    run_perms(ID, NC, T)


def run_perms(ID, NC, T):
    for j in range(NC):
        N_app = rd.randint(2,11)
        N_good = rd.randint(2,11)
        lambda_LM = rd.randint(2, 11)
        sigma_w = rd.uniform(0.1, 0.4)
        sigma_m = rd.uniform(0.1, 0.4)
        min_w_par = rd.uniform(0.1, 0.6)


        m = Model(T=T, alpha_2=0.25, sigma=1.5, N_app=N_app, N_good=N_good, lambda_LM=lambda_LM, sigma_m=sigma_m,
                  sigma_w= sigma_w, nu = 0.1, u_r=0.08, beta=1, lambda_exp = 0.5, F = 80, H = 500, min_w_par=min_w_par,
                  nr_to_r=False, mu_r = 0.4, gamma_nr = 0.4, sigma_delta=0.001, a = 1, f_max=1, W_r=1)
        m.run()

        with open('pretest_unemp_ID{}.csv'.format(ID), 'a', newline='') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            filewriter.writerow(m.u_r_arr)

        with open('pretest_gini_ID{}.csv'.format(ID), 'a', newline='') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            filewriter.writerow(m.gini_coeff)

        with open('pretest_price_ID{}.csv'.format(ID), 'a', newline='') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            filewriter.writerow(m.mean_p_arr)

        print("ID: {} sim: {} finished".format(ID, j))


def run_nc_with_mp(args_arr):

    start_time = time.time()

    p = Pool()
    p.map(run_nc, args_arr)

    p.close()
    p.join()

    end_time = time.time() - start_time
    print("Simulating {} mc simulations took {} time using mp".format(len(args_arr), end_time))


if __name__ == '__main__':

    # Number of periods per simulation
    T = 1000
    # Number of replications (cores)
    NR = 10
    # number of cases
    NC = 100
    args_arr = [(ID, NC, T) for ID in range(NR)]
    run_nc_with_mp(args_arr)
