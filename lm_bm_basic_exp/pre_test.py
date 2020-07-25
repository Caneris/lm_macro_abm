import numpy as np
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
        sigma_w = rd.uniform(0.1, 0.3)
        sigma_m = rd.uniform(0.1, 0.3)
        min_w_par = rd.uniform(0.1, 0.6)
        W_r = rd.uniform(1, 50)
        alpha_2 = rd.uniform(0.1, 0.5)
        lambda_exp = rd.uniform(0.25, 0.75)
        u_r = rd.uniform(0,0.4)
        sigma = rd.uniform(1.5,3)
        f_max = rd.randint(1,4)
        sigma_delta = rd.uniform(0.001, 0.1)

        m = Model(T=T, alpha_2=alpha_2, sigma=sigma, N_app=N_app, N_good=N_good, lambda_LM=lambda_LM, sigma_m=sigma_m,
                  sigma_w= sigma_w, nu = 0.1, u_r=u_r, beta=1, lambda_exp = lambda_exp, F = 80, H = 500, min_w_par=min_w_par,
                  nr_to_r=True, mu_r = 0.4, gamma_nr = 0.4, sigma_delta=sigma_delta, a = 1, f_max=f_max, W_r=W_r)
        m.run()

        with open('pretest_robust_unemp_ID{}.csv'.format(ID), 'a', newline='') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            filewriter.writerow(m.u_r_arr)

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
    NC = 200
    args_arr = [(ID, NC, T) for ID in range(NR)]
    run_nc_with_mp(args_arr)
