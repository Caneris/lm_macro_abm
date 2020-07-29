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


        m = Model(T=T, alpha_2=0.25, sigma=1.5, N_app=4, N_good=4, lambda_LM=1, sigma_m=0.1,
                  sigma_w= 0.2, nu = 0.1, u_r=0.08, beta=1, lambda_exp = 0.5, F = 80, H = 500, min_w_par=0.4,
                  nr_to_r=True, mu_r = 0.4, gamma_nr = 0.4, sigma_delta=0.001, a = 1, f_max=1, W_r=1)
        m.run()

        with open('tt_replicates_unemp_gini_ID{}.csv'.format(ID), 'a', newline='') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            filewriter.writerow(np.mean(m.u_r_arr[T-300:T]), np.mean(m.gini_coeff[T-300:T]))

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
    T = 500
    # Number of replications (cores)
    NR = 10
    # number of cases
    NC = 1000
    args_arr = [(ID, NC, T) for ID in range(NR)]
    run_nc_with_mp(args_arr)