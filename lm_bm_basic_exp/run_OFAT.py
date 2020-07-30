import numpy as np
import time
from multiprocessing import Pool
from model_class import Model
import csv


def run_nc(args):
    ID, NC, T, par_vals = args
    print('start simulation {} with NC = {}'.format(ID, NC))
    run_perms(ID, NC, T, par_vals)


def run_perms(ID, NC, T, par_vals):

    # N_app
    if ID == 0:

        with open('OFAT_N_app{}.csv'.format(ID), 'w+', newline='') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            filewriter.writerow(["N_app", "unemployment_rate", "gini_coeff"])

        for j in range(NC):
            i = j%len(par_vals)
            N_app = par_vals[i]

            m = Model(T=T, alpha_2=0.25, sigma=1.5, N_app=N_app, N_good=4, lambda_LM=1, sigma_m=0.1,
                      sigma_w= 0.2, nu = 0.1, u_r=0.08, beta=1, lambda_exp = 0.5, F = 80, H = 500, min_w_par=0.4,
                      nr_to_r=True, mu_r = 0.4, gamma_nr = 0.4, sigma_delta=0.001, a = 1, f_max=1, W_r=1)
            m.run()

            with open('OFAT_N_app{}.csv'.format(ID), 'a', newline='') as csvfile:
                filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                filewriter.writerow([N_app, np.mean(m.u_r_arr[T-300:T]), np.mean(m.gini_coeff[T-300:T])])

    # N_good
    elif ID == 1:

        with open('OFAT_N_good{}.csv'.format(ID), 'w+', newline='') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            filewriter.writerow(["N_app", "unemployment_rate", "gini_coeff"])

        for j in range(NC):
            i = j%len(par_vals)
            N_good = par_vals[i]

            m = Model(T=T, alpha_2=0.25, sigma=1.5, N_app=4, N_good=N_good, lambda_LM=1, sigma_m=0.1,
                      sigma_w= 0.2, nu = 0.1, u_r=0.08, beta=1, lambda_exp = 0.5, F = 80, H = 500, min_w_par=0.4,
                      nr_to_r=True, mu_r = 0.4, gamma_nr = 0.4, sigma_delta=0.001, a = 1, f_max=1, W_r=1)
            m.run()

            with open('OFAT_N_good{}.csv'.format(ID), 'a', newline='') as csvfile:
                filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                filewriter.writerow([N_good, np.mean(m.u_r_arr[T-300:T]), np.mean(m.gini_coeff[T-300:T])])

    # lambda_LM
    elif ID == 2:

        with open('OFAT_N_good{}.csv'.format(ID), 'w+', newline='') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            filewriter.writerow(["N_app", "unemployment_rate", "gini_coeff"])

        for j in range(NC):
            i = j%len(par_vals)
            lambda_LM = par_vals[i]

            m = Model(T=T, alpha_2=0.25, sigma=1.5, N_app=4, N_good=4, lambda_LM=lambda_LM, sigma_m=0.1,
                      sigma_w= 0.2, nu = 0.1, u_r=0.08, beta=1, lambda_exp = 0.5, F = 80, H = 500, min_w_par=0.4,
                      nr_to_r=True, mu_r = 0.4, gamma_nr = 0.4, sigma_delta=0.001, a = 1, f_max=1, W_r=1)
            m.run()

            with open('OFAT_N_good{}.csv'.format(ID), 'a', newline='') as csvfile:
                filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                filewriter.writerow([lambda_LM, np.mean(m.u_r_arr[T-300:T]), np.mean(m.gini_coeff[T-300:T])])

    # sigma_w
    elif ID == 3:

        with open('OFAT_N_good{}.csv'.format(ID), 'w+', newline='') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            filewriter.writerow(["sigma_w", "unemployment_rate", "gini_coeff"])

        for j in range(NC):
            i = j % len(par_vals)
            sigma_w = par_vals[i]

            m = Model(T=T, alpha_2=0.25, sigma=1.5, N_app=4, N_good=4, lambda_LM=1, sigma_m=0.1,
                      sigma_w=sigma_w, nu=0.1, u_r=0.08, beta=1, lambda_exp=0.5, F=80, H=500, min_w_par=0.4,
                      nr_to_r=True, mu_r=0.4, gamma_nr=0.4, sigma_delta=0.001, a=1, f_max=1, W_r=1)
            m.run()

            with open('OFAT_N_good{}.csv'.format(ID), 'a', newline='') as csvfile:
                filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                filewriter.writerow([sigma_w, np.mean(m.u_r_arr[T - 300:T]), np.mean(m.gini_coeff[T - 300:T])])

    # sigma_m
    elif ID == 4:

        with open('OFAT_N_good{}.csv'.format(ID), 'w+', newline='') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            filewriter.writerow(["sigma_m", "unemployment_rate", "gini_coeff"])

        for j in range(NC):
            i = j % len(par_vals)
            sigma_m = par_vals[i]

            m = Model(T=T, alpha_2=0.25, sigma=1.5, N_app=4, N_good=4, lambda_LM=1, sigma_m=sigma_m,
                      sigma_w=0.2, nu=0.1, u_r=0.08, beta=1, lambda_exp=0.5, F=80, H=500, min_w_par=0.4,
                      nr_to_r=True, mu_r=0.4, gamma_nr=0.4, sigma_delta=0.001, a=1, f_max=1, W_r=1)
            m.run()

            with open('OFAT_N_good{}.csv'.format(ID), 'a', newline='') as csvfile:
                filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                filewriter.writerow([sigma_m, np.mean(m.u_r_arr[T - 300:T]), np.mean(m.gini_coeff[T - 300:T])])

    # min_w_par
    elif ID == 5:

        with open('OFAT_N_good{}.csv'.format(ID), 'w+', newline='') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            filewriter.writerow(["phi_mw", "unemployment_rate", "gini_coeff"])

        for j in range(NC):
            i = j % len(par_vals)
            min_w_par = par_vals[i]

            m = Model(T=T, alpha_2=0.25, sigma=1.5, N_app=4, N_good=N_good, lambda_LM=1, sigma_m=0.1,
                      sigma_w=0.2, nu=0.1, u_r=0.08, beta=1, lambda_exp=0.5, F=80, H=500, min_w_par=min_w_par,
                      nr_to_r=True, mu_r=0.4, gamma_nr=0.4, sigma_delta=0.001, a=1, f_max=1, W_r=1)
            m.run()

            with open('OFAT_N_good{}.csv'.format(ID), 'a', newline='') as csvfile:
                filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                filewriter.writerow([min_w_par, np.mean(m.u_r_arr[T - 300:T]), np.mean(m.gini_coeff[T - 300:T])])



def run_nc_with_mp(args_arr):

    start_time = time.time()

    p = Pool()
    p.map(run_nc, args_arr)

    p.close()
    p.join()

    end_time = time.time() - start_time
    print("Simulating {} mc simulations took {} time using mp".format(len(args_arr), end_time))


if __name__ == '__main__':

    N_app = np.linspace(2, 10,10)
    N_good = np.linspace(2, 10,10)
    lambda_LM = np.linspace(1, 10,10)
    sigma_w = np.linspace(0.1, 0.5, 10)
    sigma_m = np.linspace(0.1, 0.5, 10)
    min_w_par = np.linspace(0.1, 0.8, 10)

    par_list = [N_app, N_good, lambda_LM, sigma_w, sigma_m, min_w_par]

    # Number of periods per simulation
    T = 500
    # Number of replications (cores)
    NR = 6
    # number of cases
    NC = 100
    args_arr = [(ID, NC, T, par_list[ID]) for ID in range(NR)]
    run_nc_with_mp(args_arr)