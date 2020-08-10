import numpy as np
import numpy.random as rd
import time
from multiprocessing import Pool
from setting_generator import pick_element
from model_class import Model
import csv


def run_nc(args):
    ID, NC, T, par_vals, par_names = args
    print('start simulation {} with NC = {}'.format(ID, NC))
    run_perms(ID, NC, T, par_vals, par_names)


def run_perms(ID, NC, T, par_vals, par_names):

    with open('OFAT_{}.csv'.format(par_names[ID]), 'w+', newline='') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        filewriter.writerow([par_names[ID], "unemployment_rate", "gini_coeff", "mean_price",
                             "mean_wage", "median_wage", "mean_mark_up", "Y", "DY", "C", "DC",
                             "INV", "GDP"])

    # N_app
    if ID == 0:

        perm_seq = rd.permutation(NC)
        for j in range(NC):
            num = perm_seq[j]
            i = pick_element(par_vals, num)[0]
            N_app = par_vals[i]

            m = Model(T=T, alpha_2=0.25, sigma=1.5, N_app=N_app, N_good=4, lambda_LM=10, sigma_m=0.1,
                      sigma_w= 0.2, nu = 0.1, u_r=0.08, beta=1, lambda_exp = 0.5, F = 80, H = 500, min_w_par=0.4,
                      nr_to_r=False, mu_r = 0.4, gamma_nr = 0.4, sigma_delta=0.001, a = 1, f_max=1, W_r=1)
            m.run()

            with open('OFAT_{}.csv'.format(par_names[ID]), 'a', newline='') as csvfile:
                filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                filewriter.writerow([N_app, np.mean(m.u_r_arr[T-300:T]), np.mean(m.gini_coeff[T-300:T]),
                                     np.mean(m.mean_p_arr[T-300:T]), np.mean(m.mean_nominal_w_arr[T-300:T]),
                                     np.mean(m.median_w_arr[T-300:T]), np.mean(m.mean_m_arr[T-300:T]),
                                     np.mean(m.Y_arr[T-300:T]), np.mean(m.DY_arr[T-300:T]),
                                     np.mean(m.C_arr[T-300:T]), np.mean(m.DC_arr[T-300:T]),
                                     np.mean(m.INV_arr[T-300:T]), np.mean(m.GDP[T-300:T])])

    # N_good
    elif ID == 1:

        perm_seq = rd.permutation(NC)
        for j in range(NC):
            num = perm_seq[j]
            i = pick_element(par_vals, num)[0]
            N_good = par_vals[i]

            m = Model(T=T, alpha_2=0.25, sigma=1.5, N_app=4, N_good=N_good, lambda_LM=10, sigma_m=0.1,
                      sigma_w= 0.2, nu = 0.1, u_r=0.08, beta=1, lambda_exp = 0.5, F = 80, H = 500, min_w_par=0.4,
                      nr_to_r=False, mu_r = 0.4, gamma_nr = 0.4, sigma_delta=0.001, a = 1, f_max=1, W_r=1)
            m.run()

            with open('OFAT_{}.csv'.format(par_names[ID]), 'a', newline='') as csvfile:
                filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                filewriter.writerow([N_good, np.mean(m.u_r_arr[T-300:T]), np.mean(m.gini_coeff[T-300:T]),
                                     np.mean(m.mean_p_arr[T-300:T]), np.mean(m.mean_nominal_w_arr[T-300:T]),
                                     np.mean(m.median_w_arr[T-300:T]), np.mean(m.mean_m_arr[T-300:T]),
                                     np.mean(m.Y_arr[T-300:T]), np.mean(m.DY_arr[T-300:T]),
                                     np.mean(m.C_arr[T-300:T]), np.mean(m.DC_arr[T-300:T]),
                                     np.mean(m.INV_arr[T-300:T]), np.mean(m.GDP[T-300:T])])

    # lambda_LM
    elif ID == 2:

        perm_seq = rd.permutation(NC)
        for j in range(NC):
            num = perm_seq[j]
            i = pick_element(par_vals, num)[0]
            lambda_LM = par_vals[i]

            m = Model(T=T, alpha_2=0.25, sigma=1.5, N_app=4, N_good=4, lambda_LM=lambda_LM, sigma_m=0.1,
                      sigma_w= 0.2, nu = 0.1, u_r=0.08, beta=1, lambda_exp = 0.5, F = 80, H = 500, min_w_par=0.4,
                      nr_to_r=False, mu_r = 0.4, gamma_nr = 0.4, sigma_delta=0.001, a = 1, f_max=1, W_r=1)
            m.run()

            with open('OFAT_{}.csv'.format(par_names[ID]), 'a', newline='') as csvfile:
                filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                filewriter.writerow([lambda_LM, np.mean(m.u_r_arr[T-300:T]), np.mean(m.gini_coeff[T-300:T]),
                                     np.mean(m.mean_p_arr[T-300:T]), np.mean(m.mean_nominal_w_arr[T-300:T]),
                                     np.mean(m.median_w_arr[T-300:T]), np.mean(m.mean_m_arr[T-300:T]),
                                     np.mean(m.Y_arr[T-300:T]), np.mean(m.DY_arr[T-300:T]),
                                     np.mean(m.C_arr[T-300:T]), np.mean(m.DC_arr[T-300:T]),
                                     np.mean(m.INV_arr[T-300:T]), np.mean(m.GDP[T-300:T])])

    # sigma_w
    elif ID == 3:

        perm_seq = rd.permutation(NC)
        for j in range(NC):
            num = perm_seq[j]
            i = pick_element(par_vals, num)[0]
            sigma_w = par_vals[i]

            m = Model(T=T, alpha_2=0.25, sigma=1.5, N_app=4, N_good=4, lambda_LM=10, sigma_m=0.1,
                      sigma_w=sigma_w, nu=0.1, u_r=0.08, beta=1, lambda_exp=0.5, F=80, H=500, min_w_par=0.4,
                      nr_to_r=False, mu_r=0.4, gamma_nr=0.4, sigma_delta=0.001, a=1, f_max=1, W_r=1)
            m.run()

            with open('OFAT_{}.csv'.format(par_names[ID]), 'a', newline='') as csvfile:
                filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                filewriter.writerow([sigma_w, np.mean(m.u_r_arr[T-300:T]), np.mean(m.gini_coeff[T-300:T]),
                                     np.mean(m.mean_p_arr[T-300:T]), np.mean(m.mean_nominal_w_arr[T-300:T]),
                                     np.mean(m.median_w_arr[T-300:T]), np.mean(m.mean_m_arr[T-300:T]),
                                     np.mean(m.Y_arr[T-300:T]), np.mean(m.DY_arr[T-300:T]),
                                     np.mean(m.C_arr[T-300:T]), np.mean(m.DC_arr[T-300:T]),
                                     np.mean(m.INV_arr[T-300:T]), np.mean(m.GDP[T-300:T])])

    # sigma_m
    elif ID == 4:

        perm_seq = rd.permutation(NC)
        for j in range(NC):
            num = perm_seq[j]
            i = pick_element(par_vals, num)[0]
            sigma_m = par_vals[i]

            m = Model(T=T, alpha_2=0.25, sigma=1.5, N_app=4, N_good=4, lambda_LM=10, sigma_m=sigma_m,
                      sigma_w=0.2, nu=0.1, u_r=0.08, beta=1, lambda_exp=0.5, F=80, H=500, min_w_par=0.4,
                      nr_to_r=False, mu_r=0.4, gamma_nr=0.4, sigma_delta=0.001, a=1, f_max=1, W_r=1)
            m.run()

            with open('OFAT_{}.csv'.format(par_names[ID]), 'a', newline='') as csvfile:
                filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                filewriter.writerow([sigma_m, np.mean(m.u_r_arr[T-300:T]), np.mean(m.gini_coeff[T-300:T]),
                                     np.mean(m.mean_p_arr[T-300:T]), np.mean(m.mean_nominal_w_arr[T-300:T]),
                                     np.mean(m.median_w_arr[T-300:T]), np.mean(m.mean_m_arr[T-300:T]),
                                     np.mean(m.Y_arr[T-300:T]), np.mean(m.DY_arr[T-300:T]),
                                     np.mean(m.C_arr[T-300:T]), np.mean(m.DC_arr[T-300:T]),
                                     np.mean(m.INV_arr[T-300:T]), np.mean(m.GDP[T-300:T])])

    # min_w_par
    elif ID == 5:

        perm_seq = rd.permutation(NC)
        for j in range(NC):
            num = perm_seq[j]
            i = pick_element(par_vals, num)[0]
            min_w_par = par_vals[i]

            m = Model(T=T, alpha_2=0.25, sigma=1.5, N_app=4, N_good=4, lambda_LM=10, sigma_m=0.1,
                      sigma_w=0.2, nu=0.1, u_r=0.08, beta=1, lambda_exp=0.5, F=80, H=500, min_w_par=min_w_par,
                      nr_to_r=False, mu_r=0.4, gamma_nr=0.4, sigma_delta=0.001, a=1, f_max=1, W_r=1)
            m.run()

            with open('OFAT_{}.csv'.format(par_names[ID]), 'a', newline='') as csvfile:
                filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                filewriter.writerow([min_w_par, np.mean(m.u_r_arr[T-300:T]), np.mean(m.gini_coeff[T-300:T]),
                                     np.mean(m.mean_p_arr[T-300:T]), np.mean(m.mean_nominal_w_arr[T-300:T]),
                                     np.mean(m.median_w_arr[T-300:T]), np.mean(m.mean_m_arr[T-300:T]),
                                     np.mean(m.Y_arr[T-300:T]), np.mean(m.DY_arr[T-300:T]),
                                     np.mean(m.C_arr[T-300:T]), np.mean(m.DC_arr[T-300:T]),
                                     np.mean(m.INV_arr[T-300:T]), np.mean(m.GDP[T-300:T])])



def run_nc_with_mp(args_arr):

    start_time = time.time()

    p = Pool()
    p.map(run_nc, args_arr)

    p.close()
    p.join()

    end_time = time.time() - start_time
    print("Simulating {} mc simulations took {} time using mp".format(len(args_arr), end_time))


if __name__ == '__main__':

    N_app_arr = np.linspace(2, 21, 20).astype(int)
    N_good_arr = np.linspace(2, 21, 20).astype(int)
    lambda_LM_arr = np.linspace(1, 20, 20)
    sigma_w_arr = np.linspace(0.01, 0.8, 20)
    sigma_m_arr = np.linspace(0.01, 0.8, 20)
    min_w_par_arr = np.round(np.linspace(0.01, 1, 20), 2)

    par_list = [N_app_arr, N_good_arr, lambda_LM_arr, sigma_m_arr, sigma_w_arr, min_w_par_arr]
    par_names = ["N_app", "N_good", "lambda_LM", "sigma_m", "sigma_w", "min_w_par"]

    # Number of periods per simulation
    T = 1000
    # Number of replications (cores)
    NR = 6
    # number of cases
    NC = 400
    args_arr = [(ID, NC, T, par_list[ID], par_names) for ID in range(NR)]
    run_nc_with_mp(args_arr)