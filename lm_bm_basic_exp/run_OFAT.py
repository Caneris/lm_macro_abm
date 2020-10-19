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
        filewriter.writerow([par_names[ID],
                             "unemployment_rate", "unemployment_rate_std",
                             "ur_r", "ur_r_std",
                             "unr_r", "unr_r_std",
                             "share_inactive", "share_inactive_std",
                             "gini_coeff", "gini_coeff_std",
                             "mean_price", "mean_price_std",
                             "mean_wage", "mean_wage_std",
                             "median_wage", "median_wage_std",
                             "mean_mark_up", "mean_mark_up_std",
                             "Y", "Y_std",
                             "DY", "DY_std",
                             "C", "C_std",
                             "DC", "DC_std",
                             "INV", "INV_std",
                             "GDP", "GDP_std",
                             "mean_r_wages", "mean_r_wages_std",
                             "mean_nr_wages", "mean_nr_wages_std",
                             "9/1", "9/1_std",
                             "9/5", "9/5_std",
                             "5/1", "5/1_std"])

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
                filewriter.writerow([N_app,
                                     np.mean(m.u_r_arr[T-300:T]), np.std(m.u_r_arr[T-300:T]),
                                     np.mean(m.ur_r_arr[T - 300:T]), np.std(m.ur_r_arr[T - 300:T]),
                                     np.mean(m.unr_r_arr[T - 300:T]), np.std(m.unr_r_arr[T - 300:T]),
                                     np.mean(m.share_inactive[T - 300:T]), np.std(m.share_inactive[T - 300:T]),
                                     np.mean(m.gini_coeff[T-300:T]), np.std(m.gini_coeff[T-300:T]),
                                     np.mean(m.mean_p_arr[T-300:T]), np.std(m.mean_p_arr[T-300:T]),
                                     np.mean(m.mean_nominal_w_arr[T-300:T]), np.std(m.mean_nominal_w_arr[T-300:T]),
                                     np.mean(m.median_w_arr[T-300:T]), np.std(m.median_w_arr[T-300:T]),
                                     np.mean(m.mean_m_arr[T-300:T]), np.std(m.mean_m_arr[T-300:T]),
                                     np.mean(m.Y_arr[T-300:T]), np.std(m.Y_arr[T-300:T]),
                                     np.mean(m.DY_arr[T-300:T]), np.std(m.DY_arr[T-300:T]),
                                     np.mean(m.C_arr[T-300:T]), np.std(m.C_arr[T-300:T]),
                                     np.mean(m.DC_arr[T-300:T]), np.std(m.DC_arr[T-300:T]),
                                     np.mean(m.INV_arr[T-300:T]), np.std(m.INV_arr[T-300:T]),
                                     np.mean(m.GDP[T-300:T]), np.std(m.GDP[T-300:T]),
                                     np.mean(m.mean_r_w_arr[T-300:T]), np.std(m.mean_r_w_arr[T-300:T]),
                                     np.mean(m.mean_nr_w_arr[T - 300:T]), np.std(m.mean_nr_w_arr[T - 300:T]),
                                     np.mean(m.nine_to_one[T - 300:T]), np.std(m.nine_to_one[T - 300:T]),
                                     np.mean(m.nine_to_five[T - 300:T]), np.std(m.nine_to_five[T - 300:T]),
                                     np.mean(m.five_to_one[T - 300:T]), np.std(m.five_to_one[T - 300:T])
                                     ])

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
                filewriter.writerow([N_good,
                                     np.mean(m.u_r_arr[T - 300:T]), np.std(m.u_r_arr[T - 300:T]),
                                     np.mean(m.ur_r_arr[T - 300:T]), np.std(m.ur_r_arr[T - 300:T]),
                                     np.mean(m.unr_r_arr[T - 300:T]), np.std(m.unr_r_arr[T - 300:T]),
                                     np.mean(m.share_inactive[T - 300:T]), np.std(m.share_inactive[T - 300:T]),
                                     np.mean(m.gini_coeff[T - 300:T]), np.std(m.gini_coeff[T - 300:T]),
                                     np.mean(m.mean_p_arr[T - 300:T]), np.std(m.mean_p_arr[T - 300:T]),
                                     np.mean(m.mean_nominal_w_arr[T - 300:T]), np.std(m.mean_nominal_w_arr[T - 300:T]),
                                     np.mean(m.median_w_arr[T - 300:T]), np.std(m.median_w_arr[T - 300:T]),
                                     np.mean(m.mean_m_arr[T - 300:T]), np.std(m.mean_m_arr[T - 300:T]),
                                     np.mean(m.Y_arr[T - 300:T]), np.std(m.Y_arr[T - 300:T]),
                                     np.mean(m.DY_arr[T - 300:T]), np.std(m.DY_arr[T - 300:T]),
                                     np.mean(m.C_arr[T - 300:T]), np.std(m.C_arr[T - 300:T]),
                                     np.mean(m.DC_arr[T - 300:T]), np.std(m.DC_arr[T - 300:T]),
                                     np.mean(m.INV_arr[T - 300:T]), np.std(m.INV_arr[T - 300:T]),
                                     np.mean(m.GDP[T - 300:T]), np.std(m.GDP[T - 300:T]),
                                     np.mean(m.mean_r_w_arr[T - 300:T]), np.std(m.mean_r_w_arr[T - 300:T]),
                                     np.mean(m.mean_nr_w_arr[T - 300:T]), np.std(m.mean_nr_w_arr[T - 300:T]),
                                     np.mean(m.nine_to_one[T - 300:T]), np.std(m.nine_to_one[T - 300:T]),
                                     np.mean(m.nine_to_five[T - 300:T]), np.std(m.nine_to_five[T - 300:T]),
                                     np.mean(m.five_to_one[T - 300:T]), np.std(m.five_to_one[T - 300:T])
                                     ])


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
                filewriter.writerow([lambda_LM,
                                     np.mean(m.u_r_arr[T - 300:T]), np.std(m.u_r_arr[T - 300:T]),
                                     np.mean(m.ur_r_arr[T - 300:T]), np.std(m.ur_r_arr[T - 300:T]),
                                     np.mean(m.unr_r_arr[T - 300:T]), np.std(m.unr_r_arr[T - 300:T]),
                                     np.mean(m.share_inactive[T - 300:T]), np.std(m.share_inactive[T - 300:T]),
                                     np.mean(m.gini_coeff[T - 300:T]), np.std(m.gini_coeff[T - 300:T]),
                                     np.mean(m.mean_p_arr[T - 300:T]), np.std(m.mean_p_arr[T - 300:T]),
                                     np.mean(m.mean_nominal_w_arr[T - 300:T]), np.std(m.mean_nominal_w_arr[T - 300:T]),
                                     np.mean(m.median_w_arr[T - 300:T]), np.std(m.median_w_arr[T - 300:T]),
                                     np.mean(m.mean_m_arr[T - 300:T]), np.std(m.mean_m_arr[T - 300:T]),
                                     np.mean(m.Y_arr[T - 300:T]), np.std(m.Y_arr[T - 300:T]),
                                     np.mean(m.DY_arr[T - 300:T]), np.std(m.DY_arr[T - 300:T]),
                                     np.mean(m.C_arr[T - 300:T]), np.std(m.C_arr[T - 300:T]),
                                     np.mean(m.DC_arr[T - 300:T]), np.std(m.DC_arr[T - 300:T]),
                                     np.mean(m.INV_arr[T - 300:T]), np.std(m.INV_arr[T - 300:T]),
                                     np.mean(m.GDP[T - 300:T]), np.std(m.GDP[T - 300:T]),
                                     np.mean(m.mean_r_w_arr[T - 300:T]), np.std(m.mean_r_w_arr[T - 300:T]),
                                     np.mean(m.mean_nr_w_arr[T - 300:T]), np.std(m.mean_nr_w_arr[T - 300:T]),
                                     np.mean(m.nine_to_one[T - 300:T]), np.std(m.nine_to_one[T - 300:T]),
                                     np.mean(m.nine_to_five[T - 300:T]), np.std(m.nine_to_five[T - 300:T]),
                                     np.mean(m.five_to_one[T - 300:T]), np.std(m.five_to_one[T - 300:T])
                                     ])


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
                filewriter.writerow([sigma_w,
                                     np.mean(m.u_r_arr[T - 300:T]), np.std(m.u_r_arr[T - 300:T]),
                                     np.mean(m.ur_r_arr[T - 300:T]), np.std(m.ur_r_arr[T - 300:T]),
                                     np.mean(m.unr_r_arr[T - 300:T]), np.std(m.unr_r_arr[T - 300:T]),
                                     np.mean(m.share_inactive[T - 300:T]), np.std(m.share_inactive[T - 300:T]),
                                     np.mean(m.gini_coeff[T - 300:T]), np.std(m.gini_coeff[T - 300:T]),
                                     np.mean(m.mean_p_arr[T - 300:T]), np.std(m.mean_p_arr[T - 300:T]),
                                     np.mean(m.mean_nominal_w_arr[T - 300:T]), np.std(m.mean_nominal_w_arr[T - 300:T]),
                                     np.mean(m.median_w_arr[T - 300:T]), np.std(m.median_w_arr[T - 300:T]),
                                     np.mean(m.mean_m_arr[T - 300:T]), np.std(m.mean_m_arr[T - 300:T]),
                                     np.mean(m.Y_arr[T - 300:T]), np.std(m.Y_arr[T - 300:T]),
                                     np.mean(m.DY_arr[T - 300:T]), np.std(m.DY_arr[T - 300:T]),
                                     np.mean(m.C_arr[T - 300:T]), np.std(m.C_arr[T - 300:T]),
                                     np.mean(m.DC_arr[T - 300:T]), np.std(m.DC_arr[T - 300:T]),
                                     np.mean(m.INV_arr[T - 300:T]), np.std(m.INV_arr[T - 300:T]),
                                     np.mean(m.GDP[T - 300:T]), np.std(m.GDP[T - 300:T]),
                                     np.mean(m.mean_r_w_arr[T - 300:T]), np.std(m.mean_r_w_arr[T - 300:T]),
                                     np.mean(m.mean_nr_w_arr[T - 300:T]), np.std(m.mean_nr_w_arr[T - 300:T]),
                                     np.mean(m.nine_to_one[T - 300:T]), np.std(m.nine_to_one[T - 300:T]),
                                     np.mean(m.nine_to_five[T - 300:T]), np.std(m.nine_to_five[T - 300:T]),
                                     np.mean(m.five_to_one[T - 300:T]), np.std(m.five_to_one[T - 300:T])
                                     ])


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
                filewriter.writerow([sigma_m,
                                     np.mean(m.u_r_arr[T - 300:T]), np.std(m.u_r_arr[T - 300:T]),
                                     np.mean(m.ur_r_arr[T - 300:T]), np.std(m.ur_r_arr[T - 300:T]),
                                     np.mean(m.unr_r_arr[T - 300:T]), np.std(m.unr_r_arr[T - 300:T]),
                                     np.mean(m.share_inactive[T - 300:T]), np.std(m.share_inactive[T - 300:T]),
                                     np.mean(m.gini_coeff[T - 300:T]), np.std(m.gini_coeff[T - 300:T]),
                                     np.mean(m.mean_p_arr[T - 300:T]), np.std(m.mean_p_arr[T - 300:T]),
                                     np.mean(m.mean_nominal_w_arr[T - 300:T]), np.std(m.mean_nominal_w_arr[T - 300:T]),
                                     np.mean(m.median_w_arr[T - 300:T]), np.std(m.median_w_arr[T - 300:T]),
                                     np.mean(m.mean_m_arr[T - 300:T]), np.std(m.mean_m_arr[T - 300:T]),
                                     np.mean(m.Y_arr[T - 300:T]), np.std(m.Y_arr[T - 300:T]),
                                     np.mean(m.DY_arr[T - 300:T]), np.std(m.DY_arr[T - 300:T]),
                                     np.mean(m.C_arr[T - 300:T]), np.std(m.C_arr[T - 300:T]),
                                     np.mean(m.DC_arr[T - 300:T]), np.std(m.DC_arr[T - 300:T]),
                                     np.mean(m.INV_arr[T - 300:T]), np.std(m.INV_arr[T - 300:T]),
                                     np.mean(m.GDP[T - 300:T]), np.std(m.GDP[T - 300:T]),
                                     np.mean(m.mean_r_w_arr[T - 300:T]), np.std(m.mean_r_w_arr[T - 300:T]),
                                     np.mean(m.mean_nr_w_arr[T - 300:T]), np.std(m.mean_nr_w_arr[T - 300:T]),
                                     np.mean(m.nine_to_one[T - 300:T]), np.std(m.nine_to_one[T - 300:T]),
                                     np.mean(m.nine_to_five[T - 300:T]), np.std(m.nine_to_five[T - 300:T]),
                                     np.mean(m.five_to_one[T - 300:T]), np.std(m.five_to_one[T - 300:T])
                                     ])


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
                filewriter.writerow([min_w_par,
                                     np.mean(m.u_r_arr[T - 300:T]), np.std(m.u_r_arr[T - 300:T]),
                                     np.mean(m.ur_r_arr[T - 300:T]), np.std(m.ur_r_arr[T - 300:T]),
                                     np.mean(m.unr_r_arr[T - 300:T]), np.std(m.unr_r_arr[T - 300:T]),
                                     np.mean(m.share_inactive[T - 300:T]), np.std(m.share_inactive[T - 300:T]),
                                     np.mean(m.gini_coeff[T - 300:T]), np.std(m.gini_coeff[T - 300:T]),
                                     np.mean(m.mean_p_arr[T - 300:T]), np.std(m.mean_p_arr[T - 300:T]),
                                     np.mean(m.mean_nominal_w_arr[T - 300:T]), np.std(m.mean_nominal_w_arr[T - 300:T]),
                                     np.mean(m.median_w_arr[T - 300:T]), np.std(m.median_w_arr[T - 300:T]),
                                     np.mean(m.mean_m_arr[T - 300:T]), np.std(m.mean_m_arr[T - 300:T]),
                                     np.mean(m.Y_arr[T - 300:T]), np.std(m.Y_arr[T - 300:T]),
                                     np.mean(m.DY_arr[T - 300:T]), np.std(m.DY_arr[T - 300:T]),
                                     np.mean(m.C_arr[T - 300:T]), np.std(m.C_arr[T - 300:T]),
                                     np.mean(m.DC_arr[T - 300:T]), np.std(m.DC_arr[T - 300:T]),
                                     np.mean(m.INV_arr[T - 300:T]), np.std(m.INV_arr[T - 300:T]),
                                     np.mean(m.GDP[T - 300:T]), np.std(m.GDP[T - 300:T]),
                                     np.mean(m.mean_r_w_arr[T - 300:T]), np.std(m.mean_r_w_arr[T - 300:T]),
                                     np.mean(m.mean_nr_w_arr[T - 300:T]), np.std(m.mean_nr_w_arr[T - 300:T]),
                                     np.mean(m.nine_to_one[T - 300:T]), np.std(m.nine_to_one[T - 300:T]),
                                     np.mean(m.nine_to_five[T - 300:T]), np.std(m.nine_to_five[T - 300:T]),
                                     np.mean(m.five_to_one[T - 300:T]), np.std(m.five_to_one[T - 300:T])
                                     ])

    # f_max
    elif ID == 6:

        perm_seq = rd.permutation(NC)
        for j in range(NC):
            num = perm_seq[j]
            i = pick_element(par_vals, num)[0]
            f_max = par_vals[i]

            m = Model(T=T, alpha_2=0.25, sigma=1.5, N_app=4, N_good=4, lambda_LM=10, sigma_m=0.1,
                      sigma_w=0.2, nu=0.1, u_r=0.08, beta=1, lambda_exp=0.5, F=80, H=500, min_w_par=0.4,
                      nr_to_r=False, mu_r=0.4, gamma_nr=0.4, sigma_delta=0.001, a=1, f_max=f_max, W_r=1)
            m.run()

            with open('OFAT_{}.csv'.format(par_names[ID]), 'a', newline='') as csvfile:
                filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                filewriter.writerow([f_max,
                                     np.mean(m.u_r_arr[T - 300:T]), np.std(m.u_r_arr[T - 300:T]),
                                     np.mean(m.ur_r_arr[T - 300:T]), np.std(m.ur_r_arr[T - 300:T]),
                                     np.mean(m.unr_r_arr[T - 300:T]), np.std(m.unr_r_arr[T - 300:T]),
                                     np.mean(m.share_inactive[T - 300:T]), np.std(m.share_inactive[T - 300:T]),
                                     np.mean(m.gini_coeff[T - 300:T]), np.std(m.gini_coeff[T - 300:T]),
                                     np.mean(m.mean_p_arr[T - 300:T]), np.std(m.mean_p_arr[T - 300:T]),
                                     np.mean(m.mean_nominal_w_arr[T - 300:T]), np.std(m.mean_nominal_w_arr[T - 300:T]),
                                     np.mean(m.median_w_arr[T - 300:T]), np.std(m.median_w_arr[T - 300:T]),
                                     np.mean(m.mean_m_arr[T - 300:T]), np.std(m.mean_m_arr[T - 300:T]),
                                     np.mean(m.Y_arr[T - 300:T]), np.std(m.Y_arr[T - 300:T]),
                                     np.mean(m.DY_arr[T - 300:T]), np.std(m.DY_arr[T - 300:T]),
                                     np.mean(m.C_arr[T - 300:T]), np.std(m.C_arr[T - 300:T]),
                                     np.mean(m.DC_arr[T - 300:T]), np.std(m.DC_arr[T - 300:T]),
                                     np.mean(m.INV_arr[T - 300:T]), np.std(m.INV_arr[T - 300:T]),
                                     np.mean(m.GDP[T - 300:T]), np.std(m.GDP[T - 300:T]),
                                     np.mean(m.mean_r_w_arr[T - 300:T]), np.std(m.mean_r_w_arr[T - 300:T]),
                                     np.mean(m.mean_nr_w_arr[T - 300:T]), np.std(m.mean_nr_w_arr[T - 300:T]),
                                     np.mean(m.nine_to_one[T - 300:T]), np.std(m.nine_to_one[T - 300:T]),
                                     np.mean(m.nine_to_five[T - 300:T]), np.std(m.nine_to_five[T - 300:T]),
                                     np.mean(m.five_to_one[T - 300:T]), np.std(m.five_to_one[T - 300:T])
                                     ])


def run_nc_with_mp(args_arr):

    start_time = time.time()

    p = Pool()
    p.map(run_nc, args_arr)

    p.close()
    p.join()

    end_time = time.time() - start_time
    print("Simulating {} mc simulations took {} time using mp".format(len(args_arr), end_time))


if __name__ == '__main__':

    N_app_arr = np.linspace(2, 41, 10).astype(int)
    N_good_arr = np.linspace(2, 41, 10).astype(int)
    lambda_LM_arr = np.linspace(1, 20, 10)
    sigma_w_arr = np.linspace(0.01, 0.8, 10)
    sigma_m_arr = np.linspace(0.01, 0.8, 10)
    min_w_par_arr = np.round(np.linspace(0.01, 1, 10), 2)
    f_max_arr = np.linspace(1, 10, 10).astype(int)

    par_list = [N_app_arr, N_good_arr, lambda_LM_arr, sigma_m_arr, sigma_w_arr, min_w_par_arr, f_max_arr]
    par_names = ["N_app", "N_good", "lambda_LM", "sigma_m", "sigma_w", "min_w_par", "f_max"]

    # Number of periods per simulation
    T = 1000
    # Number of replications (cores)
    NR = 7
    # number of cases
    NC = 100
    args_arr = [(ID, NC, T, par_list[ID], par_names) for ID in range(NR)]
    run_nc_with_mp(args_arr)