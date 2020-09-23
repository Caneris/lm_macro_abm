import numpy as np
import time
from multiprocessing import Pool
from model_class import Model
import csv


def run_nc(args):
    ID, N_sim, T, periods, param_ID_settings = args
    print('start simulation {} with N_sim = {}'.format(ID, N_sim))
    run_perms(ID, N_sim, T, periods, param_ID_settings)


def run_perms(ID, N_sim, T, periods, param_ID_settings):
    tt = T-periods
    with open('sobol_ID{}.csv'.format(ID), 'w+', newline='') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        filewriter.writerow(["sim_num", 'lambda_LM', 'phi_mw', 'sigma_w', 'sigma_m', 'N_app', 'N_good',
                             'unemployment_rate', 'std_unemployment_rate',
                             'u_r_r', 'std_u_r_r', 'u_r_nr', 'std_u_r_nr',
                             'share_inactive', 'std_share_inactive',
                             'gini_coeff', 'std_gini_coeff',
                             'mean_price', 'std_mean_price', 'mean_wage', 'std_mean_wage',
                             'median_wage', 'std_median_wage', 'mean_mark_up', 'std_mean_mark_up',
                             'Y', 'std_Y', 'DY', 'std_DY', 'C', 'std_C', 'DC', 'std_DC',
                             'INV', 'std_INV', 'GDP', 'std_GDP'])

    for j in range(N_sim):
        phi_mw, sigma_w, N_app = param_ID_settings[j, :]
        print('start simulating case number {}'.format(j))

        m = Model(T=T, alpha_2=0.25, sigma=1.5, N_app=int(N_app), N_good=4, lambda_LM=10, sigma_m=0.1,
                  sigma_w=sigma_w, nu=0.1, u_r=0.08, beta=1, lambda_exp=0.5, F=80, H=500, min_w_par=phi_mw,
                  nr_to_r=False, mu_r=0.4, gamma_nr=0.4, sigma_delta=0.001, a=1, f_max=1, W_r=1)
        m.run()

        with open('sobol_ID{}.csv'.format(ID), 'a', newline='') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            filewriter.writerow([j, phi_mw, sigma_w, N_app,
                                 np.mean(m.u_r_arr[tt:T]), np.std(m.u_r_arr[tt:T]),
                                 np.mean(m.ur_r_arr[tt:T]), np.std(m.ur_r_arr[tt:T]),
                                 np.mean(m.unr_r_arr[tt:T]), np.std(m.unr_r_arr[tt:T]),
                                 np.mean(m.share_inactive[tt:T]), np.std(m.share_inactive[tt:T]),
                                 np.mean(m.gini_coeff[tt:T]), np.std(m.gini_coeff[tt:T]),
                                 np.mean(m.mean_p_arr[tt:T]), np.std(m.mean_p_arr[tt:T]),
                                 np.mean(m.mean_nominal_w_arr[tt:T]), np.std(m.mean_nominal_w_arr[tt:T]),
                                 np.mean(m.median_w_arr[tt:T]), np.std(m.median_w_arr[tt:T]),
                                 np.mean(m.mean_m_arr[tt:T]), np.std(m.mean_m_arr[tt:T]),
                                 np.mean(m.Y_arr[tt:T]), np.std(m.Y_arr[tt:T]),
                                 np.mean(m.DY_arr[tt:T]), np.std(m.DY_arr[tt:T]),
                                 np.mean(m.C_arr[tt:T]), np.std(m.C_arr[tt:T]),
                                 np.mean(m.DC_arr[tt:T]), np.std(m.DC_arr[tt:T]),
                                 np.mean(m.INV_arr[tt:T]), np.std(m.INV_arr[tt:T]),
                                 np.mean(m.GDP[tt:T]), np.std(m.GDP[tt:T])])


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
    # 'lambda_LM', 'phi_mw', 'sigma_w', 'sigma_m', 'N_app', 'N_good'
    param_settings = np.loadtxt('C:/Users/atecan00/Desktop/saltelli/param_settings.csv', delimiter=',')

    n_Core = 5

    N_sim = int(param_settings.shape[0] / n_Core)
    print(N_sim)

    params_list = [0 for i in range(n_Core)]
    i_start = 0
    i_stop = 160
    i_step = 160

    for i in range(n_Core):
        params_list[i] = param_settings[i_start:i_stop, :]
        i_start += i_step
        i_stop += i_step

    # Number of periods per simulation
    T = 1000
    periods = 300
    tt = T - periods

    args_arr = [(ID, N_sim, T, periods, params_list[ID]) for ID in range(n_Core)]
    run_nc_with_mp(args_arr)
