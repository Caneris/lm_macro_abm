import numpy as np
from sys import exit


# H = 200, F = 20, u_r = 0.08, mu_r = 1, W_r = 1, gamma_nr = 0.33,
#                  m = 0.1, sigma = 0.5, delta = 1, alpha_2 = 0.25

def calibrate_model(H = 200, F = 20, Ah = 1, u_r = 0.08, mu_r = 0.3, W_r = 1, gamma_nr = 0.33, m = 0.1, sigma = 0.5, delta = 1, alpha_2 = 0.1):

    # get elasticity parameter
    rho = (sigma-1)/sigma

    if mu_r < 0:
        print("\nSorry, but mu_r has to be between 0 and 1.")
        exit()

    mu_nr = 1 - mu_r

    # 1. get Omega, mu_nr, W_nr
    Omega = (1-gamma_nr)/gamma_nr
    X_1 = mu_nr/mu_r
    W_nr = (X_1**rho)*(Omega**(1-rho))*W_r

    # 2. get Nr, Nnr
    Nnr = np.round(H*(1-u_r)/(1+Omega))
    Nr = np.round(Omega*Nnr)

    # 3. get y
    Y = (2**(1/rho))*Omega*mu_r*Nnr
    y_f = Y/F

    # 4. get uc, p
    uc = (Nr*W_r + Nnr*W_nr)/Y
    p = (1+m)*uc

    AF = uc * Y
    Af = AF / F

    # 5. get pi, DIV
    Pi = m*(Nr*W_r + Nnr*W_nr)
    pi_f = Pi/F
    DIV = delta*Pi

    div_f = DIV/F
    div_h = DIV/H

    # get I, c, C, alpha_1, AF
    AH = H*Ah
    I = Nr*W_r + Nnr*W_nr + DIV
    alpha_1 = (Y * p - alpha_2 * AH) / I

    c = Y/H # individual steady state consumption

    print("rho: {}, mu_r: {}, W_nr: {}, Ah: {}, Af: {}, uc: {}, p: {}, y_f: {}".format(rho, mu_nr, W_nr, Ah, Af, uc, p, y_f))
    print("pi_f: {}, div_h: {}, div_f: {}, c: {}, alpha_1: {}, Nr: {}, Nnr: {}".format(pi_f, div_h, div_f, c, alpha_1, Nr, Nnr))

    return mu_nr, W_nr, Af, uc, p, y_f, pi_f, div_h, div_f, c, alpha_1, Nr, Nnr






