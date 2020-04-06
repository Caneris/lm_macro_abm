import numpy as np



# H = 200, F = 20, u_r = 0.08, mu_r = 1, W_r = 1, gamma_nr = 0.33,
#                  m = 0.1, sigma = 0.5, delta = 1, alpha_2 = 0.25

def calibrate_model(H = 200, F = 20, Ar = 1, u_r = 0.08, mu_r = 1, W_r = 1, gamma_nr = 0.33, m = 0.1, sigma = 0.5, delta = 1, alpha_2 = 0.25):

    # get elasticity parameter
    rho = (sigma-1)/sigma
    # print(rho)
    koeff1 = 2**((rho-1)/rho)
    koeff2 = 2**(1/rho)

    mu_r = mu_r

    # 1. get Omega, mu_nr, W_nr
    Omega = (1-gamma_nr)/gamma_nr
    mu_nr, W_nr = Omega*mu_r, Omega*W_r

    # 2. get Nr, Nnr
    Nnr = np.round(H*(1-u_r)/(1+Omega))
    Nr = np.round(Omega*Nnr)

    # 3. get y
    y = koeff2*Omega*mu_r*Nnr

    # 4. get uc, p
    uc = koeff1*(W_r/mu_r)
    p = (1+m)*uc

    # 5. get pi, DIV
    pi = 2*Nr*W_r*m
    pi_f = pi/F
    DIV = delta*m*2*Nr*W_r

    DIV_r = 0.5 * (DIV / ((1-gamma_nr)*H))
    DIV_nr = 0.5 * (DIV / (gamma_nr*H))
    DIV_f = DIV/F

    # get I, c, C, alpha_1, AF
    Anr = Omega*Ar
    I, c = (1+delta*m)*2*Nr*W_r, y
    C = c*p

    c_r = 0.5 * (c / ((1-gamma_nr) * H))
    c_nr = 0.5 * (c / (gamma_nr * H))

    AF = uc*y
    Af = AF/F

    AH = (1 - gamma_nr)*H*Ar + gamma_nr*H*Anr
    alpha_1 = (C - alpha_2*AH)/I

    return mu_nr, W_nr, Anr, Af, p, y, pi_f, DIV_r, DIV_nr, DIV_f, c_r, c_nr, alpha_1, Nr, Nnr






