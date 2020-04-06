import numpy as np



# H = 200, F = 20, u_r = 0.08, mu_r = 1, W_r = 1, gamma_nr = 0.33,
#                  m = 0.1, sigma = 0.5, delta = 1, alpha_2 = 0.25

def calibrate_model(H = 200, F = 20, Ah = 1, u_r = 0.08, mu_r = 1, W_r = 1, gamma_nr = 0.33, m = 0.1, sigma = 0.5, delta = 1, alpha_2 = 0.25):

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
    print(y, koeff2)

    # 4. get uc, p
    uc = koeff1*(W_r/mu_r)
    p = (1+m)*uc

    # 5. get pi, DIV
    pi = 2*Nr*W_r*m
    pi_f = pi/F
    DIV = delta*m*2*Nr*W_r

    DIV_h = DIV/H
    DIV_f = DIV/F

    # get I, c, C, alpha_1, AH, AF
    I, c = (1+delta*m)*2*Nr*W_r, y
    C = c*p
    C_h = C/H

    # AH = G*(C/alpha_2)

    AF = uc*y
    Af = AF/F

    AH = H * Ah
    alpha_1 = (C - alpha_2*AH)/I


    return mu_nr, W_nr, Af, p, y, pi_f, DIV_h, DIV_f, C_h, alpha_1, Nr, Nnr


cal = calibrate_model()
print(cal)





