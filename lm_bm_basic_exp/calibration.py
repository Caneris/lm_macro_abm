import numpy as np




def calibrate_model(u_r, gamma_nr, H, F, W_r, m, sigma, delta, alpha_2, G, nu_Af):

    # get elasticity parameter
    rho = (sigma-1)/sigma
    print(rho)
    koeff1 = 2**((rho-1)/rho)
    koeff2 = 2**(1/rho)

    # Steady state W_r equal to price
    mu_r = 2*(1+m)*koeff1

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
    DIV = delta*m*2*Nr*W_r

    # get I, c, C, alpha_1, AH, AF
    I, c = (1+delta*m)*2*Nr*W_r, y
    C = c*p

    AH = G*(C/alpha_2)
    Ah = AH/H

    AF = uc*y*(1+nu_Af)
    Af = AF/F

    alpha_1 = (C - alpha_2*AH)/I


    print(mu_r, mu_nr, W_r, W_nr, p, y, p*y, pi, DIV, I, C, AH, Ah, alpha_1, AF, Af)
    print(Nr, Nnr)


calibrate_model(0.10, 0.33, 200, 20, 1, 0.1, 0.5, 1, 0.25, 0.3, 0)





