from model_class import *
from plot_tool import *
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rd
import time



T = 300
rd.seed(132356)
m = Model(T=T, alpha_2=0.1, chi_C=0.05, lambda_LM=1, sigma_m=0.05, sigma_w= 0.05, sigma_delta=0.001,
          nu=0.1, u_r=0.08, beta=1, lambda_exp = 0.25, F = 80, H = 500, N_app=4, sigma=0.5, mu_r=0.3)

start = time.time()
m.run()
end = time.time()
print(end - start)

fig1, fig2 = plot_lm(m, m.t, m.t, 1)
fig1.show()
fig2.show()

fig3 = get_wage_dist_fig(m)
fig3.show()