from model_class import *
from plot_tool import plot_lm



T = 500
rd.seed(334421457)
m = Model(T=T, alpha_2=0.1, chi_L=0.4, chi_C=0.4, lambda_LM=3, sigma_m=0.001, sigma_w= 0.005,
          nu=0.1, u_r=0.10, beta=1, lambda_exp = 0.5, F = 40, H = 250)

m.run()



f1, f2 = plot_lm(m, m.t, m.t)
f1.show()
f2.show()