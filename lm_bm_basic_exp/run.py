from model_class import *
from plot_tool import plot_lm
import matplotlib.pyplot as plt
import time



T = 1000
rd.seed(334333)
m = Model(T=T, alpha_2=0.1, chi_L=0.4, chi_C=0.4, lambda_LM=3, sigma_m=0.003, sigma_w= 0.003,
          nu=0.1, u_r=0.08, beta=1, lambda_exp = 0.5, F = 16, H = 100)

start = time.time()
m.run()
end = time.time()
print(end - start)


f1, f2 = plot_lm(m, m.t, m.t, 1)
f1.show()
f2.show()