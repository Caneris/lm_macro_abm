from model_class import *
from plot_tool import plot_lm
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rd
import time



T = 300
rd.seed(132453256)
m = Model(T=T, alpha_2=0.1, chi_L=0.1, chi_C=0.1, lambda_LM=1, sigma_m=0.05, sigma_w= 0.001, sigma_delta=0.001,
          nu=0.1, u_r=0.08, beta=1, lambda_exp = 0.25, F = 40, H = 250)

start = time.time()
m.run()
end = time.time()
print(end - start)


f1, f2 = plot_lm(m, m.t, m.t, 1)
f1.show()
f2.show()

