from model_class import *
from plot_tool import *
import numpy as np
import numpy.random as rd
import time



T = 1000
periods = T
rd.seed(13521332)
m = Model(T=T, alpha_2=0.5, chi_C=0.05, lambda_LM=1, sigma_m=0.1, sigma_w= 0.1, sigma_delta=0.001,
          nu=0.05, u_r=0.08, beta=1, lambda_exp = 1, F = 80, H = 500, N_app=4, sigma=0.25, mu_r=0.3,
          nr_to_r=True, a = 100, shock_t=600, min_realw_t=0.5, gamma_nr=0.33)

start = time.time()
m.run()
end = time.time()
print(end - start)

fig1, fig2 = plot_lm(m, m.t, m.t, 1)
fig3 = get_wage_dist_fig(m)
fig4 = get_aggregate_regs(m, m.t, m.t)

fig1.show()
plt.close(fig1)

fig2.show()
plt.close(fig2)

fig3.show()
plt.close(fig3)

fig4.show()
plt.close(fig4)

# show nr in r jobs
print(np.sum(m.emp_matrix[:,np.logical_and(np.invert(m.nr_job_arr), m.non_routine_arr)]))