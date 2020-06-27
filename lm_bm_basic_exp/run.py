from model_class import *
from plot_tool import *
import numpy as np
import numpy.random as rd
import time



T = 800
periods = T
rd.seed(1312343)
m = Model(T=T, alpha_2=0.25, chi_C=0.05, lambda_LM=1, sigma_m=0.1, sigma_w= 0.1, sigma_delta=0.001,
          nu=0.1, u_r=0.08, beta=0.8, lambda_exp = 0.5, F = 40, H = 250, N_app=4, sigma=1.5, mu_r=0.4,
          nr_to_r=True, a = 100, shock_t=400, min_realw_t=0.3, gamma_nr=0.4, minw_init_par=0.5)

start = time.time()
m.run()
end = time.time()
print(end - start)

fig1, fig2 = plot_lm(m, m.t, periods, 1)
get_wage_dist_fig(m)
get_aggregate_regs(m, m.t, 200)

fig1.show()
plt.close(fig1)

fig2.show()
plt.close(fig2)

# fig3.show()
# plt.close(fig3)

# fig4.show()
# plt.close(fig4)

# show nr in r jobs
print(np.sum(m.emp_matrix[:,np.logical_and(np.invert(m.nr_job_arr), m.non_routine_arr)]))