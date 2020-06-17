from model_class import *
from plot_tool import *
import numpy as np
import numpy.random as rd
import time



T = 500
periods = T
rd.seed(13521332)
m = Model(T=T, alpha_2=0.5, chi_C=0.1, lambda_LM=1, sigma_m=0.2, sigma_w= 0.05, sigma_delta=0.001,
          nu=0.1, u_r=0.05, beta=1, lambda_exp = 0.5, F = 40, H = 250, N_app=4, sigma=1.5, mu_r=0.33,
          nr_to_r=True, a = 100, shock_t=0, min_realw_t=None, gamma_nr=0.33, W_r=50, minw_init_par = 0.4)

start = time.time()
m.run()
end = time.time()
print(end - start)

print(np.mean(m.mean_p_arr[400:]))

fig1, fig2 = plot_lm(m, m.t, m.t, 1)
fig3 = get_wage_dist_fig(m)
fig4 = get_aggregate_regs(m, m.t, 100)

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