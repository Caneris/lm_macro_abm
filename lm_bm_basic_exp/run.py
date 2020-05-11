from model_class import *
from plot_tool import *
import numpy as np
import numpy.random as rd
import time



T = 1000
rd.seed(13561213)
m = Model(T=T, alpha_2=0.25, chi_C=0.05, lambda_LM=1, sigma_m=0.1, sigma_w= 0.05, sigma_delta=0.001,
          nu=0.1, u_r=0.08, beta=1, lambda_exp = 0.25, F = 80, H = 500, N_app=4, sigma=0.5, mu_r=0.3,
          nr_to_r=True)

start = time.time()
m.run()
end = time.time()
print(end - start)

fig1, fig2 = plot_lm(m, m.t, 300, 1)
fig3 = get_wage_dist_fig(m)
fig4 = get_aggregate_regs(m, m.t, m.t)
fig1.show()
fig2.show()
fig3.show()
fig4.show()
plt.close('all')

# show nr in r jobs
print(np.sum(m.emp_matrix[:,np.logical_and(np.invert(m.nr_job_arr), m.non_routine_arr)]))