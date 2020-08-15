from model_class import *
from plot_tool import *
import numpy as np
import numpy.random as rd
import time



T = 400
periods = T
rd.seed(132423213)
m = Model(T=T, alpha_2=0.25, N_good=4, lambda_LM=10, sigma_m=0.1, sigma_w=0.2, sigma_delta=0.001,
          nu=0.1, u_r=0.08, beta=1, lambda_exp=0.5, F=80, H=500, N_app=4, sigma=1.5, mu_r=0.4,
          nr_to_r=False, a=1, gamma_nr=0.4, min_w_par=0.4, W_r=1, f_max=1)

start = time.time()
m.run()
end = time.time()
print(end - start)

fig1, fig2 = plot_lm(m, m.t, 400, 1)
fig3 = get_wage_dist_fig(m)
fig4 = get_aggregate_regs(m, m.t, 300)

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


import seaborn as sns

fig, (ax1) = plt.subplots(1, 1, figsize=(15, 5))

ax1.clear()
ax1.grid()
sns.distplot(m.u_r_arr[150:500], kde=False, ax = ax1)
fig.show()


ax = sns.distplot(m.u_r_arr)
