from model_class import *
from plot_tool import plot_lm



T = 200
rd.seed(135)
m = Model(T=T, alpha_2=0.1, chi_C=0.4)

# initialize employment
initialize_emp(m.h_arr, m.f_arr, m.F, int(m.Nr), int(m.Nnr))

for t in range(m.T):
    m.step_function()

set_W_fs(m.f_arr, m.h_arr)
val = 0

f1, f2 = plot_lm(m, m.T, m.T)
f1.show()
f2.show()