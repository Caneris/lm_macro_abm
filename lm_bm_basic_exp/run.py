from model_class import *
from plot_tool import plot_lm



T = 500
rd.seed(135)
m = Model(T=T, alpha_2=0.1, chi_C=0.4)

# initialize employment
initialize_emp(m.h_arr, m.f_arr, m.F, int(m.Nr), int(m.Nnr))

set_W_fs(m.f_arr, m.h_arr)
val = 0

f1, f2 = plot_lm(m, 500, 500)
f1.show()
f2.show()