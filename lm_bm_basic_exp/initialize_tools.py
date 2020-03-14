from numpy import maximum
import numpy.random as rd


# initial wealth ditributions for
# firms, banks and households

init_Af = lambda : maximum(0.1, rd.randn() * 1 + 3)
init_Ai = lambda : maximum(0.01, rd.randn() * 0.01 + 0.5)


# initial price for firms
p_init = lambda: maximum(0.01, rd.randn() * 0.01 + 0.2)
# initial desire wage by households
w_init = lambda : 0.5
d_w_init = lambda : maximum(0.5, rd.randn() * 0.01 + 0.5)
# initial sales
s_init = lambda : 5
# initial expected sales
s_e_init = lambda : maximum(0.5, rd.randn() * 0.1 + 5)