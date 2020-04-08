from model_class import *
from plot_tool import plot_lm



T = 500
rd.seed(135)
m = Model(T=T, alpha_2=0.25, chi_C=0.4)

# initialize employment
set_W_fs(m.f_arr, m.h_arr)
initialize_emp(m.h_arr, m.f_arr, m.F, int(m.Nr), int(m.Nnr))

for t in range(m.T):
    m.step_function()
    dN_arr = np.array([(f.d_Nr, f.d_Nnr) for f in m.f_arr])
    Af_arr = np.array([f.A for f in m.f_arr])
    # print(dN_arr)
    if np.any(dN_arr < 0):
        print("some element of dN_arr < 0")
        break
    elif m.mean_w_arr[t] < 0:
        print("some element of m.mean_w_arr[t] < 0")
        break
    elif np.any(Af_arr < 0):
        print("some element of Af_arr < 0")
        break



f1, f2 = plot_lm(m, m.t, m.t)
f1.show()
f2.show()


hw_arr = np.array([h.w for h in m.h_arr])
print(hw_arr)

fp_arr = np.array([f.p for f in m.f_arr])
print(fp_arr)

fm_arr = np.array([f.m for f in m.f_arr])
print(fm_arr)

fuc_arr = np.array([f.uc_arr[m.t-1] for f in m.f_arr])
print(fuc_arr)

fy_arr = np.array([f.d_y for f in m.f_arr])
print(fy_arr)

dN_arr = np.array([(f.d_Nr, f.d_Nnr) for f in m.f_arr])
print(dN_arr)

fW_arr = np.array([(f.Wr, f.Wnr) for f in m.f_arr])
print(fW_arr)

fW_e_arr = np.array([(f.Wr_e, f.Wnr_e) for f in m.f_arr])
print(fW_e_arr)

Omega_arr = np.array([get_Omega(f.Wr_e, f.Wnr_e, m.mu_r, m.mu_nr, m.sigma)
                      for f in m.f_arr])

d_N_arr = np.array([get_d_Nnr_binding(f.A, f.Wr_e, f.Wnr_e, 2)
                   for f in m.f_arr])
print(d_N_arr)

print(Omega_arr)

print(m.mean_w_arr)