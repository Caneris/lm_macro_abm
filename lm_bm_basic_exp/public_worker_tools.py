import numpy as np

def update_pw_w(pw_arr, h_arr, gov, H_pw, t):
    u_arr = np.array([h.u[t] for h in h_arr])
    mask = u_arr < 1

    emp_wages = np.array([h.w for h in h_arr[mask]])
    mean_w = np.mean(emp_wages)
    gov.W_pw = mean_w*H_pw
    gov.debt = mean_w*H_pw

    for pw in pw_arr:
        if t>0:
            pw.w = mean_w
        else:
            pw.w = mean_w
            pw_e = mean_w