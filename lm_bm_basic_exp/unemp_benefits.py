import numpy as np
import numpy.random as rd


def hhs_pay_income_tax(h_arr, tau, t):
    tax_arr = np.zeros(len(h_arr))
    for h in h_arr:
        income = h.w*(h.u[t] == 0) + h.div
        tax = tau*income
        h.income_tax = tax
        tax_arr[h.id] = tax
    return tax_arr


def firms_pay_income_tax(f_arr, tau):
    tax_arr = np.zeros(len(f_arr))
    for f in f_arr:
        pi = f.pos_pi
        if pi > 0:
            income = pi
        else:
            income = 0
        tax = tau * income
        f.income_tax = tax
        tax_arr[f.id] = tax
    return tax_arr


def gov_collects_income_taxes(gov, h_tax_arr, f_tax_arr):
    gov.tax_income = np.sum(h_tax_arr) + np.sum(f_tax_arr)


def get_benefits(h_arr, period, psi):
    xi_func = lambda xi, period: np.floor(xi / period) + 1
    benefits = np.array([(psi**xi_func(h.xi, period))*h.last_w*(h.xi > 0) for h in h_arr])
    return benefits

# fehlerhaft, bitte korrigieren!
def gov_decides_for_benefits(gov, h_arr, period, psi, t):
    resources = gov.A + gov.tax_income
    benefits = get_benefits(h_arr, period, psi)
    tot_benefits = np.sum(benefits)

    if tot_benefits > 0:
        par = np.minimum(resources / tot_benefits, 1)
    else:
        par = 0
    benefits = par*benefits
    gov.paid_benefits = np.sum(benefits)

    for h in h_arr:
        if h.u[t] == 1:
            h.w = benefits[h.id]


def update_AG(gov):
    gov.A += gov.tax_income - gov.debt


def update_xi(h_arr, t):
    for h in h_arr:
        if h.u[t] == 1:
            h.xi += 1
        else:
            h.xi = 0


