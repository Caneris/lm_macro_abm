import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_lm(m, T, periods, steps):

    t = T - periods

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig2, axs2 = plt.subplots(2, 2, figsize=(12, 8))

    time_array = np.arange(0, T)

    axs[0, 0].clear()
    axs[1, 0].clear()
    axs[0, 1].clear()
    axs[1, 1].clear()

    fontsize = 15

    # GDP plot
    axs[0, 0].grid()
    axs[0, 0].set_title("GDP", fontsize=fontsize)
    axs[0, 0].plot(time_array[t:T:steps], m.GDP[t:T:steps], marker="o", markersize=3, alpha=1, label="GDP")

    # Unemployment plot
    axs[1, 0].grid()
    axs[1, 0].set_title("Unemployment rate", fontsize=fontsize)
    axs[1, 0].plot(time_array[t:T:steps], m.u_r_arr[t:T:steps], marker="o", markersize=3, alpha=0.5, label="Unemployment")

    # mean wages plot
    axs[0, 1].grid()
    axs[0, 1].set_title("Mean wages", fontsize=fontsize)
    axs[0, 1].plot(time_array[t:T:steps], m.mean_r_w_arr[t:T:steps], marker="o", markersize=2, alpha=1, label="Mean wages routine")
    axs[0, 1].plot(time_array[t:T:steps], m.mean_nr_w_arr[t:T:steps], marker="o", markersize=2, alpha=1,
                   label="Mean wages non-routine")
    axs[0, 1].legend(loc="best")

    # Beveridge curve
    axs[1, 1].grid()
    axs[1, 1].set_title("Beveridge curve", fontsize=fontsize)
    axs[1, 1].scatter(m.u_r_arr[t:T:steps], m.open_vs[t:T:steps], alpha=0.5)
    # axs[1, 1].set(xlim = (0, 0.25), ylim = (0, 80))


    # Log-differences mean real wages
    axs2[0, 0].grid()
    axs2[0, 0].set_title("Variance of Wages", fontsize=fontsize)
    log_diff_arr = np.log(m.mean_nr_w_arr[t:T:steps])-np.log(m.mean_r_w_arr[t:T:steps])
    axs2[0, 0].plot(time_array[t:T:steps], log_diff_arr, marker="o", markersize=2, alpha=1,
                   label="log-diff mean real wages", color = "k")

    # nr and r unemployment rates
    axs2[0, 1].grid()
    axs2[0, 1].set_title("Decile Comparison", fontsize=fontsize)
    axs2[0, 1].plot(time_array[t:T:steps], m.nine_to_five[t:T:steps], label="9/5")
    axs2[0, 1].plot(time_array[t:T:steps], m.five_to_one[t:T:steps], label="5/1")
    axs2[0, 1].plot(time_array[t:T:steps], m.nine_to_one[t:T:steps], label="9/1")
    axs2[0, 1].legend(loc="best")
    axs2[0, 1].set_ylabel("(decile) ratio of income")

    axs2[1, 0].grid()
    axs2[1, 0].set_title("Mean prices", fontsize=fontsize)
    axs2[1, 0].plot(time_array[t:T:steps], m.mean_p_arr[t:T:steps], marker="o",
                    markersize=2, alpha=1, color = "green")
    axs2[0, 1].set_ylabel("mean price")

    axs2[1, 1].grid()
    axs2[1, 1].set_title("default rate", fontsize=fontsize)
    axs2[1, 1].set_ylabel("share of refin. firms", color = "c")
    ax3 = axs2[1, 1].twinx()
    ax3.set_ylabel("share of inactive firms", color = "orange")
    # axs2[1, 1].plot(time_array[t:T], m.n_refinanced[t:T]/m.F, alpha=1, label = "share of refinanced firms")
    width = 40
    axs2[1, 1].bar(time_array[t:T:steps], m.n_refinanced[t:T:steps]/m.F, label = "share of refinanced firms", color = "c")
    ax3.bar(time_array[t:T:steps], m.share_inactive[t:T:steps], 10, label="share of inactive firms", color="orange", alpha = 0.3)
    # ax3.plot(time_array[t:T], m.share_inactive[t:T], alpha=1, label = "share of inactive firms", color = "orange")


    return fig, fig2

def get_wage_dist_fig(m):

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    ax1.clear()
    ax2.clear()
    ax3.clear()
    ax1.grid()
    ax2.grid()
    ax3.grid()

    fontsize = 15
    # look only at employed wages
    wages = np.array([h.w for h in m.h_arr])
    wages = wages[wages > 0]

    # wages of routine worker
    r_wages = np.array([h.w for h in m.h_arr[m.routine_arr]])
    r_wages = r_wages[r_wages > 0]

    # wages of non-routine worker
    nr_wages = np.array([h.w for h in m.h_arr[m.non_routine_arr]])
    nr_wages = nr_wages[nr_wages > 0]

    # norm_wages = (1/np.max(wages))*wages
    sns.distplot(wages, kde=False, ax = ax1)
    sns.distplot(r_wages, kde=False, ax = ax2)
    sns.distplot(nr_wages, kde=False, ax = ax3)

    # titles
    ax1.set_title("Distribution of the wages of all workers", fontsize=fontsize)
    ax2.set_title("Distribution of the wages of routine workers", fontsize=fontsize)
    ax3.set_title("Distribution of the wages of non-routine workers", fontsize=fontsize)

    return fig

def get_aggregate_regs(m, T, periods):

    t = T - periods

    GDP_growth = [m.GDP[i] / m.GDP[i-1] - 1 if i > 0 else 0 for i in range(m.T)]
    GDP_growth = GDP_growth[1:]

    unemp_growth = [np.exp(m.u_r_arr[i]-m.u_r_arr[i-1]) - 1 if i > 0 else 0 for i in range(m.T)]
    unemp_growth = unemp_growth[1:]

    wage_rate = [m.mean_nominal_w_arr[i]/m.mean_nominal_w_arr[i-1] - 1
                 if i > 0 else 0 for i in range(m.T)]

    wage_level = [m.mean_nominal_w_arr[i] for i in range(m.T)]
    # wage_rate = wage_rate[1:]


    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    ax1.clear()
    ax2.clear()
    ax3.clear()
    ax1.grid()
    ax2.grid()
    ax3.grid()

    fontsize = 15

    # Beveridge curve
    ax1.set_title("Beveridge curve", fontsize=fontsize)
    ax1.scatter(m.u_r_arr[t:T], m.open_vs[t:T], alpha=0.5, color = "red")
    ax1.set_ylabel("open vacancies")
    ax1.set_ylabel("unemployment rate")

    # Wage curve
    ax2.set_title("Wage curve", fontsize=fontsize)
    ax2.scatter(m.u_r_arr[t:T], wage_level[t:T], alpha=0.5, color = "orange")
    ax2.set_xlabel("unemployment rate")
    ax2.set_ylabel("wage level")

    # Okun curve
    ax3.set_title("Okun curve", fontsize=fontsize)
    ax3.scatter(unemp_growth[t:T], GDP_growth[t:T], alpha=0.5, color = "green")
    ax3.set_ylabel("(real) GDP growth")
    ax3.set_xlabel("unemployment growth")

    return fig