import numpy as np
import matplotlib.pyplot as plt


def plot_lm(m, T, periods, steps):

    t = T - periods

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig2, axs2 = plt.subplots(2, 2, figsize=(14, 10))

    time_array = np.arange(0, T)

    axs[0, 0].clear()
    axs[1, 0].clear()
    axs[0, 1].clear()
    axs[1, 1].clear()

    fontsize = 10

    # GDP plot
    axs[0, 0].grid()
    axs[0, 0].set_title("GDP", fontsize=fontsize)
    axs[0, 0].plot(time_array[t:T:steps], m.GDP[t:T:steps], marker="o", markersize=3, alpha=1, label="GDP")
    if m.shock_t > t:
        axs[0, 0].axvline(x=m.shock_t, color="red")

    # Unemployment plot
    axs[1, 0].grid()
    axs[1, 0].set_title("Unemployment rate", fontsize=fontsize)
    axs[1, 0].plot(time_array[t:T:steps], m.u_r_arr[t:T:steps], marker="o", markersize=3, alpha=0.5, label="Unemployment")
    if m.shock_t > t:
        axs[1, 0].axvline(x=m.shock_t, color="red")
    # axs[1, 0].set(ylim=(0, 0.25))

    # mean wages plot
    axs[0, 1].grid()
    axs[0, 1].set_title("Mean wages", fontsize=fontsize)
    axs[0, 1].plot(time_array[t:T:steps], m.mean_r_w_arr[t:T:steps], marker="o", markersize=2, alpha=1, label="Mean wages routine")
    axs[0, 1].plot(time_array[t:T:steps], m.mean_nr_w_arr[t:T:steps], marker="o", markersize=2, alpha=1,
                   label="Mean wages non-routine")
    axs[0, 1].legend(loc="best")
    if m.shock_t > t:
        axs[0, 1].axvline(x=m.shock_t, color="red")
    # axs[0, 1].set(ylim=(0, 0.25))

    # Beveridge curve
    axs[1, 1].grid()
    axs[1, 1].set_title("Beveridge curve", fontsize=fontsize)
    axs[1, 1].scatter(m.u_r_arr[t:T:steps], m.open_vs[t:T:steps], alpha=0.5)
    # axs[1, 1].set(xlim = (0, 0.25), ylim = (0, 80))


    # Log-differences mean real wages
    axs2[0, 0].grid()
    axs2[0, 0].set_title("log-diff mean real wages", fontsize=fontsize)
    log_diff_arr = np.log(m.mean_nr_w_arr[t:T:steps])-np.log(m.mean_r_w_arr[t:T:steps])
    axs2[0, 0].plot(time_array[t:T:steps], log_diff_arr, marker="o", markersize=2, alpha=1,
                   label="log-diff mean real wages", color = "k")
    if m.shock_t > t:
        axs2[0, 0].axvline(x=m.shock_t, color="red")
    # axs2[0].set(ylim=(-0.1, 1))

    # nr and r unemployment rates
    axs2[0, 1].grid()
    axs2[0, 1].set_title("Share of Nr in R jobs", fontsize=fontsize)
    axs2[0, 1].bar(np.arange(len(time_array[t:T:steps])), m.share_nr_in_r[t:T:steps], label="share r in nr")
    # axs2[0, 1].plot(time_array[t:T], m.share_nr_in_r[t:T], marker="o", markersize=2, alpha=1)
    # axs2[1].plot(time_array[t:T], m.unr_r_arr[t:T], marker="o", markersize=2, alpha=1,
    #              label="Unemployment rate nr-worker")
    if m.shock_t > t:
        axs2[0, 1].axvline(x=m.shock_t, color="red")
    # axs2[1].legend(loc="best")
    # axs2[1].set(ylim=(0, 0.25))

    axs2[1, 0].grid()
    axs2[1, 0].set_title("Mean prices$ ", fontsize=fontsize)
    axs2[1, 0].plot(time_array[t:T:steps], m.mean_p_arr[t:T:steps], marker="o",
                    markersize=2, alpha=1, color = "green")
    if m.shock_t > t:
        axs2[1, 0].axvline(x=m.shock_t, color="red")

    axs2[1, 1].grid()
    axs2[1, 1].set_title("default rate", fontsize=fontsize)
    axs2[1, 1].set_ylabel("share of refin. firms")
    ax3 = axs2[1, 1].twinx()
    ax3.set_ylabel("share of inactive firms", color = "orange")
    # axs2[1, 1].plot(time_array[t:T], m.n_refinanced[t:T]/m.F, alpha=1, label = "share of refinanced firms")
    width = 40
    axs2[1, 1].bar(time_array[t:T:steps], m.n_refinanced[t:T:steps]/m.F, label = "share of refinanced firms", color = "c")
    ax3.bar(time_array[t:T:steps], m.share_inactive[t:T:steps], 10, label="share of inactive firms", color="orange")
    # ax3.plot(time_array[t:T], m.share_inactive[t:T], alpha=1, label = "share of inactive firms", color = "orange")
    if m.shock_t > t:
        axs2[1, 1].axvline(x=m.shock_t, color="red")
    # axs2[1, 1].legend(loc="best")
    # axs2[1].set(ylim=(0, 0.25))


    return fig, fig2

