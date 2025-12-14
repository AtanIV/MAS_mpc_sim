# cbf_penalty_tuner.py
# Interactive plot for: phi(h) = w * sigmoid(kappa * (h_thresh - h)) * (eps / (h + eps))**p

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, CheckButtons

def phi_inverse(h, h_thresh=0.2, kappa=8.0, eps=0.1, p=2.0, w=1.0):
    """Inverse-power penalty with gating (sigmoid)."""
    gate = 1.0 / (1.0 + np.exp(-kappa * (h_thresh - h)))  # ~1 if h < h_thresh
    core = (eps / (h + eps)) ** p                          # steep near h -> 0
    return w * gate * core

def make_plot():
    # --- initial params ---
    params = dict(h_thresh=0.2, kappa=8.0, eps=0.1, p=2.0, w=1.0)
    h = np.linspace(0.0, 1.0, 1000)

    # --- figure & axes ---
    plt.close("all")
    fig, ax = plt.subplots(figsize=(9, 5))
    plt.subplots_adjust(left=0.08, right=0.98, top=0.90, bottom=0.28)

    y = phi_inverse(h, **params)
    [line] = ax.plot(h, y, lw=2, label=r'$\phi(h)$')
    vline = ax.axvline(params["h_thresh"], ls="--", lw=1.5, label="h_thresh")
    ax.set_xlabel("h")
    ax.set_ylabel("phi(h)")
    ax.set_title("CBF Penalty: Inverse-Power with Gating")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    # sensible initial y-limits (can be large near 0)
    ax.set_ylim(0.0, np.nanpercentile(y, 99.5) * 1.1)

    # --- sliders area ---
    axcolor = "lightgoldenrodyellow"
    s_hth = Slider(plt.axes([0.10, 0.20, 0.80, 0.03], facecolor=axcolor),
                   "h_thresh", 0.0, 1.0, valinit=params["h_thresh"], valstep=0.001)
    s_kap = Slider(plt.axes([0.10, 0.16, 0.80, 0.03], facecolor=axcolor),
                   "kappa", 1.0, 100.0, valinit=params["kappa"], valstep=0.5)
    s_eps = Slider(plt.axes([0.10, 0.12, 0.80, 0.03], facecolor=axcolor),
                   "eps", 1e-4, 0.2, valinit=params["eps"], valstep=1e-4)
    s_p   = Slider(plt.axes([0.10, 0.08, 0.80, 0.03], facecolor=axcolor),
                   "p", 1.0, 10.0, valinit=params["p"], valstep=0.1)
    s_w   = Slider(plt.axes([0.10, 0.04, 0.80, 0.03], facecolor=axcolor),
                   "w", 0.0, 10.0, valinit=params["w"], valstep=0.1)

    # --- checkbox to toggle log y-scale ---
    cb = CheckButtons(plt.axes([0.015, 0.04, 0.07, 0.10]), labels=["log y"], actives=[False])

    def update(_=None):
        params["h_thresh"] = s_hth.val
        params["kappa"]    = s_kap.val
        params["eps"]      = s_eps.val
        params["p"]        = s_p.val
        params["w"]        = s_w.val

        y_new = phi_inverse(h, **params)
        line.set_ydata(y_new)
        vline.set_xdata([params["h_thresh"], params["h_thresh"]])

        # refresh y-limits unless log scale is on (to avoid jolting)
        if ax.get_yscale() == "linear":
            ymax = np.nanpercentile(y_new, 99.5)
            ymax = 1.0 if not np.isfinite(ymax) or ymax <= 0 else ymax * 1.1
            ax.set_ylim(0.0, ymax)

        fig.canvas.draw_idle()

    for s in (s_hth, s_kap, s_eps, s_p, s_w):
        s.on_changed(update)

    def on_check(label):
        ax.set_yscale("log" if ax.get_yscale() == "linear" else "linear")
        update()

    cb.on_clicked(on_check)

    # reset button
    reset_ax = plt.axes([0.90, 0.905, 0.08, 0.05])
    reset_btn = Button(reset_ax, "Reset", color=axcolor, hovercolor="0.9")

    def reset(_):
        s_hth.reset(); s_kap.reset(); s_eps.reset(); s_p.reset(); s_w.reset()
        if ax.get_yscale() != "linear":
            ax.set_yscale("linear")
        update()

    reset_btn.on_clicked(reset)

    plt.show()

if __name__ == "__main__":
    make_plot()
