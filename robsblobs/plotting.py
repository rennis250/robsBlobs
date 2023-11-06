import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np

from .monitor import Monitor
from .cie_standard import plank_lasso
from .uv_space import plank_lasso_uvY
from .infamous_lab import lab2rgb

def xyY_fig(mon: Monitor):
    fig, ax = plt.subplots(1, 1, figsize = (7, 7))
    plt.grid()

    ax.plot(plank_lasso[:, 0], plank_lasso[:, 1], color="black")

    ax.plot([mon.monxyY[0, 0], mon.monxyY[1, 0]], [mon.monxyY[0, 1], mon.monxyY[1, 1]], color="black")
    ax.plot([mon.monxyY[1, 0], mon.monxyY[2, 0]], [mon.monxyY[1, 1], mon.monxyY[2, 1]], color="black")
    ax.plot([mon.monxyY[0, 0], mon.monxyY[2, 0]], [mon.monxyY[0, 1], mon.monxyY[2, 1]], color="black")
    ax.scatter(mon.monxyY[0, 0], mon.monxyY[0, 1], color="red")
    ax.scatter(mon.monxyY[1, 0], mon.monxyY[1, 1], color="green")
    ax.scatter(mon.monxyY[2, 0], mon.monxyY[2, 1], color="blue")

    ax.set_xlim((0, 0.8))
    ax.set_ylim((0, 0.9))

    ax.set_xlabel('x')
    ax.set_ylabel('y')

    return fig, ax


def AB_fig(mon: Monitor, xlims, ylims):
    fig, ax = plt.subplots(1, 1, figsize = (7, 7))
    plt.grid()

    plt.plot(np.array([0, 0]), np.array([ylims[0], ylims[1]]), color="black")
    plt.plot(np.array([xlims[0], xlims[1]]), np.array([0, 0]), color="black")

    ax.set_xlim(xlims)
    ax.set_ylim(ylims)

    ax.set_xlabel('a*')
    ax.set_ylabel('b*')

    return fig, ax


def LC_fig(mon: Monitor, xlims, ylims):
    fig, ax = plt.subplots(1, 1, figsize = (7, 7))
    plt.grid()

    ax.set_xlim(xlims)
    ax.set_ylim(ylims)

    ax.set_xlabel('Chroma')
    ax.set_ylabel('L*')

    return fig, ax


def uvY_fig(mon: Monitor):
    fig, ax = plt.subplots(1, 1, figsize = (7, 7))
    plt.grid()

    ax.plot(plank_lasso_uvY[:, 0], plank_lasso_uvY[:, 1], color="black")

    # ax.plot([mon.monuvY[0, 0], mon.monuvY[1, 0]], [mon.monuvY[0, 1], mon.monuvY[1, 1]], color="black", linestyle="dashed")
    # ax.plot([mon.monuvY[1, 0], mon.monuvY[2, 0]], [mon.monuvY[1, 1], mon.monuvY[2, 1]], color="black", linestyle="dashed")
    # ax.plot([mon.monuvY[0, 0], mon.monuvY[2, 0]], [mon.monuvY[0, 1], mon.monuvY[2, 1]], color="black", linestyle="dashed")
    # ax.scatter(mon.monuvY[0, 0], mon.monuvY[0, 1], color="red")
    # ax.scatter(mon.monuvY[1, 0], mon.monuvY[1, 1], color="green")
    # ax.scatter(mon.monuvY[2, 0], mon.monuvY[2, 1], color="blue")

    ax.set_xlim((0, 0.6))
    ax.set_ylim((0, 0.7))

    ax.set_xlabel('u\'')
    ax.set_ylabel('v\'')

    return fig, ax


def add_hue_plot(mon: Monitor, fig, subspec):
    position_major = [0, np.pi/6, np.pi/3, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4]
    labels = ["Red", "Pink", "Orange", "Yellow", "Chartreuse", "Green", "Cyan", "Blue", "Purple"]

    # rgbs = np.zeros((len(labels), 3))
    # for c in range(len(labels)):
        # l = 50
        # a = 80*np.cos(position_major[c])
        # b = 80*np.sin(position_major[c])
        # rgb = lab2rgb(mon, np.array([l, a, b]))
        # rgbs[c, :] = np.sqrt(rgb)

    ax = fig.add_subplot(subspec, projection='polar')
    ax.grid(True)
    ax.axis(True)
    ax.spines["polar"].set_visible(False)
    ax.xaxis.set_major_locator(ticker.FixedLocator(position_major))
    ax.xaxis.set_major_formatter(ticker.FixedFormatter(labels))
    ax.set_rgrids([0, 1], ["" for _ in range(len([0, 1]))], fontsize=16)
    ax.set_ylim(0, 1.12)
    ax.set_yticklabels([])

    # ax.scatter(position_major, np.ones((len(position_major))), c=rgbs, s=80)

    return ax