import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import pandas as pd

def create_figure_and_axes():
    fig, ax = plt.subplots(figsize=(10, 8))
    return fig, ax

def set_up_plot(ax):
    ax.set_xlabel('X', fontsize=14)
    ax.set_ylabel('Y', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)

def plot_magnitude(X, Y, data, title, filename, levels=70):
    fig, ax = create_figure_and_axes()
    triang = tri.Triangulation(X, Y)
    tpc = plt.tricontourf(triang, data, levels=levels)  # Retirer l'argument shading
    plt.colorbar(tpc)
    set_up_plot(ax)
    plt.jet()
    plt.title(title, fontweight="bold")
    plt.savefig(filename)
    plt.close(fig)  # close figure after saving


def plot_side_by_side(X, Y, u_real, u_pred, error, t, levels=70):
    """ Affiche la solution réelle et la solution prédite côte à côte """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 8))
    
    # Plot solution réelle
    plot_on_given_axis(X, Y, u_real, f'U ground truth @ t={t} seconds', ax1, levels)
    
    # Plot solution prédite
    plot_on_given_axis(X, Y, u_pred, f'U predicted with PINN @ t={t} seconds', ax2, levels)
    
    # Ajustement des niveaux pour l'erreur
    error_levels = np.linspace(0, error.max(), levels)
    
    # Plot erreur absolue
    plot_on_given_axis(X, Y, error, f'Absolute Error @ t={t} seconds', ax3, error_levels)
    
    plt.tight_layout()
    plt.savefig(f"/notebooks/pictures/Comparison_t{t}.png")
    plt.close(fig)

def plot_on_given_axis(X, Y, data, title, ax, levels=70):
    """ Trace la magnitude de U sur un axe donné """
    plt.jet()  # Palette de couleurs
    triang = tri.Triangulation(X, Y)
    tpc = ax.tricontourf(triang, data, levels=levels)
    plt.colorbar(tpc, ax=ax)
    ax.set_title(title, fontweight="bold")

