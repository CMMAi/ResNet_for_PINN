import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib import cm


def plot2D_Error(x, t, y_true, y_pred,U_test , u_predict1):
    X, T = x, t
    error = abs(y_true - y_pred)
    max_error = torch.max(torch.abs(U_test - u_predict1)).item()
    plt.figure()
    cp = plt.contourf(T, X, error, 30, cmap='brg',
                      linewidths=1, antialiased=True, alpha=0.6, vmax=max_error)
    colorbar = plt.colorbar(cp)
    colorbar.ax.tick_params(labelsize=20)
    plt.xlabel('x', fontsize=16)
    plt.ylabel('y', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title('Error Profile', fontsize=16)
    plt.tight_layout()
    plt.show()

    
    
    
def plot2D_Matrix(x, t, y):
    X, T = x, t
    F_xt = y
    plt.figure()
    cp = plt.contourf(T, X, F_xt, 30, cmap='brg',
                      linewidths=1, antialiased=True, alpha=0.6)
    # plt.contour(T, X, F_xt, 30, colors='w', linewidths=0.4)  # Add contour lines in black
    colorbar = plt.colorbar(cp)
    colorbar.ax.tick_params(labelsize=18)
    plt.xlabel('x', fontsize=16)
    plt.ylabel('y', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title('with skip connection', fontsize=20)
    plt.tight_layout()
    plt.show()

def plot3D_Matrix(x, t, y):
    X, T = x, t
    F_xt = y
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(T, X, F_xt, cmap=cm.brg,
                    linewidth=0, antialiased=False, alpha=1)
    ax.set_xlabel('x', fontsize=16)
    ax.set_ylabel('y', fontsize=16)
    # ax.set_zlabel('F', fontsize=16)
    ax.set_zlabel('F', fontsize=16, labelpad=15)  # Adjust labelpad to position the label above z-axis tick labels
    ax.tick_params(axis='z', labelsize=16)  # Set font size of z-axis tick labels
    ax.tick_params(axis='both', which='major', labelsize=16)  # Set font size of x and y-axis tick labels
    plt.title('with skip connection', fontsize=20)
    plt.tight_layout()