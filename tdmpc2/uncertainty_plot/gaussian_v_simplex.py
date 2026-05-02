import numpy as np
import matplotlib.pyplot as plt

def plot_2d_vs_extended_simplex():
    # Set up the figure with two side-by-side subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # ==========================================
    # Plot 1: 2D Gaussian Distribution
    # ==========================================
    mean = [0, 0]
    cov = [[1, 0], [0, 1]]
    x_g, y_g = np.random.multivariate_normal(mean, cov, 2000).T

    ax1.scatter(x_g, y_g, alpha=0.3, color='blue', s=10, zorder=2)
    # Added pad=30 to push the title up and away from the y-axis arrow
    #ax1.set_title('2D Gaussian', pad=30, fontsize=24)

    ax1.set_xlim(-4, 4)
    ax1.set_ylim(-4, 4)

    # ==========================================
    # Plot 2: Extended "Simplex" Line (Affine Hull)
    # ==========================================
    x_s = np.random.uniform(-0.5, 1.5, 2000)
    y_s = 1 - x_s

    ax2.scatter(x_s, y_s, alpha=0.3, color='red', s=10, zorder=2)

    # Draw the solid line from x=-0.5 to x=1.5
    ax2.plot([-0.5, 1.5], [1.5, -0.5], color='darkred', linewidth=2, zorder=3)

    # Added pad=30 here as well for symmetry and spacing
    #ax2.set_title('2D Simplex: $x + y = 1$', pad=30, fontsize=24)

    ax2.set_xlim(-0.5, 1.5)
    ax2.set_ylim(-0.5, 1.5)

    # ==========================================
    # Custom Axis Formatting for Both Plots
    # ==========================================
    for ax in [ax1, ax2]:
        # Remove tick marks and numbers
        ax.set_xticks([])
        ax.set_yticks([])

        # Hide the default boundary box (spines)
        for spine in ['top', 'right', 'bottom', 'left']:
            ax.spines[spine].set_visible(False)

        # Get the limits of the plot to draw the arrows appropriately
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()

        # Draw prominent X and Y Cartesian axes with double-ended arrows
        arrow_props = dict(arrowstyle="<|-|>", color='black', lw=3, shrinkA=0, shrinkB=0)

        # X-axis
        ax.annotate('', xy=(xmax, 0), xytext=(xmin, 0), arrowprops=arrow_props, zorder=4)
        # Y-axis
        ax.annotate('', xy=(0, ymax), xytext=(0, ymin), arrowprops=arrow_props, zorder=4)

        # Place the axis labels relative to the arrow tips
        # x-label: exactly at xmax horizontally, shifted 10 points UP
        ax.annotate('$x$', xy=(xmax, 0), xytext=(0, 10), textcoords='offset points',
                    fontsize=24, va='bottom', ha='center')

        # y-label: exactly at ymax vertically, shifted 10 points RIGHT
        ax.annotate('$y$', xy=(0, ymax), xytext=(10, 0), textcoords='offset points',
                    fontsize=24, va='center', ha='left')

        # Draw a faint grid to keep spatial context
        ax.grid(True, linestyle='--', alpha=0.3, zorder=1)

    # Render
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_2d_vs_extended_simplex()
