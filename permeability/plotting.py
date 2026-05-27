import numpy as np
import matplotlib.pyplot as plt

steps = np.array([40, 50, 60])
labels = ["30→40", "40→50", "50→60"]

# --- DATA ---
porosity = np.array([
    [1.9939, 0.1861, 1.0198],
    [2.2897, 0.9325, 0.4795],
    [2.4433, 0.7826, 0.6095],
    [0.8608, 2.5907, 1.2088],
    [6.9265, 0.0476, 0.4342],
    [6.8408, 2.4152, 0.3044],
    [4.2296, 1.1022, 1.0935],
    [1.6485, 0.1475, 0.3890],
    [5.6063, 0.6773, 0.1871],
    [2.7804, 2.6905, 0.4269],
    [1.9463, 0.1396, 0.1857],
    [0.2111, 1.7711, 0.6072],
])

K = np.array([
    [26.8955, 2.6547, 16.2423],
    [29.6657, 14.4018, 7.5251],
    [23.6574, 8.5695, 6.8054],
    [7.5189, 20.2426, 10.6915],
    [41.2190, 0.3282, 3.0484],
    [40.3999, 18.1243, 2.1437],
    [24.4720, 6.6851, 6.5391],
    [10.5104, 0.9777, 2.6072],
    [33.4990, 4.6678, 1.2807],
    [18.6505, 21.4133, 3.1447],
    [15.8465, 1.0809, 1.4540],
    [1.2256, 10.8935, 3.5473],
])

def plot_panel(ax, data, title):
    mean = data.mean(axis=0)
    std = data.std(axis=0)

    # individual REVs
    for i in range(data.shape[0]):
        ax.plot(steps, data[i], marker="o", alpha=0.35)

        # label at last point
        ax.text(
            steps[-1] + 0.5,
            data[i, -1],
            f"{i+1}",
            fontsize=8,
            alpha=0.7,
            verticalalignment="center"
        )

    # std hull
    ax.fill_between(steps, mean - std, mean + std, alpha=0.25)

    # mean line
    ax.plot(steps, mean, marker="o", linewidth=3)

    ax.set_xticks(steps)
    ax.set_xticklabels(labels)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)


# -----------------------
# SIDE-BY-SIDE PLOT
# -----------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

plot_panel(axes[0], porosity, "Porosity convergence")
plot_panel(axes[1], K, "Permeability K convergence")

axes[0].set_ylabel("Relative change [%]")

plt.tight_layout()
plt.show()