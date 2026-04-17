"""
Plot training loss vs temporal encoding strength for MAE and JEPA.
Two-panel figure showing both models' losses alongside temporal encoding.
The smoking gun: both losses evolve, but only MAE's temporal encoding collapses.

Usage:
    python scripts/neurips/plot_recon_vs_temporal.py
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams.update({
    "font.size": 10,
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# --- MAE Data ---
# Reconstruction loss from training logs
mae_epochs_loss = [0, 4, 9, 14, 19, 24, 29, 34, 39, 44, 49, 54,
                   59, 64, 69, 74, 79, 84, 89, 94, 99, 104, 109, 114]
mae_recon_loss = [0.4900, 0.3474, 0.2892, 0.2022, 0.1758, 0.1628, 0.1544, 0.1487,
                  0.1448, 0.1429, 0.1413, 0.1388, 0.1389, 0.1368, 0.1360, 0.1345,
                  0.1333, 0.1324, 0.1312, 0.1307, 0.1301, 0.1297, 0.1296, 0.1290]

# Temporal encoding strength (clean R² - shuffled R²)
mae_epochs_temporal = [24, 50, 74, 99]
mae_temporal = [0.0454, 0.4431, 0.0601, 0.0164]

# --- JEPA Data ---
# Latent prediction loss (note: non-monotonic due to improving EMA teacher)
jepa_epochs_loss = list(range(1, 101))
jepa_latent_loss = [
    0.509, 0.388, 0.409, 0.427, 0.441, 0.451, 0.460, 0.466, 0.463, 0.466,
    0.464, 0.464, 0.465, 0.461, 0.459, 0.460, 0.461, 0.461, 0.461, 0.460,
    0.462, 0.465, 0.462, 0.466, 0.466, 0.470, 0.470, 0.472, 0.475, 0.476,
    0.484, 0.478, 0.480, 0.481, 0.483, 0.483, 0.485, 0.490, 0.490, 0.490,
    0.487, 0.487, 0.485, 0.484, 0.481, 0.478, 0.474, 0.468, 0.466, 0.464,
    0.461, 0.461, 0.460, 0.457, 0.459, 0.455, 0.454, 0.454, 0.454, 0.455,
    0.454, 0.456, 0.456, 0.454, 0.456, 0.457, 0.458, 0.459, 0.460, 0.461,
    0.462, 0.461, 0.460, 0.460, 0.461, 0.458, 0.460, 0.458, 0.461, 0.461,
    0.461, 0.462, 0.462, 0.464, 0.464, 0.464, 0.465, 0.464, 0.467, 0.467,
    0.461, 0.488, 0.495, 0.494, 0.490, 0.480, 0.475, 0.474, 0.472, 0.474,
]

# Temporal encoding strength
jepa_epochs_temporal = [25, 50, 75, 100]
jepa_temporal = [0.0517, 0.2124, 0.1662, 0.1028]

# --- Colors ---
COLOR_MAE = "#E67E22"       # orange
COLOR_JEPA = "#2E86C1"      # blue
COLOR_LOSS_MAE = "#B0B0B0"  # light gray for MAE loss
COLOR_LOSS_JEPA = "#A0C4E0" # light blue for JEPA loss

# --- Figure: two panels ---
fig, (ax_mae, ax_jepa) = plt.subplots(1, 2, figsize=(12, 4.5))

# ============ MAE Panel ============
ax1 = ax_mae
ax1.plot(mae_epochs_loss, mae_recon_loss, color=COLOR_LOSS_MAE, linewidth=2,
         marker="o", markersize=2.5, label="Pixel reconstruction loss", zorder=2)
ax1.set_xlabel("Training epoch", fontsize=11)
ax1.set_ylabel("Training loss", color="#666666", fontsize=11)
ax1.tick_params(axis="y", labelcolor="#666666")
ax1.set_ylim(0.10, 0.52)
ax1.axvspan(50, 99, alpha=0.05, color=COLOR_MAE, zorder=0)

ax1r = ax1.twinx()
ax1r.plot(mae_epochs_temporal, mae_temporal, color=COLOR_MAE, linewidth=2.5,
          marker="s", markersize=9, label="Temporal encoding\nstrength ($\\Delta R^2$)", zorder=3)
ax1r.set_ylabel("Temporal encoding strength\n(clean $R^2$ $-$ shuffled $R^2$)",
                color=COLOR_MAE, fontsize=10)
ax1r.tick_params(axis="y", labelcolor=COLOR_MAE)
ax1r.set_ylim(-0.02, 0.50)
ax1r.spines["right"].set_visible(True)
ax1r.spines["right"].set_color(COLOR_MAE)

ax1r.annotate("Peak temporal\nencoding",
              xy=(50, 0.443), xytext=(65, 0.40),
              fontsize=8.5, color=COLOR_MAE, fontweight="bold",
              arrowprops=dict(arrowstyle="->", color=COLOR_MAE, lw=1.3), ha="left")
ax1r.annotate("Temporal shortcut\ncomplete",
              xy=(99, 0.016), xytext=(78, 0.13),
              fontsize=8.5, color=COLOR_MAE,
              arrowprops=dict(arrowstyle="->", color=COLOR_MAE, lw=1.3), ha="center")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1r.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=8.5,
           framealpha=0.9, edgecolor="#CCCCCC")
ax1.set_xlim(-2, 118)
ax1.set_xticks([0, 25, 50, 75, 100])
ax1.set_title("MAE (pixel reconstruction)", fontsize=12, fontweight="bold", color=COLOR_MAE, pad=10)

# ============ JEPA Panel ============
ax2 = ax_jepa
ax2.plot(jepa_epochs_loss, jepa_latent_loss, color=COLOR_LOSS_JEPA, linewidth=2,
         marker="o", markersize=1.5, label="Latent prediction loss", zorder=2)
ax2.set_xlabel("Training epoch", fontsize=11)
ax2.set_ylabel("Training loss", color="#666666", fontsize=11)
ax2.tick_params(axis="y", labelcolor="#666666")
ax2.set_ylim(0.35, 0.52)

ax2r = ax2.twinx()
ax2r.plot(jepa_epochs_temporal, jepa_temporal, color=COLOR_JEPA, linewidth=2.5,
          marker="s", markersize=9, label="Temporal encoding\nstrength ($\\Delta R^2$)", zorder=3)
ax2r.set_ylabel("Temporal encoding strength\n(clean $R^2$ $-$ shuffled $R^2$)",
                color=COLOR_JEPA, fontsize=10)
ax2r.tick_params(axis="y", labelcolor=COLOR_JEPA)
ax2r.set_ylim(-0.02, 0.50)
ax2r.spines["right"].set_visible(True)
ax2r.spines["right"].set_color(COLOR_JEPA)

ax2r.annotate("Peak temporal\nencoding",
              xy=(50, 0.212), xytext=(62, 0.35),
              fontsize=8.5, color=COLOR_JEPA, fontweight="bold",
              arrowprops=dict(arrowstyle="->", color=COLOR_JEPA, lw=1.3), ha="left")
ax2r.annotate("Consolidation:\ntemporal features\ncompressed, not lost",
              xy=(100, 0.103), xytext=(60, 0.07),
              fontsize=8, color=COLOR_JEPA, style="italic",
              arrowprops=dict(arrowstyle="->", color=COLOR_JEPA, lw=1.3), ha="center")

lines3, labels3 = ax2.get_legend_handles_labels()
lines4, labels4 = ax2r.get_legend_handles_labels()
ax2.legend(lines3 + lines4, labels3 + labels4, loc="upper right", fontsize=8.5,
           framealpha=0.9, edgecolor="#CCCCCC")
ax2.set_xlim(-2, 104)
ax2.set_xticks([0, 25, 50, 75, 100])
ax2.set_title("JEPA (latent prediction)", fontsize=12, fontweight="bold", color=COLOR_JEPA, pad=10)

plt.tight_layout(w_pad=3)

# Save
outpath = "uhn_echo/nature_medicine/assets/neurips/recon_vs_temporal.png"
fig.savefig(outpath, dpi=300, bbox_inches="tight", facecolor="white")
print(f"Saved to {outpath}")

outpath_pdf = outpath.replace(".png", ".pdf")
fig.savefig(outpath_pdf, bbox_inches="tight", facecolor="white")
print(f"Saved to {outpath_pdf}")

plt.close()
