"""
Variant of plot_recon_vs_temporal.py: plots clean R² and fully-shuffled R² as
separate lines instead of the difference (temporal Δ).

Data: Protocol C (severity gradient, fraction 0.00 vs 1.00, 3-seed mean).
See claude/neurips/experiments/severity-gradient.md.
"""

import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    "font.size": 10,
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# --- MAE ---
mae_epochs_loss = [0, 4, 9, 14, 19, 24, 29, 34, 39, 44, 49, 54,
                   59, 64, 69, 74, 79, 84, 89, 94, 99, 104, 109, 114]
mae_recon_loss = [0.4900, 0.3474, 0.2892, 0.2022, 0.1758, 0.1628, 0.1544, 0.1487,
                  0.1448, 0.1429, 0.1413, 0.1388, 0.1389, 0.1368, 0.1360, 0.1345,
                  0.1333, 0.1324, 0.1312, 0.1307, 0.1301, 0.1297, 0.1296, 0.1290]

mae_epochs_r2 = [24, 50, 74, 99]
mae_clean    = [0.221,  0.141, 0.390, 0.445]
mae_shuffled = [0.176, -0.301, 0.330, 0.428]

# --- JEPA ---
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

jepa_epochs_r2 = [25, 50, 75, 100]
jepa_clean    = [0.383, 0.503, 0.537, 0.591]
jepa_shuffled = [0.331, 0.290, 0.370, 0.488]

# --- Colors ---
COLOR_MAE = "#E67E22"
COLOR_JEPA = "#2E86C1"
COLOR_LOSS_MAE = "#B0B0B0"
COLOR_LOSS_JEPA = "#A0C4E0"

fig, (ax_mae, ax_jepa) = plt.subplots(1, 2, figsize=(12, 4.5))

# ============ MAE ============
ax_mae.plot(mae_epochs_loss, mae_recon_loss, color=COLOR_LOSS_MAE, linewidth=2,
            marker="o", markersize=2.5, label="Pixel reconstruction loss", zorder=2)
ax_mae.set_xlabel("Training epoch", fontsize=11)
ax_mae.set_ylabel("Training loss", color="#666666", fontsize=11)
ax_mae.tick_params(axis="y", labelcolor="#666666")
ax_mae.set_ylim(0.10, 0.52)
ax_mae.axvspan(50, 99, alpha=0.05, color=COLOR_MAE, zorder=0)

ax_mae_r = ax_mae.twinx()
ax_mae_r.plot(mae_epochs_r2, mae_clean, color=COLOR_MAE, linewidth=2.5,
              marker="s", markersize=9, label="Clean $R^2$", zorder=4)
ax_mae_r.plot(mae_epochs_r2, mae_shuffled, color=COLOR_MAE, linewidth=2.5,
              marker="o", markersize=9, linestyle="--", fillstyle="none",
              markeredgewidth=2, label="Shuffled $R^2$", zorder=3)
ax_mae_r.axhline(0, color="#AAAAAA", linewidth=0.7, linestyle=":", zorder=1)
ax_mae_r.set_ylabel("LVEF $R^2$ (EchoNet-Dynamic)",
                    color=COLOR_MAE, fontsize=10)
ax_mae_r.tick_params(axis="y", labelcolor=COLOR_MAE)
ax_mae_r.set_ylim(-0.40, 0.65)
ax_mae_r.spines["right"].set_visible(True)
ax_mae_r.spines["right"].set_color(COLOR_MAE)

ax_mae_r.annotate("Clean and shuffled\nconverge: shortcut",
                  xy=(99, 0.437), xytext=(70, 0.25),
                  fontsize=8.5, color=COLOR_MAE,
                  arrowprops=dict(arrowstyle="->", color=COLOR_MAE, lw=1.3),
                  ha="center")
ax_mae_r.annotate("Gap = $-$0.44\n(shuffle collapses)",
                  xy=(50, -0.15), xytext=(18, -0.30),
                  fontsize=8.5, color=COLOR_MAE, fontweight="bold",
                  arrowprops=dict(arrowstyle="->", color=COLOR_MAE, lw=1.3),
                  ha="left")

lines1, labels1 = ax_mae.get_legend_handles_labels()
lines2, labels2 = ax_mae_r.get_legend_handles_labels()
ax_mae.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8.5,
              framealpha=0.9, edgecolor="#CCCCCC", bbox_to_anchor=(0.01, 0.98))
ax_mae.set_xlim(-2, 118)
ax_mae.set_xticks([0, 25, 50, 75, 100])
ax_mae.set_title("MAE (pixel reconstruction)", fontsize=12, fontweight="bold",
                 color=COLOR_MAE, pad=10)

# ============ JEPA ============
ax_jepa.plot(jepa_epochs_loss, jepa_latent_loss, color=COLOR_LOSS_JEPA, linewidth=2,
             marker="o", markersize=1.5, label="Latent prediction loss", zorder=2)
ax_jepa.set_xlabel("Training epoch", fontsize=11)
ax_jepa.set_ylabel("Training loss", color="#666666", fontsize=11)
ax_jepa.tick_params(axis="y", labelcolor="#666666")
ax_jepa.set_ylim(0.35, 0.52)

ax_jepa_r = ax_jepa.twinx()
ax_jepa_r.plot(jepa_epochs_r2, jepa_clean, color=COLOR_JEPA, linewidth=2.5,
               marker="s", markersize=9, label="Clean $R^2$", zorder=4)
ax_jepa_r.plot(jepa_epochs_r2, jepa_shuffled, color=COLOR_JEPA, linewidth=2.5,
               marker="o", markersize=9, linestyle="--", fillstyle="none",
               markeredgewidth=2, label="Shuffled $R^2$", zorder=3)
ax_jepa_r.set_ylabel("LVEF $R^2$ (EchoNet-Dynamic)",
                     color=COLOR_JEPA, fontsize=10)
ax_jepa_r.tick_params(axis="y", labelcolor=COLOR_JEPA)
ax_jepa_r.set_ylim(0.20, 0.65)
ax_jepa_r.spines["right"].set_visible(True)
ax_jepa_r.spines["right"].set_color(COLOR_JEPA)

ax_jepa_r.annotate("Gap preserved:\nconsolidation, not abandonment",
                   xy=(100, 0.54), xytext=(35, 0.60),
                   fontsize=8.5, color=COLOR_JEPA, style="italic",
                   arrowprops=dict(arrowstyle="->", color=COLOR_JEPA, lw=1.3),
                   ha="left")
ax_jepa_r.annotate("Shuffled $R^2$ at e100\n$\\approx$ BYOL clean (0.468)",
                   xy=(100, 0.488), xytext=(55, 0.30),
                   fontsize=8, color=COLOR_JEPA,
                   arrowprops=dict(arrowstyle="->", color=COLOR_JEPA, lw=1.3),
                   ha="center")

lines3, labels3 = ax_jepa.get_legend_handles_labels()
lines4, labels4 = ax_jepa_r.get_legend_handles_labels()
ax_jepa.legend(lines3 + lines4, labels3 + labels4, loc="lower right", fontsize=8.5,
               framealpha=0.9, edgecolor="#CCCCCC")
ax_jepa.set_xlim(-2, 104)
ax_jepa.set_xticks([0, 25, 50, 75, 100])
ax_jepa.set_title("JEPA (latent prediction)", fontsize=12, fontweight="bold",
                  color=COLOR_JEPA, pad=10)

plt.tight_layout(w_pad=3)

outpath = "uhn_echo/nature_medicine/assets/neurips/recon_vs_clean_shuffled.png"
fig.savefig(outpath, dpi=300, bbox_inches="tight", facecolor="white")
print(f"Saved to {outpath}")

outpath_pdf = outpath.replace(".png", ".pdf")
fig.savefig(outpath_pdf, bbox_inches="tight", facecolor="white")
print(f"Saved to {outpath_pdf}")

plt.close()
