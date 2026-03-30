"""Degradation curve plots for EchoBench results."""

from echobench.noise import NOISE_TYPES, SEVERITY_LEVELS


def plot_degradation_curves(results_list, metric="mae", output_path=None):
    """
    Plot degradation curves: one subplot per noise type, one line per model.

    Args:
        results_list: list of JSON result dicts (one per model, with "meta.model_name")
        metric: metric key to plot (default: mae)
        output_path: path to save figure (optional; shows interactively if None)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for plotting. Install with: pip install echobench[reporting]")

    fig, axes = plt.subplots(1, len(NOISE_TYPES), figsize=(4 * len(NOISE_TYPES), 4), sharey=True)
    if len(NOISE_TYPES) == 1:
        axes = [axes]

    x_labels = ["Clean"] + [s.capitalize() for s in SEVERITY_LEVELS]

    for ax, nt in zip(axes, NOISE_TYPES):
        for result in results_list:
            model_name = result.get("meta", {}).get("model_name", "Unknown")
            conditions = {c["condition"]: c.get("metrics", {}) for c in result.get("conditions", [])}

            clean_val = conditions.get("clean", {}).get(metric)
            if clean_val is None:
                continue

            y_vals = [clean_val]
            for sev in SEVERITY_LEVELS:
                val = conditions.get(f"{nt}/{sev}", {}).get(metric)
                y_vals.append(val if val is not None else float("nan"))

            ax.plot(range(len(x_labels)), y_vals, "o-", label=model_name, markersize=4)

        ax.set_title(_short_name(nt), fontsize=11, fontweight="bold")
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel(metric.upper(), fontsize=11)
    axes[-1].legend(fontsize=8, loc="upper left")

    fig.suptitle("EchoBench Degradation Curves", fontsize=13, fontweight="bold")
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"Plot saved to {output_path}")
    else:
        plt.show()

    plt.close(fig)


def _short_name(noise_type):
    return {
        "depth_attenuation": "Depth Attenuation",
        "gaussian_shadow": "Gaussian Shadow",
        "haze_artifact": "Haze Artifact",
        "speckle_reduction": "Speckle Reduction",
    }.get(noise_type, noise_type)
