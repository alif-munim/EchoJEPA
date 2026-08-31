"""Table generation for EchoBench results."""

from echobench.noise import NOISE_TYPES, SEVERITY_LEVELS


def _extract_table_data(results_list, metric="mae"):
    """Extract a structured table from a list of JSON result dicts (one per model).

    Returns:
        list of dicts, each with "model", "clean", noise_type/severity values, "avg_deg"
    """
    rows = []
    for result in results_list:
        model_name = result.get("meta", {}).get("model_name", "Unknown")
        conditions = {c["condition"]: c.get("metrics", {}) for c in result.get("conditions", [])}

        clean_val = conditions.get("clean", {}).get(metric)
        if clean_val is None:
            continue

        row = {"model": model_name, "clean": clean_val}
        noised_vals = []

        for nt in NOISE_TYPES:
            for sev in SEVERITY_LEVELS:
                key = f"{nt}/{sev}"
                val = conditions.get(key, {}).get(metric)
                row[key] = val
                if val is not None:
                    noised_vals.append(val)

        # Avg degradation (for MAE: higher = worse)
        if noised_vals and clean_val != 0:
            avg_noised = sum(noised_vals) / len(noised_vals)
            avg_deg = (avg_noised - clean_val) / abs(clean_val) * 100
            row["avg_deg"] = avg_deg
        else:
            row["avg_deg"] = None

        rows.append(row)
    return rows


def generate_markdown_table(results_list, metric="mae"):
    """Generate a Markdown table from multiple model results.

    Args:
        results_list: list of JSON result dicts (one per model)
        metric: metric key to tabulate (default: mae)

    Returns:
        str: Markdown-formatted table
    """
    rows = _extract_table_data(results_list, metric)
    if not rows:
        return "No data."

    # Build header
    header_parts = ["Model", "Clean"]
    for nt in NOISE_TYPES:
        for sev in SEVERITY_LEVELS:
            header_parts.append(f"{_short_name(nt)}/{sev[0].upper()}")
    header_parts.append("Avg Deg")

    header = "| " + " | ".join(header_parts) + " |"
    sep = "| " + " | ".join(["---"] * len(header_parts)) + " |"

    lines = [header, sep]
    for row in rows:
        parts = [row["model"], _fmt(row["clean"])]
        for nt in NOISE_TYPES:
            for sev in SEVERITY_LEVELS:
                parts.append(_fmt(row.get(f"{nt}/{sev}")))
        deg = row.get("avg_deg")
        parts.append(f"+{deg:.1f}%" if deg is not None else "N/A")
        lines.append("| " + " | ".join(parts) + " |")

    return "\n".join(lines)


def generate_latex_table(results_list, metric="mae"):
    """Generate a LaTeX table from multiple model results.

    Args:
        results_list: list of JSON result dicts (one per model)
        metric: metric key to tabulate (default: mae)

    Returns:
        str: LaTeX-formatted table
    """
    rows = _extract_table_data(results_list, metric)
    if not rows:
        return "% No data."

    n_noise = len(NOISE_TYPES)
    n_sev = len(SEVERITY_LEVELS)
    n_cols = 2 + n_noise * n_sev + 1  # model + clean + noise cols + avg deg

    lines = []
    lines.append("\\begin{table}[t]")
    lines.append(f"\\caption{{\\textbf{{EchoBench robustness}} ({metric.upper()}). "
                 "Avg.\\ Deg.\\ reports relative increase from clean performance.}}")
    lines.append("\\centering")

    col_spec = "lc|" + "|".join(["c" * n_sev] * n_noise) + "|c"
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\hline")

    # Header row 1: noise type groups
    header1_parts = [" ", " "]
    for nt in NOISE_TYPES:
        header1_parts.append(f"\\multicolumn{{{n_sev}}}{{c|}}{{\\textbf{{{_short_name(nt)}}}}}")
    header1_parts.append(" ")
    lines.append(" & ".join(header1_parts) + " \\\\")

    # Header row 2: severity levels
    header2_parts = ["\\textbf{Model}", "\\textbf{Clean}"]
    for _ in NOISE_TYPES:
        for sev in SEVERITY_LEVELS:
            header2_parts.append(f"\\textbf{{{sev.capitalize()}}}")
    header2_parts.append("\\textbf{Avg.\\ Deg.\\ $\\downarrow$}")
    lines.append(" & ".join(header2_parts) + " \\\\")
    lines.append("\\hline")

    # Data rows
    for row in rows:
        parts = [row["model"], _fmt(row["clean"])]
        for nt in NOISE_TYPES:
            for sev in SEVERITY_LEVELS:
                parts.append(_fmt(row.get(f"{nt}/{sev}")))
        deg = row.get("avg_deg")
        parts.append(f"+{deg:.1f}\\%" if deg is not None else "N/A")
        lines.append(" & ".join(parts) + " \\\\")

    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    return "\n".join(lines)


def _short_name(noise_type):
    """Abbreviate noise type names for table headers."""
    return {
        "depth_attenuation": "Depth Atten.",
        "gaussian_shadow": "Gauss. Shadow",
        "haze_artifact": "Haze",
        "speckle_reduction": "Speckle Red.",
    }.get(noise_type, noise_type)


def _fmt(val):
    """Format a numeric value for display."""
    if val is None:
        return "N/A"
    return f"{val:.2f}"
