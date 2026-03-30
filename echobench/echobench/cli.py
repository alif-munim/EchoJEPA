"""EchoBench command-line interface."""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="echobench",
        description="Acoustic robustness benchmark for echocardiography foundation models",
    )
    subparsers = parser.add_subparsers(dest="command")

    # --- evaluate ---
    eval_parser = subparsers.add_parser(
        "evaluate", help="Run robustness evaluation on a model"
    )
    eval_parser.add_argument("--adapter", required=True, choices=["echojepa", "videomae"],
                             help="Encoder adapter name")
    eval_parser.add_argument("--checkpoint", required=True, help="Path to encoder checkpoint")
    eval_parser.add_argument("--probe", required=True, help="Path to probe checkpoint")
    eval_parser.add_argument("--task", default="lvef", choices=["lvef"],
                             help="Evaluation task (default: lvef)")
    eval_parser.add_argument("--task-type", default="regression",
                             choices=["regression", "classification"],
                             help="Task type (default: regression)")
    eval_parser.add_argument("--data-csv", required=True,
                             help="Path to test CSV (space-delimited: path label)")
    eval_parser.add_argument("--output", default=None, help="Output JSON path")
    eval_parser.add_argument("--device", default="cuda", help="Device (default: cuda)")
    eval_parser.add_argument("--max-cases", type=int, default=None,
                             help="Limit number of test videos per condition")
    eval_parser.add_argument("--noise-types", nargs="+", default=None,
                             help="Noise types to evaluate (default: all)")
    eval_parser.add_argument("--severity-levels", nargs="+", default=None,
                             help="Severity levels to evaluate (default: all)")
    eval_parser.add_argument("--resolution", type=int, default=224,
                             help="Input resolution (default: 224)")
    eval_parser.add_argument("--frames", type=int, default=16,
                             help="Frames per clip (default: 16)")
    eval_parser.add_argument("--frame-step", type=int, default=2,
                             help="Frame sampling stride (default: 2)")
    # EchoJEPA-specific
    eval_parser.add_argument("--model-name", default="vit_large",
                             help="ViT variant for echojepa adapter (default: vit_large)")
    eval_parser.add_argument("--checkpoint-key", default="target_encoder",
                             help="Checkpoint dict key for echojepa (default: target_encoder)")

    # --- report ---
    report_parser = subparsers.add_parser(
        "report", help="Generate tables/plots from result JSON files"
    )
    report_parser.add_argument("results", nargs="+", help="JSON result files (one per model)")
    report_parser.add_argument("--format", default="markdown", choices=["markdown", "latex", "csv"],
                               help="Output format (default: markdown)")
    report_parser.add_argument("--metric", default="mae", help="Primary metric (default: mae)")
    report_parser.add_argument("--output", default=None, help="Output file path")

    # --- list ---
    subparsers.add_parser("list-noise", help="List available noise types and severity levels")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "evaluate":
        _run_evaluate(args)
    elif args.command == "report":
        _run_report(args)
    elif args.command == "list-noise":
        _run_list_noise()


def _run_evaluate(args):
    import torch

    from echobench.adapters import get_adapter
    from echobench.evaluate import evaluate
    from echobench.tasks.lvef import LVEFTask

    # Build adapter
    adapter_kwargs = dict(
        checkpoint=args.checkpoint,
        device=args.device,
        resolution=args.resolution,
        frames=args.frames,
    )
    if args.adapter == "echojepa":
        adapter_kwargs["model_name"] = args.model_name
        adapter_kwargs["checkpoint_key"] = args.checkpoint_key

    print(f"Loading {args.adapter} adapter...")
    adapter = get_adapter(args.adapter, **adapter_kwargs)

    # Build task
    if args.task == "lvef":
        task = LVEFTask(
            data_csv=args.data_csv,
            probe_checkpoint=args.probe,
            task_type=args.task_type,
            resolution=args.resolution,
            frames=args.frames,
            frame_step=args.frame_step,
        )

    # Run
    with torch.no_grad():
        evaluate(
            adapter=adapter,
            task=task,
            noise_types=args.noise_types,
            severity_levels=args.severity_levels,
            device=args.device,
            max_cases=args.max_cases,
            output_path=args.output,
        )


def _run_report(args):
    import json

    from echobench.reporting.tables import generate_latex_table, generate_markdown_table

    results = []
    for path in args.results:
        with open(path) as f:
            results.append(json.load(f))

    if args.format == "markdown":
        table = generate_markdown_table(results, metric=args.metric)
    elif args.format == "latex":
        table = generate_latex_table(results, metric=args.metric)
    else:
        print("CSV output not yet implemented.")
        return

    if args.output:
        with open(args.output, "w") as f:
            f.write(table)
        print(f"Table saved to {args.output}")
    else:
        print(table)


def _run_list_noise():
    from echobench.noise import NOISE_TYPES, PERTURBATIONS, SEVERITY_LEVELS

    print("Noise types:")
    for nt in NOISE_TYPES:
        print(f"  {nt}")
        for sev in SEVERITY_LEVELS:
            params = PERTURBATIONS[nt][sev]
            param_str = ", ".join(f"{k}={v}" for k, v in params.items())
            print(f"    {sev}: {param_str}")
    print(f"\nSeverity levels: {SEVERITY_LEVELS}")


if __name__ == "__main__":
    main()
