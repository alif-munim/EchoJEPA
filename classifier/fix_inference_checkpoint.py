import torch
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Fix checkpoint keys by removing 'module.' prefix.")
    parser.add_argument(
        "--input", 
        type=str, 
        required=True, 
        help="Path to the input .pt checkpoint file"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default=None, 
        help="Path to save the fixed checkpoint. If not provided, appends '_fixed' to the input filename."
    )

    args = parser.parse_args()

    # Determine output path if not provided
    if args.output is None:
        base, ext = os.path.splitext(args.input)
        args.output = f"{base}_fixed{ext}"

    print(f"Loading checkpoint from: {args.input}")
    checkpoint = torch.load(args.input, map_location="cpu") # map_location="cpu" prevents GPU errors on non-GPU nodes

    # Remove "module." prefix from classifier weights
    if "classifiers" in checkpoint:
        fixed_classifiers = []
        for classifier_dict in checkpoint["classifiers"]:
            fixed_dict = {k.replace("module.", ""): v for k, v in classifier_dict.items()}
            fixed_classifiers.append(fixed_dict)
        
        checkpoint["classifiers"] = fixed_classifiers
        print("Successfully removed 'module.' prefixes.")
    else:
        print("Warning: Key 'classifiers' not found in checkpoint.")

    print(f"Saving fixed checkpoint to: {args.output}")
    torch.save(checkpoint, args.output)

if __name__ == "__main__":
    main()