import timm
import os

# 1. Define the model name
model_name = "convnext_base.fb_in22k_ft_in1k_384"

# 2. Trigger the download (pretrained=True fetches the weights)
# This will cache the file locally.
print(f"Downloading {model_name}...")
model = timm.create_model(model_name, pretrained=True)
print("Download complete.")

# 3. Find where it saved the file (Optional, useful for verifying)
# Timm usually saves to ~/.cache/huggingface/hub/models--timm--...
print("Check your ~/.cache/huggingface/hub/ directory.")