import torch  
  
checkpoint = torch.load("/home/sagemaker-user/user-default-efs/vjepa2/evals/vitg-384/classifier/video_classification_frozen/uhn22k-classifier-fs2-ns2-nvs1-echojepa/latest.pt")  
  
# Remove "module." prefix from classifier weights  
fixed_classifiers = []  
for classifier_dict in checkpoint["classifiers"]:  
    fixed_dict = {k.replace("module.", ""): v for k, v in classifier_dict.items()}  
    fixed_classifiers.append(fixed_dict)  
  
checkpoint["classifiers"] = fixed_classifiers  
torch.save(checkpoint, "/home/sagemaker-user/user-default-efs/vjepa2/evals/vitg-384/classifier/video_classification_frozen/uhn22k-classifier-fs2-ns2-nvs1-echojepa/latest_fixed.pt")