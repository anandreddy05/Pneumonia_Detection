
from torchvision.models import efficientnet_b0, efficientnet_b2, EfficientNet_B0_Weights, EfficientNet_B2_Weights
import torchvision
import torch
from torch import nn

def create_effnetb0(out_features: int):
    """Creates a EfficientNet-B0 model with custom classifier.
    
    Args:
        out_features: Number of output features for the classifier.
        
    Returns:
        A modified EfficientNet-B0 model with frozen feature extractor.
    """
    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=weights)
       
    for param in model.features.parameters():
        param.requires_grad = False
    
   
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=1280, out_features=out_features)
    )
    
    
    model.name = "effnetb0"
    print(f"[INFO] Created new {model.name} model.")
    return model

def create_effnetb2(out_features: int):
    """Creates a EfficientNet-B2 model with custom classifier.
    
    Args:
        out_features: Number of output features for the classifier.
        
    Returns:
        A modified EfficientNet-B2 model with frozen feature extractor.
    """
    weights = EfficientNet_B2_Weights.DEFAULT
    model = efficientnet_b2(weights=weights)
    
    for param in model.features.parameters():
        param.requires_grad = False

    
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),  
        nn.Linear(in_features=1408, out_features=out_features)
    )

    
    model.name = "effnetb2"
    print(f"[INFO] Created new {model.name} model.")
    return model
