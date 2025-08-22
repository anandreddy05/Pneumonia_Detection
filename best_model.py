from dataloader import create_dataloaders
from utils import save_model,plot_loss_curves
from train_test_fn import train,test_step
from pre_trained_models import create_effnetb0

import torch
import torchvision
from torchvision import transforms
from torch import nn,optim
from pathlib import Path
from tqdm.auto import tqdm
import os


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    data_path = Path("data/chest_xray")
    train_path = data_path/"train"
    test_path = data_path/"test"
    val_path = data_path/"val"
    best_model_path = './models/data_split_20_effnet_b0_model_5_epochs.pth'


    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                        std=[0.229, 0.224, 0.225]) 

    simple_transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(), 
        normalize 
    ])

    # Load Data and Create Dataloaders

    train_loader,val_loader,test_loader,class_names = create_dataloaders(train_path=train_path,
                                                test_path=test_path,
                                                valid_path=val_path,
                                                transform_data=simple_transform,
                                                )

    # Load Best Chosen Model
    best_model = create_effnetb0(out_features=len(class_names))
    best_model.load_state_dict(torch.load(best_model_path))

    effnet_b0_model = Path(best_model_path).stat().st_size // (1024*1024)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(best_model.parameters(),lr=0.0001)

    results = train(
        num_epochs=40,
        model=best_model,
        train_dataloader=train_loader,
        valid_dataloader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        patience=5,
        early_stopping=True,
        test_dataloader=test_loader
    )

    os.makedirs("best_model",exist_ok=True)
    save_model(model=best_model,target_dir="best_model",model_name="effnet_b0_fully_trained.pth")

    loss_fn = nn.CrossEntropyLoss()
    test_loss,test_acc = test_step(model=best_model.to(device),
                                dataloader=test_loader,
                                loss_fn=loss_fn,
                                device=device)
    
    os.makedirs("plots", exist_ok=True)
    plot_loss_curves(results, save_path="plots/effnet_b0_loss_curves.png")

    print(f"Final Test Performance → Loss: {test_loss:.3f} | Acc: {test_acc:.3f}")