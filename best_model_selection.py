from train_test_fn import train
from pre_trained_models import create_effnetb0, create_effnetb2
from dataloader import create_dataloaders
from utils import create_summary_writer
from pathlib import Path
from torchvision import transforms
from utils import save_model
import torch
import os

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # List of dataset folders you want to train on
    dataset_folders = [
        Path("data_split_10"),
        Path("data_split_20")
    ]

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                     std=[0.229, 0.224, 0.225]) 

    simple_transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(), 
        normalize 
    ])

    # Define models
    models = {
        "effnet_b0_model": create_effnetb0,
        "effnet_b2_model": create_effnetb2
    }

    NUM_EPOCHS = [5, 10]

    for dataset_path in dataset_folders:
        print(f"\n===== Training on dataset: {dataset_path} =====")

        train_path = dataset_path / "train"
        val_path = dataset_path / "val"
        test_path = dataset_path / "test"

        # Create dataloaders
        train_dataloader, valid_dataloader, test_dataloader, class_names = create_dataloaders(
            train_path=train_path,
            test_path=test_path,
            valid_path=val_path,
            batch_size=32,
            transform_data=simple_transform
        )

        for name, model_fn in models.items():
            for epoch in NUM_EPOCHS:
                
                model = model_fn(out_features=len(class_names))

                # TensorBoard writer (log by dataset, model, and epochs)
                writer = create_summary_writer(
                    experiment_name=f"{dataset_path.name}_{name}_{epoch}",
                    model_name=name
                )

                # Train
                results = train(
                    model=model,
                    num_epochs=epoch,
                    train_dataloader=train_dataloader,
                    valid_dataloader=valid_dataloader,
                    loss_fn=torch.nn.CrossEntropyLoss(),
                    optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
                    device=device,
                    writer=writer
                )

                # Save model
                os.makedirs("models", exist_ok=True)
                save_filepath = f"{dataset_path.name}_{name}_{epoch}_epochs.pth"
                save_model(model=model, target_dir="models", model_name=save_filepath)

                print("-"*60 + "\n")
        print("================ x Completed x ===================")
        
# BEST MODEL => 2025-08-22\data_split_20_effnet_b0_model_5\effnet_b0_model