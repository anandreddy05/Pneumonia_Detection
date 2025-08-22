import os
import shutil
import random
from pathlib import Path
from torch.utils.tensorboard.writer import SummaryWriter
import torch
from typing import Optional
import matplotlib.pyplot as plt
from typing import Dict, List

import shutil
import random
from pathlib import Path
from collections import defaultdict


def split_data(train_path: Path,
               val_path: Path,
               test_path: Path,
               split_percent: int,
               save_destination: Path,
               seed: int = 42):
    """
    Splits medical image dataset (NORMAL/PNEUMONIA) into train/val/test sets.

    Args:
        train_path (Path): Path to original training data directory
        val_path (Path): Path to original validation data directory  
        test_path (Path): Path to original test data directory
        split_percent (int): Percent of data to move from training set into validation/test
        save_destination (Path): Path where new split dataset will be saved
        seed (int): Random seed for reproducible splits

    Returns:
        dict: Count of images in each split:
        {
            'train': {'NORMAL': count, 'PNEUMONIA': count},
            'val': {'NORMAL': count, 'PNEUMONIA': count}, 
            'test': {'NORMAL': count, 'PNEUMONIA': count}
        }
    """
    random.seed(seed)
    save_destination = Path(save_destination)
    
    if save_destination.exists():
        shutil.rmtree(save_destination)
    save_destination.mkdir(parents=True, exist_ok=True)

    counts = {
        "train": defaultdict(int),
        "val": defaultdict(int),
        "test": defaultdict(int)
    }

    original_splits = {
        "train": train_path,
        "val": val_path,
        "test": test_path
    }

    for split, split_dir in original_splits.items():
        for class_dir in ["NORMAL", "PNEUMONIA"]:
            src_dir = Path(split_dir) / class_dir
            images = list(src_dir.glob("*.jpeg")) + list(src_dir.glob("*.jpg")) + list(src_dir.glob("*.png"))
            
            # Shuffle for reproducibility
            random.shuffle(images)
            
            if split == "train":
                # Apply split percent (into val & test)
                split_size = int(len(images) * split_percent / 100)
                val_images = images[:split_size // 2]
                test_images = images[split_size // 2:split_size]
                train_images = images[split_size:]

                split_map = {
                    "train": train_images,
                    "val": val_images,
                    "test": test_images
                }
            else:
                # For val & test, just copy as is
                split_map = {split: images}
            
            # Copy images to new destination
            for dst_split, img_list in split_map.items():
                dst_dir = save_destination / dst_split / class_dir
                dst_dir.mkdir(parents=True, exist_ok=True)

                for img_path in img_list:
                    shutil.copy(img_path, dst_dir)
                    counts[dst_split][class_dir] += 1

    return counts





def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
    """Saves a PyTorch model to a target directory.

    Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
      either ".pth" or ".pt" as the file extension.

    Example usage:
    save_model(model=model_0,
               target_dir="models",
               model_name="05_going_modular_tingvgg_model.pth")
    """
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,
                        exist_ok=True)

    # Create model save path
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(),
             f=model_save_path)
    

def create_summary_writer(experiment_name: str,
                          model_name: str,
                          extra: Optional[str]=None):
    """Creates a torch.utils.tensorboard.writer.SummaryWriter() instance saving to a specific log_dir.

    log_dir is a combination of runs/timestamp/experiment_name/model_name/extra.

    Where timestamp is the current date in YYYY-MM-DD format.

    Args:
        experiment_name (str): Name of experiment.
        model_name (str): Name of model.
        extra (str, optional): Anything extra to add to the directory. Defaults to None.

    Returns:
        torch.utils.tensorboard.writer.SummaryWriter(): Instance of a writer saving to log_dir.

    Example usage:
        # Create a writer saving to "runs/2022-06-04/data_10_percent/effnetb2/5_epochs/"
        writer = create_writer(experiment_name="data_10_percent",
                               model_name="effnetb2",
                               extra="5_epochs")
        # The above is the same as:
        writer = SummaryWriter(log_dir="runs/2022-06-04/data_10_percent/effnetb2/5_epochs/")
    """
    from datetime import datetime
    import os
    
    timestamp = datetime.now().strftime("%Y-%m-%d")
    
    if extra:
        log_dir = os.path.join("runs",timestamp,experiment_name,model_name,extra)
    else:
        log_dir = os.path.join("runs",timestamp,experiment_name,model_name)
    print(f"[INFO] Created SummaryWriter, saving to: {log_dir}...")
    return SummaryWriter(log_dir=log_dir)

class EarlyStopper:
    def __init__(self, patience=5, min_delta=0.0):
        """
        Args:
            patience (int): how many epochs to wait after last improvement.
            min_delta (float): minimum change to qualify as improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0

    def step(self, current_loss: float) -> bool:
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.counter = 0
            return False  # don't stop
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True  # stop training
            return False



import matplotlib.pyplot as plt
from typing import Dict, List

def plot_loss_curves(results: Dict[str, List[float]], save_path: str):
    """Plots saves training/validation curves.

    Args:
        results: Dictionary containing train/val (and optionally test) metrics.
        save_path: saves the plot to this file (e.g. "plots/loss_curves.png").
    """
    loss = results["train_loss"]
    val_loss = results["val_loss"]

    train_acc = results["train_acc"]
    val_acc = results["val_acc"]

    epochs = range(len(results["train_loss"]))

    plt.figure(figsize=(15, 4))

    # --- Loss Plot ---
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, val_loss, label="val_loss")
    if "test_loss" in results and results["test_loss"] is not None:
        plt.axhline(results["test_loss"], color="r", linestyle="--", label="test_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # --- Accuracy Plot ---
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, label="train_acc")
    plt.plot(epochs, val_acc, label="val_acc")
    if "test_acc" in results and results["test_acc"] is not None:
        plt.axhline(results["test_acc"], color="g", linestyle="--", label="test_acc")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()

    plt.tight_layout()

    # Save or Show
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"📊 Plot saved to {save_path}")
    else:
        plt.show()
