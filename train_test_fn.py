from torch import nn, optim
import torch
import torchvision
from tqdm.auto import tqdm
from torch.utils.tensorboard.writer import SummaryWriter
from utils import EarlyStopper

def train_step(model: nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device = torch.device("cpu"),
               ):
    """ Turns a PyTorch model into training mode
    
    Args:
        model: A PyTorch model to be trained.
        dataloader: A DataLoader instance that our model will be trained on.
        loss_fn: A loss function used to calculate loss.
        optimizer: An optimizer used to minimize the loss.
        device: A target device to compute on (e.g. "cuda" or "cpu").
    Returns:
        A tuple of training loss and training accuracy metrics.
        In the form (train_loss, train_accuracy).
    """
    model.train()
    train_loss, train_acc = 0, 0
    
    for batch, (X, y) in tqdm(enumerate(dataloader), desc="Training", leave=False):
        X, y = X.to(device), y.to(device)
        
        y_logits = model(X)
        loss = loss_fn(y_logits, y)
        train_loss += loss.item()
        
        y_pred = torch.argmax(y_logits, dim=1)
        train_acc += (y_pred == y).sum().item() 
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    train_loss /= len(dataloader)
    train_acc /= len(dataloader.dataset)
    print(f"Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.3f}")
    return train_loss, train_acc 

def val_step(model: nn.Module,
             dataloader: torch.utils.data.DataLoader,
             loss_fn: nn.Module,
             device: torch.device = torch.device("cpu")
             ):
    """Evaluates model on validation data.
    
    Args:
        model: A PyTorch model to be evaluated.
        dataloader: A DataLoader instance for validation data.
        loss_fn: A loss function used to calculate loss.
        device: A target device to compute on (e.g. "cuda" or "cpu").
    Returns:
        A tuple of validation loss and validation accuracy metrics.
        In the form (val_loss, val_accuracy).
    """
    model.eval()
    val_loss, val_acc = 0, 0

    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            y_logits = model(X)
            loss = loss_fn(y_logits, y)
            val_loss += loss.item()
            y_pred = torch.argmax(y_logits, dim=1)
            val_acc += (y_pred == y).sum().item()

    val_loss /= len(dataloader)
    val_acc /= len(dataloader.dataset)
    print(f"Val Loss: {val_loss:.3f} | Val Acc: {val_acc:.3f}")
    return val_loss, val_acc


def test_step(model: nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: nn.Module,
              device: torch.device = torch.device("cpu")
              ):
    """ Evaluates a PyTorch model in testing mode.
    
    Args:
        model: A PyTorch model to be tested.
        dataloader: A DataLoader instance for the model to be tested on.
        loss_fn: A loss function used to calculate loss.
        device: A target device to compute on (e.g. "cuda" or "cpu").
    Returns:
        A tuple of test loss and test accuracy metrics.
        In the form (test_loss, test_accuracy).
    """
    model.eval()
    test_loss, test_acc = 0, 0
    
    with torch.inference_mode():
        for X, y in dataloader:  
            X, y = X.to(device), y.to(device)
            y_logits = model(X)
            loss = loss_fn(y_logits, y)  
            test_loss += loss.item()   
            
            y_pred = torch.argmax(y_logits, dim=1)
            test_acc += (y == y_pred).sum().item()
            
    test_loss /= len(dataloader) 
    test_acc /= len(dataloader.dataset)
    print(f"Test Loss: {test_loss:.3f} | Test Acc: {test_acc:.3f}")
    return test_loss, test_acc  

def train(num_epochs: int,
          model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          valid_dataloader: torch.utils.data.DataLoader,
          loss_fn: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          device: torch.device,
          writer: SummaryWriter = None,
          patience: int = 5,
          early_stopping: bool = True,
          test_dataloader: torch.utils.data.DataLoader = None
          ):
    """Trains and validates a PyTorch model.

    Passes a target PyTorch model through train_step() and val_step()
    functions for a number of epochs, training and validating the model
    in the same epoch loop.

    Calculates, prints and stores evaluation metrics throughout.
    Stores metrics to specified writer log_dir if present.

    Args:
        num_epochs: An integer indicating how many epochs to train for.
        model: A PyTorch model to be trained and tested.
        train_dataloader: A DataLoader instance for the model to be trained on.
        valid_dataloader: A DataLoader instance for validation data.
        loss_fn: A PyTorch loss function to calculate loss on datasets.
        optimizer: A PyTorch optimizer to help minimize the loss function.
        device: A target device to compute on (e.g. "cuda" or "cpu").
        writer: Optional SummaryWriter() instance to log model results to.

    Returns:
        A dictionary of training and validation loss as well as training and
        validation accuracy metrics. Each metric has a value in a list for 
        each epoch. If test_dataloader provided, includes final test metrics.
        In the form: {train_loss: [...], train_acc: [...], val_loss: [...], 
                     val_acc: [...]}
    """
    results = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [], 
        "val_acc": []
    }
    model = model.to(device)
    stopper = EarlyStopper(patience=patience, min_delta=0.0) if early_stopping else None
    best_val_loss = float("inf")
    best_model_wts = None
    
    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        print(f"Epoch: {epoch+1}/{num_epochs}")
        
        train_loss, train_acc = train_step(
            model=model,  
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device
        )
        
        val_loss, val_acc = val_step(
            model=model,
            dataloader=valid_dataloader,
            loss_fn=loss_fn,
            device=device
        )
        
        results['train_loss'].append(train_loss)
        results['train_acc'].append(train_acc)
        results['val_loss'].append(val_loss)
        results['val_acc'].append(val_acc)
        
        if writer:
            writer.add_scalars(main_tag="Loss", 
                               tag_scalar_dict={"train_loss": train_loss,
                                                "val_loss": val_loss},
                               global_step=epoch)
            writer.add_scalars(main_tag="Accuracy", 
                               tag_scalar_dict={"train_acc": train_acc,
                                                "val_acc": val_acc}, 
                               global_step=epoch)
            
            
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = model.state_dict().copy()
    
        if stopper and stopper.step(val_loss):
            print(f"⏹ Early stopping at epoch {epoch+1}")
            break
    
    if writer:
        writer.close()
    if best_model_wts:
        model.load_state_dict(best_model_wts)
    if test_dataloader:
        test_loss, test_acc = test_step(model=model, dataloader=test_dataloader,
                                        loss_fn=loss_fn, device=device)
        results["test_loss"] = test_loss
        results["test_acc"] = test_acc
        print(f"Final Test → Loss: {test_loss:.3f} | Acc: {test_acc:.3f}")
    print("Best model restored with val_loss:", best_val_loss)
    return results