# Chest X-Ray Pneumonia Detection

A deep learning project for automated pneumonia detection from chest X-ray images using PyTorch and pre-trained EfficientNet models.

## 🎯 Project Overview

This project implements a computer vision solution to classify chest X-ray images as either NORMAL or PNEUMONIA using transfer learning with EfficientNet architectures. The system includes comprehensive model selection, training pipelines, and evaluation metrics.

## 🏗️ Project Structure

```
chest-xray-pneumonia-detection/
├── data/
│   └── chest_xray/          # Original dataset
│       ├── train/
│       ├── val/
│       └── test/
├── data_split_10/           # 10% data split
├── data_split_20/           # 20% data split
├── models/                  # Saved trained models
├── best_model/             # Best performing model
├── plots/                  # Training curves and visualizations
├── runs/                   # TensorBoard logs
├── src/
│   ├── best_model.py           # Final model training script
│   ├── best_model_selection.py # Model comparison and selection
│   ├── dataloader.py           # Data loading utilities
│   ├── pre_trained_models.py   # Model architectures
│   ├── train_test_fn.py        # Training/testing functions
│   └── utils.py                # Utility functions
├── .gitignore
├── requirements.txt
└── README.md
```

## 📋 Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Core Dependencies
- `torch` - PyTorch deep learning framework
- `torchvision` - Computer vision utilities
- `tensorboard` - Training visualization
- `matplotlib` - Plotting and visualization
- `tqdm` - Progress bars
- `pathlib` - Path handling

## 🚀 Quick Start

### 1. Data Preparation

Place your chest X-ray dataset in the following structure:
```
data/chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

### 2. Model Selection and Training

Run the model selection pipeline to compare different architectures:

```bash
python best_model_selection.py
```

This script will:
- Train EfficientNet-B0 and EfficientNet-B2 models
- Test on different data splits (10% and 20%)
- Evaluate with different epoch counts (5 and 10)
- Save all models and TensorBoard logs

### 3. Final Model Training

Train the best performing model with extended epochs:

```bash
python best_model.py
```

This will:
- Load the best model from selection phase
- Train for 40 epochs with early stopping
- Save the final model and generate loss curves

## 🧠 Model Architectures

### EfficientNet-B0
- **Input Size**: 224×224×3
- **Feature Extraction**: Frozen pre-trained layers
- **Classifier**: Dropout(0.2) + Linear(1280 → num_classes)

### EfficientNet-B2
- **Input Size**: 224×224×3
- **Feature Extraction**: Frozen pre-trained layers
- **Classifier**: Dropout(0.3) + Linear(1408 → num_classes)

Both models use ImageNet pre-trained weights for transfer learning.

## 🔧 Key Features

### Data Handling
- **Flexible Data Splits**: Custom data splitting with configurable percentages
- **Image Preprocessing**: Standardized resizing, normalization using ImageNet statistics
- **Efficient Loading**: Multi-worker DataLoader with proper shuffling

### Training Features
- **Early Stopping**: Prevents overfitting with configurable patience
- **Best Model Checkpointing**: Automatically saves best performing weights
- **TensorBoard Integration**: Real-time training monitoring
- **Progress Tracking**: Visual progress bars with tqdm

### Evaluation & Monitoring
- **Comprehensive Metrics**: Loss and accuracy for train/validation/test
- **Loss Curve Plotting**: Automatic generation of training visualizations
- **Model Comparison**: Systematic evaluation across architectures and hyperparameters


## 📁 File Descriptions

| File | Purpose |
|------|---------|
| `best_model_selection.py` | Systematic model comparison across architectures and hyperparameters |
| `best_model.py` | Final training of the best performing model with extended epochs |
| `dataloader.py` | Data loading utilities and DataLoader creation |
| `pre_trained_models.py` | EfficientNet model definitions with custom classifiers |
| `train_test_fn.py` | Core training, validation, and testing functions |
| `utils.py` | Utility functions for data splitting, model saving, visualization |

## 🔍 Monitoring Training

### TensorBoard
Launch TensorBoard to monitor training progress:
```bash
tensorboard --logdir=runs
```

### Generated Plots
Training curves are automatically saved to `plots/` directory showing:
- Training vs Validation Loss
- Training vs Validation Accuracy
- Test performance (if available)

## 🎛️ Customization

### Custom Data Splits
Use the `split_data()` function in `utils.py` to create custom train/val/test splits.

- Check existing issues in the repository
- Create a new issue with detailed description
- Include system specifications and error logs
