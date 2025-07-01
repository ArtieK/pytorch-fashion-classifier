# PyTorch Fashion Classifier

A neural network implementation for classifying fashion items using the Fashion-MNIST dataset. Built with PyTorch, this project demonstrates feedforward neural networks for multi-class image classification.

## Overview

This project implements two neural network architectures to classify grayscale images of clothing items into 10 categories. The Fashion-MNIST dataset contains 60,000 training images and 10,000 test images, each 28x28 pixels.

### Fashion Categories
- T-shirt/top
- Trouser
- Pullover
- Dress
- Coat
- Sandal
- Shirt
- Sneaker
- Bag
- Ankle Boot

## Features

- **Dual Architecture Support**: Basic and deeper network architectures for comparison
- **Training Pipeline**: Complete training loop with SGD optimization
- **Model Evaluation**: Comprehensive evaluation metrics including loss and accuracy
- **Prediction Interface**: Single-image prediction with top-k confidence scores
- **Data Handling**: Automated dataset download and preprocessing

## Architecture

### Basic Model
- **Input Layer**: 784 nodes (28×28 flattened)
- **Hidden Layer 1**: 128 nodes with ReLU activation
- **Hidden Layer 2**: 64 nodes with ReLU activation
- **Output Layer**: 10 nodes (one per class)

### Deeper Model
- **Input Layer**: 784 nodes (28×28 flattened)
- **Hidden Layer 1**: 256 nodes with ReLU activation
- **Hidden Layer 2**: 128 nodes with ReLU activation
- **Hidden Layer 3**: 64 nodes with ReLU activation
- **Hidden Layer 4**: 32 nodes with ReLU activation
- **Output Layer**: 10 nodes (one per class)

## Requirements

- Python 3.8+
- PyTorch 2.1.0
- torchvision 0.16.0
- NumPy < 2.0

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ArtieK/pytorch-fashion-classifier.git
cd pytorch-fashion-classifier
```

2. Create and activate a virtual environment:
```bash
python3 -m venv Pytorch
source Pytorch/bin/activate  # On Windows: Pytorch\Scripts\activate
```

3. Install dependencies:
```bash
pip install --upgrade pip
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0
pip install "numpy<2"
```

## Usage

### Basic Example

```python
import torch
import torch.nn as nn
from pytorch import get_data_loader, build_model, train_model, evaluate_model, predict_label

# Initialize
criterion = nn.CrossEntropyLoss()

# Load data
train_loader = get_data_loader(training=True)
test_loader = get_data_loader(training=False)

# Build and train model
model = build_model()
train_model(model, train_loader, criterion, epochs=5)

# Evaluate
evaluate_model(model, test_loader, criterion, show_loss=True)

# Make predictions
test_images = next(iter(test_loader))[0]
predict_label(model, test_images, index=0)
```

### Training Output

```
Train Epoch: 0 Accuracy: 42269/60000(70.45%) Loss: 0.898
Train Epoch: 1 Accuracy: 49400/60000(82.33%) Loss: 0.506
Train Epoch: 2 Accuracy: 50553/60000(84.25%) Loss: 0.450
Train Epoch: 3 Accuracy: 51264/60000(85.44%) Loss: 0.417
Train Epoch: 4 Accuracy: 51645/60000(86.08%) Loss: 0.394
```

### Prediction Output

```
Pullover: 98.77%
Shirt: 1.08%
Coat: 0.11%
```

## API Reference

### `get_data_loader(training=True)`
Returns a DataLoader for the Fashion-MNIST dataset.

**Parameters:**
- `training` (bool): If True, returns training set; if False, returns test set

**Returns:**
- `torch.utils.data.DataLoader`: DataLoader with batch_size=64

### `build_model()`
Constructs the basic neural network model.

**Returns:**
- `nn.Sequential`: Untrained neural network model

### `build_deeper_model()`
Constructs a deeper neural network model with additional hidden layers.

**Returns:**
- `nn.Sequential`: Untrained deeper neural network model

### `train_model(model, train_loader, criterion, T)`
Trains the model using SGD optimization.

**Parameters:**
- `model`: Neural network model to train
- `train_loader`: DataLoader containing training data
- `criterion`: Loss function (e.g., CrossEntropyLoss)
- `T` (int): Number of training epochs

### `evaluate_model(model, test_loader, criterion, show_loss=True)`
Evaluates model performance on test data.

**Parameters:**
- `model`: Trained neural network model
- `test_loader`: DataLoader containing test data
- `criterion`: Loss function
- `show_loss` (bool): Whether to display loss metric

### `predict_label(model, test_images, index)`
Predicts the top 3 most likely classes for a single image.

**Parameters:**
- `model`: Trained neural network model
- `test_images`: Tensor of shape (N, 1, 28, 28)
- `index` (int): Index of image to predict (0 ≤ index < N)

## Performance

The basic model achieves approximately **84-86%** accuracy on the test set after 5 epochs of training with the following hyperparameters:

- **Optimizer**: SGD
- **Learning Rate**: 0.001
- **Momentum**: 0.9
- **Batch Size**: 64
- **Loss Function**: CrossEntropyLoss

## Project Structure

```
pytorch-fashion-classifier/
├── pytorch.py          # Main implementation
├── README.md           # Project documentation
├── .gitignore         # Git ignore rules
├── data/              # Dataset (auto-downloaded)
└── Pytorch/           # Virtual environment
```

## Technical Details

### Data Preprocessing
- Images are converted to tensors and normalized with mean=0.1307, std=0.3081
- Training data is shuffled; test data maintains consistent order
- Batch processing with size 64 for efficient GPU utilization

### Training Strategy
- **Stochastic Gradient Descent (SGD)**: Uses random batches for faster convergence
- **Momentum**: Accelerates gradient descent in relevant directions
- **CrossEntropyLoss**: Combines LogSoftmax and NLLLoss for classification

### Model Considerations
- ReLU activations introduce non-linearity for complex pattern learning
- No activation on final layer (handled internally by CrossEntropyLoss)
- Sequential architecture for straightforward feedforward processing

## License

This project was developed as part of academic coursework at the University of Wisconsin-Madison (CS540: Introduction to Artificial Intelligence). The code is provided for educational and reference purposes.

## Acknowledgments

- Fashion-MNIST dataset by Zalando Research
- PyTorch framework by Meta AI
- Course materials from UW-Madison CS540

## Author

**Artie Krishnan**
- GitHub: [@ArtieK](https://github.com/ArtieK)

---

*Built with PyTorch • Fashion-MNIST • Neural Networks*