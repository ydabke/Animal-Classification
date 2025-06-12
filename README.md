# CNN Animal Classifier Using PyTorch

## Overview

This project is a convolutional neural network (CNN) built using PyTorch for classifying animal faces. The dataset used is the [Animal Faces (AFHQ)](https://www.kaggle.com/datasets/andrewmvd/animal-faces), which includes images of cats, dogs, and wild animals. The goal was to understand how to apply CNNs for image classification and to deepen my understanding of PyTorch fundamentals, from data preprocessing to model evaluation.

## Dataset

The dataset was downloaded directly from Kaggle using the `opendatasets` library. It consists of images grouped into subdirectories by class (i.e., cat, dog, wild). Each image is resized to 128x128 and transformed into a tensor before being fed into the model.

## Key Concepts Learned

* Custom Dataset creation using `torch.utils.data.Dataset`
* Data preprocessing with `torchvision.transforms`
* CNN architecture design in PyTorch
* Efficient training loops including validation
* Model evaluation and metric plotting
* GPU utilization with `.to(device)`

## Project Workflow

### 1. Data Loading and Preprocessing

* Extract image paths and labels into a DataFrame.
* Use `LabelEncoder` to convert string labels to numerical values.
* Apply image transformations using `transforms.Compose`:

  * Resize
  * ToTensor
  * ConvertImageDtype

### 2. Custom Dataset Class

Implemented a `CustomImageDataset` class to interface with PyTorch's `DataLoader`, providing flexibility in loading and transforming images.

### 3. Data Splitting

* 70% training
* 15% validation
* 15% testing

### 4. CNN Model Architecture

```python
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pooling = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(128 * 16 * 16, 128)
        self.output = nn.Linear(128, 3)

    def forward(self, x):
        x = self.pooling(self.relu(self.conv1(x)))
        x = self.pooling(self.relu(self.conv2(x)))
        x = self.pooling(self.relu(self.conv3(x)))
        x = self.flatten(x)
        x = self.linear(x)
        return self.output(x)
```

### 5. Training Loop

* Loss: `CrossEntropyLoss`
* Optimizer: `Adam`
* Metrics: Accuracy and Loss (both training and validation)

```python
for epoch in range(EPOCHS):
    # Training phase
    # Validation phase (with torch.no_grad())
    # Store loss and accuracy for each phase
```

### 6. Model Evaluation

* Achieved \~96% accuracy on the test set
* Used `matplotlib` to visualize loss and accuracy over epochs

### 7. Inference Function

```python
def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).to(device)
    output = model(image.unsqueeze(0))
    return label_encoder.inverse_transform([torch.argmax(output, 1).item()])
```

## How This Project Helped Me Learn PyTorch

* **Model Design**: I learned how to structure convolutional neural networks from scratch using PyTorch modules.
* **Dataset Handling**: Creating a custom `Dataset` class helped me understand how PyTorch expects input.
* **Efficient Training**: Implementing separate loops for training and validation with GPU support showed how to manage memory and computation efficiently.
* **Evaluation & Debugging**: Visualization of loss/accuracy helped in tracking the model's learning behavior.
* **Real-World Dataset**: Handling a realistic, complex image dataset provided hands-on experience with preprocessing and model generalization.

## Future Improvements

* Add data augmentation
* Implement early stopping
* Try transfer learning using pretrained CNNs (e.g., ResNet)
* Improve test accuracy with hyperparameter tuning

## Requirements

* PyTorch
* torchvision
* opendatasets
* scikit-learn
* matplotlib
* PIL

## Run in Google Colab

This notebook was developed and run in Google Colab, using a GPU runtime.

---

*This project was an invaluable learning experience and significantly strengthened my understanding of convolutional neural networks and PyTorch.*
