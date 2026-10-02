# Neural Network From Scratch

A basic fully connected neural network implemented from scratch using NumPy for handwritten digit classification.

## Overview

This project implements a simple neural network without using high-level deep learning frameworks such as TensorFlow or PyTorch.

The model is trained on the Scikit-learn Digits dataset and classifies handwritten digits from 0 to 9.

## Dataset

The project uses the Digits dataset from Scikit-learn.

* Total samples: 1797
* Image size: 8 × 8 pixels
* Input features: 64
* Number of classes: 10
* Classes: 0–9

Each 8 × 8 image is flattened into 64 input features.

## Model Architecture

```text
Input Layer
64 Features
     |
     v
Hidden Layer
128 Neurons
ReLU Activation
     |
     v
Output Layer
10 Neurons
Softmax Activation
     |
     v
Predicted Digit
```

Architecture:

```text
64 → 128 → 10
```

## Preprocessing

The dataset is divided into training and testing sets.

The pixel values are normalized by dividing them by 16:

```python
X = X / 16.0
```

The target labels are converted into one-hot encoded vectors for calculating the loss.

## Forward Propagation

The first layer is calculated as:

```python
Z1 = X @ W1 + b1
A1 = ReLU(Z1)
```

The output layer is calculated as:

```python
Z2 = A1 @ W2 + b2
A2 = Softmax(Z2)
```

The class with the highest probability is selected as the prediction.

## Activation Functions

### ReLU

The hidden layer uses the ReLU activation function:

```text
ReLU(x) = max(0, x)
```

It converts negative values to zero while keeping positive values unchanged.

### Softmax

The output layer uses Softmax to convert the output scores into probabilities for the 10 digit classes.

## Loss Function

Cross-Entropy Loss is used to measure the difference between the actual labels and the predicted probabilities.

```text
Loss = -Σ y_true × log(y_pred)
```

The objective of training is to minimize this loss.

## Backpropagation

Gradients are calculated manually using backpropagation.

The output-layer gradient is:

```python
dZ2 = A2 - y_train_encoded
```

The gradients for the second layer are:

```python
dW2 = (A1.T @ dZ2) / m
db2 = np.sum(dZ2, axis=0, keepdims=True) / m
```

The error is then propagated back to the hidden layer:

```python
dA1 = dZ2 @ W2.T
dZ1 = dA1 * (Z1 > 0)
```

The first-layer gradients are calculated as:

```python
dW1 = (X.T @ dZ1) / m
db1 = np.sum(dZ1, axis=0, keepdims=True) / m
```

## Gradient Descent

The weights and biases are updated using gradient descent:

```python
W -= learning_rate * dW
b -= learning_rate * db
```

The training process is repeated for multiple epochs.

## Prediction

After training, the model performs forward propagation on the test data and selects the class with the highest probability:

```python
prediction = np.argmax(A2, axis=1)
```

Test accuracy is calculated by comparing the predicted labels with the actual test labels.

## Project Files

```text
Neural-Network/
│
├── neural.ipynb
├── neural.pdf
└── README.md
```

### neural.ipynb

Contains the complete implementation and execution of the neural network.

### neural.pdf

Contains the exported version of the Jupyter Notebook for easy viewing and submission.

## Technologies Used

* Python
* NumPy
* Matplotlib
* Scikit-learn
* Jupyter Notebook

## Learning Outcomes

This project helped in understanding:

* Data preprocessing
* One-hot encoding
* Weight initialization
* Forward propagation
* ReLU activation
* Softmax activation
* Cross-entropy loss
* Backpropagation
* Gradient calculation
* Gradient descent
* Prediction
* Training and testing accuracy

## Conclusion

This project demonstrates the implementation of a basic neural network from scratch using NumPy. It provides a practical understanding of how forward propagation, loss calculation, backpropagation, and parameter updates work together to train a classification model.

## Author

**Syed Kaysan Ul Islam**
B.Tech in Artificial Intelligence & Machine Learning
