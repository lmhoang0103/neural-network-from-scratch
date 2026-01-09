# Neural Network From Scratch (NumPy Only)

This project implements a fully functional neural network **from scratch**, using only NumPy.  
The goal is to demonstrate conceptual and mathematical understanding of neural network internals rather than framework usage.

The implementation covers model definition, training dynamics, optimization, and regularization, and is applied to a non-linear multi-class classification problem.

---

## Scope and Constraints

**Deliberate constraints**
- No deep learning frameworks (TensorFlow, PyTorch, scikit-learn)
- NumPy only
- Explicit forward and backward passes
- Manual optimization logic

**What this project demonstrates**
- Understanding of neural network mechanics
- Gradient-based optimization
- Stability and generalization techniques
- Practical application to a non-trivial classification task

---

## Implemented Components

### Core Neural Network
- Fully connected (dense) layers
- Forward propagation
- Backpropagation with analytical gradients

### Activation Functions
- ReLU
- Softmax

### Loss Function
- Categorical Cross-Entropy Loss
- Numerical stability handling

### Optimization Algorithms
- Vanilla Gradient Descent
- Momentum
- AdaGrad
- Adam

Each optimizer is implemented manually and operates directly on parameter gradients.

### Regularization
- L1 and L2 penalty added directly to loss
- Dropout applied during training

---

## Applied Problem

The model is applied to a **multi-class classification task**:

- Input: 2D points
- Output: 3 discrete classes
- Decision boundary: non-linear and geometrically complex

This problem setup prevents trivial linear separation and forces the network to learn meaningful internal representations.

---
## How to run

1. Intall dependencies:
```bash
pip install -r requirements.txt
```
2. Run model
All the code along with classification problems and experiments along with plot showing to demonstrate the problem and statistical result of the model

---
## Result

The Model achieve 85% accuracy in Testing
