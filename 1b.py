"""
The dataset from part C was used to evaluate 1 epoch of the model. The accuracy ended up being 11% for 1 epoch. Code for part c should include indications of it being used for part b as well.
"""

import numpy as np
import matplotlib.pyplot as plt

# Set seed for reproducibility
np.random.seed(42) # the seed allows us to generate a random value, but the same random value each time it is run. This is to help with reproducibility.

# Activation functions
def relu(x): # 1st activation function; adds non-linearity
    return np.maximum(0, x)

def relu_derivative(x): # 2nd activation function; required for backpropogation; 1 for when the input is larger than 0
    return x > 0

def softmax(x): # Turns the final layer scores into probablilites, summing to 1
    exps = np.exp(x - np.max(x))  # stability
    return exps / np.sum(exps, axis=1, keepdims=True)

# Loss function
def cross_entropy(predictions, labels): # A loss function determined by softmax output and true labels; go from 0-9
    m = labels.shape[0]
    p = softmax(predictions)
    log_likelihood = -np.log(p[range(m), labels])
    return np.sum(log_likelihood) / m

def accuracy(predictions, labels): # Finds the model's accuracy
    pred_labels = np.argmax(predictions, axis=1)
    return np.mean(pred_labels == labels) # compares the predicted vs. true labels

# Initialize weights
def init_weights(input_size, hidden_sizes, output_size): # Initalizes weights with small random numbers and bias as 0; for any number of layers
    layers = [input_size] + hidden_sizes + [output_size]
    weights = []
    biases = []
    for i in range(len(layers) - 1):
        weights.append(np.random.randn(layers[i], layers[i+1]) * 0.01)
        biases.append(np.zeros((1, layers[i+1])))
    return weights, biases

# Forward pass
def forward_pass(x, weights, biases): # Runs the inputs through each layer
    activations = [x]
    zs = [] # zs is weightes sums
    for i in range(len(weights) - 1):
        z = activations[-1] @ weights[i] + biases[i] # the @ is matrix multiplication
        zs.append(z)
        activations.append(relu(z))
    z = activations[-1] @ weights[-1] + biases[-1]
    zs.append(z)
    activations.append(z)  # No softmax here; done in loss
    return activations, zs # activations are output of each layer after relu

# Backward pass
def backward_pass(x, y, weights, biases, activations, zs, lr): # Computes gradients and updates weights using gradient descent
    m = x.shape[0]
    grads_w = [0] * len(weights)
    grads_b = [0] * len(biases)

    # One-hot encode labels
    y_onehot = np.zeros_like(activations[-1])
    y_onehot[np.arange(m), y] = 1

    # Output layer gradient
    dz = softmax(activations[-1]) - y_onehot
    for i in reversed(range(len(weights))):
        grads_w[i] = activations[i].T @ dz / m
        grads_b[i] = np.sum(dz, axis=0, keepdims=True) / m
        if i > 0:
            dz = (dz @ weights[i].T) * relu_derivative(zs[i-1]) # use chain rule with ReLU derivative

    # Update weights
    for i in range(len(weights)):
        weights[i] -= lr * grads_w[i]
        biases[i] -= lr * grads_b[i] # Updates are scaled by learning rate (lr)

# Training loop
def train(x_train, y_train, x_val, y_val, epochs=10, batch_size=32, lr=0.1): # Trains using mini-batches
    weights, biases = init_weights(100, [10, 10, 10], 10) # Initializes the model with 100 → 10 → 10 → 10 → 10 perceptrons
    history = [] # Tracks performance in history

    for epoch in range(epochs):
        indices = np.arange(x_train.shape[0])
        np.random.shuffle(indices)

        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i:i+batch_size]
            x_batch, y_batch = x_train[batch_idx], y_train[batch_idx]

            activations, zs = forward_pass(x_batch, weights, biases)
            backward_pass(x_batch, y_batch, weights, biases, activations, zs, lr)

        # Evaluate
        val_logits, _ = forward_pass(x_val, weights, biases)
        acc = accuracy(val_logits[-1], y_val)
        loss = cross_entropy(val_logits[-1], y_val)
        print(f"Epoch {epoch+1}: Val Acc = {acc:.4f}, Val Loss = {loss:.4f}") # After each epoch, prints accuracy and loss on validation data
        history.append((acc, loss))

    return weights, biases, history
