import math
import random
import struct
import numpy as np

# Load MNIST Training and Testing Data
def read_images(filename):
    with open(filename, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows * cols)

def read_labels(filename):
    with open(filename, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        return np.frombuffer(f.read(), dtype=np.uint8)

x_train = read_images('Dataset/train-images.idx3-ubyte') / 255.0
y_train = read_labels('Dataset/train-labels.idx1-ubyte')
x_test = read_images('Dataset/t10k-images.idx3-ubyte') / 255.0
y_test = read_labels('Dataset/t10k-labels.idx1-ubyte')

# Define different activation functions and helper funcitons
def relu(z):
    return np.maximum(0, z)

def derivative_relu(z):
    return (z > 0).astype(float)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def derivative_sigmoid(z):
    s = sigmoid(z)
    return s * (1 - s)

def softmax(z):
    z_max = np.max(z, axis=0, keepdims=True)
    e_z = np.exp(z - z_max)
    return e_z / np.sum(e_z, axis=0, keepdims=True)

def derivative_softmax(output, target):
    return output - target

def one_hot_batch(y, num_classes=10):
    m = y.shape[0]
    out = np.zeros((num_classes, m))
    out[y, np.arange(m)] = 1
    return out

# Define a scalable fully connected Neural Network class with variable depth and layer size optimized for the classification task
class NeuralNetwork:
    def __init__(self, input_size, num_hidden_layers, hidden_layer_size, output_size):
        self.input_size = input_size
        self.num_layers = num_hidden_layers
        self.hidden_layers_weights = []
        self.hidden_layers_biases = []
        self.output_size = output_size
        self.hidden_layer_size = hidden_layer_size

    # Initialize weight and bias values between 0.1 and -0.1
    def initialize_layers(self):
        layer_sizes = [self.input_size] + [self.hidden_layer_size[i] for i in range(self.num_layers)] + [self.output_size]
        for i in range(len(layer_sizes) - 1):
            weights = (np.random.rand(layer_sizes[i+1], layer_sizes[i]) * 2 - 1) / 10 
            biases = (np.random.rand(layer_sizes[i+1], 1) * 2 - 1) / 10
            self.hidden_layers_weights.append(weights)
            self.hidden_layers_biases.append(biases)
    
    # Feedforward step using ReLU activation for hidden layers and softmax for output layer
    def feed_forward(self, input_batch):
        self.activations = [input_batch]
        self.z_values = []

        current = input_batch
        for i in range(len(self.hidden_layers_weights)):
            z = self.hidden_layers_weights[i] @ current + self.hidden_layers_biases[i]
            self.z_values.append(z)
            if i == len(self.hidden_layers_weights) - 1:
                current = softmax(z)
            else:
                current = relu(z)
            self.activations.append(current)
        return current
    
    def backpropagate(self, desired_batch, learning_rate):
        m = desired_batch.shape[1]

        output = self.activations[-1]
        delta = derivative_softmax(output, desired_batch)
        deltas = [delta]

        for l in range(len(self.hidden_layers_weights) - 2, -1, -1):
            z = self.z_values[l]
            w_next = self.hidden_layers_weights[l + 1]
            delta = (w_next.T @ deltas[0]) * derivative_relu(z)
            deltas.insert(0, delta)

        for l in range(len(self.hidden_layers_weights)):
            a_prev = self.activations[l]
            delta = deltas[l]

            grad_w = (delta @ a_prev.T) / m
            grad_b = np.mean(delta, axis=1, keepdims=True)

            self.hidden_layers_weights[l] -= learning_rate * grad_w
            self.hidden_layers_biases[l] -= learning_rate * grad_b

    def train(self, learning_rate, input_batch, desired_batch):
        self.feed_forward(input_batch)
        self.backpropagate(desired_batch, learning_rate)

### Example use case for MNIST classification ###

LEARNING_RATE = 0.01
EPOCHS = 10
BATCH_SIZE = 32

NN = NeuralNetwork(784, 2, [16, 16], 10)
NN.initialize_layers()

for epoch in range(EPOCHS):

    print(epoch)

    indices = np.arange(len(x_train))
    np.random.shuffle(indices)

    for i in range(0, len(x_train), BATCH_SIZE):
        batch_indices = indices[i : i + BATCH_SIZE]
        batch_x = x_train[batch_indices].T
        batch_y = one_hot_batch(y_train[batch_indices])

        NN.train(LEARNING_RATE, batch_x, batch_y)

    correct = 0
    for j in range(len(x_test)):
        input = x_test[j].reshape((784, 1))
        output = NN.feed_forward(input)
        if np.argmax(output) == y_test[j]:
            correct += 1
    accuracy = correct / len(x_test)
    print(f"Accuracy: {accuracy * 100:.2f} %")
