import numpy as np
import companion_gpt

# Simple neural network for XOR
class SimpleNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros((1, output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, X):
        # Use Rust for dot product and sigmoid
        self.z1 = companion_gpt.dot_product_rust(X, self.W1) + self.b1
        self.a1 = companion_gpt.sigmoid_rust(self.z1.flatten())
        self.a1 = self.a1.reshape(self.z1.shape)  # Reshape back
        self.z2 = companion_gpt.dot_product_rust(self.a1, self.W2) + self.b2
        self.a2 = companion_gpt.sigmoid_rust(self.z2.flatten())
        self.a2 = self.a2.reshape(self.z2.shape)
        return self.a2

    def backward(self, X, y, output, learning_rate=0.1):
        output_error = y - output
        output_delta = output_error * self.sigmoid_derivative(output)

        hidden_error = output_delta.dot(self.W2.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.a1)

        self.W2 += self.a1.T.dot(output_delta) * learning_rate
        self.b2 += np.sum(output_delta, axis=0, keepdims=True) * learning_rate
        self.W1 += X.T.dot(hidden_delta) * learning_rate
        self.b1 += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate

    def train(self, X, y, epochs=10000):
        for _ in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)

# XOR data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
y = np.array([[0], [1], [1], [0]], dtype=np.float64)

nn = SimpleNN(2, 2, 1)
nn.train(X, y)

# Test
predictions = nn.forward(X)
print("Predictions:")
print(predictions)
print("Rounded:")
print(np.round(predictions))

# Simple task execution
def perform_task(task):
    if "add" in task.lower():
        parts = task.split()
        try:
            a = float(parts[1])
            b = float(parts[3])
            return a + b
        except:
            return "Invalid task"
    elif "multiply" in task.lower():
        parts = task.split()
        try:
            a = float(parts[1])
            b = float(parts[3])
            return a * b
        except:
            return "Invalid task"
    else:
        return "Unknown task"

# Example
print("Task result:", perform_task("add 3 and 4"))