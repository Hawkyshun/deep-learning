import numpy as np

class NeuralNetwork:
    def __init__(self, learning_rate=0.1):
        self.weights = {
            'h1': 0.5,  # w11
            'h2': 0.5,  # w21
            'o': np.array([0.5, 0.5])  # w11, w12 for output layer
        }
        self.learning_rate = learning_rate
        self.iterations = 0
        self.errors = []

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward_propagation(self, x):
        # First layer
        self.z1 = x * self.weights['h1']
        self.h1 = self.sigmoid(self.z1)
        
        self.z2 = x * self.weights['h2']
        self.h2 = self.sigmoid(self.z2)
        
        # Output layer
        self.z_o = self.h1 * self.weights['o'][0] + self.h2 * self.weights['o'][1]
        self.output = self.sigmoid(self.z_o)
        
        return self.output

    def backward_propagation(self, x, y):
        # Calculate gradients
        error = float(y - self.output)
        self.errors.append(abs(error))
        
        # Output layer gradients
        d_output = error * self.output * (1 - self.output)
        
        # Hidden layer gradients
        d_h1 = d_output * self.weights['o'][0] * self.h1 * (1 - self.h1)
        d_h2 = d_output * self.weights['o'][1] * self.h2 * (1 - self.h2)
        
        # Update weights
        self.weights['o'][0] += self.learning_rate * d_output * self.h1
        self.weights['o'][1] += self.learning_rate * d_output * self.h2
        self.weights['h1'] += self.learning_rate * d_h1 * x
        self.weights['h2'] += self.learning_rate * d_h2 * x

    def train(self, x, y, epochs=1):
        for _ in range(epochs):
            self.iterations += 1
            output = self.forward_propagation(x)
            self.backward_propagation(x, y)
            
            if self.iterations % 100 == 0:
                print(f"Iterasyon {self.iterations}: Hata = {self.errors[-1]:.6f}")

# Test the implementation
if __name__ == "__main__":
    # Initialize network
    nn = NeuralNetwork(learning_rate=0.1)
    
    # Training data
    X = 1  # input
    y = 0.5  # desired output
    
    # Train for 1000 iterations
    nn.train(X, y, epochs=1000)
    
    # Final results
    print("\nEğitim Tamamlandı!")
    print(f"Toplam İterasyon: {nn.iterations}")
    print(f"Son Hata: {nn.errors[-1]:.6f}")
    print(f"Son Çıktı: {nn.forward_propagation(X):.6f}") 