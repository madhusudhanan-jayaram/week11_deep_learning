# Simple program to understand ReLU activation function
import numpy as np
import matplotlib.pyplot as plt

# Relu stands for Rectified linear Unit

print("🔥 UNDERSTANDING ReLU ACTIVATION FUNCTION")
print("=" * 50)

# 1. Define ReLU function
def relu_function(x):
    """Simple ReLU implementation"""
    return np.maximum(0, x)

# 2. Test ReLU with different inputs
print("\n📊 ReLU Function Examples:")
test_values = [-5, -2, -0.5, 0, 0.5, 2, 5, 10]

for value in test_values:
    result = relu_function(value)
    print(f"ReLU({value:4}) = {result:4}")

# 3. Visualize ReLU function
print("\n📈 Plotting ReLU Function:")
x = np.linspace(-10, 10, 100)
y = relu_function(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y, 'b-', linewidth=3, label='ReLU(x)')
plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
plt.grid(True, alpha=0.3)
plt.xlabel('Input (x)')
plt.ylabel('Output ReLU(x)')
plt.title('ReLU Activation Function: max(0, x)')
plt.legend()
plt.show()

# 4. Simulate a mini neural network with ReLU
print("\n🧠 Mini Neural Network Example:")
print("Input → Dense Layer → ReLU → Output")

# Simulate input (like flattened pixels)
input_data = np.array([-2, -1, 0, 1, 2, 3])
print(f"Input data:     {input_data}")

# Simulate weights and bias (randomly initialized)
weights = np.array([0.5, -0.3, 0.8, 0.2, -0.6, 0.9])
bias = 0.1
print(f"Weights:        {weights}")
print(f"Bias:           {bias}")

# Dense layer computation: weighted sum + bias
dense_output = np.dot(input_data, weights) + bias
print(f"Dense output:   {dense_output:.3f}")

# Apply ReLU activation
relu_output = relu_function(dense_output)
print(f"After ReLU:     {relu_output:.3f}")

# 5. Compare with other activation functions
print("\n🔄 Comparison with other activations:")
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

test_input = 2.5
print(f"Input: {test_input}")
print(f"ReLU:    {relu_function(test_input):.3f}")
print(f"Sigmoid: {sigmoid(test_input):.3f}")
print(f"Tanh:    {tanh(test_input):.3f}")

# 6. Why ReLU is popular
print("\n✨ Why ReLU is Popular:")
print("✅ Simple: max(0, x)")
print("✅ Fast: No complex calculations")
print("✅ Sparse: ~50% of outputs are zero")
print("✅ No vanishing gradient for positive values")
print("✅ Works well in deep networks")

# 7. ReLU in action with your digit data
print("\n🔢 ReLU with Digit Recognition:")
print("Pixel values → Dense layer → ReLU → Pattern detection")
print("Negative activations = irrelevant features (turned off)")
print("Positive activations = important features (kept)")