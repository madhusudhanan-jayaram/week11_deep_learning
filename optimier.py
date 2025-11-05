import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
import matplotlib.pyplot as plt
import numpy as np

print("🔧 OPTIMIZER PARAMETERS - Simple Demonstration")
print("=" * 60)

# Load and prepare simple dataset (Fashion-MNIST for variety)
print("📦 Loading Fashion-MNIST dataset...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# Normalize data
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Use smaller dataset for faster demonstration
x_train_small = x_train[:5000]
y_train_small = y_train[:5000]
x_test_small = x_test[:1000]
y_test_small = y_test[:1000]

print(f"Training samples: {len(x_train_small)}")
print(f"Test samples: {len(x_test_small)}")

# Simple model function
def create_simple_model():
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])
    return model

print("\n" + "=" * 60)
print("🎯 EXPERIMENT 1: Learning Rate Impact")
print("=" * 60)

learning_rates = [0.1, 0.01, 0.001, 0.0001]
results = {}

for lr in learning_rates:
    print(f"\n🔍 Testing Learning Rate: {lr}")
    
    model = create_simple_model()
    optimizer = Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    
    # Train for 5 epochs
    history = model.fit(x_train_small, y_train_small,
                       validation_data=(x_test_small, y_test_small),
                       epochs=5, verbose=0)
    
    final_accuracy = history.history['val_accuracy'][-1]
    results[lr] = final_accuracy
    print(f"   Final Accuracy: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")

# Display results
print(f"\n📊 LEARNING RATE COMPARISON:")
print("-" * 40)
for lr, acc in results.items():
    stars = "⭐" * int(acc * 10)
    print(f"LR {lr:6}: {acc:.4f} {stars}")

best_lr = max(results, key=results.get)
print(f"\n🏆 Best Learning Rate: {best_lr} (Accuracy: {results[best_lr]:.4f})")

print("\n" + "=" * 60)
print("🎯 EXPERIMENT 2: Adam Parameters Fine-tuning")
print("=" * 60)

# Test different Adam configurations
adam_configs = [
    {'name': 'Default Adam', 'lr': 0.001, 'beta_1': 0.9, 'beta_2': 0.999},
    {'name': 'Fast Adam', 'lr': 0.01, 'beta_1': 0.9, 'beta_2': 0.999},
    {'name': 'Conservative Adam', 'lr': 0.0001, 'beta_1': 0.9, 'beta_2': 0.999},
    {'name': 'High Beta1 Adam', 'lr': 0.001, 'beta_1': 0.95, 'beta_2': 0.999},
    {'name': 'Low Beta2 Adam', 'lr': 0.001, 'beta_1': 0.9, 'beta_2': 0.99}
]

adam_results = {}

for config in adam_configs:
    print(f"\n🧪 Testing {config['name']}")
    print(f"   Parameters: lr={config['lr']}, beta_1={config['beta_1']}, beta_2={config['beta_2']}")
    
    model = create_simple_model()
    optimizer = Adam(learning_rate=config['lr'], 
                     beta_1=config['beta_1'], 
                     beta_2=config['beta_2'])
    
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    history = model.fit(x_train_small, y_train_small,
                       validation_data=(x_test_small, y_test_small),
                       epochs=5, verbose=0)
    
    final_acc = history.history['val_accuracy'][-1]
    adam_results[config['name']] = final_acc
    print(f"   Final Accuracy: {final_acc:.4f}")

print(f"\n📊 ADAM CONFIGURATION COMPARISON:")
print("-" * 50)
for name, acc in adam_results.items():
    print(f"{name:20}: {acc:.4f} ({'⭐' * int(acc * 10)})")

print("\n" + "=" * 60)
print("🎯 EXPERIMENT 3: SGD with Momentum")
print("=" * 60)

# Test SGD with different momentum values
momentum_values = [0.0, 0.5, 0.9, 0.99]
sgd_results = {}

for momentum in momentum_values:
    print(f"\n🔍 Testing SGD with Momentum: {momentum}")
    
    model = create_simple_model()
    optimizer = SGD(learning_rate=0.01, momentum=momentum)
    
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    history = model.fit(x_train_small, y_train_small,
                       validation_data=(x_test_small, y_test_small),
                       epochs=5, verbose=0)
    
    final_acc = history.history['val_accuracy'][-1]
    sgd_results[momentum] = final_acc
    print(f"   Final Accuracy: {final_acc:.4f}")

print(f"\n📊 MOMENTUM COMPARISON:")
print("-" * 30)
for momentum, acc in sgd_results.items():
    print(f"Momentum {momentum:4}: {acc:.4f}")

print("\n" + "=" * 60)
print("🎯 EXPERIMENT 4: Optimizer Comparison")
print("=" * 60)

# Compare different optimizers with their best settings
optimizers_to_test = [
    ('Adam (Default)', Adam(learning_rate=0.001)),
    ('Adam (Tuned)', Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999)),
    ('SGD (No Momentum)', SGD(learning_rate=0.01)),
    ('SGD (With Momentum)', SGD(learning_rate=0.01, momentum=0.9)),
    ('RMSprop', RMSprop(learning_rate=0.001))
]

optimizer_results = {}
training_histories = {}

for name, optimizer in optimizers_to_test:
    print(f"\n🏃‍♂️ Training with {name}...")
    
    model = create_simple_model()
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    history = model.fit(x_train_small, y_train_small,
                       validation_data=(x_test_small, y_test_small),
                       epochs=8, verbose=0)
    
    final_acc = history.history['val_accuracy'][-1]
    optimizer_results[name] = final_acc
    training_histories[name] = history.history
    print(f"   Final Accuracy: {final_acc:.4f}")

# Plot training curves
plt.figure(figsize=(12, 5))

# Plot accuracy
plt.subplot(1, 2, 1)
for name, history in training_histories.items():
    plt.plot(history['val_accuracy'], label=name, marker='o')
plt.title('Validation Accuracy Comparison')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Plot loss
plt.subplot(1, 2, 2)
for name, history in training_histories.items():
    plt.plot(history['val_loss'], label=name, marker='o')
plt.title('Validation Loss Comparison')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

print("\n" + "=" * 60)
print("🏆 FINAL RESULTS SUMMARY")
print("=" * 60)

# Sort results by accuracy
sorted_results = sorted(optimizer_results.items(), key=lambda x: x[1], reverse=True)

print("\n🥇 OPTIMIZER RANKING:")
for i, (name, acc) in enumerate(sorted_results, 1):
    medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
    print(f"{medal} {name:20}: {acc:.4f} ({acc*100:.2f}%)")

print("\n" + "=" * 60)
print("💡 KEY INSIGHTS & RECOMMENDATIONS")
print("=" * 60)

print("\n🎯 LEARNING RATE INSIGHTS:")
print("• Too high (>0.1): Training becomes unstable")
print("• Too low (<0.0001): Learning is very slow")
print("• Sweet spot: 0.001-0.01 for most problems")

print("\n🚀 ADAM PARAMETER TIPS:")
print("• learning_rate=0.001: Good default")
print("• beta_1=0.9: Controls momentum (rarely change)")
print("• beta_2=0.999: Controls adaptation (rarely change)")

print("\n🏃‍♂️ SGD MOMENTUM TIPS:")
print("• momentum=0: No acceleration (slower)")
print("• momentum=0.9: Good balance (recommended)")
print("• momentum=0.99: High acceleration (might overshoot)")

print("\n" + "=" * 60)
print("🔧 READY-TO-USE CODE TEMPLATES")
print("=" * 60)

print("\n# 🥇 Best general purpose optimizer:")
print("optimizer = Adam(learning_rate=0.001)")

print("\n# 🎯 For fine-tuning pre-trained models:")
print("optimizer = Adam(learning_rate=0.0001)")

print("\n# 🏃‍♂️ For stable training from scratch:")
print("optimizer = SGD(learning_rate=0.01, momentum=0.9)")

print("\n# ⚡ For faster convergence (risky):")
print("optimizer = Adam(learning_rate=0.01)")

print("\n# 🕰️ For RNNs and time series:")
print("optimizer = RMSprop(learning_rate=0.001)")

print("\n🎓 Remember: Start with Adam(lr=0.001), then experiment!")