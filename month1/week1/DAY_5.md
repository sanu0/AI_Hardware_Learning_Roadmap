# Week 1 , Day 5: Neural Networks from First Principles
## Build a Neural Network from Scratch — Then See It Run on Your GPU

Yesterday you wrote CUDA kernels. Today you build the thing those kernels
are designed to accelerate: a neural network. By the end of this session,
you'll have built a neural network from raw math in NumPy (no frameworks),
then rebuilt it in PyTorch and run it on GPU.

**Time: ~2.5-3 hours**
**Setup: Google Colab (GPU runtime for PyTorch exercises)**

---

# PART 0: NEW TERMS FOR TODAY

```
NEURON         = The basic unit: takes inputs, multiplies by weights, adds bias,
                 applies activation function. Outputs one number.
                 Inspired by brain neurons but that's where the analogy ends.

WEIGHT         = A learnable number that controls how much an input matters.
                 If weight is large → that input matters a lot.
                 If weight is near zero → that input is ignored.

BIAS           = A learnable number added after the weighted sum.
                 Lets the neuron "shift" its activation.
                 Like the y-intercept in y = mx + b.

ACTIVATION     = A function applied after the weighted sum.
FUNCTION         Without it, stacking layers would be pointless
                 (a stack of linear transforms = still one linear transform).
                 Adds NON-LINEARITY so the network can learn complex patterns.

LAYER          = A group of neurons. All take the same inputs but produce
                 different outputs (because they have different weights).

MLP            = Multi-Layer Perceptron. Layers stacked: input → hidden → output.
                 The simplest type of neural network. Also called "feedforward."

FORWARD PASS   = Running the input through the network to get a prediction.
                 Just matrix multiplications + activations.

LOSS FUNCTION  = Measures how WRONG the prediction is.
                 Low loss = good prediction. High loss = bad prediction.
                 Training = minimizing the loss.

BACKWARD PASS  = Computing gradients: how should each weight change
(BACKPROP)       to reduce the loss? Uses the chain rule from Day 3.

GRADIENT       = The direction and magnitude to nudge a weight.
DESCENT          Repeatedly do: weight = weight - learning_rate × gradient

EPOCH          = One complete pass through all training data.
                 Training typically runs for 10-100+ epochs.

BATCH          = A subset of training data processed together.
                 Instead of updating weights after every single example,
                 we average gradients over a batch (e.g., 32 examples).
                 This is more stable and leverages GPU parallelism.

LEARNING RATE  = How big a step to take when updating weights.
                 Too large → overshoots, training explodes.
                 Too small → takes forever to converge.
                 Typical: 0.001 to 0.01.
```

---

# PART 1: THE NEURON — The Atomic Unit

## 1.1 What a Single Neuron Does

```
A single neuron computes:

  output = activation(w₁×x₁ + w₂×x₂ + w₃×x₃ + ... + bias)
         = activation(dot_product(weights, inputs) + bias)
         = activation(W · X + b)

Visually:
  x₁ ──w₁──╲
  x₂ ──w₂───→ [Σ + b] → [activation] → output
  x₃ ──w₃──╱
             weighted     non-linear
             sum          function

Example with numbers:
  inputs  = [0.5, 0.3, 0.8]
  weights = [0.2, -0.1, 0.4]
  bias    = 0.1
  
  weighted_sum = 0.5×0.2 + 0.3×(-0.1) + 0.8×0.4 + 0.1
               = 0.10 + (-0.03) + 0.32 + 0.1
               = 0.49
  
  output = activation(0.49)
  
  If activation is ReLU: output = max(0, 0.49) = 0.49
  If activation is sigmoid: output = 1/(1+e^(-0.49)) = 0.620
```

## 1.2 Why Activation Functions Exist

```
WITHOUT activation:
  Layer 1: y₁ = W₁ × x + b₁
  Layer 2: y₂ = W₂ × y₁ + b₂ = W₂ × (W₁ × x + b₁) + b₂
                                = (W₂×W₁) × x + (W₂×b₁ + b₂)
                                = W_combined × x + b_combined
  
  TWO layers collapsed into ONE! Stacking is pointless!
  You can only learn LINEAR relationships (straight lines).

WITH activation (e.g., ReLU):
  Layer 1: y₁ = ReLU(W₁ × x + b₁)      ← non-linear!
  Layer 2: y₂ = ReLU(W₂ × y₁ + b₂)
  
  These DON'T collapse. Two layers > one layer.
  Can learn curves, corners, complex patterns.
  
  This is the UNIVERSAL APPROXIMATION THEOREM:
  An MLP with one hidden layer and non-linear activation can
  approximate ANY continuous function (given enough neurons).
```

## 1.3 The Activation Functions You Need to Know

```
SIGMOID: σ(x) = 1 / (1 + e^(-x))
  Range: (0, 1)
  Pros:  Output is a probability
  Cons:  Vanishing gradient for large |x| → slow training
  Used:  Binary classification output, gates in LSTM
  In LLMs: Inside the "sigmoid" part of SiLU activation

TANH: tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
  Range: (-1, 1)
  Pros:  Zero-centered (better than sigmoid)
  Cons:  Still vanishing gradient at extremes
  Used:  Older RNNs, some normalizations
  In LLMs: Rarely used directly

ReLU: ReLU(x) = max(0, x)
  Range: [0, ∞)
  Pros:  Simple, fast, no vanishing gradient for positive values
  Cons:  "Dying ReLU" — neurons that go negative stay dead forever
  Used:  The default for most deep learning (2012-2020)
  In LLMs: Replaced by GELU/SiLU in modern Transformers

GELU: GELU(x) = x × Φ(x)    where Φ is the standard normal CDF
  Range: (-0.17, ∞)  approximately
  Pros:  Smooth, non-monotonic, probabilistic interpretation
  Cons:  Slightly more expensive to compute than ReLU
  Used:  BERT, GPT-2, many Transformers
  In LLMs: Used in BERT-family models

SiLU (Swish): SiLU(x) = x × sigmoid(x) = x / (1 + e^(-x))
  Range: (-0.28, ∞)  approximately
  Pros:  Smooth, self-gated, works great in practice
  Cons:  Needs exp() → uses SFU on GPU (from Day 2!)
  Used:  LLaMA, Mistral, most modern LLMs
  In LLMs: THE activation function in modern LLMs (inside SwiGLU FFN)
  
  Hardware note: SiLU needs the exp() function → runs on SFU.
  You benchmarked this yesterday in your CUDA ops library!
  ReLU is a simple comparison → runs on CUDA cores (faster).
  But SiLU gives better model quality, so it's worth the cost.
```

---

# PART 2: THE MULTI-LAYER PERCEPTRON (MLP)

## 2.1 Stacking Layers

```
MLP = stack of layers, each doing: output = activation(W × input + b)

Example: 2-layer MLP for classifying handwritten digits (MNIST)

Input: 784 numbers (28×28 pixel image, flattened)
Hidden layer: 128 neurons
Output layer: 10 neurons (one per digit 0-9)

  ┌─────────┐     ┌───────────┐     ┌───────────┐     ┌──────────┐
  │  Input   │     │  Hidden   │     │   Output  │     │Prediction│
  │  784     │────→│  128      │────→│   10      │────→│  digit 7 │
  │  pixels  │  W₁ │  neurons  │  W₂ │  scores   │     │          │
  └─────────┘     └───────────┘     └───────────┘     └──────────┘
                   activation         softmax
                   (ReLU)             (probabilities)

Forward pass:
  h = ReLU(W₁ × x + b₁)      [784] → [784×128] → [128] → ReLU → [128]
  y = softmax(W₂ × h + b₂)   [128] → [128×10]  → [10]  → softmax → [10]

Matrix shapes:
  W₁: [128 × 784]  = 100,352 parameters
  b₁: [128]        = 128 parameters
  W₂: [10 × 128]   = 1,280 parameters
  b₂: [10]         = 10 parameters
  Total:            = 101,770 parameters (~102K)

Compare with LLaMA-7B: 6,700,000,000 parameters. That's 66,000x larger!
```

## 2.2 How This Connects to LLMs

```
The FFN (Feed-Forward Network) inside every Transformer layer IS an MLP!

LLaMA FFN:
  up   = SiLU(x × W_up)        [4096] → [4096×11008] → SiLU → [11008]
  gate = x × W_gate             [4096] → [4096×11008] → [11008]
  h    = up * gate              element-wise multiply → [11008]
  out  = h × W_down             [11008] → [11008×4096] → [4096]

This is a 2-layer MLP with SiLU activation and a gating mechanism!
The pattern is identical to what you're building today,
just with much larger matrices and a fancier activation.
```

---

# PART 3: BUILD AN MLP FROM SCRATCH IN NUMPY

No PyTorch. No frameworks. Just math and NumPy. This is how you truly understand it.

## 3.1 The Forward Pass

```python
import numpy as np

# ============ HELPER FUNCTIONS ============

def relu(x):
    """ReLU activation: max(0, x)"""
    return np.maximum(0, x)

def relu_derivative(x):
    """Derivative of ReLU: 1 if x > 0, else 0"""
    return (x > 0).astype(float)

def softmax(x):
    """Convert scores to probabilities"""
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # numerical stability
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def cross_entropy_loss(probs, labels):
    """How wrong are our predictions?"""
    n = labels.shape[0]
    # probs[i, labels[i]] = probability assigned to the correct class
    correct_probs = probs[np.arange(n), labels]
    return -np.mean(np.log(correct_probs + 1e-8))

# ============ THE MLP CLASS ============

class MLP_from_scratch:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights randomly (small values)
        # Xavier initialization: scale by sqrt(2 / fan_in)
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros(output_size)
        
        print(f"MLP created:")
        print(f"  Layer 1: [{input_size} × {hidden_size}] + [{hidden_size}] = {input_size*hidden_size + hidden_size} params")
        print(f"  Layer 2: [{hidden_size} × {output_size}] + [{output_size}] = {hidden_size*output_size + output_size} params")
        print(f"  Total: {input_size*hidden_size + hidden_size + hidden_size*output_size + output_size} parameters")
    
    def forward(self, X):
        """
        Forward pass: input → hidden → output
        X shape: [batch_size, input_size]
        """
        # Layer 1: linear + ReLU
        self.z1 = X @ self.W1 + self.b1          # [batch, hidden]
        self.a1 = relu(self.z1)                    # [batch, hidden]
        
        # Layer 2: linear + softmax
        self.z2 = self.a1 @ self.W2 + self.b2     # [batch, output]
        self.a2 = softmax(self.z2)                 # [batch, output] (probabilities)
        
        # Save input for backward pass
        self.X = X
        
        return self.a2
    
    def backward(self, labels, learning_rate=0.01):
        """
        Backward pass: compute gradients and update weights
        This IS backpropagation — the chain rule applied layer by layer
        """
        batch_size = labels.shape[0]
        
        # ---- Output layer gradient ----
        # For softmax + cross-entropy, the gradient simplifies to:
        # dL/dz2 = predictions - one_hot(labels)
        dz2 = self.a2.copy()
        dz2[np.arange(batch_size), labels] -= 1   # subtract 1 from correct class
        dz2 /= batch_size                          # average over batch
        
        # Gradients for W2 and b2
        dW2 = self.a1.T @ dz2                      # [hidden × output]
        db2 = np.sum(dz2, axis=0)                   # [output]
        
        # ---- Hidden layer gradient (chain rule!) ----
        # dL/da1 = dz2 × W2^T  (propagate gradient back through W2)
        da1 = dz2 @ self.W2.T                      # [batch × hidden]
        
        # dL/dz1 = dL/da1 * relu_derivative(z1)
        dz1 = da1 * relu_derivative(self.z1)       # [batch × hidden]
        
        # Gradients for W1 and b1
        dW1 = self.X.T @ dz1                       # [input × hidden]
        db1 = np.sum(dz1, axis=0)                   # [hidden]
        
        # ---- Update weights (gradient descent) ----
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
```

## 3.2 Train It on a Simple Dataset

```python
# Generate synthetic data: 2D points belonging to 3 classes
np.random.seed(42)
N = 300  # 100 points per class
D = 2    # 2D input
K = 3    # 3 classes

X = np.zeros((N, D))
y = np.zeros(N, dtype=int)

# Class 0: cluster around (-1, -1)
X[:100] = np.random.randn(100, D) * 0.4 + [-1, -1]
y[:100] = 0

# Class 1: cluster around (1, -1)
X[100:200] = np.random.randn(100, D) * 0.4 + [1, -1]
y[100:200] = 1

# Class 2: cluster around (0, 1)
X[200:] = np.random.randn(100, D) * 0.4 + [0, 1]
y[200:] = 2

print(f"Dataset: {N} points, {D} features, {K} classes")
print(f"X shape: {X.shape}, y shape: {y.shape}")

# Create and train the MLP
model = MLP_from_scratch(input_size=2, hidden_size=32, output_size=3)

# Training loop
losses = []
for epoch in range(200):
    # Forward pass
    probs = model.forward(X)
    
    # Compute loss
    loss = cross_entropy_loss(probs, y)
    losses.append(loss)
    
    # Backward pass (updates weights)
    model.backward(y, learning_rate=0.5)
    
    # Compute accuracy
    predictions = np.argmax(probs, axis=1)
    accuracy = np.mean(predictions == y) * 100
    
    if epoch % 20 == 0:
        print(f"Epoch {epoch:3d} | Loss: {loss:.4f} | Accuracy: {accuracy:.1f}%")

print(f"\nFinal: Loss={losses[-1]:.4f}, Accuracy={accuracy:.1f}%")
print(f"\nThis is EXACTLY how LLMs train — just with 7 billion parameters")
print(f"instead of {model.W1.size + model.b1.size + model.W2.size + model.b2.size}.")
```

## 3.3 What Each Line Does to the Hardware (Connecting to Day 1-4)

```
Line of code                    What happens on hardware
──────────────────────────────────────────────────────────────────
self.z1 = X @ self.W1 + self.b1
  X @ self.W1                   Matrix multiply → CUDA cores or Tensor Cores
                                (on GPU: thousands of threads in parallel)
  + self.b1                     Element-wise add → one CUDA kernel
                                (on GPU: each thread adds one element)

self.a1 = relu(self.z1)
  np.maximum(0, x)              Element-wise comparison → one CUDA kernel
                                (same as your vec_relu from yesterday!)

self.z2 = self.a1 @ self.W2
  Another matrix multiply       Same as above

self.a2 = softmax(self.z2)
  np.exp(x)                     Uses SFU for exp() (from Day 2!)
  np.sum(...)                   Parallel reduction (you'll implement Week 3)
  exp / sum                     Element-wise divide

loss = -np.mean(np.log(probs))
  np.log()                      Uses SFU for log()
  np.mean()                     Parallel reduction (sum / N)

TOTAL per training step:
  2 matrix multiplies (forward)
  2 matrix multiplies (backward) 
  Several element-wise operations
  A few reductions (softmax, loss)
  
On CPU (NumPy): each operation runs sequentially
On GPU (PyTorch): each operation launches a kernel → thousands of threads
This is why training on GPU is 10-100x faster.
```

---

# PART 4: REBUILD IT IN PYTORCH (on GPU!)

Now let's build the exact same network in PyTorch and run it on the GPU.

## 4.1 PyTorch MLP with nn.Module

```python
import torch
import torch.nn as nn
import torch.optim as optim

# ============ DEFINE THE MODEL ============

class MLP(nn.Module):
    """
    nn.Module is the base class for ALL neural networks in PyTorch.
    You define layers in __init__ and the forward pass in forward().
    PyTorch automatically computes the backward pass (autograd)!
    """
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()   # always call parent __init__
        
        # nn.Linear = one fully-connected layer (matrix multiply + bias)
        # It creates W and b automatically with proper initialization
        self.layer1 = nn.Linear(input_size, hidden_size)   # W1: [input × hidden], b1: [hidden]
        self.layer2 = nn.Linear(hidden_size, output_size)  # W2: [hidden × output], b2: [output]
        self.relu = nn.ReLU()
        
        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        print(f"PyTorch MLP: {total_params} parameters")
    
    def forward(self, x):
        """
        Define the forward pass. PyTorch tracks all operations
        so it can automatically compute gradients (backward pass).
        """
        x = self.layer1(x)    # linear: x @ W1.T + b1
        x = self.relu(x)      # activation: max(0, x)
        x = self.layer2(x)    # linear: x @ W2.T + b2
        return x               # raw scores (logits), NOT softmax
        # Note: CrossEntropyLoss in PyTorch includes softmax internally

# ============ CREATE MODEL AND MOVE TO GPU ============

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = MLP(input_size=2, hidden_size=32, output_size=3).to(device)
# .to(device) moves ALL weights to GPU memory (cudaMemcpy under the hood!)

# ============ PREPARE DATA ============
# Reuse the NumPy data from Part 3, convert to PyTorch tensors

X_tensor = torch.FloatTensor(X).to(device)     # [300, 2] → GPU
y_tensor = torch.LongTensor(y).to(device)       # [300]    → GPU

# ============ DEFINE LOSS AND OPTIMIZER ============

criterion = nn.CrossEntropyLoss()
# This computes: softmax(logits) → cross_entropy(probs, labels)
# Combines softmax + loss in one step (numerically more stable)

optimizer = optim.Adam(model.parameters(), lr=0.01)
# Adam optimizer: adaptive learning rate per parameter
# Better than plain SGD — converges faster, less sensitive to learning rate
# This is what ALL LLM training uses (specifically AdamW variant)

# ============ TRAINING LOOP ============

print("\nTraining on GPU..." if device.type == 'cuda' else "\nTraining on CPU...")

for epoch in range(200):
    # Forward pass
    logits = model(X_tensor)           # model(x) calls model.forward(x)
    loss = criterion(logits, y_tensor) # compute loss
    
    # Backward pass
    optimizer.zero_grad()    # clear old gradients (they accumulate by default!)
    loss.backward()          # compute gradients (backpropagation via autograd)
    optimizer.step()         # update weights: w = w - lr * grad
    
    if epoch % 20 == 0:
        predictions = logits.argmax(dim=1)     # which class has highest score?
        accuracy = (predictions == y_tensor).float().mean() * 100
        print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f} | Accuracy: {accuracy:.1f}%")

# Final result
predictions = model(X_tensor).argmax(dim=1)
accuracy = (predictions == y_tensor).float().mean() * 100
print(f"\nFinal Accuracy: {accuracy:.1f}%")
print(f"GPU memory used: {torch.cuda.memory_allocated()/1024**2:.1f} MB")
```

## 4.2 What PyTorch Does Under the Hood

```
YOUR CODE                          WHAT PYTORCH ACTUALLY DOES
──────────────────────────────────────────────────────────────────
model.to(device)                   cudaMemcpy for W1, b1, W2, b2 → GPU HBM

X_tensor.to(device)                cudaMemcpy for input data → GPU HBM

logits = model(X_tensor)           
  self.layer1(x)                   Launches cuBLAS GEMM kernel (matrix multiply)
                                   → runs on Tensor Cores if FP16/BF16
  self.relu(x)                     Launches element-wise kernel (your vec_relu!)
  self.layer2(x)                   Another cuBLAS GEMM kernel

loss = criterion(logits, labels)   Launches softmax kernel + cross-entropy kernel
                                   → SFU for exp(), parallel reduction for sum

loss.backward()                    PyTorch replays the computation graph BACKWARDS
                                   Launches kernels for each gradient:
                                   → GEMM for weight gradients
                                   → element-wise for activation gradients
                                   → All tracked by autograd automatically

optimizer.step()                   Launches kernel to update each parameter:
                                   w = w - lr * grad (for SGD)
                                   More complex for Adam (momentum + variance)
```

---

# PART 5: AUTOGRAD — PyTorch's Magic

## 5.1 How It Works

```python
# When you create a tensor with requires_grad=True,
# PyTorch tracks every operation on it.

a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(3.0, requires_grad=True)

c = a * b       # PyTorch remembers: c = a * b
d = c + a       # PyTorch remembers: d = c + a = a*b + a

d.backward()     # Compute all gradients

print(a.grad)    # dd/da = b + 1 = 3 + 1 = 4.0 ✓
print(b.grad)    # dd/db = a = 2.0 ✓

# PyTorch built a computation graph:
#   a ──→ [*] ──→ c ──→ [+] ──→ d
#   b ──↗           a ──↗
#
# backward() walks this graph in reverse, applying chain rule at each node.
# This IS backpropagation. You don't write it — PyTorch does it for you.
```

## 5.2 Why This Matters for LLMs

```
A Transformer with 32 layers has a computation graph like:

  input → embed → [attention → FFN] × 32 → output → loss
          ↓         ↓          ↓              ↓        ↓
         W_emb    W_Q,K,V    W_up,gate,down  W_out    cross_entropy

  loss.backward() walks back through ALL of these:
  ∂loss/∂W_out → ∂loss/∂W_down → ... → ∂loss/∂W_Q_layer1 → ∂loss/∂W_emb

  Computing gradients for 7 BILLION parameters automatically.
  Thousands of CUDA kernels launched in sequence.
  Each kernel runs on thousands of threads.
  
  THIS is why GPU memory is so important during training:
  PyTorch must STORE the intermediate values (activations)
  from the forward pass to use in the backward pass.
  That's where the 10-50+ GB of activation memory comes from.
```

## 5.3 Key Autograd Patterns

```python
# PATTERN 1: torch.no_grad() — skip gradient tracking (inference)
with torch.no_grad():
    output = model(input)  # faster, uses less memory
# Use during: inference, evaluation, anything that doesn't need gradients

# PATTERN 2: .detach() — stop gradient flow
hidden = encoder(x)
hidden_detached = hidden.detach()  # gradient won't flow back through encoder
output = decoder(hidden_detached)

# PATTERN 3: optimizer.zero_grad() — MUST call before each backward
# Without it, gradients ACCUMULATE across iterations (usually a bug)
optimizer.zero_grad()   # clear old gradients
loss.backward()          # compute new gradients
optimizer.step()         # update weights

# PATTERN 4: .item() — get a Python number from a 0-dim tensor
loss_value = loss.item()  # converts tensor(0.342) → float 0.342
# Use for logging/printing (doesn't track gradients)
```

---

# PART 6: MNIST — Your First Real Task

## 6.1 Loading MNIST

```python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# MNIST: 60,000 training images + 10,000 test images
# Each image: 28×28 pixels, grayscale, handwritten digit (0-9)

transform = transforms.Compose([
    transforms.ToTensor(),           # convert PIL image → tensor [0,1]
    transforms.Normalize((0.5,), (0.5,))  # normalize to [-1, 1]
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# DataLoader: handles batching, shuffling, parallel loading
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print(f"Training samples: {len(train_dataset)}")
print(f"Test samples:     {len(test_dataset)}")
print(f"Image shape:      {train_dataset[0][0].shape}")  # [1, 28, 28]
print(f"Batches per epoch: {len(train_loader)}")  # 60000/64 = 938
```

## 6.2 MNIST MLP

```python
class MNIST_MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()           # [1, 28, 28] → [784]
        self.layer1 = nn.Linear(784, 256)     # 784 → 256
        self.layer2 = nn.Linear(256, 128)     # 256 → 128
        self.layer3 = nn.Linear(128, 10)      # 128 → 10 (digits)
        self.relu = nn.ReLU()
        
        total = sum(p.numel() for p in self.parameters())
        print(f"MNIST MLP: {total:,} parameters")
    
    def forward(self, x):
        x = self.flatten(x)       # flatten image to vector
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)        # output logits (no activation)
        return x

# Create model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MNIST_MLP().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(5):
    model.train()  # set to training mode (affects dropout, batchnorm)
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        # Forward
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track accuracy
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    train_acc = 100. * correct / total
    avg_loss = total_loss / len(train_loader)
    
    # Test accuracy
    model.eval()  # set to evaluation mode
    test_correct = 0
    test_total = 0
    with torch.no_grad():  # no gradients needed for evaluation
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()
    
    test_acc = 100. * test_correct / test_total
    print(f"Epoch {epoch+1}/5 | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.1f}% | Test Acc: {test_acc:.1f}%")

print(f"\nGPU memory used: {torch.cuda.memory_allocated()/1024**2:.1f} MB")
print(f"Expected test accuracy: ~97-98% (just from a simple MLP!)")
```

---

# PART 7: TODAY'S MINI-PROJECT 🔨

## Project: "CPU vs GPU Neural Network Showdown"

Train the SAME MNIST model on CPU and GPU, time everything, and show
exactly where the GPU wins and by how much.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time

# Setup data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )
    def forward(self, x):
        return self.net(x)

def train_and_time(device_name, num_epochs=3, batch_size=64):
    device = torch.device(device_name)
    
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                        num_workers=2 if device_name == 'cuda' else 0,
                        pin_memory=(device_name == 'cuda'))
    test_loader = DataLoader(test_dataset, batch_size=256)
    
    model = SimpleMLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    params = sum(p.numel() for p in model.parameters())
    
    # Training
    if device_name == 'cuda':
        torch.cuda.synchronize()
    start = time.time()
    
    for epoch in range(num_epochs):
        model.train()
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
    
    if device_name == 'cuda':
        torch.cuda.synchronize()
    train_time = time.time() - start
    
    # Test accuracy
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            pred = model(images).argmax(1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    
    accuracy = 100 * correct / total
    
    return {
        'device': device_name.upper(),
        'params': params,
        'epochs': num_epochs,
        'time': train_time,
        'accuracy': accuracy,
        'samples_per_sec': len(train_dataset) * num_epochs / train_time,
    }

# Run both
print("=" * 60)
print("        CPU vs GPU Neural Network Showdown")
print("=" * 60)

results = []
for dev in ['cpu', 'cuda']:
    if dev == 'cuda' and not torch.cuda.is_available():
        print("No GPU available, skipping")
        continue
    print(f"\nTraining on {dev.upper()}...")
    r = train_and_time(dev)
    results.append(r)
    print(f"  Time: {r['time']:.2f}s | Accuracy: {r['accuracy']:.1f}% | {r['samples_per_sec']:.0f} samples/sec")

if len(results) == 2:
    speedup = results[0]['time'] / results[1]['time']
    print(f"\n{'=' * 60}")
    print(f"  GPU is {speedup:.1f}x faster than CPU!")
    print(f"  CPU: {results[0]['time']:.2f}s ({results[0]['samples_per_sec']:.0f} samples/sec)")
    print(f"  GPU: {results[1]['time']:.2f}s ({results[1]['samples_per_sec']:.0f} samples/sec)")
    print(f"  Both achieve ~{results[1]['accuracy']:.1f}% accuracy")
    print(f"{'=' * 60}")
    print(f"\n  For this small MLP, GPU speedup is modest (~2-5x).")
    print(f"  For large LLMs (billions of params), GPU speedup is 100-1000x")
    print(f"  because the matrix multiplies are MUCH larger → Tensor Cores dominate.")
```

---

# CHECKLIST

After Day 5:
- [ ] Understand what a neuron does (weighted sum + bias + activation)
- [ ] Know 5 activation functions and which LLMs use (SiLU for LLaMA, GELU for BERT)
- [ ] Understand why activation functions exist (non-linearity, universal approximation)
- [ ] Built MLP from scratch in NumPy (forward + backward pass manually)
- [ ] Understand backpropagation as chain rule applied layer by layer
- [ ] Rebuilt the MLP in PyTorch using nn.Module, nn.Linear
- [ ] Understand autograd: PyTorch tracks operations and computes gradients automatically
- [ ] Know the training loop: forward → loss → zero_grad → backward → step
- [ ] Trained MNIST digit classifier achieving ~97% accuracy
- [ ] Benchmarked CPU vs GPU training speed
- [ ] Understand how each Python line maps to GPU hardware (matmul → Tensor Cores, etc.)

**Tomorrow is SATURDAY — Weekly Project Day!**
You'll combine everything from this week into a bigger project.
Check the Readme.md for the Week 1 Saturday project.

---

*Status: ⬜ NOT YET COMPLETED*
*Date completed: ___________*
