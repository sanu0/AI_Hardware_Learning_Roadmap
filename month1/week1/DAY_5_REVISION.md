# W1D5 Revision — Neural Networks from First Principles
## ⏱️ 10-minute speed-revision sheet

> **One-line summary:** A neural network is **stacked linear layers + non-linear activations**, trained by **gradient descent** with gradients computed via **chain rule (backpropagation)**. Today you built one in NumPy from scratch, then rebuilt it in PyTorch on a GPU — and saw both produce the same answer.

---

## 🧠 30-Second Mental Model

```
ONE NEURON:    output = activation(W · x + b)
                       │            │   │
                       │            │   └─ bias (offset)
                       │            └─── weights (learnable matrix)
                       └────────────── non-linearity (ReLU, GELU, SiLU, ...)

ONE LAYER:     a vector of neurons computed in parallel = matmul + bias + activation

MLP:           Input → [Linear → Activation] × N → Output
                                                    │
                                            (last layer: softmax for classification,
                                             linear for regression)
                                            
TRAINING LOOP: forward pass → loss → backward pass (chain rule) → update weights
```

If you remember **only one thing**: a neural network is `repeat(matmul + activation)`, trained by `loss → ∂loss/∂weights via chain rule → step weights downhill`.

---

## 1️⃣ The Neuron — The Atomic Unit (30 sec)

```
output = f(w · x + b)
       = f(Σ w_i x_i + b)

where f is a non-linear activation function
```

- Without `f` (the activation), stacking layers gives you... another linear layer. The whole point of activations is to introduce **non-linearity** so the network can learn complex shapes.
- Modern LLMs use **SiLU (Swish)** or **GELU**.

---

## 2️⃣ Activation Functions — Memorize These (60 sec)

| Activation | Formula | Range | Use case |
|---|---|---|---|
| **ReLU** | `max(0, x)` | `[0, ∞)` | Default for CNNs, simple MLPs (cheap, "dies" at 0 input) |
| **Sigmoid** | `1 / (1 + e^-x)` | `(0, 1)` | Binary output (rarely in hidden layers anymore) |
| **Tanh** | `(e^x - e^-x) / (e^x + e^-x)` | `(-1, 1)` | RNNs (older), centered at 0 |
| **Softmax** | `e^z_i / Σ e^z_j` | sums to 1 | Final layer for multi-class classification |
| **GELU** | `x · Φ(x)` (Φ = Gaussian CDF) | smooth | Transformers (BERT, GPT) |
| **SiLU / Swish** | `x · sigmoid(x)` | smooth | LLaMA's FFN |

### Why GELU/SiLU > ReLU for LLMs?
- ReLU has a hard zero at x < 0 → gradient is 0 → "dead neurons"
- GELU/SiLU are smooth, differentiable everywhere → cleaner gradients during training

### Memorable picture:
- **ReLU**: hockey stick (flat then linear up)
- **Sigmoid**: S-curve (saturates)
- **GELU/SiLU**: looks like ReLU but smoothly bent at the corner — keeps a small negative tail

---

## 3️⃣ The MLP (Multi-Layer Perceptron) (45 sec)

```python
# An MLP with hidden_size=64, 3 classes:
class MLP:
    Z1 = X @ W1 + b1                  # [batch, 64]   — linear
    A1 = ReLU(Z1)                      # [batch, 64]   — activation
    Z2 = A1 @ W2 + b2                  # [batch, 3]    — linear
    probs = softmax(Z2)                # [batch, 3]    — probabilities
```

### Shape rule (memorize):
- `Linear(in_features=A, out_features=B)` has weights of shape `[A, B]`
- For input `X` of shape `[batch, A]`: `X @ W = output` of shape `[batch, B]`

### Connection to LLMs:
- Inside every Transformer block there's an **FFN**: typically `Linear(d → 4d) → SiLU/GELU → Linear(4d → d)` — that's literally a 2-layer MLP
- The "4× expansion" is standard (`d=4096` → `d_ff=16384` in LLaMA-7B)

---

## 4️⃣ Forward Pass + Loss (45 sec)

### Forward pass = run input through the network → predicted probabilities
### Loss = how wrong are the predictions?

For multi-class classification:
```python
# Numerically stable softmax
def softmax(x):
    x = x - x.max(axis=-1, keepdims=True)   # prevent overflow
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)

# Cross-entropy (one-hot labels)
def cross_entropy(probs, labels):
    eps = 1e-9                              # prevent log(0)
    return -np.mean(np.log(probs[np.arange(len(labels)), labels] + eps))
```

### Why subtract max in softmax?
- `exp(1000)` overflows. `exp(z - max)` keeps numbers bounded since the largest is now 0.
- The softmax output is unchanged because `e^a / e^b = e^(a-b)` (the constant cancels).

### Why `+ eps` in cross-entropy?
- If model predicts `prob = 0` for the correct class, `log(0) = -∞` → NaN. Adding tiny eps avoids this.

---

## 5️⃣ Backpropagation — The Chain Rule, Applied Layer-by-Layer (90 sec)

### The intuition (no calculus required first):
1. Loss is at the end of the network. We want to know: *"if I nudge weight `W1[i,j]` by ε, how much does loss change?"*
2. That's exactly the partial derivative `∂L/∂W1[i,j]`.
3. To compute it, we walk **backward** through the network, applying the chain rule.

### The chain rule mental model:
```
L  ←  Z2  ←  A1  ←  Z1  ←  W1
∂L/∂W1 = (∂L/∂Z2) · (∂Z2/∂A1) · (∂A1/∂Z1) · (∂Z1/∂W1)
```
You compute each factor separately, then multiply.

### The math for our MLP (forward → backward, mirroring):
```
Forward:                          Backward (gradients):
Z1 = X @ W1 + b1                  dZ2 = (probs - one_hot(labels)) / N
A1 = ReLU(Z1)                     dW2 = A1.T @ dZ2
Z2 = A1 @ W2 + b2                 db2 = dZ2.sum(0)
probs = softmax(Z2)               dA1 = dZ2 @ W2.T
                                  dZ1 = dA1 * (Z1 > 0)        # ReLU derivative
                                  dW1 = X.T @ dZ1
                                  db1 = dZ1.sum(0)
```

### One beautiful identity (memorize):
- For **softmax + cross-entropy combined**: `dZ2 = (probs - one_hot(labels)) / N` — the gradient simplifies to "predicted minus true" / batch size. This is why these two are always paired in classification.

### Update step (SGD):
```python
W1 -= lr * dW1
b1 -= lr * db1
W2 -= lr * dW2
b2 -= lr * db2
```

---

## 6️⃣ NumPy → PyTorch (the rebuild) (45 sec)

```python
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

device = "cuda" if torch.cuda.is_available() else "cpu"
model = MLP(2, 64, 3).to(device)         # .to(device) = cudaMemcpy under the hood

criterion = nn.CrossEntropyLoss()         # combines softmax + CE in one numerically stable step
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

for epoch in range(100):
    optimizer.zero_grad()                 # PATTERN: clear old grads
    logits = model(X)                     # forward
    loss = criterion(logits, y)           # loss
    loss.backward()                       # backward (autograd computes ALL gradients)
    optimizer.step()                      # update weights
```

### What PyTorch does that you did manually in NumPy:
- `loss.backward()` = walks the computation graph in reverse, applies chain rule, fills `.grad` on every tensor with `requires_grad=True`
- `optimizer.step()` = applies the update rule (SGD, Adam, AdamW, ...)
- `.to(device)` = `cudaMemcpy` under the hood

---

## 7️⃣ Autograd Patterns — MEMORIZE THESE 4 (45 sec)

| Pattern | When to use |
|---|---|
| **`with torch.no_grad():`** | Inference / evaluation — skip gradient tracking, save memory + time |
| **`.detach()`** | Stop gradient flow at a specific tensor (e.g., teacher in distillation) |
| **`optimizer.zero_grad()`** | Call BEFORE every `loss.backward()` — gradients accumulate by default |
| **`.item()`** | Convert single-element tensor → Python float (for printing/logging) |

### Common training-loop bug:
- **Forgetting `optimizer.zero_grad()`** → gradients accumulate across batches → exploding updates → loss = NaN. Most common training bug.

---

## 8️⃣ MNIST Walkthrough (30 sec)

```python
# Data: 60K train + 10K test, 28×28 grayscale digits → flatten to 784
mlp = MLP(in_dim=784, hidden=128, out_dim=10).to(device)

for epoch in range(5):
    for X, y in train_loader:
        X = X.view(-1, 784).to(device)    # flatten
        y = y.to(device)
        optimizer.zero_grad()
        loss = criterion(mlp(X), y)
        loss.backward()
        optimizer.step()

# Evaluation
with torch.no_grad():                      # PATTERN 1
    acc = (mlp(X_test).argmax(1) == y_test).float().mean()
```

A 1-hidden-layer MLP achieves ~97% accuracy on MNIST in 5 epochs. **A 2-layer Transformer would do similarly well — same principle.**

---

## 9️⃣ CPU vs GPU Performance (15 sec)

For an MLP `[784 → 128 → 10]` trained 5 epochs on 60K MNIST images:
- **CPU NumPy**: ~30 seconds
- **GPU PyTorch (RTX 3090 / A100 / T4)**: ~2-3 seconds → **~10× speedup**

The win comes from: parallel matmul on Tensor Cores + parallel ReLU on CUDA cores + batching → 60K images move through the GPU as one big tensor.

---

## 🔍 Quick Recall (60 sec)

1. What does a single neuron compute? *(`activation(W · x + b)`)*
2. Why do we need activation functions? *(introduce non-linearity; without them stacked layers = one linear layer)*
3. Which activation does LLaMA's FFN use? *(SiLU / Swish)*
4. What's the FFN expansion factor in LLMs? *(4× — `d → 4d → d`)*
5. Why subtract `max` before exp in softmax? *(numerical stability — prevent overflow)*
6. What's the gradient of `softmax + cross-entropy` combined? *(`(probs - one_hot) / N`)*
7. What does `loss.backward()` do? *(walks computation graph backward, fills `.grad` on each parameter via chain rule)*
8. What does `optimizer.zero_grad()` prevent? *(accumulating gradients from previous batches)*
9. When do you use `with torch.no_grad():`? *(inference / evaluation, to skip gradient tracking)*
10. What does `.to('cuda')` do under the hood? *(`cudaMemcpy` from host to device)*

---

## 🎯 If You Remember Only Three Things

1. **A neural network is just `repeat(matmul + activation)` followed by a final softmax for classification.** The whole field is built on this. LLMs are no exception — they're just bigger and add attention.
2. **Backprop is the chain rule applied layer-by-layer.** PyTorch's `loss.backward()` does it automatically, but you've now done it by hand in NumPy. You will never be confused by a gradient bug again.
3. **The training loop is always the same 4 steps:** `zero_grad → forward → backward → step`. Forgetting `zero_grad` is the #1 training bug. This 4-step loop runs hundreds of millions of times when training an LLM.

---

*Revision file generated from `DAY_5.md`. For deep dive + full worked examples, see the original DAY_5.md.*
