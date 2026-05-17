# W1D3 Revision — Math Foundations for LLMs
## ⏱️ 10-minute speed-revision sheet

> **One-line summary:** LLMs are massive matrix multiplications and softmax probability distributions, optimized via gradient descent (chain rule = backprop), and graded by cross-entropy. Every operation has a geometric meaning — not just a formula.

---

## 🧠 30-Second Mental Model

```
LLM forward pass = repeated transformations in "meaning space":
  
  text  →  tokens (integers)
        →  embeddings (vectors in a 4096-D space)
        →  matmul + activation + matmul + ...   (Transformer blocks)
        →  logits (raw scores per token in vocabulary)
        →  softmax → probability distribution over next token
  
LLM training = nudge millions of weights to make the right token's
probability go up. Done via gradient descent (chain rule).
```

If you remember **only one thing**: a Transformer is a stack of matrix multiplications with non-linearities sprinkled in, ending in a softmax. All four math fields (linear algebra, calculus, probability, information theory) plug directly into one of those steps.

---

## 1️⃣ Linear Algebra — The Language of LLMs (90 sec)

| Concept | Geometric meaning | LLM use |
|---|---|---|
| **Vector** | A point/arrow in N-dim space | Token embedding (4096-D in LLaMA) |
| **Matrix multiply** `Y = X · W` | Transform X into a new space defined by W | Linear layers, attention projections |
| **Transpose** `M^T` | Flip rows ↔ columns | `Q · K^T` in attention |
| **Dot product** `a · b` | Similarity (= `‖a‖·‖b‖·cos θ`) | Attention scores between Q and K |
| **Norm** `‖v‖` | Length / magnitude of vector | LayerNorm/RMSNorm divides by this |
| **Eigenvector** | A direction unchanged by transformation (only scaled) | PCA, conditioning analysis (mostly theoretical for LLMs) |

### Key facts to remember:
- **Matmul shape rule:** `[M×K] · [K×N] = [M×N]`. The K's must match.
- **Dot product is the heart of attention:** large dot product ⇒ Q and K are similar ⇒ that token attends to that key
- **Why scale by `√d_k`** in attention: keeps dot-product magnitudes reasonable so softmax doesn't saturate
- LLaMA-7B's hidden size is **4096-dimensional** — your "meaning space" is a 4096-D space

---

## 2️⃣ Calculus — How Models Learn (60 sec)

### Gradient = "which direction increases the function fastest"
- For a function `f(x, y, z, ...)` of many variables, the gradient `∇f = [∂f/∂x, ∂f/∂y, ...]` points in the direction of steepest increase
- Gradient **descent**: take small steps in the *opposite* direction → loss goes down
- `w_new = w_old - lr · ∇L(w)`

### Chain rule = how backprop works
- If `y = f(g(x))`, then `dy/dx = (df/dg) · (dg/dx)`
- For a deep network with N layers, the gradient with respect to layer 1's weights = product of N partial derivatives, computed by walking backward through the graph
- **This IS backpropagation** — you don't invent a new algorithm, you just apply the chain rule layer-by-layer

### Jacobian = full matrix of partial derivatives
- For a vector function `f: R^n → R^m`, the Jacobian is an `m × n` matrix of all `∂f_i / ∂x_j`
- Modern auto-grad frameworks (PyTorch) never materialize the full Jacobian — they only compute Jacobian-vector products

### Memorable analogy:
- **Gradient descent = hiking down a foggy mountain.** You can't see far, but you can feel the slope under your feet. Step in the steepest-down direction. Repeat. Eventually you reach a valley (local minimum).

---

## 3️⃣ Probability — The Output of Every LLM (60 sec)

### Softmax — turn raw scores into a probability distribution
```
softmax(z_i) = exp(z_i) / Σ_j exp(z_j)
```
- Properties: every output ∈ (0, 1), all outputs sum to 1
- Numerical stability: subtract the max first → `exp(z_i - max(z))` (prevents overflow)
- LLMs always end with `softmax(logits)` over the vocabulary (50K+ tokens) to get next-token probabilities

### Temperature — controls randomness
```
softmax(z / T)
```
- **T = 1**: normal probabilities
- **T → 0**: sharp distribution (basically argmax — always picks most likely token, deterministic)
- **T → ∞**: flat distribution (uniform random)
- LLM sampling: `T = 0.7-1.0` typical for creative; `T = 0.1` for deterministic

### Sampling strategies (briefly):
- **Greedy** (`T=0` or argmax): always pick top token. Boring, repetitive.
- **Top-k**: keep only top-k tokens, renormalize. Common k = 40-50.
- **Top-p (nucleus)**: keep the smallest set of tokens whose cumulative prob ≥ p. Common p = 0.9.

---

## 4️⃣ Information Theory — The Loss Function of LLMs (90 sec)

### Cross-Entropy — how wrong are our predictions?
```
CE(p, q) = -Σ_i  p_i · log(q_i)
```
- `p` = true distribution (one-hot for the correct token)
- `q` = predicted distribution (softmax output)
- For one-hot labels, this simplifies to: `CE = -log(q[correct_idx])`
- **Lower is better.** Perfect prediction → CE = 0. Uniform random over V tokens → CE ≈ log(V).

### Why log? (the intuition)
- log makes products of probabilities (per-token) → sums → numerically stable + mathematically clean
- log heavily penalizes confident wrong predictions: `-log(0.01) ≈ 4.6` vs `-log(0.5) ≈ 0.69`
- **This is why LLMs are sometimes "afraid" to be confident — being confident and wrong is heavily punished by CE.**

### KL Divergence — distance between two distributions
```
KL(p ‖ q) = Σ_i  p_i · log(p_i / q_i)
```
- Measures how "different" `q` is from `p`
- Asymmetric: `KL(p‖q) ≠ KL(q‖p)`
- KL ≥ 0, with 0 only when p = q exactly
- **Used in:** RLHF (KL penalty keeps fine-tuned model close to base model), DPO loss, knowledge distillation

### Entropy — how uncertain is a single distribution?
```
H(p) = -Σ_i  p_i · log(p_i)
```
- Max when uniform (most uncertain). Min when one-hot (most certain).
- **Connection:** `CE(p, q) = H(p) + KL(p ‖ q)` — cross-entropy = base uncertainty + extra penalty for being wrong

### Perplexity — the LLM-eval-friendly form of CE
```
perplexity = exp(CE)
```
- Interpretation: "the model is as confused as if it were picking uniformly among `perplexity` tokens"
- LLaMA-7B's perplexity on Wikipedia ≈ 5-7 (very confident next-token predictor)

---

## 5️⃣ Operations That Matter for LLMs — Summary (45 sec)

| Operation | Where in Transformer | Math foundation |
|---|---|---|
| **Embedding lookup** | Token IDs → vectors | Linear algebra (matrix indexing) |
| **Matmul** | All linear layers, attention projections | Linear algebra |
| **Dot product** | `Q · K^T` for attention scores | Linear algebra |
| **Scale by `1/√d_k`** | Pre-softmax in attention | Probability (prevents softmax saturation) |
| **Softmax** | Attention weights, final output | Probability |
| **Activation (GELU/SiLU)** | Inside FFN | Calculus (smooth, differentiable) |
| **LayerNorm / RMSNorm** | Before each block | Linear algebra (normalization) |
| **Cross-entropy** | Loss function during training | Information theory |
| **Gradient** | Backprop signal | Calculus (chain rule) |

---

## 6️⃣ The Core Equations (memorize these)

```
Attention(Q, K, V) = softmax(Q · K^T / √d_k) · V        ← THE Transformer line

softmax(z_i) = exp(z_i) / Σ_j exp(z_j)                   ← turn scores → probs

CE(target, predicted) = -log(predicted[target_idx])      ← LLM loss

w_new = w_old - lr · ∂L/∂w                                ← gradient descent step
```

---

## 🔍 Quick Recall (60 sec)

1. What's the shape rule for matmul `A · B = C`? *(A:[M×K] · B:[K×N] = C:[M×N])*
2. What does `Q · K^T` measure in attention? *(similarity between query and key vectors)*
3. Why divide attention scores by `√d_k`? *(prevents softmax saturation)*
4. What's the formula for softmax? *(exp(z_i) / Σ exp(z_j))*
5. What's cross-entropy for one-hot labels? *(-log(q[correct]))*
6. What is KL divergence used for in LLMs? *(RLHF KL penalty, DPO, distillation)*
7. What is the chain rule's role in deep learning? *(it IS backpropagation)*
8. What's the temperature trick? *(divide logits by T before softmax — controls sharpness)*
9. What's perplexity? *(exp of cross-entropy — "how confused is the model")*
10. What dimension is LLaMA-7B's "meaning space"? *(4096-D)*

---

## 🎯 If You Remember Only Three Things

1. **A Transformer is `softmax(Q · K^T / √d) · V` + matmul + activation, repeated.** Every word in that sentence is one of the math concepts you learned today.
2. **Backprop is just the chain rule applied layer-by-layer.** No magic. PyTorch's `.backward()` walks the computation graph in reverse, multiplying partial derivatives.
3. **Cross-entropy is the LLM loss because it heavily penalizes confident-wrong predictions** via the `log`. Perplexity = `exp(CE)` is the same thing in human-friendly form.

---

*Revision file generated from `DAY_3.md`. For deep dive + worked code, see the original DAY_3.md.*
