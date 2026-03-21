# Week 1, Day 3: Math Foundations for LLMs
## Geometric Intuition — What Each Operation MEANS, Not Just the Formula

If you have a CS/math background, you know most of this math already.
This document skips derivations and focuses on: **what does this operation
mean geometrically, and WHERE does it show up in LLMs?**

**Time: ~2 hours reading + 30 min coding**

---

# PART 1: LINEAR ALGEBRA — The Language of LLMs

## Vectors = Points in Meaning Space

```
In LLMs, a vector IS a word's meaning.

"cat"   → [0.8, 0.1, 0.9, -0.3, ...]   (4096 numbers)
"dog"   → [0.7, 0.2, 0.8, -0.2, ...]   (nearby — similar meaning!)
"car"   → [-0.1, 0.9, -0.3, 0.7, ...]  (far away — different meaning)

Geometric meaning:
  Each of the 4096 dimensions captures SOME aspect of meaning.
  Dimension 47 might loosely correspond to "is it alive?"
  Dimension 203 might correspond to "is it physical?"
  (In practice dimensions aren't interpretable, but the geometry works)

  "king" - "man" + "woman" ≈ "queen"
  This ACTUALLY WORKS because vector arithmetic = meaning arithmetic.
```

**Where in LLMs:** The embedding table maps token IDs → vectors. Every computation in the Transformer operates on these meaning-vectors.

## Matrix Multiplication = Transformation of Meaning

```
When you multiply a vector by a matrix:

  y = W × x

Geometrically: W TRANSFORMS x into a new space.
  - Rotation: change what the dimensions mean
  - Scaling: stretch or shrink along certain directions
  - Projection: collapse some dimensions, emphasize others

In LLMs, EVERY matrix multiply is a transformation:

  Q = x × W_Q    "Transform meaning-vector into a QUERY:
                   what am I looking for?"

  K = x × W_K    "Transform meaning-vector into a KEY:
                   what do I have to offer?"

  V = x × W_V    "Transform meaning-vector into a VALUE:
                   what information do I carry?"

  FFN_up = x × W_up    "Project into a HIGHER dimensional space (4096 → 11008)
                         to capture more complex patterns"

  FFN_down = h × W_down  "Project back DOWN (11008 → 4096)
                           to compress the learned patterns"

Each weight matrix W was LEARNED during training to perform the
specific transformation that makes the model understand language.
```

## Transpose = Flip Rows and Columns

```
A = [[1, 2, 3],      A^T = [[1, 4],
     [4, 5, 6]]             [2, 5],
                             [3, 6]]

Geometric meaning: Mirror across the diagonal.

In LLMs:
  Attention scores = Q × K^T
  
  Why transpose K? Because Q is [seq_len × d_head] and K is [seq_len × d_head].
  To get [seq_len × seq_len] compatibility scores, you need:
    [seq × d] × [d × seq] = [seq × seq]
  K^T converts [seq × d] → [d × seq]. That's it.
```

## Dot Product = Similarity Measure

```
a · b = a₁b₁ + a₂b₂ + ... + aₙbₙ = |a| × |b| × cos(θ)

Geometric meaning:
  - Positive dot product → vectors point same direction → SIMILAR
  - Zero dot product → vectors are perpendicular → UNRELATED
  - Negative dot product → vectors point opposite → OPPOSITE meaning

In LLMs:
  Attention score(i,j) = Q[i] · K[j]
  
  "How much should token i attend to token j?"
  = "How similar is token i's query to token j's key?"
  = dot product of their vectors!
  
  High score → "France" key matches "is ___" query → high attention.
```

## Eigenvalues/Eigenvectors = Principal Directions of a Transformation

```
Av = λv

Geometric meaning: v is a direction that the matrix A DOESN'T ROTATE,
only scales by factor λ.

In LLMs (where it matters):
  - PCA for visualization: reduce 4096-D embeddings to 2D for plotting
  - Spectral analysis of attention patterns
  - Understanding weight matrix structure
  
  Not used directly in forward/backward pass,
  but useful for ANALYZING trained models.
```

---

# PART 2: CALCULUS — How Models Learn

## Gradient = Direction of Steepest Increase

```
∇f(x) = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]

Geometric meaning:
  The gradient points in the direction where f increases FASTEST.
  Its magnitude tells you HOW FAST it increases.

  To MINIMIZE loss: go OPPOSITE the gradient direction.
  
  w_new = w_old - lr × ∇Loss(w_old)
  
  This is gradient descent. Every LLM is trained this way.
  
  7 billion weights, each gets a gradient, each gets nudged
  in the direction that reduces the loss. Repeat trillions of times.
```

## Chain Rule = Backpropagation

```
If f(g(x)), then df/dx = df/dg × dg/dx

Geometric meaning:
  To know how x affects f, multiply how x affects g
  by how g affects f. Chain the local effects.

In LLMs:
  A Transformer has 32 layers stacked.
  To know how weight in layer 1 affects the final loss:
  
  ∂Loss/∂W₁ = ∂Loss/∂out₃₂ × ∂out₃₂/∂out₃₁ × ... × ∂out₂/∂out₁ × ∂out₁/∂W₁
  
  That's 32 multiplications chained together.
  This IS backpropagation. PyTorch autograd does this automatically.
  
  Problem: if each ∂outₖ/∂outₖ₋₁ < 1, the product shrinks → VANISHING GRADIENT
           if each > 1, the product explodes → EXPLODING GRADIENT
  
  Solution: Residual connections! output = layer(x) + x
    This makes ∂outₖ/∂outₖ₋₁ = 1 + something
    The gradient always has a "1" term that flows straight through.
    This is WHY Transformers have residual connections. Not optional.
```

## Jacobian = Matrix of All Partial Derivatives

```
If f: ℝⁿ → ℝᵐ, then J is an m×n matrix where J[i][j] = ∂fᵢ/∂xⱼ

Geometric meaning:
  The Jacobian tells you how each output changes
  when you wiggle each input. It's the "sensitivity matrix."

In LLMs:
  You rarely compute the full Jacobian (too large: 4096×4096 per layer).
  Instead, backpropagation computes Jacobian-vector products efficiently.
  This is what PyTorch autograd actually does under the hood.
```

---

# PART 3: PROBABILITY — The Output of Every LLM

## Softmax = Turn Numbers into Probabilities

```
softmax(zᵢ) = exp(zᵢ) / Σⱼ exp(zⱼ)

Geometric meaning:
  Takes ANY vector of numbers → outputs a probability distribution.
  The LARGEST input gets the LARGEST probability.
  The GAP between inputs controls how "confident" the distribution is.

  z = [5.0, 2.0, 1.0]
  softmax(z) = [0.935, 0.047, 0.017]     ← very confident (big gap)
  
  z = [2.1, 2.0, 1.9]
  softmax(z) = [0.367, 0.332, 0.301]     ← uncertain (small gap)

In LLMs — used in TWO critical places:

  1. ATTENTION WEIGHTS:
     Raw scores [0.1, 0.3, 0.8, 0.2] → softmax → [0.14, 0.18, 0.41, 0.17]
     "41% attention to token 3, 17% to token 4, etc."

  2. NEXT TOKEN PREDICTION:
     Logits for 32,000 tokens → softmax → probability of each token
     Token "Paris" gets 0.82, "London" gets 0.03, etc.

Hardware connection:
  softmax needs exp() → runs on SFU (Special Function Units)
  softmax needs sum → runs as parallel REDUCTION (Week 3 topic)
  softmax is memory-bound (reads/writes all elements)
```

## Temperature = Control Randomness

```
softmax(zᵢ / T)

T = 1.0:  normal behavior
T = 0.1:  sharpen → almost always pick the top token (deterministic)
T = 2.0:  flatten → more random, creative, diverse outputs

Geometric meaning: Temperature SCALES the logits before softmax.
  Low T → gaps between logits are magnified → confident
  High T → gaps are reduced → uncertain → more random sampling

This is what the "temperature" slider in ChatGPT controls.
```

---

# PART 4: INFORMATION THEORY — The Loss Function of LLMs

## Cross-Entropy = How Wrong Are Our Predictions?

```
H(p, q) = -Σ p(x) × log(q(x))

Where:
  p = true distribution (the actual next token is "Paris" → one-hot [0,0,...,1,...,0])
  q = model's predicted distribution (softmax output)

Geometric meaning:
  Measures the "surprise" when using model q to predict reality p.
  If q predicts "Paris" with probability 0.99 → low surprise → low loss.
  If q predicts "Paris" with probability 0.01 → high surprise → high loss.

  Perfectly predicted:  -1 × log(1.0) = 0        (zero loss!)
  Completely wrong:     -1 × log(0.001) = 6.9     (huge loss!)

In LLMs:
  THIS IS THE TRAINING LOSS.
  
  For every token in training data:
    Model predicts probability distribution over 32,000 tokens
    Cross-entropy measures how far the prediction is from truth
    Average this over billions of tokens = training loss
    Minimize this loss = the model gets better at predicting next tokens
    
  Perplexity = exp(cross-entropy loss)
    Perplexity of 10 means "the model is as confused as if it had
    to randomly pick from 10 equally likely tokens"
    Lower perplexity = better model.
```

## KL Divergence = Distance Between Two Distributions

```
KL(p || q) = Σ p(x) × log(p(x) / q(x))

Geometric meaning:
  "How different is distribution q from distribution p?"
  KL = 0 means they're identical.
  KL > 0 means they differ (always non-negative).
  NOT symmetric: KL(p||q) ≠ KL(q||p)

In LLMs:
  1. RLHF: KL penalty prevents the fine-tuned model from drifting
     too far from the base model.
     Loss = reward - β × KL(fine_tuned || base_model)
     "Be helpful, but don't forget what you learned in pre-training"

  2. Knowledge distillation: student model tries to match
     teacher model's output distribution.
     Loss = KL(teacher_output || student_output)
```

## Entropy = How Uncertain Is a Distribution?

```
H(p) = -Σ p(x) × log(p(x))

High entropy = very uncertain (uniform distribution)
Low entropy = very certain (one token dominates)

In LLMs:
  When the model outputs high-entropy distribution →
  it's "confused" about what comes next.
  Useful for detecting when the model is uncertain
  and might hallucinate.
```

---

# PART 5: THE OPERATIONS THAT MATTER FOR LLMs — Summary

```
OPERATION            GEOMETRIC MEANING              WHERE IN LLM
────────────────────────────────────────────────────────────────────
Matrix multiply      Transform meaning vectors       Q,K,V projections, FFN,
                                                     output projection
                                                     (90% of compute)

Dot product          Similarity between vectors      Attention scores
                                                     (Q · K for each pair)

Softmax              Numbers → probabilities         Attention weights,
                                                     next-token prediction

Cross-entropy        How wrong are predictions?      Training loss function

Gradient             Direction to reduce loss         Backpropagation
                                                     (training)

Residual add         Skip connection (y = f(x) + x)  Every Transformer layer
                     Keeps gradient flowing            (prevents vanishing grad)

Layer/RMS Norm       Normalize to unit scale          Before every sublayer
                     Prevents values from growing      (stabilizes training)

Exp, Sin, Cos        Transcendental functions         Softmax, RoPE, GELU

Sum (reduction)      Collapse a dimension             Softmax denominator,
                                                     loss averaging
```

---

# PART 6: CODING EXERCISES (NumPy — no GPU needed)

Run these on Colab with **CPU runtime** (no GPU needed, save compute units).

## Exercise 1: Softmax From Scratch

```python
import numpy as np

def softmax(z):
    """Numerically stable softmax"""
    z_shifted = z - np.max(z)  # subtract max for numerical stability
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z)

# Test 1: Confident prediction
logits = np.array([5.0, 2.0, 1.0, 0.5, -1.0])
probs = softmax(logits)
print("Confident logits:", logits)
print("Probabilities:   ", np.round(probs, 3))
print("Sum:             ", np.sum(probs))  # should be 1.0

# Test 2: Uncertain prediction
logits_flat = np.array([2.1, 2.0, 1.9, 1.8, 1.7])
probs_flat = softmax(logits_flat)
print("\nUncertain logits:", logits_flat)
print("Probabilities:   ", np.round(probs_flat, 3))

# Test 3: Temperature scaling
def softmax_with_temp(z, temperature):
    return softmax(z / temperature)

logits = np.array([5.0, 2.0, 1.0])
print("\nTemperature effect on [5.0, 2.0, 1.0]:")
for T in [0.1, 0.5, 1.0, 2.0, 5.0]:
    p = softmax_with_temp(logits, T)
    print(f"  T={T:<4} → {np.round(p, 3)}  (max prob: {p.max():.3f})")
# Low T → deterministic, High T → random
```

## Exercise 2: Cross-Entropy Loss From Scratch

```python
def cross_entropy_loss(predicted_probs, true_index):
    """
    predicted_probs: output of softmax (probability for each token)
    true_index: the index of the correct token
    """
    return -np.log(predicted_probs[true_index])

# Simulate: model predicting next token after "The capital of France is"
vocab = ["Paris", "London", "Berlin", "the", "a", "cat"]

# Good model: high probability on correct answer
good_probs = softmax(np.array([8.0, 2.0, 1.5, 0.1, -0.5, -2.0]))
true_token = 0  # "Paris"
loss_good = cross_entropy_loss(good_probs, true_token)

# Bad model: low probability on correct answer
bad_probs = softmax(np.array([1.0, 3.0, 2.5, 4.0, 0.5, -1.0]))
loss_bad = cross_entropy_loss(bad_probs, true_token)

print("Good model predictions:", {v: f"{p:.3f}" for v, p in zip(vocab, good_probs)})
print(f"Loss (good model): {loss_good:.4f}")
print(f"Perplexity:        {np.exp(loss_good):.2f}")

print(f"\nBad model predictions:", {v: f"{p:.3f}" for v, p in zip(vocab, bad_probs)})
print(f"Loss (bad model):  {loss_bad:.4f}")
print(f"Perplexity:        {np.exp(loss_bad):.2f}")

print(f"\nBad model loss is {loss_bad/loss_good:.1f}x worse")
```

## Exercise 3: Attention Scores (Dot Product)

```python
np.random.seed(42)
d_model = 8  # small for visualization (real LLMs use 4096)

# 4 token embeddings
tokens = ["The", "capital", "of", "France"]
X = np.random.randn(4, d_model)

# Fake Q, K projections (random weights — not trained)
W_Q = np.random.randn(d_model, d_model) * 0.5
W_K = np.random.randn(d_model, d_model) * 0.5

Q = X @ W_Q  # [4 × 8]
K = X @ W_K  # [4 × 8]

# Attention scores = Q × K^T / sqrt(d)
scores = Q @ K.T / np.sqrt(d_model)  # [4 × 4]

# Apply softmax per row
attention = np.array([softmax(row) for row in scores])

print("Attention matrix (each row = who does this token attend to):\n")
print(f"{'':>10}", end="")
for t in tokens:
    print(f"{t:>10}", end="")
print()

for i, token in enumerate(tokens):
    print(f"{token:>10}", end="")
    for j in range(4):
        val = attention[i][j]
        bar = "█" * int(val * 20)
        print(f"{val:>6.2f} {bar:>5}", end="")
    print()

print("\nEach row sums to 1.0 (it's a probability distribution)")
print("In a TRAINED model, 'France' would attend heavily to 'capital' and 'of'")
```

## Exercise 4: Matrix Multiply = Transformation (Visual)

```python
import numpy as np

# 2D example to visualize (LLMs use 4096D, same principle)
# A square of 4 points
points = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]).T  # 2×5

# Transformation matrix: rotate 45° and scale 1.5x
theta = np.pi / 4
W = 1.5 * np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta),  np.cos(theta)]])

transformed = W @ points  # matrix multiply = apply transformation

print("Original points (a square):")
for i in range(4):
    print(f"  ({points[0,i]:.1f}, {points[1,i]:.1f})")

print(f"\nTransformation matrix W (rotate {int(np.degrees(theta))}° + scale 1.5x):")
print(f"  {W}")

print(f"\nTransformed points (rotated, scaled rhombus):")
for i in range(4):
    print(f"  ({transformed[0,i]:.2f}, {transformed[1,i]:.2f})")

print(f"\nIn LLMs: W_Q does this same thing to 4096-dimensional meaning vectors.")
print(f"It rotates/scales them in ways that make 'queries' useful for finding relevant keys.")
print(f"The specific rotation was LEARNED during training on trillions of tokens.")
```

---

# PART 7: TODAY'S MINI-PROJECT 🔨

## Project: "LLM Math Visualizer"

**Build a script that visualizes how attention works step-by-step:**

```python
import numpy as np

def full_attention_demo(sentence, d_model=16, seed=42):
    """
    Demonstrate the full attention computation for a sentence.
    Uses random weights (not trained) but shows the correct math.
    """
    np.random.seed(seed)
    tokens = sentence.split()
    n = len(tokens)
    
    print(f"Sentence: '{sentence}'")
    print(f"Tokens: {tokens}")
    print(f"Embedding dim: {d_model}\n")
    
    # Step 1: Random embeddings (in real LLM: looked up from trained table)
    X = np.random.randn(n, d_model)
    print(f"1. Token embeddings X: shape {X.shape}")
    
    # Step 2: Q, K, V projections
    W_Q = np.random.randn(d_model, d_model) * 0.3
    W_K = np.random.randn(d_model, d_model) * 0.3
    W_V = np.random.randn(d_model, d_model) * 0.3
    
    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V
    print(f"2. Q = X × W_Q: shape {Q.shape}  (what each token is looking for)")
    print(f"   K = X × W_K: shape {K.shape}  (what each token advertises)")
    print(f"   V = X × W_V: shape {V.shape}  (what info each token carries)")
    
    # Step 3: Attention scores
    scores = Q @ K.T / np.sqrt(d_model)
    print(f"\n3. Raw attention scores (Q × K^T / √d): shape {scores.shape}")
    
    # Step 4: Causal mask (each token only sees past + itself)
    mask = np.triu(np.ones((n, n)) * -1e9, k=1)
    masked_scores = scores + mask
    print(f"4. Applied causal mask (future tokens = -infinity)")
    
    # Step 5: Softmax
    def softmax_rows(x):
        e = np.exp(x - x.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)
    
    attention = softmax_rows(masked_scores)
    
    print(f"\n5. Attention weights (after softmax):")
    print(f"   {'':>12}", end="")
    for t in tokens:
        print(f"{t[:6]:>8}", end="")
    print()
    for i, t in enumerate(tokens):
        print(f"   {t[:6]:>12}", end="")
        for j in range(n):
            w = attention[i][j]
            print(f"{w:>8.3f}", end="")
        print(f"  (sum={attention[i].sum():.3f})")
    
    # Step 6: Weighted sum of values
    output = attention @ V
    print(f"\n6. Output = Attention × V: shape {output.shape}")
    print(f"   Each token's output is a weighted mix of all value vectors")
    print(f"   weighted by how much it attended to each other token.")
    
    # Step 7: What the model "decided"
    print(f"\n7. What each token focused on:")
    for i, t in enumerate(tokens):
        top_j = np.argmax(attention[i])
        pct = attention[i][top_j] * 100
        print(f"   '{t}' → mostly attended to '{tokens[top_j]}' ({pct:.0f}%)")
    
    return attention

# Run it!
attn = full_attention_demo("The capital of France is")
print("\n" + "="*60)
attn = full_attention_demo("The cat sat on the mat")
```

**What you'll see:** The full attention computation with real matrix shapes and values. The attention patterns won't be meaningful (random weights, not trained) but the MATH is exactly what happens inside every LLM.

---

# CHECKLIST

After Day 3:
- [ ] You know what matrix multiply MEANS geometrically (transformation of meaning vectors)
- [ ] You know dot product = similarity (used in attention scores)
- [ ] You know softmax converts logits → probabilities (used twice: attention + prediction)
- [ ] You know cross-entropy = training loss (how wrong the model's prediction is)
- [ ] You know gradient = direction to improve (used in backprop)
- [ ] You know why residual connections exist (prevent vanishing gradient via chain rule)
- [ ] You know KL divergence = distance between distributions (used in RLHF)
- [ ] You've implemented softmax, cross-entropy, and attention from scratch in NumPy

**Day 4** is where you write your FIRST CUDA kernel. The theory phase is done —
tomorrow you start programming the GPU directly.

---

*Status: ⬜ NOT YET COMPLETED*
*Date completed: ___________*
