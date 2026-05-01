# Week 2, Day 3: Backpropagation and Gradient Descent
## How a Network Learns — and Why Training a GPU Looks Different from Running One

**Time: ~2.5-3 hours**
**Setup: Google Colab with GPU runtime (PyTorch + a little NumPy)**

---

## Today's Mental Model

The last two days were about feeding the GPU's compute units. Today is about
**why we feed them at all**.

A neural network is not born knowing anything. It starts as a pile of random
numbers. Every weight in the model is a knob, and there are billions of knobs.
Training is the process of adjusting those knobs so the model's outputs match
the data we care about.

That process is built on three ideas:

1. **Forward pass** — push an input through the network to get a prediction.
2. **Loss** — measure how wrong the prediction is.
3. **Backward pass** — figure out, for every weight, *how to nudge it* so the
   loss gets smaller.

The third step is **backpropagation**. It is the chain rule from calculus,
applied carefully and efficiently to a graph of operations. Combined with
**gradient descent**, it is the algorithm that made deep learning work.

By the end of today you should be able to look at any neural network kernel
and ask:

1. **What activations does this layer produce, and which of them must we keep
   for the backward pass?**
2. **What is the gradient of the loss with respect to each input of this layer?**
3. **How do those gradients flow back into the weights?**
4. **How much extra memory does training cost compared to inference?**
5. **Why are the GPU lessons from Day 1 and Day 2 even more important here?**

This is the day where ML stops feeling like "magic library calls" and starts
feeling like a graph of small, understandable operations.

---

# PART 1: THE STORY

## 1.1 The Problem That Stalled AI for Decades

By the late 1960s, researchers knew how to build a single artificial neuron.
They knew how to wire many neurons together into a multi-layer network. And
they knew, in principle, that such a network could represent very rich
functions.

But they could not train it.

A single-layer model was easy. There was a clear formula for "how should this
weight change to reduce the error?" — you could read it directly off the data.
But once you stacked layers, an awkward question appeared:

> If layer 5 made a bad prediction, who is to blame?
>
> The weights in layer 5? Layer 4? Layer 3? Layer 1?

This is called the **credit assignment problem**. With a deep network, the
final error is the result of *every* weight in *every* layer cooperating in
*every* example. There is no obvious way to point at one weight and say
"that one is wrong by this amount."

Without a way to assign blame, you could not update the weights, and the
network could not learn. AI research went into a long winter.

## 1.2 The Insight That Restarted Deep Learning

In the 1970s and 80s, several researchers (Werbos, Rumelhart, Hinton, Williams,
LeCun and others) realized something that, in hindsight, sounds simple:

> The chain rule.

If you know how the final output depends on the last layer, and you know how
the last layer depends on the layer before, and so on... then you can
**propagate the derivative backward** through the network, layer by layer,
operation by operation.

```
forward direction →
   x ──► layer 1 ──► layer 2 ──► layer 3 ──► loss

backward direction ←
        ∂loss/∂x ◄── ∂loss/∂h1 ◄── ∂loss/∂h2 ◄── ∂loss/∂h3 ◄── loss
```

That backward sweep is **backpropagation**. Each operation in the network
gets a small recipe:

```text
"If somebody hands me the gradient of the loss with respect to my output,
 I can produce the gradient of the loss with respect to my input AND
 with respect to my own weights."
```

If every operation knows that local recipe, you can chain them together and
produce gradients for *every* weight in the network in one backward pass —
in roughly the same time as the forward pass.

That is the algorithm that powers almost every modern deep learning system,
including the one running ChatGPT, Llama, Stable Diffusion, AlphaFold, and the
matmul kernels you'll write later this week.

## 1.3 Backprop in One Sentence

A more precise way to say it:

> Backpropagation is the chain rule, applied to a directed acyclic graph of
> primitive operations, in the order that reuses already-computed values.

The key word is **reuses**. Backprop is fast because it does not recompute
intermediate values for each weight — it computes them once, on the forward
pass, and reuses them on the backward pass.

That single design choice has a direct consequence on the GPU:

> Training stores activations from the forward pass so the backward pass can
> use them. That's why training a model takes much more memory than running it.

This is the bridge from yesterday's lesson (memory hierarchies, shared memory,
bandwidth) into today's lesson (backprop). The math forces you to keep extra
data alive on the GPU. The hardware then forces you to manage that data
carefully.

## 1.4 Today's Roadmap

```
PART 2  — The chain rule, by hand, on a tiny network
PART 3  — Computational graphs and reverse-mode autodiff
PART 4  — Gradient descent (and why learning rate is so dangerous)
PART 5  — Common loss functions and where they show up in LLMs
PART 6  — Hands-on: backprop a 2-layer MLP in NumPy
PART 7  — PyTorch autograd and how to read it
PART 8  — Memory cost of training vs. inference (the GPU connection)
PART 9  — Mini-project: train a digit classifier and inspect every gradient
PART 10 — How this scales to LLM training
```

---

# PART 2: THE CHAIN RULE, BY HAND

We will walk through the smallest interesting network: one input, one hidden
neuron, one output, one loss. Once you can do backprop on this, the rest is
just more arithmetic.

## 2.1 The Network

```
   x ───[w1]───► z1 = w1 · x
                 │
                 ▼
              h = ReLU(z1)
                 │
                 ▼
   h ───[w2]───► z2 = w2 · h
                 │
                 ▼
             ŷ = z2
                 │
                 ▼
   loss L = (ŷ - y)²            (squared error vs. target y)
```

What this picture shows:

- `x` is the input, a single number.
- `w1` and `w2` are the two weights we want to learn.
- `z1` is the pre-activation of the hidden neuron.
- `h` is the activation, after ReLU.
- `z2` is the pre-activation of the output.
- `ŷ` ("y-hat") is the prediction.
- `y` is the true target value.
- `L` is the squared-error loss between prediction and target.

We want to know, for a given `(x, y)` example:

```text
∂L / ∂w1 = ?
∂L / ∂w2 = ?
```

These are the gradients. They tell us how to nudge `w1` and `w2` to make `L`
smaller.

## 2.2 The Forward Pass

Pick concrete numbers so we can see what is happening:

```text
x  = 2.0
y  = 3.0
w1 = 0.5
w2 = 1.5
```

Now compute layer by layer:

```text
z1 = w1 · x          = 0.5 · 2.0       = 1.0
h  = ReLU(z1)        = ReLU(1.0)        = 1.0
z2 = w2 · h          = 1.5 · 1.0        = 1.5
ŷ  = z2                                   = 1.5
L  = (ŷ - y)²        = (1.5 - 3.0)²     = 2.25
```

So the prediction is 1.5, the target is 3.0, and the loss is 2.25.

Important habit:

> Save these intermediate values. Backprop will need them.

In a hand calculation we just remember `x`, `z1`, `h`, `z2`, `ŷ`. In a real
framework, this is exactly what gets stored in the autograd graph.

## 2.3 The Backward Pass — One Link at a Time

We want gradients of `L`. The trick is to walk backward from `L` and use the
chain rule at each step. Read each step like a domino fall.

### Step 1. ∂L / ∂ŷ

`L = (ŷ - y)²`. Differentiate with respect to `ŷ`:

```text
∂L / ∂ŷ = 2 · (ŷ - y)
        = 2 · (1.5 - 3.0)
        = -3.0
```

What this means:

- The loss is `(ŷ - y)²`.
- Its derivative with respect to `ŷ` is `2(ŷ - y)`.
- A negative number means: *increasing ŷ would decrease the loss*.
- Magnitude `3.0` means we are quite a bit off and gradient descent will push
  hard.

### Step 2. ∂L / ∂z2

Here `ŷ = z2`, so:

```text
∂ŷ / ∂z2 = 1
```

By the chain rule:

```text
∂L / ∂z2 = ∂L / ∂ŷ · ∂ŷ / ∂z2
         = -3.0 · 1
         = -3.0
```

What this means:

- The output layer is just `ŷ = z2`.
- So gradient through this step is just a pass-through.
- The "blame" arriving at `z2` is exactly the same as the blame at `ŷ`.

### Step 3. ∂L / ∂w2 and ∂L / ∂h

`z2 = w2 · h`, so:

```text
∂z2 / ∂w2 = h    = 1.0
∂z2 / ∂h  = w2   = 1.5
```

Chain rule again:

```text
∂L / ∂w2 = ∂L / ∂z2 · ∂z2 / ∂w2
         = -3.0 · 1.0
         = -3.0

∂L / ∂h  = ∂L / ∂z2 · ∂z2 / ∂h
         = -3.0 · 1.5
         = -4.5
```

What this means:

- `z2` was made by multiplying `w2` and `h`. There are two parents, so two
  gradients flow upstream.
- `∂L / ∂w2 = -3.0` says: *increasing `w2` would decrease the loss by about
  3.0 per unit.* That is the signal we will use to update `w2`.
- `∂L / ∂h  = -4.5` says: *increasing `h` would also decrease the loss.* That
  signal continues flowing back into the previous layer.

This is the heartbeat of backprop:

```
gradient at output × local derivative = gradient at input
```

### Step 4. ∂L / ∂z1

`h = ReLU(z1)`. ReLU has a piecewise derivative:

```text
∂h / ∂z1 = 1   if z1 > 0
∂h / ∂z1 = 0   if z1 ≤ 0
```

We had `z1 = 1.0 > 0`, so:

```text
∂h / ∂z1 = 1
```

Chain rule:

```text
∂L / ∂z1 = ∂L / ∂h · ∂h / ∂z1
         = -4.5 · 1
         = -4.5
```

What this means:

- ReLU acts like a gate. If it was "on" during the forward pass, it lets the
  gradient pass through unchanged.
- If it was "off" (`z1 ≤ 0`), the gate was closed, and the gradient becomes
  zero — the upstream layer learns nothing about that example.
- This is exactly why "dying ReLU" is a thing: a neuron stuck at `z1 ≤ 0`
  receives no gradient and never updates.

### Step 5. ∂L / ∂w1 and ∂L / ∂x

`z1 = w1 · x`, so:

```text
∂z1 / ∂w1 = x   = 2.0
∂z1 / ∂x  = w1  = 0.5
```

Chain rule one last time:

```text
∂L / ∂w1 = ∂L / ∂z1 · ∂z1 / ∂w1
         = -4.5 · 2.0
         = -9.0

∂L / ∂x  = ∂L / ∂z1 · ∂z1 / ∂x
         = -4.5 · 0.5
         = -2.25
```

What this means:

- `∂L / ∂w1 = -9.0`. This is the gradient we will use to update `w1`. It is
  bigger than `∂L / ∂w2` because the input `x = 2.0` was bigger than `h = 1.0`,
  amplifying the chain.
- `∂L / ∂x` exists too, but we usually do not update `x` (the input data is
  fixed). For LLMs, however, `x` is itself a learned embedding, so this
  upstream gradient *does* matter — it flows back into the embedding table.

## 2.4 Summary of the Tiny Network

```text
forward:
  x=2 → z1=1 → h=1 → z2=1.5 → ŷ=1.5 → L=2.25

backward:
  ∂L/∂ŷ  = -3.0
  ∂L/∂z2 = -3.0
  ∂L/∂h  = -4.5
  ∂L/∂z1 = -4.5
  ∂L/∂w2 = -3.0
  ∂L/∂w1 = -9.0
  ∂L/∂x  = -2.25
```

Two things to internalize:

1. **The backward pass walks the graph in reverse.** Outputs at each step
   become inputs to the previous step.
2. **Each operation has a tiny local derivative recipe.** Multiplication's
   recipe is "multiply by the other operand." ReLU's recipe is "pass through
   if positive, kill if not." Add's recipe is "pass through to both inputs."
   That is all.

If you ever forget what backprop is, come back to this picture. Everything
else — autograd, PyTorch, CUDA backward kernels — is automation of this exact
hand calculation, scaled up to billions of operations.

---

# PART 3: COMPUTATIONAL GRAPHS AND AUTODIFF

## 3.1 What a Computational Graph Is

A neural network is, mathematically, a composition of functions:

```text
L(x; w) = loss( f₃( f₂( f₁(x; w₁); w₂); w₃ ), y )
```

In practice we draw it as a **directed acyclic graph** of small operations:

```
   x
    \
     [matmul w1]
        │
        ▼
       z1
        │
        ▼
       [ReLU]
        │
        ▼
        h
        │
         \
          [matmul w2]
              │
              ▼
             z2
              │
              ▼
             [- y]
              │
              ▼
            (ŷ - y)
              │
              ▼
            [square]
              │
              ▼
              L
```

What this picture shows:

- Each box is a primitive operation (matmul, ReLU, subtract, square).
- Each arrow is a tensor flowing between operations.
- The graph is acyclic: data flows one direction in the forward pass.
- Backprop walks this same graph in reverse.

## 3.2 Reverse-Mode Automatic Differentiation

Given the graph, there are two natural ways to compute gradients.

**Forward-mode autodiff:** start from `x` and propagate derivatives forward.
Good when there are few inputs and many outputs.

**Reverse-mode autodiff:** start from the loss and propagate derivatives
backward. Good when there are many inputs (parameters) and one output (loss).

Neural networks have:

- **One scalar output** (loss).
- **Millions or billions of parameters** (weights).

So reverse-mode is the right choice. That is what every deep learning
framework actually does. People say "autograd" or "backprop" or "reverse-mode
autodiff" — they are all the same idea.

The cost analysis is striking:

```text
forward pass over the graph = 1 unit of compute
reverse-mode backward pass  = ~1-3 units of compute (same order)
```

So you can compute gradients for *every* parameter in the same time it took
to compute the loss. That is what makes training big models feasible.

## 3.3 Two Tape Modes: Eager vs. Static

You will hear "dynamic graph" vs. "static graph" in deep learning forever.
Today's mental model:

- **Static graph (TensorFlow 1.x, JAX `jit`, ONNX):** you describe the graph
  ahead of time, the framework compiles it, then you run it many times.
  Pros: high optimization potential. Cons: harder to debug, control flow is
  awkward.
- **Dynamic / eager graph (PyTorch, TensorFlow 2.x):** the graph is built as
  the forward pass runs, by recording each operation on a "tape." Backward is
  then a walk over that tape in reverse. Pros: feels like normal Python.
  Cons: slightly less optimization out-of-the-box.

PyTorch is dynamic. When you do:

```python
y = x @ W + b
loss = (y - target).pow(2).mean()
loss.backward()
```

What this snippet is doing:

- Each tensor operation is recorded on PyTorch's autograd tape.
- The tape stores enough information to compute each operation's local
  gradient.
- `loss.backward()` walks the tape in reverse, accumulating gradients into
  every parameter that has `requires_grad=True`.

The "tape" idea is just the computational graph plus the saved values from
the forward pass. It is the same thing you would write down on paper — the
framework just does it for you.

## 3.4 Why Saving Activations Costs Memory

Notice what reverse-mode autodiff implies:

> To compute gradients on the backward pass, you need activations from the
> forward pass.

For example, the gradient of `z2 = w2 · h` with respect to `w2` is `h`. To
multiply by it on the backward pass, the framework had to keep `h` around in
memory until `loss.backward()` was called.

For a tiny network this is irrelevant. For an LLM with thousands of layers
worth of intermediate tensors, this is the dominant memory cost of training.
We will return to this in Part 8 with real numbers.

---

# PART 4: GRADIENT DESCENT

Backprop tells you which direction to move each weight to decrease the loss.
**Gradient descent** is the rule that uses that direction to actually move.

## 4.1 The Update Rule

The simplest form:

```text
w := w - η · ∂L / ∂w
```

What this means:

- `w` is a weight (or any parameter).
- `∂L / ∂w` is the gradient computed by backprop.
- `η` (eta) is the **learning rate**, a small positive number you pick.
- The minus sign is what makes it *descent* — you go opposite to the gradient.

In our tiny example we had `∂L / ∂w1 = -9.0`. With `η = 0.01`:

```text
w1_new = 0.5 - 0.01 · (-9.0)
       = 0.5 + 0.09
       = 0.59
```

What this means:

- The gradient was negative, meaning "increasing `w1` reduces loss."
- The update rule subtracts a negative, which adds a positive.
- So `w1` increased from 0.5 to 0.59 — exactly the direction the gradient told
  us would help.

If you re-run the forward pass with `w1 = 0.59` and `w2 = 1.5`, you'll see the
loss has dropped. That is one step of training.

## 4.2 The Geometry: Walking Down a Hill

```
         loss
          ▲
          │      ●  (start, high loss)
          │       \
          │        \
          │         ●─────●
          │          \    \    ← gradient descent steps
          │           \    \
          │            ●────●─── ●  (low loss)
          │
          └─────────────────────────► weights
```

What this picture shows:

- The vertical axis is the loss.
- The horizontal axis is some weight (in reality it is a billion-dimensional
  surface; we just draw 1D for intuition).
- Gradient descent takes steps proportional to the slope. Steeper slopes →
  bigger steps. Flat regions → tiny steps.
- The destination is a valley — a local minimum of the loss.

In high dimensions, the picture is much wilder, but the basic idea — *follow
the slope downhill* — is the same.

## 4.3 The Learning Rate Is the Most Dangerous Hyperparameter

Pick `η` too large:

```
w1_new = 0.5 - 5.0 · (-9.0) = 45.5      (massive jump)
```

Now compute the next forward pass — the loss will *explode*. You overshot the
minimum, landed on a worse part of the loss surface, and your next step will
be even bigger. Training diverges.

Pick `η` too small:

```
w1_new = 0.5 - 0.0000001 · (-9.0) ≈ 0.5
```

The weight barely moves. After thousands of steps, the model has not learned
anything visible. Training crawls.

The rough behaviors are:

| Learning rate | Symptom |
|---------------|---------|
| Too large | Loss explodes to NaN, weights blow up, gradients become huge or zero |
| A bit too large | Loss oscillates, sometimes going down, sometimes going up |
| Just right | Loss decreases smoothly |
| Too small | Loss decreases very slowly, never reaches a good minimum in available time |

Practical defaults:

- For most MLPs and CNNs trained from scratch, `η` between `1e-3` and `1e-2`
  with SGD is a reasonable starting point.
- For LLMs and transformer training with Adam-style optimizers, `η` is often
  between `1e-5` and `5e-4`, and is usually warmed up (start small, increase,
  then decay).
- When a new training run blows up, the *first* thing to lower is the
  learning rate.

## 4.4 Stochastic Gradient Descent and Mini-Batches

The pure gradient descent rule says: compute the gradient using the *entire*
dataset, then take one step. For LLMs, the dataset has trillions of tokens.
That is impractical.

So we use **stochastic gradient descent** (SGD) with **mini-batches**:

```text
1. Pick a random batch of B examples.
2. Compute the average loss over the batch.
3. Backprop to get gradients.
4. Take one step of gradient descent.
5. Repeat with a new batch.
```

Why this works:

- Each batch's gradient is a *noisy estimate* of the true gradient.
- The noise averages out over many steps.
- The noise actually helps escape some bad local minima.

On a GPU, larger batches are more efficient because they reuse the same
weights across many examples. This is the same idea you saw on Day 1: large
batches turn a memory-bound decode into a more compute-bound problem.

## 4.5 Beyond Plain SGD: Why Adam Won

Plain SGD has two practical issues:

1. The same learning rate for every parameter is rarely ideal. Some parameters
   want big steps, some want tiny ones.
2. Pure gradient direction is noisy from one batch to the next.

Modern optimizers patch both:

- **Momentum** (1980s, Polyak / Nesterov): keep a running average of past
  gradients so noisy single-batch estimates get smoothed out. The intuition is
  a heavy ball rolling downhill — it does not change direction every time the
  ground tilts slightly.
- **Adaptive learning rates** (RMSProp, Adagrad, Adam): scale each parameter's
  step by an estimate of its historical gradient magnitude. Frequently updated
  parameters get smaller steps; rarely updated ones get bigger steps.

**Adam** combines momentum with adaptive learning rates. It is the default for
training nearly every transformer in the open-source world. The exact update
rule:

```text
m := β1 · m + (1 - β1) · g           (momentum)
v := β2 · v + (1 - β2) · g²          (squared gradient running average)
m̂ := m / (1 - β1ᵗ)                   (bias correction)
v̂ := v / (1 - β2ᵗ)
w := w - η · m̂ / (√v̂ + ε)
```

What this says, in words:

- `m` is a smoothed version of the gradient (momentum).
- `v` is a smoothed version of the squared gradient (rough variance estimate).
- `m̂` and `v̂` correct for the fact that the running averages start at zero.
- The actual step size is bigger when momentum is big and the gradient
  variance is small.

You do not need to memorize this. You should know that Adam:

- has its own state per parameter (`m` and `v`),
- doubles the optimizer's memory cost compared to SGD,
- converges much faster on transformer-style problems.

This optimizer state matters a lot for LLM training memory budgets — an
optimizer like AdamW typically takes about 8 bytes per parameter on top of
the model itself. We will see this in Part 8.

---

# PART 5: LOSS FUNCTIONS

The loss is the only place where the *task* enters the gradient computation.
Backprop is mechanical. The loss is what gives backprop something meaningful
to chase.

## 5.1 Mean Squared Error (Regression)

```text
L = (1 / N) · Σ (ŷᵢ - yᵢ)²
```

What this means:

- For each example, take the squared difference between prediction and target.
- Average over the batch.
- Used for regression problems where the output is a continuous number.

Gradient w.r.t. each prediction:

```text
∂L / ∂ŷᵢ = (2 / N) · (ŷᵢ - yᵢ)
```

Linear, simple, and the workhorse for "predict a real number" tasks.

## 5.2 Cross-Entropy (Classification)

For a classifier that outputs a probability distribution `p` over `K`
classes, with true class `y`:

```text
L = -log(p_y)
```

What this means:

- Look at the probability the model assigned to the *correct* class.
- Take the negative log of that.
- A perfect model assigns probability 1 → `log 1 = 0` → loss is 0.
- A confidently wrong model assigns probability near 0 → `-log` of a tiny
  number is huge → loss is huge.

Cross-entropy is what every LLM is trained with. The "predicted distribution"
is the softmax over the model's output logits, and the "true class" is the
next token in the training text.

A property to remember: in modern frameworks, you almost always see the
fused operation **softmax + cross-entropy** rather than computing them
separately. The fused version has a beautifully simple gradient:

```text
∂L / ∂logitᵢ = pᵢ - 1{i = y}
```

What this means:

- `pᵢ` is the predicted probability of class `i`.
- `1{i = y}` is 1 if `i` is the true class and 0 otherwise.
- So the gradient is "predicted probability minus the one-hot target."
- Small, clean, and numerically stable.

This is so common that PyTorch's `nn.CrossEntropyLoss` *requires* you to pass
unnormalized logits, not probabilities. It does the fused softmax + cross
entropy internally.

## 5.3 What Loss Functions Look Like Inside an LLM

LLMs are trained as **next-token predictors**. For a sequence of tokens
`x₁, x₂, ..., x_T`:

```text
L = -(1 / T) · Σ_t log p(x_{t+1} | x_1, ..., x_t)
```

What this means:

- For every position `t` in the sequence, the model predicts a probability
  distribution over the next token.
- The loss at that position is the negative log of the probability assigned to
  the actual next token in the training text.
- The total loss is the average over all positions in the batch.

This is just cross-entropy applied at every position simultaneously. The
backward pass through this loss is what defines what the gradients mean: every
parameter of the model is being nudged so that the model assigns higher
probability to actually-observed sequences in the training data.

That single objective, applied to enough text, is enough to make the model
write code, answer questions, and pass the bar exam. The reason it works is
not a deep secret about cross-entropy — it is that the *gradient signal* from
this loss is informative enough to teach the model the structure of language.

---

# PART 6: HANDS-ON — BACKPROP IN NUMPY

Theory is nice. A working backward pass you wrote with your own hands is much
nicer.

## 6.1 The Plan

We will train a 2-layer MLP to learn a simple non-linear function. The whole
program does:

1. Make synthetic data (`x` in [0, 1], `y = sin(2πx)`).
2. Define the network: `x → linear → ReLU → linear → ŷ`.
3. Forward pass.
4. Compute MSE loss.
5. Backward pass — by hand, using chain rule.
6. SGD update.
7. Repeat.

We will deliberately avoid PyTorch autograd here. You will write every line
of the backward pass yourself. Then, in Part 7, we will compare gradients to
PyTorch.

## 6.2 The Network, on Paper

```text
forward:
  z1 = x @ W1 + b1
  h  = ReLU(z1)
  z2 = h @ W2 + b2
  ŷ  = z2
  L  = mean((ŷ - y)²)

backward:
  ∂L / ∂ŷ  = 2 (ŷ - y) / N

  ∂L / ∂z2 = ∂L / ∂ŷ                 (identity activation on output)

  ∂L / ∂W2 = h.T @ ∂L / ∂z2
  ∂L / ∂b2 = sum over batch of ∂L / ∂z2
  ∂L / ∂h  = ∂L / ∂z2 @ W2.T

  ∂L / ∂z1 = ∂L / ∂h * (z1 > 0)      (ReLU mask)

  ∂L / ∂W1 = x.T @ ∂L / ∂z1
  ∂L / ∂b1 = sum over batch of ∂L / ∂z1
```

A few things to notice for matrix-form backprop:

- A matmul `Y = X @ W` has gradients
  `∂L/∂X = ∂L/∂Y @ W.T` and `∂L/∂W = X.T @ ∂L/∂Y`.
  This is one of the most reused identities in deep learning.
- A bias `Y = X + b` (broadcast) has gradient `∂L/∂b = sum_over_batch(∂L/∂Y)`.
- ReLU's backward is the elementwise mask `(z1 > 0)`. The forward activation
  acts as a gate.

## 6.3 The Code

```python
import numpy as np

np.random.seed(0)

N = 1024
x = np.random.rand(N, 1).astype(np.float32)
y = np.sin(2 * np.pi * x).astype(np.float32)

D_in, D_hidden, D_out = 1, 64, 1
W1 = np.random.randn(D_in,     D_hidden).astype(np.float32) * 0.1
b1 = np.zeros((1, D_hidden), dtype=np.float32)
W2 = np.random.randn(D_hidden, D_out   ).astype(np.float32) * 0.1
b2 = np.zeros((1, D_out),    dtype=np.float32)

lr = 1e-2
epochs = 2000

for epoch in range(epochs):
    z1 = x @ W1 + b1
    h  = np.maximum(z1, 0.0)
    z2 = h @ W2 + b2
    yhat = z2

    loss = np.mean((yhat - y) ** 2)

    dL_dyhat = 2.0 * (yhat - y) / N
    dL_dz2   = dL_dyhat
    dL_dW2   = h.T @ dL_dz2
    dL_db2   = dL_dz2.sum(axis=0, keepdims=True)
    dL_dh    = dL_dz2 @ W2.T
    dL_dz1   = dL_dh * (z1 > 0).astype(np.float32)
    dL_dW1   = x.T @ dL_dz1
    dL_db1   = dL_dz1.sum(axis=0, keepdims=True)

    W1 -= lr * dL_dW1
    b1 -= lr * dL_db1
    W2 -= lr * dL_dW2
    b2 -= lr * dL_db2

    if epoch % 200 == 0:
        print(f"epoch {epoch:4d}  loss {loss:.6f}")
```

### Code walkthrough

```text
import numpy as np
np.random.seed(0)
```

What this is doing:

- Imports NumPy.
- Fixes the random seed so your numbers match the discussion.

```text
N = 1024
x = np.random.rand(N, 1).astype(np.float32)
y = np.sin(2 * np.pi * x).astype(np.float32)
```

What this is doing:

- Creates `N` examples.
- `x` is a column vector of `N` random numbers in [0, 1].
- `y` is `sin(2πx)`, a smooth nonlinear target. We want the network to learn
  this function.

```text
D_in, D_hidden, D_out = 1, 64, 1
W1 = np.random.randn(D_in, D_hidden).astype(np.float32) * 0.1
b1 = np.zeros((1, D_hidden), dtype=np.float32)
W2 = np.random.randn(D_hidden, D_out).astype(np.float32) * 0.1
b2 = np.zeros((1, D_out), dtype=np.float32)
```

What this is doing:

- The network has 1 input, 64 hidden units, 1 output.
- Weights are initialized to small Gaussian noise scaled by 0.1. Biases start
  at zero. Random small weights are essential — symmetric weights (e.g., all
  zeros) would make all hidden units learn the same thing.

```text
z1 = x @ W1 + b1
h  = np.maximum(z1, 0.0)
z2 = h @ W2 + b2
yhat = z2
```

What this is doing:

- Forward pass, exactly as written on paper.
- `@` is matmul. `z1` shape is `(N, 64)`. `h` is the ReLU activation.
  `z2` and `yhat` are `(N, 1)`.

```text
loss = np.mean((yhat - y) ** 2)
```

What this is doing:

- Mean squared error over the batch.

```text
dL_dyhat = 2.0 * (yhat - y) / N
```

What this is doing:

- The derivative of `mean((yhat - y)²)` with respect to `yhat`.
- Note the `/ N` — that comes from differentiating the mean.
- Forgetting this `/ N` is a classic mistake. The gradient still points the
  right direction, but the magnitude is off by a factor of `N`, and your
  effective learning rate becomes wildly wrong.

```text
dL_dz2 = dL_dyhat
dL_dW2 = h.T @ dL_dz2
dL_db2 = dL_dz2.sum(axis=0, keepdims=True)
dL_dh  = dL_dz2 @ W2.T
```

What this is doing:

- Backward through the second linear layer.
- Output activation is identity, so `dL_dz2 = dL_dyhat`.
- Matmul gradient identities: `(X @ W) → dL/dW = X.T @ dL/dY`,
  `dL/dX = dL/dY @ W.T`.
- Bias gradient is the sum over the batch dimension.

```text
dL_dz1 = dL_dh * (z1 > 0).astype(np.float32)
```

What this is doing:

- Backward through ReLU.
- The mask `(z1 > 0)` zeroes out gradients for hidden units that were "off"
  during the forward pass.
- This is the mathematical reason ReLU networks have *sparse* gradients.

```text
dL_dW1 = x.T @ dL_dz1
dL_db1 = dL_dz1.sum(axis=0, keepdims=True)
```

What this is doing:

- Backward through the first linear layer.
- Same matmul gradient identities.

```text
W1 -= lr * dL_dW1
b1 -= lr * dL_db1
W2 -= lr * dL_dW2
b2 -= lr * dL_db2
```

What this is doing:

- Plain SGD update.
- All four parameters use the same learning rate.

```text
if epoch % 200 == 0:
    print(f"epoch {epoch:4d}  loss {loss:.6f}")
```

What this is doing:

- Prints the loss occasionally so you can watch it decrease.

## 6.4 What You Should See

```text
epoch    0  loss 0.500
epoch  200  loss 0.119
epoch  400  loss 0.054
epoch  600  loss 0.031
epoch  800  loss 0.020
epoch 1000  loss 0.013
epoch 1200  loss 0.009
...
```

The loss should drop steadily. By epoch 2000 it should be under 0.005. If
yours is not decreasing:

- Lower the learning rate (try 1e-3 first).
- Check the `/ N` in `dL_dyhat`.
- Check the ReLU mask uses `z1 > 0`, not `h > 0` (works either way for ReLU,
  but pay attention to which input you use; for some activations this matters).
- Re-seed and try again. With small `N` you can get unlucky initializations.

## 6.5 What You Just Learned

You have written, by hand, every line of the backward pass for an MLP. Every
deep-learning framework in the world is doing exactly this internally —
matmul backward, bias backward, activation backward, loss backward — just
many more layers and on the GPU. From here on out, autograd will feel less
like a black box and more like an intern doing your bookkeeping.

---

# PART 7: PYTORCH AUTOGRAD

Now that you have done backprop by hand, autograd will feel almost trivial.

## 7.1 Same Network, with Autograd

```python
import torch
import torch.nn as nn

torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"

N = 1024
x = torch.rand(N, 1, device=device)
y = torch.sin(2 * torch.pi * x)

model = nn.Sequential(
    nn.Linear(1, 64),
    nn.ReLU(),
    nn.Linear(64, 1),
).to(device)

optim = torch.optim.SGD(model.parameters(), lr=1e-2)
loss_fn = nn.MSELoss()

for epoch in range(2000):
    yhat = model(x)
    loss = loss_fn(yhat, y)

    optim.zero_grad()
    loss.backward()
    optim.step()

    if epoch % 200 == 0:
        print(f"epoch {epoch:4d}  loss {loss.item():.6f}")
```

### Code walkthrough

```text
import torch
import torch.nn as nn
torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
```

What this is doing:

- Imports PyTorch and its neural network module.
- Seeds the random number generator so results are reproducible.
- Picks the GPU if available, otherwise the CPU.

```text
x = torch.rand(N, 1, device=device)
y = torch.sin(2 * torch.pi * x)
```

What this is doing:

- Creates the same input data, but as PyTorch tensors directly on the GPU.
- Note: data is generated on the device — no CPU-to-GPU copy needed. This is
  the kind of small habit that adds up to big wins, as we discussed on Day 1.

```text
model = nn.Sequential(
    nn.Linear(1, 64),
    nn.ReLU(),
    nn.Linear(64, 1),
).to(device)
```

What this is doing:

- Builds the same MLP. `nn.Sequential` chains modules in order.
- `.to(device)` moves all parameters to the GPU.
- `nn.Linear` already includes a bias by default.

```text
optim = torch.optim.SGD(model.parameters(), lr=1e-2)
loss_fn = nn.MSELoss()
```

What this is doing:

- Picks plain SGD as the optimizer, with the same learning rate as before.
- `model.parameters()` returns an iterator over every learnable tensor in the
  model — these are the things gradient descent will update.

```text
yhat = model(x)
loss = loss_fn(yhat, y)
```

What this is doing:

- Forward pass. PyTorch records every operation on the autograd tape.
- `loss` is a scalar tensor. Importantly, it has `requires_grad=True` because
  it depends on parameters that are tracked.

```text
optim.zero_grad()
loss.backward()
optim.step()
```

What this is doing:

- `optim.zero_grad()` clears any gradients left over from the previous step.
  This matters because PyTorch *accumulates* gradients into `.grad` by default,
  rather than overwriting them. Forgetting this is one of the most common
  bugs in PyTorch code.
- `loss.backward()` walks the autograd tape backward and fills in `.grad` on
  every tensor with `requires_grad=True`. This is what your NumPy code did by
  hand.
- `optim.step()` applies the SGD update using the gradients in `.grad`.

## 7.2 Inspecting the Tape

You can see exactly what autograd recorded:

```python
yhat = model(x)
loss = loss_fn(yhat, y)
print(loss.grad_fn)
print(loss.grad_fn.next_functions)
```

What this is doing:

- `grad_fn` is the function that produced this tensor on the forward pass.
- `next_functions` tells you what to call next on the backward pass.
- Walking these `grad_fn` chains is exactly the reverse-mode autodiff
  algorithm.

You will not normally need this. But seeing it once removes any remaining
mystery from autograd.

## 7.3 Verifying Your NumPy Backprop with Autograd

This is one of the most useful debugging habits a deep learning engineer can
have: write a forward pass twice, compute gradients twice, compare. If they
disagree, your math is wrong somewhere.

```python
import torch

torch.manual_seed(0)

x = torch.rand(1024, 1, requires_grad=False)
y = torch.sin(2 * torch.pi * x)

W1 = torch.randn(1, 64,  requires_grad=True) * 0.1
b1 = torch.zeros(1, 64,  requires_grad=True)
W2 = torch.randn(64, 1,  requires_grad=True) * 0.1
b2 = torch.zeros(1, 1,   requires_grad=True)

W1 = W1.detach().clone().requires_grad_()
b1 = b1.detach().clone().requires_grad_()
W2 = W2.detach().clone().requires_grad_()
b2 = b2.detach().clone().requires_grad_()

z1 = x @ W1 + b1
h  = torch.relu(z1)
z2 = h @ W2 + b2
loss = ((z2 - y) ** 2).mean()

loss.backward()

print("W1.grad shape:", W1.grad.shape)
print("W1.grad mean:", W1.grad.mean().item())
print("W2.grad mean:", W2.grad.mean().item())
```

What this is doing:

- Builds the same MLP using bare tensor ops, with `requires_grad=True` on the
  parameters.
- Calls `.backward()` on the loss.
- Prints the gradient tensors PyTorch produced.
- If you also build the same network in NumPy with the *same* initial values
  and run your hand-coded backward pass, the numbers should match to many
  decimal places.

This kind of check is how serious deep learning code is debugged. When a new
custom layer is being added to a framework, the very first thing that gets
written is a *gradient check* — usually a finite-difference comparison
against the analytic gradient. We will write one of these on Saturday.

---

# PART 8: THE GPU MEMORY COST OF TRAINING

This is where today's lesson loops back to Days 1 and 2 of this week.
Backprop forces a memory pattern that *only* makes sense if you have already
internalized HBM, caches, and shared memory.

## 8.1 What Lives in GPU Memory During Training

For a model with `P` parameters trained in mixed-precision (FP16 with FP32
master copies), here is the rough breakdown for a transformer with the AdamW
optimizer:

| Live in HBM during training | Bytes per parameter |
|------------------------------|---------------------|
| Model weights (FP16)         | 2 |
| Master weights (FP32)        | 4 |
| Adam moment `m` (FP32)       | 4 |
| Adam moment `v` (FP32)       | 4 |
| Gradients (FP16)             | 2 |
| **Subtotal (per parameter)** | **16** |

Then, separately, **activations** for the forward pass that must be kept
alive for backward:

```text
Activation memory ≈ batch_size × seq_len × hidden_dim × num_layers × bytes_per_element
                     × constant_for_attention_etc.
```

For a 7B-parameter model trained at sequence length 2048 with a healthy batch
size, activations alone can take *tens of gigabytes*. Sometimes more than the
model itself.

So the honest size of a 7B training job in HBM is something like:

```text
parameters     ≈ 7B  × 16 B = 112 GB
activations    ≈ tens of GB depending on batch and sequence length
```

That is why training a 7B model on a single 80 GB H100 is hard, and why
training larger models requires multiple GPUs and tricks like ZeRO sharding,
activation checkpointing, FSDP, and pipeline parallelism. Inference, by
contrast, only needs the model itself and a small KV cache. That is the
whole reason inference is "easier" than training.

## 8.2 Why Activations Dominate

For each layer, PyTorch keeps the activation tensor produced on the forward
pass so that backward can use it. Roughly:

```text
forward of linear:  saves the input X    so backward can compute dL/dW = X.T @ dL/dY
forward of ReLU:    saves the input z    so backward can compute the mask (z > 0)
forward of softmax: saves the output p   so backward can compute Jacobian-times-vector
forward of attention: saves intermediate matrices so backward of attention works
```

Multiply this by:

- the number of layers,
- the batch size,
- the sequence length,
- the hidden dimension,

and activations grow fast. For a transformer:

```text
activation memory ∝ batch × seq_len × hidden × num_layers
```

This is why **activation checkpointing** is such a popular trick. The idea is
to *not* save certain activations on the forward pass, and instead recompute
them on the fly during backward. You spend a little extra compute to save a
lot of memory:

```text
without checkpointing: store all activations
with checkpointing:    store only some (e.g., one per layer block)
                       recompute the rest as needed during backward
```

If you remember Day 1's mantra ("LLM training is memory-bound, not
compute-bound"), this trick should make immediate sense: trading compute for
memory is almost always worth it on a GPU at the scale of LLM training.

## 8.3 Why Gradient Memory Patterns Look Like Day 1

The backward pass is filled with **the same kinds of tensor operations** as
the forward pass:

```text
forward:   z2 = h @ W2     (matmul)
backward:  dL/dW2 = h.T @ dL/dz2     (matmul, with transposed input)
           dL/dh  = dL/dz2 @ W2.T    (matmul, with transposed weight)
```

So backward kernels are mostly matmuls and elementwise ops, just with
transposed access patterns. That is why everything you learned about
coalescing and bank conflicts still applies — it just applies *twice per
iteration* in training, instead of once.

Three concrete consequences:

1. **Backward matmul kernels are also sensitive to memory layout.** A naive
   transposed access on `W2.T` will be uncoalesced. cuBLAS, Triton, and
   custom kernels all use shared-memory tiling to fix this — exactly the
   technique you learned yesterday.
2. **A "fast forward, slow backward" kernel is a real failure mode.** Some
   custom layers profile great for inference and tank in training. The reason
   is almost always that the backward access pattern is uncoalesced.
3. **`torch.compile` and similar tools fuse and reorder backward ops.** Many
   "training is faster than I expected" wins come from the compiler fusing
   adjacent backward kernels so intermediate gradients never round-trip
   through HBM.

You are now in a position to read those compiler notes and have them mean
something.

## 8.4 The "Forward = Inference, Forward + Backward = Training" Formula

A handy mental shortcut:

```text
inference time ≈ T_forward
training time  ≈ T_forward + T_backward + T_optimizer_step
                ≈ 3 × T_forward    (rough rule of thumb)
```

So a 100 ms forward pass implies roughly a 300 ms training step, give or
take. Memory works similarly:

```text
inference memory ≈ model + small KV cache
training memory  ≈ model + master weights + optimizer state + gradients + activations
                ≈ 4-12 × inference memory      (depending on optimizer and batch)
```

Inference and training are not the same workload. Memorize this difference.

---

# PART 9: MINI-PROJECT — TRAIN A DIGIT CLASSIFIER

## "The Three-Way Gradient Match"

This project ties everything together:

1. Train a small MLP on MNIST digits.
2. Watch the loss decrease.
3. For a single forward pass, compute gradients three ways and compare:
   - PyTorch autograd
   - Hand-rolled NumPy backprop on the same weights
   - Numerical gradient via finite differences on a couple of weights

If all three agree, you have just verified backprop end-to-end.

## 9.1 The Training Script

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

train = datasets.MNIST("./data", train=True,  download=True, transform=transform)
test  = datasets.MNIST("./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train, batch_size=128, shuffle=True)
test_loader  = DataLoader(test,  batch_size=512, shuffle=False)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

model = MLP().to(device)
optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(5):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)

        logits = model(xb)
        loss = loss_fn(logits, yb)

        optim.zero_grad()
        loss.backward()
        optim.step()

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb).argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.numel()
    print(f"epoch {epoch}  test_acc {correct/total:.4f}")
```

### Code walkthrough

```text
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])
```

What this is doing:

- Converts the MNIST images to PyTorch tensors with values in [0, 1].
- Normalizes pixel values using MNIST's known mean and standard deviation.
- Normalization helps gradient descent because the inputs are roughly
  zero-centered and unit-variance — gradient magnitudes are more uniform
  across features.

```text
train_loader = DataLoader(train, batch_size=128, shuffle=True)
test_loader  = DataLoader(test,  batch_size=512, shuffle=False)
```

What this is doing:

- Wraps the dataset in a `DataLoader` that yields mini-batches.
- `shuffle=True` for training is important — it makes the SGD noise more
  helpful and avoids the network learning patterns based on file order.
- The test loader uses a larger batch because we are not training and there
  is no gradient memory to worry about.

```text
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
```

What this is doing:

- Defines a 3-layer MLP: 784 → 256 → 64 → 10.
- The final 10 outputs are class logits — *unnormalized* scores for each digit.
- We do not apply softmax in `forward` — `nn.CrossEntropyLoss` will do the
  fused softmax + cross-entropy in a numerically stable way.
- `x.view(x.size(0), -1)` flattens the 28×28 image to a 784-vector.

```text
model = MLP().to(device)
optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
loss_fn = nn.CrossEntropyLoss()
```

What this is doing:

- Moves the model to GPU.
- Picks SGD with momentum 0.9. This is the simplest "good" optimizer for
  small classification tasks. For LLMs we would use AdamW.
- Picks cross-entropy loss for classification.

```text
for xb, yb in train_loader:
    xb, yb = xb.to(device), yb.to(device)
    logits = model(xb)
    loss = loss_fn(logits, yb)

    optim.zero_grad()
    loss.backward()
    optim.step()
```

What this is doing:

- Standard PyTorch training loop:
  1. Move batch to GPU.
  2. Forward pass through model.
  3. Compute loss.
  4. Zero out previous gradients.
  5. Backprop fills `.grad` on every parameter.
  6. Optimizer applies the update.
- Every step is doing exactly the math you wrote in NumPy in Part 6 — just
  with more layers and on GPU.

```text
model.eval()
with torch.no_grad():
    ...
```

What this is doing:

- `.eval()` puts the model in evaluation mode (matters for layers like
  dropout / batchnorm; for plain MLPs it is mostly a no-op but a good habit).
- `torch.no_grad()` disables autograd. This is critical for inference because
  it stops PyTorch from saving activations on the tape — saving a lot of
  memory and a bit of time. This is the same idea as Part 8: inference
  memory < training memory.

You should reach ~96-98% test accuracy after 5 epochs.

## 9.2 The Three-Way Gradient Match

Now do something that almost no tutorial walks through, but every framework
contributor does on day one:

```python
import numpy as np
import torch

torch.manual_seed(0)
np.random.seed(0)

D_in, D_hid, D_out = 4, 5, 3

W1 = np.random.randn(D_in,  D_hid).astype(np.float64) * 0.1
b1 = np.zeros((1, D_hid),         dtype=np.float64)
W2 = np.random.randn(D_hid, D_out).astype(np.float64) * 0.1
b2 = np.zeros((1, D_out),         dtype=np.float64)

x = np.random.randn(8, D_in).astype(np.float64)
y = np.random.randint(0, D_out, size=(8,))

def softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)

def forward_numpy(x, W1, b1, W2, b2, y):
    z1 = x @ W1 + b1
    h  = np.maximum(z1, 0)
    z2 = h @ W2 + b2
    p  = softmax(z2)
    log_p = np.log(p[np.arange(len(y)), y] + 1e-12)
    loss = -log_p.mean()
    return loss, (z1, h, z2, p)

def backward_numpy(x, y, W2, cache):
    z1, h, z2, p = cache
    N = x.shape[0]
    dz2 = p.copy()
    dz2[np.arange(N), y] -= 1.0
    dz2 /= N

    dW2 = h.T @ dz2
    db2 = dz2.sum(axis=0, keepdims=True)
    dh  = dz2 @ W2.T
    dz1 = dh * (z1 > 0)
    dW1 = x.T @ dz1
    db1 = dz1.sum(axis=0, keepdims=True)
    return dW1, db1, dW2, db2

loss_np, cache = forward_numpy(x, W1, b1, W2, b2, y)
dW1_np, db1_np, dW2_np, db2_np = backward_numpy(x, y, W2, cache)

xt  = torch.tensor(x)
yt  = torch.tensor(y, dtype=torch.long)
W1t = torch.tensor(W1, requires_grad=True)
b1t = torch.tensor(b1, requires_grad=True)
W2t = torch.tensor(W2, requires_grad=True)
b2t = torch.tensor(b2, requires_grad=True)

logits = (torch.relu(xt @ W1t + b1t)) @ W2t + b2t
loss_t = torch.nn.functional.cross_entropy(logits, yt)
loss_t.backward()

print("loss numpy : ", loss_np)
print("loss torch : ", loss_t.item())
print("max |dW1 diff| :", np.max(np.abs(dW1_np - W1t.grad.numpy())))
print("max |dW2 diff| :", np.max(np.abs(dW2_np - W2t.grad.numpy())))
print("max |db1 diff| :", np.max(np.abs(db1_np - b1t.grad.numpy())))
print("max |db2 diff| :", np.max(np.abs(db2_np - b2t.grad.numpy())))

eps = 1e-6
i, j = 1, 2
W2_plus  = W2.copy(); W2_plus[i, j]  += eps
W2_minus = W2.copy(); W2_minus[i, j] -= eps
loss_plus, _  = forward_numpy(x, W1, b1, W2_plus,  b2, y)
loss_minus, _ = forward_numpy(x, W1, b1, W2_minus, b2, y)
finite_diff = (loss_plus - loss_minus) / (2 * eps)

print("dW2[1,2] numpy        :", dW2_np[i, j])
print("dW2[1,2] torch autograd:", W2t.grad.numpy()[i, j])
print("dW2[1,2] finite diff   :", finite_diff)
```

### Code walkthrough

```text
def softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)
```

What this is doing:

- Numerically stable softmax. Subtracting the max prevents `exp` from
  overflowing on large logits.
- This is the same trick PyTorch uses internally.

```text
def forward_numpy(x, W1, b1, W2, b2, y):
    z1 = x @ W1 + b1
    h  = np.maximum(z1, 0)
    z2 = h @ W2 + b2
    p  = softmax(z2)
    log_p = np.log(p[np.arange(len(y)), y] + 1e-12)
    loss = -log_p.mean()
    return loss, (z1, h, z2, p)
```

What this is doing:

- Forward pass for a 2-layer MLP with cross-entropy loss.
- `p[np.arange(len(y)), y]` selects the predicted probability of the true
  class for each example.
- The `+ 1e-12` avoids `log(0)` when a probability is extremely small.
- The cache `(z1, h, z2, p)` stores everything backward will need.

```text
def backward_numpy(x, y, W2, cache):
    ...
    dz2 = p.copy()
    dz2[np.arange(N), y] -= 1.0
    dz2 /= N
    ...
```

What this is doing:

- Implements the fused softmax + cross-entropy backward.
- The gradient is `p - one_hot(y)`, divided by `N` because the loss is the
  *mean* over the batch.
- This is the cleanest gradient in deep learning. Memorize it.

```text
xt  = torch.tensor(x)
W1t = torch.tensor(W1, requires_grad=True)
...
loss_t.backward()
```

What this is doing:

- Repeats the same forward pass with PyTorch tensors.
- `requires_grad=True` makes them part of the autograd tape.
- `loss_t.backward()` runs reverse-mode autodiff and stores gradients in
  `.grad` on each tensor.

```text
print("max |dW1 diff| :", np.max(np.abs(dW1_np - W1t.grad.numpy())))
```

What this is doing:

- Compares your hand-coded gradient with PyTorch's gradient elementwise.
- The expected difference is on the order of machine precision (1e-12 for
  float64). If it is larger, your backprop has a bug.

```text
eps = 1e-6
W2_plus  = W2.copy(); W2_plus[i, j]  += eps
W2_minus = W2.copy(); W2_minus[i, j] -= eps
loss_plus,  _ = forward_numpy(x, W1, b1, W2_plus,  b2, y)
loss_minus, _ = forward_numpy(x, W1, b1, W2_minus, b2, y)
finite_diff = (loss_plus - loss_minus) / (2 * eps)
```

What this is doing:

- A finite-difference gradient check on a single weight.
- Definition of derivative: `(f(w + ε) - f(w - ε)) / (2ε)`.
- This is the most direct way possible to estimate `∂L / ∂W2[1, 2]`.
- It is slow (you need two forward passes per weight), so we only do it for
  a few entries — but it is the ground truth that backprop is supposed to
  match.

If all three numbers — NumPy backprop, PyTorch autograd, and finite
differences — agree to several decimal places, you have just done what every
framework engineer does to verify a new operator.

## 9.3 What This Project Cements

By doing this once, you have:

- written a forward pass twice, in two libraries,
- written a backward pass by hand,
- verified it against autograd,
- verified that against the definition of the derivative,
- and watched a real classifier train.

You will never again wonder whether autograd is doing something mysterious.
It is doing the thing you just did, just with a billion more operations.

---

# PART 10: HOW THIS SCALES TO LLM TRAINING

The same machinery you wrote scales up to GPT, Llama, Claude, and friends.
Three things change.

## 10.1 The Model Is Bigger

Instead of `(W1, b1, W2, b2)`, an LLM has hundreds of layers, each with:

- attention projections (`Wq`, `Wk`, `Wv`, `Wo`)
- the attention mechanism itself (`softmax(QK^T / √d) · V`)
- feed-forward projections (`W_up`, `W_gate`, `W_down`)
- layer-norm parameters
- positional information (often baked into attention as RoPE)

But every single one of those operations has a forward and a backward, and
backprop chains them together exactly as in our toy network.

## 10.2 The Loss Is Cross-Entropy at Every Position

For each position in the training sequence, the model produces a logit
distribution over the vocabulary. The loss is cross-entropy against the actual
next token. The gradient of this loss is exactly what you computed in Part 9:

```text
∂L / ∂logitᵢ = pᵢ - 1{i = y}
```

And it flows back through every layer.

## 10.3 The Optimizer Is Bigger and the Memory Pattern Matters Even More

Llama-7B trained with AdamW under typical settings consumes (very roughly):

```text
weights        : 14 GB (FP16)
master weights : 28 GB (FP32)
gradients      : 14 GB
optimizer (m,v): 56 GB (FP32 each, two tensors per param)
activations    : tens of GB depending on batch and sequence
─────────────────────────────────────
total          : easily > 120 GB
```

This is why training a 7B model on a single GPU is hard, and why LLMs depend
on parallelism (FSDP, ZeRO, tensor parallelism, pipeline parallelism). Every
single one of those techniques is a rearrangement of "where does each
parameter, gradient, optimizer-state, and activation live?" — that is, a
careful answer to the questions you have been learning to ask all week.

## 10.4 Training and Inference Are Different Beasts

A useful summary:

| Phase | What runs | Memory dominated by | Performance bound by |
|-------|-----------|----------------------|----------------------|
| Inference (decode, batch=1) | forward only | weights + KV cache | HBM bandwidth |
| Inference (decode, large batch) | forward only | weights + KV caches | weights HBM bandwidth |
| Inference (prefill) | forward only | activations of prompt | compute |
| Training (forward) | forward only with tape | + activations | mixed |
| Training (backward) | matmuls in reverse | + gradients | mixed |
| Training (optimizer) | elementwise updates | + optimizer state | HBM bandwidth |

Day 1 explained inference. Today explained training. Both run on the same
silicon. The difference is what is in HBM at any given moment.

---

# PART 11: WHAT YOU NOW UNDERSTAND

Close the document and try to answer these without looking:

1. **What is the credit assignment problem and how does backprop solve it?**
2. **What is the chain rule, in two sentences?**
3. **For a matmul `Y = X @ W`, what are the gradients of the loss with respect
   to `X` and `W`?**
4. **Why is reverse-mode autodiff a better fit for neural networks than
   forward-mode?**
5. **What does `loss.backward()` actually do?**
6. **Why must you call `optim.zero_grad()` between training steps?**
7. **What are the three things in HBM during training that are not there
   during inference?**
8. **Why is activation checkpointing a good idea on a GPU?**
9. **What is the gradient of fused softmax + cross-entropy w.r.t. logits?**
10. **Why does the learning rate matter so much, and how do you debug a
    training run that diverges?**

If you can answer these, you understand backprop deeper than most engineers
who have used PyTorch for years. The rest of deep learning is mostly applying
this same set of ideas, with bigger graphs and more careful memory choices.

---

# CHECKLIST

- [ ] Can describe backprop as the chain rule applied to a computational graph
- [ ] Did the tiny network's forward and backward pass by hand
- [ ] Understands matmul gradient identities (`X.T @ dL/dY`, `dL/dY @ W.T`)
- [ ] Understands ReLU backward as a mask of the forward pre-activations
- [ ] Knows the gradient of fused softmax + cross-entropy
- [ ] Knows why the `/ N` in MSE backward matters
- [ ] Implemented backprop for an MLP in pure NumPy
- [ ] Used PyTorch autograd and inspected `grad_fn`
- [ ] Verified hand-coded gradients against autograd and finite differences
- [ ] Trained an MNIST classifier from scratch in PyTorch
- [ ] Knows why training memory is much larger than inference memory
- [ ] Understands activation checkpointing as a compute-for-memory trade
- [ ] Connects backward kernels to coalescing and shared memory from Days 1-2
- [ ] Knows the components of an Adam optimizer's state per parameter
- [ ] Knows the difference between SGD and Adam at a mental-model level

---

# DETAILED ANSWERS

## 1. What is the credit assignment problem and how does backprop solve it?

In a deep network, the loss depends on every weight in every layer, and the
contribution of any single weight is "filtered" through all the layers
downstream of it. The credit assignment problem is the question:

> Given that the final output was wrong by some amount, how much of that
> error is due to each individual weight?

Backprop solves it by walking the computational graph **in reverse**, applying
the chain rule at each operation. At every node, the gradient that arrived
from the output side is multiplied by the local derivative of that operation
to produce the gradient for the next node upstream. The result is that, for
every weight in the network, you end up with a precise number — its share of
the blame.

The reason this is fast is reuse. Computing each weight's gradient
independently would be enormously redundant. By walking the graph backward
once and reusing intermediate gradients, the cost of the whole backward pass
is roughly the cost of one forward pass.

## 2. What is the chain rule, in two sentences?

If `y = f(g(x))`, then `dy/dx = f'(g(x)) · g'(x)`. Equivalently, the
derivative of a composition is the product of derivatives, and that pattern
keeps multiplying as compositions get longer.

For neural networks this means: the gradient of the loss with respect to any
intermediate value is the product of all local derivatives along every path
from that intermediate value to the loss.

## 3. For a matmul `Y = X @ W`, what are the gradients?

Two of the most useful formulas in deep learning:

```text
∂L / ∂X = ∂L / ∂Y @ W.T
∂L / ∂W = X.T @ ∂L / ∂Y
```

What these mean:

- The gradient w.r.t. the input `X` is the upstream gradient times the
  weight, transposed.
- The gradient w.r.t. the weights `W` is the input, transposed, times the
  upstream gradient.

These hold for batched matmul too. They are the entire reason GPU GEMM
kernels also need fast transposed access patterns — backward of a forward
GEMM is itself a GEMM with one operand transposed.

## 4. Why is reverse-mode autodiff a better fit for neural networks?

A neural network has many parameters (millions to trillions) and one scalar
output (the loss). Forward-mode autodiff costs roughly one extra pass per
input dimension. Reverse-mode autodiff costs roughly one extra pass per
output dimension.

So:

```text
forward-mode cost  ≈ #parameters × forward pass    (impossibly slow)
reverse-mode cost  ≈ 1 × forward pass + 1 × backward pass    (feasible)
```

Reverse-mode wins by an enormous margin. That is why every deep learning
framework computes gradients in reverse-mode.

## 5. What does `loss.backward()` actually do?

`loss.backward()`:

1. Looks up the autograd tape that was built during the forward pass.
2. Starts with `dL/dL = 1` at the loss tensor.
3. Walks the tape in reverse topological order.
4. At each operation, calls that operation's backward formula, computing
   gradients with respect to the inputs from the gradient with respect to
   the output.
5. When a gradient reaches a tensor with `requires_grad=True` and `is_leaf`
   (typically a parameter), it is **accumulated** into that tensor's `.grad`
   attribute.
6. After the call, every parameter has its gradient ready in `.grad`, so
   the optimizer can use it.

The key word is *accumulated*. PyTorch adds new gradients to whatever was in
`.grad` previously. That is why the next answer matters.

## 6. Why must you call `optim.zero_grad()` between training steps?

Because `.grad` accumulates. If you do not zero it out, each step's gradient
will be added on top of the previous step's, and your optimizer will use the
sum of all gradients ever computed — which is wrong.

Why does PyTorch accumulate by default instead of overwriting? Because some
training patterns *want* accumulation. For example:

- Gradient accumulation across micro-batches when a desired batch size does
  not fit in memory.
- Multi-task learning where two losses contribute gradients to the same
  parameters.

For ordinary single-loss training, you must call `zero_grad()` exactly once
per step. Forgetting it is one of the most common bugs in PyTorch code.

## 7. What lives in HBM during training that does not during inference?

For a typical mixed-precision training setup with AdamW:

- **Master weights (FP32):** kept alongside the FP16 weights so that small
  gradient updates don't get lost to FP16 rounding.
- **Gradients:** one tensor per parameter, same shape as the parameter.
- **Optimizer state:** for AdamW, two extra tensors per parameter (momentum
  `m` and squared-gradient running average `v`), usually in FP32.
- **Activations from the forward pass:** stored on the autograd tape so the
  backward pass can use them.

Inference does not need master weights, gradients, or optimizer state. It
only needs the model weights themselves and a small KV cache. That is why a
model that takes 14 GB to do inference can take 100+ GB to train.

## 8. Why is activation checkpointing a good idea on a GPU?

Activation checkpointing trades compute for memory:

- Without checkpointing, every forward activation that is needed by the
  backward pass is stored in HBM.
- With checkpointing, only some activations are stored. The rest are
  recomputed on the fly during the backward pass.

Why is this a *good* trade on a GPU?

1. From Day 1, we know LLM training is heavily memory-pressure-limited. Memory
   buys you bigger batch sizes, longer sequences, or fewer GPUs.
2. The recompute cost is a small fraction of total compute (typically ~33%
   extra compute per step).
3. Modern GPUs have plenty of compute headroom relative to HBM capacity.

So spending a little more arithmetic to save a lot of memory is almost
always worth it. This is why frameworks like PyTorch and DeepSpeed expose
`torch.utils.checkpoint` and similar APIs.

## 9. What is the gradient of fused softmax + cross-entropy?

For logits `z` over `K` classes, predicted distribution `p = softmax(z)`,
true class `y`, and loss `L = -log(p_y)`:

```text
∂L / ∂zᵢ = pᵢ - 1{i = y}
```

What this means:

- Subtract 1 from the predicted probability of the *correct* class.
- Leave all other entries equal to their predicted probabilities.
- Done.

This is so clean that:

- PyTorch's `nn.CrossEntropyLoss` requires logits, not probabilities, so it
  can use this fused gradient.
- LLM training loss kernels almost always implement softmax + cross-entropy
  together.
- The backward pass at the very last step of an LLM is essentially this one
  formula, applied to every position in every sequence in the batch.

If you understand only one gradient in deep learning, make it this one.

## 10. Why does the learning rate matter so much, and how do you debug a
training run that diverges?

The learning rate `η` controls the *step size* of the gradient descent
update. It is the multiplier between "how wrong we are" and "how big a change
we make."

If `η` is too large:

- Updates overshoot the minimum.
- The next step's gradient is even larger.
- The loss explodes, often to NaN.

If `η` is too small:

- Updates are tiny.
- The model barely changes per step.
- After a long time, the loss has barely moved.

Debugging a divergent training run, in priority order:

1. **Lower the learning rate by 10× and try again.** If that fixes it, you
   were too aggressive. Often this is enough.
2. **Check for NaNs in the data, the loss, or the activations.** A single bad
   sample can poison a step. Some engineers add `assert torch.isfinite(loss)`
   in the training loop.
3. **Use gradient clipping** (`torch.nn.utils.clip_grad_norm_`). This
   bounds the size of any single update step, which protects against rare
   gradient spikes. Standard for LLM training.
4. **Check normalization layers.** A misconfigured layer-norm or batch-norm
   can produce huge activations.
5. **Re-check the loss function.** A common mistake is feeding probabilities
   into a loss that expects logits, or vice versa.
6. **Re-check optimizer state initialization.** Carrying old optimizer state
   across an architectural change can blow things up.

Once a run is stable, training-rate scheduling matters: warming up from a
small `η` to the target `η` over a few thousand steps prevents the first few
updates from being too large at random initialization. Then, after a long
flat phase, decaying `η` toward the end of training squeezes out a bit more
quality.

This is the kind of thing that looks like superstition until you have lived
through one diverged 7B-parameter run.

---

**Tomorrow (Day 4): Warps, Divergence, and the SIMT Execution Model** — why
your CUDA kernel sometimes runs at a fraction of its theoretical speed even
when memory is well-coalesced and shared memory is conflict-free. Welcome to
how the GPU actually executes your code.

---

*Status: ⬜ NOT YET COMPLETED*
*Date completed: ___________*
