# LLM + CUDA + Hardware Mastery Roadmap

> **Daily: 2-3 hours** | **Saturday: 3-4 hrs project** | **Sunday: 1-2 hrs papers (optional)**
> **Duration:** ~18-24 months | **6-month milestone:** serious systems projects + strong LLM/GPU fundamentals
> **After each month:** 1 week revision buffer to revisit weak areas + complete monthly projects
> **Projects:** 1 weekly project + **2 monthly projects** (novel, cumulative, useful to other people)

---

## North Star

Become the kind of LLM systems engineer who can reason across the full stack:

```text
model behavior -> Transformer math -> PyTorch ops -> CUDA/Triton kernels
-> GPU memory behavior -> serving/runtime tradeoffs -> product-quality systems
```

After this roadmap, you should not merely "know LLMs." You should be able to:

- Build models, training loops, inference systems, RAG systems, agents, and GPU kernels from first principles
- Explain performance using hardware facts instead of vague guesses
- Read new papers and turn them into working code
- Build tools that save other engineers time, money, debugging effort, or learning effort
- Ship portfolio projects that look like real open-source products, not course assignments

The journey is deliberately hard. The reward is rare judgment: you will understand both **what the model is doing** and **why the GPU/system behaves the way it does**.

---

## How To Use This File

- **Check off items** as you complete them: change `[ ]` to `[x]`
- Each **day = 2-3 hours** of focused work
- Days alternate: **CUDA/Hardware** days and **ML/DL/LLM** days (or split 50/50)
- **Saturday** = weekly project (bigger coding session)
- **Sunday** = rest + optionally read the listed paper
- If a day takes longer, split it across 2 days — no rush
- **After every month:** take 1 extra week to revise everything from that month + previous months
- **Monthly projects:** complete 2 per month — each must use skills from current AND previous months
- **Use AI (Cursor/Codex/Claude)** as your tutor, code reviewer, paper explainer, and ruthless reviewer
- Keep a short engineering journal: what you built, what broke, what you measured, what you would change
- For every serious project, write a public-quality README and at least one technical note or blog post

---

## Project Philosophy: Novel + Useful + Cumulative

Every month has two projects because expertise needs two muscles:

- **Project A:** deep implementation of the main technical theme
- **Project B:** a useful tool/product that packages the learning for other people

For Month `N`, both projects must use Month `1...N` skills. The project should have visible evidence of that cumulative learning:

- **Month 1+ hardware proof:** benchmarks, roofline thinking, memory/compute bottleneck analysis
- **Month 2+ model proof:** Transformer/LLM internals, training/inference behavior, profiling
- **Month 3+ optimization proof:** custom CUDA/Triton extension, Flash Attention idea, distributed/fine-tuning awareness
- **Month 4+ compression proof:** quantization/PEFT/cost-vs-quality tradeoff
- **Month 5+ product proof:** serving, RAG, latency, evaluation, deployment ergonomics
- **Month 6+ reasoning proof:** tool use, memory, observability, safety, RLHF/reasoning evaluation
- **Month 7+ multimodal proof:** text/image/document/audio routing where useful
- **Month 9+ kernel proof:** measurable custom kernel or compiler/fusion insight
- **Month 10+ scale proof:** distributed, multi-GPU, failure diagnosis, cost model
- **Month 11+ research proof:** paper-to-code implementation or reproducibility analysis
- **Month 13+ production proof:** monitoring, routing, guardrails, API design, cost tracking
- **Month 16+ community proof:** external users, docs, issues, PRs, adoption

### Novelty Scorecard

Before starting any monthly project, fill this in. If the score is weak, redesign the project.

| Question | Must Be True |
|----------|--------------|
| Who is this for? | A specific user exists: ML engineer, researcher, student, infra engineer, startup, etc. |
| What pain does it remove? | It saves time, money, debugging effort, learning effort, or deployment risk |
| What is new about it? | A new combination, clearer visualization, better automation, better benchmark, or missing open-source tool |
| What will I measure? | Speed, cost, quality, accuracy, memory, latency, user time saved, or reproducibility |
| What previous months are visible? | The README explicitly says which Month 1...N skills appear and where |
| Can someone else use it? | It has setup instructions, examples, tests, and sane defaults |
| Would I be proud to show it in an interview? | The answer must be yes before calling it done |

### Definition Of Done For Monthly Projects

A monthly project is not done when the code "runs once." It is done when it has:

- [ ] Clear problem statement: who it helps and why it matters
- [ ] Working MVP with a reproducible setup command
- [ ] Benchmarks or evaluation against at least one baseline
- [ ] Architecture diagram or system overview
- [ ] Cumulative-learning section: "Skills from Month 1...N used here"
- [ ] Tests or validation scripts for the core behavior
- [ ] Failure analysis: where it breaks, limits, tradeoffs, next improvements
- [ ] Demo: notebook, CLI, web UI, screenshots, or video
- [ ] Public-quality README with install, usage, examples, and results
- [ ] One short technical write-up explaining the hardest idea

### Anti-Tutorial Rule

You may learn from tutorials, but final projects must not look like tutorials. To pass:

- Add one original feature that solves a real user pain
- Compare against a baseline and publish the numbers
- Explain a design decision using hardware/model/system reasoning
- Make the project reusable by someone who is not you

### Expertise Gates

Use these gates to decide whether you are truly leveling up:

| Point | You Should Be Able To Do |
|-------|---------------------------|
| End of Month 1 | Explain why a CUDA kernel is slow using memory access, occupancy, and bandwidth evidence |
| End of Month 3 | Build and fine-tune a small Transformer, profile it, and replace one slow operation with custom CUDA/Triton |
| End of Month 6 | Ship an end-to-end AI application with serving, RAG/agents, reasoning, safety, observability, and evaluation |
| End of Month 10 | Reason about multi-GPU training/inference systems, cost, throughput, bottlenecks, and failure modes |
| End of Month 13 | Design production LLM infrastructure with routing, caching, monitoring, alignment, and guardrails |
| End of Month 18 | Lead a serious LLM systems project from idea to benchmarked, documented, usable release |

If a gate feels weak, pause and build one more small project before moving on. Speed is less important than compounding correctly.

---

## Progress Overview

| Phase | Weeks | Months | Status |
|-------|-------|--------|--------|
| Phase 1: Foundations | 1-12 | 1-3 | 🟡 In Progress |
| Phase 2: Intermediate | 13-26 | 4-6 | ⬜ Not Started |
| Phase 3: Advanced | 27-40 | 7-10 | ⬜ Not Started |
| Phase 4: Expert | 41-52 | 11-13 | ⬜ Not Started |
| Phase 5: Mastery | 53-65 | 14-16 | ⬜ Not Started |
| Phase 6: Capstone | 66-78 | 17-18 | ⬜ Not Started |

---

# ═══════════════════════════════════════════════════════════
# MONTHLY CAPSTONE PROJECT CATALOG (Build Only When Reached)
# ═══════════════════════════════════════════════════════════

> **TWO projects per month.** Each must combine skills from the current month AND
> all previous months. Each must have a **novelty angle** — something that isn't
> just a tutorial copy-paste but demonstrates real understanding and creativity.
> This is a catalog of future project briefs. Build each project only during the
> **revision week** after the required weeks have been completed.
> These are the projects you put on your resume and GitHub.
>
> **Project A** = structured (detailed below). **Project B** = novel second project (also detailed below).
> Both must use current month + all previous month skills. Built during revision week.

Each project description below is a starting brief, not a cage. If you discover a sharper user pain while studying, improve the project, but keep the cumulative rule: Month `N` must visibly use Month `1...N`.

For strict chronological study, follow the **Phase 1 → Phase 6** weekly sections
below. Treat this catalog as the project reference that each revision week points
back to.

Every monthly project README should include:

- **User:** who this helps
- **Pain:** what annoying/expensive/confusing thing it fixes
- **Novelty:** what is different from existing tools or tutorials
- **Cumulative skills:** which months are used and where in the code
- **Evidence:** benchmark, evaluation, demo, screenshots, or reproducibility report
- **Next users:** how someone else can install it, run it, and trust it

---

## Month 1 Project: "GPU Matrix Math Engine"
**When to build:** Month 1 revision week, after Weeks 1-4.
**What:** A CUDA library that implements matrix operations and GEMM using only Month 1 concepts, with a benchmarking dashboard.
**Novelty:** Auto-selects the best Month-1 kernel based on matrix dimensions and memory pattern. Generates a roofline-style plot for YOUR specific GPU using measured bandwidth/FLOPS from your own kernels.
**Deliverables:**
- [ ] 4 GEMM implementations: CPU baseline, naive GPU, coalesced GPU, shared-memory tiled GPU
- [ ] Block-size and tile-size benchmark sweep with an automatic kernel selection heuristic
- [ ] Python benchmark script generating roofline-style plot (matplotlib)
- [ ] CUDA-event timing and effective bandwidth/FLOPS report for each kernel
- [ ] README with performance analysis and architecture diagrams
- [ ] **Future extension after Month 2:** add cuBLAS baseline and Nsight Compute report
- [ ] **Future extension after Month 3:** add register-tiled or Triton kernel
- [ ] **Publish to GitHub**

## Month 1 Project B: "LLM Inference Cost Simulator"
**When to build:** Month 1 revision week, after Weeks 1-4.
**What:** A visual, interactive tool that estimates LLM weight memory, bandwidth-bound decode speed, and GPU bottlenecks using only hardware facts you learned in Month 1.
**Novelty:** Make the hidden bottleneck visible: "a 7B FP16 model needs roughly this many bytes, this much HBM bandwidth per generated token, and this is why decode can become memory-bound." Every number must come from a transparent formula.
**Skills used:** CUDA memory hierarchy (Month 1), HBM bandwidth (Month 1), roofline thinking (Month 1), arithmetic intensity (Month 1), GPU architecture (Month 1)
**Deliverables:**
- [ ] Input: parameter count, bytes per parameter, GPU HBM bandwidth, HBM capacity, approximate FLOPS
- [ ] Output: model size, maximum memory-bound tokens/sec, memory fit/not-fit, bottleneck explanation
- [ ] Animated visualization of data flowing: HBM → L2/cache → SM → registers/shared memory
- [ ] Compare: 4-byte vs 2-byte weight storage scenarios (FP32-like vs FP16-like) and bandwidth pressure
- [ ] Explain assumptions clearly: batch=1 decode, weights read once per token, ignoring KV-cache until Month 4/5
- [ ] **Future extension after Month 4:** add INT8/INT4 quantization
- [ ] **Future extension after Month 5:** add KV-cache, prefill/decode split, batching, and serving metrics
- [ ] Web UI (Streamlit) or interactive notebook
- [ ] **Publish to GitHub**

---

## Month 2 Project A: "nanoLLM — Your Own GPT from Scratch"
**What:** A complete, clean, well-documented GPT-2 (124M) implementation trained on a real corpus with a chat interface.
**Novelty:** Not just the model — include a **hardware utilization dashboard** that shows during training: GPU utilization %, memory usage, Tensor Core utilization, tokens/sec. This is what distinguishes you from everyone who followed Karpathy's tutorial.
**Deliverables:**
- [ ] Clean GPT-2 implementation: LLaMA-style (RMSNorm, SwiGLU, RoPE)
- [ ] Training with AMP, gradient checkpointing, cosine LR schedule
- [ ] KV-cache inference with all sampling strategies (temp, top-k, top-p)
- [ ] Real-time training dashboard (wandb or custom) showing hardware metrics
- [ ] Chat interface (terminal-based) with proper chat template
- [ ] Training loss curves + sample generations at checkpoints
- [ ] **Publish to GitHub with trained weights**

## Month 2 Project B: "TransformerScope — LLM Behavior Debugger"
**What:** An interactive debugger for small Transformer models that shows what changed inside the model when an output changed.
**Novelty:** Combines interpretability with GPU profiling: not only "which head attended where?" but "which layer/head/MLP changed the prediction, and what did it cost on the GPU?"
**Skills used:** Transformer architecture (Month 2), residual stream/logit lens (Month 2), GPU profiling (Month 2), CUDA memory + matmul (Month 1)
**Deliverables:**
- [ ] Load nanoLLM checkpoints and small HuggingFace Transformer models
- [ ] Attention heatmaps across layers/heads
- [ ] Residual stream + logit lens view: how token probabilities evolve by layer
- [ ] Head/MLP ablation: disable a component and measure output probability changes
- [ ] Prompt-diff mode: compare two prompts and show which internal activations changed most
- [ ] GPU overlay: per-layer latency, memory, and tensor shapes
- [ ] Interactive web UI with saved debugging reports
- [ ] **Publish with 5 case studies: factual recall, copying, induction, refusal, hallucination**

---

## Month 3 Project A: "LLM Surgery — Fine-Tuning & Alignment Toolkit"
**What:** Take a pre-trained model and build a complete fine-tuning → alignment → deployment pipeline.
**Novelty:** **Side-by-side comparison tool** — fine-tune the SAME base model with SFT only, SFT+DPO, and prompt-only approaches, then generate a comparative evaluation report showing exactly where alignment improves outputs (with specific examples and metrics).
**Deliverables:**
- [ ] Custom CUDA kernel for at least one operation (fused LayerNorm or fused attention)
- [ ] SFT fine-tuning on instruction dataset
- [ ] DPO alignment on preference dataset
- [ ] Automated comparison tool: runs same prompts through all 3 versions
- [ ] Evaluation report: perplexity, response quality (LLM-as-judge), safety
- [ ] Triton kernel for one operation, benchmarked vs PyTorch native
- [ ] **Publish with comparison report as interactive HTML**

## Month 3 Project B: "TokenScope — Tokenizer Analyzer & Comparison Tool"
**What:** A tool that takes any text and shows how different tokenizers (GPT-2, LLaMA, Mistral, tiktoken) split it, with efficiency metrics, custom Triton kernel for fast tokenization stats, and insights on why some models handle certain languages better.
**Novelty:** Visual side-by-side comparison of 4+ tokenizers on the same text + a "tokenization efficiency score" per language/domain, focused on practical model-selection decisions.
**Skills used:** BPE tokenization (Month 1), Triton kernels (Month 3), custom CUDA extensions (Month 3), Flash Attention understanding (Month 3), LLM fine-tuning (Month 3)
**Deliverables:**
- [ ] Side-by-side tokenization of 4+ tokenizers on same input
- [ ] Efficiency metrics: tokens per word, vocab coverage, compression ratio
- [ ] Language analysis: which tokenizer handles Hindi, code, math best?
- [ ] Custom Triton kernel for batch tokenization stats (GPU-accelerated)
- [ ] Web UI showing colored token boundaries on input text
- [ ] **Publish to GitHub**

---

## Month 4 Project A: "QuantBench — The Quantization Analyzer"
**What:** A tool that takes ANY HuggingFace model and quantizes it across multiple methods, then generates a comprehensive quality-vs-speed report.
**Novelty:** A single workflow that runs GPTQ, AWQ, GGUF, and FP8 on the same model and generates a unified quality/speed/memory dashboard. You're building the quantization benchmark tool you wish existed.
**Deliverables:**
- [ ] Support for GPTQ (4-bit, 8-bit), AWQ (4-bit), GGUF (Q4_K_M, Q5_K_M, Q8_0)
- [ ] FP8 quantization with Transformer Engine (if Hopper GPU available)
- [ ] Automated benchmarking: perplexity, MMLU subset, throughput (tok/s), memory (GB), TTFT, TPOT
- [ ] LoRA fine-tuning on quantized base (QLoRA) included in comparison
- [ ] Interactive HTML dashboard with comparison charts
- [ ] CLI tool: `quantbench --model meta-llama/Llama-3-8B --methods gptq,awq,gguf`
- [ ] **Publish as pip-installable package**

## Month 4 Project B: "LLM-Speedometer — Real-Time Inference Profiler"
**What:** A middleware that wraps any LLM serving endpoint (vLLM, TensorRT-LLM, local) and shows real-time: tokens/sec, TTFT, TPOT, GPU utilization, KV-cache usage, and identifies the bottleneck (compute vs memory vs network).
**Novelty:** Not just static benchmarks — LIVE profiling during actual conversations. Shows a real-time dashboard that updates as you chat. Identifies "your model is 73% memory-bound, quantizing to INT4 would give 3.2x speedup."
**Skills used:** Quantization (Month 4), inference serving (Month 4), CUDA profiling (Month 2), memory bandwidth analysis (Month 1), KV-cache (Month 4)
**Deliverables:**
- [ ] Middleware: wraps any OpenAI-compatible endpoint
- [ ] Real-time dashboard: tok/s, TTFT, TPOT, GPU util, KV-cache occupancy
- [ ] Bottleneck identification: "memory-bound: quantize" or "compute-bound: batch more"
- [ ] Recommendation engine: suggests optimization based on profiling data
- [ ] Historical tracking: plot performance over time
- [ ] **Publish to GitHub with demo GIF**

---

## Month 5 Project A: "DeepRAG — Production RAG with NVIDIA Stack"
**What:** End-to-end RAG system that can ingest any document type and answer questions with source citations, using NVIDIA tools throughout.
**Novelty:** **RAG strategy auto-selector** — the system automatically decides whether to use naive retrieval, hybrid retrieval, graph RAG, or agentic RAG based on question complexity. Includes a built-in evaluation pipeline that scores every response.
**Deliverables:**
- [ ] Document ingestion: PDF (with tables/images), markdown, web scraping, code repos
- [ ] Multiple retrieval backends: FAISS GPU, Milvus, with NV-EmbedQA embeddings
- [ ] Hybrid retrieval (dense + BM25) with NV-RerankQA cross-encoder
- [ ] Graph RAG: auto-build knowledge graph from documents
- [ ] Agentic RAG: agent decides when to retrieve, when to ask clarifying questions
- [ ] **Strategy auto-selector**: classifies query complexity → picks retrieval strategy
- [ ] Built-in RAGAS evaluation on every query (show confidence score to user)
- [ ] FastAPI server + simple web UI
- [ ] NIM for LLM inference backend
- [ ] **Publish with Docker Compose for one-command deployment**

## Month 5 Project B: "RAGTrace — Retrieval Debugger & Evaluation Lab"
**What:** A diagnostic tool that wraps any RAG pipeline and records every retrieval decision behind an answer.
**Novelty:** Turns RAG from a black box into a flight recorder: query rewrite, retrieved chunks, reranker scores, context packing, citations, answer claims, and failure reason in one trace.
**Skills used:** RAG (Month 5), vector search (Month 5), RAG evaluation (Month 5), serving (Month 5), quantization/cost tradeoffs (Month 4), CUDA memory/perf basics (Month 1)
**Deliverables:**
- [ ] Plug-in wrapper for LangChain/LlamaIndex/custom RAG pipelines
- [ ] Trace viewer: query → rewritten query → retrieved chunks → reranked chunks → final context → answer
- [ ] Failure classifier: missing retrieval, bad chunking, weak reranking, unsupported claim, stale document
- [ ] Chunking/reranking experiment runner with side-by-side quality metrics
- [ ] Citation verifier: every generated claim mapped to source span or marked unsupported
- [ ] Cost/latency breakdown per query: embedding, retrieval, rerank, generation
- [ ] Public benchmark on at least 3 document sets
- [ ] **Publish as a reusable RAG debugging tool**

---

## Month 6 Project A: "AgentForge — Reasoning Agent Platform" ⭐ (MILESTONE PROJECT)
**When to build:** Month 6 revision week, after Weeks 21-26.
**What:** A platform where tool-using agents solve difficult tasks with ReAct, chain-of-thought prompting, self-consistency, verifier scoring, and long-context retrieval.
**Novelty:** **Reasoning performance profiler** — tracks every LLM call, tool use, reasoning branch, verifier score, retrieval step, and cost. It shows exactly why an agent succeeded or failed, not just the final answer.
**Deliverables:**
- [ ] Agent framework: define agents with roles, tools, system prompts
- [ ] Tools: code execution (sandboxed), web search, file system, RAG retrieval, API calls
- [ ] Reasoning modes: ReAct, chain-of-thought prompting, Tree of Thought, self-consistency
- [ ] Multi-agent: supervisor pattern + peer collaboration for hard tasks
- [ ] Memory: short-term (conversation), long-term (vector store), shared (between agents)
- [ ] Verifier/reward scoring: rank candidate answers and reasoning paths
- [ ] Long-context mode: retrieve or compress context when tasks exceed the model window
- [ ] NeMo Guardrails integration for safety
- [ ] **Visual execution trace**: Gantt-chart-style view of agents, tools, reasoning branches, verifier decisions
- [ ] Cost/latency tracker: tokens used per agent, cost per task, TTFT/TPOT where available
- [ ] Streaming REST API with WebSocket for real-time agent output
- [ ] Evaluate on 3 real tasks: research report, code debugging, math/reasoning task
- [ ] **Publish with demo video and live hosted demo**

## Month 6 Project B: "EvalArena — Model, Cost, and Safety Battleground"
**What:** A self-hosted arena that compares models across answer quality, latency, cost, safety, and tool-use reliability.
**Novelty:** Most arenas ask "which answer is better?" This asks "which model should I deploy for this workload?" by combining human votes, LLM-as-judge, safety checks, latency, and dollar cost.
**Skills used:** Agents/tool use (Month 5), safety/guardrails (Month 5), RAG (Month 5), serving metrics (Month 5), quantization/cost tradeoffs (Month 4), RLHF/reasoning (Month 6), long-context evaluation (Month 6), evaluation basics (all prior)
**Deliverables:**
- [ ] Load HuggingFace, GGUF, NIM, vLLM, or OpenAI-compatible endpoints
- [ ] Side-by-side comparison UI with blind human voting
- [ ] Automated LLM-as-judge with agreement analysis against human votes
- [ ] Cost/latency/TTFT/TPOT dashboard per model
- [ ] Safety and refusal tests using guardrail policies
- [ ] Tool-use benchmark: function calling correctness and argument validity
- [ ] Reasoning benchmark: CoT/self-consistency/process-reward comparison on hard tasks
- [ ] Long-context benchmark: passkey/NIAH/document QA with cost and latency
- [ ] ELO-style leaderboard plus workload-specific recommendation: cheapest acceptable model
- [ ] **Publish with web UI, seed eval set, and reproducible reports**

---

## Month 7 Project A: "IncidentLens — Multimodal AI Debugging Assistant"
**What:** A multimodal assistant for debugging ML/system incidents from screenshots, logs, traces, code snippets, PDFs, and spoken notes.
**Novelty:** Turns messy incident evidence into a root-cause timeline: "screenshot shows OOM, logs show batch size jump, profiler trace shows HBM pressure, docs explain the config knob."
**Deliverables:**
- [ ] Input router for screenshots, logs, PDFs, traces, code snippets, and voice notes
- [ ] Image understanding for error screenshots and dashboard captures
- [ ] Log parser that extracts exceptions, timestamps, GPU IDs, OOMs, latency spikes
- [ ] Document/PDF lookup for relevant runbooks or API docs
- [ ] RAG-backed root-cause report with cited evidence
- [ ] Suggested fix plan ranked by risk and expected impact
- [ ] GPU-accelerated preprocessing with DALI where useful, NIM/VLM for inference
- [ ] **Publish with 5 real or simulated incident case studies**

## Month 7 Project B: "GPU-Accelerated PDF Intelligence"
**What:** A pipeline that takes any PDF (research papers, financial reports, technical docs) and extracts ALL content — text, tables, charts, images, equations — using GPU-accelerated processing, then makes it searchable and question-answerable.
**Novelty:** Most PDF tools lose tables and charts. Yours extracts tables as structured data, interprets charts as data, reads equations as LaTeX, and stores everything in a multimodal RAG index.
**Skills used:** Multi-modal (Month 7), RAG (Month 5), NVIDIA DALI (Month 7), vision models (Month 7), all prior
**Deliverables:**
- [ ] PDF → extract text (OCR for scanned), tables (structured), charts (interpreted), equations (LaTeX)
- [ ] GPU-accelerated: DALI for image processing, NIM VLM for chart interpretation
- [ ] Multimodal RAG: query text, tables, and images with natural language
- [ ] "Explain this chart" feature: point to a chart, get a text explanation
- [ ] **Publish to GitHub**

---

## Month 8 Project A: "MoE-Lab — Mixture of Experts Playground"
**What:** Train and analyze Mixture of Experts models at small scale, with tools to visualize expert specialization.
**Novelty:** **Expert specialization visualizer** — shows WHAT each expert learned: which token types, topics, or tasks activate each expert, and how routing changes during training.
**Deliverables:**
- [ ] MoE Transformer implementation (top-2 routing, load balancing loss)
- [ ] Training on diverse text corpus (code + English + math)
- [ ] Expert activation analysis: heatmap of which experts fire for which input types
- [ ] Expert pruning experiment: remove experts and measure quality impact
- [ ] Compare: MoE vs dense model with same active parameters
- [ ] Interactive visualization dashboard
- [ ] **Publish with analysis report as blog post**

## Month 8 Project B: "ArchSearch — Neural Architecture Comparator"
**What:** A tool that lets you define any Transformer variant (dense, MoE, SSM hybrid, different attention patterns) at small scale, trains each on the same data, and generates a comprehensive comparison.
**Novelty:** One-command architecture comparison. "Is MoE 8x7B better than dense 7B for code? For math? For multilingual?" with hard data, not vibes.
**Skills used:** MoE (Month 8), Transformer variants (Month 8), Mamba/SSM (prior), training (Month 2-3), profiling (Month 2), all prior
**Deliverables:**
- [ ] Define architectures in YAML: dense, MoE, hybrid attention+SSM
- [ ] Auto-train each at matched compute budget
- [ ] Comparison dashboard: loss curves, per-domain accuracy, throughput, memory
- [ ] "Architecture recommendation" given constraints (memory, speed, task type)
- [ ] **Publish with blog post comparing 5+ architectures**

---

## Month 9 Project A: "KernelSmith — Custom Triton Kernel Library for LLMs"
**What:** A library of hand-optimized Triton kernels that speed up LLM training and inference, as a drop-in replacement for PyTorch operations.
**Novelty:** **Automatic performance regression testing** — CI pipeline that benchmarks every kernel against PyTorch native on multiple GPU architectures and shows speedup/regression in a dashboard. This is how real NVIDIA engineers work.
**Deliverables:**
- [ ] Fused attention (simplified Flash Attention in Triton)
- [ ] Fused RMSNorm + residual add
- [ ] Fused SwiGLU
- [ ] Fused rotary position embedding
- [ ] Fused cross-entropy loss
- [ ] Fused AdamW optimizer step
- [ ] Benchmark suite: each kernel vs PyTorch native, across batch sizes and sequence lengths
- [ ] Integration: drop-in replacement for `nn.Module` subclasses
- [ ] **Publish as pip-installable library with CI benchmarks**

## Month 9 Project B: "FusionLab — Transparent PyTorch-to-Triton Fusion Playground"
**What:** A tool that takes common LLM op patterns and shows how they can be fused into Triton kernels, with generated code, benchmarks, and explanations.
**Novelty:** A transparent mini-compiler for learning and benchmarking fusion. Instead of pretending to replace `torch.compile`, it teaches exactly why fusion helps and when it fails.
**Skills used:** Triton (Month 9), torch.compile/Inductor concepts (Month 9), CUDA kernels (Month 1-4), profiling (Month 2), all prior
**Deliverables:**
- [ ] Pattern matcher for common LLM blocks: RMSNorm+residual, RoPE, SwiGLU, GELU, dropout+add
- [ ] Generated Triton code shown side-by-side with original PyTorch
- [ ] Benchmark vs eager PyTorch and `torch.compile`
- [ ] Memory-traffic report: bytes read/written before and after fusion
- [ ] Correctness tests with tolerance checks across FP32/FP16/BF16
- [ ] Explanation panel: why this fusion helps, why it may not, and what hardware limit appears
- [ ] **Publish as an educational benchmarking tool**

---

## Month 10 Project A: "TrainScale — Distributed LLM Training Framework"
**What:** A simplified but functional distributed training framework that supports data parallelism + tensor parallelism.
**Novelty:** **Training cost estimator** — given a model config, dataset size, and GPU type, predicts training time, cost, and optimal parallelism strategy BEFORE you start training.
**Deliverables:**
- [ ] Data-parallel training with gradient sync
- [ ] Tensor-parallel linear layers (column + row parallel)
- [ ] Mixed-precision training (BF16 + FP32 master weights)
- [ ] Efficient data loading with pre-tokenized datasets
- [ ] Training cost estimator: time, cost, optimal DP/TP/PP config
- [ ] Comprehensive logging: loss, gradient norms, throughput, GPU utilization
- [ ] Checkpointing with resume support
- [ ] **Publish with calculator web tool**

## Month 10 Project B: "TrainWatch — Distributed Training Debugger"
**What:** A real-time monitoring tool for distributed training that shows per-GPU utilization, communication overhead, gradient norms per layer, and automatically detects training issues (loss spikes, dead layers, communication bottlenecks).
**Novelty:** Existing tools (wandb, tensorboard) show loss curves. Yours shows HARDWARE-level training health: "GPU 3 is 20% slower because NVLink bandwidth is saturated" or "Layer 15 has exploding gradients."
**Skills used:** Distributed training (Month 10), NCCL (Month 3), profiling (Month 2), all CUDA fundamentals, all prior
**Deliverables:**
- [ ] Real-time per-GPU dashboard: SM util, memory, communication time
- [ ] Per-layer gradient norm tracking with anomaly detection
- [ ] Communication profiler: AllReduce time vs compute time per step
- [ ] Auto-diagnose: "training unstable at step 4500 — gradient spike in layer 22"
- [ ] **Publish to GitHub**

---

## Month 11 Project A: "PaperBot — AI Research Paper Implementer"
**What:** An AI agent that takes an arxiv paper URL, reads it, explains it, and generates a working implementation skeleton.
**Novelty:** Goes beyond summarization — actually **extracts the algorithm, generates pseudocode, then generates runnable PyTorch code** for the key contribution. Uses multiple agent steps: parse → understand → plan → implement → verify.
**Deliverables:**
- [ ] Paper ingestion: arxiv URL → download → parse PDF
- [ ] Section-by-section summarization with key insights
- [ ] Algorithm extraction: find the novel method, extract pseudocode
- [ ] Code generation: PyTorch implementation of the key algorithm
- [ ] Self-verification: run generated code, check for errors, fix
- [ ] Test on 5 recent papers from different areas (attention, training, inference, alignment, agents)
- [ ] **Publish with demo on 10 implemented papers**

## Month 11 Project B: "Paper2Benchmark — Auto-Generate Evaluations from Papers"
**What:** Given an AI paper, automatically extract the claimed results, generate a benchmark script, and verify whether an open-source model achieves the reported numbers.
**Novelty:** "Does LLaMA-3 actually score 79.2% on MMLU as the paper claims?" — auto-run the evaluation and compare. A reproducibility verification tool.
**Skills used:** Paper reading (Month 11), agents (Month 5), evaluation (Month 10), RAG (Month 5), all prior
**Deliverables:**
- [ ] Parse paper → extract claimed benchmark scores
- [ ] Auto-generate evaluation script using lm-evaluation-harness
- [ ] Run evaluation on specified model
- [ ] Compare: "Paper claims 79.2%, we measured 78.8% (within margin ✓)"
- [ ] Reproducibility report with confidence intervals
- [ ] **Publish to GitHub**

---

## Month 12 Project A: "ReasonEngine — Test-Time Compute for Better Answers"
**What:** A system that makes any LLM dramatically smarter at hard problems by spending more compute at inference time.
**Novelty:** **Adaptive compute budget** — automatically detects question difficulty and allocates compute accordingly. Easy question = 1 pass. Hard math = 64 samples + MCTS + verification.
**Deliverables:**
- [ ] Difficulty classifier: predicts how hard a question is (few-shot, embedding-based)
- [ ] Multiple strategies: best-of-N, self-consistency, Tree of Thought, MCTS
- [ ] Process reward model: scores each reasoning step
- [ ] Adaptive router: selects strategy based on difficulty + compute budget
- [ ] Evaluation: GSM8K, MATH, HumanEval — show accuracy vs compute curve
- [ ] Cost tracking: show $/question for each difficulty level
- [ ] **Publish with interactive demo**

## Month 12 Project B: "ThinkTrace — Reasoning Chain Visualizer"
**What:** A tool that takes any reasoning model's output (CoT, ToT, MCTS) and visualizes the reasoning TREE — showing which paths were explored, which were pruned, where the model backtracked, and the final answer path highlighted.
**Novelty:** Reasoning is a black box. Your tool makes it visible. See the tree of thoughts, the dead ends, the winning path — all in an interactive graph.
**Skills used:** Reasoning/MCTS (Month 12), process reward models (Month 12), agents (Month 5), all prior
**Deliverables:**
- [ ] Capture reasoning traces from CoT, ToT, MCTS, best-of-N
- [ ] Interactive tree visualization (D3.js or similar)
- [ ] Color-coded paths: green (selected), red (pruned), yellow (explored)
- [ ] Per-step scores from process reward model
- [ ] Compare reasoning strategies visually on the same problem
- [ ] **Publish to GitHub with demo**

---

## Month 13 Project A: "NVServe — Production LLM Serving Platform"
**What:** A production-grade LLM serving platform built on the NVIDIA stack, with everything you'd need to run LLMs in a real company.
**Novelty:** **Unified dashboard** that shows everything in one place: model performance, cost per query, safety violations, user satisfaction, A/B test results. Most serving setups have monitoring scattered across 5 tools.
**Deliverables:**
- [ ] NIM-based inference with auto-scaling
- [ ] Multi-model serving: route by complexity, cost, or latency target
- [ ] Semantic caching: reuse answers for similar questions
- [ ] NeMo Guardrails: input/output safety filtering
- [ ] A/B testing: compare model versions with statistical significance
- [ ] Unified monitoring dashboard (Grafana or custom)
- [ ] API key management and rate limiting
- [ ] **Publish with Kubernetes Helm chart for one-command deployment**

## Month 13 Project B: "CostGuard — LLM Cost Optimizer"
**What:** A proxy that sits between your app and any LLM API, automatically routing queries to the cheapest model that can handle the complexity, with semantic caching, prompt compression, and cost tracking.
**Novelty:** Automatic cost optimization. "This query is simple → route to Llama-8B ($0.0001). This needs reasoning → route to GPT-4 ($0.01)." Saves 60-80% on API costs.
**Skills used:** Production serving (Month 13), model routing (Month 13), agents (Month 5), RAG (Month 5), quantization (Month 4), all prior
**Deliverables:**
- [ ] Proxy server: OpenAI-compatible input, routes to cheapest capable model
- [ ] Difficulty classifier: routes easy/medium/hard queries to appropriate model
- [ ] Semantic cache: reuse answers for similar queries (embedding similarity)
- [ ] Prompt compression: shorten prompts without losing meaning
- [ ] Cost dashboard: $/day, $/query, savings vs single-model baseline
- [ ] **Publish to GitHub**

---

## Month 14 Project A: "SynthData — GPU-Accelerated Synthetic Data Factory"
**What:** A pipeline that generates high-quality synthetic training data for any domain, using LLMs + NVIDIA tools.
**Novelty:** **Quality-aware generation loop** — generates data, scores quality with a trained classifier, filters bad examples, and iteratively improves. Uses NeMo Curator for GPU-accelerated deduplication and filtering. Not just "call GPT-4 and save outputs."
**Deliverables:**
- [ ] Instruction data generation (Self-Instruct + Evol-Instruct)
- [ ] Preference pair generation for DPO training
- [ ] Quality scoring: train classifier to predict data quality
- [ ] GPU-accelerated: NeMo Curator for dedup, PII filtering, quality filtering
- [ ] Domain adaptation: generate domain-specific data from seed examples
- [ ] End-to-end: generate data → filter → train model → evaluate improvement
- [ ] **Publish with comparison: model trained on synthetic vs human data**

## Month 14 Project B: "DataScope — Training Data Quality Analyzer"
**What:** A tool that analyzes any text dataset and generates a quality report: deduplication rate, language distribution, toxicity score, domain balance, contamination check (is benchmark data leaked?), and recommends optimal data mix.
**Novelty:** "Your dataset is 40% duplicate, 12% low-quality, and contains 3% MMLU test questions (contaminated!)" — a one-command data quality audit with actionable fixes.
**Skills used:** Synthetic data (Month 14), NeMo Curator (Month 14), embeddings (Month 5), GPU-accelerated processing (Month 1+), all prior
**Deliverables:**
- [ ] Input: any text dataset (jsonl, parquet, HuggingFace dataset)
- [ ] Dedup analysis: exact + near-duplicate detection (MinHash on GPU)
- [ ] Quality scoring: perplexity-based + classifier-based quality filters
- [ ] Contamination check: compare against major benchmarks (MMLU, GSM8K, HumanEval)
- [ ] HTML quality report with visualizations
- [ ] **Publish to GitHub**

---

## Month 15 Project A: "AgentX — Enterprise AI Agent with Full NVIDIA Stack"
**What:** The definitive enterprise AI agent that uses every relevant NVIDIA tool.
**Novelty:** **Agent capability benchmark** — includes a test suite that measures your agent's capabilities across 10 dimensions (reasoning, tool use, retrieval, safety, multi-step, etc.) and generates a radar chart. Build the agent AND the evaluation.
**Deliverables:**
- [ ] NIM for LLM inference (multiple models: fast + powerful)
- [ ] NeMo Retriever for RAG (embedding + reranking)
- [ ] NeMo Guardrails for safety (Colang 2.0 rules)
- [ ] NVIDIA AIQ / Agent Intelligence Toolkit patterns for agent orchestration
- [ ] Riva for voice interface (ASR + TTS)
- [ ] Tools: code execution, web search, database query, file operations
- [ ] Multi-agent: specialized sub-agents for different tasks
- [ ] Capability benchmark: 50-question test suite across 10 dimensions
- [ ] Radar chart visualization of agent capabilities
- [ ] **Publish with live demo + benchmark results**

## Month 15 Project B: "AgentEval — Automated Agent Testing Framework"
**What:** A framework that automatically generates test scenarios for AI agents, runs them, grades performance across 10 dimensions, and identifies failure modes.
**Novelty:** Agents are hard to test because tasks are open-ended. Your tool generates test cases from templates ("book a flight" → 50 variations) and grades each execution automatically.
**Skills used:** Enterprise agents (Month 15), full NVIDIA stack (Month 15), evaluation (Month 10), all 14 months of prior skills
**Deliverables:**
- [ ] Test case generator: templates → diverse scenarios (easy/medium/hard)
- [ ] Auto-grader: checks task completion, tool use correctness, safety
- [ ] 10-dimension scoring: reasoning, retrieval, safety, efficiency, cost, etc.
- [ ] Failure mode analysis: "agent fails 80% of multi-step math tasks"
- [ ] Regression testing: compare agent v1 vs v2 across all test cases
- [ ] **Publish to GitHub**

---

## Month 16 Project A: "OpenContrib — Major Open-Source Contribution"
**What:** Make a meaningful contribution to a major AI open-source project.
**Novelty:** This IS the novelty — you're contributing to real tools used by thousands of people.
**Deliverables:**
- [ ] Identify 2-3 projects (vLLM, TensorRT-LLM, NeMo, HuggingFace)
- [ ] Find issues labeled "good first issue" or performance improvements
- [ ] Submit at least 2 merged PRs
- [ ] Write blog post about your contribution (what you learned, how the codebase works)
- [ ] **Published PRs + blog post**

---

## Month 17-18 Project: "Magnum Opus — Your Signature Project"
**What:** Your final capstone — a project that represents the BEST of everything you've learned. This is the project people remember you by.
**Choose ONE (or combine):**

**Option A: "LLM-from-Scratch-to-Production"**
- [ ] Train a domain-specific LLM (1-3B params)
- [ ] Full pipeline: data curation → pre-training → SFT → DPO → quantization → deployment
- [ ] Custom CUDA kernels for 3+ operations
- [ ] Deployed on NIM with Guardrails, RAG, and voice interface
- [ ] Comprehensive evaluation + benchmark

**Option B: "AI Agent OS"**
- [ ] Operating system for AI agents: define agents in YAML, auto-deploy with tools
- [ ] Plugin architecture: anyone can add new tools
- [ ] Built-in evaluation, monitoring, cost tracking
- [ ] Multi-model support: route to best model per task
- [ ] Self-improving: agents learn from their failures

**Option C: "GPU Inference Engine"**
- [ ] Custom inference engine rivaling vLLM for a specific model family
- [ ] Hand-optimized CUDA/Triton kernels for attention, GEMM, normalization
- [ ] PagedAttention or your own KV-cache management
- [ ] Speculative decoding, continuous batching
- [ ] Benchmark showing competitive with vLLM/TensorRT-LLM on specific workloads

**Deliverables for any option:**
- [ ] Complete GitHub repo with excellent documentation
- [ ] Technical blog post (2000+ words)
- [ ] Demo video (5-10 minutes)
- [ ] Performance benchmarks and analysis
- [ ] **This is your portfolio centerpiece**

---

### Monthly Project Tracker (2 projects per month)

| Month | Project A | ✓ | Project B | ✓ |
|-------|-----------|---|-----------|---|
| 1 | GPU Matrix Math Engine | ⬜ | LLM Inference Cost Simulator | ⬜ |
| 2 | nanoLLM — GPT from Scratch | ⬜ | TransformerScope — LLM Behavior Debugger | ⬜ |
| 3 | LLM Surgery — Fine-Tuning Toolkit | ⬜ | TokenScope — Tokenizer Analyzer | ⬜ |
| 4 | QuantBench — Quantization Analyzer | ⬜ | LLM-Speedometer — Inference Profiler | ⬜ |
| 5 | DeepRAG — Production RAG | ⬜ | RAGTrace — Retrieval Debugger | ⬜ |
| 6 | AgentForge — Reasoning Agent Platform ⭐ | ⬜ | EvalArena — Model/Cost/Safety Battleground | ⬜ |
| 7 | IncidentLens — Multimodal Debugging Assistant | ⬜ | GPU-Accelerated PDF Intelligence | ⬜ |
| 8 | MoE-Lab — Expert Specialization | ⬜ | ArchSearch — Architecture Comparator | ⬜ |
| 9 | KernelSmith — Triton Kernel Library | ⬜ | FusionLab — Triton Fusion Playground | ⬜ |
| 10 | TrainScale — Distributed Training | ⬜ | TrainWatch — Training Debugger | ⬜ |
| 11 | PaperBot — Paper Implementer | ⬜ | Paper2Benchmark — Reproducibility Tool | ⬜ |
| 12 | ReasonEngine — Test-Time Compute | ⬜ | ThinkTrace — Reasoning Visualizer | ⬜ |
| 13 | NVServe — Production Serving | ⬜ | CostGuard — LLM Cost Optimizer | ⬜ |
| 14 | SynthData — Synthetic Data Factory | ⬜ | DataScope — Data Quality Analyzer | ⬜ |
| 15 | AgentX — Enterprise Agent (NVIDIA) | ⬜ | AgentEval — Agent Testing Framework | ⬜ |
| 16 | OpenContrib — Open Source | ⬜ | *(contribute to 2nd OSS project)* | ⬜ |
| 17-18 | Magnum Opus — Signature Project | ⬜ | *(your masterpiece)* | ⬜ |

---

# ═══════════════════════════════════════════════════════════
# MONTH-BY-MONTH FOCUS & CAPABILITIES
# "After this month, I can..."
# ═══════════════════════════════════════════════════════════

---

### Month 1 — GPU Architecture + CUDA Fundamentals + Neural Network Basics
**Focus:** CPU vs GPU architecture, CUDA programming model (kernels, threads, warps, blocks, grids), GPU memory hierarchy (global, shared, constant), memory coalescing, warp execution & divergence, parallel reduction, CUDA streams & graphs, neural networks from scratch, attention mechanism, tokenization (BPE)
**Project:** GPU Matrix Math Engine

**After this month, you can:**
- [ ] Write and launch CUDA kernels — vector ops, reductions, tiled matrix multiplication
- [ ] Explain GPU architecture from memory: SMs, warps, shared memory, L1/L2, HBM
- [ ] Optimize memory access: coalescing, shared memory tiling, bank conflict avoidance
- [ ] Use CUDA streams for overlapping computation and data transfer
- [ ] Build an MLP from scratch in NumPy (forward + backward pass, no framework)
- [ ] Implement the attention mechanism and BPE tokenizer from scratch
- [ ] Benchmark CPU vs GPU and explain roofline model analysis
- [ ] Profile basic CUDA programs and identify compute-bound vs memory-bound kernels

---

### Month 2 — Transformers + GPT from Scratch + GPU Libraries + Profiling
**Focus:** Full Transformer architecture (encoder + decoder), multi-head attention, positional encodings (sinusoidal, RoPE), GPT-2 implementation, cuBLAS, cuDNN, Tensor Cores (WMMA), Nsight Systems & Compute profiling, LLaMA architecture (RMSNorm, SwiGLU, GQA), basic interpretability (attention maps, residual stream, logit lens), scaling laws, mixed precision training (AMP, BF16), efficient data loading, CNNs
**Project:** nanoLLM — Your Own GPT from Scratch

**After this month, you can:**
- [ ] Implement a complete Transformer (encoder + decoder) from scratch — no `nn.MultiheadAttention`
- [ ] Build and train GPT-2 (124M params) with KV-cache inference and all sampling strategies
- [ ] Implement LLaMA-style model: RMSNorm, RoPE, SwiGLU, Grouped Query Attention
- [ ] Build basic interpretability tools: attention heatmaps, residual-stream probes, logit lens
- [ ] Use cuBLAS for high-performance GEMM and Tensor Cores via WMMA API
- [ ] Profile GPU workloads: Nsight Systems (timeline) and Nsight Compute (kernel-level SOL analysis)
- [ ] Train with mixed precision (FP16/BF16 AMP) and understand loss scaling
- [ ] Use HuggingFace ecosystem: AutoModel, Trainer, tokenizers, datasets
- [ ] Build efficient data pipelines: pinned memory, num_workers, prefetch, gradient accumulation

---

### Month 3 — Distributed Training + Fine-Tuning + Flash Attention + Triton
**Focus:** NCCL collectives (AllReduce, AllGather), DDP, tensor/pipeline/data parallelism, ZeRO, pre-training pipeline, SFT, DPO, RLHF overview, custom CUDA extensions for PyTorch, Flash Attention (tiling + online softmax), Triton GPU programming, PTX/SASS analysis, tokenizer deep dive, multi-modal overview, GPU cluster architecture
**Project:** LLM Surgery — Fine-Tuning & Alignment Toolkit

**After this month, you can:**
- [ ] Set up distributed training with DDP and convert any training loop to multi-GPU
- [ ] Understand all parallelism strategies: data, tensor, pipeline, ZeRO stages
- [ ] Build a complete pre-training pipeline: data curation → tokenization → training → evaluation
- [ ] Fine-tune LLMs with SFT on instruction datasets (with loss masking on prompts)
- [ ] Align models with DPO using preference data
- [ ] Write custom CUDA kernels callable from PyTorch via `torch.utils.cpp_extension`
- [ ] Write GPU kernels in Triton: fused softmax, fused LayerNorm, simplified Flash Attention
- [ ] Explain Flash Attention's algorithm, online softmax trick, and IO complexity
- [ ] Inspect PTX/SASS assembly and identify optimization opportunities
- [ ] Build a chat interface with proper chat templates (ChatML, LLaMA format)

---

### Month 4 — Parameter-Efficient Fine-Tuning + Quantization
**Focus:** LoRA from scratch, QLoRA (NF4 + double quantization), prefix tuning, adapters, IA³, adapter merging (TIES, DARE), quantization theory (INT4/INT8/FP8, symmetric/asymmetric, per-tensor/per-channel/per-group), GPTQ, AWQ, GGUF/llama.cpp, FP8 Transformer Engine, CUDA kernels for quantized inference
**Project:** QuantBench — The Quantization Analyzer

**After this month, you can:**
- [ ] Implement LoRA from scratch and apply it to any model's attention layers
- [ ] Fine-tune 7B+ parameter models with QLoRA on consumer GPUs (4-bit base + FP16 LoRA)
- [ ] Merge, swap, and stack multiple LoRA adapters for different tasks
- [ ] Quantize any HuggingFace model with GPTQ (4/8-bit), AWQ (4-bit), and GGUF (Q4_K_M through Q8_0)
- [ ] Use FP8 training with Transformer Engine on Hopper GPUs
- [ ] Write CUDA kernels for INT8 GEMM with dequantize-on-the-fly
- [ ] Produce comprehensive quality-vs-speed-vs-memory reports for any quantization method
- [ ] Build CLI tools that automate model compression workflows

---

### Month 5 — Inference Serving + RAG + Agents + NVIDIA Ecosystem
**Focus:** Inference analysis (prefill vs decode, TTFT, TPOT, throughput), TensorRT-LLM, vLLM, PagedAttention, speculative decoding, continuous batching, Triton Inference Server, NVIDIA Dynamo, vector search (FAISS GPU, cuVS), embeddings, RAG architecture, chunking, hybrid retrieval, re-ranking, Graph RAG, agentic RAG, Self-RAG, RAGAS evaluation, function calling, ReAct, agent memory, multi-agent systems, NIM, NeMo, Guardrails, Riva, ACE, AI Blueprints, NGC, build.nvidia.com
**Project:** DeepRAG — Production RAG with NVIDIA Stack

**After this month, you can:**
- [ ] Deploy LLMs for production on vLLM, TensorRT-LLM, and Triton Inference Server
- [ ] Implement speculative decoding for 2-3x faster autoregressive generation
- [ ] Build end-to-end RAG: document ingestion → chunking → embedding → retrieval → generation
- [ ] Use GPU-accelerated vector search (FAISS GPU, cuVS/CAGRA)
- [ ] Implement hybrid retrieval (dense + BM25) with cross-encoder re-ranking
- [ ] Build agentic RAG (model decides when to retrieve) and Graph RAG (knowledge graph traversal)
- [ ] Evaluate RAG quality with RAGAS: faithfulness, relevance, correctness
- [ ] Build AI agents with tools, memory, ReAct loops, and basic multi-agent coordination
- [ ] Use NVIDIA NIM, NeMo Retriever, NeMo Guardrails, Riva, AI Blueprints, and NGC at a practical level
- [ ] Serve models via REST API with benchmarked throughput and latency

---

### Month 6 — Distillation + RLHF + Reasoning + Long Context ⭐ (6-MONTH MILESTONE)
**Focus:** Knowledge distillation, model merging, synthetic instruction data, RL foundations, PPO/GRPO, RLHF pipeline, chain-of-thought, Tree of Thought, self-consistency, process reward models, long-context evaluation, RoPE scaling, KV-cache compression, Mamba/SSM overview
**Project:** AgentForge — Reasoning Agent Platform

**After this month, you can:**
- [ ] Distill a larger model into a smaller one and measure quality/speed tradeoffs
- [ ] Merge fine-tuned models using SLERP, TIES, DARE, or model soups
- [ ] Explain RLHF from SFT → reward model → PPO/GRPO with KL control
- [ ] Implement reasoning methods: CoT, self-consistency, Tree of Thought, verifier scoring
- [ ] Build agents that combine tools, memory, retrieval, and explicit reasoning strategies
- [ ] Evaluate long-context behavior with NIAH/passkey/document-QA style tests
- [ ] Explain RoPE scaling, KV-cache compression, and when Mamba/SSM-style models matter
- [ ] **You now have 12 monthly capstones plus weekly builds, and can build real AI applications end-to-end**

---

### Month 7 — Multi-Modal AI + Diffusion Overview
**Focus:** CLIP (contrastive image-text), Vision Transformers (ViT), LLaVA (visual instruction tuning), document understanding, video understanding, audio (Whisper, Riva), DALI for image loading, diffusion models overview (DDPM, latent diffusion, Stable Diffusion), DiT
**Project:** IncidentLens — Multimodal AI Debugging Assistant

**After this month, you can:**
- [ ] Build AI that understands images, documents, audio, and video in a single conversation
- [ ] Fine-tune CLIP on custom datasets and build vision-language models (CLIP + projection + LLM)
- [ ] Extract information from PDFs with tables, charts, and images
- [ ] Understand how diffusion models work (forward/reverse process, latent space, guidance)
- [ ] Build multi-modal chatbots with automatic modality routing

---

### Month 8 — Mixture of Experts + Advanced Architectures
**Focus:** MoE Transformer implementation, expert parallelism, load balancing loss, router architectures (hash, learned, expert-choice), DeepSeek-V3 architecture (Multi-head Latent Attention, DeepSeekMoE), sparse vs dense MoE, auxiliary losses, expert communication patterns
**Project:** MoE-Lab — Mixture of Experts Playground

**After this month, you can:**
- [ ] Implement MoE Transformers with top-k routing and load balancing
- [ ] Visualize which experts specialize in which types of input (code, math, language)
- [ ] Implement Multi-head Latent Attention (MLA) from DeepSeek-V3
- [ ] Prune experts and measure quality impact
- [ ] Compare MoE vs dense models at equal active parameter counts
- [ ] Profile and optimize expert routing communication on GPU
- [ ] Understand how frontier models (Mixtral, DeepSeek-V3, GPT-4) work internally

---

### Month 9 — Advanced Kernel Engineering + Compiler Optimization
**Focus:** Triton compiler internals, torch.compile (TorchDynamo graph capture + TorchInductor code gen), operator fusion patterns, kernel autotuning, CUDA cooperative groups, warp specialization (Hopper), fused training kernels, gradient compression
**Project:** KernelSmith — Custom Triton Kernel Library for LLMs

**After this month, you can:**
- [ ] Write production-quality Triton kernels for all common LLM operations
- [ ] Build: fused attention, fused RMSNorm, fused SwiGLU, fused RoPE, fused cross-entropy, fused AdamW
- [ ] Understand torch.compile internals and how TorchInductor generates GPU code
- [ ] Automate kernel benchmarking with CI (every kernel tested against PyTorch native)
- [ ] Build drop-in `nn.Module` replacements that are measurably faster than stock PyTorch
- [ ] Implement gradient compression for communication optimization
- [ ] **Your kernels are pip-installable and usable by anyone**

---

### Month 10 — Distributed Training at Scale + Production Systems
**Focus:** Megatron-LM (3D parallelism: DP+TP+PP, sequence parallelism, context parallelism), DeepSpeed ZeRO-3, FSDP2, distributed checkpointing, training failure diagnosis, evaluation frameworks (lm-eval-harness, HELM, MMLU, HumanEval, MT-Bench), production MLOps, auto-scaling, A/B testing, NVIDIA AI Enterprise
**Project:** TrainScale — Distributed LLM Training Framework

**After this month, you can:**
- [ ] Configure Megatron-LM for large-scale multi-node training
- [ ] Train models larger than GPU memory using DeepSpeed ZeRO-3 and FSDP2
- [ ] Implement fault-tolerant distributed checkpointing with resume
- [ ] Build training cost estimators: predict time, cost, optimal DP/TP/PP config before training
- [ ] Evaluate models comprehensively (MMLU, HumanEval, GSM8K, MT-Bench, and more)
- [ ] Build production serving with auto-scaling, monitoring, and alerting
- [ ] Implement semantic caching and multi-model routing (easy → small, hard → large)
- [ ] Diagnose common training failures: loss spikes, gradient explosions, OOM

---

### Month 11 — Research Paper Implementation + Data Engineering
**Focus:** Paper reading technique (figures → method → experiments), implementing algorithms from scratch, reproducing results, synthetic data pipelines (Self-Instruct, Evol-Instruct), NeMo Curator (GPU-accelerated curation), constrained decoding (outlines, GBNF), function calling training, code generation, Text-to-SQL
**Project:** PaperBot — AI Research Paper Implementer

**After this month, you can:**
- [ ] Read any AI paper and extract the key algorithm into working code
- [ ] Implement novel architectures and training techniques from publications
- [ ] Reproduce experimental results and identify when papers are misleading
- [ ] Build GPU-accelerated data curation pipelines: scrape → clean → dedupe → filter → tokenize
- [ ] Generate high-quality synthetic instruction and preference data
- [ ] Implement grammar-constrained generation (guaranteed valid JSON output)
- [ ] Train function-calling and code generation models
- [ ] **You can now turn any paper into a working prototype**

---

### Month 12 — Test-Time Compute + Reasoning + RL for LLMs
**Focus:** RL foundations (MDP, policy gradient, REINFORCE, Actor-Critic, advantage, GAE), PPO, GRPO (DeepSeek), full RLHF pipeline (SFT → reward model → PPO + KL penalty), RLAIF, process reward models, chain-of-thought (zero-shot, few-shot), Tree of Thought, self-consistency, STaR, test-time compute scaling, MCTS for reasoning, adaptive compute allocation, DeepSeek-R1
**Project:** ReasonEngine — Test-Time Compute for Better Answers

**After this month, you can:**
- [ ] Implement PPO and GRPO from scratch (RL fundamentals + LLM application)
- [ ] Build full RLHF pipelines: SFT → reward model → PPO with KL penalty
- [ ] Train process reward models that score individual reasoning steps
- [ ] Build MCTS-based reasoning systems for hard math/logic problems
- [ ] Implement adaptive compute: easy question = 1 pass, hard question = 64 samples + verification
- [ ] Show accuracy vs compute curves (more inference compute = better answers)
- [ ] Understand how reasoning models (o1, DeepSeek-R1) achieve step-by-step thinking
- [ ] **You can make any LLM dramatically smarter on hard problems**

---

### Month 13 — Production LLM Serving + Advanced Alignment + Enterprise Agents
**Focus:** NIM auto-scaling, multi-model serving, semantic caching, A/B testing with statistical significance, Guardrails advanced (Colang 2.0), NVIDIA AIQ enterprise patterns, MCP (Model Context Protocol), agent fine-tuning, tool-use training, advanced alignment (SPIN, ORPO, SimPO, iterative DPO), reward hacking prevention
**Project:** NVServe — Production LLM Serving Platform

**After this month, you can:**
- [ ] Build production-grade serving with auto-scaling on Kubernetes
- [ ] Route requests across multiple models based on complexity, cost, or latency targets
- [ ] A/B test model versions with statistical significance
- [ ] Implement MCP servers and clients for standardized tool interfaces
- [ ] Compare 5+ alignment methods (DPO, PPO, GRPO, ORPO, SimPO) on the same base model
- [ ] Detect and prevent reward hacking in alignment training
- [ ] Build self-improving model pipelines (generate → evaluate → retrain)
- [ ] **You can deploy and manage LLMs at enterprise scale**

---

### Month 14 — Synthetic Data Engineering + Advanced RAG
**Focus:** Quality-aware synthetic data generation loops, data contamination detection, NeMo Curator at scale (dedup, PII, quality scoring), advanced retrieval (ColBERT, SPLADE, RAPTOR), multi-hop RAG, structured data RAG (SQL generation, table retrieval), RAG + tool use, domain adaptation
**Project:** SynthData — GPU-Accelerated Synthetic Data Factory

**After this month, you can:**
- [ ] Build quality-aware data generation loops: generate → score → filter → iterate
- [ ] Use NeMo Curator for GPU-accelerated deduplication, PII filtering, quality scoring
- [ ] Detect data contamination in training sets
- [ ] Build state-of-the-art RAG with ColBERT late interaction and RAPTOR hierarchical retrieval
- [ ] Handle multi-hop questions requiring multiple retrieval steps
- [ ] Demonstrate measurable model improvement from synthetic data vs human-curated data
- [ ] Build domain-specific data pipelines from seed examples to full training set
- [ ] **Your synthetic data is good enough to train production models**

---

### Month 15 — Enterprise AI Agents (Full NVIDIA Stack)
**Focus:** NIM (LLM + embedding + reranking), NeMo Guardrails (Colang 2.0), NVIDIA AIQ orchestration, Riva voice I/O, multi-agent coordination, agent capability benchmarking across 10 dimensions, advanced evaluation
**Project:** AgentX — Enterprise AI Agent with Full NVIDIA Stack

**After this month, you can:**
- [ ] Build enterprise-grade AI agents using every relevant NVIDIA tool simultaneously
- [ ] Design and run 50-question agent capability benchmarks across 10 dimensions
- [ ] Build voice-enabled agents: Riva ASR → LLM NIM → Riva TTS
- [ ] Orchestrate specialized sub-agents for different task types
- [ ] Generate radar chart visualizations of agent capabilities vs baselines
- [ ] Deploy production agents with monitoring, safety rails, and per-task cost tracking
- [ ] **You can architect and deliver complete AI agent solutions for enterprise use**

---

### Month 16 — Open-Source Contributions
**Focus:** Contributing to major AI projects (vLLM, TensorRT-LLM, NeMo, HuggingFace transformers/trl/PEFT), navigating large codebases, finding and fixing performance issues, technical writing
**Project:** OpenContrib — Major Open-Source Contribution

**After this month, you can:**
- [ ] Navigate and understand codebases with 100K+ lines (vLLM, Megatron, NeMo)
- [ ] Identify performance bottlenecks in production AI tools
- [ ] Submit meaningful PRs that get merged into projects used by thousands of developers
- [ ] Write technical blog posts that explain complex systems clearly
- [ ] **You are a recognized contributor to the AI open-source ecosystem**

---

### Months 17-18 — Magnum Opus: Your Signature Project
**Focus:** Combining all 16 months of skills into one definitive project. Technical writing, demo creation, portfolio polish. Choose: LLM-from-Scratch-to-Production, AI Agent OS, or GPU Inference Engine.
**Project:** Magnum Opus — The Project People Remember You By

**After these months, you can:**
- [ ] Design and execute a large-scale AI project from zero to production
- [ ] Train domain-specific LLMs (1-3B params): data curation → pre-training → SFT → DPO → quantize → deploy
- [ ] OR build a custom inference engine competitive with vLLM/TensorRT-LLM
- [ ] OR build an agent OS with plugin architecture, self-improvement, and multi-model routing
- [ ] Write 2000+ word technical deep-dives with architecture diagrams
- [ ] Present your work with polished demo videos and live hosted demos
- [ ] **You have a portfolio demonstrating mastery of CUDA + LLMs + production systems + NVIDIA stack**
- [ ] **You can lead AI infrastructure projects at NVIDIA or any top AI company**

---

# ═══════════════════════════════════════════════
# PHASE 1: FOUNDATIONS (Weeks 1-12, Months 1-3)
# GPU Architecture, CUDA, Neural Nets, Transformers
# ═══════════════════════════════════════════════

---

## Week 1: CPU vs GPU & Math Foundations ✅ COMPLETED

### Day 1 — CPU vs GPU Architecture ✅
- [x] Von Neumann architecture: fetch-decode-execute cycle
- [x] CPU: few powerful cores, large caches, branch prediction, out-of-order execution
- [x] GPU: thousands of simple cores, SIMT (Single Instruction Multiple Thread)
- [x] Throughput vs latency oriented design — why GPUs win at parallel workloads
- [x] ALU ratio: GPU ~80% ALUs vs CPU ~5%
- [x] **Code:** Write CPU matrix multiply, measure FLOPS

### Day 2 — NVIDIA GPU Physical Architecture ✅
- [x] Streaming Multiprocessor (SM) internals: CUDA cores, Tensor Cores, SFUs, Load/Store units
- [x] Warp schedulers: how 2-4 schedulers issue instructions per cycle
- [x] Register file: 65536 32-bit registers per SM
- [x] L1 cache / Shared memory (configurable split)
- [x] L2 cache, memory controllers, HBM2/HBM3 interface
- [x] NVIDIA GPU lineage: Tesla → Fermi → Kepler → Maxwell → Pascal → Volta → Turing → Ampere → Hopper → Blackwell
- [x] **Explore:** NGC Catalog (catalog.ngc.nvidia.com) — browse containers, models, resources
- [x] **Code:** `cudaGetDeviceProperties()` — query and understand every field

### Day 3 — Math Foundations for Deep Learning ✅
- [x] Linear algebra: vectors, matrices, matrix multiplication, transpose, inverse
- [x] Eigenvalues/eigenvectors intuition
- [x] Probability: Bayes theorem, Gaussian, Bernoulli, Categorical distributions
- [x] Information theory: entropy, cross-entropy, KL divergence
- [x] Calculus: partial derivatives, chain rule, Jacobian, gradient
- [x] **Code:** Implement matrix multiply, softmax, cross-entropy loss in NumPy

### Day 4 — CUDA Programming Model ✅
- [x] Host (CPU) vs Device (GPU) code
- [x] Kernel launch: `kernel<<<grid, block>>>(args)`
- [x] Thread hierarchy: Thread → Warp (32) → Block → Grid
- [x] `threadIdx`, `blockIdx`, `blockDim`, `gridDim`
- [x] 1D, 2D, 3D thread configurations
- [x] nvcc compilation: PTX → SASS
- [x] **Code:** "Hello World" kernel + vector addition with error checking

### Day 5 — Neural Networks from First Principles ✅
- [x] Perceptron: weighted sum + activation = decision boundary
- [x] Multi-layer perceptron (MLP), universal approximation theorem
- [x] Activation functions: Sigmoid, tanh, ReLU, GELU, SiLU/Swish — what each does and WHY
- [x] **Code:** Implement 2-layer MLP from scratch in NumPy (forward + backward manually)

### 🔨 Saturday Project
- [ ] **CUDA Vector Math Library** — add, subtract, multiply, dot product on GPU
- [ ] Benchmark CPU vs naive GPU vs optimized GPU
- [ ] Generate performance plots with matplotlib
- [ ] Write 1-page analysis of memory bandwidth utilization

### 📄 Sunday Paper
- [ ] Read: "An Even Easier Introduction to CUDA" (NVIDIA Blog, Mark Harris)
- [ ] Read: Michael Nielsen "Neural Networks and Deep Learning" Ch 1-2 (online book)

---

## Week 2 : CUDA Memory & Backpropagation

### Day 1 — CUDA Memory Model: Global Memory
- [ ] Global memory: off-chip HBM, high bandwidth, high latency (400-600 cycles)
- [ ] Memory transactions: 32-byte and 128-byte cache lines
- [ ] Coalesced vs uncoalesced access patterns
- [ ] Memory alignment requirements
- [ ] `cudaMalloc`, `cudaFree`, `cudaMemcpy` (H2D, D2H, D2D)
- [ ] **Code:** Vector addition with bandwidth measurement

### Day 2 — CUDA Memory Model: Shared & Constant Memory
- [ ] Shared memory: on-chip SRAM, ~100x faster than global
- [ ] 32 memory banks, bank conflicts, padding trick `[32][33]`
- [ ] Static `__shared__` vs dynamic `extern __shared__`
- [ ] Constant memory: 64KB, broadcast to entire warp
- [ ] Local memory: per-thread, spills to global (slow)
- [ ] **Code:** Shared memory tiled operations, verify no bank conflicts

### Day 3 — Backpropagation & Gradient Descent
- [ ] Forward pass → compute loss → backward pass (chain rule)
- [ ] Computational graphs: how frameworks track operations
- [ ] SGD: `w = w - lr * grad`
- [ ] Momentum, Adam, AdamW — what each improves
- [ ] Learning rate schedules: warmup, cosine decay
- [ ] **Code:** Implement backprop manually for 2-layer MLP, verify with PyTorch autograd

### Day 4 — Memory Coalescing & SoA vs AoS
- [ ] Aligned vs misaligned access patterns
- [ ] Structure of Arrays (SoA) vs Array of Structures (AoS) on GPU
- [ ] Why SoA wins: each field is contiguous → coalesced reads
- [ ] **Code:** AoS vs SoA particle simulation kernel, benchmark the difference

### Day 5 — PyTorch Fundamentals
- [ ] Tensors: creation, indexing, reshaping, broadcasting
- [ ] Device management: `.to('cuda')`, `torch.device`
- [ ] Autograd: `requires_grad`, `.backward()`, `.grad`, `torch.no_grad()`
- [ ] `nn.Module`: defining custom layers, `nn.Parameter`
- [ ] Model saving: `state_dict`, `torch.save`, `torch.load`
- [ ] **Code:** Rewrite NumPy MLP in PyTorch, train on synthetic data, move to GPU

### 🔨 Saturday Project
- [ ] **Tiled Matrix Multiplication in CUDA**
  - [ ] Naive CPU, naive GPU, coalesced GPU, shared-memory tiled GPU
  - [ ] Time with CUDA events and report effective bandwidth/FLOPS
  - [ ] Generate a simple bandwidth/FLOPS plot for your GPU
  - [ ] Refine into a real roofline plot after Sunday's roofline paper
  - [ ] Revisit after Week 6/7 to add cuBLAS and Nsight Compute

### 📄 Sunday Paper
- [ ] Read: "Roofline: An Insightful Visual Performance Model" (Williams et al., 2009)

---

## Week 3 : Warp Execution & Training Neural Networks

### Day 1 — Warp Execution Model
- [ ] Warps: 32 threads executing in lockstep (SIMT)
- [ ] Warp scheduling: SM time-multiplexes warps to hide memory latency
- [ ] Occupancy: active warps / max warps per SM
- [ ] Why high occupancy helps: more warps to switch to
- [ ] **Code:** Experiment with block sizes, measure occupancy impact

### Day 2 — Warp Divergence & Synchronization
- [ ] Branch divergence: both paths execute, inactive threads masked
- [ ] Strategies: sort data, replace branches with arithmetic, uniform control flow
- [ ] `__syncthreads()`: block-level barrier
- [ ] Warp-level: `__syncwarp()`, `__shfl_sync()`, `__ballot_sync()`
- [ ] Shuffle instructions: intra-warp register communication (no shared memory needed)
- [ ] **Code:** Kernel with intentional divergence → profile → optimize

### Day 3 — Loss Functions & Training Dynamics
- [ ] MSE Loss, Cross-Entropy Loss, Binary Cross-Entropy
- [ ] Label smoothing
- [ ] Overfitting vs underfitting, bias-variance tradeoff
- [ ] Regularization: L1, L2, Dropout
- [ ] BatchNorm vs LayerNorm — why Transformers use LayerNorm
- [ ] **Code:** Train MLP on MNIST with early stopping, plot loss curves

### Day 4 — Parallel Reduction
- [ ] Tree reduction: parallel O(log N) time
- [ ] 6 optimization levels: interleaved → sequential → first-add-during-load → unroll-last-warp → complete-unroll → multi-element-per-thread
- [ ] **Code:** Implement all 6 levels, benchmark each, understand why each helps

### Day 5 — Atomic Operations & Embeddings
- [ ] `atomicAdd`, `atomicMax`, `atomicCAS` — why atomics are slow
- [ ] Privatization pattern: local histogram in shared mem → atomic merge
- [ ] Word embeddings: one-hot → dense vectors, Word2Vec intuition
- [ ] `nn.Embedding`: lookup table, gradient computation
- [ ] **Code:** Parallel histogram (naive vs privatized) + train word embeddings on small corpus

### 🔨 Saturday Project
- [ ] **CUDA Reduction Library + MNIST Classifier**
  - [ ] Complete parallel reduction library (sum, min, max, argmax)
  - [ ] MLP on MNIST achieving >97% accuracy
  - [ ] Time the training loop with PyTorch profiler/CUDA events
  - [ ] Revisit after Week 7 to profile end-to-end with Nsight Systems

### 📄 Sunday Paper
- [ ] Read: "Efficient Estimation of Word Representations in Vector Space" (Mikolov, 2013) — Word2Vec

---

## Week 4 : CUDA Streams & Sequence Models

### Day 1 — CUDA Streams & Events
- [ ] Default stream (stream 0): all operations serialized
- [ ] Non-default streams: enable concurrent execution
- [ ] `cudaStreamCreate`, `cudaStreamSynchronize`
- [ ] Overlapping: H2D + kernel + D2H on different streams
- [ ] CUDA events: `cudaEventRecord`, `cudaEventElapsedTime` for precise timing
- [ ] **Code:** Pipelined processing with 3 streams, measure overlap benefits

### Day 2 — CUDA Graphs
- [ ] Problem: kernel launch overhead (5-10 μs) adds up for small kernels
- [ ] CUDA Graphs: capture operation sequence, replay with minimal overhead
- [ ] Stream capture: `cudaStreamBeginCapture` / `cudaStreamEndCapture`
- [ ] When to use: inference pipelines with fixed topology
- [ ] **Code:** Convert multi-kernel pipeline to CUDA Graph, measure overhead reduction

### Day 3 — Recurrent Neural Networks
- [ ] Why feedforward nets fail on sequences → RNNs
- [ ] RNN cell: `h_t = tanh(W_hh * h_{t-1} + W_xh * x_t + b)`
- [ ] Vanishing/exploding gradient problem
- [ ] LSTM: forget gate, input gate, output gate, cell state — gradient highway
- [ ] GRU: simplified with reset and update gates
- [ ] **Code:** Implement LSTM cell from scratch, verify against `nn.LSTMCell`

### Day 4 — Attention Mechanism (Bahdanau)
- [ ] Fixed-size bottleneck problem in seq2seq
- [ ] Query-Key-Value intuition: query asks, keys answer, values provide content
- [ ] Attention weights: softmax over compatibility scores
- [ ] Context vector: weighted sum of encoder hidden states
- [ ] **Code:** Implement Bahdanau attention from scratch in PyTorch

### Day 5 — Tokenization for NLP
- [ ] Character-level vs word-level vs subword tokenization
- [ ] BPE (Byte Pair Encoding): iteratively merge most frequent pairs
- [ ] WordPiece, SentencePiece, tiktoken
- [ ] Vocabulary size tradeoffs
- [ ] **Code:** Implement BPE tokenizer from scratch, compare with HuggingFace tokenizers

### 🔨 Saturday Project
- [ ] **Character-Level Language Model**
  - [ ] Train character-level LSTM on Shakespeare text
  - [ ] Generate text at different temperatures
  - [ ] Track training speed, memory use, and loss curves
  - [ ] Revisit after Week 7 to profile training with Nsight Systems

### 📄 Sunday Paper
- [ ] Read: "Neural Machine Translation by Jointly Learning to Align and Translate" (Bahdanau, 2014)

---

## 🔄 Buffer Week (Month 1 Revision)
- [ ] Revise all CUDA concepts: kernels, memory hierarchy, coalescing, warps, SMs
- [ ] Revise all ML concepts: MLP, backprop, softmax, cross-entropy, activation functions
- [ ] Re-run your best CUDA kernels, try to optimize further
- [ ] **Build Monthly Project A:** GPU Matrix Math Engine
- [ ] **Build Monthly Project B:** LLM Inference Cost Simulator
- [ ] Push both projects to GitHub

---

## Week 5 : The Transformer — Attention Is All You Need

### Day 1 — Scaled Dot-Product Attention
- [ ] Why replace RNNs: parallelization, long-range dependencies
- [ ] Self-attention: every position attends to every other position
- [ ] QKV formulation: `Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V`
- [ ] Why scale by sqrt(d_k): prevent softmax saturation
- [ ] **Code:** Implement scaled dot-product attention from scratch

### Day 2 — Multi-Head Attention
- [ ] Multiple heads: attend to different aspects of input
- [ ] Splitting dimensions: d_model → h heads of d_k each
- [ ] Concatenation and linear projection
- [ ] **Code:** Implement MultiHeadAttention class from scratch (no `nn.MultiheadAttention`)

### Day 3 — Positional Encoding & Feed-Forward Networks
- [ ] Self-attention is permutation-invariant → need position info
- [ ] Sinusoidal PE: `PE(pos, 2i) = sin(pos/10000^(2i/d_model))`
- [ ] Learned positional embeddings (GPT, BERT)
- [ ] Rotary Position Embedding (RoPE) — used in modern LLMs
- [ ] Position-wise FFN: two linear layers with activation
- [ ] **Code:** Implement sinusoidal PE and RoPE

### Day 4 — Full Transformer Block
- [ ] Encoder block: MHA → Add & LayerNorm → FFN → Add & LayerNorm
- [ ] Residual connections: critical for gradient flow
- [ ] Pre-norm vs Post-norm: modern LLMs use pre-norm
- [ ] RMSNorm: simplified LayerNorm used in LLaMA
- [ ] **Code:** Implement complete Transformer encoder block

### Day 5 — Transformer Decoder & Full Architecture
- [ ] Masked self-attention: causal mask prevents future token attention
- [ ] Cross-attention: decoder attends to encoder output
- [ ] Autoregressive generation: one token at a time
- [ ] Teacher forcing during training
- [ ] **Code:** Implement full Transformer (encoder + decoder)

### 🔨 Saturday Project
- [ ] **Build GPT from Scratch (Part 1)**
  - [ ] Decoder-only Transformer
  - [ ] Causal masked self-attention
  - [ ] Train on tiny Shakespeare dataset
  - [ ] Generate text samples

### 📄 Sunday Paper
- [ ] Read: **"Attention Is All You Need"** (Vaswani et al., 2017) — THE paper
- [ ] Read: "The Illustrated Transformer" (Jay Alammar blog)

---

## Week 6 : cuBLAS, cuDNN, Tensor Cores

### Day 1 — cuBLAS
- [ ] BLAS levels: L1 (vec-vec), L2 (mat-vec), L3 (mat-mat)
- [ ] `cublasSgemm`, `cublasGemmEx` for mixed precision
- [ ] Column-major vs row-major (cuBLAS is column-major!)
- [ ] cuBLASLt: lightweight GEMM with epilogue fusion
- [ ] **Code:** Compare your tiled GEMM vs cuBLAS

### Day 2 — cuDNN & Tensor Cores
- [ ] cuDNN: convolution, normalization, activation, attention primitives
- [ ] cuDNN algorithm autotuning
- [ ] Tensor Cores: `D = A * B + C` for small matrices in one cycle
- [ ] Supported precisions: FP16, BF16, TF32, FP8, INT8
- [ ] WMMA API: `nvcuda::wmma` fragment types
- [ ] **Code:** Implement GEMM using WMMA, compare with FP32 CUDA core GEMM

### Day 3 — GPT Architecture Deep Dive
- [ ] Decoder-only Transformer: why GPT drops the encoder
- [ ] Causal language modeling: predict next token
- [ ] GPT-1 → GPT-2 → GPT-3: what changed (scale, data, few-shot)
- [ ] Architecture: token emb + pos emb → N blocks → LM head → softmax
- [ ] **Code:** Implement clean GPT-2 small (124M) architecture

### Day 4 — GPT Training & Inference
- [ ] Training: minimize cross-entropy of next-token prediction
- [ ] Loss computation: shift logits and labels by 1
- [ ] Gradient clipping, LR warmup + cosine decay
- [ ] KV-cache: store past key-value pairs to avoid recomputation
- [ ] Sampling: greedy, temperature, top-k, top-p, repetition penalty
- [ ] **Code:** Implement training loop + all sampling strategies

### Day 5 — BERT & HuggingFace Ecosystem
- [ ] BERT: Masked Language Modeling, bidirectional, [CLS]/[SEP] tokens
- [ ] BERT vs GPT: understanding vs generation
- [ ] HuggingFace: `AutoModel`, `AutoTokenizer`, `pipeline()`, `Trainer`
- [ ] `datasets` library
- [ ] **Explore:** build.nvidia.com — try NVIDIA-hosted LLMs via API (free tier)
- [ ] **Code:** Fine-tune BERT for sentiment analysis with HF Trainer

### 🔨 Saturday Project
- [ ] **GPT-2 Small from Scratch**
  - [ ] Full 124M parameter model implementation
  - [ ] Clean FP32 training loop with checkpoint save/resume
  - [ ] KV-cache inference
  - [ ] Benchmark tokens/second
  - [ ] Revisit after Week 8 to add AMP and gradient checkpointing

### 📄 Sunday Papers
- [ ] "Language Models are Unsupervised Multitask Learners" (GPT-2, Radford 2019)
- [ ] "BERT: Pre-training of Deep Bidirectional Transformers" (Devlin, 2018)

---

## Week 7 : Profiling & Modern LLM Architectures

### Day 1 — Nsight Systems
- [ ] Timeline view: CPU + GPU activity, memory transfers, kernel launches
- [ ] Identifying bottlenecks: compute-bound vs memory-bound
- [ ] CPU-GPU synchronization issues
- [ ] **Code:** Profile GPT training with Nsight Systems, find top 5 bottlenecks

### Day 2 — Nsight Compute
- [ ] Kernel-level profiling: instruction throughput, memory throughput
- [ ] Speed of Light (SOL) analysis
- [ ] Warp stall analysis, roofline analysis
- [ ] **Code:** Profile GEMM kernels, compare SOL metrics between naive and tiled

### Day 3 — LLaMA Architecture
- [ ] Key differences from GPT: RMSNorm, SwiGLU, RoPE, no bias
- [ ] RMSNorm: `x / sqrt(mean(x²) + eps) * gamma`
- [ ] SwiGLU activation: `(xW₁ ⊙ Swish(xV)) W₂`
- [ ] Grouped Query Attention (GQA): share KV heads → smaller KV-cache
- [ ] Basic interpretability: residual stream, logit lens, attention heatmaps
- [ ] **Code:** Implement LLaMA model architecture from scratch

### Day 4 — Scaling Laws & MoE Overview
- [ ] Chinchilla scaling laws: optimal compute allocation (model size vs data)
- [ ] Loss = f(N, D, C) — implications for training decisions
- [ ] Emergent abilities at scale
- [ ] Mixture of Experts: increase params without proportional compute
- [ ] Top-k routing, load balancing loss
- [ ] **Code:** Implement simple MoE layer with top-2 routing

### Day 5 — Kernel Optimization & torch.compile
- [ ] Loop unrolling, instruction-level parallelism, register pressure
- [ ] `__launch_bounds__`, `__ldg()` for read-only memory
- [ ] Kernel fusion: reduce memory traffic
- [ ] `torch.compile` (TorchDynamo + TorchInductor): automatic kernel fusion
- [ ] Triton: Python GPU kernel language used by PyTorch
- [ ] **Code:** Compare `torch.compile` vs eager mode, profile both; add simple logit-lens probes to your small model

### 🔨 Saturday Project
- [ ] **LLaMA-Style Model with All Modern Features**
  - [ ] RMSNorm, RoPE, SwiGLU, GQA
  - [ ] KV-cache for inference
  - [ ] Train on small dataset, profile with Nsight Systems
  - [ ] Compare training speed with GPT-2 implementation

### 📄 Sunday Papers
- [ ] "LLaMA: Open and Efficient Foundation Language Models" (Touvron, 2023)
- [ ] "Training Compute-Optimal Large Language Models" (Chinchilla, Hoffmann 2022)

---

## Week 8 : GPU Memory Deep Dive & Data Pipelines

### Day 1 — GPU Memory Architecture
- [ ] HBM2 vs HBM2e vs HBM3: bandwidth, capacity, stack architecture
- [ ] NVLink: GPU-to-GPU interconnect bandwidth and topology
- [ ] PCIe Gen 3/4/5 bandwidth
- [ ] NVSwitch: full-bisection bandwidth fabric
- [ ] **Code:** Measure PCIe bandwidth (H2D, D2H), compare theoretical vs actual

### Day 2 — GPU Architecture Comparison
- [ ] A100: 108 SMs, 432 Tensor Cores, 40/80GB HBM2e
- [ ] H100: 132 SMs, 528 Tensor Cores, 80GB HBM3, Transformer Engine, FP8
- [ ] B100/B200 (Blackwell): latest features
- [ ] Calculate: theoretical peak FLOPS, memory bandwidth, roofline
- [ ] **Code:** Build roofline model for your specific GPU

### Day 3 — Mixed Precision Training
- [ ] FP32 vs FP16 vs BF16: bit layouts, range, precision
- [ ] Why BF16 preferred: same exponent range as FP32
- [ ] Tensor Cores: 4x-16x throughput for FP16/BF16/FP8
- [ ] PyTorch AMP: `torch.cuda.amp.autocast()` + `GradScaler`
- [ ] Loss scaling: prevent underflow in FP16 gradients
- [ ] **Code:** Train MNIST MLP with AMP, compare speed FP32 vs AMP

### Day 4 — Efficient Data Loading
- [ ] PyTorch `Dataset` and `DataLoader`
- [ ] `num_workers`, `pin_memory=True`, `prefetch_factor`
- [ ] `non_blocking=True` for async H2D transfer
- [ ] Gradient accumulation: simulate larger batch sizes
- [ ] Gradient checkpointing: trade compute for memory
- [ ] **Code:** Build efficient CIFAR-10 pipeline, monitor GPU utilization

### Day 5 — CNNs & Vision (needed to understand ViT, CLIP, LLaVA later)
- [ ] Convolution operation: sliding filter over input, producing feature maps
- [ ] Kernel (filter) sizes: 3x3, 5x5, 1x1 — what each captures
- [ ] Stride, padding, dilation — how they control output size
- [ ] Pooling layers: max pooling, average pooling — spatial downsampling
- [ ] Key CNN architectures (know what they contributed):
  - [ ] LeNet (1998): first practical CNN
  - [ ] AlexNet (2012): deep CNNs + GPU training started the DL revolution
  - [ ] ResNet (2015): residual connections (same idea used in Transformers!)
  - [ ] EfficientNet: compound scaling
- [ ] Why convolutions work: translation invariance, parameter sharing, hierarchical features
- [ ] Feature visualization: early layers detect edges, mid layers detect textures, deep layers detect objects
- [ ] Why Transformers replaced CNNs in vision: ViT showed attention over patches beats convolutions
- [ ] **Code:** Build CNN for CIFAR-10 (Conv → BatchNorm → ReLU → Pool → FC), achieve >85% accuracy

### 🔨 Saturday Project
- [ ] **Mixed Precision Training Benchmark**
  - [ ] Train same model: FP32, FP16+AMP, BF16
  - [ ] Measure: training speed, memory usage, final accuracy
  - [ ] Profile Tensor Core utilization with Nsight
  - [ ] Generate comprehensive comparison report

### 📄 Sunday Paper
- [ ] Skim: "Efficient Processing of Deep Neural Networks" (Sze et al., 2020) — hardware chapter

---

## 🔄 Buffer Week (Month 2 Revision)
- [ ] Revise Transformer architecture: attention, multi-head, positional encoding, FFN
- [ ] Revise GPU libraries: cuBLAS, Tensor Cores, Nsight profiling
- [ ] Revise LLaMA architecture: RMSNorm, RoPE, SwiGLU, GQA
- [ ] Re-read your GPT-2 implementation, understand every line
- [ ] **Build Monthly Project A:** nanoLLM — GPT from Scratch
- [ ] **Build Monthly Project B:** TransformerScope — LLM Behavior Debugger
- [ ] Push both projects to GitHub

---

## Week 9 : Distributed CUDA & Pre-training Concepts

### Day 1 — NCCL & Data Parallelism
- [ ] NCCL: AllReduce, AllGather, ReduceScatter, Broadcast
- [ ] Ring AllReduce: how gradients sync across GPUs
- [ ] PyTorch DDP: process groups, ranks, gradient bucketing
- [ ] `torchrun` for launching distributed training
- [ ] **Code:** Convert GPT training to DDP

### Day 2 — Model & Pipeline Parallelism
- [ ] Tensor parallelism: split layers across GPUs (column-parallel, row-parallel)
- [ ] Pipeline parallelism: split model layers across GPUs, micro-batching
- [ ] ZeRO stages 1/2/3: partition optimizer states, gradients, parameters
- [ ] DeepSpeed, FSDP
- [ ] **Code:** Implement simple tensor-parallel linear layer

### Day 3 — Pre-training Data & Tokenization
- [ ] Data sources: Common Crawl, Wikipedia, Books, Code
- [ ] Deduplication, filtering, quality scoring
- [ ] Data mixing proportions
- [ ] Training tokenizer on large corpus
- [ ] **Code:** Process a small text corpus end-to-end: clean → dedupe → tokenize

### Day 4 — Pre-training Implementation
- [ ] Training objective: next token prediction (causal LM)
- [ ] Packing documents with separator tokens
- [ ] LR schedule: warmup + cosine decay
- [ ] Weight initialization for deep networks
- [ ] **Code:** Complete pre-training script with all details

### Day 5 — Training Stability & Evaluation
- [ ] Loss spikes, gradient norm monitoring, weight norms
- [ ] Z-loss: prevent logit growth
- [ ] Perplexity: exp(avg cross-entropy loss)
- [ ] Benchmarks: MMLU, HellaSwag, ARC
- [ ] **Code:** Add comprehensive training logging, evaluate with lm-evaluation-harness

### 🔨 Saturday Project
- [ ] **Distributed GPT Training Pipeline**
  - [ ] DDP training
  - [ ] wandb logging
  - [ ] Checkpointing and resumption
  - [ ] Evaluation hooks every N steps

### 📄 Sunday Papers
- [ ] "Megatron-LM: Training Multi-Billion Parameter Models" (Shoeybi, 2019)
- [ ] "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models" (Rajbhandari, 2019)

---

## Week 10 : Custom CUDA for PyTorch & Instruction Following

### Day 1 — CUTLASS & TensorRT Basics
- [ ] CUTLASS: template library for GEMM, maps to GPU hierarchy
- [ ] TensorRT: layer fusion, precision calibration, kernel auto-tuning
- [ ] ONNX → TensorRT workflow
- [ ] **Code:** Convert PyTorch model to ONNX → TensorRT, benchmark speedup

### Day 2 — Custom CUDA Extensions for PyTorch
- [ ] `torch.utils.cpp_extension`
- [ ] Writing CUDA kernels callable from PyTorch
- [ ] `torch.autograd.Function` for custom forward/backward
- [ ] JIT compilation with `load_inline`
- [ ] **Code:** Write custom fused LayerNorm+ReLU CUDA kernel for PyTorch

### Day 3 — Supervised Fine-Tuning (SFT)
- [ ] Pre-trained → instruction-following via SFT
- [ ] Instruction data formats: instruction/input/output, conversation
- [ ] SFT details: lower LR, shorter training, loss masking on prompts only
- [ ] **Code:** Prepare instruction dataset, fine-tune small GPT on it

### Day 4 — RLHF & DPO
- [ ] RLHF overview: SFT → Reward Model → PPO
- [ ] Bradley-Terry model for preferences
- [ ] DPO: simpler, no separate reward model
- [ ] DPO loss: binary cross-entropy on preference pairs
- [ ] **Code:** Implement DPO training, fine-tune with preference data

### Day 5 — Chat Templates & Evaluation
- [ ] ChatML, LLaMA chat format, system prompts
- [ ] Multi-turn conversation formatting
- [ ] `apply_chat_template()`
- [ ] MT-Bench, AlpacaEval evaluation
- [ ] **Code:** Build chat interface for fine-tuned model with proper templates

### 🔨 Saturday Project
- [ ] **Complete LLM Fine-Tuning Pipeline**
  - [ ] Pre-trained model → SFT on instructions
  - [ ] DPO on preferences
  - [ ] Evaluate before/after alignment
  - [ ] Simple chat interface

### 📄 Sunday Papers
- [ ] "Training language models to follow instructions with human feedback" (InstructGPT, 2022)
- [ ] "Direct Preference Optimization" (Rafailov, 2023)

---

## Week 11 : Flash Attention & Triton

### Day 1 — Flash Attention Theory
- [ ] Standard attention: O(N²) memory for attention matrix
- [ ] Flash Attention: O(N) memory via tiling + online softmax
- [ ] Online softmax trick: incremental computation without full matrix
- [ ] Keep everything in SRAM (shared memory)
- [ ] IO complexity: standard O(N²d) vs Flash O(N²d²/M)

### Day 2 — Flash Attention 2 & 3
- [ ] FA2: better parallelism over sequence length
- [ ] FA3: Hopper features (TMA, warp specialization)
- [ ] **Code:** Study Flash Attention Triton implementation, understand each optimization

### Day 3 — Triton GPU Programming
- [ ] Triton: Python DSL → optimized GPU code
- [ ] Automatic memory coalescing, shared memory management
- [ ] `@triton.jit`, block pointers
- [ ] **Code:** Implement fused softmax in Triton, compare with PyTorch native

### Day 4 — PTX, SASS & Low-Level Analysis
- [ ] PTX: virtual ISA, SASS: actual machine code
- [ ] `cuobjdump` for kernel inspection
- [ ] Understanding instruction mapping
- [ ] Vectorized memory: `float4` for wider transactions
- [ ] **Code:** Inspect PTX/SASS of your kernels, identify optimization opportunities

### Day 5 — Advanced Memory Patterns
- [ ] Software pipelining: overlap loads with computation
- [ ] Double buffering: load next tile while computing current
- [ ] **Code:** Implement double-buffered GEMM kernel

### 🔨 Saturday Project
- [ ] **Triton Kernel Library**
  - [ ] Fused softmax
  - [ ] Fused LayerNorm
  - [ ] Fused attention (simplified Flash Attention)
  - [ ] Benchmark all vs PyTorch native

### 📄 Sunday Papers
- [ ] "FlashAttention: Fast and Memory-Efficient Exact Attention" (Dao, 2022)
- [ ] "Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations" (Tillet, 2019)

---

## Week 12 : Tokenizer Mastery & Multi-Modal Overview

### Day 1 — Tokenizer Deep Dive
- [ ] BPE algorithm implementation details
- [ ] SentencePiece vs tiktoken: byte-level handling differences
- [ ] Special tokens: `<bos>`, `<eos>`, `<pad>`, `<unk>`, chat templates
- [ ] Tokenizer training: building vocabulary from corpus
- [ ] **Code:** Train BPE tokenizer with `tokenizers` library, compare with tiktoken

### Day 2 — Multi-Modal Models Overview
- [ ] CLIP: contrastive image-text pre-training
- [ ] Vision-Language Models: LLaVA, GPT-4V architecture
- [ ] How images become tokens: patch embedding
- [ ] **Code:** Run CLIP for zero-shot classification

### Day 3 — GPU Cluster Architecture & NVIDIA Systems
- [ ] DGX systems: DGX A100 (8xA100 NVLink), DGX H100 (8xH100 NVSwitch), DGX B200
- [ ] DGX SuperPOD: cluster of DGX systems
- [ ] HGX: GPU baseboard design (what OEMs use)
- [ ] InfiniBand (ConnectX, BlueField DPUs) for multi-node
- [ ] NVIDIA Base Command: cluster management, job scheduling
- [ ] `nvidia-smi topo -m`: understanding topology
- [ ] **Code:** Run `nvidia-smi`, `nvidia-smi topo`, understand your system topology

### Day 4 — Unified Memory & Advanced Allocation
- [ ] `cudaMallocManaged`: single pointer for CPU and GPU
- [ ] Page migration, prefetching: `cudaMemPrefetchAsync`
- [ ] Pinned memory: `cudaMallocHost`
- [ ] Memory pools: `cudaMallocAsync` / `cudaFreeAsync`
- [ ] **Code:** Compare transfer speeds: pageable vs pinned vs unified

### Day 5 — Review & Consolidation
- [ ] Review all CUDA concepts from Phase 1
- [ ] Review all DL/LLM concepts from Phase 1
- [ ] Identify weak areas for extra study
- [ ] **Code:** Revisit and optimize your best project

### 🔨 Saturday Project
- [ ] **Phase 1 Capstone: Mini LLM from Scratch**
  - [ ] Custom tokenizer
  - [ ] LLaMA-style architecture
  - [ ] Train on a text corpus
  - [ ] Custom CUDA kernel for at least one operation
  - [ ] Profile and optimize
  - [ ] Chat-style inference with KV-cache

### 📄 Sunday
- [ ] Review all papers from Phase 1
- [ ] Write summary of key learnings

---

### ✅ Phase 1 Completion Checklist
- [ ] Can explain GPU SM architecture from memory
- [ ] Can write and optimize CUDA kernels (shared memory, coalescing, reduction)
- [ ] Can implement Transformer from scratch
- [ ] Can train a small LLM (GPT-2 style)
- [ ] Can use Nsight Systems/Compute for profiling
- [ ] Understand FP32/FP16/BF16 and Tensor Cores
- [ ] Can fine-tune models with SFT and DPO

---

# ═══════════════════════════════════════════════════
# PHASE 2: INTERMEDIATE (Weeks 13-26, Months 4-6)
# PEFT, Quantization, Inference, RAG, Agents, Reasoning
# ═══════════════════════════════════════════════════

---

## 🔄 Buffer Week (Month 3 Revision)
- [ ] Revise distributed: DDP, ZeRO, tensor parallelism, NCCL
- [ ] Revise post-training: SFT, DPO, chat templates
- [ ] Revise Flash Attention algorithm and Triton programming
- [ ] Re-read custom CUDA extensions code
- [ ] **Build Monthly Project A:** LLM Surgery — Fine-Tuning & Alignment Toolkit
- [ ] **Build Monthly Project B:** TokenScope — Tokenizer Analyzer
- [ ] Push both projects to GitHub

---

## Week 13 : LoRA & Parameter-Efficient Fine-Tuning

### Day 1 — KV-Cache Implementation
- [ ] Why KV-cache: avoid recomputing past key-value pairs
- [ ] Memory layout: `[batch, num_heads, seq_len, head_dim]`
- [ ] Pre-allocated buffers, cache management
- [ ] Memory overhead calculation formula
- [ ] **Code:** Implement KV-cache, benchmark inference speedup

### Day 2 — LoRA from Scratch
- [ ] Full fine-tuning problem: expensive, catastrophic forgetting
- [ ] LoRA: `ΔW = BA` where B∈ℝ^{d×r}, A∈ℝ^{r×k}
- [ ] Rank selection (r=8 to 64), alpha scaling
- [ ] Which layers: typically Q, K, V, O projections
- [ ] Merging: `W' = W + BA` for zero-overhead deployment
- [ ] **Code:** Implement LoRA from scratch in PyTorch

### Day 3 — QLoRA
- [ ] 4-bit quantization of base model + LoRA in FP16/BF16
- [ ] NF4 (NormalFloat 4-bit): optimized for normal distribution
- [ ] Double quantization, paged optimizers
- [ ] bitsandbytes library
- [ ] **Code:** Fine-tune 7B model with QLoRA using PEFT library

### Day 4 — Other PEFT Methods
- [ ] Prefix tuning, prompt tuning, adapters, IA³
- [ ] Comparing methods: parameter efficiency vs task performance
- [ ] **Code:** Implement prefix tuning, compare with LoRA on same task

### Day 5 — Multi-LoRA & Best Practices
- [ ] Multiple LoRA adapters for different tasks
- [ ] LoRA merging: TIES, DARE
- [ ] Fine-tuning best practices: data quality, hyperparameters, evaluation
- [ ] **Code:** Train multiple LoRA adapters, experiment with merging

### 🔨 Saturday Project
- [ ] **LoRA Fine-Tuning Toolkit**
  - [ ] LoRA + QLoRA from scratch
  - [ ] Multiple target module support
  - [ ] Adapter merging
  - [ ] Compare with HuggingFace PEFT results

### 📄 Sunday Papers
- [ ] "LoRA: Low-Rank Adaptation of Large Language Models" (Hu, 2021)
- [ ] "QLoRA: Efficient Finetuning of Quantized Language Models" (Dettmers, 2023)

---

## Week 14 : Quantization Deep Dive

### Day 1 — Quantization Fundamentals
- [ ] Number formats: FP32, FP16, BF16, FP8 (E4M3/E5M2), INT8, INT4
- [ ] `q = round(x / scale + zero_point)`
- [ ] Symmetric vs asymmetric, per-tensor vs per-channel vs per-group
- [ ] **Code:** Implement symmetric & asymmetric quantization in PyTorch

### Day 2 — GPTQ & AWQ
- [ ] Post-Training Quantization: weight-only vs W+A
- [ ] Calibration: MinMax, percentile, MSE-optimal
- [ ] GPTQ: one-shot weight quantization using Hessian info
- [ ] AWQ: protect salient channels based on activation awareness
- [ ] **Code:** Quantize a model with GPTQ and AWQ, compare

### Day 3 — CUDA Kernels for Quantized Inference
- [ ] INT4 matmul: packing two INT4 in one byte
- [ ] Dequantize-on-the-fly: load INT4 → FP16 → compute
- [ ] Marlin kernel, ExLlama v2 kernels
- [ ] **Code:** Write simple INT8 GEMM kernel in CUDA

### Day 4 — FP8 & Transformer Engine
- [ ] FP8 E4M3 vs E5M2
- [ ] Per-tensor scaling, delayed scaling
- [ ] H100 Transformer Engine: hardware FP8 with auto-scaling
- [ ] **Code:** Use Transformer Engine for FP8 training

### Day 5 — Practical Quantization
- [ ] GGUF & llama.cpp: Q4_0, Q5_1, Q8_0, K-quant
- [ ] AutoGPTQ, AutoAWQ libraries
- [ ] SmoothQuant: smoothing activation outliers
- [ ] **Code:** Quantize model to GGUF, benchmark perplexity, speed, memory

### 🔨 Saturday Project
- [ ] **Quantization Benchmark Suite**
  - [ ] Quantize 7B model: GPTQ 4-bit, AWQ 4-bit, GGUF Q4/Q5/Q8
  - [ ] Evaluate: perplexity, MMLU subset, throughput, memory
  - [ ] Generate comparison tables and plots

### 📄 Sunday Papers
- [ ] "GPTQ: Accurate Post-Training Quantization" (Frantar, 2022)
- [ ] "AWQ: Activation-aware Weight Quantization" (Lin, 2023)

---

## Week 15 : Inference Serving

### Day 1 — Inference Performance Analysis
- [ ] Prefill (compute-bound) vs Decode (memory-bandwidth-bound)
- [ ] TTFT, TPOT, throughput — what each measures
- [ ] Throughput vs latency tradeoffs
- [ ] **Code:** Build inference benchmarking tool measuring TTFT, TPOT, throughput

### Day 2 — TensorRT-LLM
- [ ] Architecture: Python model definition → C++ runtime → CUDA kernels
- [ ] Build vs runtime phase
- [ ] In-flight batching, KV-cache management, multi-GPU
- [ ] **Code:** Build TensorRT-LLM engine for LLaMA, benchmark

### Day 3 — vLLM & PagedAttention
- [ ] Static batching waste → continuous batching
- [ ] PagedAttention: KV-cache as virtual memory pages
- [ ] Block tables, prefix caching, chunked prefill
- [ ] **Code:** Deploy model on vLLM, compare with TensorRT-LLM

### Day 4 — Speculative Decoding
- [ ] Problem: autoregressive = memory-bandwidth-bound, GPU underutilized
- [ ] Draft model generates candidates, large model verifies in parallel
- [ ] Medusa: multiple draft heads
- [ ] **Code:** Implement basic speculative decoding

### Day 5 — Other Serving Frameworks & NVIDIA Dynamo
- [ ] Triton Inference Server: model repository, dynamic batching, ensemble
- [ ] NVIDIA Dynamo: open-source multi-node inference (disaggregated prefill/decode)
- [ ] TGI (HuggingFace), SGLang
- [ ] **Code:** Deploy on Triton, quick test on TGI, explore Dynamo repo

### 🔨 Saturday Project
- [ ] **LLM Inference Serving Platform**
  - [ ] Deploy on TensorRT-LLM, vLLM, TGI
  - [ ] Benchmark: batch sizes, sequence lengths, quantization
  - [ ] REST API wrapper
  - [ ] Performance comparison report with graphs

### 📄 Sunday Papers
- [ ] "Efficient Memory Management for LLM Serving with PagedAttention" (vLLM, Kwon 2023)
- [ ] "Efficiently Scaling Transformer Inference" (Pope, 2022)

---

## Week 16 : RAG Foundations

### Day 1 — Vector Search & Embeddings
- [ ] Cosine similarity, L2 distance, inner product
- [ ] Embedding models: e5-large, BGE, GTE, nomic-embed, **NV-EmbedQA** (NVIDIA)
- [ ] Reranking models: **NV-RerankQA** (NVIDIA), Cohere reranker
- [ ] FAISS: flat index, IVF, PQ, HNSW
- [ ] FAISS GPU, cuVS / CAGRA (NVIDIA vector search)
- [ ] NeMo Retriever NIM: NVIDIA's embedding + reranking microservices
- [ ] **Code:** Use NV-EmbedQA via NIM or build.nvidia.com API, compare with FAISS GPU

### Day 2 — RAG Architecture
- [ ] Query → retrieve → augment prompt → generate
- [ ] Chunking: fixed-size, recursive, semantic
- [ ] Chunk size tradeoffs
- [ ] **Code:** Build basic RAG pipeline end-to-end

### Day 3 — Vector Databases & Hybrid Search
- [ ] ChromaDB, Milvus, Qdrant, Pinecone
- [ ] Hybrid search: dense (embedding) + sparse (BM25)
- [ ] Re-ranking with cross-encoder
- [ ] **Code:** Set up Milvus, implement hybrid search + re-ranking

### Day 4 — Advanced Retrieval
- [ ] Multi-query retrieval, HyDE
- [ ] Metadata filtering, caching
- [ ] **Code:** Implement HyDE and multi-query retrieval

### Day 5 — RAG Evaluation
- [ ] Retrieval: recall@k, MRR, NDCG
- [ ] Generation: faithfulness, relevance, correctness
- [ ] RAGAS framework
- [ ] **Code:** Evaluate RAG pipeline with RAGAS

### 🔨 Saturday Project
- [ ] **Production RAG System**
  - [ ] Document ingestion (PDF, markdown, web)
  - [ ] Chunking with overlap
  - [ ] GPU-accelerated embedding + indexing
  - [ ] Hybrid retrieval + re-ranking
  - [ ] LLM generation with source citations
  - [ ] FastAPI REST API
  - [ ] RAGAS evaluation pipeline

### 📄 Sunday Papers
- [ ] "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (Lewis, 2020)
- [ ] "Lost in the Middle: How Language Models Use Long Contexts" (Liu, 2023)

---

## 🔄 Buffer Week (Month 4 Revision)
- [ ] Revise LoRA/QLoRA: math, implementation, merging strategies
- [ ] Revise quantization: GPTQ, AWQ, GGUF, FP8, INT4 CUDA kernels
- [ ] Revise inference serving: TensorRT-LLM, vLLM, PagedAttention, speculative decoding
- [ ] Revise RAG: chunking, hybrid retrieval, re-ranking, evaluation
- [ ] **Build Monthly Project A:** QuantBench — Quantization Analyzer
- [ ] **Build Monthly Project B:** LLM-Speedometer — Inference Profiler
- [ ] Push both projects to GitHub

---

## Week 17 : Advanced RAG

### Day 1 — Agentic RAG & Self-RAG
- [ ] RAG as a tool: LLM decides when/how to retrieve
- [ ] Self-RAG: model decides if retrieval is needed
- [ ] CRAG: evaluate retrieval quality, fallback to web search
- [ ] **Code:** Implement self-RAG with quality self-evaluation

### Day 2 — Graph RAG
- [ ] Knowledge graphs from documents using LLMs
- [ ] Graph-based retrieval: traverse relationships
- [ ] Microsoft's GraphRAG: hierarchical community summarization
- [ ] **Code:** Build knowledge graph, implement graph-based RAG

### Day 3 — Multi-Modal & Code RAG
- [ ] Images, tables, charts in documents
- [ ] ColPali/ColQwen for document retrieval
- [ ] Code RAG: function-level chunking, AST parsing
- [ ] **Code:** Multi-modal RAG handling PDFs with images

### Day 4 — Fine-tuning for RAG
- [ ] Fine-tuning embedding models: contrastive learning on domain data
- [ ] RAFT: train model to cite sources
- [ ] **Code:** Fine-tune embedding model on domain-specific data

### Day 5 — GPU-Accelerated RAG
- [ ] GPU-accelerated embedding generation (batch processing)
- [ ] cuVS CAGRA vs FAISS GPU IVF-PQ comparison
- [ ] TensorRT for embedding model optimization
- [ ] **Code:** Optimize full RAG pipeline for GPU

### 🔨 Saturday Project
- [ ] **Advanced RAG System**
  - [ ] Compare: naive RAG, agentic RAG, graph RAG
  - [ ] Multi-document type support
  - [ ] GPU-accelerated retrieval
  - [ ] Evaluation framework for all approaches

### 📄 Sunday Papers
- [ ] "Self-RAG: Learning to Retrieve, Generate, and Critique" (Asai, 2023)
- [ ] "From Local to Global: A Graph RAG Approach" (Microsoft, 2024)

---

## Week 18 : AI Agents — Foundations

### Day 1 — Tool Use & Function Calling
- [ ] Function calling: LLMs generating structured tool calls
- [ ] JSON schema for function definitions
- [ ] Parallel tool calling
- [ ] **Code:** Implement function calling parser

### Day 2 — ReAct Pattern
- [ ] Reasoning + Acting: Thought → Action → Observation cycle
- [ ] Grounding in real tool outputs
- [ ] **Code:** Implement ReAct agent from scratch (no framework)

### Day 3 — Agent Memory & Planning
- [ ] Short-term, working, long-term memory
- [ ] Memory retrieval with embeddings
- [ ] Task decomposition, plan-and-execute
- [ ] Reflexion, Tree of Thought
- [ ] **Code:** Agent with short-term + long-term memory

### Day 4 — LangChain & LlamaIndex
- [ ] LangChain: chains, agents, tools, LCEL
- [ ] LlamaIndex: document ingestion, query engines, data agents
- [ ] **Code:** Multi-tool agent with LangChain

### Day 5 — Multi-Agent Systems & NVIDIA AIQ
- [ ] CrewAI: role assignment, task definition
- [ ] AutoGen: conversable agents, group chat
- [ ] NVIDIA AIQ / Agent Intelligence Toolkit: toolkit for building enterprise AI agents
- [ ] Human-in-the-loop
- [ ] **Code:** Multi-agent research + writing system, explore NVIDIA AIQ examples

### 🔨 Saturday Project
- [ ] **Research Agent**
  - [ ] Search web, read papers, analyze code, write summaries
  - [ ] Multi-step reasoning with ReAct
  - [ ] Long-term memory for research context
  - [ ] Evaluate on research-style tasks

### 📄 Sunday Papers
- [ ] "ReAct: Synergizing Reasoning and Acting in Language Models" (Yao, 2022)
- [ ] "Toolformer: Language Models Can Teach Themselves to Use Tools" (Schick, 2023)

---

## Week 19 : AI Agents — Advanced

### Day 1 — Code-Executing & Browser-Using Agents
- [ ] Sandboxed execution: Docker, E2B
- [ ] Code generation + execution + error handling loop
- [ ] Web automation: Playwright, vision-based interaction
- [ ] **Code:** Agent that generates and runs Python safely

### Day 2 — Agent Orchestration
- [ ] Supervisor pattern, hierarchical agents
- [ ] Consensus, checkpoint and recovery
- [ ] **Code:** Hierarchical agent system

### Day 3 — Structured Output & Safety
- [ ] JSON mode, Pydantic, grammar-constrained generation (outlines)
- [ ] Prompt injection defense, sandboxing
- [ ] **Code:** Grammar-constrained generation + safety checks

### Day 4 — Agent Monitoring & Deployment
- [ ] LangSmith tracing, custom telemetry, cost tracking
- [ ] NeMo Guardrails integration for agent safety
- [ ] REST API with streaming (WebSocket/SSE)
- [ ] **Code:** Deploy agent with streaming API + NeMo Guardrails

### Day 5 — Agent Evaluation
- [ ] SWE-bench, GAIA, AgentBench, ToolBench
- [ ] Success rate, efficiency, reliability metrics
- [ ] **Code:** Build evaluation harness

### 🔨 Saturday Project
- [ ] **Full-Featured AI Agent Platform**
  - [ ] Code execution + file system + web browsing
  - [ ] RAG for knowledge, guardrails for safety
  - [ ] Monitoring dashboard
  - [ ] REST API with streaming

### 📄 Sunday Papers
- [ ] "Voyager: An Open-Ended Embodied Agent with LLMs" (Wang, 2023)
- [ ] "SWE-agent: Agent-Computer Interfaces Enable Automated SE" (Yang, 2024)

---

## Week 20 : NVIDIA AI Ecosystem — Part 1

### Day 1 — NVIDIA NGC & build.nvidia.com
- [ ] NGC (NVIDIA GPU Cloud): GPU-optimized container registry
- [ ] NGC Catalog: pre-trained models, containers, Helm charts, resources
- [ ] build.nvidia.com: API playground for NVIDIA-hosted models
- [ ] NVIDIA API Catalog: browse and test LLMs, embedding models, rerankers
- [ ] NV API keys: how to use NVIDIA's hosted inference endpoints
- [ ] **Code:** Get NV API key, call models via build.nvidia.com APIs, explore NGC catalog

### Day 2 — NVIDIA NIM (Inference Microservices)
- [ ] NIM: optimized, containerized inference for LLMs, VLMs, embedding, reranking
- [ ] NIM architecture: TensorRT-LLM backend, OpenAI-compatible REST API
- [ ] NIM for LLMs: deploy Llama, Mistral, etc. with single `docker run`
- [ ] NIM for embeddings: NV-EmbedQA, NV-RerankQA
- [ ] NIM profiles: selecting optimization for your GPU (FP8, INT4, etc.)
- [ ] **Code:** Pull and run NIM container, benchmark latency/throughput, compare with vLLM

### Day 3 — NeMo Framework
- [ ] NeMo 2.0: pre-training, fine-tuning (SFT, PEFT, DPO), alignment
- [ ] NeMo Curator: GPU-accelerated data curation (dedup, filter, classify, PII redaction)
- [ ] NeMo Guardrails: Colang language, topical/safety/fact-checking rails
- [ ] NeMo Retriever: embedding + reranking microservices for RAG
- [ ] NeMo Customizer: API for fine-tuning NIM models
- [ ] **Code:** Fine-tune a model with NeMo, add Guardrails, use NeMo Retriever for RAG

### Day 4 — RAPIDS & NVIDIA DALI
- [ ] cuDF: pandas on GPU (100x speedup for dataframe ops)
- [ ] cuML: GPU-accelerated ML (sklearn-compatible)
- [ ] cuGraph: GPU-accelerated graph analytics
- [ ] NVIDIA DALI: GPU-accelerated data loading pipeline for training
- [ ] **Code:** Data preprocessing pipeline: RAPIDS for tabular + DALI for images, benchmark vs CPU

### Day 5 — TensorRT-LLM Advanced & Triton Inference Server
- [ ] TensorRT-LLM: custom model support, plugin system, quantization workflows
- [ ] Triton Inference Server: model repository, dynamic batching, ensemble models
- [ ] Triton + TensorRT-LLM integration: production serving stack
- [ ] Model Analyzer: profiling model configurations for optimal serving
- [ ] **Code:** Deploy TensorRT-LLM model on Triton, use Model Analyzer to optimize

### 🔨 Saturday Project
- [ ] **NVIDIA Full-Stack AI Application**
  - [ ] NIM for LLM inference
  - [ ] NeMo Retriever for embeddings + reranking
  - [ ] NeMo Guardrails for safety
  - [ ] Triton for serving
  - [ ] Compare with non-NVIDIA alternatives (vLLM, HF embeddings)

### 📄 Sunday
- [ ] NVIDIA NeMo documentation
- [ ] NVIDIA NIM documentation
- [ ] Explore build.nvidia.com API catalog

---

## Week 20.5 : NVIDIA AI Ecosystem — Part 2

### Day 1 — NVIDIA AI Workbench & AI Enterprise
- [ ] AI Workbench: unified dev environment for AI projects
- [ ] AI Enterprise: production deployment platform (support, security, management)
- [ ] Base Command Manager: managing GPU clusters and workloads
- [ ] Fleet Command: managing edge AI deployments
- [ ] **Code:** Set up AI Workbench project, explore Base Command if available

### Day 2 — NVIDIA Riva & ACE
- [ ] Riva: ASR (speech-to-text) + TTS (text-to-speech) on GPU
- [ ] Riva NIM: containerized speech AI microservices
- [ ] NVIDIA ACE (Avatar Cloud Engine): AI-powered digital humans
- [ ] Audio2Face, Omniverse integration for digital avatars
- [ ] **Code:** Build voice-enabled chatbot: Riva ASR → LLM NIM → Riva TTS

### Day 3 — NVIDIA AI Blueprints
- [ ] Pre-built reference architectures for common AI workflows
- [ ] RAG Blueprint: complete RAG pipeline with NIM + NeMo Retriever
- [ ] Digital Human Blueprint: ACE + Riva + LLM
- [ ] Agentic AI Blueprint: agents with tools and guardrails
- [ ] **Code:** Deploy and customize a RAG AI Blueprint

### Day 4 — NVIDIA Dynamo & Inference Optimization
- [ ] NVIDIA Dynamo: open-source inference serving framework for multi-node/multi-GPU
- [ ] Disaggregated serving: separate prefill and decode phases across GPUs
- [ ] Smart routing: intelligent request routing to optimal GPUs
- [ ] KV-cache offloading: extend effective context via CPU/NVMe
- [ ] **Code:** Explore NVIDIA Dynamo, benchmark disaggregated vs colocated serving

### Day 5 — NVIDIA cuOpt, Morpheus & Other Tools
- [ ] cuOpt: GPU-accelerated optimization solver (routing, scheduling)
- [ ] Morpheus: AI-powered cybersecurity framework
- [ ] NVIDIA Nsight DL Designer: visual tool for DL model design (if available)
- [ ] NVIDIA Nsight AI: profiling AI workloads
- [ ] cuda-python: Python bindings for CUDA driver and runtime APIs
- [ ] **Code:** Explore Morpheus for log analysis, try cuOpt for a simple optimization

### 🔨 Saturday Project
- [ ] **Enterprise AI Agent with Full NVIDIA Stack**
  - [ ] NIM for LLM + embedding + reranking
  - [ ] NeMo Guardrails for safety
  - [ ] Riva for voice I/O
  - [ ] NVIDIA AI Blueprint as starting point
  - [ ] RAG with NeMo Retriever
  - [ ] Production deployment on Triton

---

## 🔄 Buffer Week (Month 5 Revision)
- [ ] Revise advanced RAG: agentic, graph, multi-modal
- [ ] Revise agents: ReAct, tool use, memory, multi-agent, orchestration
- [ ] Revise full NVIDIA stack: NIM, NeMo, Guardrails, Retriever, Riva, ACE
- [ ] Re-run your best agent, try to improve it
- [ ] **Build Monthly Project A:** DeepRAG — Production RAG
- [ ] **Build Monthly Project B:** RAGTrace — Retrieval Debugger
- [ ] Push both projects to GitHub

---

## Week 21-22 : Knowledge Distillation & Model Merging

### Week 21 Topics
- [ ] Teacher-student paradigm, soft targets, KD loss
- [ ] Feature distillation, attention transfer
- [ ] Progressive distillation, self-distillation
- [ ] **Code:** Distill large GPT → smaller GPT, implement feature distillation

### Week 22 Topics
- [ ] Model merging: linear interpolation, SLERP, TIES, DARE
- [ ] Model soups: average fine-tuned checkpoints
- [ ] Synthetic data generation: self-instruct, Evol-Instruct, Magpie
- [ ] Curriculum learning
- [ ] **Code:** Merge models, generate synthetic dataset, train model on it

### 🔨 Saturday Projects
- [ ] **Week 21:** Distill 7B → 1.3B model, evaluate on benchmarks
- [ ] **Week 22:** Synthetic data pipeline → fine-tune → evaluate

### 📄 Sunday Papers
- [ ] "Distilling the Knowledge in a Neural Network" (Hinton, 2015)
- [ ] "DARE: Language Models are Super Mario" (2023)

---

## Week 23-24 : RL Foundations, RLHF & Reasoning

### Week 23 — RL Foundations + RLHF Pipeline

**RL Foundations (Days 1-2) — needed to truly understand RLHF/PPO/GRPO:**
- [ ] What is RL: agent, environment, state, action, reward, policy
- [ ] Markov Decision Process (MDP): states, transitions, discount factor
- [ ] Value function V(s): expected total reward from state s
- [ ] Action-value function Q(s,a): expected reward for taking action a in state s
- [ ] Policy gradient: directly optimize the policy (REINFORCE algorithm)
- [ ] Actor-Critic: policy network (actor) + value network (critic)
- [ ] Advantage function A(s,a) = Q(s,a) - V(s): "how much better is this action than average"
- [ ] GAE (Generalized Advantage Estimation): balance bias-variance in advantage estimates
- [ ] **Code:** Implement REINFORCE on CartPole in PyTorch (simple, ~50 lines)

**PPO & GRPO (Days 3-4):**
- [ ] PPO: clipped objective — prevent policy from changing too much per update
- [ ] PPO hyperparameters: clip ratio, epochs per batch, batch size
- [ ] Why PPO works for LLMs: stable updates, doesn't collapse
- [ ] GRPO (DeepSeek): no value model needed, group relative scoring
- [ ] **Code:** Implement PPO from scratch on CartPole, then understand how it applies to LLMs

**Full RLHF Pipeline (Day 5 + continuing):**
- [ ] Full RLHF: SFT → reward model → PPO with KL penalty
- [ ] Reward model: train on human preferences (Bradley-Terry model)
- [ ] KL penalty: prevent policy from diverging too far from SFT model
- [ ] RLAIF: LLM as judge instead of humans
- [ ] Process reward models: reward each reasoning step
- [ ] **Code:** Full RLHF pipeline with trl library, implement GRPO for LLM alignment

### Week 24 — Reasoning & Chain-of-Thought
- [ ] Zero-shot & few-shot CoT
- [ ] Tree of Thought, self-consistency, least-to-most prompting
- [ ] Training for reasoning: STaR, RL for reasoning
- [ ] DeepSeek-R1: RL-driven reasoning emergence
- [ ] **Code:** Implement ToT + self-consistency, STaR training loop

### 🔨 Saturday Projects
- [ ] **Week 23:** Compare PPO vs DPO vs GRPO alignment on same model
- [ ] **Week 24:** Reasoning agent: CoT + tool use + process reward model

### 📄 Sunday Papers
- [ ] "Constitutional AI: Harmlessness from AI Feedback" (Bai, 2022)
- [ ] "Chain-of-Thought Prompting Elicits Reasoning" (Wei, 2022)
- [ ] "DeepSeek-R1: Incentivizing Reasoning via RL" (DeepSeek, 2025)

---

## Week 25-26 : Long Context & Efficient Architectures

### Week 25 — Long Context
- [ ] RoPE scaling: linear, NTK-aware, YaRN, Dynamic NTK, ABF
- [ ] Flash Attention tiling + online softmax in detail
- [ ] Ring Attention: distribute across GPUs
- [ ] KV-cache compression: INT4/INT8, eviction (H2O)
- [ ] Sliding window attention (Mistral), StreamingLLM
- [ ] **Code:** Implement RoPE scaling, KV-cache quantization, NIAH test

### Week 26 — State Space Models & Alternatives
- [ ] S4, Mamba: linear-time sequence modeling
- [ ] Mamba-2, Jamba (hybrid attention + Mamba)
- [ ] RWKV, RetNet
- [ ] When to use attention vs SSM
- [ ] **Code:** Implement Mamba layer, compare with Transformer

### 🔨 Saturday Projects
- [ ] **Week 25:** Long context evaluation suite (NIAH, passkey, doc QA at 4K-128K)
- [ ] **Week 26:** Hybrid attention+Mamba model, train and evaluate

### 📄 Sunday Papers
- [ ] "FlashAttention-2" (Dao, 2023)
- [ ] "Mamba: Linear-Time Sequence Modeling" (Gu & Dao, 2023)
- [ ] "Efficient Streaming Language Models with Attention Sinks" (Xiao, 2023)

---

### ✅ Phase 2 Completion Checklist (6-MONTH MILESTONE)
- [ ] Can fine-tune LLMs with LoRA/QLoRA
- [ ] Can quantize models (GPTQ, AWQ, GGUF, FP8)
- [ ] Can deploy models on vLLM/TensorRT-LLM/NIM
- [ ] Can build production RAG systems (with NeMo Retriever)
- [ ] Can build AI agents with tools, memory, multi-agent coordination
- [ ] Can use NVIDIA NIM, NeMo, TensorRT-LLM, RAPIDS, Guardrails
- [ ] Know NGC Catalog, build.nvidia.com, AI Blueprints, Dynamo, Riva, ACE
- [ ] Understand RL fundamentals (policy, value, advantage, PPO)
- [ ] Can implement RLHF/DPO/GRPO alignment pipelines
- [ ] Can implement chain-of-thought and reasoning techniques
- [ ] Can read and understand most AI papers
- [ ] **Have 12+ impressive monthly capstones plus weekly builds in portfolio**

## 🔄 Buffer Week (Month 6 Revision) ⭐ 6-MONTH MILESTONE
- [ ] Revise RLHF pipeline: RL basics, PPO, GRPO, reward models
- [ ] Revise reasoning: CoT, ToT, self-consistency, STaR
- [ ] Revise long context: RoPE scaling, Flash Attention, Mamba/SSM
- [ ] Revise distillation, model merging, synthetic data
- [ ] **Build Monthly Project A:** AgentForge — Reasoning Agent Platform
- [ ] **Build Monthly Project B:** EvalArena — Model/Cost/Safety Battleground
- [ ] Push both projects to GitHub
- [ ] **Write a "6-month retrospective" blog post summarizing your journey**

---

# ═══════════════════════════════════════════════════
# PHASE 3: ADVANCED (Weeks 27-40, Months 7-10)
# Deep Systems, Research Implementation, Specialization
# ═══════════════════════════════════════════════════

---

## Week 27-28 : Vision-Language & Multi-Modal AI

### Topics
- [ ] CLIP: contrastive learning, image encoder + text encoder
- [ ] ViT: Vision Transformer, patch embedding
- [ ] LLaVA: visual instruction tuning (CLIP ViT + projection + LLM)
- [ ] Document understanding: OCR + layout-aware models
- [ ] Video understanding: frame sampling, temporal modeling
- [ ] NVIDIA DALI: GPU-accelerated image loading
- [ ] Audio: Whisper architecture, ASR + TTS

### Code Exercises
- [ ] Fine-tune CLIP on custom dataset
- [ ] Build simple vision-language model: CLIP + projection + small LLM
- [ ] Document Q&A with vision-language model
- [ ] Multi-modal chatbot with speech I/O

### 🔨 Saturday Projects
- [ ] **Week 27:** Multi-modal AI agent (images + documents + audio)
- [ ] **Week 28:** Document understanding pipeline with table/chart extraction

### 📄 Sunday Papers
- [ ] "Visual Instruction Tuning" (LLaVA, Liu 2023)
- [ ] "Learning Transferable Visual Models From Natural Language" (CLIP, Radford 2021)

---

## Week 28.5 : Diffusion Models & Generative AI Overview

> You don't need to be a diffusion expert for LLMs, but you need to understand how
> image/video generation works because multi-modal AI is converging and NVIDIA serves
> these models too (NIM for Stable Diffusion, NVIDIA Picasso).

### Topics
- [ ] **What are diffusion models**: gradually add noise to data, train to reverse the process
- [ ] Forward process: clean image → progressively noisier → pure Gaussian noise
- [ ] Reverse process: learn to denoise step by step → generate images from noise
- [ ] U-Net architecture: the neural network that predicts noise at each step
- [ ] Training objective: predict the noise that was added (simple MSE loss!)
- [ ] DDPM (Denoising Diffusion Probabilistic Models): the foundational paper
- [ ] DDIM: faster sampling (fewer steps needed)
- [ ] **Latent Diffusion (Stable Diffusion)**: run diffusion in compressed latent space (not pixel space)
  - [ ] VAE encoder: compress image to latent space
  - [ ] Diffusion in latent space: much cheaper than pixel space
  - [ ] VAE decoder: decompress latent back to image
- [ ] **Text conditioning**: CLIP text encoder provides the "prompt" that guides generation
- [ ] Classifier-free guidance: trade diversity for quality by scaling conditioning
- [ ] **How this connects to LLMs**:
  - [ ] Text encoder in Stable Diffusion IS a Transformer (CLIP/T5)
  - [ ] Modern image gen uses Transformer backbone instead of U-Net (DiT)
  - [ ] Multi-modal LLMs generate text about images; diffusion models generate images from text
  - [ ] NVIDIA NIM serves both LLMs and diffusion models
- [ ] Flux, DALL-E 3, Midjourney: current state of the art (know what exists)
- [ ] **Key difference from LLMs**: diffusion = continuous space, LLMs = discrete tokens

### Code Exercises
- [ ] Run Stable Diffusion inference using `diffusers` library
- [ ] Experiment with different prompts, guidance scales, number of steps
- [ ] Implement simplified DDPM from scratch on MNIST (generate digits)
- [ ] **Optional:** Fine-tune Stable Diffusion with LoRA on custom images (DreamBooth/textual inversion)

### 📄 Sunday Papers
- [ ] "Denoising Diffusion Probabilistic Models" (Ho et al., 2020)
- [ ] "High-Resolution Image Synthesis with Latent Diffusion Models" (Rombach et al., 2022) — Stable Diffusion paper

---

## Week 29-30 : MoE & Advanced Architectures

### Topics
- [ ] MoE implementation: expert parallelism, load balancing on GPU
- [ ] Router architectures: hash, learned, expert-choice
- [ ] DeepSeek-V3: Multi-head Latent Attention (MLA), DeepSeekMoE
- [ ] Auxiliary losses: load balancing, router z-loss
- [ ] Sparse vs dense MoE
- [ ] Communication patterns for expert routing

### Code Exercises
- [ ] Full MoE Transformer with expert parallelism
- [ ] Compare MoE vs dense: same total params, same active params
- [ ] Implement MLA from DeepSeek-V3

### 🔨 Saturday Projects
- [ ] **Week 29:** Train MoE model, profile expert routing
- [ ] **Week 30:** Reproduce DeepSeek-V3 architecture at small scale

### 📄 Sunday Papers
- [ ] "Mixtral of Experts" (Jiang, 2024)
- [ ] "DeepSeek-V3 Technical Report" (DeepSeek, 2024)

---

## Week 31-32 : Compiler & Kernel Optimization

### Topics
- [ ] Triton compiler internals
- [ ] torch.compile: TorchDynamo graph capture → TorchInductor code gen
- [ ] Operator fusion: pattern matching
- [ ] Kernel autotuning: systematic search for optimal configs
- [ ] CUDA cooperative groups
- [ ] Warp specialization (Hopper)
- [ ] Custom training kernels: fused optimizer, fused LayerNorm

### Code Exercises
- [ ] Write 5 optimized Triton kernels for LLM operations
- [ ] Fused AdamW optimizer in CUDA
- [ ] Communication optimization: gradient compression

### 🔨 Saturday Projects
- [ ] **Week 31:** Triton kernel library for all common LLM ops
- [ ] **Week 32:** Custom torch.compile backend for specific optimization

### 📄 Sunday Papers
- [ ] "TorchInductor: A PyTorch Native Compiler" (Meta blog)
- [ ] "Reducing Activation Recomputation in Large Transformer Models" (Korthikanti, 2022)

---

## Week 33-34 : Training Infrastructure

### Topics
- [ ] Megatron-LM: 3D parallelism (DP + TP + PP), sequence parallelism, context parallelism
- [ ] DeepSpeed: ZeRO stages, offloading, DeepSpeed-Chat
- [ ] PyTorch FSDP2: per-parameter sharding, DTensor
- [ ] Distributed checkpointing: async, elastic training
- [ ] Training monitoring: loss curves, gradient norms, layer-wise analysis
- [ ] Common training failures and diagnosis

### Code Exercises
- [ ] Configure Megatron-LM training (small scale)
- [ ] DeepSpeed ZeRO-3 for model larger than GPU memory
- [ ] FSDP2 training, compare with DeepSpeed
- [ ] Implement async distributed checkpointing

### 🔨 Saturday Projects
- [ ] **Week 33:** Complete distributed training pipeline with fault tolerance
- [ ] **Week 34:** Training monitoring dashboard with automated issue detection

### 📄 Sunday Papers
- [ ] "PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel" (Zhao, 2023)

---

## Week 35-36 : Evaluation & Benchmarking

### Topics
- [ ] lm-evaluation-harness, HELM, OpenCompass
- [ ] Benchmarks: MMLU, ARC, HellaSwag, WinoGrande, GSM8K, HumanEval, MATH
- [ ] LLM-as-judge: GPT-4/Claude evaluation
- [ ] MT-Bench, AlpacaEval, Chatbot Arena
- [ ] Safety: red-teaming, toxicity, bias
- [ ] Multi-modal evaluation: VQA, image captioning
- [ ] Agent evaluation: task completion, tool accuracy, efficiency
- [ ] Building custom domain-specific benchmarks

### Code Exercises
- [ ] Comprehensive eval suite for your trained models
- [ ] Custom domain-specific benchmark
- [ ] Automated evaluation pipeline with CI/CD

### 🔨 Saturday Projects
- [ ] **Week 35:** Complete evaluation framework (all benchmark types)
- [ ] **Week 36:** Domain-specific benchmark + leaderboard

### 📄 Sunday Papers
- [ ] "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena" (Zheng, 2023)
- [ ] "Holistic Evaluation of Language Models" (HELM, Liang 2022)

---

## Week 37-38 : Production Systems

### Topics
- [ ] Load balancing, auto-scaling for GPU inference
- [ ] Multi-tenant serving, isolation
- [ ] Cost optimization: model routing (small → large), semantic caching
- [ ] A/B testing model versions
- [ ] MLOps: model registries, CI/CD, monitoring, drift detection
- [ ] HuggingFace Hub, MLflow, DVC
- [ ] NVIDIA AI Enterprise: production-grade support, security, certified containers
- [ ] NVIDIA Base Command Manager: GPU cluster orchestration, job scheduling
- [ ] NVIDIA Fleet Command: edge deployment management
- [ ] NVIDIA Dynamo for multi-node serving in production

### Code Exercises
- [ ] Production serving with auto-scaling (NIM + Kubernetes)
- [ ] Multi-model routing system
- [ ] Semantic cache implementation
- [ ] Production monitoring with alerting
- [ ] Deploy with NVIDIA AI Enterprise stack (if available)

### 🔨 Saturday Projects
- [ ] **Week 37:** Production LLM system: NIM + Triton + auto-scaling + monitoring
- [ ] **Week 38:** Multi-model router: easy queries → small NIM, hard → large NIM

---

## Week 39-40 : Synthetic Data & Structured Generation

### Topics
- [ ] Data quality assessment: perplexity filtering, dedup, classifier quality
- [ ] Synthetic instruction generation: Self-Instruct, Evol-Instruct
- [ ] Synthetic preference data
- [ ] Data contamination detection
- [ ] NVIDIA NeMo Curator: GPU-accelerated curation (dedup, filter, PII, quality scoring)
- [ ] NVIDIA NeMo Data Curator containers on NGC
- [ ] Constrained decoding: GBNF, outlines
- [ ] Function calling training (Gorilla-style)
- [ ] Code generation: StarCoder, DeepSeek-Coder
- [ ] Text-to-SQL

### Code Exercises
- [ ] Complete data curation pipeline: scrape → clean → dedupe → filter → tokenize
- [ ] Train function-calling model from scratch
- [ ] Grammar-constrained generation with guaranteed JSON schema

### 🔨 Saturday Projects
- [ ] **Week 39:** Synthetic dataset generation + model training + evaluation
- [ ] **Week 40:** Structured output system with function calling

### 📄 Sunday Papers
- [ ] "Textbooks Are All You Need" (Phi-1, Gunasekar 2023)
- [ ] "Self-Instruct: Aligning LMs with Self-Generated Instructions" (Wang, 2022)

---

### ✅ Phase 3 Completion Checklist
- [ ] Can build multi-modal AI systems (vision + audio + text)
- [ ] Understand diffusion models and how they connect to LLMs
- [ ] Understand MoE architectures and can implement them
- [ ] Can write optimized Triton/CUDA kernels for LLM ops
- [ ] Can set up distributed training (Megatron, DeepSpeed, FSDP)
- [ ] Can evaluate models comprehensively
- [ ] Can build production LLM systems with MLOps
- [ ] Can generate and curate synthetic training data

---

# ═══════════════════════════════════════════════════
# PHASE 4: EXPERT (Weeks 41-52, Months 11-13)
# Research, Test-Time Compute, Enterprise Agents
# ═══════════════════════════════════════════════════

---

## Week 41-42 : Reading & Implementing Papers

### How to Read AI Papers
- [ ] Master the technique: Title/Abstract → Figures/Tables → Intro → Method → Experiments → Related Work
- [ ] Implement key components from 3 recent papers
- [ ] Reproduce main results of 1 paper end-to-end

### Paper Sources
- [ ] Set up arxiv alerts for cs.CL, cs.LG, cs.AI
- [ ] Follow top ML researchers on Twitter/X
- [ ] Bookmark: Papers With Code, NeurIPS/ICML/ICLR/ACL proceedings

### 🔨 Saturday Projects
- [ ] **Week 41:** Implement 3 paper algorithms
- [ ] **Week 42:** Full paper reproduction with blog post

---

## Week 43-44 : Test-Time Compute & Reasoning Scaling

### Topics
- [ ] Test-time compute scaling: more inference compute → better answers
- [ ] Search at inference: beam search, MCTS for reasoning
- [ ] Verifier models, process reward models
- [ ] Self-refinement, consensus methods
- [ ] Compute-optimal inference: adapt compute to difficulty
- [ ] Speculative decoding for reasoning workloads

### Code Exercises
- [ ] MCTS-based reasoning for math problems
- [ ] Adaptive compute: easy = 1 shot, hard = 64+
- [ ] Process reward model scoring

### 🔨 Saturday Projects
- [ ] **Week 43:** MCTS reasoning agent for GSM8K/MATH
- [ ] **Week 44:** Adaptive compute system with quality-aware routing

### 📄 Sunday Papers
- [ ] "Scaling LLM Test-Time Compute Optimally" (Snell, 2024)
- [ ] "Let's Verify Step by Step" (Lightman, 2023)
- [ ] "Tree of Thoughts" (Yao, 2023)

---

## Week 45-46 : Enterprise AI Agents with NVIDIA

### Topics
- [ ] NVIDIA ACE (Avatar Cloud Engine): AI-powered digital humans
- [ ] NVIDIA AIQ / Agent Intelligence Toolkit: toolkit for building enterprise AI agents
- [ ] NVIDIA NeMo Guardrails: advanced Colang 2.0, custom actions, KB integration
- [ ] NVIDIA NIM Agent Blueprints: pre-built agentic AI workflows
- [ ] Model Context Protocol (MCP): standardized tool interfaces for agents
- [ ] NVIDIA Tokkio: customer service AI agent platform
- [ ] Multi-agent coordination: hierarchical, consensus, debate, marketplace
- [ ] Agent memory: episodic, semantic, procedural systems
- [ ] Agent fine-tuning: training for agentic tasks (NeMo Customizer)
- [ ] Tool-use training data generation

### Code Exercises
- [ ] Enterprise agent: NIM (LLM + embed + rerank) + Guardrails + RAG
- [ ] MCP server + client implementation
- [ ] Agent with episodic + semantic memory
- [ ] Deploy using NVIDIA NIM Agent Blueprint

### 🔨 Saturday Projects
- [ ] **Week 45:** Enterprise AI agent with full NVIDIA stack (NIM + Guardrails + AIQ + Retriever)
- [ ] **Week 46:** Multi-agent software engineering system (SWE-bench style)

### 📄 Sunday Papers
- [ ] "Gorilla: Large Language Model Connected with Massive APIs" (Patil, 2023)
- [ ] "AgentTuning: Enabling Generalized Agent Abilities For LLMs" (Zeng, 2023)

---

## Week 47-48 : Advanced Fine-Tuning & Alignment

### Topics
- [ ] SPIN: self-play fine-tuning
- [ ] ORPO: odds ratio preference optimization
- [ ] SimPO: without reference model
- [ ] Iterative DPO: generate → retrain cycles
- [ ] Online vs offline RL comparison
- [ ] Reward hacking detection and prevention
- [ ] Advanced alignment: SteerLM, attribute-conditioned generation

### Code Exercises
- [ ] Implement ORPO, SimPO, compare with DPO
- [ ] Iterative DPO pipeline
- [ ] Reward hacking detection system

### 🔨 Saturday Projects
- [ ] **Week 47:** Compare 5 alignment methods on same base model
- [ ] **Week 48:** Self-improving model pipeline (generate → evaluate → retrain)

---

## Week 49-50 : Advanced RAG & Information Retrieval

### Topics
- [ ] ColBERT, ColBERTv2: late interaction models
- [ ] SPLADE: learned sparse retrieval
- [ ] Cross-encoder re-ranking at scale
- [ ] RAG fusion, corrective RAG
- [ ] Multi-hop RAG: questions requiring multiple retrieval steps
- [ ] RAPTOR: recursive abstractive processing for tree-organized retrieval
- [ ] Structured data RAG: SQL generation, table retrieval
- [ ] RAG + tool use combination

### Code Exercises
- [ ] ColBERT-based retrieval system
- [ ] Multi-hop reasoning RAG
- [ ] RAPTOR-style hierarchical retrieval

### 🔨 Saturday Projects
- [ ] **Week 49:** State-of-the-art RAG with ColBERT + re-ranking + multi-hop
- [ ] **Week 50:** Complex QA system requiring multi-hop reasoning

### 📄 Sunday Papers
- [ ] "ColBERT: Efficient Passage Search via Late Interaction" (Khattab, 2020)
- [ ] "RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval" (Sarthi, 2024)

---

## Week 51-52 : Phase 4 Capstone

### Topics
- [ ] Review everything from Phase 4
- [ ] Identify personal strengths and remaining gaps
- [ ] Plan capstone project

### 🔨 Saturday Projects
- [ ] **Week 51:** Integration project combining multiple Phase 4 skills
- [ ] **Week 52:** Open-source contribution to vLLM, TensorRT-LLM, or HuggingFace

---

### ✅ Phase 4 Completion Checklist
- [ ] Can read, understand, and implement AI papers
- [ ] Can build test-time compute systems (MCTS, process reward)
- [ ] Can build enterprise-grade AI agents with NVIDIA tools
- [ ] Can implement advanced alignment techniques
- [ ] Can build state-of-the-art RAG systems
- [ ] Have contributed to open-source AI projects

---

# ═══════════════════════════════════════════════════
# PHASE 5: MASTERY (Weeks 53-65, Months 14-16)
# Deep Specialization & Innovation
# ═══════════════════════════════════════════════════

---

## Week 53-56 : Capstone Project

### Choose ONE:

**Option A: Full-Stack LLM Platform**
- [ ] Custom training pipeline
- [ ] Multiple fine-tuned models
- [ ] RAG with advanced retrieval
- [ ] AI agent with tools
- [ ] Production deployment with monitoring
- [ ] NVIDIA NIM + TensorRT-LLM backend

**Option B: AI Agent Framework**
- [ ] Custom agent runtime
- [ ] Multi-agent coordination
- [ ] Tool integration framework
- [ ] Memory systems (short + long term)
- [ ] Safety guardrails + evaluation

**Option C: Efficient Inference Engine**
- [ ] Custom CUDA kernels for attention, GEMM
- [ ] Quantization pipeline (FP8, INT4)
- [ ] Speculative decoding
- [ ] Continuous batching with PagedAttention
- [ ] Benchmark against vLLM/TensorRT-LLM

**Option D: Domain-Specific LLM**
- [ ] Data curation for specific domain
- [ ] Pre-training or continued pre-training
- [ ] Domain fine-tuning + RAG
- [ ] Domain evaluation benchmarks

### Milestones
- [ ] Week 53: Architecture design + setup
- [ ] Week 54: Core implementation
- [ ] Week 55: Optimization + testing
- [ ] Week 56: Polish + documentation

---

## Week 57-60 : Cutting Edge Research

### Topics (explore based on latest papers)
- [ ] Latest architecture innovations (check arxiv)
- [ ] New training techniques
- [ ] Efficiency breakthroughs
- [ ] Agent framework developments
- [ ] Multi-modal advances
- [ ] Implement 2-3 cutting-edge techniques

### 🔨 Saturday Projects
- [ ] **Week 57-58:** Implement latest breakthrough paper
- [ ] **Week 59-60:** Build novel combination of techniques

---

## Week 61-65 : Open Source & Community

### Activities
- [ ] Contribute meaningful PRs to 2-3 open-source projects
- [ ] Write 3-5 technical blog posts
- [ ] Create educational content (tutorials, videos)
- [ ] Mentor others starting their journey
- [ ] Build and share tools with the community

### 🔨 Saturday Projects
- [ ] **Week 61-62:** Major open-source contribution
- [ ] **Week 63-64:** Technical blog series
- [ ] **Week 65:** Community presentation or workshop

---

# ═══════════════════════════════════════════════════
# PHASE 6: CAPSTONE & PORTFOLIO (Weeks 66-78)
# Months 17-18: Final Projects & Career Readiness
# ═══════════════════════════════════════════════════

---

## Week 66-72: Second Capstone + Portfolio

- [ ] Build second capstone project (different option from first)
- [ ] Create GitHub portfolio showcasing all projects
- [ ] Write comprehensive README for each project
- [ ] Record demo videos for top projects
- [ ] Update resume with specific skills

## Week 73-78: Staying Current & Mastery

- [ ] Follow latest research (2-3 papers/week)
- [ ] Participate in ML competitions
- [ ] Network with AI community
- [ ] Plan next learning goals
- [ ] Consider writing your own research paper

---

# ═══════════════════════════════════════════════════
# APPENDIX: ESSENTIAL RESOURCES
# ═══════════════════════════════════════════════════

## Books
- [ ] "Programming Massively Parallel Processors" (Kirk & Hwu)
- [ ] "Deep Learning" (Goodfellow, Bengio, Courville)
- [ ] "Designing Machine Learning Systems" (Huyen)

## Online Courses
- [ ] Karpathy "Neural Networks: Zero to Hero"
- [ ] Karpathy "Let's build GPT"
- [ ] Stanford CS224n — NLP with Deep Learning
- [ ] Stanford CS336 — Language Modeling from Scratch
- [ ] NVIDIA DLI: "Fundamentals of Accelerated Computing with CUDA"
- [ ] NVIDIA DLI: "Building RAG Agents with LLMs"
- [ ] NVIDIA DLI: "Generative AI with Diffusion Models"
- [ ] NVIDIA DLI: "Accelerating CUDA C++ Applications with Nsight Systems"

## GitHub Repos to Study
- [ ] nanoGPT (Karpathy)
- [ ] llama.cpp
- [ ] vLLM
- [ ] TensorRT-LLM (NVIDIA)
- [ ] Megatron-LM (NVIDIA)
- [ ] NeMo (NVIDIA)
- [ ] NeMo-Guardrails (NVIDIA)
- [ ] NeMo-Curator (NVIDIA)
- [ ] NVIDIA/GenerativeAIExamples (NVIDIA RAG & agent examples)
- [ ] NVIDIA/cuda-python
- [ ] NVIDIA/cutlass
- [ ] NVIDIA/Dynamo
- [ ] NVIDIA/NVFlare (federated learning)
- [ ] DeepSpeed (Microsoft)
- [ ] transformers (HuggingFace)
- [ ] trl (HuggingFace)
- [ ] PEFT (HuggingFace)
- [ ] flash-attention (Dao-AILab)

## Master Reading List (38 Papers)
- [ ] 1. "Attention Is All You Need" (Vaswani, 2017)
- [ ] 2. "BERT" (Devlin, 2018)
- [ ] 3. "GPT-2" (Radford, 2019)
- [ ] 4. "GPT-3: Language Models are Few-Shot Learners" (Brown, 2020)
- [ ] 5. "Chinchilla: Training Compute-Optimal LLMs" (Hoffmann, 2022)
- [ ] 6. "LLaMA" (Touvron, 2023)
- [ ] 7. "Llama 2" (Touvron, 2023)
- [ ] 8. "Mistral 7B" (Jiang, 2023)
- [ ] 9. "Mixtral of Experts" (Jiang, 2024)
- [ ] 10. "DeepSeek-V3 Technical Report" (2024)
- [ ] 11. "InstructGPT / RLHF" (Ouyang, 2022)
- [ ] 12. "DPO: Direct Preference Optimization" (Rafailov, 2023)
- [ ] 13. "Constitutional AI" (Bai, 2022)
- [ ] 14. "DeepSeek-R1" (2025)
- [ ] 15. "FlashAttention" (Dao, 2022)
- [ ] 16. "FlashAttention-2" (Dao, 2023)
- [ ] 17. "LoRA" (Hu, 2021)
- [ ] 18. "QLoRA" (Dettmers, 2023)
- [ ] 19. "GPTQ" (Frantar, 2022)
- [ ] 20. "AWQ" (Lin, 2023)
- [ ] 21. "RAG" (Lewis, 2020)
- [ ] 22. "ReAct" (Yao, 2022)
- [ ] 23. "Toolformer" (Schick, 2023)
- [ ] 24. "Voyager" (Wang, 2023)
- [ ] 25. "Chain-of-Thought Prompting" (Wei, 2022)
- [ ] 26. "Tree of Thoughts" (Yao, 2023)
- [ ] 27. "Let's Verify Step by Step" (Lightman, 2023)
- [ ] 28. "CLIP" (Radford, 2021)
- [ ] 29. "LLaVA" (Liu, 2023)
- [ ] 30. "Whisper" (Radford, 2022)
- [ ] 31. "Megatron-LM" (Shoeybi, 2019)
- [ ] 32. "ZeRO" (Rajbhandari, 2019)
- [ ] 33. "vLLM / PagedAttention" (Kwon, 2023)
- [ ] 34. "Triton" (Tillet, 2019)
- [ ] 35. "Denoising Diffusion Probabilistic Models" (Ho, 2020)
- [ ] 36. "High-Resolution Image Synthesis with Latent Diffusion Models" (Rombach, 2022) — Stable Diffusion
- [ ] 37. "An Image is Worth 16x16 Words: Transformers for Image Recognition" (ViT, Dosovitskiy, 2020)
- [ ] 38. "Deep Residual Learning for Image Recognition" (ResNet, He, 2015)

---

## Hardware Knowledge Checklist
- [ ] SM internals (CUDA cores, Tensor Cores, SFUs, warp schedulers)
- [ ] Warp execution (SIMT, divergence, scheduling)
- [ ] Register file and register pressure
- [ ] Shared memory banks and conflict avoidance
- [ ] L1/L2 cache hierarchy
- [ ] Memory coalescing requirements
- [ ] HBM2/HBM3 architecture and bandwidth
- [ ] PCIe vs NVLink vs NVSwitch
- [ ] FP32/FP16/BF16/TF32/FP8/INT8/INT4 formats
- [ ] Tensor Core operation and supported formats
- [ ] DGX system topology (DGX A100, H100, B200)
- [ ] HGX baseboard design
- [ ] NCCL collective operations
- [ ] InfiniBand / ConnectX / BlueField DPU basics
- [ ] Roofline model analysis
- [ ] Nsight Systems timeline analysis
- [ ] Nsight Compute kernel analysis

---

## NVIDIA AI Tools & Ecosystem Checklist

### Inference & Serving
- [ ] **NIM** (NVIDIA Inference Microservices) — deploy optimized LLM/VLM/embedding containers
- [ ] **TensorRT-LLM** — build optimized LLM engines for inference
- [ ] **TensorRT** — general model optimization (ONNX → engine)
- [ ] **Triton Inference Server** — production model serving (batching, ensemble, multi-model)
- [ ] **NVIDIA Dynamo** — open-source multi-node inference (disaggregated prefill/decode)
- [ ] **Model Analyzer** — profiling Triton model configs for optimal throughput/latency

### Training & Fine-Tuning
- [ ] **NeMo Framework** — pre-training, SFT, PEFT, DPO, RLHF
- [ ] **NeMo Customizer** — API for fine-tuning NIM models
- [ ] **Megatron-LM** — distributed training (TP, PP, DP, SP, CP)
- [ ] **Transformer Engine** — FP8 training on Hopper GPUs
- [ ] **NVIDIA DALI** — GPU-accelerated data loading
- [ ] **NCCL** — GPU collective communication library

### Data & Retrieval
- [ ] **NeMo Curator** — GPU-accelerated data curation (dedup, filter, PII, quality)
- [ ] **NeMo Retriever** — embedding + reranking microservices
- [ ] **cuVS / CAGRA** — GPU-accelerated vector search
- [ ] **RAPIDS** (cuDF, cuML, cuGraph) — GPU DataFrames, ML, graph analytics

### Safety & Guardrails
- [ ] **NeMo Guardrails** — Colang 2.0, topical/safety/fact-checking rails
- [ ] **NV-Shield** — content safety classification (if available)

### Agents & Applications
- [ ] **NVIDIA AIQ / Agent Intelligence Toolkit** — enterprise AI agent toolkit, formerly AgentIQ
- [ ] **NIM Agent Blueprints** — pre-built agentic AI workflows
- [ ] **NVIDIA AI Blueprints** — reference architectures (RAG, digital human, etc.)
- [ ] **NVIDIA Tokkio** — customer service AI agent platform

### Speech & Avatar
- [ ] **Riva** — ASR + TTS microservices on GPU
- [ ] **ACE** (Avatar Cloud Engine) — AI-powered digital humans
- [ ] **Audio2Face** — lip-sync from audio

### Platforms & Infrastructure
- [ ] **NGC Catalog** — GPU-optimized containers, models, Helm charts
- [ ] **build.nvidia.com** — API playground for NVIDIA-hosted models
- [ ] **NVIDIA AI Enterprise** — production platform (support, security, certification)
- [ ] **AI Workbench** — unified AI development environment
- [ ] **Base Command Manager** — GPU cluster orchestration
- [ ] **Fleet Command** — edge AI deployment management
- [ ] **DGX Cloud** — cloud-hosted DGX systems

### Profiling & Development
- [ ] **Nsight Systems** — system-wide CPU+GPU profiling
- [ ] **Nsight Compute** — kernel-level GPU profiling
- [ ] **cuda-python** — Python bindings for CUDA driver/runtime
- [ ] **CUTLASS** — CUDA templates for GEMM and linear algebra
- [ ] **cuBLAS** — GPU-accelerated BLAS
- [ ] **cuDNN** — GPU-accelerated DL primitives

### Other
- [ ] **Morpheus** — AI-powered cybersecurity
- [ ] **cuOpt** — GPU-accelerated optimization solver
- [ ] **NVFlare** — federated learning framework
- [ ] **Omniverse** — simulation platform (Isaac Sim for robotics)

---

*Current week: Week 2*
*Weeks completed: Week 1 ✅*
*Total items completed: ~35 / ~1270*
*Monthly projects completed: 0 / 34 (17 Project A + 17 Project B)*
