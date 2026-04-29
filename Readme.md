# LLM × CUDA × Hardware × AI Silicon — Full-Stack Mastery Roadmap

> **Daily: 2-3 hours (LLM/CUDA track Mon–Fri)** | **Day 6 (Sat morning, 1.5-2 hrs): Silicon track from Week 3** | **Saturday afternoon: 3-4 hrs main project** | **Sunday: 1-2 hrs papers (optional)**
> **Total duration: ~36 months (156 weeks)** split into two stages:
>   – **Stage 1 — LLM core + interleaved silicon (Months 1-24, Weeks 1-104):** the entire original LLM roadmap, eased from 18 → 24 months so nothing is rushed, with chip-design Day 6 woven in from Week 3 as the silicon-level explanation of every LLM/CUDA topic
>   – **Stage 2 — Silicon mastery (Months 25-36, Weeks 105-156):** full-time chip design — advanced microarchitecture, physical design (RTL→GDS in OpenLane on Sky130), TinyTapeout/Efabless real-silicon tape-out, and your final **Silicon Magnum Opus**: a custom transformer/LLM inference accelerator
> **HARD CONSTRAINT — LLM Magnum Opus deadline:** completed inside Month 24 (will not slip past 2 years)
> **HARD CONSTRAINT — Final Silicon Magnum Opus:** completed inside Month 36 (~3 years)
> **6-month milestone:** strong LLM/GPU fundamentals + can write synthesizable Verilog (HDLBits done) + understand what a Tensor Core *is* in silicon
> **12-month milestone:** production-grade LLM systems + 5-stage pipelined RV32I CPU on FPGA with hazard detection + first MAC unit RTL
> **18-month milestone:** advanced LLM (papers, MoE, training infra) + working 4×4 INT8 systolic array on FPGA + first formal verification proofs
> **24-month milestone:** **LLM Magnum Opus shipped** + 8×8 / 16×16 INT8 systolic array + first full OpenLane Sky130 RTL→GDSII run on a small AI block
> **30-month milestone:** transformer block accelerator on FPGA (attention + FFN + LayerNorm in RTL), TinyTapeout submission
> **36-month milestone (FINAL):** **Silicon Magnum Opus** — a custom AI inference accelerator with paper, open-source RTL, FPGA demo, and (optionally) physical silicon back from a shuttle
> **After each month:** 1 week revision buffer to revisit weak areas + complete monthly projects
> **Projects:** 1 weekly project + **3 monthly projects** (Project A = LLM core, Project B = LLM novelty/tool, **Project C = Silicon counterpart that hardware-explains the same month's LLM work**)

---

## North Star

Become the rare engineer who can reason across **every layer of the AI stack — from transformer math down to transistors — as one continuous chain of cause and effect**, and ship at every layer.

```text
LLM behavior -> Transformer math -> PyTorch ops -> CUDA/Triton kernels
-> GPU memory & SM microarchitecture -> tensor-core PTX (WGMMA/tcgen05)
-> AI accelerator microarchitecture (systolic arrays, dataflow, NoC)
-> RTL/SystemVerilog -> functional & formal verification
-> physical design (synthesis, P&R, STA, signoff) -> GDSII -> silicon
```

This is **not two careers stacked on top of each other**. It is **one career that goes one layer deeper than anyone else**. Every week's chip-design content (the **Day 6 — Silicon block**) is the **physical-level explanation of the LLM/CUDA topic you just covered that week**. They are the same idea seen from different abstraction layers:

| LLM/CUDA topic (existing weekly track) | Silicon topic (Day 6 — same week) | Why they are the same thing |
|---|---|---|
| GPU vs CPU, SMs, warps (Week 1) | Transistors → CMOS gates → NAND/NOR (Week 3 Day 6) | An SM is built bottom-up from billions of these gates |
| CUDA memory coalescing (Week 2) | SRAM 6T cell, DRAM 1T1C cell, row buffers (Week 3-4 Day 6) | Coalescing exists *because* of how SRAM banks/DRAM rows are wired |
| Warp execution & divergence (Week 3) | Combinational logic, FSMs, control unit design (Week 5 Day 6) | A warp scheduler IS an FSM with priority logic |
| Reduction, atomics (Week 3) | Adders (ripple/CLA/Kogge-Stone), multipliers (Booth/Wallace) (Week 5-6 Day 6) | Every reduction is silicon adders chained together |
| cuBLAS GEMM, Tensor Cores WMMA (Week 6) | Systolic MAC array, mixed-precision PE (Week 6-7 Day 6) | A Tensor Core literally **is** a small systolic array on silicon |
| Nsight profiling, critical path (Week 7) | STA: setup/hold, slack, clock skew (Week 7 Day 6) | "Why is my kernel slow?" → "Why is my critical path long?" — same question |
| LLaMA architecture, RMSNorm (Week 7) | Reciprocal & sqrt circuits, FP MAC fusion (Week 8 Day 6) | RMSNorm needs hardware that does these in one shot |
| HBM bandwidth, GPU memory deep dive (Week 8) | HBM3 PHY, stacked-die, IR drop, memory controller (Week 8 Day 6) | The "memory wall" is a physical wire and stacking problem |
| Distributed CUDA, NCCL (Week 9) | NoC topologies (mesh/ring), NVSwitch silicon, UCIe die-to-die (Week 9 Day 6) | All-reduce is a routing problem that lives on metal traces |
| Custom CUDA extensions (Week 10) | First synthesizable Verilog: ALU, FIFO, UART (Week 10 Day 6) | Your CUDA op eventually becomes a sequence of cycles on these primitives |
| Flash Attention (Week 11) | On-chip SRAM tiling + dataflow architectures (Eyeriss) (Week 11 Day 6) | Flash Attention is software *imitating* what dataflow accelerators do natively |
| Tokenization, KV-cache (Week 12-13) | Scratchpad + DMA + paged-buffer RTL (Week 12-13 Day 6) | KV-cache management *is* a silicon memory-controller problem |
| LoRA / QLoRA (Week 13-14) | Low-rank MAC datapath, weight streaming (Day 6) | LoRA's compute pattern shapes the *ideal* MAC unit |
| Quantization GPTQ/AWQ/FP8 (Week 14) | Mixed-precision MAC, dequant-on-the-fly hardware (Day 6) | Software quant is useless without matching silicon |
| TensorRT-LLM, vLLM (Week 15) | In-flight batching scheduler in RTL, paged KV controller (Day 6) | vLLM's PagedAttention is a hardware allocator pattern |
| RAG, vector search (Week 16) | CAM/TCAM, approximate-NN hardware (Day 6) | "Find nearest" was a silicon problem before it was an LLM problem |
| MoE expert routing (Week 29-30) | All-to-all NoC patterns, multicast hardware (Day 6) | MoE routing efficiency = NoC bandwidth efficiency |
| Triton compiler internals (Week 31-32) | LLVM/MLIR backend for accelerators, custom dialects (Day 6) | A custom chip needs a custom compiler — they co-evolve |
| Megatron 3D parallelism (Week 33-34) | NVLink/NVSwitch silicon, multi-die scaling (Day 6) | Tensor parallelism only works because of fast die-to-die links |
| FlashAttention-3, Hopper WGMMA (Phase 4) | Warp specialization in silicon, TMA hardware unit (Day 6) | FA-3 only exists because Hopper added these silicon units |
| Test-time compute (Phase 4) | Inference-only ASIC architecture (Etched Sohu style) (Day 6) | Specialized inference silicon makes TTC economically feasible |

After this roadmap, you should not merely "know LLMs" or "know chips." You should be able to:

- **LLM side (by Month 24):** build models, training loops, inference systems, RAG, agents, and GPU kernels from first principles; explain performance using hardware facts; read new papers and turn them into working code; ship the **LLM Magnum Opus** — a portfolio centerpiece that looks like a real open-source product
- **Silicon side (by Month 36):** write synthesizable Verilog/SystemVerilog; design and verify a pipelined RISC-V CPU; design an AI accelerator (systolic array / dataflow) from architecture spec → RTL → FPGA → ASIC layout in OpenLane on Sky130 → optional TinyTapeout/Efabless tape-out; understand modern NVIDIA SM internals down to PTX (WGMMA, TMA, tcgen05, FP4/FP6 on Blackwell); read and reproduce ideas from ISSCC/ISCA/HotChips papers
- **Co-design (the rare layer):** look at a transformer block and reason about *what hardware would make this 10× faster* — and prove it with measured benchmarks on **real silicon you can hold in your hand**, not slides

The journey is deliberately hard. Three years of compounding study, daily projects, weekend builds, and eventually real silicon. The reward is rare judgment: you will understand **what the model is doing**, **why the GPU/system behaves the way it does**, and **how the silicon underneath was designed and could be designed better.** This is the engineer NVIDIA, Etched, Cerebras, Groq, Tenstorrent, AMD, and Google TPU teams hire as a senior architect — not as an entry-level rotation.

---

## How To Use This File

- **Check off items** as you complete them: change `[ ]` to `[x]`
- Each **weekday Day 1-5 = 2-3 hours** of focused LLM/CUDA work (alternating CUDA/Hardware and ML/DL/LLM as before — nothing removed)
- **Day 6 (Saturday morning, 1.5-2 hrs) = Silicon Day** — the chip-design topic that **complements** that week's LLM/CUDA learning. **Starts Week 3** (after you finish Week 2). It is the *same lesson seen one layer deeper* — never disconnected from what you just learned.
- **Saturday afternoon (3-4 hrs) = main weekly project** (LLM-focused, as before). On selected weeks, the silicon Day 6 grows into a fuller silicon mini-project (clearly marked `🔧 Silicon Saturday`)
- **Sunday = rest + listed papers.** Some weeks list a Silicon Sunday paper alongside the LLM paper — both are 30-45 min reads, optional but recommended.
- If a day takes longer, split it across 2 days — no rush. The plan is eased to 24 months for LLM precisely so you don't need to rush.
- **After every month:** take 1 extra week to revise everything + complete monthly projects (now THREE per month)
- **Monthly projects (THREE per month from Month 2 onwards):**
  - **Project A** — deep LLM implementation of the main technical theme
  - **Project B** — useful LLM tool/product that packages the learning for other people
  - **Project C — Silicon counterpart** — the same month's idea expressed in RTL or hardware: small, focused, and *visibly tied to that month's LLM work* (e.g., when LLM month is "Tensor Cores," Project C is "design a 4×4 INT8 MAC array in Verilog and run it on FPGA"). Project C is short the first few months and grows over time as your HDL skill grows.
- **Use AI (Cursor/Codex/Claude)** as your tutor, code reviewer, paper explainer, RTL reviewer, and ruthless critic
- Keep a short engineering journal: what you built, what broke, what you measured, what you would change
- For every serious project (A, B, *and* C), write a public-quality README and at least one technical note or blog post
- **Hardware budget:** you'll need an FPGA dev board around Month 6-8 (recommended: Digilent Arty A7-100T ~$220 or Nexys A7 ~$320 for Xilinx; or Tang Nano 20K ~$30 for entry-level Gowin). Around Month 22-24 you'll optionally pay ~$300 for a TinyTapeout submission (real silicon back). All EDA software is free/open (Yosys, OpenLane, Verilator, Vivado WebPACK, Icarus). Total hardware cost across 3 years: **$300-700**.

---

## Project Philosophy: Novel + Useful + Cumulative + Silicon-Grounded

Every month has **three** projects because expertise now needs three muscles:

- **Project A:** deep LLM implementation of the main technical theme
- **Project B:** a useful LLM tool/product that packages the learning for other people
- **Project C:** the **silicon counterpart** — the same month's idea expressed in HDL/RTL/FPGA, grounding software intuition in real hardware

For Month `N`, all three projects must use Month `1...N` skills. The project should have visible evidence of that cumulative learning:

- **Month 1+ hardware proof:** benchmarks, roofline thinking, memory/compute bottleneck analysis
- **Month 2+ model proof:** Transformer/LLM internals, training/inference behavior, profiling
- **Month 3+ silicon-foundations proof:** digital logic / first synthesizable Verilog (Project C tier 1)
- **Month 3+ optimization proof:** custom CUDA/Triton extension, Flash Attention idea, distributed/fine-tuning awareness
- **Month 4+ compression proof:** quantization/PEFT/cost-vs-quality tradeoff
- **Month 5+ product proof:** serving, RAG, latency, evaluation, deployment ergonomics
- **Month 6+ reasoning proof:** tool use, memory, observability, safety, RLHF/reasoning evaluation
- **Month 6+ HDL proof:** MAC unit / register file / FSM in synthesizable Verilog with testbench (Project C tier 2)
- **Month 7+ multimodal proof:** text/image/document/audio routing where useful
- **Month 9+ kernel proof:** measurable custom kernel or compiler/fusion insight
- **Month 9+ CPU proof:** working single-cycle then 5-stage pipelined RISC-V on FPGA (Project C tier 3)
- **Month 10+ scale proof:** distributed, multi-GPU, failure diagnosis, cost model
- **Month 11+ research proof:** paper-to-code implementation or reproducibility analysis
- **Month 12+ accelerator proof:** small INT8 systolic array (4×4 → 8×8) on FPGA (Project C tier 4)
- **Month 13+ production proof:** monitoring, routing, guardrails, API design, cost tracking
- **Month 16+ community proof:** external users, docs, issues, PRs, adoption
- **Month 18+ verification proof:** UVM testbench + functional coverage on a non-trivial RTL block (Project C tier 5)
- **Month 22+ physical design proof:** first OpenLane Sky130 RTL→GDSII run on a small AI block, with timing/area/power reports
- **Month 24+ LLM Magnum Opus shipped:** the LLM-side capstone is *done* — full repo, paper, demo
- **Month 27+ LLM-aware accelerator proof:** transformer block accelerator on FPGA (attention + FFN + LN in RTL)
- **Month 30+ tape-out proof:** TinyTapeout submission accepted (or Efabless chipIgnite), waiting for silicon
- **Month 36+ Silicon Magnum Opus:** custom AI inference accelerator — RTL repo + FPGA demo + GDSII + paper + (optionally) physical chip back from a shuttle

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

Use these gates to decide whether you are truly leveling up. Each gate has an **LLM side** and a **Silicon side** — both must hold.

| Point | LLM/CUDA side | Silicon side (one layer deeper) |
|-------|---------------|----------------------------------|
| End of Month 1 | Explain why a CUDA kernel is slow using memory access, occupancy, and bandwidth evidence | (Silicon track has not started yet — Week 3 onwards) |
| End of Month 2 | Build a clean Transformer + GPT-2 from scratch, profile it with Nsight | Comfortable with binary/hex, Boolean algebra, K-maps, half/full adders, basic FSMs (Harris & Harris Ch 1-3) |
| End of Month 4 | Build and fine-tune a small Transformer, replace one slow op with custom CUDA/Triton | Pass HDLBits modules 1-50 (combinational + sequential Verilog) and explain setup/hold violation in your own words |
| End of Month 6 | Ship an end-to-end AI application with serving, RAG/agents, reasoning, safety, observability | Build a single-cycle RV32I CPU in Verilog (works in simulation); explain what a Tensor Core *is* in silicon |
| End of Month 9 | Comfortable with MoE, Triton compiler internals, distributed training basics | 5-stage pipelined RISC-V on FPGA with hazard detection + forwarding; first MAC unit and 4×4 INT8 GEMM |
| End of Month 12 | Reason about multi-GPU training/inference systems, cost, throughput, bottlenecks | Working 8×8 INT8 systolic array on FPGA; running a tiny matmul from PyTorch via custom AXI driver |
| End of Month 15 | Design production LLM infrastructure with routing, caching, monitoring, alignment, and guardrails | UVM testbench with constrained-random + functional coverage on your systolic array; first formal SVA |
| End of Month 18 | Read state-of-the-art papers, implement them, reproduce results | Run Yosys synthesis + OpenSTA timing on your design; understand floorplanning and cell placement |
| End of Month 21 | Lead a serious LLM systems project from idea to benchmarked, documented, usable release | First full RTL→GDSII run in OpenLane on Sky130 for a small AI block (e.g., 4×4 INT8 GEMM); report area/power/timing |
| **End of Month 24 — LLM MAGNUM OPUS SHIPPED** | LLM Magnum Opus complete: domain-specific LLM trained, fine-tuned, quantized, deployed, with custom CUDA kernels and full evaluation | Submitted (or ready to submit) a TinyTapeout block on Sky130 that reflects something you learned in the LLM stack |
| End of Month 27 | (LLM track done — only refresh + papers) | Transformer-block accelerator on FPGA: attention + FFN + LayerNorm in RTL, mapped to a real LLaMA layer's compute budget |
| End of Month 30 | (LLM track done — only refresh + papers) | First TinyTapeout/Efabless submission accepted; running OpenLane on a non-trivial multi-block design |
| End of Month 33 | (LLM track done — only refresh + papers) | Full transformer inference accelerator (RTL + FPGA demo) running quantized LLaMA-style block end-to-end at measurable tokens/sec |
| **End of Month 36 — SILICON MAGNUM OPUS SHIPPED** | Speak about your work fluently across all 5 stack layers in a senior-architect interview | Silicon Magnum Opus complete: custom AI inference accelerator — RTL repo + FPGA demo + GDSII + paper + (optionally) physical chip back from a shuttle |

If a gate feels weak, pause and build one more small project before moving on. Speed is less important than compounding correctly. The 24-month LLM deadline is firm; the 36-month silicon deadline is firm. Buffer weeks exist precisely to keep both promises.

---

## Progress Overview

The plan is now eight phases across **~156 weeks (≈ 36 months)**. Phases 1-6 are the original LLM/CUDA roadmap, eased from 18 → 24 months and silicon-infused via Day 6 + Project C. Phases 7-8 are the silicon-mastery stage that builds on top of the now-complete LLM Magnum Opus.

| Phase | Weeks | Months | Track focus | Status |
|-------|-------|--------|-------------|--------|
| Phase 1: Foundations | 1-16 | 1-4 | GPU+CUDA+Neural Nets+Transformers (LLM) · Digital logic + first Verilog (Silicon, from W3) | 🟡 In Progress |
| Phase 2: Intermediate | 17-34 | 5-8 | PEFT, Quant, Inference, RAG, Agents (LLM) · Single-cycle RV32I + MAC unit (Silicon) | ⬜ Not Started |
| Phase 3: Advanced | 35-52 | 9-12 | Multi-modal, MoE, kernels, training infra (LLM) · 5-stage pipelined RISC-V + 4×4 INT8 GEMM on FPGA (Silicon) | ⬜ Not Started |
| Phase 4: Expert | 53-68 | 13-16 | Papers, test-time compute, enterprise agents (LLM) · 8×8 systolic array + UVM verification (Silicon) | ⬜ Not Started |
| Phase 5: Mastery | 69-86 | 17-20 | Cutting edge research, OSS, deep specialization (LLM) · OpenLane RTL→GDSII on Sky130 (Silicon) | ⬜ Not Started |
| Phase 6: LLM Magnum Opus | 87-104 | 21-24 | **LLM Magnum Opus shipped (HARD deadline Month 24)** · TinyTapeout-ready block (Silicon) | ⬜ Not Started |
| Phase 7: Silicon Deep Dive | 105-130 | 25-30 | (LLM in maintenance mode) · Advanced microarch, AI accelerator architecture, transformer-block accelerator on FPGA, tape-out submission | ⬜ Not Started |
| Phase 8: Silicon Magnum Opus | 131-156 | 31-36 | **Final Silicon Magnum Opus: custom AI inference accelerator** — RTL + FPGA demo + GDSII + paper + (optional) physical silicon | ⬜ Not Started |

---

# ═══════════════════════════════════════════════════════════
# MONTHLY CAPSTONE PROJECT CATALOG (Build Only When Reached)
# ═══════════════════════════════════════════════════════════

> **THREE projects per month from Month 2 onwards.** Each must combine skills from the current month AND
> all previous months. Each must have a **novelty angle** — something that isn't
> just a tutorial copy-paste but demonstrates real understanding and creativity.
> This is a catalog of future project briefs. Build each project only during the
> **revision week** after the required weeks have been completed.
> These are the projects you put on your resume and GitHub.
>
> - **Project A** = structured LLM/CUDA implementation (detailed below)
> - **Project B** = novel LLM/CUDA tool/product (detailed below)
> - **Project C** = **silicon counterpart** — the same month's idea expressed in HDL / RTL / FPGA. Project C is intentionally **small** the first few months (Month 2-5: just exercises and tiny modules), grows into HDL builds (Month 6-10), then real accelerators (Month 11-24), then full chip designs in Phase 7-8 (Month 25-36). Project C must always **visibly relate** to that month's LLM topic — they are the same lesson at two layers.
>
> All three must use current month + all previous month skills. Built during revision week.

Each project description below is a starting brief, not a cage. If you discover a sharper user pain while studying, improve the project, but keep the cumulative rule: Month `N` must visibly use Month `1...N`.

For strict chronological study, follow the **Phase 1 → Phase 8** weekly sections
below. Treat this catalog as the project reference that each revision week points
back to.

Every monthly project README should include:

- **User:** who this helps (Project A/B) or what hardware module is delivered (Project C)
- **Pain:** what annoying/expensive/confusing thing it fixes (A/B) or what silicon idea it makes concrete (C)
- **Novelty:** what is different from existing tools or tutorials
- **Cumulative skills:** which months are used and where in the code
- **Evidence:** benchmark, evaluation, demo, screenshots, or reproducibility report (A/B); waveform / FPGA demo / synthesis report (C)
- **Co-design link:** how Project C explains, accelerates, or hardware-grounds the same month's Project A and B

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

## Month 1 Project C: *(Silicon track has not started yet — begins Week 3 / Month 2)*
> The Silicon track begins Day 6 of **Week 3**, after you finish your existing Week 1 + Week 2. Month 1 has no Project C; instead, it has a **Silicon Onboarding** task done during the Month 1 revision week:
- [ ] Order an FPGA board (recommended: Digilent Arty A7-100T or Tang Nano 20K — see Appendix > Hardware Setup)
- [ ] Install free toolchain on your machine: **Vivado WebPACK** (Xilinx) or Gowin EDA (Tang Nano), plus open-source **Icarus Verilog + Verilator + GTKWave**
- [ ] Read Harris & Harris *Digital Design and Computer Architecture, RISC-V Edition* — Chapter 1 (binary, hex, gates) ~3 hrs
- [ ] Skim the *VLSI Facts* "VLSI Roadmap 2026" page once to mentally map the territory ahead
- [ ] **No checkbox here counts as Project C** — but onboarding done means Day 6 of Week 3 starts smoothly

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

## Month 2 Project C — Silicon: "Logic-Lab — Gates to GEMM"
**Co-design link:** Project A trains GPT-2 using Tensor Cores; Project C peels back what a Tensor Core literally *is* in silicon — a tiny array of multiply-accumulate (MAC) cells made of adders, AND gates, and registers. By the end of this month you should hold a CPU/GPU schematic in your head down to the gate level.
**What:** A bench of tiny digital-logic exercises that culminate in a hand-drawn 1-bit MAC schematic + a Verilog half-adder/full-adder/4-bit ripple-carry adder + 4-bit multiplier with testbench. Run on Icarus Verilog and view waves in GTKWave. No FPGA needed yet.
**Skills used:** Boolean algebra, K-maps, gate-level design, two's complement, Verilog basics
**Deliverables:**
- [ ] Solve HDLBits modules 1-15 (Getting Started → Basic Gates) — all green
- [ ] Half-adder + Full-adder + 4-bit ripple-carry adder in Verilog with testbench
- [ ] 2:1 MUX, 4:1 MUX, 3-to-8 decoder
- [ ] 4-bit multiplier (shift-add) in Verilog
- [ ] **1-bit MAC unit** (a × b + c) with synthesizable Verilog + testbench
- [ ] Hand-drawn schematic of an N-bit MAC array (the "Tensor Core in 1 page")
- [ ] One-page write-up: "How a Tensor Core is just my MAC × 64"
- [ ] **Publish to GitHub** (folder: `silicon/month02/`)

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

## Month 3 Project C — Silicon: "ALU + Register File — The Datapath of an SM"
**Co-design link:** Project A writes a custom CUDA kernel for fused LayerNorm; Project C builds the *exact silicon primitives* the SM uses to execute that kernel — an ALU and a register file. After this month you will read PTX and see where each instruction lands in silicon.
**What:** A 32-bit RISC-V-style integer ALU + a 32×32 register file in SystemVerilog, with full testbench. Synthesize via Yosys (open-source) and read the resulting gate netlist for the first time.
**Skills used:** Verilog/SystemVerilog basics (Days 6 weeks 3-12), combinational + sequential design, Yosys synthesis intro
**Deliverables:**
- [ ] HDLBits modules 16-40 (combinational + sequential) all green
- [ ] **32-bit ALU** supporting ADD, SUB, AND, OR, XOR, SLL, SRL, SRA, SLT (RV32I subset) in SystemVerilog
- [ ] **32×32 register file** (2 read ports, 1 write port) — the same shape as a real CPU's regfile
- [ ] Synchronous testbench with constrained-random tests, verified bit-exact against a Python reference
- [ ] First Yosys synthesis run — print the gate count, identify what each gate is doing
- [ ] One-page write-up: "What my ALU has in common with `mma.sync.aligned`"
- [ ] **Publish to GitHub** (folder: `silicon/month03/`)

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

## Month 4 Project C — Silicon: "INT4/INT8 MAC Unit + Multi-Precision Datapath"
**Co-design link:** Project A quantizes models to INT4/INT8/FP8 in software; Project C builds the *silicon* that makes those formats fast — a multi-precision MAC unit that can switch between INT8 and INT4 (with packed-byte storage) at runtime. This is exactly what NVIDIA Marlin / ExLlama v2 kernels assume the hardware does.
**What:** A configurable MAC unit (parameter `WIDTH = 4 | 8`) in SystemVerilog with on-the-fly dequantization (load INT4 → expand → MAC → accumulate in INT32). Synthesize via Yosys and report area for each precision.
**Skills used:** Verilog datapath, fixed-point arithmetic, two's complement, parameterized modules
**Deliverables:**
- [ ] **Configurable MAC unit** in SystemVerilog: `mac #(.W(4|8|16)) mac0(...)` — runtime-switchable precision
- [ ] **Packed INT4 storage**: two INT4 in one byte, with dequant logic
- [ ] **Saturating accumulator**: prevent overflow with explicit saturation logic
- [ ] Testbench comparing your MAC against a Python reference for 100k random inputs
- [ ] Yosys synthesis report: area in gate-equivalents for INT4 vs INT8 vs INT16 — make the trade-off visible
- [ ] One-page write-up: "Why GPTQ/AWQ are useless without my MAC unit"
- [ ] **Publish to GitHub** (folder: `silicon/month04/`)

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

## Month 5 Project C — Silicon: "FIFO + AXI-Lite — The Plumbing of an LLM Server"
**Co-design link:** Project A serves a RAG system over FastAPI with KV-cache and PagedAttention; Project C builds the *silicon plumbing* that makes any inference serving system possible — synchronous & asynchronous FIFOs, and a small AXI-Lite slave (the bus protocol every accelerator uses to talk to the host CPU). vLLM's PagedAttention is essentially a FIFO of memory pages.
**What:** Synchronous + asynchronous (CDC-safe) FIFO in SystemVerilog, plus an AXI4-Lite slave that exposes 4 control registers. First step toward the inference accelerators of Months 11-12.
**Skills used:** Sequential design, clock domain crossing, handshake protocols, parameterized depth
**Deliverables:**
- [ ] **Synchronous FIFO** (parameterized depth & width) with full/empty flags + tests
- [ ] **Asynchronous FIFO** with Gray-coded pointers (clock-domain-crossing safe)
- [ ] **AXI4-Lite slave** with 4 32-bit registers (read + write), tested with a SV testbench acting as master
- [ ] Verify: write all 4 regs, read back, check FIFO push/pop with backpressure
- [ ] One-page write-up: "vLLM's PagedAttention is just my FIFO of pointers"
- [ ] **Publish to GitHub** (folder: `silicon/month05/`)

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

## Month 6 Project C — Silicon: "Single-Cycle RV32I CPU on FPGA" 🔧 (FIRST FPGA BUILD)
**Co-design link:** Project A builds an agent platform that runs on a CPU + GPU; Project C builds **your own CPU from scratch in Verilog** — a single-cycle RISC-V (RV32I) processor that runs real C code (compiled with riscv32-gcc) on a real FPGA. After this month you understand exactly what executes underneath every Python `if`/`for`/`while`.
**What:** A working RV32I single-cycle CPU on an FPGA dev board (Arty A7 or Tang Nano 20K), with UART output. Runs Hello-World and a small matmul written in C.
**Skills used:** All prior Verilog (Months 2-5), instruction decode, datapath, control, FPGA toolchain (Vivado / Gowin EDA)
**Deliverables:**
- [ ] **All RV32I instructions** (R-type, I-type, S-type, B-type, U-type, J-type) implemented and tested in simulation
- [ ] **Datapath**: PC, IMem, RegFile, ALU (from M3), DMem, immediate generator, branch unit
- [ ] **Control unit** (FSM): generate control signals from opcode/funct3/funct7
- [ ] Cross-compile a C "Hello World" with `riscv32-gcc`, load `.hex` into IMem, see "Hello" come out the UART
- [ ] Run a 4×4 INT8 matmul written in C (no hardware acceleration — pure CPU loops). Measure cycles per multiply.
- [ ] Synthesis report: LUTs, FFs, BRAMs used; max clock freq (timing report)
- [ ] One-page write-up: "Why CUDA exists" — using the cycle count of your matmul as the punchline
- [ ] **Publish to GitHub** (folder: `silicon/month06_rv32i_cpu/`)
- [ ] **Buffer week extension if needed** — this is the largest Project C so far; take the second buffer week if Month 6 revision week isn't enough

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

## Month 7 Project C — Silicon: "Pipelined RV32I (5-Stage) — IF/ID/EX/MEM/WB"
**Co-design link:** Project A and B chain GPU operations as a pipeline (preprocess → embed → retrieve → generate); Project C upgrades your single-cycle CPU into a **5-stage pipelined CPU** — the same overlapping-execution principle GPUs use across warps and PyTorch uses across layers. Pipeline hazards are the *exact same problem* as inter-kernel dependencies in CUDA streams.
**What:** Convert your Month 6 single-cycle RV32I into a 5-stage pipeline (IF / ID / EX / MEM / WB) with hazard detection and forwarding, on the same FPGA.
**Skills used:** Pipeline registers, data hazards, control hazards, forwarding/bypassing, stalling
**Deliverables:**
- [ ] **5 pipeline stages** with explicit pipeline registers (IF/ID, ID/EX, EX/MEM, MEM/WB)
- [ ] **Hazard detection unit**: detect RAW hazards on `lw → use` (1-cycle stall)
- [ ] **Forwarding unit**: bypass EX/MEM and MEM/WB to ALU inputs
- [ ] **Branch handling**: flush 2 instructions on taken branch (predict-not-taken)
- [ ] Re-run the C matmul from M6 — measure ~4× cycle reduction (per the textbook prediction)
- [ ] Verify against Spike (the official RISC-V ISA simulator) for 1000 random programs
- [ ] One-page write-up: "Why CUDA streams exist" — pipelining at scale
- [ ] **Publish to GitHub** (folder: `silicon/month07_pipelined_rv32i/`)

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

## Month 8 Project C — Silicon: "I-Cache + D-Cache + AXI Memory Subsystem"
**Co-design link:** Project A and B compare different model architectures running on GPUs with memory hierarchies; Project C teaches that hierarchy in silicon — by adding a small **direct-mapped instruction cache and data cache** to your pipelined CPU. After this month "L1 cache miss" becomes a thing you can *see in waveforms*.
**What:** Add a 4 KB direct-mapped I-cache and a 4 KB direct-mapped D-cache (with valid bits, dirty bits, write-back policy) to your Month 7 pipelined RISC-V CPU. Connect to a backing BRAM via a small AXI-Lite bus.
**Skills used:** Memory hierarchy, cache state machines, write policies, AXI bus (from M5)
**Deliverables:**
- [ ] **Direct-mapped I-cache**: 4 KB, 32-byte lines, 128 sets — read-only, refill on miss
- [ ] **Direct-mapped D-cache**: 4 KB, write-back with dirty bit, write-allocate on miss
- [ ] **Cache controller FSMs**: IDLE → COMPARE_TAG → ALLOCATE → WRITE_BACK
- [ ] **AXI-Lite memory bus** between cache and BRAM "main memory"
- [ ] **Performance counters**: hit count, miss count, write-back count — read via CSR
- [ ] Run the matmul C program; measure hit rate and IPC improvement vs no-cache version
- [ ] One-page write-up: "Why CUDA shared memory exists" using your cache miss waveforms
- [ ] **Publish to GitHub** (folder: `silicon/month08_cached_cpu/`)

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

## Month 9 Project C — Silicon: "4×4 INT8 Systolic GEMM on FPGA" 🔧 (FIRST AI ACCELERATOR)
**Co-design link:** Project A writes Triton kernels that target Tensor Cores; Project C builds **the Tensor Core itself** in silicon — a 4×4 weight-stationary systolic array of MAC units (from M2/M4) that performs INT8 GEMM. After this month you have built, in RTL, a tiny version of *the exact circuit that runs every Triton matmul*.
**What:** A 4×4 systolic array of INT8 MAC units, weight-stationary dataflow, with input/weight/output buffers. Memory-mapped via AXI4-Lite so a host program can write inputs, kick off compute, and read results. Synthesize on FPGA + measure cycles vs your Month 7 CPU running the same matmul in C.
**Skills used:** MAC unit (M2/M4), FIFO/AXI (M5), pipelined design (M7), parameterized SystemVerilog
**Deliverables:**
- [ ] **4×4 systolic array** in SystemVerilog (parameterized to scale to 8×8 later) — weight-stationary
- [ ] **3 BRAM buffers**: input matrix A, weights W, output C
- [ ] **Tile controller FSM**: load weights → stream activations → drain outputs
- [ ] **AXI4-Lite control registers**: A/W/C base addresses, M/N/K dimensions, start/done bits
- [ ] **Test harness**: Python script writes random INT8 matrices, kicks off, reads result, compares against numpy
- [ ] **Benchmark on FPGA**: cycles for 16×16 matmul vs your M7 CPU running C-loop matmul → expect ~50–100× speedup
- [ ] One-page write-up: "What I just built is what every Triton kernel ultimately drives"
- [ ] **Publish to GitHub** (folder: `silicon/month09_systolic_4x4/`)
- [ ] **Future extension after Month 12:** scale to 8×8 and 16×16

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

## Month 10 Project C — Silicon: "Mesh NoC for Multi-Tile Accelerator"
**Co-design link:** Project A and B build distributed training across GPUs connected by NVLink/NVSwitch; Project C builds **a 2D mesh on-chip network (NoC)** that connects 4 of your Month 9 systolic tiles. All-reduce on a NoC = all-reduce on NVLink, just at a smaller scale. After this month you understand why tensor parallelism and NoC topology are the same problem.
**What:** A 2×2 mesh NoC with router modules, virtual channels, and credit-based flow control, connecting 4 of your Month 9 systolic-array tiles. Implements an all-reduce primitive across the 4 tiles in hardware.
**Skills used:** NoC topologies, packet routing, deadlock avoidance, parameterized router modules
**Deliverables:**
- [ ] **2×2 mesh router** in SystemVerilog with XY routing
- [ ] **Virtual channels** (2 VCs minimum) for deadlock avoidance
- [ ] **Credit-based flow control** between routers
- [ ] **All-reduce primitive**: ring or recursive-doubling across 4 tiles, implemented in HW
- [ ] **Test**: 4 tiles each compute partial sums, NoC all-reduces them, single tile verifies result
- [ ] Synthesis + STA report: throughput in flits/cycle, max frequency
- [ ] One-page write-up: "Why NCCL exists" using your NoC waveforms
- [ ] **Publish to GitHub** (folder: `silicon/month10_noc/`)

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

## Month 11 Project C — Silicon: "Reproduce a Hardware Paper — Eyeriss-Style Row-Stationary PE"
**Co-design link:** Project A and B reproduce LLM papers; Project C reproduces the foundational AI-accelerator paper — **Eyeriss (Chen et al., ISCA 2016)**. You implement a single row-stationary Processing Element from the paper, the same dataflow that informs every modern AI chip. Reading hardware papers is now part of your weekly routine.
**What:** Implement a row-stationary Processing Element in SystemVerilog: 16 MAC units sharing weights along rows, with input/partial-sum FIFOs. Compare its energy/cycle profile against your Month 9 weight-stationary array on the same convolution workload.
**Skills used:** Dataflow architecture, paper reading, parameterized SystemVerilog, energy modeling
**Deliverables:**
- [ ] Read Eyeriss paper end-to-end and write a 1-page summary in your own words
- [ ] **Row-stationary PE** in SystemVerilog (16 MAC units, weight-shared rows)
- [ ] Run a 3×3 convolution on a small image and verify against numpy
- [ ] **Compare with M9 weight-stationary**: cycles, FF count, BRAM usage on the same convolution
- [ ] One-page write-up: "Why the dataflow you choose decides what models run well on your chip"
- [ ] Set up a weekly arxiv hardware feed (cs.AR + select cs.LG hardware papers) — start reading 1 paper/week from now on
- [ ] **Publish to GitHub** (folder: `silicon/month11_eyeriss_pe/`)

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

## Month 12 Project C — Silicon: "8×8 INT8 Systolic Array + PyTorch Bridge"
**Co-design link:** Project A and B do test-time compute that pushes more inference work to silicon; Project C is **the silicon that benefits from it** — a scaled-up 8×8 INT8 systolic array with a real PyTorch bridge (via PCIe XDMA on FPGA, or CocoTB simulation if no PCIe). Run an actual fine-tuned LLM's MLP weight matmul on your chip.
**What:** Scale your Month 9 4×4 array to 8×8, add output-stationary mode (configurable dataflow), and write a PyTorch C++ extension that calls your FPGA accelerator for `torch.matmul` of small INT8 matrices.
**Skills used:** All prior silicon, PyTorch C++ extensions (M3 LLM-side), AXI/PCIe DMA, dataflow modes
**Deliverables:**
- [ ] **8×8 systolic array** with both **weight-stationary** and **output-stationary** modes (runtime selectable)
- [ ] **PyTorch C++ extension** wrapping the FPGA accelerator: `myaccel.matmul(a_int8, w_int8) -> c_int32`
- [ ] **Real LLM workload**: take one MLP weight matrix from your Month 3 fine-tuned model, INT8-quantize it, run prefill of 1 batch×64 token through your accelerator. Verify against PyTorch.
- [ ] **Benchmark**: GOPS/W on FPGA vs the same workload on your laptop CPU and GPU
- [ ] One-page write-up: "What changes when I add a second dataflow mode" — argue *which* dataflow wins for which layer
- [ ] **Publish to GitHub** (folder: `silicon/month12_systolic_8x8_torch/`)

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

## Month 13 Project C — Silicon: "Production-Ready Accelerator Wrapper + UVM Testbench (Tier 1)"
**Co-design link:** Project A is a production LLM serving platform; Project C makes your Month 12 8×8 systolic array **production-ready** by wrapping it in a CSR + interrupt + DMA control plane and writing your **first UVM testbench** for it. Enterprise silicon is 70% verification — start now.
**What:** Wrap the 8×8 systolic array in a real "accelerator" interface: AXI4 (full, not just Lite) DMA for streaming, CSR space, an interrupt line, and a UVM-Lite verification environment.
**Skills used:** AXI4 full, DMA, CSRs, interrupts, **first SystemVerilog OOP for verification** (UVM-Lite from chipverify.com)
**Deliverables:**
- [ ] **AXI4-full DMA reader/writer** for streaming inputs/weights/outputs
- [ ] **CSR space**: control, status, interrupt enable, doorbell registers
- [ ] **Interrupt logic**: assert when DMA done or compute done
- [ ] **UVM-Lite testbench**: agent + driver + monitor + scoreboard for the AXI bus, constrained-random transactions
- [ ] **Functional coverage**: covergroups for tile sizes, dataflow modes, INT4/INT8 selection — closure ≥ 90%
- [ ] First SystemVerilog assertions (SVA) on the doorbell handshake
- [ ] One-page write-up: "Why verification is half the work" — paste your bug-find log
- [ ] **Publish to GitHub** (folder: `silicon/month13_uvm/`)

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

## Month 14 Project C — Silicon: "Formal Verification + Sparse 2:4 GEMM Hardware"
**Co-design link:** Project A and B engineer training data quality and detect contamination — the "data integrity" problem. Project C does the silicon equivalent: **formal verification** (mathematical proof that your hardware is correct, no possible bug) plus **2:4 structured sparsity hardware** (Ampere+ NVIDIA's hardware feature that works *only because the data has a structural property*).
**What:** Use **SymbiYosys** (open-source formal verification) to prove safety properties on your AXI handshake, then add 2:4 structured sparsity support to your 8×8 systolic array.
**Skills used:** Formal verification, SVA properties, structured sparsity, parameterized HW
**Deliverables:**
- [ ] Write 5+ SVA properties for your AXI4 DMA (no deadlock, no spurious done, etc.)
- [ ] **Run SymbiYosys** to formally prove them — find at least 1 corner-case bug your UVM tests missed
- [ ] **2:4 sparse mode** in the systolic array: every group of 4 weights has exactly 2 zeros, encode with 2-bit indices, hardware decompresses on the fly
- [ ] Compare dense vs 2:4 sparse: throughput should ~2× for the same area
- [ ] One-page write-up: "What I learned from the bug formal found that my testbench didn't"
- [ ] **Publish to GitHub** (folder: `silicon/month14_formal_sparse/`)

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

## Month 15 Project C — Silicon: "FlashAttention-Inspired Hardware Block (Tile-Based Attention RTL)"
**Co-design link:** Phase 2's Flash Attention paper showed how to redesign attention so it fits on-chip; Project C builds the **silicon version of that same insight** — an attention engine in RTL that tiles Q, K, V in BRAM and computes attention with on-the-fly softmax. This is the most exciting Project C so far and a direct prerequisite for the LLM Magnum Opus.
**What:** A simplified Flash-Attention-style hardware block in SystemVerilog: configurable tile size, online softmax (running max + running sum), tiled QKV streaming, INT8/FP16 modes. Run a single attention layer of a 4-head 64-dim transformer on FPGA.
**Skills used:** Online softmax (M11 LLM), systolic GEMM (M9-M12 silicon), AXI DMA (M13), pipelined design
**Deliverables:**
- [ ] **Tile loader**: stream Q tile (TM × d), K tile (TN × d), V tile (TN × d) into BRAM
- [ ] **QK^T systolic block**: reuse your 8×8 systolic array
- [ ] **Online softmax unit**: track running max and running sum; rescale partial outputs (the FA-2 trick)
- [ ] **PV systolic block**: second pass on the V tile
- [ ] **Output writer**: stream tile output to HBM/DDR via AXI
- [ ] Verify against PyTorch's `scaled_dot_product_attention` for 100 random inputs
- [ ] **Run a real LLaMA-style attention layer** (4 heads × 64 dim × 256 tokens) on FPGA
- [ ] One-page write-up: "What FlashAttention is — at the gate level"
- [ ] **Publish to GitHub** (folder: `silicon/month15_flashattn_hw/`)

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

## Month 16 Project C — Silicon: "First Yosys+OpenSTA Synthesis Run on Sky130"
**Co-design link:** Project A contributes RTL to open-source AI projects; Project C makes the same leap on the silicon side — **your first synthesis on a real PDK** (SkyWater 130 nm), with timing, area, and power reports. By the end of Month 16 your accelerator is no longer "FPGA-only" — it's *physically synthesizable on real silicon*.
**What:** Take your Month 12 8×8 systolic array, run **Yosys synthesis on the Sky130 PDK**, then run **OpenSTA** with realistic constraints. Generate area, power, and timing reports. This is your first real chip-design flow run end-to-end.
**Skills used:** Yosys, OpenSTA, Sky130 PDK, .lib/.lef files, SDC constraint files
**Deliverables:**
- [ ] **Install Sky130 PDK** + Yosys + OpenSTA (open-source, free)
- [ ] Write an **SDC constraint file**: 200 MHz clock, IO delays, false paths
- [ ] **Synthesize** the 8×8 array — read the gate count, tech-mapped netlist
- [ ] **Run OpenSTA** in MCMM (multi-corner): SS/TT/FF, report worst slack
- [ ] **Power report**: dynamic + leakage, identify high-toggle nets
- [ ] **Area report**: gate-equivalents, breakdown by block
- [ ] If timing fails: use retiming or pipeline insertion to fix it (real chip-design move)
- [ ] One-page write-up: "What I learned about my own design from synthesis"
- [ ] **Publish to GitHub** (folder: `silicon/month16_sky130_synth/`)

---

## Month 17 Project A: "PolyglotLM — Multi-Language Domain LLM" (NEW — Month 17 of LLM track)
**What:** Train (or continue-pretrain) a small LLM (~700M-1.5B) on a *non-English* + code domain (e.g., Hindi + Python, Spanish + finance, Japanese + chemistry). Includes BPE tokenizer trained on the domain, full data curation, SFT, DPO, quantization, and serving.
**Novelty:** Most LLMs are English-first. Your project shows the full pipeline of *bringing an LLM into a new language and domain* — from data scraping → tokenizer training → continued pre-training → SFT → DPO → eval → deployment. This is what every non-US AI company actually does.
**Skills used:** All Phase 1-3 + advanced fine-tuning (M14), synthetic data (M14), serving (M13)
**Deliverables:**
- [ ] Domain-specific BPE tokenizer trained from scratch with explicit efficiency report
- [ ] Continued pre-training pipeline on a curated domain corpus (≥1B tokens)
- [ ] SFT + DPO on synthetic + filtered preference data
- [ ] FP8/INT4 quantized inference deployed on NIM/vLLM
- [ ] Domain benchmark suite (write 5-10 evaluation tasks specific to the language/domain)
- [ ] **Publish to GitHub with model on HuggingFace + benchmark report**

## Month 17 Project B: "ContextWeaver — 1M-Token Context Extension Toolkit"
**What:** A toolkit that extends any open LLM's context to 1M tokens via RoPE scaling, YaRN, or LongRoPE, with eval (NIAH, passkey, document QA) and a serving stack that handles paged KV-cache offload to NVMe.
**Novelty:** Combines training-time context extension *and* serving-time NVMe-backed paged KV-cache — the two halves nobody bundles together. Lets a small org run 1M-context LLMs on a single H100.
**Skills used:** Long context (Phase 2), RoPE scaling, KV-cache management, serving (M13), NIAH eval
**Deliverables:**
- [ ] Extend a 7B model to 1M context using LongRoPE / NTK-aware scaling
- [ ] NIAH + passkey eval at 4K, 32K, 256K, 1M
- [ ] NVMe-backed paged KV-cache with prefetcher
- [ ] Latency study: how much slower is decode at 256K vs 4K?
- [ ] **Publish with reproducible training + serving Docker stack**

## Month 17 Project C — Silicon: "Floorplan + Place-and-Route with OpenLane"
**Co-design link:** Project A and B build production-grade language and context-handling capabilities; Project C steps from synthesis into **physical implementation** — your first OpenLane run that takes RTL → GDSII layout (the literal silicon mask file) on Sky130.
**What:** Run the **OpenLane / LibreLane flow** end-to-end on your 8×8 systolic array on Sky130 PDK. Get a floorplan, placement, CTS, routing, and a viewable GDSII file in **KLayout**.
**Skills used:** Floorplanning (utilization, aspect ratio, IO ring), placement, CTS, routing, OpenLane TCL config
**Deliverables:**
- [ ] **OpenLane config**: target 50% utilization, 1×1 aspect ratio, 200 MHz clock
- [ ] Successful **floorplan** → **placement** → **CTS** → **routing** with no DRC errors
- [ ] **GDSII** opened in KLayout — screenshot of your chip's actual layout
- [ ] Synthesis vs post-layout timing comparison (real wires hurt timing — see by how much)
- [ ] One-page write-up: "What I learned from looking at my own silicon layout"
- [ ] **Publish to GitHub** (folder: `silicon/month17_openlane/`)

---

## Month 18 Project A: "EvolveLM — Continual Learning + Live Updating LLM"
**What:** A framework for keeping an LLM up to date *after* deployment without catastrophic forgetting — using LoRA-stack updates, periodic distillation, and a retrieval-augmented "knowledge cutoff extension" layer.
**Novelty:** "Models go stale" is a known problem with no clean answer. Your project provides a working answer: a deployment that *learns* in production, with safety/regression gates.
**Skills used:** PEFT (M4), distillation (Phase 2), RAG (M5), evaluation (M10), all prior
**Deliverables:**
- [ ] LoRA-stack continual learning loop (new adapter per week of new data)
- [ ] Forgetting detection: per-domain regression eval before promoting an adapter
- [ ] RAG-as-knowledge-extension for ultra-recent facts
- [ ] Production guardrails: don't promote a regressed adapter
- [ ] **Publish with 4-week simulated update log + dashboards**

## Month 18 Project B: "AlignBench — Alignment Method Showdown"
**What:** Train the same base model with 5+ alignment methods (SFT, DPO, IPO, ORPO, SimPO, KTO, RLAIF) and produce a unified comparison report (cost, quality, safety, refusal-rate).
**Novelty:** A single replicable harness for the alignment landscape. Useful for any team picking an alignment recipe.
**Skills used:** All alignment methods (M3, M6, Phase 4), evaluation (M10), all prior
**Deliverables:**
- [ ] Reproducible training scripts for each method
- [ ] Unified eval suite + cost report
- [ ] **Publish as the alignment-method reference project**

## Month 18 Project C — Silicon: "First TinyTapeout-Sized Block (Sky130) — Real Tape-Out Prep"
**Co-design link:** Project A and B push deployed-model quality further; Project C pushes you toward **real silicon you can hold in your hand** — a TinyTapeout-shaped (160 µm × 100 µm tile) Sky130 block that you'll actually submit in Month 22 for fabrication. This month is the pre-flight build.
**What:** Take a small but meaningful AI block — say a 4×4 INT8 MAC array or a small softmax unit — and squeeze it into TinyTapeout's tile constraints. Get it through DRC + LVS clean.
**Skills used:** TinyTapeout template, area-constrained design, LVS, DRC fixes
**Deliverables:**
- [ ] Fork the official `tt09-` (or current) TinyTapeout template repo
- [ ] Pick a small AI block that fits the tile budget (≈ 1000 standard cells)
- [ ] Run the TT GitHub Actions flow — get a clean GDSII
- [ ] **DRC + LVS clean** — no violations
- [ ] Write the datasheet markdown the TT submission requires
- [ ] One-page write-up: "What I'll submit to TinyTapeout in Month 22"
- [ ] **Publish to GitHub** (folder: `silicon/month18_tt_prep/`)

---

## Month 19 Project A: "DistillForge — Multi-Teacher Distillation Workbench"
**What:** A workbench to distill a small student model from multiple specialized teacher models (e.g., one math teacher, one code teacher, one chat teacher) using attention transfer, feature distillation, and synthetic data.
**Novelty:** Most distillation is single-teacher. Yours is multi-teacher with automatic dataset routing — for each query type, picks the right teacher's logits.
**Skills used:** Distillation (Phase 2), synthetic data (M14), training (M10), all prior
**Deliverables:** [ ] 3-teacher distillation, [ ] benchmark vs single-teacher, [ ] **publish + blog post**

## Month 19 Project B: "EnergyAuditor — LLM Carbon + Cost Tracker"
**What:** A wrapper around inference + training infrastructure that reports **energy used per token, kWh per training run, $/token, gCO₂e/token** based on your GPU model and grid mix.
**Novelty:** No mainstream LLM tool reports energy. Yours does, and pairs energy savings with quantization + serving optimizations from prior months.
**Skills used:** Profiling (M2), quantization (M4), serving (M13), all prior
**Deliverables:** [ ] integrate with vLLM/Triton/NIM, [ ] dashboard, [ ] **publish**

## Month 19 Project C — Silicon: "Multi-Block Floorplan + Power Distribution Network"
**Co-design link:** Project B *measures* energy per token; Project C *designs* the power distribution that determines that energy at the silicon level. This month you start treating your accelerator as a real chip with multiple blocks (GEMM + softmax + DMA + control) sharing a power grid.
**What:** Lay out a multi-block design — e.g., GEMM tile + softmax unit + DMA engine + control FSM — with a proper power-ground (PG) grid, multiple voltage areas, and clock gating.
**Skills used:** Multi-block floorplanning, PG ring + straps, IR drop estimation, clock gating
**Deliverables:**
- [ ] **Multi-block floorplan** in OpenLane: 4 blocks placed with halos
- [ ] **PG grid**: top-level VDD/GND straps + macro-level rings
- [ ] **Clock gating** inserted on idle blocks
- [ ] **IR drop estimation** report (use OpenROAD's PDN analysis)
- [ ] One-page write-up: "Where my chip burns most power and why"
- [ ] **Publish to GitHub** (folder: `silicon/month19_pdn/`)

---

## Month 20 Project A: "OpenLLM-Stack — Self-Hosted Production LLM Platform v1"
**What:** A turnkey self-hosted LLM platform: NIM/vLLM inference + RAG + agents + guardrails + monitoring + cost tracking, deployable with one Helm chart on a small Kubernetes cluster. The "self-hosted ChatGPT" for an enterprise.
**Novelty:** All the pieces from Months 1-19, packaged as one product.
**Skills used:** Everything LLM so far
**Deliverables:** [ ] Helm chart, [ ] docs site, [ ] demo video, [ ] **public release**

## Month 20 Project B: "DocAgent — Multi-Modal Document Analysis Agent (NVIDIA Stack)"
**What:** An agent that ingests 1000-page documents (PDFs, slides, video transcripts) and answers questions with citations, using multi-modal RAG, table understanding, chart understanding, and the full NVIDIA stack.
**Novelty:** Most doc-QA is text-only. Yours handles tables, charts, images, formulas, and slide layouts.
**Deliverables:** [ ] full pipeline, [ ] eval suite, [ ] **publish with demos on real reports**

## Month 20 Project C — Silicon: "Static Timing Analysis Mastery — Closing Timing on Sky130"
**Co-design link:** Project A and B push for *production-quality* LLM systems; Project C pushes for **production-quality silicon timing** — a clean STA closure on your multi-block design at a target frequency. This is the bar a real chip team clears every day.
**What:** Take your Month 19 multi-block design and push the clock target from 200 MHz to 400 MHz. Use multi-cycle paths, clock skew, retiming, register pipelining, and (if needed) architectural changes. Document every fix.
**Skills used:** Advanced STA, MCMM, multi-cycle / false-path constraints, retiming
**Deliverables:**
- [ ] Pre-fix timing summary at 400 MHz: WNS, TNS, failing endpoints
- [ ] Apply each fix one at a time, log the WNS improvement after each
- [ ] **Final WNS ≥ 0** at 400 MHz across SS/TT/FF corners
- [ ] One-page write-up: "Timing closure as a debugging discipline"
- [ ] **Publish to GitHub** (folder: `silicon/month20_sta/`)

---

## Month 21 Project A: "ResearchLab — A Reproducible-Paper Pipeline for Yourself"
**What:** Take 3 recent (2026) research papers in different areas (efficient attention, alignment, agents) and reproduce all main results end-to-end on a single GPU, with reproducible Docker images and write-ups.
**Novelty:** Reproducibility is rare. Your repo is a public proof of how to do it cleanly.
**Skills used:** Phase 4 paper reading, all prior
**Deliverables:** [ ] 3 reproductions, [ ] each with a 1500-word write-up, [ ] **public release**

## Month 21 Project B: "OSS-Citizen — Sustained Open-Source Maintenance"
**What:** Become a recognized contributor to one major project (vLLM / TensorRT-LLM / NeMo / HuggingFace / Triton). Goal: **5+ merged PRs in this month alone**, plus issue triage and reviews.
**Novelty:** Sustained contribution > single PR. This builds reputation.
**Deliverables:** [ ] 5+ merged PRs, [ ] 1 review on someone else's PR/week, [ ] blog post

## Month 21 Project C — Silicon: "TinyTapeout Submission Live — Press Submit"
**Co-design link:** Project A and B graduate you into the OSS world; Project C graduates you into the **silicon-fabrication** world. By the end of this month you have **submitted a real chip block to TinyTapeout** that will be fabricated at SkyWater on a real wafer. This is the pre-Magnum-Opus tape-out — the dress rehearsal for Month 36.
**What:** Polish the Month 18 TT-prep block, write the full TT datasheet, integrate with TT's I/O harness, run TT's official GitHub Actions verification end-to-end, and **press submit on the next TinyTapeout shuttle**.
**Skills used:** Everything silicon so far + TT submission flow
**Deliverables:**
- [ ] Block functionally verified on FPGA
- [ ] Block passes TT's automated DRC + LVS + GitHub Actions checks
- [ ] Datasheet written (function, I/O pinout, register map)
- [ ] **Submission accepted** on the official TT shuttle 🎉
- [ ] One-page blog post: "I just submitted a real AI chip block to be fabricated"
- [ ] **Publish to GitHub** (folder: `silicon/month21_tt_submitted/`)

---

## Month 22-24 Project: "LLM Magnum Opus — Your Signature LLM Project" ⭐ (HARD DEADLINE — BY END OF MONTH 24)
> **This is the LLM-side capstone. It MUST be complete by end of Month 24 (= 2 years). The 24-month constraint is non-negotiable. The 6-month buffer (Month 19-21) was specifically created so this 3-month build window is comfortable, not rushed.**

**What:** Your final LLM capstone — a project that represents the BEST of everything you've learned in Phases 1-5. This is the LLM project people remember you by.
**Choose ONE (or combine):**

**Option A: "LLM-from-Scratch-to-Production"**
- [ ] Train a domain-specific LLM (1-3B params)
- [ ] Full pipeline: data curation → pre-training → SFT → DPO → quantization → deployment
- [ ] Custom CUDA kernels for 3+ operations (with Nsight Compute reports)
- [ ] Deployed on NIM with Guardrails, RAG, and voice interface
- [ ] Comprehensive evaluation + benchmark + reproducibility report
- [ ] **+ silicon hook:** include a section showing how your chosen Month 22 silicon co-design (Project C) accelerates one operation in this LLM

**Option B: "AI Agent OS"**
- [ ] Operating system for AI agents: define agents in YAML, auto-deploy with tools
- [ ] Plugin architecture: anyone can add new tools
- [ ] Built-in evaluation, monitoring, cost tracking
- [ ] Multi-model support: route to best model per task
- [ ] Self-improving: agents learn from their failures
- [ ] **+ silicon hook:** show a demo of the agent calling your custom FPGA accelerator for one tool

**Option C: "GPU Inference Engine"**
- [ ] Custom inference engine rivaling vLLM for a specific model family
- [ ] Hand-optimized CUDA/Triton kernels for attention, GEMM, normalization
- [ ] PagedAttention or your own KV-cache management
- [ ] Speculative decoding, continuous batching
- [ ] Benchmark showing competitive with vLLM/TensorRT-LLM on specific workloads
- [ ] **+ silicon hook:** drop-in support for your FPGA accelerator as one of the backends, with a benchmark

**Option D: "Edge LLM — On-Device Inference Stack"**
- [ ] A serving stack for ≤4-bit quantized 1-3B models on Jetson / mobile / browser (WebGPU)
- [ ] Custom kernels per platform
- [ ] Privacy-first features (no telemetry, on-device RAG)
- [ ] Battery-life and latency benchmark vs cloud baseline
- [ ] **+ silicon hook:** discuss what an ASIC would gain vs the CPU/NPU baseline

**Deliverables for any option:**
- [ ] Complete GitHub repo with excellent documentation
- [ ] Technical blog post (2500+ words)
- [ ] Demo video (5-10 minutes)
- [ ] Performance benchmarks and analysis (with reproducibility scripts)
- [ ] **This is your LLM portfolio centerpiece — the project a hiring manager remembers**

## Month 22 Project C — Silicon: "Co-Design Block for the LLM Magnum Opus"
**What:** A small but real silicon block that **plugs into the LLM Magnum Opus chosen above** — e.g., if you chose Option C (GPU Inference Engine), Project C is the FlashAttention hardware block from M15 polished into a deployable IP block. The silicon and the LLM project ship as one repo.
**Deliverables:** [ ] Polished RTL, [ ] FPGA demo, [ ] integrated with Magnum Opus, [ ] benchmark showing speedup on the integrated path

## Month 23 Project C — Silicon: "Layout Optimization + DFT Insertion"
**What:** Add **scan chains** (Design For Test) to your accelerator and re-run physical implementation. Your design is now testable like real production silicon.
**Deliverables:** [ ] scan-chain insertion, [ ] ATPG run with ≥95% fault coverage, [ ] post-layout timing closed, [ ] **publish**

## Month 24 Project C — Silicon: "LLM Magnum Opus Companion Paper — Hardware Section"
**What:** Write the **hardware-companion section** of your LLM Magnum Opus paper/blog: which ops are memory-bound, what an ASIC would change, what your FPGA prototype showed. By Month 24 your LLM Magnum Opus is shipped *with* a credible silicon perspective.
**Deliverables:** [ ] 1500-word hardware section, [ ] roofline-style charts for the LLM workload, [ ] integration with the Magnum Opus repo

---

## Month 25-27 Projects (Phase 7 begins — Silicon Deep Dive)
> **From Month 25 the LLM track is done. You revisit it weekly only as paper-reading and OSS upkeep. Your full daily attention now shifts to silicon.**

### Month 25 Project: "Out-of-Order RISC-V Mini-Core"
**What:** Extend your pipelined RV32I CPU with **register renaming + reservation stations + reorder buffer** (Tomasulo-style). Single-issue OoO is enough.
**Skills used:** Advanced microarchitecture, Hennessy & Patterson Ch 3
**Deliverables:** [ ] OoO RV32I-M with 8-entry ROB, [ ] verified vs Spike, [ ] benchmarked vs in-order baseline, [ ] **publish**

### Month 26 Project: "Branch Predictor Lab + Cache Coherence Toy"
**What:** Implement gshare → TAGE-style branch predictor, then a 2-core MESI coherence protocol on a shared L2.
**Deliverables:** [ ] gshare + TAGE, [ ] 2-core MESI in SystemVerilog, [ ] **publish**

### Month 27 Project: "Transformer-Block Accelerator on FPGA — Full Layer"
**What:** A complete transformer layer (multi-head attention + FFN + LayerNorm + residual + RoPE) running on FPGA with INT8 weights, computing one layer of a quantized LLaMA-style block at measurable tokens/sec.
**Skills used:** All silicon prior + LLM kernel knowledge from Phase 1-5
**Deliverables:** [ ] Full layer RTL, [ ] FPGA demo, [ ] benchmark vs PyTorch CPU, [ ] **publish — this is the headline silicon artifact before the Magnum Opus**

---

## Month 28-30 Projects (Phase 7 continues)

### Month 28 Project: "Architecture Comparison Paper — H100 vs MI300X vs TPUv5 vs Sohu for LLM Inference"
**What:** Use **Timeloop + Accelergy** to model 4 commercial AI chip architectures on a 70B-parameter LLM inference workload. Write a public technical paper comparing them.
**Deliverables:** [ ] Timeloop models for each chip, [ ] paper (~6000 words) on arxiv-ready style, [ ] **publish**

### Month 29 Project: "Compiler for Your Accelerator (MLIR / TVM)"
**What:** A small MLIR dialect (or TVM-style schedule) that lowers a transformer block IR onto your Month 27 accelerator. Software–hardware co-design proven by both halves working.
**Deliverables:** [ ] MLIR dialect, [ ] working compile of a tiny LLaMA layer, [ ] **publish**

### Month 30 Project: "Chiplet Design Study — Compute Die + Memory Die over UCIe"
**What:** Model your compute die + a simulated HBM-like memory die, define the UCIe interface, model die-to-die bandwidth/latency. Optionally implement a SystemC or RTL UCIe-PHY-emulator.
**Deliverables:** [ ] Chiplet model, [ ] UCIe interface spec, [ ] performance projection, [ ] **publish**

---

## Month 31-36 Project: "Silicon Magnum Opus — Custom AI Inference Accelerator" ⭐⭐⭐ (FINAL)
> **Six-month build. This is the chip-design career piece — the one you take to Etched, Tenstorrent, Cerebras, Groq, NVIDIA architecture interviews. By Month 36 this is your most important artifact.**

**What:** Design and implement a **custom AI inference accelerator** targeting LLM decode workloads. The full stack from architecture spec → RTL → verification → FPGA prototype → physical implementation in OpenLane on Sky130 → and (if your TT submission from Month 21 came back) characterization on real silicon.

**Choose ONE primary architecture (or hybrid):**

**Option A: "TinyTransformer Engine — Decode-Optimized ASIC"**
- A specialized inference accelerator for LLM decode (memory-bound) on small (1-3B) quantized models
- Multi-precision systolic array (FP8 / INT8 / INT4 / 2:4 sparse)
- On-chip paged KV-cache controller (PagedAttention in hardware)
- FlashDecoding-style attention block
- AXI-Lite control + AXI-Full DMA + interrupt
- Optionally: a small RISC-V control core (your Month 25 OoO core)

**Option B: "FlashAttention ASIC"**
- A pure attention accelerator: handles only `softmax(Q @ K.T) @ V`, but does it at 2-3× the silicon efficiency of a GPU
- Hopper-WGMMA-style async tile loads
- Online softmax in dedicated hardware
- 2:4 + INT4 sparse-aware
- Plug into a PyTorch-compatible runtime via `torch.utils.cpp_extension`

**Option C: "MoE Router + Expert Tile Array"**
- A spatial MoE architecture: one router + N expert compute tiles connected over a 2D NoC
- Hardware top-K selector
- All-to-all routing in NoC
- Scales to many tiles

**Option D: "CIM-Inspired LLM Decode Block"**
- A Compute-In-Memory–style block for matrix-vector multiply (the inner kernel of LLM decode)
- SRAM array doubles as both storage and bit-line MAC
- Especially aggressive on energy

**Required deliverables (regardless of option):**
- [ ] **Architecture spec document** (3000+ words): workload analysis (use your Month 28 paper), block diagram, dataflow, ISA / register map, performance/area/power targets
- [ ] **Synthesizable RTL** in SystemVerilog (10K+ lines), parameterized
- [ ] **Verification**: UVM testbench with ≥95% functional coverage + formal proofs on critical interfaces
- [ ] **FPGA prototype** running real LLaMA-class workload (single layer or single block, INT4/INT8 quantized) at measurable tokens/sec
- [ ] **PyTorch integration** as a backend — `torch.compile` custom backend or C++ extension
- [ ] **OpenLane / LibreLane RTL→GDSII flow** on Sky130 with timing/area/power signoff
- [ ] **DFT inserted** (scan chains), ATPG ≥ 95% fault coverage
- [ ] **Architecture comparison paper** (8000+ words): your design vs H100, MI300X, TPUv5, Sohu — performance per watt, area efficiency, key trade-offs
- [ ] **Open-source GitHub release** with reproducible everything (RTL, testbench, OpenLane config, FPGA bitstream build, PyTorch integration)
- [ ] **(Optional but ideal)**: TinyTapeout-fabricated Month 21 block returns from SkyWater this year — characterize it, post-mortem, write up
- [ ] **Demo video** (10-15 min): walkthrough of the architecture, FPGA running, OpenLane flow, performance numbers
- [ ] **Optional final tape-out** via Efabless chipIgnite (~$10K) — if you have the budget, this is the headline-grabber

**This project belongs in 4 places when shipped:** Hot Chips paper-track submission · arxiv preprint · GitHub trending · your CV

---

### Monthly Project Tracker (3 projects per month from Month 2 onwards)

| Month | Project A (LLM) | ✓ | Project B (LLM) | ✓ | Project C (Silicon) | ✓ |
|-------|------------------|---|------------------|---|---------------------|---|
| 1 | GPU Matrix Math Engine | ⬜ | LLM Inference Cost Simulator | ⬜ | *(Silicon onboarding only)* | ⬜ |
| 2 | nanoLLM — GPT from Scratch | ⬜ | TransformerScope — Behavior Debugger | ⬜ | Logic-Lab — Gates to GEMM | ⬜ |
| 3 | LLM Surgery — Fine-Tuning Toolkit | ⬜ | TokenScope — Tokenizer Analyzer | ⬜ | ALU + Register File | ⬜ |
| 4 | QuantBench — Quantization Analyzer | ⬜ | LLM-Speedometer — Inference Profiler | ⬜ | INT4/INT8 MAC Unit | ⬜ |
| 5 | DeepRAG — Production RAG | ⬜ | RAGTrace — Retrieval Debugger | ⬜ | FIFO + AXI-Lite | ⬜ |
| 6 | AgentForge ⭐ | ⬜ | EvalArena | ⬜ | Single-Cycle RV32I CPU on FPGA 🔧 | ⬜ |
| 7 | IncidentLens — Multimodal Debug | ⬜ | GPU PDF Intelligence | ⬜ | 5-Stage Pipelined RV32I | ⬜ |
| 8 | MoE-Lab | ⬜ | ArchSearch | ⬜ | I-Cache + D-Cache + AXI Memory | ⬜ |
| 9 | KernelSmith — Triton Library | ⬜ | FusionLab | ⬜ | 4×4 INT8 Systolic GEMM 🔧 | ⬜ |
| 10 | TrainScale — Distributed Training | ⬜ | TrainWatch | ⬜ | Mesh NoC for Multi-Tile | ⬜ |
| 11 | PaperBot | ⬜ | Paper2Benchmark | ⬜ | Eyeriss-style Row-Stationary PE | ⬜ |
| 12 | ReasonEngine — Test-Time Compute | ⬜ | ThinkTrace | ⬜ | 8×8 Systolic + PyTorch Bridge | ⬜ |
| 13 | NVServe — Production Serving | ⬜ | CostGuard | ⬜ | Production Wrapper + UVM | ⬜ |
| 14 | SynthData — Synthetic Data Factory | ⬜ | DataScope | ⬜ | Formal Verif + 2:4 Sparse HW | ⬜ |
| 15 | AgentX — Enterprise Agent (NVIDIA) | ⬜ | AgentEval | ⬜ | FlashAttention HW Block | ⬜ |
| 16 | OpenContrib — Open Source | ⬜ | *(2nd OSS project)* | ⬜ | First Yosys+OpenSTA on Sky130 | ⬜ |
| 17 | PolyglotLM — Multi-Lang Domain LLM | ⬜ | ContextWeaver — 1M-Token | ⬜ | OpenLane RTL→GDSII Floorplan | ⬜ |
| 18 | EvolveLM — Continual Learning | ⬜ | AlignBench | ⬜ | TinyTapeout Sky130 Block (prep) | ⬜ |
| 19 | DistillForge — Multi-Teacher | ⬜ | EnergyAuditor | ⬜ | Multi-Block Floorplan + PDN | ⬜ |
| 20 | OpenLLM-Stack — Self-Hosted | ⬜ | DocAgent — Multi-Modal | ⬜ | STA Mastery — Close Timing | ⬜ |
| 21 | ResearchLab — Reproducible Papers | ⬜ | OSS-Citizen — Sustained OSS | ⬜ | **TinyTapeout Submission Live** 🎉 | ⬜ |
| 22 | LLM Magnum Opus (build) ⭐ | ⬜ | *(integrated)* | — | Co-Design Block for Magnum Opus | ⬜ |
| 23 | LLM Magnum Opus (build) ⭐ | ⬜ | *(integrated)* | — | DFT Insertion + Layout Opt | ⬜ |
| 24 | **LLM Magnum Opus shipped** ⭐ | ⬜ | *(integrated)* | — | LLM Magnum Opus HW Companion | ⬜ |
| 25 | *(LLM track in maintenance — papers only)* | — | — | — | Out-of-Order RV32I Mini-Core | ⬜ |
| 26 | — | — | — | — | Branch Predictor + MESI Coherence | ⬜ |
| 27 | — | — | — | — | Transformer-Block Accelerator on FPGA | ⬜ |
| 28 | — | — | — | — | Architecture Comparison Paper | ⬜ |
| 29 | — | — | — | — | MLIR/TVM Compiler for Your Accelerator | ⬜ |
| 30 | — | — | — | — | Chiplet Design Study (UCIe) | ⬜ |
| 31-36 | — | — | — | — | **Silicon Magnum Opus** ⭐⭐⭐ | ⬜ |

---

# ═══════════════════════════════════════════════════════════
# MONTH-BY-MONTH FOCUS & CAPABILITIES
# "After this month, I can..."
# ═══════════════════════════════════════════════════════════

---

### Month 1 — GPU Architecture + CUDA Fundamentals + Neural Network Basics
**Focus:** CPU vs GPU architecture, CUDA programming model (kernels, threads, warps, blocks, grids), GPU memory hierarchy (global, shared, constant), memory coalescing, warp execution & divergence, parallel reduction, CUDA streams & graphs, neural networks from scratch, attention mechanism, tokenization (BPE)
**Silicon Day-6 Focus:** *(Silicon track does not start until Week 3 / Month 2)*. Use Month 1 revision week to do **Silicon onboarding only**: order an FPGA board (Arty A7-100T or Tang Nano 20K), install free toolchain (Vivado WebPACK / Gowin EDA + Icarus Verilog + Verilator + GTKWave), read Harris & Harris Ch 1 (gates, Boolean algebra, K-maps).
**Project:** GPU Matrix Math Engine + LLM Inference Cost Simulator

**After this month, you can:**
- [ ] Write and launch CUDA kernels — vector ops, reductions, tiled matrix multiplication
- [ ] Explain GPU architecture from memory: SMs, warps, shared memory, L1/L2, HBM
- [ ] Optimize memory access: coalescing, shared memory tiling, bank conflict avoidance
- [ ] Use CUDA streams for overlapping computation and data transfer
- [ ] Build an MLP from scratch in NumPy (forward + backward pass, no framework)
- [ ] Implement the attention mechanism and BPE tokenizer from scratch
- [ ] Benchmark CPU vs GPU and explain roofline model analysis
- [ ] Profile basic CUDA programs and identify compute-bound vs memory-bound kernels
- [ ] **Silicon onboarding:** FPGA board ordered, toolchain installed, gates/Boolean algebra reviewed

---

### Month 2 — Transformers + GPT from Scratch + GPU Libraries + Profiling
**Focus:** Full Transformer architecture (encoder + decoder), multi-head attention, positional encodings (sinusoidal, RoPE), GPT-2 implementation, cuBLAS, cuDNN, Tensor Cores (WMMA), Nsight Systems & Compute profiling, LLaMA architecture (RMSNorm, SwiGLU, GQA), basic interpretability (attention maps, residual stream, logit lens), scaling laws, mixed precision training (AMP, BF16), efficient data loading, CNNs
**Silicon Day-6 Focus:** *(Silicon track BEGINS this month — Week 3 onwards.)* Binary/hex/two's complement, Boolean algebra, half/full adders, ripple-carry adder, 4-bit multiplier (shift-add), 1-bit MAC unit. **Why this complements LLM track:** by the end of Month 2 you have built — *with your own hands in Verilog* — the smallest version of what cuBLAS and Tensor Cores execute.
**Projects:** A: nanoLLM · B: TransformerScope · **C: Logic-Lab — Gates to GEMM**

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
- [ ] **Silicon side:** explain a half-adder/full-adder schematic; write synthesizable Verilog for a 4-bit ripple-carry adder, a 4-bit shift-add multiplier, and a 1-bit MAC unit, with passing testbenches in Icarus Verilog

---

### Month 3 — Distributed Training + Fine-Tuning + Flash Attention + Triton
**Focus:** NCCL collectives (AllReduce, AllGather), DDP, tensor/pipeline/data parallelism, ZeRO, pre-training pipeline, SFT, DPO, RLHF overview, custom CUDA extensions for PyTorch, Flash Attention (tiling + online softmax), Triton GPU programming, PTX/SASS analysis, tokenizer deep dive, multi-modal overview, GPU cluster architecture
**Silicon Day-6 Focus:** RV32I ISA fundamentals, datapath design, control unit FSM, 32-bit ALU implementing the full RV32I subset, 32-register register file, first Yosys synthesis run with gate-count inspection. **Complement to LLM track:** writing custom CUDA extensions in Project A is the same idea as designing custom ALU operations in Project C — both add a new instruction to a hardware target.
**Projects:** A: LLM Surgery · B: TokenScope · **C: ALU + Register File**

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
- [ ] **Silicon side:** finish HDLBits modules 1-40 (combinational + sequential), build a 32-bit RV32I-subset ALU + 32×32 register file, run first Yosys synthesis and read the gate netlist

---

### Month 4 — Parameter-Efficient Fine-Tuning + Quantization
**Focus:** LoRA from scratch, QLoRA (NF4 + double quantization), prefix tuning, adapters, IA³, adapter merging (TIES, DARE), quantization theory (INT4/INT8/FP8, symmetric/asymmetric, per-tensor/per-channel/per-group), GPTQ, AWQ, GGUF/llama.cpp, FP8 Transformer Engine, CUDA kernels for quantized inference
**Silicon Day-6 Focus:** Configurable INT4/INT8/INT16 MAC units, packed-byte storage, on-the-fly dequantization, saturating accumulators, parameterized SystemVerilog. **Complement to LLM track:** Project A makes a model fast through software quantization; Project C builds the silicon that *makes that quantization fast in real hardware* — without it, INT4 weights would be slower than FP16.
**Projects:** A: QuantBench · B: LLM-Speedometer · **C: INT4/INT8 MAC Unit + Multi-Precision Datapath**

**After this month, you can:**
- [ ] Implement LoRA from scratch and apply it to any model's attention layers
- [ ] Fine-tune 7B+ parameter models with QLoRA on consumer GPUs (4-bit base + FP16 LoRA)
- [ ] Merge, swap, and stack multiple LoRA adapters for different tasks
- [ ] Quantize any HuggingFace model with GPTQ (4/8-bit), AWQ (4-bit), and GGUF (Q4_K_M through Q8_0)
- [ ] Use FP8 training with Transformer Engine on Hopper GPUs
- [ ] Write CUDA kernels for INT8 GEMM with dequantize-on-the-fly
- [ ] Produce comprehensive quality-vs-speed-vs-memory reports for any quantization method
- [ ] Build CLI tools that automate model compression workflows
- [ ] **Silicon side:** Design a runtime-switchable INT4/INT8/INT16 MAC unit with packed-byte storage and on-the-fly dequantization; report area in gate-equivalents per precision

---

### Month 5 — Inference Serving + RAG + Agents + NVIDIA Ecosystem
**Focus:** Inference analysis (prefill vs decode, TTFT, TPOT, throughput), TensorRT-LLM, vLLM, PagedAttention, speculative decoding, continuous batching, Triton Inference Server, NVIDIA Dynamo, vector search (FAISS GPU, cuVS), embeddings, RAG architecture, chunking, hybrid retrieval, re-ranking, Graph RAG, agentic RAG, Self-RAG, RAGAS evaluation, function calling, ReAct, agent memory, multi-agent systems, NIM, NeMo, Guardrails, Riva, ACE, AI Blueprints, NGC, build.nvidia.com
**Silicon Day-6 Focus:** Synchronous + asynchronous (CDC-safe) FIFO with Gray pointers, AXI4-Lite slave with CSRs, host-bus communication patterns. **Complement to LLM track:** vLLM's PagedAttention is a software-managed FIFO of memory pages — Project C builds the *hardware version of the same idea*. The serving stack you build in Project A is a thin abstraction over silicon AXI buses.
**Projects:** A: DeepRAG · B: RAGTrace · **C: FIFO + AXI-Lite — The Plumbing of an LLM Server**

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
- [ ] **Silicon side:** Build a synchronous FIFO + an async CDC-safe FIFO + an AXI4-Lite slave with 4 CSRs, all verified with constrained-random testbenches

---

### Month 6 — Distillation + RLHF + Reasoning + Long Context ⭐ (6-MONTH MILESTONE)
**Focus:** Knowledge distillation, model merging, synthetic instruction data, RL foundations, PPO/GRPO, RLHF pipeline, chain-of-thought, Tree of Thought, self-consistency, process reward models, long-context evaluation, RoPE scaling, KV-cache compression, Mamba/SSM overview
**Silicon Day-6 Focus:** 🔧 **FIRST FPGA BUILD** — Single-cycle RV32I CPU on FPGA running real C code. Cross-compile via `riscv32-gcc`, run Hello World over UART, run a small matmul in pure C. This is the silicon counterpart of nanoLLM — your *own* CPU you built and put on real hardware.
**Projects:** A: AgentForge ⭐ · B: EvalArena · **C: Single-Cycle RV32I CPU on FPGA** 🔧

**After this month, you can:**
- [ ] Distill a larger model into a smaller one and measure quality/speed tradeoffs
- [ ] Merge fine-tuned models using SLERP, TIES, DARE, or model soups
- [ ] Explain RLHF from SFT → reward model → PPO/GRPO with KL control
- [ ] Implement reasoning methods: CoT, self-consistency, Tree of Thought, verifier scoring
- [ ] Build agents that combine tools, memory, retrieval, and explicit reasoning strategies
- [ ] Evaluate long-context behavior with NIAH/passkey/document-QA style tests
- [ ] Explain RoPE scaling, KV-cache compression, and when Mamba/SSM-style models matter
- [ ] **Silicon milestone:** Run YOUR OWN CPU (single-cycle RV32I) on a real FPGA board, with C code printing Hello World over UART
- [ ] **You now have 18 monthly capstones (Project A/B/C × 6 months) plus weekly builds, and can build real AI applications end-to-end with hardware grounding**

---

### Month 7 — Multi-Modal AI + Diffusion Overview
**Focus:** CLIP (contrastive image-text), Vision Transformers (ViT), LLaVA (visual instruction tuning), document understanding, video understanding, audio (Whisper, Riva), DALI for image loading, diffusion models overview (DDPM, latent diffusion, Stable Diffusion), DiT
**Silicon Day-6 Focus:** Pipelined RV32I (5-stage IF/ID/EX/MEM/WB) with hazard detection + forwarding + branch flush. The same C matmul from M6 should now run ~4× faster — measure exactly. **Complement to LLM track:** Project A chains GPU pipelines (preprocess → embed → retrieve → generate); Project C builds the *original CPU pipeline* — the universal pattern.
**Projects:** A: IncidentLens · B: GPU PDF Intelligence · **C: 5-Stage Pipelined RV32I**

**After this month, you can:**
- [ ] Build AI that understands images, documents, audio, and video in a single conversation
- [ ] Fine-tune CLIP on custom datasets and build vision-language models (CLIP + projection + LLM)
- [ ] Extract information from PDFs with tables, charts, and images
- [ ] Understand how diffusion models work (forward/reverse process, latent space, guidance)
- [ ] Build multi-modal chatbots with automatic modality routing
- [ ] **Silicon side:** Run a pipelined RV32I (with hazard + forwarding) on FPGA, verify bit-exact against Spike for 1000 random programs, demonstrate ~4× IPC improvement vs single-cycle

---

### Month 8 — Mixture of Experts + Advanced Architectures
**Focus:** MoE Transformer implementation, expert parallelism, load balancing loss, router architectures (hash, learned, expert-choice), DeepSeek-V3 architecture (Multi-head Latent Attention, DeepSeekMoE), sparse vs dense MoE, auxiliary losses, expert communication patterns
**Silicon Day-6 Focus:** Add **direct-mapped I-cache and D-cache** (4 KB each) to the pipelined CPU, with write-back policy + AXI memory bus. Performance counters expose hit rate. **Complement to LLM track:** Project A and B compare model architectures running on GPUs with deep memory hierarchies; Project C builds that hierarchy from scratch — "L1 cache miss" becomes a thing visible in your waveforms.
**Projects:** A: MoE-Lab · B: ArchSearch · **C: I-Cache + D-Cache + AXI Memory Subsystem**

**After this month, you can:**
- [ ] Implement MoE Transformers with top-k routing and load balancing
- [ ] Visualize which experts specialize in which types of input (code, math, language)
- [ ] Implement Multi-head Latent Attention (MLA) from DeepSeek-V3
- [ ] Prune experts and measure quality impact
- [ ] Compare MoE vs dense models at equal active parameter counts
- [ ] Profile and optimize expert routing communication on GPU
- [ ] Understand how frontier models (Mixtral, DeepSeek-V3, GPT-4) work internally
- [ ] **Silicon side:** Add I-cache + D-cache to your pipelined CPU, measure hit rate / IPC improvement, expose performance counters via CSRs

---

### Month 9 — Advanced Kernel Engineering + Compiler Optimization
**Focus:** Triton compiler internals, torch.compile (TorchDynamo graph capture + TorchInductor code gen), operator fusion patterns, kernel autotuning, CUDA cooperative groups, warp specialization (Hopper), fused training kernels, gradient compression
**Silicon Day-6 Focus:** 🔧 **FIRST AI ACCELERATOR** — A 4×4 INT8 systolic array of MAC units, weight-stationary dataflow, AXI4-Lite controlled, BRAM-buffered, running on FPGA. Compare cycles to your M7 CPU running pure-C matmul: ~50–100× speedup expected. **Complement to LLM track:** Project A writes Triton kernels that target Tensor Cores; Project C builds the silicon *that Tensor Cores actually are*.
**Projects:** A: KernelSmith · B: FusionLab · **C: 4×4 INT8 Systolic GEMM on FPGA** 🔧

**After this month, you can:**
- [ ] Write production-quality Triton kernels for all common LLM operations
- [ ] Build: fused attention, fused RMSNorm, fused SwiGLU, fused RoPE, fused cross-entropy, fused AdamW
- [ ] Understand torch.compile internals and how TorchInductor generates GPU code
- [ ] Automate kernel benchmarking with CI (every kernel tested against PyTorch native)
- [ ] Build drop-in `nn.Module` replacements that are measurably faster than stock PyTorch
- [ ] Implement gradient compression for communication optimization
- [ ] **Your kernels are pip-installable and usable by anyone**
- [ ] **Silicon milestone:** 4×4 INT8 systolic array running on FPGA, measured ~50–100× speedup vs your CPU C-loop matmul

---

### Month 10 — Distributed Training at Scale + Production Systems
**Focus:** Megatron-LM (3D parallelism: DP+TP+PP, sequence parallelism, context parallelism), DeepSpeed ZeRO-3, FSDP2, distributed checkpointing, training failure diagnosis, evaluation frameworks (lm-eval-harness, HELM, MMLU, HumanEval, MT-Bench), production MLOps, auto-scaling, A/B testing, NVIDIA AI Enterprise
**Silicon Day-6 Focus:** 2D mesh **NoC** with virtual channels and credit-based flow control, connecting 4 of your M9 systolic tiles, with hardware all-reduce. **Complement to LLM track:** Project A does AllReduce over NCCL/NVLink across GPUs; Project C does AllReduce over a NoC across silicon tiles — the *same algorithm at a different scale*.
**Projects:** A: TrainScale · B: TrainWatch · **C: Mesh NoC for Multi-Tile Accelerator**

**After this month, you can:**
- [ ] Configure Megatron-LM for large-scale multi-node training
- [ ] Train models larger than GPU memory using DeepSpeed ZeRO-3 and FSDP2
- [ ] Implement fault-tolerant distributed checkpointing with resume
- [ ] Build training cost estimators: predict time, cost, optimal DP/TP/PP config before training
- [ ] Evaluate models comprehensively (MMLU, HumanEval, GSM8K, MT-Bench, and more)
- [ ] Build production serving with auto-scaling, monitoring, and alerting
- [ ] Implement semantic caching and multi-model routing (easy → small, hard → large)
- [ ] Diagnose common training failures: loss spikes, gradient explosions, OOM
- [ ] **Silicon side:** A 2×2 mesh NoC with XY routing, virtual channels, and a hardware all-reduce primitive across 4 of your systolic tiles

---

### Month 11 — Research Paper Implementation + Data Engineering
**Focus:** Paper reading technique (figures → method → experiments), implementing algorithms from scratch, reproducing results, synthetic data pipelines (Self-Instruct, Evol-Instruct), NeMo Curator (GPU-accelerated curation), constrained decoding (outlines, GBNF), function calling training, code generation, Text-to-SQL
**Silicon Day-6 Focus:** Reproduce a hardware paper — implement the **Eyeriss row-stationary PE** (Chen et al., ISCA 2016), compare its energy/cycle profile against your M9 weight-stationary array. Begin a weekly arxiv hardware-paper feed (cs.AR + select cs.LG hardware). **Complement to LLM track:** Project A reproduces ML papers; Project C reproduces a hardware paper — same skill at a different layer.
**Projects:** A: PaperBot · B: Paper2Benchmark · **C: Eyeriss-style Row-Stationary PE**

**After this month, you can:**
- [ ] Read any AI paper and extract the key algorithm into working code
- [ ] Implement novel architectures and training techniques from publications
- [ ] Reproduce experimental results and identify when papers are misleading
- [ ] Build GPU-accelerated data curation pipelines: scrape → clean → dedupe → filter → tokenize
- [ ] Generate high-quality synthetic instruction and preference data
- [ ] Implement grammar-constrained generation (guaranteed valid JSON output)
- [ ] Train function-calling and code generation models
- [ ] **Silicon side:** Reproduce Eyeriss-style row-stationary PE in SystemVerilog and write a comparison report against your weight-stationary array
- [ ] **You can now turn any paper — ML or hardware — into a working prototype**

---

### Month 12 — Test-Time Compute + Reasoning + RL for LLMs
**Focus:** RL foundations (MDP, policy gradient, REINFORCE, Actor-Critic, advantage, GAE), PPO, GRPO (DeepSeek), full RLHF pipeline (SFT → reward model → PPO + KL penalty), RLAIF, process reward models, chain-of-thought (zero-shot, few-shot), Tree of Thought, self-consistency, STaR, test-time compute scaling, MCTS for reasoning, adaptive compute allocation, DeepSeek-R1
**Silicon Day-6 Focus:** Scale your systolic array to **8×8 INT8 with both weight-stationary and output-stationary modes**, write a **PyTorch C++ extension** that drives the FPGA from `torch.matmul` and runs a real LLaMA-style MLP weight matrix through it. **Complement to LLM track:** Project A pushes more inference work onto silicon (test-time compute); Project C is the silicon that benefits — and it's the first time PyTorch on your laptop calls *your* chip.
**Projects:** A: ReasonEngine · B: ThinkTrace · **C: 8×8 Systolic Array + PyTorch Bridge**

**After this month, you can:**
- [ ] Implement PPO and GRPO from scratch (RL fundamentals + LLM application)
- [ ] Build full RLHF pipelines: SFT → reward model → PPO with KL penalty
- [ ] Train process reward models that score individual reasoning steps
- [ ] Build MCTS-based reasoning systems for hard math/logic problems
- [ ] Implement adaptive compute: easy question = 1 pass, hard question = 64 samples + verification
- [ ] Show accuracy vs compute curves (more inference compute = better answers)
- [ ] Understand how reasoning models (o1, DeepSeek-R1) achieve step-by-step thinking
- [ ] **Silicon side:** 8×8 systolic array with selectable dataflow modes, integrated as a PyTorch C++ extension, running a real LLM MLP weight matmul end-to-end
- [ ] **You can make any LLM dramatically smarter on hard problems — and run a real LLM op on silicon you built**

---

### Month 13 — Production LLM Serving + Advanced Alignment + Enterprise Agents
**Focus:** NIM auto-scaling, multi-model serving, semantic caching, A/B testing with statistical significance, Guardrails advanced (Colang 2.0), NVIDIA AIQ enterprise patterns, MCP (Model Context Protocol), agent fine-tuning, tool-use training, advanced alignment (SPIN, ORPO, SimPO, iterative DPO), reward hacking prevention
**Silicon Day-6 Focus:** Wrap your accelerator in a **production-quality control plane**: AXI4-full DMA, CSRs, interrupt line, and your **first UVM-Lite testbench** with constrained-random + functional coverage + SVA. **Complement to LLM track:** Project A is production-grade LLM serving; Project C makes your hardware production-grade with the verification rigor real silicon teams use.
**Projects:** A: NVServe · B: CostGuard · **C: Production Wrapper + UVM Testbench (Tier 1)**

**After this month, you can:**
- [ ] Build production-grade serving with auto-scaling on Kubernetes
- [ ] Route requests across multiple models based on complexity, cost, or latency targets
- [ ] A/B test model versions with statistical significance
- [ ] Implement MCP servers and clients for standardized tool interfaces
- [ ] Compare 5+ alignment methods (DPO, PPO, GRPO, ORPO, SimPO) on the same base model
- [ ] Detect and prevent reward hacking in alignment training
- [ ] Build self-improving model pipelines (generate → evaluate → retrain)
- [ ] **Silicon side:** UVM-Lite verification of your accelerator with ≥90% functional coverage and your first SVA properties
- [ ] **You can deploy and manage LLMs — and silicon — at enterprise scale**

---

### Month 14 — Synthetic Data Engineering + Advanced RAG
**Focus:** Quality-aware synthetic data generation loops, data contamination detection, NeMo Curator at scale (dedup, PII, quality scoring), advanced retrieval (ColBERT, SPLADE, RAPTOR), multi-hop RAG, structured data RAG (SQL generation, table retrieval), RAG + tool use, domain adaptation
**Silicon Day-6 Focus:** **Formal verification with SymbiYosys** + add **2:4 structured sparsity hardware** (the same Ampere/Hopper feature) to your systolic array. **Complement to LLM track:** Project A and B engineer data integrity; Project C engineers *silicon integrity* — formal proofs find corner-case bugs your testbench missed.
**Projects:** A: SynthData · B: DataScope · **C: Formal Verification + 2:4 Sparse GEMM**

**After this month, you can:**
- [ ] Build quality-aware data generation loops: generate → score → filter → iterate
- [ ] Use NeMo Curator for GPU-accelerated deduplication, PII filtering, quality scoring
- [ ] Detect data contamination in training sets
- [ ] Build state-of-the-art RAG with ColBERT late interaction and RAPTOR hierarchical retrieval
- [ ] Handle multi-hop questions requiring multiple retrieval steps
- [ ] Demonstrate measurable model improvement from synthetic data vs human-curated data
- [ ] Build domain-specific data pipelines from seed examples to full training set
- [ ] **Silicon side:** Formal proofs on your AXI handshake (with at least 1 caught bug) + a 2:4 sparse mode in the systolic array showing ~2× throughput
- [ ] **Your synthetic data is good enough to train production models, your silicon is provably correct**

---

### Month 15 — Enterprise AI Agents (Full NVIDIA Stack)
**Focus:** NIM (LLM + embedding + reranking), NeMo Guardrails (Colang 2.0), NVIDIA AIQ orchestration, Riva voice I/O, multi-agent coordination, agent capability benchmarking across 10 dimensions, advanced evaluation
**Silicon Day-6 Focus:** **FlashAttention-inspired hardware block** — a tile-based attention engine in SystemVerilog with online softmax, INT8/FP16 modes, running a real LLaMA-style attention layer on FPGA. **Complement to LLM track:** Project A is the headline LLM enterprise agent; Project C is the headline silicon block that an LLM agent would call as a tool — the link is concrete.
**Projects:** A: AgentX · B: AgentEval · **C: FlashAttention Hardware Block**

**After this month, you can:**
- [ ] Build enterprise-grade AI agents using every relevant NVIDIA tool simultaneously
- [ ] Design and run 50-question agent capability benchmarks across 10 dimensions
- [ ] Build voice-enabled agents: Riva ASR → LLM NIM → Riva TTS
- [ ] Orchestrate specialized sub-agents for different task types
- [ ] Generate radar chart visualizations of agent capabilities vs baselines
- [ ] Deploy production agents with monitoring, safety rails, and per-task cost tracking
- [ ] **Silicon side:** A FlashAttention-inspired tile-based attention engine in RTL, running a real 4-head × 64-dim attention layer on FPGA, verified against PyTorch
- [ ] **You can architect and deliver complete AI agent solutions for enterprise use, with silicon co-design**

---

### Month 16 — Open-Source Contributions + First Real Synthesis Run
**Focus:** Contributing to major AI projects (vLLM, TensorRT-LLM, NeMo, HuggingFace transformers/trl/PEFT), navigating large codebases, finding and fixing performance issues, technical writing
**Silicon Day-6 Focus:** **First Yosys synthesis on Sky130 PDK + OpenSTA** with realistic SDC constraints, MCMM corners. Your accelerator is no longer "FPGA-only" — it's *physically synthesizable on real silicon*.
**Projects:** A: OpenContrib · B: 2nd OSS · **C: First Yosys+OpenSTA Synthesis Run on Sky130**

**After this month, you can:**
- [ ] Navigate and understand codebases with 100K+ lines (vLLM, Megatron, NeMo)
- [ ] Identify performance bottlenecks in production AI tools
- [ ] Submit meaningful PRs that get merged into projects used by thousands of developers
- [ ] Write technical blog posts that explain complex systems clearly
- [ ] **Silicon side:** Run Yosys + OpenSTA on Sky130 PDK with full timing/area/power reports for your 8×8 systolic array
- [ ] **You are a recognized contributor to the AI open-source ecosystem and have produced your first synthesis-ready chip block**

---

### Month 17 — Domain-Specific LLMs + First Physical Layout (Sky130)
**Focus (LLM):** Multi-language and domain pre-training, BPE training, continued pre-training, domain SFT/DPO, domain eval. Long-context extension via RoPE scaling / YaRN / LongRoPE, NIAH eval, NVMe-backed paged KV-cache.
**Silicon Day-6 Focus:** **First full OpenLane / LibreLane RTL→GDSII run** on Sky130 — floorplan, placement, CTS, routing, viewable in KLayout. Compare synthesis vs post-layout timing.
**Projects:** A: PolyglotLM · B: ContextWeaver · **C: OpenLane RTL→GDSII Floorplan + Place-and-Route**

**After this month, you can:**
- [ ] Train domain-specific BPE tokenizers and continued-pre-train domain models
- [ ] Extend any open LLM to 1M-token context with reproducible eval
- [ ] Build NVMe-backed paged KV-cache for long-context serving
- [ ] **Silicon side:** Open KLayout, see your own chip layout, understand the post-layout timing penalty (real wires hurt — by exactly how much, in your design)

---

### Month 18 — Continual Learning + Alignment Bench + TinyTapeout Prep
**Focus (LLM):** LoRA-stack continual learning with regression gates, RAG-as-knowledge-extension. Alignment method showdown: DPO, IPO, ORPO, SimPO, KTO, RLAIF — same base, unified comparison.
**Silicon Day-6 Focus:** **TinyTapeout-shaped Sky130 block** (160 µm × 100 µm tile) — squeeze a small AI block into TT's tile constraints, get DRC + LVS clean, write the datasheet. This is the dress rehearsal for the Month 21 actual submission.
**Projects:** A: EvolveLM · B: AlignBench · **C: TinyTapeout Sky130 Block (Prep)**

**After this month, you can:**
- [ ] Run continual learning in production with regression-gated promotion
- [ ] Compare 6+ alignment methods on the same base model with cost/quality/safety metrics
- [ ] **Silicon side:** A DRC-clean, LVS-clean Sky130 block fitting TinyTapeout's tile budget, with a written datasheet — ready for shuttle submission

---

### Month 19 — Distillation + Energy Audit + Power Distribution Network
**Focus (LLM):** Multi-teacher distillation, attention transfer, feature distillation. Energy/carbon tracking for inference and training: kWh/run, gCO₂e/token.
**Silicon Day-6 Focus:** **Multi-block floorplan + power-ground (PG) grid + clock gating + IR drop estimation** on a multi-block design (GEMM + softmax + DMA + control).
**Projects:** A: DistillForge · B: EnergyAuditor · **C: Multi-Block Floorplan + PDN**

**After this month, you can:**
- [ ] Distill a small student from multiple specialized teachers
- [ ] Report energy and carbon per token across your inference stack
- [ ] **Silicon side:** A multi-block design with proper PG grid, clock gating, and IR-drop analysis — production chip-design hygiene

---

### Month 20 — Self-Hosted LLM Stack + Multi-Modal Doc Agent + Timing Closure
**Focus (LLM):** Turnkey self-hosted LLM platform (NIM/vLLM + RAG + agents + guardrails + monitoring + cost tracking) deployable as one Helm chart. Multi-modal document analysis agent with table/chart/image understanding.
**Silicon Day-6 Focus:** **Push your accelerator from 200 MHz → 400 MHz** with multi-cycle paths, retiming, register pipelining. Document every WNS-improving fix.
**Projects:** A: OpenLLM-Stack · B: DocAgent · **C: STA Mastery — Close Timing at 400 MHz**

**After this month, you can:**
- [ ] Deploy a turnkey self-hosted LLM platform with one Helm chart
- [ ] Build doc-QA agents that handle tables, charts, formulas, slide layouts
- [ ] **Silicon side:** Close timing at a real frequency target across SS/TT/FF corners — the bar a real chip team clears every day

---

### Month 21 — Reproducible Research + Sustained OSS + 🎉 TINYTAPEOUT SUBMITTED
**Focus (LLM):** Reproduce 3 recent (2026) papers end-to-end with Docker images. Sustained OSS contribution: 5+ merged PRs in one project this month + reviews on others' PRs.
**Silicon Day-6 Focus:** **Submit a real chip block to TinyTapeout shuttle** — polished RTL, datasheet written, integrated with TT I/O harness, GitHub Actions verification clean, **submission accepted**.
**Projects:** A: ResearchLab · B: OSS-Citizen · **C: 🎉 TinyTapeout Submission Live**

**After this month, you can:**
- [ ] Reproduce SOTA research papers reliably
- [ ] Sustain meaningful OSS contributions in a major project
- [ ] **Silicon milestone:** Your block is on a SkyWater wafer, fabrication in progress (you'll get the chip back ~6-9 months later)

---

### Months 22-24 — LLM Magnum Opus ⭐ (HARD DEADLINE BY END OF MONTH 24)
**Focus:** Combining all 21 months of LLM/CUDA skills into ONE definitive LLM project. Choose: LLM-from-Scratch-to-Production, AI Agent OS, GPU Inference Engine, or Edge LLM. Includes a "silicon hook" that connects to a Project C silicon block. Three months for the build, *six months* of buffer (Phase 5 and Months 19-21) make this comfortable, not rushed.
**Silicon Day-6 Focus:** Polish your accelerator to ship as the Magnum Opus's hardware companion (M22 polish, M23 DFT/scan-chain insertion + post-layout timing closure, M24 hardware-companion paper section).
**Projects:** A: LLM Magnum Opus ⭐ (build M22-23, ship M24) · **C: Co-design Block + DFT + Hardware Companion Paper**

**After these months, you can:**
- [ ] **LLM Magnum Opus shipped** — 2500-word blog post, demo video, GitHub repo, reproducibility report
- [ ] **Silicon side:** Production-quality accelerator with DFT (scan chains + ATPG ≥95%), post-layout timing closed, integrated with your Magnum Opus
- [ ] **You have a portfolio demonstrating mastery of CUDA + LLMs + production systems + NVIDIA stack + a real silicon block**
- [ ] **You can lead AI infrastructure projects at NVIDIA or any top AI company** — and reason about silicon co-design while doing so

---

### Months 25-30 — Phase 7: Silicon Deep Dive (LLM in Maintenance Mode)
> The LLM track is **done**. From here you keep up with papers (~2/week) and maintain your OSS commitments — that's 1–2 hrs/week max. The other 95% of your effort is full-time silicon mastery.

**Month 25 — Out-of-Order RV32I Core**: Tomasulo-style register renaming, reservation stations, 8-entry reorder buffer; verified vs Spike on benchmarks.
**Month 26 — Branch Predictor Lab + MESI Coherence**: gshare → TAGE-style branch predictor; 2-core MESI cache coherence in SystemVerilog.
**Month 27 — Transformer-Block Accelerator on FPGA**: Full transformer layer (multi-head attention + FFN + LayerNorm + residual + RoPE) in RTL, running a quantized LLaMA-style block on FPGA at measurable tokens/sec. **Headline silicon artifact before the Magnum Opus.**
**Month 28 — Architecture Comparison Paper**: Use Timeloop + Accelergy to model H100 vs MI300X vs TPUv5 vs Sohu on a 70B-parameter LLM workload. Write a 6000-word arxiv-ready paper.
**Month 29 — Compiler for Your Accelerator**: A small MLIR dialect (or TVM schedule) that lowers a transformer-block IR onto your Month 27 accelerator.
**Month 30 — Chiplet Design Study**: Compute die + simulated HBM-like die, UCIe-style die-to-die interface, performance projection.

**After Phase 7, you can:**
- [ ] Build advanced microarchitecture (out-of-order, branch predictors, cache coherence)
- [ ] Build a *full transformer-block accelerator* on FPGA running real LLM ops
- [ ] Model and compare commercial AI chips at architectural level (Timeloop / Accelergy)
- [ ] Build a custom MLIR/TVM compiler for your hardware
- [ ] Design chiplets and die-to-die interfaces

---

### Months 31-36 — Phase 8: Silicon Magnum Opus ⭐⭐⭐ (FINAL — by end of Month 36)
**Focus:** Six-month build of your **custom AI inference accelerator** — the chip-design career piece you take to Etched, Tenstorrent, Cerebras, Groq, or NVIDIA architecture interviews.

**Choose ONE primary architecture:** TinyTransformer Engine (decode-optimized) · FlashAttention ASIC · Spatial MoE Router · CIM-Inspired LLM Decode Block.

**Required deliverables:** architecture spec doc (3000+ words) · synthesizable RTL (10K+ lines) · UVM verification (≥95% coverage) · formal proofs on critical interfaces · FPGA prototype running real LLaMA workload at measurable tokens/sec · PyTorch backend integration · OpenLane RTL→GDSII signoff on Sky130 · DFT (scan + ATPG ≥95%) · 8000-word architecture comparison paper · open-source GitHub release · 10-15 min demo video · **(optional)** Efabless chipIgnite tape-out (~$10K) — and **(optional)** characterize your TinyTapeout chip from Month 21 returning from SkyWater.

**After Phase 8, you can:**
- [ ] Design, verify, and physically implement an AI accelerator end-to-end on real silicon
- [ ] Reason fluently about transformer math → CUDA → microarchitecture → RTL → silicon
- [ ] Walk into any AI-chip company (NVIDIA, Etched, Tenstorrent, Cerebras, Groq, AMD, Google TPU, AWS Trainium) at the senior-architect level — not entry-level
- [ ] **Submit your architecture comparison paper to Hot Chips / arxiv preprint / ISSCC**
- [ ] **You have built the AI stack from electrons to GPT-4-class systems, with your own hands**

---

# ═══════════════════════════════════════════════
# 🔧 SILICON TRACK — HOW IT WEAVES INTO EVERY WEEK
# ═══════════════════════════════════════════════

> **Starts Week 3 (after you finish Week 2). Runs through Week 104. Then takes over fully Weeks 105-156.**

Each week's Saturday morning (or Friday evening — your choice) you spend **1.5-2 hours on Day 6 — Silicon**. The topic for that day is the silicon-level explanation of the LLM/CUDA topic that week. **It's the same lesson at a deeper layer**, never disconnected.

**Tools needed (all free / open-source):**
- Simulation: **Icarus Verilog**, **Verilator**, **GTKWave**
- FPGA toolchain: **Vivado WebPACK** (Xilinx) or **Gowin EDA** (Tang Nano)
- Open-source synthesis & physical design: **Yosys**, **OpenSTA**, **OpenROAD**, **OpenLane / LibreLane**, **Magic**, **KLayout**, **Netgen**
- PDK: **SkyWater Sky130** (open-source, real fabrication possible via TinyTapeout / Efabless)
- Verification: **CocoTB**, **SymbiYosys** (formal), **UVM-1.2/2017** (later)
- Practice site: **HDLBits** (https://hdlbits.01xz.net) — solve modules every Day 6 from Week 3-30

**Hardware needed:**
- An FPGA dev board around Month 6-8. Recommended: **Digilent Arty A7-100T** (~$220) or **Nexys A7-100T** (~$320), or for tighter budget **Sipeed Tang Nano 20K** (~$30) or **Pynq-Z2** (~$200)
- USB-UART cable (most boards include one)
- Optional later (Month 21-22): **TinyTapeout submission slot** (~$300, real silicon comes back)
- Optional final (Month 36): **Efabless chipIgnite tape-out** (~$10K, your final silicon)

**Books (free or cheap):**
- *Digital Design and Computer Architecture, RISC-V Edition* — Harris & Harris (Months 2-9)
- *Computer Organization and Design RISC-V* — Patterson & Hennessy (Months 6-15)
- *SystemVerilog for Verification* — Chris Spear (Months 13-18)
- *Efficient Processing of Deep Neural Networks* — Sze, Chen, Yang, Emer (Months 14-30)
- *Computer Architecture: A Quantitative Approach* (6th/7th ed) — Hennessy & Patterson (Months 25-36)
- *VLSI Physical Design: From Graph Partitioning to Timing Closure* — Kahng et al. (Months 17-30)

**Free YouTube courses to follow alongside:**
- Onur Mutlu's *Computer Architecture* (CMU/ETH Zürich) — full lectures, the deepest free course on the planet
- MIT 6.5930 *Hardware Architecture for Deep Learning* (Vivienne Sze)
- Berkeley CS152 / CS252
- *nand2tetris* (build a computer from NAND gates up — fast warm-up, ~10 hrs)

**Day 6 weekly cadence (Week 3 → Week 30):**
1. ~30 min reading from Harris & Harris (or Patterson & Hennessy after M9)
2. ~45 min HDLBits problems for the week's topic (or guided Verilog tutorial)
3. ~30 min connecting back to LLM track: write 2-3 sentences in your engineering journal: *"This week's silicon topic explains [what] in this week's LLM/CUDA topic"*

**From Week 31 onwards** (Phase 4-6 silicon work), Day 6 grows to ~3 hours and absorbs more of Saturday because you are building real accelerators (UVM testbenches, OpenLane runs, FPGA prototypes).

**From Week 105 onwards** (Phase 7-8), the entire weekday cadence flips — 5 days are silicon, 1 day is LLM-paper-reading + OSS.

---

# ═══════════════════════════════════════════════
# PHASE 1: FOUNDATIONS (Weeks 1-16, Months 1-4)
# GPU Architecture, CUDA, Neural Nets, Transformers (LLM)
# Digital Logic + first synthesizable Verilog (Silicon, Week 3+)
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

### 🔧 Day 6 — Silicon: Digital Logic Foundations (START OF SILICON TRACK)
> **Connection to LLM track:** You just spent 5 days learning how warps execute reductions and atomics on millions of transistors. Day 6 zooms into a single transistor and the gates around it — because everything you did this week ultimately runs on this layer.
- [ ] Read Harris & Harris **Ch 1** (binary, hex, two's complement, Boolean algebra, K-maps) — ~45 min
- [ ] HDLBits modules **1 ("Getting Started") through 5 ("Vectors")** — solve all of them in your browser
- [ ] First Verilog: write a 4-bit MUX module + testbench, simulate with Icarus, view in GTKWave
- [ ] Engineering journal: 3 sentences on how a half-adder is the smallest version of `atomicAdd`

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

### 🔧 Day 6 — Silicon: Combinational Building Blocks (Adders & Multipliers)
> **Connection to LLM track:** CUDA streams overlap kernel execution; on the silicon side, every kernel ultimately fires adder + multiplier units inside the SM. Today you build them in Verilog.
- [ ] Read Harris & Harris **Ch 2 (combinational logic) + Ch 5.2 (arithmetic circuits)** — ~45 min
- [ ] HDLBits modules **6-15 ("Modules", "Procedures", "More Verilog Features")** — solve all
- [ ] Write a **half-adder + full-adder + 4-bit ripple-carry adder** in Verilog with testbench
- [ ] Bonus: read about Kogge-Stone fast adders (don't implement, just understand why they exist)
- [ ] Engineering journal: 3 sentences on how a 32-bit adder is what FP32 mantissa addition uses inside an FMA unit

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
- [ ] **Silicon onboarding:** finish FPGA board ordering + toolchain install (if not already done)
- [ ] **Build Monthly Project A:** GPU Matrix Math Engine
- [ ] **Build Monthly Project B:** LLM Inference Cost Simulator
- [ ] **Build Monthly Project C:** *(none yet — Project C starts at Month 2)*
- [ ] Push all projects to GitHub

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

### 🔧 Day 6 — Silicon: Sequential Logic + FSMs + The 4-bit Multiplier
> **Connection to LLM track:** The Transformer block has stateful pieces (KV-cache, residual stream); silicon expresses state as flip-flops + FSMs. The autoregressive decoder is conceptually a finite state machine, ticking forward one token per cycle.
- [ ] Read Harris & Harris **Ch 3 (sequential logic) + Ch 4 (HDL)** — ~45 min
- [ ] HDLBits modules **16-30 ("Multiplexers", "Latches & FFs", "Counters", "FSMs")** — solve all
- [ ] Write a **4-bit shift-add multiplier** in Verilog (sequential, multi-cycle) with testbench
- [ ] Write a **traffic-light FSM** (Moore-style) with testbench
- [ ] Engineering journal: 3 sentences on how the autoregressive decoder loop is just an FSM

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

### 🔧 Day 6 — Silicon: The MAC Unit (a Tensor Core in 1 Page)
> **Connection to LLM track:** Today's LLM track is cuBLAS + cuDNN + **Tensor Cores via WMMA**. Today's silicon track builds the *primitive that a Tensor Core is made of*: a multiply-accumulate (MAC) unit. After today, when you write `mma.sync.aligned`, you'll picture exactly what circuit it lights up.
- [ ] Read Harris & Harris **Ch 5 sections on multiplication + multiply-accumulate** — ~30 min
- [ ] Read the *VLSIFacts* "Matrix Multiply Unit Architecture" article (10 min)
- [ ] Write a **1-bit MAC unit** in Verilog (a × b + c → next c), with testbench
- [ ] Extend it to **8-bit signed MAC** with saturating accumulate
- [ ] Hand-draw a schematic of an **N-bit MAC array** (your own "Tensor Core in 1 page")
- [ ] **Project C — Month 2 — START:** publish your repo `silicon/month02/` with adder, multiplier, and 1-bit MAC

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
- [ ] **Silicon paper (optional):** "Matrix Multiply Unit: Architecture, Pipelining, and Verification" — 30 min, ties back to today's MAC

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

### 🔧 Day 6 — Silicon: Static Timing Analysis (Setup, Hold, Slack, Critical Path)
> **Connection to LLM track:** Today's Nsight kernel-level "speed of light" analysis asks: "what's the slowest path through the kernel?" — that's the *exact same question* STA asks at the gate level: "what's the longest path between two flip-flops?" Both define maximum frequency. The vocabulary is identical: critical path, slack, throughput.
- [ ] Read Harris & Harris **Ch 3.5 (timing & metastability)** — ~30 min
- [ ] Read the *VLSIFacts* "Setup-Hold Time Tutorial" — 15 min
- [ ] HDLBits modules **31-40 ("Counters", "Shift Registers", "FSMs")** — solve all
- [ ] Hand-compute setup/hold slack on a 3-stage pipeline: clock period 5 ns, FF Tcq=0.5 ns, comb delay=3 ns, FF setup=0.4 ns → what's the slack?
- [ ] **Project C — Month 3 — START:** begin the 32-bit RV32I-subset ALU + 32×32 register file (2 weeks of Day 6)

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

### 🔧 Day 6 — Silicon: SRAM Cells, DRAM, HBM Stack — Why Memory Matters
> **Connection to LLM track:** This week's LLM topic is HBM2/HBM3 bandwidth, NVLink, and the GPU memory architecture. Day 6 zooms into the *physics* of those memories: why a 6T SRAM cell is fast and small, why DRAM is dense but slow, why HBM stacks dies vertically to get more bandwidth per mm² than any flat die can.
- [ ] Read *Efficient Processing of Deep Neural Networks* (Sze et al.) **memory chapter** — ~45 min
- [ ] Read the NVIDIA Blackwell whitepaper section on HBM3 + memory subsystem — ~30 min (the PDF you saved earlier)
- [ ] Hand-draw a 6T SRAM cell + 1T1C DRAM cell; explain why DRAM needs refresh
- [ ] HDLBits modules **41-50 ("Memory", "Larger Circuits")** — solve all
- [ ] Engineering journal: 3 sentences on why HBM3 vs DDR5 changes what kernels are memory-bound

### 🔨 Saturday Project
- [ ] **Mixed Precision Training Benchmark**
  - [ ] Train same model: FP32, FP16+AMP, BF16
  - [ ] Measure: training speed, memory usage, final accuracy
  - [ ] Profile Tensor Core utilization with Nsight
  - [ ] Generate comprehensive comparison report

### 📄 Sunday Paper
- [ ] Skim: "Efficient Processing of Deep Neural Networks" (Sze et al., 2020) — hardware chapter
- [ ] **Silicon paper (optional):** "Microbenchmarking NVIDIA's Blackwell Architecture" (arXiv 2512.02189) — 30 min

---

## 🔄 Buffer Week (Month 2 Revision)
- [ ] Revise Transformer architecture: attention, multi-head, positional encoding, FFN
- [ ] Revise GPU libraries: cuBLAS, Tensor Cores, Nsight profiling
- [ ] Revise LLaMA architecture: RMSNorm, RoPE, SwiGLU, GQA
- [ ] Re-read your GPT-2 implementation, understand every line
- [ ] **Build Monthly Project A:** nanoLLM — GPT from Scratch
- [ ] **Build Monthly Project B:** TransformerScope — LLM Behavior Debugger
- [ ] **Build Monthly Project C — Silicon:** Logic-Lab — Gates to GEMM (`silicon/month02/`)
- [ ] Push all 3 projects to GitHub

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

### 🔧 Day 6 — Silicon: NoC Topologies + Why NVSwitch Looks the Way It Does
> **Connection to LLM track:** Today's LLM track is NCCL, AllReduce, NVLink/NVSwitch — multi-GPU communication. Day 6 explains the *silicon* of multi-chip communication: NoC topologies (mesh, ring, torus, dragonfly), routing, virtual channels, deadlock avoidance. NVSwitch is a giant crossbar in silicon; UCIe is a die-to-die PHY. Same concepts, different scale.
- [ ] Read *Efficient Processing of DNNs* — **NoC chapter** (~30 min)
- [ ] Skim a recent UCIe / NVSwitch architecture article (~15 min)
- [ ] Sketch a 2×2 mesh router (input ports, crossbar, output buffers, credit logic) — pen and paper only
- [ ] HDLBits — work on the **Larger Circuits** problems
- [ ] **Project C — Month 3 — DEEP WORK:** continue the 32-bit ALU + register file. Add a constrained-random testbench checking against a Python reference for 100K random instructions

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

### 🔧 Day 6 — Silicon: Custom CUDA Extension ↔ Custom HW Block — The Same Idea Twice
> **Connection to LLM track:** Today's LLM track wrote a custom CUDA extension callable from PyTorch (`torch.utils.cpp_extension`). On Day 6 you build the silicon equivalent: a **memory-mapped accelerator block** that another module (eventually a CPU) writes to via AXI registers and reads results back. Same pattern: extend the platform with a new operator.
- [ ] Read Harris & Harris **Ch 8 (Memory and IO Systems)** — ~30 min
- [ ] HDLBits — finish remaining sequential / FSM problems
- [ ] Write a **memory-mapped multiplier accelerator**: AXI-Lite slave with 3 registers (operand A, operand B, result) — write A and B, kick off, read result
- [ ] **Project C — Month 3 — FINISH:** wrap up the 32-bit ALU + register file project, run **Yosys synthesis**, count gates, publish to `silicon/month03/`

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

### 🔧 Day 6 — Silicon: Dataflow Architectures (Eyeriss) — The Idea That Inspired Flash Attention
> **Connection to LLM track:** Flash Attention's whole insight is "tile and reuse SRAM" — that exact idea was published in 2016 as **Eyeriss row-stationary dataflow** for CNN accelerators. Flash Attention is FlashAttention because Tri Dao read the Eyeriss paper. Today you read it too.
- [ ] Read **Eyeriss paper** (Chen et al., ISCA 2016) — ~60 min, take notes on the 5 dataflow categories (Weight-Stationary, Output-Stationary, Input-Stationary, Row-Stationary, No-Local-Reuse)
- [ ] Sketch (pen + paper) a 4×4 Weight-Stationary array doing matrix multiply. Trace where each input/weight/output goes per cycle.
- [ ] Compare with Flash Attention's tile flow (Q tile, K tile, V tile, online softmax) — write 5 sentences mapping FA tiles to Eyeriss dataflows
- [ ] HDLBits — finish the **Verification: Reading Simulations** problems
- [ ] Engineering journal: "Why FlashAttention is just Eyeriss for attention"

### 🔨 Saturday Project
- [ ] **Triton Kernel Library**
  - [ ] Fused softmax
  - [ ] Fused LayerNorm
  - [ ] Fused attention (simplified Flash Attention)
  - [ ] Benchmark all vs PyTorch native

### 📄 Sunday Papers
- [ ] "FlashAttention: Fast and Memory-Efficient Exact Attention" (Dao, 2022)
- [ ] "Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations" (Tillet, 2019)
- [ ] **Silicon paper:** "Eyeriss: A Spatial Architecture for Energy-Efficient Dataflow for CNNs" (Chen et al., ISCA 2016)

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

### 🔧 Day 6 — Silicon: AXI-Lite Slave + FIFO — The Plumbing Behind Every Accelerator
> **Connection to LLM track:** Today's LLM track is unified memory + advanced allocation; on the silicon side, every accelerator that talks to a CPU does so via AXI buses and FIFOs. You'll need this for all future Project C builds.
- [ ] Read the **AMBA AXI4-Lite spec** Sections 1-3 (~30 min, available free from ARM)
- [ ] Read about Gray-code pointers for async FIFOs (~15 min)
- [ ] Write a **synchronous FIFO** with full/empty flags + testbench
- [ ] **Project C — Month 5 — START** (taking advantage of Buffer Week extension if needed)

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
- [ ] Write summary of key learnings (LLM + silicon)

---

### ✅ Phase 1 Completion Checklist (LLM track ~ 3 months / Silicon track ~ 2.5 months)
**LLM side:**
- [ ] Can explain GPU SM architecture from memory
- [ ] Can write and optimize CUDA kernels (shared memory, coalescing, reduction)
- [ ] Can implement Transformer from scratch
- [ ] Can train a small LLM (GPT-2 style)
- [ ] Can use Nsight Systems/Compute for profiling
- [ ] Understand FP32/FP16/BF16 and Tensor Cores
- [ ] Can fine-tune models with SFT and DPO

**Silicon side:**
- [ ] Can write synthesizable Verilog from scratch (modules, FSMs, pipelined designs)
- [ ] HDLBits modules 1-50 done (combinational + sequential basics)
- [ ] Built: half/full adder, 4-bit multiplier, 1-bit MAC, 32-bit ALU, 32×32 register file, 4-bit MUX/decoder, FIFO
- [ ] Can explain setup/hold timing, slack, critical path in your own words
- [ ] First Yosys synthesis run, gate count understood
- [ ] Read Eyeriss + the Blackwell architecture deep-dive
- [ ] FPGA board has arrived and toolchain is installed (ready for Phase 2 single-cycle CPU build)

---

# ═══════════════════════════════════════════════════
# PHASE 2: INTERMEDIATE (Weeks 17-34, Months 5-8)
# Original Phase 2 content (LLM weeks 13-26) eased to fit chip-design Day 6 + buffer
# PEFT, Quantization, Inference, RAG, Agents, Reasoning (LLM)
# Single-cycle RV32I CPU on FPGA → 5-stage pipelined CPU → I/D-cache (Silicon)
# ═══════════════════════════════════════════════════

---

## 🔧 Phase 2 — Silicon Day 6 Master Schedule (Weeks 13-26 of LLM content, calendar Months 4-8)

> Each entry below is **one Day 6 (Saturday morning, 1.5-2 hrs)**. The topic is chosen to be the silicon-level explanation of the same week's LLM topic — same lesson, deeper layer.

| LLM Week | LLM/CUDA Topic | Silicon Day 6 (1.5-2 hrs) | Month-end Project C |
|---|---|---|---|
| 13 | LoRA + KV-cache | RV32I ISA spec, pipelining concepts (Patterson Ch 4 read-through), HDLBits — finish FSM/Memory problems | (M4 cont.) MAC Unit polish + tape-out-style README |
| 14 | Quantization (GPTQ/AWQ) | Multi-precision MAC: design INT4/INT8/INT16 selectable MAC, Yosys synth report per precision | M4 — INT4/INT8 MAC Unit submitted |
| 15 | Inference Serving (vLLM/TensorRT-LLM) | AXI4-Lite slave with 4 CSRs + interrupt line; testbench acting as master | (M5 in progress) |
| 16 | RAG Foundations | Synchronous FIFO + asynchronous (Gray-coded CDC) FIFO; both verified | M5 — FIFO + AXI-Lite submitted |
| 17 | Advanced RAG | Single-cycle RV32I — start: PC, IMem, decoder, RegFile (from M3) | (M6 in progress — biggest C yet) |
| 18 | AI Agents Foundations | Single-cycle RV32I — middle: ALU integration (from M3), DMem, branch unit, control FSM | (M6 in progress) |
| 19 | AI Agents Advanced | Single-cycle RV32I — finish: assemble on FPGA, run Hello World over UART, run a C matmul, get cycle counts | M6 — 🔧 **Single-Cycle RV32I on FPGA** ⭐ |
| 20 | NVIDIA AI Ecosystem Pt 1 | Pipelined RV32I — start: insert IF/ID, ID/EX, EX/MEM, MEM/WB pipeline registers, get a non-hazardous program working | (M7 in progress) |
| 20.5 | NVIDIA AI Ecosystem Pt 2 | Pipelined RV32I — middle: add hazard detection unit, forwarding unit, branch flush | (M7 in progress) |
| 21-22 | Distillation & Model Merging | Pipelined RV32I — verify against **Spike** (RISC-V ISA simulator) on 1000 random programs; benchmark IPC vs single-cycle | M7 — 🔧 **5-Stage Pipelined RV32I** |
| 23 | RL & RLHF | Cache fundamentals: read Patterson Ch 5; design a 4 KB direct-mapped I-cache with valid bits | (M8 in progress) |
| 24 | Reasoning & CoT | Add a 4 KB direct-mapped D-cache with write-back + dirty bits; integrate with pipelined CPU | (M8 in progress) |
| 25 | Long Context | Cache controller FSM: IDLE → COMPARE_TAG → ALLOCATE → WRITE_BACK; AXI memory bus to BRAM | (M8 in progress) |
| 26 | State Space Models | Performance counters on cache (hit/miss/writeback) exposed as CSRs; re-run matmul, measure hit rate + IPC improvement | M8 — **I-Cache + D-Cache + AXI Memory** |

> **Reminder:** the LLM weekly Day 1-5 content from the existing roadmap is **fully preserved**. Day 6 is an *additional* Saturday-morning slot.

---

## 🔄 Buffer Week (Month 3 Revision)
- [ ] Revise distributed: DDP, ZeRO, tensor parallelism, NCCL
- [ ] Revise post-training: SFT, DPO, chat templates
- [ ] Revise Flash Attention algorithm and Triton programming
- [ ] Re-read custom CUDA extensions code
- [ ] **Build Monthly Project A:** LLM Surgery — Fine-Tuning & Alignment Toolkit
- [ ] **Build Monthly Project B:** TokenScope — Tokenizer Analyzer
- [ ] **Build Monthly Project C — Silicon:** ALU + Register File (`silicon/month03/`) + first Yosys synthesis report
- [ ] Push all 3 projects to GitHub

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
- [ ] **Build Monthly Project C — Silicon:** INT4/INT8 MAC Unit + Multi-Precision Datapath (`silicon/month04/`)
- [ ] Push all 3 projects to GitHub

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
- [ ] **Build Monthly Project C — Silicon:** FIFO + AXI-Lite (`silicon/month05/`)
- [ ] Push all 3 projects to GitHub

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
- [ ] **Build Monthly Project C — Silicon ⭐:** Single-Cycle RV32I CPU on FPGA (`silicon/month06_rv32i_cpu/`) — first FPGA build, runs C Hello World over UART
- [ ] Push all 3 projects to GitHub
- [ ] **Write a "6-month retrospective" blog post summarizing your journey** — make sure to include the silicon side: "I built my own CPU and ran C code on it"

---

# ═══════════════════════════════════════════════════
# PHASE 3: ADVANCED (Weeks 35-52, Months 9-12)
# Deep Systems, Research Implementation, Specialization (LLM)
# Pipelined RV32I + I/D-cache → AI accelerators (4×4, 8×8 systolic, NoC) (Silicon)
# ═══════════════════════════════════════════════════

---

## 🔧 Phase 3 — Silicon Day 6 Master Schedule

| LLM Week | LLM/CUDA Topic | Silicon Day 6 (1.5-2 hrs) | Month-end Project C |
|---|---|---|---|
| 27-28 | Vision-Language + Multi-Modal | NVIDIA SM microarchitecture deep-dive: read CUTLASS 3.x docs, study WMMA/MMA/WGMMA layouts | (M7 cont.) |
| 28.5 | Diffusion Overview | Hopper TMA + warp specialization (read NVIDIA Hopper whitepaper); sketch how WGMMA + TMA enable FA-3 | M7 — 5-Stage Pipelined RV32I shipped |
| 29-30 | MoE & Advanced Architectures | All-to-all NoC patterns; sketch a multicast tree for top-K routing in HW | (M8 in progress) |
| 31-32 | Compiler & Kernel Optimization | LLVM IR + MLIR intro: read MLIR docs, study the linalg dialect | (M8 in progress) |
| 33-34 | Training Infrastructure | NVLink/NVSwitch silicon: read NVSwitch whitepaper; sketch a hierarchical NoC topology | M8 — I-Cache + D-Cache + AXI Memory |
| 35-36 | Evaluation & Benchmarking | 4×4 INT8 systolic array — start: PE design, weight loading FSM | (M9 in progress — biggest Project C yet) |
| 37-38 | Production Systems | 4×4 INT8 systolic array — middle: 3 BRAM buffers, AXI4-Lite control regs, tile FSM | (M9 in progress) |
| 39-40 | Synthetic Data & Structured Generation | 4×4 INT8 systolic array — finish: integration on FPGA, Python test harness, benchmark vs M7 CPU C-loop matmul | M9 — 🔧 **4×4 INT8 Systolic GEMM on FPGA** ⭐ |

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
**LLM side:**
- [ ] Can build multi-modal AI systems (vision + audio + text)
- [ ] Understand diffusion models and how they connect to LLMs
- [ ] Understand MoE architectures and can implement them
- [ ] Can write optimized Triton/CUDA kernels for LLM ops
- [ ] Can set up distributed training (Megatron, DeepSpeed, FSDP)
- [ ] Can evaluate models comprehensively
- [ ] Can build production LLM systems with MLOps
- [ ] Can generate and curate synthetic training data

**Silicon side (after 12 months):**
- [ ] **5-stage pipelined RV32I CPU on FPGA** with hazard detection, forwarding, branch flush — verified vs Spike on 1000 random programs
- [ ] **Direct-mapped I-cache + D-cache** with write-back policy, AXI memory bus, performance counters
- [ ] **4×4 INT8 systolic GEMM array on FPGA** — first AI accelerator, ~50–100× faster than your CPU C-matmul
- [ ] **2×2 mesh NoC** with virtual channels and hardware all-reduce — 4 systolic tiles wired together
- [ ] **Eyeriss-style row-stationary PE** reproduced and compared against your weight-stationary array
- [ ] Read NVIDIA Hopper + Blackwell whitepapers in depth; comfortable reading CUTLASS 3.x source

---

# ═══════════════════════════════════════════════════
# PHASE 4: EXPERT (Weeks 53-68, Months 13-16)
# Research, Test-Time Compute, Enterprise Agents (LLM)
# 8×8 systolic + PyTorch Bridge → UVM → Formal → FlashAttention HW Block (Silicon)
# ═══════════════════════════════════════════════════

---

## 🔧 Phase 4 — Silicon Day 6 Master Schedule

| LLM Week | LLM Topic | Silicon Day 6 (1.5-2 hrs) | Month-end Project C |
|---|---|---|---|
| 41-42 | Reading & Implementing Papers | Read **TPU paper** (Jouppi 2017) + **Cerebras WSE** architecture overview | (M11 in progress) |
| 43-44 | Test-Time Compute & Reasoning | Eyeriss row-stationary PE — start: PE design with 16 MAC units, weight-shared rows | (M11 in progress) |
| 45-46 | Enterprise AI Agents (NVIDIA) | Eyeriss PE — finish + comparison report vs your weight-stationary array on a small convolution | M11 — Eyeriss-style Row-Stationary PE |
| 47-48 | Advanced Fine-Tuning & Alignment | 8×8 systolic array — start: scale up M9 array to 8×8, parameterized for runtime mode select | (M12 in progress — biggest C yet) |
| 49-50 | Advanced RAG & Information Retrieval | 8×8 systolic — middle: PyTorch C++ extension that drives the FPGA via UART/PCIe; one matmul end-to-end | (M12 in progress) |
| 51-52 | Phase 4 Capstone | 8×8 systolic — finish: run a real LLaMA-style MLP weight matmul through your FPGA accelerator, benchmarked | M12 — 🔧 **8×8 Systolic Array + PyTorch Bridge** |

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
**LLM side:**
- [ ] Can read, understand, and implement AI papers
- [ ] Can build test-time compute systems (MCTS, process reward)
- [ ] Can build enterprise-grade AI agents with NVIDIA tools
- [ ] Can implement advanced alignment techniques
- [ ] Can build state-of-the-art RAG systems
- [ ] Have contributed to open-source AI projects

**Silicon side (after 16 months):**
- [ ] **8×8 INT8 systolic array on FPGA** running a real LLM MLP weight matmul, callable from PyTorch
- [ ] **UVM-Lite testbench** with constrained-random + functional coverage ≥90% + first SVA properties
- [ ] **Formal verification** with SymbiYosys on AXI handshake (caught at least 1 bug)
- [ ] **2:4 structured sparse GEMM** showing ~2× throughput
- [ ] **FlashAttention-inspired HW block** running a real attention layer on FPGA
- [ ] **Yosys + OpenSTA on Sky130 PDK** with timing/area/power reports
- [ ] Comfortable reading hardware papers from ISCA/HPCA/MICRO weekly

---

# ═══════════════════════════════════════════════════
# PHASE 5: MASTERY (Weeks 69-86, Months 17-20)
# Cutting-edge research, OSS, deep specialization (LLM)
# OpenLane RTL→GDSII on Sky130 → TinyTapeout submission (Silicon)
# ═══════════════════════════════════════════════════

---

## 🔧 Phase 5 — Silicon Day 6 Master Schedule

| LLM Week | LLM Topic | Silicon Day 6 (now ~3 hrs each — Saturday morning + early afternoon) | Month-end Project C |
|---|---|---|---|
| 53-54 | LLM Capstone (Phase 5) options | OpenLane install + first tutorial run on Sky130 PDK | (M17 in progress) |
| 55-56 | Capstone implementation | OpenLane on M16 8×8 systolic block — get to GDSII; view in KLayout | M17 — OpenLane RTL→GDSII Floorplan + P&R |
| 57-58 | Cutting Edge Research | TinyTapeout-shaped block design — pick small AI block, fit tile budget | (M18 in progress) |
| 59-60 | Cutting Edge Research cont. | TT block — DRC + LVS clean, datasheet drafted | M18 — TinyTapeout Sky130 Block (Prep) |
| 61-62 | Open Source & Community | Multi-block floorplan: GEMM + softmax + DMA + control with proper PG grid | (M19 in progress) |
| 63-64 | OSS & Community cont. | PG grid + clock gating + IR drop estimation (OpenROAD PDN analysis) | M19 — Multi-Block Floorplan + PDN |
| 65 | OSS finale + community presentation | STA closure: push from 200 MHz → 400 MHz with retiming + multi-cycle paths | (M20 in progress) |
| 66-67 *(NEW Phase 5 weeks)* | **Production v1 LLM stack capstone** | STA closure cont.: WNS ≥ 0 across SS/TT/FF corners | M20 — STA Mastery |
| 68 *(NEW)* | **Multi-modal doc agent** | 🎉 **TinyTapeout submission live**: integrate, polish, GitHub Actions clean, **press SUBMIT** | M21 — 🎉 **TinyTapeout SUBMITTED** |

> **Note on weeks**: original LLM Phase 5 content was Weeks 53-65. New extra weeks 66-68 expand the LLM track in Phase 5 with Production-v1 LLM stack capstone work and the Multi-modal doc agent — both new Project A's for Months 17-20. The Phase 5 cap follows below.

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
# PHASE 6: LLM MAGNUM OPUS + SILICON CO-DESIGN (Weeks 87-104, Months 21-24)
# Months 21: ResearchLab + OSS-Citizen + 🎉 TinyTapeout submitted
# Months 22-24: LLM MAGNUM OPUS BUILD ⭐ (HARD DEADLINE: end of Month 24)
# Silicon: Co-design block + DFT + post-layout closure + HW companion paper
# ═══════════════════════════════════════════════════

---

## 🔧 Phase 6 — Silicon Day 6 Master Schedule

| Calendar Week | LLM Activity | Silicon Activity | Month-end |
|---|---|---|---|
| 87-89 (Month 21) | ResearchLab + OSS-Citizen | TinyTapeout submission polish, datasheet write-up, **press SUBMIT** | M21 — 🎉 TinyTapeout LIVE |
| 90-94 (Month 22) | **LLM Magnum Opus — Architecture & MVP build** | Polish co-design block (whichever silicon block from M11-M20 your Magnum Opus calls) | M22 — Co-Design Block |
| 95-99 (Month 23) | **LLM Magnum Opus — Optimization, eval, paper writing** | DFT insertion (scan chains), ATPG, post-layout timing closure | M23 — DFT + Post-Layout |
| 100-104 (Month 24) | **LLM Magnum Opus — SHIP** ⭐ blog + repo + demo + benchmarks | Hardware-companion paper section (1500 words) integrated into Magnum Opus repo | M24 — **LLM MAGNUM OPUS SHIPPED** ⭐ |

---

## Week 87-89 (Month 21): ResearchLab + OSS-Citizen + 🎉 TinyTapeout Submission

**LLM Daily (Day 1-5):**
- [ ] Reproduce 3 recent (2026) papers end-to-end with Docker images, write 1500-word reports each
- [ ] Sustained OSS contribution: 5+ merged PRs in one project, plus reviews on others' PRs (vLLM / TensorRT-LLM / NeMo / HuggingFace / Triton)
- [ ] Write technical blog: "How I reproduce papers"

**Silicon Day 6 (Saturday 1.5-3 hrs):**
- [ ] Polish your TinyTapeout block — RTL clean, datasheet written, GitHub Actions verification clean
- [ ] **Press SUBMIT on the TinyTapeout shuttle**
- [ ] Blog post: "I just submitted a chip block for fabrication on a real wafer"

**Project C (Month 21):** 🎉 **TinyTapeout Submission Live** — your block is on a SkyWater wafer, fabrication in progress, you'll receive the chip ~6-9 months later.

---

## Week 90-94 (Month 22): LLM Magnum Opus — Architecture & MVP Build

**LLM (Day 1-5 + Sat afternoon, ~80% of effort this month):**
- [ ] Pick your LLM Magnum Opus option (A: LLM-from-Scratch-to-Production · B: AI Agent OS · C: GPU Inference Engine · D: Edge LLM)
- [ ] Architecture document (3000+ words): user, pain, novelty, system diagram, Month 1-21 cumulative skills used
- [ ] Build the MVP: get the core pipeline running end-to-end, even if rough
- [ ] First evaluation pass: identify what's good and what needs polish
- [ ] Set up the repo, README skeleton, CI

**Silicon Day 6 (~3 hrs Saturday):**
- [ ] Polish your "co-design block" — whichever silicon module from M11-M20 the Magnum Opus uses (most likely the FlashAttention HW block from M15 or the 8×8 systolic from M12)
- [ ] Make sure the co-design block is integrated and demonstrably accelerates one path in the Magnum Opus
- [ ] Update the silicon block's README to point at the Magnum Opus

**Sunday:**
- [ ] Continue the weekly ISCA/HotChips/HPCA paper habit (still ~2/week)

---

## Week 95-99 (Month 23): LLM Magnum Opus — Optimization, Evaluation, Paper

**LLM:**
- [ ] Deep optimization pass: profile bottlenecks, optimize hot paths, swap in custom CUDA/Triton kernels where they help
- [ ] Comprehensive evaluation suite: 5+ benchmarks, baselines, regression-tested
- [ ] Begin writing the **2500+ word technical blog post**
- [ ] Record initial demo video (rough cut)
- [ ] First public preview to friends / Twitter / Discord — get feedback

**Silicon Day 6:**
- [ ] **DFT insertion** — add scan chains to your accelerator
- [ ] **ATPG** — generate test patterns, target ≥95% fault coverage
- [ ] **Post-layout timing closure** — re-run OpenLane on the post-DFT design, close timing
- [ ] Document everything in the silicon repo

---

## Week 100-104 (Month 24): LLM MAGNUM OPUS — SHIPPED ⭐ (HARD DEADLINE)

**LLM (full focus — this is the cap!):**
- [ ] **Final blog post** (2500+ words) polished and published
- [ ] **Final demo video** (5-10 min) recorded and edited
- [ ] **GitHub repo** — polished README, install instructions, examples, tests, all benchmarks reproducible
- [ ] **Architecture diagrams** — clear visuals
- [ ] **Reproducibility report** — one-command run, all numbers reproducible
- [ ] **Public launch**: post on Twitter/X, Hacker News, the relevant Discord
- [ ] **Reach out to 10 people**: hiring managers, researchers, OSS maintainers — share the project

**Silicon Day 6 (Month 24 silicon polish):**
- [ ] Write the **Hardware Companion Section** of the Magnum Opus paper (1500 words)
- [ ] Roofline-style charts: which ops are memory-bound on GPU vs your accelerator
- [ ] Integrate this section into the Magnum Opus repo
- [ ] **You now have a portfolio that visibly spans algorithms → silicon**

---

### ✅ Phase 6 Completion Checklist (= 24-MONTH MILESTONE)
- [ ] **LLM MAGNUM OPUS SHIPPED on time** (within Month 24)
- [ ] Blog post (2500+ words), demo video, GitHub repo, reproducibility scripts all live
- [ ] At least 10 people outside your circle have engaged with the project
- [ ] Co-design silicon block polished, DFT-inserted, post-layout-closed, and integrated
- [ ] Hardware companion paper section written and published
- [ ] **Your CV now has one headline LLM project + a meaningful silicon block + 24 monthly capstones (A+B+C × 22 months) = 66 portfolio artifacts**
- [ ] You can explain every line of every artifact at a senior-engineer interview level
- [ ] **You can apply to senior LLM-systems roles at NVIDIA, OpenAI, Anthropic, Etched, Cerebras, Groq, and others — and pass at the senior level, not the entry rotation**

---

# ═══════════════════════════════════════════════════
# PHASE 7: SILICON DEEP DIVE (Weeks 105-130, Months 25-30)
# LLM track in maintenance mode. Silicon track is full-time.
# Out-of-Order RV32I → Branch Predictors → Cache Coherence
# → Transformer-Block Accelerator on FPGA → Architecture Comparison Paper
# → Custom Accelerator Compiler → Chiplet Design
# ═══════════════════════════════════════════════════

> **What changes from Phase 6 → Phase 7:**
> - Mon-Fri Day 1-5 are now **silicon-focused** (2-3 hrs/day): RTL design, architecture reading, OpenLane runs, FPGA prototyping
> - Day 6 (Saturday) is **LLM maintenance only** — read 2 papers/week, maintain OSS commitments, that's it (1.5-2 hrs total)
> - Saturday afternoon project becomes **silicon-focused** (3-4 hrs)
> - Sunday paper alternates between LLM and silicon papers (you've earned the right to read pure-silicon papers now)

---

## Month 25 (Weeks 105-108): Out-of-Order RV32I Mini-Core
**LLM activity:** 2 papers/week, ongoing OSS reviews
**Silicon focus:** Tomasulo-style register renaming, reservation stations, reorder buffer, single-issue out-of-order RV32I
**Reading:** Hennessy & Patterson *Computer Architecture: A Quantitative Approach* — Chapter 3 (Instruction-Level Parallelism)
**Daily silicon work:**
- [ ] Week 105: Read H&P Ch 3, sketch the OoO datapath
- [ ] Week 106: Implement register rename table + reservation stations
- [ ] Week 107: Implement 8-entry reorder buffer + retire logic
- [ ] Week 108: Verify against Spike on 1000 random programs, benchmark IPC vs in-order pipelined RV32I

**Project C (Month 25):** Out-of-Order RV32I-M with 8-entry ROB, verified, benchmarked, published

---

## Month 26 (Weeks 109-112): Branch Predictor Lab + MESI Cache Coherence
**Silicon focus:** Branch prediction (gshare → TAGE), 2-core MESI cache coherence on shared L2
**Reading:** H&P Ch 5 (Memory Hierarchy and Coherence)
**Daily silicon work:**
- [ ] Week 109: gshare branch predictor + branch target buffer (BTB)
- [ ] Week 110: TAGE-style branch predictor (basic version)
- [ ] Week 111: 2-core MESI coherence: bus snooping, MESI state machine
- [ ] Week 112: Shared L2 cache controller + benchmark: false sharing, ping-pong

**Project C (Month 26):** Branch predictor lab + 2-core MESI in SystemVerilog

---

## Month 27 (Weeks 113-116): Transformer-Block Accelerator on FPGA — Full Layer 🔧 (HEADLINE BUILD)
**Silicon focus:** A complete transformer layer in RTL: multi-head attention + FFN + LayerNorm + residual + RoPE, INT8 quantized, running on FPGA at measurable tokens/sec
**Reading:** Re-read FlashAttention 1, 2, 3 papers + the Sohu (Etched) white paper
**Daily silicon work:**
- [ ] Week 113: Architecture spec — block diagram, dataflow, register map, BRAM budget
- [ ] Week 114: Multi-head attention block (reuse M15 FlashAttention HW) + RoPE unit
- [ ] Week 115: FFN block (2-layer MLP with SwiGLU activation) + LayerNorm + residual integrate
- [ ] Week 116: Run a real LLaMA-class single-layer block on FPGA, measure tokens/sec, write blog post

**Project C (Month 27):** Transformer-Block Accelerator on FPGA — the headline pre-Magnum-Opus silicon artifact

---

## Month 28 (Weeks 117-120): Architecture Comparison Paper — H100 vs MI300X vs TPUv5 vs Sohu
**Silicon focus:** Use **Timeloop + Accelergy** to model 4 commercial AI chips on a 70B-parameter LLM workload. Write a public paper.
**Reading:** Hot Chips proceedings (latest year), SemiAnalysis on AI chip landscape, Cerebras WSE-3 paper, Groq TSP paper, NVIDIA Blackwell whitepaper, Etched Sohu whitepaper
**Daily silicon work:**
- [ ] Week 117: Workload analysis: LLaMA-70B inference (decode-heavy + prefill-heavy scenarios)
- [ ] Week 118: Timeloop models for H100 + MI300X
- [ ] Week 119: Timeloop models for TPUv5 + Sohu (where data exists; otherwise reasonable estimates)
- [ ] Week 120: Write the 6000-word paper, generate roofline + perf-per-watt charts

**Project C (Month 28):** Architecture Comparison Paper — arxiv-ready style, published

---

## Month 29 (Weeks 121-124): Compiler for Your Accelerator (MLIR / TVM)
**Silicon focus:** A small MLIR dialect (or TVM schedule) that lowers a transformer-block IR onto your Month 27 accelerator
**Reading:** MLIR docs, IREE source, the LLVM/MLIR Lecture Series referenced in the AI hardware engineer roadmap
**Daily silicon work:**
- [ ] Week 121: Define a custom MLIR dialect: `myaccel.gemm`, `myaccel.softmax`, `myaccel.layernorm`
- [ ] Week 122: Lowering passes: linalg → your dialect → LLVM IR → driver code
- [ ] Week 123: End-to-end compile of a tiny LLaMA layer onto your accelerator
- [ ] Week 124: Benchmark, write blog post

**Project C (Month 29):** MLIR/TVM compiler for your accelerator

---

## Month 30 (Weeks 125-130): Chiplet Design Study (UCIe) + Phase 7 Cap
**Silicon focus:** Chiplet architecture: compute die + simulated HBM-like memory die, UCIe die-to-die interface modeling
**Reading:** UCIe specification, AMD Instinct MI300X (CoWoS-based) architecture, Intel Foveros Direct, NVIDIA Grace-Hopper coherent memory paper
**Daily silicon work:**
- [ ] Weeks 125-126: Read UCIe spec end-to-end, summarize the PHY + protocol layers
- [ ] Weeks 127-128: Model die-to-die bandwidth/latency in SystemC or Python; size for LLM workloads
- [ ] Weeks 129-130: Write the Phase 7 cap blog: "Why every AI chip after 2026 is a chiplet"

**Project C (Month 30):** Chiplet design study, performance projection, blog post

---

### ✅ Phase 7 Completion Checklist (= 30-MONTH MILESTONE)
- [ ] **Out-of-order RISC-V** with register rename + ROB on FPGA, verified
- [ ] Branch predictors (gshare + TAGE) and 2-core MESI coherence
- [ ] **Transformer-block accelerator** running real LLaMA layer on FPGA at measurable tokens/sec — your headline silicon artifact
- [ ] Architecture comparison paper modeling 4 commercial AI chips with Timeloop, posted publicly
- [ ] Custom MLIR/TVM compiler for your accelerator
- [ ] Chiplet design study with die-to-die bandwidth modeling
- [ ] **You can walk into any AI chip company architecture interview prepared**

---

# ═══════════════════════════════════════════════════
# PHASE 8: SILICON MAGNUM OPUS ⭐⭐⭐ (Weeks 131-156, Months 31-36)
# 6-MONTH BUILD: CUSTOM AI INFERENCE ACCELERATOR
# Architecture spec → RTL → Verification → FPGA → OpenLane GDSII
# → DFT → Architecture comparison paper → Open-source release
# ═══════════════════════════════════════════════════

> **This is the chip-design career piece.** The one you take to NVIDIA / Etched / Tenstorrent / Cerebras / Groq / AMD architecture interviews. By Month 36, this is your most important artifact in the entire roadmap.

---

## Month 31 (Weeks 131-134): Architecture Specification + RTL Foundation
- [ ] Pick your option: **A** TinyTransformer Engine · **B** FlashAttention ASIC · **C** Spatial MoE Router · **D** CIM-Inspired LLM Decode Block
- [ ] Write the **Architecture Spec Document** (3000+ words):
  - [ ] Workload analysis (use your Month 28 paper)
  - [ ] Block diagram
  - [ ] Dataflow choice + justification
  - [ ] ISA / register map / programming model
  - [ ] Performance / area / power targets
  - [ ] Comparison hooks to NVIDIA H100/Blackwell, Sohu, TPU
- [ ] Set up the repo: `silicon_magnum_opus/` with directory structure (rtl/, tb/, openlane/, fpga/, docs/, paper/)
- [ ] Begin RTL: top-level skeleton + parameter declarations + interface definitions

---

## Month 32 (Weeks 135-138): Core RTL Implementation
- [ ] Implement the compute datapath (systolic array / FlashAttention pipeline / MoE router as chosen)
- [ ] Implement the memory subsystem (scratchpad SRAM, paged KV-cache controller, DMA engines)
- [ ] Implement the control plane (CSRs, doorbell, interrupts, AXI4 full master/slave)
- [ ] Begin simulation testbench (CocoTB or pure SystemVerilog)
- [ ] **Synthesizable RTL ≥ 5000 lines by end of month**

---

## Month 33 (Weeks 139-142): Verification + FPGA Prototype
- [ ] **UVM testbench** with constrained-random + functional coverage targeting ≥95%
- [ ] **Formal verification (SymbiYosys)** on critical interfaces (AXI handshake, no-deadlock properties)
- [ ] FPGA bring-up: synthesize for your board, get a tiny test running
- [ ] Run a real LLaMA-class workload (single layer or single block, INT4/INT8) on FPGA at measurable tokens/sec
- [ ] **PyTorch integration** as a backend: `torch.compile` custom backend or C++ extension

---

## Month 34 (Weeks 143-146): Physical Implementation (OpenLane Sky130)
- [ ] **OpenLane / LibreLane RTL→GDSII flow** on Sky130
- [ ] Floorplan optimization, multi-block PG grid, clock-gating insertion
- [ ] Timing closure across MCMM corners (SS/TT/FF, voltage, temperature)
- [ ] **DFT insertion** — scan chains, ATPG ≥ 95% fault coverage
- [ ] Power analysis: dynamic + leakage breakdown

---

## Month 35 (Weeks 147-150): Architecture Comparison Paper + Open-Source Polish
- [ ] **Architecture comparison paper** (8000+ words):
  - [ ] Your design vs H100 / MI300X / TPUv5 / Sohu — perf/W, area efficiency, key trade-offs
  - [ ] Workload-specific analysis: where does your design win, where does it lose?
- [ ] Polish the GitHub repo: comprehensive README, build instructions, FPGA bitstream build, OpenLane config, PyTorch integration examples
- [ ] Write integration tests
- [ ] **Reproducibility script**: one command brings up FPGA + runs benchmark + compares to PyTorch reference

---

## Month 36 (Weeks 151-156): SHIP ⭐⭐⭐
- [ ] **Final demo video** (10-15 min): architecture walkthrough → FPGA running → OpenLane flow → performance numbers → silicon comparison
- [ ] **Public release**: arxiv preprint of architecture comparison paper, GitHub release tagged v1.0
- [ ] **Hot Chips paper-track submission** (or relevant venue): submit the architecture comparison paper for peer review
- [ ] **(If TinyTapeout chip from Month 21 returned)**: characterize it on a probe station or hobbyist setup, post-mortem, write up
- [ ] **(Optional final tape-out)**: Efabless chipIgnite submission (~$10K) — your custom block submitted for real fabrication. This is the headline-grabber.
- [ ] **Public launch**: post the project on Hacker News, Twitter/X, the relevant subreddits, your LinkedIn
- [ ] **Reach out**: 20+ people (NVIDIA architects, AI chip startup founders, professors)

---

### ✅ Phase 8 Completion Checklist (= 36-MONTH FINAL MILESTONE — 3 YEARS)

**You have built the AI stack from electrons to GPT-class systems with your own hands.**

- [ ] **Silicon Magnum Opus shipped** — RTL + FPGA demo + OpenLane GDSII + paper + open-source release
- [ ] (Optionally) physical silicon back from TinyTapeout shuttle — characterized
- [ ] (Optionally) Efabless chipIgnite submission in for fabrication
- [ ] Architecture comparison paper submitted to Hot Chips / arxiv
- [ ] **Your portfolio now has:**
  - 1 LLM Magnum Opus shipped (Month 24)
  - 1 Silicon Magnum Opus shipped (Month 36)
  - 22 LLM monthly capstones (Project A) — Months 2-24
  - 22 LLM monthly capstones (Project B) — Months 2-24 (some integrated)
  - 22 silicon monthly capstones (Project C) — Months 2-24
  - 12 silicon monthly capstones — Months 25-36
  - 1 architecture comparison paper (8000+ words)
  - 1 hardware companion paper (1500 words, Month 24)
  - **78+ public artifacts on GitHub**
- [ ] You can reason fluently about transformer math → CUDA → PTX → SASS → microarchitecture → RTL → standard cells → silicon
- [ ] **You can walk into NVIDIA / Etched / Tenstorrent / Cerebras / Groq / AMD / Google TPU / AWS Trainium architecture team interviews at the senior-architect level** — not as an entry-level hire
- [ ] **You are exactly the engineer the AI silicon industry desperately needs and can't find: someone who has built both ends of the stack with their own hands**

---

# ═══════════════════════════════════════════════════
# APPENDIX: ESSENTIAL RESOURCES
# ═══════════════════════════════════════════════════

## Books

### LLM / CUDA / ML Books
- [ ] "Programming Massively Parallel Processors" (Kirk & Hwu) — 4th ed
- [ ] "Deep Learning" (Goodfellow, Bengio, Courville)
- [ ] "Designing Machine Learning Systems" (Huyen)

### 🔧 Silicon / Computer Architecture / VLSI Books (in order of when to read them)
- [ ] **Months 2-9: Foundations** — *Digital Design and Computer Architecture, RISC-V Edition* (Sarah & David Harris) — the single best CSE-friendly entry book to silicon
- [ ] **Months 6-15: ISA & CPU** — *Computer Organization and Design, RISC-V Edition* (Patterson & Hennessy) — covers ISA, datapath, pipelining, hazards, memory hierarchy
- [ ] **Months 10-15: Verification** — *SystemVerilog for Verification* (Chris Spear) — the standard text for testbench + UVM
- [ ] **Months 14-30: AI Hardware** — *Efficient Processing of Deep Neural Networks* (Sze, Chen, Yang, Emer) — the AI accelerator bible (free PDF + free MIT OCW course 6.5930)
- [ ] **Months 12-25: Transistors & CMOS** — *CMOS VLSI Design* (Weste & Harris) — when you need to understand the physical layer
- [ ] **Months 17-30: Physical Design** — *VLSI Physical Design: From Graph Partitioning to Timing Closure* (Kahng, Lienig, Markov, Hu)
- [ ] **Months 20-30: Static Timing** — *Static Timing Analysis for Nanometer Designs* (Bhasker & Chadha)
- [ ] **Months 25-36: Advanced Architecture** — *Computer Architecture: A Quantitative Approach* (Hennessy & Patterson, 6th/7th ed) — DSA chapters are mandatory
- [ ] **Months 25-36: Reference** — *UVM Cookbook* (Verification Academy / Siemens, FREE PDF) — for production verification environments

## Online Courses

### LLM / ML
- [ ] Karpathy "Neural Networks: Zero to Hero"
- [ ] Karpathy "Let's build GPT"
- [ ] Stanford CS224n — NLP with Deep Learning
- [ ] Stanford CS336 — Language Modeling from Scratch
- [ ] NVIDIA DLI: "Fundamentals of Accelerated Computing with CUDA"
- [ ] NVIDIA DLI: "Building RAG Agents with LLMs"
- [ ] NVIDIA DLI: "Generative AI with Diffusion Models"
- [ ] NVIDIA DLI: "Accelerating CUDA C++ Applications with Nsight Systems"

### 🔧 Silicon / Computer Architecture (free YouTube + university)
- [ ] **Onur Mutlu — Computer Architecture** (CMU/ETH Zürich) — full lectures on YouTube, **the deepest free architecture course on the planet**, watch through Months 6-30
- [ ] **MIT 6.5930 — Hardware Architecture for Deep Learning** (Vivienne Sze) — pairs with the *Efficient Processing of DNNs* book
- [ ] **MIT 6.004 / 6.S081** — Computation Structures (warm-up if you're new to systems)
- [ ] **Berkeley CS152 / CS252** — Computer Architecture / Graduate Architecture
- [ ] **nand2tetris** — build a computer from NAND gates up (~10 hrs warm-up)
- [ ] **Carnegie Mellon CS447 / CS740** — advanced architecture
- [ ] **Stanford CS149** — Parallel Computing
- [ ] **NPTEL** (Indian govt) — VLSI Design + Digital VLSI Design + Computer Architecture (free, multiple courses)
- [ ] **HDLBits** — https://hdlbits.01xz.net — solve Verilog problems weekly from Week 3 (Day 6) onwards. **Mandatory hands-on practice.**

## 🔧 Silicon GitHub Repos to Study
- [ ] **NVIDIA CUTLASS** — production templates for WGMMA / TMA / warp specialization on Hopper
- [ ] **NVIDIA NVDLA** — open-source NVIDIA Deep Learning Accelerator (read the spec end-to-end)
- [ ] **tinygrad** — minimal lazy DAG, UOp IR, scheduler — fork-and-extend target for custom backends
- [ ] **OpenROAD / OpenLane / LibreLane** — open-source RTL→GDSII flow on Sky130
- [ ] **Yosys + ABC** — synthesis
- [ ] **TinyTapeout** template repos (`tt09-` etc.) — the path to real silicon for ~$300
- [ ] **Awais-Asghar/5-Stage-Pipelined-RISC-V-Processor-on-FPGA** — reference implementation for Months 6-7
- [ ] **VexRiscv** (SpinalHDL) — production-quality open RISC-V cores
- [ ] **PicoRV32** — minimal RISC-V core, easy to read
- [ ] **CVA6 / Ariane** — Linux-capable open RISC-V (~3000 lines)
- [ ] **ChipsAlliance / Chisel + FIRRTL** — Scala-based HDL ecosystem
- [ ] **SpinalHDL** — Scala-based HDL with VexRiscv ecosystem
- [ ] **CocoTB** — Python testbench framework
- [ ] **SymbiYosys** — open-source formal verification
- [ ] **Timeloop / Accelergy (MIT)** — analytical AI-accelerator modeling
- [ ] **Ramulator / DRAMSim3** — DRAM/HBM simulators
- [ ] **MIT Eyeriss reference** — the foundational AI accelerator design

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

## 🔧 Silicon Master Reading List (40+ Papers — read 1-2/week starting Week 11)

### Foundational AI Accelerator Papers
- [ ] 1. **"In-Datacenter Performance Analysis of a TPU"** (Jouppi et al., ISCA 2017) — the TPU paper, mandatory
- [ ] 2. **"Eyeriss: A Spatial Architecture for Energy-Efficient Dataflow for CNNs"** (Chen et al., ISCA 2016) — the dataflow taxonomy
- [ ] 3. "Eyeriss v2" (Chen et al., 2019)
- [ ] 4. "MAESTRO: An Open-source Infrastructure for Modeling Dataflows" (Kwon et al., 2020)
- [ ] 5. "Timeloop: A Systematic Approach to DNN Accelerator Evaluation" (Parashar et al., 2019)
- [ ] 6. "Cerebras WSE: Wafer-Scale Engine" architecture papers
- [ ] 7. "Groq TSP — Software-Scheduled Tensor Streaming" papers
- [ ] 8. "Etched Sohu — Transformer-Only ASIC" white paper (2024)
- [ ] 9. "SambaNova Reconfigurable Dataflow Architecture" papers
- [ ] 10. "Graphcore IPU: Bulk Synchronous Parallel" architecture

### LLM-Specific Hardware (your differentiator)
- [ ] 11. "FlashAttention: Fast and Memory-Efficient Exact Attention" (Dao, 2022)
- [ ] 12. "FlashAttention-2" (Dao, 2023)
- [ ] 13. "FlashAttention-3" (Tri Dao, 2024)
- [ ] 14. "Tawa: Automatic Warp Specialization" (arXiv 2510.14719, 2025)
- [ ] 15. "FlatAttention: A New Dataflow for Multi-Head Attention" (2025) — 4× over FA-3
- [ ] 16. "PagedAttention / vLLM" (Kwon, 2023)
- [ ] 17. "HALO: Memory-Centric Heterogeneous Accelerator with 2.5D Integration" (2025)
- [ ] 18. "Microbenchmarking NVIDIA's Blackwell Architecture" (arXiv 2512.02189)

### NVIDIA Architecture (your employer — go deep)
- [ ] 19. NVIDIA Hopper Architecture Whitepaper
- [ ] 20. NVIDIA Blackwell Architecture Whitepaper
- [ ] 21. NVIDIA Hopper TMA + WGMMA developer deep-dive
- [ ] 22. CUTLASS 3.x source code (read with the docs alongside)
- [ ] 23. NVIDIA NVDLA spec (the open-source accelerator)
- [ ] 24. "Hopper/Blackwell Tensor Core MMA Layouts" (vjkrish blog, 2026)

### Computer Architecture Classics
- [ ] 25. "Tomasulo's Algorithm" (1967) — original out-of-order paper
- [ ] 26. "MESI Protocol" papers — Stenström survey
- [ ] 27. "Branch Prediction Strategies" — Yeh & Patt, McFarling
- [ ] 28. "TAGE Branch Predictor" (Seznec, 2006)

### Compiler & Software for Accelerators
- [ ] 29. "TVM: An Automated End-to-End Optimizing Compiler" (Chen et al., OSDI 2018)
- [ ] 30. "MLIR: Multi-Level Intermediate Representation" (Lattner et al., 2020)
- [ ] 31. "Triton: An IL and Compiler for Tiled Neural Network Computations" (Tillet et al., 2019)
- [ ] 32. "Halide: Decoupling Algorithms from Schedules" (Ragan-Kelley et al., 2013)

### Memory & Interconnect
- [ ] 33. "HBM3 specification" (JEDEC) — read the bandwidth/structure section
- [ ] 34. "UCIe Specification" (1.x) — die-to-die interconnect
- [ ] 35. "NVLink/NVSwitch architecture" (NVIDIA whitepapers)

### Sparse, Quantized, and CIM
- [ ] 36. "SCNN: An Accelerator for Compressed-sparse CNNs" (Parashar, ISCA 2017)
- [ ] 37. "Cambricon-X: An Accelerator for Sparse NNs" (2016)
- [ ] 38. "Compute-in-Memory: Survey" (recent IEEE Micro)
- [ ] 39. NVIDIA 2:4 Structured Sparsity papers (Ampere+)

### Industry Reading (every quarter)
- [ ] 40. **Hot Chips proceedings** — annual, read the AI-accelerator papers each year
- [ ] 41. **ISSCC, ISCA, HPCA, MICRO** — top architecture conferences
- [ ] 42. **SemiAnalysis** (Dylan Patel) — competitive landscape and roadmaps
- [ ] 43. **Chip Architects podcast** — AI chip industry analysis

---

## 🔧 Hardware Setup (one-time, ~$300-700 total over 3 years)

### Required Around Month 6-8 — FPGA Dev Board (~$30-320)
Pick **one** based on budget:
- **Sipeed Tang Nano 20K** (~$30) — Gowin GW2A FPGA, decent for RV32I + small accelerator. Tightest budget option, **fully sufficient through Month 16**.
- **Digilent Arty A7-100T** (~$220) — Xilinx Artix-7, the reference choice. Used by most RISC-V tutorials and AI accelerator projects.
- **Pynq-Z2** (~$200) — Xilinx Zynq-7020 (FPGA + ARM CPU). Good for PyTorch C++ extension integration.
- **Digilent Nexys A7-100T** (~$320) — bigger Artix-7, more BRAM, fits your 8×8 systolic array more comfortably.
- *Avoid for this roadmap:* extreme-budget boards like Cyclone IV / IceStick — too small for the systolic arrays in Months 9-15.

### Required Software (all FREE)
- **Vivado WebPACK** (Xilinx — free for boards under $500) OR **Gowin EDA** (for Tang Nano)
- **Icarus Verilog** + **Verilator** + **GTKWave** (free, all platforms)
- **Yosys** + **OpenSTA** + **OpenROAD** + **OpenLane / LibreLane** (free, open-source)
- **Magic** + **Netgen** + **KLayout** (free, open-source)
- **CocoTB** (Python testbench framework, free)
- **SymbiYosys** (formal verification, free)
- **riscv32-gcc cross-compiler** (free, for compiling C to your CPU)
- **Spike** (RISC-V ISA simulator, free reference for verification)

### Optional Around Month 21-22 — TinyTapeout Submission (~$300)
- One TinyTapeout shuttle slot (~$300) — your Sky130 block goes on a real SkyWater wafer
- Chip returns ~6-9 months later for hobbyist characterization

### Optional Final — Month 36 — Efabless ChipIgnite Tape-Out (~$10K)
- ~$10,000 for a small custom block on a 130nm or 180nm shuttle. **This is the headline-grabber for your Silicon Magnum Opus** but is fully optional — the OpenLane GDSII alone is enough for the senior-architect interview.

### Total budget across 3 years
- **Minimum**: ~$50 (Tang Nano + free toolchain only) — gets you to a working AI accelerator on FPGA + GDSII files
- **Recommended**: ~$520 (Arty A7 + TinyTapeout) — gets you real silicon back from SkyWater
- **Premium**: ~$10,520 (above + Efabless tape-out) — your name on a custom-fabricated chip

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

## 🔧 Silicon Tools & Ecosystem Checklist

### Simulation & Wave Viewing (free)
- [ ] **Icarus Verilog** — open-source Verilog simulator (used Day 6 weeks 3-30)
- [ ] **Verilator** — fast cycle-accurate Verilog simulator
- [ ] **GTKWave** — open-source waveform viewer
- [ ] **CocoTB** — Python-based testbench framework

### Synthesis & Physical Design (free, open-source)
- [ ] **Yosys** — open-source RTL synthesis (Months 3+)
- [ ] **OpenSTA** — static timing analysis (Months 7+)
- [ ] **OpenROAD** — placement, CTS, routing (Months 17+)
- [ ] **OpenLane / LibreLane** — full RTL→GDSII flow on Sky130 (Months 17+)
- [ ] **Magic** — VLSI layout viewer/editor
- [ ] **Netgen** — LVS (Layout vs Schematic) checker
- [ ] **KLayout** — GDS viewer

### FPGA Toolchains (free)
- [ ] **Vivado WebPACK** (Xilinx — for Arty / Nexys / Pynq boards)
- [ ] **Gowin EDA** (for Tang Nano boards)

### Verification (free)
- [ ] **SymbiYosys** — open-source formal verification (Month 14+)
- [ ] **UVM-1.2 / IEEE 1800.2-2017** — Universal Verification Methodology (Month 13+)
- [ ] **Verification Academy / ChipVerify** — free UVM training resources

### PDKs (open-source process design kits)
- [ ] **SkyWater Sky130** — the open PDK used by TinyTapeout and OpenLane (mandatory)
- [ ] **GlobalFoundries gf180mcu** — alternative open PDK at 180nm

### Tape-out shuttles (real silicon, optional)
- [ ] **TinyTapeout** — ~$300/slot, Sky130, returns ~6-9 months later
- [ ] **Efabless chipIgnite** — ~$10K, more flexibility, returns ~9-12 months later

### Modeling & Analysis (free)
- [ ] **Timeloop** (MIT) — analytical AI accelerator performance modeling
- [ ] **Accelergy** (MIT) — energy modeling for accelerators
- [ ] **Gem5** — full-system architectural simulator
- [ ] **Ramulator / DRAMSim3** — DRAM/HBM simulators

### Compilers for Custom Hardware
- [ ] **LLVM** — IR, three-phase compiler, custom backends
- [ ] **MLIR** — multi-level IR, linalg/tensor/affine/vector dialects
- [ ] **TVM** — schedule-based ML compiler with auto-tuning
- [ ] **Triton** (OpenAI) — Python DSL → PTX
- [ ] **IREE** — MLIR-based ML runtime
- [ ] **tinygrad** — fork-and-extend target for adding custom hardware backends

### Cross-compilers (for putting code on YOUR CPU)
- [ ] **riscv32-unknown-elf-gcc** — compile C/C++ to your RV32I CPU
- [ ] **Spike** — official RISC-V ISA simulator (use as reference)

---

## 📁 Repo Layout Convention

For a cohesive portfolio, use this layout:

```
ai-stack-roadmap/
├── llm/                        # All LLM/CUDA monthly projects
│   ├── month01_gpu_matmul/
│   ├── month01_inference_simulator/
│   ├── month02_nanoLLM/
│   ├── month02_transformerscope/
│   └── ... (Months 1-24)
├── silicon/                    # All Project C silicon work
│   ├── month02/                # Logic-Lab — Gates to GEMM
│   ├── month03/                # ALU + Register File
│   ├── month04/                # INT4/INT8 MAC Unit
│   ├── month05/                # FIFO + AXI-Lite
│   ├── month06_rv32i_cpu/      # Single-Cycle RV32I on FPGA
│   ├── month07_pipelined_rv32i/
│   ├── month08_cached_cpu/
│   ├── month09_systolic_4x4/   # FIRST AI ACCELERATOR
│   ├── month10_noc/
│   ├── month11_eyeriss_pe/
│   ├── month12_systolic_8x8_torch/
│   ├── month13_uvm/
│   ├── month14_formal_sparse/
│   ├── month15_flashattn_hw/
│   ├── month16_sky130_synth/
│   ├── month17_openlane/
│   ├── month18_tt_prep/
│   ├── month19_pdn/
│   ├── month20_sta/
│   ├── month21_tt_submitted/
│   └── ... (Months 22-24 hardware companion)
├── llm_magnum_opus/            # The headline LLM project (Months 22-24)
└── silicon_magnum_opus/        # The headline silicon project (Months 31-36)
    ├── rtl/
    ├── tb/
    ├── openlane/
    ├── fpga/
    ├── docs/
    └── paper/
```

Each folder has its own README with: User · Pain · Novelty · Cumulative skills · Evidence · Co-design link.

---

## Progress Tracker

*Current week: Week 2 (LLM track)*
*Weeks completed: Week 1 ✅*
*Silicon Track: starts Week 3 (after current week ends)*
*LLM Magnum Opus deadline: Month 24 (Week 104)*
*Silicon Magnum Opus deadline: Month 36 (Week 156)*
*Total items completed: ~35 / ~2400*
*Monthly projects completed: 0 / **70** (24 Project A + 24 Project B + 22 Project C through Month 24, plus 12 silicon projects in Months 25-36)*

### Milestone Tracker
- [ ] Month 6 — first FPGA build (single-cycle RV32I CPU) ⭐
- [ ] Month 9 — first AI accelerator (4×4 INT8 systolic on FPGA) 🔧
- [ ] Month 12 — 8×8 systolic + PyTorch bridge (real LLM ops on your chip)
- [ ] Month 16 — first Yosys+OpenSTA on Sky130
- [ ] Month 17 — first full RTL→GDSII (OpenLane on Sky130)
- [ ] Month 21 — 🎉 TinyTapeout submission live (real silicon coming back)
- [ ] **Month 24 — LLM MAGNUM OPUS SHIPPED ⭐** (HARD DEADLINE)
- [ ] Month 27 — Transformer-block accelerator on FPGA running real LLaMA layer
- [ ] Month 28 — Architecture comparison paper (H100 vs MI300X vs TPU vs Sohu)
- [ ] **Month 36 — SILICON MAGNUM OPUS SHIPPED ⭐⭐⭐** (FINAL — 3 YEARS)
