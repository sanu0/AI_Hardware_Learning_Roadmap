<a id="toc"></a>

# Local AI & On-Device LLM Mastery Roadmap

> **Daily: 1-1.5 hours** (parallel to main LLM/CUDA roadmap) | **Saturday: 2-3 hrs hands-on project** | **Sunday: optional deep dive**
> **Duration:** ~12 months | **6-month milestone:** confidently run, fine-tune, and deploy local LLMs for real workflows
> **After each month:** 1 week revision buffer to consolidate + complete monthly projects
> **Projects:** 1 weekly mini-build + **2 monthly projects** (useful tools that solve YOUR problems first, then others')

> 🎯 **This roadmap runs PARALLEL to your main `LLM + CUDA + Hardware Mastery Roadmap`.**
> Where the main roadmap teaches deep systems theory and big-iron infrastructure, this one teaches you to **own your AI stack** on consumer hardware. Concepts from the main roadmap (quantization, LoRA, RAG, agents, inference serving) will accelerate your progress here; building real local apps will deepen your intuition there. Both feed each other.

---

## 🖥 Your Hardware Profile (Read This First)

This roadmap is calibrated to YOUR specific machine. Internalize these numbers — every model choice, quantization decision, and fine-tuning plan flows from them.

| Component | Spec | Implication for Local AI |
|-----------|------|--------------------------|
| **Primary GPU** | NVIDIA RTX 1000 Ada Generation Laptop GPU | CUDA-capable (compute capability 8.9 — Ada Lovelace). Supports FP16, BF16, INT8, FP8 (limited), Tensor Cores, Flash Attention 2 |
| **Dedicated VRAM** | **6 GB GDDR6** | The hard ceiling. Most models you run must fit (with overhead for KV-cache, context, activations) |
| **Shared GPU memory** | ~17.9 GB (system RAM borrowed) | Used as overflow by some runtimes (slow — PCIe bottleneck). Avoid relying on this for production speed |
| **System RAM** | 31.5 GB | Excellent — enough to run large GGUF models entirely on CPU at acceptable speeds, or partial GPU offload |
| **Secondary GPU** | Intel Arc Pro Graphics | Can be used by OpenVINO / DirectML for some workloads (mostly for offloading non-LLM tasks) |
| **NPU** | Intel AI Boost (Core Ultra) | ~10-13 TOPS. Usable via OpenVINO, DirectML, Windows ML. Good for embedding models and lightweight inference |
| **CPU** | Intel Core Ultra (1.84 GHz base, multi-core) | Strong AVX-512 / VNNI for CPU inference (llama.cpp benefits). Can run 7B Q4 at ~5-10 tok/s on CPU alone |
| **Storage** | SSD (RAID) | Good — GGUF models are large (3-15 GB each). SSD speed matters for cold load |
| **OS** | Windows 11 (win32 10.0.22631) | CUDA + WSL2 both work. WSL2 recommended for ML tooling (Linux-first ecosystem) |

### 📐 The 6GB VRAM Math (Memorize This)

```text
Approximate VRAM needed = (Parameters × Bits-per-parameter / 8) + KV-cache + activations + overhead

Examples for your 6GB card:
  7B model in FP16:      7B × 2 bytes  = 14.0 GB  ❌ Won't fit on GPU
  7B model in Q8 (8-bit):  7B × 1 byte = 7.0 GB   ❌ Barely won't fit (no headroom)
  7B model in Q4_K_M:     7B × 0.5 byte ≈ 4.0 GB  ✅ Fits with ~2 GB for context/KV
  3B model in FP16:      3B × 2 bytes  = 6.0 GB   ⚠ Tight, no context headroom
  3B model in Q8:        3B × 1 byte   = 3.0 GB   ✅ Fits comfortably
  3B model in Q4:        3B × 0.5 byte = 1.5 GB   ✅ Tons of room for context
  1.5B in FP16:          1.5B × 2 byte = 3.0 GB   ✅ Fits comfortably
  Embedding model (e.g., nomic-embed v1.5, 137M): ~280 MB ✅ Trivial
```

**Bottom line:** Your sweet spot is **7B models at Q4_K_M** or **3B-4B models at Q8** for chat, plus **small embedding models** for RAG. For fine-tuning, you'll use **QLoRA on 3B-7B with Unsloth's memory optimizations** to fit in 6GB.

---

## 📑 Table of Contents

### 🎯 Getting Started
- [North Star](#north-star)
- [Why Local AI Matters (and When It Doesn't)](#why-local-ai-matters-and-when-it-doesnt)
- [How To Use This File](#how-to-use-this-file)
- [Project Philosophy: Useful First, Pretty Later](#project-philosophy-useful-first-pretty-later)
- [Cross-Reference With Main Roadmap](#cross-reference-with-main-roadmap)
- [Progress Overview](#progress-overview)

### 🏆 Monthly Capstone Project Catalog
- [Month 1 — OllaMate + VRAMCalc](#month-1-project-ollamate--your-personal-local-chat-stack)
- [Month 2 — ModelStudio + PromptForge](#month-2-project-a-modelstudio--ggufquantization-lab)
- [Month 3 — LocalRAG + DocVault](#month-3-project-a-localrag--privacy-first-document-qa)
- [Month 4 — CodeCompanion + ChatVerse](#month-4-project-a-codecompanion--local-code-assistant)
- [Month 5 — TinyTuner + ModelMart](#month-5-project-a-tinytuner--qlora-on-6gb-vram)
- [Month 6 — DomainDoc + FineTuneBench ⭐](#month-6-project-a-domaindoc--fine-tuned-domain-specialist--6-month-milestone)
- [Month 7 — AgentDesk + ToolKit](#month-7-project-a-agentdesk--local-personal-agent)
- [Month 8 — CrewLocal + AgentTrace](#month-8-project-a-crewlocal--multi-agent-local-system)
- [Month 9 — VisionAssist + ScreenSage](#month-9-project-a-visionassist--local-multimodal-ai)
- [Month 10 — VoiceMate + ImageForge](#month-10-project-a-voicemate--fully-offline-voice-assistant)
- [Month 11 — HomeAI + EdgeDeploy](#month-11-project-a-homeai--household-ai-server)
- [Month 12 — MyAI Magnum Opus ⭐⭐](#month-12-project-myai--your-signature-local-ai-system)
- [Monthly Project Tracker](#monthly-project-tracker-2-projects-per-month)

### 📅 Month-by-Month Focus & Capabilities
- [Month 1 — Local AI Foundations](#month-1--local-ai-foundations-ollama-gguf-vram-math)
- [Month 2 — Tooling & Quantization](#month-2--tooling-quantization-modelfiles)
- [Month 3 — Local RAG & Knowledge](#month-3--local-rag--personal-knowledge-bases)
- [Month 4 — Local Apps & Coding Assistants](#month-4--local-applications--coding-assistants)
- [Month 5 — Local Fine-Tuning Foundations](#month-5--local-fine-tuning-foundations-qlora--unsloth)
- [Month 6 — Domain Fine-Tuning ⭐](#month-6--domain-specialization--6-month-milestone)
- [Month 7 — Local Agents Foundations](#month-7--local-agents-foundations)
- [Month 8 — Multi-Agent & Production Agents](#month-8--multi-agent-systems--reliability)
- [Month 9 — Multimodal Local AI](#month-9--multimodal-local-ai-vision)
- [Month 10 — Voice, Image & Generative](#month-10--voice-image--generative-local-ai)
- [Month 11 — Edge, NPU & Home Server](#month-11--edge-npu--home-server-deployment)
- [Month 12 — Magnum Opus + Open Source](#month-12--magnum-opus--open-source-contribution)

### 🛠 Phase 1: Foundations (Weeks 1-8, Months 1-2)
- [Week 0 — Setup & First Local Model](#week-0--setup--first-local-model-do-this-immediately)
- [Week 1 — Ollama Deep Dive](#week-1--ollama-deep-dive)
- [Week 2 — llama.cpp & GGUF Quantization Formats](#week-2--llamacpp--gguf-quantization-formats)
- [Week 3 — Model Selection & VRAM Engineering](#week-3--model-selection--vram-engineering)
- [Week 4 — Local Inference Frontends](#week-4--local-inference-frontends-lm-studio-jan-open-webui)
- [🔄 Buffer Week (Month 1 Revision)](#-buffer-week-month-1-revision-local-ai)
- [Week 5 — Modelfiles, System Prompts & API Integration](#week-5--modelfiles-system-prompts--api-integration)
- [Week 6 — Quantization Beyond GGUF (GPTQ, AWQ, EXL2)](#week-6--quantization-beyond-gguf-gptq-awq-exl2)
- [Week 7 — Prompt Engineering for Small Local Models](#week-7--prompt-engineering-for-small-local-models)
- [Week 8 — Local Benchmarking & Evaluation](#week-8--local-benchmarking--evaluation)

### 🚀 Phase 2: Local Apps & RAG (Weeks 9-16, Months 3-4)
- [🔄 Buffer Week (Month 2 Revision)](#-buffer-week-month-2-revision-local-ai)
- [Week 9 — Local Embeddings & Vector Stores](#week-9--local-embeddings--vector-stores)
- [Week 10 — Local RAG End-to-End](#week-10--local-rag-end-to-end)
- [Week 11 — Advanced Local RAG (Hybrid, Rerank, Graph)](#week-11--advanced-local-rag-hybrid-rerank-graph)
- [Week 12 — Document Processing Pipelines](#week-12--document-processing-pipelines)
- [🔄 Buffer Week (Month 3 Revision)](#-buffer-week-month-3-revision-local-ai)
- [Week 13 — Local Coding Assistants](#week-13--local-coding-assistants-continue-cody-aider)
- [Week 14 — Chat UIs & Self-Hosted Platforms](#week-14--chat-uis--self-hosted-platforms)
- [Week 15 — Privacy, Security & Network Isolation](#week-15--privacy-security--network-isolation)
- [Week 16 — Local AI Productivity Workflows](#week-16--local-ai-productivity-workflows)

### 🔧 Phase 3: Local Fine-Tuning (Weeks 17-24, Months 5-6)
- [🔄 Buffer Week (Month 4 Revision)](#-buffer-week-month-4-revision-local-ai)
- [Week 17 — QLoRA Theory for 6GB Cards](#week-17--qlora-theory-for-6gb-cards)
- [Week 18 — Unsloth Deep Dive](#week-18--unsloth-deep-dive)
- [Week 19 — Axolotl & LLaMA-Factory](#week-19--axolotl--llama-factory)
- [Week 20 — Dataset Engineering for Fine-Tuning](#week-20--dataset-engineering-for-fine-tuning)
- [🔄 Buffer Week (Month 5 Revision)](#-buffer-week-month-5-revision-local-ai)
- [Week 21 — Your First Real Fine-Tune](#week-21--your-first-real-fine-tune)
- [Week 22 — LoRA Merging, Stacking & GGUF Conversion](#week-22--lora-merging-stacking--gguf-conversion)
- [Week 23 — DPO & Preference Tuning Locally](#week-23--dpo--preference-tuning-locally)
- [Week 24 — Evaluating Fine-Tuned Models ⭐ 6-MONTH MILESTONE](#week-24--evaluating-fine-tuned-models--6-month-milestone)

### 🤖 Phase 4: Local Agents (Weeks 25-32, Months 7-8)
- [🔄 Buffer Week (Month 6 Revision) ⭐](#-buffer-week-month-6-revision-local-ai--6-month-milestone)
- [Week 25 — Function Calling with Local LLMs](#week-25--function-calling-with-local-llms)
- [Week 26 — smolagents, LangGraph & Ollama Agents](#week-26--smolagents-langgraph--ollama-agents)
- [Week 27 — Tool Use Reliability & Error Handling](#week-27--tool-use-reliability--error-handling)
- [Week 28 — Local Code-Executing Agents](#week-28--local-code-executing-agents)
- [🔄 Buffer Week (Month 7 Revision)](#-buffer-week-month-7-revision-local-ai)
- [Week 29 — Multi-Agent Systems Locally](#week-29--multi-agent-systems-locally-crewai-autogen)
- [Week 30 — Agent Memory & Persistence](#week-30--agent-memory--persistence-mem0-letta)
- [Week 31 — Agent Observability & Debugging](#week-31--agent-observability--debugging)
- [Week 32 — Browser & Desktop Automation Agents](#week-32--browser--desktop-automation-agents)

### 🎨 Phase 5: Multimodal & Generative (Weeks 33-40, Months 9-10)
- [🔄 Buffer Week (Month 8 Revision)](#-buffer-week-month-8-revision-local-ai)
- [Week 33 — Vision-Language Models Locally](#week-33--vision-language-models-locally-llava-moondream-minicpm-v)
- [Week 34 — OCR + Vision-LLM Pipelines](#week-34--ocr--vision-llm-pipelines)
- [Week 35 — Local Speech-to-Text (Whisper)](#week-35--local-speech-to-text-whisperfaster-whisperdistil-whisper)
- [Week 36 — Local Text-to-Speech (Piper, Coqui, XTTS)](#week-36--local-text-to-speech-piper-coqui-xtts)
- [🔄 Buffer Week (Month 9 Revision)](#-buffer-week-month-9-revision-local-ai)
- [Week 37 — Voice Assistants End-to-End](#week-37--voice-assistants-end-to-end)
- [Week 38 — Stable Diffusion Locally](#week-38--stable-diffusion-locally-comfyui-a1111-fooocus)
- [Week 39 — SD Fine-Tuning (DreamBooth, LoRA)](#week-39--sd-fine-tuning-dreambooth-lora-textual-inversion)
- [Week 40 — Local Video & Audio Generation](#week-40--local-video--audio-generation)

### 🏭 Phase 6: Edge, Production & Mastery (Weeks 41-52, Months 11-12)
- [🔄 Buffer Week (Month 10 Revision)](#-buffer-week-month-10-revision-local-ai)
- [Week 41 — Home Server Architecture](#week-41--home-server-architecture-for-local-ai)
- [Week 42 — NPU & Intel AI Boost (OpenVINO, DirectML)](#week-42--npu--intel-ai-boost-openvino-directml)
- [Week 43 — Mobile Inference (MLC-LLM, llama.cpp Mobile)](#week-43--mobile-inference-mlc-llm-llamacpp-mobile)
- [Week 44 — Jetson, Raspberry Pi & Edge Devices](#week-44--jetson-raspberry-pi--edge-devices)
- [🔄 Buffer Week (Month 11 Revision)](#-buffer-week-month-11-revision-local-ai)
- [Week 45-48 — Magnum Opus Build](#week-45-48--magnum-opus-build-myai-personal-ai-os)
- [Week 49-50 — Open Source Contribution](#week-49-50--open-source-contribution-llamacpp-ollama-unsloth)
- [Week 51-52 — Polish, Publish & Plan Next Phase](#week-51-52--polish-publish--plan-next-phase)

### 📦 Appendix: Local AI Knowledge Base
- [Models That Fit 6GB VRAM (Curated List)](#models-that-fit-6gb-vram-curated-list)
- [Quantization Format Cheat Sheet](#quantization-format-cheat-sheet)
- [Essential Tools & Frameworks](#essential-tools--frameworks)
- [Books, Courses & Papers](#books-courses--papers)
- [GitHub Repos to Study](#github-repos-to-study-local-ai)
- [Communities, Newsletters & Channels](#communities-newsletters--channels)
- [Troubleshooting Cheat Sheet](#troubleshooting-cheat-sheet)
- [Cost & Energy Reality Check](#cost--energy-reality-check)

---

## North Star

Become the kind of engineer who can take any LLM idea and **make it run on the hardware in front of them** — no cloud bill, no API rate limits, no data leaving the machine. Your destination:

```text
arbitrary task -> pick the smallest model that works -> quantize to fit VRAM
-> fine-tune on your data -> wrap in an agent/RAG/voice interface
-> deploy on home server / phone / NPU -> ship as a real product
```

After this roadmap, you should not just "know Ollama." You should be able to:

- Look at any new HuggingFace model and instantly know whether it fits your 6 GB card, at what quant, and how fast it will run
- Pick the right quantization format (GGUF Q4_K_M vs GPTQ vs AWQ vs EXL2) based on the deployment target
- Fine-tune a 7B model on YOUR data using QLoRA + Unsloth in under 2 hours on 6 GB VRAM
- Build a fully offline personal assistant: RAG + voice + agent + tools, no cloud anywhere
- Ship a local AI product that runs on someone else's modest laptop and is delightful to use
- Contribute meaningfully to llama.cpp, Ollama, Unsloth, or LM Studio
- Reason about the cost / privacy / latency / quality tradeoffs of local vs cloud for any workload

The advantage of local AI is **agency**: your data stays yours, your apps work on planes, your costs are zero after the electricity bill, and you understand the entire stack from disk to display.

---

## Why Local AI Matters (and When It Doesn't)

### Local AI is the right choice when:
- **Privacy is non-negotiable** — medical, legal, financial, journalism, internal company data
- **Cost matters at scale** — API calls add up; a $300 used GPU pays back in months
- **Latency must be predictable** — no network round-trips, no rate limits, no provider outages
- **You're offline / air-gapped** — planes, ships, secure facilities, rural areas, field work
- **You want to learn deeply** — running your own stack teaches more than calling an API
- **Customization is critical** — fine-tune for niche domains where cloud models won't bother
- **Hobbyist / curiosity / autonomy** — owning your AI is its own reward

### Local AI is the WRONG choice when:
- You need GPT-4-class reasoning on hard math or rare languages (no open model matches yet for some tasks)
- One-off scripts where API cost is trivial (~$1) and you just need an answer
- Workloads requiring 70B+ models and you don't have 2× consumer GPUs or a Mac with unified memory
- Production traffic > a few QPS where dedicated cloud infrastructure makes more sense
- Real-time multimodal at scale (video understanding, speech with sub-100ms latency end-to-end)

**Honest expectation setting:** A 7B Q4 model on your laptop will not match GPT-4. It WILL match (or beat) GPT-3.5 on many tasks, and it's free + private. Calibrate.

---

## How To Use This File

- **Check off items** as you complete them: change `[ ]` to `[x]`
- Each **day = 1-1.5 hours** of focused work (lighter than main roadmap because it's parallel)
- **Saturday** = the weekly project — a small useful thing you'll actually keep using
- **Sunday** = optional deep dive (paper, YouTube tutorial, exploration)
- If a day's content is light because your main roadmap had a heavy day, that's fine — keep momentum
- **After every month:** 1 buffer week to revise + ship the 2 monthly projects
- **Monthly projects:** must use what you learned that month AND in prior months (cumulative)
- **Connect both roadmaps:** when you learn quantization in Month 4 of the main roadmap, that's exactly what Month 2 here teaches you to apply with GGUF. Cross-pollinate ruthlessly.
- **Keep a journal** in this folder (`./journal/`): what model you tried, what worked, tokens/sec measured, what broke
- **Every project gets a README + a demo GIF/video** — local AI is hard to evangelize without demos

---

## Project Philosophy: Useful First, Pretty Later

Every monthly project here must pass **the personal usefulness test:** *Will I use this thing every week after I build it?* If the answer is no, redesign.

Local AI projects have a unique advantage: **you are also the user.** Build the thing you wish existed for your own workflow first, then polish for others.

- **Project A:** a deep technical implementation of the month's main theme
- **Project B:** a tool/product that solves a specific personal pain point using Project A's skills

### Local AI Novelty Scorecard

| Question | Must Be True |
|----------|--------------|
| Will I use this myself, weekly? | Yes — you are your own first customer |
| Does it run fully offline? | Yes (unless the project's explicit purpose is comparing local vs cloud) |
| Does it work on 6 GB VRAM? | Yes — that's the constraint that makes it impressive |
| What pain does it remove? | Privacy concern, API cost, latency, offline gap, customization need |
| What's the novel angle? | Most local AI tools are technical demos. Yours should solve a real workflow |
| Can a non-engineer use it? | After v1: yes. README + one-command install + sane defaults |
| Would you trust it with personal data? | Yes — verify no telemetry, no network calls without your consent |

### Definition of Done for Local AI Projects

A local AI project is done when:
- [ ] Works on a clean machine following only your README
- [ ] Has a one-command install (script, Docker, or Modelfile)
- [ ] Includes the exact models used + quantization level + measured tokens/sec on your hardware
- [ ] Documents minimum VRAM and tested hardware
- [ ] Has a "limitations" section: where it breaks, what it can't do
- [ ] Has a benchmark or evaluation (even informal: "20 test queries, 18 acceptable")
- [ ] Has a demo: GIF, video, or screenshots
- [ ] Has a "next improvements" list — what you'd add with more time / better hardware
- [ ] Network monitor confirmed: no outbound calls during normal operation (privacy proof)

### Anti-Tutorial Rule (Local AI Edition)

There are 10,000 "I built a local chatbot with Ollama" repos. To rise above:
- Solve a SPECIFIC workflow problem (e.g., "summarize my email backlog every morning" not "chat")
- Show measured numbers: tokens/sec on RTX 1000 Ada, RAM used, latency for first token
- Compare against the obvious cloud alternative: "this replaces $X/month of ChatGPT for me"
- Include a "why local?" section that's honest, not zealot
- Document the failure cases — small models hallucinate; show how you handle it

### Expertise Gates

| Point | You Should Be Able To Do |
|-------|---------------------------|
| End of Month 1 | Run any GGUF model, predict its VRAM use, and measure tokens/sec without help |
| End of Month 3 | Build a useful local RAG app over personal documents that you actually use weekly |
| End of Month 6 ⭐ | Fine-tune a 3-7B model with QLoRA on your data, evaluate it, and deploy via Ollama |
| End of Month 8 | Ship a multi-tool local agent that reliably completes 80%+ of realistic tasks |
| End of Month 10 | Build voice + vision + chat in one local assistant (Whisper + LLaVA + Llama + Piper) |
| End of Month 12 | Have a flagship local AI product, an OSS contribution, and a complete portfolio |

---

## Cross-Reference With Main Roadmap

This roadmap is designed to **reinforce and apply** what you learn in the main roadmap. Use the table below to align both — when you hit a concept in the main roadmap, the parallel application here will be hands-on.

| Main Roadmap (Month) | Topic | Apply Here (This Roadmap) |
|---------------------|-------|---------------------------|
| Month 1 — GPU/CUDA basics | Memory hierarchy, HBM, roofline | Month 1 — Understand why 6GB matters, measure your real bandwidth |
| Month 2 — Transformers, GPT, profiling | Attention, KV-cache, GPT architecture | Month 1-2 — KV-cache is why context length costs VRAM; profile local models |
| Month 3 — Distributed, SFT, DPO, Flash Attention | Fine-tuning fundamentals | Month 5-6 — Apply SFT and DPO with Unsloth on your laptop |
| Month 4 — PEFT, Quantization | LoRA, QLoRA, GPTQ, AWQ, GGUF | Month 2 + Month 5 — Quantize models yourself; understand every format |
| Month 5 — Inference Serving, RAG, Agents, NVIDIA stack | Production inference, RAG, agents | Months 3-4 + Months 7-8 — Build local versions of each |
| Month 6 — Distillation, RLHF, reasoning, long context ⭐ | Reasoning, alignment | Month 6 + Month 7 — Distill into smaller models that fit your GPU; reasoning prompts |
| Month 7 — Multi-modal, diffusion | CLIP, ViT, LLaVA, SD | Month 9-10 — Run all of these locally |
| Month 9 — Kernel engineering | Triton, custom kernels | Reflective — appreciate what llama.cpp authors did for you |
| Month 11 — Paper implementation | Reproduce research | Month 12 — Implement a recent local-AI paper as your capstone |
| Month 13 — Production serving | Auto-scaling, routing | Month 11 — Build your home AI server with similar patterns at smaller scale |

**Net effect:** by the time you finish Month 6 of the main roadmap (the milestone), you'll have completed Month 4-5 here, and your local skills will be screaming-fast because the theory backs them.

---

## Progress Overview

| Phase | Weeks | Months | Status |
|-------|-------|--------|--------|
| Phase 1: Local AI Foundations | 1-8 | 1-2 | ⬜ Not Started |
| Phase 2: Local Apps & RAG | 9-16 | 3-4 | ⬜ Not Started |
| Phase 3: Local Fine-Tuning ⭐ | 17-24 | 5-6 | ⬜ Not Started |
| Phase 4: Local Agents | 25-32 | 7-8 | ⬜ Not Started |
| Phase 5: Multimodal & Generative | 33-40 | 9-10 | ⬜ Not Started |
| Phase 6: Edge, Production & Mastery | 41-52 | 11-12 | ⬜ Not Started |

---

# ═══════════════════════════════════════════════════════════
# MONTHLY CAPSTONE PROJECT CATALOG (Build Only When Reached)
# ═══════════════════════════════════════════════════════════

> **TWO projects per month.** Each must combine the current month's skills with all prior months.
> Each must work on **YOUR 6GB hardware**, fully offline (unless explicitly comparing to cloud).
> These are projects you'll actually use AND that demonstrate real local-AI craft to interviewers.

---

## Month 1 Project: "OllaMate — Your Personal Local Chat Stack"
**When to build:** Month 1 revision week, after Weeks 1-4.
**What:** A polished local chat app on top of Ollama with multi-model switching, system prompt library, conversation history, and tokens/sec display — your daily-driver replacement for ChatGPT for non-sensitive tasks.
**Novelty:** Not just an Ollama UI — includes a **"model recommender"** that, given your prompt, picks the best model from your library based on past performance + speed + VRAM. Tracks which model answered which question best so it learns your preferences.
**Deliverables:**
- [ ] Streamlit or Gradio web UI (runs locally, accessible at `localhost:7860`)
- [ ] Multi-model switcher: Llama 3.2, Qwen 2.5, Phi-3, Gemma 2 (all Q4_K_M for 6GB)
- [ ] System prompt library: 10+ named personas (coder, writer, summarizer, tutor, etc.)
- [ ] Conversation history persisted to SQLite, searchable
- [ ] Tokens/sec + VRAM use shown live during generation
- [ ] Export conversations to markdown
- [ ] **Publish to GitHub with screenshots + setup script**

## Month 1 Project B: "VRAMCalc — Local Model Compatibility Predictor"
**When to build:** Month 1 revision week.
**What:** A web tool that, given a HuggingFace model name and your GPU (defaults to RTX 1000 Ada 6GB), predicts: will it fit? At what quant? Expected tokens/sec? Recommended context length?
**Novelty:** Most calculators are static formulas. Yours **actually pulls model metadata from HuggingFace API**, infers parameter count from config.json, accounts for KV-cache, and gives a confidence-graded answer. Includes a "cheaper alternative" suggestion if the model won't fit.
**Skills used:** VRAM math (Week 1), quantization formats (Week 2), Ollama Modelfile knowledge (Week 5), HuggingFace API
**Deliverables:**
- [ ] Input: HF model URL/ID, GPU VRAM, target context length
- [ ] Output: fits at FP16? Q8? Q5? Q4? With/without context margin
- [ ] Estimated tokens/sec based on a small benchmark database you collect
- [ ] "Alternative models" suggestion if it doesn't fit (smaller cousins)
- [ ] One-click Ollama Modelfile generation
- [ ] CLI version: `vramcalc meta-llama/Llama-3.1-8B --gpu 6`
- [ ] **Publish to GitHub**

---

## Month 2 Project A: "ModelStudio — GGUF/Quantization Lab"
**What:** A workflow tool that takes any HuggingFace model, quantizes it to multiple GGUF formats (Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q4_0, IQ4_XS, Q3_K_M), benchmarks each on your hardware, and produces a quality-vs-speed-vs-size report.
**Novelty:** A **per-quant quality estimator** that uses perplexity on a calibration set PLUS task-specific accuracy (math, code, factual recall) so you see WHICH abilities degrade at which quantization level. Most tools only show perplexity.
**Deliverables:**
- [ ] Pipeline: HF model → llama.cpp convert.py → quantize at 5+ levels
- [ ] Benchmarks per quant: perplexity (wikitext-103), GSM8K subset, HumanEval subset, MMLU-mini
- [ ] tokens/sec and first-token latency on your RTX 1000 Ada
- [ ] HTML report with charts (Plotly): quality vs file size vs speed
- [ ] CLI: `modelstudio --model meta-llama/Llama-3.2-3B --quants Q4_K_M,Q5_K_M,Q8_0`
- [ ] One-click push to Ollama: creates Modelfile and `ollama create`
- [ ] **Publish as pip-installable**

## Month 2 Project B: "PromptForge — Local Prompt Engineering Workbench"
**What:** A side-by-side tool that runs the same prompt through multiple local models, compares outputs, and saves winning prompts to a versioned library. Includes a "small model whisperer" mode that suggests prompt rewrites optimized for 3-7B models.
**Novelty:** Most prompt tools target GPT-4. Yours targets **small local models**, where prompt quality matters 10× more. Includes templates from research (CoT prompts proven to work on 3B models, structured-output tricks that help small models avoid drift).
**Skills used:** Modelfiles (Week 5), prompt engineering for small models (Week 7), local benchmarking (Week 8), Ollama API
**Deliverables:**
- [ ] UI: write prompt, select 2-4 models, get outputs side-by-side
- [ ] Rating system: thumbs up/down per output, system learns which prompts win
- [ ] "Small-model rewriter": adds CoT, structured output cues, few-shot examples
- [ ] Prompt library with versioning + tags
- [ ] Export winning prompts as Ollama Modelfile SYSTEM prompts
- [ ] Quality diff highlighter: shows where two outputs disagree
- [ ] **Publish to GitHub**

---

## Month 3 Project A: "LocalRAG — Privacy-First Document Q&A"
**What:** A complete offline RAG system you point at any folder (PDFs, markdown, code, docx) and chat with. Uses local embeddings, local vector store, local LLM, with proper citations.
**Novelty:** **Provenance-first design** — every claim in the answer is highlighted with the exact source page/line, and there's a "verify this claim" button that re-retrieves and shows the supporting evidence in context. Designed for people who actually need to trust the output (lawyers, researchers, journalists).
**Deliverables:**
- [ ] Watch-folder mode: auto-ingest new files, incremental indexing
- [ ] Multiple file types: PDF (with PyMuPDF), DOCX, MD, code (AST-aware chunks for .py/.ts/.go)
- [ ] Local embeddings: nomic-embed-text-v1.5 via Ollama or fastembed
- [ ] Vector store: LanceDB (embedded, file-based, no server)
- [ ] Hybrid retrieval: dense + BM25 (Tantivy or rank_bm25)
- [ ] Local reranker: bge-reranker-v2-m3 or similar (optional, off by default for speed)
- [ ] LLM: Qwen 2.5 7B Q4_K_M (great for citations) or Llama 3.1 8B Q4_K_M
- [ ] Citation rendering: clickable links to exact page/line in original document
- [ ] "Verify" mode: re-retrieves quoted sources, shows them in PDF viewer with highlighting
- [ ] Zero network calls (verified with `tcpdump`/Wireshark — include the verification in README)
- [ ] **Publish with Docker Compose + sample document set**

## Month 3 Project B: "DocVault — Personal Knowledge Vault"
**What:** An Obsidian/Notion-like personal knowledge base where every note is automatically embedded, semantically searchable, AI-summarizable, and you can chat with your whole vault locally.
**Novelty:** Local AI applied to PKM (Personal Knowledge Management). **"Why did past-me write this?"** mode — given a note, finds related notes you wrote, summarizes the thread of thought, surfaces forgotten connections. Replaces Mem.ai for fully-local users.
**Skills used:** Local embeddings (Week 9), RAG (Week 10), hybrid retrieval (Week 11), document processing (Week 12), Markdown editing
**Deliverables:**
- [ ] Markdown-based notes (compatible with Obsidian vault format)
- [ ] Auto-embedding on save with nomic-embed
- [ ] Semantic search across all notes
- [ ] "Chat with vault" mode using LocalRAG infrastructure
- [ ] Auto-link suggestions: "this note relates to X, Y, Z notes"
- [ ] Daily summary generator: summarize today's notes into a weekly digest
- [ ] Question-from-vault prompt: "what did I learn about X in the last 3 months?"
- [ ] Web UI (FastAPI + simple HTML) accessible from phone on local network
- [ ] **Publish to GitHub with sample vault**

---

## Month 4 Project A: "CodeCompanion — Local Code Assistant"
**What:** A VS Code extension (or Aider/Continue config) that uses local Qwen 2.5 Coder 7B for autocomplete + chat + refactor, with full project context via local RAG over your codebase.
**Novelty:** **Per-repo learning** — keeps a per-project memory of conventions (your import style, your test patterns, your naming) and gradually includes them as few-shot examples in prompts. Effectively does "implicit fine-tuning via prompting." Plus benchmarks first-token latency to keep autocomplete snappy on 6GB.
**Deliverables:**
- [ ] VS Code extension OR Continue.dev config OR Aider integration
- [ ] FIM (fill-in-middle) autocomplete using Qwen 2.5 Coder 1.5B (fast)
- [ ] Chat panel using Qwen 2.5 Coder 7B Q4 (better quality, slower)
- [ ] Codebase context: tree-sitter AST chunking + embeddings
- [ ] Per-project conventions file: regenerated weekly from your code
- [ ] "Explain this function" with full call-graph context
- [ ] "Write tests" using your existing test patterns as examples
- [ ] Latency dashboard: FIM should be <500ms TTFT
- [ ] **Publish to GitHub + Marketplace**

## Month 4 Project B: "ChatVerse — Multi-Model Local Chat Router"
**What:** A chat UI where you don't pick a model — the router picks for you based on the query. Easy chitchat → Phi-3 Mini (super fast). Code → Qwen 2.5 Coder. Math → DeepSeek Math. Reasoning → Llama 3.1 8B with CoT system prompt. Vision → Moondream/LLaVA.
**Novelty:** **Local "smart routing"** with a tiny classifier (TinyLlama or sklearn TF-IDF) that decides which specialist to invoke. Saves load time by keeping the routing decision <100ms and only loading the chosen model into VRAM. Includes a UI panel that shows WHY it routed there.
**Skills used:** Multiple models (Months 1-2), Modelfiles (Week 5), prompt engineering (Week 7), classifier basics
**Deliverables:**
- [ ] Router classifier: lightweight (sklearn or distilled BERT) for intent detection
- [ ] Model registry: catalog of specialist models with their strengths
- [ ] Lazy model loading: only one big model in VRAM at a time, swap as needed
- [ ] UI shows route decision + alternative models the user can override
- [ ] Latency budget: routing decision < 100ms, then full generation
- [ ] **Publish to GitHub**

---

## Month 5 Project A: "TinyTuner — QLoRA on 6GB VRAM"
**What:** A push-button fine-tuning tool that uses Unsloth to QLoRA-train any 3-7B model on YOUR data, on YOUR 6GB GPU, in under 2 hours. Outputs a GGUF ready for Ollama.
**Novelty:** **Memory autopilot** — given your VRAM, model, and dataset size, it auto-selects batch size, gradient accumulation, sequence length, LoRA rank, and gradient checkpointing settings to fit. Most fine-tuning fails for newbies because of OOM; this tool refuses to crash.
**Deliverables:**
- [ ] Wrapper around Unsloth with VRAM-aware config selection
- [ ] Supported bases: Llama 3.2 3B, Qwen 2.5 3B/7B, Phi-3 Mini, Gemma 2 2B
- [ ] Dataset formats: JSONL (Alpaca, ChatML, ShareGPT), CSV → auto-conversion
- [ ] Training UI: live loss curve, GPU utilization, ETA
- [ ] Auto-evaluation: held-out test set perplexity + 5 sample generations
- [ ] LoRA merge → GGUF Q4_K_M → Ollama Modelfile in one command
- [ ] "Failed fine-tune doctor": if loss is NaN or stuck, suggests fixes
- [ ] **Publish to GitHub with example dataset (a small personal-style dataset)**

## Month 5 Project B: "ModelMart — Personal Model & Adapter Hub"
**What:** A local registry for all the models, LoRA adapters, and Modelfiles you've created. Tag, version, search, and one-click load. Like a personal HuggingFace Hub on your machine.
**Novelty:** **Adapter stacking explorer** — you can experimentally combine 2-3 LoRA adapters (e.g., "my-writing-style" + "technical-jargon" + "concise-format") and see what comes out. Inspired by Stable Diffusion LoRA stacking, but for text.
**Skills used:** LoRA merging (Week 22), GGUF conversion (Week 22), Modelfile generation (Week 5), file management
**Deliverables:**
- [ ] CLI + web UI: `modelmart list / push / pull / try`
- [ ] SQLite registry with model metadata, training data, eval scores
- [ ] Adapter stacking: load base + multiple LoRAs at runtime via PEFT
- [ ] Storage management: track disk usage, garbage-collect old adapters
- [ ] Auto-Modelfile generation per registered model
- [ ] **Publish to GitHub**

---

## Month 6 Project A: "DomainDoc — Fine-Tuned Domain Specialist" ⭐ 6-MONTH MILESTONE
**When to build:** Month 6 revision week.
**What:** Pick a domain you genuinely care about (e.g., Indian tax law, your codebase, your personal writing voice, Marathi/Hindi grammar, NVIDIA CUDA docs, a hobby like chess openings). Build a fine-tuned 7B model that crushes a generic LLM on this domain, deployed via Ollama with RAG fallback for facts.
**Novelty:** **End-to-end "domain pipeline"** including: data scraping/curation, synthetic data generation with a teacher model, QLoRA fine-tuning, DPO preference tuning, GGUF conversion, RAG-augmented serving, and a domain-specific eval suite. This is the moment everything from Months 1-6 comes together.
**Deliverables:**
- [ ] Domain choice + 1-pager: why this domain, who it helps, sample queries
- [ ] Data pipeline: scrape/collect → clean → format → synthesize → ~5-10K examples
- [ ] QLoRA training with TinyTuner
- [ ] DPO with preference pairs (you label ~200, generate the rest via LLM-as-judge)
- [ ] Custom eval suite: 50 hand-curated domain questions with gold answers
- [ ] Comparison: base model vs your fine-tune vs cloud GPT-4 on your evals
- [ ] Deployment: Ollama Modelfile + RAG over domain docs for hard facts
- [ ] **Publish with blog post: "How I built a domain-specialist LLM on a 6GB laptop"**

## Month 6 Project B: "FineTuneBench — Adapter Evaluation & Comparison Toolkit"
**What:** Take any base model + 2-5 LoRA adapters and produce a comparison report: perplexity, instruction-following score, style adherence, knowledge retention from base, hallucination rate.
**Novelty:** **"Catastrophic forgetting detector"** — automatically checks whether fine-tuning broke base-model abilities (math, coding, multilingual) by running a "general capabilities" eval before/after. Most local fine-tuners discover too late that their fine-tune nuked the model's ability to do anything else.
**Skills used:** Eval methods (Week 24), QLoRA (Week 17-21), DPO (Week 23), all prior
**Deliverables:**
- [ ] Eval harness with: TruthfulQA mini, MMLU mini, HumanEval mini, MT-Bench subset
- [ ] Style metrics: avg sentence length, vocab diversity, formality (textstat)
- [ ] Knowledge retention check: compare base vs fine-tuned on general questions
- [ ] HTML report card per adapter with strengths/weaknesses
- [ ] CLI: `ftbench --base llama3.1-8b --adapters style.lora,domain.lora`
- [ ] **Publish to GitHub**

---

## Month 7 Project A: "AgentDesk — Local Personal Agent"
**What:** A personal assistant agent that runs entirely on your laptop, with tools to: search local files, read/send email (via local IMAP/SMTP, opt-in), execute Python in sandbox, browse the web (Playwright), and maintain a long-term memory of you.
**Novelty:** **"Trust budget"** system — you grant the agent specific permissions (e.g., "can read my Documents folder," "can run code in `/tmp/agent-sandbox`," "cannot access bank tabs in browser"). Visual permission audit every time agent crosses a boundary. Designed for actual daily use, not demo theater.
**Deliverables:**
- [ ] Agent framework: smolagents OR LangGraph + Ollama function-calling-tuned model
- [ ] Tools: filesystem (read-only by default), Python sandbox (Docker or Firejail), Playwright browser, email (opt-in)
- [ ] Memory: SQLite with embeddings for episodic memory; structured profile for semantic memory
- [ ] Permission system with visible audit log
- [ ] Daily routine: morning briefing (calendar + email digest + news from local RSS)
- [ ] Voice trigger optional (uses Whisper if installed)
- [ ] **Publish to GitHub with security audit checklist**

## Month 7 Project B: "ToolKit — Reliable Function Calling for Small Local Models"
**What:** A library that wraps any local LLM and makes it reliably emit valid function calls — even when the base model wasn't trained for tools. Uses outlines/lm-format-enforcer + retry/repair logic + tool-use prompts proven on small models.
**Novelty:** Most function-calling libraries assume GPT-class models. Yours **targets 3-7B models** with structured output guarantees (grammar-constrained), self-repair on schema violations, and a "tool-use confidence" score that flags when the model is guessing.
**Skills used:** Function calling (Week 25), structured output, prompt engineering for small models (Week 7)
**Deliverables:**
- [ ] Wrapper for Ollama / llama-cpp-python with function_calling=True
- [ ] Schema-constrained generation via outlines or lm-format-enforcer
- [ ] Retry loop with error-feedback prompting
- [ ] Confidence scoring (token probabilities at decision points)
- [ ] Benchmark: 50 tool-use tasks across 4 small models, success rate report
- [ ] **Publish as pip-installable**

---

## Month 8 Project A: "CrewLocal — Multi-Agent Local System"
**What:** A local crew of specialized agents (researcher, writer, critic, coder) that collaborate on tasks. Uses small fast models for orchestrator, larger model for synthesis. All on 6GB VRAM via model swapping.
**Novelty:** **VRAM-aware orchestration** — knows the current model in GPU memory, batches calls to same-model agents together to avoid swap thrashing. Includes a "thinking budget" so agents don't loop forever (a real risk with small models).
**Deliverables:**
- [ ] 4 agent personas with distinct system prompts + tool access
- [ ] Orchestrator pattern: planner → workers → critic → finalizer
- [ ] Model registry: which agent uses which model
- [ ] Token + time budget enforced per task
- [ ] 3 end-to-end use cases: research a topic, write a blog post, debug a code issue
- [ ] **Publish to GitHub with demo video**

## Month 8 Project B: "AgentTrace — Local Agent Observability"
**What:** A tracing/debugging tool for local agents. Records every LLM call, tool call, retrieval, retry. Visualizes the agent's "thinking" as a tree. Identifies failure modes: loops, hallucinated tools, wrong arguments.
**Novelty:** Like LangSmith but **fully local** + adds an automated **"failure pattern classifier"** trained on a hand-labeled set of agent failure modes. Tells you "this agent got stuck because it hallucinated a tool that doesn't exist" rather than dumping raw logs.
**Skills used:** Agents (Weeks 25-32), local SQLite, web UI patterns, classifier basics
**Deliverables:**
- [ ] Decorator/middleware for smolagents, LangGraph, custom agents
- [ ] SQLite store for traces, web UI (FastAPI + Alpine.js or Streamlit)
- [ ] Tree visualization of agent thinking (LLM call → tool → LLM call ...)
- [ ] Failure mode classifier: 10 common patterns (loop, fake tool, schema violation, etc.)
- [ ] Replay mode: re-run a failed trace step by step
- [ ] **Publish to GitHub**

---

## Month 9 Project A: "VisionAssist — Local Multimodal AI"
**What:** A local vision assistant: you drag an image (screenshot, photo, chart, diagram, PDF page) into the UI and ask questions. Uses Moondream2 (1.86B params, runs FAST on 6GB) or LLaVA 1.6 Mistral 7B Q4 for harder tasks. Pipeline routes by image complexity.
**Novelty:** **Auto-router for vision** — uses a tiny model first to estimate "is this a simple photo or a complex chart/document?" and routes to the cheapest model that handles it. Plus a **"local ALT text"** mode for accessibility: generates descriptive alt-text for any image batch.
**Deliverables:**
- [ ] UI: drag-image-and-ask
- [ ] Models: Moondream2 (fast path), LLaVA 1.6 7B Q4 (quality path), Phi-3 Vision (alt)
- [ ] Auto-router (image classifier: chart vs photo vs document)
- [ ] Batch mode: process a folder of images → CSV of descriptions
- [ ] ALT text generator for web/blog images
- [ ] OCR fallback (Tesseract or PaddleOCR) integrated for text-heavy images
- [ ] **Publish to GitHub**

## Month 9 Project B: "ScreenSage — Screenshot Q&A & Workflow Assistant"
**What:** A background app that lets you screenshot anything (Ctrl+Shift+S) → ask a question → get a local-VLM answer in seconds. "What is this error?" "Translate this menu." "Summarize this email I'm looking at."
**Novelty:** Uses **OS-level hotkeys + persistent vision model warm in RAM** for sub-3-second response. Lightweight enough to leave running 24/7 on your laptop without noticeable battery drain.
**Skills used:** Vision models (Week 33), OCR (Week 34), system tray apps, OS automation
**Deliverables:**
- [ ] System tray app (PyQt or tauri)
- [ ] Global hotkey for screenshot capture
- [ ] Pre-loaded Moondream2 in memory for fast inference
- [ ] Result overlay window with copy-to-clipboard
- [ ] History of past screenshots + questions, searchable
- [ ] Battery-friendly: unload model after 10 min idle
- [ ] **Publish to GitHub**

---

## Month 10 Project A: "VoiceMate — Fully Offline Voice Assistant"
**What:** Wake-word detection → Whisper ASR → local LLM (Llama 3.1 8B Q4) with tool use → Piper TTS. The Alexa/Siri replacement that doesn't phone home. Runs on your laptop or a Raspberry Pi 5.
**Novelty:** **Full pipeline latency budget** with measured numbers: wake word < 100ms, ASR < 1s for typical query, LLM < 3s TTFT, TTS streaming. Most "local voice assistant" projects take 15+ seconds end-to-end; yours is built for actual conversation.
**Deliverables:**
- [ ] Wake word: openwakeword or Porcupine (offline)
- [ ] ASR: faster-whisper (small or distil-small-en model) — INT8 on CPU is fast
- [ ] LLM: Llama 3.1 8B Q4_K_M with function calling for smart home / search / timer / calendar
- [ ] TTS: Piper (low latency, decent quality) or Coqui XTTS for voice cloning
- [ ] Streaming TTS: start speaking before LLM finishes generating
- [ ] Tool integration: timers, calendar (local ICS), weather (cached or local RSS), smart home (Home Assistant API if available)
- [ ] Measured end-to-end latency report
- [ ] **Publish to GitHub with demo video**

## Month 10 Project B: "ImageForge — Local Image Generation Studio"
**What:** A self-hosted Stable Diffusion studio (ComfyUI base) with curated workflows, your own trained LoRAs, and an "everyday image" preset library (logos, illustrations, photos for blog, profile pics).
**Novelty:** **6GB-optimized workflows** — most SD UIs assume 8GB+. Yours has tested presets for SDXL Lightning/Turbo, SD 1.5 with VAE tiling, and Cascade Stage C lowmem mode. Plus a **"prompt-to-product"** preset library where each preset is tuned for a specific output (blog hero image, podcast cover, etc.).
**Skills used:** Stable Diffusion (Week 38), LoRA training (Week 39), workflow tools (ComfyUI)
**Deliverables:**
- [ ] ComfyUI setup with low-VRAM presets
- [ ] 10 curated workflows for common needs
- [ ] 2-3 self-trained LoRAs (your style, a specific subject)
- [ ] Batch generation mode with prompt variations
- [ ] Watermark + metadata embedding for tracking AI-generated images
- [ ] **Publish workflows + LoRAs to GitHub**

---

## Month 11 Project A: "HomeAI — Household AI Server"
**What:** Deploy a local AI server on your home network (your laptop running headless, or an old PC, or a dedicated mini-PC) that all your family devices can use. Web UI for chat, RAG over family documents (recipes, manuals, schedules), photo search ("find that picture from Goa trip"), voice assistant accessible from anywhere on Wi-Fi.
**Novelty:** **Multi-user with profiles** — each family member has their own conversation history, preferences, and isolated RAG vault. **Network-only access** (LAN, never internet). Includes a phone-friendly PWA so kids/parents who aren't techies can use it.
**Deliverables:**
- [ ] Server setup: Docker Compose with Ollama, Open WebUI, LocalRAG, ImageForge
- [ ] Reverse proxy (Caddy) with mDNS hostname (e.g., `homeai.local`)
- [ ] User profiles: 4-5 family member accounts with isolation
- [ ] Family knowledge base: medical records (private per user), recipes (shared), manuals (shared)
- [ ] Photo search using CLIP embeddings of your photo library
- [ ] Voice access from phones via PWA + WebRTC
- [ ] Privacy audit: confirmed no outbound traffic
- [ ] **Publish architecture diagram + setup guide**

## Month 11 Project B: "EdgeDeploy — Deploy Local AI to Mobile & Edge"
**What:** Take one of your fine-tuned models and deploy it to a mobile phone (via MLC-LLM or llama.cpp on Android/iOS), a Raspberry Pi 5, and the Intel AI Boost NPU. Document every step with benchmarks per platform.
**Novelty:** **Cross-platform comparison report** — same model, 4 platforms (RTX 1000 Ada, Intel NPU, Raspberry Pi 5, Pixel phone), reporting tokens/sec, watts, latency, and the practical trade-offs. Useful artifact for the community.
**Skills used:** Mobile inference (Week 43), edge devices (Week 44), NPU/OpenVINO (Week 42), all deployment skills
**Deliverables:**
- [ ] Same model (e.g., Qwen 2.5 1.5B Q4) on 4 platforms
- [ ] Mobile: MLC-LLM Android APK or llama.cpp termux build
- [ ] Pi 5: llama.cpp with ARM NEON optimization
- [ ] NPU: OpenVINO conversion + benchmarks
- [ ] Comparison table + write-up
- [ ] **Publish to GitHub with benchmark CSV + photos of running on each device**

---

## Month 12 Project: "MyAI — Your Signature Local AI System" ⭐⭐
**What:** Your magnum opus. The flagship project that uses everything from Months 1-11. Build something you'd be proud to demo at NVIDIA, in a job interview, or to a friend who knows nothing about AI.
**Choose ONE (or combine):**

**Option A: "Personal AI Operating System"**
- [ ] Voice + chat + vision in one local app
- [ ] RAG over all your documents, code, emails, notes
- [ ] Fine-tuned model in YOUR voice/style
- [ ] Multi-agent for complex tasks (research, write, code, schedule)
- [ ] Daily proactive briefings (morning routine, weekly review)
- [ ] Plugin system: others can add tools/personas
- [ ] Runs on your laptop, accessible from your phone

**Option B: "Domain-Specialist AI Product"**
- [ ] Pick a niche audience (e.g., Indian students preparing for JEE, indie writers, Marathi-language users, hobby chess players)
- [ ] Fine-tuned domain model + RAG over domain corpus
- [ ] Polished web UI optimized for the user's workflow
- [ ] Local deployment instructions + cloud option for non-techies
- [ ] Real users (5-10) testing and giving feedback
- [ ] Monetization story (if any): donations, paid tier, hosted version

**Option C: "Local AI Developer Tool"**
- [ ] A tool that helps other local-AI developers
- [ ] Examples: a better Modelfile editor, a fine-tuning recipe library, a model recommendation engine
- [ ] Polish: good README, examples, tests, CI
- [ ] Get adopted: at least one external user or contributor

**Deliverables for any option:**
- [ ] GitHub repo with excellent documentation and architecture diagrams
- [ ] 2000+ word technical blog post explaining the design
- [ ] 5-10 minute demo video
- [ ] Benchmarks: latency, VRAM use, quality, comparison to alternatives
- [ ] **At least one external user trying it**
- [ ] **This is your portfolio centerpiece for local AI**

## Month 12 Project B: "Open Source Contribution to Local AI Stack"
**What:** Make a meaningful contribution to one of: llama.cpp, Ollama, Unsloth, LM Studio (closed source — no), Open WebUI, KoboldCPP, ComfyUI, MLC-LLM, smolagents.
**Novelty:** This IS the novelty — you're now a contributor to the tools you've been using.
**Deliverables:**
- [ ] Identify 2-3 candidate projects from your daily use
- [ ] Find issues labeled "good first issue" OR fix a bug YOU hit OR add a feature YOU need
- [ ] Submit at least 1 merged PR (preferably 2)
- [ ] Write a blog post: "How I added X to Y, and how Y's codebase works"
- [ ] **Visible, merged contribution to a tool used by thousands**

---

### Monthly Project Tracker (2 projects per month)

| Month | Project A | ✓ | Project B | ✓ |
|-------|-----------|---|-----------|---|
| 1 | OllaMate — Local Chat Stack | ⬜ | VRAMCalc — Compatibility Predictor | ⬜ |
| 2 | ModelStudio — GGUF/Quant Lab | ⬜ | PromptForge — Prompt Workbench | ⬜ |
| 3 | LocalRAG — Privacy-First Doc Q&A | ⬜ | DocVault — Personal Knowledge Base | ⬜ |
| 4 | CodeCompanion — Local Code Assistant | ⬜ | ChatVerse — Multi-Model Router | ⬜ |
| 5 | TinyTuner — QLoRA on 6GB | ⬜ | ModelMart — Model & Adapter Hub | ⬜ |
| 6 | DomainDoc — Fine-Tuned Specialist ⭐ | ⬜ | FineTuneBench — Adapter Eval | ⬜ |
| 7 | AgentDesk — Local Personal Agent | ⬜ | ToolKit — Reliable Function Calls | ⬜ |
| 8 | CrewLocal — Multi-Agent System | ⬜ | AgentTrace — Agent Observability | ⬜ |
| 9 | VisionAssist — Local Multimodal | ⬜ | ScreenSage — Screenshot Q&A | ⬜ |
| 10 | VoiceMate — Voice Assistant | ⬜ | ImageForge — Image Gen Studio | ⬜ |
| 11 | HomeAI — Household Server | ⬜ | EdgeDeploy — Cross-Platform Benchmark | ⬜ |
| 12 | MyAI — Magnum Opus ⭐⭐ | ⬜ | OSS Contribution to Local AI Tool | ⬜ |

[⬆ Back to Table of Contents](#toc)

---

# ═══════════════════════════════════════════════════════════
# MONTH-BY-MONTH FOCUS & CAPABILITIES
# "After this month, I can..."
# ═══════════════════════════════════════════════════════════

---

### Month 1 — Local AI Foundations (Ollama, GGUF, VRAM Math)
**Focus:** Install and master Ollama + llama.cpp. Understand GGUF quantization. Internalize VRAM math. Run 5+ models. Measure speed. Pick the right model for the task.
**Project:** OllaMate (chat stack) + VRAMCalc (compatibility predictor)

**After this month, you can:**
- [ ] Install Ollama on Windows/WSL2 and run any model with one command
- [ ] Predict whether a model will fit in your 6GB VRAM without trial-and-error
- [ ] Explain GGUF, the K-quant family (Q2_K through Q8_0), and pick the right level
- [ ] Measure tokens/sec and first-token latency consistently
- [ ] Use the Ollama API from Python/curl programmatically
- [ ] Choose between Ollama, llama.cpp directly, LM Studio, Jan, and Open WebUI for any use case
- [ ] Explain the difference between context length, KV-cache, and parameter memory
- [ ] Build a polished local-chat web app you actually use daily

---

### Month 2 — Tooling, Quantization, Modelfiles
**Focus:** Quantize models yourself with llama.cpp's `quantize` binary. Compare quantization formats (GGUF vs GPTQ vs AWQ vs EXL2). Write Modelfiles. Master system prompts for small models. Benchmark.
**Project:** ModelStudio (quantization lab) + PromptForge (prompt workbench)

**After this month, you can:**
- [ ] Convert any HuggingFace model to GGUF and quantize at multiple levels
- [ ] Read a Modelfile and write your own (system prompts, parameters, templates)
- [ ] Explain GPTQ vs AWQ vs GGUF vs EXL2 and when each wins
- [ ] Run llama.cpp / Ollama with custom parameters (num_ctx, num_gpu, num_thread, temperature)
- [ ] Benchmark a model on YOUR hardware with reproducible numbers
- [ ] Engineer prompts that work on 3-7B models (smaller models need more guidance than GPT-4)
- [ ] Use structured output techniques (JSON mode, grammars) to control small-model behavior

---

### Month 3 — Local RAG & Personal Knowledge Bases
**Focus:** Local embeddings (nomic-embed, BGE, mxbai). Vector stores that don't need a server (LanceDB, ChromaDB embedded, Qdrant local). End-to-end RAG. Document processing.
**Project:** LocalRAG (privacy-first doc Q&A) + DocVault (personal knowledge vault)

**After this month, you can:**
- [ ] Run local embedding models efficiently and explain the speed/quality tradeoffs
- [ ] Set up an embedded vector store (LanceDB) without managing a server
- [ ] Build end-to-end RAG over PDFs, markdown, code with proper citations
- [ ] Implement hybrid retrieval (dense + BM25) for better recall
- [ ] Run a local reranker (bge-reranker-v2) for quality boost when needed
- [ ] Process diverse documents: PDFs with tables, OCR for scans, AST-aware code chunking
- [ ] Verify a RAG system is truly offline (network monitoring)
- [ ] Build a personal knowledge management system you'd trust with your private notes

---

### Month 4 — Local Applications & Coding Assistants
**Focus:** Real apps you'll keep using. Code assistants (Continue, Aider, Cody offline). Chat UIs (Open WebUI, LibreChat, Anything LLM). Privacy and security audit. Productivity workflows.
**Project:** CodeCompanion (local code assistant) + ChatVerse (multi-model router)

**After this month, you can:**
- [ ] Replace GitHub Copilot with a fully local code completion + chat system (FIM with Qwen 2.5 Coder)
- [ ] Self-host Open WebUI or LibreChat for a polished daily-driver chat experience
- [ ] Verify your local AI stack makes ZERO outbound calls (Wireshark, Little Snitch / GlassWire)
- [ ] Route queries to specialist models (code → Qwen Coder, math → DeepSeek Math, etc.)
- [ ] Integrate local AI into your daily workflow: writing, coding, research
- [ ] Decide which tasks are worth local vs cloud based on quality/privacy/cost analysis

---

### Month 5 — Local Fine-Tuning Foundations (QLoRA + Unsloth)
**Focus:** QLoRA theory for 6GB cards. Unsloth deep dive (the game-changer for low-VRAM fine-tuning). Axolotl & LLaMA-Factory. Dataset engineering. Your first real fine-tune.
**Project:** TinyTuner (push-button QLoRA) + ModelMart (model & adapter hub)

**After this month, you can:**
- [ ] Explain QLoRA from first principles: NF4 base + LoRA in FP16, gradient flow
- [ ] Use Unsloth to fine-tune a 7B model on 6GB VRAM (Unsloth's memory optimizations are essential)
- [ ] Use Axolotl or LLaMA-Factory for more complex configs (multi-GPU, advanced schedulers)
- [ ] Prepare a clean instruction dataset (Alpaca, ChatML, ShareGPT formats)
- [ ] Diagnose common fine-tuning failures: NaN loss, no improvement, catastrophic forgetting, OOM
- [ ] Auto-tune training hyperparameters based on available VRAM

---

### Month 6 — Domain Specialization ⭐ (6-MONTH MILESTONE)
**Focus:** Bring everything together — pick a domain you care about, build dataset, fine-tune, evaluate, deploy. Add DPO for preference tuning. Build domain-specific evals. Convert to GGUF + Ollama.
**Project:** DomainDoc (fine-tuned domain specialist) + FineTuneBench (adapter evaluation)

**After this month, you can:**
- [ ] Take a domain idea from zero to deployed model in 2 weeks
- [ ] Build a curated dataset from scratch (scrape + clean + synthesize)
- [ ] Apply SFT then DPO to fine-tune for both knowledge and style
- [ ] Detect catastrophic forgetting and mitigate it (mix in general data, lower LR, fewer epochs)
- [ ] Build a domain-specific evaluation suite that proves your model is genuinely better
- [ ] Convert LoRA → merged GGUF → Ollama Modelfile → daily-use chat
- [ ] **You have a real, deployed, fine-tuned model that works on YOUR data, on YOUR hardware**
- [ ] **6-MONTH MILESTONE: you can credibly claim "Local AI Practitioner" status**

---

### Month 7 — Local Agents Foundations
**Focus:** Reliable function calling on small models. smolagents, LangGraph + Ollama. Tool use with retries and error handling. Code-executing agents safely.
**Project:** AgentDesk (local personal agent) + ToolKit (reliable function-calling library)

**After this month, you can:**
- [ ] Build agents on local 7B models that reliably call tools (a real challenge — small models hallucinate tools)
- [ ] Use grammar-constrained generation (outlines, lm-format-enforcer) for guaranteed schema validity
- [ ] Build retry/repair loops that recover from malformed tool calls
- [ ] Sandbox code execution safely (Docker, Firejail, restricted Python)
- [ ] Use smolagents or LangGraph with Ollama as the backend
- [ ] Design a permission/trust system for agents acting on your behalf

---

### Month 8 — Multi-Agent Systems & Reliability
**Focus:** CrewAI / AutoGen with local LLMs. Agent memory (mem0, Letta, custom). Observability and debugging. Browser/desktop automation. Patterns for actually-shippable agents.
**Project:** CrewLocal (multi-agent crew) + AgentTrace (agent observability)

**After this month, you can:**
- [ ] Orchestrate 3-5 specialized local agents on a single 6GB GPU via smart model swapping
- [ ] Add long-term memory to agents (episodic + semantic + procedural)
- [ ] Trace and visualize an agent's "thinking" — see why it succeeded or failed
- [ ] Classify common agent failure modes and design defenses
- [ ] Build agents that automate browser tasks (Playwright) and desktop apps (PyAutoGUI, pywinauto)
- [ ] Decide when to use single-agent vs multi-agent (most problems don't need multi-agent)

---

### Month 9 — Multimodal Local AI (Vision)
**Focus:** Vision-Language Models that run on 6GB: Moondream2 (1.86B), LLaVA 1.6 Mistral 7B Q4, Phi-3 Vision, MiniCPM-V. OCR pipelines (Tesseract, PaddleOCR, Marker). Document AI.
**Project:** VisionAssist (local multimodal) + ScreenSage (screenshot Q&A)

**After this month, you can:**
- [ ] Run multiple VLMs locally and pick the right one per task (speed vs quality vs domain)
- [ ] Build vision pipelines: photo description, chart interpretation, document Q&A, ALT text generation
- [ ] Integrate OCR (Tesseract for simple, PaddleOCR for harder, Marker for PDFs) with VLMs
- [ ] Build a system-tray screenshot assistant that responds in seconds
- [ ] Understand VLM architectures: vision encoder + projector + LLM, and why Moondream is so efficient

---

### Month 10 — Voice, Image & Generative Local AI
**Focus:** Whisper (faster-whisper, distil-whisper, whisper.cpp). TTS (Piper, Coqui, XTTS). End-to-end voice assistant. Stable Diffusion local (ComfyUI, A1111, Fooocus). SD LoRAs.
**Project:** VoiceMate (offline voice assistant) + ImageForge (local image gen studio)

**After this month, you can:**
- [ ] Run Whisper variants (faster-whisper INT8) on CPU at real-time speed
- [ ] Use Piper TTS for low-latency voice output, or XTTS for voice cloning
- [ ] Build a complete voice assistant: wake word → ASR → LLM → TTS, all offline, <5s end-to-end
- [ ] Run SDXL or SD 1.5 on 6GB with low-VRAM tricks (tiling, sequential CPU offload)
- [ ] Train SD LoRAs for custom styles/subjects in <1 hour on your GPU
- [ ] Build ComfyUI workflows for repeated production tasks (blog images, avatars, illustrations)

---

### Month 11 — Edge, NPU & Home Server Deployment
**Focus:** Architect a home AI server. Leverage Intel AI Boost NPU via OpenVINO/DirectML. Deploy to mobile (MLC-LLM, llama.cpp Android). Jetson / Raspberry Pi 5 / mini-PC edge nodes.
**Project:** HomeAI (household server) + EdgeDeploy (cross-platform benchmark)

**After this month, you can:**
- [ ] Architect a home AI server: Docker Compose stack, reverse proxy, mDNS, multi-user
- [ ] Use Intel AI Boost NPU via OpenVINO Runtime (great for embedding inference, small LLMs)
- [ ] Convert and deploy models to mobile via MLC-LLM (compiles to TVM-optimized kernels)
- [ ] Run local AI on Raspberry Pi 5 with llama.cpp (Pi 5 with 8GB can run 3B Q4 at 3-5 tok/s)
- [ ] Compare deployment targets quantitatively: tokens/sec, watts, $/hour amortized
- [ ] Build privacy-first family/household AI products that respect network boundaries

---

### Month 12 — Magnum Opus + Open Source Contribution
**Focus:** Combine all 11 months into one signature project. Polish documentation. Get external users. Make a real contribution to llama.cpp, Ollama, Unsloth, or another major local AI tool.
**Project:** MyAI (your signature project) + Open Source Contribution

**After this month, you can:**
- [ ] Ship a polished local-AI product end-to-end and have it used by real people
- [ ] Navigate the codebases of major local-AI projects (llama.cpp, Ollama, Unsloth)
- [ ] Submit a merged PR to a project used by thousands
- [ ] Write technical blog posts that other practitioners learn from
- [ ] Reason about the local-AI ecosystem trends and what's coming next (NPUs, MoE local, 1-bit, etc.)
- [ ] **You are a Local AI Expert. You can answer "should this be local?" for any new project with rigor.**

[⬆ Back to Table of Contents](#toc)

---

# ═══════════════════════════════════════════════
# PHASE 1: FOUNDATIONS (Weeks 1-8, Months 1-2)
# Ollama, llama.cpp, GGUF, VRAM math, prompt mastery
# ═══════════════════════════════════════════════

---

## Week 0 — Setup & First Local Model (Do This Immediately)

> This week is shorter (~3-4 hours total) but unblocks everything. Do it the day you start.

### Day 1 — Install Ollama (30 min)
- [ ] Download Ollama for Windows: https://ollama.com/download
- [ ] Verify install: `ollama --version`
- [ ] Optional but recommended: install WSL2 Ubuntu and ALSO install Ollama in WSL — Linux ecosystem is easier for ML tooling
- [ ] Set `OLLAMA_MODELS` env var if you want models stored on a different drive (they're big — 3-15GB each):
  ```powershell
  [System.Environment]::SetEnvironmentVariable("OLLAMA_MODELS", "D:\ollama-models", "User")
  ```
- [ ] **Code:** `ollama serve` (starts the server) then in another terminal: `ollama --help`

### Day 2 — Run Your First Local Model (30 min)
- [ ] `ollama run llama3.2:3b` — Llama 3.2 3B Instruct, perfect first model (1.9GB, fits easily)
- [ ] Ask it questions, observe quality vs ChatGPT
- [ ] Exit with `/bye`
- [ ] `ollama run qwen2.5:3b` — Qwen 2.5 3B (great quality for size)
- [ ] `ollama list` — see what you have
- [ ] `ollama ps` — see what's loaded in memory
- [ ] **Code:** Time a 100-token generation, note tokens/sec

### Day 3 — Quick VRAM Reality Check (1 hour)
- [ ] Run `nvidia-smi` while Ollama is generating — watch VRAM use climb
- [ ] Try `ollama run llama3.1:8b` (Llama 3.1 8B, Q4_K_M default, ~4.7GB)
- [ ] Try `ollama run llama3.1:8b-instruct-q8_0` (8B at Q8, ~8.5GB → won't fit GPU-only, will spill to system RAM, slow)
- [ ] Observe: Q4_K_M is fast (~25-40 tok/s on RTX 1000 Ada). Q8 forces partial CPU = ~3-8 tok/s
- [ ] **Code:** Compare both speeds, internalize the tradeoff

### Day 4 — Install llama.cpp & Test (1 hour)
- [ ] Download prebuilt llama.cpp binary for Windows/CUDA: https://github.com/ggerganov/llama.cpp/releases (look for `cudart-llama-bin-win-cu12.x-x64.zip`)
- [ ] OR build from source in WSL2 (recommended): `git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp && make LLAMA_CUDA=1 -j`
- [ ] Download a GGUF directly from HuggingFace: e.g., `Qwen/Qwen2.5-3B-Instruct-GGUF` → `qwen2.5-3b-instruct-q4_k_m.gguf`
- [ ] Run: `./llama-cli -m qwen2.5-3b-instruct-q4_k_m.gguf -p "Hello" -n 100 -ngl 99`
- [ ] `-ngl 99` = offload all layers to GPU; experiment with `-ngl 20` (partial offload)
- [ ] **Code:** Compare GPU-only vs partial-offload vs CPU-only speeds

### 🔨 Saturday Mini-Project (1-2 hours)
- [ ] **"Hello Local AI" notebook** — a Jupyter or markdown journal documenting:
  - 5 models you tried (name, size, quant, GPU/CPU split, tok/s, vibes)
  - First-token latency for each
  - Your subjective quality ranking on 3 test questions
  - Which model you'd use for: chat, code, summarizing, creative writing
- [ ] Save this in `./LOCAL-AI/journal/week-0-baseline.md` — you'll come back to it later

### 📄 Sunday Resources
- [ ] Watch: "Run Local LLMs" by NetworkChuck or Matt Williams (Ollama channel) — 30 min overview
- [ ] Read: Ollama README + "Getting Started" docs on ollama.com/docs

---

## Week 1 — Ollama Deep Dive

### Day 1 — Ollama Architecture
- [ ] Ollama = llama.cpp under the hood + Go server + Modelfile abstraction
- [ ] How models are stored: blob store, manifest, layers (similar to Docker images)
- [ ] Where models live on disk (Windows: `C:\Users\<user>\.ollama\models`; Linux: `~/.ollama/models`)
- [ ] `ollama show <model>` — see Modelfile, parameters, template
- [ ] `ollama show --modelfile llama3.2:3b`
- [ ] **Code:** Inspect 3 models' Modelfiles, note differences in templates

### Day 2 — Ollama API
- [ ] REST API runs on `localhost:11434` by default
- [ ] `/api/generate` — single completion
- [ ] `/api/chat` — chat-style with message history
- [ ] `/api/embeddings` — get embeddings (for RAG later)
- [ ] `/api/tags` — list models
- [ ] `/api/pull` — pull a model programmatically
- [ ] OpenAI-compatible endpoint at `/v1/chat/completions` — drop-in replacement!
- [ ] **Code:** Hit each endpoint with `curl` or `requests` in Python, parse responses

### Day 3 — Streaming Generation
- [ ] Why streaming: shows tokens as they're generated, much better UX
- [ ] Set `"stream": true` in the API; iterate response chunks
- [ ] Server-Sent Events (SSE) pattern
- [ ] OpenAI-compatible streaming: handle `data: {...}\n\n` chunks
- [ ] **Code:** Write a streaming Python client that prints tokens as they arrive

### Day 4 — Model Parameters
- [ ] `temperature` — randomness (0 = deterministic, 1+ = creative); typical 0.6-0.8
- [ ] `top_k` (40), `top_p` (0.9) — sampling restrictions
- [ ] `repeat_penalty` (1.1) — discourage repetition (small models loop more!)
- [ ] `num_ctx` — context window (default 2048 in Ollama!). For longer convos: 4096, 8192, 16384...
- [ ] `num_predict` — max tokens to generate (-1 = until done)
- [ ] `seed` — for reproducibility
- [ ] **WARNING:** Ollama defaults `num_ctx=2048` which surprises everyone. Set it explicitly!
- [ ] **Code:** Experiment with each parameter, observe output changes

### Day 5 — `ollama` CLI Mastery
- [ ] `ollama pull <model>` — download
- [ ] `ollama create <name> -f <Modelfile>` — build a custom model
- [ ] `ollama rm <model>` — delete (free disk space)
- [ ] `ollama push <model>` — push to Ollama Hub (requires account)
- [ ] `ollama cp <src> <dst>` — copy/rename
- [ ] `OLLAMA_NUM_PARALLEL` env var — parallel requests
- [ ] `OLLAMA_MAX_LOADED_MODELS` — keep multiple models in VRAM (careful, 6GB is tight)
- [ ] **Code:** Set `OLLAMA_KEEP_ALIVE=-1` so models stay loaded (faster repeated queries)

### 🔨 Saturday Project
- [ ] **Build a Streamlit chat UI** wrapping Ollama
  - [ ] Model selector dropdown (loads `/api/tags`)
  - [ ] Streaming generation
  - [ ] Conversation history in session state
  - [ ] Display tokens/sec live in sidebar
  - [ ] Export conversation as markdown

### 📄 Sunday
- [ ] Read: Ollama API docs end-to-end
- [ ] Watch: "Ollama Course" by Matt Williams (Ollama YouTube channel) — pick 2-3 videos

---

## Week 2 — llama.cpp & GGUF Quantization Formats

### Day 1 — Why llama.cpp Exists
- [ ] Born from Georgi Gerganov's curiosity: "can I run LLaMA on a Mac?"
- [ ] Pure C/C++, no Python deps, runs on CPU/CUDA/Metal/ROCm/Vulkan/OpenCL
- [ ] Powers Ollama, LM Studio, KoboldCPP, Jan, GPT4All, and dozens of frontends
- [ ] Designed for inference, not training (use Unsloth/Axolotl for that)
- [ ] **Read:** GGML/llama.cpp design docs in the repo

### Day 2 — GGUF Format Deep Dive
- [ ] GGUF = "GGML Unified Format" — successor to GGML, GGJT, GGMF
- [ ] Single file contains: weights, tokenizer, chat template, metadata
- [ ] Memory-mapped (mmap): can load partially, swap pages to disk
- [ ] Compatible across llama.cpp versions (the GGML formats weren't)
- [ ] **Code:** Use `gguf-dump.py` (from llama.cpp/gguf-py) to inspect a GGUF file's metadata

### Day 3 — K-Quant Family Explained
- [ ] **Q2_K** — 2.5 bits/param avg, very small, noticeable quality loss
- [ ] **Q3_K_S/M/L** — 3-3.5 bits, big quality drop
- [ ] **Q4_0** — legacy 4-bit, simple, slightly worse than K-quants
- [ ] **Q4_K_S/M** — 4.5-4.8 bits, the sweet spot for most users ⭐
- [ ] **Q5_K_S/M** — 5.5-6 bits, near-FP16 quality
- [ ] **Q6_K** — ~6.5 bits, very close to FP16
- [ ] **Q8_0** — 8 bits, essentially lossless for inference
- [ ] **IQ-quants** (IQ4_XS, IQ3_S, IQ2_XS) — "importance quants" using a calibration dataset, smaller for same quality
- [ ] **Code:** Read the llama.cpp quantize source (`ggml-quants.c`) — comments explain each variant

### Day 4 — Converting and Quantizing Models
- [ ] HuggingFace model → GGUF using `convert_hf_to_gguf.py` (in llama.cpp repo)
- [ ] `python convert_hf_to_gguf.py /path/to/hf-model --outfile model.gguf --outtype f16`
- [ ] Then quantize: `./llama-quantize model-f16.gguf model-q4_k_m.gguf Q4_K_M`
- [ ] iMatrix (importance matrix) for IQ-quants: generated from calibration data
- [ ] `./llama-imatrix -m model-f16.gguf -f wiki.train.raw -o imatrix.dat`
- [ ] **Code:** Convert + quantize a small model (e.g., Qwen2.5-0.5B) to 3 quant levels

### Day 5 — llama.cpp Runtime Flags You Must Know
- [ ] `-ngl N` — number of layers to offload to GPU (try 99 for "all", then reduce if OOM)
- [ ] `-c N` — context size (must be ≤ model's trained max; 4096, 8192, etc.)
- [ ] `-b N` — logical batch size
- [ ] `-t N` — CPU threads (set to physical core count, not logical)
- [ ] `-fa` — flash attention (CUDA, big speedup if supported)
- [ ] `--mlock` — lock model in RAM (Linux only; avoid swap)
- [ ] `--no-mmap` — disable memory mapping (sometimes faster on Windows, more RAM use)
- [ ] `-p "prompt"` and `-n N` (max tokens), `--temp`, `--top-p`, `--top-k`, `--repeat-penalty`
- [ ] `llama-server` — HTTP server with OpenAI-compatible API + web UI
- [ ] **Code:** Run `llama-server -m model.gguf -c 4096 -ngl 99 --port 8080`, hit `http://localhost:8080`

### 🔨 Saturday Project
- [ ] **Quantize a model 4 ways, benchmark on YOUR hardware**
  - [ ] Pick Qwen 2.5 3B (small enough to do all quants in a day)
  - [ ] Quantize: Q4_0, Q4_K_M, Q5_K_M, Q8_0
  - [ ] For each: file size, VRAM use (`nvidia-smi`), tokens/sec, perplexity (`llama-perplexity` on wikitext)
  - [ ] Produce a markdown table + observations
  - [ ] Notes: where does each break down? Q4_0 vs Q4_K_M is the most interesting comparison

### 📄 Sunday Reading
- [ ] "GGUF Quantization Comparison" articles on HuggingFace blog
- [ ] Skim llama.cpp's README.md and `examples/` directory

---

## Week 3 — Model Selection & VRAM Engineering

### Day 1 — The "Will It Fit" Algorithm
- [ ] Formula: `VRAM ≈ params × bytes_per_param + KV_cache + activations + 0.5GB overhead`
- [ ] `KV_cache_per_token ≈ 2 × num_layers × num_kv_heads × head_dim × bytes_per_kv (2 for FP16, 1 for INT8 KV-cache)`
- [ ] Example: Llama 3.1 8B, 32 layers, 8 KV heads, 128 head_dim, FP16 KV cache, 4096 ctx:
  - `2 × 32 × 8 × 128 × 2 × 4096 ≈ 0.5 GB` for KV-cache alone at 4k context
  - At 32k context: 4 GB just for KV-cache! Almost as much as the model weights
- [ ] **This is why long context is expensive on small GPUs**
- [ ] **Code:** Write a Python function that takes model config + ctx_len + dtype and returns VRAM estimate

### Day 2 — KV-Cache Optimization
- [ ] KV cache quantization: `-ctk q8_0 -ctv q8_0` in llama.cpp halves KV memory
- [ ] More aggressive: `-ctk q4_0 -ctv q4_0` (some quality loss for very long context)
- [ ] Flash Attention (`-fa`) reduces activation memory
- [ ] Trade-off: longer context vs higher batch vs larger model
- [ ] **Code:** Try 32k context on a 3B model with KV quantization; measure VRAM saved

### Day 3 — Curated Model Catalog for 6GB
- [ ] **Tier 1 — runs effortlessly (3-5GB):**
  - Llama 3.2 3B Instruct Q5_K_M / Q6_K
  - Qwen 2.5 3B Instruct Q5_K_M
  - Phi-3 Mini 3.8B Q5_K_M
  - Gemma 2 2B Q8_0
- [ ] **Tier 2 — sweet spot (4.5-5.5GB, leaves room for context):**
  - Llama 3.1 8B Instruct Q4_K_M ⭐
  - Qwen 2.5 7B Instruct Q4_K_M ⭐
  - Mistral 7B Instruct v0.3 Q4_K_M
  - Qwen 2.5 Coder 7B Q4_K_M (code specialist)
  - DeepSeek-R1-Distill-Qwen-7B Q4_K_M (reasoning)
- [ ] **Tier 3 — fits with care (5.5-6GB):**
  - Qwen 2.5 14B Q3_K_M (quality drops noticeably)
  - Phi-3 Medium 14B Q3_K_M
  - Yi 1.5 9B Q4_K_M
- [ ] **Tier 4 — partial GPU offload only (slow but works):**
  - Llama 3.3 70B Q3_K_S (CPU-heavy, ~1-2 tok/s — for batch tasks only)
- [ ] **Specialist:**
  - nomic-embed-text-v1.5 (embedding, 137M) — basically free VRAM
  - bge-reranker-v2-m3 (reranker, 568M)
  - Moondream2 1.86B (vision)
  - LLaVA 1.6 Mistral 7B Q4_K_M (vision)
- [ ] **Code:** Save this catalog as `models-catalog.md` in your `LOCAL-AI/` folder

### Day 4 — Choosing Between Models
- [ ] **General chat:** Llama 3.1 8B Q4 or Qwen 2.5 7B Q4 (both excellent)
- [ ] **Coding:** Qwen 2.5 Coder 7B Q4 ⭐ (Qwen Coder series is currently SOTA for local code)
- [ ] **Math/reasoning:** DeepSeek-R1-Distill-Qwen-7B Q4 (reasoning chains), Qwen 2.5 Math 7B
- [ ] **Long context:** Phi-3 Mini 128k or Llama 3.1 8B (128k native, but expensive on 6GB)
- [ ] **Fast/cheap:** Llama 3.2 1B or 3B for quick tasks
- [ ] **Multilingual:** Aya Expanse 8B, Qwen 2.5 (good Chinese), Llama 3.1 (decent)
- [ ] **Function calling:** Qwen 2.5 Instruct (best small-model tool use), Llama 3.1 8B Instruct
- [ ] **Indian languages (Hindi/Marathi):** Sarvam-1, Krutrim, Llama 3.1 (passable Hindi)
- [ ] **Code:** Make YOUR personal model decision tree (a flowchart `if domain=X then model=Y`)

### Day 5 — Benchmarking Methodology
- [ ] Three numbers to always measure: TTFT (time to first token), TPOT (time per output token), peak VRAM
- [ ] TTFT depends on context length (prefill is compute-bound)
- [ ] TPOT depends on memory bandwidth (decode is memory-bound)
- [ ] Use 5+ identical prompts of varying lengths, average results
- [ ] Set `seed` for reproducibility
- [ ] Always include: model + quant + ctx + offload level + GPU + driver version
- [ ] **Code:** Write a `bench.py` that runs a model through 5 prompts and prints a benchmark table

### 🔨 Saturday Project
- [ ] **Personal Model Catalog with Benchmarks**
  - [ ] Pull 8-10 models matching your needs
  - [ ] Run identical benchmark on each: TTFT, TPOT, peak VRAM, quality on 5 test prompts
  - [ ] Produce a markdown table you can refer back to forever
  - [ ] Include a "use this model when..." column

### 📄 Sunday
- [ ] Read: HuggingFace's "Open LLM Leaderboard" methodology
- [ ] Explore: lmsys.org (Chatbot Arena), and note which leaderboard models you can actually run

---

## Week 4 — Local Inference Frontends (LM Studio, Jan, Open WebUI)

### Day 1 — LM Studio
- [ ] Closed-source, free, polished GUI for local LLMs
- [ ] Built-in model browser (HF GGUF search)
- [ ] OpenAI-compatible API server (similar to Ollama)
- [ ] Strengths: easy for non-devs, great for quick exploration
- [ ] Weaknesses: closed source, can't customize as much as Ollama
- [ ] **Code:** Install LM Studio, download Qwen 2.5 7B Q4_K_M, chat, start the server, hit it from Python

### Day 2 — Jan
- [ ] Open-source desktop app (Electron + llama.cpp backend)
- [ ] Plugin/extension system
- [ ] Multiple model engines: llama.cpp, TensorRT-LLM (limited), remote APIs
- [ ] Strengths: open source, plugins, configurable
- [ ] Weaknesses: less polish than LM Studio
- [ ] **Code:** Install Jan, try a model, explore the extension system

### Day 3 — Open WebUI ⭐ (probably your daily driver)
- [ ] Web app (FastAPI + Svelte) designed to wrap Ollama
- [ ] Multi-user, conversation history, RAG, prompts, models all in one
- [ ] Self-hosted, runs in Docker
- [ ] Best for "I want a polished ChatGPT-like UI for my Ollama"
- [ ] **Code:** Run via Docker: `docker run -d -p 3000:8080 --add-host=host.docker.internal:host-gateway -v open-webui:/app/backend/data --name open-webui ghcr.io/open-webui/open-webui:main`
- [ ] Configure OLLAMA_BASE_URL, log in, chat

### Day 4 — Alternative Frontends
- [ ] **KoboldCPP** — fork of llama.cpp with creative writing focus (storytelling, RP)
- [ ] **Text Generation Web UI (oobabooga)** — Swiss army knife, many backends, complex
- [ ] **AnythingLLM** — RAG-first chat platform, beautiful UI
- [ ] **LibreChat** — multi-provider chat, very polished, can do local + cloud
- [ ] **GPT4All** — desktop app, includes its own model curation
- [ ] **Code:** Try 2 of these, decide what you'll keep

### Day 5 — Comparing & Choosing
- [ ] **Quick chat:** Ollama CLI + a simple Streamlit script (OllaMate Project A)
- [ ] **Polished daily driver:** Open WebUI ⭐
- [ ] **Drag-and-drop with built-in RAG:** AnythingLLM
- [ ] **Multi-provider (local + cloud):** LibreChat
- [ ] **Creative writing/RP:** KoboldCPP
- [ ] **Quick model exploration:** LM Studio
- [ ] **Embeddable in your own app:** llama-cpp-python or Ollama API
- [ ] **Code:** Document your "personal stack" — which tool for which job

### 🔨 Saturday Project
- [ ] **OllaMate v1** — start your Month 1 project
  - [ ] Streamlit web UI
  - [ ] Multi-model dropdown (auto-fetch from Ollama)
  - [ ] Conversation history (SQLite or session state)
  - [ ] System prompt library (5-10 personas)
  - [ ] Tokens/sec display
  - [ ] Export conversation as markdown
  - [ ] Push to GitHub with screenshots

### 📄 Sunday
- [ ] Watch: "Self-Host ChatGPT with Open WebUI" by Network Chuck or similar
- [ ] Decide: which frontend will you keep as your daily driver?

---

## 🔄 Buffer Week (Month 1 Revision) Local AI
- [ ] Revise: Ollama CLI + API, llama.cpp flags, GGUF formats, K-quant family
- [ ] Revise: VRAM math, KV-cache, model catalog for 6GB
- [ ] Re-run a few benchmarks — they should feel routine now
- [ ] **Build Monthly Project A:** OllaMate — Local Chat Stack (polish to publishable quality)
- [ ] **Build Monthly Project B:** VRAMCalc — Compatibility Predictor
- [ ] Push both projects to GitHub with clear READMEs + screenshots
- [ ] Write a journal entry: "What I know now that I didn't 4 weeks ago"

---

## Week 5 — Modelfiles, System Prompts & API Integration

### Day 1 — Modelfile Anatomy
- [ ] `FROM` — base model (a tag or another Modelfile)
- [ ] `PARAMETER` — set generation params (temperature, num_ctx, num_predict, etc.)
- [ ] `SYSTEM` — system prompt baked into the model
- [ ] `TEMPLATE` — chat template (Go template syntax, with `.System`, `.Prompt`, `.Response`)
- [ ] `MESSAGE` — embed few-shot examples
- [ ] `ADAPTER` — apply a LoRA adapter
- [ ] `LICENSE` — embed license text
- [ ] **Code:** Inspect `ollama show --modelfile llama3.1:8b` and dissect every line

### Day 2 — Custom Modelfiles for Personas
- [ ] Create `code-tutor.Modelfile`:
  ```
  FROM qwen2.5-coder:7b
  PARAMETER temperature 0.3
  PARAMETER num_ctx 8192
  SYSTEM """You are a patient code tutor. Always explain step-by-step.
  Show code with comments. Suggest improvements after working solution.
  If unsure, say so explicitly rather than guess."""
  ```
- [ ] `ollama create code-tutor -f code-tutor.Modelfile`
- [ ] `ollama run code-tutor`
- [ ] **Code:** Build 5 personas you'd actually use (writer, debugger, summarizer, brainstormer, fact-checker)

### Day 3 — Chat Templates Deep Dive
- [ ] Why templates matter: each model family expects a specific format (ChatML, Llama-3, Mistral, etc.)
- [ ] ChatML (Qwen, OpenAI): `<|im_start|>role\ncontent<|im_end|>`
- [ ] Llama 3: `<|begin_of_text|><|start_header_id|>system<|end_header_id|>...`
- [ ] Mistral: `[INST] {prompt} [/INST]`
- [ ] Gemma: `<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n`
- [ ] **WARNING:** Using wrong template = model performs much worse. Verify your Modelfile's TEMPLATE matches the model
- [ ] **Code:** Compare same prompt with correct vs wrong template — observe quality drop

### Day 4 — Ollama Python Library
- [ ] `pip install ollama`
- [ ] `import ollama; ollama.chat(model='llama3.1:8b', messages=[...])`
- [ ] Streaming: `for chunk in ollama.chat(..., stream=True): print(chunk['message']['content'])`
- [ ] Function calling (Ollama supports tools format similar to OpenAI):
  ```python
  ollama.chat(model='llama3.1:8b', messages=[...], tools=[{...}])
  ```
- [ ] **Code:** Build a Python module wrapping common patterns (chat, stream, embed, structured output)

### Day 5 — OpenAI-Compatible Drop-In Replacement
- [ ] Ollama serves OpenAI-compatible API at `http://localhost:11434/v1/`
- [ ] Use `openai` Python SDK:
  ```python
  from openai import OpenAI
  client = OpenAI(base_url="http://localhost:11434/v1/", api_key="ollama")
  client.chat.completions.create(model="llama3.1:8b", messages=[...])
  ```
- [ ] Same for: LangChain (`ChatOpenAI` with custom base_url), LlamaIndex, instructor
- [ ] This means **any tool that supports OpenAI works with your local Ollama**
- [ ] **Code:** Take an existing OpenAI-based script you have, redirect to Ollama, observe what works/breaks

### 🔨 Saturday Project
- [ ] **"Persona Library"** — a versioned collection of 10 custom Modelfiles
  - [ ] code-tutor, writer-assistant, summarizer, brainstormer, translator, fact-checker, debate-partner, etc.
  - [ ] Each has documented use case, base model, parameters
  - [ ] All stored in `~/.ollama/personas/` and a Git repo
  - [ ] Build a `persona-switcher.py` CLI that picks the right one based on intent

### 📄 Sunday
- [ ] Read: Ollama Modelfile reference docs in full
- [ ] Read: 2-3 well-crafted system prompts from awesome-chatgpt-prompts and adapt for local models

---

## Week 6 — Quantization Beyond GGUF (GPTQ, AWQ, EXL2)

### Day 1 — GPTQ
- [ ] Post-training quantization using Hessian information of weights
- [ ] Weights only (activations stay FP16)
- [ ] 4-bit or 8-bit, per-channel or per-group
- [ ] Library: AutoGPTQ (deprecated), GPTQModel (active fork)
- [ ] Best for: pure inference on GPU (transformers, vLLM, ExLlamaV2)
- [ ] **Code:** Try loading a GPTQ model with `transformers + auto-gptq` (or GPTQModel)
- [ ] Note: needs a GPU with enough VRAM; doesn't benefit from CPU offload like GGUF does

### Day 2 — AWQ (Activation-aware Weight Quantization)
- [ ] Identifies "salient" channels using activation statistics; preserves them at higher precision
- [ ] 4-bit, achieves better quality than GPTQ at the same bit-width often
- [ ] Library: AutoAWQ
- [ ] Best for: vLLM inference (vLLM has excellent AWQ kernel)
- [ ] **Code:** Try `pip install autoawq`, quantize a small model

### Day 3 — EXL2 (ExLlamaV2)
- [ ] Variable bit-width: different layers at different precisions (per-tensor)
- [ ] Calibration-based, very high quality per bit
- [ ] Library: exllamav2 (turboderp)
- [ ] Best for: maximum quality at given VRAM target, NVIDIA only
- [ ] Used in TabbyAPI, oobabooga
- [ ] **Code:** Quantize a small model to EXL2, run with `exllamav2` directly

### Day 4 — Comparing Quantization Formats
| Format | Best For | Pros | Cons |
|--------|----------|------|------|
| GGUF | CPU + partial GPU, Ollama, llama.cpp | Universal, easy, mmap-friendly | Slightly slower on pure GPU than AWQ/EXL2 |
| GPTQ | Pure GPU inference, vLLM/transformers | Fast on GPU, mature ecosystem | Pure-GPU only, weight-only |
| AWQ | vLLM production serving | Best 4-bit quality, vLLM fast | Pure-GPU only |
| EXL2 | Maximum quality / VRAM target | Best quality, flexible bit-width | NVIDIA only, smaller ecosystem |
| FP8 | H100/H200/4000-series advanced | Hardware-accelerated | Limited support, needs Ada+/Hopper |
- [ ] **For YOUR 6GB Ada GPU: GGUF Q4_K_M is the daily driver. EXL2 4bpw can be slightly faster for pure GPU inference if you want to squeeze every tok/sec.**

### Day 5 — Bitsandbytes (Different Beast)
- [ ] `bitsandbytes` library: runtime quantization, used during fine-tuning (QLoRA)
- [ ] NF4 (NormalFloat 4-bit) — paired with QLoRA
- [ ] LLM.int8() — 8-bit inference for transformers (deprecated, use GPTQ/AWQ for prod)
- [ ] Not great for inference (slower than GGUF/AWQ), but essential for training
- [ ] **Code:** Install bitsandbytes (works on Windows with `bitsandbytes-windows-webui` fork, or use WSL2)

### 🔨 Saturday Project
- [ ] **Quantization Format Showdown**
  - [ ] Same model (e.g., Llama 3.1 8B), quantize to: GGUF Q4_K_M, GPTQ 4-bit, AWQ 4-bit, EXL2 4bpw
  - [ ] Benchmark on YOUR RTX 1000 Ada: tokens/sec, peak VRAM, perplexity, MMLU mini
  - [ ] Produce a one-page report
  - [ ] Decide: which format will you use for which scenario?

### 📄 Sunday
- [ ] Read: "GPTQ" paper (Frantar et al., 2022) — overlaps with main roadmap Month 4
- [ ] Read: "AWQ" paper (Lin et al., 2023)
- [ ] Read: ExLlamaV2 README + blog posts by turboderp

---

## Week 7 — Prompt Engineering for Small Local Models

### Day 1 — Why Small Models Need Better Prompts
- [ ] GPT-4 forgives sloppy prompts; 7B Q4 will fail
- [ ] Common failures of small models: verbosity, repetition, off-topic, format drift, hallucination, refusal of safe queries
- [ ] Mitigations: structured prompts, few-shot examples, explicit format specs, low temperature
- [ ] **Code:** Run the same vague prompt on GPT-4 (if you have access) vs Llama 3.1 8B Q4. Observe the gap

### Day 2 — Structured Prompts
- [ ] Role + context + task + format + constraints + examples
- [ ] Example:
  ```
  You are a senior Python developer.
  Context: I have a file `data.csv` with columns id, name, age.
  Task: Write a function to load it, filter age > 18, return as DataFrame.
  Format: Single function, type hints, docstring.
  Constraints: Use pandas only. No external libs. Handle missing values.
  Example output:
    def load_filtered(path: str) -> pd.DataFrame: ...
  ```
- [ ] **Code:** Create 10 templates following this structure for your common tasks

### Day 3 — Chain-of-Thought (CoT) for Small Models
- [ ] Adding "Think step by step before answering" boosts small models on math/logic
- [ ] Even stronger: "Let's think step by step. First, identify the key facts. Second, ..."
- [ ] Few-shot CoT: include 2-3 worked examples with reasoning
- [ ] **Code:** Test on GSM8K-style problems: bare prompt vs CoT prompt. Measure accuracy delta

### Day 4 — Structured Output (Critical for Small Models)
- [ ] JSON mode in Ollama: set `format: "json"` parameter
- [ ] Grammar-constrained generation: GBNF format (llama.cpp), outlines, lm-format-enforcer
- [ ] Schema-based: provide a Pydantic schema, get guaranteed valid JSON
- [ ] **Code:** Use `outlines`:
  ```python
  import outlines
  model = outlines.models.transformers("Qwen/Qwen2.5-3B")
  schema = '{"name": str, "age": int, "skills": [str]}'
  generator = outlines.generate.json(model, schema)
  result = generator("Extract: 'Alice is 30, knows Python and Go'")
  ```

### Day 5 — Prompt Patterns Worth Stealing
- [ ] **Self-critique loop:** "Now critique your answer. List 3 weaknesses. Provide an improved version."
- [ ] **Confidence calibration:** "Rate confidence 1-10. If <7, explain why."
- [ ] **Refusal handling:** "If you can't answer, say 'I don't know because [reason]'. Don't guess."
- [ ] **Format anchoring:** "Begin every response with '##'." (helps small models stay on format)
- [ ] **Length control:** "Answer in EXACTLY 3 sentences." (small models often ignore but adding it helps)
- [ ] **Persona persistence:** Repeat persona reminder every 5-10 turns in long convos
- [ ] **Code:** Build a "prompt patterns library" markdown file with examples

### 🔨 Saturday Project
- [ ] **PromptForge v1** — start your Month 2 project
  - [ ] Side-by-side prompt testing across 3-4 models
  - [ ] Save winning prompts with metadata
  - [ ] "Rewrite for small model" button using a meta-prompt
  - [ ] Export to Modelfile

### 📄 Sunday
- [ ] Read: "The Prompt Report" (Schulhoff et al., 2024) — exhaustive prompt technique survey
- [ ] Watch: "Prompt Engineering Mastery" videos by anthropic/openai/youtube tutorials

---

## Week 8 — Local Benchmarking & Evaluation

### Day 1 — What Numbers Matter
- [ ] **Speed:** TTFT, TPOT, total throughput (input + output tokens / sec)
- [ ] **Memory:** peak VRAM, peak RAM, model file size
- [ ] **Quality:** perplexity on hold-out, task-specific accuracy (GSM8K, HumanEval, MMLU), human ratings
- [ ] **Reliability:** % valid JSON, % follows format, hallucination rate
- [ ] **Cost (local):** electricity (~watts × hours), opportunity cost of your laptop
- [ ] **Code:** Build a metrics-tracking dataclass

### Day 2 — Perplexity Locally
- [ ] llama.cpp has `llama-perplexity`: `./llama-perplexity -m model.gguf -f wiki.test.raw`
- [ ] Wikitext-103 test set is standard
- [ ] Lower = better, but small differences (<0.1) are usually noise
- [ ] **Code:** Run perplexity on your favorite models, compare to published numbers

### Day 3 — Task-Specific Mini-Benchmarks
- [ ] Don't run full MMLU (14k questions, hours). Use subsets:
  - MMLU-Mini: 5-15 questions per subject
  - GSM8K-50: 50 math problems
  - HumanEval-30: 30 Python coding problems
- [ ] Use `lm-evaluation-harness` (overlap with main roadmap):
  ```bash
  lm_eval --model openai-completions --model_args base_url=http://localhost:11434/v1/,model=llama3.1:8b --tasks gsm8k,humaneval --limit 50
  ```
- [ ] **Code:** Run a 3-task eval on 3 of your models, save results to a CSV

### Day 4 — LLM-as-Judge for Subjective Tasks
- [ ] For chat quality, writing, summarization: use a larger model (or GPT-4 via API) as judge
- [ ] Pairwise comparison: "given prompt and answers A and B, which is better and why?"
- [ ] Watch for biases: judges prefer longer answers, prefer their own style, etc.
- [ ] **Code:** Build a `judge.py` that scores 20 answers per model on a 1-5 scale

### Day 5 — Building Your Personal Eval Suite
- [ ] 20-50 questions covering YOUR use cases (not generic benchmarks)
- [ ] Include: easy chat, hard reasoning, code task, factual recall, creative writing, summarization, refusal test
- [ ] Save gold-standard answers (your own, polished)
- [ ] Every time you try a new model or fine-tune, run this suite
- [ ] **Code:** Save as `personal-eval.jsonl` in your repo; a CI step can rerun it

### 🔨 Saturday Project
- [ ] **ModelStudio v1** — start Month 2 main project
  - [ ] Wrap llama.cpp quantize binary
  - [ ] Auto-quantize a HF model to multiple GGUF formats
  - [ ] Run perplexity + your personal eval on each
  - [ ] HTML report with charts

### 📄 Sunday
- [ ] Read: "Holistic Evaluation of Language Models" (HELM) — pick the section on LLM evaluation
- [ ] Read: "Chatbot Arena" methodology paper
- [ ] Reflect: write Phase 1 completion notes — what shifted in your mental model?

---

### ✅ Phase 1 Completion Checklist (End of Month 2)
- [ ] Can run any GGUF model and predict VRAM use before launching
- [ ] Mastered Ollama (CLI + API + Modelfiles + parameters)
- [ ] Mastered llama.cpp (build, quantize, server, all major flags)
- [ ] Know GGUF, GPTQ, AWQ, EXL2 — pros, cons, when to use
- [ ] Have a personal model catalog with measured benchmarks on YOUR hardware
- [ ] Can write effective prompts for 3-7B models
- [ ] Have a personal eval suite you run on every new model
- [ ] Built and use OllaMate + VRAMCalc daily
- [ ] Built ModelStudio + PromptForge for the Month 2 capstone

[⬆ Back to Table of Contents](#toc)

---

# ═══════════════════════════════════════════════════
# PHASE 2: LOCAL APPS & RAG (Weeks 9-16, Months 3-4)
# Knowledge bases, coding assistants, real workflows
# ═══════════════════════════════════════════════════

---

## 🔄 Buffer Week (Month 2 Revision) Local AI
- [ ] Revise: Modelfiles, chat templates, all quantization formats
- [ ] Revise: prompt engineering patterns for small models
- [ ] Revise: benchmarking methodology, personal eval suite
- [ ] **Build Monthly Project A:** ModelStudio — GGUF Quantization Lab
- [ ] **Build Monthly Project B:** PromptForge — Prompt Engineering Workbench
- [ ] Push both to GitHub with clear READMEs + screenshots
- [ ] Journal: "I am now comfortable with the local AI inference stack. What's next?"

---

## Week 9 — Local Embeddings & Vector Stores

### Day 1 — Why Local Embeddings Matter
- [ ] OpenAI embeddings cost $ per million tokens — adds up for large corpora
- [ ] Cloud embeddings = your text is sent away. Privacy violation for personal data
- [ ] Local embeddings: free, fast on GPU, totally private
- [ ] Modern open embeddings (nomic, BGE, mxbai) match or beat OpenAI's older models on benchmarks
- [ ] **Code:** Compare local nomic-embed vs OpenAI ada-002 on a sample retrieval task

### Day 2 — Top Local Embedding Models (Choose Wisely)
- [ ] **nomic-embed-text-v1.5** — 137M params, ~280MB VRAM, fast, high quality, 8k context ⭐
- [ ] **bge-large-en-v1.5** — 335M, slower but slightly better, 512 token context
- [ ] **bge-m3** — multilingual (100+ languages), dense + sparse + multi-vector, 8k context (slower)
- [ ] **mxbai-embed-large-v1** — 335M, top of MTEB leaderboard for general English
- [ ] **all-MiniLM-L6-v2** — 22M, tiny, fast, lower quality (good for prototyping)
- [ ] **snowflake-arctic-embed-l** — newer, competitive
- [ ] For YOUR 6GB GPU: **nomic-embed-text-v1.5** is the daily driver; BGE-M3 if you need multilingual
- [ ] **Code:** Try 3 embedding models, observe speed and quality differences on YOUR sample data

### Day 3 — Running Embeddings (Multiple Paths)
- [ ] **Via Ollama:** `ollama pull nomic-embed-text` then `/api/embeddings` endpoint
- [ ] **Via sentence-transformers:** `from sentence_transformers import SentenceTransformer; m = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)`
- [ ] **Via fastembed:** `pip install fastembed` — uses ONNX, very fast on CPU, no PyTorch dep
- [ ] **Via Transformers + bitsandbytes:** for advanced control / quantized embedding (rare)
- [ ] **Speed comparison:** fastembed CPU ≈ sentence-transformers GPU for small models, much less setup
- [ ] **Code:** Benchmark all 3 ways for embedding 1000 documents

### Day 4 — Vector Stores That Don't Need a Server
- [ ] **LanceDB** ⭐ — embedded, file-based, columnar (Apache Arrow), Python+JS+Rust, fast
  ```python
  import lancedb
  db = lancedb.connect("./mydb")
  tbl = db.create_table("docs", data=[{"vector": [...], "text": "...", "source": "..."}])
  tbl.search(query_vec).limit(10).to_list()
  ```
- [ ] **ChromaDB (embedded)** — popular, simple Python API, less performant at scale
- [ ] **Qdrant (local mode)** — open source server, but has local file mode; very feature-rich
- [ ] **Weaviate (embedded)** — heavier
- [ ] **Milvus Lite** — embedded version of Milvus
- [ ] **FAISS** — Facebook's pure-vector library, fast but no persistence/metadata built-in
- [ ] **For YOUR daily work: LanceDB or Qdrant local. ChromaDB for super quick prototypes.**

### Day 5 — Vector Store Operations Mastery
- [ ] Insert: batch insert (1000s at a time) for speed
- [ ] Search: top-K with cosine/L2/inner-product
- [ ] Hybrid search: combine vector + keyword (BM25)
- [ ] Filtering: metadata pre-filters (`where source = 'pdf' and date > 2024`)
- [ ] Indexing: IVF, HNSW, PQ — when each matters (only at >100K vectors usually)
- [ ] Update/delete: incremental, no full rebuild
- [ ] **Code:** Build a `vector-store-bench.py` comparing insert and search speeds on 10K docs across 3 stores

### 🔨 Saturday Project
- [ ] **Embedding & Vector Store Lab**
  - [ ] Embed all 5000 markdown notes from a vault (use your own Obsidian/notes if you have them)
  - [ ] Index in 2 stores: LanceDB and ChromaDB
  - [ ] Build a CLI: `notes-search "query"` returns top 10 with snippets
  - [ ] Compare retrieval quality (manual: are the top 5 results good?)

### 📄 Sunday
- [ ] Read: "Nomic Embed: Training a Reproducible Long Context Text Embedder" (Nussbaum, 2024)
- [ ] Read: BGE technical report (BAAI)
- [ ] Explore: MTEB leaderboard on HuggingFace (compare local embeddings vs cloud)

---

## Week 10 — Local RAG End-to-End

### Day 1 — RAG Architecture Recap (Local-First Lens)
- [ ] **Ingest:** documents → chunks → embeddings → vector store
- [ ] **Retrieve:** query → embed → top-K vectors → optional rerank
- [ ] **Generate:** stuff context into prompt → LLM → answer + citations
- [ ] **The local twist:** every step happens on your machine. No API key. No data leaving.
- [ ] **Code:** Sketch your pipeline on paper, identify failure points

### Day 2 — Chunking Strategies
- [ ] **Fixed-size:** simple, e.g., 500 tokens with 50 overlap
- [ ] **Recursive character splitter:** splits on `\n\n`, then `\n`, then ` `, then chars
- [ ] **Semantic chunking:** uses embeddings to group sentences with similar meaning
- [ ] **Document-structure-aware:** PDF sections, Markdown headers, code functions
- [ ] **Sliding window:** overlapping chunks reduce "cut in the middle" issues
- [ ] **For YOUR use:** RecursiveCharacterTextSplitter with 500-token chunks + 50 overlap is a great default
- [ ] **Code:** Try 3 chunking strategies on the same doc, observe how it affects retrieval

### Day 3 — Building Your First Local RAG (LangChain or LlamaIndex)
- [ ] LangChain example:
  ```python
  from langchain_community.embeddings import OllamaEmbeddings
  from langchain_community.vectorstores import LanceDB
  from langchain_community.chat_models import ChatOllama
  from langchain.text_splitter import RecursiveCharacterTextSplitter
  
  embeddings = OllamaEmbeddings(model="nomic-embed-text")
  llm = ChatOllama(model="llama3.1:8b")
  ```
- [ ] LlamaIndex tends to be more RAG-focused; LangChain more general
- [ ] **Or skip the framework:** ~100 lines of pure Python is often clearer
- [ ] **Code:** Build the same RAG with LangChain AND raw Python; compare

### Day 4 — Prompt Engineering for RAG
- [ ] Critical: tell the model to ONLY use the provided context
- [ ] Force citations: "After each claim, add [source: page_5]"
- [ ] Refusal handling: "If the context doesn't have the answer, say 'I don't know based on the provided documents.'"
- [ ] Example RAG prompt:
  ```
  You are a helpful assistant. Answer the question using ONLY the provided context.
  Cite sources as [source: filename, page X] inline.
  If the context lacks the answer, say so clearly. Do not use external knowledge.
  
  Context:
  {retrieved_chunks}
  
  Question: {user_question}
  Answer:
  ```
- [ ] **Code:** Test with on-topic and off-topic questions; observe whether your model refuses correctly

### Day 5 — Retrieval Quality Diagnostics
- [ ] Common failure: the right chunk is in the corpus but didn't get retrieved
- [ ] Debug tools: log top-10 chunks, their scores, their text snippets
- [ ] Try: lower chunk size, larger overlap, different embedding model, hybrid search, rerank
- [ ] **Code:** Build a "retrieval debugger" that shows top-K chunks for any query

### 🔨 Saturday Project
- [ ] **LocalRAG v1** — start your Month 3 main project
  - [ ] Point at a folder of PDFs/markdown
  - [ ] Ingest into LanceDB with nomic-embed
  - [ ] FastAPI backend + simple HTML frontend
  - [ ] Chat with citations
  - [ ] Verify zero network calls (Wireshark/tcpdump)

### 📄 Sunday
- [ ] Read: "RAG" original paper (Lewis et al., 2020) — overlaps main roadmap
- [ ] Read: "Lost in the Middle" (Liu et al., 2023) — why context placement matters
- [ ] Watch: "Building Production RAG" by Jerry Liu / LlamaIndex

---

## Week 11 — Advanced Local RAG (Hybrid, Rerank, Graph)

### Day 1 — Hybrid Retrieval (Dense + Sparse)
- [ ] Dense (embeddings): semantic similarity, but can miss exact terms (acronyms, names, IDs)
- [ ] Sparse (BM25): keyword matching, great for exact terms
- [ ] Hybrid: take top-K from both, merge with reciprocal rank fusion (RRF) or weighted score
- [ ] Libraries: `rank_bm25` (Python, simple), Tantivy (Rust, fast), Elasticsearch (overkill for local)
- [ ] **Code:** Implement RRF on top of LanceDB results + BM25 results; compare to dense-only

### Day 2 — Local Reranking
- [ ] Cross-encoder reranker: re-scores top-50 retrievals using a more expensive model
- [ ] Local rerankers:
  - **bge-reranker-v2-m3** (568M, multilingual, fits in 6GB with overhead)
  - **mxbai-rerank-large-v1** (435M, English)
  - **jina-reranker-v2** (smaller, fast)
- [ ] Trade-off: ~200-500ms extra latency, but big quality boost on hard queries
- [ ] **Code:** Add reranker stage; measure quality lift and latency cost

### Day 3 — Query Rewriting
- [ ] User queries are often ambiguous, abbreviated, or contain typos
- [ ] Rewrite with the LLM before retrieval: expand acronyms, add context, generate sub-queries
- [ ] **HyDE (Hypothetical Document Embeddings):** generate an imaginary answer, embed THAT, retrieve based on the imagined answer's embedding
- [ ] **Multi-query retrieval:** generate 3-5 paraphrased queries, union the results
- [ ] **Code:** Implement HyDE; compare on hard queries vs naive retrieval

### Day 4 — Graph RAG (Lightweight Local Version)
- [ ] Extract entities + relationships from documents using LLM
- [ ] Build a graph (NetworkX or Kùzu for local graph DB)
- [ ] Retrieval: traverse the graph from query entities
- [ ] Microsoft's GraphRAG is the canonical version; lighter local versions exist
- [ ] When to use: highly relational data (research, knowledge networks, biographies)
- [ ] **Code:** Try GraphRAG-lite on a small Wikipedia subset; compare to vector RAG

### Day 5 — Self-RAG & Agentic RAG
- [ ] Model decides WHETHER to retrieve (sometimes the question is general-knowledge)
- [ ] Model decides WHAT to retrieve (rewrites query)
- [ ] Model evaluates retrieval quality (self-critique)
- [ ] Local implementation: function-calling pattern with retrieve() as a tool
- [ ] **Code:** Wrap RAG as a tool the LLM can choose to call; test on mixed query set

### 🔨 Saturday Project
- [ ] **LocalRAG v2 — Advanced Retrieval**
  - [ ] Add hybrid dense + BM25 retrieval
  - [ ] Add bge-reranker-v2-m3 stage
  - [ ] Add query rewriting (HyDE or multi-query)
  - [ ] Comparison report: naive vs hybrid vs hybrid+rerank on 20 test queries

### 📄 Sunday
- [ ] Read: "Self-RAG" (Asai et al., 2023)
- [ ] Read: "From Local to Global: A Graph RAG Approach" (Microsoft, 2024)
- [ ] Read: Anthropic's "Contextual Retrieval" blog post

---

## Week 12 — Document Processing Pipelines

### Day 1 — PDFs (The Hard Case)
- [ ] **PyMuPDF (fitz):** fast, good for text-based PDFs, handles tables OK
- [ ] **pdfplumber:** great for tables, slower
- [ ] **Unstructured:** ML-based, handles layouts well, slower, more deps
- [ ] **Marker (VikParuchuri):** state-of-the-art for PDF → Markdown, local models, fast
- [ ] **Docling (IBM):** open source, growing fast, handles tables/equations/figures
- [ ] **For YOUR daily use:** Marker or Docling for high-quality conversion, PyMuPDF for speed
- [ ] **Code:** Convert 3 sample PDFs (text, scientific paper with equations, scanned doc) with each tool; compare

### Day 2 — OCR for Scanned Documents
- [ ] **Tesseract:** classic, free, decent quality, fast
- [ ] **PaddleOCR:** Baidu's, multilingual (100+ langs incl. Hindi/Marathi), excellent accuracy
- [ ] **EasyOCR:** Python-first, easy install, slower
- [ ] **Surya OCR:** newer, specifically for documents, supports many languages
- [ ] **Code:** OCR a scanned PDF page with each, compare extracted text quality

### Day 3 — Markdown, DOCX, Code Files
- [ ] **Markdown:** trivial, just read the file. Mind YAML frontmatter
- [ ] **DOCX:** `python-docx` or convert with Pandoc to Markdown first
- [ ] **HTML:** BeautifulSoup + readability-lxml to strip nav/ads
- [ ] **Code files:** chunk by function/class using tree-sitter or AST
  - Python: `ast.parse` → walk nodes
  - JS/TS/Go/Rust: tree-sitter parsers
- [ ] **Code:** Build a chunker that's file-type aware (different strategy per extension)

### Day 4 — Tables, Charts, Equations
- [ ] Tables: extract as structured data (CSV/JSON), keep alongside the text chunks
- [ ] Charts: caption them with a VLM (LLaVA), store the description + image path
- [ ] Equations: LaTeX form is RAG-friendly; convert images of equations using Nougat or similar
- [ ] **Code:** Extract tables from a financial report PDF; show them as JSON

### Day 5 — Putting It All Together: An Ingest Pipeline
- [ ] Stage 1: File type detection (`python-magic` or extension)
- [ ] Stage 2: Conversion to text (per type)
- [ ] Stage 3: Chunking (per type)
- [ ] Stage 4: Embedding (batched)
- [ ] Stage 5: Vector store insert (batched)
- [ ] Watch folder: `watchdog` library for incremental updates
- [ ] **Code:** Build a unified `ingest.py` that accepts any folder and processes everything

### 🔨 Saturday Project
- [ ] **LocalRAG v3 — Full Document Support**
  - [ ] Watch folder for new files
  - [ ] Multi-format ingestion (PDF, MD, DOCX, code)
  - [ ] Incremental indexing (don't re-ingest unchanged files)
  - [ ] Table-aware retrieval (return tables as structured chunks)
  - [ ] Demo on a real-world doc set (e.g., your textbook PDFs)

### 📄 Sunday
- [ ] Read: Marker / Docling READMEs and design docs
- [ ] Explore: how do RAG-as-a-Service tools (LlamaCloud, etc.) handle docs? What do they do that you don't?

---

## 🔄 Buffer Week (Month 3 Revision) Local AI
- [ ] Revise: embeddings (nomic, BGE), vector stores (LanceDB, Chroma)
- [ ] Revise: RAG basics, chunking, hybrid retrieval, reranking
- [ ] Revise: PDF/OCR pipelines
- [ ] **Build Monthly Project A:** LocalRAG — Privacy-First Document Q&A (polished, with verification)
- [ ] **Build Monthly Project B:** DocVault — Personal Knowledge Vault
- [ ] Push both projects to GitHub
- [ ] Run privacy audit: no network calls, document it in the README

---

## Week 13 — Local Coding Assistants (Continue, Cody, Aider)

### Day 1 — The Local Code-Assistant Landscape
- [ ] **Continue.dev** ⭐ — open source VS Code/JetBrains extension, Ollama-native
- [ ] **Cody (Sourcegraph)** — has local model support, more enterprise-y
- [ ] **Aider** — CLI-based, works directly with git, very effective for refactoring
- [ ] **Tabby** — self-hosted GitHub Copilot replacement, FIM optimized
- [ ] **TwinnyCode** — pure VS Code + Ollama
- [ ] **For YOUR daily use:** Continue.dev (best balance of features and integration) + Aider (for "make this big change")

### Day 2 — Setting Up Continue.dev with Local Models
- [ ] Install Continue.dev VS Code extension
- [ ] Configure `~/.continue/config.json`:
  ```json
  {
    "models": [
      { "title": "Qwen Coder", "provider": "ollama", "model": "qwen2.5-coder:7b" }
    ],
    "tabAutocompleteModel": {
      "title": "FIM",
      "provider": "ollama",
      "model": "qwen2.5-coder:1.5b-base"
    }
  }
  ```
- [ ] Use 1.5B for FIM (fill-in-middle autocomplete, must be fast)
- [ ] Use 7B for chat / longer reasoning
- [ ] **Code:** Set this up, use for 1 hour of real coding, take notes

### Day 3 — Fill-In-Middle (FIM) Models
- [ ] FIM = model trained to predict middle text given prefix + suffix
- [ ] Critical for code autocomplete: cursor often in middle of function
- [ ] Best local FIM models (Q4_K_M / Q5_K_M):
  - **Qwen 2.5 Coder 1.5B Base** — best speed/quality ratio for autocomplete
  - **Qwen 2.5 Coder 3B Base** — better quality, still fast
  - **StarCoder2 3B / 7B** — good fallback
  - **DeepSeek Coder V2 Lite 16B (MoE)** — only 2.4B active params, fits in 6GB!
- [ ] **Code:** Test FIM latency — should be <500ms for snappy UX

### Day 4 — Aider for Codebase-Wide Changes
- [ ] `pip install aider-chat`
- [ ] `aider --model ollama_chat/qwen2.5-coder:7b --no-auto-commits` in a git repo
- [ ] Aider reads files, makes diffs, applies them, you review
- [ ] Powerful for refactoring, adding features across files
- [ ] Pro tip: use `/add` to add specific files to context to save tokens
- [ ] **Code:** Use Aider on a small project, make a non-trivial change with it

### Day 5 — Per-Project Configuration
- [ ] Continue.dev: `.continuerc.json` in repo root for per-project models/prompts
- [ ] Aider: `.aider.conf.yml` for per-project config
- [ ] Custom slash commands: define `/explain`, `/test`, `/refactor` with project-specific prompts
- [ ] Codebase indexing: Continue.dev can embed your repo for semantic code search
- [ ] **Code:** Configure your favorite project with a custom prompt and slash commands

### 🔨 Saturday Project
- [ ] **CodeCompanion v1** — start your Month 4 main project
  - [ ] Document your full local code-assistant setup
  - [ ] Custom Continue.dev config with 3 models (FIM, chat, code-review)
  - [ ] Custom slash commands for your workflow
  - [ ] Measured latency: FIM TTFT, chat TTFT
  - [ ] Per-project conventions doc

### 📄 Sunday
- [ ] Read: Continue.dev docs (especially "Customizing")
- [ ] Read: Aider's "How it works" + leaderboard (Aider has a famous coding leaderboard)
- [ ] Explore: Qwen 2.5 Coder paper/technical report

---

## Week 14 — Chat UIs & Self-Hosted Platforms

### Day 1 — Open WebUI Deep Dive
- [ ] Multi-user with auth
- [ ] Workspaces, models, prompts, knowledge (built-in RAG)
- [ ] Function/tool integration
- [ ] Settings management
- [ ] Pipelines: custom plugins for arbitrary logic
- [ ] **Code:** Set up Open WebUI with 2 users, attach a knowledge base, try a pipeline

### Day 2 — AnythingLLM
- [ ] Workspaces with isolated RAG
- [ ] Built-in document upload + processing
- [ ] Multiple LLM backends (Ollama, local, cloud)
- [ ] Agent skills
- [ ] **Code:** Set up AnythingLLM, ingest a document collection, compare RAG quality to LocalRAG

### Day 3 — LibreChat
- [ ] Multi-provider (Ollama + OpenAI + Anthropic + Google + etc.)
- [ ] ChatGPT-style UI
- [ ] Plugin system, code interpreter
- [ ] Great for "local-first but cloud-fallback" workflows
- [ ] **Code:** Set up LibreChat with Ollama as one provider; compare UX

### Day 4 — Comparison Matrix
| Feature | Open WebUI | AnythingLLM | LibreChat | Custom (OllaMate) |
|---------|------------|-------------|-----------|-------------------|
| Polish | high | high | high | depends |
| Setup ease | Docker one-liner | Installer | Docker | code yourself |
| Multi-user | ✅ | ✅ | ✅ | hard |
| RAG built-in | ✅ | ✅✅ (focus) | partial | depends |
| Plugins | ✅ | ✅ | ✅ | depends |
| Self-hosted | ✅ | ✅ | ✅ | ✅ |
| Mobile-friendly | ✅ | ✅ | ✅ | depends |
- [ ] **Code:** Decide which one you'll keep running long-term as your daily driver

### Day 5 — Mobile Access to Your Local AI
- [ ] Option 1: Expose Open WebUI on your LAN, access from phone browser
- [ ] Option 2: Tailscale (free) — secure access from anywhere as if on home LAN
- [ ] Option 3: Cloudflare Tunnel — public URL, careful with auth
- [ ] Option 4: Native mobile apps that hit Ollama API (Enchanted for iOS, Ollama Android, etc.)
- [ ] **Code:** Set up Tailscale + Open WebUI; chat from your phone via your local laptop

### 🔨 Saturday Project
- [ ] **Daily-Driver Chat Setup**
  - [ ] Pick your preferred frontend (probably Open WebUI)
  - [ ] Configure 5+ Ollama models accessible
  - [ ] Set up a knowledge base for personal docs
  - [ ] Add Tailscale for phone access
  - [ ] Replace ChatGPT/Claude for non-sensitive daily use for a week — log what works and what doesn't

### 📄 Sunday
- [ ] Read: Open WebUI documentation cover-to-cover
- [ ] Watch: comparison reviews on YouTube

---

## Week 15 — Privacy, Security & Network Isolation

### Day 1 — Why Privacy Audit Matters
- [ ] Many "local" tools secretly phone home (telemetry, update checks, analytics)
- [ ] Real privacy means you've verified, not assumed
- [ ] Threat model: what data do you not want leaving? Who would intercept it?
- [ ] **Code:** Write your personal threat model (1 page): data classes, who can see what

### Day 2 — Tools to Audit Network Traffic
- [ ] **Wireshark** — packet capture, definitive but verbose
- [ ] **tcpdump (Linux/WSL2)** — CLI packet capture
- [ ] **Little Snitch (Mac), GlassWire (Windows)** — per-app firewall
- [ ] **Windows Firewall + Process Hacker** — quick checks
- [ ] **netstat / `ss -tunap`** — see open connections
- [ ] **Code:** Run Ollama, fire `tcpdump -i any port 11434` while making requests, observe purely local traffic

### Day 3 — Sandboxing & Isolation
- [ ] Run untrusted code from LLMs in Docker, Firejail, or restricted user
- [ ] WSL2 = a soft sandbox (separate kernel, less risk than running everything on Windows host)
- [ ] Virtual environments per project (`venv`, `conda`, `uv venv`)
- [ ] Don't `pip install` random unverified packages an LLM recommends — inspect first
- [ ] **Code:** Set up a Docker container with restricted network + read-only filesystem for code-executing agents

### Day 4 — Firewall Rules for Local AI Server
- [ ] Allow inbound only on chosen ports (e.g., 11434 for Ollama, 3000 for Open WebUI)
- [ ] Restrict to LAN only (192.168.x.x, 10.x.x.x): don't expose to internet by accident
- [ ] Windows: `New-NetFirewallRule -DisplayName "Ollama LAN" -Direction Inbound -LocalPort 11434 -Protocol TCP -Action Allow -RemoteAddress 192.168.0.0/16`
- [ ] Linux: `ufw allow from 192.168.0.0/16 to any port 11434`
- [ ] **Code:** Audit your current firewall — what's exposed to the internet right now?

### Day 5 — Encrypted Storage for Sensitive RAG Data
- [ ] If RAG indexes your tax docs / medical records — should they be encrypted at rest?
- [ ] Options: BitLocker (Windows), LUKS (Linux), encrypted Veracrypt container
- [ ] Per-app password encryption: SQLite + SQLCipher
- [ ] Trade-off: encrypted DB is slower for embeddings, fine for RAG-scale workloads
- [ ] **Code:** Move sensitive RAG vector store onto an encrypted container; document the workflow

### 🔨 Saturday Project
- [ ] **Privacy Audit Report for Your Local Stack**
  - [ ] Test each tool you use (Ollama, Open WebUI, Continue, etc.) for outbound calls
  - [ ] Document findings: which tools made network calls, to where, why
  - [ ] Configure firewall + DNS to block any unintended outbound
  - [ ] Publish anonymized version: helpful to community

### 📄 Sunday
- [ ] Read: "How to verify your local AI is actually local" — search relevant blog posts
- [ ] Read: Tailscale docs (if you didn't already in Week 14)

---

## Week 16 — Local AI Productivity Workflows

### Day 1 — Identify YOUR Workflows
- [ ] What do you do every day that's repetitive + uses words/text?
- [ ] Examples to brainstorm: email triage, meeting note summarization, code review, journal entries, research notes, code documentation, daily standups, technical writing, learning summaries
- [ ] **Code:** Write a list of 10 workflows; rank by frequency × annoyance

### Day 2 — Build: Email Summarizer
- [ ] Connect to local email (Thunderbird's IMAP, Outlook, or Gmail via IMAP with app password)
- [ ] Read last N emails (don't reply yet — read-only first)
- [ ] LLM summarizes: who, what, urgency, action items
- [ ] Output: daily morning briefing in your terminal or markdown file
- [ ] **Code:** ~150 lines: imaplib + Ollama + Markdown output

### Day 3 — Build: Meeting Notes Assistant
- [ ] Record audio (locally — Audacity, OBS, or `ffmpeg` from mic)
- [ ] Transcribe with faster-whisper (Week 35 preview)
- [ ] Summarize + extract action items + decisions with LLM
- [ ] Save as markdown alongside the audio
- [ ] **Code:** End-to-end script: record 5-min audio → transcript → summary

### Day 4 — Build: Daily Journal Companion
- [ ] Prompt yourself each morning/evening; LLM asks follow-up questions
- [ ] Save responses as markdown
- [ ] Weekly: LLM summarizes what's happened this week, what you should focus on
- [ ] Monthly: LLM finds patterns across journal entries
- [ ] **Code:** Simple CLI or Streamlit; saves to `~/journal/YYYY-MM-DD.md`

### Day 5 — Workflow Automation Patterns
- [ ] Schedule with Task Scheduler (Windows) or cron (Linux)
- [ ] Trigger on file change with watchdog
- [ ] Trigger on shortcut (Windows hotkeys + AutoHotkey, Linux + xbindkeys)
- [ ] Output destinations: terminal, markdown file, system notification, email
- [ ] **Code:** Make ONE workflow run automatically every morning, untouched

### 🔨 Saturday Project
- [ ] **ChatVerse v1 + 3 Real Workflows**
  - [ ] Multi-model router (Project B for Month 4)
  - [ ] Integrate at least 3 of your productivity workflows
  - [ ] Measurable goal: replace ~30 minutes/day of manual work
  - [ ] Track results for 2 weeks

### 📄 Sunday
- [ ] Read: "Local AI for Personal Productivity" blog posts (search Reddit r/LocalLLaMA, HN)
- [ ] Reflect: how is your daily life different now that local AI is in your workflow?

---

### ✅ Phase 2 Completion Checklist (End of Month 4)
- [ ] Can build production-ready local RAG with hybrid retrieval and reranking
- [ ] Master at least one chat UI platform (Open WebUI / AnythingLLM / LibreChat)
- [ ] Use a local code assistant daily (Continue.dev / Aider with Qwen Coder)
- [ ] Verified your local AI stack is truly offline (privacy audit)
- [ ] Built 3+ personal productivity workflows powered by local AI
- [ ] Built LocalRAG + DocVault as Month 3 projects
- [ ] Built CodeCompanion + ChatVerse as Month 4 projects
- [ ] **You now PREFER local AI for many daily tasks over cloud — calibrate where it wins**

[⬆ Back to Table of Contents](#toc)

---

# ═══════════════════════════════════════════════════
# PHASE 3: LOCAL FINE-TUNING (Weeks 17-24, Months 5-6)
# QLoRA, Unsloth, dataset engineering, domain models ⭐
# ═══════════════════════════════════════════════════

> This is the phase that separates "local AI users" from "local AI builders."
> By the end of Month 6, you will have **fine-tuned a real model on your data, on your 6GB GPU**, evaluated it, and deployed it. That's the 6-month milestone.

---

## 🔄 Buffer Week (Month 4 Revision) Local AI
- [ ] Revise: RAG architecture, hybrid search, document processing
- [ ] Revise: chat platforms (Open WebUI), coding assistants (Continue + Aider)
- [ ] Revise: privacy audit methodology
- [ ] Hands-on: improve your LocalRAG or CodeCompanion based on 2 weeks of real use
- [ ] Read: skim Unsloth docs and the "Tutorial" notebook to preview Month 5
- [ ] Journal: "I'm about to start fine-tuning. What domain do I want a specialist for?"

---

## Week 17 — QLoRA Theory for 6GB Cards

### Day 1 — Why Fine-Tune at All?
- [ ] Pre-trained models are generalists. Your domain is specific.
- [ ] Fine-tuning bakes knowledge / style / format into the weights → no need to repeat in prompt
- [ ] Three reasons to fine-tune locally:
  - **Style:** make a model sound like YOU (or your brand, or a specific tone)
  - **Format:** reliably output exact structures (JSON schemas, code, specific dialect)
  - **Knowledge:** less common — only worth it for high-volume, narrow domain
- [ ] **Don't fine-tune for:** simple facts (use RAG), changing personality per query (use system prompts)
- [ ] **Code:** List 3 concrete fine-tuning goals you actually want

### Day 2 — Full Fine-Tuning vs LoRA vs QLoRA
- [ ] **Full fine-tuning:** update ALL parameters. For 8B model = 16 GB params + 32-64 GB optimizer state. NO chance on 6GB GPU
- [ ] **LoRA:** freeze base, add low-rank adapter matrices (B @ A) to specific layers. ~1% of params trained. Optimizer state much smaller. Still needs base in FP16 = ~14GB for 7B. Still won't fit
- [ ] **QLoRA:** quantize base to NF4 (4-bit), keep LoRA in FP16. Base = ~4GB for 7B. LoRA + optimizer = ~1-2 GB. **Fits in 6GB! This is the technique you'll use.**
- [ ] **Code:** Draw the memory layout on paper — visualize it

### Day 3 — LoRA Math
- [ ] Original layer: `W ∈ ℝ^(d × k)`, output `y = Wx`
- [ ] LoRA: add `ΔW = B @ A` where `B ∈ ℝ^(d × r)` and `A ∈ ℝ^(r × k)`, `r << d, k`
- [ ] During training: only `A` and `B` get gradients. `W` is frozen.
- [ ] At inference: either keep adapter separate (`y = Wx + B @ A @ x * alpha/r`) or merge: `W_new = W + B @ A`
- [ ] **Rank r:** controls capacity. Common values: 8, 16, 32, 64. Higher r = more params trained, more memory
- [ ] **Alpha:** scaling factor, typically alpha = r or alpha = 2r
- [ ] **Target modules:** usually q_proj, k_proj, v_proj, o_proj (attention), plus optionally MLP layers
- [ ] **Code:** Read the LoRA paper or HuggingFace PEFT docs; compute parameter count for r=16 on Llama 3.1 8B

### Day 4 — QLoRA-Specific Details
- [ ] **NF4 quantization:** NormalFloat 4-bit, optimized for normally-distributed weights
- [ ] **Double quantization:** quantize the quantization constants too — saves ~0.4 bits/param
- [ ] **Paged optimizers:** offload Adam states to CPU when not in use, prevents OOM spikes
- [ ] **Crucial detail:** LoRA must be in higher precision (FP16/BF16) for stable training; only base is in NF4
- [ ] **Why it works:** quantization noise + gradient noise are both small enough that the LoRA can adapt around them
- [ ] **Code:** Estimate memory for QLoRA on Llama 3.1 8B, r=32, batch 1, seq 2048: base ~4.5GB + LoRA ~120MB + activations ~500MB + optimizer ~250MB ≈ 5.5GB ✅ fits 6GB

### Day 5 — Hyperparameters That Matter
- [ ] **Learning rate:** 1e-4 to 5e-4 for LoRA (much higher than full fine-tuning's 1e-5)
- [ ] **Batch size:** as large as fits, use gradient accumulation to simulate larger
- [ ] **Sequence length:** longer = more memory. Use 1024 or 2048 for most use cases
- [ ] **Number of epochs:** 2-3 is usually plenty. Beyond that = overfitting
- [ ] **LoRA rank/alpha:** start with r=16, alpha=32. Increase r if undertraining
- [ ] **LR schedule:** linear warmup (3-10% of steps) then cosine decay
- [ ] **Gradient checkpointing:** trade compute for memory (essential on 6GB) — slows ~30% but enables larger batch
- [ ] **Code:** Write a "memory budget" worksheet: given model + seq_len + r + batch, predict VRAM use

### 🔨 Saturday Project
- [ ] **QLoRA Memory Budget Calculator**
  - [ ] Inputs: model size, seq length, batch, rank, gradient checkpointing on/off
  - [ ] Outputs: predicted peak VRAM, recommended adjustments if over budget
  - [ ] Validate against actual measurements from Week 18+

### 📄 Sunday Reading
- [ ] Paper: "LoRA: Low-Rank Adaptation of Large Language Models" (Hu et al., 2021)
- [ ] Paper: "QLoRA: Efficient Finetuning of Quantized LLMs" (Dettmers et al., 2023)
- [ ] (Both will be revisited in main roadmap Month 4; the math sticks deeper after you USE it)

---

## Week 18 — Unsloth Deep Dive

> Unsloth is the single most important tool for low-VRAM fine-tuning. It's 2-5x faster and uses 70% less VRAM than vanilla HuggingFace TRL via hand-written Triton kernels + smart memory management. Master it.

### Day 1 — What Makes Unsloth Special
- [ ] Custom Triton kernels for: attention, RMSNorm, RoPE, cross-entropy, MLP layers
- [ ] Smart gradient checkpointing (only re-computes what's needed)
- [ ] Memory-efficient backward pass
- [ ] Bug-fixed gradient accumulation (HuggingFace had a bug Unsloth caught and fixed)
- [ ] Native QLoRA, full fine-tune, DPO, ORPO support
- [ ] Works on consumer GPUs (4060, 1660, T4 in Colab, and YES your RTX 1000 Ada)
- [ ] **Code:** Install: `pip install unsloth` (in WSL2 strongly recommended; Windows native works but more brittle)

### Day 2 — Unsloth Notebook Walkthrough
- [ ] Clone: https://github.com/unslothai/unsloth
- [ ] Open the example notebook for Llama 3.1 8B
- [ ] Walk through cell by cell, understand each step:
  - Load base model with `FastLanguageModel.from_pretrained(...)` (auto-applies 4-bit quant)
  - Apply LoRA with `FastLanguageModel.get_peft_model(...)`
  - Prepare dataset (chat format)
  - Train with `SFTTrainer` (from TRL)
  - Save and convert to GGUF
- [ ] **Code:** Run the notebook end-to-end on a tiny dataset (50 examples) — verify it trains on YOUR GPU

### Day 3 — Unsloth on Llama 3.2 3B (Quick Win)
- [ ] 3B model trains FAST on 6GB — perfect first real fine-tune
- [ ] Dataset: pick something simple, e.g., Alpaca subset (1k examples) or your own ~200 examples
- [ ] Target: train in <1 hour, observable loss decrease, sample generations clearly affected
- [ ] **Code:** Fine-tune Llama 3.2 3B on Alpaca-1k; save checkpoints

### Day 4 — Unsloth on Llama 3.1 8B (The Real Test)
- [ ] 8B at 4-bit + LoRA r=16 should fit comfortably with seq_len=1024
- [ ] If OOM: reduce seq_len, enable gradient checkpointing (`use_gradient_checkpointing="unsloth"`), reduce r
- [ ] **Code:** Fine-tune Llama 3.1 8B on a 500-example dataset; measure training time, GPU util

### Day 5 — Saving and Deploying Trained Models
- [ ] Save LoRA adapter only: `model.save_pretrained("lora-adapter")` (~100MB)
- [ ] Save merged 16-bit: `model.save_pretrained_merged(..., save_method="merged_16bit")`
- [ ] Save merged 4-bit: `model.save_pretrained_merged(..., save_method="merged_4bit")`
- [ ] Save as GGUF: `model.save_pretrained_gguf(..., quantization_method="q4_k_m")`
- [ ] Push to HuggingFace Hub: `model.push_to_hub_gguf(...)` (optional, only if you want to share)
- [ ] **Code:** Take your Week 18 trained model → GGUF → Ollama Modelfile → `ollama create my-tuned -f Modelfile` → chat

### 🔨 Saturday Project
- [ ] **First Real Fine-Tune End-to-End**
  - [ ] Pick a small task you care about (e.g., "convert any text to my writing style", or "answer questions about my favorite framework")
  - [ ] Curate 200-500 examples (manual + LLM-assisted)
  - [ ] QLoRA fine-tune with Unsloth on Llama 3.1 8B
  - [ ] Convert to GGUF, push to Ollama
  - [ ] Compare 5 prompts: base model vs your fine-tune
  - [ ] **Write up the experience as a blog post**

### 📄 Sunday
- [ ] Read: Unsloth blog posts (multiple, all gold)
- [ ] Watch: Daniel Han's interviews / talks on Unsloth design
- [ ] Bookmark: Unsloth Discord for help

---

## Week 19 — Axolotl & LLaMA-Factory

> Once you know Unsloth, learn Axolotl + LLaMA-Factory. They're more configurable and used by more research projects. Unsloth is faster on 1 GPU; Axolotl scales better and supports more advanced techniques.

### Day 1 — Axolotl Overview
- [ ] YAML-config-driven fine-tuning, much more options than Unsloth
- [ ] Supports: LoRA, QLoRA, full FT, DPO, ORPO, SimPO, ReLoRA, etc.
- [ ] Supports: DeepSpeed ZeRO, FSDP, multi-GPU
- [ ] Used by many open models you've seen (Hermes, Dolphin, OpenHermes, etc.)
- [ ] **Code:** Install: `pip install axolotl` (in WSL2)

### Day 2 — Axolotl YAML Anatomy
- [ ] Base model + tokenizer
- [ ] Dataset config (path, format, prompts)
- [ ] LoRA config (r, alpha, target_modules, dropout)
- [ ] Training config (lr, batch, epochs, scheduler, gradient_accumulation)
- [ ] Optimization (load_in_4bit, flash_attn, gradient_checkpointing)
- [ ] Save / push config
- [ ] **Code:** Write a YAML config for Llama 3.1 8B QLoRA on your dataset

### Day 3 — Run Axolotl Training
- [ ] `accelerate launch -m axolotl.cli.train config.yaml`
- [ ] Monitor: tensorboard, wandb, or terminal logs
- [ ] Compare: same dataset on Unsloth vs Axolotl — speed, final loss, generation quality
- [ ] **Code:** Same fine-tune as last Saturday, redone with Axolotl. Document differences

### Day 4 — LLaMA-Factory (UI-First Alternative)
- [ ] WebUI for fine-tuning (Gradio-based)
- [ ] All major techniques: SFT, DPO, KTO, ORPO, PPO, reward modeling
- [ ] Supports nearly every model family on HuggingFace
- [ ] CLI also available for production
- [ ] **Code:** Try the WebUI for a fine-tune — useful when you want to compare configs quickly without YAML

### Day 5 — Choosing Your Tool
- [ ] **Use Unsloth when:** single GPU (6-24 GB), want fastest training, want simplest code
- [ ] **Use Axolotl when:** multi-GPU, complex configs, advanced techniques, research-y workflows
- [ ] **Use LLaMA-Factory when:** want a UI, lots of experimentation, beginner-friendly
- [ ] For YOUR 6GB GPU: **Unsloth is your daily driver.** Axolotl for occasional advanced runs.
- [ ] **Code:** Document your choice in your local journal with reasons

### 🔨 Saturday Project
- [ ] **TinyTuner v1** — start Month 5 main project
  - [ ] Wrapper around Unsloth with auto-config based on VRAM
  - [ ] CLI: `tinytune --model llama3.1-8b --dataset my-data.jsonl --task chat`
  - [ ] Auto-selects: r, alpha, batch, seq_len, grad_checkpoint
  - [ ] Refuses to start if config will OOM (predicted)
  - [ ] After training: auto-merges + auto-converts to GGUF + auto-creates Ollama Modelfile

### 📄 Sunday
- [ ] Read: Axolotl docs (especially "Dataset Formats")
- [ ] Read: LLaMA-Factory README
- [ ] Compare: top fine-tuned community models — which framework did they use?

---

## Week 20 — Dataset Engineering for Fine-Tuning

> Data quality matters more than model choice. A small (500-2000 examples) high-quality dataset will beat a 100k noisy dataset every time on local-AI scale.

### Day 1 — Dataset Formats
- [ ] **Alpaca:** `{"instruction": ..., "input": ..., "output": ...}` — simple, popular
- [ ] **ChatML / OpenAI:** `{"messages": [{"role": "user", "content": ...}, {"role": "assistant", "content": ...}]}`
- [ ] **ShareGPT:** similar to OpenAI but with `from`/`value` keys (multi-turn)
- [ ] **DPO:** `{"prompt": ..., "chosen": ..., "rejected": ...}` — for preference tuning
- [ ] **Code:** Convert between formats using simple Python scripts

### Day 2 — Where to Get Data
- [ ] **Existing datasets on HuggingFace:**
  - Alpaca (52k), Dolly (15k), OpenAssistant Conversations, Capybara
  - Domain-specific: Magicoder (code), MathInstruct (math), Aya (multilingual)
- [ ] **Your own data:**
  - Personal writings (blog posts, emails sent, journal)
  - Codebases (your repos with commit messages as labels)
  - Domain corpus (research papers, docs)
- [ ] **Synthetic data:** generate with a teacher model (more powerful local or one-time GPT-4 API)
- [ ] **Code:** Inspect 3 popular datasets, see how they're structured

### Day 3 — Synthetic Data Generation Patterns
- [ ] **Self-Instruct:** seed examples → LLM generates similar examples
- [ ] **Evol-Instruct (WizardLM):** start with simple Qs, iteratively make them harder
- [ ] **Magpie:** prompt the LLM with empty user turn, harvest what it generates
- [ ] **Distillation:** ask a stronger model to answer; use its answers as training data
- [ ] **Code:** Generate 100 synthetic examples in your domain using Self-Instruct pattern

### Day 4 — Data Quality Filtering
- [ ] **Length filter:** drop too-short (< 10 tokens) or too-long (> seq_len) examples
- [ ] **Deduplication:** exact match dedup + near-dedup (MinHash via `datasketch`)
- [ ] **Quality scoring:** ask the LLM to rate each example 1-5; keep 4+
- [ ] **Toxicity filter:** Detoxify or Llama Guard if relevant
- [ ] **PII filter:** detect emails, phone numbers, names; redact or drop
- [ ] **Code:** Apply all filters to a synthetic dataset; observe how many remain

### Day 5 — Format-Aware Augmentation
- [ ] System prompts: include them in training data → model learns to follow them
- [ ] Conversation length: include both single-turn and multi-turn examples
- [ ] Negative examples: include refusals, "I don't know," partial answers — model needs these
- [ ] Format diversity: train on multiple legitimate output formats so model adapts at inference
- [ ] **Code:** Audit your dataset: does it represent ALL ways your model should respond?

### 🔨 Saturday Project
- [ ] **Domain Dataset Curation**
  - [ ] Pick your Month 6 domain (research it, journal about it)
  - [ ] Collect/generate 500-1000 training examples
  - [ ] Apply quality filters
  - [ ] Hold out 50 for evaluation, never use in training
  - [ ] Document the data lineage and filtering choices

### 📄 Sunday
- [ ] Paper: "Self-Instruct" (Wang et al., 2022)
- [ ] Paper: "Magpie" (Xu et al., 2024)
- [ ] Read: HuggingFace blog on "DataTrove" and "FineWeb"

---

## 🔄 Buffer Week (Month 5 Revision) Local AI
- [ ] Revise: QLoRA theory (NF4, LoRA math, paged optimizers)
- [ ] Revise: Unsloth notebooks; you should be able to write a training script from memory
- [ ] Revise: dataset formats, quality filters, synthetic data
- [ ] **Build Monthly Project A:** TinyTuner — push-button QLoRA on 6GB
- [ ] **Build Monthly Project B:** ModelMart — your personal model & adapter hub
- [ ] Push to GitHub
- [ ] Journal: "I now know how to fine-tune. What domain model will define my Month 6?"

---

## Week 21 — Your First Real Fine-Tune

> This week the training wheels come off. You will train, evaluate, iterate, and ship a model for a real use case.

### Day 1 — Pick a Real Use Case
- [ ] Examples (pick one):
  - "Marathi customer service bot" (multilingual + style)
  - "Indian tax law assistant" (domain knowledge + format)
  - "Code reviewer in your team's style" (style + format)
  - "JEE physics tutor" (domain + format + reasoning)
  - "Your personal writing assistant" (style transfer)
- [ ] Write a 1-pager: user, pain, success criteria, sample queries with desired answers
- [ ] **Code:** This is the spec doc you'll refer back to

### Day 2 — Dataset Plan
- [ ] How many examples? Minimum 500 for visible effect, 1000-3000 ideal, beyond 5000 diminishing returns at this scale
- [ ] Sources: manual + synthetic + scraped + filtered
- [ ] Output format: chat or instruction or both
- [ ] Splits: 90% train, 5% val, 5% test (test is golden — never look at it until final eval)
- [ ] **Code:** Build your dataset; sanity-check a few examples manually

### Day 3 — Training Run #1 (Baseline)
- [ ] Use TinyTuner or raw Unsloth
- [ ] Conservative config: r=16, alpha=32, lr=2e-4, 2 epochs, batch 1 + grad accum 8
- [ ] Run; monitor loss curve
- [ ] **Code:** Save the run. Compare val loss to base model val loss (run val on base model first as baseline)

### Day 4 — Evaluation
- [ ] Run 5 sample prompts manually — vibes check first
- [ ] Run your held-out test set through both base and fine-tune
- [ ] Score: perplexity (lower better), format compliance (regex check), task accuracy (custom rubric)
- [ ] LLM-as-judge: ask Llama 3.1 8B (or even GPT-4 if you have access) to rate base vs fine-tune outputs
- [ ] **Code:** Build an `eval.py` you'll re-run every iteration

### Day 5 — Iterate
- [ ] If undertrained: more epochs, higher lr, larger r
- [ ] If overfit: fewer epochs, lower lr, dropout in LoRA (0.05-0.1)
- [ ] If catastrophic forgetting (model forgot general abilities): mix in some general SFT data (e.g., 20% Alpaca)
- [ ] If format issues: add more format-anchoring examples; use structured output at inference
- [ ] **Code:** Document each iteration with config + results + reflection — this becomes your training journal

### 🔨 Saturday Project
- [ ] **Real Fine-Tune Shipped**
  - [ ] Pick best checkpoint
  - [ ] Convert to GGUF Q4_K_M
  - [ ] Push to Ollama
  - [ ] Document everything: dataset stats, training config, eval results, sample outputs
  - [ ] Optional: push to HuggingFace Hub (model card with proper attribution)

### 📄 Sunday
- [ ] Read: "Catastrophic Forgetting in LLM Fine-tuning" articles
- [ ] Watch: Daniel Han talks on common fine-tuning failures

---

## Week 22 — LoRA Merging, Stacking & GGUF Conversion

### Day 1 — Why Merge LoRAs?
- [ ] Inference with adapter has overhead (extra matmuls)
- [ ] Merged model: `W' = W + B @ A` — same architecture as base, no LoRA at inference
- [ ] Trade-off: lose flexibility (can't swap adapters) for simpler deployment
- [ ] **Code:** Merge a LoRA into base, save full model

### Day 2 — Multi-LoRA Inference (No Merge)
- [ ] PEFT library supports loading multiple LoRAs and switching at runtime
- [ ] Useful for: same base, multiple specialized adapters (writing, code, tutor)
- [ ] **Code:** Load base + 2 adapters; switch between them in one Python session

### Day 3 — LoRA Stacking & Merging Methods
- [ ] **Naive addition:** `W' = W + α₁ · LoRA₁ + α₂ · LoRA₂` (works for orthogonal tasks)
- [ ] **TIES merging:** trim, elect sign, disjoint merge (handles conflicts better)
- [ ] **DARE:** drop-and-rescale before merge (regularizes)
- [ ] **MergeKit:** the toolkit for all of these (`pip install mergekit`)
- [ ] **Code:** Stack 2 LoRAs (e.g., "Hindi style" + "polite tone"); test if they compose

### Day 4 — Converting Merged Model to GGUF
- [ ] `python convert_hf_to_gguf.py /path/to/merged-model --outfile out.gguf --outtype f16`
- [ ] Then quantize: `./llama-quantize out.gguf out-q4_k_m.gguf Q4_K_M`
- [ ] Unsloth has a shortcut: `model.save_pretrained_gguf(...)` does both
- [ ] **Code:** Convert your fine-tuned model to 2-3 quantization levels

### Day 5 — Ollama Modelfile for Fine-Tuned Model
- [ ] ```Modelfile
FROM ./my-tuned-q4_k_m.gguf
TEMPLATE """<|im_start|>system
{{ .System }}<|im_end|>
<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
"""
PARAMETER temperature 0.7
PARAMETER num_ctx 4096
SYSTEM """You are MyDomainBot, a specialist in [domain]."""
```
- [ ] `ollama create mydomain -f Modelfile`
- [ ] `ollama run mydomain`
- [ ] **Code:** Get your fine-tune running in Ollama; chat with it

### 🔨 Saturday Project
- [ ] **ModelMart v1**
  - [ ] Registry for all your LoRA adapters + merged models
  - [ ] CLI: `modelmart list`, `modelmart deploy <name>`, `modelmart stack <a> <b>`
  - [ ] SQLite metadata + filesystem storage
  - [ ] Auto-Modelfile generation per registered model

### 📄 Sunday
- [ ] Read: "TIES-Merging" paper (Yadav et al., 2023)
- [ ] Read: MergeKit README and examples
- [ ] Explore: top community-merged models on HuggingFace; what techniques did they use?

---

## Week 23 — DPO & Preference Tuning Locally

### Day 1 — Why DPO After SFT?
- [ ] SFT teaches "what's a valid answer"
- [ ] DPO teaches "which valid answer is better"
- [ ] Use cases: tone (concise vs verbose), correctness preference, safety, format strictness
- [ ] DPO doesn't need a separate reward model (vs PPO/RLHF)
- [ ] **Code:** Read DPO paper (overlaps main roadmap Month 3); understand the loss

### Day 2 — DPO Dataset Format
- [ ] `{"prompt": "...", "chosen": "good answer", "rejected": "bad answer"}`
- [ ] How to get pairs:
  - **Manual:** label 200 yourself
  - **Synthetic:** generate 2 answers, use stronger model as judge
  - **Public datasets:** UltraFeedback, HH-RLHF, Anthropic preference data
- [ ] **Code:** Build a 100-example preference dataset for your domain

### Day 3 — DPO Training with Unsloth
- [ ] Unsloth supports DPO natively (`DPOTrainer`)
- [ ] Start from your SFT-fine-tuned model (not the base!)
- [ ] Hyperparams: `beta=0.1` (KL strength), lr=5e-6 (much lower than SFT)
- [ ] **Code:** Run DPO on top of your Week 21 model; observe loss curves

### Day 4 — Beyond DPO: KTO, ORPO, SimPO
- [ ] **KTO (Kahneman-Tversky Optimization):** uses single-rating data (not pairs!), simpler to label
- [ ] **ORPO (Odds Ratio Preference Optimization):** combines SFT and preference in one stage
- [ ] **SimPO:** length-normalized DPO, often better for short answers
- [ ] All supported by Unsloth (`ORPOTrainer`, `KTOTrainer`)
- [ ] **Code:** Try ORPO on the same dataset; compare to SFT+DPO sequential

### Day 5 — Evaluating Preference-Tuned Models
- [ ] Pairwise win rate vs base SFT model (LLM-as-judge)
- [ ] Length: are answers getting verbose or concise as desired?
- [ ] Refusal rate: is it appropriately careful?
- [ ] **Code:** Run pairwise eval comparing SFT vs SFT+DPO on 50 prompts

### 🔨 Saturday Project
- [ ] **Preference-Tuned Domain Model**
  - [ ] Take your Week 21 SFT model
  - [ ] Build a preference pair dataset (100-200 pairs)
  - [ ] Run DPO (or ORPO) on top
  - [ ] Compare base vs SFT vs SFT+DPO
  - [ ] Choose your favorite; convert to GGUF; deploy to Ollama

### 📄 Sunday
- [ ] Paper: "DPO" (Rafailov et al., 2023) — main roadmap will cover deeply
- [ ] Paper: "ORPO" (Hong et al., 2024)
- [ ] Paper: "SimPO" (Meng et al., 2024)
- [ ] Reflect: did preference tuning materially improve your model? When does it not?

---

## Week 24 — Evaluating Fine-Tuned Models ⭐ 6-MONTH MILESTONE

### Day 1 — Why Evaluation Is Hard
- [ ] LLM outputs are open-ended — no easy "correct/incorrect"
- [ ] Need multiple eval axes: helpfulness, correctness, format, style, safety
- [ ] Benchmarks lie: a model that "scores well" might be unusable
- [ ] **Code:** Write your own eval philosophy: what does "better" mean for YOUR fine-tune?

### Day 2 — Multi-Axis Evaluation
- [ ] **Task-specific accuracy:** custom rubric for your domain
- [ ] **General capabilities (regression check):** MMLU mini, HellaSwag mini, ARC easy — has fine-tuning broken general abilities?
- [ ] **Style adherence:** length, vocab, formality (textstat)
- [ ] **Safety:** does it refuse appropriate prompts but answer benign ones?
- [ ] **Speed:** tokens/sec (should be similar to base + quant overhead)
- [ ] **Code:** Build an eval harness covering all axes

### Day 3 — Comparison Reports
- [ ] Side-by-side: base | SFT | SFT+DPO | (optional cloud) for 20 hand-picked queries
- [ ] HTML report with model names, prompts, outputs, your ratings
- [ ] Per-axis scores in a table
- [ ] **Code:** Build a `compare-models.py` that produces this report

### Day 4 — Catastrophic Forgetting Detection
- [ ] Test on tasks UNRELATED to your fine-tune: math, code, multilingual
- [ ] If accuracy drops > 10% on these, you have CF
- [ ] Mitigations: lower epochs, lower lr, smaller r, mix in general data
- [ ] **Code:** Run regression check on your final model; report deltas

### Day 5 — Deployment Decision
- [ ] Is your fine-tune actually better than RAG + base model? Be honest.
- [ ] Sometimes the answer is: "the SFT model + my RAG is sufficient, save the DPO work"
- [ ] Sometimes the answer is: "fine-tune for style + RAG for facts"
- [ ] Document the decision: this model is for X, deploys as Y, expected to handle Z, fallback for W
- [ ] **Code:** Write the deployment doc; commit to repo

### 🔨 Saturday Project ⭐ THE 6-MONTH MILESTONE
- [ ] **DomainDoc — Fine-Tuned Domain Specialist (Month 6 Project A)**
  - [ ] Full pipeline: domain → data → SFT → DPO → eval → deploy
  - [ ] 50-question domain eval suite with gold answers
  - [ ] Comparison report: base vs SFT vs SFT+DPO vs (optional) cloud
  - [ ] Deployed on Ollama, integrated with LocalRAG for facts
  - [ ] **Blog post: "How I built a domain-specialist LLM on a 6GB laptop"**
  - [ ] **FineTuneBench (Project B):** adapter evaluation toolkit

### 📄 Sunday — Reflect on 6 Months
- [ ] Write a "6-Month Retrospective" blog post:
  - What you knew 6 months ago vs now
  - The 3 biggest surprises (positive and negative)
  - Your favorite local model and why
  - What you'd tell a beginner starting now
- [ ] Publish on dev.to, Medium, or your blog
- [ ] Cross-post to r/LocalLLaMA — get feedback from the community

---

### ✅ Phase 3 Completion Checklist (End of Month 6 — 6-MONTH MILESTONE) ⭐
- [ ] Understand QLoRA from first principles (NF4 + LoRA math + memory layout)
- [ ] Can fine-tune any 3-8B model on 6GB VRAM using Unsloth
- [ ] Know Axolotl + LLaMA-Factory for advanced configs
- [ ] Can curate datasets (collect, synthesize, filter, format) for any domain
- [ ] Have completed at least one real SFT fine-tune deployed to Ollama
- [ ] Have applied DPO/ORPO on top of SFT
- [ ] Can detect and mitigate catastrophic forgetting
- [ ] Built a multi-axis evaluation harness
- [ ] **Shipped DomainDoc: a fine-tuned domain specialist with measurable improvement over base**
- [ ] **Published a 6-month retrospective blog post**
- [ ] **🎯 You are a credible Local AI Practitioner. You can have substantive conversations with anyone in the field.**

[⬆ Back to Table of Contents](#toc)

---

# ═══════════════════════════════════════════════════
# PHASE 4: LOCAL AGENTS (Weeks 25-32, Months 7-8)
# Function calling, tool use, multi-agent, observability
# ═══════════════════════════════════════════════════

> Agents are hard. Local small-model agents are HARDER. This phase is about
> making them reliable on 6GB VRAM — the rare skill that separates the
> production-ready local AI engineer from the demo-builder.

---

## 🔄 Buffer Week (Month 6 Revision) Local AI ⭐ 6-MONTH MILESTONE
- [ ] Review your DomainDoc model — use it for a week, note real wins/failures
- [ ] Iterate based on real use feedback (more data? different prompts? merge with base?)
- [ ] Write the comprehensive blog post documenting your 6-month journey
- [ ] Share to community: r/LocalLLaMA, Twitter, HN
- [ ] Read: papers on function calling and tool use to prepare for Month 7
- [ ] Journal: "I've leveled up. The next phase is agents — what's my goal?"

---

## Week 25 — Function Calling with Local LLMs

### Day 1 — What Is Function Calling?
- [ ] LLM emits structured JSON describing which function to call with what arguments
- [ ] Caller runs the function, returns result, LLM continues
- [ ] Building block of all agents
- [ ] **Two implementations:**
  - **Native (model trained for it):** Llama 3.1 Instruct, Qwen 2.5 Instruct, Mistral Instruct
  - **Prompt-engineered:** any model can be coerced with the right system prompt + parsing
- [ ] **Code:** Run a single function call example with Ollama using a function-calling-trained model

### Day 2 — Ollama Function Calling API
- [ ] Ollama supports OpenAI-format `tools` parameter (and Anthropic-style)
- [ ] ```python
  response = ollama.chat(
    model='llama3.1:8b',
    messages=[{'role':'user', 'content':'What is the weather in NYC?'}],
    tools=[{
      'type':'function',
      'function':{
        'name':'get_weather',
        'parameters':{'type':'object','properties':{'city':{'type':'string'}},'required':['city']}
      }
    }]
  )
  ```
- [ ] Parse `response.message.tool_calls`
- [ ] **Code:** Build 3 tool functions (weather, calculator, search) and chain calls

### Day 3 — Which Small Models Are Good at Tools?
- [ ] **Best for tools (Q4_K_M):**
  - **Qwen 2.5 7B Instruct** ⭐ — best small-model tool use in my experience
  - **Llama 3.1 8B Instruct** — solid, well-documented
  - **Hermes 3 (Llama 3 fine-tune)** — Nous Research's strong tool-use model
  - **Functionary** — explicitly trained for function calling
- [ ] **Not ideal:** very small models (Phi-3 Mini, Gemma 2B) — they hallucinate tool names
- [ ] **Code:** Benchmark all 4 on 20 tool-use scenarios; track success rate

### Day 4 — When the Model Lies About Tools
- [ ] Small models often: invent tool names, miss required args, malform JSON, call tools that should be plain text
- [ ] Defenses:
  - Validate against schema; reject + reprompt with error
  - Use grammar-constrained generation (forces valid JSON via outlines/lm-format-enforcer)
  - Give very clear tool descriptions in system prompt
  - Few-shot examples of correct tool use
- [ ] **Code:** Add schema validation + retry loop to your tool-using code

### Day 5 — Structured Output Libraries
- [ ] **outlines:** grammar-constrained generation, supports Pydantic, JSON schema
- [ ] **lm-format-enforcer:** lower-level, more flexible
- [ ] **instructor:** patches OpenAI SDK to enforce Pydantic models (works with Ollama via OpenAI-compatible endpoint)
- [ ] **Code:** Use instructor:
  ```python
  import instructor
  from openai import OpenAI
  from pydantic import BaseModel
  
  class WeatherQuery(BaseModel):
      city: str
      units: str
  
  client = instructor.patch(OpenAI(base_url="http://localhost:11434/v1/", api_key="ollama"))
  result = client.chat.completions.create(
    model="llama3.1:8b",
    response_model=WeatherQuery,
    messages=[{"role":"user","content":"Get NYC weather in Fahrenheit"}],
  )
  ```

### 🔨 Saturday Project
- [ ] **Function-Calling Benchmark Suite**
  - [ ] 30 tool-use scenarios across 5 domains (math, search, file ops, code, API)
  - [ ] Test on 4 local models (Q4_K_M each)
  - [ ] Track: schema validity %, correct args %, semantic success %, retries needed
  - [ ] Publish the report; this is a useful artifact for the community

### 📄 Sunday
- [ ] Read: Ollama tools docs + Qwen 2.5 function calling docs
- [ ] Read: outlines / instructor READMEs
- [ ] Watch: "Function Calling with Open Source LLMs" talks

---

## Week 26 — smolagents, LangGraph & Ollama Agents

### Day 1 — smolagents (HuggingFace)
- [ ] HuggingFace's minimalist agent framework (recent, ~3000 LOC)
- [ ] Two patterns: CodeAgent (writes Python to solve tasks), ToolCallingAgent (classical tool calling)
- [ ] Native Ollama support
- [ ] Designed to be readable and educational
- [ ] **Code:** Install: `pip install smolagents`; build a `ToolCallingAgent` with Ollama as backend

### Day 2 — CodeAgent Pattern (Powerful)
- [ ] Agent writes and executes Python code rather than calling discrete tools
- [ ] More expressive, less brittle, but needs sandboxed execution
- [ ] HuggingFace's E2B integration for sandboxed code execution
- [ ] **Code:** Build a CodeAgent that solves data analysis tasks; sandbox with Docker

### Day 3 — LangGraph (More Structured)
- [ ] State machine-based agent framework from LangChain team
- [ ] Define nodes (steps) and edges (transitions)
- [ ] Supports cycles (an agent can loop, retry, branch)
- [ ] More boilerplate than smolagents but more reliable for complex workflows
- [ ] Native Ollama support via `ChatOllama`
- [ ] **Code:** Build a simple LangGraph agent: researcher node → writer node → reviewer node

### Day 4 — Comparing Frameworks
| Framework | Best For | Pros | Cons |
|-----------|----------|------|------|
| smolagents | Quick projects, learning | Minimal, readable, CodeAgent is powerful | Less battle-tested |
| LangGraph | Complex flows | Mature, observable, debuggable | Steeper learning curve |
| AutoGen (Microsoft) | Multi-agent conversations | Strong multi-agent abstractions | Heavier, more opinionated |
| CrewAI | Multi-agent with roles | Good for "team of specialists" pattern | Less low-level control |
| Custom (raw Python) | Specific patterns | Total control | You build everything |
- [ ] **For YOUR daily use:** smolagents for single-agent, LangGraph for production flows, custom for everything else

### Day 5 — Building Your First Useful Agent
- [ ] Use case: a "research summarizer" — searches your local notes, summarizes findings, writes a structured report
- [ ] Tools: search_notes (RAG), get_full_note (file read), draft_summary (LLM call), format_report (template)
- [ ] Pattern: plan → search → read → draft → review → finalize
- [ ] **Code:** Build it; run on a real question you have; iterate until acceptable

### 🔨 Saturday Project
- [ ] **AgentDesk v0** — start your Month 7 main project
  - [ ] Multi-tool agent with 5 tools: filesystem (read-only), web search (DuckDuckGo), calculator, datetime, your LocalRAG
  - [ ] Built on smolagents or LangGraph + Ollama (Qwen 2.5 7B Instruct)
  - [ ] CLI interface: `agentdesk "your task"`
  - [ ] Verbose mode showing each step
  - [ ] Trace log persisted to SQLite

### 📄 Sunday
- [ ] Read: smolagents docs (short, finishable in an evening)
- [ ] Read: LangGraph "Concepts" docs
- [ ] Watch: "Building Agents with Local Models" talks

---

## Week 27 — Tool Use Reliability & Error Handling

### Day 1 — Common Agent Failure Modes
- [ ] **Hallucinated tools:** "calls a tool that doesn't exist"
- [ ] **Wrong args:** "tool exists but model passed garbage"
- [ ] **Infinite loops:** "agent keeps calling search forever"
- [ ] **Premature finish:** "agent stops before task done"
- [ ] **Tool result ignored:** "model proceeds as if tool failed"
- [ ] **Refusal cascade:** "model refuses + can't recover"
- [ ] **Code:** Reproduce each on purpose with a flaky agent; document what triggers each

### Day 2 — Defensive Patterns
- [ ] **Schema enforcement:** outlines / instructor (already in Week 25)
- [ ] **Tool budget:** max N tool calls; if hit, summarize and exit
- [ ] **Time budget:** max T seconds; force finalize
- [ ] **Retry with feedback:** if tool returns error, retry with error message in prompt
- [ ] **Result validation:** after tool returns, ask LLM if result actually answered the sub-question
- [ ] **Code:** Add all defenses to AgentDesk; observe behavior change on the failure cases

### Day 3 — Self-Critique & Repair
- [ ] After generating a tool call, add a "verify" step: "Is this tool call correct? If not, fix it."
- [ ] After final answer, add a "review" step: "Does this answer fully address the user's question?"
- [ ] Trade-off: more LLM calls = slower + more compute. Apply selectively
- [ ] **Code:** Add a critique step to AgentDesk; measure success rate change

### Day 4 — Confidence & Refusal Handling
- [ ] If model's tool-call probability is low, log it (sample multiple times, see variance)
- [ ] If model wants to refuse but task is benign: explicit system prompt allowing the task
- [ ] If genuinely unsafe: refuse cleanly, don't pretend
- [ ] **Code:** Sample 3 generations of a tool call; if they disagree, surface uncertainty to user

### Day 5 — Logging Best Practices for Agents
- [ ] Log every: LLM call (prompt + completion + latency + tokens), tool call (name + args + result + latency), errors
- [ ] Structure: JSON lines, one per event, indexed by trace_id
- [ ] Storage: SQLite for personal, Postgres if you scale
- [ ] **Code:** Build a structured logging layer for AgentDesk

### 🔨 Saturday Project
- [ ] **ToolKit v1** — Project B for Month 7
  - [ ] Library that wraps Ollama + your favorite agent framework
  - [ ] Provides: schema enforcement, retry-with-feedback, budgets, validation
  - [ ] Models supported: Qwen 2.5 Instruct, Llama 3.1 Instruct, Hermes 3
  - [ ] Reliability benchmark: 30 tool-use tasks across 3 models, before/after ToolKit
  - [ ] **Publish as pip-installable**

### 📄 Sunday
- [ ] Read: "Reliable Agents at Production Scale" blog posts (search for case studies)
- [ ] Read: Anthropic's "Effective Agents" engineering blog
- [ ] Reflect: what failure modes did you hit that I didn't list?

---

## Week 28 — Local Code-Executing Agents

### Day 1 — Why Code Execution Is Powerful
- [ ] Code = general-purpose tool. One tool replaces 100 specific tools
- [ ] But: risk. Malicious or buggy code can damage your system
- [ ] Sandbox before execution
- [ ] Local code agents replace AutoGPT, Open Interpreter for sensitive workflows

### Day 2 — Sandboxing Options
- [ ] **Docker container:** isolated FS, restricted network, easy to set up
- [ ] **Firejail (Linux):** lightweight sandbox using kernel namespaces
- [ ] **gVisor:** stronger isolation than Docker
- [ ] **E2B:** cloud sandbox-as-a-service (NOT local, but useful for prototyping)
- [ ] **Restricted Python:** RestrictedPython, but easily escaped
- [ ] **For YOUR use:** Docker with `--read-only --network none --memory 2g --cap-drop=ALL`
- [ ] **Code:** Build a sandbox runner: receives Python code, runs in Docker, returns stdout/stderr

### Day 3 — Open Interpreter (Study, Maybe Use)
- [ ] Open Interpreter = local code-executing agent, Ollama compatible
- [ ] Defaults to executing on host (DANGEROUS — don't do this on a machine that matters)
- [ ] Configure to use sandbox or use carefully
- [ ] **Code:** Try Open Interpreter on a non-critical machine / VM; observe what it tries to do

### Day 4 — Building Your Own Code-Executing Agent
- [ ] Loop: prompt → LLM → emit code → sandbox.run(code) → return stdout → LLM → emit more code or finalize
- [ ] Track state: variables persist across cells (like Jupyter)
- [ ] Limit code generation: max lines, no imports of dangerous libs, no shell calls
- [ ] **Code:** Build a Jupyter-like agent that solves a data analysis task on a CSV

### Day 5 — Code Agents for Real Tasks
- [ ] Use cases that benefit:
  - Data analysis ("what's the trend in this CSV?")
  - File operations ("rename all photos by date")
  - Math/symbolic computation
  - Generating + running tests
- [ ] Use cases to avoid:
  - System administration (use deterministic scripts)
  - Network operations (security risk)
- [ ] **Code:** Build a "personal data analyst" agent for a real CSV you have

### 🔨 Saturday Project
- [ ] **AgentDesk v1 with Code Execution**
  - [ ] Add Docker-sandboxed Python execution tool
  - [ ] 3 demo workflows: analyze CSV, summarize folder structure, refactor a small code snippet
  - [ ] Permission audit log: every code execution logged with diff of what changed
  - [ ] Polish: README, demo video, security disclaimers

### 📄 Sunday
- [ ] Read: Open Interpreter design docs
- [ ] Read: "Lessons from Building Code Agents" blog posts
- [ ] Watch: "Code Agents in Production" talks

---

## 🔄 Buffer Week (Month 7 Revision) Local AI
- [ ] Revise: function calling, structured output, defensive patterns
- [ ] Revise: smolagents and LangGraph
- [ ] Revise: sandbox patterns for code execution
- [ ] **Build Monthly Project A:** AgentDesk — Local Personal Agent (full version)
- [ ] **Build Monthly Project B:** ToolKit — Reliable Function Calling
- [ ] Push to GitHub
- [ ] Journal: "Where do my agents still fail? What patterns work consistently?"

---

## Week 29 — Multi-Agent Systems Locally (CrewAI, AutoGen)

### Day 1 — When Multi-Agent Helps (And When It Doesn't)
- [ ] **Helps when:** clear role specialization (researcher + writer + critic), tasks decompose naturally
- [ ] **Doesn't help when:** single agent + good prompts handles it (which is most cases)
- [ ] **Common trap:** building multi-agent for cool factor when single agent is simpler and works
- [ ] **Honest:** for personal use on 6GB GPU, you'll rarely truly need multi-agent. But it's good to know.

### Day 2 — VRAM-Aware Orchestration (Critical for 6GB)
- [ ] Two patterns on 6GB:
  - **Single model, multiple agents:** all agents share same Ollama-loaded model, different system prompts → cheapest
  - **Multi-model with swapping:** orchestrator unloads/loads models → slower but more specialized
- [ ] Use `OLLAMA_KEEP_ALIVE=0` to force unload between calls (saves VRAM)
- [ ] Use `OLLAMA_MAX_LOADED_MODELS=1` to prevent thrashing
- [ ] **Code:** Build a multi-agent crew where each agent uses a different model via swapping

### Day 3 — CrewAI
- [ ] Role-based: "researcher," "writer," "critic" each a class with goal + tools
- [ ] Process: Sequential, Hierarchical, Async
- [ ] Native Ollama support
- [ ] **Code:** Build a 3-agent crew for blog post writing: researcher → writer → editor

### Day 4 — AutoGen (Microsoft)
- [ ] Conversation-based: agents talk to each other in chat format
- [ ] GroupChat: multiple agents in a conversation with a manager
- [ ] Code execution baked in
- [ ] Pricier (more LLM calls per task) but more flexible
- [ ] **Code:** Same blog post task with AutoGen; compare to CrewAI

### Day 5 — When To Build From Scratch
- [ ] Both CrewAI and AutoGen are opinionated; you might prefer custom code
- [ ] Simple recipe: orchestrator function calls specialist functions, each backed by an LLM call
- [ ] No framework needed for 90% of multi-agent
- [ ] **Code:** Same blog post task in raw Python (~200 lines); decide which approach you prefer

### 🔨 Saturday Project
- [ ] **CrewLocal v0** — start Month 8 main project
  - [ ] 4 agent roles: researcher, writer, critic, finalizer
  - [ ] Token budget per task (cap at ~10k tokens of LLM time)
  - [ ] VRAM-aware: defaults to single-model with different prompts; can swap on demand
  - [ ] Demo: research + write 500-word blog post on a topic
  - [ ] Demo: code review for a small file

### 📄 Sunday
- [ ] Read: CrewAI docs (short, focused)
- [ ] Read: AutoGen tutorial
- [ ] Watch: "When Multi-Agent Helps and When It Doesn't" critical talks

---

## Week 30 — Agent Memory & Persistence (mem0, Letta)

### Day 1 — Why Agents Need Memory
- [ ] Context window is finite; long conversations exceed it
- [ ] Across sessions, agent forgets you
- [ ] Memory types:
  - **Short-term (in context):** what's in this conversation
  - **Episodic:** "what happened" — events, facts, conversations
  - **Semantic:** "what I know" — preferences, facts about the user
  - **Procedural:** "how I do things" — learned patterns, scripts
- [ ] **Code:** Sketch what you'd want a personal agent to remember about you

### Day 2 — mem0 (Easy Start)
- [ ] Python library that adds memory to any LLM
- [ ] Stores facts in vector DB, retrieves relevant at chat time
- [ ] Auto-extracts memories from conversations
- [ ] Local backend support (Chroma, Qdrant, sqlite-vec)
- [ ] **Code:** `pip install mem0ai`; build an agent that remembers facts across sessions

### Day 3 — Letta (More Structured)
- [ ] Letta (formerly MemGPT) — context window management for "infinite memory"
- [ ] Agent decides what to store, what to retrieve, what to swap in/out of context
- [ ] More principled than mem0; heavier
- [ ] **Code:** Try Letta with Ollama backend; compare to mem0

### Day 4 — Custom Memory Layer
- [ ] You don't need a library; here's the pattern:
  - Every turn, LLM extracts facts (`{"fact": ..., "topic": ..., "confidence": ...}`)
  - Store in SQLite + vector index
  - At new turn, retrieve top-K relevant facts, inject into system prompt
- [ ] Better control + less dependency
- [ ] **Code:** Build a custom memory layer for AgentDesk; ~150 lines of Python

### Day 5 — Memory Best Practices
- [ ] Time-stamped facts (some go stale)
- [ ] Conflict resolution (newer facts overwrite older when contradictory)
- [ ] Forgetting: explicitly delete stale memories
- [ ] Privacy: let user view + delete all stored memories
- [ ] **Code:** Add a "show me what you remember" command to your agent

### 🔨 Saturday Project
- [ ] **AgentDesk v2 with Memory**
  - [ ] Add long-term memory (episodic + semantic)
  - [ ] User can view, delete, export memories
  - [ ] Demo: agent remembers your preferences after 5 conversations
  - [ ] Compare with-memory vs without-memory on a task that needs context

### 📄 Sunday
- [ ] Read: mem0 paper / blog
- [ ] Read: MemGPT paper (Letta predecessor)
- [ ] Reflect: which memory pattern works for your use case?

---

## Week 31 — Agent Observability & Debugging

### Day 1 — Tracing What Agents Do
- [ ] Without traces, agents are black boxes
- [ ] Trace = sequence of events: LLM call, tool call, decision, error
- [ ] Each event has: timestamp, inputs, outputs, latency, cost (tokens)
- [ ] **Code:** Define a `Trace` and `Event` dataclass; instrument your agent

### Day 2 — Visualization
- [ ] Tree view: agent's thinking as a tree
- [ ] Timeline: when each event happened
- [ ] Graph: relationships between LLM calls and tools
- [ ] Tools: LangSmith (cloud), LangFuse (self-hosted, OSS!), Phoenix (Arize, OSS)
- [ ] **For YOUR use:** LangFuse (self-hosted, Postgres + UI in Docker)
- [ ] **Code:** Run LangFuse locally; instrument AgentDesk to send traces

### Day 3 — Failure Mode Classification
- [ ] After collecting 100 traces, label common failure patterns
- [ ] Train a tiny classifier (sklearn or a small LLM) to auto-label
- [ ] Failure types: tool hallucination, infinite loop, schema error, premature finish, refusal cascade, off-topic
- [ ] **Code:** Hand-label 50 traces; train a logistic regression classifier on bag-of-events features

### Day 4 — Replay & Debug
- [ ] Given a trace, replay the agent step by step
- [ ] Modify a step (e.g., different tool output) and re-run from there
- [ ] Useful for finding "what if?" — what would the agent have done with different data?
- [ ] **Code:** Add replay capability to AgentTrace

### Day 5 — Cost / Latency Dashboard
- [ ] Per task: total tokens, total time, # tool calls, # retries
- [ ] Per model: average latency, average tokens
- [ ] Per tool: usage count, failure rate
- [ ] Long-term: cost over time, performance trends
- [ ] **Code:** Build a Streamlit dashboard pulling from your trace SQLite

### 🔨 Saturday Project
- [ ] **AgentTrace v1** — Project B for Month 8
  - [ ] Decorator/middleware for smolagents, LangGraph, custom
  - [ ] LangFuse + custom UI for traces
  - [ ] Failure mode classifier with 10+ classes
  - [ ] Replay mode
  - [ ] Cost dashboard

### 📄 Sunday
- [ ] Read: LangFuse docs
- [ ] Read: "Debugging Agents in Production" blog posts
- [ ] Watch: Phoenix and LangSmith demos

---

## Week 32 — Browser & Desktop Automation Agents

### Day 1 — Browser Automation with Playwright
- [ ] Playwright = headless browser control via Python
- [ ] Agent generates: click X, fill Y, screenshot Z
- [ ] Use cases: scraping, form filling, testing, research
- [ ] **Code:** Install: `pip install playwright; playwright install`; basic script that opens a page

### Day 2 — Vision-Based Browser Agents
- [ ] Screenshot the page → VLM (Moondream2) describes it / extracts text
- [ ] LLM decides next action based on description
- [ ] More robust to changing page structure than CSS selectors
- [ ] Tools: browser-use, Skyvern (cloud), Magnitude
- [ ] **Code:** Build a vision-driven browser agent that completes a 3-step web task

### Day 3 — Desktop Automation
- [ ] PyAutoGUI: mouse/keyboard control
- [ ] pywinauto (Windows): window-aware automation
- [ ] AutoHotkey integration: trigger Python scripts from hotkeys
- [ ] **Code:** Build an agent that opens a specific app, types a query, captures result

### Day 4 — Safety & Limits for Automation Agents
- [ ] Sandbox: separate user account or VM
- [ ] Read-only mode by default; ask before destructive actions
- [ ] Confirmation prompts for: file deletion, sending messages, financial actions
- [ ] Timeout: kill the agent if running too long
- [ ] **Code:** Add a confirmation layer to your automation agent

### Day 5 — Realistic Personal Use Cases
- [ ] Research agent: opens 5 tabs, summarizes findings
- [ ] Inbox triage: classifies emails, drafts responses (you approve)
- [ ] Form-filler: fills repetitive forms with stored data
- [ ] Avoid: anything financial, password-related, irreversible
- [ ] **Code:** Pick ONE realistic personal task; build an automation agent for it

### 🔨 Saturday Project
- [ ] **Browser Research Agent**
  - [ ] Given a topic, opens DuckDuckGo, reads top 5 results, summarizes
  - [ ] Saves notes to local markdown file
  - [ ] Vision-based (Moondream2) for resilience to layout changes
  - [ ] Trace all actions to AgentTrace

### 📄 Sunday
- [ ] Read: browser-use repo / blog
- [ ] Read: PyAutoGUI examples
- [ ] Reflect: which automation will you actually use weekly?

---

### ✅ Phase 4 Completion Checklist (End of Month 8)
- [ ] Master function calling with local 7B models (Qwen 2.5, Llama 3.1)
- [ ] Use grammar-constrained generation (outlines/instructor) for guaranteed valid output
- [ ] Built defensive patterns for all common agent failure modes
- [ ] Built code-executing agents with proper sandboxing
- [ ] Built multi-agent crews with VRAM-aware orchestration
- [ ] Added long-term memory to agents
- [ ] Built local observability with LangFuse + custom dashboards
- [ ] Built browser/desktop automation agents safely
- [ ] **You can answer "should I use an agent here?" with rigor**
- [ ] Built AgentDesk + ToolKit (Month 7) + CrewLocal + AgentTrace (Month 8)

[⬆ Back to Table of Contents](#toc)

---

# ═══════════════════════════════════════════════════
# PHASE 5: MULTIMODAL & GENERATIVE (Weeks 33-40)
# Vision, voice, image generation — all on 6GB
# ═══════════════════════════════════════════════════

> By the end of this phase you'll have a local assistant that can SEE, HEAR, and SPEAK,
> plus you'll be generating images locally. Every modality, fully offline.

---

## 🔄 Buffer Week (Month 8 Revision) Local AI
- [ ] Revise: function calling, agent patterns, multi-agent, memory, observability
- [ ] Use your agents for a week of real tasks; document what works and what fails
- [ ] Make one PR or contribution to smolagents / LangGraph / similar (good warmup for Month 12)
- [ ] Read: papers on vision-language models to prepare for Month 9
- [ ] Journal: "I have a local AI brain. What I want next is local AI senses."

---

## Week 33 — Vision-Language Models Locally (LLaVA, Moondream, MiniCPM-V)

### Day 1 — How VLMs Work (Mental Model)
- [ ] Architecture: image encoder (CLIP ViT) → projector (MLP) → LLM (text decoder)
- [ ] The image becomes a series of "visual tokens" that the LLM consumes alongside text tokens
- [ ] Training: pair (image, caption) data + (image, instruction, answer) data
- [ ] Quality depends on: image encoder quality + LLM strength + projector training
- [ ] **Code:** Draw the architecture; identify which parts you can swap

### Day 2 — Local VLM Catalog for 6GB
- [ ] **Moondream2** ⭐ — 1.86B params, ~3-4GB VRAM, FAST, surprisingly capable
- [ ] **LLaVA 1.6 Mistral 7B Q4** — 4-5GB, slow but high quality
- [ ] **MiniCPM-V 2.6** — 8B but with efficient design, competitive
- [ ] **Phi-3.5 Vision** — 4.2B, good balance, Microsoft's small VLM
- [ ] **Qwen 2 VL 7B Q4** — strong, supports video, large variant
- [ ] **PaliGemma** — Google's small VLM, useful for fine-tuning
- [ ] **For YOUR daily use:** Moondream2 (speed) + LLaVA 1.6 7B Q4 (quality), switch based on task

### Day 3 — Running VLMs via Ollama
- [ ] `ollama pull moondream:1.8b`
- [ ] Send image:
  ```python
  ollama.chat(model='moondream', messages=[{
    'role':'user',
    'content':'What is in this image?',
    'images':['./photo.jpg']
  }])
  ```
- [ ] LLaVA, MiniCPM-V also available in Ollama
- [ ] **Code:** Test all 4 VLMs on 10 sample images (photos, charts, screenshots, documents); compare quality + speed

### Day 4 — Running VLMs via Transformers
- [ ] More control, more configuration, but heavier
- [ ] Useful when Ollama doesn't have your preferred VLM
- [ ] ```python
  from transformers import AutoModel, AutoTokenizer
  model = AutoModel.from_pretrained('vikhyatk/moondream2', trust_remote_code=True, torch_dtype=torch.float16).cuda()
  ```
- [ ] **Code:** Run Moondream2 via Transformers; compare to Ollama wrapper performance

### Day 5 — Vision Tasks & Prompts
- [ ] **Describe image:** "What is in this image? Be detailed."
- [ ] **Extract text (OCR-ish):** "Transcribe all text in this image."
- [ ] **Compare images:** "How do these two images differ?" (needs multi-image VLM like Qwen 2 VL)
- [ ] **Visual reasoning:** "If the temperature is shown as 25°C and it's raining, should I take an umbrella?"
- [ ] **Chart analysis:** "What does this chart show? What's the trend?"
- [ ] **Document QA:** "What's the total invoice amount?"
- [ ] **Code:** Build a prompt library for vision tasks

### 🔨 Saturday Project
- [ ] **VisionAssist v0** — start Month 9 main project
  - [ ] Streamlit UI: drag-image, ask question
  - [ ] 3 model options: Moondream2 (fast), LLaVA 1.6 7B (quality), Phi-3.5 Vision (balanced)
  - [ ] Auto-router: classifies image as photo/chart/document, picks model
  - [ ] Batch mode: process a folder, output CSV of descriptions
  - [ ] Alt-text generator mode for accessibility

### 📄 Sunday
- [ ] Read: LLaVA paper (Liu et al., 2023)
- [ ] Read: Moondream2 announcement and design (Vik Khedkar's blog)
- [ ] Compare: Open VLM Leaderboard on HuggingFace

---

## Week 34 — OCR + Vision-LLM Pipelines

### Day 1 — When OCR Beats VLM (and Vice Versa)
- [ ] **OCR wins:** text-heavy docs, exact transcription, multilingual scripts, scanned PDFs
- [ ] **VLM wins:** understanding context, summarizing visuals, answering questions
- [ ] Best pattern: **OCR + VLM combined** — OCR extracts exact text, VLM provides context/understanding
- [ ] **Code:** Process a complex document with both; compare results

### Day 2 — Tesseract Mastery
- [ ] Install: `tesseract-ocr` on Linux, installer on Windows
- [ ] Python wrapper: `pytesseract`
- [ ] Languages: download data files for Hindi (`hin`), Marathi (`mar`), Telugu (`tel`), etc.
- [ ] Config tuning: page segmentation mode, OCR engine mode
- [ ] **Code:** OCR a multi-lingual PDF; tune for best accuracy

### Day 3 — PaddleOCR (Higher Quality)
- [ ] Baidu's, generally more accurate than Tesseract
- [ ] Excellent multilingual support including Indic scripts
- [ ] Has table recognition and structure extraction
- [ ] Slower setup but better results
- [ ] **Code:** Install PaddleOCR, OCR same documents; compare to Tesseract

### Day 4 — Document Understanding Pipelines
- [ ] Marker / Docling (revisit from Week 12)
- [ ] These pipelines combine OCR + VLM + layout analysis
- [ ] Output: clean Markdown with tables, equations, figure references
- [ ] **Code:** Run Marker on a scientific paper; observe output quality

### Day 5 — Building Your Own Pipeline
- [ ] Stage 1: Layout detection (DocLayNet model, identifies regions)
- [ ] Stage 2: OCR per region (Tesseract for text, table-specific OCR for tables)
- [ ] Stage 3: VLM captions for figures/charts
- [ ] Stage 4: Markdown assembly
- [ ] **Code:** Build a custom pipeline for YOUR document type (textbook page, research paper, invoice)

### 🔨 Saturday Project
- [ ] **Document Understanding Pipeline**
  - [ ] Choose a document type relevant to you (e.g., textbook, research paper, invoice)
  - [ ] Build the pipeline: layout → OCR → VLM → assembly
  - [ ] Test on 5 documents; measure: char accuracy, table accuracy, figure description quality
  - [ ] Compare to Marker / Docling baseline

### 📄 Sunday
- [ ] Read: DocLayNet paper (IBM)
- [ ] Read: PaddleOCR documentation
- [ ] Watch: "Modern OCR with VLMs" talks

---

## Week 35 — Local Speech-to-Text (Whisper/faster-whisper/distil-whisper)

### Day 1 — Whisper Model Family
- [ ] OpenAI's Whisper: open source ASR model family
- [ ] Sizes: tiny (39M), base (74M), small (244M), medium (769M), large-v3 (1550M)
- [ ] Multilingual versions and English-only (`.en` suffix; smaller, faster, English only)
- [ ] **Distil-Whisper:** distilled, ~50% smaller, 6× faster, slight accuracy drop
- [ ] **For YOUR daily use:** `distil-small.en` for English, `large-v3` for multilingual
- [ ] **Code:** Run all sizes on a 1-minute test audio; measure accuracy + speed

### Day 2 — faster-whisper (CTranslate2)
- [ ] `faster-whisper` library uses CTranslate2 for inference
- [ ] 4-8× faster than vanilla Whisper, lower memory
- [ ] INT8 quantization for even faster CPU inference
- [ ] ```python
  from faster_whisper import WhisperModel
  model = WhisperModel("distil-small.en", compute_type="int8")
  segments, info = model.transcribe("audio.mp3")
  ```
- [ ] **Code:** Install, transcribe a 5-minute audio file; measure RTF (real-time factor)

### Day 3 — Whisper.cpp (Pure C++)
- [ ] GGML-based, runs Whisper without Python
- [ ] Apple Neural Engine support, CUDA support, very fast on CPU
- [ ] Useful for embedding in apps (Electron, mobile)
- [ ] **Code:** Build whisper.cpp, run from CLI; observe performance

### Day 4 — Real-Time Transcription
- [ ] Stream audio from mic → chunk → transcribe → emit
- [ ] Latency target: < 2 seconds per chunk
- [ ] VAD (Voice Activity Detection): only transcribe when someone's speaking
- [ ] Libraries: pyaudio + WebRTC VAD + faster-whisper
- [ ] **Code:** Build a real-time terminal transcriber

### Day 5 — Speaker Diarization
- [ ] Who said what? — separate speakers in a multi-person recording
- [ ] **pyannote.audio:** state-of-the-art open source diarization
- [ ] **whisperX:** combines Whisper + diarization + word-level timestamps
- [ ] **Code:** Diarize a podcast snippet; output "Speaker A: ..., Speaker B: ..."

### 🔨 Saturday Project
- [ ] **Meeting Transcript Tool**
  - [ ] Records local audio (Audacity or `ffmpeg` from mic)
  - [ ] Transcribes with faster-whisper or whisperX
  - [ ] Diarizes (if multi-speaker)
  - [ ] Summarizes with local LLM
  - [ ] Outputs markdown: transcript + summary + action items
  - [ ] Process: voice memo → useful notes in 60 seconds

### 📄 Sunday
- [ ] Read: Whisper paper (Radford et al., 2022)
- [ ] Read: distil-whisper announcement
- [ ] Watch: Whisper architecture explainers

---

## Week 36 — Local Text-to-Speech (Piper, Coqui, XTTS)

### Day 1 — Local TTS Landscape
- [ ] **Piper** ⭐ — lightweight, low latency, decent voices, 30+ languages
- [ ] **Coqui TTS / XTTS v2** — higher quality, voice cloning capability, slower
- [ ] **Bark** — emotional speech, music, sound effects, but slower and larger
- [ ] **OpenVoice (MyShell)** — voice cloning, multilingual
- [ ] **Tortoise TTS** — very high quality, very slow
- [ ] **For YOUR daily use:** Piper for general TTS, XTTS for voice cloning if needed

### Day 2 — Piper Mastery
- [ ] `pip install piper-tts` (Python wrapper) or use the binary directly
- [ ] Download voices: https://huggingface.co/rhasspy/piper-voices
- [ ] ```python
  echo "Hello world" | piper --model en_US-amy-medium.onnx --output_file out.wav
  ```
- [ ] Available voices: dozens of languages including Hindi, Marathi, Telugu
- [ ] **Code:** Install, generate audio in 3 languages; observe latency (<200ms typically)

### Day 3 — XTTS for Voice Cloning
- [ ] Coqui's XTTS v2: clone a voice from a 6-second sample
- [ ] Higher quality than Piper, but ~3-5× slower
- [ ] ```python
  from TTS.api import TTS
  tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")
  tts.tts_to_file(text="...", speaker_wav="sample.wav", language="en", file_path="out.wav")
  ```
- [ ] **Code:** Clone your own voice from a 6-second sample; play around with it ethically

### Day 4 — Streaming TTS for Conversations
- [ ] Don't wait for full sentence: start speaking after first phrase
- [ ] Strategy: chunk LLM output by punctuation, send each chunk to TTS, queue audio
- [ ] Piper supports streaming natively
- [ ] **Code:** Build a streaming-LLM + streaming-TTS pipeline

### Day 5 — TTS Quality Tradeoffs
- [ ] For conversational AI: latency is king. Piper at ~150ms TTFT wins.
- [ ] For long-form (audiobook, podcast): quality matters more. XTTS or Bark.
- [ ] For language coverage: Piper > Coqui > Bark
- [ ] **Code:** Build a TTS "router" that picks based on use case

### 🔨 Saturday Project
- [ ] **Local TTS Sandbox**
  - [ ] UI: type text, pick voice/language, generate audio
  - [ ] Includes: Piper (5 voices) + XTTS (your cloned voice optional)
  - [ ] Streaming demo
  - [ ] Save as part of your local stack — you'll use this in VoiceMate

### 📄 Sunday
- [ ] Read: Piper README and design
- [ ] Read: XTTS paper
- [ ] Listen: compare voice quality between systems

---

## 🔄 Buffer Week (Month 9 Revision) Local AI
- [ ] Revise: vision (Moondream, LLaVA), OCR, document pipelines
- [ ] Revise: Whisper variants and streaming patterns
- [ ] Revise: TTS systems
- [ ] **Build Monthly Project A:** VisionAssist — Local Multimodal AI
- [ ] **Build Monthly Project B:** ScreenSage — Screenshot Q&A
- [ ] Push to GitHub with demo videos
- [ ] Journal: "Local AI now sees. Next: it speaks and creates."

---

## Week 37 — Voice Assistants End-to-End

### Day 1 — Architecture of an Offline Voice Assistant
- [ ] Pipeline: **Wake word → ASR → LLM → TTS**
- [ ] Each stage has latency budget:
  - Wake word: < 100ms
  - ASR: < 1s for typical query
  - LLM TTFT: < 3s
  - TTS streaming: < 200ms to first audio
  - **Total end-to-end: < 5s for a snappy assistant**
- [ ] **Code:** Sketch your pipeline with latency budgets

### Day 2 — Wake Word Detection
- [ ] **openwakeword** — open source, custom wake words, free
- [ ] **Porcupine (Picovoice)** — proprietary but generous free tier
- [ ] **Snowboy** — discontinued but forks exist
- [ ] **For YOUR use:** openwakeword (fully open, custom training possible)
- [ ] **Code:** Install openwakeword; train or use a pre-trained "hey jarvis" model

### Day 3 — Putting It All Together
- [ ] Loop:
  1. Listen for wake word (always-on, low CPU)
  2. On detection: record audio until silence (VAD)
  3. Transcribe with faster-whisper
  4. Generate response with Ollama
  5. Stream to Piper TTS
  6. Play audio
  7. Return to step 1
- [ ] **Code:** ~300-400 lines of Python; functioning voice assistant

### Day 4 — Tool Use in Voice Assistants
- [ ] Tools: timer, calendar (local ICS), weather (cached), smart home (Home Assistant API), Wikipedia (local dump), math
- [ ] Function calling + voice = magic
- [ ] **Code:** Add 5 tools; demo: "set a 10 minute timer," "what's on my calendar today," "convert 100 USD to INR"

### Day 5 — Latency Optimization
- [ ] Keep Ollama warm (`OLLAMA_KEEP_ALIVE=-1`)
- [ ] Pre-warm Whisper (run a dummy transcription at startup)
- [ ] Pre-warm Piper voice
- [ ] Streaming everything where possible
- [ ] CPU pinning for ASR; GPU for LLM
- [ ] **Code:** Measure end-to-end latency at each stage; optimize the slowest one

### 🔨 Saturday Project
- [ ] **VoiceMate v1** — start Month 10 main project
  - [ ] Full pipeline: openwakeword + faster-whisper + Llama 3.1 8B + Piper
  - [ ] 5 tools integrated
  - [ ] Measured latency report
  - [ ] Runs as background process
  - [ ] Demo video: realistic conversation

### 📄 Sunday
- [ ] Read: Home Assistant + voice integration docs
- [ ] Watch: "Building Local Voice Assistants" videos
- [ ] Test: Mycroft (open source voice assistant) for inspiration

---

## Week 38 — Stable Diffusion Locally (ComfyUI, A1111, Fooocus)

### Day 1 — SD Model Family
- [ ] **SD 1.5** — original, lightest (~3-4GB), still useful
- [ ] **SD 2.x** — improved, less popular than 1.5
- [ ] **SDXL** — bigger (~6.5GB), much better quality
- [ ] **SDXL Lightning / Turbo** — 1-8 step generation, fast
- [ ] **SD 3.5** — newer, more complex, mixed reception
- [ ] **Flux.1 [schnell/dev]** — current state of the art OSS, ~12GB FP8, hard on 6GB
- [ ] **PixArt-Σ** — efficient alternative
- [ ] **For 6GB VRAM:** SD 1.5 (always fits), SDXL with tiling and CPU offload (slow but works), SDXL Lightning (sweet spot)

### Day 2 — ComfyUI (Power User) ⭐
- [ ] Node-based workflow editor, very flexible
- [ ] Built-in low-VRAM optimizations
- [ ] Massive plugin ecosystem
- [ ] **Code:** Clone https://github.com/comfyanonymous/ComfyUI; run; download SD 1.5 base model

### Day 3 — Automatic1111 (Beginner Friendly)
- [ ] Web UI for SD, very feature-rich
- [ ] Extensions, embeddings, LoRAs, ControlNet, etc.
- [ ] Less flexible than ComfyUI but easier to start
- [ ] **Code:** Set up A1111; generate first image

### Day 4 — Fooocus (Easiest)
- [ ] SDXL with great defaults; minimal UI
- [ ] Built-in optimizations for low VRAM
- [ ] For "I just want good images, no tinkering"
- [ ] **Code:** Try Fooocus; compare ease vs ComfyUI

### Day 5 — Low VRAM Tricks
- [ ] `--medvram` / `--lowvram` flags in A1111
- [ ] VAE tiling: process image in tiles to save memory
- [ ] Sequential CPU offload: load layers from CPU as needed (slower)
- [ ] FP8 weights (newer): smaller, fast on Ada+ GPUs
- [ ] SDXL Lightning / Turbo: fewer denoising steps (4-8 vs 30)
- [ ] **Code:** Get SDXL running on YOUR 6GB GPU with reasonable speed (target: <30s per image)

### 🔨 Saturday Project
- [ ] **ImageForge v0** — start Month 10 main project
  - [ ] ComfyUI setup with 5 low-VRAM workflows
  - [ ] Workflows for: blog hero image, profile pic, illustration, logo, photo-realistic
  - [ ] Document each: prompt template, model, sampler, steps, expected time

### 📄 Sunday
- [ ] Read: "How Stable Diffusion Works" articles (the math)
- [ ] Watch: tutorials on ComfyUI workflow building
- [ ] Browse: civitai for community models and LoRAs

---

## Week 39 — SD Fine-Tuning (DreamBooth, LoRA, Textual Inversion)

### Day 1 — Three Ways to Customize SD
- [ ] **Textual Inversion:** train a "new word" in the text encoder. Tiny (~5KB), limited capacity.
- [ ] **LoRA:** low-rank adaptation, ~10-200MB, good for styles/subjects, fast to train
- [ ] **DreamBooth:** full fine-tune for a specific subject, larger (~2-7GB), best fidelity, slower

### Day 2 — Training SD LoRAs Locally
- [ ] Tools: Kohya_ss (Web UI for training), OneTrainer, ai-toolkit
- [ ] Dataset: 10-30 images of subject, captions for each
- [ ] Time on 6GB: ~30-60 min for SDXL LoRA
- [ ] **Code:** Train your first LoRA on your own face or a style you like (~20 images)

### Day 3 — Caption Engineering for SD Training
- [ ] Trigger word: rare token, e.g., "ohwx" or "<my-style>"
- [ ] Caption everything else that's NOT the subject; that's what the model "shouldn't learn"
- [ ] Auto-caption tools: BLIP-2, CLIP-Interrogator, GPT-4V (cloud)
- [ ] Local: use Moondream2 to caption your training images
- [ ] **Code:** Auto-caption 20 training images, then manually clean

### Day 4 — Testing Your LoRA
- [ ] Load LoRA in ComfyUI or A1111
- [ ] Generate with trigger word + various prompts
- [ ] Check: subject fidelity, prompt adherence, can do new things with subject (not just memorized)
- [ ] **Code:** Generate 20 images with your LoRA; rate them

### Day 5 — Combining LoRAs and Embeddings
- [ ] Stack 2-3 LoRAs (subject + style + concept) with adjustable weights
- [ ] Risk: too many LoRAs degrades quality; experiment
- [ ] **Code:** Try combining your subject LoRA with a community style LoRA

### 🔨 Saturday Project
- [ ] **Your First Custom SD LoRA**
  - [ ] Pick a subject (yourself, a pet, a style)
  - [ ] Curate ~20 images
  - [ ] Train SDXL LoRA with Kohya_ss
  - [ ] Generate test set
  - [ ] Document: dataset, config, results, LoRA file

### 📄 Sunday
- [ ] Read: Kohya_ss documentation
- [ ] Read: "LoRA for SD" guides on civitai
- [ ] Watch: tutorials from Aitrepreneur, Olivio Sarikas

---

## Week 40 — Local Video & Audio Generation

### Day 1 — Video Generation (State of Local Art)
- [ ] **AnimateDiff:** SD + motion module, animations from prompts
- [ ] **CogVideoX:** local video gen, 2B and 5B variants (the 2B fits 6GB with care)
- [ ] **LTX-Video:** newer, fast, 8B params
- [ ] **Mochi 1, HunyuanVideo:** higher quality, need more VRAM
- [ ] **Reality:** local video gen is HARD on 6GB. Most needs >12GB. Lower your expectations.
- [ ] **Code:** Try AnimateDiff in ComfyUI; produce a 2-second animation

### Day 2 — Audio / Music Generation
- [ ] **Bark (Suno AI):** voice + music + sound effects, ~3GB VRAM, supports laughter, music
- [ ] **MusicGen (Meta):** music from text, 300M / 1.5B / 3.3B variants
- [ ] **AudioCraft (Meta):** umbrella of MusicGen + AudioGen + EnCodec
- [ ] **Code:** Generate a 30-second music clip from text using MusicGen 1.5B

### Day 3 — Realistic Use Cases for Local Generative
- [ ] **Blog/website assets:** illustrations, hero images (✅ practical with SDXL)
- [ ] **Background music for content:** MusicGen (✅ practical)
- [ ] **Avatar / profile pics:** LoRA of yourself (✅ very practical)
- [ ] **Animations:** experimentation only (⚠ slow, quality limited)
- [ ] **Full video production:** wait for hardware or use cloud (❌ not practical)

### Day 4 — Workflow Tools for Creators
- [ ] Krita + diffusers plugin: local SD inside paint software
- [ ] InvokeAI: pro-grade SD app
- [ ] ComfyUI for repeatable production workflows (your existing setup)
- [ ] **Code:** Pick one creator workflow you'll actually do; build a 1-click pipeline

### Day 5 — Ethics & Watermarking
- [ ] Generated content can be misleading — be honest
- [ ] Embed metadata: model, prompt, date in EXIF or PNG metadata
- [ ] C2PA (Content Provenance) standard — emerging
- [ ] **Code:** Add metadata embedding to ImageForge

### 🔨 Saturday Project
- [ ] **ImageForge v1 (Polished)** — Month 10 main project
  - [ ] 10 curated ComfyUI workflows
  - [ ] 2-3 self-trained LoRAs
  - [ ] Batch generation with prompt variations
  - [ ] Metadata embedding
  - [ ] Publish workflows + LoRAs

### 📄 Sunday
- [ ] Read: AnimateDiff paper
- [ ] Read: MusicGen paper (Copet et al., 2023)
- [ ] Reflect: where does local generative AI add real value in your workflow?

---

### ✅ Phase 5 Completion Checklist (End of Month 10)
- [ ] Run VLMs (Moondream, LLaVA, MiniCPM-V) for diverse vision tasks
- [ ] Build OCR + VLM pipelines for documents
- [ ] Run Whisper variants for real-time transcription
- [ ] Use Piper / XTTS for low-latency TTS
- [ ] Built a complete offline voice assistant (<5s end-to-end)
- [ ] Run Stable Diffusion (SDXL Lightning, SD 1.5) on 6GB
- [ ] Train custom SD LoRAs for subjects / styles
- [ ] Built VisionAssist + ScreenSage (Month 9) + VoiceMate + ImageForge (Month 10)
- [ ] **Your local AI now: sees, hears, speaks, and creates**

[⬆ Back to Table of Contents](#toc)

---

# ═══════════════════════════════════════════════════════════
# PHASE 6: EDGE, PRODUCTION & MASTERY (Weeks 41-52, Months 11-12)
# Home server, NPU, mobile, magnum opus, open source ⭐⭐
# ═══════════════════════════════════════════════════════════

> The final phase. Turn your local AI skills into a flagship product,
> deploy beyond your laptop, and contribute to the open source projects
> you've been using. End the year with a portfolio that speaks for itself.

---

## 🔄 Buffer Week (Month 10 Revision) Local AI
- [ ] Use your local stack (chat + voice + vision + RAG + agents) for a week of real work
- [ ] Identify the 3 things you'd most want to fix or expand — these become Magnum Opus seeds
- [ ] Read: edge deployment guides, NPU docs, MLC-LLM articles
- [ ] Journal: "If I could only run AI on a smaller device, what would I run? Why?"

---

## Week 41 — Home Server Architecture for Local AI

### Day 1 — Hardware Options for a Home AI Server
- [ ] **Your existing laptop (headless mode):** cheapest, but inconvenient to leave on
- [ ] **Old desktop with GPU:** great if you have one; 16+GB RAM, any CUDA GPU
- [ ] **Mini-PC (e.g., Beelink SER, Minisforum):** small, quiet, decent CPU+iGPU, can add eGPU
- [ ] **Used workstation (eBay/local):** Dell/HP with Xeon, used GPU like RTX 3060 (12GB), <$500 total
- [ ] **Mac Mini (M2/M3/M4):** unified memory = can run big models, expensive
- [ ] **NAS with GPU (Synology, Asustor, Unraid):** combines storage + AI
- [ ] **For your situation:** an old PC with a used RTX 3060 12GB is the price/performance sweet spot
- [ ] **Code:** Spec out 2 options at different budgets; document tradeoffs

### Day 2 — OS Choice
- [ ] **Linux (Ubuntu Server 24.04 LTS):** standard, well-supported, lightweight, free
- [ ] **Proxmox:** virtualization layer; run multiple VMs (one for AI, one for media, etc.)
- [ ] **Unraid:** great for combined storage + apps server
- [ ] **TrueNAS Scale:** ZFS + apps
- [ ] **Windows Server:** if you must (Hyper-V, but more overhead)
- [ ] **For your purposes:** Ubuntu Server 24.04 + Docker Compose is the simplest robust option

### Day 3 — Docker Compose Stack for AI Server
- [ ] One `docker-compose.yml` orchestrating all services
- [ ] Services: Ollama, Open WebUI, your LocalRAG, ImageForge (ComfyUI), AgentDesk, monitoring
- [ ] Volumes for persistence (models, conversations, databases)
- [ ] Network: internal Docker network + reverse proxy
- [ ] **Code:** Write a complete `docker-compose.yml` for your home AI stack

### Day 4 — Reverse Proxy & Local Domain
- [ ] **Caddy** ⭐ — automatic HTTPS, simple config, great for self-hosting
- [ ] **Traefik** — more flexible, more complex
- [ ] **Nginx** — traditional choice
- [ ] mDNS hostname (e.g., `homeai.local`) via avahi/bonjour
- [ ] Or use Pi-hole as local DNS to alias `homeai.local` to your server IP
- [ ] **Code:** Set up Caddy serving `chat.homeai.local`, `rag.homeai.local`, `image.homeai.local`

### Day 5 — Monitoring Your Server
- [ ] Prometheus + Grafana (overkill for home? Maybe. But satisfying)
- [ ] Lightweight: Netdata, Beszel, or just `htop` + `nvidia-smi`
- [ ] Alerts when: disk full, model crashes, response times degrade
- [ ] **Code:** Set up basic monitoring; configure one alert (e.g., disk > 90%)

### 🔨 Saturday Project
- [ ] **HomeAI v0** — start Month 11 main project
  - [ ] Docker Compose with Ollama + Open WebUI + LocalRAG + ImageForge + AgentDesk
  - [ ] Caddy reverse proxy with HTTPS (self-signed for LAN)
  - [ ] Monitoring with Netdata
  - [ ] One-command deploy: `docker compose up -d`

### 📄 Sunday
- [ ] Read: self-hosting subreddit highlights (r/selfhosted)
- [ ] Read: TrueNAS / Unraid getting-started guides
- [ ] Watch: home lab YouTube channels (Christian Lempa, Techno Tim)

---

## Week 42 — NPU & Intel AI Boost (OpenVINO, DirectML)

### Day 1 — Your Intel AI Boost NPU
- [ ] Intel Core Ultra processors include NPU ("Neural Processing Unit") aka AI Boost
- [ ] ~10-13 TOPS at 5-10 watts (efficient!)
- [ ] Designed for sustained AI inference at low power
- [ ] Use cases: background AI tasks while saving GPU for foreground heavy lifting
- [ ] **Code:** Verify your NPU is detected: `Task Manager → Performance → NPU 0`

### Day 2 — OpenVINO Runtime
- [ ] Intel's inference framework, optimized for Intel CPU + GPU + NPU
- [ ] Supports: LLMs, embedding models, vision models, classical ML
- [ ] Convert models from HuggingFace / PyTorch / TensorFlow → OpenVINO IR
- [ ] **Code:** Install `openvino-genai`, run a small LLM on the NPU

### Day 3 — Running LLMs on NPU
- [ ] Best for: small LLMs (3B and under, INT4) and embedding models
- [ ] Why? NPU memory is limited; large LLMs don't fit
- [ ] Example: Phi-3 Mini INT4 on NPU = ~10 tok/s at ~5W (vs GPU at higher power)
- [ ] OpenVINO has pre-converted models on HuggingFace ("OpenVINO" org)
- [ ] **Code:** Run Phi-3 Mini on NPU; compare to running on GPU; observe power difference

### Day 4 — DirectML (Windows-First)
- [ ] Microsoft's DirectX-based AI runtime; uses any DirectX-12 GPU
- [ ] Supports your Intel Arc Pro AND NVIDIA GPU
- [ ] ONNX Runtime + DirectML execution provider
- [ ] Useful for: deploying to other Windows machines without CUDA
- [ ] **Code:** Run a small model via ONNX Runtime + DirectML on the Intel iGPU

### Day 5 — NPU/GPU Workload Splitting
- [ ] Idea: NPU for background tasks (embedding model for RAG, wake-word, small LLM)
- [ ] GPU for foreground heavy LLM
- [ ] Result: smoother user experience, less battery drain
- [ ] **Code:** Design a workflow that uses both; measure power use over an hour

### 🔨 Saturday Project
- [ ] **NPU Showcase**
  - [ ] Run nomic-embed-text or Phi-3 Mini on Intel AI Boost NPU
  - [ ] Build a "RAG with NPU embeddings + GPU LLM" demo
  - [ ] Benchmark: power use, latency, quality vs all-GPU
  - [ ] Document for the community (NPUs are underused, your benchmarks help)

### 📄 Sunday
- [ ] Read: OpenVINO docs on LLM inference
- [ ] Read: Intel AI Boost developer guides
- [ ] Watch: NPU vs GPU vs CPU benchmarks on YouTube

---

## Week 43 — Mobile Inference (MLC-LLM, llama.cpp Mobile)

### Day 1 — Why Run LLMs on Phones
- [ ] Total privacy (data never leaves device)
- [ ] Offline (airplane mode works)
- [ ] No API costs
- [ ] Limits: smaller models (1-3B), slower than your laptop
- [ ] **Reality check:** A modern phone (Snapdragon 8 Gen 3, Apple A17 Pro) runs 3B models at ~10-15 tok/s

### Day 2 — MLC-LLM
- [ ] Compiles models to TVM kernels for any device (Android, iOS, WebGPU, ROCm, CUDA, Metal)
- [ ] Best-in-class on Android (uses Vulkan or OpenCL)
- [ ] Apps: MLC Chat on Play Store / TestFlight
- [ ] **Code:** Install MLC Chat on your Android phone; download Phi-3 Mini; chat

### Day 3 — llama.cpp on Android
- [ ] Termux + llama.cpp build for ARM NEON
- [ ] CLI-based, no GUI by default
- [ ] Slower than MLC but more flexible
- [ ] **Code:** Compile llama.cpp in Termux; run a Q4_K_M 1B model

### Day 4 — iOS Local LLM Apps
- [ ] **Private LLM** (paid) — best polish
- [ ] **MLC Chat for iOS** — open source, free
- [ ] **Enchanted (Mac/iOS)** — connects to your home Ollama server (not purely on-device but private)
- [ ] **Code:** If you have iPhone, try one of these; document experience

### Day 5 — Mobile-Optimized Models
- [ ] **Phi-3 Mini 3.8B (INT4):** ~2GB on phone, excellent quality
- [ ] **Llama 3.2 1B/3B:** designed for edge deployment
- [ ] **Gemma 2 2B:** another solid choice
- [ ] **Qwen 2.5 0.5B/1.5B:** smallest credible quality
- [ ] **MobiLlama:** specifically designed for mobile
- [ ] **Code:** Pick a model for mobile use; deploy via MLC-LLM with your fine-tune (if applicable)

### 🔨 Saturday Project
- [ ] **Mobile Deployment of Your Fine-Tuned Model**
  - [ ] Take your DomainDoc (Month 6) or a smaller fine-tune
  - [ ] Convert to MLC-LLM format
  - [ ] Deploy to your phone via MLC Chat
  - [ ] Demo video: your own model in your pocket

### 📄 Sunday
- [ ] Read: MLC-LLM documentation
- [ ] Watch: "Running LLMs on Phones" demos
- [ ] Reflect: what local-AI use case becomes possible because of phone deployment?

---

## Week 44 — Jetson, Raspberry Pi & Edge Devices

### Day 1 — NVIDIA Jetson Family (If You Have / Want One)
- [ ] **Jetson Orin Nano (8GB)** — ~$500, ~6 tok/s on 7B Q4
- [ ] **Jetson Orin NX (16GB)** — ~$1000, runs bigger models
- [ ] **Jetson AGX Orin (64GB)** — ~$2000, datacenter-class for edge
- [ ] CUDA + Tensor Cores in tiny package
- [ ] Great for robotics, smart cameras, edge AI products
- [ ] **Don't buy unless you have a specific use case** — your laptop already runs everything Orin Nano can

### Day 2 — Raspberry Pi 5 (Most Accessible Edge)
- [ ] Pi 5 8GB: ~$80, ~3-5 tok/s on 3B Q4_K_M, ~1-2 tok/s on 7B Q4
- [ ] No GPU acceleration for LLMs (yet, work in progress with Vulkan)
- [ ] Power efficient (~5-8W under load)
- [ ] Great for always-on local AI services
- [ ] **Code:** If you have one, install Ubuntu Server, llama.cpp, run a 1B-3B model

### Day 3 — Mini-PCs for Edge
- [ ] **Beelink SER series, Minisforum UM/HM/UH series, GMKtec, Khadas, ASUS NUC**
- [ ] 16-32GB RAM, 6-12 CPU cores, integrated GPU
- [ ] Run 7-13B GGUF Q4 entirely on CPU at 4-8 tok/s
- [ ] ~$300-800
- [ ] Silent, fits anywhere
- [ ] **Code:** Benchmark a Mini-PC if you have access; compare to your laptop

### Day 4 — Distributed Inference (Optional, Advanced)
- [ ] **exo**: distribute LLM inference across multiple devices (multiple laptops, phones, etc.)
- [ ] **Petals**: BitTorrent-style distributed LLM
- [ ] Not practical for production but fun to experiment
- [ ] **Code:** If you have 2+ devices, try exo; document experience

### Day 5 — Choosing Your Deployment Target
- [ ] Decision matrix:
  - Phone: highest privacy, smallest models
  - Laptop: dev + primary use
  - Mini-PC: always-on home server
  - Pi 5: always-on, low power, tiny services
  - Jetson: robotics, real product
  - Cloud GPU: training and very large models occasionally
- [ ] **Code:** Document your personal AI deployment strategy

### 🔨 Saturday Project
- [ ] **EdgeDeploy v1** — Project B for Month 11
  - [ ] Take your fine-tuned model (DomainDoc or smaller)
  - [ ] Deploy on at least 2 edge platforms (mobile + Pi or mobile + NPU)
  - [ ] Cross-platform benchmark report
  - [ ] Tweet / blog the comparison — this is useful info for the community

### 📄 Sunday
- [ ] Read: Jetson developer guides
- [ ] Read: Pi 5 + LLM benchmark posts
- [ ] Watch: edge AI product showcases

---

## 🔄 Buffer Week (Month 11 Revision) Local AI
- [ ] Revise: home server setup, NPU usage, mobile deployment, edge devices
- [ ] Ensure HomeAI is running stable for a week
- [ ] Use it with family/friends; collect feedback
- [ ] Decide your Magnum Opus direction
- [ ] **Build / Polish Monthly Project A:** HomeAI — Household Server
- [ ] **Build Monthly Project B:** EdgeDeploy — Cross-Platform Benchmark
- [ ] Journal: "Time to build the thing I'll be remembered for. What is it?"

---

## Week 45-48 — Magnum Opus Build (MyAI: Personal AI OS)

> Four weeks of focused, deep work on your signature project. This is the
> project you'll show in interviews, blog about, and remember.

### Week 45 — Spec, Design, Skeleton
- [ ] **Day 1 — Spec it:**
  - User: who is this for? (probably you first, then others like you)
  - Pain: what are you solving that no existing tool solves?
  - Success: how will you know it works? (specific metrics or user satisfaction)
- [ ] **Day 2 — Architecture diagram:**
  - All components (LLM, RAG, agent, voice, vision, etc.)
  - Data flow
  - External integrations (none, except your own services)
- [ ] **Day 3 — Tech stack decision:**
  - Backend: FastAPI / Express
  - Frontend: React / Svelte / Streamlit
  - Database: SQLite or Postgres
  - Deployment: Docker Compose
  - Models: pick your final set
- [ ] **Day 4 — Repository skeleton:**
  - Folders, README placeholder, license, .gitignore, CI skeleton, Docker setup
- [ ] **Day 5 — Define MVP scope:**
  - Cut features ruthlessly. MVP = simplest thing that proves the core value
  - Plan: MVP in Week 46, polish in Week 47, evaluation/users in Week 48

### Week 46 — Build the MVP
- [ ] **Day 1-2:** Backend services running, all models accessible
- [ ] **Day 3:** Core happy-path UI working
- [ ] **Day 4-5:** End-to-end demo possible — even if rough
- [ ] **Saturday:** First end-to-end demo to yourself; record video; identify top 5 issues

### Week 47 — Polish to Shippable
- [ ] **Day 1:** Error handling everywhere — graceful failures
- [ ] **Day 2:** Onboarding flow — first-run experience
- [ ] **Day 3:** Docs: README, install guide, architecture, troubleshooting
- [ ] **Day 4:** Performance pass — measure and optimize the hot paths
- [ ] **Day 5:** Security pass — privacy audit, sandbox check, secret management

### Week 48 — Evaluate, Get Users, Iterate
- [ ] **Day 1-2:** Beta with 3-5 friends/colleagues. Collect bug reports + feature requests
- [ ] **Day 3-4:** Fix critical issues, polish
- [ ] **Day 5:** Final build, screenshots, demo video (3-5 min)
- [ ] **Saturday:** Public launch — GitHub, blog post, social
- [ ] **Sunday:** Reflect — write retrospective. What worked? What didn't?

### 🎯 Magnum Opus Deliverables Checklist
- [ ] Public GitHub repo with clear README
- [ ] Architecture diagram + technical blog post (2000+ words)
- [ ] 3-5 minute demo video
- [ ] Benchmarks: latency, VRAM, quality, comparison to alternatives
- [ ] At least 3 external testers' feedback documented
- [ ] One-command install (script or Docker Compose)
- [ ] Privacy verification (no outbound calls during normal use)
- [ ] **This is your portfolio centerpiece**

---

## Week 49-50 — Open Source Contribution (llama.cpp, Ollama, Unsloth)

> The transition from "local AI user" to "local AI contributor." A merged PR
> to a major tool is rare and valuable. Aim for at least 1, ideally 2.

### Week 49 — Identify and Plan

**Day 1 — Pick Projects to Contribute To**
- [ ] Top candidates (tools you've used extensively):
  - **llama.cpp** — large codebase, but tons of opportunities
  - **Ollama** — Go-based, mature, friendly maintainers
  - **Unsloth** — newer, lots of growth, very welcoming
  - **Open WebUI** — TypeScript + Python, many "good first issues"
  - **smolagents** — minimal codebase, very approachable
  - **ComfyUI** — node-based, contributions often plugins
  - **KoboldCPP** — fork of llama.cpp with creative focus
  - **MLC-LLM** — complex but cutting-edge
- [ ] Pick 2-3 based on: your usage, codebase you'd enjoy, language match

**Day 2 — Find Issues**
- [ ] Browse "good first issue," "help wanted," "bug" labels
- [ ] Even better: fix a bug YOU hit during your year of using these tools
- [ ] Or: add a feature YOU wanted
- [ ] Read recent merged PRs to understand style + review patterns

**Day 3 — Set Up Dev Environment**
- [ ] Fork, clone, build from source
- [ ] Run tests
- [ ] Make a trivial change first (typo in docs, comment fix) to validate workflow

**Day 4-5 — Pick One and Dig In**
- [ ] Read relevant code with care
- [ ] Sketch your solution
- [ ] Implement, test, iterate
- [ ] Polish before opening PR

### Week 50 — Submit and Iterate

**Day 1 — Open Your First PR**
- [ ] Clear title and description
- [ ] Linked issue (if applicable)
- [ ] Tests if appropriate
- [ ] Be polite, patient, open to feedback

**Day 2-3 — Respond to Reviews**
- [ ] Maintainers may take days/weeks to respond — be patient
- [ ] Address feedback thoughtfully; don't just rubber-stamp
- [ ] Ask questions if unclear

**Day 4-5 — Start a Second PR**
- [ ] Different project or different area
- [ ] Apply lessons from first PR

### 🔨 Saturday — Document Your Contribution
- [ ] Blog post: "How I contributed to X" — what you fixed, how the codebase works, lessons learned
- [ ] Twitter/social: link to PR (link to MERGED PR ideally)
- [ ] Add to your portfolio/resume

### 📄 Sunday
- [ ] Read: maintainer blog posts on what makes a good PR
- [ ] Reflect: are you a Local AI Contributor now?

---

## Week 51-52 — Polish, Publish & Plan Next Phase

### Week 51 — Portfolio Polish

**Day 1 — GitHub Portfolio Audit**
- [ ] All 24 monthly projects on GitHub
- [ ] Pinned: your top 6 most impressive (Magnum Opus, DomainDoc, OllaMate, LocalRAG, VoiceMate, your OSS PR)
- [ ] Each pinned repo: great README, screenshots/GIF, install instructions, license

**Day 2 — Personal Site / Blog**
- [ ] If you don't have one: build a simple Astro/Next.js site
- [ ] List of projects, each with a description and link
- [ ] Blog posts you've written this year

**Day 3 — Resume + LinkedIn Update**
- [ ] Add: "Local AI Engineer" skills, specific tools, project links
- [ ] Highlight: "Fine-tuned and deployed LLMs on consumer 6GB GPU"
- [ ] Quantify: number of projects, models trained, OSS contributions

**Day 4 — Technical Blog Post Series**
- [ ] Write 3-5 deep technical posts:
  - "Fine-tuning LLMs on a 6GB GPU: a practical guide"
  - "Building local RAG that actually works"
  - "Function calling reliability on small local models"
  - "Lessons from a year of local AI"

**Day 5 — Share with the Community**
- [ ] Post on Reddit (r/LocalLLaMA, r/MachineLearning)
- [ ] Twitter / X
- [ ] Hacker News (Show HN)
- [ ] Discord / Slack communities

### Week 52 — Reflect and Plan

**Day 1 — Year-End Retrospective Blog Post**
- [ ] Comprehensive: what you learned, what surprised you, what didn't work
- [ ] Honest: what's still hard? what would you do differently?
- [ ] Useful: what would you tell someone starting from scratch?

**Day 2 — Update Both Roadmaps**
- [ ] Mark completed items
- [ ] Note where reality diverged from plan
- [ ] Identify gaps you want to fill

**Day 3 — What's Next?**
- [ ] Specialization options:
  - **Deeper systems:** focus on inference engines (write your own), kernel optimization
  - **Domain expert:** become THE local-AI person for medicine / law / education / etc.
  - **Research:** implement papers, contribute to research
  - **Product:** turn your Magnum Opus into a startup or paid product
  - **Open Source Maintainer:** become a regular contributor to 1-2 projects
- [ ] Pick a direction; sketch a 3-6 month plan

**Day 4 — Community Building**
- [ ] Mentor someone starting their local AI journey
- [ ] Answer questions on Stack Overflow / Discord / Reddit
- [ ] Pay forward what you learned

**Day 5 — Celebrate**
- [ ] You did the thing. Most people don't.
- [ ] Take a real break — a week off all this. Read fiction. Touch grass. Talk to humans.

### 🔨 Saturday — Final Demo Day
- [ ] Record a 10-min "year in review" demo of your top 3 projects in action
- [ ] Publish to YouTube
- [ ] You're done with the formal roadmap

### 📄 Sunday
- [ ] Thank the open source community
- [ ] Reflect: who are you now compared to a year ago?

---

### ✅ Phase 6 Completion Checklist (END OF JOURNEY)
- [ ] Designed and deployed a home AI server (HomeAI) used by you / family
- [ ] Used Intel AI Boost NPU effectively for at least one workload
- [ ] Deployed local AI to mobile (MLC-LLM or similar)
- [ ] Benchmarked local AI across 3+ platforms (laptop, NPU, Pi/phone)
- [ ] Shipped your Magnum Opus with external users
- [ ] At least 1 merged PR to a major OSS local-AI project
- [ ] Updated portfolio + blog + resume + LinkedIn
- [ ] Published 3+ technical blog posts during this year
- [ ] **🎖 You are a Local AI Expert. You can architect, build, and ship local AI products that matter.**

[⬆ Back to Table of Contents](#toc)

---

# ═══════════════════════════════════════════════════════════
# APPENDIX: LOCAL AI KNOWLEDGE BASE
# ═══════════════════════════════════════════════════════════

## Models That Fit 6GB VRAM (Curated List)

> All numbers are approximate for Q4_K_M quantization unless noted. Update as new models drop.

### General-Purpose Chat (Daily Drivers)
| Model | Size | VRAM | Tok/s (est) | Strengths |
|-------|------|------|-------------|-----------|
| **Llama 3.1 8B Instruct Q4_K_M** ⭐ | ~4.7 GB | ~5.5 GB w/ ctx | 25-35 | Balanced, well-aligned |
| **Qwen 2.5 7B Instruct Q4_K_M** ⭐ | ~4.7 GB | ~5.5 GB | 25-35 | Top general quality, multilingual |
| **Mistral 7B Instruct v0.3 Q4_K_M** | ~4.4 GB | ~5.3 GB | 25-35 | Fast, classic choice |
| **Llama 3.2 3B Instruct Q5_K_M** | ~2.3 GB | ~3.5 GB | 50-70 | Fast, quality good for size |
| **Phi-3 Mini 3.8B Q5_K_M** | ~2.7 GB | ~4 GB | 40-60 | Microsoft's small powerhouse |
| **Gemma 2 9B Instruct Q4_K_M** | ~5.4 GB | ~6 GB tight | 20-30 | Google's offering, careful w/ ctx |
| **Hermes 3 (Llama 3.1 fine-tune) Q4_K_M** | ~4.7 GB | ~5.5 GB | 25-35 | Better tool use than base |

### Coding Specialists
| Model | Size | VRAM | Strengths |
|-------|------|------|-----------|
| **Qwen 2.5 Coder 7B Instruct Q4_K_M** ⭐ | ~4.7 GB | ~5.5 GB | Best small-model coder |
| **Qwen 2.5 Coder 1.5B Base Q5_K_M** | ~1.1 GB | ~1.5 GB | Fast FIM autocomplete |
| **DeepSeek Coder V2 Lite (MoE) Q4_K_M** | ~10 GB total / 2.4B active | ~7 GB | MoE, partial offload |
| **StarCoder 2 7B Q4_K_M** | ~4.5 GB | ~5.5 GB | Older but solid |

### Reasoning Specialists
| Model | Size | VRAM | Strengths |
|-------|------|------|-----------|
| **DeepSeek-R1-Distill-Qwen-7B Q4_K_M** ⭐ | ~4.7 GB | ~5.5 GB | Step-by-step reasoning |
| **Qwen 2.5 Math 7B Q4_K_M** | ~4.7 GB | ~5.5 GB | Math-specific |
| **Phi-3 Mini 128k Q4_K_M** | ~2.5 GB | ~5 GB w/ long ctx | Long context reasoning |

### Vision-Language Models
| Model | Size | VRAM | Strengths |
|-------|------|------|-----------|
| **Moondream2 1.86B FP16** ⭐ | ~3.7 GB | ~4 GB | FAST, surprisingly capable |
| **LLaVA 1.6 Mistral 7B Q4_K_M** | ~4.7 GB | ~5.5 GB | High quality, slower |
| **MiniCPM-V 2.6 Q4** | ~5 GB | ~5.7 GB | Strong open VLM |
| **Phi-3.5 Vision Q4** | ~2.8 GB | ~4 GB | Microsoft's small VLM |
| **Qwen 2 VL 7B Q4** | ~5 GB | ~5.7 GB | Supports video |

### Embedding Models (Trivial VRAM)
| Model | Params | VRAM | Strengths |
|-------|--------|------|-----------|
| **nomic-embed-text-v1.5** ⭐ | 137M | ~280 MB | Fast, 8k context, top quality |
| **bge-large-en-v1.5** | 335M | ~700 MB | Higher quality, slower |
| **bge-m3** | 568M | ~1.2 GB | Multilingual, dense+sparse |
| **mxbai-embed-large-v1** | 335M | ~700 MB | Top MTEB |
| **all-MiniLM-L6-v2** | 22M | ~50 MB | Tiny, fast prototypes |

### Rerankers
| Model | Params | VRAM | Strengths |
|-------|--------|------|-----------|
| **bge-reranker-v2-m3** ⭐ | 568M | ~1.2 GB | Multilingual, accurate |
| **mxbai-rerank-large-v1** | 435M | ~900 MB | English, fast |
| **jina-reranker-v2-base-multilingual** | 278M | ~600 MB | Multilingual, smaller |

### Multilingual / Indian Language Models
| Model | Size | VRAM | Strengths |
|-------|------|------|-----------|
| **Sarvam-1** (2B) Q5 | ~1.5 GB | ~2 GB | 10 Indian languages |
| **Aya Expanse 8B Q4_K_M** | ~5 GB | ~5.7 GB | 23 languages, Cohere |
| **Krutrim** (when available) | varies | varies | Indian, multilingual |
| **Llama 3.1 8B Q4_K_M** | ~4.7 GB | ~5.5 GB | Decent Hindi |
| **Qwen 2.5 7B Q4_K_M** | ~4.7 GB | ~5.5 GB | Excellent Chinese, good multilingual |

---

## Quantization Format Cheat Sheet

| Format | Bits | File Size (7B) | VRAM | Quality vs FP16 | Best For |
|--------|------|----------------|------|-----------------|----------|
| **FP16 / BF16** | 16 | ~14 GB | ~14 GB | 100% (baseline) | Reference only |
| **Q8_0** | 8 | ~7.5 GB | ~7.5 GB | ~99.9% | When you have the VRAM |
| **Q6_K** | ~6.5 | ~5.6 GB | ~5.6 GB | ~99.5% | High quality at 6GB borderline |
| **Q5_K_M** | ~5.7 | ~4.8 GB | ~4.8 GB | ~99% | Great quality, fits 6GB |
| **Q5_K_S** | ~5.5 | ~4.6 GB | ~4.6 GB | ~98.5% | |
| **Q4_K_M** ⭐ | ~4.8 | ~4.0 GB | ~4.0 GB | ~98% | **Daily driver for 6GB** |
| **Q4_K_S** | ~4.5 | ~3.8 GB | ~3.8 GB | ~97% | More context room |
| **Q4_0** | 4 | ~3.8 GB | ~3.8 GB | ~96% | Legacy, slightly worse |
| **IQ4_XS** | ~4.2 | ~3.6 GB | ~3.6 GB | ~97% | Importance-quant, smaller |
| **Q3_K_M** | ~3.5 | ~3.1 GB | ~3.1 GB | ~93% | Noticeable quality drop |
| **Q3_K_S** | ~3.3 | ~2.9 GB | ~2.9 GB | ~91% | Use only if forced |
| **Q2_K** | ~2.5 | ~2.4 GB | ~2.4 GB | ~85% | Last resort |
| **IQ3_S, IQ2_XS** | various | varies | varies | varies | Use for VRAM emergencies |

| Format | Quantize Tool | Inference Engine | Notes |
|--------|---------------|------------------|-------|
| **GGUF (Q*)** | llama.cpp `quantize` | llama.cpp / Ollama / LM Studio | Universal, mmap, CPU + GPU |
| **GPTQ** | AutoGPTQ / GPTQModel | transformers / vLLM | Pure GPU |
| **AWQ** | AutoAWQ | vLLM / transformers | Pure GPU, best 4-bit quality |
| **EXL2** | exllamav2 | exllamav2 / tabbyAPI | Pure GPU, variable bpw |
| **bitsandbytes 4/8-bit** | runtime | transformers (training) | QLoRA training only |
| **FP8** | Transformer Engine | NeMo, vLLM (some) | Hopper / Ada GPUs |

---

## Essential Tools & Frameworks

### Inference & Serving
- **Ollama** ⭐ — daily-driver runner for GGUF models with Modelfile abstraction
- **llama.cpp** ⭐ — the inference engine under everything
- **LM Studio** — polished closed-source GUI
- **Jan** — open-source desktop app
- **Open WebUI** ⭐ — self-hosted ChatGPT-style UI
- **AnythingLLM** — RAG-first chat app
- **LibreChat** — multi-provider chat
- **KoboldCPP** — creative-writing focused fork
- **TGW (oobabooga)** — power-user toolkit
- **GPT4All** — desktop with model curation
- **llama-cpp-python** — Python bindings
- **vLLM** — production GPU serving (overkill for 6GB)
- **TensorRT-LLM** — NVIDIA's production stack
- **TabbyAPI** — EXL2 API server

### Fine-Tuning
- **Unsloth** ⭐ — fastest, low-VRAM QLoRA
- **Axolotl** — flexible YAML-based
- **LLaMA-Factory** — UI + CLI, all techniques
- **HuggingFace TRL** — base library underneath most
- **PEFT** — adapter library
- **bitsandbytes** — quantization for training

### Embeddings & RAG
- **Ollama embeddings API**, **sentence-transformers**, **fastembed**
- **LanceDB** ⭐ — embedded vector DB
- **ChromaDB**, **Qdrant local**, **Weaviate embedded**, **Milvus Lite**
- **rank_bm25** — sparse retrieval
- **LangChain**, **LlamaIndex** — RAG frameworks
- **Haystack** — alternative RAG framework

### Document Processing
- **PyMuPDF (fitz)** — fast PDF
- **pdfplumber** — table-friendly PDF
- **Marker** — high-quality PDF→Markdown
- **Docling** — IBM's pipeline
- **Tesseract**, **PaddleOCR**, **Surya OCR** — OCR
- **Unstructured** — universal docs
- **tree-sitter** — code AST

### Agents
- **smolagents** ⭐ — minimal, Hugging Face
- **LangGraph** — production-grade
- **CrewAI** — role-based
- **AutoGen** — Microsoft's conversational
- **Open Interpreter** — code-executing local
- **outlines**, **lm-format-enforcer**, **instructor** — structured output
- **mem0**, **Letta** — agent memory

### Multimodal
- **faster-whisper** ⭐ — fast Whisper inference
- **whisper.cpp** — C++ Whisper
- **distil-whisper** — distilled, faster
- **whisperX** — Whisper + diarization
- **pyannote.audio** — speaker diarization
- **Piper** ⭐ — fast local TTS
- **Coqui TTS / XTTS** — higher quality, voice cloning
- **Bark** — emotional TTS
- **OpenVoice** — voice cloning
- **openwakeword** — open wake-word detection

### Image Generation
- **ComfyUI** ⭐ — node-based, flexible
- **Automatic1111** — full-featured Web UI
- **Fooocus** — easiest SDXL UI
- **InvokeAI** — pro-grade
- **diffusers (HF)** — Python library
- **Kohya_ss** — training UI
- **OneTrainer**, **ai-toolkit** — alternative trainers

### Observability
- **LangFuse** ⭐ — self-hosted OSS
- **Phoenix (Arize)** — OSS
- **LangSmith** — cloud (LangChain)
- **Weights & Biases** — training tracking
- **TensorBoard** — classic

### Mobile / Edge
- **MLC-LLM** ⭐ — multi-platform compilation
- **llama.cpp mobile builds** — Termux on Android
- **OpenVINO** — Intel CPU/iGPU/NPU
- **DirectML** — Windows DirectX-12
- **ONNX Runtime** — universal

### NVIDIA Stack (Local-Friendly Bits)
- **TensorRT-LLM** — production inference
- **NeMo Framework** — training (heavy)
- **CUDA, cuDNN, Tensor Cores** — under everything
- **Triton Inference Server** — serving

---

## Books, Courses & Papers

### Books
- "Hands-On Generative AI with Transformers and Diffusion Models" (Sanseviero, Bekman)
- "AI Engineering" (Chip Huyen)
- "Designing Machine Learning Systems" (Chip Huyen)
- "Build a Large Language Model (From Scratch)" (Sebastian Raschka)

### Courses & Channels
- **Sebastian Raschka's blog & courses** — clear, deep, practical
- **Daniel Han (Unsloth) talks** — fine-tuning master
- **Andrej Karpathy: "Let's build GPT", "nanoGPT"** — foundational
- **Networkchuck, Matt Williams (Ollama channel)** — accessible local AI
- **The Local AI Show** (podcast)
- **Hugging Face Learn** — free, hands-on
- **DeepLearning.AI short courses** (many free on local AI topics)

### Papers Worth Re-Reading (Local AI Lens)
1. "QLoRA" (Dettmers et al., 2023) — the technique that enables 6GB fine-tuning
2. "LoRA" (Hu et al., 2021) — parameter-efficient adaptation
3. "GPTQ" (Frantar et al., 2022) — post-training quantization
4. "AWQ" (Lin et al., 2023) — activation-aware quantization
5. "Llama 3" technical report — modern open base
6. "Qwen 2.5" technical report — current SOTA for many tasks
7. "Phi-3 Technical Report" — small efficient models
8. "DeepSeek-V3 / R1" — frontier open models, MoE & reasoning
9. "RAG" (Lewis et al., 2020) — RAG origin
10. "Self-RAG" (Asai et al., 2023) — agentic retrieval
11. "Whisper" (Radford et al., 2022) — local ASR foundation
12. "LLaVA" (Liu et al., 2023) — vision-language alignment
13. "DPO" (Rafailov et al., 2023) — alignment without PPO
14. "Mamba" (Gu & Dao, 2023) — efficient alternatives to attention
15. "Self-Instruct" (Wang et al., 2022) — synthetic data
16. "ReAct" (Yao et al., 2022) — agent reasoning pattern
17. "Stable Diffusion" (Rombach et al., 2022) — local image gen
18. "FlashAttention" (Dao, 2022) — efficient attention (under llama.cpp/Unsloth)
19. "RoPE" (Su et al., 2021) — positional encoding modern LLMs use
20. "Chinchilla" (Hoffmann et al., 2022) — scaling laws (helps understand why small models work)

---

## GitHub Repos to Study (Local AI)

### Inference Engines
- [ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp) — the foundation ⭐
- [ollama/ollama](https://github.com/ollama/ollama) — your daily driver
- [LostRuins/koboldcpp](https://github.com/LostRuins/koboldcpp) — llama.cpp fork
- [turboderp/exllamav2](https://github.com/turboderp/exllamav2) — EXL2
- [vllm-project/vllm](https://github.com/vllm-project/vllm) — for understanding production patterns
- [mlc-ai/mlc-llm](https://github.com/mlc-ai/mlc-llm) — cross-platform compilation

### Fine-Tuning
- [unslothai/unsloth](https://github.com/unslothai/unsloth) ⭐
- [axolotl-ai-cloud/axolotl](https://github.com/axolotl-ai-cloud/axolotl)
- [hiyouga/LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
- [huggingface/peft](https://github.com/huggingface/peft)
- [huggingface/trl](https://github.com/huggingface/trl)

### Apps / Frontends
- [open-webui/open-webui](https://github.com/open-webui/open-webui) ⭐
- [Mintplex-Labs/anything-llm](https://github.com/Mintplex-Labs/anything-llm)
- [danny-avila/LibreChat](https://github.com/danny-avila/LibreChat)
- [continuedev/continue](https://github.com/continuedev/continue) — code assistant
- [paul-gauthier/aider](https://github.com/paul-gauthier/aider)

### RAG / Embeddings
- [run-llama/llama_index](https://github.com/run-llama/llama_index)
- [langchain-ai/langchain](https://github.com/langchain-ai/langchain)
- [lancedb/lancedb](https://github.com/lancedb/lancedb)
- [chroma-core/chroma](https://github.com/chroma-core/chroma)
- [VikParuchuri/marker](https://github.com/VikParuchuri/marker)
- [DS4SD/docling](https://github.com/DS4SD/docling)

### Agents
- [huggingface/smolagents](https://github.com/huggingface/smolagents)
- [langchain-ai/langgraph](https://github.com/langchain-ai/langgraph)
- [microsoft/autogen](https://github.com/microsoft/autogen)
- [crewAIInc/crewAI](https://github.com/crewAIInc/crewAI)
- [OpenInterpreter/open-interpreter](https://github.com/OpenInterpreter/open-interpreter)
- [dottxt-ai/outlines](https://github.com/dottxt-ai/outlines)
- [mem0ai/mem0](https://github.com/mem0ai/mem0)

### Multimodal
- [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- [ggerganov/whisper.cpp](https://github.com/ggerganov/whisper.cpp)
- [m-bain/whisperX](https://github.com/m-bain/whisperX)
- [rhasspy/piper](https://github.com/rhasspy/piper)
- [coqui-ai/TTS](https://github.com/coqui-ai/TTS) — keep an eye on forks
- [vikhyat/moondream](https://github.com/vikhyat/moondream)
- [haotian-liu/LLaVA](https://github.com/haotian-liu/LLaVA)
- [comfyanonymous/ComfyUI](https://github.com/comfyanonymous/ComfyUI)

### Edge / Mobile
- [openvinotoolkit/openvino](https://github.com/openvinotoolkit/openvino)
- [mlc-ai/web-llm](https://github.com/mlc-ai/web-llm) — LLMs in browser via WebGPU
- [exo-explore/exo](https://github.com/exo-explore/exo) — distributed inference

---

## Communities, Newsletters & Channels

### Subreddits
- **r/LocalLLaMA** ⭐ — THE community for local LLMs
- **r/StableDiffusion** — for SD-related work
- **r/selfhosted** — home server / privacy
- **r/MachineLearning** — broader ML
- **r/Ollama** — Ollama-specific

### Discord Servers
- **Ollama Discord**
- **Hugging Face Discord**
- **Unsloth Discord** — Daniel Han is often active
- **LocalAI Discord**

### Twitter / X Accounts to Follow
- @ggerganov (llama.cpp)
- @ollama
- @danielhanchen (Unsloth)
- @vikhyatk (Moondream)
- @teortaxesTex (deep technical local AI commentary)
- @abacaj (practical fine-tuning)
- @awnihannun (MLX/Apple silicon)
- @Teknium1 (Nous Research)

### Newsletters / Blogs
- The Batch (DeepLearning.AI)
- Last Week in AI (substack)
- AlphaSignal
- Sebastian Raschka's "Ahead of AI"
- Simon Willison's blog
- The Local AI Newsletter

### YouTube Channels
- Matt Williams (Ollama official)
- NetworkChuck
- AI Explained
- Sam Witteveen
- Wes Roth
- Yannic Kilcher (deeper, paper reviews)
- Andrej Karpathy

### Podcasts
- The Local AI Podcast
- Latent Space (Swyx + Alessio)
- Practical AI
- The Cognitive Revolution

---

## Troubleshooting Cheat Sheet

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| OOM (CUDA out of memory) | Model too big for VRAM | Smaller quant (Q4 → Q3), reduce ctx, fewer layers (`-ngl`) |
| Very slow (< 5 tok/s on 7B Q4) | Model on CPU not GPU | Check `-ngl 99`, verify CUDA build, `nvidia-smi` shows usage |
| Garbage / repeated output | Wrong chat template | Verify Modelfile TEMPLATE matches model family |
| Hallucinated tool calls | Small model, generic prompt | Use Qwen 2.5 Instruct, add few-shot, use schema enforcement |
| Embedding too slow | Running on CPU | Use Ollama (GPU by default) or sentence-transformers w/ CUDA |
| RAG returns wrong chunks | Bad chunking or weak embed | Try different chunk size, hybrid retrieval, reranker |
| Ollama "model not found" | Wrong tag | `ollama list` to check; `ollama pull <model>` to get |
| Ollama uses too much disk | Old models | `ollama rm <model>`; check `~/.ollama/models` |
| Continue.dev FIM is sluggish | Big model for FIM | Switch to 1.5B FIM model; use `temperature: 0` |
| Fine-tune loss is NaN | LR too high, bad data | Lower LR (1e-5 to 5e-5 for QLoRA), check data formatting |
| Fine-tune "doesn't change anything" | Too few epochs, too low LR | More epochs (3-5), higher LR (2e-4 to 5e-4), more data |
| SD generates noise | Wrong sampler/scheduler | Use DPM++ 2M Karras, 20-30 steps |
| SDXL OOM on 6GB | No VAE tiling | Enable `--medvram` (A1111) or use Lightning checkpoint |
| Voice assistant slow | LLM unload between calls | Set `OLLAMA_KEEP_ALIVE=-1` |
| Whisper missing words | Audio quality / language | Try larger model, specify language, enable VAD |
| Power consumption high | Models always loaded | Set `OLLAMA_KEEP_ALIVE=10m` (unload after 10 min idle) |

---

## Cost & Energy Reality Check

### Electricity Cost of Local AI (Approximate)

Assume RTX 1000 Ada Laptop @ ~50W under LLM load (very rough; varies):
- 1 hour of heavy use = 0.05 kWh
- Indian electricity @ ₹8/kWh = ₹0.40/hour
- 4 hours/day = ₹1.60/day = ₹48/month ≈ **<$1/month for heavy local AI use**

Compare:
- GPT-4 API at typical use: $20-200/month
- Claude API typical: $20-100/month
- ChatGPT Plus: $20/month flat (limited)

**Local AI pays for itself in electricity vs paid APIs at any non-trivial volume.**

### Hardware Amortization

If you bought a $300 used PC + $250 used RTX 3060 12GB for a home AI server:
- Total: $550
- Useful life: ~3 years for AI workloads
- Cost/month: ~$15
- Plus electricity (~24/7 idle ~30W, peak 200W): ~$5-15/month depending on usage
- **Total: $20-30/month for an always-on, multi-user, private AI server**

### When Cloud Is Genuinely Cheaper
- **Very low volume** (<$5/month of API calls)
- **Burst workloads** (occasional big jobs, mostly idle)
- **Need frontier model quality** (GPT-4, Claude 3.5 Sonnet, etc.)
- **No upfront capital** for hardware

### When Local Wins (Honest)
- **Sustained heavy use** (>$20/month of API at your current rate)
- **Privacy requirement** (can't be evaluated by cost alone)
- **Predictable latency / no API failures**
- **Customization (fine-tuning) at modest scale**
- **The intellectual satisfaction of owning your stack**

---

## Final Words

> "The future is already here — it's just not evenly distributed."
> — William Gibson

You've spent a year proving that the cutting edge of AI isn't only behind expensive APIs in distant data centers. With a modest 6GB laptop GPU, a curious mind, and consistent effort, you can:
- Run capable LLMs locally
- Fine-tune them on your data
- Build RAG, agents, voice assistants, image generators
- Deploy across phone, edge, home server
- Contribute to the open source ecosystem

You started as a user of cloud AI. You're now an architect of your own AI stack.

Use that power thoughtfully. Build things that respect privacy, expand human agency, and give your community access to capabilities that previously belonged only to big tech.

---

*Local AI Roadmap | Companion to the main LLM + CUDA + Hardware Mastery Roadmap*

*Current week: Week 0*
*Hardware: NVIDIA RTX 1000 Ada Laptop GPU (6 GB VRAM) + 31.5 GB RAM + Intel AI Boost NPU*
*Weeks completed: 0*
*Monthly projects completed: 0 / 24 (12 Project A + 12 Project B)*
*Open-source contributions: 0*

[⬆ Back to Table of Contents](#toc)
