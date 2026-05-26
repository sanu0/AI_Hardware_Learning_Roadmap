# Day 0b — Python + uv + PyTorch in WSL2

> **Goal:** Set up the Python ML stack inside WSL2 — `uv` (modern package manager), a Python 3.11 virtual environment, PyTorch with CUDA support, Git config, and VS Code WSL integration. End the day with **proof that Linux Python can drive the Windows GPU through WSL2**.
>
> **Time:** ~45 min (mostly waiting on PyTorch download — it's a chunky ~2-3 GB).
>
> **Why this matters:** WSL2 + CUDA gave us a Linux dev environment with GPU access. Today we install the ML toolchain on top of it. By the end, `import torch; torch.cuda.is_available()` returns `True` — and from that moment, the entire ML ecosystem is yours.

---

## 📋 Today's Checklist

- [x] Install `uv` via the official install script
- [x] Create `~/local-ai` workspace + Python 3.11 venv
- [x] Activate the venv (prompt now shows `(.venv)`)
- [x] `uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`
- [x] Verify `torch.cuda.is_available()` returns `True` + GPU name displayed
- [x] Configure Git globally (`user.name`, `user.email`, `init.defaultBranch`)
- [x] (Optional) Generate SSH key for GitHub
- [x] Install VS Code (Windows side) + WSL extension
- [x] Open `~/local-ai` in VS Code from Ubuntu terminal (`code .`) — green `WSL: Ubuntu` indicator confirmed

---

## 🧠 Concepts I'm Learning Today

### Why `uv` instead of `pip` or `conda`
- **Speed:** 10-100× faster than pip. Resolves & installs a typical ML environment in seconds instead of minutes.
- **Correctness:** uses a real resolver (no more "broken environment" surprises).
- **Python versions:** `uv venv --python 3.11` downloads & uses 3.11 even if your system has 3.12.
- **Single binary:** no Python-bootstrap chicken-and-egg (unlike pip itself, which needs Python to install).
- It's quickly becoming the standard in 2025-26 ML workflows.

### Virtual environments — why every project gets its own `.venv`
- ML libraries pin specific torch/CUDA/numpy versions. Without venvs, project A's installs break project B.
- `(.venv)` prefix in your prompt = you're inside an isolated environment.
- `deactivate` exits it; `source .venv/bin/activate` re-enters.
- **Convention:** every project gets a `.venv` folder at its root. Never `pip install` globally.

### Why we install PyTorch with `--index-url cu121` even when toolkit is 12.6
- PyTorch ships pre-compiled wheels per CUDA version (cu118, cu121, cu124, …).
- The `cu121` wheel was compiled against CUDA 12.1, but the **CUDA runtime is forward-compatible** — a binary built against 12.1 works fine with a 12.6 toolkit installed.
- PyTorch's wheel index lags ~1-2 minor versions behind latest toolkit. Always pick the **latest cu1XX wheel that's ≤ your toolkit**. Today that's `cu121`.
- Don't try to chase the absolute latest — `cu121` is what every tutorial assumes.

### What `torch.cuda.is_available()` actually checks
1. Can it find `libcuda.so` (the driver library)? — provided by WSL2 from Windows ✅
2. Can it find the CUDA runtime? — provided by `cuda-toolkit-12-6` ✅
3. Can it enumerate at least one GPU? — your RTX 1000 Ada via passthrough ✅
- All three need to succeed for `True`. Any failure → False, often silently.

### WSL2 + VS Code = the killer dev setup
- Code runs **in Linux** (your venv, your CUDA, your tools).
- Editor runs **in Windows** (native UI, no X11 forwarding hacks).
- File access is bidirectional (`/mnt/c/` from Linux, `\\wsl$\Ubuntu\home\<user>\` from Windows).
- The green `WSL: Ubuntu` indicator at the bottom-left = VS Code knows it's connected to Linux.
- This is the same setup every serious ML engineer uses on Windows.

---

## 🛠 Step-by-Step Setup (What I Actually Ran)

### Phase 1: Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
uv --version
```

Expected output:
```
uv 0.x.y
```

### Phase 2: Create Workspace + Virtual Environment

```bash
mkdir -p ~/local-ai && cd ~/local-ai
uv venv .venv --python 3.11
source .venv/bin/activate
```

After activation, prompt changes to:
```
(.venv) <user>@<hostname>:~/local-ai$
```

The `(.venv)` prefix confirms you're inside the isolated environment.

### Phase 3: Install PyTorch with CUDA Support (the chunky one)

```bash
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Downloads ~2-3 GB of wheels (torch itself is ~2 GB because it bundles CUDA libraries).

### Phase 4: The Moment of Truth ⭐

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

Expected output:
```
CUDA available: True
Device: NVIDIA RTX 1000 Ada Generation Laptop GPU
```

**🎯 SUCCESS criteria:**
- `True` (not `False`)
- GPU name shown (not `"CPU only"`)

If you see both ✅ → the entire ML ecosystem is now usable from this venv. Time to celebrate.

### Phase 5: Configure Git Globally

```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
git config --global init.defaultBranch main
```

Verify with `git config --list`.

### Phase 6: SSH Key for GitHub (Optional)

```bash
ssh-keygen -t ed25519 -C "your.email@example.com"
# Press Enter for default location; optionally set a passphrase
cat ~/.ssh/id_ed25519.pub
```

Copy the printed key → GitHub → Settings → SSH and GPG keys → New SSH key.

### Phase 7: VS Code WSL Integration

1. Install VS Code on Windows (https://code.visualstudio.com/) if not already.
2. Open VS Code → Extensions (`Ctrl+Shift+X`) → search "WSL" → install **WSL** by Microsoft.
3. From Ubuntu terminal:
   ```bash
   cd ~/local-ai
   code .
   ```
4. First time: VS Code installs a small server inside WSL2 (~30 sec). Then it opens.
5. **Confirm:** bottom-left corner shows green `WSL: Ubuntu` indicator → you're connected.
6. Open VS Code's integrated terminal (`Ctrl+~`) — it should drop you into Ubuntu bash automatically.

---

## ⚠ Surprises & Lessons Learned

1. **PyTorch wheel is ~2 GB.** Don't panic if it takes 3-5 min. It bundles CUDA libraries.
2. **`(.venv)` prefix is your "am I in the right environment?" indicator.** Always check before installing or running.
3. **`cu121` wheel is the right pick even with a 12.6 toolkit.** PyTorch wheel indices lag behind toolkit versions. CUDA runtime is forward-compatible.
4. **VS Code's WSL integration is magic.** Code in Linux, edit in Windows, no friction.
5. **If a new terminal doesn't have `(.venv)`, just `cd ~/local-ai && source .venv/bin/activate`.** Or set up an auto-activate hook in `~/.bashrc` (later).

---

## 🐛 Troubleshooting Cheat Sheet

| Symptom | Fix |
|---|---|
| `uv: command not found` after install | `source ~/.bashrc` or open a new terminal |
| `torch.cuda.is_available()` returns `False` | Verify (a) `nvidia-smi` works in WSL2, (b) `nvcc --version` works, (c) you installed the `cu121` wheel (not CPU-only), (d) you're in the right venv |
| PyTorch install fails with "no matching distribution" | Make sure you used `--index-url https://download.pytorch.org/whl/cu121`, not a typo |
| `code .` says "command not found" | Install VS Code on Windows first; the `code` command gets installed by the WSL extension on first connection |
| VS Code opens but no WSL indicator | Open Command Palette (`Ctrl+Shift+P`) → "WSL: Connect to WSL" |
| Forgot to activate venv → `pip install` polluted system Python | Just `deactivate`, delete the broken `.venv`, `uv venv .venv --python 3.11` again, reinstall |

---

## ✅ Done When

- [x] `uv --version` works
- [x] `~/local-ai/.venv` exists and is activated (prompt shows `(.venv)`)
- [x] `python -c "import torch; print(torch.cuda.is_available())"` prints `True`
- [x] `python -c "import torch; print(torch.cuda.get_device_name(0))"` prints `NVIDIA RTX 1000 Ada Generation Laptop GPU`
- [x] `git config --get user.email` returns your email
- [x] VS Code opens with green `WSL: Ubuntu` indicator from `code .`

---

## 🔜 Next: `DAY_1_OLLAMA_INSTALL.md`

Install **Ollama** in WSL2 — the easiest way to run local LLMs. By the end of Day 1, you'll have:
- Ollama running as a service
- Your first local LLM downloaded (Llama 3.2 3B — small, fast, fits comfortably in 6GB VRAM)
- A real chat session with a model running entirely on your laptop, fully offline

This is the moment local AI stops being "concept" and becomes "I'm chatting with an LLM that lives on my computer."
