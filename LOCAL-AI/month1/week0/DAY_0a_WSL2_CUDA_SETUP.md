# Day 0a — WSL2 + CUDA Toolkit Setup

> **Goal:** Set up WSL2 + Ubuntu on Windows, verify GPU passthrough, install CUDA toolkit, and confirm Python + PyTorch can see the GPU from inside Linux.
>
> **Time:** ~1-2 hours (most spent waiting on apt downloads).
>
> **Why this matters:** 90% of the ML ecosystem (Unsloth, Axolotl, bitsandbytes, Triton, flash-attention, vLLM, llama.cpp builds) is Linux-first. WSL2 + CUDA gives you a full Linux dev environment with GPU access without leaving Windows. This is the single most important Windows-specific setup; get it right once.

---

## 📋 Today's Checklist

- [x] Verify NVIDIA driver works on Windows (`nvidia-smi` in PowerShell)
- [x] Find correct Ubuntu distro name via `wsl --list --online`
- [x] Install WSL2 + Ubuntu via `wsl --install -d <name>`
- [x] Reboot Windows; complete first-launch UNIX user setup
- [x] Verify GPU passthrough inside Ubuntu (`nvidia-smi` from WSL2)
- [ ] Update apt + install build essentials
- [ ] Install CUDA toolkit 12.6 inside WSL2
- [ ] Add CUDA to PATH + verify `nvcc --version`
- [ ] (Continued in `DAY_0b_PYTHON_PYTORCH.md`) Install uv + PyTorch, verify `torch.cuda.is_available()`

---

## 🧠 Concepts I'm Learning Today

### WSL2 isn't a "VM" in the traditional sense
- It runs a real Linux kernel on top of a thin Hyper-V layer.
- Shares **GPU, filesystem, network, and ~50% of RAM** with Windows.
- Boots in ~1 second; uses 500MB-2GB idle RAM.
- Better described as "Linux as a feature of Windows" than "a VM."

### Why we need Linux for ML, even though Ollama works on Windows
| Tool | Windows native | WSL2 (Linux) |
|---|---|---|
| Ollama, LM Studio, llama.cpp inference | ✅ Works | ✅ Works |
| Triton (GPU kernel language) | ❌ Doesn't work | ✅ Works |
| Flash Attention 2/3 | ❌ No official wheels | ✅ Works |
| bitsandbytes (proper, used by QLoRA) | ⚠ Hacky forks | ✅ Works |
| Unsloth, Axolotl | ❌ Broken | ✅ Works |
| vLLM, DeepSpeed | ❌ Linux only | ✅ Works |

So the workflow becomes: **Ollama/LM Studio/ComfyUI in Windows for daily chat & inference, WSL2 for fine-tuning and serious ML work.**

### CUDA driver version vs CUDA toolkit version (gotcha)
- `nvidia-smi` shows the **max CUDA version your DRIVER supports** (e.g., a modern driver may list CUDA 13.0 in the header).
- This is **NOT what toolkit version to install**.
- The CUDA *toolkit* is independent — it's the compilers, libraries, and headers your code links against.
- **Best practice:** install toolkit **12.6** (or 12.8). All major ML libraries (PyTorch, Unsloth, bitsandbytes) ship CUDA 12.x wheels. Don't chase the latest toolkit just because your driver supports it.
- A driver can support multiple toolkit versions; a toolkit version 12.x always works with any driver that supports CUDA 12.x or higher.

### How GPU passthrough works in WSL2
- The Windows NVIDIA driver is shared into WSL2 via the `/usr/lib/wsl/lib/` virtual mount.
- You do **NOT** install the Linux NVIDIA driver in WSL2 (this is a common mistake — installing it breaks the passthrough).
- You only install the **CUDA toolkit** (`cuda-toolkit-12-6`), which is just userspace tools and libraries.

---

## 🛠 Step-by-Step Setup (What I Actually Ran)

### Phase 1: Verify Windows Driver

```powershell
# In Windows PowerShell:
nvidia-smi
```

Expected output (shows GPU + driver version + max CUDA capability):
```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI <version>             Driver Version: <version>      CUDA Version: <max>     |
+-----------------------------------------+------------------------+----------------------+
|   0  NVIDIA RTX 1000 Ada Gene...  WDDM  |   00000000:01:00.0 Off |                  N/A |
| N/A   <temp>  P3        <pwr>   /  <max>|       0MiB /   6141MiB |      0%      Default |
+-----------------------------------------+------------------------+----------------------+
```

**Verified:** GPU detected, ~6 GB VRAM, modern driver, max CUDA capability sufficient.

---

### Phase 2: Find Correct WSL Distro Name (DON'T SKIP)

```powershell
wsl --list --online
```

**Real output I got:**
```
The following is a list of valid distributions that can be installed.
The default distribution is denoted by '*'.

  NAME                            FRIENDLY NAME
* Ubuntu                          Ubuntu
  Debian                          Debian GNU/Linux
  kali-linux                      Kali Linux Rolling
  OracleLinux_7_9                 Oracle Linux 7.9
  OracleLinux_8_10                Oracle Linux 8.10
  OracleLinux_9_5                 Oracle Linux 9.5
  SUSE-Linux-Enterprise-15-SP6    SUSE Linux Enterprise 15 SP6
  openSUSE-Tumbleweed             openSUSE Tumbleweed
```

**Surprise:** only generic `Ubuntu` was available — no `Ubuntu-24.04` or `Ubuntu-22.04` versioned names. The Windows build dictates the available names.

### Phase 3: Failed Attempt + Recovery

```powershell
# Tried this first:
wsl --install -d Ubuntu-22.04
# ❌ Error: "Invalid distribution name: 'Ubuntu-22.04'."

# Then this (worked):
wsl --install -d Ubuntu
```

**Lesson learned:** ALWAYS run `wsl --list --online` first; distribution naming varies by Windows build.

### Phase 4: WSL2 Installation Output

```
The requested operation requires elevation.
Installing: Virtual Machine Platform
Virtual Machine Platform has been installed.
Installing: Windows Subsystem for Linux
Windows Subsystem for Linux has been installed.
Installing: Ubuntu
Ubuntu has been installed.
The requested operation is successful. Changes will not be effective until the system is rebooted.
```

**Reboot Windows.** The reboot is mandatory because the Virtual Machine Platform feature needs to be enabled at boot.

### Phase 5: First Ubuntu Launch

After reboot, opened **Ubuntu** from the Start menu. First launch:
- Prompts for a new **UNIX username** (lowercase; separate from Windows username).
- Prompts for a **password** twice. **Cursor doesn't move while typing — normal Linux behavior.**
- Hostname is auto-assigned to match the Windows machine name.

Landed at the green Linux prompt:
```
<user>@<hostname>:~$
<user>@<hostname>:~$ ls
<user>@<hostname>:~$ pwd
/home/<user>
```

### Phase 6: The "It All Clicks" Moment — GPU Passthrough Verified

```bash
# Inside Ubuntu (WSL2):
nvidia-smi
```

✅ **Same GPU appeared as in Windows PowerShell.** Same model, same VRAM, same driver. This proves WSL2's GPU passthrough works without any additional driver install inside Linux.

### Phase 7: System Update + Build Tools

```bash
sudo apt update && sudo apt upgrade -y
# First sudo call asks for the UNIX password you set in Phase 5.

sudo apt install -y build-essential cmake git curl wget python3-dev python3-venv python3-pip libomp-dev pkg-config
```

These tools are needed for compiling things like `llama.cpp`, `whisper.cpp`, and various Python C-extensions later.

### Phase 8: Install CUDA Toolkit (the slow part)

```bash
# Add NVIDIA's CUDA repo for WSL Ubuntu:
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update

# Install CUDA toolkit (NOT the driver — driver is in Windows):
sudo apt-get install -y cuda-toolkit-12-6
```

**This step takes ~10-30 minutes.** It downloads ~3-4 GB of CUDA packages (compilers, libraries, headers, samples), then unpacks ~7-10 GB to `/usr/local/cuda-12.6/`.

While waiting, useful checks from a second Ubuntu terminal:
```bash
# Count downloaded CUDA packages (should grow over time):
ls -la /var/cache/apt/archives/ | grep cuda | wc -l

# Watch disk space shrinking as install grows:
df -h ~

# Check active install process:
top  # press 'q' to exit; look for dpkg/apt processes
```

### Phase 9: Add CUDA to PATH

```bash
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify:
nvcc --version
```

Expected output:
```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on ...
Cuda compilation tools, release 12.6, V12.6.x
```

---

## ⚠ Surprises & Lessons Learned

1. **WSL distro naming varies by Windows build.** ALWAYS run `wsl --list --online` first. `Ubuntu-22.04` doesn't exist on every build; the generic `Ubuntu` is safest.
2. **`nvidia-smi`'s "CUDA Version" is the DRIVER capability, not what toolkit to install.** Pick toolkit 12.6 regardless of what the driver header shows.
3. **DON'T install Linux NVIDIA driver inside WSL2.** It breaks the Windows-passthrough driver. Only install `cuda-toolkit-XX-Y` packages.
4. **MobaXterm, Windows Terminal, Ubuntu app, and VS Code integrated terminal all open the SAME Ubuntu instance.** It's purely UX preference. Windows Terminal is the recommended default (free, native, tabs).
5. **Sudo password input doesn't echo characters — this is normal**, not a hang.
6. **The reboot after `wsl --install`** is mandatory. The installer literally tells you. Don't try to skip it.
7. **CUDA toolkit install is slow.** Don't kill it; let it run. ~10-30 minutes is normal.

---

## 🐛 Troubleshooting Cheat Sheet

| Symptom | Fix |
|---|---|
| `Invalid distribution name` | Run `wsl --list --online`; use exact name listed |
| `nvidia-smi: command not found` inside WSL2 | `wsl --update` in PowerShell, then `wsl --shutdown` and reopen Ubuntu |
| `apt-get update` fails with GPG errors | `sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys <KEYID>` |
| `cuda-toolkit-12-6` not found | Try `cuda-toolkit-12-4` or `cuda-toolkit-12-8` |
| `nvcc --version` says "command not found" after install | Run `source ~/.bashrc` or open a new terminal |
| WSL2 uses too much RAM | Create `C:\Users\<you>\.wslconfig`: `[wsl2]\nmemory=24GB\nprocessors=8\nswap=8GB`, then `wsl --shutdown` |
| Install appears hung (>45 min) | `Ctrl+C`, then `sudo dpkg --configure -a`, then retry install |

---

## ✅ Done When

- [x] `nvidia-smi` works in BOTH Windows PowerShell AND inside WSL2 Ubuntu (same output)
- [ ] `nvcc --version` prints CUDA 12.6 inside WSL2
- [ ] No Linux NVIDIA driver was installed (only toolkit)
- [ ] You feel comfortable opening/closing WSL2 and navigating Linux basic commands (`ls`, `cd`, `pwd`)

---

## 🔜 Next: `DAY_0b_PYTHON_PYTORCH.md`

Install `uv` (modern Python package manager), create a `~/local-ai` workspace with a Python 3.11 virtual environment, install PyTorch with CUDA support, and run the "moment of truth" verification:

```python
import torch
print(torch.cuda.is_available())          # Should print: True
print(torch.cuda.get_device_name(0))      # Should print your RTX 1000 Ada
```

When that prints `True` + the GPU name → **Linux Python is talking to the Windows GPU through WSL2. The entire ML ecosystem is now available to you.**
