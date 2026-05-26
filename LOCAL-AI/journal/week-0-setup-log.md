# Week 0 Setup Log — Full Command History

> Detailed, verbatim command-by-command log of the Week 0 setup. The main README has the **distilled** version; this file has the **complete** raw log including outputs, errors, fixes. Refer here when troubleshooting or when redoing this on a new machine.

---

## Hardware Baseline (Captured May 20, 2026)

```
PS C:\Users\ksanu> nvidia-smi
Thu May 21 11:22:53 2026
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 581.95                 Driver Version: 581.95         CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                  Driver-Model | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|=========================================+========================+======================|
|   0  NVIDIA RTX 1000 Ada Gene...  WDDM  |   00000000:01:00.0 Off |                  N/A |
| N/A   45C    P3            588W /   43W |       0MiB /   6141MiB |      0%      Default |
+-----------------------------------------+------------------------+----------------------+
```

**Key facts to remember:**
- GPU: NVIDIA RTX 1000 Ada Generation Laptop GPU
- Driver version: 581.95 (very recent)
- Max CUDA supported: 13.0 (driver capability — NOT what toolkit to install)
- VRAM: 6141 MiB ≈ 6 GB (the design constraint of this entire roadmap)
- System RAM: 31.5 GB total → WSL2 default sees ~16 GB

---

## Step-by-Step Real Command History

### Attempt 1 (failed — bad distro name)

```powershell
PS C:\Users\ksanu> wsl --install -d Ubuntu-22.04
Invalid distribution name: 'Ubuntu-22.04'.
To get a list of valid distributions, use 'wsl --list --online'.
```

### Recovery — check available distros first

```powershell
PS C:\Users\ksanu> wsl --list --online
The following is a list of valid distributions that can be installed.
The default distribution is denoted by '*'.
Install using 'wsl --install -d <Distro>'.

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

**Note:** Only generic `Ubuntu` available (no `Ubuntu-24.04` or `Ubuntu-22.04`). This Windows build keeps it simple.

### Attempt 2 (success)

```powershell
PS C:\Users\ksanu> wsl --install -d Ubuntu
The requested operation requires elevation.
Installing: Virtual Machine Platform
Virtual Machine Platform has been installed.
Installing: Windows Subsystem for Linux
Windows Subsystem for Linux has been installed.
Installing: Ubuntu
Ubuntu has been installed.
The requested operation is successful. Changes will not be effective until the system is rebooted.
```

### Reboot Windows. After reboot, opened Ubuntu from Start menu.

- First-launch prompted for UNIX username + password.
- Used username: `ksanu`
- Hostname auto-assigned: `NV-D8M4FB4` (matches Windows hostname)
- Landing prompt:
  ```
  ksanu@NV-D8M4FB4:~$
  ksanu@NV-D8M4FB4:~$ ls
  ksanu@NV-D8M4FB4:~$ pwd
  /home/ksanu
  ```
- Home directory empty, as expected.

### Phase 2 onwards — TODO (fill in as I run each step)

#### nvidia-smi from inside WSL2 Ubuntu

```bash
# Output: (paste here when verified)
```

#### apt update + upgrade

```bash
# Output / any errors: (paste here)
```

#### Build essentials install

```bash
# Output: (paste here)
```

#### CUDA toolkit install

```bash
# Output: (paste here)
```

#### nvcc --version

```bash
# Output: (paste here)
```

#### uv install

```bash
# Output: (paste here)
```

#### uv venv + PyTorch install + verify torch.cuda.is_available()

```bash
# Output: (paste here)
```

#### Git config + SSH key

```bash
# Output: (paste here)
```

#### VS Code WSL extension working

```bash
# Notes: (anything weird?)
```

---

## Surprises & Lessons Learned

1. **Distribution naming** — `Ubuntu-22.04` doesn't exist on my Windows build. Always run `wsl --list --online` first.
2. **CUDA driver version vs. toolkit version** — Driver supports 13.0, but I install toolkit 12.6 because PyTorch/Unsloth target 12.x. They are independent.
3. **NV-D8M4FB4 hostname** — WSL2 inherits the Windows machine name (this is an NVIDIA-issued laptop, hence the NV prefix).
4. **MobaXterm + Windows Terminal + Ubuntu app** all open the SAME Ubuntu instance — choice is purely UX preference.
5. **Sudo password input** — cursor doesn't move while typing. Normal Linux behavior.

---

## If I Need to Redo This on Another Machine

Condensed sequence (assuming Windows 11 + NVIDIA GPU + admin rights):

```powershell
# In PowerShell (admin):
nvidia-smi                    # verify GPU + driver
wsl --list --online           # see what distros are available
wsl --update                  # update WSL itself
wsl --install -d Ubuntu       # or whatever name appeared in --list
wsl --set-default-version 2
# REBOOT WINDOWS
# After reboot: open Ubuntu from Start menu, set username/password
```

```bash
# Inside Ubuntu:
nvidia-smi                    # verify GPU passthrough
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential cmake git curl wget python3-dev python3-venv python3-pip libomp-dev pkg-config

# CUDA toolkit
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-6
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
nvcc --version

# uv + PyTorch + verify
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
mkdir -p ~/local-ai && cd ~/local-ai
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# Git
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
git config --global init.defaultBranch main
```

Total time on a clean machine: **~45-60 min** (most spent on apt updates and CUDA download).
