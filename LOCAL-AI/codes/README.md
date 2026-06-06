# Code — Local AI Roadmap

Scripts and projects that accompany the daily lessons in `../month*/week*/DAY_*.md`.
The folder layout **mirrors the docs**: code for `month1/week1/DAY_2.md` lives in `month1/week1/day2/`,
so the code for any lesson sits right next to where that lesson is described.

## Structure

```
codes/
└── month1/
    └── week1/
        ├── day1/                  # Install & first run
        │   └── check_setup.sh     # one-shot stack sanity check
        ├── day2/                  # Driving Ollama from Python + benchmarking
        │   ├── benchmark.sh       # TPS benchmark, one model
        │   ├── benchmark_ctx.sh   # TPS across context sizes
        │   ├── 01_basic_chat.py
        │   ├── 02_streaming_chat.py
        │   ├── 03_raw_api.py
        │   ├── 04_openai_compat.py
        │   └── 05_compare_models.py
        ├── day3/                  # Streaming, multi-turn, memory
        │   ├── 06_stream_lib.py
        │   ├── 07_stream_requests.py
        │   ├── 08_stream_openai.py
        │   ├── 08b_stream_sse_raw.py
        │   ├── 09_ttft.py
        │   ├── 10_stateless_proof.py
        │   ├── 11_multiturn.py
        │   ├── 12_system_prompts.py
        │   ├── 13_sampling_intro.py
        │   └── chat.py            # terminal chatbot project
        └── day4/                  # Model parameters (sampling knobs)
            ├── 14_temperature_sweep.py
            ├── 15_seed_demo.py
            ├── 16_repeat_penalty.py
            ├── 17_length_and_stop.py
            ├── playground.py      # sweep any one parameter
            └── presets.py         # factual / balanced / creative presets
```

## Running

These run inside WSL2 (Ubuntu) with Ollama installed and a Python venv:

```bash
cd ~/local-ai && source .venv/bin/activate     # activate venv (every new shell)
uv pip install ollama requests openai           # one-time
python <script>.py
```

The Python venv (`.venv/`) and downloaded models are **not** committed — see `../.gitignore`.
Only source code lives here.

## Hardware context

Developed and benchmarked on an RTX 1000 Ada Laptop GPU (6 GB VRAM) + 32 GB RAM.
See `../Readme.md` for the full hardware profile and the 12-month roadmap.
