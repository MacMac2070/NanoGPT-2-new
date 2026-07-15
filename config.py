"""Configuration for the paper-faithful decoder-only GPT.

Single source of truth for hyperparameters, device and paths.
Model dimensions follow the base model of "Attention Is All You Need"
(Vaswani et al. 2017), Table 3: d_model=512, N=6, h=8, d_ff=2048, P_drop=0.1.

Note: vocab_size deliberately does NOT live here. It comes from the BPE
tokenizer trained on the corpus by data.py (get_tokenizer), never hardcoded;
bpe_vocab_size below is only the trainer's target.
"""

from pathlib import Path

import torch

# Paths are anchored to this file so every script works from any working
# directory.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = _PROJECT_ROOT / "files" / "input.txt"
CKPT_DIR = Path(__file__).resolve().parent / "checkpoints"
# The trained BPE tokenizer (see data.py). Saved once, then reused by both
# train.py and generate.py so they always tokenise identically.
TOKENIZER_PATH = Path(__file__).resolve().parent / "tokenizer" / "bpe.json"

# Device auto-select: cuda > mps > cpu.
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

# ---------------------------------------------------------------------------
# EDIT HYPERPARAMETERS HERE
# ---------------------------------------------------------------------------
# Architecture (paper terms in brackets)
n_embd = 512          # d_model, the width of every token vector
n_layer = 6           # N, how many blocks are stacked (section 3.1)
n_head = 8            # h, attention heads per block (section 3.2.2)
block_size = 256      # maximum context length in tokens (subwords here)
d_ff = 4 * n_embd     # feed-forward inner width, 2048 (section 3.3)
dropout = 0.1         # P_drop (section 5.4)

# Tokenizer: target vocabulary for the BPE trained on the corpus. The model
# always sizes itself from tokenizer.get_vocab_size(), never this constant.
bpe_vocab_size = 5000

# Training
batch_size = 64
max_iters = 5000
eval_interval = 250   # estimate train/val loss (and checkpoint) this often
eval_iters = 200      # batches averaged per loss estimate
seed = 1337

# Optimiser (section 5.3): Adam with the "Noam" warmup schedule,
# lrate = d_model^-0.5 * min(step^-0.5, step * warmup_steps^-1.5)
adam_betas = (0.9, 0.98)
adam_eps = 1e-9
warmup_steps = 4000
use_noam_schedule = True   # set False to fall back to a constant learning rate
constant_lr = 3e-4         # only used when use_noam_schedule is False
# ---------------------------------------------------------------------------

# Each head handles n_embd // n_head dimensions (d_k = d_v = d_model / h),
# so the width must divide evenly across heads.
assert n_embd % n_head == 0, (
    f"n_embd ({n_embd}) must be divisible by n_head ({n_head})"
)
