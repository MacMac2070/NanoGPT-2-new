"""Training script for the paper-faithful decoder-only GPT.

Optimisation follows section 5.3: Adam (not AdamW, the paper uses no weight
decay) with beta1=0.9, beta2=0.98, eps=1e-9 and the "Noam" schedule,

    lrate = d_model^-0.5 * min(step^-0.5, step * warmup_steps^-1.5)

which warms the learning rate up linearly for warmup_steps and then decays
it as 1/sqrt(step). The warmup matters here: a post-norm stack of depth 6
trains less smoothly than pre-norm without it. A constant-LR fallback is
available via config.use_noam_schedule = False, for debugging only.

Usage:
    python train.py            # full training run
    python train.py --dry-run  # one batch + one forward pass, then stop
"""

import argparse
import math
import os
import time

import torch

import config
import data
from model import GPT


def build_model(vocab_size: int) -> GPT:
    """Construct the GPT purely from config plus the data-derived vocab."""
    return GPT(
        vocab_size=vocab_size,
        n_embd=config.n_embd,
        n_layer=config.n_layer,
        n_head=config.n_head,
        block_size=config.block_size,
        d_ff=config.d_ff,
        dropout=config.dropout,
    ).to(config.DEVICE)


@torch.no_grad()
def estimate_loss(model, train_data, val_data):
    """Average the loss over eval_iters random batches for each split."""
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(config.eval_iters)
        for i in range(config.eval_iters):
            x, y = data.get_batch(split, train_data, val_data,
                                  config.block_size, config.batch_size,
                                  config.DEVICE)
            _, loss = model(x, y)
            losses[i] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def save_checkpoint(model, optimizer, scheduler, iteration, vocab_size, tokenizer):
    """Save weights plus everything generate.py needs to rebuild the model,
    so a later edit to config.py cannot silently break generation. The BPE
    tokenizer is embedded as a JSON string, keeping the checkpoint
    self-contained even if tokenizer/bpe.json is deleted.

    Written atomically: torch.save goes to a temporary file in the same
    directory, then os.replace() swaps it onto the final path, so a kill or
    disk-full mid-write cannot corrupt the checkpoint that's already there."""
    config.CKPT_DIR.mkdir(parents=True, exist_ok=True)
    path = config.CKPT_DIR / "ckpt.pt"
    tmp_path = config.CKPT_DIR / "ckpt.pt.tmp"
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "iteration": iteration,
        "hyperparams": {
            "vocab_size": vocab_size,
            "n_embd": config.n_embd,
            "n_layer": config.n_layer,
            "n_head": config.n_head,
            "block_size": config.block_size,
            "d_ff": config.d_ff,
            "dropout": config.dropout,
        },
        "tokenizer_json": tokenizer.to_str(),
    }, tmp_path)
    os.replace(tmp_path, path)
    return path


def main():
    parser = argparse.ArgumentParser(description="Train the paper GPT")
    parser.add_argument("--dry-run", action="store_true",
                        help="verify data + one forward pass, do NOT train")
    parser.add_argument("--resume", action="store_true",
                        help="resume from checkpoints/ckpt.pt instead of starting fresh")
    args = parser.parse_args()

    torch.manual_seed(config.seed)

    print(f"device: {config.DEVICE}")
    tokenizer, vocab_size, train_data, val_data = data.load_dataset()
    print(f"vocab_size: {vocab_size}")
    print(f"train: {len(train_data):,} tokens  val: {len(val_data):,} tokens")

    model = build_model(vocab_size)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model parameters: {n_params:,}")

    x, y = data.get_batch("train", train_data, val_data,
                          config.block_size, config.batch_size, config.DEVICE)

    if args.dry_run:
        # Dry verification only: shapes and a finite loss, no optimiser, no
        # scheduler, no training loop.
        print(f"batch shapes: x {tuple(x.shape)}  y {tuple(y.shape)}")
        logits, loss = model(x, y)
        expected = (config.batch_size, config.block_size, vocab_size)
        assert logits.shape == expected, (
            f"logits shape {tuple(logits.shape)} != {expected}"
        )
        assert torch.isfinite(loss), f"loss is not finite: {loss.item()}"
        print(f"logits shape: {tuple(logits.shape)}")
        print(f"loss: {loss.item():.4f} (uniform baseline ln({vocab_size}) = "
              f"{math.log(vocab_size):.4f})")
        print("note: with section 3.4 weight tying the untrained model "
              "predicts the input token itself\n(the residual stream "
              "still points at its embedding), so the starting loss sits "
              "well above\nthe uniform baseline. The section 5.3 warmup "
              "trains through this in the first few hundred steps.")
        print("DRY RUN complete: no optimiser created, no training performed.")
        return

    # Section 5.3 optimiser. lr=1.0 is a base multiplied by the schedule.
    if config.use_noam_schedule:
        optimizer = torch.optim.Adam(model.parameters(), lr=1.0,
                                     betas=config.adam_betas,
                                     eps=config.adam_eps)
        def noam(step: int) -> float:
            step = step + 1  # LambdaLR starts at 0; the formula needs step >= 1
            return config.n_embd ** -0.5 * min(
                step ** -0.5, step * config.warmup_steps ** -1.5
            )
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, noam)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=config.constant_lr,
                                     betas=config.adam_betas,
                                     eps=config.adam_eps)
        scheduler = None

    start_iteration = 0
    if args.resume:
        ckpt_path = config.CKPT_DIR / "ckpt.pt"
        if not ckpt_path.is_file():
            raise SystemExit(
                f"--resume given but no checkpoint found at {ckpt_path}; "
                f"run train.py without --resume first"
            )
        ckpt = torch.load(ckpt_path, weights_only=True, map_location=config.DEVICE)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if scheduler is not None and ckpt.get("scheduler") is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_iteration = ckpt["iteration"]
        print(f"resumed from {ckpt_path} at iteration {start_iteration}")

    print(f"schedule: {'Noam warmup' if scheduler else 'constant lr'}")
    model.train()
    t0 = time.time()

    for iteration in range(start_iteration, config.max_iters):
        if iteration % config.eval_interval == 0 or iteration == config.max_iters - 1:
            losses = estimate_loss(model, train_data, val_data)
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"iter {iteration:5d} | train {losses['train']:.4f} | "
                  f"val {losses['val']:.4f} | lr {lr_now:.2e} | "
                  f"{time.time() - t0:.0f}s")
            path = save_checkpoint(model, optimizer, scheduler, iteration,
                                   vocab_size, tokenizer)

        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        # Fetch the next batch after stepping, so the loss above always
        # corresponds to the batch it was computed on.
        x, y = data.get_batch("train", train_data, val_data,
                              config.block_size, config.batch_size,
                              config.DEVICE)

    path = save_checkpoint(model, optimizer, scheduler, config.max_iters,
                           vocab_size, tokenizer)
    print(f"training complete. checkpoint: {path}")


if __name__ == "__main__":
    main()
