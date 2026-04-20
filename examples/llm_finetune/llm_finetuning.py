# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""LLM fine-tuning with SPDL data loading pipeline.

Fine-tunes LLaMA 3.2 1B on Alpaca-style instruction data using LoRA,
with SPDL PipelineBuilder for high-performance concurrent tokenization.

SPDL Data Pipeline
^^^^^^^^^^^^^^^^^^

The core of this example is the SPDL data loading pipeline:

1. ``DistributedRandomSampler`` -- distributes sample indices across ranks
   with per-epoch reshuffling
2. ``pipe(tokenize, concurrency=N)`` -- concurrent Alpaca-format prompt
   formatting and tokenization
3. ``aggregate(batch_size)`` -- groups into batches
4. ``pipe(collate)`` -- stacks tensors
5. ``add_sink(buffer_size=3)`` -- prefetch buffer for the training loop

Data
^^^^

Download instruction-following datasets::

    # https://github.com/tatsu-lab/stanford_alpaca
    python download_alpaca.py --output /tmp/alpaca.jsonl
    # https://huggingface.co/datasets/databricks/databricks-dolly-15k
    python download_dolly.py --output /tmp/dolly.jsonl

Data format (JSONL with Alpaca-style fields)::

    {"instruction": "Explain what a linked list is.", "input": "", "output": "A linked list is..."}

Usage
^^^^^

::

    torchrun \\
      --nproc_per_node 8 \\
      -m spdl.examples.llm_finetune.llm_finetuning \\
      --model-path /path/to/Llama-3.2-1B-Instruct \\
      --data-path \\
        /tmp/alpaca.jsonl \\
        /tmp/dolly.jsonl

With the default settings (global batch size 8x32), the training throughput reaches roughly ~570
samples on H100 GPUs.
"""

from __future__ import annotations

__all__ = [
    "build_model",
    "build_pytorch_dataloader",
    "build_spdl_dataloader",
    "load_data",
    "main",
    "train",
]

# pyre-strict

import argparse
import logging
import os
import time
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    from examples.llm_finetune.utils.dataloader import (  # pyre-ignore[21]
        build_pytorch_dataloader,
    )
    from examples.llm_finetune.utils.pipeline import (  # pyre-ignore[21]
        build_spdl_dataloader,
    )
    from examples.llm_finetune.utils.utils import (  # pyre-ignore[21]
        load_data,
        report_progress,
        resolve_model_path,
    )
except ImportError:
    from spdl.examples.llm_finetune.utils.dataloader import build_pytorch_dataloader
    from spdl.examples.llm_finetune.utils.pipeline import (
        build_spdl_dataloader,
    )
    from spdl.examples.llm_finetune.utils.utils import (
        load_data,
        report_progress,
        resolve_model_path,
    )

_LG: logging.Logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------


def build_model(
    model_path: str,
    device: torch.device,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
) -> torch.nn.Module:
    """Load LLaMA model and apply LoRA."""
    from peft import get_peft_model, LoraConfig, TaskType
    from transformers import AutoModelForCausalLM

    _LG.info("Loading model from %s", model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    model = model.to(device)
    return model


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(
    *,
    model_path: str,
    data_path: list[str],
    output_dir: str,
    max_seq_len: int,
    batch_size: int,
    num_epochs: int,
    lr: float,
    weight_decay: float,
    max_grad_norm: float,
    log_interval: int,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    num_workers: int,
    dataloader: str = "spdl",
    mp_context: str = "forkserver",
    progress_fn: Callable[[int, int], None] | None = None,
) -> None:
    """Main training function, called per-rank."""
    rank: int = dist.get_rank()
    world_size: int = dist.get_world_size()
    local_rank: int = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    _LG.info(
        "Rank %d/%d on device %s (dataloader=%s)",
        rank,
        world_size,
        device,
        dataloader,
    )

    # --- Data ---
    samples = load_data(data_path)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Model ---
    model = build_model(
        model_path,
        device,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )
    ddp_model = DDP(model, device_ids=[local_rank])

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(
        ddp_model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        foreach=True,
    )

    num_steps_per_epoch = len(samples) // (batch_size * world_size)
    total_steps = num_steps_per_epoch * num_epochs
    if rank == 0:
        _LG.info(
            "Training: %d samples, %d epochs, %d steps/epoch, %d total steps",
            len(samples),
            num_epochs,
            num_steps_per_epoch,
            total_steps,
        )
        if progress_fn is not None:
            progress_fn(0, total_steps)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=lr * 0.1,
    )

    # --- Build data source ---
    if dataloader == "pytorch":
        dl = build_pytorch_dataloader(
            samples=samples,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            batch_size=batch_size,
            rank=rank,
            world_size=world_size,
            num_workers=num_workers,
            mp_context=mp_context,
            device=device,
        )
    else:
        dl = build_spdl_dataloader(
            samples=samples,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            batch_size=batch_size,
            rank=rank,
            world_size=world_size,
            num_threads=num_workers,
            mp_context=mp_context,
        )

    # --- Training loop ---
    global_step = 0
    ddp_model.train()

    for epoch in range(num_epochs):
        _LG.info("Epoch %d/%d", epoch + 1, num_epochs)

        t0 = time.monotonic()
        epoch_loss = 0.0
        num_batches = 0

        for batch in dl:
            outputs = ddp_model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1

            if rank == 0:
                if progress_fn is not None:
                    progress_fn(global_step, total_steps)
                if global_step % log_interval == 0:
                    avg_loss = epoch_loss / num_batches
                    elapsed = time.monotonic() - t0
                    _LG.info(
                        "Step %d | loss=%.4f | lr=%.2e | %.1f samples/s",
                        global_step,
                        avg_loss,
                        scheduler.get_last_lr()[0],
                        num_batches * batch_size * world_size / elapsed,
                    )

        elapsed = time.monotonic() - t0
        if rank == 0:
            avg_loss = epoch_loss / max(num_batches, 1)
            _LG.info(
                "Epoch %d complete | avg_loss=%.4f | %.1fs | %.1f samples/s",
                epoch + 1,
                avg_loss,
                elapsed,
                num_batches * batch_size * world_size / elapsed,
            )

    # --- Save ---
    if rank == 0 and output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        ddp_model.module.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
        _LG.info("Model saved to %s", output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    # Model
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to pretrained LLaMA model directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Directory to save fine-tuned LoRA weights",
    )
    # Data
    parser.add_argument(
        "--data-path",
        type=str,
        nargs="+",
        required=True,
        help="One or more paths to Alpaca-format JSONL files (local or manifold://).",
    )
    parser.add_argument("--max-seq-len", type=int, default=512)
    # Training
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=10)
    # LoRA
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    # Pipeline
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Concurrent tokenization workers in the data pipeline",
    )
    parser.add_argument(
        "--dataloader",
        type=str,
        choices=["spdl", "pytorch"],
        default="spdl",
        help="Data loading backend: 'spdl' (default) or 'pytorch' (torch DataLoader)",
    )
    parser.add_argument(
        "--mp-context",
        type=str,
        choices=["fork", "spawn", "forkserver"],
        default="forkserver",
        help="Multiprocessing context for workers (default: forkserver)",
    )
    return parser.parse_args()


def init_logging() -> None:
    """Initialize logging."""
    rank = os.environ.get("RANK", "?")
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [%(levelname).1s] [Rank{rank}] %(name)s: %(message)s",
    )


def main(args: argparse.Namespace) -> None:
    dist.init_process_group(backend="nccl", timeout=timedelta(minutes=3))
    try:
        train(
            model_path=resolve_model_path(args.model_path),
            data_path=args.data_path,
            output_dir=args.output_dir,
            max_seq_len=args.max_seq_len,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            max_grad_norm=args.max_grad_norm,
            log_interval=args.log_interval,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            num_workers=args.num_workers,
            dataloader=args.dataloader,
            mp_context=args.mp_context,
            progress_fn=report_progress,
        )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    init_logging()
    main(parse_args())
