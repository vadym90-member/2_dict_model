"""Training loop for ``DictTransformer``.

Kept deliberately compact: a single ``Trainer`` class owns the optimizer,
LR schedule, loss, evaluation, and checkpointing. No external trainer
frameworks (Lightning, HF Trainer, …) are used — everything is plain
PyTorch so the data flow stays inspectable.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .dataset import collate
from .model import DictTransformer
from .tokenizer import PAD_ID


@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 32
    lr: float = 3e-4
    warmup_steps: int = 200
    weight_decay: float = 0.01
    label_smoothing: float = 0.1
    grad_clip: float = 1.0
    log_every: int = 20
    eval_every_epoch: int = 1
    output_dir: str = "runs/exp1"
    device: str = "auto"


def _resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def _lr_lambda(step: int, warmup: int) -> float:
    """Noam-style warmup + inverse-sqrt decay, normalised so the multiplier
    is exactly 1.0 at ``step == warmup`` (peak), then decays as ``step^-0.5``.
    Returns the multiplier to apply to the optimizer's base LR."""
    step = max(step, 1)
    warmup = max(warmup, 1)
    return min(step ** -0.5, step * warmup ** -1.5) * (warmup ** 0.5)


class Trainer:
    def __init__(
        self,
        model: DictTransformer,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        cfg: TrainConfig,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = _resolve_device(cfg.device)
        self.model.to(self.device)
        self.optim = torch.optim.AdamW(
            self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.98), eps=1e-9
        )
        # Noam-style warmup + inverse-sqrt decay. The scheduler returns a
        # multiplier in [0, 1]; the AdamW base LR (``cfg.lr``) sets the peak.
        warmup = cfg.warmup_steps
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optim,
            lr_lambda=lambda step: _lr_lambda(step, warmup),
        )
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_ID, label_smoothing=cfg.label_smoothing)
        self.out_dir = Path(cfg.output_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.history: list[dict] = []
        self.global_step = 0

    # -- per-batch ---------------------------------------------------------

    def _step(self, batch: dict, *, train: bool) -> dict:
        src = batch["src"].to(self.device, non_blocking=True)
        tgt_in = batch["tgt_in"].to(self.device, non_blocking=True)
        tgt_out = batch["tgt_out"].to(self.device, non_blocking=True)
        src_kpm = batch["src_kpm"].to(self.device, non_blocking=True)
        tgt_kpm = batch["tgt_kpm"].to(self.device, non_blocking=True)
        logits = self.model(src, tgt_in, src_kpm=src_kpm, tgt_kpm=tgt_kpm)
        loss = self.loss_fn(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
        if train:
            self.optim.zero_grad(set_to_none=True)
            loss.backward()
            if self.cfg.grad_clip and self.cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.optim.step()
            self.scheduler.step()
            self.global_step += 1
        # Token-level accuracy ignoring PAD.
        with torch.no_grad():
            pred = logits.argmax(dim=-1)
            mask = tgt_out != PAD_ID
            correct = ((pred == tgt_out) & mask).sum().item()
            total = mask.sum().item()
        return {"loss": loss.item(), "correct": correct, "total": total}

    # -- epochs ------------------------------------------------------------

    def _run_epoch(self, loader: DataLoader, *, train: bool) -> dict:
        self.model.train(train)
        agg = {"loss_sum": 0.0, "n_batches": 0, "correct": 0, "total": 0}
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for i, batch in enumerate(loader):
                stats = self._step(batch, train=train)
                agg["loss_sum"] += stats["loss"]
                agg["n_batches"] += 1
                agg["correct"] += stats["correct"]
                agg["total"] += stats["total"]
                if train and (i + 1) % self.cfg.log_every == 0:
                    lr = self.scheduler.get_last_lr()[0]
                    print(
                        f"  step {self.global_step:>5d}  lr {lr:.2e}  "
                        f"loss {stats['loss']:.4f}  "
                        f"acc {stats['correct']/max(stats['total'],1):.3f}"
                    )
        n = max(agg["n_batches"], 1)
        return {
            "loss": agg["loss_sum"] / n,
            "acc": agg["correct"] / max(agg["total"], 1),
        }

    def fit(self) -> dict:
        print(f"[trainer] device={self.device}  params={self.model.num_parameters():,}")
        best_val = math.inf
        for epoch in range(1, self.cfg.epochs + 1):
            t0 = time.time()
            tr = self._run_epoch(self.train_loader, train=True)
            row = {"epoch": epoch, "train_loss": tr["loss"], "train_acc": tr["acc"]}
            if self.val_loader is not None and epoch % self.cfg.eval_every_epoch == 0:
                va = self._run_epoch(self.val_loader, train=False)
                row["val_loss"] = va["loss"]
                row["val_acc"] = va["acc"]
                if va["loss"] < best_val:
                    best_val = va["loss"]
                    self.save("best.pt")
            row["secs"] = round(time.time() - t0, 2)
            self.history.append(row)
            print(
                f"epoch {epoch:>3d}  "
                + "  ".join(f"{k}={v}" for k, v in row.items() if k != "epoch")
            )
            (self.out_dir / "history.json").write_text(json.dumps(self.history, indent=2), encoding="utf-8")
            self.save("last.pt")
        return {"best_val_loss": best_val, "history": self.history}

    # -- checkpoint --------------------------------------------------------

    def save(self, name: str) -> None:
        path = self.out_dir / name
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "model_cfg": vars(self.model.cfg),
                "train_cfg": vars(self.cfg),
                "global_step": self.global_step,
            },
            path,
        )
