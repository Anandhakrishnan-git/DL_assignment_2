"""Minimal training/eval loops used by W&B experiment scripts."""

from __future__ import annotations

import time
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .shared import dice_score_from_logits


@torch.no_grad()
def _update_confusion(
    confusion: torch.Tensor,
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> None:
    preds = preds.detach().view(-1).to(dtype=torch.int64)
    targets = targets.detach().view(-1).to(dtype=torch.int64)
    valid = (targets >= 0) & (targets < num_classes)
    if valid.any():
        encoded = (num_classes * targets[valid] + preds[valid]).cpu()
        bincount = torch.bincount(encoded, minlength=num_classes * num_classes)
        confusion += bincount.view(num_classes, num_classes)


def _mean_iou_from_confusion(confusion: torch.Tensor, eps: float = 1e-6) -> float:
    conf = confusion.to(dtype=torch.float32)
    true_pos = torch.diag(conf)
    false_pos = conf.sum(dim=0) - true_pos
    false_neg = conf.sum(dim=1) - true_pos
    denom = true_pos + false_pos + false_neg
    valid = denom > 0
    if not valid.any():
        return 0.0
    iou = true_pos[valid] / (denom[valid] + eps)
    return float(iou.mean().item())


def train_classification_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    model.train()
    start = time.perf_counter()
    running_loss = 0.0
    running_correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=device.type == "cuda")
        labels = labels.to(device, dtype=torch.long, non_blocking=device.type == "cuda")

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = int(images.size(0))
        running_loss += float(loss.item()) * batch_size
        running_correct += int((torch.argmax(logits, dim=1) == labels).sum().item())
        total += batch_size

    elapsed = time.perf_counter() - start
    return {
        "loss": running_loss / max(1, total),
        "acc": running_correct / max(1, total),
        "sec": elapsed,
    }


@torch.no_grad()
def eval_classification(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    start = time.perf_counter()
    running_loss = 0.0
    running_correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=device.type == "cuda")
        labels = labels.to(device, dtype=torch.long, non_blocking=device.type == "cuda")

        logits = model(images)
        loss = criterion(logits, labels)

        batch_size = int(images.size(0))
        running_loss += float(loss.item()) * batch_size
        running_correct += int((torch.argmax(logits, dim=1) == labels).sum().item())
        total += batch_size

    elapsed = time.perf_counter() - start
    return {
        "loss": running_loss / max(1, total),
        "acc": running_correct / max(1, total),
        "sec": elapsed,
    }


def train_segmentation_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int = 3,
    dice_class_index: int = 1,
    post_train_prepare=None,
) -> Dict[str, float]:
    model.train()
    if post_train_prepare is not None:
        post_train_prepare(model)
    start = time.perf_counter()
    running_loss = 0.0
    total_samples = 0
    pixel_correct = 0
    total_pixels = 0
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    dice_total = 0.0
    dice_batches = 0

    for images, masks in loader:
        images = images.to(device, non_blocking=device.type == "cuda")
        masks = masks.to(device, dtype=torch.long, non_blocking=device.type == "cuda")

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        batch_size = int(images.size(0))
        running_loss += float(loss.item()) * batch_size
        total_samples += batch_size

        preds = torch.argmax(logits, dim=1)
        pixel_correct += int((preds == masks).sum().item())
        total_pixels += int(masks.numel())
        _update_confusion(confusion, preds, masks, num_classes)

        dice_total += float(dice_score_from_logits(logits, masks, class_index=dice_class_index))
        dice_batches += 1

    elapsed = time.perf_counter() - start
    return {
        "loss": running_loss / max(1, total_samples),
        "pix_acc": pixel_correct / max(1, total_pixels),
        "miou": _mean_iou_from_confusion(confusion),
        "dice": dice_total / max(1, dice_batches),
        "sec": elapsed,
    }


@torch.no_grad()
def eval_segmentation(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int = 3,
    dice_class_index: int = 1,
) -> Dict[str, float]:
    model.eval()
    start = time.perf_counter()
    running_loss = 0.0
    total_samples = 0
    pixel_correct = 0
    total_pixels = 0
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    dice_total = 0.0
    dice_batches = 0

    for images, masks in loader:
        images = images.to(device, non_blocking=device.type == "cuda")
        masks = masks.to(device, dtype=torch.long, non_blocking=device.type == "cuda")

        logits = model(images)
        loss = criterion(logits, masks)
        preds = torch.argmax(logits, dim=1)

        batch_size = int(images.size(0))
        running_loss += float(loss.item()) * batch_size
        total_samples += batch_size
        pixel_correct += int((preds == masks).sum().item())
        total_pixels += int(masks.numel())
        _update_confusion(confusion, preds, masks, num_classes)

        dice_total += float(dice_score_from_logits(logits, masks, class_index=dice_class_index))
        dice_batches += 1

    elapsed = time.perf_counter() - start
    return {
        "loss": running_loss / max(1, total_samples),
        "pix_acc": pixel_correct / max(1, total_pixels),
        "miou": _mean_iou_from_confusion(confusion),
        "dice": dice_total / max(1, dice_batches),
        "sec": elapsed,
    }

