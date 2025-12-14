from __future__ import annotations

from typing import Dict


def compute_progress(global_step: int, total_steps: int) -> float:
    denom = max(1, total_steps - 1)
    return max(0.0, min(1.0, global_step / denom))


def lambda_distill(progress: float, cfg: Dict) -> float:
    if not cfg.get("enable", False):
        return cfg.get("lambda_max", 0.0)

    warmup = cfg.get("warmup_frac", 0.0)
    ramp_end = cfg.get("ramp_end_frac", warmup)
    ramp_power = cfg.get("ramp_power", 1.0)
    tail_start = cfg.get("tail_start_frac", 1.0)
    lambda_max = cfg.get("lambda_max", 0.0)
    tail_decay = cfg.get("tail_decay", False)

    if progress < warmup:
        return 0.0
    if progress < ramp_end and ramp_end > warmup:
        scaled = (progress - warmup) / max(ramp_end - warmup, 1e-6)
        return lambda_max * (scaled**ramp_power)
    if tail_decay and progress > tail_start:
        remain = max(0.0, 1.0 - progress)
        tail = max(1e-6, 1.0 - tail_start)
        return lambda_max * (remain / tail)
    return lambda_max


def layer_weight(layer_idx: int, progress: float, curriculum_cfg: Dict, layers: list[int]) -> float:
    if not curriculum_cfg.get("enable", False):
        return 1.0

    if layer_idx not in layers:
        return 0.0

    layer_window = curriculum_cfg.get("layer_curriculum", {}).get(layer_idx)
    if layer_window is None:
        return 1.0

    start, end = layer_window
    if progress < start:
        return 0.0
    if end <= start:
        return 1.0
    if progress >= end:
        return 1.0

    scale = (progress - start) / max(end - start, 1e-6)
    return max(0.0, min(1.0, scale))


def validate_curriculum_config(cfg: Dict) -> None:
    warmup = cfg.get("warmup_frac", 0.0)
    ramp_end = cfg.get("ramp_end_frac", warmup)
    tail_start = cfg.get("tail_start_frac", 1.0)
    for key in (warmup, ramp_end, tail_start):
        if not 0.0 <= key <= 1.0:
            raise ValueError("Curriculum fractions must be in [0,1].")
    if warmup > ramp_end or ramp_end > tail_start:
        raise ValueError("Require warmup <= ramp_end <= tail_start <= 1.0")
