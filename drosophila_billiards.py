#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
DIGITAL DROSOPHILA PLAYS BILLIARDS — ULTIMATE EXTENDED EDITION
================================================================================
Оцифрованный мозг дрозофилы (реальный коннектом FlyWire FAFB v783 или
биологически-инспирированная синтетическая сеть) учится играть в 2D-бильярд.

Возможности:
  * Среды: базовый бильярд, бильярд с препятствиями, с пауэрапами, с
    несколькими типами шаров (обычный, тяжёлый, лёгкий).
  * Мозги: SyntheticFlyBrain (PN→AL→KC→MBON), ConnectomeFlyBrain (FlyWire),
    HybridFlyBrain (коннектом + обучаемый readout).
  * Алгоритмы RL: REINFORCE + baseline, A2C, PPO (clipped), DQN (для
    дискретного управления).
  * Живая графика во время обучения:
      - Pygame-окно с игрой (60 FPS);
      - Matplotlib-дашборд 2×3: игра, растр спайков, мембраны, гистограмма
        весов, кривые reward/potted, реалтайм-телеметрия.
  * Callbacks: EarlyStopping, ModelCheckpoint, LRScheduler, TensorBoard.
  * Метрики и экспорт: CSV, JSON, PNG, GIF, PT.
  * CLI на подкомандах (train, eval, play, export, info).
  * Тесты (unittest) для физики и градиентов.

Запуск:
    python drosophila_billiards.py train --algo ppo --brain synthetic
    python drosophila_billiards.py play  --checkpoint runs/best.pt
    python drosophila_billiards.py test

Лицензия: MIT
================================================================================
"""

# =============================================================================
# SECTION 1 — IMPORTS
# =============================================================================
from __future__ import annotations

import os
import sys
import csv
import json
import time
import math
import copy
import random
import argparse
import logging
import warnings
import unittest
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict, fields
from typing import (
    Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union,
)
from pathlib import Path
from collections import deque, defaultdict, OrderedDict
from contextlib import contextmanager

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.distributions import Normal, Categorical, Bernoulli
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    raise

import matplotlib
if "--no-gui" in sys.argv or "play" in sys.argv and "--no-gui" in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle, Rectangle, Polygon, FancyArrow
from matplotlib.collections import LineCollection
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

try:
    import pygame
    _HAS_PYGAME = True
except ImportError:
    _HAS_PYGAME = False

# --- Опциональные зависимости для коннектома ---
try:
    import torch_geometric  # noqa: F401
    from flywire_gnn import FlyWireFAFB  # type: ignore
    _HAS_FLYWIRE = True
    _FLYWIRE_IMPORT_ERROR = ""
except Exception as _e:
    _HAS_FLYWIRE = False
    _FLYWIRE_IMPORT_ERROR = str(_e)

try:
    from fafbseg import flywire as _fafbseg_flywire  # type: ignore
    _HAS_FAFBSEG = True
except Exception:
    _HAS_FAFBSEG = False

try:
    from torch.utils.tensorboard import SummaryWriter  # type: ignore
    _HAS_TB = True
except Exception:
    _HAS_TB = False


# =============================================================================
# SECTION 2 — CONFIG
# =============================================================================
@dataclass
class EnvConfig:
    """Параметры среды."""
    table_w: float = 400.0
    table_h: float = 200.0
    pocket_r: float = 14.0
    ball_r: float = 6.0
    fly_r: float = 5.0
    n_balls: int = 3
    n_heavy_balls: int = 0
    n_light_balls: int = 0
    n_obstacles: int = 0
    obstacle_r: float = 10.0
    n_powerups: int = 0
    powerup_r: float = 7.0
    friction_fly: float = 0.90
    friction_ball: float = 0.96
    wall_bounce: float = 0.8
    ball_bounce: float = 0.95
    fly_acc: float = 0.5
    fly_mass: float = 2.0
    ball_mass: float = 1.0
    fly_max_speed: float = 4.0
    ball_max_speed: float = 8.0
    max_steps: int = 200
    physics_substeps: int = 2
    gravity: float = 0.0  # для будущих расширений
    enable_obstacles: bool = False
    enable_powerups: bool = False


@dataclass
class RewardConfig:
    """Параметры награды."""
    reward_pot: float = 10.0
    reward_shaping: float = 0.05
    reward_time_penalty: float = 0.01
    reward_collision: float = 0.05
    reward_powerup: float = 2.0
    reward_wall_hit_penalty: float = 0.02
    reward_energy_penalty: float = 0.001


@dataclass
class BrainConfig:
    """Параметры мозга."""
    mode: str = "synthetic"  # synthetic | connectome | hybrid
    snn_timesteps: int = 4
    tau: float = 2.0
    v_th: float = 1.0
    tau_syn: float = 5.0
    v_reset: float = 0.0
    # synthetic
    n_sensory: int = 30
    n_al: int = 30
    n_kc: int = 120
    n_mbon: int = 24
    # connectome
    connectome_max_neurons: int = 600
    connectome_min_synapses: int = 5
    connectome_cache_dir: str = "~/.cache/flywire_gnn"
    connectome_include_types: Tuple[str, ...] = (
        "KC", "MBON", "PN", "AL", "APL", "DAN", "GABA",
    )
    connectome_target_region: str = "mushroom_body"
    # hybrid
    hybrid_frozen_layers: int = 2


@dataclass
class AlgoConfig:
    """Параметры RL-алгоритма."""
    algo: str = "ppo"  # reinforce | a2c | ppo | dqn
    gamma: float = 0.98
    gae_lambda: float = 0.95
    lr_brain: float = 1e-3
    lr_critic: float = 1e-3
    entropy_coef: float = 0.02
    value_coef: float = 0.5
    grad_clip: float = 1.0
    ppo_clip: float = 0.2
    ppo_epochs: int = 4
    ppo_minibatch: int = 64
    dqn_buffer: int = 50000
    dqn_batch: int = 64
    dqn_target_update: int = 200
    dqn_eps_start: float = 1.0
    dqn_eps_end: float = 0.05
    dqn_eps_decay: int = 5000
    dqn_n_actions: int = 8  # для дискретного управления


@dataclass
class TrainConfig:
    """Параметры обучения."""
    n_episodes: int = 2000
    log_every: int = 10
    save_every: int = 200
    eval_every: int = 100
    eval_episodes: int = 5
    early_stop_patience: int = 20  # eval без улучшения
    early_stop_min_delta: float = 0.01
    lr_scheduler: str = "none"  # none | step | cosine
    lr_step_size: int = 500
    lr_gamma: float = 0.5
    seed: int = 42
    device: str = "cpu"


@dataclass
class VizConfig:
    """Параметры визуализации."""
    live_game_window: bool = True
    live_dashboard: bool = True
    live_dashboard_update_every: int = 5  # эпизодов
    live_game_fps: int = 60
    live_game_scale: int = 2
    matplotlib_interval_ms: int = 40
    n_frames_final: int = 200
    show_spikes: bool = True
    show_membrane: bool = True
    show_weights: bool = True
    show_value_function: bool = False
    save_gif: bool = False
    gif_path: str = "drosophila_play.gif"
    save_mp4: bool = False
    mp4_path: str = "drosophila_play.mp4"


@dataclass
class Config:
    """Общая конфигурация проекта."""
    env: EnvConfig = field(default_factory=EnvConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    brain: BrainConfig = field(default_factory=BrainConfig)
    algo: AlgoConfig = field(default_factory=AlgoConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    viz: VizConfig = field(default_factory=VizConfig)
    output_dir: str = "runs/drosophila_billiards"
    run_name: str = ""
    log_level: str = "INFO"
    resume_from: Optional[str] = None
    dry_run: bool = False

    # -------------------------------------------------------------- сериализация
    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)

    @classmethod
    def from_json(cls, s: str) -> "Config":
        d = json.loads(s)
        return cls(
            env=EnvConfig(**d.get("env", {})),
            reward=RewardConfig(**d.get("reward", {})),
            brain=BrainConfig(**d.get("brain", {})),
            algo=AlgoConfig(**d.get("algo", {})),
            train=TrainConfig(**d.get("train", {})),
            viz=VizConfig(**d.get("viz", {})),
            output_dir=d.get("output_dir", "runs/drosophila_billiards"),
            run_name=d.get("run_name", ""),
            log_level=d.get("log_level", "INFO"),
            resume_from=d.get("resume_from"),
            dry_run=d.get("dry_run", False),
        )

    def save(self, path: Union[str, Path]) -> None:
        p = Path(path).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_json(), encoding="utf-8")


# =============================================================================
# SECTION 3 — LOGGING & UTILS
# =============================================================================
def setup_logging(cfg: Config) -> logging.Logger:
    out = Path(cfg.output_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "train.log"

    logger = logging.getLogger("drosophila")
    logger.setLevel(getattr(logging, cfg.log_level.upper(), logging.INFO))
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-7s %(message)s", datefmt="%H:%M:%S"
    )
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


LOG = logging.getLogger("drosophila")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def safe_mkdir(p: Union[str, Path]) -> Path:
    p = Path(p).expanduser()
    p.mkdir(parents=True, exist_ok=True)
    return p


def human_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(seconds, 60)
    if m < 60:
        return f"{int(m)}m {s:.0f}s"
    h, m = divmod(m, 60)
    return f"{int(h)}h {int(m)}m {s:.0f}s"


def save_checkpoint(state: Dict[str, Any], path: Union[str, Path]) -> None:
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    LOG.info(f"Checkpoint saved: {path}")


def load_checkpoint(path: Union[str, Path], map_location: str = "cpu") -> Dict[str, Any]:
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location=map_location, weights_only=False)


def discounted_returns(rewards: Sequence[float], gamma: float) -> List[float]:
    G = 0.0
    out = []
    for r in reversed(rewards):
        G = r + gamma * G
        out.insert(0, G)
    return out


def compute_gae(
    rewards: Sequence[float],
    values: Sequence[float],
    dones: Sequence[bool],
    gamma: float,
    lam: float,
) -> Tuple[List[float], List[float]]:
    """Generalized Advantage Estimation."""
    T = len(rewards)
    adv = [0.0] * T
    last = 0.0
    for t in reversed(range(T)):
        nonterminal = 0.0 if dones[t] else 1.0
        next_v = values[t + 1] if t + 1 < T else 0.0
        delta = rewards[t] + gamma * next_v * nonterminal - values[t]
        last = delta + gamma * lam * nonterminal * last
        adv[t] = last
    returns = [adv[t] + values[t] for t in range(T)]
    return adv, returns


def linear_schedule(start: float, end: float, cur: int, total: int) -> float:
    frac = min(max(cur / max(total, 1), 0.0), 1.0)
    return start + frac * (end - start)


def exponential_schedule(start: float, end: float, cur: int, decay: int) -> float:
    if cur >= decay:
        return end
    return end + (start - end) * math.exp(-5.0 * cur / max(decay, 1))


@contextmanager
def timer(name: str = ""):
    t0 = time.time()
    yield
    LOG.debug(f"{name}: {time.time()-t0:.3f}s")


# =============================================================================
# SECTION 4 — ENVIRONMENT
# =============================================================================
class BallType:
    NORMAL = "normal"
    HEAVY = "heavy"
    LIGHT = "light"
    CUE = "cue"

    MASS = {NORMAL: 1.0, HEAVY: 2.5, LIGHT: 0.5, CUE: 1.0}
    COLOR = {
        NORMAL: (240, 240, 240),
        HEAVY: (200, 60, 60),
        LIGHT: (60, 200, 120),
        CUE: (255, 255, 255),
    }


class Powerup:
    """Пауэрап: ускоряет муху или замораживает шары."""

    TYPE_SPEED = "speed"
    TYPE_FREEZE = "freeze"
    TYPE_MAGNET = "magnet"

    COLORS = {
        TYPE_SPEED: (255, 200, 0),
        TYPE_FREEZE: (100, 200, 255),
        TYPE_MAGNET: (255, 100, 200),
    }

    def __init__(self, pos: np.ndarray, kind: str, duration: float = 30.0):
        self.pos = pos
        self.kind = kind
        self.duration = duration
        self.timer = 0.0
        self.active = False


class BilliardsEnv:
    """
    Расширенный 2D-бильярд.

    Особенности:
      - несколько типов шаров (обычные, тяжёлые, лёгкие);
      - статические препятствия (круги);
      - пауэрапы (ускорение, заморозка, магнит);
      - под-шаговый интегратор для устойчивости физики.

    Наблюдение (базовое, 28 + добавки):
      [0:2]   позиция мухи (норм.)
      [2:4]   скорость мухи (норм.)
      [4:4+2B] относительные позиции шаров (норм.)
      [..+2B] скорости шаров (норм.)
      [..+12] относительные позиции 6 луз (норм.)
      + 8 на тип шара (one-hot усреднённый)
      + 6 на препятствия (если есть)
      + 6 на пауэрапы (если есть)
    """

    # ---------------------------------------------------------------- init
    def __init__(self, env_cfg: EnvConfig, reward_cfg: RewardConfig):
        self.cfg = env_cfg
        self.rc = reward_cfg

        self.W = env_cfg.table_w
        self.H = env_cfg.table_h
        self.POCKET_R = env_cfg.pocket_r
        self.BALL_R = env_cfg.ball_r
        self.FLY_R = env_cfg.fly_r
        self.N_BALLS = env_cfg.n_balls
        self.FRICTION_FLY = env_cfg.friction_fly
        self.FRICTION_BALL = env_cfg.friction_ball
        self.WALL_BOUNCE = env_cfg.wall_bounce
        self.BALL_BOUNCE = env_cfg.ball_bounce
        self.FLY_ACC = env_cfg.fly_acc
        self.FLY_MASS = env_cfg.fly_mass
        self.BALL_MASS = env_cfg.ball_mass
        self.FLY_MAX_SPEED = env_cfg.fly_max_speed
        self.BALL_MAX_SPEED = env_cfg.ball_max_speed
        self.MAX_STEPS = env_cfg.max_steps
        self.SUBSTEPS = max(1, env_cfg.physics_substeps)

        self._rng = np.random.RandomState(42)

        self.pockets = np.array([
            [0.0, 0.0], [self.W / 2, 0.0], [self.W, 0.0],
            [0.0, self.H], [self.W / 2, self.H], [self.W, self.H],
        ], dtype=np.float32)

        # типы шаров
        self.ball_types = self._make_ball_types()

        self.obstacles: List[np.ndarray] = []
        self.powerups: List[Powerup] = []

        self._effects = {
            "speed_timer": 0.0,
            "freeze_timer": 0.0,
            "magnet_timer": 0.0,
        }

        self.reset()

    def _make_ball_types(self) -> List[str]:
        types = [BallType.NORMAL] * self.N_BALLS
        n_h = min(self.cfg.n_heavy_balls, self.N_BALLS)
        n_l = min(self.cfg.n_light_balls, self.N_BALLS - n_h)
        idx = list(range(self.N_BALLS))
        self._rng.shuffle(idx)
        for i in range(n_h):
            types[idx[i]] = BallType.HEAVY
        for i in range(n_h, n_h + n_l):
            types[idx[i]] = BallType.LIGHT
        return types

    def _make_obstacles(self) -> None:
        self.obstacles = []
        if not self.cfg.enable_obstacles:
            return
        for _ in range(self.cfg.n_obstacles):
            for _a in range(50):
                p = np.array([
                    self._rng.uniform(60, self.W - 60),
                    self._rng.uniform(40, self.H - 40),
                ], dtype=np.float32)
                # не пересекается с лузами и стартовой позицией мухи
                if min(np.linalg.norm(self.pockets - p, axis=1)) > self.cfg.obstacle_r + self.POCKET_R + 10:
                    break
            self.obstacles.append(p)

    def _make_powerups(self) -> None:
        self.powerups = []
        if not self.cfg.enable_powerups:
            return
        kinds = [Powerup.TYPE_SPEED, Powerup.TYPE_FREEZE, Powerup.TYPE_MAGNET]
        for i in range(self.cfg.n_powerups):
            p = np.array([
                self._rng.uniform(40, self.W - 40),
                self._rng.uniform(30, self.H - 30),
            ], dtype=np.float32)
            self.powerups.append(Powerup(p, kinds[i % len(kinds)]))

    # ---------------------------------------------------------------- reset
    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            self._rng = np.random.RandomState(seed)

        self.fly_pos = np.array([50.0, self.H / 2], dtype=np.float32)
        self.fly_vel = np.zeros(2, dtype=np.float32)

        self.ball_types = self._make_ball_types()
        balls_pos = []
        for _ in range(self.N_BALLS):
            for _a in range(200):
                p = np.array([
                    self._rng.uniform(self.BALL_R + 20, self.W - self.BALL_R - 20),
                    self._rng.uniform(self.BALL_R + 20, self.H - self.BALL_R - 20),
                ], dtype=np.float32)
                if np.linalg.norm(p - self.fly_pos) > 60:
                    ok = True
                    for q in balls_pos:
                        if np.linalg.norm(p - q) < 3 * self.BALL_R:
                            ok = False
                            break
                    if ok:
                        break
            balls_pos.append(p)
        self.balls_pos = np.array(balls_pos, dtype=np.float32)
        self.balls_vel = np.zeros((self.N_BALLS, 2), dtype=np.float32)
        self.potted = np.zeros(self.N_BALLS, dtype=bool)

        self._make_obstacles()
        self._make_powerups()

        self._effects = {"speed_timer": 0.0, "freeze_timer": 0.0, "magnet_timer": 0.0}

        self.steps = 0
        self.prev_dist = self._total_dist()
        self._episode_reward = 0.0
        self._potted_count = 0
        self._wall_hits = 0
        self._collisions = 0
        return self.get_obs()

    # -------------------------------------------------------------- helpers
    def _total_dist(self) -> float:
        d = 0.0
        for i in range(self.N_BALLS):
            if self.potted[i]:
                continue
            d += float(np.min(np.linalg.norm(self.pockets - self.balls_pos[i], axis=1)))
        return d

    def _ball_mass(self, i: int) -> float:
        return BallType.MASS[self.ball_types[i]]

    def get_obs(self) -> np.ndarray:
        pockets_rel = (self.pockets - self.fly_pos) / self.W
        balls_rel = (self.balls_pos - self.fly_pos) / self.W

        # one-hot типы
        type_feats = []
        for t in self.ball_types:
            oh = [0.0, 0.0, 0.0]
            if t == BallType.HEAVY:
                oh[1] = 1.0
            elif t == BallType.LIGHT:
                oh[2] = 1.0
            else:
                oh[0] = 1.0
            type_feats.extend(oh)

        # препятствия (ближайшие 3)
        obs_feats = [0.0] * 6
        if self.obstacles:
            dists = [float(np.linalg.norm(o - self.fly_pos)) for o in self.obstacles]
            idx = np.argsort(dists)[:3]
            for k, i in enumerate(idx):
                obs_feats[2 * k] = (self.obstacles[i][0] - self.fly_pos[0]) / self.W
                obs_feats[2 * k + 1] = (self.obstacles[i][1] - self.fly_pos[1]) / self.H

        # пауэрапы (ближайшие 3)
        pu_feats = [0.0] * 6
        if self.powerups:
            dists = [float(np.linalg.norm(p.pos - self.fly_pos)) for p in self.powerups]
            idx = np.argsort(dists)[:3]
            for k, i in enumerate(idx):
                pu_feats[2 * k] = (self.powerups[i].pos[0] - self.fly_pos[0]) / self.W
                pu_feats[2 * k + 1] = (self.powerups[i].pos[1] - self.fly_pos[1]) / self.H

        # эффекты
        eff_feats = [
            self._effects["speed_timer"] / 30.0,
            self._effects["freeze_timer"] / 30.0,
            self._effects["magnet_timer"] / 30.0,
        ]

        return np.concatenate([
            self.fly_pos / np.array([self.W, self.H], dtype=np.float32),
            self.fly_vel / 5.0,
            balls_rel.flatten(),
            self.balls_vel.flatten() / 5.0,
            pockets_rel.flatten(),
            np.array(type_feats, dtype=np.float32),
            np.array(obs_feats, dtype=np.float32),
            np.array(pu_feats, dtype=np.float32),
            np.array(eff_feats, dtype=np.float32),
        ]).astype(np.float32)

    @property
    def obs_dim(self) -> int:
        return (
            4
            + 4 * self.N_BALLS
            + 12
            + 3 * self.N_BALLS
            + 6
            + 6
            + 3
        )

    # ---------------------------------------------------------------- step
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        self.steps += 1
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)

        # эффект скорости
        acc_mult = 1.0
        if self._effects["speed_timer"] > 0:
            acc_mult = 1.6
            self._effects["speed_timer"] -= 1.0

        reward = 0.0

        # физика под-шагами
        for _sub in range(self.SUBSTEPS):
            self._physics_substep(action, acc_mult, dt=1.0 / self.SUBSTEPS)

        # эффект заморозки
        if self._effects["freeze_timer"] > 0:
            self.balls_vel *= 0.4
            self._effects["freeze_timer"] -= 1.0

        # эффект магнита
        if self._effects["magnet_timer"] > 0:
            for i in range(self.N_BALLS):
                if self.potted[i]:
                    continue
                d = self.fly_pos - self.balls_pos[i]
                dist = float(np.linalg.norm(d)) + 1e-6
                if dist < 80:
                    self.balls_vel[i] += (d / dist) * 0.15
            self._effects["magnet_timer"] -= 1.0

        # --- детект луз ---
        potted_now = False
        for i in range(self.N_BALLS):
            if self.potted[i]:
                continue
            for pk in self.pockets:
                if np.linalg.norm(self.balls_pos[i] - pk) < self.POCKET_R:
                    self.potted[i] = True
                    self.balls_vel[i] = 0.0
                    reward += self.rc.reward_pot
                    potted_now = True
                    self._potted_count += 1
                    break

        # --- пауэрапы ---
        for pu in self.powerups:
            if pu.active:
                continue
            if np.linalg.norm(pu.pos - self.fly_pos) < self.FLY_R + self.cfg.powerup_r:
                pu.active = True
                reward += self.rc.reward_powerup
                if pu.kind == Powerup.TYPE_SPEED:
                    self._effects["speed_timer"] = pu.duration
                elif pu.kind == Powerup.TYPE_FREEZE:
                    self._effects["freeze_timer"] = pu.duration
                elif pu.kind == Powerup.TYPE_MAGNET:
                    self._effects["magnet_timer"] = pu.duration

        # --- shaping ---
        cur = self._total_dist()
        reward += (self.prev_dist - cur) * self.rc.reward_shaping
        self.prev_dist = cur

        reward -= self.rc.reward_time_penalty
        reward -= self.rc.reward_energy_penalty * float(np.linalg.norm(action))

        self._episode_reward += reward

        done = potted_now or self.steps >= self.MAX_STEPS
        info = {
            "potted": int(self.potted.sum()),
            "potted_now": potted_now,
            "steps": self.steps,
            "episode_reward": self._episode_reward,
            "wall_hits": self._wall_hits,
            "collisions": self._collisions,
        }
        return self.get_obs(), reward, done, info

    # -------------------------------------------------------- physics substep
    def _physics_substep(self, action: np.ndarray, acc_mult: float, dt: float) -> None:
        # --- ускорение мухи ---
        self.fly_vel = self.fly_vel + action * self.FLY_ACC * acc_mult * dt
        sp = float(np.linalg.norm(self.fly_vel))
        if sp > self.FLY_MAX_SPEED:
            self.fly_vel = self.fly_vel / sp * self.FLY_MAX_SPEED

        # --- интеграция ---
        self.fly_pos = self.fly_pos + self.fly_vel * dt
        self.balls_pos = self.balls_pos + self.balls_vel * dt

        # --- трение ---
        self.fly_vel *= self.FRICTION_FLY ** dt
        self.balls_vel *= self.FRICTION_BALL ** dt

        # --- столкновения со стенами: шары ---
        for i in range(self.N_BALLS):
            if self.potted[i]:
                continue
            p, v = self.balls_pos[i], self.balls_vel[i]
            if p[0] < self.BALL_R:
                p[0] = self.BALL_R; v[0] = -v[0] * self.WALL_BOUNCE
                self._wall_hits += 1
            if p[0] > self.W - self.BALL_R:
                p[0] = self.W - self.BALL_R; v[0] = -v[0] * self.WALL_BOUNCE
                self._wall_hits += 1
            if p[1] < self.BALL_R:
                p[1] = self.BALL_R; v[1] = -v[1] * self.WALL_BOUNCE
                self._wall_hits += 1
            if p[1] > self.H - self.BALL_R:
                p[1] = self.H - self.BALL_R; v[1] = -v[1] * self.WALL_BOUNCE
                self._wall_hits += 1

        # --- столкновения со стенами: муха ---
        p, v = self.fly_pos, self.fly_vel
        if p[0] < self.FLY_R:
            p[0] = self.FLY_R; v[0] = -v[0] * 0.5
        if p[0] > self.W - self.FLY_R:
            p[0] = self.W - self.FLY_R; v[0] = -v[0] * 0.5
        if p[1] < self.FLY_R:
            p[1] = self.FLY_R; v[1] = -v[1] * 0.5
        if p[1] > self.H - self.FLY_R:
            p[1] = self.H - self.FLY_R; v[1] = -v[1] * 0.5

        # --- столкновения с препятствиями ---
        for o in self.obstacles:
            # муха-препятствие
            d = self.fly_pos - o
            dist = float(np.linalg.norm(d)) + 1e-6
            min_d = self.FLY_R + self.cfg.obstacle_r
            if dist < min_d:
                n = d / dist
                self.fly_pos = o + n * min_d
                vn = float(np.dot(self.fly_vel, n))
                if vn < 0:
                    self.fly_vel = self.fly_vel - 1.5 * vn * n
            # шар-препятствие
            for i in range(self.N_BALLS):
                if self.potted[i]:
                    continue
                d = self.balls_pos[i] - o
                dist = float(np.linalg.norm(d)) + 1e-6
                min_d = self.BALL_R + self.cfg.obstacle_r
                if dist < min_d:
                    n = d / dist
                    self.balls_pos[i] = o + n * min_d
                    vn = float(np.dot(self.balls_vel[i], n))
                    if vn < 0:
                        self.balls_vel[i] = self.balls_vel[i] - 1.5 * vn * n

        # --- столкновения муха-шар ---
        for i in range(self.N_BALLS):
            if self.potted[i]:
                continue
            d = self.balls_pos[i] - self.fly_pos
            dist = float(np.linalg.norm(d))
            min_dist = self.FLY_R + self.BALL_R
            if 1e-6 < dist < min_dist:
                n = d / dist
                overlap = min_dist - dist
                self.balls_pos[i] += n * overlap * 0.5
                self.fly_pos -= n * overlap * 0.5
                v_rel = self.fly_vel - self.balls_vel[i]
                v_n = float(np.dot(v_rel, n))
                if v_n > 0:
                    m1, m2 = self.FLY_MASS, self._ball_mass(i)
                    j = 2 * v_n / (m1 + m2)
                    self.fly_vel -= j * m2 * n
                    self.balls_vel[i] += j * m1 * n
                    bs = float(np.linalg.norm(self.balls_vel[i]))
                    if bs > self.BALL_MAX_SPEED:
                        self.balls_vel[i] = self.balls_vel[i] / bs * self.BALL_MAX_SPEED
                    self._collisions += 1

        # --- столкновения шар-шар ---
        for i in range(self.N_BALLS):
            if self.potted[i]:
                continue
            for j in range(i + 1, self.N_BALLS):
                if self.potted[j]:
                    continue
                d = self.balls_pos[j] - self.balls_pos[i]
                dist = float(np.linalg.norm(d))
                if 1e-6 < dist < 2 * self.BALL_R:
                    n = d / dist
                    overlap = 2 * self.BALL_R - dist
                    self.balls_pos[i] -= n * overlap * 0.5
                    self.balls_pos[j] += n * overlap * 0.5
                    v_rel = self.balls_vel[i] - self.balls_vel[j]
                    v_n = float(np.dot(v_rel, n))
                    if v_n > 0:
                        imp = v_n * self.BALL_BOUNCE
                        m1, m2 = self._ball_mass(i), self._ball_mass(j)
                        total = m1 + m2
                        self.balls_vel[i] -= (2 * m2 / total) * imp * n
                        self.balls_vel[j] += (2 * m1 / total) * imp * n
                        self._collisions += 1

    # ---------------------------------------------------------------- render
    def render_state(self) -> Dict[str, Any]:
        return {
            "fly_pos": self.fly_pos.copy(),
            "fly_vel": self.fly_vel.copy(),
            "balls_pos": self.balls_pos.copy(),
            "balls_vel": self.balls_vel.copy(),
            "potted": self.potted.copy(),
            "ball_types": list(self.ball_types),
            "obstacles": [o.copy() for o in self.obstacles],
            "powerups": [(p.pos.copy(), p.kind, p.active) for p in self.powerups],
            "effects": dict(self._effects),
        }

    def replay(self, trajectory: List[Dict[str, Any]]) -> None:
        """Проигрывает записанную траекторию (для отладки)."""
        for step in trajectory:
            self.fly_pos = step["fly_pos"]
            self.balls_pos = step["balls_pos"]
            # и т.д.


# =============================================================================
# SECTION 5 — LIVE PYGAME RENDERER
# =============================================================================
class PygameRenderer:
    """
    Живой рендер игры через Pygame. Используется во время обучения
    для отслеживания поведения агента в реальном времени.

    Открывает окно при первом вызове, обновляет 60 FPS.
    Не блокирует основной поток (обрабатывает события pygame).
    """

    def __init__(self, env: BilliardsEnv, scale: int = 2, fps: int = 60, title: str = "Drosophila"):
        if not _HAS_PYGAME:
            raise RuntimeError("pygame не установлен. pip install pygame")
        self.env = env
        self.scale = scale
        self.fps = fps
        self.title = title
        self.screen: Optional[pygame.Surface] = None
        self.clock: Optional[pygame.time.Clock] = None
        self.font: Optional[pygame.font.Font] = None
        self.small_font: Optional[pygame.font.Font] = None
        self.closed = False
        self._init()

    def _init(self):
        pygame.init()
        pygame.display.set_caption(self.title)
        W = int(self.env.W * self.scale + 40)
        H = int(self.env.H * self.scale + 100)
        self.screen = pygame.display.set_mode((W, H))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 16, bold=True)
        self.small_font = pygame.font.SysFont("consolas", 12)

    def _to_px(self, p: np.ndarray) -> Tuple[int, int]:
        return (int(p[0] * self.scale + 20), int(p[1] * self.scale + 80))

    def render(self, info: Dict[str, Any], episode: int, total_reward: float):
        if self.closed or self.screen is None:
            return
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.closed = True
                pygame.quit()
                return
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.closed = True
                pygame.quit()
                return

        self.screen.fill((18, 18, 24))

        # стол
        table_rect = pygame.Rect(20, 80, int(self.env.W * self.scale), int(self.env.H * self.scale))
        pygame.draw.rect(self.screen, (26, 92, 26), table_rect)
        pygame.draw.rect(self.screen, (60, 40, 20), table_rect, 3)

        # лузы
        for pk in self.env.pockets:
            px, py = self._to_px(pk)
            pygame.draw.circle(self.screen, (0, 0, 0), (px, py), int(self.env.POCKET_R * self.scale))

        # препятствия
        for o in self.env.obstacles:
            px, py = self._to_px(o)
            pygame.draw.circle(self.screen, (140, 100, 60), (px, py),
                               int(self.env.cfg.obstacle_r * self.scale))

        # пауэрапы
        for pu in self.env.powerups:
            if pu.active:
                continue
            px, py = self._to_px(pu.pos)
            color = Powerup.COLORS.get(pu.kind, (255, 255, 255))
            pygame.draw.circle(self.screen, color, (px, py), int(self.env.cfg.powerup_r * self.scale))

        # шары
        for i in range(self.env.N_BALLS):
            if self.env.potted[i]:
                continue
            px, py = self._to_px(self.env.balls_pos[i])
            color = BallType.COLOR[self.env.ball_types[i]]
            pygame.draw.circle(self.screen, color, (px, py), int(self.env.BALL_R * self.scale))
            pygame.draw.circle(self.screen, (0, 0, 0), (px, py), int(self.env.BALL_R * self.scale), 1)

        # муха
        fpx, fpy = self._to_px(self.env.fly_pos)
        pygame.draw.circle(self.screen, (230, 60, 60), (fpx, fpy), int(self.env.FLY_R * self.scale))
        # крылья
        pygame.draw.circle(self.screen, (255, 200, 200),
                           (fpx - int(6 * self.scale), fpy - int(3 * self.scale)),
                           int(self.env.FLY_R * self.scale * 0.7))
        pygame.draw.circle(self.screen, (255, 200, 200),
                           (fpx + int(6 * self.scale), fpy - int(3 * self.scale)),
                           int(self.env.FLY_R * self.scale * 0.7))

        # стрелка скорости
        if np.linalg.norm(self.env.fly_vel) > 0.05:
            vx, vy = self.env.fly_vel
            ex = fpx + int(vx * self.scale * 4)
            ey = fpy + int(vy * self.scale * 4)
            pygame.draw.line(self.screen, (255, 255, 0), (fpx, fpy), (ex, ey), 2)

        # заголовок
        header = f"Episode {episode}  |  reward {total_reward:7.2f}  |  potted {int(self.env.potted.sum())}/{self.env.N_BALLS}  |  step {self.env.steps}/{self.env.MAX_STEPS}"
        surf = self.font.render(header, True, (230, 230, 230))
        self.screen.blit(surf, (20, 20))

        # статус эффектов
        y = 50
        for k, v in self.env._effects.items():
            if v > 0:
                s = self.small_font.render(f"{k}: {v:.0f}", True, (255, 255, 100))
                self.screen.blit(s, (20, y))
                y += 14

        pygame.display.flip()
        if self.clock:
            self.clock.tick(self.fps)

    def close(self):
        if not self.closed:
            try:
                pygame.quit()
            except Exception:
                pass
            self.closed = True


# =============================================================================
# SECTION 6 — MATPLOTLIB LIVE DASHBOARD
# =============================================================================
class LiveDashboard:
    """
    Многопанельный live-дашборд:
      [0,0] игра
      [0,1] растр спайков
      [0,2] мембранные потенциалы
      [1,0] гистограмма весов / активности
      [1,1] кривые обучения (reward, potted)
      [1,2] телеметрия / статистика
    """

    def __init__(self, env: BilliardsEnv, brain: nn.Module, cfg: Config):
        self.env = env
        self.brain = brain
        self.cfg = cfg
        self.fig = None
        self.axes = {}
        self.artists = {}
        self.spike_history: Dict[str, List[np.ndarray]] = defaultdict(list)
        self.membrane_history: Dict[str, List[np.ndarray]] = defaultdict(list)
        self.reward_history: List[float] = []
        self.potted_history: List[int] = []
        self.last_info: Dict[str, Any] = {}
        self._max_history = 300
        self._closed = False
        self._build()

    def _build(self):
        self.fig = plt.figure(figsize=(15, 8.5), facecolor="#0e0e12")
        gs = gridspec.GridSpec(2, 3, figure=self.fig, hspace=0.35, wspace=0.28,
                               left=0.04, right=0.98, top=0.94, bottom=0.06)

        ax_game = self.fig.add_subplot(gs[:, 0])
        ax_game.set_facecolor("#1a5c1a")
        ax_game.set_xlim(-12, self.env.W + 12)
        ax_game.set_ylim(-12, self.env.H + 12)
        ax_game.set_aspect("equal")
        ax_game.set_xticks([]); ax_game.set_yticks([])
        ax_game.set_title("Бильярд", color="white")
        self.axes["game"] = ax_game
        # лузы
        for p in self.env.pockets:
            ax_game.add_patch(Circle(p, self.env.POCKET_R, color="black", zorder=1))
        # динамические артисты игры
        self.artists["balls"] = []
        for i in range(self.env.N_BALLS):
            c = Circle((0, 0), self.env.BALL_R, color="white", ec="black", zorder=2)
            ax_game.add_patch(c)
            self.artists["balls"].append(c)
        self.artists["fly"] = Circle((0, 0), self.env.FLY_R, color="red", zorder=3)
        ax_game.add_patch(self.artists["fly"])
        self.artists["obstacles"] = []
        self.artists["powerups"] = []
        self.artists["heading"] = ax_game.text(
            0.02, 1.02, "", transform=ax_game.transAxes,
            color="white", fontsize=9, verticalalignment="bottom",
        )

        # спайки
        ax_raster = self.fig.add_subplot(gs[0, 1])
        ax_raster.set_facecolor("#0e0e12")
        ax_raster.set_title("Raster спайков", color="white")
        ax_raster.set_xlabel("t", color="#aaa")
        ax_raster.set_ylabel("нейрон", color="#aaa")
        ax_raster.tick_params(colors="#aaa")
        self.axes["raster"] = ax_raster

        # мембраны
        ax_mem = self.fig.add_subplot(gs[0, 2])
        ax_mem.set_facecolor("#0e0e12")
        ax_mem.set_title("Мембранные потенциалы", color="white")
        ax_mem.set_xlabel("t", color="#aaa")
        ax_mem.set_ylabel("v", color="#aaa")
        ax_mem.tick_params(colors="#aaa")
        self.axes["membrane"] = ax_mem

        # веса
        ax_w = self.fig.add_subplot(gs[1, 1])
        ax_w.set_facecolor("#0e0e12")
        ax_w.set_title("Веса / активность", color="white")
        ax_w.tick_params(colors="#aaa")
        self.axes["weights"] = ax_w

        # кривые
        ax_c = self.fig.add_subplot(gs[1, 2])
        ax_c.set_facecolor("#0e0e12")
        ax_c.set_title("Reward / Potted", color="white")
        ax_c.tick_params(colors="#aaa")
        ax_c.set_xlabel("эпизод", color="#aaa")
        self.axes["curves"] = ax_c
        self.artists["curve_reward"] = ax_c.plot([], [], color="#4fc3f7", lw=1, label="reward")[0]
        self.artists["curve_potted"] = ax_c.plot([], [], color="#81c784", lw=1, label="potted")[0]
        ax_c.legend(facecolor="#222", labelcolor="white", fontsize=8)

        plt.ion()
        plt.show(block=False)

    # ---------------------------------------------------------------- record
    def record_spikes(self, spikes: Dict[str, torch.Tensor]):
        for name, spk in spikes.items():
            arr = spk.detach().cpu().numpy()
            if arr.ndim == 2:
                arr = arr[0]
            self.spike_history[name].append(arr)
            if len(self.spike_history[name]) > self._max_history:
                self.spike_history[name] = self.spike_history[name][-self._max_history:]

    def record_membrane(self):
        for name, layer in self._iter_lif_layers():
            v = layer.v.detach().cpu().numpy()
            if v.ndim == 2:
                v = v[0]
            self.membrane_history[name].append(v)
            if len(self.membrane_history[name]) > self._max_history:
                self.membrane_history[name] = self.membrane_history[name][-self._max_history:]

    def _iter_lif_layers(self):
        from_syn = getattr(self.brain, "layers", None)
        if from_syn is not None:
            for name, layer in zip(["pn", "al", "kc", "mbon"], from_syn):
                yield name, layer
        elif hasattr(self.brain, "v"):
            yield "connectome", self.brain

    def record_episode(self, reward: float, potted: int):
        self.reward_history.append(reward)
        self.potted_history.append(potted)
        if len(self.reward_history) > self._max_history:
            self.reward_history = self.reward_history[-self._max_history:]
            self.potted_history = self.potted_history[-self._max_history:]

    # ---------------------------------------------------------------- update
    def update_game(self):
        ax = self.axes["game"]
        for i, c in enumerate(self.artists["balls"]):
            if self.env.potted[i]:
                c.set_visible(False)
            else:
                c.set_visible(True)
                c.center = (float(self.env.balls_pos[i][0]), float(self.env.balls_pos[i][1]))
                c.set_color(np.array(BallType.COLOR[self.env.ball_types[i]]) / 255.0)
        self.artists["fly"].center = (float(self.env.fly_pos[0]), float(self.env.fly_pos[1]))

        # препятствия (пере-рисовка)
        for a in self.artists["obstacles"]:
            a.remove()
        self.artists["obstacles"].clear()
        for o in self.env.obstacles:
            c = Circle((float(o[0]), float(o[1])), self.env.cfg.obstacle_r,
                       color="#8a6a3a", zorder=1.5)
            ax.add_patch(c)
            self.artists["obstacles"].append(c)

        # пауэрапы
        for a in self.artists["powerups"]:
            a.remove()
        self.artists["powerups"].clear()
        for pu in self.env.powerups:
            if pu.active:
                continue
            color = np.array(Powerup.COLORS.get(pu.kind, (255, 255, 255))) / 255.0
            c = Circle((float(pu.pos[0]), float(pu.pos[1])), self.env.cfg.powerup_r,
                       color=color, zorder=1.2)
            ax.add_patch(c)
            self.artists["powerups"].append(c)

        header = (f"step {self.env.steps}/{self.env.MAX_STEPS} | "
                  f"potted {int(self.env.potted.sum())}/{self.env.N_BALLS}")
        self.artists["heading"].set_text(header)

    def update_raster(self):
        ax = self.axes["raster"]
        ax.clear()
        ax.set_facecolor("#0e0e12")
        ax.set_title("Raster спайков", color="white")
        ax.set_xlabel("t", color="#aaa"); ax.set_ylabel("нейрон", color="#aaa")
        ax.tick_params(colors="#aaa")
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(self.spike_history), 1)))
        offset = 0
        for i, (name, hist) in enumerate(self.spike_history.items()):
            if not hist:
                continue
            arr = np.stack(hist, axis=0)
            n_show = min(arr.shape[1], 60)
            sub = arr[:, :n_show]
            ts, ns = np.where(sub > 0.5)
            ax.scatter(ts, ns + offset, s=2, c=[colors[i]], label=name, alpha=0.85)
            offset += n_show + 5
        ax.legend(loc="upper right", fontsize=7, facecolor="#222", labelcolor="white")

    def update_membrane(self):
        ax = self.axes["membrane"]
        ax.clear()
        ax.set_facecolor("#0e0e12")
        ax.set_title("Мембранные потенциалы", color="white")
        ax.set_xlabel("t", color="#aaa"); ax.set_ylabel("v", color="#aaa")
        ax.tick_params(colors="#aaa")
        for name, hist in self.membrane_history.items():
            if not hist:
                continue
            arr = np.stack(hist, axis=0)
            n_show = min(arr.shape[1], 8)
            for j in range(n_show):
                ax.plot(arr[:, j], lw=0.8, alpha=0.7)

    def update_weights(self):
        ax = self.axes["weights"]
        ax.clear()
        ax.set_facecolor("#0e0e12")
        ax.set_title("Веса / активность", color="white")
        ax.tick_params(colors="#aaa")
        if hasattr(self.brain, "W_val"):
            w = self.brain.W_val.detach().cpu().numpy()
            ax.hist(w, bins=40, color="#4fc3f7", alpha=0.85)
            ax.set_xlabel("syn weight", color="#aaa")
        else:
            all_spk = []
            for hist in self.spike_history.values():
                if hist:
                    all_spk.append(np.concatenate(hist))
            if all_spk:
                all_spk = np.concatenate(all_spk)
                ax.hist(all_spk, bins=20, color="#81c784", alpha=0.85)
            ax.set_xlabel("spike value", color="#aaa")

    def update_curves(self):
        ax = self.axes["curves"]
        if self.reward_history:
            self.artists["curve_reward"].set_data(range(len(self.reward_history)), self.reward_history)
            self.artists["curve_potted"].set_data(range(len(self.potted_history)), self.potted_history)
            ax.relim()
            ax.autoscale_view()

    def update_all(self):
        if self._closed or self.fig is None:
            return
        try:
            self.update_game()
            self.update_raster()
            self.update_membrane()
            self.update_weights()
            self.update_curves()
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
        except Exception as e:
            LOG.warning(f"Dashboard update failed: {e}")

    def close(self):
        if self.fig is not None and not self._closed:
            self._closed = True
            try:
                plt.close(self.fig)
            except Exception:
                pass


# =============================================================================
# SECTION 7 — SPIKING NEURONS
# =============================================================================
class SurrogateSpikeFastSigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v):
        ctx.save_for_backward(v)
        return (v > 0).float()

    @staticmethod
    def backward(ctx, grad_out):
        (v,) = ctx.saved_tensors
        sg = 1.0 / (1.0 + 5.0 * v.abs()) ** 2
        return grad_out * sg


class SurrogateSpikeATan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v):
        ctx.save_for_backward(v)
        return (v > 0).float()

    @staticmethod
    def backward(ctx, grad_out):
        (v,) = ctx.saved_tensors
        alpha = 2.0
        sg = alpha / (2.0 * (1.0 + (math.pi * alpha * v / 2.0) ** 2))
        return grad_out * sg


class SurrogateSpikeTriangle(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v):
        ctx.save_for_backward(v)
        return (v > 0).float()

    @staticmethod
    def backward(ctx, grad_out):
        (v,) = ctx.saved_tensors
        gamma = 0.5
        sg = torch.clamp(1.0 - v.abs() / gamma, min=0.0) / gamma
        return grad_out * sg


SPIKE_FUNCS = {
    "fast_sigmoid": SurrogateSpikeFastSigmoid.apply,
    "atan": SurrogateSpikeATan.apply,
    "triangle": SurrogateSpikeTriangle.apply,
}


def spike_fn(v: torch.Tensor, kind: str = "fast_sigmoid") -> torch.Tensor:
    return SPIKE_FUNCS.get(kind, SurrogateSpikeFastSigmoid.apply)(v)


class LIFLayer(nn.Module):
    """Простой LIF-слой с опциональной синаптической динамикой."""

    def __init__(
        self,
        n_in: int,
        n_out: int,
        tau: float = 2.0,
        v_th: float = 1.0,
        tau_syn: float = 5.0,
        v_reset: float = 0.0,
        use_syn: bool = False,
        spike_kind: str = "fast_sigmoid",
    ):
        super().__init__()
        self.n_in, self.n_out = n_in, n_out
        self.tau, self.v_th = tau, v_th
        self.tau_syn, self.v_reset = tau_syn, v_reset
        self.use_syn = use_syn
        self.spike_kind = spike_kind

        self.fc = nn.Linear(n_in, n_out)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

        self.reset_state(1)

    def reset_state(self, batch: int = 1):
        self.v = torch.zeros(batch, self.n_out)
        if self.use_syn:
            self.i_syn = torch.zeros(batch, self.n_out)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        I = self.fc(x)
        if self.use_syn:
            self.i_syn = self.i_syn + (-self.i_syn + I) / self.tau_syn
            I_eff = self.i_syn
        else:
            I_eff = I
        self.v = self.v + (-self.v + I_eff) / self.tau
        spk = spike_fn(self.v - self.v_th, self.spike_kind)
        self.v = self.v * (1 - spk) + self.v_reset * spk
        return spk, self.v

    def detach_state(self):
        self.v = self.v.detach()
        if self.use_syn:
            self.i_syn = self.i_syn.detach()


class RecurrentLIFLayer(nn.Module):
    """LIF-слой с рекуррентными связями."""

    def __init__(self, n_in: int, n_out: int, tau: float = 2.0,
                 v_th: float = 1.0, spike_kind: str = "fast_sigmoid"):
        super().__init__()
        self.n_in, self.n_out = n_in, n_out
        self.tau, self.v_th = tau, v_th
        self.spike_kind = spike_kind

        self.fc_in = nn.Linear(n_in, n_out)
        self.fc_rec = nn.Linear(n_out, n_out, bias=False)
        nn.init.xavier_uniform_(self.fc_in.weight)
        nn.init.zeros_(self.fc_in.bias)
        nn.init.orthogonal_(self.fc_rec.weight)
        with torch.no_grad():
            self.fc_rec.weight.mul_(0.5)

        self.reset_state(1)

    def reset_state(self, batch: int = 1):
        self.v = torch.zeros(batch, self.n_out)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        I = self.fc_in(x) + self.fc_rec(self.last_spk if hasattr(self, "last_spk") else torch.zeros_like(self.v))
        self.v = self.v + (-self.v + I) / self.tau
        spk = spike_fn(self.v - self.v_th, self.spike_kind)
        self.v = self.v * (1 - spk)
        self.last_spk = spk
        return spk, self.v

    def detach_state(self):
        self.v = self.v.detach()
        if hasattr(self, "last_spk"):
            self.last_spk = self.last_spk.detach()


# =============================================================================
# SECTION 8 — BRAINS
# =============================================================================
class BaseBrain(nn.Module, ABC):
    """Базовый интерфейс мозга."""

    def __init__(self, n_obs: int, n_act: int, cfg: Config):
        super().__init__()
        self.n_obs, self.n_act = n_obs, n_act
        self.cfg = cfg
        self.log_std = nn.Parameter(torch.full((n_act,), -0.3))

    @abstractmethod
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]: ...

    @abstractmethod
    def reset_state(self, batch: int = 1) -> None: ...

    def detach_state(self) -> None:
        pass

    def act(self, obs: torch.Tensor, deterministic: bool = False):
        mean, spikes = self.forward(obs)
        std = self.log_std.exp()
        dist = Normal(mean, std)
        if deterministic:
            action = mean
            log_prob = torch.zeros(mean.shape[0], device=mean.device)
            entropy = torch.zeros(mean.shape[0], device=mean.device)
        else:
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(-1)
            entropy = dist.entropy().sum(-1)
        return action, log_prob, entropy, spikes


class SyntheticFlyBrain(BaseBrain):
    """PN→AL→KC→MBON→мотор."""

    def __init__(self, n_obs: int, n_act: int, cfg: Config):
        super().__init__(n_obs, n_act, cfg)
        bc = cfg.brain
        self.n_pn, self.n_al = bc.n_sensory, bc.n_al
        self.n_kc, self.n_mbon = bc.n_kc, bc.n_mbon

        self.pn = LIFLayer(n_obs, self.n_pn, bc.tau, bc.v_th, bc.tau_syn,
                           bc.v_reset, use_syn=True)
        self.al = LIFLayer(self.n_pn, self.n_al, bc.tau, bc.v_th, bc.tau_syn,
                           bc.v_reset, use_syn=True)
        self.kc = LIFLayer(self.n_al, self.n_kc, bc.tau, bc.v_th, bc.tau_syn,
                           bc.v_reset, use_syn=False)
        self.mbon = LIFLayer(self.n_kc, self.n_mbon, bc.tau, bc.v_th, bc.tau_syn,
                             bc.v_reset, use_syn=False)
        self.motor = nn.Linear(self.n_mbon, n_act)
        nn.init.xavier_uniform_(self.motor.weight)
        nn.init.zeros_(self.motor.bias)
        self.layers = [self.pn, self.al, self.kc, self.mbon]

    def reset_state(self, batch: int = 1):
        for layer in self.layers:
            layer.reset_state(batch)

    def forward(self, obs: torch.Tensor):
        spikes = {}
        x = obs
        for name, layer in zip(["pn", "al", "kc", "mbon"], self.layers):
            x, _ = layer(x)
            spikes[name] = x
        mean = torch.tanh(self.motor(x))
        return mean, spikes

    def detach_state(self):
        for layer in self.layers:
            layer.detach_state()


class HybridFlyBrain(BaseBrain):
    """Коннектом + обучаемый readout."""

    def __init__(self, n_obs: int, n_act: int, connectome: Dict[str, Any], cfg: Config):
        super().__init__(n_obs, n_act, cfg)
        bc = cfg.brain
        N = len(connectome["neuron_ids"])
        self.N = N
        self.register_buffer("neuron_ids", torch.tensor(connectome["neuron_ids"], dtype=torch.long))
        self.cell_types = connectome["cell_types"]
        self.spike_kind = "fast_sigmoid"

        ei = connectome["edge_index"]
        ew = connectome["edge_weight"].astype(np.float32)
        signs = np.array([
            -1.0 if ("GABA" in ct or "DAN" in ct or "APL" in ct) else 1.0
            for ct in self.cell_types
        ], dtype=np.float32)
        ew = ew / (ew.max() + 1e-8) * signs[ei[1]]

        self.register_buffer("W_idx", torch.tensor(ei, dtype=torch.long))
        self.register_buffer("W_val", torch.tensor(ew, dtype=torch.float32))

        self.W_in = nn.Linear(n_obs, N)
        nn.init.xavier_uniform_(self.W_in.weight)
        nn.init.zeros_(self.W_in.bias)

        mbon_idx = [i for i, ct in enumerate(self.cell_types) if "MBON" in ct]
        if len(mbon_idx) < n_act:
            mbon_idx = list(range(min(N, max(n_act, 24))))
        self.register_buffer("mbon_idx", torch.tensor(mbon_idx, dtype=torch.long))
        self.W_out = nn.Linear(len(mbon_idx), n_act)
        nn.init.xavier_uniform_(self.W_out.weight)
        nn.init.zeros_(self.W_out.bias)

        self.tau = bc.tau
        self.v_th = bc.v_th
        self.reset_state(1)

    def reset_state(self, batch: int = 1):
        self.v = torch.zeros(batch, self.N)
        self._last_spk = torch.zeros(batch, self.N)

    def _sparse_mm(self, spk: torch.Tensor) -> torch.Tensor:
        B = spk.shape[0]
        out = torch.zeros(B, self.N, device=spk.device, dtype=spk.dtype)
        src = self.W_idx[0]
        dst = self.W_idx[1]
        w = self.W_val
        contrib = spk[:, src] * w.unsqueeze(0)
        out.index_add_(1, dst, contrib)
        return out

    def forward(self, obs: torch.Tensor):
        I_ext = self.W_in(obs)
        rec = self._sparse_mm(self._last_spk)
        self.v = self.v + (-self.v + I_ext + rec) / self.tau
        spk = spike_fn(self.v - self.v_th, self.spike_kind)
        self.v = self.v * (1 - spk)
        self._last_spk = spk.detach()
        mbon_act = spk[:, self.mbon_idx]
        mean = torch.tanh(self.W_out(mbon_act))
        return mean, {"connectome": spk}

    def detach_state(self):
        self.v = self.v.detach()
        self._last_spk = self._last_spk.detach()


# =============================================================================
# SECTION 9 — CONNECTOME LOADER
# =============================================================================
class ConnectomeLoader:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._cache_path = Path(cfg.brain.connectome_cache_dir).expanduser() / "subset.npz"

    def load(self) -> Optional[Dict[str, Any]]:
        if not _HAS_FLYWIRE:
            LOG.warning(f"flywire-gnn не установлен ({_FLYWIRE_IMPORT_ERROR}). Fallback → synthetic.")
            return None
        if self._cache_path.exists():
            try:
                LOG.info(f"Загрузка коннектома из кэша: {self._cache_path}")
                return self._load_cache()
            except Exception as e:
                LOG.warning(f"Кэш не читается: {e}")
        try:
            LOG.info("Загрузка FlyWire FAFB v783...")
            data = self._download_and_build()
            self._save_cache(data)
            return data
        except Exception as e:
            LOG.error(f"Ошибка загрузки коннектома: {e}")
            return None

    def _load_cache(self):
        z = np.load(self._cache_path, allow_pickle=True)
        return {
            "neuron_ids": z["neuron_ids"],
            "cell_types": list(z["cell_types"]),
            "edge_index": z["edge_index"],
            "edge_weight": z["edge_weight"],
        }

    def _save_cache(self, d):
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self._cache_path, **d)
        LOG.info(f"Кэш сохранён: {self._cache_path}")

    def _download_and_build(self):
        ds = FlyWireFAFB()
        data = ds.data
        N = data.num_nodes
        neuron_ids = np.arange(N, dtype=np.int64)
        cell_types = [""] * N
        for attr in ("cell_type", "cell_class", "cell_sub_class"):
            if hasattr(ds, attr):
                try:
                    arr = getattr(ds, attr)
                    if arr is not None and len(arr) == N:
                        cell_types = [str(x) for x in arr]
                        break
                except Exception:
                    pass

        include = set(self.cfg.brain.connectome_include_types)
        mask = np.array([any(t in ct for t in include) if ct else False for ct in cell_types])
        if mask.sum() < 50:
            mask = np.ones_like(mask, dtype=bool)

        idx = np.where(mask)[0]
        max_n = self.cfg.brain.connectome_max_neurons
        if len(idx) > max_n:
            rng = np.random.RandomState(self.cfg.train.seed)
            idx = np.sort(rng.choice(idx, size=max_n, replace=False))

        ei = data.edge_index.cpu().numpy()
        ew = data.edge_attr[:, 0].cpu().numpy() if data.edge_attr is not None else np.ones(ei.shape[1])

        keep = ew >= self.cfg.brain.connectome_min_synapses
        ei = ei[:, keep]; ew = ew[keep]

        remap = -np.ones(N, dtype=np.int64)
        remap[idx] = np.arange(len(idx))
        keep2 = (remap[ei[0]] >= 0) & (remap[ei[1]] >= 0)
        ei = ei[:, keep2]; ew = ew[keep2]
        ei = remap[ei]

        LOG.info(f"Коннектом: {len(idx)} нейронов, {ei.shape[1]} связей.")
        return {
            "neuron_ids": idx,
            "cell_types": [cell_types[i] for i in idx],
            "edge_index": ei.astype(np.int64),
            "edge_weight": ew.astype(np.float32),
        }


def build_brain(cfg: Config, n_obs: int, n_act: int) -> BaseBrain:
    mode = cfg.brain.mode
    if mode in ("connectome", "hybrid"):
        loader = ConnectomeLoader(cfg)
        conn = loader.load()
        if conn is not None:
            LOG.info(f"Коннектомный мозг ({mode}).")
            return HybridFlyBrain(n_obs, n_act, conn, cfg)
        LOG.warning("Fallback → synthetic.")
    LOG.info("Synthetic мозг.")
    return SyntheticFlyBrain(n_obs, n_act, cfg)


# =============================================================================
# SECTION 10 — CRITIC
# =============================================================================
class Critic(nn.Module):
    def __init__(self, n_obs: int, hidden: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(n_obs, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 1)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc3(h).squeeze(-1)


# =============================================================================
# SECTION 11 — REPLAY BUFFER
# =============================================================================
class ReplayBuffer:
    """Циклический буфер для off-policy алгоритмов (DQN)."""

    def __init__(self, capacity: int, obs_dim: int):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(self, o, a, r, no, d):
        self.obs[self.ptr] = o
        self.actions[self.ptr] = a
        self.rewards[self.ptr] = r
        self.next_obs[self.ptr] = no
        self.dones[self.ptr] = float(d)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch: int, device: str = "cpu"):
        idx = np.random.randint(0, self.size, size=batch)
        return (
            torch.tensor(self.obs[idx], device=device),
            torch.tensor(self.actions[idx], device=device),
            torch.tensor(self.rewards[idx], device=device),
            torch.tensor(self.next_obs[idx], device=device),
            torch.tensor(self.dones[idx], device=device),
        )

    def __len__(self):
        return self.size


# =============================================================================
# SECTION 12 — ALGORITHMS
# =============================================================================
class AlgorithmBase(ABC):
    """Интерфейс RL-алгоритма."""

    def __init__(self, brain: BaseBrain, critic: Optional[Critic], env: BilliardsEnv, cfg: Config):
        self.brain = brain
        self.critic = critic
        self.env = env
        self.cfg = cfg
        self.opt_brain = torch.optim.Adam(brain.parameters(), lr=cfg.algo.lr_brain)
        self.opt_critic = (
            torch.optim.Adam(critic.parameters(), lr=cfg.algo.lr_critic) if critic else None
        )
        self.scheduler_brain = self._make_scheduler(self.opt_brain)
        self.scheduler_critic = self._make_scheduler(self.opt_critic) if self.opt_critic else None

    def _make_scheduler(self, opt):
        kind = self.cfg.train.lr_scheduler
        if kind == "step":
            return torch.optim.lr_scheduler.StepLR(
                opt, step_size=self.cfg.train.lr_step_size, gamma=self.cfg.train.lr_gamma
            )
        if kind == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.cfg.train.n_episodes)
        return None

    @abstractmethod
    def collect_episode(self, deterministic: bool = False) -> Dict[str, Any]: ...

    @abstractmethod
    def update(self, rollout: Dict[str, Any]) -> Dict[str, float]: ...

    def step_schedulers(self):
        if self.scheduler_brain:
            self.scheduler_brain.step()
        if self.scheduler_critic:
            self.scheduler_critic.step()


class REINFORCEAlgo(AlgorithmBase):
    """Vanilla REINFORCE + learned baseline."""

    def __init__(self, brain, critic, env, cfg, dashboard: Optional[LiveDashboard] = None,
                 pygame_renderer: Optional[PygameRenderer] = None):
        super().__init__(brain, critic, env, cfg)
        self.dashboard = dashboard
        self.pygame_renderer = pygame_renderer

    def collect_episode(self, deterministic: bool = False) -> Dict[str, Any]:
        env = self.env
        obs = env.reset()
        self.brain.reset_state(1)
        self.brain.detach_state()

        log_probs, entropies, rewards, obs_list, spikes_list = [], [], [], [], []
        dones = []
        done = False
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.cfg.train.device).unsqueeze(0)
            obs_list.append(obs_t.squeeze(0))
            action, lp, ent, spikes = self.brain.act(obs_t, deterministic=deterministic)
            log_probs.append(lp.squeeze(0))
            entropies.append(ent.squeeze(0))
            spikes_list.append(spikes)
            a_np = action.detach().cpu().numpy()[0]
            obs, r, done, info = env.step(a_np)
            rewards.append(r)
            dones.append(done)

            # live rendering
            if self.pygame_renderer is not None and not self.pygame_renderer.closed:
                self.pygame_renderer.render(info, episode=-1, total_reward=sum(rewards))
            if self.dashboard is not None:
                self.dashboard.record_spikes(spikes)
                self.dashboard.record_membrane()
                self.dashboard.update_game()

        return {
            "obs": torch.stack(obs_list),
            "log_probs": torch.stack(log_probs),
            "entropies": torch.stack(entropies),
            "rewards": rewards,
            "dones": dones,
            "spikes": spikes_list,
            "potted": int(env.potted.sum()),
            "episode_reward": float(sum(rewards)),
            "episode_len": len(rewards),
        }

    def update(self, rollout):
        device = self.cfg.train.device
        returns = torch.tensor(
            discounted_returns(rollout["rewards"], self.cfg.algo.gamma),
            dtype=torch.float32, device=device,
        )
        obs_batch = rollout["obs"].to(device)
        values = self.critic(obs_batch)
        adv = returns - values.detach()
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        log_probs = rollout["log_probs"].to(device)
        entropies = rollout["entropies"].to(device)
        policy_loss = -(log_probs * adv).mean() - self.cfg.algo.entropy_coef * entropies.mean()
        critic_loss = F.mse_loss(values, returns)

        self.opt_brain.zero_grad()
        self.opt_critic.zero_grad()
        (policy_loss + self.cfg.algo.value_coef * critic_loss).backward()
        torch.nn.utils.clip_grad_norm_(self.brain.parameters(), self.cfg.algo.grad_clip)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.algo.grad_clip)
        self.opt_brain.step(); self.opt_critic.step()
        self.step_schedulers()

        return {
            "loss_policy": float(policy_loss.item()),
            "loss_critic": float(critic_loss.item()),
            "entropy": float(entropies.mean().item()),
        }


class A2CAlgo(AlgorithmBase):
    """Advantage Actor-Critic (синхронный, одношаговый n-step)."""

    def __init__(self, brain, critic, env, cfg, dashboard=None, pygame_renderer=None):
        super().__init__(brain, critic, env, cfg)
        self.dashboard = dashboard
        self.pygame_renderer = pygame_renderer

    def collect_episode(self, deterministic: bool = False):
        env = self.env
        obs = env.reset()
        self.brain.reset_state(1)
        self.brain.detach_state()

        log_probs, entropies, rewards, values, obs_list, spikes_list, dones = [], [], [], [], [], [], []
        done = False
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.cfg.train.device).unsqueeze(0)
            obs_list.append(obs_t.squeeze(0))
            action, lp, ent, spikes = self.brain.act(obs_t, deterministic=deterministic)
            with torch.no_grad():
                v = self.critic(obs_t).squeeze(0)
            log_probs.append(lp.squeeze(0))
            entropies.append(ent.squeeze(0))
            values.append(v)
            spikes_list.append(spikes)
            a_np = action.detach().cpu().numpy()[0]
            obs, r, done, info = env.step(a_np)
            rewards.append(r); dones.append(done)

            if self.pygame_renderer is not None and not self.pygame_renderer.closed:
                self.pygame_renderer.render(info, episode=-1, total_reward=sum(rewards))
            if self.dashboard is not None:
                self.dashboard.record_spikes(spikes)
                self.dashboard.record_membrane()
                self.dashboard.update_game()

        return {
            "obs": torch.stack(obs_list),
            "log_probs": torch.stack(log_probs),
            "entropies": torch.stack(entropies),
            "values": torch.stack(values),
            "rewards": rewards,
            "dones": dones,
            "spikes": spikes_list,
            "potted": int(env.potted.sum()),
            "episode_reward": float(sum(rewards)),
            "episode_len": len(rewards),
        }

    def update(self, rollout):
        device = self.cfg.train.device
        # one-step TD targets
        rewards = torch.tensor(rollout["rewards"], dtype=torch.float32, device=device)
        dones = torch.tensor(rollout["dones"], dtype=torch.float32, device=device)
        values = rollout["values"].to(device)
        obs = rollout["obs"].to(device)

        with torch.no_grad():
            next_v = torch.cat([self.critic(obs[1:]), torch.zeros(1, device=device)], dim=0)
            targets = rewards + self.cfg.algo.gamma * next_v * (1 - dones)

        adv = targets - values
        log_probs = rollout["log_probs"].to(device)
        entropies = rollout["entropies"].to(device)

        policy_loss = -(log_probs * adv.detach()).mean() - self.cfg.algo.entropy_coef * entropies.mean()
        critic_loss = F.mse_loss(values, targets)

        self.opt_brain.zero_grad(); self.opt_critic.zero_grad()
        (policy_loss + self.cfg.algo.value_coef * critic_loss).backward()
        torch.nn.utils.clip_grad_norm_(self.brain.parameters(), self.cfg.algo.grad_clip)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.algo.grad_clip)
        self.opt_brain.step(); self.opt_critic.step()
        self.step_schedulers()

        return {
            "loss_policy": float(policy_loss.item()),
            "loss_critic": float(critic_loss.item()),
            "entropy": float(entropies.mean().item()),
        }


class PPOAlgo(AlgorithmBase):
    """PPO с clipped surrogate objective."""

    def __init__(self, brain, critic, env, cfg, dashboard=None, pygame_renderer=None):
        super().__init__(brain, critic, env, cfg)
        self.dashboard = dashboard
        self.pygame_renderer = pygame_renderer

    def collect_episode(self, deterministic: bool = False):
        env = self.env
        obs = env.reset()
        self.brain.reset_state(1)
        self.brain.detach_state()

        log_probs, entropies, rewards, values, obs_list, spikes_list, dones = [], [], [], [], [], [], []
        actions = []
        done = False
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.cfg.train.device).unsqueeze(0)
            obs_list.append(obs_t.squeeze(0))
            action, lp, ent, spikes = self.brain.act(obs_t, deterministic=deterministic)
            with torch.no_grad():
                v = self.critic(obs_t).squeeze(0)
            log_probs.append(lp.squeeze(0))
            entropies.append(ent.squeeze(0))
            values.append(v)
            spikes_list.append(spikes)
            actions.append(action.squeeze(0))
            a_np = action.detach().cpu().numpy()[0]
            obs, r, done, info = env.step(a_np)
            rewards.append(r); dones.append(done)

            if self.pygame_renderer is not None and not self.pygame_renderer.closed:
                self.pygame_renderer.render(info, episode=-1, total_reward=sum(rewards))
            if self.dashboard is not None:
                self.dashboard.record_spikes(spikes)
                self.dashboard.record_membrane()
                self.dashboard.update_game()

        return {
            "obs": torch.stack(obs_list),
            "actions": torch.stack(actions),
            "log_probs": torch.stack(log_probs),
            "entropies": torch.stack(entropies),
            "values": torch.stack(values),
            "rewards": rewards,
            "dones": dones,
            "spikes": spikes_list,
            "potted": int(env.potted.sum()),
            "episode_reward": float(sum(rewards)),
            "episode_len": len(rewards),
        }

    def update(self, rollout):
        device = self.cfg.train.device
        obs = rollout["obs"].to(device)
        actions = rollout["actions"].to(device)
        old_log_probs = rollout["log_probs"].to(device)
        old_values = rollout["values"].to(device)
        rewards = rollout["rewards"]
        dones = rollout["dones"]

        values_np = old_values.detach().cpu().numpy().tolist() + [0.0]
        adv, returns = compute_gae(
            rewards, values_np, dones, self.cfg.algo.gamma, self.cfg.algo.gae_lambda
        )
        adv = torch.tensor(adv, dtype=torch.float32, device=device)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=device)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        T = len(rewards)
        indices = np.arange(T)
        losses = []
        for _epoch in range(self.cfg.algo.ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, T, self.cfg.algo.ppo_minibatch):
                end = min(start + self.cfg.algo.ppo_minibatch, T)
                mb_idx = indices[start:end]
                mb_idx_t = torch.tensor(mb_idx, dtype=torch.long, device=device)

                # ВАЖНО: SNN с состоянием нельзя просто пересчитать.
                # Используем zero-state пересчёт по obs (приближённо).
                self.brain.reset_state(end - start)
                mean, spikes = self.brain(obs[mb_idx_t])
                std = self.brain.log_std.exp()
                dist = Normal(mean, std)
                new_log_probs = dist.log_prob(actions[mb_idx_t]).sum(-1)
                entropy = dist.entropy().sum(-1).mean()

                ratio = (new_log_probs - old_log_probs[mb_idx_t]).exp()
                surr1 = ratio * adv[mb_idx_t]
                surr2 = torch.clamp(ratio, 1 - self.cfg.algo.ppo_clip,
                                    1 + self.cfg.algo.ppo_clip) * adv[mb_idx_t]
                policy_loss = -torch.min(surr1, surr2).mean() - self.cfg.algo.entropy_coef * entropy

                new_values = self.critic(obs[mb_idx_t])
                value_loss = F.mse_loss(new_values, returns_t[mb_idx_t])

                self.opt_brain.zero_grad(); self.opt_critic.zero_grad()
                (policy_loss + self.cfg.algo.value_coef * value_loss).backward()
                torch.nn.utils.clip_grad_norm_(self.brain.parameters(), self.cfg.algo.grad_clip)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.algo.grad_clip)
                self.opt_brain.step(); self.opt_critic.step()
                losses.append((float(policy_loss.item()), float(value_loss.item()), float(entropy.item())))
        self.step_schedulers()

        pl = np.mean([l[0] for l in losses])
        cl = np.mean([l[1] for l in losses])
        ent = np.mean([l[2] for l in losses])
        return {"loss_policy": float(pl), "loss_critic": float(cl), "entropy": float(ent)}


class DQNAlgo(AlgorithmBase):
    """DQN для дискретного управления (8 направлений толчка)."""

    def __init__(self, brain, critic, env, cfg, dashboard=None, pygame_renderer=None):
        super().__init__(brain, critic, env, cfg)
        self.dashboard = dashboard
        self.pygame_renderer = pygame_renderer
        # Для DQN голова мозга должна выдавать Q-values по n действиям.
        # Мы используем только mean (n_act = n_actions) как Q.
        self.n_actions = cfg.algo.dqn_n_actions
        self.buffer = ReplayBuffer(cfg.algo.dqn_buffer, env.obs_dim)
        self.target_brain = copy.deepcopy(brain)
        self.steps = 0
        self.eps = cfg.algo.dqn_eps_start

    def _discrete_to_cont(self, a: int) -> np.ndarray:
        angle = 2 * math.pi * a / self.n_actions
        return np.array([math.cos(angle), math.sin(angle)], dtype=np.float32)

    def _eps_schedule(self) -> float:
        c = self.cfg.algo
        return exponential_schedule(c.dqn_eps_start, c.dqn_eps_end, self.steps, c.dqn_eps_decay)

    def collect_episode(self, deterministic: bool = False):
        env = self.env
        obs = env.reset()
        self.brain.reset_state(1)
        self.brain.detach_state()

        rewards = []
        done = False
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.cfg.train.device).unsqueeze(0)
            self.eps = self._eps_schedule()
            if not deterministic and random.random() < self.eps:
                a_disc = random.randrange(self.n_actions)
            else:
                with torch.no_grad():
                    q, _ = self.brain(obs_t)
                a_disc = int(torch.argmax(q[0, :self.n_actions]).item())

            a_cont = self._discrete_to_cont(a_disc)
            next_obs, r, done, info = env.step(a_cont)
            self.buffer.add(obs, a_disc, r, next_obs, done)
            rewards.append(r)
            obs = next_obs
            self.steps += 1

            if self.pygame_renderer is not None and not self.pygame_renderer.closed:
                self.pygame_renderer.render(info, episode=-1, total_reward=sum(rewards))

            if len(self.buffer) >= self.cfg.algo.dqn_batch:
                self._train_step()

            if self.steps % self.cfg.algo.dqn_target_update == 0:
                self.target_brain.load_state_dict(self.brain.state_dict())

        return {
            "rewards": rewards,
            "potted": int(env.potted.sum()),
            "episode_reward": float(sum(rewards)),
            "episode_len": len(rewards),
            "obs": torch.tensor(obs, dtype=torch.float32, device=self.cfg.train.device).unsqueeze(0),
            "log_probs": torch.zeros(1, device=self.cfg.train.device),
            "entropies": torch.zeros(1, device=self.cfg.train.device),
            "dones": [done],
        }

    def _train_step(self):
        device = self.cfg.train.device
        o, a, r, no, d = self.buffer.sample(self.cfg.algo.dqn_batch, device)
        with torch.no_grad():
            q_next, _ = self.target_brain(no)
            q_next = q_next[:, :self.n_actions].max(dim=1)[0]
            target = r + self.cfg.algo.gamma * q_next * (1 - d)
        q, _ = self.brain(o)
        q = q[:, :self.n_actions]
        q_selected = q.gather(1, a.unsqueeze(1)).squeeze(1)
        loss = F.smooth_l1_loss(q_selected, target)

        self.opt_brain.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.brain.parameters(), self.cfg.algo.grad_clip)
        self.opt_brain.step()

    def update(self, rollout):
        return {"loss_policy": 0.0, "loss_critic": 0.0, "entropy": 0.0}


ALGOS = {
    "reinforce": REINFORCEAlgo,
    "a2c": A2CAlgo,
    "ppo": PPOAlgo,
    "dqn": DQNAlgo,
}


def build_algo(name: str, brain, critic, env, cfg, dashboard=None, pygame_renderer=None):
    if name not in ALGOS:
        raise ValueError(f"Unknown algo: {name}. Available: {list(ALGOS)}")
    cls = ALGOS[name]
    return cls(brain, critic, env, cfg, dashboard=dashboard, pygame_renderer=pygame_renderer)


# =============================================================================
# SECTION 13 — CALLBACKS
# =============================================================================
class Callback(ABC):
    """Базовый callback."""

    def on_train_begin(self, trainer): pass
    def on_train_end(self, trainer): pass
    def on_episode_end(self, trainer, ep, metrics): pass
    def on_eval_end(self, trainer, ep, metrics): pass


class ModelCheckpointCallback(Callback):
    def __init__(self, save_every: int = 200):
        self.save_every = save_every

    def on_episode_end(self, trainer, ep, metrics):
        if (ep + 1) % self.save_every == 0:
            trainer.save_checkpoint(ep)


class EarlyStoppingCallback(Callback):
    def __init__(self, patience: int = 20, min_delta: float = 0.01):
        self.patience = patience
        self.min_delta = min_delta
        self.best = -float("inf")
        self.bad = 0
        self.stop = False

    def on_eval_end(self, trainer, ep, metrics):
        score = metrics.get("eval_potted_mean", 0.0) + 0.01 * metrics.get("eval_reward_mean", 0.0)
        if score > self.best + self.min_delta:
            self.best = score
            self.bad = 0
        else:
            self.bad += 1
            if self.bad >= self.patience:
                LOG.info(f"EarlyStopping: нет улучшений {self.patience} эпох. Останов.")
                self.stop = True


class TensorBoardCallback(Callback):
    def __init__(self, log_dir: str):
        self.writer = SummaryWriter(log_dir) if _HAS_TB else None

    def on_episode_end(self, trainer, ep, metrics):
        if self.writer is None:
            return
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                self.writer.add_scalar(f"train/{k}", v, ep)

    def on_eval_end(self, trainer, ep, metrics):
        if self.writer is None:
            return
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                self.writer.add_scalar(f"eval/{k}", v, ep)

    def on_train_end(self, trainer):
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()


# =============================================================================
# SECTION 14 — METRICS
# =============================================================================
class MetricsTracker:
    """Скользящие метрики обучения."""

    def __init__(self, window: int = 100):
        self.window = window
        self.data: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window))
        self.history: Dict[str, List[float]] = defaultdict(list)

    def add(self, **kwargs):
        for k, v in kwargs.items():
            self.data[k].append(float(v))
            self.history[k].append(float(v))

    def mean(self, key: str) -> float:
        if key not in self.data or not self.data[key]:
            return 0.0
        return float(np.mean(self.data[key]))

    def last(self, key: str) -> float:
        if key not in self.data or not self.data[key]:
            return 0.0
        return float(self.data[key][-1])

    def all(self, key: str) -> List[float]:
        return list(self.history.get(key, []))

    def save_csv(self, path: Union[str, Path]):
        path = Path(path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        keys = sorted(self.history.keys())
        if not keys:
            return
        max_len = max(len(v) for v in self.history.values())
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["step"] + keys)
            for i in range(max_len):
                row = [i]
                for k in keys:
                    vals = self.history[k]
                    row.append(vals[i] if i < len(vals) else "")
                w.writerow(row)
        LOG.info(f"CSV метрики сохранены: {path}")

    def plot(self, path: Union[str, Path]):
        path = Path(path).expanduser()
        keys = list(self.history.keys())
        if not keys:
            return
        n = len(keys)
        cols = 3
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.5 * rows))
        axes = np.array(axes).reshape(-1)
        for i, k in enumerate(keys):
            axes[i].plot(self.history[k])
            axes[i].set_title(k)
        for j in range(len(keys), len(axes)):
            axes[j].axis("off")
        plt.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        LOG.info(f"Графики метрик сохранены: {path}")


# =============================================================================
# SECTION 15 — TRAINER
# =============================================================================
class Trainer:
    """
    Оркестрирует обучение: среда + мозг + алгоритм + визуализация + callbacks.
    """

    def __init__(
        self,
        cfg: Config,
        brain: BaseBrain,
        critic: Critic,
        env: BilliardsEnv,
        algo,
        output_dir: Path,
        dashboard: Optional[LiveDashboard] = None,
        pygame_renderer: Optional[PygameRenderer] = None,
    ):
        self.cfg = cfg
        self.brain = brain
        self.critic = critic
        self.env = env
        self.algo = algo
        self.output_dir = output_dir
        self.dashboard = dashboard
        self.pygame_renderer = pygame_renderer
        self.device = cfg.train.device

        self.metrics = MetricsTracker(window=100)
        self.callbacks: List[Callback] = [
            ModelCheckpointCallback(save_every=cfg.train.save_every),
            EarlyStoppingCallback(
                patience=cfg.train.early_stop_patience,
                min_delta=cfg.train.early_stop_min_delta,
            ),
        ]
        if _HAS_TB:
            self.callbacks.append(TensorBoardCallback(str(output_dir / "tb")))

        self.best_potted = 0
        self.best_eval = -float("inf")
        self.global_step = 0
        self.t0 = time.time()

    # --------------------------------------------------------------- helpers
    def _all_callbacks(self, method: str, *args, **kwargs):
        for cb in self.callbacks:
            try:
                getattr(cb, method)(*args, **kwargs)
            except Exception as e:
                LOG.warning(f"Callback {cb.__class__.__name__}.{method} error: {e}")

    def save_checkpoint(self, ep: int):
        path = self.output_dir / f"checkpoint_ep{ep+1}.pt"
        state = {
            "episode": ep,
            "brain": self.brain.state_dict(),
            "critic": self.critic.state_dict(),
            "best_potted": self.best_potted,
            "metrics_history": dict(self.metrics.history),
            "config": self.cfg.to_json(),
        }
        save_checkpoint(state, path)

    def save_best(self):
        path = self.output_dir / "best.pt"
        state = {
            "brain": self.brain.state_dict(),
            "critic": self.critic.state_dict(),
            "best_potted": self.best_potted,
            "config": self.cfg.to_json(),
        }
        save_checkpoint(state, path)

    def save_final(self):
        path = self.output_dir / "final.pt"
        state = {
            "brain": self.brain.state_dict(),
            "critic": self.critic.state_dict(),
            "best_potted": self.best_potted,
            "metrics_history": dict(self.metrics.history),
            "config": self.cfg.to_json(),
        }
        save_checkpoint(state, path)

    # ----------------------------------------------------------------- train
    def train(self):
        cfg = self.cfg
        self._all_callbacks("on_train_begin", self)
        LOG.info(f"Начало обучения: algo={cfg.algo.algo} brain={cfg.brain.mode} "
                 f"episodes={cfg.train.n_episodes}")

        for ep in range(cfg.train.n_episodes):
            rollout = self.algo.collect_episode(deterministic=False)
            losses = self.algo.update(rollout)

            self.metrics.add(
                reward=rollout["episode_reward"],
                potted=rollout["potted"],
                episode_len=rollout["episode_len"],
                **losses,
            )

            if self.dashboard is not None:
                self.dashboard.record_episode(rollout["episode_reward"], rollout["potted"])
                if ep % cfg.viz.live_dashboard_update_every == 0:
                    self.dashboard.update_all()

            if rollout["potted"] > self.best_potted:
                self.best_potted = rollout["potted"]
                self.save_best()

            self._all_callbacks("on_episode_end", self, ep, {
                "reward": rollout["episode_reward"],
                "potted": rollout["potted"],
                **losses,
            })

            if ep % cfg.train.log_every == 0:
                avg_r = self.metrics.mean("reward")
                avg_p = self.metrics.mean("potted")
                LOG.info(
                    f"ep {ep:5d}/{cfg.train.n_episodes} | "
                    f"R {rollout['episode_reward']:8.2f} | avgR {avg_r:8.2f} | "
                    f"potted {rollout['potted']} | avgP {avg_p:.2f} | "
                    f"ent {losses.get('entropy', 0):.3f} | "
                    f"len {rollout['episode_len']} | "
                    f"elapsed {human_time(time.time()-self.t0)}"
                )

            if cfg.train.eval_every and (ep + 1) % cfg.train.eval_every == 0:
                eval_metrics = self.evaluate(n_episodes=cfg.train.eval_episodes)
                self._all_callbacks("on_eval_end", self, ep, eval_metrics)
                if any(getattr(cb, "stop", False) for cb in self.callbacks):
                    LOG.info("Ранняя остановка.")
                    break

        self._all_callbacks("on_train_end", self)
        self.save_final()
        self.metrics.save_csv(self.output_dir / "metrics.csv")
        self.metrics.plot(self.output_dir / "metrics.png")
        LOG.info(f"Обучение завершено за {human_time(time.time()-self.t0)}")

    # -------------------------------------------------------------- evaluate
    @torch.no_grad()
    def evaluate(self, n_episodes: int = 5) -> Dict[str, float]:
        rewards, potted = [], []
        for _ in range(n_episodes):
            r = self.algo.collect_episode(deterministic=True)
            rewards.append(r["episode_reward"])
            potted.append(r["potted"])
        metrics = {
            "eval_reward_mean": float(np.mean(rewards)),
            "eval_potted_mean": float(np.mean(potted)),
            "eval_potted_max": int(np.max(potted)),
        }
        LOG.info(
            f"[EVAL] reward {metrics['eval_reward_mean']:.2f} | "
            f"potted avg {metrics['eval_potted_mean']:.2f} | max {metrics['eval_potted_max']}"
        )
        if self.dashboard is not None:
            self.dashboard.update_all()
        return metrics


# =============================================================================
# SECTION 16 — FINAL VISUALIZATION (ANIMATION)
# =============================================================================
def run_final_animation(
    brain: BaseBrain,
    env: BilliardsEnv,
    cfg: Config,
    save_gif: bool = False,
    gif_path: str = "drosophila_play.gif",
):
    """
    Финальная анимация: игра + спайки.
    Запускает детерминированный эпизод и записывает всё.
    """
    brain.eval()
    brain.reset_state(1)

    env.reset()
    brain.detach_state()

    fig = plt.figure(figsize=(14, 6))
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.25)
    ax_game = fig.add_subplot(gs[0, 0])
    ax_game.set_facecolor("#1a5c1a")
    ax_game.set_xlim(-12, env.W + 12)
    ax_game.set_ylim(-12, env.H + 12)
    ax_game.set_aspect("equal")
    ax_game.set_xticks([]); ax_game.set_yticks([])
    ax_game.set_title("Digital Drosophila plays billiards")

    ax_spk = fig.add_subplot(gs[0, 1])
    ax_spk.set_facecolor("#0e0e12")
    ax_spk.set_title("Spike raster", color="white")
    ax_spk.tick_params(colors="#aaa")

    for p in env.pockets:
        ax_game.add_patch(Circle(p, env.POCKET_R, color="black", zorder=1))
    balls_patches = []
    for i in range(env.N_BALLS):
        c = Circle((0, 0), env.BALL_R, color="white", ec="black", zorder=2)
        ax_game.add_patch(c)
        balls_patches.append(c)
    fly_patch = Circle((0, 0), env.FLY_R, color="red", zorder=3)
    ax_game.add_patch(fly_patch)

    spike_history: Dict[str, List[np.ndarray]] = defaultdict(list)
    state = {"done": False, "step": 0}

    def update(frame):
        if state["done"]:
            return []
        with torch.no_grad():
            obs = env.get_obs()
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            mean, spikes = brain(obs_t)
            action = mean.cpu().numpy()[0]
        _, _, done, _ = env.step(action)
        state["done"] = done
        state["step"] += 1

        for i, c in enumerate(balls_patches):
            if env.potted[i]:
                c.set_visible(False)
            else:
                c.set_visible(True)
                c.center = (float(env.balls_pos[i][0]), float(env.balls_pos[i][1]))
                c.set_color(np.array(BallType.COLOR[env.ball_types[i]]) / 255.0)
        fly_patch.center = (float(env.fly_pos[0]), float(env.fly_pos[1]))

        for name, spk in spikes.items():
            arr = spk.detach().cpu().numpy()
            if arr.ndim == 2:
                arr = arr[0]
            spike_history[name].append(arr)
        ax_spk.clear()
        ax_spk.set_facecolor("#0e0e12")
        ax_spk.set_title("Spike raster", color="white")
        ax_spk.tick_params(colors="#aaa")
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(spike_history), 1)))
        offset = 0
        for i, (name, hist) in enumerate(spike_history.items()):
            if not hist:
                continue
            arr = np.stack(hist, axis=0)
            n_show = min(arr.shape[1], 60)
            sub = arr[:, :n_show]
            ts, ns = np.where(sub > 0.5)
            ax_spk.scatter(ts, ns + offset, s=2, c=[colors[i]], label=name, alpha=0.85)
            offset += n_show + 5
        ax_spk.legend(loc="upper right", fontsize=7, facecolor="#222", labelcolor="white")
        return []

    anim = FuncAnimation(fig, update, frames=cfg.viz.n_frames_final,
                         interval=cfg.viz.matplotlib_interval_ms, blit=False)
    if save_gif:
        LOG.info(f"Сохранение GIF: {gif_path}")
        anim.save(gif_path, writer=PillowWriter(fps=25))
    else:
        plt.show()
    return anim


# =============================================================================
# SECTION 17 — CLI
# =============================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Digital Drosophila plays billiards")
    sub = p.add_subparsers(dest="cmd", required=True)

    # --- train ---
    pt = sub.add_parser("train", help="Обучение")
    pt.add_argument("--algo", choices=list(ALGOS.keys()), default="ppo")
    pt.add_argument("--brain", choices=["synthetic", "connectome", "hybrid"], default="synthetic")
    pt.add_argument("--episodes", type=int, default=2000)
    pt.add_argument("--seed", type=int, default=42)
    pt.add_argument("--device", default="cpu")
    pt.add_argument("--output-dir", default="runs/drosophila_billiards")
    pt.add_argument("--resume", default=None)
    pt.add_argument("--no-gui", action="store_true", help="Отключить все окна")
    pt.add_argument("--no-pygame", action="store_true", help="Отключить только Pygame окно")
    pt.add_argument("--no-dashboard", action="store_true", help="Отключить matplotlib дашборд")
    pt.add_argument("--save-gif", action="store_true")
    pt.add_argument("--gif-path", default="drosophila_play.gif")
    pt.add_argument("--dry-run", action="store_true")
    pt.add_argument("--log-level", default="INFO")
    pt.add_argument("--obstacles", type=int, default=0)
    pt.add_argument("--powerups", type=int, default=0)
    pt.add_argument("--n-balls", type=int, default=3)
    pt.add_argument("--n-heavy", type=int, default=0)
    pt.add_argument("--n-light", type=int, default=0)

    # --- eval ---
    pe = sub.add_parser("eval", help="Оценка")
    pe.add_argument("--checkpoint", required=True)
    pe.add_argument("--episodes", type=int, default=10)
    pe.add_argument("--no-gui", action="store_true")

    # --- play ---
    pp = sub.add_parser("play", help="Проигрывание чекпоинта с анимацией")
    pp.add_argument("--checkpoint", required=True)
    pp.add_argument("--save-gif", action="store_true")
    pp.add_argument("--gif-path", default="drosophila_play.gif")

    # --- export ---
    px = sub.add_parser("export", help="Экспорт модели (state_dict + config) в .pt")
    px.add_argument("--checkpoint", required=True)
    px.add_argument("--out", default="exported_model.pt")

    # --- info ---
    pi = sub.add_parser("info", help="Информация о среде/мозге")
    pi.add_argument("--brain", default="synthetic")

    # --- test ---
    sub.add_parser("test", help="Запустить unit-тесты")

    return p.parse_args()


def build_config_from_args(args) -> Config:
    cfg = Config()
    if args.cmd == "train":
        cfg.algo.algo = args.algo
        cfg.brain.mode = args.brain
        cfg.train.n_episodes = args.episodes
        cfg.train.seed = args.seed
        cfg.train.device = args.device
        cfg.output_dir = args.output_dir
        cfg.log_level = args.log_level
        cfg.resume_from = args.resume
        cfg.dry_run = args.dry_run
        cfg.viz.live_game_window = not (args.no_gui or args.no_pygame)
        cfg.viz.live_dashboard = not (args.no_gui or args.no_dashboard)
        cfg.viz.save_gif = args.save_gif
        cfg.viz.gif_path = args.gif_path
        cfg.env.n_balls = args.n_balls
        cfg.env.n_heavy_balls = args.n_heavy
        cfg.env.n_light_balls = args.n_light
        cfg.env.n_obstacles = args.obstacles
        cfg.env.n_powerups = args.powerups
        cfg.env.enable_obstacles = args.obstacles > 0
        cfg.env.enable_powerups = args.powerups > 0
    return cfg


# =============================================================================
# SECTION 18 — COMMANDS
# =============================================================================
def cmd_train(args) -> int:
    global LOG
    cfg = build_config_from_args(args)
    LOG = setup_logging(cfg)
    set_seed(cfg.train.seed)

    LOG.info("=" * 74)
    LOG.info("DIGITAL DROSOPHILA PLAYS BILLIARDS — ULTIMATE EXTENDED EDITION")
    LOG.info("=" * 74)
    LOG.info(f"Config:\n{cfg.to_json()}")

    env = BilliardsEnv(cfg.env, cfg.reward)
    brain = build_brain(cfg, env.obs_dim, 2)
    critic = Critic(env.obs_dim)

    LOG.info(f"Brain: {brain.__class__.__name__} | obs_dim={env.obs_dim}")
    if hasattr(brain, "W_val"):
        LOG.info(f"  Connectome: {brain.N} neurons, {brain.W_val.numel()} synapses")

    if cfg.resume_from:
        try:
            ckpt = load_checkpoint(cfg.resume_from, map_location=cfg.train.device)
            brain.load_state_dict(ckpt["brain"])
            if "critic" in ckpt:
                critic.load_state_dict(ckpt["critic"])
            LOG.info(f"Resumed from {cfg.resume_from}")
        except Exception as e:
            LOG.error(f"Resume failed: {e}. Starting fresh.")

    if cfg.dry_run:
        LOG.info("Dry run завершён.")
        return 0

    # --- визуализация ---
    dashboard = None
    if cfg.viz.live_dashboard:
        try:
            dashboard = LiveDashboard(env, brain, cfg)
        except Exception as e:
            LOG.warning(f"LiveDashboard недоступен: {e}")
            dashboard = None

    pygame_renderer = None
    if cfg.viz.live_game_window and _HAS_PYGAME:
        try:
            pygame_renderer = PygameRenderer(env, scale=cfg.viz.live_game_scale,
                                             fps=cfg.viz.live_game_fps)
        except Exception as e:
            LOG.warning(f"PygameRenderer недоступен: {e}")
            pygame_renderer = None
    elif cfg.viz.live_game_window and not _HAS_PYGAME:
        LOG.warning("pygame не установлен — живое окно игры отключено.")

    algo = build_algo(cfg.algo.algo, brain, critic, env, cfg,
                      dashboard=dashboard, pygame_renderer=pygame_renderer)

    out_dir = safe_mkdir(cfg.output_dir)
    trainer = Trainer(cfg, brain, critic, env, algo, out_dir,
                      dashboard=dashboard, pygame_renderer=pygame_renderer)

    try:
        trainer.train()
    except KeyboardInterrupt:
        LOG.warning("Прервано пользователем. Сохраняем финальное состояние.")
        trainer.save_final()
    finally:
        if dashboard is not None:
            dashboard.close()
        if pygame_renderer is not None:
            pygame_renderer.close()

    if cfg.viz.save_gif:
        try:
            run_final_animation(brain, env, cfg, save_gif=True, gif_path=cfg.viz.gif_path)
        except Exception as e:
            LOG.error(f"GIF ошибка: {e}")

    LOG.info("Готово.")
    return 0


def cmd_eval(args) -> int:
    global LOG
    cfg = Config()
    cfg.log_level = "INFO"
    LOG = setup_logging(cfg)

    env = BilliardsEnv(cfg.env, cfg.reward)
    brain = build_brain(cfg, env.obs_dim, 2)
    critic = Critic(env.obs_dim)
    ckpt = load_checkpoint(args.checkpoint, map_location="cpu")
    brain.load_state_dict(ckpt["brain"])
    brain.eval()

    rewards, potted = [], []
    for ep in range(args.episodes):
        env.reset()
        brain.reset_state(1)
        brain.detach_state()
        done = False
        total_r = 0.0
        while not done:
            obs_t = torch.tensor(env.get_obs(), dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                mean, _ = brain(obs_t)
            _, r, done, _ = env.step(mean.cpu().numpy()[0])
            total_r += r
        rewards.append(total_r)
        potted.append(int(env.potted.sum()))
        LOG.info(f"ep {ep}: reward={total_r:.2f} potted={potted[-1]}")

    LOG.info(f"Средний reward: {np.mean(rewards):.2f}; среднее potted: {np.mean(potted):.2f}")
    return 0


def cmd_play(args) -> int:
    global LOG
    cfg = Config()
    LOG = setup_logging(cfg)

    env = BilliardsEnv(cfg.env, cfg.reward)
    brain = build_brain(cfg, env.obs_dim, 2)
    ckpt = load_checkpoint(args.checkpoint, map_location="cpu")
    brain.load_state_dict(ckpt["brain"])
    brain.eval()

    run_final_animation(brain, env, cfg, save_gif=args.save_gif, gif_path=args.gif_path)
    return 0


def cmd_export(args) -> int:
    global LOG
    cfg = Config()
    LOG = setup_logging(cfg)
    ckpt = load_checkpoint(args.checkpoint, map_location="cpu")
    torch.save({
        "brain": ckpt["brain"],
        "critic": ckpt.get("critic"),
        "config": ckpt.get("config", ""),
        "best_potted": ckpt.get("best_potted", 0),
    }, args.out)
    LOG.info(f"Экспортировано: {args.out}")
    return 0


def cmd_info(args) -> int:
    global LOG
    cfg = Config()
    cfg.brain.mode = args.brain
    LOG = setup_logging(cfg)
    env = BilliardsEnv(cfg.env, cfg.reward)
    LOG.info(f"obs_dim = {env.obs_dim}")
    LOG.info(f"n_balls = {env.n_balls if hasattr(env, 'n_balls') else env.N_BALLS}")
    LOG.info(f"table = {env.W}x{env.H}, pockets = {len(env.pockets)}")
    brain = build_brain(cfg, env.obs_dim, 2)
    n_params = sum(p.numel() for p in brain.parameters() if p.requires_grad)
    LOG.info(f"brain = {brain.__class__.__name__}, params = {n_params}")
    if hasattr(brain, "W_val"):
        LOG.info(f"connectome synapses = {brain.W_val.numel()}")
    LOG.info(f"pygame available = {_HAS_PYGAME}, flywire available = {_HAS_FLYWIRE}")
    return 0


def cmd_test(args) -> int:
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestEnvironment)
    suite.addTests(loader.loadTestsFromTestCase(TestSpikes))
    suite.addTests(loader.loadTestsFromTestCase(TestBrains))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


# =============================================================================
# SECTION 19 — TESTS
# =============================================================================
class TestEnvironment(unittest.TestCase):
    def setUp(self):
        self.cfg = EnvConfig()
        self.rc = RewardConfig()
        self.env = BilliardsEnv(self.cfg, self.rc)

    def test_reset(self):
        obs = self.env.reset()
        self.assertEqual(obs.shape[0], self.env.obs_dim)
        self.assertEqual(int(self.env.potted.sum()), 0)

    def test_step(self):
        obs, r, done, info = self.env.step(np.array([0.1, 0.0]))
        self.assertEqual(obs.shape[0], self.env.obs_dim)
        self.assertIsInstance(r, float)
        self.assertIsInstance(done, bool)

    def test_potted_when_at_pocket(self):
        # Помещаем шар прямо в лузу
        self.env.balls_pos[0] = self.env.pockets[0].copy()
        self.env.balls_vel[0] = np.array([0.0, 0.0])
        _, r, _, info = self.env.step(np.array([0.0, 0.0]))
        self.assertTrue(self.env.potted[0])
        self.assertGreater(r, 0.0)

    def test_wall_bounce(self):
        self.env.balls_pos[0] = np.array([self.env.BALL_R + 0.1, 50.0])
        self.env.balls_vel[0] = np.array([-5.0, 0.0])
        for _ in range(5):
            self.env.step(np.array([0.0, 0.0]))
        self.assertGreaterEqual(self.env.balls_pos[0, 0], self.env.BALL_R - 1e-3)

    def test_episode_terminates(self):
        env = BilliardsEnv(EnvConfig(max_steps=10), RewardConfig())
        done = False
        for _ in range(20):
            _, _, done, _ = env.step(np.array([0.0, 0.0]))
            if done:
                break
        self.assertTrue(done)

    def test_ball_type_distribution(self):
        cfg = EnvConfig(n_balls=5, n_heavy_balls=2, n_light_balls=1)
        env = BilliardsEnv(cfg, RewardConfig())
        env.reset()
        heavy = sum(1 for t in env.ball_types if t == BallType.HEAVY)
        light = sum(1 for t in env.ball_types if t == BallType.LIGHT)
        self.assertEqual(heavy, 2)
        self.assertEqual(light, 1)


class TestSpikes(unittest.TestCase):
    def test_surrogate_forward(self):
        v = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0])
        spk = spike_fn(v)
        self.assertEqual(spk.tolist(), [0.0, 0.0, 0.0, 1.0, 1.0])

    def test_surrogate_gradient_flow(self):
        v = torch.tensor([0.1, -0.1], requires_grad=True)
        spk = spike_fn(v)
        loss = spk.sum()
        loss.backward()
        self.assertIsNotNone(v.grad)
        self.assertGreater(v.grad.abs().sum().item(), 0)

    def test_all_surrogates(self):
        for kind in SPIKE_FUNCS:
            v = torch.tensor([0.5], requires_grad=True)
            spk = spike_fn(v, kind)
            spk.sum().backward()
            self.assertIsNotNone(v.grad)


class TestBrains(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()
        self.cfg.brain.mode = "synthetic"

    def test_synthetic_forward(self):
        brain = SyntheticFlyBrain(28, 2, self.cfg)
        brain.reset_state(1)
        obs = torch.randn(1, 28)
        mean, spikes = brain(obs)
        self.assertEqual(mean.shape, (1, 2))
        self.assertIn("pn", spikes)
        self.assertIn("mbon", spikes)

    def test_act_sampling(self):
        brain = SyntheticFlyBrain(28, 2, self.cfg)
        brain.reset_state(1)
        obs = torch.randn(1, 28)
        a, lp, ent, _ = brain.act(obs, deterministic=False)
        self.assertEqual(a.shape, (1, 2))
        self.assertEqual(lp.shape, (1,))

    def test_replay_buffer(self):
        buf = ReplayBuffer(100, 28)
        for i in range(50):
            buf.add(np.random.randn(28), 0, 1.0, np.random.randn(28), False)
        self.assertEqual(len(buf), 50)
        o, a, r, no, d = buf.sample(8)
        self.assertEqual(o.shape, (8, 28))

    def test_discounted_returns(self):
        rets = discounted_returns([1.0, 1.0, 1.0], 0.9)
        self.assertAlmostEqual(rets[0], 1 + 0.9 + 0.81, places=4)


# =============================================================================
# SECTION 20 — MAIN
# =============================================================================
def main() -> int:
    args = parse_args()
    handlers = {
        "train": cmd_train,
        "eval": cmd_eval,
        "play": cmd_play,
        "export": cmd_export,
        "info": cmd_info,
        "test": cmd_test,
    }
    fn = handlers.get(args.cmd)
    if fn is None:
        print(f"Unknown command: {args.cmd}")
        return 1
    return fn(args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nПрервано пользователем.")
        sys.exit(130)
    except Exception:
        traceback.print_exc()
        sys.exit(1)