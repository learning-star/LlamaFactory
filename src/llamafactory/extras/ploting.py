# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import math
import os
from typing import Any

from transformers.trainer import TRAINER_STATE_NAME

from . import logging
from .packages import is_matplotlib_available


if is_matplotlib_available():
    import matplotlib.figure
    import matplotlib.pyplot as plt


logger = logging.get_logger(__name__)


def smooth(scalars: list[float]) -> list[float]:
    r"""EMA implementation according to TensorBoard."""
    if len(scalars) == 0:
        return []

    last = scalars[0]
    smoothed = []
    weight = 1.8 * (1 / (1 + math.exp(-0.05 * len(scalars))) - 0.5)  # a sigmoid function
    for next_val in scalars:
        smoothed_val = last * weight + (1 - weight) * next_val
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def gen_metric_plot(
    trainer_log: list[dict[str, Any]], metric_key: str, ylabel: str | None = None
) -> "matplotlib.figure.Figure":
    r"""Plot metric curves in LlamaBoard."""
    plt.close("all")
    plt.switch_backend("agg")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    steps, metrics = [], []
    for log in trainer_log:
        if log.get(metric_key, None) is not None:
            steps.append(log["current_steps"])
            metrics.append(log[metric_key])

    if len(metrics) != 0:
        ax.plot(steps, metrics, color="#1f77b4", label=ylabel or metric_key)
        ax.legend()

    ax.set_xlabel("step")
    ax.set_ylabel(ylabel or metric_key)
    return fig


def gen_metric_compare_plot(
    trainer_logs: dict[str, list[dict[str, Any]]], metric_key: str, ylabel: str | None = None
) -> "matplotlib.figure.Figure":
    r"""Plot multiple smoothed metric curves in LlamaBoard."""
    plt.close("all")
    plt.switch_backend("agg")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])

    for idx, (label, trainer_log) in enumerate(trainer_logs.items()):
        steps, metrics = [], []
        for log in trainer_log:
            if log.get(metric_key, None) is not None:
                steps.append(log["current_steps"])
                metrics.append(log[metric_key])

        if len(metrics) == 0:
            continue

        color = color_cycle[idx % len(color_cycle)] if color_cycle else "#1f77b4"
        ax.plot(steps, smooth(metrics), color=color, label=label)

    if ax.lines:
        ax.legend()

    ax.set_xlabel("step")
    ax.set_ylabel(ylabel or metric_key)
    return fig


def gen_loss_plot(trainer_log: list[dict[str, Any]]) -> "matplotlib.figure.Figure":
    r"""Plot loss curves in LlamaBoard."""
    return gen_metric_plot(trainer_log, "loss", "loss")


def gen_loss_compare_plot(trainer_logs: dict[str, list[dict[str, Any]]]) -> "matplotlib.figure.Figure":
    r"""Plot multiple smoothed loss curves in LlamaBoard."""
    return gen_metric_compare_plot(trainer_logs, "loss", "loss")


def _load_plot_log_history(save_dictionary: str) -> list[dict[str, Any]]:
    r"""Load metrics from Trainer state or LlamaBoard log stream."""
    trainer_state_path = os.path.join(save_dictionary, TRAINER_STATE_NAME)
    if os.path.isfile(trainer_state_path):
        with open(trainer_state_path, encoding="utf-8") as f:
            return json.load(f).get("log_history", [])

    trainer_log_path = os.path.join(save_dictionary, "trainer_log.jsonl")
    if not os.path.isfile(trainer_log_path):
        logger.warning_rank0(f"No trainer state or trainer log found in {save_dictionary}.")
        return []

    log_history: list[dict[str, Any]] = []
    with open(trainer_log_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                log = json.loads(line)
            except json.JSONDecodeError:
                logger.warning_rank0(f"Skipping malformed trainer log line in {trainer_log_path}.")
                continue

            if "step" not in log and "current_steps" in log:
                log["step"] = log["current_steps"]

            log_history.append(log)

    return log_history


def plot_loss(save_dictionary: str, keys: list[str] = ["loss"]) -> None:
    r"""Plot loss curves and saves the image."""
    plt.switch_backend("agg")
    log_history = _load_plot_log_history(save_dictionary)
    if len(log_history) == 0:
        return

    for key in keys:
        steps, metrics = [], []
        for log in log_history:
            if key in log and "step" in log:
                steps.append(log["step"])
                metrics.append(log[key])

        if len(metrics) == 0:
            logger.warning_rank0(f"No metric {key} to plot.")
            continue

        plt.figure()
        plt.plot(steps, metrics, color="#1f77b4", label=key)
        plt.title(key)
        plt.xlabel("step")
        plt.ylabel(key)
        plt.legend()
        figure_path = os.path.join(save_dictionary, "training_{}.png".format(key.replace("/", "_")))
        plt.savefig(figure_path, format="png", dpi=100)
        print("Figure saved at:", figure_path)
