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

from collections import OrderedDict
from typing import Any

from ..extras.constants import DownloadSource


ECOPHASE_SUPPORTED_MODELS = OrderedDict(
    {
        "Qwen3-1.7B": {
            DownloadSource.DEFAULT: "/root/autodl-tmp/model/Qwen/Qwen3-1.7B",
            DownloadSource.MODELSCOPE: "/root/autodl-tmp/model/Qwen/Qwen3-1.7B",
        }
    }
)
ECOPHASE_MODEL_TEMPLATES = {"Qwen3-1.7B": "qwen3"}
ECOPHASE_SUPPORTED_DATASETS = ["scienceqa_train", "alpaca_zh_demo", "alpaca_en_demo"]

ECOPHASE_TRAINING_PRESETS: "OrderedDict[str, dict[str, Any]]" = OrderedDict(
    {
        "ScienceQA full 微调（推荐）": {
            "model_name": "Qwen3-1.7B",
            "model_path": "/root/autodl-tmp/model/Qwen/Qwen3-1.7B",
            "finetuning_type": "full",
            "template": "qwen3",
            "training_stage": "Supervised Fine-Tuning",
            "dataset_dir": "data",
            "dataset": ["scienceqa_train"],
            "cutoff_len": 4096,
            "learning_rate": "5e-5",
            "num_train_epochs": "5",
            "batch_size": 16,
            "gradient_accumulation_steps": 1,
            "compute_type": "bf16",
            "val_size": 0.1,
            "lr_scheduler_type": "cosine",
            "max_grad_norm": "1.0",
            "packing": False,
        },
        "中文指令 LoRA 快速验证": {
            "model_name": "Qwen3-1.7B",
            "model_path": "/root/autodl-tmp/model/Qwen/Qwen3-1.7B",
            "finetuning_type": "lora",
            "template": "qwen3",
            "training_stage": "Supervised Fine-Tuning",
            "dataset_dir": "data",
            "dataset": ["alpaca_zh_demo"],
            "cutoff_len": 2048,
            "learning_rate": "1e-4",
            "num_train_epochs": "3",
            "batch_size": 1,
            "gradient_accumulation_steps": 8,
            "compute_type": "bf16",
            "val_size": 0.1,
            "lr_scheduler_type": "cosine",
            "max_grad_norm": "1.0",
            "packing": False,
        },
        "英文指令 LoRA 快速验证": {
            "model_name": "Qwen3-1.7B",
            "model_path": "/root/autodl-tmp/model/Qwen/Qwen3-1.7B",
            "finetuning_type": "lora",
            "template": "qwen3",
            "training_stage": "Supervised Fine-Tuning",
            "dataset_dir": "data",
            "dataset": ["alpaca_en_demo"],
            "cutoff_len": 2048,
            "learning_rate": "1e-4",
            "num_train_epochs": "3",
            "batch_size": 1,
            "gradient_accumulation_steps": 8,
            "compute_type": "bf16",
            "val_size": 0.1,
            "lr_scheduler_type": "cosine",
            "max_grad_norm": "1.0",
            "packing": False,
        },
    }
)


def get_ecophase_training_preset(name: str) -> dict[str, Any]:
    r"""Return a copy of an EcoPhase one-click training preset."""
    return dict(ECOPHASE_TRAINING_PRESETS.get(name) or next(iter(ECOPHASE_TRAINING_PRESETS.values())))
